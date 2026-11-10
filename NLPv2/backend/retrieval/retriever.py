import os
import re
import psycopg2
import json
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

from graph.citation_resolver import (
    STOPWORDS as _NAME_STOPWORDS,
    citation_titles,
    normalize_case_name,
    parse_reporter_citations,
)


# Explicit "article N" / "section N" references in a query let us skip
# straight to the row instead of hoping dense/BM25 surface it — bare
# digits like "21" are weakly weighted by tsvector tokenization, and the
# word "article" never appears inside the article's own body text.
_ARTICLE_REF_RE = re.compile(r"\b(?:article|art\.?)\s+(\d{1,3}[a-zA-Z]?)\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\b(?:section|sec\.?)\s+(\d{1,3}[a-zA-Z]?)\b", re.IGNORECASE)

# Which statute a "section N" reference means, by the Act named in the query,
# checked in order — BNSS before BNS, since one name contains the other. A
# section of an Act that isn't ingested (CrPC, IPC, CPC, Evidence Act) maps to
# None: the Sanhitas renumbered everything, so CrPC s.41 is not BNSS s.41.
# No Act named means BNS, as before BNSS was added.
_SECTION_STATUTES = [
    ("BNSS", re.compile(r"\b(?:bnss|nagarik\s+suraksha)\b", re.IGNORECASE)),
    ("BNS", re.compile(r"\b(?:bns|nyaya\s+sanhita)\b", re.IGNORECASE)),
    (None, re.compile(
        r"\b(?:cr\.?\s?p\.?\s?c|i\.?\s?p\.?\s?c|c\.?\s?p\.?\s?c|code\s+of\s+(?:criminal|civil)\s+procedure"
        r"|penal\s+code|evidence\s+act)\b",
        re.IGNORECASE,
    )),
]


# BM25 leg query modes (benchmark/REPORT.md, 2026-11-09 results):
#   "and"     plainto_tsquery: every query word in one chunk. Precise — it is
#             what makes "AIR 1978 SC 597" work — but a passage-length query
#             matches nothing, so hybrid ran dense-only for 89% of them.
#   "or"      any informative query term, ranked by ts_rank_cd. Helps long
#             queries a little but drops citation hit@8 0.73 -> 0.02 (common
#             terms "air", "sc", "1978" are pruned) and costs ~3.7 s per call.
#   "and_or"  (default) the AND results first, then the remaining slots
#             filled from a narrower OR query under a statement timeout.
BM25_MODES = ("and", "or", "and_or")
_TSQUERY_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# OR over a term found in most chunks ("court", "appeal") makes nearly every
# chunk a match and ts_rank_cd has to score them all. Terms whose chunk
# frequency in pg_stats exceeds max_df are dropped; at most max_terms terms
# are kept, rarest first. The fill query is narrower and time-capped, since
# it runs on every query AND leaves short.
_OR_MAX_DF, _OR_MAX_TERMS = 0.05, 32
_FILL_MAX_DF, _FILL_MAX_TERMS = 0.01, 8
# A query with more distinct lexemes than this is passage-length; on the
# 2026-11-09 server run its fill timed out 856/990 times (AILA: 86/86), adding
# ~0.5 s for nothing, so the fill is skipped for it.
_FILL_MAX_QUERY_LEXEMES = 30
# 200 ms was tried (d288351): fill timeouts 140 -> 471/882 on short queries,
# hit@8 0.082 -> 0.072, for ~100 ms less BM25 time; 500 ms kept.
_FILL_TIMEOUT_MS = 500
_QUERY_CANCELED = "57014"  # Postgres SQLSTATE for statement_timeout

# Case lookup: a judgment the query names by reporter citation or "X v. Y"
# title is fetched through the citation resolver and pinned first (REPORT.md:
# chat found a named case in its top 8 40% of the time; the alias table 98%).
# Only the resolver's high-precision methods are trusted.
_LOOKUP_METHODS = ("reporter", "exact_name")
# "the Maneka Gandhi case": the party is some trailing run of the words before
# "case". Each run of up to _PARTY_MAX_TOKENS words is tried against the
# aliases' party sides, longest first (names keep commas, brackets and "A."
# initials, and some run past six words).
_CASE_WORD_RE = re.compile(r"\bcase\b", re.I)
# "Maneka Gandhi's case", "the respondents' case": a possessive before "case".
_POSSESSIVE_RE = re.compile(r"(?<=\w)(?:['\u2019]s|(?<=s)['\u2019])\s*$", re.I)
_ROLE_WORDS = frozenset("appellant appellants respondent respondents petitioner petitioners".split())
_PARTY_MAX_TOKENS = 12
_LOOKUP_MAX_SPANS = 3  # "case" occurrences tried per query
_GENERIC_CASE_WORDS = frozenset(
    "this that present instant above said landmark famous which what the a an "
    "criminal civil bail murder dowry writ appeal constitutional leading "
    # sentence openers that get captured when a question starts with them
    "is was are were did does do how why when where who explain describe summarise "
    "summarize tell discuss compare in see per under cf".split()
)
# A party name doesn't start with these (single letters are initials: "A. K.").
_NOT_PARTY_START = (_GENERIC_CASE_WORDS | {"of", "and", "v", "on", "to", "for", "by"}) - {"a"}
# Nor ends with these: "the present case", "the prosecution case", "the
# appellant's case" are about the case at hand, so no lookup is run for them.
_NOT_PARTY_END = _GENERIC_CASE_WORDS | frozenset(
    "s his her their its our my your own same such any every each other whole entire "
    "prosecution defence defense appellant appellants respondent respondents petitioner "
    "petitioners accused plaintiff plaintiffs defendant defendants complainant "
    "of and v on to for by".split()
)
# A name matching more judgments is too ambiguous to pin. The case-lookup gold
# set keeps queries with up to 5 relevant judgments (case_lookup_gold.py).
_LOOKUP_MAX_JUDGMENTS = 5


def _select_or_terms(
    lexemes: List[str], doc_freq: Dict[str, float], max_df: float, max_terms: int
) -> List[str]:
    """Keep query lexemes for an OR tsquery: drop ones in more than max_df of
    chunks, then keep the max_terms rarest. A lexeme missing from pg_stats'
    most-common list is rarer than all of them, so it sorts first; ties keep
    query order."""
    seen = set()
    unique = [l for l in lexemes if not (l in seen or seen.add(l))]
    kept = [l for l in unique if doc_freq.get(l, 0.0) <= max_df]
    kept.sort(key=lambda l: doc_freq.get(l, 0.0))  # stable sort keeps query order
    return kept[:max_terms]


def _default_db_connection_string() -> str:
    """Build the default DSN from env vars, matching the docker-compose api
    service's DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD, so this also works
    when run directly on the host against the dockerized postgres."""
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def named_case_phrases(query: str) -> List[List[str]]:
    """For each "<Name> case" in the query (at most _LOOKUP_MAX_SPANS), the
    party names it could be: the normalised runs of words ending just before
    "case", longest first. Runs that start with a connective ("of punjab") or
    have no distinctive word ("union of india") are dropped, and so is the
    whole span when the word before "case" can't end a name ("this case",
    "the bail case", "the prosecution case")."""
    out: List[List[str]] = []
    start = 0
    for m in _CASE_WORD_RE.finditer(query or ""):
        before = query[start:m.start()]
        start = m.end()
        possessive = _POSSESSIVE_RE.search(before)
        if possessive:
            before = before[: possessive.start()]
            # "the appellant's case" is the party's argument, not a case name.
            # (A bare "...Appellant(s) case" is a header label, which
            # normalize_case_name drops.)
            last = re.findall(r"[a-z]+", before.lower())[-1:]
            if last and last[0] in _ROLE_WORDS:
                continue
        tokens = normalize_case_name(before).split()[-_PARTY_MAX_TOKENS:]
        if not tokens or tokens[-1] in _NOT_PARTY_END:
            continue
        candidates = []
        for i in range(len(tokens)):
            run = tokens[i:]
            if run[0] in _NOT_PARTY_START:
                continue
            if any(
                t not in _NAME_STOPWORDS and t not in _GENERIC_CASE_WORDS and len(t) > 2 and not t.isdigit()
                for t in run
            ):
                candidates.append(" ".join(run))
        if candidates:
            out.append(candidates)
            if len(out) == _LOOKUP_MAX_SPANS:
                break
    return out


def lookup_strings(query: str) -> List[str]:
    """Citation snippets for the resolver: reporter citations and "X v. Y"
    titles written in the query (same extraction as uploaded documents)."""
    out = [c.raw for c in parse_reporter_citations(query or "")]
    out += [f"{p} v {r}".strip() for p, r in citation_titles(query or "")]
    return list(dict.fromkeys(out))


class LegalRetriever:
    bm25_mode = "and_or"
    case_lookup = True
    _chunk_doc_freq: Optional[Dict[str, float]] = None

    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        graph_manager: Optional[Any] = None,
        bm25_mode: str = "and_or",
        case_lookup: bool = True,
    ):
        if bm25_mode not in BM25_MODES:
            raise ValueError(f"bm25_mode must be one of {BM25_MODES}, got {bm25_mode!r}")
        self.db_connection_string = db_connection_string or _default_db_connection_string()
        self.embedding_model = None
        self.graph_manager = graph_manager
        self.bm25_mode = bm25_mode
        # Needs graph_manager (the citation resolver); off without one.
        self.case_lookup = case_lookup
        # BM25 counters for evals: calls, fills (AND short, OR queried), fill timeouts.
        self.bm25_stats: Counter = Counter()
        self._max_centrality: Optional[float] = None
        self._load_embedding_model()

    def _centrality_max(self) -> float:
        """Largest PageRank in the corpus, fetched once (graph stats only
        change on re-ingestion)."""
        if self._max_centrality is None:
            scores = self.graph_manager.get_centrality_scores() if self.graph_manager else {}
            self._max_centrality = max(scores.values(), default=0.0)
        return self._max_centrality

    def _load_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
            print("✓ BGE-base model loaded for retrieval")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def get_query_embedding(self, query: str) -> List[float]:
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")

    def retrieve_relevant_chunks(
        self, query: str, top_k: int = 8, similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)

        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT 
                    jc.content,
                    j.petitioner,
                    j.respondent,
                    j.court,
                    EXTRACT(YEAR FROM j.date_of_judgment) as year,
                    jc.section,
                    jc.chunk_id,
                    1 - (je.embedding <=> %s::vector) as similarity
                FROM judgment_embeddings je
                JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                JOIN judgments j ON j.id = jc.judgment_id
                WHERE 1 - (je.embedding <=> %s::vector) >= %s
                ORDER BY je.embedding <=> %s::vector
                LIMIT %s
            """,
                (
                    query_embedding,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    top_k,
                ),
            )

            results = cur.fetchall()

            retrieved_chunks = []
            for row in results:
                (
                    content,
                    petitioner,
                    respondent,
                    court,
                    year,
                    section,
                    chunk_id,
                    similarity,
                ) = row

                case_name = self._format_case_name(petitioner, respondent)
                # chunk_id is a global DB row id, not a paragraph number within
                # this judgment — label it honestly rather than implying a
                # real ¶N citation the source text doesn't actually support.
                para_id = f"(chunk {chunk_id})"

                retrieved_chunks.append(
                    {
                        "doc_type": "judgment",
                        "chunk_id": chunk_id,
                        "text": content.strip(),
                        "case": case_name,
                        "court": court if court else "Unknown Court",
                        "year": int(year) if year else None,
                        "para": para_id,
                        "section": section,
                        "similarity": float(similarity),
                    }
                )

            return retrieved_chunks

        finally:
            cur.close()
            conn.close()

    def retrieve_candidate_chunks(
        self, query: str, candidate_k: int = 30, similarity_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Stage-1 retrieval: fetch more candidates with lower threshold for reranking."""
        query_embedding = self.get_query_embedding(query)

        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT
                    jc.content,
                    j.petitioner,
                    j.respondent,
                    j.court,
                    EXTRACT(YEAR FROM j.date_of_judgment) as year,
                    jc.section,
                    jc.chunk_id,
                    jc.judgment_id,
                    1 - (je.embedding <=> %s::vector) as similarity
                FROM judgment_embeddings je
                JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                JOIN judgments j ON j.id = jc.judgment_id
                WHERE 1 - (je.embedding <=> %s::vector) >= %s
                ORDER BY je.embedding <=> %s::vector
                LIMIT %s
            """,
                (
                    query_embedding,
                    query_embedding,
                    similarity_threshold,
                    query_embedding,
                    candidate_k,
                ),
            )

            results = cur.fetchall()

            retrieved_chunks = []
            for row in results:
                (
                    content,
                    petitioner,
                    respondent,
                    court,
                    year,
                    section,
                    chunk_id,
                    judgment_id,
                    similarity,
                ) = row

                case_name = self._format_case_name(petitioner, respondent)
                # chunk_id is a global DB row id, not a paragraph number within
                # this judgment — label it honestly rather than implying a
                # real ¶N citation the source text doesn't actually support.
                para_id = f"(chunk {chunk_id})"

                retrieved_chunks.append(
                    {
                        "doc_type": "judgment",
                        "chunk_id": chunk_id,
                        "judgment_id": judgment_id,
                        "text": content.strip(),
                        "case": case_name,
                        "court": court if court else "Unknown Court",
                        "year": int(year) if year else None,
                        "para": para_id,
                        "section": section,
                        "similarity": float(similarity),
                    }
                )

            return retrieved_chunks

        finally:
            cur.close()
            conn.close()

    def _extract_explicit_section_ref(self, query: str) -> Optional[Tuple[str, str]]:
        """Detects an explicit 'article N'/'art. N' (-> Constitution) or
        'section N'/'sec. N' reference in the query. Returns
        (statute_short_title, section_number) or None. "article" only ever
        refers to the Constitution; a section belongs to the Act the query
        names (see _SECTION_STATUTES), BNS when it names none."""
        m = _ARTICLE_REF_RE.search(query)
        if m:
            return ("Constitution", m.group(1))
        m = _SECTION_REF_RE.search(query)
        if m:
            for short_title, named in _SECTION_STATUTES:
                if named.search(query):
                    return (short_title, m.group(1)) if short_title else None
            return ("BNS", m.group(1))
        return None

    def _fetch_section_by_number(
        self, cur, statute_short_title: str, section_number: str
    ) -> Optional[Dict[str, Any]]:
        cur.execute(
            """
            SELECT ss.section_id, st.title, ss.section_number, ss.content
            FROM statute_sections ss
            JOIN statutes st ON st.id = ss.statute_id
            WHERE st.short_title = %s AND ss.section_number ILIKE %s
            LIMIT 1
            """,
            (statute_short_title, section_number),
        )
        row = cur.fetchone()
        if not row:
            return None
        sid, title, sec_num, content = row
        return {
            "doc_type": "statute",
            "chunk_id": sid,
            "text": content.strip(),
            "case": title,
            "court": "Statute",
            "year": None,
            "para": f"§{sec_num}",
            "section": "statute",
            "statute_name": title,
            "section_number": sec_num,
            "similarity": 1.0,
            "rrf_score": float("inf"),
            "exact_match": True,
        }

    def retrieve_statute_candidates(
        self,
        query: str,
        candidate_k: int = 15,
        similarity_threshold: float = 0.2,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Retrieve statute sections: dense + bm25 over statute tables, fused with RRF.
        An explicit 'article N'/'section N' reference in the query is looked up
        directly and guaranteed to rank first, regardless of its dense/BM25 scores."""
        query_embedding = self.get_query_embedding(query)
        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()

        try:
            ref = self._extract_explicit_section_ref(query)
            exact_chunk = self._fetch_section_by_number(cur, ref[0], ref[1]) if ref else None

            # Dense leg
            cur.execute(
                """
                SELECT se.section_id, ss.statute_id, st.title, st.short_title,
                       ss.section_number, ss.heading, ss.content,
                       1 - (se.embedding <=> %s::vector) as similarity
                FROM statute_embeddings se
                JOIN statute_sections ss ON ss.section_id = se.section_id
                JOIN statutes st ON st.id = ss.statute_id
                WHERE 1 - (se.embedding <=> %s::vector) >= %s
                ORDER BY se.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, similarity_threshold, query_embedding, candidate_k),
            )
            dense = cur.fetchall()

            # BM25 leg
            cur.execute(
                """
                SELECT ss.section_id, ss.statute_id, st.title, st.short_title,
                       ss.section_number, ss.heading, ss.content,
                       ts_rank(ss.content_tsv, plainto_tsquery('english', %s)) as bm25_score
                FROM statute_sections ss
                JOIN statutes st ON st.id = ss.statute_id
                WHERE ss.content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY bm25_score DESC
                LIMIT %s
                """,
                (query, query, candidate_k),
            )
            bm25 = cur.fetchall()

            fused: Dict[str, Dict[str, Any]] = {}
            scores: Dict[str, float] = {}

            for rank, row in enumerate(dense):
                sid, statute_id, title, short, sec_num, heading, content, sim = row
                key = f"s_{sid}"
                fused[key] = {
                    "doc_type": "statute",
                    "chunk_id": sid,
                    "text": content.strip(),
                    "case": title,
                    "court": "Statute",
                    "year": None,
                    "para": f"\u00a7{sec_num}",
                    "section": "statute",
                    "statute_name": title,
                    "section_number": sec_num,
                    "similarity": float(sim),
                }
                scores[key] = 1.0 / (rrf_k + rank + 1)

            for rank, row in enumerate(bm25):
                sid, statute_id, title, short, sec_num, heading, content, bm = row
                key = f"s_{sid}"
                if key not in fused:
                    fused[key] = {
                        "doc_type": "statute",
                        "chunk_id": sid,
                        "text": content.strip(),
                        "case": title,
                        "court": "Statute",
                        "year": None,
                        "para": f"\u00a7{sec_num}",
                        "section": "statute",
                        "statute_name": title,
                        "section_number": sec_num,
                        "similarity": 0.0,
                    }
                else:
                    fused[key].setdefault("bm25_score", 0.0)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

            for key, s in scores.items():
                fused[key]["rrf_score"] = s

            ranked = sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)

            if exact_chunk:
                ranked = [c for c in ranked if c["chunk_id"] != exact_chunk["chunk_id"]]
                ranked = [exact_chunk] + ranked

            return ranked[:candidate_k]

        finally:
            cur.close()
            conn.close()

    def _or_query_terms(
        self, cur, query: str, max_df: float = _OR_MAX_DF, max_terms: int = _OR_MAX_TERMS
    ) -> List[str]:
        """Stemmed, stopword-free lexemes of the query (Postgres' own
        'english' config, so they match content_tsv), minus terms too common
        to be worth an OR — see _OR_MAX_DF."""
        lexemes = self._query_lexemes(cur, query)
        return _select_or_terms(lexemes, self._doc_freq(cur), max_df, max_terms) if lexemes else []

    @staticmethod
    def _query_lexemes(cur, query: str) -> List[str]:
        """Distinct stemmed, stopword-free lexemes in query order."""
        words = " ".join(_TSQUERY_WORD_RE.findall(query))
        if not words:
            return []
        cur.execute(
            "SELECT lexeme FROM unnest(to_tsvector('english', %s)) ORDER BY positions[1]",
            (words,),
        )
        return [row[0] for row in cur.fetchall()]

    def _doc_freq(self, cur) -> Dict[str, float]:
        """Share of judgment chunks containing each of the most common
        lexemes, from the planner statistics ANALYZE already keeps for
        content_tsv (no extra table). Fetched once per retriever; empty if
        the table was never analysed, which only disables pruning."""
        if self._chunk_doc_freq is None:
            cur.execute(
                """
                SELECT most_common_elems::text::text[], most_common_elem_freqs
                FROM pg_stats
                WHERE tablename = 'judgment_chunks' AND attname = 'content_tsv'
                  AND schemaname = ANY(current_schemas(false))
                LIMIT 1
                """
            )
            row = cur.fetchone()
            elems, freqs = (row or (None, None))
            # freqs has three trailing entries (min, max, null frequency).
            self._chunk_doc_freq = dict(zip(elems, freqs)) if elems and freqs else {}
        return self._chunk_doc_freq

    _BM25_SQL = """
        WITH q AS (SELECT {tsquery} AS tsq)
        SELECT
            jc.content,
            j.petitioner,
            j.respondent,
            j.court,
            EXTRACT(YEAR FROM j.date_of_judgment) as year,
            jc.section,
            jc.chunk_id,
            jc.judgment_id,
            {rank_fn}(jc.content_tsv, q.tsq) as bm25_score
        FROM q, judgment_chunks jc
        JOIN judgments j ON j.id = jc.judgment_id
        WHERE jc.content_tsv @@ q.tsq AND NOT (jc.chunk_id = ANY(%s))
        ORDER BY bm25_score DESC
        LIMIT %s
    """

    def _bm25_query(self, cur, tsquery_arg: str, use_or: bool, k: int, skip_ids: List[int]):
        if use_or:
            # Lexemes are already stemmed and alphanumeric; 'simple' keeps
            # them as they are instead of stemming a second time.
            sql = self._BM25_SQL.format(tsquery="to_tsquery('simple', %s)", rank_fn="ts_rank_cd")
        else:
            sql = self._BM25_SQL.format(tsquery="plainto_tsquery('english', %s)", rank_fn="ts_rank")
        cur.execute(sql, (tsquery_arg, list(skip_ids), k))
        return cur.fetchall()

    def retrieve_bm25_candidates(
        self, query: str, candidate_k: int = 30
    ) -> List[Dict[str, Any]]:
        """Stage-1 retrieval (keyword leg): Postgres full-text search over
        judgment_chunks.content_tsv. Catches exact terms (statute numbers,
        party names, section numbers) that dense embeddings can blur past.
        See BM25_MODES for and / or / and_or (default)."""
        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()
        self.bm25_stats["calls"] += 1

        try:
            rows: List[Tuple] = []
            if self.bm25_mode in ("and", "and_or"):
                rows = self._bm25_query(cur, query, False, candidate_k, [])
            if self.bm25_mode == "or":
                terms = self._or_query_terms(cur, query)
                if terms:
                    rows = self._bm25_query(cur, " | ".join(terms), True, candidate_k, [])
            elif self.bm25_mode == "and_or" and len(rows) < candidate_k:
                lexemes = self._query_lexemes(cur, query)
                terms = []
                if len(lexemes) > _FILL_MAX_QUERY_LEXEMES:
                    self.bm25_stats["fill_skipped_long"] += 1
                elif lexemes:
                    terms = _select_or_terms(lexemes, self._doc_freq(cur), _FILL_MAX_DF, _FILL_MAX_TERMS)
                if terms:
                    self.bm25_stats["fills"] += 1
                    cur.execute(f"SET LOCAL statement_timeout = {int(_FILL_TIMEOUT_MS)}")
                    try:
                        rows += self._bm25_query(
                            cur, " | ".join(terms), True, candidate_k - len(rows), [r[6] for r in rows]
                        )
                    except psycopg2.Error as e:
                        if getattr(e, "pgcode", None) != _QUERY_CANCELED:
                            raise
                        # Too slow: keep the AND results rather than stall the chat.
                        self.bm25_stats["fill_timeouts"] += 1
                        conn.rollback()

            retrieved_chunks = []
            for row in rows:
                (
                    content,
                    petitioner,
                    respondent,
                    court,
                    year,
                    section,
                    chunk_id,
                    judgment_id,
                    bm25_score,
                ) = row

                case_name = self._format_case_name(petitioner, respondent)
                # chunk_id is a global DB row id, not a paragraph number within
                # this judgment — label it honestly rather than implying a
                # real ¶N citation the source text doesn't actually support.
                para_id = f"(chunk {chunk_id})"

                retrieved_chunks.append(
                    {
                        "doc_type": "judgment",
                        "chunk_id": chunk_id,
                        "judgment_id": judgment_id,
                        "text": content.strip(),
                        "case": case_name,
                        "court": court if court else "Unknown Court",
                        "year": int(year) if year else None,
                        "para": para_id,
                        "section": section,
                        "bm25_score": float(bm25_score),
                    }
                )

            return retrieved_chunks

        finally:
            cur.close()
            conn.close()

    # -- case lookup ----------------------------------------------------------

    def lookup_named_judgments(self, query: str) -> List[Tuple[int, str]]:
        """(judgment_id, how) for judgments the query names: reporter
        citations and "X v. Y" titles through the citation resolver (only
        reporter / exact_name resolutions), then "<Name> case" party names
        matched against judgment_aliases when they name few enough judgments.
        Empty when there's no graph manager or nothing is named."""
        if not (self.case_lookup and self.graph_manager):
            return []
        found: Dict[int, str] = {}
        strings = lookup_strings(query)
        if strings:
            for r in self.graph_manager.resolve_citations(strings):
                if r.get("method") in _LOOKUP_METHODS and r.get("judgment_id") is not None:
                    found.setdefault(r["judgment_id"], r["method"])
        norm = normalize_case_name(query)
        title_fallback = not found and " v " in f" {norm} "
        phrases = [] if found else named_case_phrases(query)
        if title_fallback or phrases:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            try:
                if title_fallback:
                    # The title regex misses party names with commas or "M/S."
                    # and the resolver refuses names shared by connected
                    # appeals, so also match every full case name contained in
                    # the normalised query (as the Graph page search does).
                    for jid in self._contained_case_names(cur, norm):
                        found.setdefault(jid, "title_alias")
                for candidates in ([] if found else phrases):
                    for jid in self._party_judgments(cur, candidates):
                        found.setdefault(jid, "party_name")
            finally:
                cur.close()
                conn.close()
        return list(found.items())[:_LOOKUP_MAX_JUDGMENTS]

    @staticmethod
    def _party_judgments(cur, candidates: List[str]) -> List[int]:
        """Judgments with the longest candidate party name that any alias has
        on either side ("<name> v ..." or "... v <name>"). None when that
        name is on more than _LOOKUP_MAX_JUDGMENTS judgments: a shorter run
        isn't tried then, since the user named the longer one."""
        for name in candidates:
            esc = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            cur.execute(
                "SELECT DISTINCT judgment_id FROM judgment_aliases "
                "WHERE kind = 'case_name' AND (alias_norm LIKE %s OR alias_norm LIKE %s) "
                "LIMIT %s",
                (f"{esc} v %", f"% v {esc}", _LOOKUP_MAX_JUDGMENTS + 1),
            )
            ids = sorted(row[0] for row in cur.fetchall())
            if ids:
                return ids if len(ids) <= _LOOKUP_MAX_JUDGMENTS else []
        return []

    @staticmethod
    def _contained_case_names(cur, norm_query: str) -> List[int]:
        """Judgments whose full normalised case name occurs in the normalised
        query, keeping only the longest names matched (so "ram v state" inside
        "sita ram v state of punjab" doesn't count). None when that names more
        than _LOOKUP_MAX_JUDGMENTS judgments."""
        cur.execute(
            "SELECT alias_norm, judgment_id FROM judgment_aliases "
            "WHERE kind = 'case_name' AND %s LIKE '%%' || ' ' || alias_norm || ' ' || '%%'",
            (f" {norm_query} ",),
        )
        hits = cur.fetchall()
        aliases = {a for a, _ in hits}
        maximal = {a for a in aliases if not any(a != b and f" {a} " in f" {b} " for b in aliases)}
        ids = sorted({j for a, j in hits if a in maximal})
        return ids if len(ids) <= _LOOKUP_MAX_JUDGMENTS else []

    def _pinned_lookup_chunks(self, query: str) -> List[Dict[str, Any]]:
        """For each judgment the query names, its chunk closest to the query,
        marked exact_match so reranking keeps it first."""
        named = self.lookup_named_judgments(query)
        if not named:
            return []
        how = dict(named)
        emb = self.get_query_embedding(query)
        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT ON (jc.judgment_id)
                    jc.content, j.petitioner, j.respondent, j.court,
                    EXTRACT(YEAR FROM j.date_of_judgment), jc.section, jc.chunk_id, jc.judgment_id,
                    1 - (je.embedding <=> %s::vector)
                FROM judgment_chunks jc
                JOIN judgment_embeddings je ON je.chunk_id = jc.chunk_id
                JOIN judgments j ON j.id = jc.judgment_id
                WHERE jc.judgment_id = ANY(%s)
                ORDER BY jc.judgment_id, je.embedding <=> %s::vector
                """,
                (emb, list(how), emb),
            )
            rows = {r[7]: r for r in cur.fetchall()}
        finally:
            cur.close()
            conn.close()
        pinned = []
        for jid, _ in named:  # resolver order: first named, first shown
            if jid not in rows:
                continue
            content, pet, res, court, year, section, chunk_id, _, sim = rows[jid]
            pinned.append({
                "doc_type": "judgment",
                "chunk_id": chunk_id,
                "judgment_id": jid,
                "text": (content or "").strip(),
                "case": self._format_case_name(pet, res),
                "court": court if court else "Unknown Court",
                "year": int(year) if year else None,
                "para": f"(chunk {chunk_id})",
                "section": section,
                "similarity": float(sim),
                "bm25_score": 0.0,
                # Above any real RRF score (<= ~0.04) but finite: chunks reach
                # API responses (debug), and JSON has no Infinity.
                "rrf_score": 1.0,
                "graph_centrality": 0.0,
                "exact_match": True,
                "lookup_method": how[jid],
            })
        return pinned

    def retrieve_judgment_candidates(
        self,
        query: str,
        candidate_k: int = 30,
        similarity_threshold: float = 0.2,
        rrf_k: int = 60,
        graph_boost: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Stage-1 retrieval for judgments only: fuse dense (pgvector cosine)
        and BM25 (full-text) judgment candidates via Reciprocal Rank Fusion.
        Optional graph_boost > 0 reweights RRF scores using PageRank centrality.
        Judgments the query names (citation, "X v. Y", "<Name> case") are
        looked up directly and pinned first, marked exact_match — see
        lookup_named_judgments.
        Statutes are retrieved independently via retrieve_statute_candidates —
        pooling them into one ranked list structurally disadvantaged statutes
        (they only ever earned one RRF "vote" vs. judgments' two), so the two
        document types are no longer fused together at all.
        """
        dense_chunks = self.retrieve_candidate_chunks(
            query, candidate_k=candidate_k, similarity_threshold=similarity_threshold
        )
        bm25_chunks = self.retrieve_bm25_candidates(query, candidate_k=candidate_k)

        fused: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, chunk in enumerate(dense_chunks):
            key = f"j_{chunk['chunk_id']}"
            fused[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, chunk in enumerate(bm25_chunks):
            key = f"j_{chunk['chunk_id']}"
            if key not in fused:
                chunk.setdefault("similarity", 0.0)
                fused[key] = chunk
            else:
                fused[key].setdefault("bm25_score", chunk["bm25_score"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        centrality_map = {}
        if graph_boost > 0.0 and self.graph_manager:
            if hasattr(self.graph_manager, "centrality_for"):
                centrality_map = self.graph_manager.centrality_for(
                    [c.get("judgment_id") for c in fused.values()]
                )
            else:
                centrality_map = self.graph_manager.get_centrality_scores()
            # Raw PageRank sums to 1 over the corpus (max ~0.005 on 46k
            # judgments), which made the boost a no-op; rescale to [0, 1] by
            # the corpus maximum so graph_boost is the most-cited judgment's lift.
            top = self._centrality_max()
            if top > 0:
                centrality_map = {j: s / top for j, s in centrality_map.items()}

        for key, score in rrf_scores.items():
            j_id = fused[key].get("judgment_id")
            c_score = centrality_map.get(j_id, 0.0) if j_id else 0.0
            final_rrf = score * (1.0 + graph_boost * c_score)

            fused[key]["rrf_score"] = final_rrf
            fused[key].setdefault("similarity", 0.0)
            fused[key].setdefault("bm25_score", 0.0)
            fused[key]["graph_centrality"] = c_score

        ranked = sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)
        pinned = self._pinned_lookup_chunks(query)
        if pinned:
            pinned_ids = {c["chunk_id"] for c in pinned}
            ranked = pinned + [c for c in ranked if c["chunk_id"] not in pinned_ids]
        return ranked[:candidate_k]

    def _format_case_name(
        self, petitioner: Optional[str], respondent: Optional[str]
    ) -> str:
        if not petitioner or petitioner == "Unknown":
            petitioner = "Unknown"
        if not respondent or respondent == "Unknown":
            respondent = "Unknown"

        petitioner = petitioner.strip().title()
        respondent = respondent.strip().title()

        max_length = 50
        if len(petitioner) > max_length:
            petitioner = petitioner[:max_length] + "..."
        if len(respondent) > max_length:
            respondent = respondent[:max_length] + "..."

        return f"{petitioner} v. {respondent}"

    def get_retrieval_stats(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        query_embedding = self.get_query_embedding(query)

        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(1 - (embedding <=> %s::vector)) as avg_similarity,
                    MAX(1 - (embedding <=> %s::vector)) as max_similarity,
                    MIN(1 - (embedding <=> %s::vector)) as min_similarity
                FROM judgment_embeddings
            """,
                (query_embedding, query_embedding, query_embedding),
            )

            stats = cur.fetchone()

            cur.execute(
                """
                SELECT 
                    jc.section,
                    COUNT(*) as count,
                    AVG(1 - (je.embedding <=> %s::vector)) as avg_similarity
                FROM judgment_embeddings je
                JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                GROUP BY jc.section
                ORDER BY avg_similarity DESC
            """,
                (query_embedding,),
            )

            section_stats = cur.fetchall()

            return {
                "total_chunks": stats[0] if stats else 0,
                "avg_similarity": float(stats[1]) if stats and stats[1] else 0,
                "max_similarity": float(stats[2]) if stats and stats[2] else 0,
                "min_similarity": float(stats[3]) if stats and stats[3] else 0,
                "section_distribution": [
                    {
                        "section": row[0],
                        "count": row[1],
                        "avg_similarity": float(row[2]) if row[2] else 0,
                    }
                    for row in section_stats
                ],
            }

        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    retriever = LegalRetriever()
    query = "education Bill of Karnataka"

    print(f"Query: {query}")
    print("=" * 50)

    results = retriever.retrieve_relevant_chunks(query, top_k=3)

    for i, chunk in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Case: {chunk['case']}")
        print(f"Court: {chunk['court']} ({chunk['year']})")
        print(f"Section: {chunk['section']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"Content: {chunk['text'][:200]}...")

    print("\n" + "=" * 50)
    print("Retrieval Statistics:")
    stats = retriever.get_retrieval_stats(query)
    print(f"Total chunks in DB: {stats['total_chunks']}")
    print(f"Average similarity: {stats['avg_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")
    print("\nSection distribution:")
    for section in stats["section_distribution"]:
        print(
            f"  {section['section']}: {section['count']} chunks (avg sim: {section['avg_similarity']:.4f})"
        )
