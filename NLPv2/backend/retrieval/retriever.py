import os
import re
import psycopg2
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer


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


class LegalRetriever:
    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        graph_manager: Optional[Any] = None,
    ):
        self.db_connection_string = db_connection_string or _default_db_connection_string()
        self.embedding_model = None
        self.graph_manager = graph_manager
        self._load_embedding_model()

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

    def retrieve_bm25_candidates(
        self, query: str, candidate_k: int = 30
    ) -> List[Dict[str, Any]]:
        """Stage-1 retrieval (keyword leg): Postgres full-text search over
        judgment_chunks.content_tsv. Catches exact terms (statute numbers,
        party names, section numbers) that dense embeddings can blur past."""
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
                    ts_rank(jc.content_tsv, plainto_tsquery('english', %s)) as bm25_score
                FROM judgment_chunks jc
                JOIN judgments j ON j.id = jc.judgment_id
                WHERE jc.content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY bm25_score DESC
                LIMIT %s
            """,
                (query, query, candidate_k),
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

        for key, score in rrf_scores.items():
            j_id = fused[key].get("judgment_id")
            c_score = centrality_map.get(j_id, 0.0) if j_id else 0.0
            final_rrf = score * (1.0 + graph_boost * c_score)

            fused[key]["rrf_score"] = final_rrf
            fused[key].setdefault("similarity", 0.0)
            fused[key].setdefault("bm25_score", 0.0)
            fused[key]["graph_centrality"] = c_score

        ranked = sorted(fused.values(), key=lambda c: c["rrf_score"], reverse=True)
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
