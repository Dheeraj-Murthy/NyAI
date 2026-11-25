#!/usr/bin/env python3
"""Build a masked-citation gold set from the citation graph.

When judgment A cites judgment B, A's judge has already labelled B as the
authority for that passage. Take the passage around the citation, remove the
citation and B's name, and ask retrieval to find B. No human labelling and no
LLM: the ground truth is the judge's own citation.

Unlike the synthetic gold set, the query is a DIFFERENT judgment's wording,
not the target's text, so it is not paraphrase lookup. Two guards keep it
that way:
  - A itself is excluded when scoring (its own chunk contains the passage
    verbatim); judgment_eval.py over-fetches and drops it.
  - Passages that quote B at length (judges often do) are dropped: share of
    the masked query's 8-word shingles found in B's text above
    --max-quote-overlap (default 0.2).

What is masked in the passage: every reporter citation and "X v. Y" case
title (not only B's, so a neighbouring citation can't give it away),
"supra", and the identifying name words of every relevant case.

Relevant cases: every corpus judgment cited inside the passage window, not
only the edge that was sampled; all of them were masked, so all are fair
answers. The sampled one is recorded as primary_target_id.

Edges used: target resolved by reporter citation or exact case name (the
high-precision resolver methods), target != source, target not decided
after source, and the citation occurs in the body of A (after JUDGMENT:),
not in the JUDIS header, whose CITATOR INFO lists LATER cases that cite A.

Sampling: random with --seed, at most one record per citing judgment and
--max-per-target per cited judgment, so landmark cases cited thousands of
times don't dominate. Each record keeps the target's in-degree so results
can be split into rarely- vs heavily-cited targets.

Usage:
    cd NLPv2/backend
    python benchmark/masked_citation_gold.py --n 500 --seed 1 --version v1
    python benchmark/judgment_eval.py --gold benchmark/data/masked_citation_gold_v1.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_benchmark_dir)
sys.path.insert(0, _backend_dir)

from graph.citation_resolver import (  # noqa: E402
    CASE_TITLE_PATTERN,
    REPORTER_PATTERNS,
    _year_of,
    case_name_alias,
    parse_reporter_citations,
)
from commentary_gold import distinctive_name_words  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(_benchmark_dir, "data")
HIGH_PRECISION_METHODS = ("reporter", "exact_name")
LONG_MAX_WORDS = 150   # passage-style query: about one paragraph
SHORT_WORDS = 40       # short query: the words around the citation
MIN_QUERY_WORDS = 20   # after masking
DEFAULT_MAX_QUOTE_OVERLAP = 0.2
SHINGLE = 8

_BODY_START_RE = re.compile(r"^\s*JUDGMENT\s*:", re.M)
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_CASE_TITLE_RE = re.compile(CASE_TITLE_PATTERN)
_LOOSE_REPORTER_RE = re.compile(
    r"[\[(]?\b(?:1[89]\d{2}|20\d{2})\b[\])]?\s*(?:\(?\d{1,2}\)?\s*)?(?:Supp\.?\s*)?"
    r"(?:\(?\d{1,2}\)?\s*)?(?:A\.?\s?I\.?\s?R\.?|S\.?\s?C\.?\s?[CR]\.?)"
    r"(?:\s*\(?\d{1,2}\)?)?(?:\s*[A-Z][a-z]{0,4}\.?)?\s*\d{1,5}"
    r"|\b(?:A\.?\s?I\.?\s?R\.?|S\.?\s?C\.?\s?[CR]\.?)\b"
)
_SUPRA_RE = re.compile(r"\(?\bsupra\b\)?", re.I)


# ---------------------------------------------------------------------------
# Text handling (pure, unit-tested offline)
# ---------------------------------------------------------------------------


def body_start(text: str) -> int:
    """Offset where the judgment body starts in JUDIS text (the JUDGMENT:
    heading), or 0 when there is no such heading."""
    m = _BODY_START_RE.search(text)
    return m.end() if m else 0


def find_occurrence(text: str, needle: str, start: int = 0) -> Optional[Tuple[int, int]]:
    """(start, end) of the first occurrence of needle at or after start,
    tolerating case and whitespace differences (pdftotext line breaks)."""
    pattern = r"\s+".join(re.escape(part) for part in needle.split())
    if not pattern:
        return None
    m = re.compile(pattern, re.I).search(text, start)
    return (m.start(), m.end()) if m else None


def passage_window(text: str, span: Tuple[int, int], max_words: int) -> Tuple[int, int]:
    """Character span of the paragraph holding `span`, trimmed to at most
    max_words words centred on the citation."""
    para_start = 0
    for m in _PARAGRAPH_BREAK_RE.finditer(text, 0, span[0]):
        para_start = m.end()
    nxt = _PARAGRAPH_BREAK_RE.search(text, span[1])
    para_end = nxt.start() if nxt else len(text)

    tokens = [m.span() for m in re.finditer(r"\S+", text[para_start:para_end])]
    tokens = [(a + para_start, b + para_start) for a, b in tokens]
    if len(tokens) <= max_words:
        return (para_start, para_end) if tokens else span
    first = next(i for i, (a, b) in enumerate(tokens) if b > span[0])
    last = max(i for i, (a, b) in enumerate(tokens) if a < span[1])
    room = max(0, max_words - (last - first + 1))
    lo = max(0, first - room // 2)
    hi = min(len(tokens) - 1, lo + max_words - 1)
    lo = max(0, hi - max_words + 1)
    return (tokens[lo][0], tokens[hi][1])


def mask_passage(passage: str, name_words: Set[str]) -> str:
    """Remove every citation, case title, 'supra' and the given name words."""
    out = passage
    for pattern in REPORTER_PATTERNS:
        out = pattern.sub(" ", out)
    out = _LOOSE_REPORTER_RE.sub(" ", out)
    out = _CASE_TITLE_RE.sub(" ", out)
    out = _SUPRA_RE.sub(" ", out)
    if name_words:
        names = "|".join(re.escape(w) for w in sorted(name_words, key=len, reverse=True))
        out = re.sub(r"\b(?:" + names + r")(?:'s|’s)?\b", " ", out, flags=re.I)
    out = re.sub(r"\(\s*[,;.]*\s*\)|\[\s*[,;.]*\s*\]", " ", out)  # emptied brackets
    out = re.sub(r"(?:\s*[,;:]\s*){2,}", ", ", out)                # ", , ;" runs
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)          # "in , ." -> "in,."
    out = re.sub(r"[,;:]+(?=[.,;:])", "", out)        # "in,." -> "in."
    return out.strip(" ,;:")


def leaks(query: str, name_words: Set[str]) -> Optional[str]:
    if parse_reporter_citations(query):
        return "citation survived masking"
    left = name_words & set(re.findall(r"[a-z][a-z0-9']+", query.lower()))
    if left:
        return f"name word survived masking ({', '.join(sorted(left))})"
    return None


def _shingles(text: str, n: int = SHINGLE) -> Set[Tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def quote_overlap(query: str, target_text: str) -> float:
    """Share of the query's 8-word shingles that also occur in the target:
    high when the passage quotes the cited judgment."""
    q = _shingles(query)
    if not q:
        return 0.0
    return len(q & _shingles(target_text)) / len(q)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------


def build_record(
    edge: Dict[str, Any],
    source_text: str,
    target_texts: Dict[int, str],
    source_edges: Sequence[Dict[str, Any]],
    meta: Dict[int, Dict[str, Any]],
    max_quote_overlap: float = DEFAULT_MAX_QUOTE_OVERLAP,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(record, None) or (None, why it was skipped). `source_edges` are all
    resolved edges of the citing judgment; the ones whose citation falls in
    the passage window join the relevant set."""
    src, tgt = edge["source_judgment_id"], edge["target_judgment_id"]
    occ = find_occurrence(source_text, edge["cited_text"], body_start(source_text))
    if occ is None:
        return None, "citation not found in the judgment body"

    windows = {
        "long": passage_window(source_text, occ, LONG_MAX_WORDS),
        "short": passage_window(source_text, occ, SHORT_WORDS),
    }
    lo, hi = windows["long"]
    raw_long = source_text[lo:hi]

    relevant: Set[int] = {tgt}
    for other in source_edges:
        oid = other["target_judgment_id"]
        if oid is None or oid == src or oid in relevant:
            continue
        if other.get("resolution_method") not in HIGH_PRECISION_METHODS:
            continue
        o_meta, s_meta = meta.get(oid, {}), meta.get(src, {})
        if o_meta.get("year") and s_meta.get("year") and o_meta["year"] > s_meta["year"]:
            continue
        if find_occurrence(raw_long, other["cited_text"]):
            relevant.add(oid)

    name_words: Set[str] = set()
    for jid in relevant:
        name_words |= distinctive_name_words(meta.get(jid, {}).get("case_name", ""))

    queries = {}
    for kind, (a, b) in windows.items():
        queries[kind] = mask_passage(source_text[a:b], name_words)
        leak = leaks(queries[kind], name_words)
        if leak:
            return None, leak
    if len(queries["long"].split()) < MIN_QUERY_WORDS:
        return None, "passage too short after masking"

    overlap = max((quote_overlap(queries["long"], target_texts.get(j, "")) for j in relevant), default=0.0)
    if overlap > max_quote_overlap:
        return None, "passage quotes the cited judgment"

    return {
        "id": f"mc-{src}-{tgt}",
        "query": queries["long"],
        "query_short": queries["short"],
        "relevant_judgment_ids": sorted(relevant),
        "relevant_case_names": {str(j): meta.get(j, {}).get("case_name", "") for j in sorted(relevant)},
        "primary_target_id": tgt,
        "source_judgment_id": src,
        "source_case_name": meta.get(src, {}).get("case_name", ""),
        "exclude_judgment_ids": [src],
        "cited_text": edge["cited_text"],
        "resolution_method": edge["resolution_method"],
        "target_in_degree": meta.get(tgt, {}).get("in_degree", 0),
        "source_year": meta.get(src, {}).get("year"),
        "target_year": meta.get(tgt, {}).get("year"),
        "quote_overlap": round(overlap, 3),
    }, None


def eligible_edges(edges: List[Dict[str, Any]], meta: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for e in edges:
        src, tgt = e["source_judgment_id"], e["target_judgment_id"]
        if tgt is None or tgt == src or e.get("resolution_method") not in HIGH_PRECISION_METHODS:
            continue
        sy, ty = meta.get(src, {}).get("year"), meta.get(tgt, {}).get("year")
        if sy and ty and ty > sy:
            continue  # a case cannot cite a later one: header CITATOR INFO or a bad match
        out.append(e)
    return out


def sample_order(edges: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Deterministic shuffle: sorted by a stable key first, so the same
    seed gives the same order whatever order the DB returned rows in."""
    ordered = sorted(edges, key=lambda e: (e["source_judgment_id"], e["target_judgment_id"], e["cited_text"]))
    random.Random(seed).shuffle(ordered)
    return ordered


def build_gold_set(
    edges: List[Dict[str, Any]],
    meta: Dict[int, Dict[str, Any]],
    fetch_texts,
    n: int,
    seed: int,
    max_per_target: int = 2,
    max_quote_overlap: float = DEFAULT_MAX_QUOTE_OVERLAP,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """fetch_texts(ids) -> {id: judgment_text}; injected so this runs
    against the DB or a fixture."""
    by_source: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        by_source[e["source_judgment_id"]].append(e)

    records: List[Dict[str, Any]] = []
    skipped: Counter = Counter()
    used_sources: Set[int] = set()
    per_target: Counter = Counter()
    for edge in sample_order(eligible_edges(edges, meta), seed):
        if len(records) >= n:
            break
        src, tgt = edge["source_judgment_id"], edge["target_judgment_id"]
        if src in used_sources:
            continue
        if per_target[tgt] >= max_per_target:
            skipped["target already used max times"] += 1
            continue
        source_edges = by_source[src]
        candidate_ids = {src} | {e["target_judgment_id"] for e in source_edges if e["target_judgment_id"]}
        texts = fetch_texts(sorted(candidate_ids))
        record, reason = build_record(
            edge, texts.get(src, ""), texts, source_edges, meta, max_quote_overlap
        )
        if record is None:
            skipped[reason.split(" (")[0]] += 1
            continue
        records.append(record)
        used_sources.add(src)
        per_target[tgt] += 1
    return records, skipped


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


def _default_db_connection_string() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def load_from_db(conn) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, petitioner, respondent, date_of_judgment FROM judgments")
        meta = {
            r[0]: {"case_name": case_name_alias(r[1], r[2]) or "", "year": _year_of(r[3]), "in_degree": 0}
            for r in cur.fetchall()
        }
        cur.execute(
            "SELECT source_judgment_id, target_judgment_id, cited_text, resolution_method "
            "FROM citation_edges WHERE target_judgment_id IS NOT NULL"
        )
        edges = [
            {"source_judgment_id": r[0], "target_judgment_id": r[1], "cited_text": r[2], "resolution_method": r[3]}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
    for src, tgt in {(e["source_judgment_id"], e["target_judgment_id"]) for e in edges}:
        if tgt in meta and tgt != src:
            meta[tgt]["in_degree"] += 1
    return edges, meta


def make_text_fetcher(conn):
    cache: Dict[int, str] = {}

    def fetch(ids: List[int]) -> Dict[int, str]:
        missing = [i for i in ids if i not in cache]
        if missing:
            cur = conn.cursor()
            cur.execute("SELECT id, judgment_text FROM judgments WHERE id = ANY(%s)", (missing,))
            for jid, text in cur.fetchall():
                cache[jid] = text or ""
            cur.close()
        return {i: cache.get(i, "") for i in ids}

    return fetch


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a masked-citation gold set from citation_edges.")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--version", required=True, help="e.g. v1 -> masked_citation_gold_v1.json")
    parser.add_argument("--max-per-target", type=int, default=2)
    parser.add_argument("--max-quote-overlap", type=float, default=DEFAULT_MAX_QUOTE_OVERLAP)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"masked_citation_gold_{args.version}.json")
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists; gold sets are frozen once written. Use a new --version.")

    import psycopg2

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        edges, meta = load_from_db(conn)
        n_eligible = len(eligible_edges(edges, meta))
        print(f"{len(edges)} resolved edges, {n_eligible} eligible ({'/'.join(HIGH_PRECISION_METHODS)}, not later-dated)")
        records, skipped = build_gold_set(
            edges, meta, make_text_fetcher(conn), args.n, args.seed,
            args.max_per_target, args.max_quote_overlap,
        )
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM judgments")
        n_judgments = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()

    in_deg = [r["target_in_degree"] for r in records]
    gold = {
        "kind": "masked_citation",
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n": len(records),
        "config": {
            "methods": list(HIGH_PRECISION_METHODS),
            "long_max_words": LONG_MAX_WORDS,
            "short_words": SHORT_WORDS,
            "min_query_words": MIN_QUERY_WORDS,
            "max_quote_overlap": args.max_quote_overlap,
            "max_per_target": args.max_per_target,
        },
        "corpus_judgment_count": n_judgments,
        "corpus_edge_count": len(edges),
        "skipped": dict(skipped),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records -> {out}")
    if len(records) < args.n:
        print(f"WARNING: asked for {args.n}, eligible edges ran out at {len(records)}")
    print("Skipped:", dict(skipped))
    if in_deg:
        multi = sum(1 for r in records if len(r["relevant_judgment_ids"]) > 1)
        print(f"Target in-degree median {sorted(in_deg)[len(in_deg) // 2]}, max {max(in_deg)}; "
              f"{multi} records with >1 relevant case")


if __name__ == "__main__":
    main()
