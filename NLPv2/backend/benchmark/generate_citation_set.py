#!/usr/bin/env python3
"""Sample cases for the citation benchmark (citation_eval.py).

Each sampled case is a judgment in the corpus that cites at least
--min-gold other judgments that are ALSO in the corpus. Its gold set is
those cited judgments, taken from citation_edges (built by
ingestion/build_citation_edges.py). The query is the judgment's own
facts/issues text with every citation written in it redacted, so the
system has to find the precedents from the dispute itself rather than by
reading the answer out of the query.

Unlike generate_gold_set.py, no LLM is involved: the same --seed on the
same corpus and citation graph gives the same case set, query text and
gold ids every time. That avoids the same-seed-different-gold-set problem
REPORT.md's 2026-10-07 update ran into. The query text itself is saved, so
citation_eval.py never re-derives it.

Gold labels are only as good as citation resolution. By default only
reporter / exact_name / fuzzy edges count; pass --gold-methods
reporter,exact_name for a smaller, higher-precision gold set (fuzzy matches
are the least reliable — see ingestion/outputs/edge_review_sample.csv).
Citations that never resolved (cases outside the corpus, High Court and
foreign cases) can't be scored either way; each record keeps
n_citations_total vs n_gold so the report can show how much of what was
actually cited is measurable at all.

Usage:
    cd NLPv2/backend
    python benchmark/generate_citation_set.py --n 100 --seed 42
    python benchmark/generate_citation_set.py --n 100 --gold-methods reporter,exact_name
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2

# Add backend/ to the path so the dotted `graph.*` subpackage import below
# resolves whether this script is run directly or as a module — same shim
# as generate_gold_set.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from graph.citation_resolver import CASE_TITLE_PATTERN, REPORTER_PATTERNS  # noqa: E402

from generate_gold_set import _default_db_connection_string, _is_garbled  # noqa: E402

RESOLVED_METHODS = ("reporter", "exact_name", "fuzzy")
DEFAULT_QUERY_SECTIONS = ("facts", "issues")
DEFAULT_MIN_GOLD = 3
MAX_QUERY_CHUNKS = 4  # per case; each becomes one retrieval query, fused by citation_eval.py
QUERY_MAX_CHARS = 2000  # ~BGE's 512-token input limit; anything past it is truncated by the encoder anyway
MIN_QUERY_CHARS = 300  # after redaction — less than this is too little dispute to search from

REDACTION = "[CITATION]"
_TITLE_RE = re.compile(CASE_TITLE_PATTERN)

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "citation_set.json")


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def redact_citations(text: str) -> Tuple[str, int]:
    """Replace every reporter citation and "X v. Y" case title in `text`
    with a placeholder, using the same patterns citation extraction uses —
    so anything build_citation_edges.py could have turned into a gold edge
    is also hidden from the query. Returns (redacted text, redactions)."""
    count = 0
    for pattern in list(REPORTER_PATTERNS) + [_TITLE_RE]:
        text, n = pattern.subn(REDACTION, text)
        count += n
    return text, count


def _corpus_counts(cur) -> Dict[str, int]:
    cur.execute("SELECT count(*) FROM judgments")
    judgment_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM citation_edges")
    edge_count = cur.fetchone()[0]
    return {"judgment_count": judgment_count, "edge_count": edge_count}


def _citing_judgments(cur, gold_methods: Sequence[str], min_gold: int) -> List[Dict[str, Any]]:
    """Every judgment with at least min_gold distinct resolved citation
    targets, with its gold target ids and its total citation count."""
    cur.execute(
        """
        SELECT e.source_judgment_id,
               array_agg(DISTINCT e.target_judgment_id)
                   FILTER (WHERE e.target_judgment_id IS NOT NULL
                           AND e.target_judgment_id <> e.source_judgment_id
                           AND e.resolution_method = ANY(%s)) AS gold,
               count(*) AS n_total
        FROM citation_edges e
        GROUP BY e.source_judgment_id
        HAVING count(DISTINCT e.target_judgment_id)
                   FILTER (WHERE e.target_judgment_id IS NOT NULL
                           AND e.target_judgment_id <> e.source_judgment_id
                           AND e.resolution_method = ANY(%s)) >= %s
        ORDER BY e.source_judgment_id
        """,
        (list(gold_methods), list(gold_methods), min_gold),
    )
    return [{"judgment_id": r[0], "gold": sorted(r[1]), "n_total": r[2]} for r in cur.fetchall()]


def _judgment_info(cur, judgment_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    cur.execute(
        "SELECT id, petitioner, respondent, date_of_judgment FROM judgments WHERE id = ANY(%s)",
        (judgment_ids,),
    )
    return {
        r[0]: {"case_name": f"{(r[1] or 'Unknown').strip()} v. {(r[2] or 'Unknown').strip()}", "date": r[3]}
        for r in cur.fetchall()
    }


def _query_chunks(cur, judgment_id: int, sections: Sequence[str]) -> List[Dict[str, Any]]:
    cur.execute(
        "SELECT chunk_id, section, content FROM judgment_chunks "
        "WHERE judgment_id = %s AND section = ANY(%s) ORDER BY chunk_id",
        (judgment_id, list(sections)),
    )
    return [{"chunk_id": r[0], "section": r[1], "content": r[2]} for r in cur.fetchall()]


def _drop_future_targets(
    source_date: Optional[date], gold: List[int], info: Dict[int, Dict[str, Any]]
) -> Tuple[List[int], List[int]]:
    """A judgment can't cite one decided after it, so a gold target dated
    later than its source is a resolution error (usually a fuzzy name match
    to a different, later case with similar parties). Returns (kept, dropped)."""
    if source_date is None:
        return gold, []
    kept, dropped = [], []
    for jid in gold:
        target_date = info.get(jid, {}).get("date")
        (dropped if target_date is not None and target_date > source_date else kept).append(jid)
    return kept, dropped


def build_case_record(
    source: Dict[str, Any],
    source_info: Dict[str, Any],
    gold: List[int],
    dropped_future: List[int],
    chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """None when the case has no usable (non-garbled, long enough after
    redaction) query text."""
    queries = []
    query_chunk_ids = []
    query_chunk_hashes = []
    redactions = 0
    for chunk in chunks:
        if len(queries) >= MAX_QUERY_CHUNKS:
            break
        if _is_garbled(chunk["content"]):
            continue
        redacted, n = redact_citations(chunk["content"])
        redacted = redacted.strip()[:QUERY_MAX_CHARS]
        if len(redacted) < MIN_QUERY_CHARS // 2:
            continue
        queries.append(redacted)
        query_chunk_ids.append(chunk["chunk_id"])
        query_chunk_hashes.append(_content_hash(chunk["content"]))
        redactions += n
    if sum(len(q) for q in queries) < MIN_QUERY_CHARS:
        return None

    source_date = source_info.get("date")
    return {
        "source_judgment_id": source["judgment_id"],
        "case_name": source_info["case_name"],
        "date_of_judgment": source_date.isoformat() if source_date else None,
        "queries": queries,
        "query_chunk_ids": query_chunk_ids,
        "query_chunk_hashes": query_chunk_hashes,
        "redactions": redactions,
        "gold_judgment_ids": gold,
        # Gold targets dropped for being dated after the source (resolution
        # errors). Kept for inspection; never scored.
        "dropped_future_targets": dropped_future,
        # All citation edges from this judgment, resolved or not, vs. the
        # ones that made it into gold: how much of the real citation list
        # this case can score at all.
        "n_citations_total": source["n_total"],
        "n_gold": len(gold),
    }


def build_citation_set(
    n: int,
    seed: int,
    gold_methods: Sequence[str] = RESOLVED_METHODS,
    min_gold: int = DEFAULT_MIN_GOLD,
    query_sections: Sequence[str] = DEFAULT_QUERY_SECTIONS,
) -> Dict[str, Any]:
    conn = psycopg2.connect(_default_db_connection_string())
    try:
        cur = conn.cursor()
        try:
            counts = _corpus_counts(cur)
            candidates = _citing_judgments(cur, gold_methods, min_gold)
            print(f"{len(candidates)} judgments cite >= {min_gold} in-corpus judgments ({', '.join(gold_methods)}).")
            random.Random(seed).shuffle(candidates)

            records: List[Dict[str, Any]] = []
            skipped_no_text = 0
            skipped_future = 0
            for source in candidates:
                if len(records) >= n:
                    break
                info = _judgment_info(cur, [source["judgment_id"]] + source["gold"])
                source_info = info.get(source["judgment_id"])
                if source_info is None:
                    continue
                gold, dropped = _drop_future_targets(source_info["date"], source["gold"], info)
                if len(gold) < min_gold:
                    skipped_future += 1
                    continue
                record = build_case_record(
                    source, source_info, gold, dropped, _query_chunks(cur, source["judgment_id"], query_sections)
                )
                if record is None:
                    skipped_no_text += 1
                    continue
                records.append(record)
                print(
                    f"[{len(records)}/{n}] judgment {record['source_judgment_id']} "
                    f"({record['date_of_judgment']}): {record['n_gold']} gold / "
                    f"{record['n_citations_total']} cited, {record['redactions']} redactions — {record['case_name']}"
                )
        finally:
            cur.close()
    finally:
        conn.close()

    print(
        f"\n{len(records)} accepted, {skipped_no_text} skipped for no usable "
        f"{'/'.join(query_sections)} text, {skipped_future} below --min-gold after dropping future-dated targets"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gold_methods": list(gold_methods),
        "min_gold": min_gold,
        "query_sections": list(query_sections),
        "corpus_judgment_count": counts["judgment_count"],
        "corpus_edge_count": counts["edge_count"],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample cited-precedent cases from the citation graph for citation_eval.py."
    )
    parser.add_argument("--n", type=int, default=100, help="Number of cases (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case sampling (default: 42)")
    parser.add_argument(
        "--gold-methods", type=str, default=",".join(RESOLVED_METHODS),
        help="Comma-separated resolution methods whose edges count as gold "
        f"(default: {','.join(RESOLVED_METHODS)}; reporter,exact_name for higher precision)",
    )
    parser.add_argument(
        "--min-gold", type=int, default=DEFAULT_MIN_GOLD,
        help=f"Min distinct in-corpus judgments a case must cite (default: {DEFAULT_MIN_GOLD})",
    )
    parser.add_argument(
        "--query-sections", type=str, default=",".join(DEFAULT_QUERY_SECTIONS),
        help="Comma-separated sections the query text is taken from (default: facts,issues). "
        "ratio/arguments are where citations are discussed, so they leak more even after redaction.",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args()

    gold_methods = [m.strip() for m in args.gold_methods.split(",") if m.strip()]
    unknown = set(gold_methods) - set(RESOLVED_METHODS)
    if unknown:
        parser.error(f"--gold-methods: unknown or unresolved method(s) {sorted(unknown)}")

    citation_set = build_citation_set(
        args.n, args.seed, gold_methods, args.min_gold,
        [s.strip() for s in args.query_sections.split(",") if s.strip()],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(citation_set, f, indent=2)
    print(f"\nWrote {len(citation_set['records'])} cases to {args.output}")


if __name__ == "__main__":
    main()
