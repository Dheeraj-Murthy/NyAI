#!/usr/bin/env python3
"""Retrieval-quality benchmark.

For each gold query, runs the SAME two-stage retrieval the real pipeline
runs for judgments — LegalRetriever.retrieve_judgment_candidates() then
CrossEncoderReranker.rerank() — and checks whether the chunk the query was
generated from (source_chunk_id) comes back, and at what rank, at each
stage. Judgment chunks only; statutes have a separate table/id space and
are out of scope for this benchmark.

Defaults mirror rag_pipeline.py's ACTUAL hardcoded retrieval config
(stage1_k=30, stage1_threshold=0.2, stage2_k=top_k=8) — not main.py's CLI
--threshold default of 0.3, which is accepted but never passed into
retrieval (rag_pipeline.py stores it only for metrics/debug output; the
real stage-1 call always uses the hardcoded 0.2). Using 0.3 here would
silently benchmark a different configuration than what the live pipeline
actually runs.

Before scoring, every gold record is checked against the live DB
(validate_gold_set): a SERIAL chunk_id is not stable across a schema reset
+ full re-ingest, so a gold set generated before one would otherwise be
scored against wrong or missing chunks with no error at all. Stale records
are excluded from every metric and reported separately rather than
silently skewing the numbers.

Reports both CHUNK-level metrics (did we get back the exact source chunk)
and JUDGMENT-level metrics (did we get back the right case at all, in any
chunk) — a query whose retrieved top-k includes the correct judgment but a
different paragraph is a much more benign failure than missing the case
entirely, and chunk-only recall can't tell the two apart. Also reports a
per-section breakdown, since facts/issues (extractive) and arguments/ratio
(interpretive) are inherently different difficulty.

No LLM is loaded — LegalRetriever only loads the BGE encoder and
CrossEncoderReranker only loads the MiniLM cross-encoder, so this is
inherently CPU-safe.

Usage:
    cd NLPv2/backend
    python benchmark/retrieval_eval.py --gold benchmark/data/gold_set.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

# Add backend/ to the path so the dotted `retrieval.*` subpackage import and
# the flat `tracking` import below resolve whether this script is run
# directly or as a module — same shim as backend/tests/smoke_test_citation_graph.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

import psycopg2  # noqa: E402

from retrieval.retriever import BM25_MODES, LegalRetriever  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
import tracking  # noqa: E402

from metrics import aggregate, rank_of  # noqa: E402

DEFAULT_K_VALUES = (5, 8, 10, 30)

# A gold record with crowd_size >= this (see generate_gold_set.py's
# near-duplicate detection) has at least this many OTHER chunks that are
# near-identical to its source chunk — the question has no single
# discriminating answer even in principle, which is a gold-set artifact
# (formulaic disposition text repeating across unrelated judgments), not a
# retrieval weakness. Default 1 means "any near-duplicate at all" counts.
DEFAULT_CROWD_THRESHOLD = 1


def _acceptable_chunk_ids(gold: Dict[str, Any]) -> List[Any]:
    """The scoring equivalence class for a gold record's chunk-level
    target: the literal source chunk plus any near-duplicate recorded at
    gold-set generation time. Falls back to just the source chunk for a
    gold set generated before acceptable_chunk_ids existed."""
    return gold.get("acceptable_chunk_ids") or [gold["source_chunk_id"]]


def _default_db_connection_string() -> str:
    """Build the default DSN from env vars — same convention as
    retrieval/retriever.py's private helper (each module builds its own
    connection; there's no shared db.py in this repo)."""
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_gold_set(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (records, metadata). Accepts both the current
    {"records": [...], "generated_at": ..., ...} shape and a bare list, for
    a gold set generated before generate_gold_set.py added metadata."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data["records"], {k: v for k, v in data.items() if k != "records"}


def validate_gold_set(
    records: List[Dict[str, Any]], conn
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Checks each gold record's source_chunk_id still points at the same
    judgment/section/content it was generated from. A schema reset + full
    re-ingest reassigns every chunk_id from a fresh SERIAL sequence, so
    without this check a stale gold set would silently score against wrong
    or missing chunks instead of failing loudly. Returns (valid, stale)."""
    chunk_ids = [r["source_chunk_id"] for r in records]
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, judgment_id, section, content FROM judgment_chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    live = {row[0]: {"judgment_id": row[1], "section": row[2], "content": row[3]} for row in cur.fetchall()}
    cur.close()

    valid: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []
    for r in records:
        row = live.get(r["source_chunk_id"])
        if row is None:
            stale.append({**r, "stale_reason": "chunk_id no longer exists"})
        elif row["judgment_id"] != r["source_judgment_id"] or row["section"] != r["section"]:
            stale.append({**r, "stale_reason": "chunk_id now belongs to a different judgment/section"})
        elif r.get("content_hash") and _content_hash(row["content"]) != r["content_hash"]:
            stale.append({**r, "stale_reason": "chunk content changed since gold set generation"})
        else:
            valid.append(r)
    return valid, stale


def run_two_stage_retrieval(
    retriever: LegalRetriever,
    reranker: CrossEncoderReranker,
    query: str,
    candidate_k: int,
    threshold: float,
    top_k: int,
    graph_boost: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Same two calls rag_pipeline.query() makes for judgments: hybrid
    recall, then cross-encoder rerank."""
    stage1 = retriever.retrieve_judgment_candidates(
        query=query,
        candidate_k=candidate_k,
        similarity_threshold=threshold,
        graph_boost=graph_boost,
    )
    stage2 = reranker.rerank_keep_pinned(query, list(stage1), top_n=top_k)
    return stage1, stage2


def evaluate_query(
    gold: Dict[str, Any],
    stage1_chunk_ids: List[Any],
    stage2_chunk_ids: List[Any],
    stage1_judgment_ids: List[Any],
    stage2_judgment_ids: List[Any],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    target_chunk = gold["source_chunk_id"]  # strict: literal source chunk only
    target_chunk_set = _acceptable_chunk_ids(gold)  # multi-label: source + near-duplicates
    target_judgment = gold["source_judgment_id"]
    result: Dict[str, Any] = {
        "query": gold["query"],
        "source_chunk_id": target_chunk,
        "acceptable_chunk_ids": target_chunk_set,
        "crowd_size": gold.get("crowd_size", len(target_chunk_set) - 1),
        "source_judgment_id": target_judgment,
        "section": gold["section"],
        # strict: scored against the literal source chunk only, pre-fix behavior
        "stage1_rank_strict": rank_of(target_chunk, stage1_chunk_ids),
        "stage2_rank_strict": rank_of(target_chunk, stage2_chunk_ids),
        # multi-label: scored against source + near-duplicate equivalence class
        "stage1_rank": rank_of(target_chunk_set, stage1_chunk_ids),
        "stage2_rank": rank_of(target_chunk_set, stage2_chunk_ids),
        "stage1_judgment_rank": rank_of(target_judgment, stage1_judgment_ids),
        "stage2_judgment_rank": rank_of(target_judgment, stage2_judgment_ids),
    }
    for k in k_values:
        result[f"stage1_recall@{k}"] = int(result["stage1_rank"] is not None and result["stage1_rank"] <= k)
        result[f"stage2_recall@{k}"] = int(result["stage2_rank"] is not None and result["stage2_rank"] <= k)
        result[f"stage2_recall@{k}_strict"] = int(
            result["stage2_rank_strict"] is not None and result["stage2_rank_strict"] <= k
        )
    return result


def run_eval(
    gold_path: str,
    candidate_k: int = 30,
    threshold: float = 0.2,
    top_k: int = 8,
    graph_boost: float = 0.0,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    crowd_threshold: int = DEFAULT_CROWD_THRESHOLD,
    bm25_mode: str = "and_or",
    case_lookup: bool = True,
) -> Dict[str, Any]:
    records, gold_meta = load_gold_set(gold_path)
    k_values = sorted(set(k_values))

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        valid_records, stale_records = validate_gold_set(records, conn)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM judgments")
        current_judgment_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM judgment_chunks")
        current_chunk_count = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()

    # Load real models ONCE — BGE encoder + MiniLM cross-encoder are
    # expensive to construct; never re-instantiate per query.
    # Without a graph manager the retriever silently ignores graph_boost.
    graph_manager = None
    if graph_boost > 0 or case_lookup:
        from retrieval.citation_graph import CitationGraphManager
        graph_manager = CitationGraphManager()
    retriever = LegalRetriever(graph_manager=graph_manager, bm25_mode=bm25_mode, case_lookup=case_lookup)
    reranker = CrossEncoderReranker()

    per_query = []
    # "_strict": scored against the literal source chunk only (old, pre-fix
    # behavior — a near-duplicate correct chunk counts as a miss).
    # Unqualified: scored against source + near-duplicate equivalence class
    # (the multi-label fix). Comparing the two isolates how much of any
    # recall gap is a gold-set labeling artifact vs. a real retrieval miss.
    stage1_pairs: List[Tuple[Any, List[Any]]] = []
    stage1_strict_pairs: List[Tuple[Any, List[Any]]] = []
    stage2_pairs: List[Tuple[Any, List[Any]]] = []
    stage2_strict_pairs: List[Tuple[Any, List[Any]]] = []
    stage1_judgment_pairs: List[Tuple[Any, List[Any]]] = []
    stage2_judgment_pairs: List[Tuple[Any, List[Any]]] = []
    section_stage1_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    section_stage1_strict_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    section_stage2_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    section_stage2_strict_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    section_stage1_judgment_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    section_stage2_judgment_pairs: Dict[str, List[Tuple[Any, List[Any]]]] = defaultdict(list)
    # Discriminability-gate diagnostic: same stage2 strict/multi-label
    # pairs, bucketed by whether the gold record is "crowded" (crowd_size
    # >= crowd_threshold, i.e. it has near-identical chunks elsewhere in
    # the corpus — the formulaic `judgment`-section case) — per section,
    # since that's where the effect is suspected to concentrate.
    section_crowd_stage2_strict_pairs: Dict[str, Dict[str, List[Tuple[Any, List[Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    section_crowd_stage2_pairs: Dict[str, Dict[str, List[Tuple[Any, List[Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    start = time.time()
    for gold in valid_records:
        stage1, stage2 = run_two_stage_retrieval(
            retriever, reranker, gold["query"], candidate_k, threshold, top_k, graph_boost
        )
        stage1_chunk_ids = [c["chunk_id"] for c in stage1]
        stage2_chunk_ids = [c["chunk_id"] for c in stage2]
        stage1_judgment_ids = [c["judgment_id"] for c in stage1]
        stage2_judgment_ids = [c["judgment_id"] for c in stage2]
        target_chunk_strict = gold["source_chunk_id"]
        target_chunk = _acceptable_chunk_ids(gold)
        target_judgment = gold["source_judgment_id"]
        section = gold["section"]
        crowd_bucket = "high_crowd" if gold.get("crowd_size", 0) >= crowd_threshold else "low_crowd"

        stage1_pairs.append((target_chunk, stage1_chunk_ids))
        stage1_strict_pairs.append((target_chunk_strict, stage1_chunk_ids))
        stage2_pairs.append((target_chunk, stage2_chunk_ids))
        stage2_strict_pairs.append((target_chunk_strict, stage2_chunk_ids))
        stage1_judgment_pairs.append((target_judgment, stage1_judgment_ids))
        stage2_judgment_pairs.append((target_judgment, stage2_judgment_ids))
        section_stage1_pairs[section].append((target_chunk, stage1_chunk_ids))
        section_stage1_strict_pairs[section].append((target_chunk_strict, stage1_chunk_ids))
        section_stage2_pairs[section].append((target_chunk, stage2_chunk_ids))
        section_stage2_strict_pairs[section].append((target_chunk_strict, stage2_chunk_ids))
        section_stage1_judgment_pairs[section].append((target_judgment, stage1_judgment_ids))
        section_stage2_judgment_pairs[section].append((target_judgment, stage2_judgment_ids))
        section_crowd_stage2_pairs[section][crowd_bucket].append((target_chunk, stage2_chunk_ids))
        section_crowd_stage2_strict_pairs[section][crowd_bucket].append((target_chunk_strict, stage2_chunk_ids))

        per_query.append(
            evaluate_query(gold, stage1_chunk_ids, stage2_chunk_ids, stage1_judgment_ids, stage2_judgment_ids, k_values)
        )
    elapsed = time.time() - start

    per_section = {
        section: {
            "n": len(section_stage1_pairs[section]),
            "stage1": aggregate(section_stage1_pairs[section], k_values),
            "stage1_strict": aggregate(section_stage1_strict_pairs[section], k_values),
            "stage2": aggregate(section_stage2_pairs[section], k_values),
            "stage2_strict": aggregate(section_stage2_strict_pairs[section], k_values),
            "stage1_judgment": aggregate(section_stage1_judgment_pairs[section], k_values),
            "stage2_judgment": aggregate(section_stage2_judgment_pairs[section], k_values),
            # Discriminability-gate diagnostic: stage2 chunk recall split
            # by crowd bucket, strict vs multi-label. If high_crowd's
            # "stage2_strict" recall is far below its "stage2" (multi-label)
            # recall, the gap is a gold-set labeling artifact (formulaic
            # text with many valid equivalent answers), not a retrieval
            # weakness — see generate_gold_set.py's near-duplicate detection.
            "crowd_breakdown": {
                bucket: {
                    "n": len(section_crowd_stage2_pairs[section][bucket]),
                    "stage2_strict": aggregate(section_crowd_stage2_strict_pairs[section][bucket], k_values),
                    "stage2": aggregate(section_crowd_stage2_pairs[section][bucket], k_values),
                }
                for bucket in ("low_crowd", "high_crowd")
                if section_crowd_stage2_pairs[section][bucket]
            },
        }
        for section in section_stage1_pairs
    }

    return {
        "config": {
            "candidate_k": candidate_k,
            "threshold": threshold,
            "top_k": top_k,
            "graph_boost": graph_boost,
            "bm25_mode": bm25_mode,
            "case_lookup": case_lookup,
            "bm25_stats": dict(retriever.bm25_stats),
            "k_values": k_values,
            "crowd_threshold": crowd_threshold,
            "n_queries": len(valid_records),
            "n_stale": len(stale_records),
            "elapsed_s": round(elapsed, 2),
        },
        "gold_set": {
            **gold_meta,
            "current_judgment_count": current_judgment_count,
            "current_chunk_count": current_chunk_count,
        },
        "stale_records": stale_records,
        "per_query": per_query,
        "per_section": per_section,
        "aggregate_stage1_chunk": aggregate(stage1_pairs, k_values),
        "aggregate_stage1_chunk_strict": aggregate(stage1_strict_pairs, k_values),
        "aggregate_stage2_chunk": aggregate(stage2_pairs, k_values),
        "aggregate_stage2_chunk_strict": aggregate(stage2_strict_pairs, k_values),
        "aggregate_stage1_judgment": aggregate(stage1_judgment_pairs, k_values),
        "aggregate_stage2_judgment": aggregate(stage2_judgment_pairs, k_values),
    }


def _flatten_for_mlflow(report: Dict[str, Any]) -> Dict[str, float]:
    # mlflow metric names don't allow "@" — rewrite recall@8 -> recall_at_8
    # for the logged metrics; the JSON report keeps the readable "@" form.
    flat: Dict[str, float] = {"n_stale": report["config"]["n_stale"]}
    for block_name in (
        "aggregate_stage1_chunk", "aggregate_stage1_chunk_strict",
        "aggregate_stage2_chunk", "aggregate_stage2_chunk_strict",
        "aggregate_stage1_judgment", "aggregate_stage2_judgment",
    ):
        prefix = block_name.replace("aggregate_", "")
        for k, v in report[block_name].items():
            flat[f"{prefix}_{k}".replace("@", "_at_")] = v
    return flat


def _print_block(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against a synthetic gold query set."
    )
    parser.add_argument("--gold", type=str, required=True, help="Path to gold_set.json")
    parser.add_argument(
        "--candidate-k", type=int, default=30,
        help="Stage-1 candidate_k (default: 30, matches rag_pipeline.py's stage1_k)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Stage-1 similarity threshold (default: 0.2 — rag_pipeline.py's actual "
        "hardcoded stage1_threshold, not main.py's cosmetic --threshold default of 0.3)",
    )
    parser.add_argument(
        "--top-k", type=int, default=8, help="Stage-2 rerank top_k (default: 8)"
    )
    parser.add_argument(
        "--graph-boost", type=float, default=0.0,
        help="Citation-graph PageRank boost (default: 0.0 = off)",
    )
    parser.add_argument(
        "--k-values", type=str, default="5,8,10,30",
        help="Comma-separated k values for recall@k (default: 5,8,10,30)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON report path (default: benchmark/reports/retrieval_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--crowd-threshold", type=int, default=DEFAULT_CROWD_THRESHOLD,
        help=f"Min crowd_size (near-duplicate count) for a gold record to be bucketed "
        f"'high_crowd' in the discriminability-gate breakdown (default: {DEFAULT_CROWD_THRESHOLD})",
    )
    parser.add_argument(
        "--bm25-mode", choices=BM25_MODES, default="and_or",
        help="BM25 leg: and_or (default, production), and (plainto_tsquery only) or or",
    )
    parser.add_argument(
        "--case-lookup", choices=("on", "off"), default="on",
        help="Pin judgments the query names by citation / title (production: on)",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    report = run_eval(
        args.gold, args.candidate_k, args.threshold, args.top_k, args.graph_boost, k_values,
        crowd_threshold=args.crowd_threshold, bm25_mode=args.bm25_mode,
        case_lookup=args.case_lookup == "on",
    )

    output = args.output
    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output = os.path.join(reports_dir, f"retrieval_eval_{ts}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    n_stale = report["config"]["n_stale"]
    if n_stale:
        print(f"WARNING: {n_stale} gold record(s) skipped as stale (see 'stale_records' in the report):")
        for s in report["stale_records"]:
            print(f"  chunk {s['source_chunk_id']}: {s['stale_reason']}")
        print()

    gen_chunk_count = report["gold_set"].get("corpus_chunk_count")
    cur_chunk_count = report["gold_set"]["current_chunk_count"]
    if gen_chunk_count is not None and gen_chunk_count != cur_chunk_count:
        print(f"Note: corpus has {cur_chunk_count} chunks now vs {gen_chunk_count} when the gold set was generated.\n")

    print(f"Config: {report['config']}\n")
    _print_block("Stage 1 (pre-rerank) — chunk-level (multi-label):", report["aggregate_stage1_chunk"])
    _print_block("Stage 1 (pre-rerank) — chunk-level (strict, source chunk only):", report["aggregate_stage1_chunk_strict"])
    _print_block("\nStage 1 (pre-rerank) — judgment-level (right case, any chunk):", report["aggregate_stage1_judgment"])
    _print_block("\nStage 2 (post-rerank, final top_k) — chunk-level (multi-label):", report["aggregate_stage2_chunk"])
    _print_block("Stage 2 (post-rerank, final top_k) — chunk-level (strict, source chunk only):", report["aggregate_stage2_chunk_strict"])
    _print_block("\nStage 2 (post-rerank, final top_k) — judgment-level:", report["aggregate_stage2_judgment"])

    print("\nPer-section (stage 2, final top_k) — chunk-level (multi-label vs strict) vs judgment-level:")
    for section, block in sorted(report["per_section"].items()):
        s2 = block["stage2"]
        s2_strict = block["stage2_strict"]
        s2j = block["stage2_judgment"]
        print(
            f"  {section} (n={block['n']}): "
            f"chunk recall@{args.top_k}={s2.get(f'recall@{args.top_k}', float('nan')):.3f} "
            f"(strict={s2_strict.get(f'recall@{args.top_k}', float('nan')):.3f}) mrr={s2.get('mrr', float('nan')):.3f}  |  "
            f"judgment recall@{args.top_k}={s2j.get(f'recall@{args.top_k}', float('nan')):.3f} mrr={s2j.get('mrr', float('nan')):.3f}"
        )

    print(f"\nDiscriminability-gate breakdown (crowd_threshold={args.crowd_threshold}) — stage 2 chunk recall@{args.top_k}, multi-label vs strict:")
    for section, block in sorted(report["per_section"].items()):
        for bucket_name, bucket in sorted(block["crowd_breakdown"].items()):
            s2 = bucket["stage2"]
            s2_strict = bucket["stage2_strict"]
            print(
                f"  {section} / {bucket_name} (n={bucket['n']}): "
                f"recall@{args.top_k}={s2.get(f'recall@{args.top_k}', float('nan')):.3f} "
                f"(strict={s2_strict.get(f'recall@{args.top_k}', float('nan')):.3f})"
            )

    print(f"\nWrote report to {output}")

    tracking.log_benchmark_run(
        script="retrieval_eval",
        params={
            "candidate_k": args.candidate_k,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "graph_boost": args.graph_boost,
            "n_queries": report["config"]["n_queries"],
        },
        metrics=_flatten_for_mlflow(report),
    )


if __name__ == "__main__":
    main()
