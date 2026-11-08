#!/usr/bin/env python3
"""Score retrieval against a commentary gold set (commentary_gold.py).

Judgment-level only: commentary cites cases, not paragraphs. Each record has
one or more relevant judgments (every corpus case the commentary cites for
the proposition), so this reports, per stage:

  hit@k     any relevant case in the top k distinct cases
  recall@k  share of the relevant cases in the top k distinct cases
  ndcg@k    binary-relevance nDCG over the distinct cases
  mrr       reciprocal rank of the first relevant case

Ranks count DISTINCT judgments (retrieval returns chunks, and one case can
fill several of the top 8), unlike retrieval_eval.py's judgment-level
numbers, which rank chunks. Stage 2 at top_k=8 chunks can therefore hold
fewer than 8 cases.

Headline numbers carry a bootstrap 95% CI. At ~100 queries a point estimate
alone moves several points when a few queries change (see REPORT.md, the
seed-42 rerun), so compare configs by whether their CIs separate.

Same two-stage retrieval and defaults as retrieval_eval.py
(candidate_k=30, threshold=0.2, top_k=8).

Usage:
    cd NLPv2/backend
    python benchmark/commentary_eval.py --gold benchmark/data/commentary_gold_v1.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from graph.citation_resolver import case_name_alias  # noqa: E402
from metrics import aggregate_multi, bootstrap_ci, dedupe, recall_at_k, mrr  # noqa: E402

DEFAULT_K_VALUES = (1, 3, 5, 8, 10, 30)


def load_gold(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if data.get("kind") != "commentary":
        raise SystemExit(f"{path} is not a commentary gold set (use retrieval_eval.py for synthetic ones)")
    return data["records"], {k: v for k, v in data.items() if k != "records"}


def validate_gold(records: List[Dict[str, Any]], conn) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop relevant ids whose judgment is gone or now has a different case
    name (a re-ingest reassigns SERIAL ids). A record with no relevant id
    left is stale and excluded; a partly stale one is kept with the ids
    that still match, and says so."""
    ids = sorted({jid for r in records for jid in r["relevant_judgment_ids"]})
    cur = conn.cursor()
    cur.execute("SELECT id, petitioner, respondent FROM judgments WHERE id = ANY(%s)", (ids,))
    live = {row[0]: case_name_alias(row[1], row[2]) or "" for row in cur.fetchall()}
    cur.close()

    valid, stale = [], []
    for r in records:
        expected = r.get("relevant_case_names", {})
        keep, dropped = [], []
        for jid in r["relevant_judgment_ids"]:
            want = expected.get(str(jid))
            if jid not in live:
                dropped.append({"judgment_id": jid, "reason": "judgment no longer exists"})
            elif want and live[jid] != want:
                dropped.append({"judgment_id": jid, "reason": f"id now names '{live[jid]}', expected '{want}'"})
            else:
                keep.append(jid)
        if not keep:
            stale.append({**r, "stale": dropped})
        else:
            valid.append({**r, "relevant_judgment_ids": keep, "dropped_relevant": dropped})
    return valid, stale


def _ci_block(per_query: List[Dict[str, Any]], stage: str, top_k: int) -> Dict[str, Any]:
    out = {}
    for metric in (f"hit@{top_k}", "mrr"):
        values = [q[stage][metric] for q in per_query]
        lo, hi = bootstrap_ci(values)
        out[metric] = {"mean": sum(values) / len(values) if values else float("nan"), "ci95": [lo, hi]}
    return out


def run_eval(
    gold_path: str,
    candidate_k: int = 30,
    threshold: float = 0.2,
    top_k: int = 8,
    graph_boost: float = 0.0,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    import psycopg2

    from retrieval.reranker import CrossEncoderReranker
    from retrieval.retriever import LegalRetriever
    from retrieval_eval import _default_db_connection_string, run_two_stage_retrieval

    records, gold_meta = load_gold(gold_path)
    k_values = sorted(set(k_values))

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        valid, stale = validate_gold(records, conn)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM judgments")
        judgment_count = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()

    retriever = LegalRetriever()
    reranker = CrossEncoderReranker()

    pairs: Dict[str, List[Tuple[List[int], List[int]]]] = {"stage1": [], "stage2": []}
    by_source: Dict[str, Dict[str, List]] = defaultdict(lambda: {"stage1": [], "stage2": []})
    per_query = []
    start = time.time()
    for r in valid:
        stage1, stage2 = run_two_stage_retrieval(
            retriever, reranker, r["query"], candidate_k, threshold, top_k, graph_boost
        )
        ranked = {
            "stage1": dedupe([c["judgment_id"] for c in stage1]),
            "stage2": dedupe([c["judgment_id"] for c in stage2]),
        }
        relevant = r["relevant_judgment_ids"]
        row = {
            "id": r["id"],
            "query": r["query"],
            "source_id": r["source_id"],
            "relevant_judgment_ids": relevant,
            "n_relevant": len(relevant),
        }
        for stage, ids in ranked.items():
            pairs[stage].append((relevant, ids))
            by_source[r["source_id"]][stage].append((relevant, ids))
            row[stage] = {
                "ranked_judgment_ids": ids,
                f"hit@{top_k}": recall_at_k(relevant, ids, top_k),
                "mrr": mrr(relevant, ids),
            }
        per_query.append(row)
    elapsed = time.time() - start

    return {
        "config": {
            "candidate_k": candidate_k,
            "threshold": threshold,
            "top_k": top_k,
            "graph_boost": graph_boost,
            "k_values": k_values,
            "rank_unit": "distinct judgment",
            "n_queries": len(valid),
            "n_stale": len(stale),
            "n_partly_stale": sum(1 for r in valid if r["dropped_relevant"]),
            "elapsed_s": round(elapsed, 2),
        },
        "gold_set": {**gold_meta, "current_judgment_count": judgment_count},
        "aggregate_stage1": aggregate_multi(pairs["stage1"], k_values),
        "aggregate_stage2": aggregate_multi(pairs["stage2"], [k for k in k_values if k <= top_k]),
        "ci95_stage1": _ci_block(per_query, "stage1", top_k),
        "ci95_stage2": _ci_block(per_query, "stage2", top_k),
        "per_source": {
            src: {
                "n": len(p["stage2"]),
                "stage1": aggregate_multi(p["stage1"], k_values),
                "stage2": aggregate_multi(p["stage2"], [k for k in k_values if k <= top_k]),
            }
            for src, p in sorted(by_source.items())
        },
        "stale_records": stale,
        "per_query": per_query,
    }


def _flatten_for_mlflow(report: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {"n_stale": report["config"]["n_stale"]}
    for stage in ("stage1", "stage2"):
        for k, v in report[f"aggregate_{stage}"].items():
            flat[f"{stage}_{k}".replace("@", "_at_")] = v
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a commentary gold set.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--graph-boost", type=float, default=0.0)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_eval(
        args.gold, args.candidate_k, args.threshold, args.top_k, args.graph_boost,
        [int(k) for k in args.k_values.split(",")],
    )
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports",
        f"commentary_eval_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    cfg = report["config"]
    if cfg["n_stale"] or cfg["n_partly_stale"]:
        print(f"WARNING: {cfg['n_stale']} stale record(s) excluded, {cfg['n_partly_stale']} partly stale "
              "(see stale_records / dropped_relevant in the report)\n")
    if report["gold_set"].get("n_reviewed", 0) < report["gold_set"].get("n", 0):
        print("WARNING: gold set includes unreviewed records — trial-run numbers only.\n")
    print(f"Config: {cfg}\n")
    for stage, label in (("stage1", "Stage 1 (hybrid recall)"), ("stage2", "Stage 2 (reranked, final)")):
        print(f"{label}:")
        for k, v in report[f"aggregate_{stage}"].items():
            print(f"  {k}: {v:.3f}")
        for metric, ci in report[f"ci95_{stage}"].items():
            print(f"  {metric} 95% CI: [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]")
        print()
    print("Per source (stage 2):")
    for src, block in report["per_source"].items():
        s2 = block["stage2"]
        print(f"  {src} (n={block['n']}): hit@{args.top_k}={s2.get(f'hit@{args.top_k}', float('nan')):.3f} "
              f"mrr={s2.get('mrr', float('nan')):.3f}")
    print(f"\nFull report -> {output}")

    import tracking

    tracking.log_benchmark_run(
        script="commentary_eval",
        params={
            "gold_version": report["gold_set"].get("version"),
            "candidate_k": args.candidate_k,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "graph_boost": args.graph_boost,
            "n_queries": cfg["n_queries"],
        },
        metrics=_flatten_for_mlflow(report),
    )


if __name__ == "__main__":
    main()
