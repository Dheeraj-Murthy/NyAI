#!/usr/bin/env python3
"""Judgment-level retrieval eval for gold sets with several relevant cases.

Scores two kinds of gold set:
  commentary       commentary_gold.py: a proposition from a Law Commission
                   report; relevant = the corpus cases it cites.
  masked_citation  masked_citation_gold.py: a passage from judgment A with
                   its citations masked; relevant = the cases A cites there.
                   A itself is excluded from the ranking (see below).
  case_lookup      case_lookup_gold.py: a case name, a party name or a
                   reporter citation; relevant = the judgment(s) it names.
                   Broken down by query variant.
  aila             aila_benchmark.py export: a FIRE 2019 AILA fact situation;
                   relevant = its precedents that map onto the corpus.

Per stage, over DISTINCT judgments (retrieval returns chunks, and one case
can fill several of the top 8, so stage 2 can hold fewer than 8 cases):

  hit@k     any relevant case in the top k
  recall@k  share of the relevant cases in the top k
  ndcg@k    binary-relevance nDCG
  mrr       reciprocal rank of the first relevant case

Headline numbers carry a bootstrap 95% CI: at a few hundred queries a point
estimate alone moves several points when a few queries change (see
REPORT.md, the seed-42 rerun), so compare configs by whether CIs separate.

Excluded judgments (a record's exclude_judgment_ids): stage 1 is fetched
with candidate_k * --overfetch, excluded judgments' chunks are dropped, and
the list is cut back to candidate_k before reranking, approximating the
corpus without them. Without this, the citing judgment's own chunk, which
contains the query passage verbatim, would take rank 1 every time.

Also reports how often BM25 returned no usable candidate (hybrid then runs
dense-only) and BM25's mean latency. --bm25-mode and is the old
plainto_tsquery leg, which ANDs every term, so long queries matched nothing
but the citing judgment; or (default) matches any non-trivial term.

Same two-stage retrieval and defaults as retrieval_eval.py
(candidate_k=30, threshold=0.2, top_k=8).

Usage:
    cd NLPv2/backend
    python benchmark/judgment_eval.py --gold benchmark/data/masked_citation_gold_v1.json
    python benchmark/judgment_eval.py --gold benchmark/data/masked_citation_gold_v1.json --query-field query_short
    python benchmark/judgment_eval.py --gold benchmark/data/commentary_gold_v1.json
    python benchmark/judgment_eval.py --gold benchmark/data/case_lookup_gold_v1.json [--retriever lookup]
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
from metrics import aggregate_multi, bootstrap_ci, dedupe, mrr, recall_at_k  # noqa: E402

DEFAULT_K_VALUES = (1, 3, 5, 8, 10, 30)
KINDS = ("commentary", "masked_citation", "case_lookup", "aila")
# lookup = the Graph page's case search (judgment_aliases: reporter keys and
# trigram name match), not chat retrieval; it returns judgments, not chunks,
# so it has no reranker stage (stage 2 = stage 1).
RETRIEVERS = ("hybrid", "dense", "bm25", "lookup")
# Mirrors retrieval.retriever.BM25_MODES (the retriever is imported lazily, so
# the offline smoke tests run without psycopg2; tests assert the two match).
BM25_MODES = ("and", "or", "and_or")
DEFAULT_OVERFETCH = 2


def load_gold(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if data.get("kind") not in KINDS:
        raise SystemExit(
            f"{path}: kind={data.get('kind')!r}; this scores {KINDS} gold sets "
            "(use retrieval_eval.py for the synthetic chunk-level one)"
        )
    return data["records"], {k: v for k, v in data.items() if k != "records"}


def in_degree_bucket(n: int) -> str:
    if n < 5:
        return "rarely cited (<5)"
    if n < 50:
        return "cited 5-49"
    return "heavily cited (50+)"


def group_of(record: Dict[str, Any], kind: str) -> str:
    """Breakdown key: the report for commentary, how often the target is
    cited for masked citations (a PageRank boost should help the heavily
    cited bucket and could hurt the rare one)."""
    if kind == "masked_citation":
        return in_degree_bucket(record.get("target_in_degree", 0))
    if kind == "case_lookup":
        return record.get("variant", "?")
    if kind == "aila":
        return record.get("split", "?")
    return record.get("source_id", "?")


def validate_gold(records: List[Dict[str, Any]], conn) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop relevant ids whose judgment is gone or now has a different case
    name (a re-ingest reassigns SERIAL ids). A record with no relevant id
    left is stale and excluded; a partly stale one keeps the ids that still
    match, and says so."""
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


def exclude_and_cut(chunks: List[Dict[str, Any]], exclude: Sequence[int], k: int) -> List[Dict[str, Any]]:
    excluded = set(exclude)
    return [c for c in chunks if c["judgment_id"] not in excluded][:k]


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
    query_field: str = "query",
    overfetch: int = DEFAULT_OVERFETCH,
    limit: int = 0,
    retriever_mode: str = "hybrid",
    bm25_mode: str = "and_or",
    case_lookup: bool = True,
) -> Dict[str, Any]:
    import psycopg2

    from retrieval.reranker import CrossEncoderReranker
    from retrieval.retriever import LegalRetriever
    from retrieval_eval import _default_db_connection_string

    records, gold_meta = load_gold(gold_path)
    kind = gold_meta["kind"]
    if limit:
        records = records[:limit]
    missing = [r["id"] for r in records if not r.get(query_field)]
    if missing:
        raise SystemExit(f"{len(missing)} record(s) have no '{query_field}' field, e.g. {missing[0]}")
    k_values = sorted(set(k_values))
    if retriever_mode not in RETRIEVERS:
        raise SystemExit(f"--retriever must be one of {RETRIEVERS}")
    if retriever_mode == "lookup" and kind != "case_lookup":
        raise SystemExit("--retriever lookup only makes sense for a case_lookup gold set")
    if graph_boost > 0 and retriever_mode != "hybrid":
        raise SystemExit("--graph-boost only applies to --retriever hybrid")

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        valid, stale = validate_gold(records, conn)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM judgments")
        judgment_count = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()

    # Without a graph manager the retriever silently ignores graph_boost.
    graph_manager = None
    if graph_boost > 0 or retriever_mode == "lookup" or case_lookup:
        from retrieval.citation_graph import CitationGraphManager
        graph_manager = CitationGraphManager()
    retriever = LegalRetriever(graph_manager=graph_manager, bm25_mode=bm25_mode, case_lookup=case_lookup)
    reranker = CrossEncoderReranker()

    def first_stage(query: str, k: int) -> List[Dict[str, Any]]:
        if retriever_mode == "dense":
            return retriever.retrieve_candidate_chunks(query, candidate_k=k, similarity_threshold=threshold)
        if retriever_mode == "bm25":
            return retriever.retrieve_bm25_candidates(query, candidate_k=k)
        if retriever_mode == "lookup":
            return [{"judgment_id": h["judgment_id"]} for h in graph_manager.search(query, limit=k)]
        return retriever.retrieve_judgment_candidates(
            query=query, candidate_k=k, similarity_threshold=threshold, graph_boost=graph_boost
        )

    pairs: Dict[str, List[Tuple[List[int], List[int]]]] = {"stage1": [], "stage2": []}
    by_group: Dict[str, Dict[str, List]] = defaultdict(lambda: {"stage1": [], "stage2": []})
    per_query = []
    bm25_empty = 0
    bm25_seconds = 0.0
    n_pinned = n_pinned_relevant = 0
    start = time.time()
    for r in valid:
        query = r[query_field]
        exclude = r.get("exclude_judgment_ids", [])
        fetch_k = candidate_k * overfetch if exclude else candidate_k
        stage1 = first_stage(query, fetch_k)
        stage1 = exclude_and_cut(stage1, exclude, candidate_k)
        if retriever_mode == "lookup":
            stage2 = stage1[:top_k]
        else:
            stage2 = reranker.rerank_keep_pinned(query, list(stage1), top_n=top_k)
        # Count only BM25 candidates that survive the exclusion: a passage
        # query is verbatim from the citing judgment, so with AND-semantics
        # its own chunk is often the only match.
        t0 = time.time()
        bm25_only = retriever.retrieve_bm25_candidates(query, candidate_k=fetch_k)
        bm25_seconds += time.time() - t0
        bm25_hits = len(exclude_and_cut(bm25_only, exclude, 1))
        bm25_empty += bm25_hits == 0

        ranked = {
            "stage1": dedupe([c["judgment_id"] for c in stage1]),
            "stage2": dedupe([c["judgment_id"] for c in stage2]),
        }
        relevant = r["relevant_judgment_ids"]
        pinned = [c["judgment_id"] for c in stage1 if c.get("exact_match")]
        n_pinned += bool(pinned)
        n_pinned_relevant += any(j in relevant for j in pinned)
        group = group_of(r, kind)
        row = {
            "id": r["id"],
            "query": query,
            "group": group,
            "relevant_judgment_ids": relevant,
            "n_relevant": len(relevant),
            "bm25_empty": bm25_hits == 0,
        }
        for stage, ids in ranked.items():
            pairs[stage].append((relevant, ids))
            by_group[group][stage].append((relevant, ids))
            row[stage] = {
                "ranked_judgment_ids": ids,
                f"hit@{top_k}": recall_at_k(relevant, ids, top_k),
                "mrr": mrr(relevant, ids),
            }
        per_query.append(row)
    elapsed = time.time() - start
    stage2_ks = [k for k in k_values if k <= top_k]

    return {
        "config": {
            "kind": kind,
            "query_field": query_field,
            "retriever": retriever_mode,
            "bm25_mode": bm25_mode,
            "case_lookup": case_lookup,
            "candidate_k": candidate_k,
            "threshold": threshold,
            "top_k": top_k,
            "graph_boost": graph_boost,
            "overfetch": overfetch,
            "k_values": k_values,
            "rank_unit": "distinct judgment",
            "n_queries": len(valid),
            "n_stale": len(stale),
            "n_partly_stale": sum(1 for r in valid if r["dropped_relevant"]),
            "n_bm25_empty": bm25_empty,
            "bm25_mean_ms": round(1000 * bm25_seconds / len(valid), 1) if valid else None,
            # calls include the separate BM25-only call per query made for n_bm25_empty
            "bm25_stats": dict(retriever.bm25_stats),
            "n_lookup_pinned": n_pinned,
            "n_lookup_pinned_relevant": n_pinned_relevant,
            "elapsed_s": round(elapsed, 2),
        },
        "gold_set": {**gold_meta, "current_judgment_count": judgment_count},
        "aggregate_stage1": aggregate_multi(pairs["stage1"], k_values),
        "aggregate_stage2": aggregate_multi(pairs["stage2"], stage2_ks),
        "ci95_stage1": _ci_block(per_query, "stage1", top_k),
        "ci95_stage2": _ci_block(per_query, "stage2", top_k),
        "per_group": {
            g: {
                "n": len(p["stage2"]),
                "stage1": aggregate_multi(p["stage1"], k_values),
                "stage2": aggregate_multi(p["stage2"], stage2_ks),
            }
            for g, p in sorted(by_group.items())
        },
        "stale_records": stale,
        "per_query": per_query,
    }


def _flatten_for_mlflow(report: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {
        "n_stale": report["config"]["n_stale"],
        "n_bm25_empty": report["config"]["n_bm25_empty"],
    }
    for stage in ("stage1", "stage2"):
        for k, v in report[f"aggregate_{stage}"].items():
            flat[f"{stage}_{k}".replace("@", "_at_")] = v
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Judgment-level eval for commentary / masked-citation gold sets.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--query-field", default="query",
                        help="Record field used as the query (masked_citation also has query_short)")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--graph-boost", type=float, default=0.0)
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid",
                        help="Stage-1 retriever: hybrid (default, as in production), dense only, BM25 only, "
                             "or lookup (Graph page case search; case_lookup sets only)")
    parser.add_argument("--bm25-mode", choices=BM25_MODES, default="and_or",
                        help="BM25 leg: and_or (default, production), and (plainto_tsquery only) or or")
    parser.add_argument("--case-lookup", choices=("on", "off"), default="on",
                        help="Pin judgments the query names by citation / title (production: on)")
    parser.add_argument("--overfetch", type=int, default=DEFAULT_OVERFETCH,
                        help="Stage-1 fetch multiplier when a record excludes judgments (default 2)")
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N records (smoke runs)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_eval(
        args.gold, args.candidate_k, args.threshold, args.top_k, args.graph_boost,
        [int(k) for k in args.k_values.split(",")], args.query_field, args.overfetch, args.limit,
        args.retriever, args.bm25_mode, args.case_lookup == "on",
    )
    cfg = report["config"]
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reports",
        f"judgment_eval_{cfg['kind']}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    if cfg["n_stale"] or cfg["n_partly_stale"]:
        print(f"WARNING: {cfg['n_stale']} stale record(s) excluded, {cfg['n_partly_stale']} partly stale "
              "(see stale_records / dropped_relevant in the report)\n")
    gold = report["gold_set"]
    if gold.get("kind") == "commentary" and gold.get("n_reviewed", 0) < gold.get("n", 0):
        print("WARNING: gold set includes unreviewed records — trial-run numbers only.\n")
    print(f"Config: {cfg}\n")
    print(f"BM25 had no usable candidate for {cfg['n_bm25_empty']}/{cfg['n_queries']} queries "
          "after excluding the citing judgment (hybrid ran dense-only for those); "
          f"BM25 mean latency {cfg['bm25_mean_ms']} ms; {cfg['bm25_stats']}.")
    print(f"Case lookup pinned a judgment for {cfg['n_lookup_pinned']}/{cfg['n_queries']} queries, "
          f"a relevant one for {cfg['n_lookup_pinned_relevant']}.\n")
    for stage, label in (("stage1", f"Stage 1 ({cfg['retriever']}, no reranker)"), ("stage2", "Stage 2 (reranked, final)")):
        print(f"{label}:")
        for k, v in report[f"aggregate_{stage}"].items():
            print(f"  {k}: {v:.3f}")
        for metric, ci in report[f"ci95_{stage}"].items():
            print(f"  {metric} 95% CI: [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]")
        print()
    print("Per group (stage 2):")
    for g, block in report["per_group"].items():
        s2 = block["stage2"]
        print(f"  {g} (n={block['n']}): hit@{args.top_k}={s2.get(f'hit@{args.top_k}', float('nan')):.3f} "
              f"mrr={s2.get('mrr', float('nan')):.3f}")
    print(f"\nFull report -> {output}")

    import tracking

    tracking.log_benchmark_run(
        script=f"judgment_eval_{cfg['kind']}",
        params={
            "gold_version": gold.get("version"),
            "query_field": args.query_field,
            "retriever": args.retriever,
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
