#!/usr/bin/env python3
"""Citation benchmark: given a real case with its citations redacted, does
the system find the precedents the court actually cited?

Cases come from generate_citation_set.py: a judgment's facts/issues text
(citations removed) is the query, and the in-corpus judgments it cited are
the gold set. Several ways of finding citations are scored on the same
cases, so their numbers are directly comparable:

  retrieval    The pipeline's own two-stage retrieval
               (retrieve_judgment_candidates + CrossEncoderReranker), run once
               per query chunk, collapsed to judgments, fused across chunks
               with RRF. Always run. Reported pre- and post-rerank.
  popularity   Baseline: the most-cited judgments decided before the case,
               regardless of what it is about. Anything that can't beat this
               isn't using the case text. Needs judgment_graph_stats
               (compute_graph_stats.py); skipped if that table is empty.
  llm_select   --llm select: the LLM is shown the case and the top
               --select-pool retrieved precedents and picks the ones the
               court would cite. Scored as a set (precision/recall/F1), next
               to the retrieval list cut to the same length.
  closed_book  --llm closed_book: the LLM lists precedents from memory, no
               retrieval. Its answers are resolved to judgment ids with the
               same resolver edge building uses; lines that don't resolve
               are reported as `unresolved_rate` (not all hallucinations —
               real cases outside the corpus also land there).

Leakage controls, since a citation benchmark is easy to fool:
  - Query text has every reporter citation and case title redacted
    (generate_citation_set.redact_citations).
  - The case itself is never counted as a retrieved precedent — its own
    chunks would otherwise top every list.
  - Judgments decided after the case are dropped from every list; they
    can't have been cited, and many of them cite the case itself.
  - --graph-boost > 0 and the popularity baseline both use PageRank /
    in-degree computed over the full graph, which includes this case's own
    outgoing edges. That inflates their gold targets slightly; treat those
    numbers as an upper bound.

Gold is a lower bound on "correct": a precedent the court could have cited
but didn't counts as a miss. Absolute precision is therefore pessimistic;
compare configurations against each other and against the baseline, not
against 1.0.

Usage:
    cd NLPv2/backend
    python benchmark/generate_citation_set.py --n 100 --seed 42
    python benchmark/citation_eval.py --cases benchmark/data/citation_set.json
    python benchmark/citation_eval.py --cases benchmark/data/citation_set.json --llm select
    python benchmark/citation_eval.py --cases benchmark/data/citation_set.json \\
        --llm both --model claude-sonnet-5 --external-ok
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add backend/ to the path so the dotted `retrieval.*` subpackage imports and
# the flat `tracking` import below resolve whether this script is run
# directly or as a module — same shim as retrieval_eval.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

import psycopg2  # noqa: E402

from retrieval.retriever import LegalRetriever  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
import tracking  # noqa: E402

from generate_gold_set import _default_db_connection_string, _resolve_backend  # noqa: E402
from metrics import aggregate_multi, set_prf  # noqa: E402

DEFAULT_K_VALUES = (5, 10, 20, 50)
RRF_K = 60  # same constant retrieve_judgment_candidates fuses dense + BM25 with
SNIPPET_CHARS = 400  # per candidate shown to the LLM in --llm select
CASE_TEXT_CHARS = 3000  # case text shown to the LLM, both LLM modes
CLOSED_BOOK_MAX = 15  # most precedents the closed-book prompt asks for
GOLD_SIZE_BUCKETS = ((3, 5), (6, 10), (11, 10 ** 9))

SELECT_SYSTEM_PROMPT = (
    "You are an Indian legal research assistant. You are given the facts and "
    "issues of a Supreme Court of India case (citations in it have been "
    "replaced with [CITATION]) and a numbered list of earlier judgments. "
    "Choose the judgments the court in this case would most likely cite as "
    "precedent. Reply with only their numbers, most relevant first, comma "
    "separated (e.g. 3, 7, 1). Choose only judgments that are genuinely "
    "relevant; it is fine to choose few."
)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are an Indian legal research assistant. You are given the facts and "
    "issues of a Supreme Court of India case (citations in it have been "
    "replaced with [CITATION]). List the earlier Supreme Court of India "
    f"judgments this case would most likely cite as precedent, at most {CLOSED_BOOK_MAX}, "
    "most relevant first. One per line, in the form: Petitioner v. Respondent "
    "(year), followed by a reporter citation such as AIR 1973 SC 1461 or "
    "(1973) 4 SCC 225 if you know it. Only list judgments you are confident "
    "exist. Output nothing else."
)


def load_citation_set(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data["records"], {k: v for k, v in data.items() if k != "records"}


def _parse_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def load_judgment_dates(conn) -> Dict[int, Optional[date]]:
    """Every judgment's decision date, for the "decided before the case"
    filter. One query for the whole corpus (tens of thousands of rows)
    instead of one per retrieved candidate."""
    cur = conn.cursor()
    cur.execute("SELECT id, date_of_judgment FROM judgments")
    dates = dict(cur.fetchall())
    cur.close()
    return dates


def validate_citation_set(
    records: List[Dict[str, Any]], conn
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """judgment ids are SERIAL, like chunk ids — a schema reset + re-ingest
    reassigns them, and a stale case set would then score against the
    wrong judgments with no error. A case is stale if its source or any
    gold id is gone, or the source id now names a different case."""
    ids = set()
    for r in records:
        ids.add(r["source_judgment_id"])
        ids.update(r["gold_judgment_ids"])
    cur = conn.cursor()
    cur.execute("SELECT id, petitioner, respondent FROM judgments WHERE id = ANY(%s)", (list(ids),))
    live = {
        row[0]: f"{(row[1] or 'Unknown').strip()} v. {(row[2] or 'Unknown').strip()}" for row in cur.fetchall()
    }
    cur.close()

    valid, stale = [], []
    for r in records:
        if live.get(r["source_judgment_id"]) != r["case_name"]:
            stale.append({**r, "stale_reason": "source judgment id missing or now a different case"})
        elif any(g not in live for g in r["gold_judgment_ids"]):
            stale.append({**r, "stale_reason": "a gold judgment id no longer exists"})
        else:
            valid.append(r)
    return valid, stale


def _eligible(judgment_id: Any, source_id: int, source_date: Optional[date], dates: Dict[int, Optional[date]]) -> bool:
    """Not the case itself, and not decided after it. A judgment with an
    unknown date is kept — dropping it would hide real retrievals."""
    if judgment_id is None or judgment_id == source_id:
        return False
    candidate_date = dates.get(judgment_id)
    return source_date is None or candidate_date is None or candidate_date <= source_date


def collapse_to_judgments(
    chunks: List[Dict[str, Any]], source_id: int, source_date: Optional[date], dates: Dict[int, Optional[date]]
) -> List[Dict[str, Any]]:
    """Ranked chunk list -> ranked judgment list, each judgment at the rank
    of its best chunk, keeping that chunk (for the LLM snippet)."""
    seen = set()
    out = []
    for chunk in chunks:
        jid = chunk.get("judgment_id")
        if jid in seen or not _eligible(jid, source_id, source_date, dates):
            continue
        seen.add(jid)
        out.append(chunk)
    return out


def rrf_fuse(rankings: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Fuse per-query-chunk judgment rankings into one with Reciprocal Rank
    Fusion; a precedent relevant to several parts of the case rises."""
    scores: Dict[int, float] = defaultdict(float)
    best: Dict[int, Dict[str, Any]] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            jid = chunk["judgment_id"]
            scores[jid] += 1.0 / (RRF_K + rank + 1)
            best.setdefault(jid, chunk)
    return [best[jid] for jid in sorted(scores, key=lambda j: scores[j], reverse=True)]


def retrieve_for_case(
    case: Dict[str, Any],
    retriever: LegalRetriever,
    reranker: Optional[CrossEncoderReranker],
    dates: Dict[int, Optional[date]],
    candidate_k: int,
    threshold: float,
    graph_boost: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(stage-1 fused judgment ranking, stage-2 fused judgment ranking).
    Stage 2 reranks the whole stage-1 pool rather than cutting to the
    pipeline's top_k=8, so recall@k is measurable past 8."""
    source_id = case["source_judgment_id"]
    source_date = _parse_date(case["date_of_judgment"])
    stage1_rankings, stage2_rankings = [], []
    for query in case["queries"]:
        stage1 = retriever.retrieve_judgment_candidates(
            query=query, candidate_k=candidate_k, similarity_threshold=threshold, graph_boost=graph_boost
        )
        stage1_rankings.append(collapse_to_judgments(stage1, source_id, source_date, dates))
        if reranker is not None:
            stage2 = reranker.rerank(query, list(stage1), top_n=len(stage1))
            stage2_rankings.append(collapse_to_judgments(stage2, source_id, source_date, dates))
    stage1_fused = rrf_fuse(stage1_rankings)
    return stage1_fused, (rrf_fuse(stage2_rankings) if reranker is not None else stage1_fused)


def load_popularity_ranking(conn) -> List[int]:
    """All judgment ids, most-cited first. Empty if graph stats were never
    computed."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT judgment_id FROM judgment_graph_stats ORDER BY in_degree DESC, pagerank DESC")
        return [r[0] for r in cur.fetchall()]
    except psycopg2.Error:
        conn.rollback()
        return []
    finally:
        cur.close()


def popularity_for_case(
    case: Dict[str, Any], popularity: List[int], dates: Dict[int, Optional[date]], depth: int
) -> List[int]:
    source_id = case["source_judgment_id"]
    source_date = _parse_date(case["date_of_judgment"])
    out = []
    for jid in popularity:
        if _eligible(jid, source_id, source_date, dates):
            out.append(jid)
            if len(out) >= depth:
                break
    return out


def _case_text(case: Dict[str, Any]) -> str:
    return "\n\n".join(case["queries"])[:CASE_TEXT_CHARS]


def llm_select(backend, case: Dict[str, Any], pool: List[Dict[str, Any]]) -> Tuple[List[int], str]:
    """The LLM's chosen precedents from `pool`, in its order, as judgment ids."""
    lines = []
    for i, chunk in enumerate(pool, start=1):
        year = f" ({chunk['year']})" if chunk.get("year") else ""
        snippet = re.sub(r"\s+", " ", chunk.get("text", ""))[:SNIPPET_CHARS]
        lines.append(f"[{i}] {chunk.get('case', 'Unknown')}{year}: {snippet}")
    context = f"CASE:\n{_case_text(case)}\n\nCANDIDATE PRECEDENTS:\n" + "\n".join(lines)
    gen = backend.generate(
        context_block=context,
        user_query="Which candidate precedents would this court cite? Numbers only.",
        system_prompt=SELECT_SYSTEM_PROMPT,
        max_new_tokens=64,
    )
    chosen: List[int] = []
    for num in re.findall(r"\d+", gen.text):
        idx = int(num) - 1
        if 0 <= idx < len(pool):
            jid = pool[idx]["judgment_id"]
            if jid not in chosen:
                chosen.append(jid)
    return chosen, gen.text


def closed_book(
    backend, graph_manager, case: Dict[str, Any], dates: Dict[int, Optional[date]]
) -> Tuple[List[int], Dict[str, Any]]:
    """Precedents the LLM names from memory, resolved to judgment ids."""
    gen = backend.generate(
        context_block=f"CASE:\n{_case_text(case)}",
        user_query="List the precedents this case would cite.",
        system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
        max_new_tokens=512,
    )
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in gen.text.splitlines()
    ]
    lines = [line for line in lines if len(line) >= 5][:CLOSED_BOOK_MAX]
    resolved = graph_manager.resolve_citations(lines) if lines else []

    source_id = case["source_judgment_id"]
    source_date = _parse_date(case["date_of_judgment"])
    chosen: List[int] = []
    n_unresolved = n_future = 0
    for r in resolved:
        jid = r["judgment_id"]
        if jid is None:
            n_unresolved += 1
        elif not _eligible(jid, source_id, source_date, dates):
            # Resolved to the case itself or to a later judgment: wrong
            # either way, and kept out of the prediction like retrieval.
            n_future += 1
        elif jid not in chosen:
            chosen.append(jid)
    return chosen, {
        "raw": gen.text,
        "n_listed": len(lines),
        "n_unresolved": n_unresolved,
        "n_ineligible": n_future,
        "resolutions": resolved,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_prf(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {key: _mean([r[key] for r in rows]) for key in ("precision", "recall", "f1")}


def _decade(case: Dict[str, Any]) -> str:
    d = case.get("date_of_judgment")
    return f"{d[:3]}0s" if d else "unknown"


def _gold_bucket(n_gold: int) -> str:
    for lo, hi in GOLD_SIZE_BUCKETS:
        if lo <= n_gold <= hi:
            return f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}"
    return f"<{GOLD_SIZE_BUCKETS[0][0]}"


def run_eval(
    cases_path: str,
    candidate_k: int = 100,
    threshold: float = 0.2,
    graph_boost: float = 0.0,
    rerank: bool = True,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    llm: str = "none",
    model: Optional[str] = None,
    external_ok: bool = False,
    api_key: Optional[str] = None,
    select_pool: int = 20,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    records, set_meta = load_citation_set(cases_path)
    k_values = sorted(set(k_values))

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        valid, stale = validate_citation_set(records, conn)
        dates = load_judgment_dates(conn)
        popularity = load_popularity_ranking(conn)
    finally:
        conn.close()
    if limit is not None:
        valid = valid[:limit]

    graph_manager = None
    if graph_boost > 0.0 or llm in ("closed_book", "both"):
        from retrieval.citation_graph import CitationGraphManager

        graph_manager = CitationGraphManager()
    # Load real models ONCE — never re-instantiate per case.
    retriever = LegalRetriever(graph_manager=graph_manager) if graph_boost > 0.0 else LegalRetriever()
    reranker = CrossEncoderReranker() if rerank else None
    backend = _resolve_backend(model, external_ok, api_key) if llm != "none" else None
    if backend is not None and hasattr(backend, "inference"):
        # Local Qwen samples at temperature 0.9 by default (QwenInference);
        # greedy decoding makes repeated runs on the same cases comparable.
        backend.inference.do_sample = False

    pairs: Dict[str, List[Tuple[set, List[int]]]] = defaultdict(list)
    by_decade: Dict[str, List[Tuple[set, List[int]]]] = defaultdict(list)
    by_gold_size: Dict[str, List[Tuple[set, List[int]]]] = defaultdict(list)
    select_prf, select_matched_prf, closed_prf = [], [], []
    closed_unresolved: List[float] = []
    per_case = []

    start = time.time()
    for i, case in enumerate(valid, start=1):
        gold = set(case["gold_judgment_ids"])
        stage1, stage2 = retrieve_for_case(case, retriever, reranker, dates, candidate_k, threshold, graph_boost)
        stage1_ids = [c["judgment_id"] for c in stage1]
        stage2_ids = [c["judgment_id"] for c in stage2]
        pairs["stage1"].append((gold, stage1_ids))
        if reranker is not None:
            pairs["stage2"].append((gold, stage2_ids))
        by_decade[_decade(case)].append((gold, stage2_ids))
        by_gold_size[_gold_bucket(len(gold))].append((gold, stage2_ids))

        result: Dict[str, Any] = {
            "source_judgment_id": case["source_judgment_id"],
            "case_name": case["case_name"],
            "date_of_judgment": case["date_of_judgment"],
            "gold_judgment_ids": sorted(gold),
            "stage1_top": stage1_ids[: max(k_values)],
            "stage2_top": stage2_ids[: max(k_values)],
            "stage2_gold_ranks": {
                str(g): (stage2_ids.index(g) + 1 if g in stage2_ids else None) for g in sorted(gold)
            },
        }

        if popularity:
            pop_ids = popularity_for_case(case, popularity, dates, max(k_values))
            pairs["popularity"].append((gold, pop_ids))
            result["popularity_top"] = pop_ids

        if llm in ("select", "both"):
            chosen, raw = llm_select(backend, case, stage2[:select_pool])
            prf = set_prf(gold, chosen)
            # Same number of picks straight off the retrieval ranking: did
            # the LLM's choice beat just trusting the order?
            matched = set_prf(gold, stage2_ids[: len(chosen)]) if chosen else set_prf(gold, [])
            select_prf.append(prf)
            select_matched_prf.append(matched)
            result["llm_select"] = {"chosen": chosen, "raw": raw, **prf, "retrieval_same_n": matched}

        if llm in ("closed_book", "both"):
            chosen, info = closed_book(backend, graph_manager, case, dates)
            prf = set_prf(gold, chosen)
            closed_prf.append(prf)
            if info["n_listed"]:
                closed_unresolved.append(info["n_unresolved"] / info["n_listed"])
            result["closed_book"] = {"chosen": chosen, **prf, **info}

        per_case.append(result)
        found = sum(1 for g in gold if g in stage2_ids[:10])
        print(f"[{i}/{len(valid)}] judgment {case['source_judgment_id']}: {found}/{len(gold)} gold in top 10")
    elapsed = time.time() - start

    report: Dict[str, Any] = {
        "config": {
            "candidate_k": candidate_k,
            "threshold": threshold,
            "graph_boost": graph_boost,
            "rerank": rerank,
            "k_values": k_values,
            "llm": llm,
            "model": backend.model_id if backend else None,
            "select_pool": select_pool,
            "n_cases": len(valid),
            "n_stale": len(stale),
            "elapsed_s": round(elapsed, 2),
        },
        "citation_set": set_meta,
        "gold_stats": {
            "mean_gold": _mean([len(c["gold_judgment_ids"]) for c in valid]),
            # Share of everything the judgments cited that is scorable at
            # all (resolved to an in-corpus judgment). The rest is invisible
            # to this benchmark.
            "measurable_share": _mean([c["n_gold"] / c["n_citations_total"] for c in valid if c["n_citations_total"]]),
        },
        "stale_records": stale,
        "per_case": per_case,
        "ranking": {name: aggregate_multi(p, k_values) for name, p in pairs.items()},
        "stage2_by_decade": {
            d: {"n": len(p), **aggregate_multi(p, k_values)} for d, p in sorted(by_decade.items())
        },
        "stage2_by_gold_size": {
            b: {"n": len(p), **aggregate_multi(p, k_values)} for b, p in sorted(by_gold_size.items())
        },
    }
    if select_prf:
        report["llm_select"] = {
            "llm": _mean_prf(select_prf),
            "retrieval_same_n": _mean_prf(select_matched_prf),
            "mean_chosen": _mean([len(c["llm_select"]["chosen"]) for c in per_case]),
        }
    if closed_prf:
        report["closed_book"] = {
            **_mean_prf(closed_prf),
            "unresolved_rate": _mean(closed_unresolved),
            "mean_listed": _mean([c["closed_book"]["n_listed"] for c in per_case]),
        }
    return report


def _flatten_for_mlflow(report: Dict[str, Any]) -> Dict[str, float]:
    # mlflow metric names don't allow "@" — same rewrite as retrieval_eval.py.
    flat: Dict[str, float] = {"n_stale": report["config"]["n_stale"]}
    for name, block in report["ranking"].items():
        for k, v in block.items():
            flat[f"{name}_{k}".replace("@", "_at_")] = v
    for name in ("llm_select", "closed_book"):
        block = report.get(name)
        if not block:
            continue
        for k, v in block.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{name}_{k}_{kk}"] = vv
            else:
                flat[f"{name}_{k}"] = v
    return flat


def _print_ranking(title: str, metrics: Dict[str, float], k_values: Sequence[int]) -> None:
    cells = "  ".join(
        f"R@{k}={metrics[f'recall@{k}']:.3f} P@{k}={metrics[f'precision@{k}']:.3f}" for k in k_values
    )
    print(f"{title}\n  {cells}\n  MRR={metrics['mrr']:.3f}  MAP@{k_values[-1]}={metrics[f'map@{k_values[-1]}']:.3f}  "
          f"pool_recall={metrics['pool_recall']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how well the system finds the precedents real judgments cited."
    )
    parser.add_argument("--cases", type=str, required=True, help="Path to citation_set.json")
    parser.add_argument(
        "--candidate-k", type=int, default=100,
        help="Stage-1 chunks per query chunk (default: 100; more than the pipeline's 30, "
        "since a case cites many precedents, not one)",
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="Stage-1 similarity threshold (default: 0.2)")
    parser.add_argument("--graph-boost", type=float, default=0.0, help="Citation-graph PageRank boost (default: 0.0)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder (stage 1 only)")
    parser.add_argument("--k-values", type=str, default="5,10,20,50", help="Comma-separated k (default: 5,10,20,50)")
    parser.add_argument(
        "--llm", choices=("none", "select", "closed_book", "both"), default="none",
        help="LLM modes to run on top of retrieval (default: none, which needs no GPU)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM for --llm: omit for local Qwen, or claude-*/gpt-*/gemini-* with --external-ok",
    )
    parser.add_argument("--external-ok", action="store_true", help="Confirm sending case text to an external API")
    parser.add_argument("--api-key", type=str, default=None, help="API key for an external model")
    parser.add_argument(
        "--select-pool", type=int, default=20,
        help="Top retrieved judgments the LLM chooses from in --llm select (default: 20)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N cases (smoke test)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON report path (default: benchmark/reports/citation_eval_<timestamp>.json)",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    report = run_eval(
        args.cases, args.candidate_k, args.threshold, args.graph_boost, not args.no_rerank, k_values,
        args.llm, args.model, args.external_ok, args.api_key, args.select_pool, args.limit,
    )

    output = args.output
    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output = os.path.join(reports_dir, f"citation_eval_{ts}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if report["config"]["n_stale"]:
        print(f"\nWARNING: {report['config']['n_stale']} case(s) skipped as stale (see 'stale_records'):")
        for s in report["stale_records"]:
            print(f"  judgment {s['source_judgment_id']}: {s['stale_reason']}")

    k_values = report["config"]["k_values"]
    gs = report["gold_stats"]
    print(f"\nConfig: {report['config']}")
    print(f"Gold: {gs['mean_gold']:.1f} in-corpus precedents per case; "
          f"{gs['measurable_share']:.1%} of all citations are measurable\n")
    labels = {
        "stage1": "Retrieval, stage 1 (hybrid, pre-rerank):",
        "stage2": "Retrieval, stage 2 (reranked):",
        "popularity": "Baseline — most-cited judgments before the case:",
    }
    for name in ("stage1", "stage2", "popularity"):
        if name in report["ranking"]:
            _print_ranking(labels[name], report["ranking"][name], k_values)
    if "popularity" not in report["ranking"]:
        print("Baseline skipped: judgment_graph_stats is empty (run ingestion/compute_graph_stats.py).")

    k_mid = 10 if 10 in k_values else k_values[0]
    print(f"\nStage 2 by decade (recall@{k_mid}):")
    for d, block in report["stage2_by_decade"].items():
        print(f"  {d} (n={block['n']}): {block[f'recall@{k_mid}']:.3f}")
    print(f"Stage 2 by number of gold precedents (recall@{k_mid}):")
    for b, block in report["stage2_by_gold_size"].items():
        print(f"  {b} (n={block['n']}): {block[f'recall@{k_mid}']:.3f}")

    if "llm_select" in report:
        s = report["llm_select"]
        print(f"\nLLM select (from top {args.select_pool}, {s['mean_chosen']:.1f} chosen on average):")
        print(f"  LLM:                      P={s['llm']['precision']:.3f} R={s['llm']['recall']:.3f} F1={s['llm']['f1']:.3f}")
        print(f"  retrieval, same # picks:  P={s['retrieval_same_n']['precision']:.3f} "
              f"R={s['retrieval_same_n']['recall']:.3f} F1={s['retrieval_same_n']['f1']:.3f}")
    if "closed_book" in report:
        c = report["closed_book"]
        print(f"\nClosed book ({c['mean_listed']:.1f} listed on average): P={c['precision']:.3f} "
              f"R={c['recall']:.3f} F1={c['f1']:.3f}  unresolved_rate={c['unresolved_rate']:.3f}")

    print(f"\nWrote report to {output}")

    tracking.log_benchmark_run(
        script="citation_eval",
        params={
            "candidate_k": args.candidate_k,
            "threshold": args.threshold,
            "graph_boost": args.graph_boost,
            "rerank": not args.no_rerank,
            "llm": args.llm,
            "model": report["config"]["model"],
            "n_cases": report["config"]["n_cases"],
        },
        metrics=_flatten_for_mlflow(report),
    )


if __name__ == "__main__":
    main()
