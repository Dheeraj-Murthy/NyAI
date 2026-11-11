"""Pure IR metric functions for the retrieval benchmark.

No pytest suite exists in this repo (see AGENTS.md) — verify these by
running retrieval_eval.py and reading its printed output, the same
convention as ingestion/verify_search.py.

Each gold query has one PRIMARY known-relevant chunk (the chunk it was
synthetically generated from), but `rank_of` also accepts a set/list of
IDs — the primary chunk plus any near-duplicate chunks found at gold-set
generation time (see generate_gold_set.py's near-duplicate detection).
That makes this a single-equivalence-class simplification of
recall@k/MRR/nDCG@k: any one match in the accepted set counts as the hit,
scored at its best (lowest) rank. A plain scalar id still works exactly
as before — it's just treated as a one-element class.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

RelevantIds = Union[Any, Sequence[Any], Set[Any]]


def _as_id_set(relevant_ids: RelevantIds) -> Set[Any]:
    if isinstance(relevant_ids, (set, frozenset, list, tuple)):
        return set(relevant_ids)
    return {relevant_ids}


def rank_of(relevant_ids: RelevantIds, ranked_ids: Sequence[Any]) -> Optional[int]:
    """1-based rank of the first id in ranked_ids that belongs to
    relevant_ids (a single id, or a set/list of ids treated as one
    equivalence class), or None if none of them appear."""
    targets = _as_id_set(relevant_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in targets:
            return i + 1
    return None


def recall_at_k(relevant_ids: RelevantIds, ranked_ids: Sequence[Any], k: int) -> int:
    """1 if any id in the relevant equivalence class is within the top k, else 0."""
    rank = rank_of(relevant_ids, ranked_ids)
    return 1 if rank is not None and rank <= k else 0


def mrr(relevant_ids: RelevantIds, ranked_ids: Sequence[Any]) -> float:
    rank = rank_of(relevant_ids, ranked_ids)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(relevant_ids: RelevantIds, ranked_ids: Sequence[Any], k: int) -> float:
    """Single-relevant-equivalence-class nDCG@k. Ideal DCG is 1 (a relevant
    id at rank 1), so this reduces to 1/log2(rank+1) when the best-ranked
    match is within the top k, else 0."""
    rank = rank_of(relevant_ids, ranked_ids)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate(
    targets_and_ranked: List[Tuple[RelevantIds, Sequence[Any]]], k_values: Sequence[int]
) -> Dict[str, float]:
    """Mean recall@k / nDCG@k per k, and mean MRR, across all queries."""
    n = len(targets_and_ranked)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(recall_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(t, r, k) for t, r in targets_and_ranked) / n
    out["mrr"] = sum(mrr(t, r) for t, r in targets_and_ranked) / n
    return out


# ---------------------------------------------------------------------------
# Multi-relevant metrics (citation_eval.py)
#
# Unlike the retrieval gold set above, a citation-benchmark case has MANY
# independently relevant targets (every precedent the judgment actually
# cited), not one equivalence class. Each relevant id found counts
# separately, so these are the standard multi-relevant definitions.
# ---------------------------------------------------------------------------


def multi_recall_at_k(relevant_ids: Set[Any], ranked_ids: Sequence[Any], k: int) -> float:
    """Fraction of the relevant ids found within the top k."""
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & set(ranked_ids[:k])) / len(relevant_ids)


def multi_precision_at_k(relevant_ids: Set[Any], ranked_ids: Sequence[Any], k: int) -> float:
    """Fraction of the top k that is relevant. Divides by k even when fewer
    than k ids were returned, so a short list isn't rewarded for it."""
    return len(relevant_ids & set(ranked_ids[:k])) / k


def multi_hit_at_k(relevant_ids: Set[Any], ranked_ids: Sequence[Any], k: int) -> int:
    """1 if at least one relevant id is within the top k, else 0."""
    return 1 if relevant_ids & set(ranked_ids[:k]) else 0


def multi_average_precision_at_k(relevant_ids: Set[Any], ranked_ids: Sequence[Any], k: int) -> float:
    """AP@k, normalised by min(|relevant|, k) so a perfect top-k scores 1
    even when there are more relevant ids than k slots."""
    if not relevant_ids:
        return 0.0
    hits = 0
    total = 0.0
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in relevant_ids:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant_ids), k)


def multi_ndcg_at_k(relevant_ids: Set[Any], ranked_ids: Sequence[Any], k: int) -> float:
    """Binary-gain nDCG@k with several relevant ids."""
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(ranked_ids[:k]) if rid in relevant_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal else 0.0


def set_prf(relevant_ids: Set[Any], predicted_ids: Sequence[Any]) -> Dict[str, float]:
    """Precision / recall / F1 of an unranked predicted set (an LLM's chosen
    citations). An empty prediction scores precision 0, not undefined."""
    predicted = set(predicted_ids)
    tp = len(relevant_ids & predicted)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(relevant_ids) if relevant_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_multi(
    targets_and_ranked: List[Tuple[Set[Any], Sequence[Any]]], k_values: Sequence[int]
) -> Dict[str, float]:
    """Mean multi-relevant recall/precision/hit/MAP/nDCG per k, plus MRR
    (rank of the first relevant id) and full-list recall, across cases."""
    n = len(targets_and_ranked)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(multi_recall_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"precision@{k}"] = sum(multi_precision_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"hit@{k}"] = sum(multi_hit_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"map@{k}"] = sum(multi_average_precision_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"ndcg@{k}"] = sum(multi_ndcg_at_k(t, r, k) for t, r in targets_and_ranked) / n
    out["mrr"] = sum(mrr(t, r) for t, r in targets_and_ranked) / n
    # Recall over everything returned, however deep: the ceiling no
    # reranking of this candidate pool could beat.
    out["pool_recall"] = sum(multi_recall_at_k(t, r, len(r)) for t, r in targets_and_ranked) / n
    return out
