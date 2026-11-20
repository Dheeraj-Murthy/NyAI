#!/usr/bin/env python3
"""Smoke tests for the citation-pair training data builder.

Offline: the masked-citation fixtures, no DB, no model.

Run: cd NLPv2/backend && python tests/smoke_test_citation_pairs.py
"""
import json
import os
import random
import sys
import tempfile

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "tests"))
sys.path.insert(0, os.path.join(_backend_dir, "training"))

from smoke_test_masked_citation import EDGES, META, TEXTS  # noqa: E402
from citation_pairs import (  # noqa: E402
    best_chunk,
    build_pairs,
    holdout_ids,
    is_dev,
    negative_candidates,
    usable_chunk,
)

LONG = " ".join(["procedure established by law personal liberty fair just reasonable"] * 5)
CHUNKS = {
    2: [(20, "passports and travel abroad " * 10), (21, LONG), (22, "short")],
    3: [(30, "anticipatory bail discretion conditions arrest " * 8)],
}


def fetch_chunks(ids):
    return {i: CHUNKS.get(i, []) for i in ids}


def fetch_texts(ids):
    return {i: TEXTS[i] for i in ids}


def test_helpers():
    print("--- Testing helpers ---")
    assert not usable_chunk("short") and usable_chunk(LONG)
    assert not usable_chunk("க" * 200 + " " + LONG[:100])  # mostly non-ASCII: mojibake
    assert best_chunk("personal liberty procedure fair", CHUNKS[2])[0] == 21
    assert best_chunk("anything", [(1, "short")]) is None
    assert is_dev(12345, 100.0) and not is_dev(12345, 0.0)
    assert is_dev(7, 2.0) == is_dev(7, 2.0)  # stable
    ranked = [(c, c % 7) for c in range(60)]
    rng = random.Random(0)
    cands = negative_candidates(ranked, relevant={0, 1}, k=3, rng=rng)
    assert len(cands) == 9 and all(10 <= c < 50 and c % 7 not in (0, 1) for c in cands), cands
    print("✓ helpers passed")


def test_holdout():
    print("--- Testing held-out judgment collection ---")
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "masked.json")
        json.dump({"records": [{"source_judgment_id": 1, "relevant_judgment_ids": [2, 3], "exclude_judgment_ids": [1],
                                "primary_target_id": 2, "target_in_degree": 999}]}, open(a, "w"))
        b = os.path.join(d, "synthetic.json")
        json.dump({"records": [{"source_chunk_id": 77, "acceptable_chunk_ids": [77, 78], "source_judgment_id": 9}]},
                  open(b, "w"))
        ids, per_file = holdout_ids([a, b], lambda cids: {77: 40, 78: 41})
        assert ids == {1, 2, 3, 9, 40, 41}, ids  # target_in_degree is not an id
        assert per_file == {"masked.json": 3, "synthetic.json": 3}, per_file
    print("✓ holdout passed")


def test_build_pairs():
    print("--- Testing pair building ---")
    pairs, skipped = build_pairs(EDGES, META, fetch_texts, fetch_chunks, holdout=set(), short_fraction=0.0)
    # (1,2) and (1,3) are eligible; both come from the same passage.
    assert sorted((p["source_judgment_id"], p["target_judgment_id"]) for p in pairs) == [(1, 2), (1, 3)], pairs
    by_tgt = {p["target_judgment_id"]: p for p in pairs}
    assert by_tgt[2]["positive_chunk_id"] == 21 and by_tgt[3]["positive_chunk_id"] == 30
    for p in pairs:
        q = p["query"].lower()
        assert "maneka" not in q and "sibbia" not in q and "s.c.r" not in q, q  # masked
        assert set(p["relevant_judgment_ids"]) == {2, 3}
        assert p["query_kind"] == "passage" and p["negatives"] == []

    # Held out on either side -> dropped
    pairs, skipped = build_pairs(EDGES, META, fetch_texts, fetch_chunks, holdout={3})
    assert pairs == [] and skipped["held-out judgment"] == 1 and skipped["held-out judgment in passage"] == 1, (pairs, skipped)
    pairs, skipped = build_pairs(EDGES, META, fetch_texts, fetch_chunks, holdout={1})
    assert pairs == [] and skipped["held-out judgment"] == 2

    # Cap per cited judgment, short queries, determinism
    pairs, skipped = build_pairs(EDGES, META, fetch_texts, fetch_chunks, holdout=set(), max_per_target=1, short_fraction=1.0)
    assert all(p["query_kind"] == "short" and len(p["query"].split()) <= 40 for p in pairs), pairs
    again, _ = build_pairs(EDGES, META, fetch_texts, fetch_chunks, holdout=set(), max_per_target=1, short_fraction=1.0)
    assert pairs == again
    # No usable chunk in the cited judgment -> skipped
    pairs, skipped = build_pairs(EDGES, META, fetch_texts, lambda ids: {}, holdout=set())
    assert pairs == [] and skipped["no usable chunk in cited judgment"] == 2
    print("✓ pair building passed")


def test_train_columns():
    print("--- Testing training columns ---")
    try:
        from train_embedder import to_columns
    except ImportError as e:
        print(f"  skipped ({e})")
        return
    rows = [{"query": "q1", "positive": "p1", "negatives": ["n1", "n2"]},
            {"query": "q2", "positive": "p2", "negatives": ["n3"]}]
    cols = to_columns(rows, 2)
    assert cols == {"anchor": ["q1"], "positive": ["p1"], "negative_1": ["n1"], "negative_2": ["n2"]}, cols
    assert to_columns(rows, 0) == {"anchor": ["q1", "q2"], "positive": ["p1", "p2"]}
    print("✓ training columns passed")


if __name__ == "__main__":
    test_helpers()
    test_holdout()
    test_build_pairs()
    test_train_columns()
    print("\nAll citation-pair smoke tests passed.")
