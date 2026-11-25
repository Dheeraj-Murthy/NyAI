#!/usr/bin/env python3
"""Smoke tests for the FIRE 2019 AILA benchmark adapter.

Offline: a tiny fixture dataset in a temp dir, no DB, no models.

Run: cd NLPv2/backend && python tests/smoke_test_aila.py
"""
import os
import sys
import tempfile

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

from aila_benchmark import (  # noqa: E402
    BM25,
    build_gold_records,
    case_header,
    load_docs,
    load_qrels,
    load_queries,
    map_cases,
    query_ids,
    rank,
    rrf,
    run_pool,
    trec_metrics,
)
from graph.citation_resolver import Resolution  # noqa: E402
from judgment_eval import group_of  # noqa: E402


def write_fixture(root: str) -> None:
    os.makedirs(os.path.join(root, "Object_casedocs"))
    os.makedirs(os.path.join(root, "Object_statutes"))
    docs = {
        "C1": "Masud Khan v State Of Uttar Pradesh\nSupreme Court of India\n\n26 September 1973\n"
              "detention under the foreigners internment order habeas corpus",
        "C2": "Ram Kumar v State of Punjab\nSupreme Court of India\n\n3 March 1980\n"
              "bail anticipatory bail dowry death",
        "C3": "Unmatched Party v Someone\nSupreme Court of India\n\n1 January 1990\nland acquisition compensation",
    }
    for d, text in docs.items():
        with open(os.path.join(root, "Object_casedocs", f"{d}.txt"), "w") as f:
            f.write(text)
    with open(os.path.join(root, "Object_statutes", "S1.txt"), "w") as f:
        f.write("Title: Punishment for dowry death\nDesc: Whoever commits dowry death shall be punished")
    with open(os.path.join(root, "Query_doc.txt"), "w") as f:
        f.write("AILA_Q1||The accused sought anticipatory bail in a dowry death case.\n")
        f.write("AILA_Q12||A foreign national was detained under an internment order.\n")
    with open(os.path.join(root, "relevance_judgments_priorcases.txt"), "w") as f:
        for q, d, r in [("AILA_Q1", "C2", 1), ("AILA_Q1", "C1", 0), ("AILA_Q1", "C3", 0),
                        ("AILA_Q12", "C1", 1), ("AILA_Q12", "C3", 1), ("AILA_Q12", "C2", 0)]:
            f.write(f"{q} Q0 {d} {r}\n")
    with open(os.path.join(root, "relevance_judgments_statutes.txt"), "w") as f:
        f.write("AILA_Q1 Q0 S1 1\nAILA_Q12 Q0 S1 0\n")


def test_trec_metrics():
    print("--- Testing trec_eval-style metrics ---")
    judged = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0}
    m = trec_metrics(["b", "a", "x", "c"], judged)
    assert abs(m["map"] - (1 / 2 + 2 / 4) / 2) < 1e-9
    assert m["recip_rank"] == 0.5 and m["P@10"] == 0.2
    # bpref: a has 1 judged non-relevant above (b), c has 1 (b; x is unjudged).
    assert abs(m["bpref"] - ((1 - 1 / 2) + (1 - 1 / 2)) / 2) < 1e-9
    assert trec_metrics(["a"], {"a": 0})["map"] == 0.0
    print("✓ metrics passed")


def test_loading_and_pool():
    print("--- Testing loaders and BM25 pool run ---")
    with tempfile.TemporaryDirectory() as root:
        write_fixture(root)
        queries, qrels = load_queries(root), load_qrels(root, "cases")
        assert query_ids(queries, qrels, "test") == ["AILA_Q12"]
        assert query_ids(queries, qrels, "train") == ["AILA_Q1"]
        assert query_ids(queries, qrels, "all") == ["AILA_Q1", "AILA_Q12"]
        assert load_docs(root, "statutes")["S1"].startswith("Punishment for dowry death\nWhoever")

        bm25 = BM25(load_docs(root, "cases"))
        assert rank(bm25.scores(queries["AILA_Q1"]))[0] == "C2"
        report = run_pool(root, "cases", "bm25", False, "all", os.path.join(root, "work"))
        assert report["config"]["n_queries"] == 2
        assert [d for d, _ in report["run"]["AILA_Q1"]][0] == "C2"
        assert report["per_query"]["AILA_Q1"]["map"] == 1.0
        try:
            run_pool(root, "cases", "bm25", True, "all", root)
        except SystemExit:
            pass
        else:
            raise AssertionError("--rerank with bm25 accepted")
    print("✓ loaders and pool run passed")


def test_rrf():
    print("--- Testing RRF ---")
    fused = rrf([["a", "b"], ["b", "c"]])
    assert rank(fused) == ["b", "a", "c"]
    print("✓ RRF passed")


def test_export_mapping():
    print("--- Testing corpus mapping and gold records ---")
    with tempfile.TemporaryDirectory() as root:
        write_fixture(root)
        docs = load_docs(root, "cases")
        assert case_header(docs["C1"]) == ("Masud Khan v State Of Uttar Pradesh", 1973)

        class FakeResolver:
            seen = []

            def resolve(self, text):
                FakeResolver.seen.append(text)
                if text.startswith("Masud Khan"):
                    return Resolution(101, "exact_name", 100.0)
                if text.startswith("Ram Kumar"):
                    return Resolution(102, "fuzzy", 80.0)  # not trusted
                return Resolution(None, "unresolved")

        mapping, methods = map_cases(docs, FakeResolver())
        assert mapping == {"C1": 101} and methods == {"exact_name": 1, "fuzzy": 1, "unresolved": 1}
        assert "Masud Khan v State Of Uttar Pradesh (1973)" in FakeResolver.seen

        records, skipped = build_gold_records(load_queries(root), load_qrels(root, "cases"), mapping, {101: "masud khan v state of uttar pradesh"})
        assert [r["id"] for r in records] == ["AILA_Q12"] and skipped["no relevant case found in the corpus"] == 1
        r = records[0]
        assert r["relevant_judgment_ids"] == [101] and r["aila_relevant_unmapped"] == ["C3"]
        assert r["split"] == "test" and group_of(r, "aila") == "test"
        assert r["relevant_case_names"] == {"101": "masud khan v state of uttar pradesh"}
    print("✓ mapping passed")


if __name__ == "__main__":
    test_trec_metrics()
    test_loading_and_pool()
    test_rrf()
    test_export_mapping()
    print("\nAll AILA smoke tests passed.")
