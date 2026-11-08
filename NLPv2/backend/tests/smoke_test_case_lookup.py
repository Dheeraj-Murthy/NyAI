#!/usr/bin/env python3
"""Smoke tests for the case-name / citation lookup gold-set builder.

Offline: fixture judgments with JUDIS-style headers, no DB.

Run: cd NLPv2/backend && python tests/smoke_test_case_lookup.py
"""
import os
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

from case_lookup_gold import build_gold_set, display_case, format_citation, prepare, records_for  # noqa: E402
from graph.citation_resolver import case_name_alias  # noqa: E402
from judgment_eval import group_of  # noqa: E402

import random  # noqa: E402


def header(cit: str) -> str:
    return f"PETITIONER:\nX\n\nCITATION:\n {cit}\n CITATOR INFO :\n R 1990 AIR 1 (5)\n\nACT:\n..."


JUDGMENTS = [
    {"id": 1, "petitioner": "MANEKA GANDHI", "respondent": "UNION OF INDIA", "date_of_judgment": "1978-01-25",
     "citations": None, "header_text": header("1978 AIR  597\t\t  1978 SCR  (2) 621")},
    # Connected appeal reported at the same page: the citation names both.
    {"id": 2, "petitioner": "MANEKA GANDHI", "respondent": "UNION OF INDIA AND ANR.", "date_of_judgment": "1978-01-25",
     "citations": None, "header_text": header("1978 AIR  597")},
    {"id": 3, "petitioner": "STATE OF PUNJAB", "respondent": "UNION OF INDIA", "date_of_judgment": "1990-01-01",
     "citations": {"reporters": ["(1990) 3 SCC 12"]}, "header_text": "no header"},
    {"id": 4, "petitioner": "Unknown", "respondent": "UNION OF INDIA", "date_of_judgment": None,
     "citations": None, "header_text": ""},
] + [
    # Six "State of Punjab v <someone>" cases make the party-only query for
    # case 3 ambiguous beyond max_relevant.
    {"id": 10 + i, "petitioner": "STATE OF PUNJAB", "respondent": f"RESPONDENT NAME{i}", "date_of_judgment": None,
     "citations": None, "header_text": ""}
    for i in range(6)
]


def test_formatting():
    print("--- Testing query formatting ---")
    assert display_case("MANEKA GANDHI") == "Maneka Gandhi"
    assert display_case("STATE OF U.P.") == "State of U.P."
    assert display_case("M/S. TATA IRON AND STEEL CO. LTD.") == "M/S. Tata Iron and Steel Co. Ltd."
    assert display_case("Kesavananda Bharati") == "Kesavananda Bharati"  # mixed case kept
    assert format_citation("air|1978|sc|597") == "AIR 1978 SC 597"
    assert format_citation("scr|1978|2|621") == "(1978) 2 SCR 621"
    assert format_citation("scc|1990||12") == "[1990] SCC 12"
    assert format_citation("garbage") is None
    print("✓ formatting passed")


def test_prepare_and_records():
    print("--- Testing record construction ---")
    prepared, index = prepare(JUDGMENTS)
    assert 4 not in {p["id"] for p in prepared}  # one-sided name skipped
    p1 = next(p for p in prepared if p["id"] == 1)
    assert p1["reporter_keys"] == ["air|1978|sc|597", "scr|1978|2|621"]
    assert p1["alias"] == case_name_alias("MANEKA GANDHI", "UNION OF INDIA")

    recs, _ = records_for(p1, index, max_relevant=5, rng=random.Random(0))
    by = {r["variant"]: r for r in recs}
    assert set(by) == {"full_name", "question", "party_name", "citation"}
    assert by["full_name"]["query"] == "Maneka Gandhi v. Union of India"
    assert by["question"]["query"] == "What was held in Maneka Gandhi v. Union of India?"
    # "and anr" is normalised away, so both Maneka Gandhi judgments share the name.
    assert by["full_name"]["relevant_judgment_ids"] == [1, 2]
    assert by["party_name"]["query"] == "Maneka Gandhi case"  # not "Union of India case"
    if by["citation"]["query"] == "AIR 1978 SC 597":
        assert by["citation"]["relevant_judgment_ids"] == [1, 2]
    else:
        assert by["citation"]["query"] == "(1978) 2 SCR 621"
        assert by["citation"]["relevant_judgment_ids"] == [1]

    p3 = next(p for p in prepared if p["id"] == 3)
    recs3, skipped3 = records_for(p3, index, max_relevant=5, rng=random.Random(0))
    by3 = {r["variant"]: r for r in recs3}
    assert "party_name" not in by3  # "State of Punjab case" names 7 judgments
    assert any("party_name: matches more than" in k for k in skipped3)
    assert by3["citation"]["query"] == "(1990) 3 SCC 12"  # from stored reporters
    print("✓ record construction passed")


def test_build_gold_set():
    print("--- Testing build_gold_set ---")
    records, skipped = build_gold_set(JUDGMENTS, n=100, seed=1)
    again, _ = build_gold_set(JUDGMENTS, n=100, seed=1)
    assert records == again  # deterministic for a seed
    assert len({r["id"] for r in records}) == len(records)
    for r in records:
        assert set(r["relevant_case_names"]) == {str(j) for j in r["relevant_judgment_ids"]}
        assert group_of(r, "case_lookup") == r["variant"]
    assert skipped["citation: judgment has no reporter citation"] == 6
    print("✓ build_gold_set passed")


if __name__ == "__main__":
    test_formatting()
    test_prepare_and_records()
    test_build_gold_set()
    print("\nAll case-lookup smoke tests passed.")
