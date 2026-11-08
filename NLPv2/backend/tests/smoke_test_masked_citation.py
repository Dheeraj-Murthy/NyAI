#!/usr/bin/env python3
"""Smoke tests for the masked-citation gold-set builder.

Offline: fixture judgments and edges, no DB.

Run: cd NLPv2/backend && python tests/smoke_test_masked_citation.py
"""
import os
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

from masked_citation_gold import (  # noqa: E402
    body_start,
    build_gold_set,
    build_record,
    eligible_edges,
    find_occurrence,
    mask_passage,
    passage_window,
    quote_overlap,
    sample_order,
)
from judgment_eval import exclude_and_cut, in_degree_bucket  # noqa: E402

FILLER = " ".join(f"word{i}" for i in range(200))

# Judgment 1 cites 2 (Maneka Gandhi) and 3 (Sibbia) in one paragraph; its
# header's CITATOR INFO mentions a later case 4, which must not be used.
SOURCE = f"""PETITIONER:
RAM KUMAR
CITATION:
 1985 AIR 100  1985 SCR (1) 1
CITATOR INFO :
 R          1990 SC 555     (12)
HEADNOTE:
The appellant was detained. Gurbaksh Singh Sibbia v. State of Punjab, [1980] 3 S.C.R. 383 considered.
JUDGMENT:
The facts are these. {FILLER}

The procedure established by law for depriving a person of personal liberty must be fair,
just and reasonable, not fanciful, oppressive or arbitrary, as this Court explained in
Maneka Gandhi v. Union of India, [1978] 2 S.C.R. 621. The discretion to grant anticipatory
bail is wide and is not to be cut down by judicially created conditions, see Gurbaksh Singh
Sibbia v. State of Punjab, [1980] 3 S.C.R. 383, and Maneka's case (supra) applies with full force
to the arrest of a person who has not yet been convicted of any offence by a competent court.

A short paragraph citing Maneka Gandhi v. Union of India, [1978] 2 S.C.R. 621 again.

The appeal is allowed.
"""

TARGET_QUOTED = """JUDGMENT:
Liberty is precious and the procedure established by law for depriving a person of
personal liberty must be fair, just and reasonable, not fanciful, oppressive or arbitrary."""

META = {
    1: {"case_name": "ram kumar v state of haryana", "year": 1985, "in_degree": 0},
    2: {"case_name": "maneka gandhi v union of india", "year": 1978, "in_degree": 120},
    3: {"case_name": "gurbaksh singh sibbia v state of punjab", "year": 1980, "in_degree": 30},
    4: {"case_name": "later case v state", "year": 1990, "in_degree": 1},
    5: {"case_name": "other appellant v state of kerala", "year": 1986, "in_degree": 0},
}
EDGES = [
    {"source_judgment_id": 1, "target_judgment_id": 2, "cited_text": "Maneka Gandhi v. Union of India", "resolution_method": "exact_name"},
    {"source_judgment_id": 1, "target_judgment_id": 3, "cited_text": "[1980] 3 S.C.R. 383", "resolution_method": "reporter"},
    {"source_judgment_id": 1, "target_judgment_id": 4, "cited_text": "1990 SC 555", "resolution_method": "reporter"},
    {"source_judgment_id": 1, "target_judgment_id": 1, "cited_text": "1985 AIR 100", "resolution_method": "reporter"},
    {"source_judgment_id": 5, "target_judgment_id": 2, "cited_text": "Maneka Gandhi v. Union of India", "resolution_method": "fuzzy"},
]
TEXTS = {1: SOURCE, 2: "JUDGMENT:\nUnrelated text about passports.", 3: "JUDGMENT:\nAbout bail.", 4: "", 5: ""}


def test_text_helpers():
    print("--- Testing text helpers ---")
    start = body_start(SOURCE)
    assert SOURCE[start - 9 : start] == "JUDGMENT:"
    assert body_start("no heading here") == 0
    # The headnote mention comes first but the body search skips it.
    occ = find_occurrence(SOURCE, "[1980] 3 S.C.R. 383", start)
    assert occ and occ[0] > start
    # Line breaks inside a citation still match.
    assert find_occurrence("in Gurbaksh Singh\nSibbia v. State", "Gurbaksh Singh Sibbia v. State")
    occ = find_occurrence(SOURCE, "Maneka Gandhi v. Union of India", start)
    lo, hi = passage_window(SOURCE, occ, 150)
    assert SOURCE[lo:hi].startswith("The procedure established") and SOURCE[lo:hi].endswith("competent court.")
    lo, hi = passage_window(SOURCE, occ, 12)
    assert len(SOURCE[lo:hi].split()) == 12 and "Maneka" in SOURCE[lo:hi]
    print("✓ text helpers passed")


def test_masking():
    print("--- Testing masking ---")
    masked = mask_passage(
        "as held in Maneka Gandhi v. Union of India, [1978] 2 S.C.R. 621, and Maneka's case (supra) "
        "and (1980) 2 SCC 565 and AIR 1978 SC 597.",
        {"maneka", "gandhi"},
    )
    for gone in ("Maneka", "Gandhi", "S.C.R", "SCC", "AIR", "supra", "1978", "565"):
        assert gone not in masked, (gone, masked)
    assert masked.startswith("as held in") and "case" in masked, masked
    assert quote_overlap(" ".join(f"w{i}" for i in range(20)), " ".join(f"w{i}" for i in range(20))) == 1.0
    assert quote_overlap("a b c", "a b c") == 0.0  # shorter than one shingle
    print("✓ masking passed")


def test_build_record():
    print("--- Testing record building ---")
    by_source = [e for e in EDGES if e["source_judgment_id"] == 1]
    rec, reason = build_record(EDGES[0], SOURCE, TEXTS, by_source, META)
    assert reason is None, reason
    # Both cases cited in the paragraph are relevant; the later CITATOR case and self-cite are not.
    assert rec["relevant_judgment_ids"] == [2, 3], rec["relevant_judgment_ids"]
    assert rec["primary_target_id"] == 2 and rec["exclude_judgment_ids"] == [1]
    for leak in ("Maneka", "Gandhi", "Sibbia", "Gurbaksh", "S.C.R", "supra"):
        assert leak not in rec["query"], (leak, rec["query"])
    assert "procedure established by law" in rec["query"]
    assert len(rec["query_short"].split()) < len(rec["query"].split())

    # A passage that quotes the cited judgment is dropped.
    quoting = {**TEXTS, 2: TARGET_QUOTED}
    rec, reason = build_record(EDGES[0], SOURCE, quoting, by_source, META)
    assert rec is None and reason == "passage quotes the cited judgment", reason
    rec, _ = build_record(EDGES[0], SOURCE, quoting, by_source, META, max_quote_overlap=1.0)
    assert rec is not None and rec["quote_overlap"] > 0.2

    # Short paragraph -> too short after masking.
    short_edge = {**EDGES[0], "cited_text": "[1978] 2 S.C.R. 621 again"}
    rec, reason = build_record(short_edge, SOURCE, TEXTS, by_source, META)
    assert reason == "passage too short after masking", reason

    rec, reason = build_record({**EDGES[0], "cited_text": "Nowhere v. Never"}, SOURCE, TEXTS, by_source, META)
    assert reason == "citation not found in the judgment body"
    print("✓ record building passed")


def test_sampling():
    print("--- Testing edge filtering and sampling ---")
    ok = eligible_edges(EDGES, META)
    # Drops the later-dated target, the self-cite and the fuzzy match.
    assert [(e["source_judgment_id"], e["target_judgment_id"]) for e in ok] == [(1, 2), (1, 3)]
    assert sample_order(EDGES, 7) == sample_order(list(reversed(EDGES)), 7)

    records, skipped = build_gold_set(EDGES, META, lambda ids: {i: TEXTS[i] for i in ids}, n=10, seed=1)
    assert len(records) == 1, records  # one record per citing judgment
    assert records[0]["source_judgment_id"] == 1
    print("✓ sampling passed")


def test_eval_helpers():
    print("--- Testing eval helpers ---")
    chunks = [{"judgment_id": j} for j in (1, 2, 1, 3, 4)]
    assert [c["judgment_id"] for c in exclude_and_cut(chunks, [1], 2)] == [2, 3]
    assert in_degree_bucket(0) == "rarely cited (<5)" and in_degree_bucket(50) == "heavily cited (50+)"
    print("✓ eval helpers passed")


if __name__ == "__main__":
    test_text_helpers()
    test_masking()
    test_build_record()
    test_sampling()
    test_eval_helpers()
    print("\nAll masked-citation smoke tests passed.")
