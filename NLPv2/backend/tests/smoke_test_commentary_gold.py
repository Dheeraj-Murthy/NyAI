#!/usr/bin/env python3
"""Smoke tests for the commentary gold-set builder and its metrics.

Offline: no DB, no LLM, no PDFs. Uses a fixture resolver and a stub backend.

Run: cd NLPv2/backend && python tests/smoke_test_commentary_gold.py
"""
import math
import os
import sys

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

from graph.citation_resolver import CitationResolver  # noqa: E402
from commentary_gold import (  # noqa: E402
    build_gold,
    case_grounded,
    distinctive_name_words,
    has_citation_hint,
    normalize_ocr,
    parse_llm_items,
    propose_for_pages,
    question_leaks,
    resolve_case,
)
from metrics import (  # noqa: E402
    aggregate_multi,
    bootstrap_ci,
    dedupe,
    fraction_recall_at_k,
    ndcg_multi_at_k,
)

JUDGMENTS = [
    {
        "id": 10,
        "petitioner": "MANEKA GANDHI",
        "respondent": "UNION OF INDIA AND ANOTHER",
        "date_of_judgment": "1978-01-25",
        "header_text": "CITATION:\n 1978 AIR 597  1978 SCR (2) 621\n\nBENCH:",
    },
    {
        "id": 11,
        "petitioner": "GURBAKSH SINGH SIBBIA ETC",
        "respondent": "STATE OF PUNJAB",
        "date_of_judgment": "1980-04-09",
        "header_text": "CITATION:\n 1980 AIR 1632  1980 SCR (3) 383\n\nBENCH:",
    },
]

PAGE = {
    "source_id": "lcr268",
    "page": 42,
    "extraction_method": "text",
    "text": (
        "3.12 Anticipatory bail is not to be denied merely because the offence is serious; "
        "the discretion is wide and must be exercised judiciously.12\n\n"
        "12 Gurbaksh Singh Sibbia v. State of Punjab, (1980) 2 SCC 565; see also "
        "Maneka Gandhi v. Union of India, AIR 1978 SC 597; Sita Ram v. Malkiat Singh, AIR 1956 Pep 30."
    ),
}


def test_text_helpers():
    print("--- Testing OCR normalisation and page filter ---")
    assert normalize_ocr("Afzalur Rahman v. Emperor, AUR 1943 FC 18") == "Afzalur Rahman v. Emperor, AIR 1943 FC 18"
    assert normalize_ocr("Sita Ram Chandu Lail y, Malikar Singh") == "Sita Ram Chandu Lail v. Malikar Singh"
    assert has_citation_hint(PAGE["text"])
    assert not has_citation_hint("This chapter discusses the history of the provision.")
    assert distinctive_name_words("Afzalur Rahman v. Emperor, AIR 1943 FC 18") == {"afzalur", "rahman"}
    assert distinctive_name_words("Maneka Gandhi v. Union of India, (1978) 1 SCC 248") == {"maneka", "gandhi"}
    print("✓ text helpers passed")


def test_llm_parsing():
    print("--- Testing LLM output parsing ---")
    raw = 'Sure:\n```json\n[{"proposition": "P", "question": "Q?", "cases": "A v. B"}]\n```'
    assert parse_llm_items(raw) == [{"proposition": "P", "question": "Q?", "cases": ["A v. B"]}]
    assert parse_llm_items("no json here") == []
    assert parse_llm_items("[{broken") == []
    assert parse_llm_items("[]") == []
    print("✓ parsing passed")


def test_grounding_and_leaks():
    print("--- Testing grounding and leak checks ---")
    assert case_grounded("Gurbaksh Singh Sibbia v. State of Punjab, (1980) 2 SCC 565", PAGE["text"])
    assert case_grounded("Maneka Gandhi v. Union of India", PAGE["text"])
    # An invented case is not on the page.
    assert not case_grounded("Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273", PAGE["text"])
    names = distinctive_name_words("Gurbaksh Singh Sibbia v. State of Punjab")
    assert question_leaks("Can anticipatory bail be refused only because the offence is grave?", names) is None
    assert question_leaks("What did Sibbia hold on anticipatory bail?", names).startswith("question names")
    assert question_leaks("What was held in (1980) 2 SCC 565?", names) == "question contains a citation"
    print("✓ grounding and leak checks passed")


def test_resolve_fallback():
    print("--- Testing reporter -> case-name fallback ---")
    r = CitationResolver(JUDGMENTS)
    # SCC is not in the JUDIS header, so the reporter alone fails ...
    assert r.resolve("Gurbaksh Singh Sibbia v. State of Punjab, (1980) 2 SCC 565").judgment_id is None
    # ... and the name fallback finds it.
    assert resolve_case(r, "Gurbaksh Singh Sibbia v. State of Punjab, (1980) 2 SCC 565").judgment_id == 11
    assert resolve_case(r, "Maneka Gandhi v. Union of India, AIR 1978 SC 597").judgment_id == 10
    assert resolve_case(r, "Sita Ram v. Malkiat Singh, AIR 1956 Pep 30").judgment_id is None
    print("✓ resolution fallback passed")


class StubBackend:
    model_id = "stub"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, context_block, user_query, system_prompt, **_):
        class R:
            text = self.reply

        return R()


def test_propose():
    print("--- Testing propose end to end (stub LLM) ---")
    reply = """[
      {"proposition": "Anticipatory bail cannot be refused merely because the offence is serious.",
       "question": "Can a court refuse anticipatory bail only on the ground that the offence alleged is serious?",
       "cases": ["Gurbaksh Singh Sibbia v. State of Punjab, (1980) 2 SCC 565",
                 "Maneka Gandhi v. Union of India, AIR 1978 SC 597",
                 "Sita Ram v. Malkiat Singh, AIR 1956 Pep 30"]},
      {"proposition": "Leaky.", "question": "What did the court decide in Maneka Gandhi about bail and liberty?",
       "cases": ["Maneka Gandhi v. Union of India, AIR 1978 SC 597"]},
      {"proposition": "Invented.", "question": "Is arrest automatic for offences punishable up to seven years?",
       "cases": ["Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273"]}
    ]"""
    accepted, rejected = propose_for_pages([PAGE], StubBackend(reply), CitationResolver(JUDGMENTS))
    assert len(accepted) == 1, accepted
    c = accepted[0]
    assert c["relevant_judgment_ids"] == [10, 11], c
    assert [u["text"] for u in c["unresolved_cases"]] == ["Sita Ram v. Malkiat Singh, AIR 1956 Pep 30"]
    reasons = sorted(r["reason"].split(" (")[0] for r in rejected)
    assert reasons == ["no cited case found in the page text", "question names a cited party"], reasons

    _, rejected = propose_for_pages([PAGE], StubBackend("I cannot help"), CitationResolver(JUDGMENTS))
    assert rejected[0]["reason"] == "no items parsed"

    gold = build_gold(accepted, {c["candidate_id"]: {"decision": "accept", "corrected_query": ""}},
                      [{"source_id": "lcr268"}, {"source_id": "unused"}], "test", False)
    assert gold["n"] == 1 and gold["n_reviewed"] == 1 and gold["sources"] == [{"source_id": "lcr268"}]
    assert gold["records"][0]["relevant_case_names"]["11"] == "gurbaksh singh sibbia v state of punjab"
    assert build_gold(accepted, {}, [], "test", False)["n"] == 0  # unreviewed excluded by default
    trial = build_gold(accepted, {}, [], "test", True)
    assert trial["n"] == 1 and trial["n_reviewed"] == 0
    edited = build_gold(accepted, {c["candidate_id"]: {"decision": "accept", "corrected_query": "New?"}}, [], "t", False)
    assert edited["records"][0]["query"] == "New?" and edited["records"][0]["query_edited"]
    print("✓ propose / finalize passed")


def test_metrics():
    print("--- Testing multi-relevant metrics ---")
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert fraction_recall_at_k([1, 2], [5, 1, 6], 3) == 0.5
    assert ndcg_multi_at_k([1, 2], [1, 2, 3], 3) == 1.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_multi_at_k([1, 2], [9, 1, 8], 3) - expected) < 1e-9
    agg = aggregate_multi([([1, 2], [9, 1, 8]), ([4], [5, 6])], [1, 3])
    assert agg["hit@1"] == 0.0 and agg["hit@3"] == 0.5 and agg["recall@3"] == 0.25
    assert abs(agg["mrr"] - 0.25) < 1e-9
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65, (lo, hi)
    print("✓ metrics passed")


if __name__ == "__main__":
    test_text_helpers()
    test_llm_parsing()
    test_grounding_and_leaks()
    test_resolve_fallback()
    test_propose()
    test_metrics()
    print("\nAll commentary gold smoke tests passed.")
