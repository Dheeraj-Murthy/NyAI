"""
Smoke Test for the citation resolver (no database needed)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.citation_resolver import (
    CitationResolver,
    case_name_alias,
    extract_citation_header,
    normalize_case_name,
    clean_party_name,
    parse_reporter_citations,
)

JUDGMENTS = [
    {
        "id": 1,
        "petitioner": "THE STATE OF BOMBAY AND ANOTHER",
        "respondent": "F. N. BALSARA",
        "date_of_judgment": "1951-05-25",
        "header_text": "PETITIONER:\nTHE STATE OF BOMBAY\nCITATION:\n 1951 AIR 318   1951 SCR 682\n\nACT:",
    },
    {
        "id": 2,
        "petitioner": "HIS HOLINESS KESAVANANDA BHARATI SRIPADAGALVARU",
        "respondent": "STATE OF KERALA AND ANR.",
        "date_of_judgment": "1973-04-24",
        "header_text": "CITATION:\n 1973 AIR 1461  1973 SCR  (4) 1 \n CITATOR INFO :",
    },
    {
        "id": 3,
        "petitioner": "N. V. SHANMUGHAM AND CO.",
        "respondent": "COMMISSIONER OF INCOME-TAX, MADRAS",
        "date_of_judgment": "1971-03-01",
    },
    {
        "id": 4,
        "petitioner": "SUBHA LARA",
        "respondent": "STATE OF PUNJAB(AND CONNECTED APPEALS)",
        "date_of_judgment": "1960-01-01",
    },
]


def test_reporter_parsing():
    print("--- Testing reporter citation parsing ---")
    cases = {
        "AIR 1987 SC 1086": "air|1987|sc|1086",
        "A.I.R. 1953 S.C. 75": "air|1953|sc|75",
        "[1950] S.C.R. 940": "scr|1950||940",
        "[1955] 1 S.C.R. 777": "scr|1955|1|777",
        "(1992) 1 SCC 588": "scc|1992|1|588",
        "1955 SCR (1) 777": "scr|1955|1|777",
        "1953 AIR 75": "air|1953|sc|75",
    }
    for text, key in cases.items():
        got = [c.key for c in parse_reporter_citations(text)]
        assert got == [key], f"{text!r}: expected {key}, got {got}"
    header = extract_citation_header(JUDGMENTS[1]["header_text"])
    assert header == "1973 AIR 1461 1973 SCR (4) 1", header
    print("✓ reporter parsing passed")


def test_name_normalisation():
    print("--- Testing case name normalisation ---")
    assert normalize_case_name("The State of M.P. & Ors. Vs. Ram Kumar") == "state of madhya pradesh v ram kumar"
    assert normalize_case_name("UOI v. M/s. Tata Steel Ltd.") == "union of india v tata steel ltd"
    # Role labels on newer party lines are dropped, only at the end of a party.
    assert normalize_case_name("P.EKNATH ...Appellant(s) Vs. Y. Amaranatha Reddy ...Respondent(s)") == "p eknath v y amaranatha reddy"
    assert normalize_case_name("Hirabai (D) Thr. L.Rs. & Ors. ….Appellant (s)") == "hirabai d thr l rs"
    assert normalize_case_name("SHRI PRABHAT CHANDRA DAS …Petitioner v. Union of India ...Respondents") == "shri prabhat chandra das v union of india"
    assert normalize_case_name("Petitioners Association v. State") == "petitioners association v state"
    assert clean_party_name("M/s Moti Ratan Estate ..Respondents WITH CIVIL APPEAL NO. 6998 OF 2019 (Arising from SLP") == "M/s Moti Ratan Estate ..Respondents"
    assert case_name_alias("THE STATE OF BOMBAY", "BHANJI MUNJI AND ANOTHER.OCTOBER 12, 1954.[MEHR CHAND") == "state of bombay v bhanji munji"
    assert case_name_alias("In re SANT RAM", "RESPONDENT:") is None
    print("✓ normalisation passed")


def test_resolution():
    print("--- Testing citation resolution ---")
    r = CitationResolver(JUDGMENTS)
    expect = [
        ("[1951] S.C.R. 682", 1, "reporter"),
        ("A.I.R. 1973 S.C. 1461", 2, "reporter"),
        ("[1973] Supp. S.C.R. 1", None, "unresolved"),
        ("relied upon State of Bombay v. F. N. Balsara (1951) where", 1, "exact_name"),
        ("In State of Bombay v. Balsara the court", 1, "fuzzy"),
        ("Following Kesavananda Bharati v. State of Kerala, we hold", 2, "fuzzy"),
        # Shares only generic words with judgment 3.
        ("Ltd. v. Commissioner of Income", None, "unresolved"),
        # Shares one name word with judgment 4 but names a different person.
        ("In Riha Subha v. State of Punjab", None, "unresolved"),
        ("A. K. Gopalan v. State of Madras", None, "unresolved"),
    ]
    for text, jid, method in expect:
        res = r.resolve(text)
        assert (res.judgment_id, res.method) == (jid, method), f"{text!r}: got {res}"
    assert r.resolve("1951 AIR 318", source_id=1).method == "self"

    rebuilt = CitationResolver.from_alias_rows(r.alias_rows(), r.years)
    for text, jid, method in expect:
        assert rebuilt.resolve(text).judgment_id == jid, f"rebuilt resolver differs on {text!r}"
    print("✓ resolution passed")


if __name__ == "__main__":
    test_reporter_parsing()
    test_name_normalisation()
    test_resolution()
    print("\nAll citation resolver smoke tests passed.")
