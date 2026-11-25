#!/usr/bin/env python3
"""Build a case-name / citation lookup gold set.

Can a user who already knows which case they want get it back? Real queries
often just name the case ("Maneka Gandhi v. Union of India", "the Maneka
Gandhi case") or quote a citation ("AIR 1978 SC 597"). Chat retrieval has
no lookup path for these — they go through dense + BM25 like any question —
so this measures how well that works, and judgment_eval.py's
--retriever lookup scores the Graph page's alias search as the baseline a
dedicated lookup would give.

No human labelling and no LLM: the answer is the judgment whose own header
names those parties / carries that citation.

Query variants, one record each, scored as separate groups:
  full_name   "Maneka Gandhi v. Union of India"
  question    "What was held in Maneka Gandhi v. Union of India?"
  party_name  "Maneka Gandhi case" — the more distinctive party
  citation    "AIR 1978 SC 597" / "(1978) 2 SCR 621", from the judgment's
              own CITATION: header (JUDIS) or stored reporters

Relevant judgments: every corpus judgment the query names equally well —
the same normalised case name, the same party on either side, or the same
reporter citation (connected appeals share one). Queries matching more than
--max-relevant judgments are dropped as too ambiguous to score (e.g. a
"State of Punjab case" query).

Party names in JUDIS headers are often upper case; queries are re-cased the
way a user would type them, so casing doesn't decide the result.

Usage:
    cd NLPv2/backend
    python benchmark/case_lookup_gold.py --n 200 --seed 1 --version v1
    python benchmark/judgment_eval.py --gold benchmark/data/case_lookup_gold_v1.json
    python benchmark/judgment_eval.py --gold benchmark/data/case_lookup_gold_v1.json --retriever lookup
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_benchmark_dir)
sys.path.insert(0, _backend_dir)

from graph.citation_resolver import (  # noqa: E402
    STOPWORDS,
    CitationResolver,
    _year_of,
    case_name_alias,
    clean_party_name,
    normalize_case_name,
)

DEFAULT_OUTPUT_DIR = os.path.join(_benchmark_dir, "data")
VARIANTS = ("full_name", "question", "party_name", "citation")
DEFAULT_MAX_RELEVANT = 5
HEADER_CHARS = 6000  # the CITATION: header sits in the first few KB

_LOWER_WORDS = {"of", "and", "the", "for", "in", "by", "on", "at", "to", "v"}
_KEEP_UPPER = {"ltd", "pvt", "co", "llp"}


def display_case(name: str) -> str:
    """Re-case a party name the way a user would type it: "STATE OF U.P."
    -> "State of U.P.", "M/S. TATA IRON AND STEEL CO. LTD." -> "M/S. Tata
    Iron and Steel Co. Ltd.". Mixed-case names are left alone."""
    name = " ".join(name.split()).strip(" ,-")
    if not name or (name != name.upper() and name != name.lower()):
        return name
    out = []
    for i, word in enumerate(name.split()):
        bare = re.sub(r"[^a-z]", "", word.lower())
        if i and bare in _LOWER_WORDS:
            out.append(word.lower())
        elif "." in word.rstrip(".") or "/" in word:  # initials, "M/S."
            out.append(word.upper())
        else:
            out.append(word[:1].upper() + word[1:].lower())
    return " ".join(out)


def format_citation(key: str) -> Optional[str]:
    """A resolver reporter key as a user would write it:
    "air|1978|sc|597" -> "AIR 1978 SC 597", "scr|1978|2|621" -> "(1978) 2 SCR 621",
    "scr|1978||621" -> "[1978] SCR 621"."""
    parts = key.split("|")
    if len(parts) != 4:
        return None
    rep, year, mid, page = parts
    if rep == "air":
        return f"AIR {year} {mid.upper()} {page}"
    if rep in ("scr", "scc"):
        if mid:
            return f"({year}) {mid} {rep.upper()} {page}"
        return f"[{year}] {rep.upper()} {page}"
    return None


def distinctive(norm_party: str) -> Set[str]:
    return {t for t in norm_party.split() if t not in STOPWORDS and not t.isdigit() and len(t) > 1}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def prepare(judgments: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Set[int]]]]:
    """Per judgment: cleaned parties, alias and own reporter keys; plus the
    indexes used to find every judgment a query names equally well."""
    prepared = []
    index: Dict[str, Dict[str, Set[int]]] = {"alias": defaultdict(set), "party": defaultdict(set), "reporter": defaultdict(set)}
    for j in judgments:
        pet, res = clean_party_name(j.get("petitioner")), clean_party_name(j.get("respondent"))
        # From the raw columns, exactly as judgment_eval.validate_gold recomputes it.
        alias = case_name_alias(j.get("petitioner"), j.get("respondent"))
        if not alias:
            continue
        keys = sorted({c.key for c in CitationResolver._own_reporter_citations(
            {"header_text": j.get("header_text"), "citations": j.get("citations")})})
        p = {
            "id": j["id"],
            "petitioner": pet,
            "respondent": res,
            "alias": alias,
            "year": _year_of(j.get("date_of_judgment")),
            "reporter_keys": keys,
        }
        prepared.append(p)
        index["alias"][alias].add(j["id"])
        for party in (pet, res):
            index["party"][normalize_case_name(party)].add(j["id"])
        for k in keys:
            index["reporter"][k].add(j["id"])
    return prepared, index


def pick_party(p: Dict[str, Any], index: Dict[str, Dict[str, Set[int]]]) -> Optional[str]:
    """The party a user would name the case by: the one appearing in fewer
    judgments, among parties with a distinctive word (so not "Union of India")."""
    options = []
    for party in (p["petitioner"], p["respondent"]):
        norm = normalize_case_name(party)
        if distinctive(norm):
            options.append((len(index["party"][norm]), party))
    return min(options)[1] if options else None


def records_for(
    p: Dict[str, Any], index: Dict[str, Dict[str, Set[int]]], max_relevant: int, rng: random.Random
) -> Tuple[List[Dict[str, Any]], Counter]:
    skipped: Counter = Counter()
    title = f"{display_case(p['petitioner'])} v. {display_case(p['respondent'])}"
    candidates: List[Tuple[str, str, Set[int]]] = []

    by_name = index["alias"][p["alias"]]
    candidates.append(("full_name", title, by_name))
    candidates.append(("question", f"What was held in {title}?", by_name))

    party = pick_party(p, index)
    if party:
        candidates.append(("party_name", f"{display_case(party)} case", index["party"][normalize_case_name(party)]))
    else:
        skipped["party_name: no distinctive party"] += 1

    keys = [k for k in p["reporter_keys"] if format_citation(k)]
    if keys:
        key = rng.choice(keys)
        candidates.append(("citation", format_citation(key), index["reporter"][key]))
    else:
        skipped["citation: judgment has no reporter citation"] += 1

    records = []
    for variant, query, relevant in candidates:
        if len(relevant) > max_relevant:
            skipped[f"{variant}: matches more than {max_relevant} judgments"] += 1
            continue
        records.append({
            "id": f"cl-{p['id']}-{variant}",
            "variant": variant,
            "query": query,
            "target_judgment_id": p["id"],
            "relevant_judgment_ids": sorted(relevant),
            "relevant_case_names": {},  # filled in by build_gold_set
        })
    return records, skipped


def build_gold_set(
    judgments: Iterable[Dict[str, Any]], n: int, seed: int, max_relevant: int = DEFAULT_MAX_RELEVANT
) -> Tuple[List[Dict[str, Any]], Counter]:
    prepared, index = prepare(judgments)
    alias_of = {p["id"]: p["alias"] for p in prepared}
    rng = random.Random(seed)
    order = sorted(prepared, key=lambda p: p["id"])
    rng.shuffle(order)
    records: List[Dict[str, Any]] = []
    skipped: Counter = Counter()
    for p in order[:n]:
        recs, skip = records_for(p, index, max_relevant, rng)
        skipped.update(skip)
        for r in recs:
            r["relevant_case_names"] = {str(j): alias_of[j] for j in r["relevant_judgment_ids"]}
        records.extend(recs)
    return records, skipped


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


def _default_db_connection_string() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def load_from_db(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, petitioner, respondent, date_of_judgment, citations, left(judgment_text, %s) "
            "FROM judgments",
            (HEADER_CHARS,),
        )
        return [
            {"id": r[0], "petitioner": r[1], "respondent": r[2], "date_of_judgment": r[3],
             "citations": r[4], "header_text": r[5]}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a case-name / citation lookup gold set.")
    parser.add_argument("--n", type=int, default=200, help="Judgments to sample (up to 4 records each)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--version", required=True, help="e.g. v1 -> case_lookup_gold_v1.json")
    parser.add_argument("--max-relevant", type=int, default=DEFAULT_MAX_RELEVANT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"case_lookup_gold_{args.version}.json")
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists; gold sets are frozen once written. Use a new --version.")

    import psycopg2

    conn = psycopg2.connect(_default_db_connection_string())
    try:
        judgments = load_from_db(conn)
    finally:
        conn.close()

    records, skipped = build_gold_set(judgments, args.n, args.seed, args.max_relevant)
    gold = {
        "kind": "case_lookup",
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n": len(records),
        "config": {"n_judgments_sampled": args.n, "max_relevant": args.max_relevant, "variants": list(VARIANTS)},
        "corpus_judgment_count": len(judgments),
        "by_variant": dict(Counter(r["variant"] for r in records)),
        "skipped": dict(skipped),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {out}")
    print(f"By variant: {gold['by_variant']}")
    for reason, count in skipped.most_common():
        print(f"  skipped {count}: {reason}")


if __name__ == "__main__":
    main()
