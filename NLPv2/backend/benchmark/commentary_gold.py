#!/usr/bin/env python3
"""Build a judgment-level gold set from government legal commentary.

The synthetic gold set (generate_gold_set.py) writes each question from the
very chunk it is scored against, so it measures paraphrase lookup more than
legal research. Commentary is independent of the corpus: a Law Commission
report or a judicial academy compendium states a proposition of law and
cites the Supreme Court cases behind it. Each such pair becomes one gold
record, with every cited case that is in the corpus counted as relevant.

Official SCR headnotes are deliberately NOT a source: the corpus is JUDIS
text, which already contains each judgment's HEADNOTE: section, so a
question written from a headnote would be scored against its own source.

Stages (each reads the previous stage's file, so any one can be re-run):

  extract   PDFs listed in the manifest -> pages.jsonl
            pdftotext per page; pages with almost no text layer (scanned
            reports are common) are OCR'd with pdftoppm + tesseract and
            cached. Only pages with something citation-like are kept.
  propose   pages.jsonl -> candidates.jsonl (+ rejected.jsonl)
            The LLM reads a page, footnotes included, and returns
            {proposition, question, cases}. Every case it names must be
            findable in the page text (no invented citations), the question
            must not name a cited case or give a citation, and at least one
            case must resolve to a corpus judgment through the same
            CitationResolver the citation graph uses. Needs the DB.
  review    candidates.jsonl -> review.csv, for a law student to mark each
            row accept / reject and optionally rewrite the question.
  finalize  review.csv + candidates.jsonl -> the frozen gold set JSON that
            commentary_eval.py scores against. Freeze it; never regenerate
            it in place, or runs stop being comparable.

Usage:
    cd NLPv2/backend
    python benchmark/commentary_gold.py extract
    python benchmark/commentary_gold.py propose            # local Qwen, needs the DB
    python benchmark/commentary_gold.py review
    python benchmark/commentary_gold.py finalize --version v1

Source PDFs live in <repo>/data/commentary/ (DVC, like data/statutes/), named
as in benchmark/data/commentary_sources.json.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_benchmark_dir)
sys.path.insert(0, _backend_dir)

from graph.citation_resolver import (  # noqa: E402
    CASE_TITLE_PATTERN,
    CitationResolver,
    _year_of,
    citation_titles,
    parse_reporter_citations,
)

_repo_root = os.path.dirname(os.path.dirname(_backend_dir))
DEFAULT_MANIFEST = os.path.join(_benchmark_dir, "data", "commentary_sources.json")
DEFAULT_PDF_DIR = os.path.join(_repo_root, "data", "commentary")
DEFAULT_WORK_DIR = os.path.join(_benchmark_dir, "work", "commentary")
DEFAULT_GOLD_DIR = os.path.join(_benchmark_dir, "data")

# A page with fewer letters than this in its text layer is treated as scanned.
MIN_TEXT_LAYER_LETTERS = 200
OCR_DPI = 300

# Common tesseract misreads of reporter names in footnotes ("AUR 1943 FC 18").
_OCR_FIXES = [
    (re.compile(r"\bA\s?U\s?R\b"), "AIR"),
    (re.compile(r"\bA1R\b"), "AIR"),
    (re.compile(r"\bS\s?C\s?C\b"), "SCC"),
    (re.compile(r"\bS0C\b|\bSC0\b"), "SCC"),
    (re.compile(r"\bSCR\b|\bS\s?C\s?R\b"), "SCR"),
    (re.compile(r"(?<=\w)\s+y,\s+"), " v. "),  # "Lail y, Malikar" -> "Lail v. Malikar"
    (re.compile(r"(?<=\w)\s+v,\s+"), " v. "),
]
_LOOSE_REPORTER_RE = re.compile(r"\b(?:AIR|SCC|SCR|S\.C\.C\.|S\.C\.R\.|A\.I\.R\.)\b")
_CASE_TITLE_RE = re.compile(CASE_TITLE_PATTERN)

# Words that appear in many case names and say nothing about which case it
# is. A question may use these ("Can the State be held liable...").
_GENERIC_NAME_WORDS = {
    "state", "states", "union", "india", "indian", "government", "govt", "republic",
    "commissioner", "collector", "officer", "secretary", "director", "board",
    "corporation", "company", "limited", "ltd", "pvt", "private", "bank", "society",
    "municipal", "committee", "council", "authority", "court", "high", "supreme",
    "emperor", "king", "crown", "public", "prosecutor", "central", "bureau",
    "investigation", "police", "station", "other", "others", "another", "anr", "ors",
    "the", "and", "for", "with", "through", "versus", "income", "tax", "sales",
    "customs", "excise", "district", "magistrate", "judge", "registrar", "general",
    "national", "kerala", "bombay", "madras", "calcutta", "punjab", "haryana",
    "maharashtra", "gujarat", "karnataka", "rajasthan", "bihar", "orissa", "odisha",
    "pradesh", "uttar", "madhya", "andhra", "tamil", "nadu", "west", "bengal",
    "delhi", "assam", "jharkhand", "chhattisgarh", "uttarakhand", "telangana",
    "jammu", "kashmir", "goa", "himachal", "mysore", "hyderabad", "nct",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    sources = data["sources"]
    ids = [s["source_id"] for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate source_id")
    return sources


def normalize_ocr(text: str) -> str:
    for pattern, repl in _OCR_FIXES:
        text = pattern.sub(repl, text)
    return text


def has_citation_hint(text: str) -> bool:
    """Cheap page filter: a reporter citation, a reporter abbreviation, or
    a 'X v. Y' case title. Loose on purpose; the LLM and the grounding check
    do the real work."""
    return bool(
        parse_reporter_citations(text)
        or _LOOSE_REPORTER_RE.search(text)
        or _CASE_TITLE_RE.search(text)
    )


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z0-9']+", text.lower())


def distinctive_name_words(case_text: str) -> Set[str]:
    """Words that identify a case: party-name words, minus generic ones and
    the reporter/year noise. 'Afzalur Rahman v. Emperor, AIR 1943 FC 18'
    -> {'afzalur', 'rahman'}."""
    titles = citation_titles(case_text)
    if titles:
        raw = " ".join(p + " " + r for p, r in titles)
    else:
        raw = re.sub(r"[\[(]?\d{4}[\])]?.*$", "", case_text)  # drop "(1978) 1 SCC 248" tail
    return {w for w in _words(raw) if len(w) >= 4 and w not in _GENERIC_NAME_WORDS}


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' not found on PATH (install poppler-utils / tesseract-ocr)")


def _page_count(pdf_path: str) -> int:
    out = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    if not m:
        raise RuntimeError(f"pdfinfo gave no page count for {pdf_path}")
    return int(m.group(1))


def _text_layer(pdf_path: str, page: int) -> str:
    return subprocess.run(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), pdf_path, "-"],
        capture_output=True, text=True, check=True,
    ).stdout


def _ocr_page(pdf_path: str, page: int, tmp_dir: str) -> str:
    prefix = os.path.join(tmp_dir, "page")
    subprocess.run(
        ["pdftoppm", "-r", str(OCR_DPI), "-f", str(page), "-l", str(page), "-png", "-singlefile",
         pdf_path, prefix],
        capture_output=True, check=True,
    )
    image = prefix + ".png"
    try:
        return subprocess.run(
            ["tesseract", image, "-", "-l", "eng"], capture_output=True, text=True, check=True
        ).stdout
    finally:
        if os.path.exists(image):
            os.remove(image)


def extract_pages(
    source: Dict[str, Any], pdf_dir: str, cache_dir: str, force_ocr: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """All citation-bearing pages of one source, with per-page text cached
    (OCR on a 100-page report takes minutes; re-runs should not repeat it)."""
    pdf_path = os.path.join(pdf_dir, source["file"])
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    if source.get("sha256"):
        with open(pdf_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(
                f"{pdf_path}: sha256 {digest[:12]}… does not match the manifest "
                f"({source['sha256'][:12]}…); a different file would build a different gold set"
            )
    first, last = 1, _page_count(pdf_path)
    if source.get("pages"):
        first, last = source["pages"]  # e.g. skip a long statutory annexure

    src_cache = os.path.join(cache_dir, source["source_id"])
    os.makedirs(src_cache, exist_ok=True)
    stats = {"pages": 0, "ocr_pages": 0, "kept_pages": 0}
    pages = []
    for page in range(first, last + 1):
        stats["pages"] += 1
        cache_file = os.path.join(src_cache, f"p{page:04d}.json")
        if os.path.exists(cache_file) and not force_ocr:
            with open(cache_file) as f:
                cached = json.load(f)
        else:
            text, method = _text_layer(pdf_path, page), "text"
            if force_ocr or sum(ch.isalpha() for ch in text) < MIN_TEXT_LAYER_LETTERS:
                text, method = _ocr_page(pdf_path, page, src_cache), "ocr"
            cached = {"text": text, "method": method}
            with open(cache_file, "w") as f:
                json.dump(cached, f, ensure_ascii=False)
        if cached["method"] == "ocr":
            stats["ocr_pages"] += 1
        text = normalize_ocr(cached["text"]) if cached["method"] == "ocr" else cached["text"]
        if not has_citation_hint(text):
            continue
        stats["kept_pages"] += 1
        pages.append({
            "source_id": source["source_id"],
            "page": page,
            "extraction_method": cached["method"],
            "text": text,
        })
    return pages, stats


def cmd_extract(args: argparse.Namespace) -> None:
    for tool in ("pdfinfo", "pdftotext", "pdftoppm", "tesseract"):
        _require_tool(tool)
    sources = load_manifest(args.manifest)
    if args.source:
        sources = [s for s in sources if s["source_id"] in set(args.source)]
    all_pages: List[Dict[str, Any]] = []
    for source in sources:
        try:
            pages, stats = extract_pages(
                source, args.pdf_dir, os.path.join(args.work_dir, "page_cache"), args.force_ocr
            )
        except FileNotFoundError as e:
            print(f"SKIP {source['source_id']}: PDF not found at {e}")
            continue
        except ValueError as e:
            print(f"SKIP {source['source_id']}: {e}")
            continue
        print(
            f"{source['source_id']}: {stats['pages']} pages, {stats['ocr_pages']} OCR'd, "
            f"{stats['kept_pages']} with citations"
        )
        all_pages.extend(pages)
    out = os.path.join(args.work_dir, "pages.jsonl")
    print(f"Wrote {_write_jsonl(out, all_pages)} pages -> {out}")


# ---------------------------------------------------------------------------
# Stage 2: propose
# ---------------------------------------------------------------------------

PROPOSE_SYSTEM_PROMPT = """You are building evaluation data for a search engine over Indian Supreme Court judgments.

You get one page of an official Indian legal commentary (a Law Commission report or a judicial academy compendium). The text may come from OCR and contain errors. Footnotes are at the bottom of the page; footnote markers in the body may be garbled.

Find statements of law on this page that are supported by a cited Indian court case. For each, return:
- "proposition": the legal point, in one or two sentences, as the page states it.
- "question": a question a lawyer researching this point would type into a legal search engine. It must NOT contain any case name, party name, citation, year of a case, or footnote number. Ask about the legal point itself, in your own words; do not copy phrases from the page.
- "cases": the cited cases that support this proposition, copied exactly as written on the page (name and citation).

Only include Supreme Court of India (or Federal Court / Privy Council) cases if you can tell; skip High Court-only citations when the court is clear (e.g. "AIR 1967 Mad 262"). Skip propositions with no cited case. Skip statutory text, tables of contents and lists of cases.

Return ONLY a JSON array, e.g.
[{"proposition": "...", "question": "...", "cases": ["Name v. Name, (1978) 1 SCC 248"]}]
Return [] if the page has nothing usable."""

QUESTION_MIN_WORDS, QUESTION_MAX_WORDS = 6, 60


def parse_llm_items(text: str) -> List[Dict[str, Any]]:
    """The JSON array from an LLM reply, tolerating code fences or chatter
    around it. Malformed output yields [] rather than an exception: one bad
    page should not stop a batch run."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        cases = item.get("cases")
        if isinstance(cases, str):
            cases = [cases]
        items.append({
            "proposition": str(item.get("proposition") or "").strip(),
            "question": str(item.get("question") or "").strip(),
            "cases": [str(c).strip() for c in (cases or []) if str(c).strip()],
        })
    return items


def case_grounded(case_text: str, page_text: str) -> bool:
    """True if a case the LLM named is really on the page: one of its
    reporter citations is on the page, or most of its identifying name
    words are. Stops invented or half-remembered citations."""
    case_reporters = {c.loose_key for c in parse_reporter_citations(case_text)}
    if case_reporters:
        page_reporters = {c.loose_key for c in parse_reporter_citations(page_text)}
        if case_reporters & page_reporters:
            return True
    words = distinctive_name_words(case_text)
    if not words:
        return False
    page_words = set(_words(page_text))
    return len(words & page_words) / len(words) >= 2 / 3


def question_leaks(question: str, name_words: Set[str]) -> Optional[str]:
    """Why the question gives the answer away, or None."""
    if parse_reporter_citations(question) or _LOOSE_REPORTER_RE.search(question):
        return "question contains a citation"
    if _CASE_TITLE_RE.search(question):
        return "question contains a case title"
    leaked = name_words & set(_words(question))
    if leaked:
        return f"question names a cited party ({', '.join(sorted(leaked))})"
    return None


_REPORTER_SPAN_RE = re.compile(
    r"[\[(]?\b(?:1[89]\d{2}|20\d{2})\b[\])]?\s*(?:\(?\d{1,2}\)?\s*)?"
    r"(?:Supp\.?\s*)?(?:\(?\d{1,2}\)?\s*)?"
    r"(?:A\.?\s?I\.?\s?R\.?|S\.?\s?C\.?\s?[CR]\.?)\s*(?:\(?\d{1,2}\)?\s*)?(?:[A-Z][a-z]{0,4}\.?\s*)?\d{1,5}"
    r"|\bA\.?\s?I\.?\s?R\.?\s*\d{4}\s+[A-Z][A-Za-z.]{0,9}\s*\d{1,5}"
)


def resolve_case(resolver: CitationResolver, case_text: str):
    """CitationResolver.resolve, but falls back to the case name when the
    reporter citation doesn't match. resolve() only tries names when there is
    no reporter at all, and commentary mostly cites SCC while the corpus's
    JUDIS headers carry AIR/SCR, so most SCC citations would otherwise go
    unresolved even when the case is in the corpus."""
    res = resolver.resolve(case_text)
    if res.judgment_id is not None or not parse_reporter_citations(case_text):
        return res
    name_only = _REPORTER_SPAN_RE.sub(" ", case_text)
    year = re.search(r"\b(1[89]\d{2}|20\d{2})\b", case_text)
    if year:
        name_only = f"{name_only} ({year.group(1)})"  # the year still disambiguates
    by_name = resolver.resolve(name_only)
    return by_name if by_name.judgment_id is not None else res


def judgment_names(resolver: CitationResolver) -> Dict[int, str]:
    """judgment_id -> normalised case name, inverted from the resolver."""
    names: Dict[int, str] = {}
    for alias, ids in resolver.case_names.items():
        for jid in ids:
            names.setdefault(jid, alias)
    return names


def review_item(
    item: Dict[str, Any], page: Dict[str, Any], resolver: CitationResolver, names: Dict[int, str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(candidate, None) if the item passes every check, else (None, reason)."""
    question, cases = item["question"], item["cases"]
    n_words = len(question.split())
    if not item["proposition"] or not question:
        return None, "missing proposition or question"
    if not (QUESTION_MIN_WORDS <= n_words <= QUESTION_MAX_WORDS):
        return None, f"question length {n_words} words"
    if not cases:
        return None, "no cited case"

    grounded = [c for c in cases if case_grounded(c, page["text"])]
    if not grounded:
        return None, "no cited case found in the page text"

    resolved, unresolved = [], []
    for case in grounded:
        res = resolve_case(resolver, case)
        if res.judgment_id is None:
            unresolved.append({"text": case, "method": res.method})
        else:
            resolved.append({
                "text": case,
                "judgment_id": res.judgment_id,
                "method": res.method,
                "case_name": names.get(res.judgment_id, ""),
            })
    if not resolved:
        return None, "no cited case resolves to a corpus judgment"

    name_words: Set[str] = set()
    for case in grounded:
        name_words |= distinctive_name_words(case)
    for r in resolved:
        name_words |= distinctive_name_words(r["case_name"].replace(" v ", " v. "))
    leak = question_leaks(question, name_words)
    if leak:
        return None, leak

    relevant = sorted({r["judgment_id"] for r in resolved})
    return {
        "proposition": item["proposition"],
        "query": question,
        "cases": resolved,
        "unresolved_cases": unresolved,
        "ungrounded_cases": [c for c in cases if c not in grounded],
        "relevant_judgment_ids": relevant,
    }, None


def load_resolver_from_db(conn) -> CitationResolver:
    """Same construction as graph/pg_store.PgGraphStore._resolver, so cases
    resolve exactly as citation-graph edges do."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, petitioner, respondent, date_of_judgment FROM judgments")
        judgments = cur.fetchall()
        years = {r[0]: _year_of(r[3]) for r in judgments}
        cur.execute("SELECT to_regclass('judgment_aliases')")
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT alias_norm, judgment_id, kind FROM judgment_aliases")
            rows = cur.fetchall()
            if rows:
                return CitationResolver.from_alias_rows(rows, years)
    finally:
        cur.close()
    return CitationResolver(
        {"id": r[0], "petitioner": r[1], "respondent": r[2], "date_of_judgment": r[3]}
        for r in judgments
    )


def _default_db_connection_string() -> str:
    """Same env-var convention as retrieval_eval.py / generate_gold_set.py."""
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def propose_for_pages(
    pages: List[Dict[str, Any]], backend, resolver: CitationResolver
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    names = judgment_names(resolver)
    accepted, rejected = [], []
    for page in pages:
        gen = backend.generate(
            context_block=page["text"],
            user_query="Return the JSON array now.",
            system_prompt=PROPOSE_SYSTEM_PROMPT,
        )
        items = parse_llm_items(gen.text)
        if not items:
            rejected.append({"source_id": page["source_id"], "page": page["page"],
                             "reason": "no items parsed", "raw": gen.text[:2000]})
        for i, item in enumerate(items):
            candidate, reason = review_item(item, page, resolver, names)
            base = {
                "candidate_id": f"{page['source_id']}-p{page['page']}-{i}",
                "source_id": page["source_id"],
                "page": page["page"],
                "extraction_method": page["extraction_method"],
                "generator_model": backend.model_id,
            }
            if candidate is None:
                rejected.append({**base, "reason": reason, "item": item})
            else:
                accepted.append({**base, **candidate})
        print(
            f"  {page['source_id']} p{page['page']}: {len(items)} items, "
            f"{sum(1 for a in accepted if a['page'] == page['page'] and a['source_id'] == page['source_id'])} kept"
        )
    return accepted, rejected


def cmd_propose(args: argparse.Namespace) -> None:
    import psycopg2

    from generate_gold_set import _resolve_backend

    pages = _read_jsonl(os.path.join(args.work_dir, "pages.jsonl"))
    if args.source:
        pages = [p for p in pages if p["source_id"] in set(args.source)]
    if args.max_pages:
        pages = pages[: args.max_pages]
    conn = psycopg2.connect(_default_db_connection_string())
    try:
        resolver = load_resolver_from_db(conn)
    finally:
        conn.close()
    backend = _resolve_backend(args.model, args.external_ok, args.api_key)
    print(f"Proposing from {len(pages)} pages with {backend.model_id}")
    accepted, rejected = propose_for_pages(pages, backend, resolver)

    out = os.path.join(args.work_dir, "candidates.jsonl")
    rej = os.path.join(args.work_dir, "rejected.jsonl")
    _write_jsonl(out, accepted)
    _write_jsonl(rej, rejected)
    reasons: Dict[str, int] = {}
    for r in rejected:
        key = r["reason"].split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"\n{len(accepted)} candidates -> {out}")
    print(f"{len(rejected)} rejected -> {rej}")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {reason}")


# ---------------------------------------------------------------------------
# Stage 3: review export / Stage 4: finalize
# ---------------------------------------------------------------------------

REVIEW_COLUMNS = [
    "candidate_id", "decision", "corrected_query", "notes",
    "query", "proposition", "cited_cases", "source_id", "page",
]
REVIEW_HELP = (
    "decision: accept | reject. Accept only if (1) the proposition is a real point of law the "
    "page makes, (2) the listed corpus cases are the ones the page cites for it, and (3) the "
    "query asks about that point without naming the cases. Put a better question in "
    "corrected_query if needed."
)


def cmd_review(args: argparse.Namespace) -> None:
    candidates = _read_jsonl(os.path.join(args.work_dir, "candidates.jsonl"))
    out = os.path.join(args.work_dir, "review.csv")
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists (it may hold review decisions); pass --overwrite to replace it")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "candidate_id": c["candidate_id"],
                "decision": "",
                "corrected_query": "",
                "notes": "",
                "query": c["query"],
                "proposition": c["proposition"],
                "cited_cases": " | ".join(
                    f"{x['case_name'] or x['text']} [#{x['judgment_id']}]" for x in c["cases"]
                ),
                "source_id": c["source_id"],
                "page": c["page"],
            })
    print(f"Wrote {len(candidates)} rows -> {out}\n{REVIEW_HELP}")


def build_gold(
    candidates: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, str]],
    sources: List[Dict[str, Any]],
    version: str,
    allow_unreviewed: bool,
) -> Dict[str, Any]:
    used_sources: Set[str] = set()
    records = []
    for c in candidates:
        d = decisions.get(c["candidate_id"], {})
        decision = (d.get("decision") or "").strip().lower()
        if decision not in ("accept", "a", "y", "yes"):
            if not (allow_unreviewed and decision == ""):
                continue
        query = (d.get("corrected_query") or "").strip() or c["query"]
        used_sources.add(c["source_id"])
        records.append({
            "id": c["candidate_id"],
            "query": query,
            "proposition": c["proposition"],
            "relevant_judgment_ids": c["relevant_judgment_ids"],
            # Case names let commentary_eval.py notice a re-ingest that
            # reassigned judgment ids, instead of scoring against the wrong cases.
            "relevant_case_names": {str(x["judgment_id"]): x["case_name"] for x in c["cases"]},
            "source_id": c["source_id"],
            "page": c["page"],
            "extraction_method": c["extraction_method"],
            "generator_model": c["generator_model"],
            "reviewed": decision != "",
            "query_edited": bool((d.get("corrected_query") or "").strip()),
        })
    return {
        "kind": "commentary",
        "version": version,
        "generated_at": _now(),
        "n": len(records),
        "n_reviewed": sum(r["reviewed"] for r in records),
        "sources": [s for s in sources if s["source_id"] in used_sources],
        "records": records,
    }


def cmd_finalize(args: argparse.Namespace) -> None:
    candidates = _read_jsonl(os.path.join(args.work_dir, "candidates.jsonl"))
    review_path = os.path.join(args.work_dir, "review.csv")
    decisions: Dict[str, Dict[str, str]] = {}
    if os.path.exists(review_path):
        with open(review_path, newline="") as f:
            decisions = {row["candidate_id"]: row for row in csv.DictReader(f)}
    elif not args.allow_unreviewed:
        raise SystemExit(f"{review_path} not found; run `review` first or pass --allow-unreviewed")

    gold = build_gold(candidates, decisions, load_manifest(args.manifest), args.version, args.allow_unreviewed)
    out = args.output or os.path.join(DEFAULT_GOLD_DIR, f"commentary_gold_{args.version}.json")
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} exists; gold sets are frozen once written. Use a new --version.")
    with open(out, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)
    print(f"Wrote {gold['n']} records ({gold['n_reviewed']} human-reviewed) -> {out}")
    if gold["n_reviewed"] < gold["n"]:
        print("WARNING: includes unreviewed records; label any results from it as a trial run.")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="PDFs -> pages.jsonl (text layer, OCR fallback)")
    p.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    p.add_argument("--source", action="append", help="Only this source_id (repeatable)")
    p.add_argument("--force-ocr", action="store_true", help="OCR every page, ignoring the text layer and cache")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("propose", help="pages.jsonl -> candidates.jsonl (LLM + DB)")
    p.add_argument("--source", action="append", help="Only this source_id (repeatable)")
    p.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (trial runs)")
    p.add_argument("--model", default=None, help="Omit for local Qwen; claude-*/gpt-*/gemini-* need --external-ok")
    p.add_argument("--external-ok", action="store_true",
                   help="Confirm sending commentary page text to an external model API")
    p.add_argument("--api-key", default=None)
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("review", help="candidates.jsonl -> review.csv")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("finalize", help="review.csv -> frozen gold set JSON")
    p.add_argument("--version", required=True, help="e.g. v1; becomes commentary_gold_<version>.json")
    p.add_argument("--output", default=None)
    p.add_argument("--allow-unreviewed", action="store_true",
                   help="Include rows with no decision yet (trial runs only; recorded as reviewed=false)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
