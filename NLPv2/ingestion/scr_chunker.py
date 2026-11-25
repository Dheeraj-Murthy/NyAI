"""Text cleaning, paragraph rebuilding and chunking for S.C.R. law-report PDFs.

Used by ingest_aws.py. Kept separate from ingest.py so the older JUDIS-style
ingestion path is untouched.

Pipeline:  pdftotext -layout output
   -> clean_to_paragraphs()   strip running headers, A-H margin letters, rebuild paragraphs
   -> chunk_scr()             label sections by position + anchors, split on sentences, merge tiny chunks
   -> context_prefix()        one-line case label prepended to each chunk *for embedding only*
"""
import re
from typing import List, Optional, Tuple

MARGIN = "ABCDEFGH"
EM_DASH = chr(8212)
EN_DASH = chr(8211)

_RUNNING_HEADER_RES = [
    re.compile(r"^\s*\[\d{4}\]\s+(\d+\s+|SUPP\.?\s*\d*\s+)?S\.?C\.?R\.?(\s+\d+)?\s*(:\s*\d{4}\s+INSC\s+\d+)?\s*\d*\s*$", re.I),
    re.compile(r"^\s*(\d{1,4}\s+)?(S\.?C\.?R\.?\s+)?SUPREME COURT REPORTS.*$", re.I),
    re.compile(r"^\s*\d{1,4}\s*$"),  # lone page number
]


def _is_title_header(line: str) -> bool:
    """First line of pages 2+: 'STATE (NCT OF DELHI) v. AHMED JAAN   29'."""
    s = line.strip()
    if not s or len(s) > 140:
        return False
    if not (re.match(r"^\d{1,4}\s+\S", s) or re.search(r"\s\d{1,4}$", s)):
        return False
    letters = [c for c in s if c.isalpha()]
    upper = sum(c.isupper() for c in letters)
    return bool(letters) and (upper / len(letters) > 0.6 or re.search(r"\bv\.?\b|\bvs\.?\b", s, re.I) is not None)


def _page_lines(page: str, first_page: bool):
    """(indent, text) for each line of one page, headers and margin letters removed.
    A blank line is None."""
    cur = -1
    out = []
    seen_first = False
    dropped_title = False
    for raw in page.split("\n"):
        if not raw.strip():
            out.append(None)
            continue
        if any(r.match(raw) for r in _RUNNING_HEADER_RES):
            continue
        if not seen_first:
            seen_first = True
            if not first_page and _is_title_header(raw):
                dropped_title = True
                continue
        elif dropped_title:
            dropped_title = False
            s = raw.strip()
            if len(s) < 40 and s.upper() == s:      # second line of a wrapped title header
                continue
        s = raw.rstrip()
        stripped = s.strip()
        if len(stripped) == 1 and stripped in MARGIN:      # lone margin letter
            cur = max(cur, MARGIN.index(stripped))
            continue
        indent = len(s) - len(s.lstrip())
        m = re.match(r"^(\s*)([A-H])(\s{2,}|\s+(?=\S))(.*)$", s)
        if m:
            idx = MARGIN.index(m.group(2))
            if cur < idx <= cur + 3 and (cur >= 0 or (idx == 0 and len(m.group(3)) >= 2)):
                cur = idx
                s = m.group(4)
                indent = len(m.group(1)) + 1 + len(m.group(3))
        m = re.match(r"^(.*\S)\s+([A-H])$", s)
        if m:
            idx = MARGIN.index(m.group(2))
            if cur < idx <= cur + 3:
                cur = idx
                s = m.group(1)
        out.append((indent, re.sub(r"[ \t]{2,}", " ", s).strip()))
    return out


def clean_to_paragraphs(raw: str) -> List[str]:
    seq = []
    for pi, page in enumerate(raw.split("\f")):
        seq.extend(_page_lines(page, first_page=(pi == 0)))
    paras, cur, prev_indent, prev_text = [], [], 0, ""

    def flush():
        nonlocal cur
        if cur:
            txt = ""
            for ln in cur:
                if txt.endswith("-") and ln[:1].islower() and len(txt) > 1 and txt[-2].isalpha():
                    txt = txt[:-1] + ln                      # "ac-" + "cused"
                else:
                    txt = (txt + " " + ln) if txt else ln
            paras.append(txt.strip())
        cur = []

    for item in seq:
        if item is None:
            flush()
            prev_text = ""
            continue
        indent, text = item
        if cur and prev_text.endswith((".", "?", ":", ";", "]", ")")) and indent - prev_indent >= 3:
            flush()
        cur.append(text)
        prev_indent, prev_text = indent, text
    flush()
    return [p for p in paras if p]


# --------------------------------------------------------------------------
# Section labelling + chunking
# --------------------------------------------------------------------------

_JURISDICTION = re.compile(r"(CIVIL|CRIMINAL|ORIGINAL|ADVISORY|APPELLATE|WRIT)\b.{0,40}\bJURISDICTION", re.I)
_DELIVERED = re.compile(r"(judgment|order) of the court was delivered|delivered the following|\bJUDGMENT\s*$|\bORDER\s*$", re.I)
_HELD = re.compile(r"\bHELD\s*:", re.I)
_ARGUE = re.compile(r"\b(contend(ed|s|ing)?|submit(ted|s)?|argu(ed|es|ing|ment)|learned (senior )?(counsel|advocate)|on behalf of the (appellant|respondent|petitioner)|it was (urged|contended|submitted))\b", re.I)
_RATIO = re.compile(r"\b(we are of the (view|opinion)|in our (view|opinion)|we find|we are unable|it is (well )?settled|this court (has )?(held|observed)|the question (is|that arises)|the law (is|on the point)|the principle|ratio|is not tenable|cannot be sustained)\b", re.I)
_ISSUE = re.compile(r"^(the )?(question|issue|point)s?\b|^whether\b|\bpoints? for (consideration|determination)\b|\bquestions? (that )?arises?\b", re.I)
_CONCL = re.compile(r"\b(appeals? (is|are|stand|stands|shall be) (dismissed|allowed)|in the result|accordingly|for (the|these) reasons|we (therefore )?(hold|direct|allow|dismiss|set aside)|no order as to costs|(appeal|petition)s? (is|are) (dismissed|allowed|disposed))\b", re.I)

MAX_CHARS = 2400
MIN_CHARS = 200


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.;?])\s+(?=[A-Z\[(\"'0-9])", text)
    return [p for p in parts if p.strip()]


def _label(paragraphs: List[str]) -> List[str]:
    n = len(paragraphs)
    labels = ["facts"] * n
    body_start = 0
    for i, p in enumerate(paragraphs[:60]):
        if _JURISDICTION.search(p) or _DELIVERED.search(p):
            body_start = i
            break
    head_end = body_start or 0
    first_held = next((i for i, p in enumerate(paragraphs[:head_end]) if _HELD.search(p)), None)
    for i in range(head_end):
        if first_held is not None and i >= first_held:
            labels[i] = "ratio"                       # HELD: points
        else:
            has_dash = (EM_DASH in paragraphs[i][:200]) or (EN_DASH in paragraphs[i][:200])
            labels[i] = "issues" if (i < 4 or has_dash) and i < max(1, (first_held or head_end)) else "facts"
    state = "facts"
    for i in range(head_end, n):
        p = paragraphs[i]
        pos = (i - head_end) / max(1, n - head_end)
        if pos > 0.8 and _CONCL.search(p):
            state = "judgment"
        elif state != "judgment":
            if _ARGUE.search(p):
                state = "arguments"
            elif _RATIO.search(p):
                state = "ratio"
        labels[i] = "issues" if (_ISSUE.search(p[:160]) and state != "judgment") else state
    return labels


def chunk_scr(paragraphs: List[str]) -> List[Tuple[str, str]]:
    if not paragraphs:
        return []
    labels = _label(paragraphs)
    chunks: List[Tuple[str, str]] = []
    cur_label, cur = None, ""
    for lab, para in zip(labels, paragraphs):
        pieces = [para]
        if len(para) > MAX_CHARS:
            pieces, buf = [], ""
            for s in _split_sentences(para):
                if buf and len(buf) + len(s) + 1 > MAX_CHARS:
                    pieces.append(buf)
                    buf = s
                else:
                    buf = (buf + " " + s) if buf else s
            if buf:
                pieces.append(buf)
        for piece in pieces:
            if len(piece) > MAX_CHARS * 1.5:           # no sentence breaks at all: hard split on words
                sub, buf = [], ""
                for w in piece.split():
                    if buf and len(buf) + len(w) + 1 > MAX_CHARS:
                        sub.append(buf)
                        buf = w
                    else:
                        buf = (buf + " " + w) if buf else w
                if buf:
                    sub.append(buf)
            else:
                sub = [piece]
            for s in sub:
                if cur and (lab != cur_label or len(cur) + len(s) + 2 > MAX_CHARS):
                    chunks.append((cur_label, cur))
                    cur = ""
                cur = (cur + "\n\n" + s) if cur else s
                cur_label = lab
    if cur:
        chunks.append((cur_label, cur))
    merged: List[Tuple[str, str]] = []
    for lab, txt in chunks:
        if merged and len(txt) < MIN_CHARS:
            pl, pt = merged[-1]
            merged[-1] = (pl, pt + "\n\n" + txt)
        else:
            merged.append((lab, txt))
    if len(merged) > 1 and len(merged[0][1]) < MIN_CHARS:
        l1, t1 = merged[1]
        merged[1] = (l1, merged[0][1] + "\n\n" + t1)
        merged = merged[1:]
    return merged


# --------------------------------------------------------------------------
# Embedding context label
# --------------------------------------------------------------------------

def context_prefix(petitioner: str, respondent: str, date, bench, reporters: Optional[list] = None,
                   disposal: Optional[str] = None) -> str:
    """One-line case label prepended to each chunk when it is embedded.
    Not stored in judgment_chunks.content; recompute it from the judgments row
    when re-embedding with another model."""
    parts = [f"{petitioner} v. {respondent}"]
    if date:
        parts.append(str(date.year if hasattr(date, "year") else str(date)[:4]))
    if reporters:
        parts.append(", ".join(reporters[:2]))
    if bench:
        parts.append("Bench: " + ", ".join(bench[:5]))
    if disposal:
        parts.append(disposal)
    return " | ".join(parts)
