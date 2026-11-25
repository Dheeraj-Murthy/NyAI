"""
Citation resolver for the Legal RAG citation graph.

Turns a citation string pulled out of a judgment ("Kesavananda Bharati v.
State of Kerala (1973)", "[1950] S.C.R. 940", "AIR 1987 SC 1086") into the
judgment ID it refers to. Shared by ingestion (building citation_edges) and
the API (search / resolve routes), so both sides normalise names and reporter
citations the same way.

Resolution order, first hit wins:
  1. reporter citation  -> key lookup against each judgment's own CITATION:
                           header (e.g. "1953 AIR 75  1953 SCR 215")
  2. exact case name    -> Aho-Corasick scan for every known
                           "petitioner v respondent" inside the citation text
  3. fuzzy case name    -> the case title written in the citation, matched by
                           its identifying words against known case names
                           (citations shorten names), filtered by year
  4. otherwise          -> "ambiguous" (several equally good targets) or
                           "unresolved"; never a guess

pyahocorasick is optional at import time: without it the exact step falls
back to a slower pure-Python scan instead of failing.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import ahocorasick
except ImportError:  # pragma: no cover - exercised only without the wheel
    ahocorasick = None


# --------------------------------------------------------------------------
# Reporter citations
# --------------------------------------------------------------------------

_YEAR = r"(?:1[89]\d{2}|20\d{2})"
_REP = r"(?P<rep>S\.?\s?C\.?\s?[RC])\b\.?"

# Each pattern yields named groups: year, page, and optionally vol / rep / court.
REPORTER_PATTERNS = [
    # AIR in text: "AIR 1987 SC 1086", "A.I.R. 1953 S.C. 75", "A.I.R. 1950 Mad. 12"
    re.compile(
        r"\bA\.?\s?I\.?\s?R\.?\s*[\[(]?(?P<year>" + _YEAR + r")[\])]?\s+"
        r"(?P<court>[A-Z][A-Za-z]{0,9}\.?(?:\s?[A-Z][a-z]{0,3}\.)?)\s*(?P<page>\d{1,5})\b"
    ),
    # AIR in a judgment's own CITATION: header: "1953 AIR 75" (always the SC reporter)
    re.compile(r"\b(?P<year>" + _YEAR + r")\s+AIR\s+(?P<page>\d{1,5})\b"),
    # SCR / SCC with the year in brackets: "[1950] S.C.R. 940", "(1992) 1 SCC 588"
    re.compile(
        r"[\[(](?P<year>" + _YEAR + r")[\])]\s*(?P<vol>\d{1,2})?\s*" + _REP
        + r"\s*(?:p\.?\s*)?(?P<page>\d{1,5})\b"
    ),
    # SCR / SCC with a bare year, header style included: "1955 SCR (1) 777", "1970 SCC (2) 298"
    re.compile(
        r"\b(?P<year>" + _YEAR + r")\s+(?:(?P<vol_pre>\d{1,2})\s+)?" + _REP
        + r"\s*(?:\((?P<vol>\d{1,2})\)\s*)?(?P<page>\d{1,5})\b"
    ),
]


@dataclass(frozen=True)
class ReporterCitation:
    raw: str
    key: str        # exact key, e.g. "scr|1955|1|777" or "air|1953|sc|75"
    loose_key: str  # key without the volume, e.g. "scr|1955||777"


def _reporter_from_match(m: "re.Match") -> ReporterCitation:
    groups = m.groupdict()
    year, page = groups["year"], str(int(groups["page"]))
    if "court" in groups or "rep" not in groups:
        court = re.sub(r"[^a-z]", "", (groups.get("court") or "sc").lower())
        key = f"air|{year}|{court}|{page}"
        return ReporterCitation(m.group(0).strip(), key, key)
    rep = re.sub(r"[^a-z]", "", groups["rep"].lower())
    vol = groups.get("vol") or groups.get("vol_pre") or ""
    vol = str(int(vol)) if vol else ""
    return ReporterCitation(
        m.group(0).strip(), f"{rep}|{year}|{vol}|{page}", f"{rep}|{year}||{page}"
    )


def parse_reporter_citations(text: str) -> List[ReporterCitation]:
    """All reporter citations in `text`, de-duplicated by key, in text order."""
    found: List[Tuple[int, ReporterCitation]] = []
    seen: Set[str] = set()
    taken: List[Tuple[int, int]] = []
    for pattern in REPORTER_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            # Patterns overlap ("1955 1 S.C.R. 777" vs "[1955] 1 S.C.R. 777");
            # the first pattern to claim a stretch of text wins.
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            cit = _reporter_from_match(m)
            taken.append(span)
            if cit.key not in seen:
                seen.add(cit.key)
                found.append((span[0], cit))
    return [c for _, c in sorted(found, key=lambda x: x[0])]


_HEADER_RE = re.compile(
    r"CITATION:\s*(?P<body>.*?)(?=CITATOR INFO|ACT:|HEADNOTE:|BENCH:|\n\s*\n\s*\n|$)", re.S
)


def extract_citation_header(text: str) -> Optional[str]:
    """The block after a judgment's own "CITATION:" header (JUDIS format),
    e.g. "1953 AIR 75  1953 SCR 215". None when the judgment has no header."""
    if not text:
        return None
    m = _HEADER_RE.search(text[:6000])
    if not m:
        return None
    body = " ".join(m.group("body").split())
    return body or None


# --------------------------------------------------------------------------
# Case names
# --------------------------------------------------------------------------

_STATE_ABBREVIATIONS = {
    "mp": "madhya pradesh",
    "up": "uttar pradesh",
    "ap": "andhra pradesh",
    "hp": "himachal pradesh",
    "wb": "west bengal",
    "tn": "tamil nadu",
    "jk": "jammu and kashmir",
    "jandk": "jammu and kashmir",
}
_STATE_ABBR_RE = re.compile(r"\bstate of (m ?p|u ?p|a ?p|h ?p|w ?b|t ?n|j ?and ?k|j ?k)\b")
_PARTY_SUFFIX_RE = re.compile(r"\b(?:and )?(?:ors|others|anr|another|etc)\b")
_SEPARATOR_RE = re.compile(r"\b(?:versus|vs|v)\b\.?")
# Role labels newer Supreme Court headers put on the party lines
# ("P.EKNATH ...Appellant(s)", "Ratan Lal .. Respondent"); after punctuation
# is removed "(s)" is a separate "s". Only at the end of a party, so
# "Petitioners Association" keeps its name.
_ROLE_LABEL_RE = re.compile(r"\b(?:appellant|respondent|petitioner)s?(?:\s+s)?(?=\s+v\s|\s*$)")

# Words too common in Indian case titles to identify a case on their own:
# connectives, honorifics, and the government bodies and company words that
# appear as a party in thousands of cases.
STOPWORDS = frozenset(
    "v of and the state union india ltd limited co company pvt private m s mr "
    "shri sri smt dr in re ex parte by through govt government his her holiness "
    "commissioner income tax collector customs excise central sales wealth officer "
    "municipal corporation board district magistrate workmen employees bank "
    "insurance railway high court appellate tribunal authority secretary director "
    "general registrar deputy assistant additional chief controller estate".split()
)


def normalize_case_name(name: str) -> str:
    """Canonical form of a case name, applied identically to the stored
    petitioner/respondent and to citation text.

    >>> normalize_case_name("The State of M.P. & Ors. Vs. Ram Kumar")
    'state of madhya pradesh v ram kumar'
    >>> normalize_case_name("P.EKNATH ...Appellant(s) Vs. Y. Amaranatha Reddy ...Respondent(s)")
    'p eknath v y amaranatha reddy'
    """
    if not name:
        return ""
    s = name.lower().replace("&", " and ")
    s = _SEPARATOR_RE.sub(" v ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = " ".join(s.split())
    s = _STATE_ABBR_RE.sub(
        lambda m: "state of " + _STATE_ABBREVIATIONS[m.group(1).replace(" ", "")], s
    )
    s = re.sub(r"\bu ?o ?i\b", "union of india", s)
    s = _PARTY_SUFFIX_RE.sub(" ", s)
    s = _ROLE_LABEL_RE.sub(" ", s)
    s = re.sub(r"\bm s\b", " ", s)
    tokens = [t for t in s.split() if t != "the"]
    return " ".join(tokens)


_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_PARTY_JUNK_RES = [
    re.compile(r"^\s*(?:petitioner|respondent)\s*:\s*", re.I),  # leaked header label
    re.compile(r"\[.*$", re.S),                                   # "[MEHR CHAND ..." bench text
    re.compile(r"\b(?:" + _MONTHS + r")\s+\d.*$", re.I | re.S),   # "OCTOBER 12, 1954. ..."
    re.compile(r"\(\s*(?:and|with|c\b)[^)]*\)?.*$", re.I | re.S), # "(AND CONNECTED APPEALS)"
    re.compile(r"\band connected\b.*$", re.I | re.S),
    # "... WITH CIVIL APPEAL NO. 6998 OF 2019 (Arising from SLP ..." tagged-matter notes
    re.compile(r"\bwith\s+(?:civil|criminal|writ|special|transfer|contempt|review|s\.?\s?l\.?\s?p)\b.*$", re.I | re.S),
    re.compile(r"-{2,}\s*intervener.*$", re.I | re.S),
    re.compile(r"\(\s*dead\s*\)\s*(?:by|through)?\s*(?:legal representatives?|l\.?\s?rs?\.?)?", re.I),
]


def clean_party_name(raw: Optional[str]) -> str:
    """Strip text that ingestion's header parser sometimes leaves on a party
    name (connected-appeal notes, the hearing date, bench names, a leaked
    "PETITIONER:" label)."""
    s = raw or ""
    for pattern in _PARTY_JUNK_RES:
        s = pattern.sub(" ", s)
    s = s.strip()
    return "" if s.upper().rstrip(":") in ("RESPONDENT", "PETITIONER", "UNKNOWN") else s


def case_name_alias(petitioner: Optional[str], respondent: Optional[str]) -> Optional[str]:
    """Normalised "petitioner v respondent", or None when either side is
    missing — a one-sided name links far too many unrelated citations."""
    pet = normalize_case_name(clean_party_name(petitioner))
    res = normalize_case_name(clean_party_name(respondent))
    if not pet or not res or "unknown" in (pet, res):
        return None
    alias = f"{pet} v {res}"
    return alias if len(alias) > 8 else None


def _split_parties(alias: str) -> Tuple[Set[str], Set[str]]:
    """Distinctive words of each party. Initials can contain a "v"
    ("n v shanmugham and co v commissioner of income tax"), so split at the
    " v " that leaves the most distinctive words on the weaker side."""
    tokens = alias.split()
    best: Tuple[int, Set[str], Set[str]] = (-1, set(), set())
    for i, tok in enumerate(tokens):
        if tok != "v" or i == 0 or i == len(tokens) - 1:
            continue
        left, right = _distinctive(tokens[:i]), _distinctive(tokens[i + 1 :])
        score = min(len(left), len(right))
        if score > best[0]:
            best = (score, left, right)
    return best[1], best[2]


# A party name in running text: capitalised words, initials and the small
# words inside names ("State of Bombay", "F. N. Balsara", "Tata & Sons").
# Lowercase narrative ("relied upon the decision in") is never part of a name.
# At most 12 words a side: an unbounded run made an all-caps stretch of a
# judgment cost quadratic time to scan.
_NAME_WORD = r"[A-Z][A-Za-z0-9.'\u2019&\-]*"
_PARTY_WORD = r"(?:" + _NAME_WORD + r"|of|and|the|for|&)"
_PARTY = r"(" + _NAME_WORD + r"(?:\s+" + _PARTY_WORD + r"){0,11})"
_TITLE = _PARTY + r"\s*,?\s+(?:v|vs|V|Vs|VS|versus|Versus|VERSUS)\.?\s+" + _PARTY
_TITLE_RE = re.compile(_TITLE)
# A case title as written in a judgment, with its year if given:
# "Kesavananda Bharati v. State of Kerala (1973)".
CASE_TITLE_PATTERN = _TITLE + r"(?:\s*[\[(](?:1[89]\d{2}|20\d{2})[\])])?"
# Sentence openers that get capitalised in front of a case name.
_LEAD_WORDS = frozenset(
    "in see also cf per vide following relying relied followed referring "
    "applying distinguishing overruling case decision held".split()
)


def citation_titles(text: str) -> List[Tuple[str, str]]:
    """(petitioner, respondent) pairs written as case titles in `text`."""
    return [(m.group(1), m.group(2)) for m in _TITLE_RE.finditer(text or "")]


def _party_words(raw: str) -> Set[str]:
    tokens = normalize_case_name(raw).split()
    while tokens and tokens[0] in _LEAD_WORDS:
        tokens.pop(0)
    return _distinctive(tokens)


def _distinctive(tokens: Iterable[str]) -> Set[str]:
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()}


def _year_of(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", str(value))
    return int(m.group(1)) if m else None


def _cited_year(cited_text: str) -> Optional[int]:
    m = re.search(r"[\[(](1[89]\d{2}|20\d{2})[\])]", cited_text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolution:
    judgment_id: Optional[int]
    method: str  # reporter | exact_name | fuzzy | ambiguous | unresolved | self
    score: float = 0.0


# Share of a party's identifying words the citation must give for a fuzzy
# match: "Kesavananda Bharati" gives 2 of the 4 in "His Holiness Kesavananda
# Bharati Sripadagalvaru"; a bare "Ram" gives 1 of 4 in "Sita Ram Gupta Kumar".
MIN_SIDE_COVERAGE = 1 / 3
# A word shared by more judgments than this is too common to propose candidates.
MAX_TOKEN_DF = 200


class CitationResolver:
    """In-memory resolver, built once per edge-building run from the
    judgments table. Construction is O(total alias length); each resolve is
    O(citation length) for the exact steps."""

    def __init__(self, judgments: Iterable[Dict[str, Any]] = ()):
        """`judgments`: dicts with id, petitioner, respondent,
        date_of_judgment and optionally header_text (the start of
        judgment_text, for its CITATION: header) and citations (the JSONB
        column, {"reporters": [...]})."""
        self.case_names: Dict[str, Set[int]] = defaultdict(set)
        self.reporters: Dict[str, Set[int]] = defaultdict(set)
        self.reporters_loose: Dict[str, Set[int]] = defaultdict(set)
        self.years: Dict[int, Optional[int]] = {}
        self._sides: Dict[str, Tuple[Set[str], Set[str]]] = {}

        for j in judgments:
            jid = int(j["id"])
            self.years[jid] = _year_of(j.get("date_of_judgment"))

            alias = case_name_alias(j.get("petitioner"), j.get("respondent"))
            if alias:
                self.case_names[alias].add(jid)

            for cit in self._own_reporter_citations(j):
                self.reporters[cit.key].add(jid)
                self.reporters_loose[cit.loose_key].add(jid)

        self._build_indexes()

    @classmethod
    def from_alias_rows(
        cls, rows: Iterable[Tuple[str, int, str]], years: Dict[int, Optional[int]]
    ) -> "CitationResolver":
        """Rebuild a resolver from judgment_aliases rows, as written by
        alias_rows(), so the API resolves exactly as edge building did."""
        resolver = cls()
        resolver.years = dict(years)
        for alias, jid, kind in rows:
            if kind == "case_name":
                resolver.case_names[alias].add(jid)
            elif kind == "reporter":
                rep, year, vol, page = (alias.split("|") + ["", "", "", ""])[:4]
                if rep == "air" or vol:
                    resolver.reporters[alias].add(jid)
                resolver.reporters_loose[f"{rep}|{year}||{page}" if rep != "air" else alias].add(jid)
        resolver._build_indexes()
        return resolver

    def _build_indexes(self) -> None:
        # Token index for fuzzy candidate generation.
        self._token_index: Dict[str, Set[str]] = defaultdict(set)
        for alias in self.case_names:
            self._sides[alias] = _split_parties(alias)
            for tok in _distinctive(alias.split()):
                self._token_index[tok].add(alias)

        self._automaton = None
        if ahocorasick is not None and self.case_names:
            automaton = ahocorasick.Automaton()
            for alias in self.case_names:
                automaton.add_word(f" {alias} ", alias)
            automaton.make_automaton()
            self._automaton = automaton

    @staticmethod
    def _own_reporter_citations(j: Dict[str, Any]) -> List[ReporterCitation]:
        sources: List[str] = []
        header = j.get("header_text")
        if header:
            block = extract_citation_header(header)
            if block:
                sources.append(block)
        stored = j.get("citations")
        if isinstance(stored, dict):
            sources.extend(str(r) for r in stored.get("reporters") or [])
        found: List[ReporterCitation] = []
        for src in sources:
            found.extend(parse_reporter_citations(src))
        return found

    # -- DB rows ------------------------------------------------------------

    def alias_rows(self) -> List[Tuple[str, int, str]]:
        """(alias_norm, judgment_id, kind) rows for the judgment_aliases table."""
        rows = set()
        for alias, ids in self.case_names.items():
            rows.update((alias, jid, "case_name") for jid in ids)
        for table in (self.reporters, self.reporters_loose):
            for key, ids in table.items():
                rows.update((key, jid, "reporter") for jid in ids)
        return sorted(rows)

    # -- resolution -----------------------------------------------------------

    def resolve(self, cited_text: str, source_id: Optional[int] = None) -> Resolution:
        reporters = parse_reporter_citations(cited_text)
        if reporters:
            result = self._resolve_reporter(reporters, source_id)
        else:
            result = self._resolve_name(cited_text)
        if source_id is not None and result.judgment_id == source_id:
            return Resolution(None, "self", result.score)
        return result

    def _resolve_reporter(
        self, reporters: List[ReporterCitation], source_id: Optional[int] = None
    ) -> Resolution:
        ambiguous = False
        for cit in reporters:
            ids = self.reporters.get(cit.key) or self.reporters_loose.get(cit.loose_key)
            if not ids:
                continue
            if source_id in ids:
                # The judgment's own header citation; source data sometimes
                # gives two judgments the same one.
                return Resolution(source_id, "reporter", 100.0)
            if len(ids) == 1:
                return Resolution(next(iter(ids)), "reporter", 100.0)
            ambiguous = True
        return Resolution(None, "ambiguous" if ambiguous else "unresolved")

    def _pick(self, ids: Set[int], year: Optional[int]) -> Optional[int]:
        if len(ids) == 1:
            return next(iter(ids))
        if year is not None:
            same_year = {i for i in ids if self.years.get(i) == year}
            if len(same_year) == 1:
                return next(iter(same_year))
        return None

    def _resolve_name(self, cited_text: str) -> Resolution:
        norm = normalize_case_name(cited_text)
        if " v " not in f" {norm} ":
            return Resolution(None, "unresolved")
        year = _cited_year(cited_text)

        # Exact: every known full case name contained in the citation text.
        hits = self._exact_hits(norm)
        if hits:
            longest = max(len(a) for a in hits)
            ids: Set[int] = set()
            for alias in hits:
                if len(alias) == longest:
                    ids |= self.case_names[alias]
            picked = self._pick(ids, year)
            if picked is not None:
                return Resolution(picked, "exact_name", 100.0)
            return Resolution(None, "ambiguous", 100.0)

        return self._resolve_fuzzy(cited_text, year)

    def _exact_hits(self, norm: str) -> Set[str]:
        haystack = f" {norm} "
        if self._automaton is not None:
            return {alias for _, alias in self._automaton.iter(haystack)}
        return {a for a in self.case_names if f" {a} " in haystack}

    def _resolve_fuzzy(self, cited_text: str, year: Optional[int]) -> Resolution:
        """Match the case titles written in the citation against known case
        names. Citations shorten names ("Balsara" for "F. N. Balsara"), so
        every identifying word the citation gives for a party must belong to
        that party in the candidate — not the other way round."""
        best_score, best_ids = 0.0, set()
        for pet_raw, res_raw in citation_titles(cited_text):
            cp, cr = _party_words(pet_raw), _party_words(res_raw)
            if len(cp | cr) < 2:
                continue  # too little to identify a case

            candidates: Set[str] = set()
            for tok in cp | cr:
                aliases = self._token_index.get(tok)
                if aliases and len(aliases) <= MAX_TOKEN_DF:
                    candidates |= aliases

            for alias in candidates:
                pet_toks, res_toks = self._sides[alias]
                if not (cp <= pet_toks and cr <= res_toks):
                    continue
                if (pet_toks and not cp) or (res_toks and not cr):
                    continue
                coverage = [len(c) / len(t) for c, t in ((cp, pet_toks), (cr, res_toks)) if t]
                if not coverage or min(coverage) < MIN_SIDE_COVERAGE:
                    continue
                ids = self.case_names[alias]
                if year is not None:
                    ids = {i for i in ids if self.years.get(i) in (None, year - 1, year, year + 1)}
                    if not ids:
                        continue
                # How much of the case name the citation gave, 0-100.
                score = 100.0 * len(cp | cr) / len(pet_toks | res_toks)
                if score > best_score:
                    best_score, best_ids = score, set(ids)
                elif score == best_score:
                    best_ids |= ids

        if not best_ids:
            return Resolution(None, "unresolved")
        picked = self._pick(best_ids, year)
        if picked is None:
            return Resolution(None, "ambiguous", best_score)
        return Resolution(picked, "fuzzy", best_score)

