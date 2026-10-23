"""
Original documents behind citations: judgment text and PDFs, statute
sections and statute PDFs.

Text always comes from the database, so every cited judgment or statute
section can be opened. The original PDF is offered when its file can be
found under DOCUMENTS_ROOT (default: the repository root, so the DVC data
under data/ is reachable):

  - judgments: judgments.source_file, recorded by ingest.py from the next
    ingestion on. Judgments ingested before that have no PDF link.
  - statutes:  statutes.source_file, or else the PDF in data/statutes whose
    name matches the statute (the same rule ingest_statutes.py uses).

Only files inside DOCUMENTS_ROOT are ever served.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psycopg2

DOCUMENTS_ROOT = Path(
    os.getenv("DOCUMENTS_ROOT", Path(__file__).resolve().parents[2])
).resolve()
STATUTES_DIR = "data/statutes"

def relative_source_path(pdf_path: str) -> str:
    """How ingestion records a PDF: relative to DOCUMENTS_ROOT when inside
    it (portable across machines), else the absolute path, which the API
    will not serve."""
    path = Path(pdf_path).resolve()
    try:
        return path.relative_to(DOCUMENTS_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_source_file(source_file: Optional[str]) -> Optional[Path]:
    """The PDF a stored source_file points to, if it exists inside
    DOCUMENTS_ROOT."""
    if not source_file:
        return None
    path = (DOCUMENTS_ROOT / source_file).resolve()
    if not path.is_relative_to(DOCUMENTS_ROOT) or path.suffix.lower() != ".pdf":
        return None
    return path if path.is_file() else None


def statute_matches_file(short_title: Optional[str], filename: str) -> bool:
    """Which statute a PDF holds, by file name — the rule ingest_statutes.py
    uses to pick the title it stores."""
    name = filename.lower()
    is_bnss = any(k in name for k in ("bnss", "nagarik", "suraksha"))
    if (short_title or "").lower() == "constitution":
        return "constitution" in name
    if (short_title or "").lower() == "bnss":
        return is_bnss
    if (short_title or "").lower() == "bns":
        # "bns" is also inside "bnss".
        return not is_bnss and any(k in name for k in ("bns", "nyaya", "penal"))
    return False


def _statute_file_by_name(short_title: Optional[str]) -> Optional[Path]:
    folder = DOCUMENTS_ROOT / STATUTES_DIR
    if not folder.is_dir():
        return None
    for pdf in sorted(folder.glob("*.pdf")):
        if statute_matches_file(short_title, pdf.name):
            return pdf
    return None


class DocumentStore:
    def __init__(self, conn_kwargs: Dict[str, Any]):
        self._conn_kwargs = conn_kwargs
        self._columns: Dict[str, tuple] = {}

    def _connect(self):
        conn = psycopg2.connect(**self._conn_kwargs)
        conn.autocommit = True
        return conn

    def _has_column(self, cur, table: str, column: str) -> bool:
        """Cached for 5 minutes, so a migration is picked up without a restart."""
        key = f"{table}.{column}"
        hit = self._columns.get(key)
        if hit and time.monotonic() - hit[0] < 300:
            return hit[1]
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s)",
            (table, column),
        )
        found = bool(cur.fetchone()[0])
        self._columns[key] = (time.monotonic(), found)
        return found

    # -- judgments ------------------------------------------------------------

    def _judgment_row(self, cur, judgment_id: int, with_text: bool):
        source = "source_file" if self._has_column(cur, "judgments", "source_file") else "NULL"
        text = "judgment_text" if with_text else "NULL"
        cur.execute(
            f"SELECT id, petitioner, respondent, court, date_of_judgment, bench, citations, "
            f"{text}, {source} FROM judgments WHERE id = %s",
            (judgment_id,),
        )
        return cur.fetchone()

    def judgment(self, judgment_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                row = self._judgment_row(cur, judgment_id, with_text=True)
        finally:
            conn.close()
        if row is None:
            return None
        jid, pet, res, court, date, bench, citations, text, source_file = row
        if isinstance(citations, str):
            citations = json.loads(citations or "{}")
        return {
            "judgment_id": jid,
            "doc_type": "judgment",
            "title": f"{pet or 'Unknown'} v. {res or 'Unknown'}",
            "petitioner": pet,
            "respondent": res,
            "court": court,
            "date": str(date) if date else None,
            "bench": bench or [],
            "reporter_citations": (citations or {}).get("reporters", []),
            "text": text or "",
            "has_pdf": resolve_source_file(source_file) is not None,
        }

    def judgment_pdf(self, judgment_id: int) -> Optional[Path]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                row = self._judgment_row(cur, judgment_id, with_text=False)
        finally:
            conn.close()
        return resolve_source_file(row[8]) if row else None

    # -- statutes -------------------------------------------------------------

    def _statute_pdf(self, cur, statute_id: int) -> Optional[Path]:
        source = "source_file" if self._has_column(cur, "statutes", "source_file") else "NULL"
        cur.execute(f"SELECT short_title, {source} FROM statutes WHERE id = %s", (statute_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return resolve_source_file(row[1]) or _statute_file_by_name(row[0])

    def statute_section(self, section_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ss.section_id, ss.statute_id, st.title, st.short_title, "
                    "ss.section_number, ss.heading, ss.content "
                    "FROM statute_sections ss JOIN statutes st ON st.id = ss.statute_id "
                    "WHERE ss.section_id = %s",
                    (section_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                has_pdf = self._statute_pdf(cur, row[1]) is not None
        finally:
            conn.close()
        sid, statute_id, title, short, number, heading, content = row
        label = "Art." if "constitution" in (title or "").lower() else "Section"
        return {
            "section_id": sid,
            "statute_id": statute_id,
            "doc_type": "statute",
            "title": f"{title}, {label} {number}",
            "statute_title": title,
            "short_title": short,
            "section_number": number,
            "heading": heading,
            "text": content or "",
            "has_pdf": has_pdf,
        }

    def statute_pdf(self, statute_id: int) -> Optional[Path]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                return self._statute_pdf(cur, statute_id)
        finally:
            conn.close()
