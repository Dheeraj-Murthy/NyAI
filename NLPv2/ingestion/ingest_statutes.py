#!/usr/bin/env python3
"""Ingest statutes (Constitution of India, BNS, BNSS) into Legal RAG.

Usage:
    python ingest_statutes.py --dir /path/to/pdfs

PDFs are converted via `pdftotext`.  The Constitution is parsed with
`-layout` mode (clean English article markers); BNS and BNSS are parsed
in raw mode (broader section-number detection).

Each article/section becomes one chunk + embedding stored in the
`statute_sections` / `statute_embeddings` tables.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import psycopg2
import numpy as np
from dotenv import load_dotenv

# backend/ is a sibling directory, not a package ingestion/ installs — add it
# to sys.path to reuse tracking.py's MLflow setup instead of duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import tracking
from db_schema import ensure_schema
from documents import relative_source_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768
EMBED_BATCH = 128

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _default_dsn() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_conn(dsn: Optional[str] = None):
    return psycopg2.connect(dsn or _default_dsn())


def upsert_statute(cur, title: str, short_title: str, year: int,
                   source_file: Optional[str] = None) -> int:
    """Insert statute row; return id (reuses existing row by title).

    statutes.title has no UNIQUE constraint, so `ON CONFLICT DO NOTHING`
    here was a silent no-op — every call inserted a fresh row and
    re-running this script on the same PDFs kept creating duplicate
    statutes (and duplicate statute_sections under each one). Look up
    first instead of relying on a conflict that can never fire.
    """
    cur.execute("SELECT id FROM statutes WHERE title=%s", (title,))
    row = cur.fetchone()
    if row:
        if source_file:
            cur.execute("UPDATE statutes SET source_file=%s WHERE id=%s", (source_file, row[0]))
        return row[0]
    cur.execute(
        """INSERT INTO statutes (title, short_title, year, source_file)
           VALUES (%s, %s, %s, %s)
           RETURNING id""",
        (title, short_title, year, source_file),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# PDF → text extraction
# ---------------------------------------------------------------------------

def pdf_to_text(pdf_path: str, *, layout: bool = False) -> str:
    mode = ["-layout"] if layout else []
    try:
        out = subprocess.check_output(
            ["pdftotext"] + mode + [pdf_path, "-"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace")
    except FileNotFoundError:
        sys.exit("pdftotext not found – install poppler-utils first")
    except subprocess.CalledProcessError as e:
        sys.exit(f"pdftotext failed on {pdf_path}: {e}")


# ---------------------------------------------------------------------------
# Constitution parser – article splitting
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"^\s*(\d{1,3}[A-Za-z]?)\.\s+(.+?)\u2014",  # "21. Right to life.—"
    re.M,
)

# Amendment-history footnotes ("21. Ins. by the Constitution (Forty-fourth
# Amendment) Act, 1978...") match _ARTICLE_RE's shape too and recur
# throughout the document reusing the same numbers as real articles. Since
# ingestion upserts on (statute_id, section_number) with DO UPDATE, a
# footnote match occurring after the real article in document order would
# silently overwrite the real article's content — so these must never be
# treated as a header in the first place.
_FOOTNOTE_HEADING_RE = re.compile(r"^(Ins|Subs|Rep|Am|Omitted)\.\s", re.IGNORECASE)


def parse_constitution(text: str) -> List[Dict]:
    """Split text into article chunks."""
    # Find all article header positions
    headers = list(_ARTICLE_RE.finditer(text))
    sections: List[Dict] = []
    max_seen = 0
    for i, m in enumerate(headers):
        art_num = m.group(1)
        heading = m.group(2).strip()
        if _FOOTNOTE_HEADING_RE.match(heading):
            continue
        num = int(re.match(r"\d+", art_num).group())
        if max_seen > 50 and num < max_seen - 20:
            # Articles run 1→395 roughly monotonically through the document
            # (lettered insertions like 31A never move the base number
            # backwards). A sharp drop this far in means we've left the
            # Articles and entered a Schedule/Appendix, which independently
            # restarts its own paragraph numbering — e.g. the Ninth
            # Schedule's own "21. Amendment of the Schedule.—" clause would
            # otherwise overwrite the real Article 21 via the upsert.
            break
        max_seen = max(max_seen, num)
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[m.end():end].strip()
        # Remove running headers/footnotes that start with page markers
        content = re.sub(r"^\d+\s+THE CONSTITUTION OF INDIA\s*\(Part.*?\)\s*\n", "", content, flags=re.M)
        content = re.sub(r"^\s*\d+\.\s+(Ins|Subs|Rep)\.\s.*", "", content, flags=re.M)
        content = re.sub(r"\n{3,}", "\n\n", content)
        if content:
            sections.append({"section_number": art_num, "heading": heading, "content": content})
    return sections


# ---------------------------------------------------------------------------
# Sanhita parser (BNS, BNSS) – section splitting (raw text mode)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\s*(\d{1,3})\.\s*(?:\(\d+\)\s*)?([A-Z(])",
    re.M,
)

# Schedules follow the last section (BNSS has two, ~8,000 lines of tables
# and forms) — without this they'd all be glued onto the final section.
_SCHEDULE_RE = re.compile(r"^\s*THE (?:FIRST |SECOND |THIRD )?SCHEDULE\s*$", re.M)

# Sections run 1, 2, 3, ... with no lettered insertions in the new Sanhitas.
# A number already taken, or far from the last header, is a margin note that
# happens to end in a section number — e.g. BNSS s.151's note "...under
# sections 148, 149 and\n150." — and would otherwise split s.151 and, via
# the (statute_id, section_number) upsert, overwrite the real s.150. Small
# steps back are allowed: pdftotext sometimes emits a section just ahead of
# its predecessor (BNSS s.170 comes out before s.169).
_MAX_SECTION_JUMP = 5


def parse_bns(text: str) -> List[Dict]:
    """Split gazette text into section chunks."""
    schedule = _SCHEDULE_RE.search(text)
    if schedule:
        text = text[:schedule.start()]
    headers = []
    seen = set()
    last = 0
    for m in _SECTION_RE.finditer(text):
        num = int(m.group(1))
        if num not in seen and abs(num - last) <= _MAX_SECTION_JUMP:
            headers.append(m)
            seen.add(num)
            last = max(last, num)
    sections: List[Dict] = []
    for i, m in enumerate(headers):
        sec_num = m.group(1)
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[m.start():end].strip()
        # Strip running headers
        content = re.sub(r"^\d+\s+THE GAZETTE OF INDIA.*$", "", content, flags=re.M)
        content = re.sub(r"^\s*_{3,}\s*\n\s*_{3,}", "", content, flags=re.M)
        content = re.sub(r"\n{3,}", "\n\n", content)
        if content:
            sections.append({
                "section_number": sec_num,
                "heading": None,
                "content": content,
            })
    return sections


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model():
    from sentence_transformers import SentenceTransformer
    print("Loading BGE-base model…")
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed_chunks(model, chunks: List[str], batch: int = EMBED_BATCH) -> np.ndarray:
    embeddings = []
    for start in range(0, len(chunks), batch):
        batch_emb = model.encode(chunks[start:start + batch], show_progress_bar=False,
                                 normalize_embeddings=True)
        embeddings.append(batch_emb)
        if start % 500 == 0 and start > 0:
            print(f"  embedded {start}/{len(chunks)}")
    return np.vstack(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Ingest one statute
# ---------------------------------------------------------------------------

def ingest_one(pdf_path: Path, *, layout: bool, title: str, short_title: str,
               year: int, model, conn, dsn: str) -> Dict[str, int]:
    """Returns {"parsed": int, "embedded": int, "inserted": int} — main()
    aggregates these across all statute PDFs into one MLflow run."""
    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"  title={title}  year={year}  layout={layout}")

    text = pdf_to_text(str(pdf_path), layout=layout)
    if layout:
        chunks = parse_constitution(text)
    else:
        chunks = parse_bns(text)

    print(f"  Parsed {len(chunks)} sections/articles")
    if not chunks:
        print("  ⚠ Nothing to ingest – skipping")
        return {"parsed": 0, "embedded": 0, "inserted": 0}

    # Embed
    texts = [c["content"] for c in chunks]
    embeddings = embed_chunks(model, texts)
    print(f"  Embedded {len(embeddings)} vectors")

    # Batch insert
    cur = conn.cursor()
    statute_id = upsert_statute(cur, title, short_title, year,
                                source_file=relative_source_path(str(pdf_path)))
    print(f"  statute_id = {statute_id}")

    BATCH_SIZE = 500
    inserted = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch_embs = embeddings[start:start + BATCH_SIZE]

        values = []
        for c, emb in zip(batch_chunks, batch_embs):
            values.append((
                statute_id,
                c["section_number"],
                c.get("heading"),
                c["content"],
                emb.tolist(),
            ))

        cur.executemany(
            """INSERT INTO statute_sections (statute_id, section_number, heading, content)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (statute_id, section_number) DO UPDATE SET
                   heading = EXCLUDED.heading,
                   content = EXCLUDED.content""",
            [(v[0], v[1], v[2], v[3]) for v in values],
        )

        cur.executemany(
            """INSERT INTO statute_embeddings (section_id, embedding)
               SELECT section_id, %s FROM statute_sections
               WHERE statute_id=%s AND section_number=%s
               ON CONFLICT (section_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
            [(list(v[4]), v[0], v[1]) for v in values],
        )

        inserted += len(batch_chunks)
        if inserted % 1000 == 0 or inserted == len(chunks):
            print(f"  inserted {inserted}/{len(chunks)}")

    conn.commit()
    print(f"  ✓ Done – {len(chunks)} sections ingested for {short_title}")
    return {"parsed": len(chunks), "embedded": len(embeddings), "inserted": inserted}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest statutes into Legal RAG")
    parser.add_argument("--dir", required=True, help="Directory containing statute PDFs")
    parser.add_argument("--dsn", default=None, help="DB connection string override")
    args = parser.parse_args()

    pdf_dir = Path(args.dir)
    if not pdf_dir.is_dir():
        sys.exit(f"Not a directory: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir}")

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}")

    # Classify PDFs by filename
    configs = []
    for p in pdfs:
        name = p.name.lower()
        if "constitution" in name:
            configs.append((p, True,  "Constitution of India, 1950", "Constitution", 1950))
        # Before BNS: "bns" is also inside "bnss". Same rule as
        # backend/documents.py's statute_matches_file.
        elif "bnss" in name or "nagarik" in name or "suraksha" in name:
            configs.append((p, False, "Bharatiya Nagarik Suraksha Sanhita, 2023", "BNSS", 2023))
        elif "bns" in name or "nyaya" in name or "penal" in name:
            configs.append((p, False, "Bharatiya Nyaya Sanhita, 2023", "BNS", 2023))
        else:
            print(f"  ⚠ Skipping unknown statute: {p.name} (add to config dict)")

    if not configs:
        sys.exit("No recognized statute PDFs")

    model = load_model()
    dsn = args.dsn
    conn = get_conn(dsn)

    start_time = time.time()
    try:
        ensure_schema(conn)
        total_parsed = 0
        total_embedded = 0
        total_inserted = 0
        for pdf_path, layout, title, short, year in configs:
            stats = ingest_one(pdf_path, layout=layout, title=title, short_title=short,
                                year=year, model=model, conn=conn, dsn=dsn)
            total_parsed += stats["parsed"]
            total_embedded += stats["embedded"]
            total_inserted += stats["inserted"]
    finally:
        conn.close()

    tracking.log_ingestion_run(
        "ingest_statutes.py",
        params={"dir": str(pdf_dir)},
        metrics={
            "statute_pdfs": len(configs),
            "sections_parsed": total_parsed,
            "sections_embedded": total_embedded,
            "sections_inserted": total_inserted,
            "duration_seconds": round(time.time() - start_time, 2),
        },
    )

    print(f"\n{'='*60}")
    print("Statute ingestion complete ✓")


if __name__ == "__main__":
    main()
