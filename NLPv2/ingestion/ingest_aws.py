#!/usr/bin/env python3
"""Ingest the AWS open-data Indian Supreme Court judgments into Postgres.

Source: s3://indian-supreme-court-judgments (CC-BY-4.0, attribution required).
Unlike ingest.py (which parses party/date/bench out of PDF headers), this takes
all metadata from the dataset's metadata.parquet and uses the PDF only for text.

Inputs (see docs at the bottom of this file for how to fetch them):
  --metadata DIR   directory containing year=YYYY/metadata.parquet (or *.parquet)
  --pdfs DIR       directory (searched recursively) of English PDFs named
                   <path>_EN.pdf, as extracted from data/tar/year=*/english/

Run from NLPv2/ingestion. DB is chosen by DB_HOST/DB_PORT/DB_NAME/DB_USER/
DB_PASSWORD (same variables as ingest.py; DB_NAME defaults to legal_rag).
Safe to re-run: judgments whose cnr is already stored are skipped.

    DB_NAME=legal_rag_v2 python ingest_aws.py --metadata ../../data/aws/metadata \
        --pdfs ../../data/aws/pdf [--year 1995] [--limit 50] [--no-embed]
"""
import argparse
import glob
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import extract_text_from_pdf, get_embeddings_batch  # noqa: E402  (also puts backend/ on sys.path)
from scr_chunker import chunk_scr, clean_to_paragraphs, context_prefix  # noqa: E402
from db_schema import ensure_schema  # noqa: E402
from documents import relative_source_path  # noqa: E402

# --------------------------------------------------------------------------
# Metadata mapping
# --------------------------------------------------------------------------

def _clean_name(s) -> str:
    s = html.unescape(str(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def _parse_date(s):
    try:
        from datetime import datetime
        return datetime.strptime(str(s).strip(), "%d-%m-%Y").date()
    except Exception:
        return None


def _bench(judge) -> list:
    parts = re.split(r",|\bAND\b|&", html.unescape(str(judge or "")), flags=re.I)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def metadata_to_row(r) -> dict:
    cit = _clean_name(r.get("citation"))
    neutral = _clean_name(r.get("case_id"))
    reporters = [c for c in (cit,) if c]
    if "INSC" in neutral:
        reporters.append(neutral)
    return {
        "petitioner": _clean_name(r.get("petitioner")) or "Unknown",
        "respondent": _clean_name(r.get("respondent")) or "Unknown",
        "court": _clean_name(r.get("court")) or "Supreme Court of India",
        "date": _parse_date(r.get("decision_date")),
        "bench": _bench(r.get("judge")),
        "citations": {
            "header": cit,
            "reporters": reporters,
            "neutral": neutral or None,
            "cnr": _clean_name(r.get("cnr")) or None,
            "path": _clean_name(r.get("path")) or None,
        },
        "cnr": _clean_name(r.get("cnr")) or None,
        "disposal": _clean_name(r.get("disposal_nature")) or None,
    }


# --------------------------------------------------------------------------
# Loading + insert
# --------------------------------------------------------------------------

def load_metadata(meta_dir: str, year=None):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(meta_dir, "**", "*.parquet"), recursive=True))
    if year:
        files = [f for f in files if f"year={year}" in f or os.path.basename(f).startswith(str(year))]
    if not files:
        sys.exit(f"no parquet files under {meta_dir}")
    cols = ["title", "petitioner", "respondent", "judge", "citation", "case_id", "cnr",
            "decision_date", "disposal_nature", "court", "path"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("cnr").reset_index(drop=True)  # same case listed under 2 year folders
    print(f"metadata rows: {before}, unique by cnr: {len(df)}")
    return df


def index_pdfs(pdf_dir: str):
    idx = {}
    for p in Path(pdf_dir).rglob("*_EN.pdf"):
        idx[p.name[: -len("_EN.pdf")]] = p
    print(f"english pdfs found: {len(idx)}")
    return idx


def find_pdf(idx, path_key):
    return idx.get(path_key) or idx.get("S_" + path_key)


def ingest_one(conn, row: dict, pdf: Path, embed: bool):
    raw = extract_text_from_pdf(str(pdf))
    if not raw.strip():
        return {"error": "no_text"}
    paragraphs = clean_to_paragraphs(raw)
    text = "\n\n".join(paragraphs)
    if len(text) < 500:
        return {"error": "short_text"}
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO judgments (petitioner, respondent, court, date_of_judgment, bench,
                   citations, judgment_text, source_file, cnr, disposal_nature)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (row["petitioner"], row["respondent"], row["court"], row["date"], row["bench"],
             json.dumps(row["citations"]), text, relative_source_path(str(pdf)),
             row["cnr"], row["disposal"]),
        )
        jid = cur.fetchone()[0]
        chunks = [(s, c) for s, c in chunk_scr(paragraphs) if c.strip()]
        prefix = context_prefix(row["petitioner"], row["respondent"], row["date"], row["bench"],
                                row["citations"]["reporters"], row["disposal"])
        rows = execute_values(
            cur,
            "INSERT INTO judgment_chunks (judgment_id, section, content) VALUES %s RETURNING chunk_id",
            [(jid, s, c) for s, c in chunks], fetch=True,
        )
        n_emb = 0
        if embed:
            # embed with the case label attached; stored content stays clean
            embs = get_embeddings_batch([prefix + "\n" + c for _, c in chunks])
            er = [(r[0], e) for r, e in zip(rows, embs) if e is not None]
            if er:
                execute_values(cur, "INSERT INTO judgment_embeddings (chunk_id, embedding) VALUES %s", er)
            n_emb = len(er)
        conn.commit()
        return {"error": None, "chunks": len(chunks), "embeddings": n_emb}
    except Exception as e:
        conn.rollback()
        return {"error": f"db:{e}"}
    finally:
        cur.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-embed", action="store_true", help="insert text + chunks only")
    a = ap.parse_args()

    df = load_metadata(a.metadata, a.year)
    idx = index_pdfs(a.pdfs)
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"), port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ.get("DB_NAME", "legal_rag"), user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )
    ensure_schema(conn)
    with conn.cursor() as c:
        c.execute("SELECT cnr FROM judgments WHERE cnr IS NOT NULL")
        done = {r[0] for r in c.fetchall()}
    stats = {"ok": 0, "skipped_existing": 0, "no_pdf": 0, "no_text": 0, "short_text": 0, "db_error": 0,
             "chunks": 0, "embeddings": 0}
    missing = []
    t0 = time.time()
    todo = df.to_dict("records")
    if a.limit:
        todo = todo[: a.limit]
    for rec in tqdm(todo, unit="doc"):
        if rec.get("cnr") in done:
            stats["skipped_existing"] += 1
            continue
        pdf = find_pdf(idx, str(rec.get("path")))
        if pdf is None:
            stats["no_pdf"] += 1
            missing.append(str(rec.get("path")))
            continue
        res = ingest_one(conn, metadata_to_row(rec), pdf, embed=not a.no_embed)
        if res["error"] is None:
            stats["ok"] += 1
            stats["chunks"] += res["chunks"]
            stats["embeddings"] += res["embeddings"]
        elif res["error"].startswith("db:"):
            stats["db_error"] += 1
            tqdm.write(f"{rec.get('path')}: {res['error']}")
        else:
            stats[res["error"]] += 1
    conn.close()
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats, indent=2))
    if missing:
        Path("aws_missing_pdfs.txt").write_text("\n".join(missing))
        print(f"{len(missing)} metadata rows had no PDF; listed in aws_missing_pdfs.txt")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Fetching the data (public bucket, no AWS account; prefer tar + parquet for bulk)
#
#   mkdir -p data/aws/metadata data/aws/pdf && cd data/aws
#   aws s3 sync s3://indian-supreme-court-judgments/metadata/parquet metadata --no-sign-request
#   for y in $(seq 1950 2026); do
#     aws s3 cp s3://indian-supreme-court-judgments/data/tar/year=$y/english/ pdf/year=$y --recursive \
#         --no-sign-request   # english.tar and any part-*.tar
#   done
#   # then untar each: for t in pdf/year=*/*.tar; do tar -xf "$t" -C "$(dirname "$t")"; done
# ---------------------------------------------------------------------------
