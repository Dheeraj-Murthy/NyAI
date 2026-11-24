import os
import re
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from tqdm import tqdm

# backend/ is a sibling directory, not a package ingestion/ installs — add it
# to sys.path to reuse tracking.py's MLflow setup instead of duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import tracking
from graph.citation_resolver import extract_citation_header, parse_reporter_citations
from db_schema import ensure_schema
from documents import relative_source_path

# Initialize local embedding model
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    MODEL_AVAILABLE = True
    print("Local BGE-base model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"Warning: Failed to load BGE model. Embeddings will be disabled. Error: {e}")


def get_embedding(text: str):
    """Get embedding for text using local BGE model"""
    if not MODEL_AVAILABLE:
        return None
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def get_embeddings_batch(texts: List[str]) -> List[Any]:
    """Embed many chunks in one batched call instead of one .encode() per
    chunk — at full-corpus scale (tens of thousands of judgments) unbatched
    per-chunk calls turn into hundreds of thousands of individual GPU calls,
    dominated by per-call overhead rather than actual compute."""
    if not MODEL_AVAILABLE or not texts:
        return [None] * len(texts)
    try:
        embeddings = embedding_model.encode(
            texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return [None] * len(texts)


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            tqdm.write(f"pdftotext error: {result.stderr}")
        if result.stdout:
            return result.stdout
    except FileNotFoundError:
        tqdm.write("pdftotext not found - install poppler-utils")
    except Exception as e:
        tqdm.write(f"pdftotext failed: {e}")
    return ""


def clean_judgment_text(text: str) -> str:
    """
    Remove repetitive JUDIS boilerplate headers that pdftotext picks up on
    every page. These appear in two forms depending on how pdftotext lays
    out the columns:

      Form 1 (single line):
        http://JUDIS.NIC.IN    SUPREME COURT OF INDIA    Page N of N

      Form 2 (split across lines):
        http://JUDIS.NIC.IN
        SUPREME COURT OF INDIA    Page N of N
    """
    # Form 1 – all on one line
    cleaned = re.sub(
        r'http://JUDIS\.NIC\.IN\s+SUPREME COURT OF INDIA\s+Page \d+ of \d+\s*',
        '',
        text
    )
    # Form 2 – URL alone on a line
    cleaned = re.sub(r'http://JUDIS\.NIC\.IN\s*\n?', '', cleaned)
    # Form 2 – "SUPREME COURT OF INDIA  Page N of N" leftover
    cleaned = re.sub(r'SUPREME COURT OF INDIA\s+Page \d+ of \d+\s*\n?', '', cleaned)

    # Collapse runs of blank lines created by the removals
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


_MONTH_RE = re.compile(
    r'(?i)\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\b'
)
_BENCH_PAREN_RE = re.compile(r'\(([^)]*JJ?\.?)\)', re.IGNORECASE)
_CAUSE_TITLE_SEPARATOR_RE = re.compile(r'^[A-D]?\s*(v\.?|vs\.?|versus)\s*$', re.IGNORECASE)
_MARGIN_MARKER_RE = re.compile(r'^[A-D]\s+')


def _strip_margin_marker(line: str) -> str:
    """Older Supreme Court Reports-style scans (pre-JUDIS digitization,
    spanning the corpus's older decades) prefix lines with a single-letter
    side-margin marker (A/B/C/D) left over from the original two-column
    typeset layout — strip it so it doesn't leak into extracted names."""
    return _MARGIN_MARKER_RE.sub('', line).strip()


def _parse_cause_title_fallback(
    lines: List[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fallback for judgments that don't use PETITIONER:/RESPONDENT: labels
    at all — common in the older part of the corpus, which instead lays out
    a plain cause title:
        <petitioner name(s)>
                v.
        <respondent name(s)>
    Returns (petitioner, respondent, bench_text) — any of which may be None
    if the line count didn't sit within the header window or an entry is
    genuinely absent.
    """
    separator_index = None
    for i, raw_line in enumerate(lines[:20]):
        stripped = _strip_margin_marker(raw_line)
        if stripped and _CAUSE_TITLE_SEPARATOR_RE.match(stripped):
            separator_index = i
            break

    if separator_index is None:
        return None, None, None

    petitioner_lines: List[str] = []
    for raw_line in reversed(lines[:separator_index]):
        stripped = _strip_margin_marker(raw_line)
        if not stripped:
            if petitioner_lines:
                break
            continue
        petitioner_lines.insert(0, stripped)

    respondent_lines: List[str] = []
    bench_text = None
    for raw_line in lines[separator_index + 1 :]:
        stripped = _strip_margin_marker(raw_line)
        if not stripped:
            if respondent_lines:
                break
            continue
        bench_match = _BENCH_PAREN_RE.search(stripped)
        if bench_match:
            bench_text = bench_match.group(1)
            break
        if _MONTH_RE.search(stripped):
            break
        respondent_lines.append(stripped)

    petitioner = ' '.join(petitioner_lines).strip() or None
    respondent = ' '.join(respondent_lines).strip() or None
    return petitioner, respondent, bench_text


def parse_judgment_metadata(text: str, filename: str) -> Dict[str, Any]:
    """
    Extract metadata from judgment text.
    Expects already-cleaned text — no internal cleaning is done here.
    """
    metadata = {
        'petitioner': 'Unknown',
        'respondent': 'Unknown',
        'court': 'Supreme Court of India',
        'date_of_judgment': None,
        'bench': [],
        'citations': {}
    }

    header_lines = text.split('\n')[:20]

    petitioner_name = None
    respondent_name = None

    for line_index, current_line in enumerate(header_lines):
        stripped_line = current_line.strip()

        if stripped_line == 'PETITIONER:':
            petitioner_name = _extract_name_after_header(
                header_lines, line_index + 1,
                ['RESPONDENT:', 'Vs.', 'vs.', 'V.']
            )
        elif stripped_line == 'RESPONDENT:':
            respondent_name = _extract_name_after_header(
                header_lines, line_index + 1,
                ['DATE OF JUDGMENT:', 'BENCH:', 'ACT:']
            )

    fallback_bench_text = None
    if not petitioner_name or not respondent_name:
        fb_petitioner, fb_respondent, fallback_bench_text = _parse_cause_title_fallback(header_lines)
        petitioner_name = petitioner_name or fb_petitioner
        respondent_name = respondent_name or fb_respondent

    if petitioner_name:
        metadata['petitioner'] = petitioner_name
    if respondent_name:
        metadata['respondent'] = respondent_name

    date_pattern = r'DATE OF JUDGMENT:\s*(\d{2}/\d{2}/\d{4})'
    date_match = re.search(date_pattern, text)
    if date_match:
        date_str = date_match.group(1)
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
            metadata['date_of_judgment'] = date_obj.strftime('%Y-%m-%d')
        except Exception:
            pass

    # The judgment's own reporter citations, from its JUDIS "CITATION:" header
    # (e.g. "1953 AIR 75  1953 SCR 215"). Edge building reads the same header
    # from the stored text, so this column is informational.
    citation_header = extract_citation_header(text)
    if citation_header:
        metadata['citations'] = {
            'header': citation_header,
            'reporters': [c.raw for c in parse_reporter_citations(citation_header)],
        }

    bench_pattern = r'BENCH:\s*\[?([^\]\n]+)'
    bench_match = re.search(bench_pattern, text)
    bench_text = bench_match.group(1).strip() if bench_match else fallback_bench_text
    if bench_text:
        judges = re.findall(
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z])?)\s*\.?\s*J\.?',
            bench_text
        )
        if judges:
            metadata['bench'] = judges

    return metadata


def _extract_name_after_header(
    lines: List[str], start_index: int, stop_markers: List[str]
) -> str | None:
    """Extract the name that appears on the line(s) after a header keyword."""
    for next_line_index in range(start_index, len(lines)):
        line_content = lines[next_line_index].strip()
        if line_content in stop_markers:
            break
        if (
            line_content
            and not line_content.startswith('ACT:')
            and not line_content.startswith('BENCH:')
            and not line_content.startswith('Vs.')
            and not line_content.startswith('vs.')
            and not line_content.startswith('V.')
        ):
            return line_content
    return None


def chunk_judgment_text(text: str) -> List[Tuple[str, str]]:
    """Split judgment text into logical chunks based on common legal judgment structure."""

    section_identifiers = {
        'facts': [
            r'(?i)facts of the case',
            r'(?i)\bfacts\b',
            r'(?i)background',
            r'(?i)facts and circumstances',
            r'(?i)the facts are',
            r'(?i)brief facts'
        ],
        'issues': [
            r'(?i)issues? arising',
            r'(?i)questions? for consideration',
            r'(?i)points? for determination',
            r'(?i)the issue is',
            r'(?i)issues framed'
        ],
        'arguments': [
            r'(?i)arguments? advanced',
            r'(?i)submissions?',
            r'(?i)contentions?',
            r'(?i)learned counsel',
            r'(?i)learned senior counsel',
            r'(?i)learned attorney'
        ],
        'ratio': [
            r'(?i)ratio decidendi',
            r'(?i)legal principles',
            r'(?i)principles laid down',
            r'(?i)the law is',
            r'(?i)the legal position',
            r'(?i)precedent',
            r'(?i)jurisprudence'
        ],
        'judgment': [
            r'(?i)\bjudgment\b',
            r'(?i)\border\b',
            r'(?i)\bdecision\b',
            r'(?i)\bconclusion\b',
            r'(?i)\bholding\b',
            r'(?i)we hold',
            r'(?i)we conclude',
            r'(?i)\baccordingly\b',
            r'(?i)in the result'
        ]
    }

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []

    # Long stretches of text with no recognized section header (common in
    # older judgments) would otherwise accumulate into a single oversized
    # chunk — cap it so retrieval stays granular and the full chunk can
    # actually fit in the LLM's context window later.
    max_chunk_chars = 3000

    current_section = 'facts'
    current_section_text = []
    current_section_length = 0

    for paragraph in paragraphs:
        identified_section = _identify_section_type(paragraph, section_identifiers)

        if identified_section and identified_section != current_section and current_section_text:
            chunks.append((current_section, '\n'.join(current_section_text)))
            current_section_text = []
            current_section_length = 0
            current_section = identified_section

        if current_section_text and current_section_length + len(paragraph) > max_chunk_chars:
            chunks.append((current_section, '\n'.join(current_section_text)))
            current_section_text = []
            current_section_length = 0

        if len(paragraph) > max_chunk_chars:
            # A single "paragraph" (blank-line-delimited span) can itself be
            # huge when pdftotext doesn't preserve blank lines well — split
            # it on word boundaries so nothing bypasses the size cap.
            words = paragraph.split()
            piece = []
            piece_length = 0
            for word in words:
                if piece_length + len(word) + 1 > max_chunk_chars and piece:
                    chunks.append((current_section, ' '.join(piece)))
                    piece = []
                    piece_length = 0
                piece.append(word)
                piece_length += len(word) + 1
            if piece:
                current_section_text = [' '.join(piece)]
                current_section_length = piece_length
            continue

        current_section_text.append(paragraph)
        current_section_length += len(paragraph)

    if current_section_text:
        chunks.append((current_section, '\n'.join(current_section_text)))

    if not chunks:
        chunks = _create_size_based_chunks(text)

    return chunks


def _identify_section_type(
    paragraph: str, section_patterns: Dict[str, List[str]]
) -> str | None:
    """Identify which section a paragraph belongs to based on patterns."""
    for section_type, patterns in section_patterns.items():
        for pattern in patterns:
            if re.search(pattern, paragraph):
                return section_type
    return None


def _create_size_based_chunks(text: str) -> List[Tuple[str, str]]:
    """Create fixed-size chunks when no clear section headings are found."""
    words = text.split()
    chunk_size = 500
    chunks = []

    for start_index in range(0, len(words), chunk_size):
        chunk_words = words[start_index:start_index + chunk_size]
        chunk_text = ' '.join(chunk_words)

        position_ratio = start_index / len(words) if words else 0
        if position_ratio < 0.3:
            section = 'facts'
        elif position_ratio < 0.6:
            section = 'issues'
        elif position_ratio < 0.8:
            section = 'arguments'
        else:
            section = 'judgment'

        chunks.append((section, chunk_text))

    return chunks


def ingest_judgment_from_pdf(pdf_path: str, conn) -> Dict[str, Any]:
    """Ingest a single PDF judgment into the database, using a connection
    shared across the whole ingestion run (see main()) rather than opening
    a fresh one per document — at full-corpus scale that's tens of
    thousands of avoidable connection setup/teardown round-trips.

    Returns {"judgment_id": int|None, "chunks": int, "embeddings": int,
    "error": str|None} — main() aggregates these into MLflow run metrics,
    so failure needs to be distinguishable from "no text extracted" rather
    than both collapsing into the same None."""
    # 1. Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        tqdm.write(f"Warning: No text extracted from {pdf_path}")
        return {"judgment_id": None, "chunks": 0, "embeddings": 0, "error": "no_text_extracted"}

    # 2. Clean BEFORE anything else — metadata parsing, chunking, and DB storage
    #    all operate on the same clean text.
    text = clean_judgment_text(raw_text)

    # 3. Parse metadata from clean text
    metadata = parse_judgment_metadata(text, os.path.basename(pdf_path))

    cur = conn.cursor()

    try:
        # 4. Insert judgment — store the clean text, not the raw text
        cur.execute(
            """
            INSERT INTO judgments
                (petitioner, respondent, court, date_of_judgment, bench, citations, judgment_text,
                 source_file)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                metadata['petitioner'],
                metadata['respondent'],
                metadata['court'],
                metadata['date_of_judgment'],
                metadata['bench'],
                json.dumps(metadata['citations']),
                text,           # <-- clean text
                # Where the original PDF lives, so citations can link to it.
                relative_source_path(pdf_path),
            )
        )

        result = cur.fetchone()
        judgment_id = result[0] if result else None

        # 5. Chunk the clean text
        chunks = chunk_judgment_text(text)

        valid_sections = {'facts', 'issues', 'arguments', 'ratio', 'judgment'}

        normalized_chunks = [
            (section if section in valid_sections else 'judgment', chunk_text)
            for section, chunk_text in chunks
        ]

        # 5b. Insert all of this judgment's chunks in one multi-row INSERT
        # instead of one round-trip per chunk. execute_values with fetch=True
        # returns the RETURNING rows in the same order the input rows were
        # given, so chunk_ids lines up positionally with normalized_chunks.
        chunk_rows = execute_values(
            cur,
            """
            INSERT INTO judgment_chunks (judgment_id, section, content)
            VALUES %s
            RETURNING chunk_id
            """,
            [(judgment_id, section, chunk_text) for section, chunk_text in normalized_chunks],
            fetch=True,
        )
        chunk_ids = [row[0] for row in chunk_rows]
        chunk_texts = [chunk_text for _, chunk_text in normalized_chunks]

        # 6. Generate embeddings for every chunk in this judgment in one
        # batched call instead of one .encode() per chunk.
        embeddings = get_embeddings_batch(chunk_texts)

        embedding_rows = [
            (chunk_id, embedding)
            for chunk_id, embedding in zip(chunk_ids, embeddings)
            if embedding is not None
        ]
        missing = len(chunk_ids) - len(embedding_rows)
        if missing:
            tqdm.write(f"Warning: {missing} chunk(s) had no embedding generated (BGE model unavailable)")

        if embedding_rows:
            execute_values(
                cur,
                """
                INSERT INTO judgment_embeddings (chunk_id, embedding)
                VALUES %s
                """,
                embedding_rows,
            )

        conn.commit()
        return {
            "judgment_id": judgment_id,
            "chunks": len(chunk_ids),
            "embeddings": len(embedding_rows),
            "error": None,
        }

    except Exception as e:
        conn.rollback()
        tqdm.write(f"Error processing {pdf_path}: {e}")
        return {"judgment_id": None, "chunks": 0, "embeddings": 0, "error": str(e)}
    finally:
        cur.close()


def main():
    """Process all PDFs in the given --input directory (defaults to the
    bundled fixtures/sample_judgments/ set for a quick smoke test)."""
    parser = argparse.ArgumentParser(description="Ingest judgment PDFs into Postgres")
    parser.add_argument(
        "--input",
        type=str,
        default="fixtures/sample_judgments",
        help="Directory of PDF files to ingest (default: fixtures/sample_judgments)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process...")

    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ.get("DB_NAME", "legal_rag"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )
    ensure_schema(conn)
    start_time = time.time()
    try:
        successful = 0
        no_text_count = 0
        error_count = 0
        total_chunks = 0
        total_embeddings = 0

        progress = tqdm(pdf_files, desc="Ingesting", unit="doc")
        for pdf_file in progress:
            result = ingest_judgment_from_pdf(str(pdf_file), conn)
            if result["judgment_id"]:
                successful += 1
                total_chunks += result["chunks"]
                total_embeddings += result["embeddings"]
            elif result["error"] == "no_text_extracted":
                no_text_count += 1
            else:
                error_count += 1
            progress.set_postfix(ok=successful, failed=progress.n + 1 - successful)

        duration = time.time() - start_time
        failed = len(pdf_files) - successful

        print(
            f"\nProcessing complete. "
            f"Successfully ingested {successful}/{len(pdf_files)} judgments."
        )

        tracking.log_ingestion_run(
            "ingest.py",
            params={
                "input_dir": str(input_dir),
                "embedding_model": "BAAI/bge-base-en-v1.5",
            },
            metrics={
                "pdf_count": len(pdf_files),
                "successful": successful,
                "failed": failed,
                "no_text_extracted": no_text_count,
                "errors": error_count,
                "chunks_created": total_chunks,
                "embeddings_created": total_embeddings,
                "duration_seconds": round(duration, 2),
            },
        )

        # Small run manifest — gives the DVC pipeline stage a file-based
        # metrics output (dvc.yaml declares this under `metrics:`, so
        # `dvc metrics diff` can show these numbers changing across git
        # commits) since ingestion otherwise writes only to Postgres. Same
        # fields as the MLflow run above — one is the per-run dashboard,
        # the other is the git-diffable snapshot; they shouldn't diverge.
        manifest = {
            "input_dir": str(input_dir),
            "pdf_count": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "no_text_extracted": no_text_count,
            "errors": error_count,
            "chunks_created": total_chunks,
            "embeddings_created": total_embeddings,
            "duration_seconds": round(duration, 2),
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "timestamp": datetime.utcnow().isoformat(),
        }
        manifest_path = Path("outputs/ingest_manifest.json")
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
