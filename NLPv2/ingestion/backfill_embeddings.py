"""Embed the existing chunks with another model, into that model's own tables.

    python backfill_embeddings.py --model inlegal --limit 2000   # throughput sample
    python backfill_embeddings.py --model inlegal                # everything, resumable

Reads judgment_chunks.content and statute_sections.content as ingestion
already split them (same chunk_ids, so models are compared on identical
chunks) and writes <model>'s tables from backend/retrieval/embedding_models.py,
creating them if needed. It never writes the BGE baseline tables and never
changes or drops an existing table: a table whose dimension doesn't match the
spec is an error. Re-running skips chunks already embedded, so an interrupted
run just continues. The HNSW index is built once every chunk has a vector
(faster than growing it row by row); --no-index skips it.

Statute vectors embed the section content only, as ingest_statutes.py does.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import tracking  # noqa: E402
from retrieval import embedding_models  # noqa: E402
from scr_chunker import context_prefix  # noqa: E402  (same one-line case label ingest_aws.py embeds with)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# target -> (source table, id column, text to embed)
TARGETS = {
    "judgments": ("judgment_chunks", "chunk_id", "coalesce(src.content, '')"),
    "statutes": ("statute_sections", "section_id", "coalesce(src.content, '')"),
}
PAGE_ROWS = 4096


def connect():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ.get("DB_NAME", "legal_rag"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )


def check_pgvector(cur, spec) -> None:
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    if not row:
        raise SystemExit("pgvector extension is not installed in this database")
    version = tuple(int(x) for x in row[0].split(".")[:2])
    if spec.vector_type == "halfvec" and version < (0, 7):
        raise SystemExit(f"halfvec needs pgvector >= 0.7, database has {row[0]}")


def ensure_table(cur, spec, target: str, ctx_prefix: bool = False) -> str:
    src, id_col, _ = TARGETS[target]
    table = spec.judgment_table if target == "judgments" else spec.statute_table
    want = f"{spec.vector_type}({spec.dim})"
    cur.execute(
        "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(%s) AND a.attname = 'embedding' AND NOT a.attisdropped",
        (table,),
    )
    row = cur.fetchone()
    # The table's comment records which weights filled it, so a retrained
    # checkpoint never adds vectors to a table made by an older one.
    stamp = f"{spec.key} {spec.model_id} {embedding_models.model_fingerprint(spec)}"
    if ctx_prefix and target == "judgments":
        stamp += " +ctxprefix"   # embedded with the case label, so a table without it must not be mixed in
    if row:
        if row[0] != want:
            raise SystemExit(
                f"{table} exists with embedding {row[0]}, spec wants {want}. Not touching it: "
                "use a different table name in embedding_models.py, or drop it yourself if it's disposable."
            )
        cur.execute("SELECT obj_description(to_regclass(%s), 'pg_class')", (table,))
        existing = cur.fetchone()[0]
        if existing and existing != stamp:
            raise SystemExit(
                f"{table} was filled by\n  {existing}\nbut the model now is\n  {stamp}\n"
                f"Mixing them would corrupt the comparison. If the old vectors are disposable: DROP TABLE {table};"
            )
        return table
    cur.execute(
        f"CREATE TABLE {table} ("
        f" {id_col} INTEGER PRIMARY KEY REFERENCES {src}({id_col}) ON DELETE CASCADE,"
        f" embedding {want} NOT NULL,"
        f" created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    cur.execute(f"COMMENT ON TABLE {table} IS %s", (stamp,))
    return table


def missing_count(cur, target: str, table: str) -> int:
    src, id_col, _ = TARGETS[target]
    cur.execute(
        f"SELECT count(*) FROM {src} src WHERE NOT EXISTS "
        f"(SELECT 1 FROM {table} t WHERE t.{id_col} = src.{id_col})"
    )
    return cur.fetchone()[0]


def _prefixed(rows):
    """rows: (chunk_id, content, petitioner, respondent, date, bench, citations, disposal)
    -> (chunk_id, case label + newline + content), as ingest_aws.py embeds BGE."""
    out = []
    for cid, content, pet, resp, date, bench, cit, disp in rows:
        reporters = (cit or {}).get("reporters") if isinstance(cit, dict) else None
        out.append((cid, context_prefix(pet, resp, date, bench or [], reporters, disp) + "\n" + (content or "")))
    return out


def backfill(conn, model, spec, target: str, batch_size: int, limit: int, ctx_prefix: bool = False) -> dict:
    src, id_col, text_expr = TARGETS[target]
    use_ctx = ctx_prefix and target == "judgments"
    with conn.cursor() as cur:
        table = ensure_table(cur, spec, target, ctx_prefix)
        todo = missing_count(cur, target, table)
    conn.commit()
    if limit:
        todo = min(todo, limit)
    print(f"{target}: {todo} rows to embed into {table}")

    done, last_id, started = 0, -1, time.time()
    bar = tqdm(total=todo, unit="chunk")
    while done < todo:
        with conn.cursor() as cur:
            if use_ctx:
                cur.execute(
                    f"SELECT src.{id_col}, coalesce(src.content, ''), j.petitioner, j.respondent, "
                    f"j.date_of_judgment, j.bench, j.citations, j.disposal_nature "
                    f"FROM {src} src JOIN judgments j ON j.id = src.judgment_id "
                    f"WHERE src.{id_col} > %s AND NOT EXISTS "
                    f"(SELECT 1 FROM {table} t WHERE t.{id_col} = src.{id_col}) "
                    f"ORDER BY src.{id_col} LIMIT %s",
                    (last_id, min(PAGE_ROWS, todo - done)),
                )
                rows = _prefixed(cur.fetchall())
            else:
                cur.execute(
                    f"SELECT src.{id_col}, {text_expr} FROM {src} src "
                    f"WHERE src.{id_col} > %s AND NOT EXISTS "
                    f"(SELECT 1 FROM {table} t WHERE t.{id_col} = src.{id_col}) "
                    f"ORDER BY src.{id_col} LIMIT %s",
                    (last_id, min(PAGE_ROWS, todo - done)),
                )
                rows = cur.fetchall()
        if not rows:
            break
        last_id = rows[-1][0]
        # Similar lengths per batch waste less padding.
        rows.sort(key=lambda r: len(r[1]))
        vecs = embedding_models.encode(model, spec, [r[1] for r in rows], "document", batch_size=batch_size)
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"INSERT INTO {table} ({id_col}, embedding) VALUES %s ON CONFLICT ({id_col}) DO NOTHING",
                [(r[0], embedding_models.vector_literal(v)) for r, v in zip(rows, vecs)],
                template=f"(%s, %s::{spec.vector_type})",
                page_size=500,
            )
        conn.commit()
        done += len(rows)
        bar.update(len(rows))
    bar.close()
    seconds = time.time() - started
    with conn.cursor() as cur:
        remaining = missing_count(cur, target, table)
    return {"table": table, "embedded": done, "seconds": seconds, "remaining": remaining}


def build_index(conn, spec, table: str, maintenance_work_mem: str) -> None:
    name = f"{table}_hnsw_idx"
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
        if cur.fetchone()[0]:
            return
    print(f"Building HNSW index {name} (can take a while) ...")
    started = time.time()
    with conn.cursor() as cur:
        # The graph should fit in memory or the build slows sharply (pgvector
        # prints a notice when it doesn't).
        cur.execute("SELECT set_config('maintenance_work_mem', %s, false)", (maintenance_work_mem,))
        cur.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding {spec.cosine_ops})")
    conn.commit()
    print(f"  done in {time.time() - started:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=f"one of {embedding_models.available()} except the baseline")
    parser.add_argument("--target", choices=("judgments", "statutes", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="embed at most N rows per target (throughput sample)")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: sentence-transformers' choice)")
    parser.add_argument("--no-index", action="store_true", help="skip building the HNSW index at the end")
    parser.add_argument("--context-prefix", action="store_true",
                        help="judgments only: prepend the case label (parties, year, citations, bench, outcome) to each "
                             "chunk before embedding, as ingest_aws.py does for BGE. Needs the v3 columns (disposal_nature).")
    parser.add_argument("--maintenance-work-mem", default="2GB",
                        help="Postgres memory for the HNSW build (raise for large/wide tables, e.g. 8GB)")
    args = parser.parse_args()

    spec = embedding_models.get_spec(args.model)
    if spec.is_baseline:
        raise SystemExit("Refusing to backfill the baseline: its tables are written by ingest.py only.")
    try:
        spec.check_indexable()
    except ValueError as e:
        raise SystemExit(str(e))
    targets = ["judgments", "statutes"] if args.target == "all" else [args.target]

    conn = connect()
    try:
        with conn.cursor() as cur:
            check_pgvector(cur, spec)
        conn.commit()
        print(f"Loading {spec.model_id} ...")
        model = embedding_models.load_model(spec, device=args.device)
        print(f"Query prompt: {embedding_models.prompt_for(model, spec, 'query')!r}; "
              f"document prompt: {embedding_models.prompt_for(model, spec, 'document')!r}")
        metrics = {}
        for target in targets:
            res = backfill(conn, model, spec, target, args.batch_size, args.limit, args.context_prefix)
            rate = res["embedded"] / res["seconds"] if res["seconds"] else 0.0
            print(f"{target}: embedded {res['embedded']} in {res['seconds']:.0f} s "
                  f"({rate:.1f}/s), {res['remaining']} still missing")
            if res["remaining"] and rate:
                print(f"  at this rate the rest takes ~{res['remaining'] / rate / 3600:.1f} h")
            metrics.update({
                f"{target}_embedded": res["embedded"], f"{target}_seconds": res["seconds"],
                f"{target}_rows_per_s": rate, f"{target}_remaining": res["remaining"],
            })
            if res["remaining"] == 0 and not args.no_index:
                build_index(conn, spec, res["table"], args.maintenance_work_mem)
        tracking.log_ingestion_run(
            "backfill_embeddings.py",
            {"embedding_model": spec.key, "model_id": spec.model_id, "revision": spec.revision,
             "dim": spec.dim, "vector_type": spec.vector_type, "targets": ",".join(targets),
             "limit": args.limit, "batch_size": args.batch_size},
            metrics,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
