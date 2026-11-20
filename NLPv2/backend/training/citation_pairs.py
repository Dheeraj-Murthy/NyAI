#!/usr/bin/env python3
"""Mine (passage -> cited judgment) training pairs from the citation graph.

Same construction as the masked-citation benchmark (benchmark/
masked_citation_gold.py, whose code this reuses): the paragraph of judgment A
around a citation of B, with every citation, case title, "supra" and the cited
cases' name words masked. The positive is B's chunk sharing most content words
with the passage. Optionally, hard negatives come from the BGE baseline: chunks
it ranks 10-50 for the passage, from judgments that aren't relevant (the top
ranks are skipped because other judgments stating the same rule are often
right answers that just aren't labelled).

Held out, so the benchmarks stay unseen: every judgment referenced by the gold
sets passed with --holdout (sources, targets, relevant and excluded judgments,
and the judgments of any chunk ids listed). An edge touching a held-out
judgment on either side is dropped. A further --dev-percent of citing
judgments goes to dev.jsonl, never to train.jsonl.

    cd NLPv2/backend
    python training/citation_pairs.py --version v1 \\
        --holdout benchmark/data/masked_citation_gold_v1.json benchmark/data/case_lookup_gold_v1.json \\
                  benchmark/data/gold_set.json benchmark/data/aila_gold_v1.json \\
        --hard-negatives 3
    # -> <repo>/data/training/citation_pairs_v1/{train,dev}.jsonl + meta.json; then dvc add it
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_training_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_training_dir)
_repo_dir = os.path.dirname(os.path.dirname(_backend_dir))
sys.path.insert(0, _backend_dir)
sys.path.insert(0, os.path.join(_backend_dir, "benchmark"))

import masked_citation_gold as mcg  # noqa: E402

DEFAULT_OUT_ROOT = os.path.join(_repo_dir, "data", "training")
MIN_CHUNK_WORDS = 30
MAX_NON_ASCII = 0.2  # pdftotext mojibake (benchmark/REPORT.md, known issues)
NEG_RANK_LO, NEG_RANK_HI = 10, 50
_WORD_RE = re.compile(r"[a-z][a-z]{3,}")
_STOP = frozenset(
    "that this with from have been were which their there would shall such said under upon into also "
    "other than then them they what when where whom will being only made court case section appellant "
    "respondent learned counsel judgment order high supreme state".split()
)


# ---------------------------------------------------------------------------
# Pure helpers (offline-tested)
# ---------------------------------------------------------------------------


def holdout_ids(gold_files: Sequence[str], chunk_to_judgment: Callable[[List[int]], Dict[int, int]]) -> Tuple[Set[int], Dict[str, int]]:
    """Judgment ids referenced anywhere in the gold sets' records: any key
    ending in judgment_id(s), plus the judgments of keys ending in chunk_id(s)."""
    judgments: Set[int] = set()
    chunks: Set[int] = set()
    per_file: Dict[str, int] = {}

    def ints(v) -> Iterable[int]:
        if isinstance(v, int):
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, int):
                    yield x

    for path in gold_files:
        with open(path) as f:
            gold = json.load(f)
        before = len(judgments) + len(chunks)
        for rec in gold.get("records", []):
            for k, v in rec.items():
                if k.endswith(("judgment_id", "judgment_ids", "target_id")):
                    judgments.update(ints(v))
                elif k.endswith(("chunk_id", "chunk_ids")):
                    chunks.update(ints(v))
        per_file[os.path.basename(path)] = len(judgments) + len(chunks) - before
    if chunks:
        judgments.update(chunk_to_judgment(sorted(chunks)).values())
    return judgments, per_file


def is_dev(source_judgment_id: int, dev_percent: float) -> bool:
    h = int(hashlib.sha256(str(source_judgment_id).encode()).hexdigest()[:8], 16)
    return (h % 10000) < dev_percent * 100


def content_words(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def usable_chunk(text: str) -> bool:
    if not text or len(text.split()) < MIN_CHUNK_WORDS:
        return False
    return sum(1 for c in text if ord(c) > 127) / len(text) <= MAX_NON_ASCII


def best_chunk(query: str, chunks: Sequence[Tuple[int, str]]) -> Optional[Tuple[int, str]]:
    """The usable chunk sharing most content words with the query (ties: earliest)."""
    q = content_words(query)
    best, best_score = None, -1
    for cid, text in chunks:
        if not usable_chunk(text):
            continue
        score = len(q & content_words(text))
        if score > best_score:
            best, best_score = (cid, text), score
    return best


def build_pairs(
    edges: List[Dict[str, Any]],
    meta: Dict[int, Dict[str, Any]],
    fetch_texts: Callable[[List[int]], Dict[int, str]],
    fetch_chunks: Callable[[List[int]], Dict[int, List[Tuple[int, str]]]],
    holdout: Set[int],
    seed: int = 1,
    max_per_target: int = 20,
    max_pairs: int = 0,
    short_fraction: float = 0.3,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """One pair per (citing, cited) judgment pair, landmark targets capped at
    max_per_target. A short (40-word) query replaces the passage for
    short_fraction of pairs, since real user queries are short."""
    rng = random.Random(seed)
    skipped: Counter = Counter()
    eligible = mcg.eligible_edges(edges, meta)
    by_source: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        by_source[e["source_judgment_id"]].append(e)

    # One edge per (source, target): the first in a seeded order.
    seen_pair: Set[Tuple[int, int]] = set()
    chosen: List[Dict[str, Any]] = []
    for e in mcg.sample_order(eligible, seed):
        key = (e["source_judgment_id"], e["target_judgment_id"])
        if key in seen_pair:
            continue
        seen_pair.add(key)
        if key[0] in holdout or key[1] in holdout:
            skipped["held-out judgment"] += 1
            continue
        chosen.append(e)

    # Group by source so each judgment's text is fetched once.
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in chosen:
        groups[e["source_judgment_id"]].append(e)
    per_target: Counter = Counter()
    pairs: List[Dict[str, Any]] = []
    for src in sorted(groups, key=lambda s: rng.random()):
        if max_pairs and len(pairs) >= max_pairs:
            break
        group = groups[src]
        source_edges = by_source[src]
        targets = sorted({e["target_judgment_id"] for e in source_edges if e["target_judgment_id"]})
        texts = fetch_texts([src] + targets)
        chunks_by_judgment = fetch_chunks(sorted({e["target_judgment_id"] for e in group}))
        for e in group:
            tgt = e["target_judgment_id"]
            if per_target[tgt] >= max_per_target:
                skipped["target already used max times"] += 1
                continue
            record, reason = mcg.build_record(e, texts.get(src, ""), texts, source_edges, meta)
            if record is None:
                skipped[reason.split(" (")[0]] += 1
                continue
            if set(record["relevant_judgment_ids"]) & holdout:
                skipped["held-out judgment in passage"] += 1
                continue
            short = rng.random() < short_fraction and len(record["query_short"].split()) >= 15
            query = record["query_short"] if short else record["query"]
            pos = best_chunk(record["query"], chunks_by_judgment.get(tgt, []))
            if pos is None:
                skipped["no usable chunk in cited judgment"] += 1
                continue
            per_target[tgt] += 1
            pairs.append({
                "query": query,
                "query_kind": "short" if short else "passage",
                "positive": pos[1],
                "positive_chunk_id": pos[0],
                "source_judgment_id": src,
                "target_judgment_id": tgt,
                "relevant_judgment_ids": record["relevant_judgment_ids"],
                "negatives": [],
            })
            if progress:
                progress(1)
    return pairs, skipped


def negative_candidates(ranked: Sequence[Tuple[int, int]], relevant: Set[int], k: int, rng: random.Random) -> List[int]:
    """ranked: (chunk_id, judgment_id) best first. Up to 3k chunk ids, in
    random order, from ranks NEG_RANK_LO..NEG_RANK_HI of judgments that aren't
    relevant; the caller keeps the first k usable ones."""
    pool = [c for c, j in ranked[NEG_RANK_LO:NEG_RANK_HI] if j not in relevant]
    rng.shuffle(pool)
    return pool[: 3 * k]


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------


def make_chunk_fetcher(conn):
    def fetch(judgment_ids: List[int]) -> Dict[int, List[Tuple[int, str]]]:
        out: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        if not judgment_ids:
            return out
        with conn.cursor() as cur:
            cur.execute(
                "SELECT judgment_id, chunk_id, content FROM judgment_chunks WHERE judgment_id = ANY(%s) ORDER BY chunk_id",
                (judgment_ids,),
            )
            for jid, cid, content in cur.fetchall():
                out[jid].append((cid, content or ""))
        return out

    return fetch


def make_text_fetcher(conn):
    """Uncached (the full corpus' text would not fit comfortably in memory)."""
    def fetch(ids: List[int]) -> Dict[int, str]:
        with conn.cursor() as cur:
            cur.execute("SELECT id, judgment_text FROM judgments WHERE id = ANY(%s)", (ids,))
            return {jid: text or "" for jid, text in cur.fetchall()}

    return fetch


def chunk_judgments(conn):
    def fetch(chunk_ids: List[int]) -> Dict[int, int]:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, judgment_id FROM judgment_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
            return dict(cur.fetchall())

    return fetch


def mine_hard_negatives(conn, pairs: List[Dict[str, Any]], k: int, seed: int, batch_size: int) -> Counter:
    """BGE baseline (judgment_embeddings, HNSW) top-50 per query; k negatives
    from ranks 10-50. Runs the BGE model: GPU recommended."""
    from tqdm import tqdm
    from retrieval import embedding_models

    spec = embedding_models.get_spec(embedding_models.BASELINE)
    model = embedding_models.load_model(spec)
    rng = random.Random(seed)
    stats: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {NEG_RANK_HI * 2}")
    for start in tqdm(range(0, len(pairs), 256), desc="hard negatives", unit="x256"):
        batch = pairs[start:start + 256]
        vecs = embedding_models.encode(model, spec, [p["query"] for p in batch], "query", batch_size=batch_size)
        candidates: List[List[int]] = []
        with conn.cursor() as cur:
            for p, v in zip(batch, vecs):
                # ids only: chunk text is fetched for the few chunks kept
                cur.execute(
                    "SELECT je.chunk_id, jc.judgment_id FROM judgment_embeddings je "
                    "JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id "
                    "ORDER BY je.embedding <=> %s::vector LIMIT %s",
                    (embedding_models.vector_literal(v), NEG_RANK_HI),
                )
                relevant = set(p["relevant_judgment_ids"]) | {p["source_judgment_id"]}
                candidates.append(negative_candidates(cur.fetchall(), relevant, k, rng))
            wanted = sorted({c for cs in candidates for c in cs})
            cur.execute("SELECT chunk_id, content FROM judgment_chunks WHERE chunk_id = ANY(%s)", (wanted,))
            text = {c: t or "" for c, t in cur.fetchall()}
        for p, cs in zip(batch, candidates):
            p["negatives"] = [text[c] for c in cs if usable_chunk(text.get(c, ""))][:k]
            stats[f"negatives={len(p['negatives'])}"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="output folder citation_pairs_<version>")
    parser.add_argument("--holdout", nargs="*", default=[], help="gold set JSON files whose judgments are held out")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-per-target", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=0, help="stop after N pairs (0 = all)")
    parser.add_argument("--short-fraction", type=float, default=0.3)
    parser.add_argument("--dev-percent", type=float, default=2.0)
    parser.add_argument("--hard-negatives", type=int, default=0, help="BGE-mined negatives per pair (0 = none)")
    parser.add_argument("--batch-size", type=int, default=64, help="BGE batch size for negative mining")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_root, f"citation_pairs_{args.version}")
    if os.path.exists(out_dir) and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; use a new --version (or --overwrite)")
    missing = [p for p in args.holdout if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"holdout file(s) not found: {missing}")
    if not args.holdout:
        print("WARNING: no --holdout gold sets; benchmark judgments may end up in training data")

    import psycopg2
    from tqdm import tqdm

    started = time.time()
    conn = psycopg2.connect(mcg._default_db_connection_string())
    try:
        holdout, per_file = holdout_ids(args.holdout, chunk_judgments(conn))
        print(f"Held out {len(holdout)} judgments ({per_file})")
        edges, meta = mcg.load_from_db(conn)
        print(f"{len(edges)} resolved edges, {len(mcg.eligible_edges(edges, meta))} eligible")
        bar = tqdm(unit="pair")
        pairs, skipped = build_pairs(
            edges, meta, make_text_fetcher(conn), make_chunk_fetcher(conn), holdout,
            seed=args.seed, max_per_target=args.max_per_target, max_pairs=args.max_pairs,
            short_fraction=args.short_fraction, progress=bar.update,
        )
        bar.close()
        neg_stats: Counter = Counter()
        if args.hard_negatives:
            neg_stats = mine_hard_negatives(conn, pairs, args.hard_negatives, args.seed, args.batch_size)
    finally:
        conn.close()

    train = [p for p in pairs if not is_dev(p["source_judgment_id"], args.dev_percent)]
    dev = [p for p in pairs if is_dev(p["source_judgment_id"], args.dev_percent)]
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_out = {
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "n_train": len(train),
        "n_dev": len(dev),
        "n_short_queries": sum(1 for p in pairs if p["query_kind"] == "short"),
        "n_distinct_targets": len({p["target_judgment_id"] for p in pairs}),
        "holdout_judgments": len(holdout),
        "holdout_per_file": per_file,
        "holdout_judgment_ids": sorted(holdout),
        "skipped": dict(skipped),
        "negatives": dict(neg_stats),
        "elapsed_s": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"train {len(train)}, dev {len(dev)} -> {out_dir}")
    print("Skipped:", dict(skipped))
    if neg_stats:
        print("Negatives:", dict(neg_stats))


if __name__ == "__main__":
    main()
