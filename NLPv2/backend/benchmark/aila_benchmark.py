#!/usr/bin/env python3
"""FIRE 2019 AILA benchmark: precedent and statute retrieval (issue #15).

AILA gives 50 fact situations (AILA_Q1..Q50, Q1-10 released as training,
Q11-50 the official test), a pool of 2,914 Supreme Court judgments and 197
statutes, and TREC relevance judgments. Labels come from the track
organisers, not from us, and published runs exist to compare against.

Two ways to use it:

  pool    Rank AILA's own pool with our retrieval stack — BM25, BGE-base
          dense (500-word chunks, best chunk per document, as ingestion's
          size-based chunker), their RRF fusion (k=60, as production) and
          optionally the MiniLM cross-encoder on the top 30. Scored with the
          track's measures (MAP primary, P@10, BPREF, reciprocal rank) over
          the full ranking, and written as a TREC run file so trec_eval can
          check the numbers. Comparable with the published runs; needs no
          database. Best 2019 MAP on the test queries: 0.1492 (precedents,
          HLJIT2019-AILA task1 2), 0.1566 (statutes, thuir legal 2).

  export  Map AILA's relevant judgments onto OUR corpus (case title + year
          through the citation resolver, exact-name matches only) and write
          a judgment_eval.py gold set (kind "aila"), which scores production
          retrieval over the whole corpus. Statutes are not exported: AILA's
          are IPC / CrPC / Constitution sections and the corpus holds the
          new Sanhitas. Not comparable with published runs (different pool);
          queries none of whose relevant cases are in the corpus are dropped.

The dataset (CC BY 4.0, Zenodo 10.5281/zenodo.4063986) is downloaded by
`download` into <repo>/data/aila/ and pinned by sha256.

Usage:
    cd NLPv2/backend
    python benchmark/aila_benchmark.py download
    python benchmark/aila_benchmark.py pool --task cases --retriever bm25
    python benchmark/aila_benchmark.py pool --task statutes --retriever hybrid --rerank
    python benchmark/aila_benchmark.py export --version v1          # needs the DB
    python benchmark/judgment_eval.py --gold benchmark/data/aila_gold_v1.json
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_benchmark_dir)
_repo_dir = os.path.dirname(os.path.dirname(_backend_dir))
sys.path.insert(0, _backend_dir)

AILA_URL = "https://zenodo.org/records/4063986/files/AILA_2019_dataset.zip?download=1"
AILA_SHA256 = "788bef2f15e4d6212fc3c5b4c302343c52fdbf261e1599e4f2e8d3df2a5fde55"
DEFAULT_DATA_DIR = os.path.join(_repo_dir, "data", "aila")
DEFAULT_WORK_DIR = os.path.join(_benchmark_dir, "work", "aila")
DEFAULT_OUTPUT_DIR = os.path.join(_benchmark_dir, "data")

TASKS = {"cases": ("Object_casedocs", "relevance_judgments_priorcases.txt"),
         "statutes": ("Object_statutes", "relevance_judgments_statutes.txt")}
TRAIN_QUERIES = {f"AILA_Q{i}" for i in range(1, 11)}
SPLITS = ("test", "train", "all")
RETRIEVERS = ("bm25", "dense", "hybrid")
PUBLISHED_BEST_MAP = {"cases": 0.1492, "statutes": 0.1566}

CHUNK_WORDS = 500  # ingestion's size-based chunker
RRF_K = 60
RERANK_TOP = 30
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def download(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "AILA_2019_dataset.zip")
    if not os.path.exists(zpath):
        print(f"Downloading {AILA_URL}")
        tmp = zpath + ".part"
        urllib.request.urlretrieve(AILA_URL, tmp)
        os.replace(tmp, zpath)
    with open(zpath, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != AILA_SHA256:
        raise SystemExit(f"{zpath}: sha256 {digest} != pinned {AILA_SHA256}; refusing to use it")
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            # Zip-slip guard: every member must stay inside data_dir.
            target = os.path.realpath(os.path.join(data_dir, name))
            if not target.startswith(os.path.realpath(data_dir) + os.sep):
                raise SystemExit(f"unsafe path in archive: {name}")
        z.extractall(data_dir)
    print(f"AILA dataset ready in {data_dir}")
    return data_dir


def _require(data_dir: str, *parts: str) -> str:
    path = os.path.join(data_dir, *parts)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run `aila_benchmark.py download` first")
    return path


def load_queries(data_dir: str) -> Dict[str, str]:
    queries = {}
    with open(_require(data_dir, "Query_doc.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            if "||" in line:
                qid, text = line.split("||", 1)
                queries[qid.strip()] = text.strip()
    return queries


def load_qrels(data_dir: str, task: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(_require(data_dir, TASKS[task][1])) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def load_docs(data_dir: str, task: str) -> Dict[str, str]:
    folder = _require(data_dir, TASKS[task][0])
    docs = {}
    for name in sorted(os.listdir(folder)):
        if name.endswith(".txt"):
            with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as f:
                text = f.read()
            if task == "statutes":  # "Title: ...\nDesc: ..."
                text = re.sub(r"^(Title|Desc):\s*", "", text, flags=re.M)
            docs[name[:-4]] = text
    return docs


def query_ids(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], split: str) -> List[str]:
    ids = [q for q in queries if q in qrels]
    if split == "test":
        ids = [q for q in ids if q not in TRAIN_QUERIES]
    elif split == "train":
        ids = [q for q in ids if q in TRAIN_QUERIES]
    return sorted(ids, key=lambda q: int(q.rsplit("Q", 1)[1]))


# ---------------------------------------------------------------------------
# Metrics (trec_eval definitions)
# ---------------------------------------------------------------------------


def trec_metrics(ranking: Sequence[str], judged: Dict[str, int]) -> Dict[str, float]:
    """AP, P@10, bpref and reciprocal rank of one ranking, as trec_eval
    computes them: R is every relevant document in the qrels (retrieved or
    not), bpref counts only judged non-relevant documents ranked above."""
    relevant = {d for d, r in judged.items() if r > 0}
    R = len(relevant)
    N = sum(1 for r in judged.values() if r <= 0)
    if R == 0:
        return {"map": 0.0, "P@10": 0.0, "bpref": 0.0, "recip_rank": 0.0}
    hits, ap, rr, nonrel_above, bpref = 0, 0.0, 0.0, 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            hits += 1
            ap += hits / i
            if not rr:
                rr = 1.0 / i
            bpref += 1.0 - (min(nonrel_above, R) / min(R, N) if N else 0.0)
        elif d in judged:
            nonrel_above += 1
    return {
        "map": ap / R,
        "P@10": sum(1 for d in ranking[:10] if d in relevant) / 10,
        "bpref": bpref / R,
        "recip_rank": rr,
    }


def mean_metrics(per_query: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ("map", "P@10", "bpref", "recip_rank")
    n = len(per_query)
    return {k: sum(m[k] for m in per_query.values()) / n if n else float("nan") for k in keys}


# ---------------------------------------------------------------------------
# Retrieval over the pool
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in into is it its "
    "of on or she that the their them they this to was were which who will with would not no "
    "so than then there these those under upon our we you your shall may said also any all".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    """Okapi BM25 over whole documents (k1=1.2, b=0.75)."""

    def __init__(self, docs: Dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.lengths = []
        for i, doc_id in enumerate(self.ids):
            tf = Counter(tokenize(docs[doc_id]))
            self.lengths.append(sum(tf.values()))
            for term, c in tf.items():
                self.postings[term][i] = c
        n = len(self.ids)
        self.avgdl = sum(self.lengths) / n if n else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> Dict[str, float]:
        out = [0.0] * len(self.ids)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, tf in postings.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / norm
        return dict(zip(self.ids, out))


def chunk_words(text: str, size: int = CHUNK_WORDS) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


class DensePool:
    """BGE embeddings of every 500-word chunk; a document scores as its best
    chunk. Embeddings are cached under the work dir, keyed by task + model."""

    def __init__(self, docs: Dict[str, str], task: str, work_dir: str, model_name: str = EMBED_MODEL):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Tuple[str, str]] = [(d, c) for d, text in docs.items() for c in chunk_words(text)]
        os.makedirs(work_dir, exist_ok=True)
        tag = hashlib.sha256("\n".join(f"{d}:{len(c)}" for d, c in self.chunks).encode()).hexdigest()[:12]
        cache = os.path.join(work_dir, f"emb_{task}_{model_name.replace('/', '_')}_{tag}.npy")
        if os.path.exists(cache):
            self.emb = np.load(cache)
        else:
            print(f"Embedding {len(self.chunks)} chunks (cached to {cache}) ...")
            self.emb = self.model.encode(
                [c for _, c in self.chunks], batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            )
            np.save(cache, self.emb)

    def scores(self, query: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Best-chunk cosine per document, and that best chunk's text."""
        q = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        sims = self.emb @ q
        best: Dict[str, float] = {}
        best_chunk: Dict[str, str] = {}
        for (doc_id, text), s in zip(self.chunks, sims):
            if doc_id not in best or s > best[doc_id]:
                best[doc_id], best_chunk[doc_id] = float(s), text
        return best, best_chunk


def rank(scores: Dict[str, float]) -> List[str]:
    return sorted(scores, key=lambda d: (-scores[d], d))


def rrf(rankings: Iterable[List[str]], k: int = RRF_K) -> Dict[str, float]:
    fused: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, d in enumerate(ranking):
            fused[d] += 1.0 / (k + r + 1)
    return dict(fused)


def run_pool(
    data_dir: str, task: str, retriever: str, rerank: bool, split: str, work_dir: str
) -> Dict[str, Any]:
    if rerank and retriever == "bm25":
        raise SystemExit("--rerank needs a dense or hybrid retriever (it reranks each document's best chunk)")
    queries, qrels, docs = load_queries(data_dir), load_qrels(data_dir, task), load_docs(data_dir, task)
    qids = query_ids(queries, qrels, split)

    bm25 = BM25(docs) if retriever in ("bm25", "hybrid") else None
    dense = DensePool(docs, task, work_dir) if retriever in ("dense", "hybrid") else None
    reranker = None
    if rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL)

    runs: Dict[str, List[Tuple[str, float]]] = {}
    start = time.time()
    for qid in qids:
        query = queries[qid]
        rankings, best_chunk = [], {}
        if bm25:
            bm = bm25.scores(query)
            rankings.append(rank(bm))
        if dense:
            ds, best_chunk = dense.scores(query)
            rankings.append(rank(ds))
        scores = rrf(rankings) if len(rankings) > 1 else (bm if bm25 else ds)
        ranking = rank(scores)
        if reranker:
            head = ranking[:RERANK_TOP]
            ce = reranker.predict([(query, best_chunk[d]) for d in head])
            order = sorted(range(len(head)), key=lambda i: -float(ce[i]))
            # Reranked head keeps scores above every tail document.
            top = max(scores.values()) + 1.0
            scores = {**scores, **{head[i]: top + float(ce[i]) - float(min(ce)) for i in order}}
            ranking = [head[i] for i in order] + ranking[RERANK_TOP:]
        runs[qid] = [(d, scores[d]) for d in ranking]

    per_query = {qid: trec_metrics([d for d, _ in runs[qid]], qrels[qid]) for qid in qids}
    return {
        "config": {
            "task": task, "retriever": retriever, "rerank": rerank, "split": split,
            "n_queries": len(qids), "n_docs": len(docs), "chunk_words": CHUNK_WORDS,
            "embed_model": EMBED_MODEL if dense else None, "rerank_model": RERANK_MODEL if rerank else None,
            "elapsed_s": round(time.time() - start, 2),
        },
        "aggregate": mean_metrics(per_query),
        "published_best_map_test": PUBLISHED_BEST_MAP[task],
        "per_query": per_query,
        "run": runs,
    }


def write_trec_run(runs: Dict[str, List[Tuple[str, float]]], path: str, run_id: str) -> None:
    """trec_eval orders by the score column, not the rank, and breaks ties
    by docno; write a strictly decreasing score so it scores our order."""
    with open(path, "w") as f:
        for qid, ranked in runs.items():
            n = len(ranked)
            for r, (d, _) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {d} {r} {n - r + 1} {run_id}\n")


# ---------------------------------------------------------------------------
# Export onto our corpus
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Z][a-z]+)\s+(1[89]\d{2}|20\d{2})\b")


def case_header(text: str) -> Tuple[str, Optional[int]]:
    """AILA case files start "<Title>\\n<Court>\\n\\n<day> <Month> <year>":
    the title and the judgment year."""
    lines = [l.strip() for l in text.splitlines()[:8] if l.strip()]
    title = lines[0] if lines else ""
    m = _DATE_RE.search("\n".join(lines[1:4]))
    return title, (int(m.group(3)) if m else None)


def map_cases(docs: Dict[str, str], resolver) -> Tuple[Dict[str, int], Counter]:
    """AILA case id -> corpus judgment id, by case title + year through the
    citation resolver. Exact-name matches only; fuzzy and ambiguous are
    counted, not used, since a wrong mapping would mis-score silently."""
    mapping: Dict[str, int] = {}
    methods: Counter = Counter()
    for doc_id, text in docs.items():
        title, year = case_header(text)
        res = resolver.resolve(f"{title} ({year})" if year else title)
        methods[res.method] += 1
        if res.method == "exact_name" and res.judgment_id is not None:
            mapping[doc_id] = res.judgment_id
    return mapping, methods


def build_gold_records(
    queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], mapping: Dict[str, int],
    case_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], Counter]:
    records, skipped = [], Counter()
    for qid in query_ids(queries, qrels, "all"):
        rel = sorted(d for d, r in qrels[qid].items() if r > 0)
        ids = sorted({mapping[d] for d in rel if d in mapping})
        if not ids:
            skipped["no relevant case found in the corpus"] += 1
            continue
        records.append({
            "id": qid,
            "query": queries[qid],
            "split": "train" if qid in TRAIN_QUERIES else "test",
            "relevant_judgment_ids": ids,
            "relevant_case_names": {str(j): case_names.get(j, "") for j in ids},
            "aila_relevant": rel,
            "aila_relevant_unmapped": [d for d in rel if d not in mapping],
        })
    return records, skipped


def export(data_dir: str, version: str, output: Optional[str], overwrite: bool) -> None:
    out = output or os.path.join(DEFAULT_OUTPUT_DIR, f"aila_gold_{version}.json")
    if os.path.exists(out) and not overwrite:
        raise SystemExit(f"{out} exists; gold sets are frozen once written. Use a new --version.")
    import psycopg2

    from commentary_gold import _default_db_connection_string, load_resolver_from_db
    from graph.citation_resolver import case_name_alias

    queries, qrels, docs = load_queries(data_dir), load_qrels(data_dir, "cases"), load_docs(data_dir, "cases")
    conn = psycopg2.connect(_default_db_connection_string())
    try:
        resolver = load_resolver_from_db(conn)
        mapping, methods = map_cases(docs, resolver)
        cur = conn.cursor()
        cur.execute("SELECT id, petitioner, respondent FROM judgments WHERE id = ANY(%s)",
                    (sorted(set(mapping.values())),))
        case_names = {r[0]: case_name_alias(r[1], r[2]) or "" for r in cur.fetchall()}
        cur.execute("SELECT count(*) FROM judgments")
        n_judgments = cur.fetchone()[0]
        cur.close()
    finally:
        conn.close()

    records, skipped = build_gold_records(queries, qrels, mapping, case_names)
    relevant = {d for q in qrels.values() for d, r in q.items() if r > 0}
    gold = {
        "kind": "aila",
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": len(records),
        "source": {"dataset": "FIRE 2019 AILA", "doi": "10.5281/zenodo.4063986", "sha256": AILA_SHA256},
        "mapping": {
            "aila_cases": len(docs),
            "mapped": len(mapping),
            "relevant_cases": len(relevant),
            "relevant_mapped": len(relevant & set(mapping)),
            "resolver_methods": dict(methods),
        },
        "corpus_judgment_count": n_judgments,
        "skipped": dict(skipped),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(gold, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} queries to {out}")
    print(f"Mapping: {gold['mapping']}")
    if skipped:
        print(f"Skipped: {dict(skipped)}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FIRE 2019 AILA precedent / statute retrieval benchmark.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Fetch the dataset from Zenodo and verify its sha256")

    p = sub.add_parser("pool", help="Rank AILA's own pool; comparable with published runs")
    p.add_argument("--task", choices=tuple(TASKS), required=True)
    p.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    p.add_argument("--rerank", action="store_true", help=f"Cross-encoder over the top {RERANK_TOP}")
    p.add_argument("--split", choices=SPLITS, default="test", help="test = Q11-50, the official evaluation")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--output", default=None)

    e = sub.add_parser("export", help="Map onto our corpus and write a judgment_eval gold set (needs DB)")
    e.add_argument("--version", required=True)
    e.add_argument("--output", default=None)
    e.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cmd == "download":
        download(args.data_dir)
        return
    if args.cmd == "export":
        export(args.data_dir, args.version, args.output, args.overwrite)
        return

    report = run_pool(args.data_dir, args.task, args.retriever, args.rerank, args.split, args.work_dir)
    cfg = report["config"]
    run_id = f"{cfg['retriever']}{'_rerank' if cfg['rerank'] else ''}"
    out = args.output or os.path.join(
        _benchmark_dir, "reports", f"aila_{cfg['task']}_{run_id}_{cfg['split']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_trec_run(report["run"], out[:-5] + ".trec", run_id)
    with open(out, "w") as f:
        json.dump({k: v for k, v in report.items() if k != "run"}, f, indent=2)
    print(f"Config: {cfg}")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v:.4f}")
    if cfg["split"] == "test":
        print(f"  (best published MAP, 2019: {report['published_best_map_test']})")
    print(f"Report: {out}\nTREC run: {out[:-5]}.trec")


if __name__ == "__main__":
    main()
