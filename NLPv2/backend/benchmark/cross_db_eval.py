#!/usr/bin/env python3
"""Paired old-DB vs new-DB retrieval benchmark (judgment level).

Chunk ids and chunk boundaries differ between the two corpora, so chunk-level
recall is not comparable. This script instead:
  build: matches cases present in BOTH DBs (normalised petitioner+respondent,
         date-checked), samples them evenly by decade, picks one real chunk of
         each from the NEW db and has an LLM write a question for it.
  eval : runs the same two-stage retrieval (hybrid recall -> cross-encoder
         rerank) on each DB with the same questions and scores whether the
         right *case* comes back. Reports recall@k, MRR, per-decade, and a
         paired bootstrap CI for the difference.

Bias to keep in mind: questions are written from new-DB chunk text, so they
slightly favour the new DB's wording. Judgment-level scoring limits this.

Usage (cd NLPv2/backend):
  python benchmark/cross_db_eval.py build --old-db legal_rag --new-db legal_rag_v3 --n 240 --out benchmark/data/paired_gold.json
  python benchmark/cross_db_eval.py eval  --old-db legal_rag --new-db legal_rag_v3 --gold benchmark/data/paired_gold.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import psycopg2

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

_DROP = {"the", "of", "and", "ors", "anr", "others", "another", "etc", "state", "union", "india", "vs", "v", "m", "s", "shri", "smt", "sri", "mr"}


def dsn(dbname):
    return (f"host={os.environ.get('DB_HOST', 'localhost')} port={os.environ.get('DB_PORT', '5433')} "
            f"dbname={dbname} user={os.environ.get('DB_USER', 'postgres')} "
            f"password={os.environ.get('DB_PASSWORD', 'postgres')}")


def norm_name(s):
    toks = [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in _DROP]
    return " ".join(sorted(set(toks)))


def match_cases(old_cur, new_cur):
    old_cur.execute("SELECT id, petitioner, respondent, date_of_judgment FROM judgments")
    old = defaultdict(list)
    for i, p, r, d in old_cur.fetchall():
        if p and r and p != "Unknown" and r != "Unknown":
            old[(norm_name(p), norm_name(r))].append((i, d))
    new_cur.execute("SELECT id, petitioner, respondent, date_of_judgment FROM judgments")
    pairs, amb, none = [], 0, 0
    for i, p, r, d in new_cur.fetchall():
        cands = old.get((norm_name(p), norm_name(r)), [])
        if d is not None:
            exact = [c for c in cands if c[1] == d]
            cands = exact if exact else [c for c in cands if c[1] is None]
        if len(cands) == 1:
            pairs.append({"new_id": i, "old_id": cands[0][0], "date": d})
        elif not cands:
            none += 1
        else:
            amb += 1
    print(f"matched uniquely: {len(pairs)}, ambiguous: {amb}, no match: {none}")
    return pairs


def cmd_build(a):
    from benchmark.generate_gold_set import _resolve_backend, generate_question_for_chunk, _is_garbled, _question_overlap, MIN_QUESTION_OVERLAP
    oc, nc = psycopg2.connect(dsn(a.old_db)), psycopg2.connect(dsn(a.new_db))
    pairs = match_cases(oc.cursor(), nc.cursor())
    by_dec = defaultdict(list)
    for p in pairs:
        by_dec[p["date"].year // 10 * 10].append(p)
    rng = random.Random(a.seed)
    per = max(a.n // max(len(by_dec), 1), 1)
    backend = _resolve_backend(a.model, a.external_ok, a.api_key)
    cur = nc.cursor()
    records = []
    for dec in sorted(by_dec):
        pool = by_dec[dec]
        rng.shuffle(pool)
        got = 0
        for p in pool:
            if got >= per:
                break
            cur.execute("SELECT chunk_id, section, content FROM judgment_chunks WHERE judgment_id=%s AND length(content) BETWEEN 600 AND 2600", (p["new_id"],))
            chunks = [c for c in cur.fetchall() if not _is_garbled(c[2])]
            rng.shuffle(chunks)
            for cid, sec, content in chunks[:3]:
                q = generate_question_for_chunk(backend, {"section": sec, "content": content})
                if _question_overlap(q, content) < MIN_QUESTION_OVERLAP:
                    continue
                records.append({"query": q, "new_id": p["new_id"], "old_id": p["old_id"], "decade": dec, "new_chunk_id": cid})
                got += 1
                print(f"[{len(records)}] {dec}s: {q}")
                break
    json.dump({"old_db": a.old_db, "new_db": a.new_db, "records": records}, open(a.out, "w"), indent=2)
    print(f"wrote {len(records)} paired queries to {a.out}")


def first_rank(target, ids):
    for i, x in enumerate(ids):
        if x == target:
            return i + 1
    return None


def score(rank, k):
    return 1.0 if rank is not None and rank <= k else 0.0


def run_db(dbname, records, key, cand_k, thr, top_k):
    from retrieval.retriever import LegalRetriever
    from retrieval.reranker import CrossEncoderReranker
    retr, rer = LegalRetriever(db_connection_string=dsn(dbname)), CrossEncoderReranker()
    out = []
    for i, r in enumerate(records):
        s1 = retr.retrieve_judgment_candidates(query=r["query"], candidate_k=cand_k, similarity_threshold=thr)
        s2 = rer.rerank(r["query"], list(s1), top_n=top_k)
        t = r[key]
        out.append({"s1": first_rank(t, [c["judgment_id"] for c in s1]), "s2": first_rank(t, [c["judgment_id"] for c in s2])})
        if (i + 1) % 20 == 0:
            print(f"  {dbname}: {i + 1}/{len(records)}", flush=True)
    return out


def summarize(res):
    n = len(res)
    d = {"n": n, "recall@30_stage1": sum(score(r["s1"], 30) for r in res) / n}
    for k in (1, 3, 5, 8):
        d[f"recall@{k}"] = sum(score(r["s2"], k) for r in res) / n
    d["MRR@8"] = sum((1.0 / r["s2"]) if r["s2"] else 0.0 for r in res) / n
    return d


def cmd_eval(a):
    gold = json.load(open(a.gold))["records"]
    old = run_db(a.old_db, gold, "old_id", a.candidate_k, a.threshold, a.top_k)
    new = run_db(a.new_db, gold, "new_id", a.candidate_k, a.threshold, a.top_k)
    so, sn = summarize(old), summarize(new)
    print(f"\n{'metric':<18}{'old':>9}{'new':>9}{'diff':>9}")
    for k in so:
        if k == "n":
            continue
        print(f"{k:<18}{so[k]:>9.3f}{sn[k]:>9.3f}{sn[k] - so[k]:>+9.3f}")
    # paired bootstrap on recall@8 and MRR
    rng = random.Random(0)
    def stat(idx, f):
        return sum(f(new[i]) - f(old[i]) for i in idx) / len(idx)
    f_r8 = lambda r: score(r["s2"], 8)
    f_mrr = lambda r: (1.0 / r["s2"]) if r["s2"] else 0.0
    for name, f in (("recall@8", f_r8), ("MRR@8", f_mrr)):
        diffs = sorted(stat([rng.randrange(len(gold)) for _ in gold], f) for _ in range(2000))
        print(f"paired diff {name}: {stat(range(len(gold)), f):+.3f}  95% CI [{diffs[50]:+.3f}, {diffs[1949]:+.3f}]")
    wins = sum(1 for o, n_ in zip(old, new) if score(n_["s2"], 8) > score(o["s2"], 8))
    loss = sum(1 for o, n_ in zip(old, new) if score(n_["s2"], 8) < score(o["s2"], 8))
    print(f"queries where new found the case in top-8 and old didn't: {wins}; old found, new didn't: {loss}")
    by = defaultdict(list)
    for r, o, n_ in zip(gold, old, new):
        by[r["decade"]].append((o, n_))
    print("\nrecall@8 by decade (old -> new):")
    for dec in sorted(by):
        v = by[dec]
        print(f"  {dec}s  n={len(v):<4}{sum(score(o['s2'], 8) for o, _ in v) / len(v):.2f} -> {sum(score(n_['s2'], 8) for _, n_ in v) / len(v):.2f}")
    if a.out:
        json.dump({"old": so, "new": sn, "per_query": [{"query": r["query"], "old": o, "new": n_} for r, o, n_ in zip(gold, old, new)]}, open(a.out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--old-db", default="legal_rag")
        p.add_argument("--new-db", default="legal_rag_v3")
    b = sub.choices["build"]
    b.add_argument("--n", type=int, default=240)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--model", default=None)
    b.add_argument("--external-ok", action="store_true")
    b.add_argument("--api-key", default=None)
    b.add_argument("--out", default="benchmark/data/paired_gold.json")
    e = sub.choices["eval"]
    e.add_argument("--gold", required=True)
    e.add_argument("--candidate-k", type=int, default=30)
    e.add_argument("--threshold", type=float, default=0.2)
    e.add_argument("--top-k", type=int, default=8)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    (cmd_build if a.cmd == "build" else cmd_eval)(a)


if __name__ == "__main__":
    main()
