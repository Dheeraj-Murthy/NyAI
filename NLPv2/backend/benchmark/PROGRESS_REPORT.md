# Legal RAG — Retrieval Benchmark Progress Report

Consolidated from `origin/inlegal` (`NLPv2/backend/benchmark/REPORT.md`, latest, 2026-11-20), `origin/euler` and `origin/issue-19-citation-benchmark`. Local `main` still holds only the stale 2026-10-07 report. Numbers below are copied from those reports, not re-run.

## 1. Timeline

| Date | Step | Outcome |
|---|---|---|
| 2026-10-06 | Synthetic chunk-level gold set (Qwen-written questions, n=80) | Baseline: final chunk recall@8 0.525, judgment recall@8 0.688 |
| 2026-10-07 | Multi-label (near-dup) scoring + discriminability gate | Per-section result did **not** replicate under same seed (judgment section 0.250 -> 0.625); n=80 too small |
| 2026-11-08 | Masked-citation gold set v1 (n=500), commentary gold pipeline, AILA harness | Production hit@8 0.200 (passage), 0.070 (short). Reranker hurts; BM25 AND-leg dead for long queries |
| 2026-11-09/10 | Issue #12: BM25 `and_or`, case lookup in chat, alias/role-label fixes | Case lookup hit@8 0.397 -> **0.994** |
| 2026-11-11 | Citation benchmark pipeline (#19) | Built; **no results yet** |
| 2026-11-20 | Embedding-model comparison (InLegalBERT, fine-tunes, Euler/Qwen3) | Built. Euler (8B, 2048-dim) has server results (section 4); the other models have none |

## 2. Accuracy achieved (final stage, hit@8 unless noted)

| Benchmark | Start | Latest | Notes |
|---|---|---|---|
| Synthetic, chunk recall@8 (n=80) | 0.525 | ~0.575 (multi-label, seed 1) | single-seed, high variance |
| Synthetic, judgment recall@8 | 0.688 / 0.625 (AND) | 0.650 | after #12 |
| Masked citation, passage (n=500) | 0.200 | 0.200 [~0.166, 0.236] | unchanged by #12 |
| Masked citation, short query | 0.070 | **0.088** [0.064, 0.114] | |
| Case lookup, all (n=642) | 0.397 | **0.994** [0.988, 0.998] | via alias/title pinning |
| - citation / full_name / question / party_name | 0.73 / 0.38 / 0.40 / 0.31 | 0.968 / 1.000 / 1.000 / 0.989 | |
| AILA mapped to corpus (n=43) | 0.093 | 0.093 | too few queries |
| AILA pool precedents, BM25 MAP (Q11-50) | - | 0.133 | matches published BM25 0.1335; best 2019 = 0.149 |

BM25 latency after #12: ~95 ms (passage), ~143-260 ms (short); pure OR mode was 3,667 ms and was rejected.

## 3. Key findings

1. **Name/citation lookup is solved** (0.397 -> 0.994) by routing resolvable citations/titles/party names through the alias table and pinning the result through reranking. Largest gain of any benchmark.
2. **Semantic precedent retrieval is the weak spot.** Cited judgment is outside the top 30 for ~65% of masked passages; `candidate_k`=60 and graph boost (PageRank, rescaled) did nothing.
3. **Dense BGE-base is weak** on long, legal queries: AILA MAP 0.060 vs BM25 0.133; hybrid 0.106; hybrid+rerank 0.086. On masked passages hybrid == dense (BM25 contributed nothing for 89% of queries under AND).
4. **MS-MARCO MiniLM reranker hurts on precedent tasks** (masked passage: -37/+18 queries, McNemar p=0.014; AILA lowers MAP) but helps slightly on short name queries and the synthetic set.
5. **Synthetic benchmark is unreliable per section** at n=80 (same seed gave judgment-section recall 0.25 vs 0.625). Use paired comparisons and bootstrap CIs; larger n / multi-seed still outstanding.
6. Heavily cited cases (in-degree 50+) are hardest (0.141 vs ~0.21).

## 4. Euler-Legal-Embedding (8B, 2048-dim) vs BGE-base — server results

Source: results pasted by the user (server run); not re-run here.

| Setting | BGE | Euler |
|---|---|---|
| Case lookup, dense-only hit@8 (n=642) | 0.262 | **0.477** |
| Case lookup, dense-only MRR | 0.245 | 0.446 |
| Case lookup, party-name subset | 0.11 | 0.35 |
| Masked passage, end-to-end hit@8 | 0.20 | 0.23 (0.29 before the reranker) |
| Synthetic judgment recall@8 | 0.66 | 0.76 |
| AILA mapped hit@8 (n=43) | 0.093 | 0.116 |
| AILA pool MAP | 0.133 (BM25), 0.060 (BGE dense) | ~3x ours (exact figure not supplied) |
| Production case lookup | 0.994 | 0.994 (alias lookup uses no embeddings) |

- Dense retrieval gain is clear. End-to-end gain is modest and mostly inside the CIs.
- Why small: BM25 contributes nothing on most long queries, so hybrid is mostly dense-only. The MiniLM reranker lowers Euler's masked-passage score (0.29 -> 0.23).
- **AILA contamination caveat:** AILA is part of MTEB's legal benchmark (AILACasedocs / AILAStatutes), and Euler's `args.json` shows training from an MTEB folder. Its AILA numbers may be inflated. Own benchmarks (case lookup, masked citation, synthetic) are the real evidence.
- Open: gain vs serving cost (8B model, 5.3 GB index, GPU); Euler + reranker test or no-rerank; Qwen3-Embedding-8B control run; 4096/2048/1024 truncation check.

## 4b. In flight, no numbers yet

- **Other embedding models** (`inlegal`, `inlegal_ft`, `bge_ft`, `qwen3_8b`, `qwen3_06b`): per-model tables, `RAG_EMBEDDING_MODEL` switch, fine-tuning on citation pairs with benchmark gold sets held out. Cheap first look: `aila_benchmark.py pool --embedding-model <key>`.
- **Citation benchmark #19**: cases citing >=3 in-corpus judgments, redacted facts/issues as query; retrieval vs popularity baseline vs LLM select/closed-book. Report says "Results: not run yet".
- **Commentary gold set** (Law Commission reports): pipeline built, needs human review; target 100-150 queries.

## 5. Open issues / next steps

- Run the Qwen3-8B control and the remaining embedders; record latency and table size (Euler: 8B model, 5.3 GB index). Test Euler without the MiniLM reranker.
- Replace or bypass the MiniLM reranker for precedent-style queries (needs legal/cross-encoder or skip rerank when long).
- Party-name lookup now 0.989; 6 case-lookup records went stale after name cleaning (n 648 -> 642).
- Increase synthetic gold set to 300-400 or multi-seed mean +/- spread.
- Corpus mojibake rescan (corrected >127 threshold) still outstanding.
- Masked-citation v2: mask initials/more reporters, drop headnote-like passages (v1 stays frozen).
- No end-to-end answer-quality benchmark yet (`answer_eval_stub.py` is a placeholder).

## 6. Caveats

Masked-citation and citation-benchmark gold are lower bounds (other valid precedents are unlabelled); use for paired config comparison. InLegalBERT pre-training likely overlaps the corpus text (not labels).
