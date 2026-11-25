# Reingest report: old DB vs `legal_rag_v3`

Date: 2026-11-25. Uncommitted; the user commits.

## What changed

| | old DB (`legal_rag`) | new DB (`legal_rag_v3`) |
|---|---|---|
| Source | judis.nic.in dump + sci.gov.in PDFs | AWS open-data `indian-supreme-court-judgments` (CC-BY-4.0, S.C.R. reported judgments) |
| Judgments | 46,453 | 38,404 (unique by `cnr`) |
| Metadata | parsed from PDF headers; many rows lack date/year | from the dataset's parquet: parties, decision date, bench, citation, disposal, `cnr` |
| Null dates | many | 0 |
| Text cleaning | none | running headers, margin letters, control characters removed; paragraphs rebuilt (`ingestion/scr_chunker.py`) |
| Chunks | keyword chunker | 763,201 chunks, 200 to 3,057 chars, median 1,946, none under 50 chars |
| Embeddings | BGE, chunk text only | BGE, with a one-line case label prepended at embed time (parties, year, citations, bench, outcome). Stored `content` stays clean. |

Attribution: the dataset is CC-BY-4.0; credit is required.

## v3 health (`db_health_check.py`)

- Judgments 38,404; chunks and embeddings 763,201 each; orphans 0; judgments with no chunks 0.
- Duplicates: 12 identical-text groups (same 12 as in v2, so they come from the source data); 31 groups share petitioner+respondent+date (may be connected matters).
- Text: 52 judgments contain U+FFFD, 6 chunks are over 20% non-ASCII (small; probably OCR on old scans).
- Header leaks: whole-DB repeated-line query shows no running headers. Repeats are real boilerplate ("The Judgment of the Court was delivered by", "Appeal allowed.", "Case Law Reference").
- Citation graph: 632,094 edges; 68.1% have an unresolved target (expected, because the corpus holds only S.C.R. cases); 10.9% of judgments have no outgoing edges; 43.2% are never cited. Fuzzy matches (about 21% of resolved edges in v2) have not been hand-labelled.
- Ingest logs: 59 year logs report `db_error > 0`. Likely cross-year duplicate `cnr` rejected by the unique index; the total is the expected 38.4k. Confirm with `grep -h ': db:' ingest_v3_*.log | grep -vc judgments_cnr_uidx` (should be 0).

## Known weakness: section labels

Sections are guessed by keyword rules with a sticky state: facts 15.8%, issues 3.1%, arguments 42.4%, ratio 32.8%, judgment 5.9%. 152 judgments have a single label throughout. Arguments is over-assigned. The label is not used in embeddings or BM25, so retrieval is unaffected, but do not build section filters or boosts on it until the labeller is fixed and spot-checked.

## Retrieval benchmark: old vs v3 (`benchmark/cross_db_eval.py`)

Method: cases present in both DBs, matched by normalised petitioner+respondent+date (unique matches only). 221 questions, evenly by decade, each written by local Qwen from a real v3 chunk. Same two-stage pipeline on both DBs (hybrid recall, candidate_k=30, then cross-encoder rerank, top 8). Scored at **case level**: did the source judgment come back? Chunk ids differ between the DBs, so chunk-level scores are not comparable.

| metric | old | v3 | diff |
|---|---|---|---|
| stage-1 recall@30 | 0.484 | 0.679 | +0.195 |
| recall@1 | 0.362 | 0.520 | +0.158 |
| recall@3 | 0.425 | 0.629 | +0.204 |
| recall@5 | 0.443 | 0.652 | +0.208 |
| recall@8 | 0.466 | 0.670 | +0.204 |
| MRR@8 | 0.397 | 0.578 | +0.181 |

Paired bootstrap, 2,000 resamples: recall@8 diff +0.204, 95% CI [+0.136, +0.271]; MRR@8 diff +0.181, 95% CI [+0.117, +0.244]. Both exclude zero. In 55 queries only v3 found the case in the top 8; in 10 only the old DB did.

recall@8 by decade (old to v3):

| decade | n | old | v3 |
|---|---|---|---|
| 1950s | 30 | 0.60 | 0.70 |
| 1960s | 30 | 0.67 | 0.77 |
| 1970s | 30 | 0.53 | 0.67 |
| 1980s | 30 | 0.70 | 0.83 |
| 1990s | 30 | 0.43 | 0.63 |
| 2000s | 30 | 0.40 | 0.57 |
| 2010s | 30 | 0.07 | 0.50 |
| 2020s | 11 | 0.09 | 0.73 |

## Caveats

1. **Old DB collapses on the 2010s and 2020s** (recall@8 0.07 and 0.09) for cases that do exist in it. Cause not yet found (candidates: bad matched row, junk chunks, header parser failing on newer layouts). Those two decades inflate the average gain; on 1950s to 2000s alone the gain is about +0.10 to +0.20 per decade.
2. **Matched cases only.** 15,643 of 38,404 v3 cases matched; 632 were ambiguous; 22,129 had no match. The result says nothing about unmatched cases or the old DB's non-S.C.R. judgments.
3. **Bias toward v3.** Questions come from v3 chunk text, and v3 embeddings carry the party-name prefix, which helps party-name queries. This run cannot separate "better chunks and metadata" from "prefix effect". A v2 vs v3 run (same cases, `--match cnr`) would isolate chunker + prefix; it has not been run.
4. **Question quality.** Some generated questions are vague (for example one about district-judiciary independence). Both DBs see the same questions, so the comparison is fair, but absolute recall is depressed. One run, one seed, one generator model.

## Open items

- Explain the old-DB 2010s and 2020s result.
- Fix and spot-check the section labeller (about 50 hand-checked chunks).
- Hand-label `outputs/edge_review_sample.csv` for citation-graph precision.
- Review the 12 duplicate-text groups.
- Build the other embedding tables (Euler etc.) on v3 using `context_prefix()`.
- Regenerate gold sets against v3 and rerun the chunk-level benchmarks.
- Switch the API and frontend to v3; add year, judge and outcome filters.
