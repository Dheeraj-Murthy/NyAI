#!/usr/bin/env python3
"""Load one embedding model from retrieval/embedding_models.py and check it
before spending GPU hours on it. No database needed.

    cd NLPv2/backend
    python benchmark/check_embedding_model.py --embedding-model euler

Prints what the model will actually do (output width, stored width, max
tokens, query/document prompts, pooling, padding side, dtype, device), then:
  - a sanity check: each legal query must score its matching passage above
    the distractors, at the full width and at the stored width;
  - throughput on chunk-length texts and peak GPU memory, to estimate the
    corpus backfill (rows / throughput) before running it.
"""
import argparse
import os
import sys
import time

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from retrieval import embedding_models  # noqa: E402

PAIRS = [
    ("punishment for committing murder",
     "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."),
    ("when can a court grant anticipatory bail",
     "Where any person has reason to believe that he may be arrested on accusation of having committed a "
     "non-bailable offence, he may apply to the High Court or the Court of Session for a direction that in "
     "the event of such arrest he shall be released on bail."),
    ("is the right to privacy a fundamental right",
     "The right to privacy is protected as an intrinsic part of the right to life and personal liberty under "
     "Article 21 and as a part of the freedoms guaranteed by Part III of the Constitution."),
    ("death of a married woman within seven years caused by harassment for dowry",
     "Where the death of a woman is caused by any burns or bodily injury within seven years of her marriage and "
     "it is shown that soon before her death she was subjected to cruelty or harassment by her husband in "
     "connection with any demand for dowry, such death shall be called dowry death."),
]


def describe(model, spec) -> None:
    print(f"key            {spec.key}")
    print(f"model_id       {spec.model_id}  (revision {spec.revision or 'main'})")
    print(f"fingerprint    {embedding_models.model_fingerprint(spec)}")
    print(f"output dims    {model.get_sentence_embedding_dimension()}  -> stored {spec.dim} as {spec.vector_type}")
    print(f"max tokens     {model.max_seq_length}")
    print(f"query prompt   {embedding_models.prompt_for(model, spec, 'query')!r}")
    print(f"doc prompt     {embedding_models.prompt_for(model, spec, 'document')!r}")
    print(f"tables         {spec.judgment_table} / {spec.statute_table}")
    print(f"threshold      {'BGE default (caller)' if spec.dense_threshold is None else spec.dense_threshold}")
    try:
        modules = list(model)
    except TypeError:
        modules = []
    for i, module in enumerate(modules):
        cfg = getattr(module, "get_config_dict", lambda: {})()
        print(f"module {i}       {type(module).__name__} {cfg if 'Pooling' in type(module).__name__ else ''}")
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        print(f"padding side   {getattr(tok, 'padding_side', '?')}")
    try:
        p = next(model.parameters())
        print(f"dtype/device   {p.dtype} / {p.device}")
    except (StopIteration, AttributeError):
        pass


def sanity(model, spec, batch_size: int) -> bool:
    queries = [q for q, _ in PAIRS]
    docs = [d for _, d in PAIRS]
    q_full = embedding_models.encode_full(model, spec, queries, "query", batch_size)
    d_full = embedding_models.encode_full(model, spec, docs, "document", batch_size)
    ok = True
    widths = sorted({q_full.shape[1], spec.dim}, reverse=True)
    for width in widths:
        q = embedding_models.truncate(q_full, width)
        d = embedding_models.truncate(d_full, width)
        sims = q @ d.T
        right = sum(int(sims[i].argmax() == i) for i in range(len(PAIRS)))
        margin = min(sims[i, i] - max(sims[i, j] for j in range(len(PAIRS)) if j != i) for i in range(len(PAIRS)))
        print(f"  {width:>5} dims: {right}/{len(PAIRS)} queries rank their passage first; "
              f"worst margin {margin:+.3f}; matching-pair cosine {sims.diagonal().min():.3f}-{sims.diagonal().max():.3f}")
        ok &= right == len(PAIRS)
    return ok


def throughput(model, spec, n: int, batch_size: int) -> None:
    import torch

    text = " ".join(PAIRS[2][1].split() * 12)[:3200]  # ~500 words, ingestion's chunk size
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    embedding_models.encode(model, spec, [text] * min(n, batch_size), "document", batch_size)  # warm-up
    start = time.time()
    embedding_models.encode(model, spec, [text] * n, "document", batch_size)
    rate = n / (time.time() - start)
    print(f"  {rate:.1f} chunks/s at batch {batch_size} -> {3600 * rate:,.0f} chunks/hour")
    if torch.cuda.is_available():
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default=None, help=f"one of {embedding_models.available()}")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n", type=int, default=128, help="texts for the throughput test (0 = skip)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    spec = embedding_models.get_spec(args.embedding_model)
    started = time.time()
    model = embedding_models.load_model(spec, device=args.device)
    print(f"Loaded in {time.time() - started:.0f} s\n")
    describe(model, spec)
    print("\nSanity:")
    ok = sanity(model, spec, args.batch_size)
    if args.n:
        print("\nThroughput:")
        throughput(model, spec, args.n, args.batch_size)
    print("\nOK" if ok else "\nSANITY FAILED: check the prompts / pooling above before backfilling")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
