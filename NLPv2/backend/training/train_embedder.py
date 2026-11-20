#!/usr/bin/env python3
"""Fine-tune an embedding model on citation pairs (training/citation_pairs.py).

Contrastive training with sentence-transformers: each passage should embed
closer to its cited judgment's chunk than to the other chunks in the batch
(in-batch negatives) and to its BGE-mined hard negatives, if the pairs file
has them. CachedMultipleNegativesRankingLoss gives a large effective batch
(more in-batch negatives) at a fixed GPU memory cost.

    cd NLPv2/backend
    # InLegalBERT (the experiment)
    python training/train_embedder.py --base-model law-ai/InLegalBERT \\
        --pairs ../../data/training/citation_pairs_v1 --output ../../models/inlegal-ft
    # BGE-base (the control: same data, general-domain start)
    python training/train_embedder.py --base-model BAAI/bge-base-en-v1.5 \\
        --pairs ../../data/training/citation_pairs_v1 --output ../../models/bge-ft

Losses and the dev retrieval metrics go to MLflow (experiment
legal-rag-training, same mlflow.db as the other runs, so run from backend/).
The output folder gets training_info.json (base model, data, args, git
commit); `dvc add` it to version the checkpoint.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

_training_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_training_dir)
sys.path.insert(0, _backend_dir)

TRAINING_EXPERIMENT = "legal-rag-training"
DEV_MAX_QUERIES = 1000


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_columns(rows: List[Dict[str, Any]], n_neg: int) -> Dict[str, List[str]]:
    """(anchor, positive, negative_1..n) columns. With n_neg > 0, rows with
    fewer mined negatives are dropped: every row needs the same columns."""
    cols: Dict[str, List[str]] = {"anchor": [], "positive": []}
    for i in range(n_neg):
        cols[f"negative_{i + 1}"] = []
    for r in rows:
        if len(r.get("negatives", [])) < n_neg:
            continue
        cols["anchor"].append(r["query"])
        cols["positive"].append(r["positive"])
        for i in range(n_neg):
            cols[f"negative_{i + 1}"].append(r["negatives"][i])
    return cols


def dev_evaluator(dev: List[Dict[str, Any]]):
    """Retrieval over the dev pairs' positives and negatives: does each
    passage rank its cited chunk first? (Easier than the full corpus; it
    tracks training, the benchmarks measure it.)"""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    dev = dev[:DEV_MAX_QUERIES]
    corpus: Dict[str, str] = {}
    ids: Dict[str, str] = {}

    def doc_id(text: str) -> str:
        if text not in ids:
            ids[text] = f"d{len(ids)}"
            corpus[ids[text]] = text
        return ids[text]

    queries, relevant = {}, {}
    for i, r in enumerate(dev):
        queries[f"q{i}"] = r["query"]
        relevant[f"q{i}"] = {doc_id(r["positive"])}
        for n in r.get("negatives", []):
            doc_id(n)
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant, name="citation-dev",
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10], map_at_k=[100],
        show_progress_bar=False,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_backend_dir, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="e.g. law-ai/InLegalBERT or BAAI/bge-base-en-v1.5")
    parser.add_argument("--pairs", required=True, help="citation_pairs_<version> folder (train.jsonl, dev.jsonl)")
    parser.add_argument("--output", required=True, help="where the fine-tuned model is saved")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128, help="pairs per step (= in-batch negatives)")
    parser.add_argument("--mini-batch-size", type=int, default=32, help="GPU chunk for the cached loss")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--hard-negatives", type=int, default=None,
                        help="hard negatives per row (default: as many as citation_pairs mined; 0 = in-batch only). "
                             "Rows with fewer are dropped")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="train on the first N pairs only (dry runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
        raise SystemExit(f"{args.output} exists and is not empty; pick a new --output (or --overwrite)")
    train_rows = read_jsonl(os.path.join(args.pairs, "train.jsonl"))
    dev_path = os.path.join(args.pairs, "dev.jsonl")
    dev_rows = read_jsonl(dev_path) if os.path.exists(dev_path) else []
    if args.limit:
        train_rows = train_rows[: args.limit]
    meta_path = os.path.join(args.pairs, "meta.json")
    pairs_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    n_neg = args.hard_negatives
    if n_neg is None:
        n_neg = int(((pairs_meta or {}).get("args") or {}).get("hard_negatives") or 0)
    cols = to_columns(train_rows, n_neg)
    print(f"{len(cols['anchor'])} training rows ({len(train_rows) - len(cols['anchor'])} dropped for <{n_neg} negatives), "
          f"{n_neg} hard negatives each, {len(dev_rows)} dev pairs")

    import tracking

    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking.TRACKING_URI)
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", TRAINING_EXPERIMENT)
    report_to = ["mlflow"] if tracking.MLFLOW_AVAILABLE else ["none"]

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.no_bf16
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    evaluator = dev_evaluator(dev_rows) if dev_rows else None
    run_name = f"{os.path.basename(os.path.normpath(args.output))}-{time.strftime('%Y%m%d-%H%M%S')}"
    training_args = SentenceTransformerTrainingArguments(
        # Next to, not inside, the model folder: it is what gets dvc-added.
        output_dir=os.path.normpath(args.output) + "-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=bf16,
        # The same chunk twice in a batch would be its own false negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        seed=args.seed,
        report_to=report_to,
        run_name=run_name,
    )
    if evaluator:
        print("Dev before training:", evaluator(model))
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=Dataset.from_dict(cols), loss=loss, evaluator=evaluator,
    )
    started = time.time()
    trainer.train()
    model.save(args.output)
    final = evaluator(model) if evaluator else {}
    info = {
        "base_model": args.base_model,
        "pairs": os.path.abspath(args.pairs),
        "pairs_meta": pairs_meta,
        "args": vars(args),
        "n_train_rows": len(cols["anchor"]),
        "hard_negatives": n_neg,
        "bf16": bf16,
        "dev_final": final,
        "train_seconds": round(time.time() - started, 1),
        "git_commit": git_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_name": run_name,
    }
    if info["pairs_meta"]:
        info["pairs_meta"].pop("holdout_judgment_ids", None)
    with open(os.path.join(args.output, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved {args.output}\nDev after training: {final}")


if __name__ == "__main__":
    main()
