# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14",
#     "huggingface-hub[hf_transfer]>=0.20",
#     "hf-xet>=1.1.7",
#     "transformers>=4.40",
#     "torch>=2.0",
#     "accelerate>=0.27",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
# ]
# ///
"""
Fine-tune an ESM2 protein language model on a sequence-level task and push to Hub.

Supports binary/multiclass classification and regression.
Dataset must have a 'split' column (train/val/test) — run preprocess-protein-dataset.py first.

Example:
    # Binary classification (binder / non-binder)
    uv run finetune-protein-lm.py \\
        owner/antibody-dataset owner/esm2-binder-classifier \\
        --label-col label --model-id facebook/esm2_t12_35M_UR50D

    # Regression (fitness score)
    uv run finetune-protein-lm.py \\
        owner/dms-dataset owner/esm2-fitness-regressor \\
        --label-col fitness --task regression --epochs 10
"""

import argparse
import logging

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import get_token, login
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import spearmanr
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_tokenize(dataset_id: str, seq_col: str, label_col: str, tokenizer, max_len: int, task: str):
    ds = load_dataset(dataset_id)

    def encode(batch):
        enc = tokenizer(batch[seq_col], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = [float(v) if task == "regression" else int(v) for v in batch[label_col]]
        return enc

    cols_to_remove = [c for c in ds["train"].column_names if c not in (seq_col, label_col, "split")]
    ds = ds.map(encode, batched=True, remove_columns=cols_to_remove + [seq_col, "split"])
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(task: str, num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "regression":
            preds = logits.squeeze()
            rho, _ = spearmanr(labels, preds)
            return {"spearman_r": float(rho), "mse": float(np.mean((preds - labels) ** 2))}
        preds = np.argmax(logits, axis=-1)
        metrics = {"accuracy": accuracy_score(labels, preds)}
        if num_labels == 2:
            metrics["mcc"] = matthews_corrcoef(labels, preds)
        return metrics
    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 on a protein classification or regression task.")
    parser.add_argument("dataset_id",    help="HF Hub dataset (must have train/val/test splits)")
    parser.add_argument("output_repo",   help="HF Hub model repo to push to (e.g. owner/my-model)")
    parser.add_argument("--model-id",    default="facebook/esm2_t12_35M_UR50D",
                        help="Pretrained model to fine-tune (default: ESM2 35M)")
    parser.add_argument("--seq-col",     default="sequence", help="Sequence column name")
    parser.add_argument("--label-col",   required=True,      help="Label column name")
    parser.add_argument("--task",        choices=["classification", "regression"], default="classification")
    parser.add_argument("--num-labels",  type=int, default=None,
                        help="Number of classes for classification (auto-detected if omitted)")
    parser.add_argument("--max-len",     type=int, default=512)
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    login(token=get_token())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Detect num_labels from dataset
    if args.task == "classification" and args.num_labels is None:
        probe = load_dataset(args.dataset_id, split="train")
        num_labels = probe.features[args.label_col].num_classes if hasattr(probe.features[args.label_col], "num_classes") \
            else len(set(probe[args.label_col]))
        logger.info(f"Auto-detected {num_labels} classes")
    else:
        num_labels = args.num_labels or 1

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = EsmForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=num_labels,
        problem_type="single_label_classification" if (args.task == "classification" and num_labels > 1)
                     else "regression",
    )

    # Dataset
    ds = load_and_tokenize(args.dataset_id, args.seq_col, args.label_col, tokenizer, args.max_len, args.task)

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_repo,
        hub_model_id=args.output_repo,
        push_to_hub=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman_r" if args.task == "regression" else "accuracy",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation") or ds.get("val"),
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(args.task, num_labels),
    )

    logger.info("Starting training...")
    trainer.train()

    test_split = ds.get("test")
    if test_split:
        results = trainer.evaluate(test_split, metric_key_prefix="test")
        logger.info(f"Test results: {results}")

    trainer.push_to_hub(commit_message=f"Fine-tuned {args.model_id} on {args.dataset_id}")
    logger.info(f"Model pushed to https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    main()
