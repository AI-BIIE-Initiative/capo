# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.14",
#     "huggingface-hub[hf_transfer]>=0.20",
#     "hf-xet>=1.1.7",
#     "transformers>=4.40",
#     "torch>=2.0",
#     "numpy>=1.24",
# ]
# ///
"""
Run batch inference over a protein sequence dataset and push results to Hub.

Two modes:
  embeddings   — extract mean-pooled residue embeddings from a base ESM2 model
  predictions  — get predicted class or score from a fine-tuned classifier/regressor

Results are added as new columns to the dataset and pushed to Hub.

Example:
    # Extract ESM2 embeddings
    uv run protein-batch-inference.py \\
        owner/my-sequences owner/my-sequences-embedded \\
        --mode embeddings --model-id facebook/esm2_t33_650M_UR50D

    # Run a fine-tuned classifier
    uv run protein-batch-inference.py \\
        owner/my-sequences owner/my-sequences-scored \\
        --mode predictions --model-id owner/esm2-binder-classifier
"""

import argparse
import logging

import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard, get_token, login
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_embeddings(sequences: list[str], model, tokenizer, device: str, max_len: int) -> np.ndarray:
    """Mean-pool last hidden state over sequence positions (excluding BOS/EOS)."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    hidden = out.last_hidden_state          # (B, L+2, D)
    token_hidden = hidden[:, 1:-1, :]      # strip BOS and EOS
    mask = inputs["attention_mask"][:, 1:-1].unsqueeze(-1).float()
    pooled = (token_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return pooled.cpu().numpy()


@torch.no_grad()
def run_predictions(sequences: list[str], model, tokenizer, device: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_label_or_score, confidence_or_logit)."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits          # (B, num_labels) or (B, 1)

    if logits.shape[-1] == 1:               # regression
        scores = logits.squeeze(-1).cpu().numpy()
        return scores, scores
    else:                                    # classification
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        return preds, probs.max(axis=-1)


def batch_iter(lst: list, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch protein inference: embeddings or predictions.")
    parser.add_argument("input_dataset", help="HF Hub dataset ID to run inference on")
    parser.add_argument("output_repo",   help="HF Hub dataset repo for results")
    parser.add_argument("--model-id",    default="facebook/esm2_t33_650M_UR50D",
                        help="Model to use (base ESM2 for embeddings, fine-tuned for predictions)")
    parser.add_argument("--mode",        choices=["embeddings", "predictions"], default="embeddings")
    parser.add_argument("--seq-col",     default="sequence", help="Sequence column name")
    parser.add_argument("--split",       default="test",     help="Dataset split to run inference on")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--max-len",     type=int, default=1024)
    args = parser.parse_args()

    login(token=get_token())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  Mode: {args.mode}  |  Model: {args.model_id}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if args.mode == "embeddings":
        model = EsmModel.from_pretrained(args.model_id).eval().to(device)
    else:
        model = EsmForSequenceClassification.from_pretrained(args.model_id).eval().to(device)

    # Load dataset
    logger.info(f"Loading {args.input_dataset} split={args.split}")
    ds = load_dataset(args.input_dataset, split=args.split)
    sequences = ds[args.seq_col]
    logger.info(f"Running inference on {len(sequences):,} sequences")

    # Run in batches
    all_primary, all_secondary = [], []
    for i, batch in enumerate(batch_iter(sequences, args.batch_size)):
        if i % 10 == 0:
            logger.info(f"Batch {i+1} / {len(sequences) // args.batch_size + 1}")
        if args.mode == "embeddings":
            emb = run_embeddings(batch, model, tokenizer, device, args.max_len)
            all_primary.extend(emb.tolist())
        else:
            preds, conf = run_predictions(batch, model, tokenizer, device, args.max_len)
            all_primary.extend(preds.tolist())
            all_secondary.extend(conf.tolist())

    # Add result columns
    if args.mode == "embeddings":
        ds = ds.add_column("embedding", all_primary)
    else:
        ds = ds.add_column("predicted_label", all_primary)
        ds = ds.add_column("confidence", all_secondary)

    ds.push_to_hub(args.output_repo)

    card_content = f"""---
license: other
---
# {args.output_repo.split('/')[-1]}

Inference results from `{args.model_id}` in `{args.mode}` mode,
applied to [{args.input_dataset}](https://huggingface.co/datasets/{args.input_dataset}) (split: `{args.split}`).

## Columns added
{"- `embedding`: mean-pooled ESM2 residue embeddings (float list)" if args.mode == "embeddings"
 else "- `predicted_label`: predicted class index or regression score\n- `confidence`: softmax probability or logit"}
"""
    DatasetCard(card_content).push_to_hub(args.output_repo)
    logger.info(f"Results pushed to https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
