"""
prottrans_inference.py — Standalone ProtTrans inference script.

Supports one inference mode via CLI:
  embed    Extract mean-pooled protein embeddings from ProtBert or ProtT5-XL-UniRef50

Model loading:
  All models are loaded via HuggingFace transformers.
    ProtBert   — BertTokenizer + BertModel  (Rostlab/prot_bert)
    ProtT5-XL  — T5Tokenizer  + T5EncoderModel  (Rostlab/prot_t5_xl_uniref50)
                               or encoder-only half: Rostlab/prot_t5_xl_half_uniref50-enc

Critical preprocessing (applies to BOTH models):
  Sequences must be space-separated before tokenization:
      " ".join(list(re.sub(r"[UZOB]", "X", seq.upper())))
  Skipping this step causes multi-character tokens and wrong embeddings with no error.

Mean-pooling conventions:
  ProtBert  — tokenized form is [CLS] A E T … [SEP]; exclude positions 0 and seqlen-1.
  ProtT5    — tokenized form is A E T … [EOS]; exclude position seqlen-1 (EOS).

Requirements:
  pip install transformers torch
  pip install sentencepiece protobuf   # required for T5 tokenizer

License: models released under CC-BY-4.0 (ProtBert) and Research Use Agreement (ProtT5);
  check individual HuggingFace model cards before production deployment.

All heavy imports are deferred inside each function so this module can be imported
even when only some packages are installed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps CLI short name → (hf_checkpoint_id, arch)
# arch "bert" → BertTokenizer + BertModel
# arch "t5"   → T5Tokenizer  + T5EncoderModel
_PROTTRANS_REGISTRY: dict[str, tuple[str, str]] = {
    "prot-bert":       ("Rostlab/prot_bert",                    "bert"),
    "prot-t5-xl":      ("Rostlab/prot_t5_xl_uniref50",          "t5"),
    "prot-t5-xl-half": ("Rostlab/prot_t5_xl_half_uniref50-enc", "t5"),
}


# ---------------------------------------------------------------------------
# Device utility
# ---------------------------------------------------------------------------

def get_device(prefer: str = "auto") -> "torch.device":
    """
    Return the best available torch.device.

    Args:
        prefer: "auto" | "cuda" | "mps" | "cpu".
                "auto" selects cuda > mps > cpu in that order.
    """
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_sequences(sequences: list[str]) -> list[str]:
    """
    Prepare amino-acid sequences for ProtBert / ProtT5 tokenization.

    Applies two transformations required by both model families:
      1. Replace rare/ambiguous amino acids (U, Z, O, B) with X.
      2. Space-separate each character so the tokenizer treats each amino
         acid as a separate token.

    Args:
        sequences: List of raw amino-acid sequences (any case).

    Returns:
        List of preprocessed strings, e.g. "PRTEINO" → "P R T E I N O".
    """
    return [" ".join(list(re.sub(r"[UZOB]", "X", seq.upper()))) for seq in sequences]


# ---------------------------------------------------------------------------
# Embedding — ProtBert
# ---------------------------------------------------------------------------

def embed_protbert(
    sequences: list[str],
    model_name: str = "Rostlab/prot_bert",
    batch_size: int = 32,
    device: str = "auto",
) -> "np.ndarray":
    """
    Extract mean-pooled per-protein embeddings from ProtBert.

    Sequences are preprocessed (space-separation + rare-AA replacement)
    inside this function; pass raw amino-acid strings.

    Tokenized form: [CLS] A E T … [SEP] <padding>
    Mean pool excludes [CLS] (index 0) and [SEP] (last attended index).

    Args:
        sequences:   List of raw amino-acid sequences.
        model_name:  HuggingFace checkpoint. Default: "Rostlab/prot_bert".
        batch_size:  Sequences per forward pass. Reduce on OOM.
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        np.ndarray of shape (len(sequences), 1024), dtype float32.
        Row order matches input.

    Raises:
        ImportError: if transformers is not installed.
    """
    import numpy as np
    import torch
    from transformers import BertTokenizer, BertModel

    dev = get_device(device)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name).eval().to(dev)

    sequences_proc = preprocess_sequences(sequences)
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences_proc), batch_size):
        batch = sequences_proc[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        hidden  = output.last_hidden_state         # (B, L_padded, 1024)
        seqlens = inputs["attention_mask"].sum(1)  # (B,) — [CLS] + residues + [SEP]

        for i in range(hidden.size(0)):
            slen = int(seqlens[i].item())
            # positions 1 .. slen-2 are residue tokens (exclude [CLS] at 0 and [SEP] at slen-1)
            residue_emb = hidden[i, 1 : slen - 1, :]  # (n_residues, 1024)
            all_embeddings.append(residue_emb.mean(0).cpu().float().numpy())

    return np.stack(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Embedding — ProtT5
# ---------------------------------------------------------------------------

def embed_prot_t5(
    sequences: list[str],
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 8,
    device: str = "auto",
) -> "np.ndarray":
    """
    Extract mean-pooled per-protein embeddings from ProtT5-XL.

    Sequences are preprocessed (space-separation + rare-AA replacement)
    inside this function; pass raw amino-acid strings.

    Tokenized form: A E T … [EOS] <padding>  (no [CLS] token)
    Mean pool excludes EOS (last attended index per sequence).

    Args:
        sequences:   List of raw amino-acid sequences.
        model_name:  HuggingFace checkpoint.
                     "Rostlab/prot_t5_xl_uniref50" — full precision (default).
                     "Rostlab/prot_t5_xl_half_uniref50-enc" — fp16 encoder-only (faster,
                     less memory; cast to float32 automatically on CPU).
        batch_size:  Sequences per forward pass. Reduce on OOM.
                     Suggested: prot_t5_xl_uniref50 → 8, half-enc → 16.
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        np.ndarray of shape (len(sequences), 1024), dtype float32.
        Row order matches input.

    Raises:
        ImportError: if transformers or sentencepiece is not installed.
    """
    import numpy as np
    import torch
    from transformers import T5Tokenizer, T5EncoderModel

    dev = get_device(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).eval().to(dev)

    # Half-precision checkpoint loads as fp16; float32 is required on CPU.
    if dev.type == "cpu":
        model.to(torch.float32)

    sequences_proc = preprocess_sequences(sequences)
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences_proc), batch_size):
        batch = sequences_proc[start : start + batch_size]
        ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding="longest")
        input_ids      = torch.tensor(ids["input_ids"]).to(dev)
        attention_mask = torch.tensor(ids["attention_mask"]).to(dev)

        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden  = embedding_repr.last_hidden_state  # (B, L_padded, 1024)
        seqlens = attention_mask.sum(1)             # (B,) — residues + EOS

        for i in range(hidden.size(0)):
            slen = int(seqlens[i].item())
            # positions 0 .. slen-2 are residue tokens (exclude EOS at slen-1)
            residue_emb = hidden[i, : slen - 1, :]  # (n_residues, 1024)
            all_embeddings.append(residue_emb.mean(0).cpu().float().numpy())

    return np.stack(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Unified embed entry point
# ---------------------------------------------------------------------------

def embed_prottrans(
    sequences: list[str],
    model_name: str = "prot-bert",
    batch_size: int | None = None,
    device: str = "auto",
) -> "np.ndarray":
    """
    Extract mean-pooled embeddings from any supported ProtTrans model.

    Dispatches to embed_protbert() or embed_prot_t5() based on model_name.

    Args:
        sequences:   List of raw amino-acid sequences.
        model_name:  Short name from registry: "prot-bert", "prot-t5-xl",
                     "prot-t5-xl-half". Full HuggingFace checkpoint IDs are
                     also accepted.
        batch_size:  Sequences per forward pass. None → architecture default
                     (32 for BERT, 8 for T5).
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        np.ndarray of shape (len(sequences), 1024), dtype float32.

    Raises:
        ValueError:  if model_name is not recognised.
        ImportError: if transformers (or sentencepiece for T5) is not installed.
    """
    ckpt, arch = _resolve_model(model_name)

    if arch == "bert":
        bs = batch_size if batch_size is not None else 32
        return embed_protbert(sequences, model_name=ckpt, batch_size=bs, device=device)
    else:  # t5
        bs = batch_size if batch_size is not None else 8
        return embed_prot_t5(sequences, model_name=ckpt, batch_size=bs, device=device)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_model(model_name: str) -> tuple[str, str]:
    """
    Return (hf_checkpoint_id, arch) for a short name or a full HF ID.

    Short names are looked up in _PROTTRANS_REGISTRY. Unknown strings are
    passed through as full HuggingFace IDs; arch is inferred from the name.
    """
    if model_name in _PROTTRANS_REGISTRY:
        return _PROTTRANS_REGISTRY[model_name]
    # Full HF IDs: infer arch from checkpoint name
    arch = "bert" if "bert" in model_name.lower() else "t5"
    return model_name, arch


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _load_sequences(args: argparse.Namespace) -> list[str]:
    if getattr(args, "sequence", None):
        return [args.sequence.strip()]
    return [s.strip() for s in Path(args.sequences).read_text().splitlines() if s.strip()]


def _add_sequence_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", help="Single amino-acid sequence.")
    group.add_argument("--sequences", help="Path to text file, one sequence per line.")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _dispatch(args: argparse.Namespace) -> None:
    import numpy as np

    if args.command == "embed":
        seqs = _load_sequences(args)
        emb = embed_prottrans(
            seqs,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.save(args.output, emb)
        print(f"Saved embeddings {emb.shape} → {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ProtTrans inference — mean-pooled protein embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- embed ----
    p = subparsers.add_parser(
        "embed",
        help="Extract mean-pooled per-protein embeddings (ProtBert or ProtT5-XL).",
    )
    _add_sequence_args(p)
    p.add_argument(
        "--model",
        default="prot-bert",
        choices=list(_PROTTRANS_REGISTRY),
        help=(
            "Model to use. "
            "'prot-bert': Rostlab/prot_bert (BERT, 1024-d). "
            "'prot-t5-xl': Rostlab/prot_t5_xl_uniref50 (T5, 1024-d, full precision). "
            "'prot-t5-xl-half': Rostlab/prot_t5_xl_half_uniref50-enc (T5, fp16 encoder-only). "
            "Default: prot-bert."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Sequences per forward pass. Reduce on OOM. "
            "Default: 32 for ProtBert, 8 for ProtT5."
        ),
    )
    p.add_argument("--output", default="embeddings.npy", help="Output .npy file.")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
