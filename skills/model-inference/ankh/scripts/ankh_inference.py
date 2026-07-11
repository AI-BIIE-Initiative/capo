"""
ankh_inference.py — Standalone Ankh family inference script.

Supports two inference modes via CLI subcommands:
  embed       Extract protein embeddings (ankh-base, ankh-large, ankh3-large, ankh3-xl)
  complete    Ankh3 sequence completion: encoder receives first half, decoder generates second half

Model loading:
  All models are loaded via HuggingFace transformers (T5Tokenizer, T5EncoderModel,
  T5ForConditionalGeneration). T5Tokenizer must be used — AutoTokenizer resolves to the
  wrong class for Ankh checkpoints.

Ankh3 prefix tokens:
  Ankh3 models require a task prefix prepended to each sequence before tokenization.
    [NLU]  — embedding extraction (default for `embed`)
    [S2S]  — sequence completion (used automatically by `complete`); also accepted by `embed`
  Do not add prefixes to ankh-base or ankh-large inputs.

Requirements:
  embed / complete    transformers ≥4.30, torch ≥2.0

License: CC-BY-NC-SA-4.0 — non-commercial use only.

All heavy imports are deferred inside each function so this module can be imported
even when only some packages are installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps CLI short name → (hf_checkpoint_id, is_ankh3)
# is_ankh3=True means the model requires [NLU] or [S2S] prefix tokens.
_ANKH_REGISTRY: dict[str, tuple[str, bool]] = {
    "ankh-base":   ("ElnaggarLab/ankh-base",   False),
    "ankh-large":  ("ElnaggarLab/ankh-large",  False),
    "ankh3-large": ("ElnaggarLab/ankh3-large", True),
    "ankh3-xl":    ("ElnaggarLab/ankh3-xl",    True),
}

_ANKH3_MODELS = {k for k, (_, is3) in _ANKH_REGISTRY.items() if is3}


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
# Embedding — all Ankh variants
# ---------------------------------------------------------------------------

def embed_ankh(
    sequences: list[str],
    model_name: str = "ankh-base",
    prefix: str | None = None,
    batch_size: int = 16,
    device: str = "auto",
) -> "np.ndarray":
    """
    Extract mean-pooled protein embeddings from any Ankh variant.

    For ankh-base and ankh-large, sequences are tokenized as-is.
    For ankh3-large and ankh3-xl, sequences are automatically prefixed with
    "[NLU]" (or the value of `prefix`) before tokenization.

    Args:
        sequences:   List of amino-acid sequences.
        model_name:  Short model name. One of: "ankh-base", "ankh-large",
                     "ankh3-large", "ankh3-xl".
                     Full HuggingFace checkpoint IDs (e.g. "ElnaggarLab/ankh3-xl")
                     are also accepted and treated as Ankh3 when they contain "ankh3".
        prefix:      Override the prefix prepended to each sequence for Ankh3 models.
                     Accepted values: "[NLU]" (default), "[S2S]".
                     Ignored for ankh-base / ankh-large.
        batch_size:  Sequences per forward pass. Reduce on OOM.
                     Suggested starting points: ankh-base → 32, ankh-large → 16,
                     ankh3-large → 16, ankh3-xl → 8.
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        np.ndarray of shape (len(sequences), d_model), dtype float32.
        Mean-pooled over all attended positions (attention_mask handles padding).
        Row order matches input.

    Raises:
        ValueError:  if model_name is not in the supported set.
        ImportError: if transformers is not installed.
    """
    import numpy as np
    import torch
    from transformers import T5Tokenizer, T5EncoderModel

    ckpt, is_ankh3 = _resolve_model(model_name)
    resolved_prefix = _resolve_embed_prefix(prefix, is_ankh3)

    dev = get_device(device)
    tokenizer = T5Tokenizer.from_pretrained(ckpt)
    model = T5EncoderModel.from_pretrained(ckpt).eval().to(dev)

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        if resolved_prefix:
            batch = [resolved_prefix + seq for seq in batch]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=False,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state                          # (B, L, D)
        mask   = inputs["attention_mask"].unsqueeze(-1).float()     # (B, L, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # (B, D)
        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Sequence completion — Ankh3 only
# ---------------------------------------------------------------------------

def complete_ankh3(
    sequence: str,
    model_name: str = "ankh3-xl",
    split_at: int | None = None,
    device: str = "auto",
) -> str:
    """
    Complete the second half of a protein sequence using Ankh3.

    The encoder receives the first `split_at` residues (prefixed with "[S2S]");
    the decoder autoregressively generates the remainder.

    Args:
        sequence:   Full wild-type amino-acid sequence. Only the first half
                    is passed to the encoder; the second half is generated.
        model_name: "ankh3-large" or "ankh3-xl". Raises ValueError for
                    ankh-base / ankh-large (they were not trained with the
                    sequence-completion objective).
        split_at:   Number of residues to feed to the encoder.
                    None → len(sequence) // 2 (split at midpoint).
        device:     "auto" | "cuda" | "mps" | "cpu".

    Returns:
        Completed sequence string: sequence[:split_at] + <generated continuation>.
        The generated portion has length (len(sequence) - split_at).

    Raises:
        ValueError:  if model_name resolves to ankh-base or ankh-large.
        ImportError: if transformers is not installed.
    """
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from transformers.generation import GenerationConfig

    ckpt, is_ankh3 = _resolve_model(model_name)
    if not is_ankh3:
        raise ValueError(
            f"Sequence completion requires an Ankh3 model. "
            f"'{model_name}' ({ckpt}) was not trained with the [S2S] objective. "
            "Use 'ankh3-large' or 'ankh3-xl'."
        )

    half = split_at if split_at is not None else len(sequence) // 2
    target_len = len(sequence) - half  # expected length of the generated segment

    dev = get_device(device)
    tokenizer = T5Tokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt).eval().to(dev)

    s2s_input = "[S2S]" + sequence[:half]
    encoded = tokenizer(
        s2s_input,
        return_tensors="pt",
        add_special_tokens=True,
        is_split_into_words=False,
    )
    encoded = {k: v.to(dev) for k, v in encoded.items()}

    # +1 accounts for the decoder start-of-sequence token added by T5.
    gen_cfg = GenerationConfig(
        min_length=target_len + 1,
        max_length=target_len + 1,
        do_sample=False,
        num_beams=1,
    )

    with torch.no_grad():
        generated = model.generate(encoded["input_ids"], gen_cfg)

    continuation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return sequence[:half] + continuation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_model(model_name: str) -> tuple[str, bool]:
    """
    Return (hf_checkpoint_id, is_ankh3) for a short name or full HF ID.

    Short names are looked up in _ANKH_REGISTRY. Unknown names are passed
    through as full HuggingFace IDs; is_ankh3 is inferred from the name.
    """
    if model_name in _ANKH_REGISTRY:
        return _ANKH_REGISTRY[model_name]
    # Accept full HF IDs (e.g. "ElnaggarLab/ankh3-xl")
    is_ankh3 = "ankh3" in model_name.lower()
    return model_name, is_ankh3


def _resolve_embed_prefix(prefix: str | None, is_ankh3: bool) -> str | None:
    """Return the prefix string to prepend, or None if no prefix should be added."""
    if not is_ankh3:
        return None
    if prefix is None:
        return "[NLU]"
    if prefix not in ("[NLU]", "[S2S]"):
        raise ValueError(
            f"prefix must be '[NLU]' or '[S2S]' for Ankh3 models, got '{prefix!r}'."
        )
    return prefix


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
        emb = embed_ankh(
            seqs,
            model_name=args.model,
            prefix=args.prefix,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.save(args.output, emb)
        print(f"Saved embeddings {emb.shape} → {args.output}")

    elif args.command == "complete":
        result = complete_ankh3(
            args.sequence,
            model_name=args.model,
            split_at=args.split_at,
            device=args.device,
        )
        if args.output:
            Path(args.output).write_text(result)
            print(f"Wrote completed sequence → {args.output}")
        else:
            print(result)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ankh family inference — embeddings and sequence completion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- embed ----
    p = subparsers.add_parser(
        "embed",
        help="Extract mean-pooled protein embeddings (all Ankh variants).",
    )
    _add_sequence_args(p)
    p.add_argument(
        "--model",
        default="ankh-base",
        choices=list(_ANKH_REGISTRY),
        help="Ankh variant. Default: ankh-base.",
    )
    p.add_argument(
        "--prefix",
        default=None,
        choices=["[NLU]", "[S2S]"],
        help=(
            "Prefix prepended to each sequence for Ankh3 models. "
            "Default: [NLU]. Ignored for ankh-base / ankh-large."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sequences per forward pass. Reduce on OOM. Default: 16.",
    )
    p.add_argument("--output", default="embeddings.npy", help="Output .npy file.")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu.")

    # ---- complete ----
    p = subparsers.add_parser(
        "complete",
        help="Ankh3 sequence completion: generate the second half from the first.",
    )
    p.add_argument("--sequence", required=True, help="Full amino-acid sequence to split and complete.")
    p.add_argument(
        "--model",
        default="ankh3-xl",
        choices=sorted(_ANKH3_MODELS),
        help="Ankh3 variant. Default: ankh3-xl.",
    )
    p.add_argument(
        "--split-at",
        type=int,
        default=None,
        help=(
            "Number of residues fed to the encoder. "
            "Default: len(sequence) // 2 (midpoint split)."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write completed sequence to file. Default: print to stdout.",
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
