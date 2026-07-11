"""
esm_inference.py — Standalone ESM family inference script.

Supports seven inference modes via CLI subcommands:
  embed-esm2-hf       ESM2 embeddings via HuggingFace transformers (recommended)
  embed-esm2-native   ESM2 embeddings via fair-esm (use when ESMFold also needed)
  score-variants      ESM-1v masked-marginal zero-shot variant scoring
  fold                ESMFold structure prediction → PDB files
  embed-esm3          ESM3 sequence-track embeddings
  generate-esm3       ESM3 masked sequence generation / infilling
  embed-esmc          ESM Cambrian (ESM C) per-residue embeddings

Requirements by mode:
  embed-esm2-hf       transformers ≥4.30, torch ≥2.0
  embed-esm2-native   fair-esm  (pip install fair-esm)
  score-variants      fair-esm
  fold                fair-esm
  embed-esm3          esm       (pip install esm)
  generate-esm3       esm
  embed-esmc          esm       (pip install esm)

Package choice rules:
  ESM2 standard inference        → transformers (preferred)
  ESMFold / MSA / IF1 / FAIR     → fair-esm
  ESM3 open model or Forge API   → official esm package
  ESM C local or Forge API       → official esm package

All heavy imports are deferred inside each function so this module can be
imported even when only some packages are installed.
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
# ESM2 — HuggingFace (recommended)
# ---------------------------------------------------------------------------

def embed_esm2_hf(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 32,
    max_length: int = 1022,
    device: str = "auto",
    pool: str = "mean",
) -> "np.ndarray":
    """
    Compute ESM2 sequence embeddings using HuggingFace transformers.

    Recommended path for standard ESM2 inference. Does not require fair-esm.

    Args:
        sequences:  List of amino-acid sequences. Non-standard characters are
                    mapped to <unk> by the tokenizer — filter upstream if needed.
        model_name: HuggingFace model ID.
                    Options: facebook/esm2_t6_8M_UR50D (fast, CPU-friendly),
                    facebook/esm2_t12_35M_UR50D, facebook/esm2_t30_150M_UR50D,
                    facebook/esm2_t33_650M_UR50D (default, most tasks),
                    facebook/esm2_t36_3B_UR50D (max quality, LoRA only),
                    facebook/esm2_t48_15B_UR50D (research, frozen only).
        batch_size: Sequences per forward pass. Reduce on OOM.
                    Starting points: 650M → 32, 3B → 8, 15B → 2.
        max_length: Tokenizer max length. Hard limit for all ESM2 variants is
                    1022 (model was trained with BOS+EOS = 1024 total).
                    Sequences longer than this are silently truncated.
        device:     "auto" | "cuda" | "mps" | "cpu".
        pool:       "mean" — mean pool over residue positions, excluding BOS/EOS.
                    "bos"  — return the BOS (position 0) hidden state.

    Returns:
        np.ndarray of shape (len(sequences), d_model), dtype float32.
        Row order matches input.

    Raises:
        ValueError:   if pool is not "mean" or "bos".
        ImportError:  if transformers is not installed.
    """
    import numpy as np
    import torch
    from transformers import EsmModel, EsmTokenizer

    if pool not in ("mean", "bos"):
        raise ValueError(f"pool must be 'mean' or 'bos', got '{pool!r}'")

    dev = get_device(device)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().to(dev)

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (B, L+2, D)

        if pool == "bos":
            emb = hidden[:, 0, :].cpu().float().numpy()
            all_embeddings.append(emb)
        else:
            # Per-sequence mean pool excluding BOS (index 0) and EOS.
            # seqlens counts BOS + residues + EOS.
            seqlens = inputs["attention_mask"].sum(dim=1)
            B = hidden.size(0)
            batch_embs: list[np.ndarray] = []
            for i in range(B):
                L_i = max(int(seqlens[i].item()) - 2, 1)  # residue count only
                residue_h = hidden[i, 1 : 1 + L_i, :]     # (L_i, D)
                batch_embs.append(residue_h.mean(0).cpu().float().numpy())
            all_embeddings.append(np.stack(batch_embs))

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# ESM2 — native fair-esm
# ---------------------------------------------------------------------------

# Maps HuggingFace model ID → (hub_name, last_repr_layer) -> full hf registry is available here: BIIE-AI/protein-model-registry
_NATIVE_ESM2_REGISTRY: dict[str, tuple[str, int]] = {
    "facebook/esm2_t6_8M_UR50D":    ("facebook/esm2_t6_8M_UR50D",    6),
    "facebook/esm2_t12_35M_UR50D":  ("facebook/esm2_t12_35M_UR50D",  12),
    "facebook/esm2_t30_150M_UR50D": ("facebook/esm2_t30_150M_UR50D", 30),
    "facebook/esm2_t33_650M_UR50D": ("facebook/esm2_t33_650M_UR50D", 33),
    "facebook/esm2_t36_3B_UR50D":   ("facebook/esm2_t36_3B_UR50D",   36),
}


def embed_esm2_native(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    repr_layer: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> "np.ndarray":
    """
    Compute ESM2 embeddings using the native fair-esm API.

    Use this path when ESMFold, MSA Transformer, or ESM-IF1 are also needed
    in the same environment. For standalone ESM2 embeddings, prefer embed_esm2_hf().

    Args:
        sequences:   List of amino-acid sequences.
        model_name:  HuggingFace-style model ID (used as fair-esm hub identifier).
                     Supported: see _NATIVE_ESM2_REGISTRY keys.
        repr_layer:  Transformer layer to extract representations from.
                     None → last layer inferred from model name.
        batch_size:  Sequences per forward pass.
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        np.ndarray of shape (len(sequences), d_model), dtype float32.
        Mean-pooled over residue positions, excluding BOS and EOS tokens.

    Raises:
        ImportError: if fair-esm is not installed.
        ValueError:  if model_name is not in the supported set.
    """
    import numpy as np
    import torch
    import esm as fair_esm

    if model_name not in _NATIVE_ESM2_REGISTRY:
        raise ValueError(
            f"model_name '{model_name}' not supported by native path. "
            f"Supported: {list(_NATIVE_ESM2_REGISTRY)}"
        )
    hub_name, default_layer = _NATIVE_ESM2_REGISTRY[model_name]
    layer = repr_layer if repr_layer is not None else default_layer

    dev = get_device(device)
    model, alphabet = fair_esm.pretrained.load_model_and_alphabet_hub(hub_name)
    model = model.eval().to(dev)
    batch_converter = alphabet.get_batch_converter()

    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        batch_data = [(f"seq_{i}", s) for i, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(dev)

        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)

        reps = out["representations"][layer]  # (B, L+2, D)
        for i, (_, seq) in enumerate(batch_data):
            # Exclude BOS (index 0) and EOS (index len(seq)+1)
            emb = reps[i, 1 : len(seq) + 1].mean(0).cpu().float().numpy()
            all_embeddings.append(emb)

    return np.stack(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# ESM-1v — zero-shot variant scoring
# ---------------------------------------------------------------------------

_ESM1V_LOADERS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
]


def score_variants_esm1v(
    sequence: str,
    mutations: list[tuple[int, str, str]],
    model_indices: list[int] | None = None,
    device: str = "auto",
) -> list[float]:
    """
    Score single-point substitutions using ESM-1v masked-marginal log-odds.

    For each mutation, masks the position, runs a forward pass, and returns
    log P(mut_aa | context) - log P(wt_aa | context). Higher = more likely
    beneficial.

    Args:
        sequence:      Wild-type amino-acid sequence.
        mutations:     List of (mut_pos, mut_aa, wt_aa) tuples.
                       mut_pos is 1-indexed (UniProt convention).
                       BOS sits at token 0, so residue at position P is at
                       token index P — no off-by-one adjustment needed.
        model_indices: Which ESM-1v ensemble members to use (1-indexed, 1–5).
                       None → all 5 (recommended for production).
                       Pass [1] for a quick single-model run.
        device:        "auto" | "cuda" | "mps" | "cpu".

    Returns:
        List of float, one score per mutation in input order, averaged over
        the requested ensemble members.

    Raises:
        ImportError: if fair-esm is not installed.
    """
    import numpy as np
    import torch
    import esm as fair_esm

    if model_indices is None:
        model_indices = [1, 2, 3, 4, 5]

    dev = get_device(device)
    scores_by_model: list[list[float]] = []

    for idx in model_indices:
        loader_name = _ESM1V_LOADERS[idx - 1]
        loader = getattr(fair_esm.pretrained, loader_name)
        model, alphabet = loader()
        batch_converter = alphabet.get_batch_converter()
        model = model.eval().to(dev)

        model_scores: list[float] = []
        for (mut_pos, mut_aa, wt_aa) in mutations:
            # mut_pos is 1-indexed. BOS is at token 0, so residue 1 is at
            # token index 1 — token_pos = mut_pos with no further adjustment.
            token_pos = mut_pos
            _, _, tokens = batch_converter([("protein", sequence)])
            tokens = tokens.to(dev)
            tokens[0, token_pos] = alphabet.mask_idx

            with torch.no_grad():
                logits = model(tokens)["logits"][0, token_pos]  # (vocab_size,)

            score = (
                logits[alphabet.get_idx(mut_aa)] - logits[alphabet.get_idx(wt_aa)]
            ).item()
            model_scores.append(score)

        scores_by_model.append(model_scores)

    return list(np.mean(scores_by_model, axis=0))


# ---------------------------------------------------------------------------
# ESMFold — structure prediction
# ---------------------------------------------------------------------------

def fold_sequences_esmfold(
    sequences: list[str],
    chunk_size: int | None = None,
    device: str = "auto",
) -> list[str]:
    """
    Predict protein structures using ESMFold. Loads the model once and loops.

    ESMFold is a frozen predictor — do not fine-tune it.

    Args:
        sequences:   List of amino-acid sequences.
        chunk_size:  Axial attention chunk size. None → no chunking (fastest,
                     more memory). Recommended values:
                       sequences 400–800 aa on 24 GB GPU → 64
                       sequences 800–1000 aa              → 32
                       sequences >1000 aa                 → 16
        device:      "auto" | "cuda" | "mps" | "cpu".
                     ESMFold requires CUDA; MPS falls back to CPU with a warning.

    Returns:
        List of PDB-formatted strings, one per input sequence.

    Raises:
        ImportError:  if fair-esm is not installed.
        RuntimeError: on CUDA OOM — reduce chunk_size or sequence length.
    """
    import warnings
    import torch
    import esm as fair_esm

    dev = get_device(device)
    if dev.type == "mps":
        warnings.warn(
            "ESMFold does not support MPS. Falling back to CPU (slow).",
            UserWarning,
            stacklevel=2,
        )
        dev = torch.device("cpu")
    elif dev.type == "cpu":
        warnings.warn(
            "ESMFold on CPU is very slow. A CUDA GPU is strongly recommended.",
            UserWarning,
            stacklevel=2,
        )

    model = fair_esm.pretrained.esmfold_v1().eval().to(dev)
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)

    pdb_strings: list[str] = []
    with torch.no_grad():
        for seq in sequences:
            pdb_strings.append(model.infer_pdb(seq))

    return pdb_strings


# ---------------------------------------------------------------------------
# ESM3 — embeddings
# ---------------------------------------------------------------------------

def embed_esm3(
    sequences: list[str],
    model_name: str = "esm3-sm-open-v1",
    device: str = "auto",
) -> list["np.ndarray"]:
    """
    Extract sequence-track embeddings from ESM3 (EvolutionaryScale).

    Requires the official esm package (`pip install esm`), distinct from fair-esm.
    The open model `esm3-sm-open-v1` (1.4B parameters) requires ~6 GB VRAM.

    Args:
        sequences:   List of amino-acid sequences.
        model_name:  "esm3-sm-open-v1" for local open weights.
                     Forge API models are not supported here; use embed_esm3 via
                     the Forge client if needed.
        device:      "auto" | "cuda" | "mps" | "cpu".

    Returns:
        List of np.ndarray, one per input sequence, each of shape (L, d_model)
        where L is the sequence length (ESM3 SDK does not include BOS/EOS).
        Lengths vary — to get a single (d_model,) vector per sequence, use
        `.mean(axis=0)` on each array.

    Raises:
        ImportError: if the esm package is not installed.
    """
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    dev = get_device(device)
    model = ESM3.from_pretrained(model_name).to(dev).eval()

    result: list[np.ndarray] = []
    for seq in sequences:
        protein = ESMProtein(sequence=seq)
        encoded = model.encode(protein)
        emb = encoded.sequence.detach().cpu().float().numpy()  # (L, d_model)
        result.append(emb)

    return result


# ---------------------------------------------------------------------------
# ESM3 — generation
# ---------------------------------------------------------------------------

def generate_esm3(
    sequence_with_masks: str,
    model_name: str = "esm3-sm-open-v1",
    num_steps: int = 8,
    device: str = "auto",
    forge_token: str | None = None,
    forge_url: str = "https://forge.evolutionaryscale.ai",
) -> str:
    """
    Fill masked positions in a sequence using ESM3 generative inference.

    Use "_" as the mask character. The model fills all "_" positions iteratively
    in `num_steps` denoising steps.

    Args:
        sequence_with_masks: Amino-acid sequence with "_" at positions to fill.
                             Example: "MKTAY____QRQISFVK".
        model_name:          "esm3-sm-open-v1" for local inference, or a Forge
                             model name such as "esm3-medium-2024-08".
        num_steps:           Denoising steps. 8 is a reasonable default; use
                             16–32 for higher quality in production.
        device:              "auto" | "cuda" | "mps" | "cpu".
                             Ignored when using the Forge API.
        forge_token:         Forge API token. When None and model_name is not
                             "esm3-sm-open-v1", reads ESM_API_KEY env var.
        forge_url:           Forge API endpoint URL.

    Returns:
        Completed amino-acid sequence string. If not all masked positions could
        be filled, a UserWarning is issued and the partial result is returned.

    Raises:
        ImportError: if the esm package is not installed.
        ValueError:  if Forge API is required but no token is available.
    """
    import os
    import warnings
    from esm.sdk.api import ESMProtein, GenerationConfig

    protein = ESMProtein(sequence=sequence_with_masks)
    config = GenerationConfig(track="sequence", num_steps=num_steps)

    use_forge = forge_token is not None or model_name != "esm3-sm-open-v1"

    if use_forge:
        token = forge_token or os.environ.get("ESM_API_KEY")
        if not token:
            raise ValueError(
                "A Forge API token is required for non-open models. "
                "Set the ESM_API_KEY environment variable or pass forge_token."
            )
        from esm.sdk import client as esm_client
        client = esm_client(model=model_name, url=forge_url, token=token)
        output = client.generate(protein, config)
    else:
        from esm.models.esm3 import ESM3
        dev = get_device(device)
        model = ESM3.from_pretrained(model_name).to(dev).eval()
        output = model.generate(protein, config)

    result: str = output.sequence

    if "_" in result:
        warnings.warn(
            f"ESM3 generation left {result.count('_')} unresolved mask(s). "
            "Consider increasing num_steps.",
            UserWarning,
            stacklevel=2,
        )

    return result


# ---------------------------------------------------------------------------
# ESM C — embeddings
# ---------------------------------------------------------------------------

_ESMC_LOCAL_MODELS = ("esmc_300m", "esmc_600m")


def embed_esmc(
    sequences: list[str],
    model_name: str = "esmc_300m",
    device: str = "auto",
    forge_token: str | None = None,
    forge_url: str = "https://forge.evolutionaryscale.ai",
) -> list["np.ndarray"]:
    """
    Extract per-residue embeddings from ESM Cambrian (ESM C).

    ESM C is EvolutionaryScale's representation-focused family — not generative.
    It is designed as a high-performance replacement for ESM2 for embedding tasks.
    Requires the official esm package (`pip install esm`), same as ESM3.

    Args:
        sequences:   List of amino-acid sequences.
        model_name:  "esmc_300m" (300M) or "esmc_600m" (600M) for local inference.
                     Pass a Forge model name (e.g. "esmc-6b-2024-12") together with
                     forge_token for remote 6B inference.
        device:      "auto" | "cuda" | "mps" | "cpu". Ignored when using Forge API.
        forge_token: Forge API token. When None and model_name is not a local model,
                     reads ESM_API_KEY env var.
        forge_url:   Forge API endpoint URL.

    Returns:
        List of np.ndarray, one per input sequence, each of shape (L, d_model)
        where L is the sequence length. No BOS/EOS tokens in the output.
        For a single (d_model,) vector per sequence, call .mean(axis=0) on each array.

    Raises:
        ImportError: if the esm package is not installed.
        ValueError:  if Forge API is required but no token is available.
    """
    import os
    from esm.sdk.api import ESMProtein, LogitsConfig

    use_forge = forge_token is not None or model_name not in _ESMC_LOCAL_MODELS

    if use_forge:
        token = forge_token or os.environ.get("ESM_API_KEY")
        if not token:
            raise ValueError(
                "A Forge API token is required for non-local ESM C models. "
                "Set the ESM_API_KEY environment variable or pass forge_token."
            )
        from esm.sdk.forge import ESM3ForgeInferenceClient
        client = ESM3ForgeInferenceClient(model=model_name, url=forge_url, token=token)
    else:
        from esm.models.esmc import ESMC
        dev = get_device(device)
        client = ESMC.from_pretrained(model_name).to(dev).eval()

    result = []
    for seq in sequences:
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        emb = logits_output.embeddings.detach().cpu().float().numpy()  # (L, d_model)
        result.append(emb)

    return result


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_mutation(mut_str: str) -> tuple[int, str, str]:
    """Parse standard mutation notation 'A123G' → (123, 'G', 'A') = (pos, mut_aa, wt_aa)."""
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut_str.strip())
    if not m:
        raise ValueError(
            f"Cannot parse mutation '{mut_str}'. Expected format: 'A123G' "
            "(wildtype AA, 1-indexed position, mutant AA)."
        )
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return pos, mut_aa, wt_aa


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

    if args.command == "embed-esm2-hf":
        seqs = _load_sequences(args)
        emb = embed_esm2_hf(
            seqs,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            pool=args.pool,
        )
        np.save(args.output, emb)
        print(f"Saved embeddings {emb.shape} → {args.output}")

    elif args.command == "embed-esm2-native":
        seqs = _load_sequences(args)
        emb = embed_esm2_native(
            seqs,
            model_name=args.model,
            repr_layer=args.repr_layer,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.save(args.output, emb)
        print(f"Saved embeddings {emb.shape} → {args.output}")

    elif args.command == "score-variants":
        mutations: list[tuple[int, str, str]] = []
        if args.mutation:
            mutations = [_parse_mutation(args.mutation)]
        elif args.mutations:
            import pandas as pd
            df = pd.read_csv(args.mutations)
            mutations = list(zip(df["mut_pos"], df["mut_aa"], df["wt_aa"]))
        else:
            raise ValueError("Provide --mutation (single, e.g. A123G) or --mutations (CSV file).")

        model_indices = [int(x) for x in args.models.split(",")]
        scores = score_variants_esm1v(
            args.sequence,
            mutations,
            model_indices=model_indices,
            device=args.device,
        )
        import pandas as pd
        out_df = pd.DataFrame(mutations, columns=["mut_pos", "mut_aa", "wt_aa"])
        out_df["score"] = scores
        out_df.to_csv(args.output, index=False)
        print(f"Saved {len(scores)} variant score(s) → {args.output}")

    elif args.command == "fold":
        seqs = _load_sequences(args)
        ids: list[str]
        if args.ids:
            ids = [s.strip() for s in Path(args.ids).read_text().splitlines() if s.strip()]
        else:
            ids = [f"seq_{i}" for i in range(len(seqs))]

        pdb_strings = fold_sequences_esmfold(
            seqs, chunk_size=args.chunk_size, device=args.device
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for seq_id, pdb_str in zip(ids, pdb_strings):
            out_path = out_dir / f"{seq_id}.pdb"
            out_path.write_text(pdb_str)
            print(f"Wrote {out_path}")

    elif args.command == "embed-esm3":
        seqs = _load_sequences(args)
        embeddings = embed_esm3(seqs, model_name=args.model, device=args.device)
        save_dict = {f"seq_{i}": emb for i, emb in enumerate(embeddings)}
        np.savez(args.output, **save_dict)
        print(f"Saved {len(embeddings)} variable-length embedding(s) → {args.output}")

    elif args.command == "generate-esm3":
        result = generate_esm3(
            args.sequence,
            model_name=args.model,
            num_steps=args.num_steps,
            device=args.device,
            forge_token=args.forge_token,
            forge_url=args.forge_url,
        )
        if args.output:
            Path(args.output).write_text(result)
            print(f"Wrote completed sequence → {args.output}")
        else:
            print(result)

    elif args.command == "embed-esmc":
        seqs = _load_sequences(args)
        embeddings = embed_esmc(
            seqs,
            model_name=args.model,
            device=args.device,
            forge_token=args.forge_token,
            forge_url=args.forge_url,
        )
        save_dict = {f"seq_{i}": emb for i, emb in enumerate(embeddings)}
        np.savez(args.output, **save_dict)
        print(f"Saved {len(embeddings)} variable-length embedding(s) → {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ESM family inference — embeddings, scoring, folding, generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- embed-esm2-hf ----
    p = subparsers.add_parser("embed-esm2-hf", help="ESM2 HuggingFace embeddings (recommended)")
    _add_sequence_args(p)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1022)
    p.add_argument("--pool", choices=["mean", "bos"], default="mean")
    p.add_argument("--output", default="embeddings.npy")
    p.add_argument("--device", default="auto")

    # ---- embed-esm2-native ----
    p = subparsers.add_parser("embed-esm2-native", help="ESM2 native fair-esm embeddings")
    _add_sequence_args(p)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--repr-layer", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="embeddings.npy")
    p.add_argument("--device", default="auto")

    # ---- score-variants ----
    p = subparsers.add_parser("score-variants", help="ESM-1v zero-shot variant scoring")
    p.add_argument("--sequence", required=True, help="Wild-type amino-acid sequence.")
    mut_group = p.add_mutually_exclusive_group(required=True)
    mut_group.add_argument(
        "--mutation", help="Single mutation in standard notation, e.g. A123G."
    )
    mut_group.add_argument(
        "--mutations",
        help="CSV file with columns: mut_pos (int, 1-indexed), mut_aa, wt_aa.",
    )
    p.add_argument(
        "--models",
        default="1,2,3,4,5",
        help="Comma-separated ESM-1v ensemble indices (1–5). Default: all 5.",
    )
    p.add_argument("--output", default="scores.csv")
    p.add_argument("--device", default="auto")

    # ---- fold ----
    p = subparsers.add_parser("fold", help="ESMFold structure prediction → PDB files")
    _add_sequence_args(p)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Axial attention chunk size. Recommended 64 for sequences >400 aa.")
    p.add_argument("--output-dir", default="pdb_outputs",
                   help="Directory for output .pdb files.")
    p.add_argument("--ids", help="Text file with sequence IDs (used as PDB filenames).")
    p.add_argument("--device", default="auto")

    # ---- embed-esm3 ----
    p = subparsers.add_parser("embed-esm3", help="ESM3 sequence-track embeddings")
    _add_sequence_args(p)
    p.add_argument("--model", default="esm3-sm-open-v1")
    p.add_argument("--output", default="esm3_embeddings.npz",
                   help="Output .npz file (variable-length arrays, keyed seq_0, seq_1, ...).")
    p.add_argument("--device", default="auto")

    # ---- generate-esm3 ----
    p = subparsers.add_parser("generate-esm3", help="ESM3 masked sequence generation")
    p.add_argument("--sequence", required=True,
                   help="Sequence with '_' at positions to fill, e.g. MKTAY____ISFVK.")
    p.add_argument("--model", default="esm3-sm-open-v1")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--forge-token", default=None,
                   help="Forge API token. Reads ESM_API_KEY env var if not set.")
    p.add_argument("--forge-url", default="https://forge.evolutionaryscale.ai")
    p.add_argument("--output", default=None,
                   help="Write completed sequence to file. Default: print to stdout.")
    p.add_argument("--device", default="auto")

    # ---- embed-esmc ----
    p = subparsers.add_parser("embed-esmc", help="ESM Cambrian (ESM C) per-residue embeddings")
    _add_sequence_args(p)
    p.add_argument("--model", default="esmc_300m",
                   help="Local: esmc_300m, esmc_600m. Forge: esmc-6b-2024-12.")
    p.add_argument("--output", default="esmc_embeddings.npz",
                   help="Output .npz file (variable-length arrays, keyed seq_0, seq_1, ...).")
    p.add_argument("--forge-token", default=None,
                   help="Forge API token. Reads ESM_API_KEY env var if not set.")
    p.add_argument("--forge-url", default="https://forge.evolutionaryscale.ai")
    p.add_argument("--device", default="auto")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
