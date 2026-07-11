"""
Protein language model embedding backends.

Requirements: torch ≥2.0, fair-esm (pip install fair-esm)
GPU/MPS used automatically when available.

All backends return mean-pooled embeddings of shape (n_seqs, hidden_dim).
"""
import numpy as np
from typing import Iterable

# Registry: short name → (HuggingFace hub name, repr_layer)
_ESM_REGISTRY: dict[str, tuple[str, int]] = {
    "esm2_t6":  ("facebook/esm2_t6_8M_UR50D",    6),
    "esm2_t12": ("facebook/esm2_t12_35M_UR50D",  12),
    "esm2_t30": ("facebook/esm2_t30_150M_UR50D", 30),
    "esm2_t33": ("facebook/esm2_t33_650M_UR50D", 33),
    "esm2_t36": ("facebook/esm2_t36_3B_UR50D",   36),
}


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_esm_embeddings(
    sequences: Iterable[str],
    model_name: str = "esm2_t6",
    layer: int | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute mean-pooled ESM2 embeddings.

    Args:
        sequences:   Iterable of protein sequences (standard AA, no gaps).
        model_name:  One of esm2_t6 | esm2_t12 | esm2_t30 | esm2_t33 | esm2_t36.
        layer:       Representation layer index. None → last layer for the model.
        batch_size:  Sequences per forward pass. Reduce if GPU OOM.

    Returns:
        np.ndarray of shape (n_seqs, hidden_dim), float32.
    """
    import torch
    import esm

    if model_name not in _ESM_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_ESM_REGISTRY)}"
        )
    hub_name, default_layer = _ESM_REGISTRY[model_name]
    repr_layer = layer if layer is not None else default_layer

    device = get_device()
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(hub_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    seqs = list(sequences)
    embeddings: list[np.ndarray] = []

    for start in range(0, len(seqs), batch_size):
        batch_seqs = seqs[start : start + batch_size]
        batch_data = [(f"seq_{i}", s) for i, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # (B, L+2, D)
        for i, (_, seq) in enumerate(batch_data):
            # Mean-pool over sequence positions, excluding BOS/EOS tokens
            emb = reps[i, 1 : len(seq) + 1].mean(0).cpu().numpy()
            embeddings.append(emb)

    return np.stack(embeddings, axis=0).astype(np.float32)
