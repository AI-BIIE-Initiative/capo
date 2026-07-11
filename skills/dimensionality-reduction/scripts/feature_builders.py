"""
Sequence feature builders: one-hot (dense/sparse) and k-mer count matrices.

For sparse one-hot or k-mer matrices, always follow with TruncatedSVD — not PCA.
PCA would densify the sparse matrix, which is prohibitively expensive for large datasets.
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Iterable

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(STANDARD_AA)}


def build_onehot_dense(sequences: Iterable[str], max_len: int | None = None) -> np.ndarray:
    """
    Dense one-hot encoding. Shape: (n_seqs, max_len * 20).
    Use only when dataset is small enough for dense storage.
    Positions with non-standard AAs are left as all-zeros.
    """
    seqs = list(sequences)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    n_vocab = len(STANDARD_AA)
    X = np.zeros((len(seqs), max_len * n_vocab), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]):
            idx = AA_INDEX.get(aa.upper())
            if idx is not None:
                X[i, j * n_vocab + idx] = 1.0
    return X


def build_onehot_sparse(sequences: Iterable[str], max_len: int | None = None) -> sp.csr_matrix:
    """
    Sparse one-hot encoding. Shape: (n_seqs, max_len * 20).
    Memory-efficient for large datasets. Always prefer this over dense.
    Follow with TruncatedSVD, not PCA.
    """
    seqs = list(sequences)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    n_vocab = len(STANDARD_AA)
    rows, cols, data = [], [], []
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq[:max_len]):
            idx = AA_INDEX.get(aa.upper())
            if idx is not None:
                rows.append(i)
                cols.append(j * n_vocab + idx)
                data.append(1.0)
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(seqs), max_len * n_vocab),
        dtype=np.float32,
    )


def build_kmer_matrix(sequences: Iterable[str], k: int = 3) -> sp.csr_matrix:
    """
    Sparse k-mer count matrix, L2-normalized per row.
    Only counts k-mers composed entirely of standard AAs.
    Shape: (n_seqs, n_distinct_kmers_in_vocab).
    Follow with TruncatedSVD for dimensionality reduction.
    """
    from sklearn.preprocessing import normalize

    seqs = list(sequences)
    vocab: dict[str, int] = {}
    kmer_lists: list[list[str]] = []

    for seq in seqs:
        seq = seq.upper()
        kmers = [
            seq[i : i + k]
            for i in range(len(seq) - k + 1)
            if all(c in STANDARD_AA for c in seq[i : i + k])
        ]
        kmer_lists.append(kmers)
        for km in kmers:
            if km not in vocab:
                vocab[km] = len(vocab)

    rows, cols, data = [], [], []
    for i, kmers in enumerate(kmer_lists):
        counts: dict[int, int] = defaultdict(int)
        for km in kmers:
            counts[vocab[km]] += 1
        for col, cnt in counts.items():
            rows.append(i)
            cols.append(col)
            data.append(float(cnt))

    X = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(seqs), len(vocab)),
        dtype=np.float32,
    )
    return normalize(X, norm="l2")  # L2-normalize for cosine-friendly distances
