# MMseqs2 parameter choices

Quick reference for tuning when the defaults don't fit the task.

## `--min-seq-id` — sequence identity threshold

| Value | What it gives you | When to pick it |
|---|---|---|
| `0.2` | Very loose — groups distant homologs (super-families) | Pretraining splits; you want maximum separation |
| **`0.3` (default)** | Remote homologs grouped together | **Default for split generation** — matches UniRef30 / PSI-BLAST homology threshold |
| `0.5` | Moderate — strain-level / sub-family | When biology is closely related and 0.3 over-clusters |
| `0.7` | Near-duplicate dedup | Cleaning a noisy dataset; not for splits |
| `0.9` | Exact dedup | Removing literal duplicates |

Lower threshold → fewer, larger clusters → more aggressive homology removal across splits → harder, more honest evaluation.

## `-c` / `--cov-mode` — alignment coverage

`-c` is the minimum fraction of the alignment that must be covered. Default `0.8`.

| `--cov-mode` | Meaning |
|---|---|
| **`0` (default)** | Coverage of *both* query and target — strictest, prevents fragment matches |
| `1` | Coverage of query only — use when target lengths vary (e.g. searching a long DB) |
| `2` | Coverage of target only — symmetric counterpart of `1` |
| `3` | Target length must be ≥ `-c` × query length — useful with short query peptides |

For homology-safe splits on a single dataset, stick with `--cov-mode 0` and `-c 0.8`.

## `cluster` vs `linclust` runtime

Based on the MMseqs2 user guide. Wall-clock on a laptop:

| N sequences | `cluster` (sensitive, O(n¹·⁴)) | `linclust` (linear) |
|---|---|---|
| 10k  | ~30 s | ~10 s |
| 100k | ~5 min | ~30 s |
| 1M   | ~30 min | ~5 min |
| 10M  | hours | ~30 min |

`cluster` finds more remote homology pairs that `linclust` misses. At small N the extra runtime is negligible; at large N it dominates. The skill's `auto` algorithm switches at N=20k by default — override with `--auto-threshold-n` or force one with `--algorithm cluster|linclust`.

## Reproducibility

The script writes the exact `mmseqs version` string into `run_config.yaml`. MMseqs2 results are deterministic for a given binary + parameter combination, so re-running on the same input + same version + same `--seed` reproduces the splits exactly.
