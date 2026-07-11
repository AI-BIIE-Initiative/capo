"""
MMseqs2 sequence-identity clustering → homology-safe train/val/test splits.

Use when a protein dataset lacks a split column, or splits are random row-shuffles.
Reads FASTA / CSV / Parquet, runs `mmseqs cluster` or `mmseqs linclust`, assigns
entire clusters to one split, verifies leakage, writes a fixed output contract.

CLI:
    python -m scripts.run_mmseqs2 --input data/sequences.fasta \
        --id-col id --seq-col sequence \
        --algorithm auto --min-seq-id 0.3 -c 0.8 --cov-mode 0 \
        --split-ratios 0.8 0.1 0.1 --seed 42 \
        --output-dir outputs/clustering/mmseqs2/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_PRIMARY = "#1E5994"
_REF     = "#9B3208"
_NOISE   = "#AAAAAA"

_INSTALL_HINT = (
    "MMseqs2 clustering was requested but the `mmseqs` binary was not found on "
    "PATH. CAPO does NOT silently fall back to an approximate method.\n"
    "Install MMseqs2:\n"
    "  Linux (static):  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz "
    "&& tar xzf mmseqs-linux-avx2.tar.gz && export PATH=$PWD/mmseqs/bin:$PATH\n"
    "  macOS:  brew install mmseqs2\n"
    "  conda:  conda install -c bioconda mmseqs2\n"
    "Or explicitly allow the approximate fallback with "
    "--on-missing-binary fallback (NOT a substitute for true homology "
    "clustering — see the warning it prints)."
)


# ----------------------------------------------------------------------------- config

@dataclass
class MMseqs2Config:
    input: Path
    output_dir: Path
    id_col: str = "id"
    seq_col: str = "sequence"
    algorithm: Literal["auto", "cluster", "linclust"] = "auto"
    min_seq_id: float = 0.3
    coverage: float = 0.8
    cov_mode: int = 0
    split_ratios: tuple = (0.8, 0.1, 0.1)
    seed: int = 42
    keep_tmp: bool = False
    auto_threshold_n: int = 20_000
    # What to do when the mmseqs binary is missing:
    #   "error"    → fail loudly with install instructions (default; never silent)
    #   "fallback" → use the approximate k-mer Jaccard clusterer (opt-in only)
    on_missing_binary: Literal["error", "fallback"] = "error"
    # k-mer size for the fallback clusterer.
    fallback_kmer: int = 3

    def __post_init__(self):
        self.input = Path(self.input)
        self.output_dir = Path(self.output_dir)
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}")
        if not 0 < self.min_seq_id <= 1:
            raise ValueError(f"min_seq_id must be in (0, 1], got {self.min_seq_id}")
        if self.algorithm not in {"auto", "cluster", "linclust"}:
            raise ValueError(f"algorithm must be auto|cluster|linclust, got {self.algorithm}")
        if self.on_missing_binary not in {"error", "fallback"}:
            raise ValueError(
                f"on_missing_binary must be error|fallback, got {self.on_missing_binary}"
            )


# ----------------------------------------------------------------------------- preflight

def mmseqs_available() -> bool:
    """True iff the `mmseqs` binary is on PATH."""
    return shutil.which("mmseqs") is not None


def preflight(cfg: "MMseqs2Config | None" = None) -> str | None:
    """Verify the mmseqs binary; honour the configured missing-binary policy.

    Returns the mmseqs version string when present. When the binary is missing:
    - on_missing_binary == "error" (default): raise RuntimeError with install
      instructions — NEVER a silent fallback.
    - on_missing_binary == "fallback": return None (caller uses the approximate
      clusterer) after printing a loud warning.
    """
    policy = getattr(cfg, "on_missing_binary", "error")
    if not mmseqs_available():
        if policy == "fallback":
            print(
                "[mmseqs2] WARNING: mmseqs not found — using the APPROXIMATE "
                "k-mer Jaccard fallback (--on-missing-binary fallback). This is "
                "NOT a substitute for true MMseqs2 homology clustering; install "
                "MMseqs2 for production splits.",
                file=sys.stderr,
            )
            return None
        raise RuntimeError(_INSTALL_HINT)
    res = subprocess.run(
        ["mmseqs", "version"], check=True, text=True, capture_output=True
    )
    return res.stdout.strip() or res.stderr.strip()


# ----------------------------------------------------------------------------- fallback clusterer

def _read_fasta(fasta: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    rid, seq = None, []
    with fasta.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if rid is not None:
                    records.append((rid, "".join(seq)))
                rid, seq = line[1:].strip(), []
            elif line:
                seq.append(line)
    if rid is not None:
        records.append((rid, "".join(seq)))
    return records


def _kmer_set(seq: str, k: int) -> frozenset[str]:
    s = seq.upper()
    if len(s) < k:
        return frozenset({s}) if s else frozenset()
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


def fallback_cluster(fasta: Path, cfg: MMseqs2Config, id_col: str) -> pd.DataFrame:
    """Approximate, dependency-free homology clustering (greedy k-mer Jaccard).

    Greedily assigns each sequence (longest first) to the first existing cluster
    whose representative shares Jaccard(k-mer sets) >= min_seq_id, else opens a
    new cluster. O(n · n_clusters) — fine as an opt-in fallback, not for huge n.
    Returns a frame with columns [id_col, cluster, representative].
    """
    records = _read_fasta(fasta)
    # Longest-first so the representative is the most informative sequence.
    order = sorted(range(len(records)), key=lambda i: -len(records[i][1]))
    reps: list[tuple[str, frozenset[str]]] = []  # (rep_id, rep_kmers)
    assignments: dict[str, int] = {}
    threshold = cfg.min_seq_id
    k = cfg.fallback_kmer

    for i in order:
        rid, seq = records[i]
        kset = _kmer_set(seq, k)
        best_c = -1
        for c, (_, rep_kmers) in enumerate(reps):
            union = len(kset | rep_kmers)
            jacc = (len(kset & rep_kmers) / union) if union else 0.0
            if jacc >= threshold:
                best_c = c
                break
        if best_c < 0:
            best_c = len(reps)
            reps.append((rid, kset))
        assignments[rid] = best_c

    rep_ids = [reps[c][0] for c in range(len(reps))]
    rows = [
        {id_col: rid, "cluster": c, "representative": rep_ids[c]}
        for rid, c in assignments.items()
    ]
    return pd.DataFrame(rows, columns=[id_col, "cluster", "representative"])


# ----------------------------------------------------------------------------- input prep

def prepare_fasta(cfg: MMseqs2Config, workdir: Path) -> tuple[Path, int]:
    """Return (fasta_path, n_sequences). Writes a temp FASTA if input is tabular."""
    suffix = cfg.input.suffix.lower()
    if suffix in {".fasta", ".fa", ".faa", ".fna"}:
        n = _count_fasta_records(cfg.input)
        return cfg.input, n

    if suffix == ".csv":
        df = pd.read_csv(cfg.input)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(cfg.input)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(cfg.input, sep="\t")
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    for col in (cfg.id_col, cfg.seq_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in {cfg.input}; have {list(df.columns)}")

    if df[cfg.id_col].duplicated().any():
        dups = df[cfg.id_col][df[cfg.id_col].duplicated()].head(5).tolist()
        raise ValueError(f"{cfg.id_col!r} has duplicate values, e.g. {dups}")

    fasta_path = workdir / "input.fasta"
    with fasta_path.open("w") as fh:
        for _id, _seq in zip(df[cfg.id_col].astype(str), df[cfg.seq_col].astype(str)):
            fh.write(f">{_id}\n{_seq}\n")
    return fasta_path, len(df)


def _count_fasta_records(path: Path) -> int:
    n = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


# ----------------------------------------------------------------------------- mmseqs

def pick_algorithm(n_seqs: int, cfg: MMseqs2Config) -> str:
    if cfg.algorithm != "auto":
        return cfg.algorithm
    chosen = "cluster" if n_seqs < cfg.auto_threshold_n else "linclust"
    print(f"[mmseqs2] auto-select: {chosen} (n_seqs={n_seqs}, threshold={cfg.auto_threshold_n})")
    return chosen


def run_mmseqs(fasta: Path, algo: str, cfg: MMseqs2Config, workdir: Path, log_path: Path) -> Path:
    """Run createdb → cluster/linclust → createtsv. Return TSV path."""
    db        = workdir / "DB"
    clu       = workdir / "DB_clu"
    tmp       = workdir / "tmp"
    tsv_path  = workdir / "DB_clu.tsv"
    tmp.mkdir(exist_ok=True)

    cmds = [
        ["mmseqs", "createdb", str(fasta), str(db)],
        [
            "mmseqs", algo, str(db), str(clu), str(tmp),
            "--min-seq-id", f"{cfg.min_seq_id}",
            "-c", f"{cfg.coverage}",
            "--cov-mode", f"{cfg.cov_mode}",
        ],
        ["mmseqs", "createtsv", str(db), str(db), str(clu), str(tsv_path)],
    ]

    with log_path.open("a") as log:
        for cmd in cmds:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            res = subprocess.run(cmd, text=True, capture_output=True)
            log.write(res.stdout)
            if res.stderr:
                log.write("\n--- stderr ---\n" + res.stderr)
            log.flush()
            if res.returncode != 0:
                raise RuntimeError(
                    f"mmseqs failed (exit {res.returncode}): {' '.join(cmd)}\n"
                    f"See {log_path} for details."
                )
    return tsv_path


# ----------------------------------------------------------------------------- parsing + splits

def parse_tsv(tsv_path: Path, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["representative", id_col])
    codes, _ = pd.factorize(df["representative"], sort=True)
    df["cluster"] = codes
    return df[[id_col, "cluster", "representative"]]


def assign_splits(df: pd.DataFrame, ratios: tuple, seed: int) -> pd.DataFrame:
    """Assign each cluster to train, val, or test. Entire clusters stay together."""
    rng = np.random.default_rng(seed)
    clusters = sorted(df["cluster"].unique())
    n = len(clusters)
    n_train = round(ratios[0] * n)
    n_val   = round(ratios[1] * n)
    perm    = rng.permutation(n)
    arr     = np.array(clusters)
    train_c = set(arr[perm[:n_train]].tolist())
    val_c   = set(arr[perm[n_train:n_train + n_val]].tolist())

    def _split(c: int) -> str:
        if c in train_c: return "train"
        if c in val_c:   return "val"
        return "test"

    df = df.copy()
    df["split"] = df["cluster"].map(_split)
    return df


def check_leakage(df: pd.DataFrame, id_col: str) -> dict:
    seq_splits   = df.groupby(id_col)["split"].nunique()
    seq_leakage  = int((seq_splits > 1).sum())
    cluster_splits = df.groupby("cluster")["split"].nunique()
    cluster_leakage = int((cluster_splits > 1).sum())
    return {
        "sequences_in_multiple_splits": seq_leakage,
        "clusters_spanning_splits":     cluster_leakage,
        "split_counts":                 df["split"].value_counts().to_dict(),
    }


def cluster_stats(df: pd.DataFrame) -> dict:
    sizes = df.groupby("cluster").size()
    n_total = int(sizes.sum())
    return {
        "n_clusters":            int(sizes.shape[0]),
        "n_singletons":          int((sizes == 1).sum()),
        "n_sequences":           n_total,
        "largest_cluster_frac":  float(sizes.max() / n_total),
        "mean_size":             float(sizes.mean()),
        "median_size":           float(sizes.median()),
        "p99_size":              float(sizes.quantile(0.99)),
        "max_size":              int(sizes.max()),
    }


# ----------------------------------------------------------------------------- plot

def plot_size_distribution(sizes: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(sizes.values, bins=min(50, max(10, sizes.shape[0] // 5 or 10)),
            color=_PRIMARY, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.axvline(1, color=_REF, linestyle="--", linewidth=1.2, label="singleton threshold")
    ax.set_xlabel("Cluster size (# sequences)", color="black")
    ax.set_ylabel("Number of clusters (log)", color="black")
    ax.set_title("MMseqs2 cluster size distribution", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(frameon=False, labelcolor="black")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- save

def save_outputs(
    df: pd.DataFrame,
    leakage: dict,
    stats: dict,
    cfg: MMseqs2Config,
    mmseqs_version: str,
    chosen_algorithm: str,
    id_col: str,
) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df[[id_col, "cluster", "representative", "split"]].to_csv(
        out / "cluster_assignments.csv", index=False
    )
    df[[id_col, "split"]].to_csv(out / "split_assignments.csv", index=False)

    with (out / "leakage_report.json").open("w") as fh:
        json.dump(leakage, fh, indent=2)
    with (out / "cluster_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)

    cfg_dict = asdict(cfg)
    cfg_dict["input"] = str(cfg.input)
    cfg_dict["output_dir"] = str(cfg.output_dir)
    cfg_dict["split_ratios"] = list(cfg.split_ratios)
    cfg_dict["mmseqs_version"] = mmseqs_version
    cfg_dict["chosen_algorithm"] = chosen_algorithm
    with (out / "run_config.yaml").open("w") as fh:
        yaml.safe_dump(cfg_dict, fh, sort_keys=False)

    sizes = df.groupby("cluster").size()
    plot_size_distribution(sizes, out / "plots" / "cluster_size_distribution")


# ----------------------------------------------------------------------------- main

def run(cfg: MMseqs2Config) -> int:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "mmseqs.log"
    if log_path.exists():
        log_path.unlink()

    print(f"[mmseqs2] preflight…")
    mmseqs_version = preflight(cfg)
    use_fallback = mmseqs_version is None
    print(
        f"[mmseqs2] mmseqs version: {mmseqs_version}"
        if not use_fallback
        else "[mmseqs2] using approximate k-mer Jaccard fallback (no mmseqs binary)"
    )

    with tempfile.TemporaryDirectory(prefix="mmseqs2_") as tmpd:
        workdir = Path(tmpd)
        fasta, n_seqs = prepare_fasta(cfg, workdir)
        print(f"[mmseqs2] input: {fasta}  n_seqs={n_seqs}")

        if use_fallback:
            algo = "fallback_kmer_jaccard"
            df = fallback_cluster(fasta, cfg, cfg.id_col)
            mmseqs_version = "fallback (mmseqs not installed)"
        else:
            algo = pick_algorithm(n_seqs, cfg)
            tsv_path = run_mmseqs(fasta, algo, cfg, workdir, log_path)
            df = parse_tsv(tsv_path, cfg.id_col)
        print(f"[mmseqs2] parsed {len(df)} rows via {algo}")

        df = assign_splits(df, cfg.split_ratios, cfg.seed)
        leakage = check_leakage(df, cfg.id_col)
        stats = cluster_stats(df)
        save_outputs(df, leakage, stats, cfg, mmseqs_version, algo, cfg.id_col)

        if cfg.keep_tmp:
            kept = cfg.output_dir / "mmseqs_workdir"
            shutil.copytree(workdir, kept, dirs_exist_ok=True)
            print(f"[mmseqs2] kept workdir at {kept}")

    print(f"[mmseqs2] stats:   {stats}")
    print(f"[mmseqs2] leakage: {leakage}")
    print(f"[mmseqs2] outputs in {cfg.output_dir}")

    if leakage["clusters_spanning_splits"] > 0 or leakage["sequences_in_multiple_splits"] > 0:
        print("[mmseqs2] ERROR: leakage detected — see leakage_report.json", file=sys.stderr)
        return 2
    return 0


def _parse_args(argv: list[str] | None = None) -> MMseqs2Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",       required=True, type=Path)
    p.add_argument("--output-dir",  required=True, type=Path)
    p.add_argument("--id-col",      default="id")
    p.add_argument("--seq-col",     default="sequence")
    p.add_argument("--algorithm",   default="auto", choices=["auto", "cluster", "linclust"])
    p.add_argument("--min-seq-id",  type=float, default=0.3)
    p.add_argument("-c", "--coverage", type=float, default=0.8)
    p.add_argument("--cov-mode",    type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--split-ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--keep-tmp",    action="store_true")
    p.add_argument("--auto-threshold-n", type=int, default=20_000)
    p.add_argument(
        "--on-missing-binary", default="error", choices=["error", "fallback"],
        help="error (default): fail loudly if mmseqs is missing; "
             "fallback: use the approximate k-mer Jaccard clusterer instead.",
    )
    p.add_argument(
        "--allow-fallback", action="store_true",
        help="Shorthand for --on-missing-binary fallback.",
    )
    p.add_argument("--fallback-kmer", type=int, default=3)
    args = p.parse_args(argv)
    on_missing = "fallback" if args.allow_fallback else args.on_missing_binary
    return MMseqs2Config(
        input            = args.input,
        output_dir       = args.output_dir,
        id_col           = args.id_col,
        seq_col          = args.seq_col,
        algorithm        = args.algorithm,
        min_seq_id       = args.min_seq_id,
        coverage         = args.coverage,
        cov_mode         = args.cov_mode,
        split_ratios     = tuple(args.split_ratios),
        seed             = args.seed,
        keep_tmp         = args.keep_tmp,
        auto_threshold_n = args.auto_threshold_n,
        on_missing_binary = on_missing,
        fallback_kmer    = args.fallback_kmer,
    )


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
