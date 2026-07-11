"""
chai_inference.py — Chai-1 structure prediction inference helper.

Provides:
  - HuggingFace weight download (chaidiscovery/chai-1)
  - FASTA input builders (protein, ligand, DNA, RNA chains)
  - run_inference wrapper with sensible defaults
  - Output parsers for scores.model_idx_N.npz files

Installation:
  pip install chai_lab==0.6.1 huggingface_hub

Weights download automatically from chaidiscovery/chai-1 on HuggingFace on first run.
Set CHAI_DOWNLOADS_DIR before importing chai_lab to control weight storage location.

CLI subcommands:
  download-weights   Pre-fetch Chai-1 weights from HuggingFace
  fold               Run inference on a FASTA file
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HuggingFace weight management
# ---------------------------------------------------------------------------

CHAI1_HF_REPO = "chaidiscovery/chai-1"


def download_chai_weights(local_dir: str | Path | None = None) -> Path:
    """
    Download Chai-1 weights from HuggingFace Hub.

    Must be called (or CHAI_DOWNLOADS_DIR set) before importing chai_lab.
    Downloads once; subsequent calls return the cached path.

    Args:
        local_dir: Directory to store weights.
                   Defaults to CHAI_DOWNLOADS_DIR env var, then ~/.chai/weights.

    Returns:
        Path to the downloaded weights directory.

    Raises:
        ImportError:  if huggingface_hub is not installed.
        RuntimeError: if download fails.
    """
    from huggingface_hub import snapshot_download

    target = (
        Path(local_dir)
        if local_dir
        else Path(os.environ.get("CHAI_DOWNLOADS_DIR", str(Path.home() / ".chai" / "weights")))
    )
    target.mkdir(parents=True, exist_ok=True)

    repo_path = Path(
        snapshot_download(
            repo_id=CHAI1_HF_REPO,
            local_dir=str(target),
        )
    )
    return repo_path


# ---------------------------------------------------------------------------
# FASTA builders
# ---------------------------------------------------------------------------

def _fasta_entry(entity_type: str, name: str, sequence: str) -> str:
    """Format a single FASTA entry. entity_type must be protein, ligand, rna, or dna."""
    return f">{entity_type}|name={name}\n{sequence}"


def protein_entry(name: str, sequence: str) -> str:
    """
    Build a protein FASTA entry.

    Args:
        name:     Unique identifier for this chain.
        sequence: Amino-acid sequence (single-letter codes).
                  Modified residues: inline PTM codes, e.g. AAA(SEP)AAA for phosphoserine.

    Returns:
        FASTA entry string.
    """
    return _fasta_entry("protein", name, sequence)


def ligand_entry(name: str, smiles: str) -> str:
    """
    Build a small-molecule ligand FASTA entry.

    Args:
        name:   Unique identifier for this ligand.
        smiles: SMILES string (RDKit-parseable). One molecule per entry.

    Returns:
        FASTA entry string.
    """
    return _fasta_entry("ligand", name, smiles)


def rna_entry(name: str, sequence: str) -> str:
    """Build an RNA chain FASTA entry. sequence uses nucleotide bases (A/U/C/G)."""
    return _fasta_entry("rna", name, sequence)


def dna_entry(name: str, sequence: str) -> str:
    """Build a DNA chain FASTA entry. sequence uses nucleotide bases (A/T/C/G)."""
    return _fasta_entry("dna", name, sequence)


def build_fasta(entities: list[dict[str, str]]) -> str:
    """
    Build a complete FASTA string for a complex from a list of entity dicts.

    Args:
        entities: List of dicts with keys:
                    "type"     — "protein", "ligand", "rna", or "dna"
                    "name"     — unique chain identifier
                    "sequence" — sequence string or SMILES for ligands

    Returns:
        Multi-entry FASTA string ready to write to a .fasta file.

    Example:
        fasta = build_fasta([
            {"type": "protein", "name": "receptor", "sequence": "MKTAYIAK..."},
            {"type": "ligand",  "name": "drug",     "sequence": "CC(=O)Oc1ccccc1C(=O)O"},
        ])
    """
    entries = []
    for ent in entities:
        t = ent["type"]
        if t not in ("protein", "ligand", "rna", "dna"):
            raise ValueError(f"Unknown entity type '{t}'. Must be protein, ligand, rna, or dna.")
        entries.append(_fasta_entry(t, ent["name"], ent["sequence"]))
    return "\n".join(entries)


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def run(
    fasta_file: str | Path,
    output_dir: str | Path,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    device: str = "cuda:0",
    use_esm_embeddings: bool = True,
    msa_directory: str | Path | None = None,
    use_msa_server: bool = False,
    clear_output_dir: bool = True,
) -> Any:
    """
    Run Chai-1 structure prediction.

    Wraps chai_lab.chai1.run_inference with sensible defaults and automatic
    output directory management.

    Args:
        fasta_file:          Path to input .fasta file. See build_fasta() or SKILL.md
                             for the expected format.
        output_dir:          Directory for output CIF files and score .npz files.
                             Must be empty; use clear_output_dir=True (default) to
                             remove any existing contents automatically.
        num_trunk_recycles:  Refinement trunk iterations. Default 3; increase to 10
                             for higher-quality predictions at the cost of speed.
        num_diffn_timesteps: Diffusion steps. Default 200; reduce to 50–100 for speed.
        seed:                Random seed for reproducibility.
        device:              PyTorch device string ("cuda:0", "cuda:1", "cpu").
        use_esm_embeddings:  Use ESM embeddings as sequence representation. Default True.
                             Set False only if providing high-quality MSAs.
        msa_directory:       Directory containing precomputed MSA files in aligned.pqt
                             format. Mutually exclusive with use_msa_server.
        use_msa_server:      Auto-generate MSA via ColabFold MMseqs2 server. Mutually
                             exclusive with msa_directory.
        clear_output_dir:    Remove output_dir contents before running. Default True.
                             Set False if you manage the directory yourself.

    Returns:
        CandidateResult object from chai_lab with:
          .cif_paths       — list of Path objects (one per sample, ordered by score)
          .ranking_data    — list of RankingData objects with .aggregate_score

    Raises:
        ValueError:   if both msa_directory and use_msa_server are set.
        ImportError:  if chai_lab is not installed (pip install chai_lab).
        RuntimeError: if CUDA is unavailable and device="cuda:*".
    """
    if msa_directory is not None and use_msa_server:
        raise ValueError("msa_directory and use_msa_server are mutually exclusive.")

    from chai_lab.chai1 import run_inference

    output_dir = Path(output_dir)
    if clear_output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = dict(
        fasta_file=Path(fasta_file),
        output_dir=output_dir,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        seed=seed,
        device=device,
        use_esm_embeddings=use_esm_embeddings,
    )
    if msa_directory is not None:
        kwargs["msa_directory"] = Path(msa_directory)
    if use_msa_server:
        kwargs["use_msa_server"] = True

    return run_inference(**kwargs)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_scores(output_dir: str | Path, model_idx: int) -> dict[str, Any]:
    """
    Load per-sample confidence metrics from a Chai-1 scores .npz file.

    Args:
        output_dir: The output directory passed to run() or chai-lab fold.
        model_idx:  Sample index (0-based). Default run produces 5 samples (0–4).

    Returns:
        Dict with numpy arrays/scalars:
          ptm             — predicted TM-score [0,1], higher=better
          iptm            — interface TM-score [0,1], higher=better
          plddt           — per-residue confidence array [0,1], higher=better
          clash_score     — steric clash penalty, lower=better
          aggregate_score — primary ranking score (combination of above)

    Raises:
        FileNotFoundError: if the scores file does not exist.
    """
    import numpy as np

    scores_path = Path(output_dir) / f"scores.model_idx_{model_idx}.npz"
    data = np.load(scores_path)
    return dict(data)


def find_outputs(output_dir: str | Path) -> dict[str, list[Path]]:
    """
    Locate all output files in a Chai-1 predictions directory.

    Args:
        output_dir: The output directory passed to run() or chai-lab fold.

    Returns:
        Dict with keys:
          structures — list of CIF Path objects, sorted by model index
          scores     — list of .npz Path objects, sorted by model index
    """
    base = Path(output_dir)
    structures = sorted(base.glob("pred.model_idx_*.cif"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    scores = sorted(base.glob("scores.model_idx_*.npz"),
                    key=lambda p: int(p.stem.split("_")[-1]))
    return {"structures": structures, "scores": scores}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chai-1 inference helper — FASTA building, prediction, score parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- download-weights ----
    p = subparsers.add_parser(
        "download-weights",
        help="Pre-download Chai-1 weights from HuggingFace (chaidiscovery/chai-1).",
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help="Directory to store weights. Default: CHAI_DOWNLOADS_DIR or ~/.chai/weights.",
    )

    # ---- fold ----
    p = subparsers.add_parser(
        "fold",
        help="Run Chai-1 inference on a FASTA file.",
    )
    p.add_argument("fasta", help="Path to input .fasta file.")
    p.add_argument("--out-dir", default="./outputs", help="Output directory.")
    p.add_argument("--use-msa-server", action="store_true",
                   help="Auto-generate MSA via ColabFold MMseqs2 server.")
    p.add_argument("--msa-dir", default=None,
                   help="Directory of precomputed aligned.pqt MSA files.")
    p.add_argument("--trunk-recycles", type=int, default=3,
                   help="Trunk refinement iterations (default 3; 10 for higher quality).")
    p.add_argument("--diffn-timesteps", type=int, default=200,
                   help="Diffusion steps (default 200; 50-100 for speed).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0", help="PyTorch device (e.g. cuda:0, cpu).")
    p.add_argument("--no-esm", action="store_true",
                   help="Disable ESM embeddings. Only useful when providing MSAs.")
    p.add_argument("--keep-output-dir", action="store_true",
                   help="Do not clear the output directory before running.")

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "download-weights":
        path = download_chai_weights(local_dir=args.local_dir)
        print(f"Chai-1 weights downloaded to: {path}")

    elif args.command == "fold":
        candidates = run(
            fasta_file=args.fasta,
            output_dir=args.out_dir,
            num_trunk_recycles=args.trunk_recycles,
            num_diffn_timesteps=args.diffn_timesteps,
            seed=args.seed,
            device=args.device,
            use_esm_embeddings=not args.no_esm,
            msa_directory=args.msa_dir,
            use_msa_server=args.use_msa_server,
            clear_output_dir=not args.keep_output_dir,
        )
        outputs = find_outputs(args.out_dir)
        print(f"Predictions written to: {args.out_dir}")
        for cif, rd in zip(candidates.cif_paths, candidates.ranking_data):
            print(f"  {cif.name}  aggregate_score={rd.aggregate_score.item():.4f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
