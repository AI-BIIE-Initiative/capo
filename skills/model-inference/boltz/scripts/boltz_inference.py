"""
boltz_inference.py — Boltz-2 structure prediction and affinity inference helper.

Provides:
  - YAML input builders (protein, ligand, DNA/RNA chains; pocket/contact constraints)
  - HuggingFace weight download (boltz-community/boltz-2)
  - Subprocess wrapper for `boltz predict`
  - Output parsers for confidence and affinity JSON files

Installation:
  pip install boltz huggingface_hub pyyaml

Weights are downloaded from boltz-community/boltz-2 on HuggingFace.
Prefer this over cloning the git repo.

CLI subcommands:
  download-weights   Pre-fetch Boltz-2 weights from HuggingFace
  predict            Run boltz predict on a YAML file or directory
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HuggingFace weight management
# ---------------------------------------------------------------------------

BOLTZ2_HF_REPO = "boltz-community/boltz-2"
_CHECKPOINT_FILES = {
    "conf": "boltz2_conf.ckpt",    # structure prediction weights (~2.3 GB)
    "aff":  "boltz2_aff.ckpt",     # affinity prediction weights (~2.1 GB)
}


def download_boltz2_weights(local_dir: str | Path | None = None) -> dict[str, Path]:
    """
    Download Boltz-2 weights from HuggingFace Hub.

    Preferred over boltz's built-in cache download when you need explicit control
    over weight location or version pinning. Downloads once; subsequent calls
    return cached paths without re-downloading.

    Args:
        local_dir: Directory to store weights. Defaults to ~/.boltz/boltz2.
                   Set the BOLTZ_CACHE environment variable to override globally.

    Returns:
        Dict with keys "conf" and "aff" mapping to checkpoint Path objects.

    Raises:
        ImportError:  if huggingface_hub is not installed.
        RuntimeError: if download fails.
    """
    from huggingface_hub import snapshot_download

    target = Path(local_dir) if local_dir else Path.home() / ".boltz" / "boltz2"
    target.mkdir(parents=True, exist_ok=True)

    repo_path = Path(
        snapshot_download(
            repo_id=BOLTZ2_HF_REPO,
            local_dir=str(target),
        )
    )

    return {
        "conf": repo_path / _CHECKPOINT_FILES["conf"],
        "aff":  repo_path / _CHECKPOINT_FILES["aff"],
    }


# ---------------------------------------------------------------------------
# YAML input builders
# ---------------------------------------------------------------------------

def protein_chain(
    id: str | list[str],
    sequence: str,
    msa: str | None = None,
    modifications: list[dict] | None = None,
    cyclic: bool = False,
) -> dict:
    """
    Build a protein chain entry for the Boltz-2 YAML input.

    Args:
        id:            Chain ID string, or list of IDs for identical copies (e.g. ["A", "B"]).
        sequence:      Amino-acid sequence (single-letter codes).
        msa:           Path to a precomputed .a3m file, or "empty" for single-sequence mode
                       (not recommended — hurts accuracy). Omit when using --use_msa_server.
        modifications: List of modified residue dicts: [{"position": 5, "ccd": "MSE"}, ...].
                       Positions are 1-indexed.
        cyclic:        Whether the chain is cyclic.

    Returns:
        Dict suitable for inclusion in the "sequences" list of build_yaml().
    """
    entry: dict[str, Any] = {"id": id, "sequence": sequence}
    if msa is not None:
        entry["msa"] = msa
    if modifications:
        entry["modifications"] = modifications
    if cyclic:
        entry["cyclic"] = True
    return {"protein": entry}


def ligand_smiles(id: str | list[str], smiles: str) -> dict:
    """
    Build a small-molecule ligand entry using a SMILES string.

    Args:
        id:     Chain ID or list of IDs for identical copies.
        smiles: SMILES string (quoted; RDKit-parseable).

    Returns:
        Dict for the "sequences" list of build_yaml().
    """
    return {"ligand": {"id": id, "smiles": smiles}}


def ligand_ccd(id: str | list[str], ccd: str) -> dict:
    """
    Build a small-molecule ligand entry using a CCD code.

    Args:
        id:  Chain ID or list of IDs.
        ccd: CCD code (e.g. "ATP", "SAH"). Mutually exclusive with SMILES.

    Returns:
        Dict for the "sequences" list of build_yaml().
    """
    return {"ligand": {"id": id, "ccd": ccd}}


def dna_chain(id: str | list[str], sequence: str) -> dict:
    """Build a DNA chain entry. sequence is nucleotide bases (e.g. "ATCGATCG")."""
    return {"dna": {"id": id, "sequence": sequence}}


def rna_chain(id: str | list[str], sequence: str) -> dict:
    """Build an RNA chain entry. sequence is nucleotide bases (e.g. "AUCGAUCG")."""
    return {"rna": {"id": id, "sequence": sequence}}


# ---------------------------------------------------------------------------
# Constraint builders
# ---------------------------------------------------------------------------

def pocket_constraint(
    binder: str,
    contacts: list[list],
    max_distance: float = 6.0,
    force: bool = False,
) -> dict:
    """
    Specify residues forming the binding site for a given binder chain.

    Args:
        binder:       Chain ID of the binder (ligand, protein, DNA, or RNA).
        contacts:     List of [chain_id, residue_index] pairs (1-indexed).
                      For ligand chains, use atom names instead: [chain_id, atom_name].
        max_distance: Max distance in Angstroms between any binder atom and each contact.
                      Must be between 4 and 20; default 6.
        force:        If True, use a potential to enforce the constraint during diffusion.

    Returns:
        Dict for the "constraints" list of build_yaml().
    """
    return {
        "pocket": {
            "binder": binder,
            "contacts": contacts,
            "max_distance": max_distance,
            "force": force,
        }
    }


def contact_constraint(
    token1: list,
    token2: list,
    max_distance: float = 6.0,
    force: bool = False,
) -> dict:
    """
    Specify a distance constraint between two residues or atoms.

    Args:
        token1:       [chain_id, residue_index_or_atom_name] for the first residue/atom.
        token2:       [chain_id, residue_index_or_atom_name] for the second.
        max_distance: Max distance in Angstroms (4–20).
        force:        If True, enforce with a potential.
    """
    return {
        "contact": {
            "token1": token1,
            "token2": token2,
            "max_distance": max_distance,
            "force": force,
        }
    }


def bond_constraint(
    chain1: str, residue1: int, atom1: str,
    chain2: str, residue2: int, atom2: str,
) -> dict:
    """
    Specify a covalent bond between two atoms.

    Supported for CCD ligands and canonical residues only. Atom names can be
    verified in CIF files on RCSB. Residue indices are 1-indexed (ligands use 1).

    Returns:
        Dict for the "constraints" list of build_yaml().
    """
    return {
        "bond": {
            "atom1": [chain1, residue1, atom1],
            "atom2": [chain2, residue2, atom2],
        }
    }


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

def build_yaml(
    sequences: list[dict],
    constraints: list[dict] | None = None,
    templates: list[dict] | None = None,
    compute_affinity_for: str | None = None,
) -> str:
    """
    Build a Boltz-2 input YAML string.

    Args:
        sequences:             List of chain dicts from protein_chain(), ligand_smiles(),
                               ligand_ccd(), dna_chain(), or rna_chain().
        constraints:           Optional list of constraint dicts from pocket_constraint(),
                               contact_constraint(), or bond_constraint().
        templates:             Optional list of structural template dicts. Each dict must
                               have a "cif" or "pdb" key pointing to a file path.
                               See SKILL.md for the full template schema.
        compute_affinity_for:  Chain ID of the small-molecule ligand for which to compute
                               binding affinity. Sets the "properties" section.
                               Only one ligand per run. Ligand must have ≤128 atoms.

    Returns:
        YAML-formatted string ready to write to a .yaml file.

    Raises:
        ImportError: if pyyaml is not installed.
    """
    import yaml

    data: dict[str, Any] = {"version": 1, "sequences": sequences}

    if constraints:
        data["constraints"] = constraints
    if templates:
        data["templates"] = templates
    if compute_affinity_for:
        data["properties"] = [{"affinity": {"binder": compute_affinity_for}}]

    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------

def predict(
    input_path: str | Path,
    out_dir: str | Path = "./predictions",
    use_msa_server: bool = True,
    use_potentials: bool = False,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    output_format: str = "mmcif",
    checkpoint: str | Path | None = None,
    affinity_checkpoint: str | Path | None = None,
    no_kernels: bool = False,
    override: bool = False,
    devices: int = 1,
    accelerator: str = "gpu",
    extra_args: list[str] | None = None,
) -> dict[str, Path]:
    """
    Run `boltz predict` via subprocess.

    Args:
        input_path:          Path to a .yaml file or directory of .yaml files.
        out_dir:             Output directory. Boltz creates subdirs per input file.
        use_msa_server:      Auto-generate MSA via mmseqs2. Required when the YAML
                             does not specify an msa path.
        use_potentials:      Apply inference-time potentials for better pose quality.
        recycling_steps:     Refinement iterations. 3 = fast; 10 = AF3-quality.
        diffusion_samples:   Number of structural samples per input. Default 1; use
                             25 for AF3-equivalent diversity.
        sampling_steps:      Diffusion steps. Default 200; reduce for speed.
        output_format:       "mmcif" (default) or "pdb".
        checkpoint:          Path to structure prediction weights. If None, boltz
                             downloads automatically from HuggingFace on first run.
        affinity_checkpoint: Path to affinity prediction weights. Ignored if no
                             affinity is requested in the YAML.
        no_kernels:          Disable cuequivariance kernels. Required on old NVIDIA
                             GPUs that raise a cuequivariance error.
        override:            Re-run even if predictions already exist in out_dir.
        devices:             Number of GPUs to use.
        accelerator:         "gpu", "cpu", or "tpu".
        extra_args:          Additional CLI flags passed verbatim to boltz predict.

    Returns:
        Dict mapping "out_dir" → Path of the predictions directory.

    Raises:
        subprocess.CalledProcessError: if boltz predict exits with a non-zero code.
        FileNotFoundError: if boltz is not installed (pip install boltz).
    """
    cmd = ["boltz", "predict", str(input_path)]

    cmd += ["--out_dir", str(out_dir)]
    cmd += ["--recycling_steps", str(recycling_steps)]
    cmd += ["--diffusion_samples", str(diffusion_samples)]
    cmd += ["--sampling_steps", str(sampling_steps)]
    cmd += ["--output_format", output_format]
    cmd += ["--devices", str(devices)]
    cmd += ["--accelerator", accelerator]

    if use_msa_server:
        cmd.append("--use_msa_server")
    if use_potentials:
        cmd.append("--use_potentials")
    if no_kernels:
        cmd.append("--no_kernels")
    if override:
        cmd.append("--override")
    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
    if affinity_checkpoint:
        cmd += ["--affinity_checkpoint", str(affinity_checkpoint)]
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)

    return {"out_dir": Path(out_dir)}


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_confidence(json_path: str | Path) -> dict[str, Any]:
    """
    Parse a Boltz-2 confidence JSON file.

    Args:
        json_path: Path to confidence_<name>_model_N.json.

    Returns:
        Dict with keys (all floats unless noted):
          confidence_score   — primary sorting score: 0.8*complex_plddt + 0.2*iptm
          ptm                — predicted TM-score for the complex [0,1]
          iptm               — interface TM-score [0,1]
          ligand_iptm        — ipTM at protein-ligand interfaces [0,1]
          protein_iptm       — ipTM at protein-protein interfaces [0,1]
          complex_plddt      — mean pLDDT [0,1]
          complex_iplddt     — interface-upweighted pLDDT [0,1]
          complex_pde        — mean PDE in Angstroms (lower=better)
          complex_ipde       — interface PDE in Angstroms (lower=better)
          chains_ptm         — dict mapping chain index → within-chain TM-score
          pair_chains_iptm   — dict of dicts: chain i → chain j → interface TM-score
    """
    with open(json_path) as f:
        return json.load(f)


def parse_affinity(json_path: str | Path) -> dict[str, float]:
    """
    Parse a Boltz-2 affinity JSON file.

    Args:
        json_path: Path to affinity_<name>.json.

    Returns:
        Dict with keys:
          affinity_probability_binary — P(binder) ∈ [0,1].
                                        Use to separate binders from non-binders (hit discovery).
          affinity_pred_value         — log10(IC50 in μM).
                                        Lower = stronger binder.
                                        -3 → IC50 1 nM (strong), 0 → 1 μM, 2 → 100 μM (weak).
                                        Use only to rank active molecules vs. each other.
                                        Convert to pIC50 (kcal/mol): (6 - y) * 1.364.
          affinity_pred_value1        — First ensemble member prediction.
          affinity_probability_binary1
          affinity_pred_value2        — Second ensemble member prediction.
          affinity_probability_binary2

    Raises:
        FileNotFoundError: if affinity was not requested in the input YAML.
    """
    with open(json_path) as f:
        return json.load(f)


def find_outputs(out_dir: str | Path, input_stem: str) -> dict[str, list[Path] | Path | None]:
    """
    Locate all output files for a given input within the predictions directory.

    Args:
        out_dir:     The --out_dir passed to predict().
        input_stem:  Stem of the input file (e.g. "my_complex" for "my_complex.yaml").

    Returns:
        Dict with keys:
          structures   — list of CIF/PDB Path objects sorted by model index
          confidence   — list of confidence JSON Paths (one per sample)
          affinity     — Path to affinity JSON, or None if not computed
          plddt        — list of pLDDT .npz Paths
          pae          — list of PAE .npz Paths
          pde          — list of PDE .npz Paths
    """
    base = Path(out_dir) / "predictions" / input_stem

    structures  = sorted(base.glob(f"{input_stem}_model_*.cif")) or \
                  sorted(base.glob(f"{input_stem}_model_*.pdb"))
    confidence  = sorted(base.glob(f"confidence_{input_stem}_model_*.json"))
    plddt       = sorted(base.glob(f"plddt_{input_stem}_model_*.npz"))
    pae         = sorted(base.glob(f"pae_{input_stem}_model_*.npz"))
    pde         = sorted(base.glob(f"pde_{input_stem}_model_*.npz"))
    aff_path    = base / f"affinity_{input_stem}.json"
    affinity    = aff_path if aff_path.exists() else None

    return {
        "structures": structures,
        "confidence": confidence,
        "affinity":   affinity,
        "plddt":      plddt,
        "pae":        pae,
        "pde":        pde,
    }


def cif_to_pdb(cif_path: str | Path, pdb_path: str | Path) -> None:
    """
    Convert a CIF structure file to PDB format using Biopython.

    Boltz-2 outputs CIF by default. Use this when downstream tools require PDB.

    Args:
        cif_path: Path to input .cif file.
        pdb_path: Path to write output .pdb file.

    Raises:
        ImportError: if Biopython (pip install biopython) is not installed.
    """
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Boltz-2 inference helper — YAML building, prediction, output parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- download-weights ----
    p = subparsers.add_parser(
        "download-weights",
        help="Pre-download Boltz-2 weights from HuggingFace (boltz-community/boltz-2).",
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help="Directory to store weights. Default: ~/.boltz/boltz2.",
    )

    # ---- predict ----
    p = subparsers.add_parser(
        "predict",
        help="Run boltz predict on a YAML file or directory of YAML files.",
    )
    p.add_argument("input", help="Path to a .yaml file or directory of .yaml files.")
    p.add_argument("--out-dir", default="./predictions", help="Output directory.")
    p.add_argument("--use-msa-server", action="store_true",
                   help="Auto-generate MSA via mmseqs2.")
    p.add_argument("--use-potentials", action="store_true",
                   help="Apply potentials for better physical quality.")
    p.add_argument("--recycling-steps", type=int, default=3)
    p.add_argument("--diffusion-samples", type=int, default=1,
                   help="Number of structural samples per input.")
    p.add_argument("--sampling-steps", type=int, default=200)
    p.add_argument("--output-format", choices=["mmcif", "pdb"], default="mmcif")
    p.add_argument("--checkpoint", default=None,
                   help="Path to structure prediction weights.")
    p.add_argument("--affinity-checkpoint", default=None,
                   help="Path to affinity prediction weights.")
    p.add_argument("--no-kernels", action="store_true",
                   help="Disable cuequivariance kernels (required on old NVIDIA GPUs).")
    p.add_argument("--override", action="store_true",
                   help="Re-run even if predictions already exist.")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "tpu"])

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "download-weights":
        paths = download_boltz2_weights(local_dir=args.local_dir)
        print("Downloaded Boltz-2 weights:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    elif args.command == "predict":
        result = predict(
            input_path=args.input,
            out_dir=args.out_dir,
            use_msa_server=args.use_msa_server,
            use_potentials=args.use_potentials,
            recycling_steps=args.recycling_steps,
            diffusion_samples=args.diffusion_samples,
            sampling_steps=args.sampling_steps,
            output_format=args.output_format,
            checkpoint=args.checkpoint,
            affinity_checkpoint=args.affinity_checkpoint,
            no_kernels=args.no_kernels,
            override=args.override,
            devices=args.devices,
            accelerator=args.accelerator,
        )
        print(f"Predictions written to: {result['out_dir']}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
