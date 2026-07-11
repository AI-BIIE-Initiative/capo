"""
boltzgen_inference.py — BoltzGen generative protein design helper.

Provides:
  - YAML input builders (designed protein entities with length ranges, target structure files)
  - HuggingFace weight download (boltzgen/boltzgen-1) and repo clone instructions
  - Subprocess wrapper for `boltzgen run`
  - Merge and artifact-download helpers
  - Output parsers for metrics CSVs and final ranked design CIF files

Installation:
  pip install boltzgen huggingface_hub pyyaml
  git clone https://github.com/HannesStark/boltzgen  # run commands from repo root

Weights download automatically from boltzgen/boltzgen-1 on HuggingFace on first run.
Pre-download with the download-weights subcommand for explicit version control.

CLI subcommands:
  download-weights   Pre-fetch BoltzGen weights from HuggingFace
  run                Run boltzgen run on a YAML spec file
  merge              Merge multiple boltzgen output directories
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HuggingFace weight management
# ---------------------------------------------------------------------------

BOLTZGEN_HF_REPO = "boltzgen/boltzgen-1"
BOLTZGEN_GIT_REPO = "https://github.com/HannesStark/boltzgen"

_CHECKPOINT_FILES = {
    "diverse":   "boltzgen1_diverse.ckpt",    # diversity-optimized design weights
    "adherence": "boltzgen1_adherence.ckpt",  # specification-adherent design weights
    "ifold":     "boltzgen1_ifold.ckpt",      # inverse folding weights
    "conf":      "boltz2_conf_final.ckpt",    # bundled Boltz-2 structure prediction
    "aff":       "boltz2_aff.ckpt",           # bundled Boltz-2 affinity prediction
}


def download_boltzgen_weights(local_dir: str | Path | None = None) -> dict[str, Path]:
    """
    Download BoltzGen weights from HuggingFace Hub (boltzgen/boltzgen-1).

    Preferred over relying on boltzgen's built-in automatic download when you
    need explicit control over weight location or version pinning. Downloads
    once; subsequent calls return cached paths without re-downloading.

    Note: boltzgen/boltzgen-1 is not deployed by any HuggingFace Inference
    Provider — weights must always be used locally via the boltzgen CLI.

    Also note: the boltzgen CLI must be run from the cloned repo root when
    YAML files reference structure files with relative paths. Clone from:
    BOLTZGEN_GIT_REPO

    Args:
        local_dir: Directory to store weights. Defaults to ~/.boltzgen/boltzgen1.
                   Alternatively, set HF_HOME or pass --cache to `boltzgen run`.

    Returns:
        Dict mapping checkpoint keys to Path objects:
          "diverse"   → boltzgen1_diverse.ckpt
          "adherence" → boltzgen1_adherence.ckpt
          "ifold"     → boltzgen1_ifold.ckpt
          "conf"      → boltz2_conf_final.ckpt
          "aff"       → boltz2_aff.ckpt

    Raises:
        ImportError:  if huggingface_hub is not installed.
        RuntimeError: if download fails.
    """
    from huggingface_hub import snapshot_download

    target = Path(local_dir) if local_dir else Path.home() / ".boltzgen" / "boltzgen1"
    target.mkdir(parents=True, exist_ok=True)

    repo_path = Path(
        snapshot_download(
            repo_id=BOLTZGEN_HF_REPO,
            local_dir=str(target),
        )
    )

    return {key: repo_path / filename for key, filename in _CHECKPOINT_FILES.items()}


# ---------------------------------------------------------------------------
# YAML entity builders
# ---------------------------------------------------------------------------

def designed_protein(
    id: str,
    length_range: tuple[int, int] | int,
    msa: str | None = None,
) -> dict:
    """
    Build a designed protein entity for the BoltzGen YAML input.

    This entity represents the protein BoltzGen will generate. Specify a length
    range to allow the model to choose chain length during design, or a single
    integer for fixed-length output.

    Args:
        id:           Chain ID (e.g. "B"). Must be distinct from any target chains.
        length_range: (min_len, max_len) tuple for variable-length design (e.g. (80, 140)),
                      serialized as "80..140" in the YAML; or a single int for
                      fixed-length output (e.g. 120), serialized as "120".
        msa:          Path to a precomputed .a3m MSA for this chain. Rarely needed for
                      de novo design; use when redesigning a known sequence region.

    Returns:
        Dict for the "entities" list of build_yaml().
    """
    if isinstance(length_range, tuple):
        sequence_str = f"{length_range[0]}..{length_range[1]}"
    else:
        sequence_str = str(length_range)

    entry: dict[str, Any] = {"id": id, "sequence": sequence_str}
    if msa is not None:
        entry["msa"] = msa
    return {"protein": entry}


def target_structure(
    path: str | Path,
    chains: list[str] | None = None,
) -> dict:
    """
    Build a target structure entity from a CIF or PDB file.

    The target structure guides the design — BoltzGen generates proteins that
    interact with or complement this structure.

    Args:
        path:   Path to the target structure file (.cif or .pdb).
                BoltzGen resolves this path relative to the working directory
                where `boltzgen run` is executed (the cloned repo root when
                using relative paths). Use absolute paths to avoid ambiguity.
        chains: List of chain IDs to include from the file (e.g. ["A", "C"]).
                If None, all chains in the file are included.

    Returns:
        Dict for the "entities" list of build_yaml().
    """
    entry: dict[str, Any] = {"path": str(path)}
    if chains is not None:
        entry["include"] = [{"chain": {"id": c}} for c in chains]
    return {"file": entry}


def protein_sequence(id: str, sequence: str) -> dict:
    """
    Build a known protein sequence entity.

    Use when you want to include a receptor or partner by sequence rather than
    by structure file. Prefer target_structure() when a structure file is
    available — it provides more geometric context to the model.

    Args:
        id:       Chain ID (e.g. "A").
        sequence: Amino-acid sequence in single-letter code.

    Returns:
        Dict for the "entities" list of build_yaml().
    """
    return {"protein": {"id": id, "sequence": sequence}}


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

VALID_PROTOCOLS = frozenset({
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "antibody-anything",
    "nanobody-anything",
    "protein-redesign",
})

PIPELINE_STEPS = (
    "design",
    "inverse_folding",
    "folding",
    "design_folding",
    "affinity",
    "analysis",
    "filtering",
)


def build_yaml(
    entities: list[dict],
    protocol: str,
    settings: dict | None = None,
) -> str:
    """
    Build a BoltzGen input YAML string.

    For small-molecule targets (protein-small_molecule protocol), include the
    ligand-bound receptor CIF via target_structure() rather than adding a
    separate ligand entity — BoltzGen handles ligand geometry from the file.

    Args:
        entities: List of entity dicts from designed_protein(), target_structure(),
                  or protein_sequence().
        protocol: Design protocol. Must be one of VALID_PROTOCOLS:
                  "protein-anything", "peptide-anything", "protein-small_molecule",
                  "antibody-anything", "nanobody-anything", "protein-redesign".
        settings: Optional dict of additional top-level YAML keys (e.g.
                  {"num_designs": 50}). Merged into the top-level document;
                  keys here override CLI flags only when embedded in the YAML.

    Returns:
        YAML-formatted string ready to write to a .yaml file.

    Raises:
        ValueError:  if protocol is not in VALID_PROTOCOLS.
        ImportError: if pyyaml is not installed.
    """
    import yaml

    if protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Valid protocols: {sorted(VALID_PROTOCOLS)}"
        )

    data: dict[str, Any] = {"protocol": protocol, "entities": entities}
    if settings:
        data.update(settings)

    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Design runner
# ---------------------------------------------------------------------------

def run(
    yaml_path: str | Path,
    output: str | Path = "./boltzgen_output",
    protocol: str | None = None,
    num_designs: int = 50,
    budget: int | None = None,
    steps: list[str] | None = None,
    cache: str | Path | None = None,
    devices: int = 1,
    diffusion_batch_size: int | None = None,
    alpha: float | None = None,
    filter_biased: bool = False,
    additional_filters: str | None = None,
    refolding_rmsd_threshold: float | None = None,
    force_download: bool = False,
    reuse: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Path]:
    """
    Run `boltzgen run` via subprocess.

    IMPORTANT: BoltzGen must be run from the cloned repo root when the YAML
    spec file references structure files with relative paths (as all example
    YAMLs do). Either run this function from the repo root, or use absolute
    paths in your YAML file.

    Args:
        yaml_path:                 Path to the BoltzGen YAML spec file.
        output:                    Output directory. BoltzGen creates a structured
                                   hierarchy of subdirectories inside this path.
        protocol:                  Design protocol override. If provided, must be one
                                   of VALID_PROTOCOLS; overrides the protocol embedded
                                   in the YAML. If None, uses the YAML-embedded value.
        num_designs:               Total intermediate designs to generate before
                                   filtering. Use 50 for quick validation; 10,000–60,000
                                   for production runs.
        budget:                    Final diversity-optimized set size. Must be ≤
                                   num_designs. If None, BoltzGen uses its default.
        steps:                     List of pipeline steps to run. Valid values are the
                                   elements of PIPELINE_STEPS. If None, all steps run.
                                   Example: ["design", "inverse_folding"] to stop before
                                   refolding.
        cache:                     Local directory for model storage. Alternative to
                                   setting the HF_HOME environment variable.
        devices:                   Number of GPUs to use.
        diffusion_batch_size:      Samples per diffusion batch. Reduce if GPU runs OOM.
        alpha:                     Quality-diversity tradeoff for the filtering step.
                                   0.0 = maximize quality, 1.0 = maximize diversity.
        filter_biased:             Remove composition-biased designs if True.
        additional_filters:        Hard filter threshold string, e.g. "ALA_fraction<0.3".
                                   Multiple filters can be combined: "ALA_fraction<0.3,plddt>0.7".
        refolding_rmsd_threshold:  RMSD cutoff for the refolding filter step.
        force_download:            Force redownload of model weights even if cached.
        reuse:                     If True, resume an interrupted run without losing
                                   previously completed designs.
        extra_args:                Additional CLI flags passed verbatim to boltzgen run.

    Returns:
        Dict mapping "out_dir" → Path(output).

    Raises:
        ValueError:                     if protocol is provided but not in VALID_PROTOCOLS.
        subprocess.CalledProcessError:  if boltzgen exits with a non-zero code.
        FileNotFoundError:              if boltzgen is not installed (pip install boltzgen).
    """
    if protocol is not None and protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {protocol!r}. Valid protocols: {sorted(VALID_PROTOCOLS)}"
        )

    cmd = ["boltzgen", "run", str(yaml_path)]

    cmd += ["--output", str(output)]
    cmd += ["--num_designs", str(num_designs)]
    cmd += ["--devices", str(devices)]

    if protocol is not None:
        cmd += ["--protocol", protocol]
    if budget is not None:
        cmd += ["--budget", str(budget)]
    if steps:
        cmd += ["--steps"] + steps
    if cache is not None:
        cmd += ["--cache", str(cache)]
    if diffusion_batch_size is not None:
        cmd += ["--diffusion_batch_size", str(diffusion_batch_size)]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if filter_biased:
        cmd.append("--filter_biased")
    if additional_filters is not None:
        cmd += ["--additional_filters", additional_filters]
    if refolding_rmsd_threshold is not None:
        cmd += ["--refolding_rmsd_threshold", str(refolding_rmsd_threshold)]
    if force_download:
        cmd.append("--force_download")
    if reuse:
        cmd.append("--reuse")
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)

    return {"out_dir": Path(output)}


# ---------------------------------------------------------------------------
# Additional CLI commands
# ---------------------------------------------------------------------------

_VALID_ARTIFACTS = frozenset({
    "inverse-fold", "design-diverse", "design-adherence",
    "folding", "affinity", "moldir", "all",
})


def download_artifact(
    artifact: str = "all",
    cache: str | Path | None = None,
    force: bool = False,
) -> None:
    """
    Pre-download a specific BoltzGen model artifact via `boltzgen download`.

    Artifact names and their corresponding checkpoints:
      "design-diverse"   → boltzgen1_diverse.ckpt
      "design-adherence" → boltzgen1_adherence.ckpt
      "inverse-fold"     → boltzgen1_ifold.ckpt
      "folding"          → boltz2_conf_final.ckpt
      "affinity"         → boltz2_aff.ckpt
      "moldir"           → molecule dictionary
      "all"              → all of the above

    For full snapshot downloads with explicit path control, use
    download_boltzgen_weights() instead.

    Args:
        artifact: Artifact name. One of: "inverse-fold", "design-diverse",
                  "design-adherence", "folding", "affinity", "moldir", "all".
        cache:    Local directory to store downloaded models. Alternative to HF_HOME.
        force:    Re-download even if the artifact is already cached.

    Raises:
        ValueError:                    if artifact is not recognized.
        subprocess.CalledProcessError: if boltzgen download exits non-zero.
    """
    if artifact not in _VALID_ARTIFACTS:
        raise ValueError(
            f"Unknown artifact {artifact!r}. Valid artifacts: {sorted(_VALID_ARTIFACTS)}"
        )

    cmd = ["boltzgen", "download", artifact]
    if cache is not None:
        cmd += ["--cache", str(cache)]
    if force:
        cmd.append("--force")

    subprocess.run(cmd, check=True)


def merge(*run_dirs: str | Path, output: str | Path) -> Path:
    """
    Merge multiple boltzgen output directories via `boltzgen merge`.

    Use to pool results from parallel or interrupted runs before final filtering.

    Args:
        *run_dirs: Two or more output directories to merge (at least 2 required).
        output:    Destination directory for the merged results.

    Returns:
        Path to the merged output directory.

    Raises:
        ValueError:                    if fewer than two run_dirs are provided.
        subprocess.CalledProcessError: if boltzgen merge exits non-zero.
    """
    if len(run_dirs) < 2:
        raise ValueError(f"merge() requires at least 2 run directories; got {len(run_dirs)}.")

    cmd = ["boltzgen", "merge"] + [str(d) for d in run_dirs] + ["--output", str(output)]
    subprocess.run(cmd, check=True)

    return Path(output)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def find_outputs(out_dir: str | Path) -> dict[str, list[Path] | Path | None]:
    """
    Locate all output files within a boltzgen output directory.

    Args:
        out_dir: The directory passed as --output to run().

    Returns:
        Dict with keys:
          intermediate_designs               — CIF files from the diffusion step, sorted
          intermediate_designs_inverse_folded — CIF files after sequence design, sorted
          refold_cif                         — Refolded complex CIFs, sorted
          final_ranked_designs               — Filtered, diversity-ranked final CIFs, sorted
                                               (filenames encode rank; lower index = higher quality)
          metrics_csvs                       — metrics_*.csv files in out_dir, sorted
          report                             — Path to results.pdf if it exists, else None

    Note:
        final_ranked_designs is populated only after the filtering step completes.
        If you ran only --steps design inverse_folding, check intermediate_designs
        or intermediate_designs_inverse_folded instead.
    """
    base = Path(out_dir)

    def _cifs(subdir: str) -> list[Path]:
        d = base / subdir
        if not d.exists():
            return []
        return sorted(d.glob("*.cif"))

    report_path = base / "results.pdf"

    return {
        "intermediate_designs": _cifs("intermediate_designs"),
        "intermediate_designs_inverse_folded": _cifs("intermediate_designs_inverse_folded"),
        "refold_cif": _cifs("refold_cif"),
        "final_ranked_designs": _cifs("final_ranked_designs"),
        "metrics_csvs": sorted(base.glob("metrics_*.csv")),
        "report": report_path if report_path.exists() else None,
    }


def parse_metrics(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a BoltzGen metrics CSV into a list of per-design dicts.

    Args:
        csv_path: Path to a metrics_*.csv file from the analysis or filtering step.

    Returns:
        List of dicts, one per design row. Numeric values are coerced to float
        where possible; non-numeric values remain as strings.

        Available columns depend on which pipeline steps were run. Common fields
        include pLDDT, RMSD, iPTM, diversity metrics, and filter pass/fail flags.
        Iterate the keys of a returned row to discover the available columns.

    Raises:
        FileNotFoundError: if csv_path does not exist.
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coerced: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    coerced[k] = float(v)
                except (ValueError, TypeError):
                    coerced[k] = v
            rows.append(coerced)
    return rows


def load_final_designs(out_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load final ranked designs with metrics and structure paths.

    Combines find_outputs() and parse_metrics() to return a merged list of
    design records. Each record includes all metric fields from the CSV plus
    a "cif_path" key pointing to the corresponding CIF file.

    Args:
        out_dir: The directory passed as --output to run().

    Returns:
        List of dicts (one per design), each containing:
          All columns from the metrics CSV (floats where numeric, strings otherwise)
          "cif_path" — Path to the design CIF file, or None if not matched

        If final_ranked_designs/ is empty (filtering step not yet complete),
        metrics rows are returned without CIF paths. Check intermediate_designs/
        or refold_cif/ via find_outputs() for earlier-stage structures.
    """
    outputs = find_outputs(out_dir)
    cif_by_stem = {p.stem: p for p in outputs["final_ranked_designs"]}

    all_rows: list[dict[str, Any]] = []
    for csv_path in outputs["metrics_csvs"]:
        all_rows.extend(parse_metrics(csv_path))

    for row in all_rows:
        design_id = str(row.get("design_id", row.get("id", "")))
        row["cif_path"] = cif_by_stem.get(design_id)

    return all_rows


def cif_to_pdb(cif_path: str | Path, pdb_path: str | Path) -> None:
    """
    Convert a CIF structure file to PDB format using Biopython.

    BoltzGen outputs CIF by default. Use this when downstream tools require PDB.

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
        description="BoltzGen inference helper — YAML building, design runs, output parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- download-weights ----
    p = subparsers.add_parser(
        "download-weights",
        help="Pre-download BoltzGen weights from HuggingFace (boltzgen/boltzgen-1).",
    )
    p.add_argument(
        "--local-dir",
        default=None,
        help="Directory to store weights. Default: ~/.boltzgen/boltzgen1.",
    )

    # ---- run ----
    p = subparsers.add_parser(
        "run",
        help="Run boltzgen run on a YAML spec file.",
    )
    p.add_argument("yaml", help="Path to the BoltzGen YAML spec file.")
    p.add_argument("--output", default="./boltzgen_output", help="Output directory.")
    p.add_argument(
        "--protocol",
        choices=sorted(VALID_PROTOCOLS),
        default=None,
        help="Design protocol. Overrides protocol embedded in the YAML.",
    )
    p.add_argument(
        "--num-designs",
        type=int,
        default=50,
        help="Total intermediate designs to generate. Use 50 for validation; 10000–60000 for production.",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Final set size after diversity selection. Must be ≤ --num-designs.",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=list(PIPELINE_STEPS),
        default=None,
        metavar="STEP",
        help=(
            "Run only specific pipeline steps. "
            f"Valid values: {', '.join(PIPELINE_STEPS)}. "
            "Default: run all steps."
        ),
    )
    p.add_argument("--cache", default=None, help="Model cache directory (alternative to HF_HOME).")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs.")
    p.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=None,
        help="Samples per diffusion batch. Reduce if GPU runs OOM.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Quality-diversity tradeoff: 0.0=quality, 1.0=diversity.",
    )
    p.add_argument(
        "--filter-biased",
        action="store_true",
        help="Remove composition-biased designs.",
    )
    p.add_argument(
        "--additional-filters",
        default=None,
        help="Hard filter thresholds, e.g. 'ALA_fraction<0.3'.",
    )
    p.add_argument(
        "--refolding-rmsd-threshold",
        type=float,
        default=None,
        help="RMSD cutoff for the refolding filter.",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload model weights even if cached.",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Resume an interrupted run without losing progress.",
    )

    # ---- merge ----
    p = subparsers.add_parser(
        "merge",
        help="Merge multiple boltzgen output directories.",
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        metavar="RUN_DIR",
        help="Two or more output directories to merge.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination directory for merged results.",
    )

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "download-weights":
        paths = download_boltzgen_weights(local_dir=args.local_dir)
        print("Downloaded BoltzGen weights:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    elif args.command == "run":
        result = run(
            yaml_path=args.yaml,
            output=args.output,
            protocol=args.protocol,
            num_designs=args.num_designs,
            budget=args.budget,
            steps=args.steps,
            cache=args.cache,
            devices=args.devices,
            diffusion_batch_size=args.diffusion_batch_size,
            alpha=args.alpha,
            filter_biased=args.filter_biased,
            additional_filters=args.additional_filters,
            refolding_rmsd_threshold=args.refolding_rmsd_threshold,
            force_download=args.force_download,
            reuse=args.reuse,
        )
        print(f"Designs written to: {result['out_dir']}")

    elif args.command == "merge":
        merged = merge(*args.run_dirs, output=args.output)
        print(f"Merged output: {merged}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
