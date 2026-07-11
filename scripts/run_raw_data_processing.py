"""
Run raw FASTQ data preprocessing on a new Lambda instance via the CAPO framework.

Three modes:

* **Harness mode (default when systems: is set in the config).** Runs the
  leakage-isolated external evaluation: CAPO and the General Coding Agent each
  preprocess the same raw inputs under matched budgets, then a deterministic
  Stage-2 phase loads the held-out gold set and produces summary / per-species
  / efficiency / error / statistical CSVs.

* **Legacy single-system mode** (when systems: is absent). Runs only the
  CAPO preprocessing agent and returns the labeled CSV. Kept for backward
  compatibility.

* **Stage-2-only mode (--rerun-eval EVAL_RUN_DIR).** Skips all preprocessing
  and re-runs evaluation against the existing frozen processed_examples.parquet
  files under <EVAL_RUN_DIR>/systems/*/. Uses the evaluation: block from
  --config, so the contract can be updated (e.g. flipping species_filter
  to null) without redoing the agent runs.

Usage::

    # Full harness run (provisioning + preprocessing + evaluation)
    python scripts/run_raw_data_processing.py --config scripts/configs/raw_data_processing.yaml

    # Re-evaluate an existing eval-run dir without re-running the agents
    python scripts/run_raw_data_processing.py \\
        --config scripts/configs/raw_data_processing.yaml \\
        --rerun-eval lambda/runs/data-processing-eval/ace2-eval-20260610-2031-7a1f

Environment:
    LAMBDA_API_KEY — required for provisioning new instances
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capo.evaluation import (
    CandidateAdapter,
    SystemRun,
    canonicalize_species,
    run_stage2_evaluation,
    write_processed_examples,
)
from capo.orchestration.data_processing_orchestrator import (
    DataProcessingOrchestrator,
    EvaluationHarnessOrchestrator,
    SystemSpec,
    _guess_species_from_basename,
)

_REPO_ROOT = Path(__file__).parent.parent


def _expand(path: str | None) -> str | None:
    return str(Path(path).expanduser()) if path else None


_REQUIRED_TOP_LEVEL_KEYS = (
    "key_path",
    "ssh_key_name",
    "input_dir",
    "output_dir",
    "max_cost_usd",
    "max_runtime_hours",
    "instance_type",
    "terminate_after",
)

# Only required when `systems:` is absent (legacy single-system path).
_REQUIRED_LEGACY_KEYS = ("model_name", "max_turns")


def _require(parser: argparse.ArgumentParser, cfg: dict, keys) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        parser.error(f"Config is missing required keys: {missing}")


_FASTQ_SUFFIXES = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


def _scan_fastq_species(input_dir: str) -> tuple[Counter, int, int]:
    """Recursively scan input_dir for FASTQ files and infer species per filename.

    Returns (counter, files_scanned, files_no_species).
    """
    root = Path(input_dir)
    counter: Counter = Counter()
    files_scanned = 0
    no_species = 0
    if not root.exists():
        return counter, 0, 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if not any(name_lower.endswith(suf) for suf in _FASTQ_SUFFIXES):
            continue
        files_scanned += 1
        guess = _guess_species_from_basename(path.name)
        if guess:
            counter[guess] += 1
        else:
            no_species += 1
    return counter, files_scanned, no_species


def _preflight_species_overlap(
    input_dir: str,
    eval_cfg: dict,
    parser: argparse.ArgumentParser,
) -> None:
    """Surface input/species mismatches BEFORE any agent is launched.

    Hard-errors if species_filter is set and shares zero species with the
    files in input_dir. Otherwise just prints the inferred species set so
    the user can sanity-check what the agents will see.
    """
    counter, files_scanned, no_species = _scan_fastq_species(input_dir)

    print(f"Preflight — input_dir: {input_dir}")
    print(f"  files scanned: {files_scanned}")
    if counter:
        print(f"  inferred species: {dict(counter)}")
    else:
        print("  inferred species: {} (none recognised)")
    print(f"  files with no species inferred: {no_species}")

    gold_set = eval_cfg.get("gold_set") or {}
    species_filter = gold_set.get("species_filter")

    if not species_filter:
        print("  species_filter: null (will evaluate any gold species)")
        return

    canon_filter = {canonicalize_species(s) for s in species_filter}
    canon_inferred = {canonicalize_species(s) for s in counter.keys()}
    overlap = canon_filter & canon_inferred

    if not overlap:
        parser.error(
            "Preflight failure: species_filter "
            f"({sorted(canon_filter)}) has no overlap with species inferred "
            f"from input_dir ({sorted(canon_inferred) or '[]'}). "
            "Either point input_dir at FASTQs for the filtered species, "
            "or set species_filter: null to evaluate any overlap."
        )

    print(f"  species_filter overlap: {sorted(overlap)}")
    missing = canon_filter - canon_inferred
    if missing:
        print(
            f"  WARNING: species_filter species not found in input: "
            f"{sorted(missing)}"
        )


def _refreeze_system_parquet(
    sys_dir: Path,
    system_name: str,
    run_id: str,
) -> bool:
    """Rebuild processed_examples.parquet from agent CSVs in outputs/.

    Called by --rerun-eval so any post-hoc edits to the agent CSVs (or
    fixes to CandidateAdapter aliases) take effect without redoing
    preprocessing. Returns True if a parquet was written, False if no CSVs
    existed under outputs/.
    """
    import pandas as pd

    outputs_dir = sys_dir / "outputs"
    if not outputs_dir.exists():
        return False
    csvs = sorted(outputs_dir.glob("*.csv"))
    if not csvs:
        return False

    frames: list[pd.DataFrame] = []
    for csv_path in csvs:
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception as exc:
            print(f"  warn {sys_dir.name}: could not read {csv_path.name}: {exc}")
    if not frames:
        return False
    raw_concat = pd.concat(frames, ignore_index=True)

    adapter = CandidateAdapter(run_id=run_id, system_name=system_name)
    canonical = adapter.adapt(raw_concat)
    write_processed_examples(sys_dir, canonical)
    print(
        f"  refroze {system_name}: {len(canonical):,} rows "
        f"from {len(csvs)} CSV(s) under outputs/"
    )
    return True


def _rerun_stage2(
    eval_run_dir: Path,
    eval_cfg: dict,
    parser: argparse.ArgumentParser,
) -> dict[str, Path]:
    """Re-run Stage-2 evaluation against an existing eval-run directory.

    Skips all preprocessing — re-freezes processed_examples.parquet from
    each system's outputs/ CSVs (so post-hoc CSV edits and adapter fixes
    take effect), then rewrites the evaluation/ artifacts in place using
    eval_cfg (the current YAML's contract, *not* the frozen one — see note
    printed at runtime).
    """
    if not eval_run_dir.exists():
        parser.error(f"--rerun-eval directory does not exist: {eval_run_dir}")
    systems_root = eval_run_dir / "systems"
    if not systems_root.exists():
        parser.error(f"No systems/ subdirectory under {eval_run_dir}")

    system_runs: list[SystemRun] = []
    for sys_dir in sorted(p for p in systems_root.iterdir() if p.is_dir()):
        meta_path = sys_dir / "run_metadata.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  warn {sys_dir.name}: could not read run_metadata.json: {exc}")
        else:
            print(f"  warn {sys_dir.name}: run_metadata.json missing — using defaults")

        system_name = meta.get("system_name") or sys_dir.name.replace("_", " ")
        run_id = meta.get("run_id", f"rerun-{sys_dir.name}")

        # Always re-freeze parquet from the agent CSVs under outputs/ so any
        # post-hoc CSV edits and CandidateAdapter fixes take effect. Falls
        # back to the existing parquet if no CSVs are present.
        try:
            refroze = _refreeze_system_parquet(sys_dir, system_name, run_id)
        except ValueError as exc:
            print(f"  ERROR refreezing {sys_dir.name}: {exc}")
            print(f"  (skipping {sys_dir.name} from Stage-2)")
            continue
        parquet = sys_dir / "processed_examples.parquet"
        if not refroze and not parquet.exists():
            print(
                f"  skip {sys_dir.name}: no outputs/*.csv to refreeze AND "
                f"no existing processed_examples.parquet"
            )
            continue
        if not refroze:
            print(f"  {sys_dir.name}: no outputs/*.csv found — using existing parquet")

        system_runs.append(SystemRun(
            name=system_name,
            setting=meta.get("comparison_setting", "budget_matched"),
            run_dir=sys_dir,
            runtime_seconds=float(meta.get("runtime_seconds", 0.0)),
            cost_usd=float(meta.get("estimated_cost_usd", 0.0)),
            raw_reads_processed=meta.get("raw_reads_processed"),
            reads_retained_after_qc=meta.get("reads_retained_after_qc"),
            failure_count=0,
            total_attempts=1,
        ))

    if not system_runs:
        parser.error(f"No usable system runs under {systems_root}")

    out_dir = eval_run_dir / "evaluation"
    print(
        f"Re-running Stage-2 on {len(system_runs)} system(s) → {out_dir}\n"
        f"  contract source: --config (NOT the frozen evaluation_config.yaml)"
    )
    rng_seed = int(eval_cfg.get("rng_seed", 0))
    paths = run_stage2_evaluation(
        systems=system_runs,
        eval_config=eval_cfg,
        out_dir=out_dir,
        rng_seed=rng_seed,
    )
    return paths


def _print_rerun_summary(eval_run_dir: Path, paths: dict[str, Path]) -> None:
    print()
    print(f"eval_run_dir       : {eval_run_dir}")
    for key, val in paths.items():
        print(f"  {key:24}: {val}")
    summary_csv = paths.get("summary_metrics_csv") or paths.get("summary_metrics")
    if summary_csv and Path(summary_csv).exists():
        import pandas as pd
        try:
            df = pd.read_csv(summary_csv)
            macro = df[df["species_weighting"] == "macro"]
            if len(macro):
                print()
                print("Headline (macro-averaged annotation exact match):")
                for _, row in macro.iterrows():
                    print(
                        f"  - {row['system_name']:24}"
                        f"  AEM={row['annotation_exact_match']:.4f}"
                        f"  coverage={row['gold_coverage']:.4f}"
                        f"  setting={row['comparison_setting']}"
                    )
        except Exception as exc:
            print(f"(could not read summary_metrics.csv: {exc})")


def _print_single_system_summary(result) -> None:
    print()
    print(f"run_id          : {result.run_id}")
    print(f"state           : {result.state}")
    print(f"local_run_dir   : {result.local_run_dir}")
    if result.output_csv:
        print(f"output_csv      : {result.output_csv}")
    if result.total_sequences is not None:
        print(f"total_sequences : {result.total_sequences:,}")
    if result.bind_count is not None:
        print(f"bind            : {result.bind_count:,}")
    if result.non_count is not None:
        print(f"non             : {result.non_count:,}")
    if result.instance_type:
        print(f"instance_type   : {result.instance_type}")
    if result.actual_cost_usd is not None:
        print(f"infra_cost      : ${result.actual_cost_usd:.2f}")
    if result.agent_cost_usd is not None:
        print(f"agent_cost      : ${result.agent_cost_usd:.4f}")
    if result.actual_cost_usd is not None and result.agent_cost_usd is not None:
        print(f"total_cost      : ${result.actual_cost_usd + result.agent_cost_usd:.2f}")


def _print_harness_summary(result) -> None:
    print()
    print(f"eval_run_id        : {result.eval_run_id}")
    print(f"eval_run_dir       : {result.eval_run_dir}")
    print(f"state              : {result.state}")
    print(f"systems            : {len(result.system_results)}")
    for sr in result.system_results:
        line = f"  - {sr.run_id:42}  state={sr.state}"
        if sr.actual_cost_usd is not None:
            line += f"  infra=${sr.actual_cost_usd:.2f}"
        if sr.agent_cost_usd is not None:
            line += f"  agent=${sr.agent_cost_usd:.4f}"
        print(line)
    if result.summary_metrics_csv:
        print(f"summary_metrics    : {result.summary_metrics_csv}")
    if result.per_species_metrics_csv:
        print(f"per_species_metrics: {result.per_species_metrics_csv}")
    if result.efficiency_metrics_csv:
        print(f"efficiency_metrics : {result.efficiency_metrics_csv}")
    if result.error_analysis_csv:
        print(f"error_analysis     : {result.error_analysis_csv}")
    if result.statistical_tests_csv:
        print(f"statistical_tests  : {result.statistical_tests_csv}")
    if result.gold_alignment_review_csv:
        print(f"alignment_review   : {result.gold_alignment_review_csv}")

    # Print the headline annotation_exact_match per system if the harness
    # actually wrote the summary table.
    if result.summary_metrics_csv and Path(result.summary_metrics_csv).exists():
        import pandas as pd
        try:
            df = pd.read_csv(result.summary_metrics_csv)
            macro = df[df["species_weighting"] == "macro"]
            if len(macro):
                print()
                print("Headline (macro-averaged annotation exact match):")
                for _, row in macro.iterrows():
                    print(
                        f"  - {row['system_name']:24}"
                        f"  AEM={row['annotation_exact_match']:.4f}"
                        f"  coverage={row['gold_coverage']:.4f}"
                        f"  setting={row['comparison_setting']}"
                    )
        except Exception as exc:
            print(f"(could not read summary_metrics.csv: {exc})")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (see scripts/configs/raw_data_processing.yaml).",
    )
    p.add_argument(
        "--rerun-eval",
        metavar="EVAL_RUN_DIR",
        default=None,
        help=(
            "Skip all preprocessing and re-run Stage-2 evaluation against the "
            "existing frozen processed_examples.parquet files under "
            "<EVAL_RUN_DIR>/systems/*/. Uses the `evaluation:` block from "
            "--config (so the contract can be updated without redoing the "
            "agent runs). Overwrites <EVAL_RUN_DIR>/evaluation/."
        ),
    )
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Stage-2-only mode — re-run evaluation on an existing eval-run dir
    # ------------------------------------------------------------------
    if args.rerun_eval:
        eval_cfg = cfg.get("evaluation")
        if not eval_cfg:
            p.error("--rerun-eval requires `evaluation:` in --config.")
        paths = _rerun_stage2(Path(args.rerun_eval).expanduser(), eval_cfg, p)
        _print_rerun_summary(Path(args.rerun_eval).expanduser(), paths)
        return

    _require(p, cfg, _REQUIRED_TOP_LEVEL_KEYS)

    systems_cfg = cfg.get("systems")

    # ------------------------------------------------------------------
    # Harness mode — leakage-isolated CAPO vs General Coding Agent
    # ------------------------------------------------------------------
    if systems_cfg:
        eval_cfg = cfg.get("evaluation")
        if not eval_cfg:
            p.error("`systems:` is set but no `evaluation:` block found. "
                    "Add the pre-registered evaluation contract or remove `systems:`.")

        # Surface input/species mismatches before any agent or Lambda call.
        _preflight_species_overlap(_expand(cfg["input_dir"]), eval_cfg, p)

        specs: list[SystemSpec] = []
        for entry in systems_cfg:
            if not entry.get("name") or not entry.get("kind"):
                p.error(f"Each `systems` entry needs `name` and `kind`; got {entry!r}")
            specs.append(SystemSpec(
                name=entry["name"],
                kind=entry["kind"],
                setting=entry.get("setting", "budget_matched"),
                model_name=entry.get("model_name", "claude-sonnet-4-6"),
                max_turns=int(entry.get("max_turns", 300)),
            ))

        harness = EvaluationHarnessOrchestrator(
            key_path=_expand(cfg["key_path"]),
            ssh_key_name=cfg["ssh_key_name"],
            input_dir=_expand(cfg["input_dir"]),
            output_dir=_expand(cfg["output_dir"]),
            systems=specs,
            eval_config=eval_cfg,
            max_cost_usd=cfg["max_cost_usd"],
            max_runtime_hours=cfg["max_runtime_hours"],
            instance_type=cfg["instance_type"] or None,
            terminate_after=cfg["terminate_after"],
            input_hf_ref=cfg.get("input_hf_ref") or None,
        )
        result = harness.run_sync(eval_run_id=cfg.get("eval_run_id"))
        _print_harness_summary(result)
        return

    # ------------------------------------------------------------------
    # Legacy single-system path (no `systems:` block)
    # ------------------------------------------------------------------
    _require(p, cfg, _REQUIRED_LEGACY_KEYS)
    orch = DataProcessingOrchestrator(
        key_path=_expand(cfg["key_path"]),
        ssh_key_name=cfg["ssh_key_name"],
        input_dir=_expand(cfg["input_dir"]),
        output_dir=_expand(cfg["output_dir"]),
        max_cost_usd=cfg["max_cost_usd"],
        max_runtime_hours=cfg["max_runtime_hours"],
        instance_type=cfg["instance_type"] or None,
        terminate_after=cfg["terminate_after"],
        model_name=cfg["model_name"],
        max_turns=cfg["max_turns"],
        input_hf_ref=cfg.get("input_hf_ref") or None,
    )

    result = orch.run_sync(run_id=cfg.get("run_id"))
    _print_single_system_summary(result)


if __name__ == "__main__":
    main()
