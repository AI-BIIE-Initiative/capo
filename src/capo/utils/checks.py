"""
checks.py — Canonical run-directory structure validator + active repair.

Run as a module:
    python -m capo.utils.checks --run-dir <path> [--stage preflight|postrun]
    python -m capo.utils.checks --run-dir <path> --stage postrun --repair

`--repair` does more than create missing dirs: it physically moves loose files
out of the run root into their canonical subdir (e.g. `train.log` → `outputs/`,
`eval_metrics.csv` → `results/`, `evaluation_report.md` → `reports/`), removes
forbidden subdirs like `fine-tuning/` after relocating their contents, and
deletes the deprecated `README.md` stub. Repaired moves are reported.

Canonical layout (single source of truth — keep in sync with
prepare_remote_run_dir() and _setup_run_dir()):

    <run_id>/
    ├── checkpoints/{best,last}/
    ├── compaction/
    ├── configs/{experiment,training,evaluation}.yaml
    ├── outputs/        (*.log, status.json, metrics.jsonl, train.pid)
    ├── pricing/
    ├── probe/
    ├── profile/plots/
    ├── reports/health/ (+ manifests, summaries, evaluation_report.md)
    ├── results/{plots,predictions}/
    ├── scripts/
    ├── src/{data,models,train,eval,utils}/__init__.py
    ├── infra.json, manifest.json, state.json, task.md, RUN_REPORT.md
    └── probe.py, train.py, requirements.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

REQUIRED_DIRS_PREFLIGHT: list[str] = [
    "checkpoints",
    "compaction",
    "configs",
    "outputs",
    "pricing",
    "probe",
    "profile",
    "reports",
    "results",
    "scripts",
    "src",
    "src/data",
    "src/models",
    "src/train",
    "src/eval",
    "src/utils",
]

REQUIRED_DIRS_POSTRUN: list[str] = REQUIRED_DIRS_PREFLIGHT + [
    "checkpoints/best",
    "checkpoints/last",
    "results/plots",
]

REQUIRED_FILES_PREFLIGHT: list[str] = [
    "manifest.json",
    "state.json",
    "task.md",
    "probe.py",
    "train.py",
    "requirements.txt",
    "configs/experiment.yaml",
    "configs/training.yaml",
    "configs/evaluation.yaml",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/models/__init__.py",
    "src/train/__init__.py",
    "src/eval/__init__.py",
    "src/utils/__init__.py",
]

REQUIRED_FILES_POSTRUN: list[str] = REQUIRED_FILES_PREFLIGHT + [
    "outputs/status.json",
    "outputs/train.log",
    "outputs/metrics.jsonl",
    "results/eval_metrics.csv",
    "results/metrics.json",
    "reports/evaluation_report.md",
    "reports/plot_manifest.json",
    "reports/final_summary.json",
    "RUN_REPORT.md",
]

FORBIDDEN_TOP_LEVEL_DIRS: set[str] = {
    "fine-tuning",
    "finetuning",
    "training",
    "ft",
    "logs",                # merged into outputs/
    "probes",              # singular: probe/
    "figures",             # merged into results/plots/ + profile/plots/
    "archive",
    "repairs",
    "environment",
}

ALLOWED_TOP_LEVEL_FILES: set[str] = {
    "infra.json",
    "manifest.json",
    "state.json",
    "task.md",
    "RUN_REPORT.md",
    "probe.py",
    "train.py",
    "requirements.txt",
    # Mirrored from reports/ for convenience during pre-launch; not required
    "prior_runs.md",
}

# File-move rules used by --repair: map (predicate on basename) → target dir.
# Order matters — first match wins.
_REPAIR_RULES: list[tuple[str, str]] = [
    # operational logs and runtime state → outputs/
    (".log", "outputs"),
    ("status.json", "outputs"),
    ("train.pid", "outputs"),
    ("metrics.jsonl", "outputs"),
    # scientific outputs → results/
    ("eval_metrics.csv", "results"),
    ("eval_per_class.csv", "results"),
    ("train_metrics.csv", "results"),
    ("metrics.json", "results"),
    ("predictions.csv", "results/predictions"),
    # reports
    ("evaluation_report.md", "reports"),
    ("plot_manifest.json", "reports"),
    ("final_summary.json", "reports"),
    ("structure_validation.json", "reports"),
    ("trackio_check.json", "reports"),
    ("trackio_url.txt", "reports"),
    ("research_findings.json", "reports"),
    ("research_findings_agent.json", "reports"),
    ("model_selection.json", "reports"),
    ("handoff.json", "reports"),
    ("gate_state.json", "reports"),
    ("epoch_plan.json", "reports"),
    ("evaluation_report.md", "reports"),
    # pricing
    ("cost_report.json", "pricing"),
    # probe
    ("probe_result.json", "probe"),
    ("probe.log", "probe"),
    ("probe_batch_recipe.json", "probe"),
    # scripts
    ("launch_command.sh", "scripts"),
]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructureReport:
    run_dir: Path
    stage: str
    missing_dirs: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    forbidden_items: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repaired_files: list[str] = field(default_factory=list)
    repaired_dirs: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not (self.missing_dirs or self.missing_files or self.forbidden_items)

    def summary(self) -> str:
        lines = [f"Structure check [{self.stage}]: {self.run_dir}"]
        if self.valid:
            lines.append("  PASS — all required dirs and files present, no forbidden items.")
        else:
            lines.append("  FAIL")
        for d in self.missing_dirs:
            lines.append(f"  [MISSING DIR]  {d}/")
        for f in self.missing_files:
            lines.append(f"  [MISSING FILE] {f}")
        for item in self.forbidden_items:
            lines.append(f"  [FORBIDDEN]    {item}")
        for w in self.warnings:
            lines.append(f"  [WARN]         {w}")
        for r in self.repaired_files:
            lines.append(f"  [REPAIRED]     {r}")
        for r in self.repaired_dirs:
            lines.append(f"  [REPAIRED DIR] {r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "stage": self.stage,
            "valid": self.valid,
            "missing_dirs": self.missing_dirs,
            "missing_files": self.missing_files,
            "forbidden_items": self.forbidden_items,
            "warnings": self.warnings,
            "repaired_files": self.repaired_files,
            "repaired_dirs": self.repaired_dirs,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_run_structure(
    run_dir: str | Path,
    stage: str = "preflight",
) -> StructureReport:
    """Validate the canonical run-directory structure (read-only)."""
    run_dir = Path(run_dir).expanduser().resolve()
    report = StructureReport(run_dir=run_dir, stage=stage)

    required_dirs = REQUIRED_DIRS_POSTRUN if stage == "postrun" else REQUIRED_DIRS_PREFLIGHT
    required_files = REQUIRED_FILES_POSTRUN if stage == "postrun" else REQUIRED_FILES_PREFLIGHT

    for d in required_dirs:
        if not (run_dir / d).is_dir():
            report.missing_dirs.append(d)

    for f in required_files:
        if not (run_dir / f).is_file():
            report.missing_files.append(f)

    if run_dir.exists():
        for item in run_dir.iterdir():
            name = item.name
            if item.is_dir() and name in FORBIDDEN_TOP_LEVEL_DIRS:
                report.forbidden_items.append(f"{name}/")
                continue
            if item.is_file():
                if name == "README.md":
                    report.forbidden_items.append("README.md (use RUN_REPORT.md)")
                elif name.endswith(".py") and name not in {"probe.py", "train.py"}:
                    report.forbidden_items.append(f"{name} (no Python at run root)")
                elif name not in ALLOWED_TOP_LEVEL_FILES and "." in name:
                    # *.log, *.csv, *.json, *.jsonl, etc that aren't whitelisted
                    ext = name.rsplit(".", 1)[-1].lower()
                    if ext in {"log", "csv", "jsonl", "tsv", "pid", "txt"}:
                        report.forbidden_items.append(f"{name} (loose at run root)")
                    elif ext == "json" and name not in ALLOWED_TOP_LEVEL_FILES:
                        report.forbidden_items.append(f"{name} (loose at run root)")
                    elif ext == "md" and name not in ALLOWED_TOP_LEVEL_FILES:
                        report.forbidden_items.append(f"{name} (loose at run root)")

    # State.json status sanity
    state_json = run_dir / "state.json"
    if state_json.is_file():
        try:
            state = json.loads(state_json.read_text(encoding="utf-8"))
            status = state.get("status") or state.get("current_phase") or ""
            valid_statuses = {
                "init", "initialized", "configured", "data_ready",
                "pre_launch", "training", "evaluating", "finalizing",
                "completed", "failed", "archived", "paused",
            }
            if status and status not in valid_statuses:
                report.warnings.append(
                    f"state.json has unrecognised status={status!r}; "
                    f"expected one of {sorted(valid_statuses)}"
                )
        except (json.JSONDecodeError, OSError):
            report.warnings.append("state.json is present but unreadable")

    return report


# ---------------------------------------------------------------------------
# Active repair — physically moves misplaced files into canonical subdirs
# ---------------------------------------------------------------------------

def _target_dir_for(name: str) -> str | None:
    """Look up the canonical subdir for a loose root-level file name."""
    lname = name.lower()
    for needle, target in _REPAIR_RULES:
        if lname == needle.lower() or lname.endswith(needle.lower()):
            return target
    return None


def _move_with_collision(src: Path, dst_dir: Path) -> tuple[Path, bool]:
    """Move src into dst_dir. On collision, append .moved-<n> to the basename.
    Returns (new_path, ok)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        # Avoid clobbering: append a numeric suffix.
        stem = src.stem
        suffix = src.suffix
        n = 1
        while (dst_dir / f"{stem}.moved-{n}{suffix}").exists():
            n += 1
        dst = dst_dir / f"{stem}.moved-{n}{suffix}"
    try:
        shutil.move(str(src), str(dst))
        return dst, True
    except (OSError, shutil.Error):
        return src, False


def repair_run_structure(run_dir: str | Path) -> StructureReport:
    """Actively move misplaced files into canonical subdirs and create missing
    dirs. Returns a StructureReport with `.repaired_files` and `.repaired_dirs`
    populated, then re-validates."""
    run_dir = Path(run_dir).expanduser().resolve()
    repaired_files: list[str] = []
    repaired_dirs: list[str] = []

    # 1) Create all required dirs (use POSTRUN superset since repair is most
    #    commonly invoked post-run by the finalizer).
    for d in REQUIRED_DIRS_POSTRUN:
        target = run_dir / d
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            repaired_dirs.append(f"created {d}/")

    # 2) Stub package __init__.py files.
    for pkg in ("src", "src/data", "src/models", "src/train", "src/eval", "src/utils"):
        init = run_dir / pkg / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
            repaired_files.append(f"created {pkg}/__init__.py")

    # 3) Flatten any forbidden top-level dirs by moving their contents up.
    for item in list(run_dir.iterdir()):
        if not item.is_dir() or item.name not in FORBIDDEN_TOP_LEVEL_DIRS:
            continue
        forbidden = item
        for child in list(forbidden.rglob("*")):
            if child.is_dir():
                continue
            rel = child.relative_to(forbidden)
            target_dir_name = _target_dir_for(child.name)
            if target_dir_name is None:
                # No specific rule — preserve the relative path under run root.
                dst_dir = run_dir / rel.parent
            else:
                dst_dir = run_dir / target_dir_name
            new_path, ok = _move_with_collision(child, dst_dir)
            if ok:
                repaired_files.append(
                    f"moved {forbidden.name}/{rel} -> {new_path.relative_to(run_dir)}"
                )
        # Remove the now-empty forbidden tree.
        try:
            shutil.rmtree(forbidden)
            repaired_dirs.append(f"removed {forbidden.name}/")
        except OSError:
            pass

    # 4) Delete deprecated README.md (RUN_REPORT.md is the contract).
    readme = run_dir / "README.md"
    if readme.is_file():
        try:
            readme.unlink()
            repaired_files.append("removed README.md")
        except OSError:
            pass

    # 5) Move loose root-level files into their canonical subdir.
    for item in list(run_dir.iterdir()):
        if not item.is_file():
            continue
        name = item.name
        if name in ALLOWED_TOP_LEVEL_FILES:
            continue
        if name in {"probe.py", "train.py"}:
            continue
        target = _target_dir_for(name)
        if target is None:
            continue
        new_path, ok = _move_with_collision(item, run_dir / target)
        if ok:
            repaired_files.append(f"moved {name} -> {new_path.relative_to(run_dir)}")

    report = validate_run_structure(run_dir, stage="postrun")
    report.repaired_files = repaired_files
    report.repaired_dirs = repaired_dirs
    return report


# Backward-compat shim — older callers used `repair_structure` and got back
# a list of created dirs.
def repair_structure(run_dir: str | Path) -> list[str]:
    report = repair_run_structure(run_dir)
    return report.repaired_dirs + report.repaired_files


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m capo.utils.checks",
        description="Validate (and optionally repair) the canonical CAPO run-directory structure.",
    )
    parser.add_argument("--run-dir", required=True, help="Path to the run directory")
    parser.add_argument(
        "--stage",
        choices=["preflight", "postrun"],
        default="preflight",
        help="Validation stage (default: preflight)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Actively move misplaced files into canonical subdirs",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    if args.repair:
        report = repair_run_structure(args.run_dir)
    else:
        report = validate_run_structure(args.run_dir, stage=args.stage)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    sys.exit(0 if report.valid else 1)


if __name__ == "__main__":
    _cli()
