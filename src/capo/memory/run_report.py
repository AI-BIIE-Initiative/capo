"""run_report — `RUN_REPORT.md` + `runs_index.md` helpers for episodic memory.

The cross-run index lives at `<repo>/runs/runs_index.md` as a sequence of YAML
frontmatter blocks (one per completed run), delimited by `---` lines. Each
block is the same frontmatter that sits at the top of the run's
`RUN_REPORT.md`. This file is the cheap discovery layer the memory-consultant
subagent scans before deciding which full reports to load (progressive
disclosure, mirroring the SKILL.md pattern).

Public API:
  parse_frontmatter(report_path)      Read a single RUN_REPORT.md frontmatter.
  read_index_blocks()                 Parse every block in runs_index.md.
  append_index_block(frontmatter)     Atomic, fcntl-locked, dedup by run_id.
  validate_frontmatter(d)             Raise ValueError on missing required fields.

CLI:
  python -m capo.memory.run_report append-from-report --run-dir <path>
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = _REPO_ROOT / "runs" / "runs_index.md"
INDEX_LOCK_PATH = _REPO_ROOT / "runs" / ".runs_index.lock"

FRONTMATTER_FIELDS: tuple[str, ...] = (
    "run_id",
    "task_summary",
    "modality",
    "target",
    "organism",
    "assay",
    "best_metric_name",
    "best_metric_value",
    "final_val_loss",
    "key_decisions",
    "key_findings",
    "key_pitfalls",
    "report_path",
)

REQUIRED_FIELDS: tuple[str, ...] = ("run_id", "task_summary", "report_path")


def validate_frontmatter(fm: dict[str, Any]) -> None:
    """Raise ValueError if any required field is missing or empty."""
    missing = [f for f in REQUIRED_FIELDS if not fm.get(f)]
    if missing:
        raise ValueError(
            f"RUN_REPORT.md frontmatter missing required fields: {missing}. "
            f"Got keys: {sorted(fm.keys())}"
        )
    unknown = sorted(set(fm.keys()) - set(FRONTMATTER_FIELDS))
    if unknown:
        # Strict 13-field schema — extra keys indicate a finalizer drift.
        raise ValueError(
            f"RUN_REPORT.md frontmatter has unsupported fields: {unknown}. "
            f"Allowed: {list(FRONTMATTER_FIELDS)}"
        )


def parse_frontmatter(report_path: Path) -> dict[str, Any] | None:
    """Return the YAML frontmatter at the top of a RUN_REPORT.md, or None."""
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding="utf-8")
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None
    # Find the closing `---` line.
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    block = "\n".join(lines[1:end])
    data = yaml.safe_load(block)
    if not isinstance(data, dict):
        return None
    return data


def read_index_blocks(path: Path = INDEX_PATH) -> list[dict[str, Any]]:
    """Parse all YAML blocks in runs_index.md. Skip malformed blocks with a warning."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    blocks: list[dict[str, Any]] = []
    # Walk lines; collect content between matched `---` markers.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "---":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "---":
                j += 1
            if j >= len(lines):
                break
            block_text = "\n".join(lines[i + 1 : j])
            try:
                data = yaml.safe_load(block_text)
            except yaml.YAMLError as exc:
                print(
                    f"[runs_index] Skipping malformed block at line {i + 1}: {exc}",
                    file=sys.stderr,
                )
                data = None
            if isinstance(data, dict):
                blocks.append(data)
            i = j + 1
        else:
            i += 1
    return blocks


def _dump_block(fm: dict[str, Any]) -> str:
    """Format a frontmatter dict as a YAML block delimited by `---` lines."""
    ordered: dict[str, Any] = {k: fm.get(k) for k in FRONTMATTER_FIELDS if k in fm}
    body = yaml.safe_dump(
        ordered,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )
    return f"---\n{body}---\n"


def _write_all_blocks(blocks: list[dict[str, Any]], path: Path = INDEX_PATH) -> None:
    """Rewrite the index file atomically (temp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_dump_block(b) for b in blocks)
    if content and not content.endswith("\n"):
        content += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def append_index_block(
    frontmatter: dict[str, Any],
    *,
    index_path: Path = INDEX_PATH,
    lock_path: Path = INDEX_LOCK_PATH,
) -> None:
    """Append a frontmatter block to runs_index.md, deduplicating by run_id.

    Atomic against concurrent finalizers via an exclusive fcntl lock on a
    sibling lock file. If a block with the same run_id already exists, it is
    replaced in place (rewrite via temp file).
    """
    validate_frontmatter(frontmatter)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode so the lock file is created if missing.
    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            blocks = read_index_blocks(index_path)
            run_id = frontmatter["run_id"]
            replaced = False
            for i, existing in enumerate(blocks):
                if existing.get("run_id") == run_id:
                    blocks[i] = frontmatter
                    replaced = True
                    break
            if not replaced:
                blocks.append(frontmatter)
            _write_all_blocks(blocks, index_path)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def remove_index_blocks(
    run_ids: "list[str] | set[str] | tuple[str, ...]",
    *,
    index_path: Path = INDEX_PATH,
    lock_path: Path = INDEX_LOCK_PATH,
) -> int:
    """Drop every block whose run_id is in *run_ids*; return the count removed.

    This is the inverse of append_index_block — it makes the memory system stop
    surfacing those runs (the memory-consultant scans this index), WITHOUT
    touching the run directory, RUN_REPORT.md, checkpoints or any artifact. Same
    fcntl-locked, atomic-rewrite discipline as append.
    """
    targets = {r for r in run_ids if r}
    if not targets or not index_path.exists():
        return 0
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            blocks = read_index_blocks(index_path)
            kept = [b for b in blocks if b.get("run_id") not in targets]
            removed = len(blocks) - len(kept)
            if removed:
                _write_all_blocks(kept, index_path)
            return removed
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_append_from_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    report_path = run_dir / "RUN_REPORT.md"
    if not report_path.exists():
        print(f"ERROR: {report_path} does not exist", file=sys.stderr)
        return 2
    fm = parse_frontmatter(report_path)
    if fm is None:
        print(
            f"ERROR: {report_path} has no parseable YAML frontmatter",
            file=sys.stderr,
        )
        return 3
    try:
        # Look up module-level paths at call time so tests can monkeypatch them.
        append_index_block(fm, index_path=INDEX_PATH, lock_path=INDEX_LOCK_PATH)
    except ValueError as exc:
        print(f"ERROR: invalid frontmatter — {exc}", file=sys.stderr)
        return 4
    print(f"Appended {fm['run_id']} to {INDEX_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capo.memory.run_report")
    sub = parser.add_subparsers(dest="cmd", required=True)
    append_p = sub.add_parser(
        "append-from-report",
        help="Append a run's RUN_REPORT.md frontmatter to runs_index.md.",
    )
    append_p.add_argument(
        "--run-dir",
        required=True,
        help="Path to the local run directory containing RUN_REPORT.md.",
    )
    append_p.set_defaults(func=_cli_append_from_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
