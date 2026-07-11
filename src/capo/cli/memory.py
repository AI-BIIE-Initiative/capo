"""
Prune CAPO episodic memory — capo prune-memory and the /prune-memory slash
command.

"Memory" here is the cross-run index at runs/runs_index.md — the cheap
discovery layer the memory-consultant scans before deciding which full reports
to load. Pruning a run removes only its entry from that index, so the model no
longer rediscovers it through memory search. It NEVER deletes the run directory,
RUN_REPORT.md, the final report, checkpoints or any artifact — the run still
exists on disk and in capo history, it is simply forgotten by memory.

Two entry shapes, both deterministic (no model call):
  • prune_memory(index_path, run_id="abc") — remove one run directly.
  • prune_memory(index_path) — open an interactive multi-select that mirrors
    the config editor: arrow keys move, Enter ticks/unticks a run, then
    [ Confirm ] / [ Cancel ] at the bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from capo.memory.run_report import read_index_blocks, remove_index_blocks

from .colors import console


def _summary(block: dict) -> str:
    """One-line label for a run: 'run_id — task summary'."""
    rid = str(block.get("run_id", "?"))
    summ = str(block.get("task_summary") or "").strip().replace("\n", " ")
    return f"{rid} — {summ}" if summ else rid


def _prune_one(index_path: Path, run_id: str) -> None:
    """Direct form: remove a single run by id, with a clear result line."""
    removed = remove_index_blocks([run_id], index_path=index_path)
    if removed:
        console.print(f"\n  [ok]✓[/] Removed [brand.dim]{run_id}[/] from CAPO memory index.")
        console.print("  [muted]The run directory and its artifacts are untouched.[/]\n")
    else:
        console.print(f"\n  [muted]No memory entry found for [/][brand.dim]{run_id}[/][muted].[/]\n")


def _prune_interactive(index_path: Path, blocks: list[dict]) -> None:
    """Multi-select selector (same interaction model as the config editor)."""
    from .widgets import _opt_lines, _select

    console.print("  [prompt.hint]↑/↓ move · Enter tick/untick · select Confirm / Cancel[/]\n")
    selected: set[int] = set()  # indices of ticked runs, buffered until Confirm
    n = len(blocks)
    last = 0

    def _rows() -> list[str]:
        rows = []
        for i, b in enumerate(blocks):
            mark = "x" if i in selected else " "
            rows.append(f"[{mark}] {_summary(b)}"[:96])
        k = len(selected)
        confirm = "[ Confirm ]" + (f"  (remove {k})" if k else "")
        return rows + [confirm, "[ Cancel ]"]

    while True:
        rows = _rows()
        idx = _select(lambda cur: _opt_lines(rows, cur), len(rows), default_idx=last)
        if idx is None or idx == n + 1:  # Esc or [ Cancel ]
            console.print("  [muted]Cancelled — nothing removed.[/]\n")
            return
        if idx == n:  # [ Confirm ]
            if not selected:
                console.print("  [muted]Nothing selected.[/]\n")
                return
            run_ids = [str(blocks[i].get("run_id")) for i in selected]
            removed = remove_index_blocks(run_ids, index_path=index_path)
            console.print(
                f"\n  [ok]✓[/] Removed {removed} run{'' if removed == 1 else 's'} from CAPO memory."
            )
            console.print("  [muted]Run directories and artifacts are untouched.[/]\n")
            return
        last = idx
        selected.symmetric_difference_update({idx})  # toggle this run


def prune_memory(index_path: Path, run_id: Optional[str] = None) -> None:
    """Prune the runs_index.md memory at *index_path* (one run, or interactive)."""
    console.print()
    console.rule("[brand]CAPO memory[/]", style="brand.dim")
    console.print(f"  [muted]{index_path}[/]\n")

    if run_id:
        _prune_one(index_path, run_id.strip())
        return

    blocks = read_index_blocks(index_path)
    if not blocks:
        console.print("  [muted]No runs are stored in memory yet — nothing to prune.[/]\n")
        return
    _prune_interactive(index_path, blocks)
