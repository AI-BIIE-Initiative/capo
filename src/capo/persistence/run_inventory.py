"""
run_inventory.py — read-only census of a fine-tuning run directory.

Resume was historically checkpoint-only: with no *training* checkpoint, the
resume path had "nothing to do" — even when an expensive earlier stage (Boltz
embedding generation) had left valid artifacts on disk. The run that motivated
this module died 20% into its only expensive stage with 13 valid ~100 MB
embeddings_*.npz on disk; a re-run would have recomputed all of it.

This module takes a cheap, read-only inventory of what a run actually produced,
classifies each stage (complete | partial | missing | failed), and emits a
stage-level resume plan that REUSES valid artifacts and recomputes only what is
missing.

Design constraints: 
  * Pure & read-only: build_inventory / valid_embedding / plan_resume never
    write, delete, launch, or mutate anything. The only writer is
    write_run_state, which emits a *derived* run_state.json that is always
    regenerable from disk — so a missing/stale run_state.json is never fatal.
  * No new on-disk layout: reads the existing standardized run dir only.
  * Backward compatible: older run dirs (no run_state.json, no recipe sidecar)
    inventory correctly; missing optional inputs degrade, never block.
  * Model-agnostic: a non-Boltz run has no YAML/embedding artifacts, so the
    embeddings stage is simply absent and the plan falls back to checkpoint /
    fresh resume. Boltz embedding reuse is the high-value special case, not a
    hard dependency.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Stage status vocabulary.
COMPLETE = "complete"
PARTIAL = "partial"
MISSING = "missing"
FAILED = "failed"

# A real boltz embedding is ~100 MB; anything under this floor is a truncated
# write or a stub, not a reusable artifact.
_MIN_EMBEDDING_BYTES = 1024
# The big pair-representation member (z) is ~100 MB; even the small single
# representation (s) is ~0.7 MB. Require at least one substantial array member
# so a near-empty zip is rejected without reading any array bytes.
_MIN_ARRAY_BYTES = 10_000

# Probe-measured boltz predict throughput (fixes.md §1: ~23 s/complex on A100).
# Used only for the advisory CLI cost estimate — never a gate.
_DEFAULT_SEC_PER_COMPLEX = 23.0
_DEFAULT_GPU_HOURLY_USD = 1.99  # A100 SXM4 fallback when infra/pricing absent


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class StageInfo:
    status: str
    expected: int = 0
    done: int = 0
    missing: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        d = {"status": self.status, "detail": self.detail}
        if self.expected or self.done or self.missing:
            d.update(expected=self.expected, done=self.done, missing=self.missing)
        return d


@dataclass
class RunInventory:
    run_dir: Path
    stages: dict[str, StageInfo]
    expected_complexes: list[str]
    done_complexes: list[str]
    missing_complexes: list[str]
    predictions_dir: Path | None
    processed_inputs: int
    has_checkpoint: bool
    infra: dict | None
    last_error: dict | None
    terminal_state: str | None
    current_phase: str | None

    @property
    def is_boltz_embedding_run(self) -> bool:
        return bool(self.expected_complexes)

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "expected": len(self.expected_complexes),
            "done": len(self.done_complexes),
            "missing": len(self.missing_complexes),
            "processed_inputs": self.processed_inputs,
            "has_checkpoint": self.has_checkpoint,
            "infra": self.infra,
            "last_error": self.last_error,
            "terminal_state": self.terminal_state,
            "current_phase": self.current_phase,
        }


@dataclass
class StageAction:
    action: str
    detail: str
    targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"action": self.action, "detail": self.detail}
        if self.targets:
            d["targets"] = self.targets
        return d


@dataclass
class ResumePlan:
    actions: list[StageAction]
    reuse: list[str]
    recompute: list[str]
    next_resume_point: str
    est_recompute_cost_usd: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "reuse": self.reuse,
            "recompute": self.recompute,
            "next_resume_point": self.next_resume_point,
            "est_recompute_cost_usd": self.est_recompute_cost_usd,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Low-level probes (all read-only)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _expected_complexes(run_dir: Path, prefer: str = "train_val") -> list[str]:
    """The expected complex name-set = stems of the written train/val YAMLs."""
    yroot = run_dir / "outputs" / "yamls"
    for sub in (prefer, "."):
        d = yroot / sub if sub != "." else yroot
        if d.is_dir():
            stems = sorted(p.stem for p in d.glob("*.yaml"))
            if stems:
                return stems
    return []


def _find_predictions_dir(run_dir: Path, prefer: str = "train_val") -> Path | None:
    """Locate the boltz predictions/ dir (per-complex outputs live under it)."""
    emb = run_dir / "outputs" / "embeddings"
    if not emb.is_dir():
        return None
    cands = [p for p in emb.rglob("predictions") if p.is_dir()]
    if not cands:
        return None
    preferred = [p for p in cands if prefer in str(p)]
    return (preferred or cands)[0]


def valid_embedding(predictions_dir: Path, name: str) -> bool:
    """Cheap, pragmatic validity check for one boltz embedding (fixes.md §C).

    Existence + size floor + intact-zip-directory + a substantial array member.
    Reads only the zip central directory (microseconds), never the ~100 MB of
    array data, and never loads tensors into RAM. A truncated/half-synced write
    fails ZipFile() open → correctly invalid. Boltz-2 writes s and z members;
    we require any non-trivial member rather than exact names, to stay robust
    across boltz versions.
    """
    npz = Path(predictions_dir) / name / f"embeddings_{name}.npz"
    try:
        if not npz.is_file() or npz.stat().st_size < _MIN_EMBEDDING_BYTES:
            return False
        with zipfile.ZipFile(npz) as zf:
            infos = zf.infolist()
            if not infos:
                return False
            return any(info.file_size >= _MIN_ARRAY_BYTES for info in infos)
    except (zipfile.BadZipFile, OSError):
        return False


def _count_processed(run_dir: Path, prefer: str = "train_val") -> int:
    """Count tokenised/preprocessed structure inputs (boltz preprocessing)."""
    emb = run_dir / "outputs" / "embeddings"
    if not emb.is_dir():
        return 0
    npzs = list(emb.rglob("processed/structures/*.npz"))
    preferred = [p for p in npzs if prefer in str(p)]
    return len(preferred or npzs)


def _has_checkpoint(run_dir: Path) -> bool:
    """True iff a non-empty checkpoint file exists locally. Empty best/last dirs
    (training never started) correctly return False."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    for p in ckpt_dir.rglob("*"):
        try:
            if p.is_file() and p.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _gpu_hourly_usd(infra: dict | None) -> float:
    if infra:
        for k in ("hourly_cost_usd", "hourly_usd", "price_per_hour", "hourly_rate"):
            v = infra.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return _DEFAULT_GPU_HOURLY_USD


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_inventory(run_dir: Path) -> RunInventory:
    """Read-only census of a run directory. Never mutates anything."""
    run_dir = Path(run_dir)
    state = _read_json(run_dir / "state.json") or {}
    infra = _read_json(run_dir / "infra.json")
    last_error = _read_json(run_dir / "reports" / "post_launch_failure.json")

    expected = _expected_complexes(run_dir)
    preds_dir = _find_predictions_dir(run_dir)
    done = (
        [n for n in expected if valid_embedding(preds_dir, n)]
        if expected and preds_dir is not None
        else []
    )
    done_set = set(done)
    missing = [n for n in expected if n not in done_set]
    processed = _count_processed(run_dir)
    has_ckpt = _has_checkpoint(run_dir)

    stages: dict[str, StageInfo] = {}

    # Embeddings stage — boltz-specific; present only when YAMLs were written.
    if expected:
        if done and len(done) == len(expected):
            emb_status = COMPLETE
        elif done:
            emb_status = PARTIAL
        else:
            emb_status = MISSING
        stages["embeddings"] = StageInfo(
            status=emb_status,
            expected=len(expected),
            done=len(done),
            missing=missing,
            detail=f"{len(done)}/{len(expected)} valid embeddings; "
            f"{processed} inputs preprocessed",
        )

    # Training stage — works for any model: checkpoint present ⇒ resumable.
    stages["training"] = StageInfo(
        status=PARTIAL if has_ckpt else MISSING,
        detail="checkpoint present" if has_ckpt else "no local checkpoint",
    )

    # Evaluation stage.
    results = run_dir / "results" / "metrics.json"
    stages["evaluation"] = StageInfo(
        status=COMPLETE if results.exists() else MISSING,
        detail="results/metrics.json present" if results.exists() else "no eval metrics",
    )

    return RunInventory(
        run_dir=run_dir,
        stages=stages,
        expected_complexes=expected,
        done_complexes=done,
        missing_complexes=missing,
        predictions_dir=preds_dir,
        processed_inputs=processed,
        has_checkpoint=has_ckpt,
        infra=infra,
        last_error=last_error,
        terminal_state=state.get("terminal_state"),
        current_phase=state.get("current_phase"),
    )


def _estimate_cost(n_missing: int, inv: RunInventory) -> float:
    rate = _gpu_hourly_usd(inv.infra)
    return round(n_missing * _DEFAULT_SEC_PER_COMPLEX / 3600.0 * rate, 2)


def plan_resume(inv: RunInventory) -> ResumePlan:
    """Turn an inventory into an ordered, reuse-first resume plan (fixes.md §B).

    Invariant: a valid artifact is NEVER scheduled for recompute. --override
    is never part of any action — the executor must run boltz predict on the
    missing names only.
    """
    actions: list[StageAction] = []
    reuse: list[str] = []
    recompute: list[str] = []
    notes: list[str] = []
    est_cost = 0.0

    emb = inv.stages.get("embeddings")
    if emb is not None:
        if emb.status == COMPLETE:
            actions.append(
                StageAction("skip_embeddings", f"all {emb.expected} embeddings valid — reuse")
            )
            reuse += list(inv.done_complexes)
        elif emb.status == PARTIAL:
            actions.append(
                StageAction(
                    "resume_embeddings",
                    f"boltz predict on {len(emb.missing)} missing complexes "
                    f"(reuse {emb.done}; no --override)",
                    targets=list(emb.missing),
                )
            )
            reuse += list(inv.done_complexes)
            recompute += list(emb.missing)
            est_cost = _estimate_cost(len(emb.missing), inv)
        else:  # MISSING
            actions.append(
                StageAction(
                    "run_embeddings",
                    f"compute all {emb.expected} embeddings (none on disk)",
                    targets=list(emb.missing),
                )
            )
            recompute += list(emb.missing)
            est_cost = _estimate_cost(len(emb.missing), inv)

    # Training.
    if inv.has_checkpoint:
        actions.append(
            StageAction("resume_training_from_checkpoint", "resume training from checkpoint")
        )
    elif emb is not None and emb.status in (COMPLETE, PARTIAL):
        actions.append(
            StageAction("start_training_from_embeddings", "train head on (reused) embeddings")
        )
    elif emb is None:
        # Non-Boltz run with no checkpoint and no embeddings → no cheap resume.
        notes.append(
            "No reusable embeddings or checkpoint found — a resume would re-run from scratch."
        )
        actions.append(
            StageAction("rerun_from_start", "no reusable expensive artifacts; re-run the pipeline")
        )

    # Evaluation.
    ev = inv.stages.get("evaluation")
    if ev is not None and ev.status != COMPLETE and any(
        a.action != "rerun_from_start" for a in actions
    ):
        actions.append(StageAction("run_eval", "run evaluation after training"))

    # next_resume_point = first action that does real work (skip is a no-op).
    next_point = "nothing"
    for a in actions:
        if not a.action.startswith("skip_"):
            next_point = a.action
            break

    return ResumePlan(
        actions=actions,
        reuse=reuse,
        recompute=recompute,
        next_resume_point=next_point,
        est_recompute_cost_usd=est_cost,
        notes=notes,
    )


def format_plan(inv: RunInventory, plan: ResumePlan) -> str:
    """Render the human-facing resume card (fixes.md §H)."""
    lines: list[str] = []
    lines.append(f"Resuming {inv.run_dir.name}")
    lines.append("  Detected artifacts:")
    emb = inv.stages.get("embeddings")
    if emb is not None:
        verdict = "reuse" if emb.done else "recompute"
        lines.append(
            f"    embeddings : {emb.done}/{emb.expected} valid"
            f"  ({len(emb.missing)} missing)        ← {verdict}"
        )
        lines.append(
            f"    processed  : {inv.processed_inputs}/{emb.expected} inputs parsed              ← reuse"
        )
    ckpt = "present → resume from checkpoint" if inv.has_checkpoint else "none → train from embeddings"
    lines.append(f"    checkpoints: {ckpt}")
    lines.append("  Plan:")
    for a in plan.actions:
        lines.append(f"    {a.action.upper():<32} {a.detail}")
    if plan.recompute:
        lines.append(
            f"  Recompute: {len(plan.recompute)} complexes "
            f"(~${plan.est_recompute_cost_usd:.2f} of GPU time)"
        )
    if plan.reuse:
        lines.append(
            f"  Recompute risk: LOW — {len(plan.reuse)} valid artifacts will be REUSED, not recomputed."
        )
    for note in plan.notes:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


def write_run_state(run_dir: Path, inv: RunInventory, plan: ResumePlan) -> Path:
    """Write the *derived* run_state.json (fixes.md §F).

    Atomic (tmp + rename). Derived and regenerable from disk via
    build_inventory, so it is a convenience for the CLI/finalizer, never a
    source of truth. No lock: this is written single-writer at resume/finalize
    time, and a torn read just gets regenerated.
    """
    run_dir = Path(run_dir)
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {k: v.to_dict() for k, v in inv.stages.items()},
        "expected": len(inv.expected_complexes),
        "done": len(inv.done_complexes),
        "missing": inv.missing_complexes,
        "processed_inputs": inv.processed_inputs,
        "has_checkpoint": inv.has_checkpoint,
        "infra": inv.infra,
        "last_error": inv.last_error,
        "terminal_state": inv.terminal_state,
        "current_phase": inv.current_phase,
        "next_resume_point": plan.next_resume_point,
        "plan": plan.to_dict(),
    }
    dest = run_dir / "run_state.json"
    tmp = run_dir / "run_state.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, dest)
    return dest
