"""Resolve the user's model specification → run model selection only when needed.

CAPO's model-selector subagent is expensive and unnecessary when the config
already pins a deterministic choice. This module classifies model_id (and an
optional architecture) into one of four modes and decides whether to bypass
the selector:

- **exact**  — a concrete HF repo id (facebook/esm2_t6_8M_UR50D) or an exact
  registry id (esm2_t6_8m) → bypass, use it directly (validated at probe).
- **architecture** — a family/architecture token (esm2, ankh):
  one registry match → bypass and use it; several → run the selector to choose
  among exactly those candidates.
- **custom** — model_id: custom → bypass the registry; the path/import is
  validated downstream.
- **auto** — nothing pinned (null/empty/auto) → run the selector.

Pure functions over the model registry JSONL — no network, no agent calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# skills/model-selection/model_registry/registry_src/models.jsonl
# This module lives at src/capo/utils/model_resolution.py → repo root is 3 up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = (
    _REPO_ROOT
    / "skills"
    / "model-selection"
    / "model_registry"
    / "registry_src"
    / "models.jsonl"
)

_AUTO_TOKENS = {"", "auto", "none", "null"}


@dataclass
class ModelResolution:
    """Outcome of resolving the user's model spec."""

    mode: str                                   # exact | architecture | custom | auto
    bypass_selection: bool
    resolved_model_id: str | None = None        # concrete HF repo id when known
    registry_entry: dict | None = None          # the matched registry row, if any
    candidates: list[dict] = field(default_factory=list)  # >1 match → selector chooses
    fine_tune_strategy: str | None = None
    needs_validation: bool = False              # availability/import checked downstream
    reason: str = ""


def load_registry(path: Path | str | None = None) -> list[dict]:
    """Load the model registry JSONL into a list of dicts (best-effort)."""
    p = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _looks_like_hf_repo_id(model_id: str) -> bool:
    """An org/name HF repo id (one slash, no spaces)."""
    return "/" in model_id and " " not in model_id.strip()


def find_by_hf_id(registry: list[dict], hf_id: str) -> dict | None:
    target = hf_id.strip().lower()
    for row in registry:
        if str(row.get("hf_repo_id", "")).strip().lower() == target:
            return row
    return None


def find_by_registry_id(registry: list[dict], rid: str) -> dict | None:
    target = rid.strip().lower()
    for row in registry:
        if str(row.get("registry_id", "")).strip().lower() == target:
            return row
    return None


def match_architecture(registry: list[dict], token: str) -> list[dict]:
    """Registry rows whose family / architecture / id mentions token.

    Matches when the lowercased token is a substring of registry_id or
    hf_repo_id, or equals family / architecture_type. Result order
    follows the registry file order (stable).
    """
    t = token.strip().lower()
    if not t:
        return []
    out: list[dict] = []
    for row in registry:
        rid = str(row.get("registry_id", "")).lower()
        hf = str(row.get("hf_repo_id", "")).lower()
        fam = str(row.get("family", "")).lower()
        arch = str(row.get("architecture_type", "")).lower()
        if t in rid or t in hf or t == fam or t == arch:
            out.append(row)
    return out


def _candidate_from_entry(entry: dict, fine_tune_strategy: str | None) -> dict:
    """Shape a registry row into the model_selection.json candidate schema."""
    pc = entry.get("parameter_count")
    return {
        "model_id": entry.get("hf_repo_id"),
        "registry_id": entry.get("registry_id"),
        "param_count_b": (round(pc / 1e9, 4) if isinstance(pc, (int, float)) else None),
        "min_vram_gb": None,
        "fine_tune_strategy": fine_tune_strategy,
        "lora_r": None,
        "driver_script": None,
        "score": None,
        "flags": entry.get("exclusion_tags") or [],
        "rationale": "user-specified model (selection bypassed)",
    }


def resolve_model(
    model_id: str | None,
    *,
    architecture: str | None = None,
    fine_tune_strategy: str | None = None,
    registry: list[dict] | None = None,
    registry_path: Path | str | None = None,
) -> ModelResolution:
    """Classify the model spec and decide whether to run the selector."""
    if registry is None:
        registry = load_registry(registry_path)

    spec = (model_id or "").strip()
    spec_l = spec.lower()

    # Architecture given explicitly and no concrete model pinned → architecture path.
    if architecture and spec_l in _AUTO_TOKENS:
        return _resolve_architecture(architecture, fine_tune_strategy, registry)

    # Case D — nothing pinned → run model selection.
    if spec_l in _AUTO_TOKENS:
        return ModelResolution(
            mode="auto",
            bypass_selection=False,
            fine_tune_strategy=fine_tune_strategy,
            reason="no model specified → run model selection",
        )

    # Case C — custom model → bypass registry, validate downstream.
    if spec_l == "custom":
        return ModelResolution(
            mode="custom",
            bypass_selection=True,
            resolved_model_id=None,
            fine_tune_strategy=fine_tune_strategy,
            needs_validation=True,
            reason="custom model → bypass registry; validate path/import/config downstream",
        )

    # Case A — exact HF repo id.
    if _looks_like_hf_repo_id(spec):
        entry = find_by_hf_id(registry, spec)
        return ModelResolution(
            mode="exact",
            bypass_selection=True,
            resolved_model_id=spec,
            registry_entry=entry,
            fine_tune_strategy=fine_tune_strategy,
            needs_validation=True,
            reason=(
                "exact HF model id → bypass model selection; validate availability"
                + ("" if entry else " (not in registry — treated as a custom HF model)")
            ),
        )

    # Exact registry id (e.g. "esm2_t6_8m") → deterministic.
    entry = find_by_registry_id(registry, spec)
    if entry is not None:
        return ModelResolution(
            mode="exact",
            bypass_selection=True,
            resolved_model_id=entry.get("hf_repo_id"),
            registry_entry=entry,
            fine_tune_strategy=fine_tune_strategy,
            needs_validation=True,
            reason=f"exact registry id '{spec}' → bypass model selection",
        )

    # Case B — architecture/family token.
    return _resolve_architecture(spec, fine_tune_strategy, registry)


def _resolve_architecture(
    token: str,
    fine_tune_strategy: str | None,
    registry: list[dict],
) -> ModelResolution:
    matches = match_architecture(registry, token)
    if not matches:
        return ModelResolution(
            mode="auto",
            bypass_selection=False,
            fine_tune_strategy=fine_tune_strategy,
            reason=f"architecture '{token}' matched no registry entry → run model selection",
        )
    if len(matches) == 1:
        only = matches[0]
        return ModelResolution(
            mode="architecture",
            bypass_selection=True,
            resolved_model_id=only.get("hf_repo_id"),
            registry_entry=only,
            fine_tune_strategy=fine_tune_strategy,
            needs_validation=True,
            reason=f"architecture '{token}' → single registry match → bypass model selection",
        )
    return ModelResolution(
        mode="architecture",
        bypass_selection=False,
        candidates=matches,
        fine_tune_strategy=fine_tune_strategy,
        reason=(
            f"architecture '{token}' matched {len(matches)} registry entries "
            "→ run model selection among those candidates"
        ),
    )


def build_model_selection_json(resolution: ModelResolution) -> dict:
    """Build a minimal model_selection.json for a bypassed selection.

    Only valid for bypass_selection resolutions that pin a concrete model
    (exact / single-architecture-match / custom). The downstream main agent
    reads preferred → the candidate's model_id + fine_tune_strategy.
    """
    if resolution.mode == "custom":
        candidate = {
            "model_id": "custom",
            "registry_id": None,
            "param_count_b": None,
            "min_vram_gb": None,
            "fine_tune_strategy": resolution.fine_tune_strategy,
            "lora_r": None,
            "driver_script": None,
            "score": None,
            "flags": ["custom"],
            "rationale": "custom model specified by user (selection bypassed)",
        }
    elif resolution.registry_entry is not None:
        candidate = _candidate_from_entry(
            resolution.registry_entry, resolution.fine_tune_strategy
        )
    else:
        # exact HF id not in registry
        candidate = {
            "model_id": resolution.resolved_model_id,
            "registry_id": None,
            "param_count_b": None,
            "min_vram_gb": None,
            "fine_tune_strategy": resolution.fine_tune_strategy,
            "lora_r": None,
            "driver_script": None,
            "score": None,
            "flags": [],
            "rationale": "exact HF model id specified by user (selection bypassed)",
        }
    return {
        "best_fit": candidate,
        "budget": None,
        "frontier": None,
        "preferred": "best_fit",
        "preferred_rationale": resolution.reason,
        "user_preference_match": True,
        "scoring_note": "model selection bypassed (user-specified model)",
        "selection_bypassed": True,
    }
