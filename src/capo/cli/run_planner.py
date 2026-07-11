"""
Pre-orchestration RunIntent synthesis.

Runs after the questionnaire, before orch.run_sync(). Infers modality, checks
local run history for the dataset, and builds an enriched task description. That
enriched string becomes the task_description passed to run_sync() — i.e. it is
written to task.md, which the memory consultant, HF researcher and the main
agent all read. Starting them with modality + architecture + prior-run context
makes their first pass tighter and cheaper.

All inference here is best-effort and advisory; current-run artifacts always
override it downstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .colors import console

# ── modality inference ────────────────────────────────────────────────────────

_PROTEIN_CUES = re.compile(
    r"\b(protein|peptide|amino.?acid|binding|bind|esm|ankh|prot[_-]?bert|prot[_-]?t5|"
    r"antibod|cdr|epitope|sequence|fasta|uniprot|variant|fitness|solubil|thermostab)\b",
    re.I,
)
_TEXT_CUES = re.compile(r"\b(nlp|text|sentiment|biomedical.?text|pubmed|bert|gpt|llm)\b", re.I)
_TABULAR_CUES = re.compile(r"\b(tabular|csv|parquet|structured|features|xgb|gradient.?boost)\b", re.I)
_SINGLECELL_CUES = re.compile(r"\b(single.?cell|scrna|h5ad|anndata|scanpy|transcriptom)\b", re.I)

_MODALITY_LABELS = {
    "protein_sequence": "protein sequences",
    "single_cell": "single-cell RNA",
    "text": "text",
    "tabular": "tabular / structured data",
    "unknown": "unknown modality",
}

_MODALITY_MODEL_HINT = {
    "protein_sequence": "ESM2",
    "single_cell": None,
    "text": "BERT / DistilBERT",
    "tabular": None,
    "unknown": None,
}


def _infer_modality(dataset_ref: str, task_description: str) -> str:
    combined = f"{dataset_ref} {task_description}"
    if _PROTEIN_CUES.search(combined):
        return "protein_sequence"
    if _SINGLECELL_CUES.search(combined):
        return "single_cell"
    if _TEXT_CUES.search(combined):
        return "text"
    if _TABULAR_CUES.search(combined):
        return "tabular"
    return "unknown"


# ── history check ─────────────────────────────────────────────────────────────


def _check_history(dataset_ref: str, runs_root: Path) -> tuple[bool, Optional[str]]:
    """Return (seen_before, most_recent_run_id) for this dataset_ref.

    Any prior run of the dataset counts — even a failed one is worth flagging,
    since the memory consultant may hold known issues for it.
    """
    if not dataset_ref or not runs_root.exists():
        return False, None
    state_files = sorted(
        runs_root.glob("*/state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sf in state_files:
        try:
            d = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("dataset_ref") == dataset_ref:
            return True, d.get("run_id")
    return False, None


# ── RunIntent ─────────────────────────────────────────────────────────────────


@dataclass
class RunIntent:
    dataset_ref: str
    task_description: str
    enriched_description: str
    fine_tune_strategy: str
    max_cost_usd: float
    gpu_preference: Optional[str]
    inferred_modality: str
    model_hint: Optional[str]
    seen_before: bool
    prior_run_id: Optional[str]


def build_run_intent(
    dataset_ref: str,
    task_description: str,
    fine_tune_strategy: str,
    max_cost_usd: float,
    gpu_preference: Optional[str],
    model_hint: Optional[str],
    runs_root: Path,
) -> RunIntent:
    modality = _infer_modality(dataset_ref, task_description)
    resolved_model_hint = model_hint or _MODALITY_MODEL_HINT.get(modality)
    seen, prior = _check_history(dataset_ref, runs_root)

    # build the enriched description fed to the orchestrator (→ task.md).
    parts = [task_description.strip()]
    if modality != "unknown":
        parts.append(f"data modality: {_MODALITY_LABELS[modality]}")
    if resolved_model_hint:
        parts.append(f"suggested architecture: {resolved_model_hint}")
    if seen and prior:
        parts.append(f"dataset seen before (prior run: {prior}) — check memory for known issues")
    if gpu_preference:
        parts.append(f"GPU: {gpu_preference}")
    parts.append(f"strategy: {fine_tune_strategy}, budget ceiling: ${max_cost_usd:.0f}")
    enriched = "; ".join(p for p in parts if p) + "."

    return RunIntent(
        dataset_ref=dataset_ref,
        task_description=task_description,
        enriched_description=enriched,
        fine_tune_strategy=fine_tune_strategy,
        max_cost_usd=max_cost_usd,
        gpu_preference=gpu_preference,
        inferred_modality=modality,
        model_hint=resolved_model_hint,
        seen_before=seen,
        prior_run_id=prior,
    )


# ── structured task document (→ task.md) ──────────────────────────────────────

# per-modality fallbacks used only when the user/agent left evaluation blank.
_DEFAULT_EVAL = {
    "protein_sequence": (
        "Hold out a validation split (homology-aware if the data allows) and report "
        "validation loss plus task-appropriate metrics — AUROC, MCC, accuracy and F1 for "
        "classification, or Spearman/Pearson and RMSE for regression."
    ),
    "single_cell": "Report validation loss and held-out accuracy / macro-F1 on a stratified split.",
    "text": "Report validation loss and held-out accuracy / F1 on a stratified split.",
    "tabular": "Report validation loss and held-out accuracy / AUROC (RMSE for regression).",
    "unknown": "Report validation loss and task-appropriate metrics on a held-out validation split.",
}

_DEFAULT_DELIVERABLES = (
    "- Best model checkpoint, selected by the validation metric\n"
    "- Evaluation report with the final metrics\n"
    "- Training and validation curves\n"
    "- A concise run summary (decisions, cost, pitfalls)\n"
    "- Push of the best checkpoint to the Hugging Face Hub when configured"
)


def build_task_markdown(
    *,
    objective: str,
    mode: str,
    dataset_ref: str,
    fine_tune_strategy: str,
    max_cost_usd: float,
    gpu_preference: Optional[str],
    model_id: Optional[str],
    runs_root: Path,
    title: Optional[str] = None,
    organism: Optional[str] = None,
    target: Optional[str] = None,
    evaluation: Optional[str] = None,
    deliverables: Optional[str] = None,
    constraints: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[str, RunIntent]:
    """Render a structured scientific task brief (Markdown) and the RunIntent.

    The brief is what gets written to task.md — what the memory consultant, HF
    researcher and main agent read. Modality, architecture hint and prior-run
    context come from build_run_intent; everything the user/agent specified is
    slotted into the matching section, with sensible per-modality defaults for
    evaluation and deliverables. It is still free text the downstream agents
    parse, so no orchestrator contract changes — only a cleaner, fuller brief.
    """
    intent = build_run_intent(
        dataset_ref=dataset_ref,
        task_description=objective or "fine-tune a protein language model",
        fine_tune_strategy=fine_tune_strategy,
        max_cost_usd=max_cost_usd,
        gpu_preference=gpu_preference,
        model_hint=model_id or None,
        runs_root=runs_root,
    )
    modality_label = _MODALITY_LABELS.get(intent.inferred_modality, "unknown modality")
    is_pretrain = mode == "pre-train"
    title = (title or objective or "Protein model training").strip().rstrip(".") or "Protein model training"
    model_line = model_id or intent.model_hint or "auto-select (model-selection agent decides)"

    dataset = [
        f"- Reference: {dataset_ref or 'to be selected'}",
        f"- Data modality: {modality_label}",
    ]
    if organism:
        dataset.append(f"- Organism / species: {organism}")
    if target:
        dataset.append(f"- Target / property: {target}")

    strategy = [
        "- Approach: "
        + ("pre-train from a custom architecture" if is_pretrain else "fine-tune an existing model")
    ]
    if not is_pretrain:
        strategy.append(f"- Fine-tune strategy: {fine_tune_strategy}")
    strategy.append(f"- {'Architecture' if is_pretrain else 'Suggested model'}: {model_line}")
    strategy.append(f"- GPU: {gpu_preference or 'auto-select within budget'}")
    strategy.append(f"- Budget ceiling: ${max_cost_usd:.0f}")

    cons = [f"- Stay within the ${max_cost_usd:.0f} budget ceiling."]
    if constraints:
        cons.append(f"- {constraints.strip()}")
    if intent.seen_before and intent.prior_run_id:
        cons.append(
            f"- This dataset was used before (prior run: {intent.prior_run_id}); "
            "consult episodic memory for known pitfalls."
        )

    eval_body = (evaluation or _DEFAULT_EVAL.get(intent.inferred_modality, _DEFAULT_EVAL["unknown"])).strip()
    deliver_body = (deliverables or _DEFAULT_DELIVERABLES).strip()

    def _section(name: str, body: str) -> str:
        return f"## {name}\n\n{body.strip()}\n"

    parts = [
        f"# Task: {title}\n",
        _section("Objective", objective or "Train a protein language model for the stated goal."),
        _section("Dataset", "\n".join(dataset)),
        _section("Training Strategy", "\n".join(strategy)),
        _section("Evaluation", eval_body),
        _section("Deliverables", deliver_body),
        _section("Constraints", "\n".join(cons)),
    ]
    if notes:
        parts.append(_section("Notes", notes))
    return "\n".join(parts).strip() + "\n", intent


def show_run_intent(intent: RunIntent) -> None:
    """Print the RunIntent card (interactive mode only)."""
    modality_label = _MODALITY_LABELS.get(intent.inferred_modality, "unknown")

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="metric.key", no_wrap=True)
    t.add_column("Value", style="brand.dim")
    rows: list[tuple[str, str]] = [
        ("Dataset", intent.dataset_ref),
        ("Modality", modality_label),
        ("Task", intent.task_description),
        ("Strategy", intent.fine_tune_strategy),
        ("Budget", f"${intent.max_cost_usd:.0f}"),
    ]
    if intent.gpu_preference:
        rows.append(("GPU", intent.gpu_preference))
    if intent.model_hint:
        rows.append(("Model hint", intent.model_hint))
    if intent.seen_before and intent.prior_run_id:
        rows.append(("Prior run", f"[ok]{intent.prior_run_id}[/]  (reusing memory)"))
    for k, v in rows:
        t.add_row(k, v)

    console.print()
    console.print(
        Panel(
            t,
            title=Text.assemble((" Run Plan", "brand"), ("  — review before launch", "muted")),
            border_style="brand.dim",
            padding=(0, 1),
        )
    )
    console.print(
        "  [muted]Enriched context fed to orchestrator (written to task.md):[/]\n"
        f"  [brand.dim]{intent.enriched_description}[/]\n"
    )
