"""
Initial free-form intent capture.

Shown once before the questionnaire so the user can describe the task in plain
language. A light regex parser extracts hints (dataset ref, budget, strategy,
GPU, model) that pre-fill the questionnaire — the user still sees and confirms
every field, this just saves typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.panel import Panel
from rich.text import Text

from .colors import PURPLE_0, console


@dataclass
class IntentHints:
    dataset_ref: Optional[str] = None
    task_description: Optional[str] = None
    fine_tune_strategy: Optional[str] = None
    gpu_preference: Optional[str] = None
    max_cost_usd: Optional[float] = None
    model_hint: Optional[str] = None  # display only


_HF_RE = re.compile(r"\b([\w-]+/[\w._-]+)\b")
# A local dataset path or a fetch URL. Captured verbatim into dataset_ref. 
# the orchestrator's resolve_dataset_source() classifies it (local / uri) downstream.
_DATA_EXT = r"(?:csv|tsv|parquet|pq|jsonl|ndjson|json|fasta|fa|faa|fna|fastq)(?:\.gz)?"
_PATH_URL_RE = re.compile(
    r"(?<!\S)("                                    # token start
    r"(?:https?|gs|s3|ftp|file)://\S+"            # a URL
    r"|(?:\.{0,2}/|~/)\S+"                         # ./  ../  /  ~/  prefixed path
    r"|\S+\." + _DATA_EXT +                        # any token ending in a data ext
    r")(?!\S)",                                    # token end
    re.I,
)
_BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
_STRATEGY_RE = re.compile(r"\b(linear[\s_-]?probe|lora|full[\s_-]?fine[\s_-]?tun\w*)\b", re.I)
_GPU_RE = re.compile(r"\b(a10g?|a100|h100|h200|gh200|v100|l40s?|l4|4090|3090)\b", re.I)
_MODEL_RE = re.compile(r"\b(esm[\w_-]*|ankh[\w_-]*|prot[\w_-]*|bert[\w_-]*|t5[\w_-]*|llama[\w_-]*)\b", re.I)


def _parse(text: str) -> IntentHints:
    h = IntentHints(task_description=text.strip())

    # A HF owner/name id and a local path / fetch URL are equally strong dataset
    # signals, whichever the user mentions FIRST wins (position decides; no kind
    # is privileged). The resolver classifies the winner (hf / local / uri) later.
    candidates: list[tuple[int, str, bool]] = []
    pm = _PATH_URL_RE.search(text)
    if pm:
        candidates.append((pm.start(), pm.group(1).rstrip(".,;:)"), False))
    for m in _HF_RE.finditer(text):
        cand = m.group(1)
        prev = text[m.start() - 1] if m.start() > 0 else " "
        # a dataset ref is owner/name; a match preceded by / ~ . is mid-path
        # (e.g. ~/data/foo matches "data/foo") —> skip those.
        if cand.count("/") == 1 and prev not in "/~.":
            candidates.append((m.start(), cand, bool(_MODEL_RE.search(cand))))
    if candidates:
        # Tie-break away from an owner/name that looks like a model checkpoint
        # (e.g. facebook/esm2_t6_8M_UR50D) only when a real dataset signal also
        # exists, a lone model-looking id is still kept so nothing is dropped.
        non_model = [c for c in candidates if not c[2]]
        pool = non_model or candidates
        h.dataset_ref = min(pool, key=lambda c: c[0])[1]

    bm = _BUDGET_RE.search(text)
    if bm:
        h.max_cost_usd = float(bm.group(1))

    sm = _STRATEGY_RE.search(text)
    if sm:
        raw = sm.group(1).lower()
        if "linear" in raw or "probe" in raw:
            h.fine_tune_strategy = "linear-probe"
        elif "lora" in raw:
            h.fine_tune_strategy = "lora"
        elif "full" in raw:
            h.fine_tune_strategy = "full"

    gm = _GPU_RE.search(text)
    if gm:
        h.gpu_preference = gm.group(1).upper()

    mm = _MODEL_RE.search(text)
    if mm:
        h.model_hint = mm.group(1)

    return h


def parse_intent(text: str) -> IntentHints:
    """Public alias for the parser (used in tests / auto mode)."""
    return _parse(text)


def capture_intent() -> IntentHints:
    """Show the intent prompt, parse the response, echo what was extracted."""
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("  What would you like to fine-tune?\n\n", "brand"),
                ("  Describe the dataset, task, model preference, budget — ", "muted"),
                ("anything you already know.\n", "muted"),
                ("  Leave out what you don't know; CAPO will ask.\n", "muted"),
            ),
            border_style="brand.dim",
            padding=(0, 1),
        )
    )
    console.print()

    try:
        text = pt_prompt(
            HTML(f'<style fg="{PURPLE_0}"><b>  ❯ </b></style>'),
            history=InMemoryHistory(),
            multiline=False,
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[muted]Aborted.[/]\n")
        raise SystemExit(0)

    if not text.strip():
        return IntentHints()

    h = _parse(text)

    picked: list[tuple[str, str]] = []
    if h.dataset_ref:
        picked.append(("dataset", h.dataset_ref))
    if h.max_cost_usd:
        picked.append(("budget", f"${h.max_cost_usd:.0f}"))
    if h.fine_tune_strategy:
        picked.append(("strategy", h.fine_tune_strategy))
    if h.gpu_preference:
        picked.append(("GPU", h.gpu_preference))
    if h.model_hint:
        picked.append(("model", h.model_hint))

    if picked:
        parts = "   ".join(f"[metric.key]{k}[/] [brand.dim]{v}[/]" for k, v in picked)
        console.print(f"  [muted]Understood:[/]  {parts}")
    console.print()

    return h
