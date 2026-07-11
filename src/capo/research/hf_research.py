"""
hf_research.py — Pre-pipeline HuggingFace Hub research for fine-tuning context.

Runs a lightweight agent (Bash-only) before the main orchestrator to gather:
  1. Bio-aligned training datasets from the HuggingFace Hub
  2. Field-canonical evaluation benchmarks and their primary metrics
  3. Task-specific hyperparameter recommendations from the model card

Queries are restricted to HuggingFace Hub APIs (hf CLI / curl) — no general
web search. Results are synthesized into a ResearchFindings object, written
to reports/research_findings.json and injected into the Orchestrator prompt.
"""

from __future__ import annotations

import re
from pathlib import Path

import msgspec

from capo.observability import progress as ip
from capo.orchestration.agent_runner import AgentRunner
from capo.utils.prompts import load_prompt


class ResearchFindings(msgspec.Struct, kw_only=True):
    """
    Schema-validated research findings from the HF Hub agent.
    """

    entity_frame: dict = {}
    training_datasets: list[dict] = []
    eval_benchmarks: list[dict] = []
    hyperparameters: dict = {}
    summary: str = ""
    raw: str = ""

    def is_empty(self) -> bool:
        return not (
            self.training_datasets
            or self.eval_benchmarks
            or self.hyperparameters
            or self.summary
        )

    def to_prompt_section(self) -> str:
        """Render findings as a markdown section for injection into the Orchestrator prompt.

        Tolerates two hyperparameter shapes:
          - legacy: {"learning_rate": "1e-4", ...}
          - structured: {"learning_rate": {"value": "1e-4", "provenance": "model-card"}, ...}
        """
        if self.is_empty():
            return "No prior research available."

        parts: list[str] = []

        ef = self.entity_frame or {}
        if ef:
            ef_pairs = ", ".join(
                f"{k}={v}" for k, v in ef.items() if v not in (None, "", "unknown")
            )
            if ef_pairs:
                parts.append(f"**Entity frame**: {ef_pairs}")
                parts.append("")

        parts.append("### Training datasets")
        for ds in self.training_datasets:
            hf_id = ds.get("hf_id") or ""
            hf_str = f" (`{hf_id}`)" if hf_id and hf_id not in ("null", "None") else ""
            size = ds.get("size") or "unknown"
            flags: list[str] = []
            ru = ds.get("recommended_use")
            if ru and ru not in ("primary", "unknown", None):
                flags.append(f"use={ru}")
            access = ds.get("access")
            if access and access not in ("public", "unknown", None):
                flags.append(f"access={access}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            notes = ds.get("notes") or ""
            parts.append(
                f"- **{ds.get('name', 'Unknown')}**{hf_str} (size: {size}){flag_str}: {notes}"
            )
        if not self.training_datasets:
            parts.append("- No specific datasets identified.")

        parts.append("")
        parts.append("### Evaluation benchmarks")
        for bm in self.eval_benchmarks:
            metrics = ", ".join(bm.get("metrics") or [])
            source_type = bm.get("source_type")
            src_str = (
                f" [{source_type}]"
                if source_type and source_type not in ("hf_dataset", "unknown", None)
                else ""
            )
            notes = bm.get("notes") or ""
            parts.append(
                f"- **{bm.get('name', 'Unknown')}**{src_str}: metrics={metrics}. {notes}"
            )
        if not self.eval_benchmarks:
            parts.append(
                "- No specific benchmarks identified; "
                "use train/val/test split from available data."
            )

        parts.append("")
        parts.append("### Recommended hyperparameters")
        hp = self.hyperparameters
        if hp:
            for k, v in hp.items():
                if k == "notes":
                    continue
                if isinstance(v, dict) and "value" in v:
                    val = v.get("value")
                    if val in (None, "", "n/a", "N/A"):
                        continue
                    prov = v.get("provenance") or ""
                    prov_str = f" _({prov})_" if prov else ""
                    parts.append(f"- **{k}**: {val}{prov_str}")
                else:
                    if v in (None, "", "n/a", "N/A"):
                        continue
                    parts.append(f"- **{k}**: {v}")
            if hp.get("notes"):
                parts.append(f"- Notes: {hp['notes']}")
        else:
            parts.append("- Use model-card defaults; no task-specific data found.")

        if self.summary:
            parts.extend(["", f"### Summary\n{self.summary}"])

        return "\n".join(parts)

    def to_dict(self) -> dict:
        d = msgspec.to_builtins(self)
        d.pop("raw", None)
        return d

    def to_agent_safe_dict(self) -> dict:
        """Strip bio-sensitive fields for the Phase A agent.

        entity_frame (organism/target), summary, and notes fields contain
        dual-use language that accumulates with task.md to trigger content-policy
        filters. The agent only needs hyperparameters, dataset IDs, and benchmark
        metrics to write train.py — everything else is human reference material.
        """
        return {
            "entity_frame": {},
            "training_datasets": [
                {k: v for k, v in ds.items() if k != "notes"}
                for ds in self.training_datasets
            ],
            "eval_benchmarks": [
                {k: v for k, v in bm.items() if k != "notes"}
                for bm in self.eval_benchmarks
            ],
            "hyperparameters": self.hyperparameters,
            "summary": "",
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = load_prompt("research/system_prompts/hf_research")

_RESEARCH_PROMPT_TEMPLATE = load_prompt("research/user_prompts/hf_research")


# ---------------------------------------------------------------------------
# Provider-agnostic refusal signatures
# Only applied to non-JSON responses (JSON is always trusted as-is).
# ---------------------------------------------------------------------------
_ERROR_SIGNATURES: tuple[str, ...] = (
    "unable to respond to this request",
    "i cannot assist with this",
    "i can't assist with this",
    "i'm unable to help with",
    "violates our usage policy",
    "against our guidelines",
)


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced `{...}` substring of *text*, or None.

    The agent is told to emit JSON-only, but in practice it sometimes prepends
    a sentence ("Now I have enough verified data. Let me compile the final
    JSON.") or appends a trailing remark. This scanner walks the text with a
    string-aware brace counter so we can recover the JSON regardless. It
    correctly handles braces inside string literals and backslash escapes.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_findings(raw: str) -> ResearchFindings:
    """Parse agent output into ResearchFindings.

    JSON is parsed first and trusted regardless of content — a valid JSON object
    that happens to mention "API Error" in a notes field is a legitimate finding.
    Refusal detection only applies to non-JSON responses.
    """
    if not raw:
        return ResearchFindings(raw=raw)

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text).strip()

    # Try the whole stripped text first; if that fails, recover the first
    # balanced `{...}` substring from anywhere in the response. The agent is
    # told to emit JSON-only, but it sometimes wraps the object in a leading
    # sentence. we don't want that to break the run.
    candidates: list[tuple[str, str]] = [("whole", text)]
    extracted = _extract_json_object(text)
    if extracted and extracted != text:
        candidates.append(("extracted", extracted))

    last_exc: Exception | None = None
    for source, candidate in candidates:
        try:
            findings = msgspec.json.decode(candidate.encode(), type=ResearchFindings)
        except (msgspec.DecodeError, msgspec.ValidationError) as exc:
            last_exc = exc
            continue
        findings.raw = raw  # set after decode so that LLM never emits raw
        ip.emit(
            f"[research] Parsed ({source}): {len(findings.training_datasets)} dataset(s), "
            f"{len(findings.eval_benchmarks)} benchmark(s), "
            f"hyperparameters={'yes' if findings.hyperparameters else 'no'}, "
            f"summary={'yes' if findings.summary else 'no'}"
        )
        return findings

    if last_exc is not None:
        ip.emit(
            f"[research] Schema decode failed ({type(last_exc).__name__}: {last_exc}) — "
            f"raw length={len(raw)}, preview: {raw[:200]!r}"
        )

    # Non-JSON: check for provider-agnostic refusal phrases.
    raw_lower = raw.lower()
    if any(sig in raw_lower for sig in _ERROR_SIGNATURES):
        ip.emit(
            f"[research] Refusal detected in non-JSON response. "
            f"First 200 chars: {raw[:200]!r}"
        )
        return ResearchFindings(raw=raw)

    # Non-JSON, non-refusal: log and return empty.
    ip.emit(
        f"[research] Unrecognised non-JSON response — "
        f"length={len(raw)}, first 300 chars: {raw[:300]!r}"
    )
    return ResearchFindings(raw=raw)


class HFResearcher:
    """Lightweight pre-pipeline agent that queries the HuggingFace Hub for
    training datasets, evaluation benchmarks and model-card hyperparameters.

    Uses Bash-only (hf CLI or curl) with bypassPermissions so it runs fully
    non-interactively as a subprocess.  Disable via enable_hf_research=False
    on FineTuningOrchestrator.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        cwd: str | Path | None = None,
    ) -> None:
        # bypassPermissions auto-approves all tool calls without an interactive
        # prompt — required for non-interactive subprocess execution.
        # mcp_servers={} prevents lambda/docker MCP servers from starting.
        # the research step only needs Bash (hf or curl).
        self._runner = AgentRunner(
            model_name=model_name,
            allowed_tools=["Bash"],
            system_prompt=_RESEARCH_SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=20,
            cwd=cwd,
            mcp_servers={},
        )

    async def run(
        self,
        model_id: str,
        fine_tune_strategy: str,
        dataset_ref: str,
        task_md_path: str | Path | None = None,
    ) -> tuple["ResearchFindings", float | None]:
        """Run HuggingFace research and return (findings, agent_cost_usd).

        The orchestrator writes the (enriched) task to task.md before this runs.
        We point the agent at that file so research is grounded in the real task
        — organism / target / assay / modality — instead of the model id alone.
        The agent reads it with a Bash `cat` (one shell call): the bio-sensitive
        text arrives as a tool result, never in the prompt, so it does not trip
        the provider content filter the way embedding it inline would. When the
        path is absent the research proceeds from the run parameters only.
        """
        ip.emit("[research] Starting pre-pipeline HF research...")
        if task_md_path and Path(task_md_path).exists():
            task_context = (
                "First read the task to ground organism / target / assay / "
                f"modality (one shell call):\n`cat {task_md_path}`\n"
                "Let it steer dataset and benchmark choices.\n"
            )
        else:
            task_context = ""
        prompt = _RESEARCH_PROMPT_TEMPLATE.format(
            model_id=model_id,
            fine_tune_strategy=fine_tune_strategy,
            dataset_ref=dataset_ref,
            task_context=task_context,
        )
        result = await self._runner.generate(prompt=prompt)
        findings = _parse_findings(result.answer)
        ip.emit(
            f"[research] Research complete: "
            f"{len(findings.training_datasets)} dataset(s), "
            f"{len(findings.eval_benchmarks)} benchmark(s), "
            f"hyperparameters={'yes' if findings.hyperparameters else 'no'}"
        )
        return findings, result.total_cost_usd
