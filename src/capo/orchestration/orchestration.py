from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from capo.orchestration.agent_runner import AgentRunResult, AgentRunner
from capo.remote import lambda_session as ls
from capo.remote import tmux_manager as tm
from capo.remote import rsync_manager as rm
from capo.remote.run_manager import (
    RunSpec,
    RunStatus,
    prepare_remote_run_dir,
    write_remote_spec,
    start_remote_inference,
    start_remote_finetune,
    read_remote_run_status,
    get_remote_run_paths,
)
from capo.remote.config import (
    LOCAL_TMUX_SESSION,
    LOCAL_WINDOW_REMOTE,
    LOCAL_WINDOW_SYNC,
    LOCAL_WINDOW_LOCAL,
    REMOTE_TMUX_SESSION,
    REMOTE_RUN_ROOT,
    LOCAL_ARTIFACTS_ROOT,
)
from capo.observability import progress as ip
from capo.utils.prompts import load_prompt


# ---------------------------------------------------------------------------
# Lambda workflow dataclass
# ---------------------------------------------------------------------------

@dataclass
class LambdaWorkflowResult:
    run_id: str
    instance_id: str
    ssh_alias: str
    local_run_dir: Path
    status: RunStatus | None = None
    summary: dict | None = None


# ---------------------------------------------------------------------------
# Lambda workflow entry points
# ---------------------------------------------------------------------------

def start_lambda_inference_workflow(
    instance_type: str,
    ssh_key_name: str,
    key_path: str | Path,
    command: str,
    model_name: str,
    local_workdir: str | Path,
    run_config: dict | None = None,
    run_id: str | None = None,
    region: str | None = None,
    api_key: str | None = None,
    artifacts_dir: str | Path | None = None,
) -> LambdaWorkflowResult:
    """
    Full inference workflow:
    1. provision_instance
    2. wait_for_instance_ip
    3. wait_for_ssh_ready
    4. ensure_ssh_alias  →  ssh_alias = "lambda-<instance_id>"
    5. ensure_local_workspace (capo session, 3 windows)
    6. ensure_remote_tmux (capo_remote)
    7. prepare_remote_run_dir + write_remote_spec
    8. upload_run_inputs via sync window
    9. start_remote_inference → sends to capo_remote via remote window
    10. Returns LambdaWorkflowResult (monitoring/fetch separate)
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    key_path = Path(key_path)
    local_run_dir = Path(artifacts_dir or LOCAL_ARTIFACTS_ROOT) / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    # 1–3: Provision + wait
    ip.emit(f"Creating Lambda instance {instance_type} (run_id={run_id})")
    instance = ls.provision_instance(instance_type, ssh_key_name, region=region, api_key=api_key)
    instance_ip = ls.wait_for_instance_ip(instance.instance_id, api_key=api_key)
    ls.wait_for_ssh_ready(instance_ip, key_path)

    # 4: SSH alias
    ssh_alias = f"lambda-{instance.instance_id}"
    ls.ensure_ssh_alias(ssh_alias, instance_ip, key_path)

    # 5: Local tmux workspace
    ip.emit("[tmux] launching local workspace")
    tm.ensure_local_workspace()

    # 6: Remote tmux
    ip.emit(f"[tmux] ensuring remote session on {ssh_alias}")
    tm.ensure_remote_tmux(ssh_alias, key_path=str(key_path))

    # 7: Run directory + spec
    ip.emit(f"Preparing remote run directory {run_id}")
    prepare_remote_run_dir(ssh_alias, run_id, key_path=key_path)
    spec = RunSpec(
        run_id=run_id,
        task="inference",
        command=command,
        model_name=model_name,
        config=run_config or {},
    )
    write_remote_spec(ssh_alias, spec, key_path=key_path)

    # 8: Upload inputs from sync window
    workdir = Path(local_workdir)
    ip.emit("[rsync] uploading run inputs")
    rm.upload_run_inputs(ssh_alias, workdir, get_remote_run_paths(run_id).run_root, key_path=str(key_path))

    # 9: Start run inside capo_remote via remote window
    start_remote_inference(ssh_alias, run_id, command, key_path=key_path)
    ip.emit(f"Inference job submitted (run_id={run_id})")

    return LambdaWorkflowResult(
        run_id=run_id,
        instance_id=instance.instance_id,
        ssh_alias=ssh_alias,
        local_run_dir=local_run_dir,
    )


def start_lambda_finetune_workflow(
    instance_type: str,
    ssh_key_name: str,
    key_path: str | Path,
    command: str,
    model_name: str,
    local_workdir: str | Path,
    run_config: dict | None = None,
    run_id: str | None = None,
    region: str | None = None,
    api_key: str | None = None,
    artifacts_dir: str | Path | None = None,
    sync_checkpoints: bool = False,
) -> LambdaWorkflowResult:
    """Same as start_lambda_inference_workflow. sync_checkpoints=True adds checkpoints/ to download_run_outputs."""
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    key_path = Path(key_path)
    local_run_dir = Path(artifacts_dir or LOCAL_ARTIFACTS_ROOT) / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    ip.emit(f"Creating Lambda instance {instance_type} (run_id={run_id})")
    instance = ls.provision_instance(instance_type, ssh_key_name, region=region, api_key=api_key)
    instance_ip = ls.wait_for_instance_ip(instance.instance_id, api_key=api_key)
    ls.wait_for_ssh_ready(instance_ip, key_path)

    ssh_alias = f"lambda-{instance.instance_id}"
    ls.ensure_ssh_alias(ssh_alias, instance_ip, key_path)

    ip.emit("[tmux] launching local workspace")
    tm.ensure_local_workspace()
    ip.emit(f"[tmux] ensuring remote session on {ssh_alias}")
    tm.ensure_remote_tmux(ssh_alias, key_path=str(key_path))

    ip.emit(f"Preparing remote run directory {run_id}")
    prepare_remote_run_dir(ssh_alias, run_id, key_path=key_path)
    spec = RunSpec(
        run_id=run_id,
        task="finetune",
        command=command,
        model_name=model_name,
        config=run_config or {},
    )
    write_remote_spec(ssh_alias, spec, key_path=key_path)

    workdir = Path(local_workdir)
    ip.emit("[rsync] uploading run inputs")
    rm.upload_run_inputs(ssh_alias, workdir, get_remote_run_paths(run_id).run_root, key_path=str(key_path))

    start_remote_finetune(ssh_alias, run_id, command, key_path=key_path)
    ip.emit(f"Fine-tuning job submitted (run_id={run_id})")

    return LambdaWorkflowResult(
        run_id=run_id,
        instance_id=instance.instance_id,
        ssh_alias=ssh_alias,
        local_run_dir=local_run_dir,
    )


def sync_lambda_run_workflow(
    ssh_alias: str,
    run_id: str,
    local_run_dir: str | Path,
    key_path: str | Path,
    ssh_target: str | None = None,
) -> RunStatus:
    """
    Pull status.json, metrics.jsonl, train.log, train_err.log via sync_run_status.
    Read and return local status.
    """
    local_dir = Path(local_run_dir)
    target = ssh_target or ssh_alias
    remote_paths = get_remote_run_paths(run_id)
    rm.sync_run_status(target, remote_paths.run_root, local_dir, key_path=str(key_path))
    return read_remote_run_status(ssh_alias, run_id, key_path=key_path)


def fetch_lambda_results_workflow(
    ssh_alias: str,
    run_id: str,
    local_run_dir: str | Path,
    key_path: str | Path,
    ssh_target: str | None = None,
) -> dict:
    """
    download_run_outputs (outputs/ + results/ + checkpoints/).
    summarize_outputs(local_run_dir).
    Return summary dict.
    """
    from capo.results.io import summarize_outputs
    from dataclasses import asdict

    local_dir = Path(local_run_dir)
    target = ssh_target or ssh_alias
    remote_paths = get_remote_run_paths(run_id)
    rm.download_run_outputs(target, remote_paths.run_root, local_dir, key_path=str(key_path))
    summary = summarize_outputs(local_dir)
    return asdict(summary)


def shutdown_lambda_workflow(
    instance_id: str,
    api_key: str | None = None,
) -> None:
    """terminate_instance(instance_id, api_key)."""
    ls.terminate_instance(instance_id, api_key=api_key)


# ---------------------------------------------------------------------------
# InferenceOrchestrator — agent-driven protein model inference on Lambda
# ---------------------------------------------------------------------------

_REPO_ROOT_ORCH = Path(__file__).resolve().parents[3]   # src/capo/orchestration/ → 3 up → repo root
_SKILLS_DIR = _REPO_ROOT_ORCH / "skills"

_MODEL_SLUG_PATTERNS: list[tuple[str, str]] = [
    # Check more-specific names before their prefixes (boltzgen before boltz, esm2 before esm)
    (r"\bboltzgen\b",  "boltzgen"),
    (r"\bboltz\b",     "boltz"),
    (r"\besm2\b",      "esm2"),
    (r"\besmfold\b",   "esmfold"),
    (r"\besm\b",       "esm"),
    (r"\bchai\b",      "chai"),
    (r"\bankh\b",      "ankh"),
    (r"\bprottrans\b", "prottrans"),
    # (r"\baf3\b",       "af3"),
    # (r"\baf2\b",       "af2"),
    # (r"\balphafold\b", "af"),
]


def _extract_model_slug(task_description: str) -> str:
    """Return a short lowercase model slug from task_description, or 'model' if none matched."""
    lower = task_description.lower()
    for pattern, slug in _MODEL_SLUG_PATTERNS:
        if re.search(pattern, lower):
            return slug
    return "model"

_INFERENCE_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/inference")

_INFERENCE_PROMPT_TEMPLATE = load_prompt("orchestrator/user_prompts/inference")


@dataclass
class InferenceResult:
    run_id: str
    local_run_dir: Path
    output_files: list[Path]       # all files under outputs/
    state: str                     # "completed" | "failed" | "stopped" | "unknown"
    answer: str                    # full agent response text
    cost_usd: float | None


class InferenceOrchestrator:
    """
    Agent-driven protein model inference on Lambda GPU.

    Accepts a free-text task_description. The claude-sonnet-4-6 agent reads
    the relevant skill script from skills/model-inference/, uploads it, and
    runs it inside capo_remote using the CAPO multi-session framework.

    Supported model families (via skills): esm2, boltz, boltzgen, chai, ankh, prottrans.

    Example::

        orch = InferenceOrchestrator(key_path="~/.ssh/lambda_key", ssh_key_name="my-key")
        result = orch.run_sync(
            task_description=(
                "Embed sequence MKTAYIAKQR... with ESM2 8M model "
                "(facebook/esm2_t6_8M_UR50D). Use embed-esm2-hf subcommand."
            )
        )
        print(result.output_files)
    """

    def __init__(
        self,
        key_path: str | Path,
        ssh_key_name: str,
        model_name: str = "claude-sonnet-4-6",
        instance_type: str | None = None,
        instance_name: str | None = None,
        max_turns: int = 500,
        cwd: str | Path | None = None,
    ) -> None:
        self.key_path = str(Path(key_path).expanduser().resolve())
        self.ssh_key_name = ssh_key_name
        self.instance_type = instance_type or "auto"
        self.instance_name = instance_name or "auto"
        self.api_key = os.environ.get("LAMBDA_API_KEY", "")
        self._runner = AgentRunner(
            model_name=model_name,
            allowed_tools=["Read", "Bash", "mcp__lambda-repl__*"],
            system_prompt=_INFERENCE_SYSTEM_PROMPT,
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=str(cwd or _REPO_ROOT_ORCH),
        )

    async def run(
        self,
        task_description: str,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> InferenceResult:
        if run_id:
            rid = run_id
        else:
            ts   = datetime.now().strftime("%Y%m%d-%H%M")
            slug = _extract_model_slug(task_description)
            h    = uuid.uuid4().hex[:4]
            rid  = f"infer-{slug}-{ts}-{h}"
        local_run_dir = Path(output_dir or (_REPO_ROOT_ORCH / "lambda" / "runs" / "inference" / rid))
        local_run_dir.mkdir(parents=True, exist_ok=True)

        outputs_dir = local_run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        emitter = ip.ProgressEmitter(
            stdout_log=outputs_dir / ip.RUN_LOG_NAME,
            stderr_log=outputs_dir / ip.RUN_ERR_LOG_NAME,
        )

        if self.api_key is None:
            emitter.emit("[warning] LAMBDA_API_KEY not set in environment; provisioning will fail if no existing instances are available")
            raise ValueError("LAMBDA_API_KEY not set in environment - required for Lambda provisioning. Please set it and try again.")
        
        token = ip.set_emitter(emitter)
        try:
            ip.emit(f"Starting inference run {rid}")

            prompt = _INFERENCE_PROMPT_TEMPLATE.format(
                run_id=rid,
                task_description=task_description,
                ssh_key_name=self.ssh_key_name,
                key_path=self.key_path,
                local_run_dir=str(local_run_dir),
                skills_dir=str(_SKILLS_DIR),
                instance_type=self.instance_type,
                instance_name=self.instance_name,
            )

            result = await self._runner.generate(prompt=prompt)
            # cost is emitted inside agent_runner on ResultMessage

            output_files = sorted(outputs_dir.iterdir()) if outputs_dir.exists() else []
            state = "completed" if output_files else "unknown"

            # Mark results ready (records billing end, emits [results] line)
            if output_files:
                emitter.mark_results_ready()

            # Emit per-file sizes
            ip.emit(f"[results] Output files in {outputs_dir}:")
            for f in output_files:
                try:
                    size = ip.format_size(f.stat().st_size)
                except OSError:
                    size = "?"
                ip.emit(f"[results]   {f.name}  ({size})")

            # Final summary with inference cost
            emitter.emit_final_summary(rid, state, result.total_cost_usd)

            return InferenceResult(
                run_id=rid,
                local_run_dir=local_run_dir,
                output_files=output_files,
                state=state,
                answer=result.answer,
                cost_usd=result.total_cost_usd,
            )
        except Exception as exc:
            ip.error(f"Inference run {rid} failed: {exc}")
            raise
        finally:
            ip._emitter.reset(token)

    def run_sync(
        self,
        task_description: str,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> InferenceResult:
        coro = self.run(
            task_description=task_description,
            run_id=run_id,
            output_dir=output_dir,
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Scout prompt constants
# ---------------------------------------------------------------------------

_MODEL_SCOUT_PROMPT = load_prompt("orchestrator/user_prompts/model_scout")

_PROVIDER_SCOUT_PROMPT = load_prompt("orchestrator/user_prompts/provider_scout")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSelectionResult:
    selected_model: str | None
    fallback_model: str | None
    estimated_cost: str | None
    reasons: list[str]
    rejected_candidates: list[str]
    raw_text: str
    parse_error: str | None = None


@dataclass
class ProviderSelectionResult:
    selected_provider: str | None
    selected_instance_type: str | None
    instance_specs: dict[str, Any] | None
    estimated_cost_hourly: float | None
    estimated_cost_total: float | None
    budget_remaining_or_gap: str | None
    connection_method: str | None
    connection_steps: list[str]
    script_or_template: str | None
    rejected_cheaper_options: list[str]
    risks_or_missing_config: list[str]
    raw_text: str
    parse_error: str | None = None


@dataclass
class OrchestrationContext:
    model_selection: ModelSelectionResult | None
    provider_selection: ProviderSelectionResult | None
    has_model_selection: bool
    has_provider_selection: bool
    warnings: list[str]


@dataclass
class PhaseTrace:
    phase: str
    status: str
    duration_s: float
    cost_usd: float | None
    notes: str | None = None


# ---------------------------------------------------------------------------
# PhasedOrchestrator
# ---------------------------------------------------------------------------

class PhasedOrchestrator:
    def __init__(
        self,
        runner: AgentRunner,
        pipeline: AgentProcessingPipeline,
        scout_max_turns: int = 3,
    ) -> None:
        self.runner = runner
        self.pipeline = pipeline
        self.scout_max_turns = scout_max_turns

    async def run(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        base_prompt: str,
        extra_instructions: str | None = None,
        output_path: str | Path | None = None,
        generation_kwargs: dict | None = None,
        workload_description: str | None = None,
        budget_constraint: str = "unspecified",
    ) -> tuple[GenerationResult, list[PhaseTrace]]:
        all_traces: list[PhaseTrace] = []

        # Phase 1: parallel scouting
        workload_desc = workload_description or base_prompt
        t1 = time.perf_counter()
        try:
            model_raw, provider_raw = await self._run_phase1(workload_desc, budget_constraint)
        except Exception as exc:
            duration = round(time.perf_counter() - t1, 4)
            all_traces.append(PhaseTrace(
                phase="scout_model",
                status="failed",
                duration_s=duration,
                cost_usd=None,
                notes=f"Phase 1 gather failed: {exc}",
            ))
            model_raw = provider_raw = None  # type: ignore[assignment]

        # Phase 2: synthesize
        t2 = time.perf_counter()
        if model_raw is not None and provider_raw is not None:
            ctx, phase_traces = self._synthesize(model_raw, provider_raw)
            all_traces.extend(phase_traces)
        else:
            ctx = OrchestrationContext(
                model_selection=None,
                provider_selection=None,
                has_model_selection=False,
                has_provider_selection=False,
                warnings=["Phase 1 scouting failed — proceeding without orchestration context"],
            )
        all_traces.append(PhaseTrace(
            phase="synthesis",
            status="ok" if (ctx.has_model_selection or ctx.has_provider_selection) else "degraded",
            duration_s=round(time.perf_counter() - t2, 4),
            cost_usd=None,
        ))

        # Phase 3: code generation with enriched context
        t3 = time.perf_counter()
        result = await self.pipeline.generate_preprocess_function(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            base_prompt=base_prompt,
            extra_instructions=extra_instructions,
            output_path=output_path,
            generation_kwargs=generation_kwargs,
            orchestration_context=ctx,
        )
        all_traces.append(PhaseTrace(
            phase="codegen",
            status="ok",
            duration_s=round(time.perf_counter() - t3, 4),
            cost_usd=result.cost_usd,
        ))

        return result, all_traces

    async def _run_phase1(
        self, workload_description: str, budget_constraint: str
    ) -> tuple[AgentRunResult, AgentRunResult]:
        model_prompt = _MODEL_SCOUT_PROMPT.format(workload_description=workload_description)
        provider_prompt = _PROVIDER_SCOUT_PROMPT.format(
            workload_description=workload_description,
            budget_constraint=budget_constraint,
        )
        model_raw, provider_raw = await asyncio.gather(
            self.runner.generate(prompt=model_prompt, max_turns=self.scout_max_turns),
            self.runner.generate(prompt=provider_prompt, max_turns=self.scout_max_turns),
        )
        return model_raw, provider_raw

    def _parse_model_selection(self, raw: AgentRunResult) -> ModelSelectionResult:
        text = raw.answer
        try:
            data = self._extract_json_block(text)
            if data is None:
                raise ValueError("No JSON block found")
            return ModelSelectionResult(
                selected_model=data.get("selected_model"),
                fallback_model=data.get("fallback_model"),
                estimated_cost=data.get("estimated_cost"),
                reasons=data.get("reasons") or [],
                rejected_candidates=data.get("rejected_candidates") or [],
                raw_text=text,
            )
        except Exception as exc:
            return ModelSelectionResult(
                selected_model=None,
                fallback_model=None,
                estimated_cost=None,
                reasons=[],
                rejected_candidates=[],
                raw_text=text,
                parse_error=str(exc),
            )

    def _parse_provider_selection(self, raw: AgentRunResult) -> ProviderSelectionResult:
        text = raw.answer
        try:
            data = self._extract_json_block(text)
            if data is None:
                raise ValueError("No JSON block found")

            def _float(val: Any) -> float | None:
                if val is None:
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            return ProviderSelectionResult(
                selected_provider=data.get("selected_provider"),
                selected_instance_type=data.get("selected_instance_type"),
                instance_specs=data.get("instance_specs"),
                estimated_cost_hourly=_float(data.get("estimated_cost_hourly")),
                estimated_cost_total=_float(data.get("estimated_cost_total")),
                budget_remaining_or_gap=data.get("budget_remaining_or_gap"),
                connection_method=data.get("connection_method"),
                connection_steps=data.get("connection_steps") or [],
                script_or_template=data.get("script_or_template"),
                rejected_cheaper_options=data.get("rejected_cheaper_options") or [],
                risks_or_missing_config=data.get("risks_or_missing_config") or [],
                raw_text=text,
            )
        except Exception as exc:
            return ProviderSelectionResult(
                selected_provider=None,
                selected_instance_type=None,
                instance_specs=None,
                estimated_cost_hourly=None,
                estimated_cost_total=None,
                budget_remaining_or_gap=None,
                connection_method=None,
                connection_steps=[],
                script_or_template=None,
                rejected_cheaper_options=[],
                risks_or_missing_config=[],
                raw_text=text,
                parse_error=str(exc),
            )

    def _synthesize(
        self, model_raw: AgentRunResult, provider_raw: AgentRunResult
    ) -> tuple[OrchestrationContext, list[PhaseTrace]]:
        traces: list[PhaseTrace] = []
        warnings: list[str] = []

        t_m = time.perf_counter()
        model_result = self._parse_model_selection(model_raw)
        model_ok = model_result.parse_error is None
        traces.append(PhaseTrace(
            phase="scout_model",
            status="ok" if model_ok else "degraded",
            duration_s=round(time.perf_counter() - t_m, 4),
            cost_usd=model_raw.total_cost_usd,
            notes=model_result.parse_error,
        ))
        if not model_ok:
            warnings.append(f"Model scout parse failed: {model_result.parse_error}")

        t_p = time.perf_counter()
        provider_result = self._parse_provider_selection(provider_raw)
        provider_ok = provider_result.parse_error is None
        traces.append(PhaseTrace(
            phase="scout_provider",
            status="ok" if provider_ok else "degraded",
            duration_s=round(time.perf_counter() - t_p, 4),
            cost_usd=provider_raw.total_cost_usd,
            notes=provider_result.parse_error,
        ))
        if not provider_ok:
            warnings.append(f"Provider scout parse failed: {provider_result.parse_error}")

        ctx = OrchestrationContext(
            model_selection=model_result,
            provider_selection=provider_result,
            has_model_selection=model_ok,
            has_provider_selection=provider_ok,
            warnings=warnings,
        )
        return ctx, traces

    @staticmethod
    def _extract_json_block(text: str) -> dict | None:
        # Find the first balanced { ... } block
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None

    def run_sync(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        base_prompt: str,
        extra_instructions: str | None = None,
        output_path: str | Path | None = None,
        generation_kwargs: dict | None = None,
        workload_description: str | None = None,
        budget_constraint: str = "unspecified",
    ) -> tuple[GenerationResult, list[PhaseTrace]]:
        coro = self.run(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            base_prompt=base_prompt,
            extra_instructions=extra_instructions,
            output_path=output_path,
            generation_kwargs=generation_kwargs,
            workload_description=workload_description,
            budget_constraint=budget_constraint,
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
