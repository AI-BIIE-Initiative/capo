from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query
from claude_agent_sdk._errors import CLINotFoundError, ProcessError
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

from capo.context.prompt_caching import (
    cached_streaming_prompt,
    extract_cache_stats,
)
from capo.orchestration.retry import with_retry
from capo.observability import progress as ip
from capo.utils.prompts import load_prompt as _load_prompt

try:
    from claude_agent_sdk.types import ToolUseBlock as _ToolUseBlock
except ImportError:
    _ToolUseBlock = None  # type: ignore[assignment,misc]

try:
    from claude_agent_sdk.types import UserMessage as _UserMessage
except ImportError:
    _UserMessage = None  # type: ignore[assignment,misc]


class AnthropicAuthError(RuntimeError):
    """Anthropic API rejected the request (missing/invalid key or out of credit).

    The Claude Agent SDK does not raise on these errors — it returns a
    ResultMessage with subtype="success" and the assistant text contains the
    real error ("Credit balance is too low", "invalid x-api-key", …). Detected
    in ``_run_query_once`` and raised so the orchestrator fails with a clear
    cause instead of a downstream "infra.json missing" symptom.
    """


_ANTHROPIC_AUTH_PATTERNS: tuple[tuple[str, str], ...] = (
    ("credit balance is too low", "Anthropic account is out of credit"),
    ("invalid x-api-key", "ANTHROPIC_API_KEY is invalid or revoked"),
    ("invalid api key", "ANTHROPIC_API_KEY is invalid"),
    ("authentication_error", "Anthropic API authentication failed"),
    ("authentication failed", "Anthropic API authentication failed"),
    ("api key not found", "ANTHROPIC_API_KEY not found by the SDK"),
    ("unauthorized", "Anthropic API returned 401 Unauthorized"),
    ("quota exceeded", "Anthropic API quota exceeded"),
)


def _detect_anthropic_auth_error(answer: str) -> str | None:
    """Return a short human-readable cause when the answer text indicates an
    Anthropic API auth/credit failure, else None.
    """
    if not answer:
        return None
    low = answer.lower()
    for needle, cause in _ANTHROPIC_AUTH_PATTERNS:
        if needle in low:
            return cause
    return None

# allow subagents parallelization, define the 
# SUBAGENTS registry
# ---------------------------------------------------------------------------
# Full step-by-step instructions for each subagent are in the "prompt" field of each AgentDefinition below.
# ---------------------------------------------------------------------------
SUBAGENTS: dict[str, AgentDefinition] = {
    "model-selector": AgentDefinition(
        description=(
            "Selects the best PLM and fine-tuning strategy from the BIIE-AI model "
            "registry. Returns top candidates with scores, GPU requirements, and rationale."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/model_selector")
        ),
        tools=["Read", "Grep", "Glob", "Bash"],
        model="claude-sonnet-4-6",
        skills=["model-selection"],
        memory="project",
    ),
    "infrastructure": AgentDefinition(
        description=(
            "CAPO Phase 0: resolves Lambda GPU tier, attaches or provisions a cloud "
            "instance, ensures the remote tmux session, fetches hourly pricing, and "
            "writes infra.json + pricing/lambda-<gpu>.json under local_run_dir."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/infrastructure")
        ),
        tools=[
            "Read", "Grep", "Glob", "Bash", "Write",
            "mcp__lambda-repl__*",
        ],
        model="claude-sonnet-4-6",
        effort="high",
        skills=["lambda-session", "cost-estimation", "model-selection"],
        memory="project",
    ),
    "data-profiler": AgentDefinition(
        description=(
            "CAPO Phase 1: profiles a dataset (format detection, load, modality analysis, "
            "preprocessing recommendations). Produces profile.json, plots, and length percentiles."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/data_profiler")
        ),
        tools=["Read", "Grep", "Glob", "Bash"],
        model="claude-sonnet-4-6",
        effort="high",
        skills=[
            "profiling-datasets",
            "analysis/analyze-protein-sequences",
            "analysis/analyze-tabular",
            "analysis/analyze-fcs",
            "analysis/analyze-single-cell",
            "analysis/analyze-fastq-reads",
            "clustering/mmseqs2",
        ],
        memory="project",
    ),
    "memory-consultant": AgentDefinition(
        description=(
            "Episodic memory consultant. Scans <repo>/runs/runs_index.md (a "
            "concatenation of past RUN_REPORT.md YAML frontmatters) using "
            "progressive disclosure: cheap fingerprint scan first, then selective "
            "full-body Reads of the 0–3 most relevant past runs. Writes a curated "
            "prior_runs.md that downstream pre-launch subagents (infra, profiler, "
            "model-selector) and the Sonnet pre-launch agent consume as advisory "
            "priors."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/memory_consultant")
        ),
        tools=["Read", "Write", "Bash", "Glob"],
        model="claude-sonnet-4-6",
        skills=[],
        memory="project",
    ),
    "experiment-tracker": AgentDefinition(
        description=(
            "Initialises a trackio run for fine-tuning experiments using the "
            "tracking-experiments/trackio skill. Confirms config, fires trackio.init, "
            "and returns the dashboard URL for real-time metric monitoring."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/experiment_tracker")
        ),
        tools=["Read", "Bash"],
        model="claude-haiku-4-5-20251001",
        skills=["tracking-experiments/trackio"],
        memory="project",
    ),
    "training-health-monitor": AgentDefinition(
        description=(
            "Cheap periodic health check for a live Lambda fine-tuning job. "
            "Given ssh_alias, remote_run_dir and previous report, issues ONE SSH "
            "round-trip, assesses health over two temporal windows (short and "
            "long), classifies trend, and returns a strict JSON report. "
            "Read-only — never writes, kills, or launches."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/training_health_monitor")
        ),
        tools=["Bash", "mcp__lambda-repl__lambda_run_command"],
        model="claude-haiku-4-5-20251001",
        skills=[],
        memory="project",
    ),
    "code-repair-critic": AgentDefinition(
        description=(
            "Long-tail repair specialist for the CAPO 3-step gate. Invoked ONLY "
            "at Attempt 3 of the repair ladder, after the orchestrator has "
            "self-repaired twice without success, and only for "
            "failure_category=script_bug. Receives a compact failure packet "
            "(≤8 KB) and emits a unified diff that fixes the root cause. "
            "Never runs scripts, never touches Lambda, never reasons about "
            "infra or cost — pure code synthesis from a structured spec."
        ),
        prompt=(
            _load_prompt("subagent/system_prompts/code_repair_critic")
        ),
        tools=["Read", "Write", "Edit", "Glob"],
        model="claude-opus-4-8",
        effort="xhigh",
        skills=["code-writing"],
        memory=None,
    ),
}


@dataclass
class AgentRunResult:
    model_name: str
    prompt: str
    answer: str
    code: str | None
    raw_messages: list
    session_id: str | None = None
    total_cost_usd: float | None = None
    subtype: str = "success"
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    input_tokens: int | None = None       # uncached (full-price) prompt tokens
    output_tokens: int | None = None


class AgentRunner:
    SUPPORTED_MODELS: set[str] = {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    }

    SYSTEM_PROMPT = (
        _load_prompt("subagent/system_prompts/agent_runner_default")
    )

    CODE_SYSTEM_PROMPT = (
        _load_prompt("subagent/system_prompts/agent_runner_code")
    )

    def __init__(
        self,
        model_name: str,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        permission_mode: str = "acceptEdits",
        max_turns: int = 10,
        cwd: str | Path | None = None,
        subagents: dict[str, AgentDefinition] | None = None,
        mcp_servers: dict | None = None,
        emit_cost_per_call: bool = True,
        effort: str | None = None,
        skills: list[str] | str | None = None,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name!r}. "
                f"Supported: {sorted(self.SUPPORTED_MODELS)}"
            )
        self.model_name = model_name
        self.allowed_tools = allowed_tools or []
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.cwd = cwd
        self.subagents = subagents
        # None = SDK default (reads .mcp.json from cwd); {} = no MCP servers.
        self.mcp_servers = mcp_servers
        self.emit_cost_per_call = emit_cost_per_call
        # #4 (config-gated, default off): reasoning depth + skill exposure. Both
        # None → _build_options is identical to before, so behavior is unchanged
        # unless a caller opts in. effort: low|medium|high|xhigh|max.
        self.effort = effort or None
        self.skills = self._normalize_skills(skills)

    @staticmethod
    def _normalize_skills(skills: list[str] | str | None) -> list[str] | str | None:
        """Coerce a config value (None | 'all' | 'a,b' | ['a','b']) into the
        shape ClaudeAgentOptions.skills expects (None | 'all' | list[str])."""
        if skills is None:
            return None
        if isinstance(skills, str):
            s = skills.strip()
            if not s:
                return None
            return "all" if s.lower() == "all" else [p.strip() for p in s.split(",") if p.strip()]
        return list(skills) or None

    def _build_options(
        self,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> ClaudeAgentOptions:
        tools = list(self.allowed_tools)
        if self.subagents and "Agent" not in tools:
            tools.append("Agent")
        kwargs: dict = dict(
            model=self.model_name,
            system_prompt=system_prompt or self.system_prompt,
            allowed_tools=tools,
            permission_mode=self.permission_mode,  # type: ignore[arg-type]
            max_turns=max_turns if max_turns is not None else self.max_turns,
            cwd=self.cwd,
            agents=self.subagents or {},
            # Route subprocess (Claude Code CLI) stderr to our run_err.log so
            # startup errors, MCP failures and API refusals are captured.
            stderr=lambda line: ip.error(f"[cli] {line}"),
        )
        if self.mcp_servers is not None:
            kwargs["mcp_servers"] = self.mcp_servers
        # #4: only pass these when explicitly configured, so the default options
        # are byte-for-byte what they were before this knob existed.
        if self.effort is not None:
            kwargs["effort"] = self.effort
        if self.skills is not None:
            kwargs["skills"] = self.skills
        return ClaudeAgentOptions(**kwargs)

    async def _run_query(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        cached_prefix: str | None = None,
    ) -> AgentRunResult:
        async def attempt() -> AgentRunResult:
            return await self._run_query_once(
                prompt, system_prompt, max_turns, cached_prefix
            )

        try:
            return await with_retry(attempt)
        except CLINotFoundError as exc:
            raise RuntimeError(
                "Claude CLI not found. Ensure 'claude' is installed and on PATH."
            ) from exc
        except ProcessError as exc:
            raise RuntimeError(f"Claude SDK process error: {exc}") from exc

    async def _run_query_once(
        self,
        prompt: str,
        system_prompt: str | None,
        max_turns: int | None,
        cached_prefix: str | None,
    ) -> AgentRunResult:
        options = self._build_options(system_prompt=system_prompt, max_turns=max_turns)
        raw_messages: list = []
        answer_parts: list[str] = []
        session_id: str | None = None
        total_cost_usd: float | None = None
        subtype = "success"

        # Track the most recent tool name so we can correlate results.
        _last_tool_name: str = ""

        # When a cached_prefix is supplied, switch to streaming-mode input so
        # we can attach an ephemeral cache breakpoint to the stable prefix.
        # Otherwise pass the raw string (existing behavior, no overhead).
        query_prompt: Any
        if cached_prefix:
            query_prompt = cached_streaming_prompt(cached_prefix, prompt)
        else:
            query_prompt = prompt

        try:
            async for msg in query(prompt=query_prompt, options=options):
                raw_messages.append(msg)

                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            answer_parts.append(block.text)
                            # Emit short agent commentary in real-time so the
                            # user sees "Loading dataset…" before the tool call,
                            # not just tool-call labels. Skip JSON answers
                            # (start with { or [) and long reasoning blocks.
                            txt = block.text.strip()
                            if txt and len(txt) <= 300 and txt[0] not in ("{", "["):
                                ip.emit(f"[status] {txt}")
                        elif (
                            (_ToolUseBlock is not None and isinstance(block, _ToolUseBlock))
                            or (
                                not isinstance(block, TextBlock)
                                and hasattr(block, "name")
                                and hasattr(block, "input")
                            )
                        ):
                            name = getattr(block, "name", "") or ""
                            tinput = getattr(block, "input", {}) or {}
                            _last_tool_name = name
                            ip.emit_tool_call(name, tinput)

                elif isinstance(msg, ResultMessage):
                    session_id = msg.session_id
                    total_cost_usd = msg.total_cost_usd
                    subtype = msg.subtype
                    if total_cost_usd is not None and self.emit_cost_per_call:
                        ip.emit(f"[summary] agent_cost=${total_cost_usd:.4f}")

                else:
                    # Try to extract tool results from UserMessage or similar.
                    # The SDK may or may not expose these. handle gracefully.
                    content = getattr(msg, "content", None)
                    if isinstance(content, list):
                        for block in content:
                            # ToolResultBlock has output or content + tool_use_id
                            result_text = (
                                getattr(block, "output", None)
                                or getattr(block, "content", None)
                            )
                            if result_text and isinstance(result_text, str):
                                ip.emit_tool_result(_last_tool_name, result_text)
        except Exception as exc:
            # Claude Code CLI exits with code 1 when max_turns is reached: it
            # first emits a ResultMessage (already captured above), then the
            # process exits non-zero and the SDK raises here. Treat that as a
            # graceful completion. Otherwise propagate so with_retry can
            # classify and decide whether to retry.
            if session_id is not None or total_cost_usd is not None:
                ip.emit(
                    f"[warning] Agent process exited with error after ResultMessage "
                    f"(subtype={subtype!r}, cost=${total_cost_usd:.4f}): {exc}"
                )
                # Mirror to run_err.log with full detail for post-mortem diagnosis.
                ip.error(
                    f"[agent-exit] subtype={subtype!r} cost=${total_cost_usd:.4f} "
                    f"last_tool={_last_tool_name!r} exc={exc}"
                )
                # Log the partial answer — this is the actual text the agent
                # emitted before the process died, which often contains the
                # real error (API refusal, OOM message, etc.).
                if answer_parts:
                    partial = "".join(answer_parts)[:4000]
                    ip.error(f"[agent-answer] {partial}")
                ip.emit(
                    "[warning] See [agent-answer] above for the actual cause. "
                    "If training was launched on the remote before the error, "
                    "it may still be running."
                )
                if subtype == "success":
                    subtype = "max_turns"
            else:
                ip.error(
                    f"Agent SDK error before any ResultMessage — last_tool={_last_tool_name!r}: {exc}"
                )
                raise

        cache_stats = extract_cache_stats(raw_messages)
        if cache_stats["cache_read"] or cache_stats["cache_creation"]:
            ip.emit(
                f"[cache] read={cache_stats['cache_read']} "
                f"creation={cache_stats['cache_creation']} "
                f"input={cache_stats['input']} output={cache_stats['output']}"
            )

        full_answer = "".join(answer_parts)
        auth_cause = _detect_anthropic_auth_error(full_answer)
        if auth_cause is not None:
            ip.error(f"[anthropic-auth] {auth_cause}: {full_answer.strip()[:300]}")
            ip.emit(
                f"[error] Anthropic API call failed — {auth_cause}. "
                "Fix the underlying credential/billing issue and re-run; "
                "this run cannot proceed."
            )
            raise AnthropicAuthError(
                f"{auth_cause} (agent answer: {full_answer.strip()[:200]!r})"
            )

        return AgentRunResult(
            model_name=self.model_name,
            prompt=prompt,
            answer="".join(answer_parts),
            code=None,
            raw_messages=raw_messages,
            session_id=session_id,
            total_cost_usd=total_cost_usd,
            subtype=subtype,
            cache_read_tokens=cache_stats["cache_read"],
            cache_creation_tokens=cache_stats["cache_creation"],
            input_tokens=cache_stats["input"],
            output_tokens=cache_stats["output"],
        )

    async def generate(
        self,
        prompt: str | None = None,
        prompt_path: str = None,
        max_turns: int | None = None,
        cached_prefix: str | None = None,
        **_: Any,
    ) -> AgentRunResult:
        if prompt is None:
            prompt = self.load_prompt(prompt_path)
        return await self._run_query(
            prompt, max_turns=max_turns, cached_prefix=cached_prefix
        )

    async def generate_proc_code(
        self,
        prompt: str | None = None,
        prompt_path: str = None,
        max_turns: int | None = None,
        cached_prefix: str | None = None,
        **_: Any,
    ) -> AgentRunResult:
        if (prompt is None) == (prompt_path is None):
            raise ValueError("Provide exactly one of prompt or prompt_path")

        if prompt is None:
            prompt = self.load_prompt(prompt_path)

        result = await self._run_query(
            prompt,
            system_prompt=self.CODE_SYSTEM_PROMPT,
            max_turns=max_turns,
            cached_prefix=cached_prefix,
        )
        code = self._extract_code(result.answer)
        return AgentRunResult(
            model_name=result.model_name,
            prompt=result.prompt,
            answer=result.answer,
            code=code,
            raw_messages=result.raw_messages,
            session_id=result.session_id,
            total_cost_usd=result.total_cost_usd,
            subtype=result.subtype,
            cache_read_tokens=result.cache_read_tokens,
            cache_creation_tokens=result.cache_creation_tokens,
        )

    def generate_sync(self, prompt: str | None = None, prompt_path: str = None, max_turns: int | None = None, cached_prefix: str | None = None, **kw: Any) -> AgentRunResult:
        import concurrent.futures
        coro = self.generate(prompt=prompt, prompt_path=prompt_path, max_turns=max_turns, cached_prefix=cached_prefix, **kw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def generate_proc_code_sync(self, prompt: str | None = None, prompt_path: str = None, max_turns: int | None = None, cached_prefix: str | None = None, **kw: Any) -> AgentRunResult:
        import concurrent.futures
        coro = self.generate_proc_code(prompt=prompt, prompt_path=prompt_path, max_turns=max_turns, cached_prefix=cached_prefix, **kw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    @staticmethod
    def load_prompt(prompt_path: str = "data/diabetes/agent_prompt.md") -> str:
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt is empty: {prompt_path}")
        return text

    @staticmethod
    def _extract_code(text: str) -> str:
        code = re.sub(r"^```(?:python)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
        code = re.sub(r"\n?```\s*$", "", code.strip())
        return code.strip()
