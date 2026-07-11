"""Run cost accounting — agent (LLM) cost + Lambda infrastructure cost.

Pure data + arithmetic + markdown/CLI rendering. No I/O, no network. The
orchestrator collects per-agent SDK-reported cost and the resolved instance
type/rate/runtime, builds a :class:RunCostReport, and renders it into both
RUN_REPORT.md (the "## Cost Report" section) and the CLI summary block.

Two cost sources:

1. Agent cost: the authoritative total_cost_usd that every Claude Agent
   SDK call reports. The SDK/CLI computes the real per-call price, already
   accounting for cache reads/writes and any per-model discounts, so this is the
   *only* source we use for agent cost — we never re-derive a price from a
   hard-coded rate table. That keeps the ledger equal to what Anthropic actually
   billed. When the SDK does not report a cost we keep the token counts and
   leave cost_usd=None (never fabricate a price).

2. Infrastructure cost; runtime_seconds * hourly_rate_usd for the one
   Lambda instance the run used (this module is the post-run ledger; the live
   elapsed-time estimate during a run lives in capo.remote).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from capo.report.tables import (
    EMDASH,
    fmt_cost,
    fmt_cost_precise,
    fmt_int,
    fmt_runtime,
    markdown_table,
)


@dataclass
class AgentCost:
    """One LLM agent's contribution to the run's cost.

    cost_usd is the SDK-reported total_cost_usd for the call(s) — never a
    re-derived estimate. The prompt side is split by price tier so the token
    columns reconcile with the cost (a long agentic run is mostly cache reads
    billed at ~0.1x, which is why total input looks huge next to a small cost):

    - input_tokens          uncached prompt tokens (full price, ~1x)
    - cache_creation_tokens cache writes (~1.25x)
    - cache_read_tokens     cache reads (~0.1x) — usually the bulk of the total

    Token fields are None for cost-only entries (e.g. the health monitor's
    aggregate Haiku spend, where per-call token counts aren't retained).
    """

    agent_name: str
    model: str
    input_tokens: int | None            # uncached prompt tokens (full price)
    output_tokens: int | None
    cost_usd: float | None              # SDK-reported total_cost_usd, or None when unreported
    cache_read_tokens: int | None = None      # served from cache (~0.1x)
    cache_creation_tokens: int | None = None  # written to cache (~1.25x)

    @property
    def full_price_input_tokens(self) -> int | None:
        """Uncached input + cache writes — the tokens billed at roughly full
        price. This is the Input column in the report; cache reads are
        reported separately as Cached. None for cost-only entries."""
        if self.input_tokens is None and self.cache_creation_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.cache_creation_tokens or 0)


@dataclass
class InfraCost:
    """One compute instance's contribution to the run's cost."""

    provider: str
    instance_id: str | None
    instance_type: str
    gpu_type: str | None
    gpu_count: int | None
    runtime_seconds: float
    hourly_rate_usd: float | None
    cost_usd: float | None


@dataclass
class RunCostReport:
    """Complete cost ledger for a run: agent + infra breakdown and totals."""

    agent_costs: list[AgentCost] = field(default_factory=list)
    infra_costs: list[InfraCost] = field(default_factory=list)

    @property
    def total_agent_cost_usd(self) -> float:
        return sum(c.cost_usd or 0.0 for c in self.agent_costs)

    @property
    def total_infra_cost_usd(self) -> float:
        return sum(c.cost_usd or 0.0 for c in self.infra_costs)

    @property
    def total_cost_usd(self) -> float:
        return self.total_agent_cost_usd + self.total_infra_cost_usd

    @property
    def has_unknown_costs(self) -> bool:
        """True when any line item carries tokens/runtime but no price — the
        totals understate actual spend and should be flagged as such."""
        return any(c.cost_usd is None for c in self.agent_costs) or any(
            c.cost_usd is None for c in self.infra_costs
        )

    def to_dict(self) -> dict:
        return {
            "agent_costs": [vars(c) for c in self.agent_costs],
            "infra_costs": [vars(c) for c in self.infra_costs],
            "total_agent_cost_usd": round(self.total_agent_cost_usd, 6),
            "total_infra_cost_usd": round(self.total_infra_cost_usd, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "has_unknown_costs": self.has_unknown_costs,
        }


def make_agent_cost(
    agent_name: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    sdk_cost_usd: float | None = None,
) -> AgentCost:
    """Build an :class:AgentCost from the SDK-reported cost.

    sdk_cost_usd is the authoritative total_cost_usd from the Claude
    Agent SDK ResultMessage — the only price we record for an agent. The
    token counts are kept for the breakdown table but never drive the cost;
    when sdk_cost_usd is None the cost stays None (tokens retained).

    The three prompt-side counts are stored separately (they carry three
    different prices): input_tokens is the uncached remainder, cache_read_tokens
    the reads (~0.1x), cache_creation_tokens the writes (~1.25x). The report
    folds writes into the full-price Input column and shows reads as
    Cached; cost_report.json keeps all three raw.
    """
    return AgentCost(
        agent_name=agent_name,
        model=model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cost_usd=(float(sdk_cost_usd) if sdk_cost_usd is not None else None),
        cache_read_tokens=int(cache_read_tokens),
        cache_creation_tokens=int(cache_creation_tokens),
    )


def make_agent_cost_from_total(
    agent_name: str,
    model: str,
    sdk_cost_usd: float | None,
) -> AgentCost:
    """Cost-only :class:AgentCost (token counts unknown → None).

    For agents whose per-call token usage isn't retained but whose aggregate
    SDK cost is (the health monitor, finalizer, researcher)."""
    return AgentCost(
        agent_name=agent_name,
        model=model,
        input_tokens=None,
        output_tokens=None,
        cost_usd=(float(sdk_cost_usd) if sdk_cost_usd is not None else None),
    )


def make_infra_cost(
    *,
    provider: str = "lambda",
    instance_id: str | None = None,
    instance_type: str = "unknown",
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    runtime_seconds: float = 0.0,
    hourly_rate_usd: float | None = None,
) -> InfraCost:
    """Build an :class:InfraCost; cost is runtime * rate or None."""
    if hourly_rate_usd is not None and runtime_seconds is not None:
        cost: float | None = max(0.0, float(runtime_seconds)) / 3600.0 * float(hourly_rate_usd)
    else:
        cost = None
    return InfraCost(
        provider=provider,
        instance_id=instance_id,
        instance_type=instance_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        runtime_seconds=float(runtime_seconds or 0.0),
        hourly_rate_usd=hourly_rate_usd,
        cost_usd=cost,
    )


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def render_cost_report_markdown(report: RunCostReport) -> str:
    """Render the full ## Cost Report section for RUN_REPORT.md."""
    summary_rows = [
        {"category": "Agent cost", "cost": fmt_cost(report.total_agent_cost_usd)},
        {"category": "Infrastructure cost", "cost": fmt_cost(report.total_infra_cost_usd)},
        {"category": "Total", "cost": fmt_cost(report.total_cost_usd)},
    ]
    summary = markdown_table(
        summary_rows,
        columns=["category", "cost"],
        headers={"category": "Category", "cost": "Cost USD"},
        align={"cost": "right"},
    )

    agent_rows = [
        {
            "agent": c.agent_name,
            "model": c.model,
            "in": fmt_int(c.full_price_input_tokens),
            "cached": fmt_int(c.cache_read_tokens),
            "out": fmt_int(c.output_tokens),
            "cost": fmt_cost_precise(c.cost_usd),
        }
        for c in report.agent_costs
    ]
    agent_table = markdown_table(
        agent_rows,
        columns=["agent", "model", "in", "cached", "out", "cost"],
        headers={
            "agent": "Agent",
            "model": "Model",
            "in": "Input Tokens",
            "cached": "Cached Tokens",
            "out": "Output Tokens",
            "cost": "Cost USD",
        },
        align={"in": "right", "cached": "right", "out": "right", "cost": "right"},
    )
    # Explain why the cost is far below Input Tokens x list price: most of a long
    # agentic run's prompt is re-read from cache at ~10% of the input rate.
    cache_note = (
        "> Input Tokens = uncached prompt + cache writes (billed at roughly the "
        "full input rate). Cached Tokens = prompt-cache reads, billed at ~10% of "
        "that rate — which is why total cost is far below Input Tokens x list "
        "price. Cost USD is the SDK-reported amount actually billed."
    )

    infra_rows = [
        {
            "provider": c.provider,
            "instance_id": c.instance_id or EMDASH,
            "instance_type": c.instance_type,
            "gpu": (
                f"{c.gpu_count}x {c.gpu_type}"
                if c.gpu_type and c.gpu_count
                else (c.gpu_type or EMDASH)
            ),
            "runtime": fmt_runtime(c.runtime_seconds),
            "rate": fmt_cost(c.hourly_rate_usd),
            "cost": fmt_cost(c.cost_usd),
        }
        for c in report.infra_costs
    ]
    infra_table = markdown_table(
        infra_rows,
        columns=["provider", "instance_id", "instance_type", "gpu", "runtime", "rate", "cost"],
        headers={
            "provider": "Provider",
            "instance_id": "Instance ID",
            "instance_type": "Instance Type",
            "gpu": "GPU",
            "runtime": "Runtime",
            "rate": "Hourly Rate",
            "cost": "Cost USD",
        },
        align={"runtime": "right", "rate": "right", "cost": "right"},
    )

    out = [
        "## Cost Report",
        "",
        "### Summary",
        "",
        summary,
        "",
        "### Agent Costs",
        "",
        agent_table,
        "",
        cache_note,
        "",
        "### Infrastructure Costs",
        "",
        infra_table,
    ]
    if report.has_unknown_costs:
        out += [
            "",
            "> Note: one or more line items lack a known unit price; the totals "
            "above are a lower bound (unpriced items counted as $0.00).",
        ]
    return "\n".join(out)


def render_cli_summary(report: RunCostReport) -> str:
    """Render the compact CLI cost block printed at run completion."""
    return _render_cli_summary(
        agent=report.total_agent_cost_usd,
        infra=report.total_infra_cost_usd,
        total=report.total_cost_usd,
        has_unknown=report.has_unknown_costs,
    )


def render_cli_summary_dict(report_dict: dict) -> str:
    """Render the compact CLI cost block from a :meth:RunCostReport.to_dict."""
    return _render_cli_summary(
        agent=float(report_dict.get("total_agent_cost_usd") or 0.0),
        infra=float(report_dict.get("total_infra_cost_usd") or 0.0),
        total=float(report_dict.get("total_cost_usd") or 0.0),
        has_unknown=bool(report_dict.get("has_unknown_costs")),
    )


def _render_cli_summary(*, agent: float, infra: float, total: float, has_unknown: bool) -> str:
    lines = [
        "Cost summary:",
        f"  Agent cost: {fmt_cost_precise(agent)}",
        f"  Infra cost: {fmt_cost(infra)}",
        f"  Total cost: {fmt_cost(total)}",
    ]
    if has_unknown:
        lines.append("  (some items had no known price — total is a lower bound)")
    lines.append("Full breakdown written to RUN_REPORT.md")
    return "\n".join(lines)
