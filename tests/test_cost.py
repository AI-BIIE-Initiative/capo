"""Tests for capo.report.cost — SDK-reported agent cost, infra cost, rendering.

Agent cost is taken *exclusively* from the Claude Agent SDK's per-call
``total_cost_usd`` — there is no online-price rate table and no re-derivation
from token counts. Tokens are retained for the breakdown table only.
"""

from __future__ import annotations

import pytest

from capo.report.cost import (
    RunCostReport,
    make_agent_cost,
    make_agent_cost_from_total,
    make_infra_cost,
    render_cli_summary,
    render_cost_report_markdown,
)


# ---------------------------------------------------------------------------
# make_agent_cost — SDK cost is the only price
# ---------------------------------------------------------------------------

def test_agent_cost_uses_sdk_cost():
    ac = make_agent_cost(
        "infra", "claude-sonnet-4-6",
        input_tokens=1_000_000, output_tokens=1_000_000, sdk_cost_usd=0.42,
    )
    assert ac.cost_usd == pytest.approx(0.42)
    # input_tokens is the uncached remainder; no cache here, so full-price == input.
    assert ac.input_tokens == 1_000_000
    assert ac.full_price_input_tokens == 1_000_000
    assert ac.output_tokens == 1_000_000


def test_agent_cost_no_sdk_cost_is_none_but_keeps_tokens():
    # No sdk_cost_usd → cost is None; we never fabricate a price from tokens.
    ac = make_agent_cost("infra", "claude-haiku-4-5", input_tokens=500, output_tokens=10)
    assert ac.cost_usd is None
    assert ac.input_tokens == 500
    assert ac.output_tokens == 10


def test_agent_cost_zero_sdk_cost_is_recorded():
    # An explicit $0.00 (e.g. fully cached) is a known cost, not "unknown".
    ac = make_agent_cost("x", "claude-haiku-4-5", input_tokens=100, sdk_cost_usd=0.0)
    assert ac.cost_usd == 0.0


def test_agent_cost_splits_cache_by_price_tier():
    ac = make_agent_cost(
        "x", "claude-haiku-4-5",
        input_tokens=100, cache_read_tokens=900, cache_creation_tokens=50,
        sdk_cost_usd=0.0,
    )
    # The three prompt-side counts are kept separate (three different prices).
    assert ac.input_tokens == 100
    assert ac.cache_read_tokens == 900
    assert ac.cache_creation_tokens == 50
    # "Input" column = uncached + cache writes (full-price tier); reads shown as "Cached".
    assert ac.full_price_input_tokens == 150


def test_agent_cost_from_total_is_cost_only():
    ac = make_agent_cost_from_total("monitor", "claude-haiku-4-5", 0.07)
    assert ac.cost_usd == pytest.approx(0.07)
    assert ac.input_tokens is None
    assert ac.output_tokens is None
    assert ac.cache_read_tokens is None
    # cost-only rows render an em dash in every token column
    assert ac.full_price_input_tokens is None


def test_agent_cost_from_total_none_stays_none():
    ac = make_agent_cost_from_total("monitor", "claude-haiku-4-5", None)
    assert ac.cost_usd is None


# ---------------------------------------------------------------------------
# make_infra_cost
# ---------------------------------------------------------------------------

def test_infra_cost_runtime_times_rate():
    ic = make_infra_cost(
        instance_id="i-1", instance_type="gpu_1x_a100",
        runtime_seconds=3600, hourly_rate_usd=1.49,
    )
    assert ic.cost_usd == pytest.approx(1.49)


def test_infra_cost_half_hour():
    ic = make_infra_cost(runtime_seconds=1800, hourly_rate_usd=2.0)
    assert ic.cost_usd == pytest.approx(1.0)


def test_infra_cost_unknown_rate_is_none():
    ic = make_infra_cost(runtime_seconds=3600, hourly_rate_usd=None)
    assert ic.cost_usd is None


# ---------------------------------------------------------------------------
# RunCostReport totals
# ---------------------------------------------------------------------------

def test_report_totals():
    report = RunCostReport(
        agent_costs=[
            make_agent_cost("a", "claude-haiku-4-5", input_tokens=1_000_000, sdk_cost_usd=1.0),
            make_agent_cost("b", "claude-sonnet-4-6", sdk_cost_usd=0.5),
        ],
        infra_costs=[
            make_infra_cost(runtime_seconds=3600, hourly_rate_usd=2.0),  # $2
        ],
    )
    assert report.total_agent_cost_usd == pytest.approx(1.5)
    assert report.total_infra_cost_usd == pytest.approx(2.0)
    assert report.total_cost_usd == pytest.approx(3.5)
    assert report.has_unknown_costs is False


def test_report_flags_unknown_costs():
    report = RunCostReport(
        # agent line with tokens but no SDK cost → cost unknown
        agent_costs=[make_agent_cost("a", "claude-haiku-4-5", input_tokens=10)],
        infra_costs=[],
    )
    assert report.has_unknown_costs is True
    # unknown counted as 0 in the total (lower bound)
    assert report.total_cost_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_markdown_sections_present():
    report = RunCostReport(
        agent_costs=[make_agent_cost("infra", "claude-sonnet-4-6", sdk_cost_usd=0.12)],
        infra_costs=[
            make_infra_cost(
                instance_id="i-1", instance_type="gpu_1x_a100",
                gpu_type="A100", gpu_count=1,
                runtime_seconds=3600, hourly_rate_usd=1.49,
            )
        ],
    )
    md = render_cost_report_markdown(report)
    assert "## Cost Report" in md
    assert "### Summary" in md
    assert "### Agent Costs" in md
    assert "### Infrastructure Costs" in md
    assert "$1.49" in md          # infra cost
    assert "gpu_1x_a100" in md
    assert "1x A100" in md


def test_render_markdown_splits_input_and_cached():
    report = RunCostReport(
        agent_costs=[
            make_agent_cost(
                "orchestrator", "claude-opus-4-8",
                input_tokens=1_200_000, cache_creation_tokens=2_500_000,
                cache_read_tokens=34_500_000, output_tokens=49_620,
                sdk_cost_usd=23.70,
            )
        ],
    )
    md = render_cost_report_markdown(report)
    # Two-tier columns are present and separated.
    assert "Input Tokens" in md and "Cached Tokens" in md
    # Input = uncached + cache writes (1.2M + 2.5M); reads shown separately.
    assert "3,700,000" in md
    assert "34,500,000" in md
    # Footnote explains why cost << Input x list price.
    assert "billed at ~10%" in md


def test_render_cli_summary():
    report = RunCostReport(
        agent_costs=[make_agent_cost("a", "claude-haiku-4-5", input_tokens=1_000_000, sdk_cost_usd=0.3)],
        infra_costs=[make_infra_cost(runtime_seconds=3600, hourly_rate_usd=2.0)],
    )
    out = render_cli_summary(report)
    assert "Cost summary:" in out
    assert "Agent cost:" in out
    assert "Infra cost: $2.00" in out
    assert "Total cost:" in out
    assert "RUN_REPORT.md" in out


def test_render_cli_summary_flags_unknown():
    report = RunCostReport(
        agent_costs=[make_agent_cost("a", "claude-haiku-4-5", input_tokens=10)],
        infra_costs=[],
    )
    out = render_cli_summary(report)
    assert "lower bound" in out
