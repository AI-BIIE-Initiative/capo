"""
Unit tests for src/capo/compaction/.

Tests cover the pure logic -> threshold predicate, dedupe, message formatting,
case-file merge, JSON parsing and disk persistence. 
The summarizer is injected as a fake so no API calls are made.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from claude_agent_sdk.types import (
    AssistantMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from capo.context.compaction import (
    CaseFile,
    Compactor,
    CompactionConfig,
    CompactionStore,
    NullCompactor,
    dedupe_tool_results,
    format_messages_for_summarizer,
    merge_case_file,
)
from capo.context.compaction.compactor import _parse_summary_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_use(name: str, input_dict: dict, id_: str = "tool-x") -> ToolUseBlock:
    return ToolUseBlock(id=id_, name=name, input=input_dict)


def _assistant(blocks: list) -> AssistantMessage:
    return AssistantMessage(content=blocks, model="claude-sonnet-4-6")


def _tool_result(tool_use_id: str, text: str) -> UserMessage:
    return UserMessage(
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=text)]
    )


def _make_long_trace(n_pairs: int) -> list:
    """Build n_pairs (assistant tool_use, user tool_result) pairs."""
    msgs: list = []
    for i in range(n_pairs):
        tu_id = f"tu-{i}"
        msgs.append(_assistant([_tool_use("Bash", {"cmd": f"ls -la dir_{i}"}, id_=tu_id)]))
        msgs.append(_tool_result(tu_id, f"output for dir_{i}"))
    return msgs


# ---------------------------------------------------------------------------
# should_compact predicate
# ---------------------------------------------------------------------------


def test_below_threshold_no_compaction(tmp_path: Path):
    cfg = CompactionConfig(
        enabled=True,
        threshold_input_tokens=80_000,
        keep_recent_messages=5,
        min_older_messages=8,
    )
    c = Compactor(cfg, run_id="test", local_run_dir=tmp_path)

    # Threshold not met: 50k < 80k
    assert c.should_compact(last_input_tokens=50_000, n_messages=20) is False
    # Threshold met but too few messages
    assert c.should_compact(last_input_tokens=100_000, n_messages=10) is False
    # Both met
    assert c.should_compact(last_input_tokens=100_000, n_messages=20) is True


def test_disabled_never_compacts(tmp_path: Path):
    cfg = CompactionConfig(enabled=False, threshold_input_tokens=0)
    c = Compactor(cfg, run_id="test", local_run_dir=tmp_path)
    assert c.should_compact(last_input_tokens=10_000_000, n_messages=10_000) is False


def test_null_compactor_is_noop(tmp_path: Path):
    null = NullCompactor()
    assert null.should_compact(last_input_tokens=10_000_000, n_messages=10_000) is False


def test_maybe_compact_returns_none_below_threshold(tmp_path: Path):
    cfg = CompactionConfig(threshold_input_tokens=80_000, keep_recent_messages=5)
    c = Compactor(cfg, run_id="test", local_run_dir=tmp_path)
    result = asyncio.run(c.maybe_compact(
        phase_label="t",
        raw_messages=_make_long_trace(20),
        prior_case_file=None,
        last_input_tokens=10,
    ))
    assert result is None


# ---------------------------------------------------------------------------
# dedupe_tool_results
# ---------------------------------------------------------------------------


def test_dedupes_repeated_tool_calls():
    """Two identical Bash calls with the same input — only the last survives."""
    msgs = [
        _assistant([_tool_use("Bash", {"cmd": "nvidia-smi"}, id_="a1")]),
        _tool_result("a1", "output 1"),
        _assistant([_tool_use("Read", {"file": "/x"}, id_="a2")]),  # different tool
        _tool_result("a2", "x contents"),
        _assistant([_tool_use("Bash", {"cmd": "nvidia-smi"}, id_="a3")]),  # dup of a1
        _tool_result("a3", "output 3"),
    ]
    deduped = dedupe_tool_results(msgs)
    # The first Bash(nvidia-smi) and its result should be gone.
    tool_use_ids = []
    tool_result_ids = []
    for m in deduped:
        if isinstance(m, AssistantMessage):
            for b in m.content:
                if isinstance(b, ToolUseBlock):
                    tool_use_ids.append(b.id)
        elif isinstance(m, UserMessage) and isinstance(m.content, list):
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    tool_result_ids.append(b.tool_use_id)
    assert "a1" not in tool_use_ids, "first duplicate Bash call should be dropped"
    assert "a3" in tool_use_ids, "last Bash call should survive"
    assert "a2" in tool_use_ids, "non-duplicate tool should pass through"
    assert "a1" not in tool_result_ids
    assert "a3" in tool_result_ids


def test_dedupe_preserves_order_when_no_dupes():
    msgs = [
        _assistant([_tool_use("Read", {"file": "/a"}, id_="a")]),
        _tool_result("a", "..."),
        _assistant([_tool_use("Read", {"file": "/b"}, id_="b")]),
        _tool_result("b", "..."),
    ]
    deduped = dedupe_tool_results(msgs)
    assert len(deduped) == len(msgs)


# ---------------------------------------------------------------------------
# format_messages_for_summarizer
# ---------------------------------------------------------------------------


def test_format_messages_truncates_long_tool_results():
    big = "x" * 10_000
    msgs = [
        _assistant([_tool_use("Bash", {"cmd": "echo big"}, id_="z")]),
        _tool_result("z", big),
    ]
    text = format_messages_for_summarizer(msgs)
    assert "tool_call] Bash" in text
    assert "tool_result]" in text
    # Big tool result must be truncated.
    assert big not in text
    assert "…" in text


def test_format_handles_string_user_messages():
    msgs = [UserMessage(content="hello there")]
    text = format_messages_for_summarizer(msgs)
    assert "[user] hello there" in text


# ---------------------------------------------------------------------------
# CaseFile round-trip + markdown
# ---------------------------------------------------------------------------


def test_case_file_round_trip():
    cf = CaseFile(
        run_id="ft-x",
        updated_at="2026-05-05T12:00:00Z",
        decisions=["use 1xH100", "linear-probe"],
        constraints=["budget $50"],
        file_findings={"infra.json": "instance type is gh200"},
        artifacts_produced=["/tmp/profile.json"],
        narrative="Profiled dataset, ran probe, gated cost.",
        cumulative_messages_summarized=12,
        cumulative_tokens_before=50_000,
        cumulative_tokens_after=4_000,
        compactions=1,
    )
    text = cf.to_json()
    rebuilt = CaseFile.from_json(text)
    assert rebuilt.decisions == cf.decisions
    assert rebuilt.constraints == cf.constraints
    assert rebuilt.file_findings == cf.file_findings
    assert rebuilt.compactions == 1


def test_case_file_to_markdown_renders_required_sections():
    cf = CaseFile(
        run_id="ft-x",
        updated_at="2026-05-05T12:00:00Z",
        decisions=["use H100"],
        constraints=["budget $50"],
        file_findings={"a.json": "summary"},
        artifacts_produced=["/x"],
        errors_resolved=["fixed OOM"],
        open_questions=["which checkpoint?"],
        narrative="story",
    )
    md = cf.to_markdown()
    assert "## Decisions" in md
    assert "## Constraints" in md
    assert "## Files visited" in md
    assert "## Artifacts produced" in md
    assert "## Errors resolved" in md
    assert "## Open questions" in md
    assert "## Narrative" in md


def test_case_file_markdown_omits_empty_sections():
    cf = CaseFile(
        run_id="ft-x",
        updated_at="2026-05-05T12:00:00Z",
        decisions=["only thing"],
    )
    md = cf.to_markdown()
    assert "## Decisions" in md
    assert "## Constraints" not in md
    assert "## Files visited" not in md


# ---------------------------------------------------------------------------
# merge_case_file
# ---------------------------------------------------------------------------


def test_merge_case_file_extends_unique_lists_and_overrides_findings():
    prior = CaseFile(
        run_id="r",
        updated_at="2026-01-01T00:00:00Z",
        decisions=["d1"],
        constraints=["c1"],
        file_findings={"a.json": "old"},
        narrative="old story",
        cumulative_messages_summarized=5,
        cumulative_tokens_before=10_000,
        cumulative_tokens_after=1_000,
        compactions=1,
    )
    additions = {
        "decisions": ["d1", "d2"],          # d1 is duplicate, should not double
        "constraints": ["c2"],
        "file_findings": {"a.json": "new", "b.json": "fresh"},
        "open_questions": [],
        "errors_resolved": ["fixed"],
        "artifacts_produced": ["/tmp/x"],
        "narrative": "new story",
    }
    merged = merge_case_file(
        prior, run_id="r", additions=additions, metrics_delta=(7, 20_000, 2_000)
    )
    assert merged.decisions == ["d1", "d2"]
    assert merged.constraints == ["c1", "c2"]
    assert merged.file_findings == {"a.json": "new", "b.json": "fresh"}
    assert merged.errors_resolved == ["fixed"]
    assert merged.narrative == "new story"
    assert merged.compactions == 2
    assert merged.cumulative_messages_summarized == 12
    assert merged.cumulative_tokens_before == 30_000
    assert merged.cumulative_tokens_after == 3_000


def test_merge_case_file_from_none_creates_fresh():
    merged = merge_case_file(
        None,
        run_id="r",
        additions={"decisions": ["d1"], "narrative": "n"},
        metrics_delta=(3, 1_000, 100),
    )
    assert merged.decisions == ["d1"]
    assert merged.compactions == 1
    assert merged.cumulative_tokens_before == 1_000


# ---------------------------------------------------------------------------
# _parse_summary_json
# ---------------------------------------------------------------------------


def test_parse_summary_json_clean_json():
    text = json.dumps({"decisions": ["d"], "constraints": [], "file_findings": {}, "narrative": ""})
    parsed = _parse_summary_json(text)
    assert parsed is not None
    assert parsed["decisions"] == ["d"]


def test_parse_summary_json_strips_fences():
    text = '```json\n{"decisions": ["d"]}\n```'
    parsed = _parse_summary_json(text)
    assert parsed is not None
    assert parsed["decisions"] == ["d"]


def test_parse_summary_json_extracts_from_prose():
    text = 'Sure! Here is the JSON: {"decisions": ["d"]} — hope this helps!'
    parsed = _parse_summary_json(text)
    assert parsed is not None


def test_parse_summary_json_rejects_unrelated_dict():
    text = '{"some_unrelated_key": 1}'
    parsed = _parse_summary_json(text)
    assert parsed is None


def test_parse_summary_json_rejects_garbage():
    assert _parse_summary_json("not json at all") is None
    assert _parse_summary_json("") is None


# ---------------------------------------------------------------------------
# Compactor end-to-end with a fake summarizer
# ---------------------------------------------------------------------------


def test_compactor_end_to_end_with_fake_summarizer(tmp_path: Path):
    cfg = CompactionConfig(
        enabled=True,
        threshold_input_tokens=100,   # easy to cross
        keep_recent_messages=2,
        min_older_messages=2,
    )

    fake_response = json.dumps({
        "decisions": ["use H100"],
        "constraints": ["budget $50"],
        "file_findings": {"infra.json": "gh200 reused"},
        "open_questions": [],
        "errors_resolved": [],
        "artifacts_produced": ["/tmp/profile.json"],
        "narrative": "Phase A wrapped up cleanly.",
    })

    async def fake_summarizer(system_prompt, user_message, model):
        # Sanity-check the prompts get assembled correctly.
        assert "Output format" in system_prompt
        assert "Messages to summarize" in user_message
        return fake_response, 0.0123

    c = Compactor(
        cfg,
        run_id="ft-test",
        local_run_dir=tmp_path,
        summarizer=fake_summarizer,
    )

    msgs = _make_long_trace(6)  # 12 messages: 10 older + 2 tail
    case_file = asyncio.run(c.maybe_compact(
        phase_label="phase_a_pre_launch",
        raw_messages=msgs,
        prior_case_file=None,
        last_input_tokens=200,
    ))

    assert case_file is not None
    assert case_file.run_id == "ft-test"
    assert "use H100" in case_file.decisions
    assert "infra.json" in case_file.file_findings
    assert case_file.compactions == 1
    assert case_file.cumulative_tokens_before > 0
    assert case_file.cumulative_tokens_after > 0

    # Persisted artifacts on disk.
    store = CompactionStore(tmp_path)
    persisted = store.load_case_file()
    assert persisted is not None
    assert persisted.decisions == case_file.decisions

    # History event written.
    history_lines = store.history.read_text().strip().splitlines()
    assert len(history_lines) == 1
    event = json.loads(history_lines[0])
    assert event["phase_label"] == "phase_a_pre_launch"
    assert event["messages_summarized"] > 0
    assert event["summarizer_cost_usd"] == pytest.approx(0.0123)


def test_compactor_skips_when_summarizer_returns_garbage(tmp_path: Path):
    cfg = CompactionConfig(threshold_input_tokens=100, keep_recent_messages=2, min_older_messages=2)

    async def garbage_summarizer(system_prompt, user_message, model):
        return "this is not json", 0.0

    c = Compactor(cfg, run_id="r", local_run_dir=tmp_path, summarizer=garbage_summarizer)
    out = asyncio.run(c.maybe_compact(
        phase_label="t",
        raw_messages=_make_long_trace(6),
        prior_case_file=None,
        last_input_tokens=200,
    ))
    assert out is None
    # No case file persisted.
    assert not CompactionStore(tmp_path).case_file_json.exists()
