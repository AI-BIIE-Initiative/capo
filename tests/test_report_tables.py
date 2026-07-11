"""Tests for capo.report.tables — markdown table rendering + value formatting."""

from __future__ import annotations

from capo.report.tables import (
    EMDASH,
    fmt_cost,
    fmt_cost_precise,
    fmt_int,
    fmt_loss,
    fmt_metric,
    fmt_runtime,
    fmt_value,
    markdown_table,
)


# ---------------------------------------------------------------------------
# Value formatters
# ---------------------------------------------------------------------------

def test_fmt_metric_four_decimals():
    assert fmt_metric(0.87654321) == "0.8765"
    assert fmt_metric(1) == "1.0000"


def test_fmt_metric_missing_and_bad():
    assert fmt_metric(None) == EMDASH
    assert fmt_metric("n/a") == EMDASH


def test_fmt_loss_alias():
    assert fmt_loss(0.123456) == "0.1235"
    assert fmt_loss(None) == EMDASH


def test_fmt_cost():
    assert fmt_cost(1.2) == "$1.20"
    assert fmt_cost(0) == "$0.00"
    assert fmt_cost(None) == EMDASH


def test_fmt_cost_precise():
    assert fmt_cost_precise(0.01234) == "$0.0123"
    assert fmt_cost_precise(None) == EMDASH


def test_fmt_int_thousands():
    assert fmt_int(1234567) == "1,234,567"
    assert fmt_int(None) == EMDASH


def test_fmt_runtime():
    assert fmt_runtime(0) == "0s"
    assert fmt_runtime(45) == "45s"
    assert fmt_runtime(95) == "1m 35s"
    assert fmt_runtime(3725) == "1h 2m 5s"
    assert fmt_runtime(3600) == "1h"
    assert fmt_runtime(None) == EMDASH
    assert fmt_runtime(-5) == EMDASH


def test_fmt_value_dispatch():
    assert fmt_value(None) == EMDASH
    assert fmt_value(True) == "yes"
    assert fmt_value(False) == "no"
    assert fmt_value(1500) == "1,500"
    assert fmt_value(0.5) == "0.5000"
    assert fmt_value("hello") == "hello"


# ---------------------------------------------------------------------------
# markdown_table
# ---------------------------------------------------------------------------

def test_markdown_table_basic_shape():
    rows = [{"a": "x", "b": "y"}, {"a": "z", "b": "w"}]
    out = markdown_table(rows, columns=["a", "b"])
    lines = out.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| :--- | :--- |"
    assert lines[2] == "| x | y |"
    assert lines[3] == "| z | w |"


def test_markdown_table_headers_and_align():
    rows = [{"k": "Agent", "v": 1.5}]
    out = markdown_table(
        rows,
        columns=["k", "v"],
        headers={"k": "Name", "v": "Value"},
        align={"v": "right"},
    )
    lines = out.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| :--- | ---: |"
    # raw float gets the default 4dp formatter
    assert lines[2] == "| Agent | 1.5000 |"


def test_markdown_table_missing_values_are_emdash():
    rows = [{"a": "x"}]  # 'b' absent
    out = markdown_table(rows, columns=["a", "b"])
    assert out.splitlines()[2] == f"| x | {EMDASH} |"


def test_markdown_table_escapes_pipes_and_newlines():
    rows = [{"a": "a|b", "b": "line1\nline2"}]
    out = markdown_table(rows, columns=["a", "b"])
    body = out.splitlines()[2]
    assert "a\\|b" in body
    assert "\n" not in body
    assert "line1 line2" in body


def test_markdown_table_no_columns_is_empty():
    assert markdown_table([{"a": 1}], columns=[]) == ""


def test_markdown_table_no_rows_keeps_header():
    out = markdown_table([], columns=["a", "b"])
    assert out.splitlines() == ["| a | b |", "| :--- | :--- |"]
