"""Scientific report rendering helpers.

Pure, I/O-free utilities for building the paper-style markdown that the
finalizer writes into RUN_REPORT.md — tables, metric formatting, cost and
recovery sections.
"""

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

__all__ = [
    "EMDASH",
    "fmt_cost",
    "fmt_cost_precise",
    "fmt_int",
    "fmt_loss",
    "fmt_metric",
    "fmt_runtime",
    "fmt_value",
    "markdown_table",
]
