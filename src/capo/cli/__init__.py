"""
capo.cli — interactive command-line presentation layer for CAPO.

This package is a thin layer on top of the existing orchestrator. Bare capo
runs a Sonnet-backed assistant that shapes the task, then constructs the same
FineTuningOrchestrator that scripts/run_fine_tuning.py does and calls run_sync();
everything else here (chat layer, run-plan card, log streaming, slash-command
console, config editor, health/history views) is presentation only. No
orchestrator behaviour changes.
"""

from __future__ import annotations
