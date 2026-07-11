"""capo.evaluation — leakage-isolated external evaluation framework.

The public surface is intentionally small. Stage-2 evaluation (the part that is
allowed to see the held-out golden set) lives entirely in :mod:`leakage_firewall`,
which is pure deterministic Python — no agent, no SDK, no LLM calls.
"""

from capo.evaluation.leakage_firewall import (
    CANDIDATE_SCHEMA,
    CandidateAdapter,
    GoldEvaluator,
    SystemRun,
    canonicalize_species,
    run_stage2_evaluation,
    write_evaluation_config,
    write_processed_examples,
    write_raw_data_manifest,
    write_run_metadata,
)
from capo.evaluation.harness_plots import generate_harness_plots

__all__ = [
    "CANDIDATE_SCHEMA",
    "CandidateAdapter",
    "GoldEvaluator",
    "SystemRun",
    "canonicalize_species",
    "run_stage2_evaluation",
    "write_evaluation_config",
    "write_processed_examples",
    "write_raw_data_manifest",
    "write_run_metadata",
    "generate_harness_plots",
]
