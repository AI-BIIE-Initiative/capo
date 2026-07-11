"""
RunConfig — the fully-resolved settings the CLI hands to the orchestrator.

The old fixed 4-question questionnaire is gone; the Sonnet chat layer (chat.py)
now gathers intent. This module keeps only the dataclass that carries every
field FineTuningOrchestrator needs, identical to the args
scripts/run_fine_tuning.py passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import CapoConfig


@dataclass
class RunConfig:
    """Fully resolved run settings — everything app.py needs to build the
    orchestrator, identical to the args scripts/run_fine_tuning.py passes."""

    # --- task-shaping (set by the chat layer / auto mode) ---
    dataset_ref: str = ""
    task_description: str = ""
    fine_tune_strategy: str = "linear-probe"
    max_cost_usd: float = 50.0
    # --- passthrough from config ---
    model_id: str = ""
    model_name: str = "claude-sonnet-4-6"
    gpu_preference: Optional[str] = None
    key_path: str = ""
    ssh_key_name: str = ""
    ssh_alias: Optional[str] = None
    allow_reuse_existing: bool = True
    tolerance_threshold: float = 0.1
    trackio_space_id: Optional[str] = None
    probe_max_retries: int = 3
    max_turns: int = 1000
    enable_hf_research: bool = True
    enable_memory: bool = True
    # (config-gated, default off): main-orchestrator reasoning + skills.
    orchestrator_effort: Optional[str] = None
    orchestrator_skills: Optional[str] = None
    hub_push: dict = field(default_factory=dict)
    compaction_enabled: bool = True
    compaction_threshold_input_tokens: int = 80_000
    compaction_keep_recent_messages: int = 5
    output_dir: Optional[str] = None
    run_id: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: CapoConfig) -> "RunConfig":
        return cls(
            dataset_ref=cfg.dataset_ref,
            fine_tune_strategy=cfg.fine_tune_strategy,
            max_cost_usd=cfg.max_cost_usd,
            model_id=cfg.model_id,
            model_name=cfg.model_name,
            gpu_preference=cfg.gpu_preference,
            key_path=cfg.key_path,
            ssh_key_name=cfg.ssh_key_name,
            ssh_alias=cfg.ssh_alias,
            allow_reuse_existing=cfg.allow_reuse_existing,
            tolerance_threshold=cfg.tolerance_threshold,
            trackio_space_id=cfg.trackio_space_id,
            probe_max_retries=cfg.probe_max_retries,
            max_turns=cfg.max_turns,
            enable_hf_research=cfg.enable_hf_research,
            enable_memory=cfg.enable_memory,
            orchestrator_effort=cfg.orchestrator_effort,
            orchestrator_skills=cfg.orchestrator_skills,
            hub_push=dict(cfg.hub_push),
            compaction_enabled=cfg.compaction_enabled,
            compaction_threshold_input_tokens=cfg.compaction_threshold_input_tokens,
            compaction_keep_recent_messages=cfg.compaction_keep_recent_messages,
            output_dir=cfg.output_dir,
            run_id=cfg.run_id,
        )
