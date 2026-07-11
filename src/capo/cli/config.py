"""
YAML config for the capo CLI.

By design the CLI reads the SAME config as scripts/run_fine_tuning.py —
scripts/configs/fine_tuning.yaml — so capo and the script stay in lockstep.
CapoConfig is a typed view of that file carrying every field the orchestrator
needs (including the required tolerance_threshold and enable_memory / compaction
/ hub_push), plus the CLI-only cli_mode toggle.

capo config is intentionally a viewer + a surgical cli_mode toggle: the YAML
is heavily commented and yaml.dump would discard those comments, so we never
rewrite the whole file — we edit only the one line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from rich.table import Table

from .colors import console

# src/capo/cli/config.py → parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "scripts" / "configs" / "fine_tuning.yaml"


@dataclass
class CapoConfig:
    """Typed view of scripts/configs/fine_tuning.yaml.

    Defaults mirror scripts/run_fine_tuning.py's cfg.get(...) fallbacks so a
    CLI run with a sparse config behaves identically to the script.
    """

    # --- infrastructure ---
    key_path: str = "~/.ssh/lambda_ed25519"
    ssh_key_name: str = ""
    ssh_alias: Optional[str] = None
    gpu_preference: Optional[str] = "1x GH200"
    allow_reuse_existing: bool = True

    # --- model / training ---
    model_id: str = "facebook/esm2_t6_8M_UR50D"
    fine_tune_strategy: str = "linear-probe"
    dataset_ref: str = "BIIE-AI/ace2_binding"
    probe_max_retries: int = 3

    # --- cost / budget ---
    max_cost_usd: float = 50.0
    tolerance_threshold: float = 0.1  # α = 1 + tolerance_threshold (3-step gate)

    # --- tracking ---
    trackio_space_id: Optional[str] = None

    # --- hub push (finalizer pushes checkpoints/best/) ---
    hub_push: dict = field(default_factory=dict)

    # --- agent ---
    model_name: str = "claude-sonnet-4-6"
    max_turns: int = 1000
    enable_hf_research: bool = True
    enable_memory: bool = True
    
    # effort: low|medium|high|xhigh|max; skills: "all" or "a,b,c".
    orchestrator_effort: Optional[str] = None
    orchestrator_skills: Optional[str] = None

    # --- compaction ---
    compaction_enabled: bool = True
    compaction_threshold_input_tokens: int = 80_000
    compaction_keep_recent_messages: int = 5

    # --- run control ---
    task: Optional[str] = None
    task_file: Optional[str] = None
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    restart_from_checkpoint: bool = False
    resume: Optional[str] = None

    # --- CLI only ---
    cli_mode: str = "interactive"  # interactive | auto

    # --- resolved at load time ---
    config_path: Path = DEFAULT_CONFIG_PATH
    repo_root: Path = _REPO_ROOT

    @property
    def runs_root(self) -> Path:
        """Local directory holding per-run dirs (where the orchestrator writes)."""
        if self.output_dir:
            # output_dir points at a single run dir; its parent is the root.
            return Path(self.output_dir).expanduser().parent
        return self.repo_root / "runs"

    @property
    def key_path_expanded(self) -> str:
        return str(Path(self.key_path).expanduser()) if self.key_path else ""


def load_config(path: Path | str | None = None) -> CapoConfig:
    """Load CapoConfig from the YAML at *path* (default fine_tuning.yaml)."""
    cfg_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    cfg = CapoConfig(config_path=cfg_path)
    if not cfg_path.exists():
        # no config yet — return defaults so bare capo still renders.
        return cfg
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    def g(key: str, default):
        v = raw.get(key, default)
        return default if v is None and default is not None else v

    cfg.key_path = raw.get("key_path") or cfg.key_path
    cfg.ssh_key_name = raw.get("ssh_key_name") or cfg.ssh_key_name
    cfg.ssh_alias = raw.get("ssh_alias")  # may be None
    cfg.gpu_preference = raw.get("gpu_preference", cfg.gpu_preference)
    cfg.allow_reuse_existing = bool(g("allow_reuse_existing", cfg.allow_reuse_existing))
    cfg.model_id = g("model_id", cfg.model_id)
    cfg.fine_tune_strategy = g("fine_tune_strategy", cfg.fine_tune_strategy)
    cfg.dataset_ref = g("dataset_ref", cfg.dataset_ref)
    cfg.probe_max_retries = int(g("probe_max_retries", cfg.probe_max_retries))
    cfg.max_cost_usd = float(g("max_cost_usd", cfg.max_cost_usd))
    cfg.tolerance_threshold = float(g("tolerance_threshold", cfg.tolerance_threshold))
    cfg.trackio_space_id = raw.get("trackio_space_id") or None
    cfg.hub_push = raw.get("hub_push") or {}
    cfg.model_name = g("model_name", cfg.model_name)
    cfg.max_turns = int(g("max_turns", cfg.max_turns))
    cfg.enable_hf_research = bool(g("enable_hf_research", cfg.enable_hf_research))
    cfg.enable_memory = bool(g("enable_memory", cfg.enable_memory))
    cfg.orchestrator_effort = raw.get("orchestrator_effort") or None
    cfg.orchestrator_skills = raw.get("orchestrator_skills") or None
    cfg.compaction_enabled = bool(g("compaction_enabled", cfg.compaction_enabled))
    cfg.compaction_threshold_input_tokens = int(
        g("compaction_threshold_input_tokens", cfg.compaction_threshold_input_tokens)
    )
    cfg.compaction_keep_recent_messages = int(
        g("compaction_keep_recent_messages", cfg.compaction_keep_recent_messages)
    )
    cfg.task = raw.get("task")
    cfg.task_file = raw.get("task_file")
    cfg.run_id = raw.get("run_id")
    cfg.output_dir = raw.get("output_dir")
    cfg.restart_from_checkpoint = bool(g("restart_from_checkpoint", cfg.restart_from_checkpoint))
    cfg.resume = raw.get("resume")
    cfg.cli_mode = (raw.get("cli_mode") or cfg.cli_mode).strip().lower()
    return cfg


def _fmt_yaml_scalar(value) -> str:
    """Render a Python scalar as an unquoted YAML scalar, quoting when needed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    s = str(value)
    if s == "" or s[0] in "!&*?{}[],#|>@'\"%`" or ": " in s or s.strip() != s:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def set_yaml_value(path: Path, key: str, value) -> None:
    """Set a top-level key: in the YAML, preserving every comment and line.

    Rewrites only the existing key: line (regex, single occurrence); appends
    one if the key is absent. Top-level scalars only — never touches nested maps.
    """
    text = path.read_text(encoding="utf-8")
    line = f"{key}: {_fmt_yaml_scalar(value)}"
    pat = re.compile(rf"(?m)^{re.escape(key)}:.*$")
    if pat.search(text):
        # lambda replacement so backslashes in line aren't treated as group refs
        text = pat.sub(lambda _m: line, text, count=1)
    else:
        text = text.rstrip("\n") + f"\n{line}\n"
    path.write_text(text, encoding="utf-8")


def set_cli_mode(path: Path, mode: str) -> None:
    """Set cli_mode: in the YAML, preserving every comment and other line."""
    if mode not in ("interactive", "auto"):
        raise ValueError(f"cli_mode must be interactive|auto, got {mode!r}")
    set_yaml_value(path, "cli_mode", mode)


def _summary_table(cfg: CapoConfig) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="metric.key", no_wrap=True)
    t.add_column("Value", style="brand.dim")
    rows = [
        ("CLI mode", cfg.cli_mode),
        ("SSH key name", cfg.ssh_key_name or "[muted]unset[/muted]"),
        ("SSH key path", cfg.key_path),
        ("GPU preference", cfg.gpu_preference or "[muted]auto[/muted]"),
        ("Model", cfg.model_id),
        ("Strategy", cfg.fine_tune_strategy),
        ("Dataset", cfg.dataset_ref),
        ("Max cost", f"${cfg.max_cost_usd:.2f}"),
        ("Tolerance", f"{cfg.tolerance_threshold}  (α={1 + cfg.tolerance_threshold:.2f})"),
        ("Reuse session", "yes" if cfg.allow_reuse_existing else "no"),
        ("HF research", "yes" if cfg.enable_hf_research else "no"),
        ("Episodic memory", "yes" if cfg.enable_memory else "no"),
        ("Trackio space", cfg.trackio_space_id or "[muted]none[/muted]"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


def print_config_summary(cfg: CapoConfig) -> None:
    """Render the resolved config as a table (used by bare capo and editor)."""
    console.print(_summary_table(cfg))


# editable fields: (label, yaml/attr key, kind, choices). kind ∈
# choice|bool|text|int|float. attr name == YAML key for every row.
_FIELDS: list[tuple[str, str, str, list[str] | None]] = [
    ("CLI mode", "cli_mode", "choice", ["interactive", "auto"]),
    ("Dataset ref", "dataset_ref", "text", None),
    ("Model id", "model_id", "text", None),
    ("Strategy", "fine_tune_strategy", "choice", ["linear-probe", "lora", "full"]),
    ("GPU preference", "gpu_preference", "text", None),
    ("Max cost (USD)", "max_cost_usd", "float", None),
    ("Tolerance", "tolerance_threshold", "float", None),
    ("SSH key name", "ssh_key_name", "text", None),
    ("SSH key path", "key_path", "text", None),
    ("Reuse instance", "allow_reuse_existing", "bool", None),
    ("HF research", "enable_hf_research", "bool", None),
    ("Episodic memory", "enable_memory", "bool", None),
    ("Trackio space", "trackio_space_id", "text", None),
    ("Probe retries", "probe_max_retries", "int", None),
    ("Max turns", "max_turns", "int", None),
]


def _fmt_value(v) -> str:
    """Render a field value for the editor list (bool → yes/no, empty → —)."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "—" if v in (None, "") else str(v)


def _row_value(cfg: CapoConfig, attr: str) -> str:
    return _fmt_value(getattr(cfg, attr))


def _edit_field(label: str, kind: str, choices: list[str] | None, cur) -> object | None:
    """Prompt for a new value; return it (or None to keep the current value)."""
    from .widgets import select_one, text_input

    if kind == "choice" and choices:
        return select_one(label, choices, default=str(cur), allow_other=False)
    if kind == "bool":
        return select_one(label, ["yes", "no"], default=("yes" if cur else "no"),
                          allow_other=False) == "yes"
    raw = text_input(label, default="" if cur in (None, "") else str(cur)).strip()
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            return None
    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            return None
    return raw  # text


def interactive_config_editor(cfg: CapoConfig) -> CapoConfig:
    """Arrow-key config editor: navigate fields, edit values, Save or Cancel.

    Edits are buffered in memory (a changed field is marked ●). [ Save ] writes
    every pending change to the YAML with comment-preserving single-line edits
    (set_yaml_value) — the heavily-commented file is never re-dumped. [ Cancel ]
    (or Esc) discards everything. Called by capo config and /config.
    """
    from .widgets import _opt_lines, _select

    console.print()
    console.rule("[brand]CAPO configuration[/]", style="brand.dim")
    console.print(f"  [muted]{cfg.config_path}[/]\n")

    if not cfg.config_path.exists():
        print_config_summary(cfg)
        console.print("\n  [err]Config file not found — nothing to edit.[/]\n")
        return cfg

    console.print("  [prompt.hint]↑/↓ move · Enter edit · select Save / Cancel to finish[/]\n")
    pending: dict[str, object] = {}  # attr → new value, buffered until Save
    last = 0
    n_fields = len(_FIELDS)

    def _rows() -> list[str]:
        field_rows = []
        for label, attr, _k, _c in _FIELDS:
            eff = pending.get(attr, getattr(cfg, attr))
            mark = "●" if attr in pending else " "
            field_rows.append(f"{mark} {label:<16}{_fmt_value(eff)}")
        n = len(pending)
        save = "[ Save ]" + (f"  ({n} change{'' if n == 1 else 's'})" if n else "")
        return field_rows + [save, "[ Cancel ]"]

    while True:
        rows = _rows()
        idx = _select(lambda cur: _opt_lines(rows, cur), len(rows), default_idx=last)
        if idx is None or idx == n_fields + 1:  # Esc or [ Cancel ]
            if pending:
                console.print("  [muted]Cancelled — no changes written.[/]\n")
            return cfg
        if idx == n_fields:  # [ Save ]
            for attr, new in pending.items():
                set_yaml_value(cfg.config_path, attr, new)
                setattr(cfg, attr, new)
            if pending:
                console.print(f"  [ok]✓[/] Saved {len(pending)} change"
                              f"{'' if len(pending) == 1 else 's'} → [brand.dim]{cfg.config_path}[/]\n")
            else:
                console.print("  [muted]No changes to save.[/]\n")
            return cfg

        last = idx
        label, attr, kind, choices = _FIELDS[idx]
        eff = pending.get(attr, getattr(cfg, attr))
        try:
            new = _edit_field(label, kind, choices, eff)
        except KeyboardInterrupt:
            continue
        if new is None:
            continue
        if new == getattr(cfg, attr):
            pending.pop(attr, None)  # reverted to the on-disk value
        else:
            pending[attr] = new
