"""Friendly preflight validation for the three API keys the pipeline needs.

The orchestrator depends on:

* ANTHROPIC_API_KEY — used transparently by the Claude Agent SDK. When it is
  missing or invalid the SDK still returns a successful ResultMessage and
  the agent's answer text contains the error ("Credit balance is too low",
  "invalid x-api-key", …). Without an upfront check the user only sees a
  generic Pre-launch failed: infra.json missing later in the pipeline.
* LAMBDA_API_KEY — required for any new GPU provisioning.
* HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) — required for dataset access,
  trackio dashboards, and the final HF Hub model push.

This module loads .env once, classifies every key as ok / missing /
blank, prints a single readable status block, and raises
:class:`MissingAPIKeyError` listing every missing key (instead of failing on
the first one) so the user can fix them in a single edit.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


_REQUIRED_KEYS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "ANTHROPIC_API_KEY",
        ("ANTHROPIC_API_KEY",),
        "Claude Agent SDK auth — get one at https://console.anthropic.com/settings/keys",
    ),
    (
        "LAMBDA_API_KEY",
        ("LAMBDA_API_KEY",),
        "Lambda GPU provisioning — create at https://cloud.lambda.ai/api-keys/cloud-api",
    ),
    (
        "HF_TOKEN",
        ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        "HF Hub dataset/model access + trackio — https://huggingface.co/settings/tokens",
    ),
)


@dataclass(frozen=True)
class KeyStatus:
    label: str           # User-facing key name (e.g. "ANTHROPIC_API_KEY").
    status: str          # "ok" | "missing" | "blank"
    source: str          # "env" | "-"
    purpose: str         # Short description shown when missing.

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class MissingAPIKeyError(RuntimeError):
    """Raised when one or more required API keys are missing or blank."""

    def __init__(self, missing: list[KeyStatus]):
        self.missing = missing
        super().__init__(self._format_message(missing))

    @staticmethod
    def _format_message(missing: list[KeyStatus]) -> str:
        lines = ["Missing required API key(s):"]
        for ks in missing:
            verb = "not set" if ks.status == "missing" else "set but blank"
            lines.append(f"  - {ks.label} ({verb}) — {ks.purpose}")
        lines.append("")
        lines.append(
            "Add the missing values to your .env file at the repo root "
            "(or export them in your shell), then re-run."
        )
        return "\n".join(lines)


def load_env(repo_root: Path | None = None) -> Path | None:
    """Load .env from the repo root. Returns the path that was loaded, or None."""
    root = repo_root or Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return env_path
    # Fall back to default search (cwd, parents) — covers atypical layouts.
    load_dotenv(override=False)
    return None


def _classify(label: str, aliases: Iterable[str], purpose: str) -> KeyStatus:
    for name in aliases:
        raw = os.environ.get(name)
        if raw is None:
            continue
        if raw.strip() == "":
            return KeyStatus(label=label, status="blank", source="env", purpose=purpose)
        return KeyStatus(label=label, status="ok", source="env", purpose=purpose)
    return KeyStatus(label=label, status="missing", source="-", purpose=purpose)


def check_api_keys() -> list[KeyStatus]:
    """Return one :class:`KeyStatus` per required key, in declaration order."""
    return [_classify(label, aliases, purpose) for label, aliases, purpose in _REQUIRED_KEYS]


def render_status(checks: list[KeyStatus], env_path: Path | None) -> str:
    """Format a short, readable status block."""
    src = str(env_path) if env_path else "no .env file found (using process environment)"
    width = max(len(ks.label) for ks in checks)
    lines = [f"API key preflight ({src}):"]
    for ks in checks:
        marker = "OK " if ks.ok else "!! "
        detail = "set" if ks.ok else ("not set" if ks.status == "missing" else "blank")
        lines.append(f"  [{marker}] {ks.label:<{width}}  {detail}")
    return "\n".join(lines)


def assert_api_keys(*, stream=sys.stderr) -> list[KeyStatus]:
    """Load .env, validate every required key, print a status block, raise on missing.

    Returns the list of :class:`KeyStatus` on success so the caller can log it.
    """
    env_path = load_env()
    checks = check_api_keys()
    print(render_status(checks, env_path), file=stream)
    missing = [ks for ks in checks if not ks.ok]
    if missing:
        raise MissingAPIKeyError(missing)
    return checks
