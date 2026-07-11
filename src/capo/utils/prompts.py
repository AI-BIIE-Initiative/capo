"""Load prompt text from the capo prompt library (src/capo/prompts/).

All LLM prompts used by the capo system live as .md files under
src/capo/prompts/. each constant becomes a one-line
load_prompt(...) call. this helper only returns the raw
template text (with {{/}} brace escaping preserved verbatim).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the raw text of a prompt file by slash-path, without extension.

    name is a forward-slash path relative to src/capo/prompts/ with no
    .md suffix, e.g. "orchestrator/system_prompts/infrastructure". The
    content is returned exactly as stored (no stripping) so leading/trailing
    whitespace and brace escaping are preserved.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
