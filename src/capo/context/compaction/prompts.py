"""
Compaction prompts.

The summarizer is invoked with these prompts to convert a chunk of message
history (formatted as plain text) into a structured JSON case-file delta.

Tuning notes
------------
- Recall first, precision second. The system prompt explicitly instructs
  exhaustive extraction before deduplication. We'd rather over-include
  than silently drop a constraint we'll need three phases later.
- File paths are preserved verbatim. They are the durable index back into
  the disk artifacts the orchestrator writes.
- Tool calls that were superseded by a later identical call are dropped.
  Only the final outcome matters for downstream phases.
- The narrative is capped to ~500 tokens so it stays cheap to ship as a
  prompt preamble even after many compactions.
"""

from __future__ import annotations
from capo.utils.prompts import load_prompt


COMPACTION_SYSTEM_PROMPT = load_prompt("context/system_prompts/compaction")


def build_user_message(
    *,
    prior_case_file_markdown: str | None,
    older_messages_text: str,
    rolling_tail_summary: str,
) -> str:
    """Assemble the user-facing prompt for one compaction call.

    `older_messages_text` is the formatted text of the messages we want
    summarized. `rolling_tail_summary` is a brief description of the most
    recent N messages that we are NOT summarizing — included only so the
    summarizer understands where the cutoff is and doesn't try to
    distill items that the agent will see verbatim.
    """
    parts: list[str] = []

    if prior_case_file_markdown:
        parts.append("## Prior case file (already distilled — extend, do not repeat)")
        parts.append(prior_case_file_markdown.strip())
        parts.append("")

    parts.append("## Messages to summarize (older portion)")
    parts.append(older_messages_text.strip() or "(empty)")
    parts.append("")

    if rolling_tail_summary:
        parts.append("## Rolling tail (NOT to be summarized — context only)")
        parts.append(
            "These messages are kept verbatim by the orchestrator and will be "
            "visible to the next phase. They are listed here only so you "
            "understand the cutoff."
        )
        parts.append(rolling_tail_summary.strip())
        parts.append("")

    parts.append("## Now produce the JSON case file delta. JSON only.")
    return "\n".join(parts)
