"""Context compaction subsystem.

Compresses long agent message histories into a structured "case file" that
is carried forward across phase boundaries, so downstream agent calls
don't re-pay the full input-token cost of the prior phase.

See ``compactor.py`` for the core algorithm and the docstring in each
module for tuning notes.
"""

from capo.context.compaction.compactor import (
    Compactor,
    NullCompactor,
    dedupe_tool_results,
    format_messages_for_summarizer,
)
from capo.context.compaction.store import CompactionStore
from capo.context.compaction.types import (
    CaseFile,
    CompactionConfig,
    CompactionEvent,
    merge_case_file,
)

__all__ = [
    "CaseFile",
    "Compactor",
    "CompactionConfig",
    "CompactionEvent",
    "CompactionStore",
    "NullCompactor",
    "dedupe_tool_results",
    "format_messages_for_summarizer",
    "merge_case_file",
]
