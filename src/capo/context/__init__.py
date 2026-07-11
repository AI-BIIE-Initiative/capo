"""Context-management primitives for LLM calls.

This package holds primitives that shape what the model sees on each
call:

- ``prompt_caching`` — Anthropic ephemeral cache breakpoints and the
  helper that splits a prompt into a cached stable prefix + uncached
  mutable tail.
- ``compaction`` — case-file + rolling-tail context compaction that
  carries a structured summary across orchestrator phases instead of
  re-paying the full message-history input cost.

Modules import from these sub-packages directly
(e.g. ``from capo.context.compaction import Compactor``); this
``__init__`` intentionally re-exports nothing to keep the import graph
predictable and match the convention of every other top-level package
in ``capo/``.
"""
