"""FastMCP registration shim for the Lambda Cloud GPU tools.

All functional behaviour lives in :mod:`capo.mcp.tools.lambda_tools` as
plain Python functions returning ``dict``. This module:

- registers each canonical tool with FastMCP
- wraps it with :func:`logged_tool` for structured trace logs in
  ``logs/agent_tool_calls/lambda_mcp_server/...``

FastMCP validates the wrapper's return value against the wrapped function's
type annotation (``dict[str, Any]``) and handles the JSON envelope itself,
so the wrapper returns the dict unchanged.

The ``RUN_CTX`` is owned by the tools module so trace events emitted from
within the tool functions (e.g. ``session_start`` in ``lambda_start_session``)
are recorded against the same run as the wrapper-level tool calls.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from capo.utils.logging_utils import logged_tool
from capo.mcp.tools import lambda_tools as _tools

mcp = FastMCP("lambda-repl")
RUN_CTX = _tools.RUN_CTX


def _register(name: str, fn: Callable[..., dict[str, Any]]) -> None:
    """Wrap ``fn`` with :func:`logged_tool` and register it with FastMCP.

    The wrapper preserves ``fn``'s signature so FastMCP can introspect the
    parameter schema and the return type for serialization.
    """
    sig = inspect.signature(fn)

    @logged_tool(RUN_CTX, name)
    def _logged(**kwargs: Any) -> dict[str, Any]:
        return fn(**kwargs)

    _logged.__signature__ = sig  # type: ignore[attr-defined]
    _logged.__name__ = name
    _logged.__doc__ = fn.__doc__ or ""
    mcp.tool(name=name)(_logged)


for tool_name, tool_fn in _tools.CANONICAL_TOOLS:
    _register(tool_name, tool_fn)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
