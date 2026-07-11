"""
Local REPL environment for executing Python code.

This module provides a class for a local, sandboxed Python REPL environment.
It supports persistent state, context management, and communication with a
language model.
"""

import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

from capo.mcp.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from capo.mcp.core.types import REPLResult, LanguageModelChatCompletion
from capo.mcp.environments.base_env import NonIsolatedEnv
from capo.utils.logging_utils import log_trace_event
from capo.mcp.environments.constants import SAFE_BUILTINS


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.

    Executes code in a sandboxed namespace with access to context data,
    and supports communication with a language model. The state of the
    REPL can be persisted across sessions.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        work_dir: str | None = None,
        depth: int = 1,
        **kwargs,
    ):
        """
        Initializes the LocalREPL environment.

        Args:
            lm_handler_address: The address of the language model handler.
            context_payload: Data to be loaded into the REPL's context.
            setup_code: Code to be executed upon initialization.
            persistent: Whether to persist the REPL state across sessions.
            depth: The recursion depth for LLM calls.
        """
        super().__init__(persistent=persistent, depth=depth, **kwargs)

        self.lm_handler_address = lm_handler_address
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self.work_dir = work_dir
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0

        # Setup globals, locals, and modules in environment.
        self.setup()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment by creating a sandboxed namespace."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[LanguageModelChatCompletion] = []

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched

    def _final_var(self, variable_name: str) -> str:
        """Return the value of a variable as a final answer."""
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])

        # Provide helpful error message with available variables
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"No variables have been created yet. "
            f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {
            k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")
        }
        if not available:
            return (
                "No variables created yet. Use ```repl``` blocks to create variables."
            )
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """
        Query the LM via socket connection to the handler.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            # Track this LLM call
            self._pending_llm_calls.append(
                response.chat_completion,
            )

            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        """
        Query the LM with multiple prompts concurrently.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input prompts.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        try:
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model, depth=self.depth
            )

            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    # Track this LLM call
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add.
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(
                f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()"
            )
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call.
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        target_dir = self.work_dir or self.temp_dir
        try:
            os.chdir(target_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def execute_code(self, code: str) -> REPLResult:
        """
        Execute code in the persistent namespace and return result.

        Args:
            code: The Python code to execute.

        Returns:
            A REPLResult object containing the execution output and state.
        """
        start_time = time.perf_counter()
        if self._logger:
            payload = {
                "kind": "local_repl_execute",
                "code_size": len(code.encode("utf-8", errors="replace")),
            }
            if len(code) > 16384:
                payload["code_ref"] = self._logger.write_payload(
                    uuid.uuid4().hex, "code", code
                )
            else:
                payload["code"] = code
            log_trace_event(
                self._logger, "code_execution_start", self._session_id, payload
            )

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        result = REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            llm_calls=self._pending_llm_calls.copy(),
        )

        if self._logger:
            payload = {
                "kind": "local_repl_execute",
                "stdout": stdout
                if len(stdout) <= 16384
                else stdout[:16384] + "...<truncated>",
                "stderr": stderr
                if len(stderr) <= 16384
                else stderr[:16384] + "...<truncated>",
                "execution_time": result.execution_time,
            }
            if len(stdout) > 16384:
                payload["stdout_ref"] = self._logger.write_payload(
                    uuid.uuid4().hex, "stdout", stdout
                )
            if len(stderr) > 16384:
                payload["stderr_ref"] = self._logger.write_payload(
                    uuid.uuid4().hex, "stderr", stderr
                )
            log_trace_event(
                self._logger, "code_execution_end", self._session_id, payload
            )

        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        self.globals.clear()
        self.locals.clear()

    def __del__(self):
        self.cleanup()
