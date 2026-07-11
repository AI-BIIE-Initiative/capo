"""
Base classes for REPL-like environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from capo.mcp.core.types import REPLResult


class BaseEnv(ABC):
    """
    Base REPL-like environment that the system uses to interact with.
    The primary types are isolated and non-isolated, where isolated
    environments are on a separate machine.
    """

    def __init__(self, persistent: bool = False, depth: int = 1, **kwargs):
        self.persistent = persistent
        self.depth = depth
        self.kwargs = kwargs
        self._logger = None
        self._session_id = None

    def set_logger(self, logger, session_id: str | None = None) -> None:
        self._logger = logger
        if session_id is not None:
            self._session_id = session_id

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class IsolatedEnv(BaseEnv, ABC):
    """
    These environments sit on a completely separate machine,
    guaranteeing complete isolation from the main process.
    """

    def __init__(self, persistent: bool = False, **kwargs):
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class NonIsolatedEnv(BaseEnv, ABC):
    """
    These environments run on the same machine as the main process,
    and provide different levels of isolation depending on the choice
    of environment. The simplest is a local Python REPL that runs
    as a subprocess.
    """

    def __init__(self, persistent: bool = False, **kwargs):
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


@runtime_checkable
class SupportsPersistence(Protocol):
    """Protocol for environments that support persistent multi-turn sessions.

    CHECKING SUPPORT:
        Use isinstance(env, SupportsPersistence) to check if an environment
        supports persistence capabilities.

    IMPLEMENTING THIS PROTOCOL:
        To add persistence to your environment, implement these 5 methods.

    VERSIONING BEHAVIOR:
        Contexts and histories are versioned with numeric suffixes:
        - First context  -> context_0, context_1, context_2, ...
        - First history  -> history_0, history_1, history_2, ...

    ALIASING BEHAVIOR:
        The unversioned names always point to index 0:
        - context  -> context_0 (first context)
        - history  -> history_0 (first history)

    EXAMPLE IMPLEMENTATION:
        See environments/local_repl_env.py for a complete reference.
    """

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for nested LLM calls.

        Called when the handler address changes between completions.
        Store the address so llm_query() calls from executed code can reach
        the LM handler.

        Args:
            address: (host, port) tuple for the LM handler server.
        """
        ...

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """Add a context payload, making it available as context_N in code.

        Versioning:
            - context_index=None: auto-increment (0, 1, 2, ...)
            - context_index=N: use specific index N

        Storage:
            Must store so executed code can access:
            - context_0, context_1, etc. (versioned)
            - context (alias to context_0)

        Args:
            context_payload: The context data (string, dict, or list).
            context_index: Optional specific index, or None to auto-increment.

        Returns:
            The index used (for auto-increment, returns the assigned index).
        """
        ...

    def get_context_count(self) -> int:
        """Return the number of contexts added so far."""
        ...

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """Add a message history, making it available as history_N in code.

        Versioning:
            - history_index=None: auto-increment (0, 1, 2, ...)
            - history_index=N: use specific index N

        Storage:
            Must store so executed code can access:
            - history_0, history_1, etc. (versioned)
            - history (alias to history_0)

        IMPORTANT: Store a deep copy, not a reference. The caller may
        modify the list after calling this method.

        Args:
            message_history: List of message dicts (role, content).
            history_index: Optional specific index, or None to auto-increment.

        Returns:
            The index used.
        """
        ...

    def get_history_count(self) -> int:
        """Return the number of histories added so far."""
        ...
