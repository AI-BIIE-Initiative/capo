from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LanguageModelChatCompletion:
    """Represents a chat completion from the language model."""

    response: str


@dataclass
class REPLResult:
    """Represents the result of executing code in a REPL environment."""

    stdout: str
    stderr: str
    locals: Dict[str, Any]
    execution_time: float
    llm_calls: list[LanguageModelChatCompletion]
