from dataclasses import dataclass
from typing import Optional
from capo.mcp.core.types import LanguageModelChatCompletion


@dataclass
class LMRequest:
    """Represents a request to the language model."""

    prompt: str
    model: Optional[str]
    depth: int


@dataclass
class LMResponse:
    """Represents a response from the language model."""

    success: bool
    chat_completion: "LanguageModelChatCompletion"
    error: Optional[str] = None


def send_lm_request(address: tuple[str, int], request: LMRequest) -> LMResponse:
    """Sends a single language model request."""
    # This is a placeholder. In a real implementation, this would
    # make a network request to the language model handler.
    from capo.mcp.core.types import LanguageModelChatCompletion

    print(f"Sending LM request to {address}: {request.prompt}")
    return LMResponse(
        success=True,
        chat_completion=LanguageModelChatCompletion(
            response="This is a dummy response."
        ),
    )


def send_lm_request_batched(
    address: tuple[str, int], prompts: list[str], model: Optional[str], depth: int
) -> list[LMResponse]:
    """Sends a batch of language model requests."""
    # This is a placeholder. In a real implementation, this would
    # make a network request to the language model handler.
    from capo.mcp.core.types import LanguageModelChatCompletion

    print(f"Sending batched LM requests to {address}: {prompts}")
    return [
        LMResponse(
            success=True,
            chat_completion=LanguageModelChatCompletion(response=f"Dummy response for: {p}"),
        )
        for p in prompts
    ]
