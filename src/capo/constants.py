from capo.viz.palette import (
    PURPLE_0, PURPLE_50, PURPLE_90,
    ORANGE_0, ORANGE_50, ORANGE_90,
    BLUE_0, BLUE_50, BLUE_90,
    GREEN_0, GREEN_50, GREEN_90,
)

BASE_MODEL_COLOURS: dict[str, str] = {
    # Claude agent models — dark palette shades
    "claude-opus-4-6":           PURPLE_0,
    "claude-sonnet-4-6":         BLUE_0,
    "claude-haiku-4-5-20251001": GREEN_0,
    # OpenAI
    "gpt-5.2-codex":             ORANGE_0,
}

# Base seed for reproducible runs
SEED = 42