"""
Canonical color palette for all autoimmunolab plots.

Every plot produced by the system — profiling, evaluation, clustering,
dimensionality reduction — must use only these colors.
Text, spines, and tick labels stay black (#000000).
"""
from matplotlib.colors import LinearSegmentedColormap

# --- Core palette ---
PURPLE_0  = "#713D8F"
PURPLE_50 = "#C694E1"
PURPLE_90 = "#EAD5F6"

ORANGE_0  = "#9B3208"
ORANGE_50 = "#E6905B"
ORANGE_90 = "#FAC19E"

BLUE_0    = "#1E5994"
BLUE_50   = "#8DB8E2"
BLUE_90   = "#BDD9F5"

GREEN_0   = "#0E625C"
GREEN_50  = "#78B5B0"
GREEN_90  = "#C8DFD9"

BLACK     = "#000000"
WHITE     = "#FFFFFF"

# Neutral grey for noise, missing, or background points
NOISE     = "#AAAAAA"

# --- Categorical sequence ---
# Ordered for maximum contrast when cycling through up to 12 categories.
# Cycle: dark blues → dark oranges → dark purples → dark greens → medium shades
CATEGORICAL: list[str] = [
    BLUE_0, ORANGE_0, PURPLE_0, GREEN_0,
    BLUE_50, ORANGE_50, PURPLE_50, GREEN_50,
    BLUE_90, ORANGE_90, PURPLE_90, GREEN_90,
]

# --- Colormaps ---
# Sequential: low-intensity white/light → high-intensity dark green (entropy, counts)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "capo_seq", [GREEN_90, GREEN_50, GREEN_0]
)

# Diverging: blue ↔ orange centered on white (correlations, signed scores)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "capo_div", [BLUE_0, WHITE, ORANGE_0]
)

# Single-hue blue: light → dark (confusion matrices, probability maps)
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "capo_blue", [BLUE_90, BLUE_50, BLUE_0]
)
