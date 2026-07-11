"""
comparison plots for the CAPO evaluation harness.

Reads the Stage-2 CSVs and produces four figures under <eval_dir>/plots/:
    performance.{pdf,png}       — macro metric bars + per-species AEM dumbbell
    error_taxonomy.{pdf,png}    — error-type rates per system (aggregate)
    efficiency.{pdf,png}        — pipeline data yield + coverage-quality scatter
    statistical_tests.{pdf,png} — bootstrap 95% CI forest plot (when data exist)

All colors come from capo.viz.palette.  CAPO = PURPLE_0, GCA = ORANGE_50, 3rd system = BLUE_0.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from capo.viz.palette import (
    BLUE_0, BLUE_50,
    ORANGE_50, ORANGE_90,
    PURPLE_0, PURPLE_50,
    NOISE, BLACK,
)

# ---------------------------------------------------------------------------
# System identity helpers
# ---------------------------------------------------------------------------

# Ordered color cycle: CAPO=purple, GCA=orange, any third system=blue.
# Unknown systems beyond the first three fall back to NOISE.
_COLOR_CYCLE: list[str] = [PURPLE_0, ORANGE_50, BLUE_0]
_LIGHT_CYCLE: list[str] = [PURPLE_50, ORANGE_90, BLUE_50]

# Explicit per-system overrides (names must match harness system_name exactly).
_COLORS: dict[str, str] = {
    "CAPO":                  PURPLE_0,
    "General Coding Agent":  ORANGE_50,
}
_LIGHT: dict[str, str] = {
    "CAPO":                  PURPLE_50,
    "General Coding Agent":  ORANGE_90,
}
_LABELS: dict[str, str] = {
    "CAPO":                  "CAPO",
    "General Coding Agent":  "Gen. Coding Agent",
}

_METRIC_LABELS: dict[str, str] = {
    "gold_coverage":          "Gold Coverage",
    "annotation_exact_match": "Ann. Exact Match",
    "label_accuracy":         "Label Accuracy",
    "field_f1":               "Binder F1",
}

# Runtime index so unknown system names get a stable cycle position.
_system_order: list[str] = list(_COLORS)


def _ensure_system(s: str) -> None:
    """Register an unknown system so it gets a stable cycle color."""
    if s not in _system_order:
        _system_order.append(s)


def _color(s: str) -> str:
    _ensure_system(s)
    i = _system_order.index(s)
    return _COLORS.get(s, _COLOR_CYCLE[i] if i < len(_COLOR_CYCLE) else NOISE)


def _light(s: str) -> str:
    _ensure_system(s)
    i = _system_order.index(s)
    return _LIGHT.get(s, _LIGHT_CYCLE[i] if i < len(_LIGHT_CYCLE) else NOISE)


def _label(s: str) -> str:
    return _LABELS.get(s, s)


# ---------------------------------------------------------------------------
# Style helpers (must be called after import matplotlib)
# ---------------------------------------------------------------------------

def _set_style() -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.edgecolor": "#888888",
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.titlepad": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)


def _grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, which="major", color="#EEEEEE", linewidth=0.8, zorder=0)


def _macro_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if "species_weighting" in summary.columns:
        return summary[summary["species_weighting"] == "macro"].copy()
    return summary.copy()


# ---------------------------------------------------------------------------
# Figure 1 — performance
# ---------------------------------------------------------------------------

def _make_performance(summary: pd.DataFrame, per_species: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.0, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.36},
    )

    # Panel A — macro-averaged grouped bar chart
    macro = _macro_rows(summary)
    systems = [s for s in _COLORS if s in macro["system_name"].unique()]
    metrics = [m for m in _METRIC_LABELS if m in macro.columns]

    bar_h, gap = 0.32, 0.10
    group_h = len(systems) * bar_h + gap

    for gi, metric in enumerate(metrics):
        for si, sys in enumerate(systems):
            row = macro[macro["system_name"] == sys]
            val = float(row[metric].iloc[0]) if len(row) else 0.0
            y = gi * group_h + si * bar_h
            ax_l.barh(y, val, height=bar_h - 0.02,
                      color=_color(sys), edgecolor="white", linewidth=0.5, zorder=3)
            ax_l.text(val + 0.008, y + bar_h / 2 - 0.01,
                      f"{val:.3f}", va="center", ha="left", fontsize=8.5, color=BLACK, zorder=4)

    tick_pos = [gi * group_h + (len(systems) * bar_h) / 2 - bar_h / 2 for gi in range(len(metrics))]
    ax_l.set_yticks(tick_pos)
    ax_l.set_yticklabels([_METRIC_LABELS[m] for m in metrics])
    ax_l.set_xlim(0, 1.22)
    ax_l.set_xlabel("Score (macro-averaged across species)")
    ax_l.set_title("Preprocessing quality — macro averages")
    ax_l.xaxis.set_major_locator(MultipleLocator(0.2))
    _grid(ax_l, "x")
    ax_l.legend(
        handles=[mpatches.Patch(facecolor=_color(s), edgecolor="white", label=_label(s)) for s in systems],
        loc="lower right", frameon=False, fontsize=9, handlelength=1.1, handleheight=0.85,
    )
    sns.despine(ax=ax_l, top=True, right=True)

    # Panel B — per-species annotation exact match dumbbell
    if "species" in per_species.columns and "annotation_exact_match" in per_species.columns:
        ps_systems = [s for s in _COLORS if s in per_species["system_name"].unique()]
        pivot = per_species.pivot_table(
            index="species", columns="system_name", values="annotation_exact_match",
        ).reset_index()

        sort_col = next((s for s in ps_systems if s in pivot.columns), None)
        if sort_col:
            pivot = pivot.sort_values(sort_col, ascending=True).reset_index(drop=True)
        n = len(pivot)
        y_pos = list(range(n))

        if len(ps_systems) == 2 and all(s in pivot.columns for s in ps_systems):
            for i, row in pivot.iterrows():
                ax_r.plot(
                    [row[ps_systems[0]], row[ps_systems[1]]], [i, i],
                    color=NOISE, linewidth=1.0, alpha=0.55, zorder=1,
                )

        for si, sys in enumerate(ps_systems):
            if sys not in pivot.columns:
                continue
            ax_r.scatter(
                pivot[sys], y_pos,
                s=85 - si * 20,
                facecolor=_color(sys) if si == 0 else "white",
                edgecolor=_color(sys),
                linewidth=1.3, zorder=3 + si,
                label=_label(sys),
            )

        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(pivot["species"].str.replace("_", " ").str.title())
        ax_r.set_xlabel("Annotation exact match")
        ax_r.set_title("Per-species annotation exact match")
        ax_r.set_xlim(-0.05, 1.12)
        ax_r.xaxis.set_major_locator(MultipleLocator(0.2))
        _grid(ax_r, "x")
        ax_r.legend(loc="lower right", frameon=False, fontsize=9, handletextpad=0.5)

    sns.despine(ax=ax_r, top=True, right=True)
    _save(fig, out_dir, "performance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — error taxonomy
# ---------------------------------------------------------------------------

def _make_error_taxonomy(error_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator

    agg = (
        error_df[error_df["species"] == "all"].copy()
        if "species" in error_df.columns
        else pd.DataFrame()
    )
    if agg.empty and not error_df.empty:
        agg = (
            error_df.groupby(["system_name", "error_type"], as_index=False)
            .agg(error_count=("error_count", "sum"), total_examples=("total_examples", "max"))
        )
        agg["error_rate"] = agg["error_count"] / agg["total_examples"].clip(lower=1)

    if agg.empty:
        return

    systems = [s for s in _COLORS if s in agg["system_name"].unique()]
    nonzero = agg.groupby("error_type")["error_rate"].max().loc[lambda s: s > 0].index.tolist()
    if not nonzero:
        return

    agg = agg[agg["error_type"].isin(nonzero)]
    et_max = {et: agg[agg["error_type"] == et]["error_rate"].max() for et in nonzero}
    error_types = sorted(nonzero, key=lambda t: et_max[t], reverse=True)

    bar_w, gap = 0.30, 0.10
    group_w = len(systems) * bar_w + gap

    fig, ax = plt.subplots(figsize=(max(8.0, len(error_types) * 1.4), 4.6))

    for gi, et in enumerate(error_types):
        for si, sys in enumerate(systems):
            row = agg[(agg["system_name"] == sys) & (agg["error_type"] == et)]
            val = float(row["error_rate"].iloc[0]) if len(row) else 0.0
            x = gi * group_w + si * bar_w
            ax.bar(x, val, width=bar_w - 0.02,
                   color=_color(sys), edgecolor="white", linewidth=0.5, zorder=3)

    xtick_pos = [gi * group_w + (len(systems) * bar_w) / 2 - bar_w / 2 for gi in range(len(error_types))]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([et.replace("_", "\n") for et in error_types], fontsize=8.5, ha="center")
    ax.set_ylabel("Error rate")
    ax.set_title("Error taxonomy — aggregate across species")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    _grid(ax, "y")
    ax.legend(
        handles=[mpatches.Patch(facecolor=_color(s), edgecolor="white", label=_label(s)) for s in systems],
        loc="upper right", frameon=False, fontsize=9,
    )
    sns.despine(ax=ax, top=True, right=True)
    _save(fig, out_dir, "error_taxonomy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — efficiency
# ---------------------------------------------------------------------------

def _make_efficiency(
    efficiency: pd.DataFrame,
    summary: pd.DataFrame,
    per_species: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11.0, 4.4),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.38},
    )

    systems = [s for s in _COLORS if s in efficiency["system_name"].unique()]

    # Panel A — data yield funnel (normalized to raw reads per system)
    stage_cols = [
        ("raw_reads_processed",         "Raw reads"),
        ("reads_retained_after_qc",     "Post-QC"),
        ("unique_model_ready_sequences", "Unique seqs"),
    ]
    avail = [(c, l) for c, l in stage_cols if c in efficiency.columns]

    raw_vals: dict[str, float] = {}
    for sys in systems:
        row = efficiency[efficiency["system_name"] == sys]
        raw = float(row["raw_reads_processed"].iloc[0]) if len(row) and "raw_reads_processed" in row.columns else 1.0
        raw_vals[sys] = raw if raw > 0 else 1.0

    bar_w, gap = 0.28, 0.10
    group_w = len(systems) * bar_w + gap

    for gi, (col, label) in enumerate(avail):
        for si, sys in enumerate(systems):
            row = efficiency[efficiency["system_name"] == sys]
            val = float(row[col].iloc[0]) if len(row) else 0.0
            frac = val / raw_vals[sys]
            x = gi * group_w + si * bar_w
            alpha = max(0.40, 1.0 - gi * 0.22)
            ax_l.bar(x, frac, width=bar_w - 0.02,
                     color=_color(sys), edgecolor="white", linewidth=0.5, alpha=alpha, zorder=3)
            ax_l.text(x + (bar_w - 0.02) / 2, frac + 0.012, f"{frac:.0%}",
                      ha="center", va="bottom", fontsize=8, color=BLACK)

    xtick_pos = [gi * group_w + (len(systems) * bar_w) / 2 - bar_w / 2 for gi in range(len(avail))]
    ax_l.set_xticks(xtick_pos)
    ax_l.set_xticklabels([l for _, l in avail], fontsize=9)
    ax_l.set_ylabel("Fraction of raw reads")
    ax_l.set_ylim(0, 1.26)
    ax_l.set_title("Pipeline data yield")
    ax_l.yaxis.set_major_locator(MultipleLocator(0.2))
    _grid(ax_l, "y")
    ax_l.legend(
        handles=[mpatches.Patch(facecolor=_color(s), edgecolor="white", label=_label(s)) for s in systems],
        loc="upper right", frameon=False, fontsize=9,
    )
    sns.despine(ax=ax_l, top=True, right=True)

    # Panel B — coverage vs annotation exact match scatter
    macro = _macro_rows(summary)
    macro_systems = [s for s in _COLORS if s in macro["system_name"].unique()]

    for sys in macro_systems:
        if "species" in per_species.columns:
            sp = per_species[per_species["system_name"] == sys]
            need = {"gold_coverage", "annotation_exact_match"}
            if not sp.empty and need.issubset(sp.columns):
                ax_r.scatter(
                    sp["gold_coverage"], sp["annotation_exact_match"],
                    s=28, color=_light(sys), alpha=0.55, zorder=2, edgecolors="none",
                )

        row = macro[macro["system_name"] == sys]
        if row.empty:
            continue
        need = {"gold_coverage", "annotation_exact_match"}
        if not need.issubset(row.columns):
            continue
        cx = float(row["gold_coverage"].iloc[0])
        cy = float(row["annotation_exact_match"].iloc[0])
        cost = (
            float(row["estimated_cost_usd"].iloc[0])
            if "estimated_cost_usd" in row.columns and pd.notna(row["estimated_cost_usd"].iloc[0])
            else None
        )

        ax_r.scatter(cx, cy, s=170, color=_color(sys), edgecolors=BLACK, linewidth=0.6, zorder=4)
        annot = _label(sys)
        if cost is not None:
            annot += f"\n${cost:.2f}"
        ax_r.annotate(annot, xy=(cx, cy), xytext=(cx + 0.02, cy + 0.03),
                      fontsize=8.5, color=_color(sys), fontweight="semibold")

    ax_r.set_xlabel("Gold coverage (fraction of gold matched)")
    ax_r.set_ylabel("Annotation exact match")
    ax_r.set_xlim(-0.04, 1.15)
    ax_r.set_ylim(-0.04, 1.15)
    ax_r.set_title("Coverage vs. annotation quality")
    ax_r.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_r.yaxis.set_major_locator(MultipleLocator(0.2))
    _grid(ax_r, "both")
    sns.despine(ax=ax_r, top=True, right=True)

    _save(fig, out_dir, "efficiency")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — statistical tests (bootstrap CI forest plot)
# ---------------------------------------------------------------------------

def _make_statistical_tests(tests: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    needed = {"metric", "absolute_difference", "ci_lower_95", "ci_upper_95"}
    if tests.empty or not needed.issubset(tests.columns):
        return

    rows = tests.dropna(subset=["absolute_difference", "ci_lower_95", "ci_upper_95"])
    if rows.empty:
        return

    metrics = rows["metric"].unique().tolist()
    n = len(metrics)
    fig, ax = plt.subplots(figsize=(7.5, 1.5 + n * 0.95))

    for i, metric in enumerate(metrics):
        r = rows[rows["metric"] == metric].iloc[0]
        diff = float(r["absolute_difference"])
        ci_lo, ci_hi = float(r["ci_lower_95"]), float(r["ci_upper_95"])
        p = float(r["p_value"]) if "p_value" in r and pd.notna(r.get("p_value")) else None

        color = BLUE_0 if diff >= 0 else ORANGE_50
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=2.4, zorder=2, solid_capstyle="round")
        ax.scatter([diff], [i], s=95, color=color, zorder=3, edgecolors=BLACK, linewidth=0.5)

        sig = ""
        if p is not None:
            if p < 0.001:   sig = " ***"
            elif p < 0.01:  sig = " **"
            elif p < 0.05:  sig = " *"
        label_txt = f"{metric.replace('_', ' ').title()}  {diff:+.3f}{sig}"
        ax.text(ci_hi + 0.008, i, label_txt, va="center", ha="left", fontsize=9, color=BLACK)

    ax.axvline(0, color=NOISE, linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks([])
    ax.set_xlabel("CAPO − General Coding Agent (absolute difference)")
    ax.set_title("Bootstrap 95% CI — paired metric differences (CAPO vs GCA)")
    x_pad = 0.08
    ax.set_xlim(rows["ci_lower_95"].min() - x_pad, rows["ci_upper_95"].max() + 0.28)
    _grid(ax, "x")
    sns.despine(ax=ax, top=True, right=True, left=True)
    _save(fig, out_dir, "statistical_tests")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_harness_plots(eval_dir: Path) -> Path | None:
    """Generate all comparison plots from Stage-2 CSVs in eval_dir.

    Writes PDFs and PNGs under eval_dir/plots/.  Returns the plots directory
    on success, None if matplotlib is unavailable or all source CSVs are absent.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        warnings.warn("matplotlib not installed — skipping harness plots", stacklevel=2)
        return None

    plots_dir = eval_dir / "plots"

    def _read(name: str) -> pd.DataFrame:
        p = eval_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    _set_style()

    summary     = _read("summary_metrics.csv")
    per_species = _read("per_species_metrics.csv")
    error_df    = _read("error_analysis.csv")
    efficiency  = _read("efficiency_metrics.csv")
    tests       = _read("statistical_tests.csv")

    built: list[str] = []

    if not summary.empty and not per_species.empty:
        _make_performance(summary, per_species, plots_dir)
        built.append("performance")

    if not error_df.empty:
        _make_error_taxonomy(error_df, plots_dir)
        built.append("error_taxonomy")

    if not efficiency.empty and not summary.empty:
        _make_efficiency(efficiency, summary, per_species, plots_dir)
        built.append("efficiency")

    if not tests.empty:
        _make_statistical_tests(tests, plots_dir)
        built.append("statistical_tests")

    if not built:
        return None

    return plots_dir
