"""Shared plotting theme for TAMR-DTI manuscript figures."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


TEXT = "#1F2937"
MUTED_TEXT = "#64748B"
GRID = "#E5E7EB"
SPINE = "#94A3B8"
BASELINE = "#A7B0BA"
BASELINE_DARK = "#374151"
TAMR = "#005F99"

METRIC_COLORS = {
    "AUROC": "#005F99",
    "AUPRC": "#00876C",
    "Acc@0.5": "#D97706",
    "F1@0.5": "#B91C1C",
}

GROUP_COLORS = {
    "TP": "#005F99",
    "TN": "#00876C",
    "FP": "#D97706",
    "FN": "#B91C1C",
}

ABLATION_COLORS = ["#005F99", "#00876C", "#D97706", "#7C3AED", "#B91C1C"]

DROP_CMAP = LinearSegmentedColormap.from_list(
    "tamr_drop",
    ["#FFF7ED", "#FED7AA", "#FB923C", "#EA580C", "#7C2D12"],
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "tamr_sequential",
    ["#F8FAFC", "#DBEAFE", "#60A5FA", "#0284C7", "#083344"],
)


def apply_paper_theme(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": TEXT,
            "axes.edgecolor": SPINE,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax, *, grid_axis: str = "y", labelsize: float = 8.0) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=labelsize, colors=TEXT)
