#!/usr/bin/env python3
"""Plot BioSNAP interpretability outputs extracted from TAMR-DTI."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

from plot_theme import (
    GROUP_COLORS,
    SEQUENTIAL_CMAP,
    TAMR,
    apply_paper_theme,
    style_axes as theme_style_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    REPO_ROOT
    / "TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction"
)
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs/interpretability/biosnap_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--paper_fig_dir", default=str(PAPER_DIR / "figures"))
    parser.add_argument("--heatmap_cases", type=int, default=18)
    parser.add_argument("--case_count", type=int, default=8)
    return parser.parse_args()


def select_cases(df: pd.DataFrame, total: int) -> pd.DataFrame:
    parts = []
    quotas = [("TP", max(4, total // 2)), ("TN", max(3, total // 4)), ("FP", 2), ("FN", 2)]
    for case_type, quota in quotas:
        subset = df[df["case_type"] == case_type].copy()
        if subset.empty:
            continue
        subset["confidence"] = np.where(subset["label"] == 1, subset["pred"], 1.0 - subset["pred"])
        parts.append(subset.sort_values("confidence", ascending=False).head(quota))

    selected = pd.concat(parts, ignore_index=True) if parts else df.head(total)
    return selected.head(total).reset_index(drop=True)


def style_axes(ax) -> None:
    theme_style_axes(ax, grid_axis="y", labelsize=8)


def save_all(fig, prefixes: list[Path]) -> None:
    for prefix in prefixes:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_conformer_heatmap(df: pd.DataFrame, output_prefixes: list[Path], heatmap_cases: int) -> pd.DataFrame:
    selected = select_cases(df, heatmap_cases)
    conf_cols = [f"conf_{idx}" for idx in range(1, 9)]
    values = selected[conf_cols].to_numpy(dtype=float)
    labels = [
        f"{row.case_type}{i + 1} ({row.pred:.2f})"
        for i, row in enumerate(selected.itertuples(index=False))
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        values,
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=max(0.5, float(values.max()))),
    )
    ax.set_xlabel("Conformer index", fontsize=10)
    ax.set_ylabel("Selected test cases", fontsize=10)
    ax.set_xticks(np.arange(len(conf_cols)))
    ax.set_xticklabels([str(idx) for idx in range(1, 9)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Selection weight", fontsize=9)
    save_all(fig, output_prefixes)
    return selected


def plot_module_statistics(df: pd.DataFrame, output_prefixes: list[Path]) -> None:
    groups = [group for group in ("TP", "TN", "FP", "FN") if (df["case_type"] == group).any()]
    colors = GROUP_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)

    entropy_data = [
        df[df["case_type"] == group]["conformer_entropy_norm"].clip(upper=1.0).to_numpy()
        for group in groups
    ]
    bp = axes[0].boxplot(entropy_data, labels=groups, patch_artist=True, showfliers=False)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.72)
    axes[0].set_ylabel("Normalized entropy", fontsize=9)
    axes[0].set_title("Conformer selection", fontsize=10)
    axes[0].set_ylim(0.84, 1.01)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    style_axes(axes[0])

    mamba_data = [df[df["case_type"] == group]["mamba_refine_mean"].to_numpy() for group in groups]
    bp = axes[1].boxplot(mamba_data, labels=groups, patch_artist=True, showfliers=False)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.72)
    gate = df["mamba_gate"].dropna().mean()
    axes[1].set_ylabel("Mean gated refinement", fontsize=9)
    axes[1].set_title(f"Protein Mamba (gate={gate:.3f})", fontsize=10)
    style_axes(axes[1])

    film_cols = [col for col in df.columns if col.startswith("film_l") and col.endswith("_strength")]
    layer_ids = [col.split("_")[1].replace("l", "L") for col in film_cols]
    film_means = df[film_cols].mean().to_numpy(dtype=float) if film_cols else np.array([])
    film_stds = df[film_cols].std().fillna(0).to_numpy(dtype=float) if film_cols else np.array([])
    axes[2].bar(np.arange(len(film_cols)), film_means, yerr=film_stds, color=TAMR, alpha=0.84, capsize=3)
    axes[2].set_xticks(np.arange(len(film_cols)))
    axes[2].set_xticklabels(layer_ids)
    axes[2].set_ylabel(r"$||\gamma||_2 + ||\beta||_2$", fontsize=9)
    axes[2].set_title("FiLM modulation", fontsize=10)
    style_axes(axes[2])

    save_all(fig, output_prefixes)


def plot_mamba_heatmap(
    df: pd.DataFrame,
    profiles: np.ndarray,
    output_prefixes: list[Path],
    case_count: int,
) -> pd.DataFrame:
    selected = select_cases(df, case_count)
    row_indices = selected["row_index"].to_numpy(dtype=int)
    values = profiles[row_indices]
    row_max = np.maximum(values.max(axis=1, keepdims=True), 1e-8)
    values = values / row_max
    labels = [
        f"{row.case_type}{i + 1} ({row.pred:.2f})"
        for i, row in enumerate(selected.itertuples(index=False))
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    im = ax.imshow(
        values,
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        norm=PowerNorm(gamma=1.6, vmin=0.0, vmax=1.0),
    )
    ax.set_xlabel("Protein token position", fontsize=10)
    ax.set_ylabel("Selected test cases", fontsize=10)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 31, 63, 95, 127])
    ax.set_xticklabels(["1", "32", "64", "96", "128"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Normalized refinement", fontsize=9)
    save_all(fig, output_prefixes)
    return selected


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    paper_fig_dir = Path(args.paper_fig_dir)
    df = pd.read_csv(input_dir / "biosnap_seed42_interpretability_samples.csv")
    profiles = np.load(input_dir / "biosnap_seed42_mamba_profiles.npy")

    output_prefixes = [
        input_dir / "interpretability_conformer_weights",
        paper_fig_dir / "interpretability_conformer_weights",
    ]
    conformer_cases = plot_conformer_heatmap(df, output_prefixes, args.heatmap_cases)
    conformer_cases.to_csv(input_dir / "biosnap_seed42_conformer_cases.csv", index=False)

    output_prefixes = [
        input_dir / "interpretability_module_statistics",
        paper_fig_dir / "interpretability_module_statistics",
    ]
    plot_module_statistics(df, output_prefixes)

    output_prefixes = [
        input_dir / "interpretability_mamba_refinement",
        paper_fig_dir / "interpretability_mamba_refinement",
    ]
    mamba_cases = plot_mamba_heatmap(df, profiles, output_prefixes, args.case_count)
    mamba_cases.to_csv(input_dir / "biosnap_seed42_mamba_cases.csv", index=False)

    print("Saved interpretability figures:")
    print(f"- {input_dir}")
    print(f"- {paper_fig_dir}")


if __name__ == "__main__":
    apply_paper_theme(plt)
    main()
