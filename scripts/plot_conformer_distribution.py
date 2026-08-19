"""Plan A — population-level conformer-weight distribution per case_type.

Replaces the cherry-picked PowerNorm heatmap in figures/interpretability_conformer_weights.pdf.

Reads:  outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv
Writes: outputs/interpretability/biosnap_seed42/figs/plan_A_conformer_distribution.{png,svg}
        outputs/interpretability/biosnap_seed42/figs/plan_A_stats.json
        TAMR-DTI__.../figures/interpretability_conformer_weights.pdf   (paper figure)

For each case_type (TP/TN/FP/FN):
  - Sort the 8 conformer weights per row in descending order → ranks 1..8.
  - Plot median + IQR (25–75%) + 5–95% band across all rows in that group.
  - Dashed reference at 1/K = 0.125 (uniform).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse project palette
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_theme import GROUP_COLORS, apply_paper_theme, style_axes  # noqa: E402

apply_paper_theme(plt)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv"
FIG_DIR = ROOT / "outputs/interpretability/biosnap_seed42/figs"
PAPER_PDF = (
    ROOT
    / "TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction"
    / "figures"
    / "interpretability_conformer_weights.pdf"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 8
CONF_COLS = [f"conf_{i+1}" for i in range(K)]
GROUPS = ["TP", "TN", "FP", "FN"]


def main() -> None:
    df = pd.read_csv(CSV)
    assert all(c in df.columns for c in CONF_COLS), f"missing cols in {CSV}"
    print(f"[data] loaded {len(df)} rows from {CSV}")
    counts = df["case_type"].value_counts().to_dict()
    print(f"[data] case_type counts: {counts}")

    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.45), sharey=True)
    ranks = np.arange(1, K + 1)

    stats: dict = {"K": K, "n_total": int(len(df)), "groups": {}}

    for ax, g in zip(axes, GROUPS):
        sub_df = df[df["case_type"] == g]
        sub = sub_df[CONF_COLS].to_numpy()
        if sub.shape[0] == 0:
            ax.set_title(f"{g} (n=0)", fontsize=10, color=GROUP_COLORS[g])
            continue

        sorted_desc = -np.sort(-sub, axis=1)  # (n, K) descending per row
        med = np.median(sorted_desc, axis=0)
        q25, q75 = np.percentile(sorted_desc, [25, 75], axis=0)
        q05, q95 = np.percentile(sorted_desc, [5, 95], axis=0)
        top1 = sorted_desc[:, 0]
        # Use the entropy as stored in the CSV (normalized by log of #effective
        # conformers, matching the reported paper statistic).
        ent = sub_df["conformer_entropy_norm"].to_numpy()

        c = GROUP_COLORS[g]
        ax.fill_between(ranks, q05, q95, color=c, alpha=0.18, linewidth=0,
                        label="5–95%")
        ax.fill_between(ranks, q25, q75, color=c, alpha=0.35, linewidth=0,
                        label="IQR (25–75%)")
        ax.plot(ranks, med, color=c, lw=1.8, marker="o", ms=4.0, label="median")
        ax.axhline(1 / K, color="#94A3B8", lw=0.8, ls="--", label=f"1/K={1/K:.3f}")
        ax.set_xticks(ranks)
        ax.set_xlabel("Rank (within sample, descending)", fontsize=9)
        ax.set_title(f"{g}  (n={sub.shape[0]})", fontsize=10, color=c)
        style_axes(ax)

        stats["groups"][g] = {
            "n": int(sub.shape[0]),
            "rank_median": med.round(6).tolist(),
            "rank_iqr_low": q25.round(6).tolist(),
            "rank_iqr_high": q75.round(6).tolist(),
            "rank_q05": q05.round(6).tolist(),
            "rank_q95": q95.round(6).tolist(),
            "top1_mean": float(top1.mean()),
            "top1_std": float(top1.std()),
            "entropy_mean": float(ent.mean()),
            "entropy_std": float(ent.std()),
            "rank1_median": float(med[0]),
            "rank2_median": float(med[1]),
            "rank4_median": float(med[3]),
        }
        # Per-axis annotation: top-1 + entropy
        ax.text(
            0.96, 0.95,
            f"top-1={top1.mean():.3f}\nH/logK={ent.mean():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=c,
            bbox=dict(facecolor="white", edgecolor=c, alpha=0.85,
                      boxstyle="round,pad=0.25", linewidth=0.6),
        )

    axes[0].set_ylabel("Conformer selection weight", fontsize=10)
    # Place the shared legend below all panels so it does not cover the FN curve.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        labelcolor=plt.rcParams.get("axes.labelcolor", "black"),
    )

    fig.suptitle(
        "Population-level conformer selection across BioSNAP test set (n=5493)\n"
        "Sorted weights within each sample, then aggregated by case type",
        fontsize=10.5,
        y=0.97,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.78, bottom=0.24, wspace=0.18)

    # save artifacts
    fig.savefig(FIG_DIR / "plan_A_conformer_distribution.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / "plan_A_conformer_distribution.svg", bbox_inches="tight")
    fig.savefig(PAPER_PDF, bbox_inches="tight")
    plt.close(fig)

    with (FIG_DIR / "plan_A_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[ok] wrote {FIG_DIR/'plan_A_conformer_distribution.png'}")
    print(f"[ok] wrote {FIG_DIR/'plan_A_conformer_distribution.svg'}")
    print(f"[ok] wrote {PAPER_PDF}")
    print(f"[ok] wrote {FIG_DIR/'plan_A_stats.json'}")
    print("\nrank-medians per group:")
    for g in GROUPS:
        if g in stats["groups"]:
            s = stats["groups"][g]
            print(f"  {g} n={s['n']:4d}  r1={s['rank1_median']:.3f}  "
                  f"r2={s['rank2_median']:.3f}  r4={s['rank4_median']:.3f}  "
                  f"top1={s['top1_mean']:.3f}  H/logK={s['entropy_mean']:.3f}")


if __name__ == "__main__":
    main()
