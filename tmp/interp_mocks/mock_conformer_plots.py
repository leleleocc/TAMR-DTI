"""Mock three proposed conformer-weight figures using synthetic data
calibrated to the BioSNAP statistics reported in the paper:
  mean top-1 = 0.241, mean normalized entropy = 0.931
  FP: top-1 = 0.267, entropy = 0.898
  FN: top-1 = 0.284, entropy = 0.879
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# Reuse project palette
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_theme import (  # noqa: E402
    GROUP_COLORS,
    SEQUENTIAL_CMAP,
    TAMR,
    apply_paper_theme,
    style_axes,
)

apply_paper_theme(plt)

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)

K = 8
N_TOTAL = 5493
# Group sizes (roughly realistic; just for mock visual)
N_TP, N_TN, N_FP, N_FN = 2200, 2700, 320, 273
assert N_TP + N_TN + N_FP + N_FN == N_TOTAL


def dirichlet_weights(n: int, alpha: float) -> np.ndarray:
    """Symmetric Dirichlet(alpha) over K slots → (n, K) row-stochastic."""
    return RNG.dirichlet(alpha=np.full(K, alpha), size=n)


def normalized_entropy(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 1e-12, 1.0)
    H = -(w * np.log(w)).sum(axis=1)
    return H / np.log(K)


# Tune alpha per group to roughly hit the paper's stats
# Smaller alpha → more concentrated (lower entropy, higher top-1)
W_TP = dirichlet_weights(N_TP, alpha=2.65)
W_TN = dirichlet_weights(N_TN, alpha=2.75)
W_FP = dirichlet_weights(N_FP, alpha=1.85)
W_FN = dirichlet_weights(N_FN, alpha=1.55)

groups = ["TP"] * N_TP + ["TN"] * N_TN + ["FP"] * N_FP + ["FN"] * N_FN
weights = np.vstack([W_TP, W_TN, W_FP, W_FN])
df = pd.DataFrame(weights, columns=[f"conf_{i+1}" for i in range(K)])
df["case_type"] = groups
# Synthetic prediction prob (only for case labels; not used in shape)
df["pred"] = np.concatenate([
    RNG.uniform(0.6, 0.99, N_TP),
    RNG.uniform(0.01, 0.4, N_TN),
    RNG.uniform(0.6, 0.99, N_FP),
    RNG.uniform(0.01, 0.4, N_FN),
])
df["label"] = [1] * N_TP + [0] * N_TN + [0] * N_FP + [1] * N_FN

top1 = np.sort(weights, axis=1)[:, -1]
ent = normalized_entropy(weights)
print(f"[mock stats] overall top1={top1.mean():.3f}  entropy={ent.mean():.3f}")
for g, W in [("TP", W_TP), ("TN", W_TN), ("FP", W_FP), ("FN", W_FN)]:
    t = np.sort(W, axis=1)[:, -1].mean()
    h = normalized_entropy(W).mean()
    print(f"  {g}: top1={t:.3f}  entropy={h:.3f}")


# ---------------------------------------------------------------------------
# Reference: replicate the CURRENT (cherry-pick + PowerNorm) figure for compare
# ---------------------------------------------------------------------------
def current_figure():
    total = 18
    quotas = [("TP", max(4, total // 2)), ("TN", max(3, total // 4)), ("FP", 2), ("FN", 2)]
    parts = []
    for g, q in quotas:
        sub = df[df["case_type"] == g].copy()
        sub["confidence"] = np.where(sub["label"] == 1, sub["pred"], 1.0 - sub["pred"])
        parts.append(sub.sort_values("confidence", ascending=False).head(q))
    selected = pd.concat(parts, ignore_index=True).head(total)
    values = selected[[f"conf_{i+1}" for i in range(K)]].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        values,
        aspect="auto",
        cmap=SEQUENTIAL_CMAP,
        norm=PowerNorm(gamma=0.85, vmin=0.0, vmax=max(0.5, float(values.max()))),
    )
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([str(i + 1) for i in range(K)])
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([
        f"{r.case_type}{i+1} ({r.pred:.2f})"
        for i, r in enumerate(selected.itertuples(index=False))
    ], fontsize=8)
    ax.set_xlabel("Conformer index", fontsize=10)
    ax.set_ylabel("Selected test cases", fontsize=10)
    ax.set_title("CURRENT — 18 most-confident cases + PowerNorm(γ=0.85)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Selection weight", fontsize=9)
    fig.savefig(OUT / "0_current.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plan A — sorted-weight percentile bands per group
# ---------------------------------------------------------------------------
def plan_A():
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1),
                             sharey=True, constrained_layout=True)
    ranks = np.arange(1, K + 1)
    for ax, g in zip(axes, ["TP", "TN", "FP", "FN"]):
        sub = df[df["case_type"] == g][[f"conf_{i+1}" for i in range(K)]].to_numpy()
        sorted_desc = -np.sort(-sub, axis=1)
        med = np.median(sorted_desc, axis=0)
        q25, q75 = np.percentile(sorted_desc, [25, 75], axis=0)
        q05, q95 = np.percentile(sorted_desc, [5, 95], axis=0)
        c = GROUP_COLORS[g]
        ax.fill_between(ranks, q05, q95, color=c, alpha=0.18, linewidth=0)
        ax.fill_between(ranks, q25, q75, color=c, alpha=0.35, linewidth=0)
        ax.plot(ranks, med, color=c, lw=1.8, marker="o", ms=4)
        ax.axhline(1 / K, color="#94A3B8", lw=0.8, ls="--")
        ax.set_xlabel("Rank (sorted within sample)", fontsize=9)
        ax.set_title(f"{g}  (n={ (df['case_type']==g).sum() })", fontsize=10, color=c)
        ax.set_xticks(ranks)
        style_axes(ax)
    axes[0].set_ylabel("Conformer weight", fontsize=10)
    fig.suptitle("Plan A — Sorted-weight distribution across all test samples\n"
                 "(median • IQR • 5–95% band; dashed = uniform 1/K)",
                 fontsize=10)
    fig.savefig(OUT / "1_plan_A.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plan B — heatmap, BUT stratified sampling + LINEAR norm
# ---------------------------------------------------------------------------
def plan_B():
    n_per_quintile = {"TP": 4, "TN": 4, "FP": 3, "FN": 3}  # 5 quintiles × n
    rows = []
    for g, n_each in n_per_quintile.items():
        sub = df[df["case_type"] == g].copy()
        sub["confidence"] = np.where(sub["label"] == 1, sub["pred"], 1.0 - sub["pred"])
        sub["quintile"] = pd.qcut(sub["confidence"], q=5, labels=False, duplicates="drop")
        # take ceil(n_each / 5) per quintile, then trim to n_each*... we just want
        # roughly uniform spread across confidence
        picked = (
            sub.groupby("quintile", group_keys=False)
            .apply(lambda d: d.sample(min(len(d), max(1, n_each // 5 + 1)), random_state=0))
        )
        # sort within group by top-1 weight (descending) to reveal the hard→soft gradient
        w = picked[[f"conf_{i+1}" for i in range(K)]].to_numpy()
        picked = picked.assign(top1=w.max(axis=1)).sort_values("top1", ascending=False)
        picked = picked.head(8)
        rows.append(picked)
    selected = pd.concat(rows, ignore_index=True)
    values = selected[[f"conf_{i+1}" for i in range(K)]].to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap=SEQUENTIAL_CMAP,
                   vmin=0.0, vmax=float(values.max()))
    # group separators
    cuts = np.cumsum([8, 8, 8])
    for c in cuts:
        ax.axhline(c - 0.5, color="#94A3B8", lw=0.8)
    # left-edge color strip indicating group
    for i, r in enumerate(selected.itertuples(index=False)):
        ax.add_patch(plt.Rectangle((-0.7, i - 0.45), 0.35, 0.9,
                                   facecolor=GROUP_COLORS[r.case_type],
                                   edgecolor="none", clip_on=False))
    ax.set_xlim(-1.0, K - 0.5)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([str(i + 1) for i in range(K)])
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([f"{r.case_type}  p={r.pred:.2f}"
                        for r in selected.itertuples(index=False)], fontsize=7.5)
    ax.set_xlabel("Conformer index", fontsize=10)
    ax.set_title("Plan B — Stratified sample (4×8 = TP/TN/FP/FN × confidence quintiles)\n"
                 "Linear color norm; within each group sorted by top-1 weight",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Selection weight (linear)", fontsize=9)
    fig.savefig(OUT / "2_plan_B.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plan C — A + B side by side
# ---------------------------------------------------------------------------
def plan_C():
    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6])

    ranks = np.arange(1, K + 1)
    for i, g in enumerate(["TP", "TN", "FP", "FN"]):
        ax = fig.add_subplot(gs[0, i])
        sub = df[df["case_type"] == g][[f"conf_{j+1}" for j in range(K)]].to_numpy()
        sorted_desc = -np.sort(-sub, axis=1)
        med = np.median(sorted_desc, axis=0)
        q25, q75 = np.percentile(sorted_desc, [25, 75], axis=0)
        q05, q95 = np.percentile(sorted_desc, [5, 95], axis=0)
        c = GROUP_COLORS[g]
        ax.fill_between(ranks, q05, q95, color=c, alpha=0.18, lw=0)
        ax.fill_between(ranks, q25, q75, color=c, alpha=0.35, lw=0)
        ax.plot(ranks, med, color=c, lw=1.6, marker="o", ms=3.5)
        ax.axhline(1 / K, color="#94A3B8", lw=0.8, ls="--")
        ax.set_title(f"{g}", fontsize=10, color=c)
        ax.set_xticks(ranks)
        ax.set_xlabel("Rank", fontsize=9)
        if i == 0:
            ax.set_ylabel("Weight", fontsize=10)
        style_axes(ax)

    # right panel — small stratified heatmap (10 cases)
    ax = fig.add_subplot(gs[0, 4])
    rows = []
    for g, n_each in [("TP", 3), ("TN", 3), ("FP", 2), ("FN", 2)]:
        sub = df[df["case_type"] == g].copy()
        sub["confidence"] = np.where(sub["label"] == 1, sub["pred"], 1.0 - sub["pred"])
        sub["quintile"] = pd.qcut(sub["confidence"], q=5, labels=False, duplicates="drop")
        # one per quintile, top n_each spread
        picked = (
            sub.groupby("quintile", group_keys=False)
            .apply(lambda d: d.sample(1, random_state=0))
            .head(n_each)
        )
        rows.append(picked)
    selected = pd.concat(rows, ignore_index=True)
    values = selected[[f"conf_{j+1}" for j in range(K)]].to_numpy()
    im = ax.imshow(values, aspect="auto", cmap=SEQUENTIAL_CMAP,
                   vmin=0.0, vmax=float(values.max()))
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([str(i + 1) for i in range(K)])
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([f"{r.case_type}  p={r.pred:.2f}"
                        for r in selected.itertuples(index=False)], fontsize=7.5)
    for i, r in enumerate(selected.itertuples(index=False)):
        ax.add_patch(plt.Rectangle((-0.8, i - 0.45), 0.35, 0.9,
                                   facecolor=GROUP_COLORS[r.case_type],
                                   edgecolor="none", clip_on=False))
    ax.set_xlim(-1.1, K - 0.5)
    ax.set_xlabel("Conformer index", fontsize=10)
    ax.set_title("Stratified case examples", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Weight", fontsize=9)

    fig.suptitle("Plan C — Population-level distribution (left 4 panels) + per-case examples (right)",
                 fontsize=11)
    fig.savefig(OUT / "3_plan_C.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    current_figure()
    plan_A()
    plan_B()
    plan_C()
    print(f"\nMocks saved to {OUT}/")
