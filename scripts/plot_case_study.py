"""Figure 4 — Case-study interpretability for TAMR-DTI.

Panel A (top-left): Residue-level Mamba refinement profile R(r) for a
representative DHFR TP row, with NADP+/substrate binding sites overlaid.

Panel B (top-right): Forest plot of per-target enrichment fold at top-20%
residues across all 9 UniProt targets with TP matches.

Panel C (bottom): Conformer-selection weights for the 4 DHFR TP rows,
showing that conformer selection varies row-to-row even though residue
attention is protein-dominated.

Reads:
  outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy
  outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv
  outputs/interpretability/biosnap_seed42/case_study/per_row.csv
  outputs/interpretability/biosnap_seed42/case_study/aggregate.csv
  tmp/case_study_candidates/candidates.json
  data/datasets/biosnap/random/test.csv

Writes:
  outputs/interpretability/biosnap_seed42/figs/case_study_figure4.{png,svg}
  TAMR-DTI__.../figures/case_study_dhfr.pdf
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_theme import (  # noqa: E402
    GROUP_COLORS,
    TAMR,
    apply_paper_theme,
    style_axes,
)

apply_paper_theme(plt)

INTERP_CSV = ROOT / "outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv"
PROFILES = ROOT / "outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy"
TEST_CSV = ROOT / "data/datasets/biosnap/random/test.csv"
CANDIDATES = ROOT / "tmp/case_study_candidates/candidates.json"
PER_ROW = ROOT / "outputs/interpretability/biosnap_seed42/case_study/per_row.csv"
AGG = ROOT / "outputs/interpretability/biosnap_seed42/case_study/aggregate.csv"

FIG_DIR = ROOT / "outputs/interpretability/biosnap_seed42/figs"
PAPER_PDF = (
    ROOT
    / "TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction"
    / "figures"
    / "case_study_dhfr.pdf"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_TOKENS = 128
TOP_FRAC = 0.20


def project_tokens_to_residues(rt: np.ndarray, L: int, K: int = K_TOKENS) -> np.ndarray:
    out = np.zeros(L, dtype=np.float64)
    for t in range(K):
        s = int(math.floor(t * L / K))
        e = min(int(math.ceil((t + 1) * L / K)), L)
        if e > s:
            out[s:e] = rt[t]
    return out


def norm_prot(s: str) -> str:
    return "".join(str(s).upper().split())


def main() -> None:
    candidates = json.loads(CANDIDATES.read_text())
    dhfr = next(c for c in candidates if c["target_uniprot"] == "P00374")
    nadp_residues = sorted({r for s in dhfr["binding_sites"]
                            if s.get("ligand", "") == "NADP(+)"
                            for r in s["residues"]})
    substrate_residues = sorted({r for s in dhfr["binding_sites"]
                                  if s.get("ligand", "") == "substrate"
                                  for r in s["residues"]})
    all_binding = sorted(set(nadp_residues) | set(substrate_residues))

    df_interp = pd.read_csv(INTERP_CSV)
    profiles = np.load(PROFILES)
    df_test = pd.read_csv(TEST_CSV)
    df_test["_p"] = df_test["Protein"].astype(str).apply(norm_prot)
    df_per_row = pd.read_csv(PER_ROW)
    df_agg = pd.read_csv(AGG)

    # Find DHFR TP rows
    dhfr_seq = norm_prot(dhfr["target_sequence"])
    dhfr_mask = df_test["_p"].apply(lambda p: p and (p == dhfr_seq or p in dhfr_seq or dhfr_seq in p))
    dhfr_test_idx = df_test.index[dhfr_mask].tolist()
    dhfr_tp = df_interp[
        (df_interp["row_index"].isin(dhfr_test_idx))
        & (df_interp["case_type"] == "TP")
    ].sort_values("pred", ascending=False).reset_index(drop=True)
    print(f"[dhfr] {len(dhfr_tp)} TP rows")

    # Representative row = highest confidence (closest to a true positive)
    rep_row_idx = int(dhfr_tp.iloc[0]["row_index"])
    L_rep = int(len(df_test.loc[rep_row_idx, "_p"]))
    residue_R = project_tokens_to_residues(profiles[rep_row_idx], L_rep)

    # Identify top-20% residues for shading
    k_top = max(1, int(math.ceil(TOP_FRAC * L_rep)))
    top_thresh = np.sort(residue_R)[-k_top]

    # -----------------------------------------------------------------------
    # Figure layout: 2 rows, top: 2 wide panels (A + B), bottom: panel C
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(13.6, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.8])

    # ----- Panel A: residue-level R(r) for representative DHFR TP row -----
    axA = fig.add_subplot(gs[0, 0])
    positions = np.arange(1, L_rep + 1)
    # baseline line
    axA.plot(positions, residue_R, color=TAMR, lw=1.2, alpha=0.85)
    axA.fill_between(positions, 0.0, residue_R, color=TAMR, alpha=0.18,
                     linewidth=0)
    # threshold line for top-20%
    axA.axhline(top_thresh, ls="--", color="#94A3B8", lw=0.8,
                label=f"top-20% threshold")
    # Overlay binding-site residues
    for r in nadp_residues:
        if 1 <= r <= L_rep:
            axA.axvline(r, color="#00876C", lw=1.2, alpha=0.55, zorder=1)
    for r in substrate_residues:
        if 1 <= r <= L_rep:
            axA.axvline(r, color="#D97706", lw=1.2, alpha=0.55, zorder=1)
    # Build legend entries for binding categories
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=TAMR, lw=1.5, alpha=0.85,
               label=r"Mamba refinement $R(r)$"),
        Line2D([0], [0], color="#00876C", lw=1.5, alpha=0.85,
               label="NADP$^+$ binding (UniProt)"),
        Line2D([0], [0], color="#D97706", lw=1.5, alpha=0.85,
               label="Substrate binding (UniProt)"),
        Line2D([0], [0], color="#94A3B8", lw=0.8, ls="--",
               label="Top-20% threshold"),
    ]
    axA.legend(handles=legend_handles, loc="upper right",
               fontsize=8.0, frameon=True, framealpha=0.92,
               edgecolor="#94A3B8")
    axA.set_xlim(0, L_rep + 1)
    axA.set_xlabel("Residue position (UniProt numbering)", fontsize=10)
    axA.set_ylabel(r"$R(r)$  =  $\sigma(g)\,\|h^{\mathrm{M}}_t - h_t\|_2$",
                   fontsize=10)
    pred_v = float(dhfr_tp.iloc[0]["pred"])
    top1_v = float(dhfr_tp.iloc[0]["conformer_top1"])
    axA.set_title(
        f"(A) DHFR (P00374) — TP row {rep_row_idx}, pred={pred_v:.3f}, "
        f"top-1 conformer={top1_v:.2f}",
        fontsize=10,
    )
    style_axes(axA, grid_axis="y")

    # ----- Panel B: forest plot of fold across targets -----
    axB = fig.add_subplot(gs[0, 1])
    # Sort agg by fold descending, but only show targets with n>=2 TP rows
    df_show = df_agg[df_agg["n_rows"] >= 1].copy().sort_values(
        "top20_fold_mean", ascending=True
    )
    y = np.arange(len(df_show))
    folds = df_show["top20_fold_mean"].to_numpy()
    p_combined = df_show["top20_p_combined"].to_numpy()
    n_rows = df_show["n_rows"].to_numpy()
    # Color: green if p<0.05, gray otherwise
    colors = ["#00876C" if p < 0.05 else "#94A3B8" for p in p_combined]
    bars = axB.barh(y, folds, color=colors, height=0.6, alpha=0.9,
                    edgecolor="white", linewidth=0.5)
    axB.axvline(1.0, color="#374151", ls="--", lw=0.8,
                label="no enrichment")
    for i, (yp, fold, p, n, lbl) in enumerate(
        zip(y, folds, p_combined, n_rows, df_show["label"].tolist())
    ):
        p_txt = f"p={p:.1e}" if p < 0.05 else "n.s."
        axB.text(max(fold, 0.05) + 0.07, yp,
                 f"{p_txt}  n={n}",
                 va="center", ha="left", fontsize=7.5,
                 color="#1F2937")
    labels = [f"{r['label']} ({r['uniprot']})" for _, r in df_show.iterrows()]
    axB.set_yticks(y)
    axB.set_yticklabels(labels, fontsize=8)
    axB.set_xlabel(r"Enrichment fold of binding residues in top-20% $R(r)$",
                   fontsize=9)
    axB.set_xlim(0, max(2.6, folds.max() * 1.15))
    axB.set_title("(B) Per-target binding-site enrichment (TP only)",
                  fontsize=10)
    style_axes(axB, grid_axis="x")
    axB.legend(loc="lower right", fontsize=7.5, frameon=False)

    # ----- Panel C: conformer weights across 4 DHFR rows -----
    axC = fig.add_subplot(gs[1, :])
    K = 8
    rows_idx = dhfr_tp["row_index"].tolist()
    conf_mat = np.zeros((len(rows_idx), K))
    for i, ri in enumerate(rows_idx):
        ir = df_interp[df_interp["row_index"] == ri].iloc[0]
        conf_mat[i] = [ir[f"conf_{j+1}"] for j in range(K)]
    # Sort within row in descending order to align top-1 conformer at index 0
    sorted_conf = -np.sort(-conf_mat, axis=1)
    x_pos = np.arange(K)
    width = 0.20
    for i, ri in enumerate(rows_idx):
        ir = df_interp[df_interp["row_index"] == ri].iloc[0]
        axC.bar(x_pos + (i - 1.5) * width, sorted_conf[i], width,
                label=f"row {ri}  pred={float(ir['pred']):.3f}",
                color=plt.cm.viridis(0.15 + 0.6 * i / max(1, len(rows_idx) - 1)),
                edgecolor="white", linewidth=0.4)
    axC.axhline(1 / K, color="#94A3B8", ls="--", lw=0.8,
                label=f"uniform 1/K={1/K:.3f}")
    axC.set_xticks(x_pos)
    axC.set_xticklabels([f"r{i+1}" for i in range(K)])
    axC.set_xlabel("Rank within sample (descending)", fontsize=9)
    axC.set_ylabel("Conformer selection weight", fontsize=9)
    axC.set_title(
        "(C) Conformer weights across the 4 DHFR TP rows (same protein, different drugs)",
        fontsize=10,
    )
    axC.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
    style_axes(axC, grid_axis="y")

    # ----- super title -----
    p_dhfr = float(df_agg[df_agg["uniprot"] == "P00374"]["top20_p_combined"].iloc[0])
    fold_dhfr = float(df_agg[df_agg["uniprot"] == "P00374"]["top20_fold_mean"].iloc[0])
    fig.suptitle(
        "Case study: DHFR is the one target where TAMR-DTI's Mamba refinement "
        f"concentrates on annotated binding residues (fold={fold_dhfr:.2f}, "
        f"Fisher's-exact $p_\\mathrm{{Stouffer}}$={p_dhfr:.1e})",
        fontsize=11,
    )

    # save
    fig.savefig(FIG_DIR / "case_study_figure4.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / "case_study_figure4.svg", bbox_inches="tight")
    fig.savefig(PAPER_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] wrote {FIG_DIR/'case_study_figure4.png'}")
    print(f"[ok] wrote {FIG_DIR/'case_study_figure4.svg'}")
    print(f"[ok] wrote {PAPER_PDF}")


if __name__ == "__main__":
    main()
