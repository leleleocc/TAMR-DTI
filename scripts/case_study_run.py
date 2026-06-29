"""Population-level binding-site enrichment for TAMR-DTI Mamba refinement.

Reads:
  - outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy   (5493, 128)
  - outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv
  - data/datasets/biosnap/random/test.csv     (for sequence length per row)
  - tmp/case_study_candidates/candidates.json (UniProt annotations + binding sites)

For each TP row whose protein matches one of the 9 UniProt targets:
  1. Project the 128-token Mamba refinement profile R_t back onto residues
     using start = floor(t * L_eff / 128), end = ceil((t+1) * L_eff / 128).
  2. Take the top-K% residues (default top-20%) as "model-attended set".
  3. Compare against UniProt-annotated binding residues (intersected with
     [1, L_eff] since BioSNAP truncates proteins to 1200).
  4. Fisher's exact test (one-sided "greater") for enrichment.

Aggregates per UniProt target (n=number of TP rows) using:
  - mean fold (a/b vs c/d normalised)
  - Stouffer-combined fisher p-value across rows
  - min/median p

Writes:
  outputs/interpretability/biosnap_seed42/case_study/results.json
  outputs/interpretability/biosnap_seed42/case_study/per_row.csv
  outputs/interpretability/biosnap_seed42/case_study/aggregate.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/interpretability/biosnap_seed42/case_study"
OUT.mkdir(parents=True, exist_ok=True)

INTERP_CSV = ROOT / "outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv"
PROFILES = ROOT / "outputs/interpretability/biosnap_seed42/biosnap_seed42_mamba_profiles.npy"
TEST_CSV = ROOT / "data/datasets/biosnap/random/test.csv"
CANDIDATES = ROOT / "tmp/case_study_candidates/candidates.json"

K_TOKENS = 128
TOPK_FRACTIONS = [0.10, 0.20, 0.30]
BIOSNAP_MAX_LEN = 1200  # known BioSNAP truncation


def norm_prot(s: str) -> str:
    return "".join(str(s).upper().split())


def project_tokens_to_residues(rt: np.ndarray, L_eff: int, K: int = K_TOKENS) -> np.ndarray:
    """Inverse of adaptive_avg_pool1d: spread each token's R_t value uniformly
    over the residues it pools."""
    out = np.zeros(L_eff, dtype=np.float64)
    for t in range(K):
        s = int(math.floor(t * L_eff / K))
        e = int(math.ceil((t + 1) * L_eff / K))
        e = min(e, L_eff)
        if e > s:
            out[s:e] = rt[t]
    return out


def fisher_topk(values: np.ndarray, binding_set: set[int], frac: float) -> dict:
    """One-sided Fisher's exact test: are binding residues over-represented
    in the top frac% of |R_t|?"""
    L = values.size
    k = max(1, int(math.ceil(frac * L)))
    # Top-k indices (1-indexed residues)
    order = np.argsort(-values)  # descending
    top_set = set((order[:k] + 1).astype(int).tolist())  # 1-indexed
    a = len(top_set & binding_set)            # hit ∩ binding
    b = k - a                                  # hit ∩ non-binding
    n_binding = len(binding_set)
    c = n_binding - a                          # non-hit ∩ binding
    d = L - k - c                              # non-hit ∩ non-binding
    table = np.array([[a, b], [c, d]], dtype=int)
    if d < 0 or any(x < 0 for x in (a, b, c, d)):
        return {"k": k, "a": a, "b": b, "c": c, "d": d, "fold": float("nan"),
                "p": float("nan"), "L": L, "n_binding": n_binding}
    odds, p = fisher_exact(table, alternative="greater")
    # Fold = (a / k) / (n_binding / L)
    fold = (a / k) / (n_binding / L) if (k > 0 and n_binding > 0 and L > 0) else float("nan")
    return {"k": k, "a": a, "b": b, "c": c, "d": d, "fold": float(fold),
            "p": float(p), "L": L, "n_binding": n_binding}


def stouffer_combine(pvals: list[float]) -> float:
    """Stouffer's z-method, equal weights."""
    if not pvals:
        return float("nan")
    from scipy.stats import norm
    # Clip extreme values to keep z finite
    pvals = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0 - 1e-12)
    z = norm.isf(pvals)
    z_sum = z.sum() / math.sqrt(len(pvals))
    return float(norm.sf(z_sum))


def main() -> None:
    print(f"[data] loading {INTERP_CSV.name}")
    df_interp = pd.read_csv(INTERP_CSV)
    print(f"  rows={len(df_interp)}")
    profiles = np.load(PROFILES)
    print(f"  profiles={profiles.shape}  dtype={profiles.dtype}")

    df_test = pd.read_csv(TEST_CSV)
    df_test["_p"] = df_test["Protein"].astype(str).apply(norm_prot)

    candidates = json.loads(CANDIDATES.read_text())
    # Collapse to per-UniProt target
    target_meta: dict[str, dict] = {}
    for c in candidates:
        u = c["target_uniprot"]
        if u in target_meta:
            continue
        b = sorted({r for site in c.get("binding_sites", []) for r in site["residues"]})
        a_ = sorted({r for site in c.get("active_sites", []) for r in site["residues"]})
        target_meta[u] = {
            "label": c["target_label"],
            "seq": norm_prot(c["target_sequence"]),
            "L_uniprot": int(c["target_length"]),
            "binding_residues_uniprot": b,
            "active_residues_uniprot": a_,
            "pdb_refs": c.get("pdb_refs", []),
        }
    print(f"[meta] unique UniProt targets: {len(target_meta)}")

    per_row_rows = []
    agg = {}
    for u, meta in target_meta.items():
        seq = meta["seq"]
        mask = df_test["_p"].apply(lambda p: bool(p) and bool(seq) and (p == seq or p in seq or seq in p))
        matched_idx = df_test.index[mask].tolist()
        # Restrict to TP rows
        sub_interp = df_interp[
            df_interp["row_index"].isin(matched_idx)
            & (df_interp["case_type"] == "TP")
        ]
        if sub_interp.empty:
            print(f"  -- {u} ({meta['label']}): no TP matches")
            continue

        binding_uniprot = meta["binding_residues_uniprot"]
        rows_for_target = []
        for _, ir in sub_interp.iterrows():
            r_idx = int(ir["row_index"])
            L_eff = int(len(df_test.loc[r_idx, "_p"]))
            # Restrict binding residues to those that survived the truncation
            binding_in_range = [r for r in binding_uniprot if 1 <= r <= L_eff]
            if not binding_in_range:
                continue
            rt = profiles[r_idx]
            residue_R = project_tokens_to_residues(rt, L_eff, K=K_TOKENS)
            row_results = {
                "row_index": r_idx, "uniprot": u, "label_target": meta["label"],
                "L_eff": L_eff, "L_uniprot": meta["L_uniprot"],
                "n_binding_in_range": len(binding_in_range),
                "n_binding_uniprot": len(binding_uniprot),
                "pred": float(ir["pred"]),
                "conformer_top1": float(ir["conformer_top1"]),
                "conformer_entropy_norm": float(ir["conformer_entropy_norm"]),
                "mamba_refine_mean_seq": float(np.mean(residue_R)),
                "mamba_refine_max_seq": float(np.max(residue_R)),
                "smiles": str(ir["smiles"])[:60],
            }
            for frac in TOPK_FRACTIONS:
                ft = fisher_topk(residue_R, set(binding_in_range), frac)
                row_results[f"top{int(frac*100)}_a"] = ft["a"]
                row_results[f"top{int(frac*100)}_k"] = ft["k"]
                row_results[f"top{int(frac*100)}_fold"] = ft["fold"]
                row_results[f"top{int(frac*100)}_p"] = ft["p"]
            rows_for_target.append(row_results)
            per_row_rows.append(row_results)

        if not rows_for_target:
            print(f"  -- {u} ({meta['label']}): all binding residues out of range")
            continue

        # Aggregate per target
        df_t = pd.DataFrame(rows_for_target)
        n_rows = len(df_t)
        agg_target = {
            "uniprot": u, "label": meta["label"], "n_rows": int(n_rows),
            "L_uniprot": meta["L_uniprot"], "n_binding": len(binding_uniprot),
        }
        for frac in TOPK_FRACTIONS:
            tag = f"top{int(frac*100)}"
            agg_target[f"{tag}_fold_mean"] = float(df_t[f"{tag}_fold"].mean())
            agg_target[f"{tag}_fold_median"] = float(df_t[f"{tag}_fold"].median())
            pvs = df_t[f"{tag}_p"].tolist()
            agg_target[f"{tag}_p_combined"] = stouffer_combine(pvs)
            agg_target[f"{tag}_p_median"] = float(np.median(pvs))
            agg_target[f"{tag}_p_min"] = float(np.min(pvs))
            agg_target[f"{tag}_n_sig"] = int(sum(p < 0.05 for p in pvs))
        agg[u] = agg_target

    # Save outputs
    df_per_row = pd.DataFrame(per_row_rows)
    df_per_row.to_csv(OUT / "per_row.csv", index=False)
    df_agg = pd.DataFrame(list(agg.values()))
    df_agg.to_csv(OUT / "aggregate.csv", index=False)
    with (OUT / "results.json").open("w") as f:
        json.dump({"per_target_aggregate": agg,
                   "n_rows_total": int(len(df_per_row))},
                  f, indent=2)

    print(f"\n[done] {len(df_per_row)} TP rows analyzed across {len(agg)} UniProt targets")
    print(f"  → {OUT/'per_row.csv'}")
    print(f"  → {OUT/'aggregate.csv'}")
    print(f"  → {OUT/'results.json'}")

    print("\nPer-target enrichment (top-20% residues):")
    df_show = df_agg.sort_values("top20_p_combined")
    for _, r in df_show.iterrows():
        n_sig = r["top20_n_sig"]; n = r["n_rows"]
        print(f"  {r['uniprot']} ({r['label']:<14s}) n={n:<2d}  "
              f"fold={r['top20_fold_mean']:.2f}  "
              f"p_comb={r['top20_p_combined']:.3e}  "
              f"sig={n_sig}/{n}")


if __name__ == "__main__":
    main()
