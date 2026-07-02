"""Preview the updated plot_conformer_distribution.py logic locally with
synthetic data that mimics the *actual* skew observed in production:
median rank-1 ≈ 0.14, mean rank-1 ≈ 0.24."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# Build a synthetic CSV that matches the production skew, then call the real
# script's main() with the file path monkey-patched.
RNG = np.random.default_rng(42)
K = 8
GROUPS = ["TP", "TN", "FP", "FN"]
NS = {"TP": 2341, "TN": 2404, "FP": 341, "FN": 407}


def synth_group(n: int, alpha_uniform: float, alpha_concentrated: float,
                frac_concentrated: float) -> np.ndarray:
    """Right-skewed mixture: most samples ~uniform-on-7, minority concentrated.
    Most BioSNAP samples have 7 valid conformers (slot 8 masked), so for the
    'uniform' subset we directly assign 1/7 to slots 0..6 and 0 to slot 7
    (Dirichlet would jitter and shift the median off 0.143)."""
    n_conc = int(n * frac_concentrated)
    n_uni = n - n_conc
    uni = np.zeros((n_uni, K))
    uni[:, :K - 1] = 1.0 / (K - 1)
    b = np.full(K, alpha_concentrated)
    conc = RNG.dirichlet(b, size=n_conc)
    # Mask slot 8 in some concentrated samples too (mimic conf_mask)
    mask_conc = RNG.random(n_conc) < 0.5
    if mask_conc.any():
        idx = np.where(mask_conc)[0]
        conc[idx, -1] = 0.0
        conc[idx] /= conc[idx].sum(axis=1, keepdims=True)
    out = np.vstack([uni, conc])
    RNG.shuffle(out)
    return out


rows = []
for g in GROUPS:
    n = NS[g]
    # Target stats from paper_macros.tex:
    # TP/TN: top1_mean≈0.24, H≈0.93;  FP/FN: top1_mean≈0.25/0.26, H≈0.92/0.90
    frac = {"TP": 0.22, "TN": 0.22, "FP": 0.27, "FN": 0.34}[g]
    w = synth_group(n, alpha_uniform=20.0, alpha_concentrated=0.8,
                    frac_concentrated=frac)
    for i in range(n):
        valid = (w[i] > 0)
        valid_count = int(valid.sum())
        # Normalized entropy by log(valid_count)
        p = w[i][valid]
        H = -(p * np.log(p + 1e-12)).sum()
        H_norm = H / np.log(max(valid_count, 2))
        row = {f"conf_{j+1}": w[i, j] for j in range(K)}
        row["case_type"] = g
        row["conformer_entropy_norm"] = H_norm
        rows.append(row)
df = pd.DataFrame(rows)
out_csv = Path(__file__).resolve().parent / "synth_interp.csv"
df.to_csv(out_csv, index=False)
print(f"wrote synthetic CSV: {out_csv}  rows={len(df)}")

# Now monkey-patch the real script and invoke main()
import importlib.util
script_path = ROOT / "scripts/plot_conformer_distribution.py"
spec = importlib.util.spec_from_file_location("plot_mod", script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Override AFTER load so main()'s globals see the new values
mod.CSV = out_csv
preview_dir = Path(__file__).resolve().parent / "preview_out"
preview_dir.mkdir(parents=True, exist_ok=True)
mod.FIG_DIR = preview_dir
mod.PAPER_PDF = preview_dir / "interpretability_conformer_weights.pdf"
mod.main()
print(f"\npreview PNG: {mod.FIG_DIR/'plan_A_conformer_distribution.png'}")
