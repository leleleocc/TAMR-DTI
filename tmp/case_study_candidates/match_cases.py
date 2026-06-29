"""Match curated case-study candidates against the BioSNAP test split.

Usage (on the remote machine, where data/ lives):
    python match_cases.py \
        --test_csv data/biosnap/random/test.csv \
        --candidates tmp/case_study_candidates/candidates.json \
        [--interp_csv outputs/interpretability/biosnap_seed42/biosnap_seed42_interpretability_samples.csv]

For each candidate, reports the row_index, label, and (if --interp_csv given)
the model's predicted probability and TP/TN/FP/FN class.

Match strategy:
  - Drug: canonicalize via RDKit and compare InChIKeys (most robust).
  - Protein: equality on uppercase/whitespace-stripped sequence; if no equality
    hit, fall back to substring (BioSNAP truncates to 1200, UniProt full).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


def normalize_protein(s: str) -> str:
    """Match pre_extract.format_protein_sequence (without inserting spaces)."""
    s = "".join(str(s).upper().split())
    return s


def canonical_inchikey(s: str) -> str:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToInchiKey(mol)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", required=True, type=Path)
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--interp_csv", type=Path, default=None,
                   help="optional: per-sample model outputs to join in")
    p.add_argument("--max_test_rows", type=int, default=0,
                   help="for quick smoke testing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    candidates = json.loads(args.candidates.read_text())
    df = pd.read_csv(args.test_csv)
    if args.max_test_rows:
        df = df.head(args.max_test_rows)
    print(f"[data] loaded test.csv  rows={len(df)}  columns={list(df.columns)}")

    # Normalize test rows once
    print("[data] canonicalizing test SMILES via RDKit ...")
    df["__inchikey"] = df["SMILES"].astype(str).apply(canonical_inchikey)
    df["__prot_norm"] = df["Protein"].astype(str).apply(normalize_protein)
    bad = (df["__inchikey"] == "").sum()
    if bad:
        print(f"  warning: {bad} rows had invalid SMILES (skipped in matching)")

    # Optionally join interp outputs
    interp = None
    if args.interp_csv and args.interp_csv.exists():
        interp = pd.read_csv(args.interp_csv)
        print(f"[interp] loaded {len(interp)} interp rows  cols={list(interp.columns)[:8]}…")

    # Match
    print("\n=== matches ===")
    hits = []
    for cand in candidates:
        target_norm = normalize_protein(cand["target_sequence"])
        drug_ik = cand["drug_inchikey"]
        # equality
        eq_mask = (df["__inchikey"] == drug_ik) & (df["__prot_norm"] == target_norm)
        if eq_mask.any():
            match_rows = df[eq_mask].index.tolist()
            match_type = "exact"
        else:
            # substring fallback: BioSNAP sequence may be truncated to 1200
            # Try forward containment in both directions.
            def contains(p_norm: str) -> bool:
                return (p_norm and target_norm and
                        (p_norm in target_norm or target_norm in p_norm))
            sub_mask = (df["__inchikey"] == drug_ik) & df["__prot_norm"].apply(contains)
            match_rows = df[sub_mask].index.tolist()
            match_type = "substring" if match_rows else "none"

        if not match_rows:
            print(f"  -- {cand['drug_name']:14s} + {cand['target_label']:18s}  no match")
            continue

        for row_idx in match_rows:
            label = df.iloc[row_idx].get("Y", df.iloc[row_idx].get("label"))
            extra = ""
            if interp is not None and "row_index" in interp.columns:
                m = interp[interp["row_index"] == row_idx]
                if not m.empty:
                    r = m.iloc[0]
                    extra = (f"  | pred={float(r['pred']):.3f}"
                             f"  case={r['case_type']}")
            print(f"  ++ {cand['drug_name']:14s} + {cand['target_label']:18s} "
                  f" row={row_idx:5d}  label={label}  [{match_type}]{extra}")
            hits.append({
                "drug": cand["drug_name"], "target": cand["target_label"],
                "uniprot": cand["target_uniprot"], "row_index": int(row_idx),
                "label": int(label) if label is not None else -1,
                "match_type": match_type,
                "test_protein_length": int(len(df.iloc[row_idx]["__prot_norm"])),
                "uniprot_length": int(len(target_norm)),
            })

    if hits:
        out = args.candidates.parent / "match_hits.csv"
        pd.DataFrame(hits).to_csv(out, index=False)
        print(f"\nSaved {len(hits)} hits → {out}")
    else:
        print("\nNo hits found.  Check column names in test.csv.")


if __name__ == "__main__":
    main()
