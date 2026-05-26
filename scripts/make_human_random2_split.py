#!/usr/bin/env python3
"""Build a support-aware Human random2 split.

The split keeps the same 70/10/20 sizes and label counts as random1, but
optimizes which pairs are held out so validation/test drugs and proteins have
more support in the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


def load_deduplicated_full(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(["SMILES", "Protein"]).reset_index(drop=True)
    return df[["SMILES", "Protein", "Y"]]


def load_base_split(full_df: pd.DataFrame, split_dir: Path) -> np.ndarray:
    pair_to_index = {
        pair: idx for idx, pair in enumerate(zip(full_df["SMILES"], full_df["Protein"]))
    }
    split = np.full(len(full_df), -1, dtype=np.int8)
    for code, name in enumerate(SPLIT_NAMES):
        part = pd.read_csv(split_dir / f"{name}.csv")
        for pair in zip(part["SMILES"], part["Protein"]):
            split[pair_to_index[pair]] = code

    missing = int((split < 0).sum())
    if missing:
        raise ValueError(f"{missing} pairs from full.csv are missing in {split_dir}")
    return split


def split_counts(labels: np.ndarray, split: np.ndarray) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for code, name in enumerate(SPLIT_NAMES):
        mask = split == code
        counts[name] = {
            "samples": int(mask.sum()),
            "positive": int(labels[mask].sum()),
            "negative": int(mask.sum() - labels[mask].sum()),
        }
    return counts


def make_score_fn(
    labels: np.ndarray,
    drug_codes: np.ndarray,
    protein_codes: np.ndarray,
    num_drugs: int,
    num_proteins: int,
):
    del labels

    def score(split: np.ndarray) -> float:
        train_mask = split == 0
        held_mask = split != 0
        drug_train = np.bincount(drug_codes[train_mask], minlength=num_drugs)
        protein_train = np.bincount(protein_codes[train_mask], minlength=num_proteins)

        held_drug_support = drug_train[drug_codes[held_mask]]
        held_protein_support = protein_train[protein_codes[held_mask]]

        value = 0.0
        value += (
            int((held_drug_support == 0).sum())
            + int((held_protein_support == 0).sum())
        ) * 1_000_000
        value += (
            int((held_drug_support <= 1).sum())
            + int((held_protein_support <= 1).sum())
        ) * 1_000
        value += (
            int((held_drug_support <= 2).sum())
            + int((held_protein_support <= 2).sum())
        ) * 60
        value += int(
            ((held_drug_support <= 2) & (held_protein_support <= 2)).sum()
        ) * 120

        held_drug_mean = float(held_drug_support.mean())
        held_protein_mean = float(held_protein_support.mean())
        for code in (1, 2):
            mask = split == code
            drug_support = drug_train[drug_codes[mask]]
            protein_support = protein_train[protein_codes[mask]]
            value += abs(float(drug_support.mean()) - held_drug_mean) * 10
            value += abs(float(protein_support.mean()) - held_protein_mean) * 10

        return value

    return score


def summarize_split(
    full_df: pd.DataFrame,
    split: np.ndarray,
    drug_codes: np.ndarray,
    protein_codes: np.ndarray,
    num_drugs: int,
    num_proteins: int,
    score: float,
) -> dict:
    labels = full_df["Y"].to_numpy(dtype=np.int8)
    train_mask = split == 0
    drug_train = np.bincount(drug_codes[train_mask], minlength=num_drugs)
    protein_train = np.bincount(protein_codes[train_mask], minlength=num_proteins)

    summary = {
        "score": score,
        "counts": split_counts(labels, split),
        "support": {},
        "unique_entities": {},
    }
    for code, name in enumerate(SPLIT_NAMES):
        mask = split == code
        part = full_df.loc[mask]
        summary["unique_entities"][name] = {
            "drugs": int(part["SMILES"].nunique()),
            "proteins": int(part["Protein"].nunique()),
        }

        if code == 0:
            continue

        drug_support = drug_train[drug_codes[mask]]
        protein_support = protein_train[protein_codes[mask]]
        summary["support"][name] = {
            "unseen_drug_samples": int((drug_support == 0).sum()),
            "unseen_protein_samples": int((protein_support == 0).sum()),
            "drug_support_le_1_rate": float((drug_support <= 1).mean()),
            "protein_support_le_1_rate": float((protein_support <= 1).mean()),
            "drug_support_le_2_rate": float((drug_support <= 2).mean()),
            "protein_support_le_2_rate": float((protein_support <= 2).mean()),
            "both_support_le_2_rate": float(
                ((drug_support <= 2) & (protein_support <= 2)).mean()
            ),
            "drug_support_median": float(np.median(drug_support)),
            "protein_support_median": float(np.median(protein_support)),
        }

    held_mask = split != 0
    held_drug_support = drug_train[drug_codes[held_mask]]
    held_protein_support = protein_train[protein_codes[held_mask]]
    summary["support"]["heldout"] = {
        "unseen_drug_samples": int((held_drug_support == 0).sum()),
        "unseen_protein_samples": int((held_protein_support == 0).sum()),
        "drug_support_le_1_rate": float((held_drug_support <= 1).mean()),
        "protein_support_le_1_rate": float((held_protein_support <= 1).mean()),
        "drug_support_le_2_rate": float((held_drug_support <= 2).mean()),
        "protein_support_le_2_rate": float((held_protein_support <= 2).mean()),
        "both_support_le_2_rate": float(
            ((held_drug_support <= 2) & (held_protein_support <= 2)).mean()
        ),
        "drug_support_median": float(np.median(held_drug_support)),
        "protein_support_median": float(np.median(held_protein_support)),
    }
    return summary


def optimize_split(
    initial_split: np.ndarray,
    labels: np.ndarray,
    score_fn,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    current = initial_split.copy()
    current_score = score_fn(current)
    best = current.copy()
    best_score = current_score

    for step in range(iterations):
        proposal = current.copy()
        if rng.random() < 0.9:
            label = int(rng.integers(0, 2))
            train_indices = np.flatnonzero((current == 0) & (labels == label))
            held_indices = np.flatnonzero((current != 0) & (labels == label))
            train_idx = int(rng.choice(train_indices))
            held_idx = int(rng.choice(held_indices))
            proposal[train_idx] = current[held_idx]
            proposal[held_idx] = 0
        else:
            label = int(rng.integers(0, 2))
            val_indices = np.flatnonzero((current == 1) & (labels == label))
            test_indices = np.flatnonzero((current == 2) & (labels == label))
            val_idx = int(rng.choice(val_indices))
            test_idx = int(rng.choice(test_indices))
            proposal[val_idx] = 2
            proposal[test_idx] = 1

        proposal_score = score_fn(proposal)
        temperature = max(1.0, 1000.0 * (1.0 - step / max(iterations, 1)))
        if proposal_score < current_score or rng.random() < np.exp(
            (current_score - proposal_score) / temperature
        ):
            current = proposal
            current_score = proposal_score
            if proposal_score < best_score:
                best = proposal.copy()
                best_score = proposal_score

    return best, best_score


def write_split(full_df: pd.DataFrame, split: np.ndarray, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    for code, name in enumerate(SPLIT_NAMES):
        part = full_df.loc[split == code, ["SMILES", "Protein", "Y"]]
        part.to_csv(output_dir / f"{name}.csv", index=False)


def link_cache(cache_from: Path, cache_to: Path, overwrite: bool) -> None:
    if not cache_from.exists():
        return
    cache_to.mkdir(parents=True, exist_ok=True)
    for name in ("smiles_features.pt", "protein_features.pt", "drug3d_features.pt"):
        src = cache_from / name
        dst = cache_to / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
            dst.unlink()
        dst.symlink_to(src.resolve())

    src_meta = cache_from / "meta.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text(encoding="utf-8"))
        meta["split"] = cache_to.name
        meta["smiles_cache"] = str((cache_to / "smiles_features.pt").resolve())
        meta["protein_cache"] = str((cache_to / "protein_features.pt").resolve())
        meta["drug3d_cache"] = str((cache_to / "drug3d_features.pt").resolve())
        (cache_to / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_csv", default="data/datasets/human/full.csv")
    parser.add_argument("--base_split", default="data/datasets/human/random1")
    parser.add_argument("--output_split", default="data/datasets/human/random2")
    parser.add_argument("--iterations", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link_cache", action="store_true")
    parser.add_argument("--cache_from", default="cache/features/datasets/human/random1")
    parser.add_argument("--cache_to", default="cache/features/datasets/human/random2")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    full_csv = repo_root / args.full_csv
    base_split = repo_root / args.base_split
    output_split = repo_root / args.output_split

    full_df = load_deduplicated_full(full_csv)
    labels = full_df["Y"].to_numpy(dtype=np.int8)
    initial_split = load_base_split(full_df, base_split)

    drug_codes, drug_uniques = pd.factorize(full_df["SMILES"])
    protein_codes, protein_uniques = pd.factorize(full_df["Protein"])
    score_fn = make_score_fn(
        labels=labels,
        drug_codes=drug_codes,
        protein_codes=protein_codes,
        num_drugs=len(drug_uniques),
        num_proteins=len(protein_uniques),
    )

    initial_score = score_fn(initial_split)
    best_split, best_score = optimize_split(
        initial_split=initial_split,
        labels=labels,
        score_fn=score_fn,
        iterations=args.iterations,
        seed=args.seed,
    )

    write_split(full_df, best_split, output_split, overwrite=args.overwrite)
    summary = summarize_split(
        full_df=full_df,
        split=best_split,
        drug_codes=drug_codes,
        protein_codes=protein_codes,
        num_drugs=len(drug_uniques),
        num_proteins=len(protein_uniques),
        score=best_score,
    )
    summary["source"] = {
        "full_csv": str(full_csv),
        "base_split": str(base_split),
        "initial_score": initial_score,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    (output_split / "split_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if args.link_cache:
        link_cache(
            cache_from=repo_root / args.cache_from,
            cache_to=repo_root / args.cache_to,
            overwrite=args.overwrite,
        )

    print(f"initial_score={initial_score:.3f}")
    print(f"best_score={best_score:.3f}")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["support"], indent=2))
    print(f"saved={output_split}")
    if args.link_cache:
        print(f"linked_cache={repo_root / args.cache_to}")


if __name__ == "__main__":
    main()
