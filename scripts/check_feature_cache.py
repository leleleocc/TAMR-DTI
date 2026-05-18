#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

FEATURE_SCHEMA_VERSION = 3
DRUG_1D_SHAPE = (354, 128)
PROTEIN_SHAPE = (128, 1024)
PROTEIN_MASK_SHAPE = (128,)
DRUG3D_NUM_CONFORMERS = 8
DRUG3D_FEATURE_SHAPE = (DRUG3D_NUM_CONFORMERS, 64, 128)
DRUG3D_COOR_SHAPE = (DRUG3D_NUM_CONFORMERS, 64, 3)
DRUG3D_CONF_MASK_SHAPE = (DRUG3D_NUM_CONFORMERS,)
DRUG3D_ENERGY_SHAPE = (DRUG3D_NUM_CONFORMERS,)


def feature_of(entry):
    if isinstance(entry, dict):
        return entry.get("feature")
    return entry


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous().float()
    return hashlib.sha1(tensor.numpy().tobytes()).hexdigest()


def load_cache(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def shape_counter(cache, extractor):
    counts = Counter()
    for entry in cache.values():
        tensor = extractor(entry)
        counts[tuple(tensor.shape) if torch.is_tensor(tensor) else None] += 1
    return counts


def diversity_stats(cache, extractor, max_items):
    hashes = []
    first = None
    max_abs_diff = 0.0
    for idx, entry in enumerate(cache.values()):
        if max_items > 0 and idx >= max_items:
            break

        tensor = extractor(entry)
        if not torch.is_tensor(tensor):
            continue

        tensor = tensor.float()
        hashes.append(tensor_hash(tensor))
        if first is None:
            first = tensor
        elif first.shape == tensor.shape:
            max_abs_diff = max(max_abs_diff, (tensor - first).abs().max().item())

    return len(set(hashes)), max_abs_diff, len(hashes)


def most_common_shapes(counter):
    return ", ".join(f"{shape}:{count}" for shape, count in counter.most_common(4))


def check_shape(cache_name, counter, expected_shape, failures):
    bad_count = sum(count for shape, count in counter.items() if shape != expected_shape)
    if bad_count:
        failures.append(f"{cache_name}: {bad_count} entries have shape != {expected_shape}")


def check_diversity(cache_name, count, unique_hashes, failures):
    if count <= 1:
        return
    min_unique = max(10, int(count * 0.01))
    if unique_hashes < min_unique:
        failures.append(
            f"{cache_name}: only {unique_hashes} unique feature hashes among {count} checked entries"
        )


def main():
    parser = argparse.ArgumentParser(description="Validate offline feature cache health")
    parser.add_argument("--data", required=True, help="dataset path under data/, e.g. datasets/bindingdb")
    parser.add_argument("--split", default="random")
    parser.add_argument("--cache_root", default="cache/features")
    parser.add_argument(
        "--max_hash_items",
        type=int,
        default=0,
        help="limit diversity hashing; 0 means check all entries",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / args.cache_root / args.data / args.split
    meta_path = cache_dir / "meta.json"
    smiles_cache = load_cache(cache_dir / "smiles_features.pt")
    protein_cache = load_cache(cache_dir / "protein_features.pt")
    drug3d_cache = load_cache(cache_dir / "drug3d_features.pt")

    failures = []
    print(f"[cache] dir={cache_dir}")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[meta] feature_schema_version={meta.get('feature_schema_version')}")
        if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            failures.append(f"meta: feature_schema_version != {FEATURE_SCHEMA_VERSION}")
    else:
        failures.append(f"meta: missing {meta_path}")

    checks = [
        ("smiles", smiles_cache, feature_of, DRUG_1D_SHAPE),
        ("protein", protein_cache, feature_of, PROTEIN_SHAPE),
        ("drug3d", drug3d_cache, lambda entry: entry.get("feature") if isinstance(entry, dict) else None, DRUG3D_FEATURE_SHAPE),
    ]
    for name, cache, extractor, expected_shape in checks:
        counter = shape_counter(cache, extractor)
        unique_hashes, max_abs_diff, checked = diversity_stats(cache, extractor, args.max_hash_items)
        print(
            f"[{name}] count={len(cache)} checked={checked} shapes={most_common_shapes(counter)} "
            f"unique_hashes={unique_hashes} max_abs_diff_from_first={max_abs_diff:.6g}"
        )
        check_shape(name, counter, expected_shape, failures)
        check_diversity(name, checked, unique_hashes, failures)

    protein_mask_shapes = shape_counter(
        protein_cache,
        lambda entry: entry.get("mask") if isinstance(entry, dict) else None,
    )
    drug3d_coor_shapes = shape_counter(
        drug3d_cache,
        lambda entry: entry.get("coor") if isinstance(entry, dict) else None,
    )
    drug3d_conf_mask_shapes = shape_counter(
        drug3d_cache,
        lambda entry: entry.get("conf_mask") if isinstance(entry, dict) else None,
    )
    drug3d_energy_shapes = shape_counter(
        drug3d_cache,
        lambda entry: entry.get("energy") if isinstance(entry, dict) else None,
    )
    drug3d_versions = Counter(
        entry.get("schema_version") if isinstance(entry, dict) else None for entry in drug3d_cache.values()
    )
    drug3d_ok = sum(1 for entry in drug3d_cache.values() if isinstance(entry, dict) and entry.get("ok") is True)
    print(f"[protein_mask] shapes={most_common_shapes(protein_mask_shapes)}")
    print(f"[drug3d_coor] shapes={most_common_shapes(drug3d_coor_shapes)}")
    print(f"[drug3d_conf_mask] shapes={most_common_shapes(drug3d_conf_mask_shapes)}")
    print(f"[drug3d_energy] shapes={most_common_shapes(drug3d_energy_shapes)}")
    print(f"[drug3d] ok={drug3d_ok}/{len(drug3d_cache)} schema_versions={dict(drug3d_versions)}")
    check_shape("protein_mask", protein_mask_shapes, PROTEIN_MASK_SHAPE, failures)
    check_shape("drug3d_coor", drug3d_coor_shapes, DRUG3D_COOR_SHAPE, failures)
    check_shape("drug3d_conf_mask", drug3d_conf_mask_shapes, DRUG3D_CONF_MASK_SHAPE, failures)
    check_shape("drug3d_energy", drug3d_energy_shapes, DRUG3D_ENERGY_SHAPE, failures)
    if set(drug3d_versions) != {FEATURE_SCHEMA_VERSION}:
        failures.append(f"drug3d: schema_version must be {FEATURE_SCHEMA_VERSION}")

    if failures:
        print("[result] FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("[result] OK")


if __name__ == "__main__":
    main()
