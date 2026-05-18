#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def crop_entry(entry, n: int, slots: int):
    if not isinstance(entry, dict):
        return entry

    out = dict(entry)
    for key in ("feature", "coor", "conf_mask", "energy"):
        value = out.get(key)
        if torch.is_tensor(value):
            out[key] = value.clone()

    feature = out.get("feature")
    coor = out.get("coor")
    conf_mask = out.get("conf_mask")
    energy = out.get("energy")

    keep = min(n, slots)
    if torch.is_tensor(feature) and feature.size(0) == slots:
        feature[keep:] = 0
    if torch.is_tensor(coor) and coor.size(0) == slots:
        coor[keep:] = 0
    if torch.is_tensor(conf_mask) and conf_mask.size(0) == slots:
        conf_mask[keep:] = 0
        if conf_mask.sum().item() == 0:
            conf_mask[0] = 1.0
    if torch.is_tensor(energy) and energy.size(0) == slots:
        energy[keep:] = 0

    out["effective_num_conformers"] = n
    out["derived_from_num_conformers"] = slots
    return out


def derive_cache(src_dir: Path, dst_root: Path, data: str, split: str, n: int, link_mode: str) -> Path:
    src_cache_dir = src_dir / data / split
    dst_cache_dir = dst_root / data / split
    dst_cache_dir.mkdir(parents=True, exist_ok=True)

    drug3d_path = src_cache_dir / "drug3d_features.pt"
    if not drug3d_path.exists():
        raise FileNotFoundError(drug3d_path)

    drug3d_cache = torch.load(drug3d_path, map_location="cpu", weights_only=True)
    if not isinstance(drug3d_cache, dict):
        raise TypeError(f"Expected dict cache at {drug3d_path}, got {type(drug3d_cache)}")

    slots = None
    for entry in drug3d_cache.values():
        if isinstance(entry, dict) and torch.is_tensor(entry.get("feature")):
            slots = int(entry["feature"].size(0))
            break
    if slots is None:
        raise ValueError("Could not infer conformer slot count from drug3d cache.")
    if n < 1 or n > slots:
        raise ValueError(f"n must be in [1, {slots}], got {n}")

    derived = {key: crop_entry(value, n, slots) for key, value in drug3d_cache.items()}
    torch.save(derived, dst_cache_dir / "drug3d_features.pt")

    for name in ("smiles_features.pt", "protein_features.pt"):
        link_or_copy(src_cache_dir / name, dst_cache_dir / name, link_mode)

    meta_path = src_cache_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["source_cache_dir"] = str(src_cache_dir)
    meta["derived_cache_dir"] = str(dst_cache_dir)
    meta["effective_valid_conformers"] = n
    meta.setdefault("drug3d_feature", {})
    meta["drug3d_feature"]["slot_conformers"] = slots
    meta["drug3d_feature"]["effective_valid_conformers"] = n
    meta["drug3d_feature"]["derivation"] = f"keep first {n} conformer slots, zero out the rest"
    meta["drug3d_cache"] = str((dst_cache_dir / "drug3d_features.pt").resolve())
    meta["smiles_cache"] = str((dst_cache_dir / "smiles_features.pt").resolve())
    meta["protein_cache"] = str((dst_cache_dir / "protein_features.pt").resolve())
    (dst_cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return dst_cache_dir


def main():
    parser = argparse.ArgumentParser(description="Derive n-effective-conformer caches from an existing 8-slot cache.")
    parser.add_argument("--src_root", default="cache/features")
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--data", default="datasets/bindingdb")
    parser.add_argument("--split", default="random")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--link_mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    args = parser.parse_args()

    dst = derive_cache(
        src_dir=Path(args.src_root),
        dst_root=Path(args.dst_root),
        data=args.data,
        split=args.split,
        n=args.n,
        link_mode=args.link_mode,
    )
    print(f"derived n={args.n} cache: {dst}")


if __name__ == "__main__":
    main()
