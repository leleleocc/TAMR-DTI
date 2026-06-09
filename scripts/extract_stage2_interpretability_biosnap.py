#!/usr/bin/env python3
"""Extract and plot BioSNAP interpretability signals for TAMR-DTI."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_DIR = (
    REPO_ROOT
    / "TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction"
)
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/interpretability/biosnap_seed42"

from src.config.configs import get_cfg_defaults  # noqa: E402
from src.core.utils import graph_collate_func, set_seed  # noqa: E402
from src.data.dataloader import DTIDataset  # noqa: E402
from src.models.models import BINDTI, binary_cross_entropy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/biosnap", help="dataset path under data/")
    parser.add_argument("--split", default="random", help="dataset split name")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/experiments/stage2_main_06_full_tamr_dti_n8_biosnap_seed42.yaml"),
        help="experiment config yaml",
    )
    parser.add_argument("--cache_root", default=str(REPO_ROOT / "cache/features"), help="feature cache root")
    parser.add_argument("--checkpoint_dir", default=str(DEFAULT_RUN_DIR), help="DeepSpeed checkpoint directory")
    parser.add_argument("--checkpoint_tag", default=None, help="checkpoint tag; defaults to best_epoch in best_metrics.txt")
    parser.add_argument(
        "--fp32_state_dict",
        default=str(DEFAULT_OUTPUT_DIR / "fp32_state_dict.pt"),
        help="cached consolidated fp32 state_dict path",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="analysis output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full test set")
    parser.add_argument("--threshold", type=float, default=None, help="classification threshold; defaults to best_metrics")
    parser.add_argument("--allow_cpu", action="store_true", help="allow CPU execution for non-Mamba debug runs")
    return parser.parse_args()


def load_best_metadata(run_dir: Path) -> tuple[str | None, float | None]:
    best_metrics = run_dir / "best_metrics.txt"
    if not best_metrics.exists():
        return None, None

    best_epoch = None
    threshold = None
    for line in best_metrics.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_epoch="):
            best_epoch = int(line.split("=", 1)[1])
        elif line.startswith("best_test="):
            metrics = ast.literal_eval(line.split("=", 1)[1])
            threshold = float(metrics.get("threshold", 0.5))

    tag = f"epoch_{best_epoch:04d}" if best_epoch is not None else None
    return tag, threshold


def normalize_state_dict(raw_state: dict) -> dict:
    state = raw_state
    if isinstance(raw_state, dict):
        for key in ("module", "model", "state_dict"):
            value = raw_state.get(key)
            if isinstance(value, dict):
                state = value
                break

    normalized = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_fp32_state_dict(checkpoint_dir: Path, tag: str, cache_path: Path | None) -> dict:
    if cache_path is not None and cache_path.exists():
        return normalize_state_dict(torch.load(cache_path, map_location="cpu"))

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_dir), tag=tag)
    state_dict = normalize_state_dict(state_dict)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, cache_path)
    return state_dict


def move_batch_to_device(batch, device: torch.device):
    feature_vectors, feature, coor, conf_mask, energy, bg_d, v_p, protein_mask, y = batch
    return (
        feature_vectors.to(device, non_blocking=True),
        feature.to(device, non_blocking=True),
        coor.to(device, non_blocking=True),
        conf_mask.to(device, non_blocking=True),
        energy.to(device, non_blocking=True),
        bg_d.to(device),
        v_p.to(device, non_blocking=True),
        protein_mask.to(device, non_blocking=True),
        y.float().to(device, non_blocking=True),
    )


def register_film_hooks(model, buffers: dict):
    handles = []
    protein_extractor = model.protein_extractor

    for layer_idx, film_layer in enumerate(protein_extractor.film_layers):
        def make_hook(idx):
            def hook(_module, _inputs, output):
                gamma_raw, beta_raw = output.detach().chunk(2, dim=-1)
                gamma = protein_extractor.film_scale * torch.tanh(gamma_raw)
                beta = protein_extractor.film_scale * beta_raw
                buffers[idx] = {
                    "gamma_norm": gamma.norm(dim=-1).detach().cpu(),
                    "beta_norm": beta.norm(dim=-1).detach().cpu(),
                    "strength": (gamma.norm(dim=-1) + beta.norm(dim=-1)).detach().cpu(),
                }

            return hook

        handles.append(film_layer.register_forward_hook(make_hook(layer_idx)))
    return handles


def register_mamba_hook(model, buffer: dict):
    cross = getattr(model, "cross_intention", None)
    protein_mamba = getattr(cross, "protein_mamba", None)
    if protein_mamba is None:
        return []

    def hook(_module, inputs, output):
        protein_input = inputs[0].detach()
        protein_output = output.detach()
        buffer["input"] = protein_input
        buffer["output"] = protein_output

    return [protein_mamba.register_forward_hook(hook)]


def classify_case(label: int, pred_label: int) -> str:
    if label == 1 and pred_label == 1:
        return "TP"
    if label == 0 and pred_label == 0:
        return "TN"
    if label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def entropy(weights: np.ndarray, mask: np.ndarray) -> float:
    valid = weights[mask.astype(bool)]
    valid = valid[valid > 0]
    if valid.size == 0:
        return 0.0
    return float(-(valid * np.log(valid + 1e-12)).sum())


def normalized_entropy(weights: np.ndarray, mask: np.ndarray) -> float:
    valid_count = int(mask.astype(bool).sum())
    if valid_count <= 1:
        return 0.0
    return entropy(weights, mask) / math.log(valid_count)


def collect_interpretability(model, loader, df_test, device, threshold, max_samples=0) -> tuple[list[dict], np.ndarray]:
    rows = []
    mamba_profiles = []
    sample_offset = 0
    film_buffers: dict[int, dict] = {}
    mamba_buffer: dict[str, torch.Tensor] = {}
    handles = register_film_hooks(model, film_buffers) + register_mamba_hook(model, mamba_buffer)

    try:
        with torch.no_grad():
            for batch in loader:
                if max_samples and sample_offset >= max_samples:
                    break

                (
                    feature_vectors,
                    feature,
                    coor,
                    conf_mask,
                    energy,
                    bg_d,
                    v_p,
                    protein_mask,
                    y,
                ) = move_batch_to_device(batch, device)

                film_buffers.clear()
                mamba_buffer.clear()
                score, aux = model(
                    feature_vectors,
                    feature,
                    coor,
                    conf_mask,
                    energy,
                    bg_d,
                    v_p,
                    protein_mask,
                    mode="eval",
                    return_aux=True,
                )
                pred, _ = binary_cross_entropy(score, y)

                pred_np = pred.detach().cpu().numpy()
                label_np = y.detach().cpu().numpy().astype(int)
                conf_weight_np = aux["conformer_weight"].detach().cpu().numpy()
                conf_mask_np = conf_mask.detach().cpu().numpy()
                protein_mask_np = protein_mask.detach().cpu().numpy()

                if "input" in mamba_buffer and "output" in mamba_buffer:
                    raw_delta = (mamba_buffer["output"] - mamba_buffer["input"]).norm(dim=-1)
                    gate_logit = getattr(model.cross_intention, "protein_mamba_gate_logit", None)
                    gate_value = float(torch.sigmoid(gate_logit.detach()).cpu()) if gate_logit is not None else 1.0
                    mamba_delta = (gate_value * raw_delta).detach().cpu().numpy()
                else:
                    gate_value = float("nan")
                    mamba_delta = np.zeros_like(protein_mask_np, dtype=np.float32)

                batch_size = int(y.size(0))
                for batch_idx in range(batch_size):
                    if max_samples and len(rows) >= max_samples:
                        break
                    row_idx = sample_offset + batch_idx
                    source = df_test.iloc[row_idx]
                    weights = conf_weight_np[batch_idx]
                    mask = conf_mask_np[batch_idx]
                    sorted_weights = np.sort(weights[mask.astype(bool)])[::-1]
                    top1 = float(sorted_weights[0]) if sorted_weights.size else 0.0
                    top2 = float(sorted_weights[1]) if sorted_weights.size > 1 else 0.0
                    pred_label = int(pred_np[batch_idx] >= threshold)
                    case_type = classify_case(int(label_np[batch_idx]), pred_label)

                    profile = mamba_delta[batch_idx]
                    valid_profile = profile[protein_mask_np[batch_idx].astype(bool)]
                    if valid_profile.size == 0:
                        valid_profile = profile
                    top_token = int(np.argmax(profile))

                    row = {
                        "row_index": row_idx,
                        "case_type": case_type,
                        "label": int(label_np[batch_idx]),
                        "pred": float(pred_np[batch_idx]),
                        "pred_label": pred_label,
                        "threshold": float(threshold),
                        "smiles": str(source["SMILES"]),
                        "protein_prefix": str(source["Protein"])[:60],
                        "conformer_entropy": entropy(weights, mask),
                        "conformer_entropy_norm": normalized_entropy(weights, mask),
                        "conformer_top1": top1,
                        "conformer_margin": top1 - top2,
                        "mamba_gate": gate_value,
                        "mamba_refine_mean": float(np.mean(valid_profile)),
                        "mamba_refine_max": float(np.max(valid_profile)),
                        "mamba_top_token": top_token,
                    }
                    for conf_idx, value in enumerate(weights, start=1):
                        row[f"conf_{conf_idx}"] = float(value)
                    for layer_idx, values in sorted(film_buffers.items()):
                        row[f"film_l{layer_idx + 1}_gamma_norm"] = float(values["gamma_norm"][batch_idx])
                        row[f"film_l{layer_idx + 1}_beta_norm"] = float(values["beta_norm"][batch_idx])
                        row[f"film_l{layer_idx + 1}_strength"] = float(values["strength"][batch_idx])

                    rows.append(row)
                    mamba_profiles.append(profile.astype(np.float32))

                sample_offset += batch_size
    finally:
        for handle in handles:
            handle.remove()

    return rows, np.stack(mamba_profiles, axis=0) if mamba_profiles else np.empty((0, 0), dtype=np.float32)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    film_cols = [col for col in df.columns if col.startswith("film_l") and col.endswith("_strength")]
    lines = []
    lines.append(f"num_samples={len(df)}")
    lines.append(f"threshold={df['threshold'].iloc[0]:.4f}")
    lines.append(f"mamba_gate={df['mamba_gate'].dropna().mean():.6f}")
    lines.append(f"conformer_entropy_norm_mean={df['conformer_entropy_norm'].mean():.6f}")
    lines.append(f"conformer_top1_mean={df['conformer_top1'].mean():.6f}")
    lines.append(f"conformer_margin_mean={df['conformer_margin'].mean():.6f}")
    lines.append(f"mamba_refine_mean={df['mamba_refine_mean'].mean():.6f}")
    lines.append(f"mamba_refine_max_mean={df['mamba_refine_max'].mean():.6f}")
    for col in film_cols:
        lines.append(f"{col}_mean={df[col].mean():.6f}")
    for case_type in ("TP", "TN", "FP", "FN"):
        subset = df[df["case_type"] == case_type]
        if subset.empty:
            continue
        lines.append(f"{case_type}_count={len(subset)}")
        lines.append(f"{case_type}_entropy_norm_mean={subset['conformer_entropy_norm'].mean():.6f}")
        lines.append(f"{case_type}_top1_mean={subset['conformer_top1'].mean():.6f}")
        lines.append(f"{case_type}_mamba_refine_mean={subset['mamba_refine_mean'].mean():.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    fp32_cache = Path(args.fp32_state_dict) if args.fp32_state_dict else None
    best_tag, best_threshold = load_best_metadata(checkpoint_dir)
    checkpoint_tag = args.checkpoint_tag or best_tag
    threshold = float(args.threshold if args.threshold is not None else (best_threshold or 0.5))
    if checkpoint_tag is None:
        raise ValueError("Could not infer checkpoint tag; pass --checkpoint_tag explicitly.")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not visible in this session. This model uses Mamba CUDA kernels; "
            "run the script on a GPU node or pass --allow_cpu only for debugging non-Mamba configs."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    set_seed(cfg.SOLVER.SEED)

    data_dir = REPO_ROOT / "data" / args.data / args.split
    cache_dir = Path(args.cache_root) / args.data / args.split
    df_test = pd.read_csv(data_dir / "test.csv")
    if args.max_samples:
        df_for_dataset = df_test.head(args.max_samples).copy()
    else:
        df_for_dataset = df_test

    dataset = DTIDataset(df_for_dataset.index.values, df_for_dataset, cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=graph_collate_func,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loading checkpoint {checkpoint_dir} tag={checkpoint_tag}")
    state_dict = load_fp32_state_dict(checkpoint_dir, checkpoint_tag, fp32_cache)
    model = BINDTI(device=device, **cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()

    rows, profiles = collect_interpretability(
        model=model,
        loader=loader,
        df_test=df_for_dataset,
        device=device,
        threshold=threshold,
        max_samples=args.max_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "biosnap_seed42_interpretability_samples.csv")
    np.save(output_dir / "biosnap_seed42_mamba_profiles.npy", profiles)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "biosnap_seed42_interpretability_samples.csv", index=False)
    write_summary(df, output_dir / "biosnap_seed42_interpretability_summary.txt")

    print("Saved interpretability outputs:")
    print(f"- {output_dir}")
    print("Next: python scripts/plot_stage2_interpretability_biosnap.py")
    print((output_dir / "biosnap_seed42_interpretability_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
