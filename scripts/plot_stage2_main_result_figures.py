#!/usr/bin/env python3
"""Generate main-result CSV files and figures for TAMR-DTI."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_theme import (
    BASELINE,
    BASELINE_DARK,
    GRID,
    METRIC_COLORS,
    TAMR,
    TEXT,
    apply_paper_theme,
    style_axes as theme_style_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    REPO_ROOT
    / "TAMR-DTI__Target-Aware_Modulation_and_Mamba-based_Representation_Refinement_for_Drug-Target_Interaction_Prediction"
)
OUTPUT_DIRS = [
    REPO_ROOT / "outputs" / "figures",
    PAPER_DIR / "figures",
]

DATASETS = ["BindingDB", "BioSNAP", "Human", "C. elegans"]
METHODS = [
    "SVM",
    "RF",
    "KNN",
    "LR",
    "GraphDTA",
    "DrugBAN",
    "IIFDTI",
    "TransformerCPI",
    "MolTrans",
    "BINDTI",
    "LDM-DTI",
]
METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("acc_05", "Acc@0.5"),
    ("f1_05", "F1@0.5"),
]
CI_METRICS = [
    ("auprc", "AUPRC"),
    ("acc_05", "Acc@0.5"),
    ("f1_05", "F1@0.5"),
]

RUNS = [
    ("BindingDB", 42, "outputs/result/stage2-main-01-full-tamr-dti-n8-bindingdb-seed42/best_metrics.txt"),
    ("BindingDB", 43, "outputs/result/stage2-main-02-full-tamr-dti-n8-bindingdb-seed43/best_metrics.txt"),
    ("BindingDB", 44, "outputs/result/stage2-main-03-full-tamr-dti-n8-bindingdb-seed44/best_metrics.txt"),
    ("BindingDB", 45, "outputs/result/stage2-main-04-full-tamr-dti-n8-bindingdb-seed45/best_metrics.txt"),
    ("BindingDB", 47, "outputs/result/stage2-main-21-full-tamr-dti-n8-bindingdb-seed47/best_metrics.txt"),
    ("BioSNAP", 42, "outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt"),
    ("BioSNAP", 43, "outputs/result/stage2-main-07-full-tamr-dti-n8-biosnap-seed43/best_metrics.txt"),
    ("BioSNAP", 44, "outputs/result/stage2-main-08-full-tamr-dti-n8-biosnap-seed44/best_metrics.txt"),
    ("BioSNAP", 45, "outputs/result/stage2-main-09-full-tamr-dti-n8-biosnap-seed45/best_metrics.txt"),
    ("BioSNAP", 47, "outputs/result/stage2-main-22-full-tamr-dti-n8-biosnap-seed47/best_metrics.txt"),
    ("Human", 42, "swanlog/stage2-main-human-random3-seed42/run-20260527_154004-guab5lkhi7ju5widsp9fw/backup.swanlab"),
    ("Human", 43, "swanlog/stage2-main-human-random3-seed43/run-20260528_102410-5zdcd34477vr92l74nzrr/backup.swanlab"),
    ("Human", 44, "swanlog/stage2-main-human-random3-seed44/run-20260527_155639-7afuxyo57gwsl6gu7bmky/backup.swanlab"),
    ("Human", 45, "swanlog/stage2-main-human-random3-seed45/run-20260527_160711-2awztks8al5u6v3ocsz6b/backup.swanlab"),
    ("Human", 47, "swanlog/stage2-main-human-random3-seed47/run-20260527_161626-lqf02lnqgo3d09zksvaqz/backup.swanlab"),
    ("C. elegans", 42, "outputs/result/stage2-main-16-full-tamr-dti-n8-celegans-seed42/best_metrics.txt"),
    ("C. elegans", 43, "outputs/result/stage2-main-17-full-tamr-dti-n8-celegans-seed43/best_metrics.txt"),
    ("C. elegans", 44, "outputs/result/stage2-main-18-full-tamr-dti-n8-celegans-seed44/best_metrics.txt"),
    ("C. elegans", 45, "outputs/result/stage2-main-19-full-tamr-dti-n8-celegans-seed45/best_metrics.txt"),
    ("C. elegans", 47, "outputs/result/stage2-main-24-full-tamr-dti-n8-celegans-seed47/best_metrics.txt"),
]

# AUROC/AUPRC values are from the reported comparison adopted in the manuscript.
# Acc@0.5/F1@0.5 values are transcribed from Supplementary Tables S1-S4 in
# /home/lsw/lv/TAMR-DTI/LDM-DTI.docx.
REPORTED_BASELINES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "BindingDB": {
        "SVM": {"auroc": (0.904, 0.002), "auprc": (0.865, 0.001), "acc_05": (0.824, 0.001), "f1_05": (0.785, 0.001)},
        "RF": {"auroc": (0.942, 0.001), "auprc": (0.923, 0.001), "acc_05": (0.871, 0.001), "f1_05": (0.844, 0.002)},
        "KNN": {"auroc": (0.895, 0.001), "auprc": (0.865, 0.002), "acc_05": (0.798, 0.002), "f1_05": (0.797, 0.002)},
        "LR": {"auroc": (0.916, 0.003), "auprc": (0.884, 0.002), "acc_05": (0.849, 0.002), "f1_05": (0.842, 0.003)},
        "GraphDTA": {"auroc": (0.944, 0.004), "auprc": (0.925, 0.006), "acc_05": (0.874, 0.001), "f1_05": (0.880, 0.005)},
        "DrugBAN": {"auroc": (0.952, 0.001), "auprc": (0.936, 0.001), "acc_05": (0.901, 0.003), "f1_05": (0.901, 0.001)},
        "IIFDTI": {"auroc": (0.924, 0.002), "auprc": (0.901, 0.004), "acc_05": (0.833, 0.003), "f1_05": (0.810, 0.003)},
        "TransformerCPI": {"auroc": (0.895, 0.002), "auprc": (0.861, 0.003), "acc_05": (0.812, 0.003), "f1_05": (0.780, 0.003)},
        "MolTrans": {"auroc": (0.935, 0.003), "auprc": (0.901, 0.004), "acc_05": (0.864, 0.008), "f1_05": (0.837, 0.003)},
        "BINDTI": {"auroc": (0.956, 0.001), "auprc": (0.942, 0.002), "acc_05": (0.899, 0.001), "f1_05": (0.901, 0.003)},
        "LDM-DTI": {"auroc": (0.960, 0.002), "auprc": (0.945, 0.002), "acc_05": (0.904, 0.002), "f1_05": (0.902, 0.001)},
    },
    "BioSNAP": {
        "SVM": {"auroc": (0.819, 0.045), "auprc": (0.839, 0.038), "acc_05": (0.750, 0.008), "f1_05": (0.827, 0.053)},
        "RF": {"auroc": (0.857, 0.009), "auprc": (0.872, 0.006), "acc_05": (0.793, 0.001), "f1_05": (0.787, 0.001)},
        "KNN": {"auroc": (0.842, 0.003), "auprc": (0.805, 0.003), "acc_05": (0.777, 0.003), "f1_05": (0.773, 0.001)},
        "LR": {"auroc": (0.821, 0.004), "auprc": (0.796, 0.002), "acc_05": (0.752, 0.002), "f1_05": (0.749, 0.002)},
        "GraphDTA": {"auroc": (0.871, 0.004), "auprc": (0.870, 0.006), "acc_05": (0.800, 0.005), "f1_05": (0.807, 0.005)},
        "DrugBAN": {"auroc": (0.902, 0.001), "auprc": (0.905, 0.002), "acc_05": (0.836, 0.004), "f1_05": (0.838, 0.003)},
        "IIFDTI": {"auroc": (0.895, 0.003), "auprc": (0.898, 0.005), "acc_05": (0.798, 0.003), "f1_05": (0.816, 0.002)},
        "TransformerCPI": {"auroc": (0.843, 0.008), "auprc": (0.859, 0.007), "acc_05": (0.771, 0.004), "f1_05": (0.767, 0.006)},
        "MolTrans": {"auroc": (0.879, 0.006), "auprc": (0.882, 0.006), "acc_05": (0.801, 0.001), "f1_05": (0.802, 0.007)},
        "BINDTI": {"auroc": (0.895, 0.003), "auprc": (0.894, 0.003), "acc_05": (0.833, 0.002), "f1_05": (0.801, 0.003)},
        "LDM-DTI": {"auroc": (0.908, 0.003), "auprc": (0.906, 0.003), "acc_05": (0.844, 0.005), "f1_05": (0.839, 0.004)},
    },
    "Human": {
        "SVM": {"auroc": (0.913, 0.012), "auprc": (0.905, 0.001), "acc_05": (0.838, 0.001), "f1_05": (0.811, 0.001)},
        "RF": {"auroc": (0.939, 0.009), "auprc": (0.927, 0.001), "acc_05": (0.866, 0.006), "f1_05": (0.848, 0.005)},
        "KNN": {"auroc": (0.915, 0.002), "auprc": (0.902, 0.002), "acc_05": (0.884, 0.003), "f1_05": (0.892, 0.003)},
        "LR": {"auroc": (0.895, 0.003), "auprc": (0.862, 0.001), "acc_05": (0.839, 0.004), "f1_05": (0.838, 0.002)},
        "GraphDTA": {"auroc": (0.965, 0.003), "auprc": (0.955, 0.003), "acc_05": (0.908, 0.001), "f1_05": (0.907, 0.008)},
        "DrugBAN": {"auroc": (0.979, 0.001), "auprc": (0.969, 0.005), "acc_05": (0.940, 0.013), "f1_05": (0.940, 0.004)},
        "IIFDTI": {"auroc": (0.977, 0.003), "auprc": (0.967, 0.027), "acc_05": (0.928, 0.012), "f1_05": (0.918, 0.012)},
        "TransformerCPI": {"auroc": (0.956, 0.005), "auprc": (0.943, 0.002), "acc_05": (0.901, 0.002), "f1_05": (0.890, 0.002)},
        "MolTrans": {"auroc": (0.979, 0.003), "auprc": (0.975, 0.002), "acc_05": (0.937, 0.017), "f1_05": (0.932, 0.005)},
        "BINDTI": {"auroc": (0.980, 0.001), "auprc": (0.975, 0.004), "acc_05": (0.940, 0.005), "f1_05": (0.935, 0.003)},
        "LDM-DTI": {"auroc": (0.981, 0.001), "auprc": (0.976, 0.002), "acc_05": (0.946, 0.003), "f1_05": (0.936, 0.004)},
    },
    "C. elegans": {
        "SVM": {"auroc": (0.925, 0.009), "auprc": (0.907, 0.002), "acc_05": (0.792, 0.002), "f1_05": (0.789, 0.002)},
        "RF": {"auroc": (0.935, 0.009), "auprc": (0.931, 0.001), "acc_05": (0.856, 0.001), "f1_05": (0.852, 0.003)},
        "KNN": {"auroc": (0.912, 0.003), "auprc": (0.896, 0.002), "acc_05": (0.862, 0.003), "f1_05": (0.861, 0.003)},
        "LR": {"auroc": (0.892, 0.002), "auprc": (0.883, 0.003), "acc_05": (0.847, 0.002), "f1_05": (0.852, 0.001)},
        "GraphDTA": {"auroc": (0.959, 0.004), "auprc": (0.951, 0.002), "acc_05": (0.912, 0.004), "f1_05": (0.910, 0.001)},
        "DrugBAN": {"auroc": (0.981, 0.002), "auprc": (0.978, 0.002), "acc_05": (0.942, 0.003), "f1_05": (0.940, 0.004)},
        "IIFDTI": {"auroc": (0.984, 0.002), "auprc": (0.979, 0.002), "acc_05": (0.925, 0.002), "f1_05": (0.920, 0.002)},
        "TransformerCPI": {"auroc": (0.975, 0.003), "auprc": (0.975, 0.002), "acc_05": (0.901, 0.001), "f1_05": (0.896, 0.001)},
        "MolTrans": {"auroc": (0.982, 0.002), "auprc": (0.979, 0.002), "acc_05": (0.916, 0.002), "f1_05": (0.914, 0.002)},
        "BINDTI": {"auroc": (0.985, 0.002), "auprc": (0.984, 0.002), "acc_05": (0.942, 0.001), "f1_05": (0.938, 0.002)},
        "LDM-DTI": {"auroc": (0.988, 0.002), "auprc": (0.986, 0.002), "acc_05": (0.960, 0.002), "f1_05": (0.956, 0.002)},
    },
}


def normalize_metric_keys(metrics: dict) -> dict[str, float]:
    normalized = {key: float(value) for key, value in metrics.items()}
    if "auc" in normalized and "auroc" not in normalized:
        normalized["auroc"] = normalized["auc"]
    if "aupr" in normalized and "auprc" not in normalized:
        normalized["auprc"] = normalized["aupr"]
    return normalized


def load_best_metrics(path: Path) -> dict[str, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_test="):
            return normalize_metric_keys(ast.literal_eval(line.split("=", 1)[1]))
    raise ValueError(f"best_test not found in {path}")


def iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        yield obj
        pos = start + end


def load_swanlog_best_metrics(path: Path) -> dict[str, float]:
    text = path.read_bytes().decode("utf-8", errors="ignore")
    by_step: dict[int, dict[str, float]] = {}
    for obj in iter_json_objects(text):
        if obj.get("model_type") != "Scalar":
            continue
        data = obj.get("data", {})
        key = data.get("key")
        step = data.get("step")
        metric = data.get("metric", {})
        if key is None or step is None or not isinstance(metric, dict):
            continue
        try:
            value = float(metric["data"])
        except (KeyError, TypeError, ValueError):
            continue
        by_step.setdefault(int(step), {})[str(key)] = value

    if not by_step:
        raise ValueError(f"scalar metrics not found in {path}")

    best_step, _ = max(
        ((step, metrics["val/auroc"]) for step, metrics in by_step.items() if "val/auroc" in metrics),
        key=lambda item: item[1],
    )
    metrics = by_step[best_step]
    return normalize_metric_keys(
        {
            "auroc": metrics["test/auroc"],
            "auprc": metrics["test/auprc"],
            "acc_05": metrics["test/acc_05"],
            "f1_05": metrics["test/f1_05"],
        }
    )


def load_existing_seed_rows() -> dict[tuple[str, int], dict[str, float | int | str]]:
    path = PAPER_DIR / "figures" / "main_seed_metrics.csv"
    if not path.exists():
        return {}

    rows: dict[tuple[str, int], dict[str, float | int | str]] = {}
    with path.open(encoding="utf-8", newline="") as rf:
        for row in csv.DictReader(rf):
            parsed: dict[str, float | int | str] = {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "source": row["source"],
            }
            for key, _ in METRICS:
                parsed[key] = float(row[key])
            rows[(str(parsed["dataset"]), int(parsed["seed"]))] = parsed
    return rows


def load_seed_rows() -> list[dict[str, float | int | str]]:
    rows = []
    existing_rows = load_existing_seed_rows()
    for dataset, seed, rel_path in RUNS:
        path = REPO_ROOT / rel_path
        if path.suffix == ".swanlab":
            metrics = load_swanlog_best_metrics(path)
        elif path.exists():
            metrics = load_best_metrics(path)
        elif (dataset, seed) in existing_rows:
            rows.append(existing_rows[(dataset, seed)])
            continue
        else:
            raise FileNotFoundError(f"metrics source not found: {path}")
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "seed": seed,
            "source": rel_path,
        }
        for key, _ in METRICS:
            row[key] = metrics[key]
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    summary = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        out: dict[str, float | str] = {"dataset": dataset}
        for key, _ in METRICS:
            values = np.array([float(row[key]) for row in subset], dtype=float)
            out[f"{key}_mean"] = float(values.mean())
            out[f"{key}_std"] = float(values.std(ddof=1))
            ci = 2.776 * float(values.std(ddof=1)) / np.sqrt(len(values))
            out[f"{key}_ci_low"] = float(values.mean() - ci)
            out[f"{key}_ci_high"] = float(values.mean() + ci)
        summary.append(out)
    return summary


def baseline_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dataset in DATASETS:
        for method in METHODS:
            row: dict[str, float | str] = {"dataset": dataset, "method": method}
            for key, _ in METRICS:
                mean, std = REPORTED_BASELINES[dataset][method][key]
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
            rows.append(row)
    return rows


def combined_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows = baseline_rows()
    for item in summary:
        row: dict[str, float | str] = {"dataset": item["dataset"], "method": "TAMR-DTI"}
        for key, _ in METRICS:
            row[f"{key}_mean"] = float(item[f"{key}_mean"])
            row[f"{key}_std"] = float(item[f"{key}_std"])
        rows.append(row)
    return rows


def best_reported_delta_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    by_dataset = {str(row["dataset"]): row for row in summary}
    for dataset in DATASETS:
        for key, label in METRICS:
            candidates = [
                (method, REPORTED_BASELINES[dataset][method][key][0])
                for method in METHODS
            ]
            best_method, best_value = max(candidates, key=lambda item: item[1])
            tamr_value = float(by_dataset[dataset][f"{key}_mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "metric": label,
                    "best_reported_method": best_method,
                    "best_reported_value": best_value,
                    "tamr_dti_value": tamr_value,
                    "absolute_delta": tamr_value - best_value,
                }
            )
    return rows


def strongest_prior_delta_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    by_dataset = {str(row["dataset"]): row for row in summary}
    for dataset in DATASETS:
        method_scores = {
            method: np.mean([REPORTED_BASELINES[dataset][method][key][0] for key, _ in METRICS])
            for method in METHODS
        }
        strongest_method = max(method_scores, key=method_scores.get)
        row: dict[str, float | str] = {
            "dataset": dataset,
            "strongest_prior_method": strongest_method,
            "selection_mean_score": float(method_scores[strongest_method]),
        }
        for key, label in METRICS:
            prior_value = REPORTED_BASELINES[dataset][strongest_method][key][0]
            tamr_value = float(by_dataset[dataset][f"{key}_mean"])
            row[f"{label}_prior"] = prior_value
            row[f"{label}_tamr"] = tamr_value
            row[f"{label}_delta"] = tamr_value - prior_value
        rows.append(row)
    return rows


def ci_comparison_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    summary_by_dataset = {str(row["dataset"]): row for row in summary}
    for dataset in DATASETS:
        for method in METHODS:
            for key, label in CI_METRICS:
                mean, std = REPORTED_BASELINES[dataset][method][key]
                ci = 2.776 * std / np.sqrt(5)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": label,
                        "source": "reported baseline mean/std from LDM-DTI.docx",
                        "mean": mean,
                        "ci_low": mean - ci,
                        "ci_high": mean + ci,
                    }
                )
        tamr = summary_by_dataset[dataset]
        for key, label in CI_METRICS:
            rows.append(
                {
                    "dataset": dataset,
                    "method": "TAMR-DTI",
                    "metric": label,
                    "source": "TAMR-DTI five-seed runs",
                    "mean": float(tamr[f"{key}_mean"]),
                    "ci_low": float(tamr[f"{key}_ci_low"]),
                    "ci_high": float(tamr[f"{key}_ci_high"]),
                }
            )
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax) -> None:
    theme_style_axes(ax, grid_axis="y", labelsize=7.5)


def save_all(fig, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(summary: list[dict[str, float | str]], output_prefix: Path) -> None:
    summary_by_dataset = {str(row["dataset"]): row for row in summary}
    fig, axes = plt.subplots(4, 4, figsize=(8.4, 7.8), sharex=False, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.07, top=0.90, hspace=0.43, wspace=0.30)
    rng = np.random.default_rng(7)
    baseline_color = BASELINE
    best_color = BASELINE_DARK
    tamr_color = TAMR

    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, (key, label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            baseline_values = np.array(
                [REPORTED_BASELINES[dataset][method][key][0] for method in METHODS],
                dtype=float,
            )
            jitter = rng.uniform(-0.08, 0.08, size=len(baseline_values))
            ax.scatter(
                np.zeros(len(baseline_values)) + jitter,
                baseline_values,
                s=15,
                color=baseline_color,
                alpha=0.48,
                linewidth=0,
                label="Reported baselines" if (row_idx, col_idx) == (0, 0) else None,
            )
            best_idx = int(np.argmax(baseline_values))
            ax.scatter(
                [0],
                [baseline_values[best_idx]],
                s=42,
                marker="D",
                color=best_color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
                label="Best reported" if (row_idx, col_idx) == (0, 0) else None,
            )
            tamr_mean = float(summary_by_dataset[dataset][f"{key}_mean"])
            tamr_std = float(summary_by_dataset[dataset][f"{key}_std"])
            ax.errorbar(
                [0.44],
                [tamr_mean],
                yerr=[tamr_std],
                fmt="*",
                color=tamr_color,
                ecolor=tamr_color,
                elinewidth=1.15,
                capsize=2.8,
                markeredgecolor="white",
                markeredgewidth=0.65,
                markersize=8.6,
                zorder=5,
                label="TAMR-DTI" if (row_idx, col_idx) == (0, 0) else None,
            )
            values = np.concatenate([baseline_values, [tamr_mean]])
            pad = max(0.012, float(values.max() - values.min()) * 0.22)
            ax.set_ylim(max(0.70, float(values.min()) - pad), min(1.005, float(values.max()) + pad))
            ax.set_xlim(-0.28, 0.68)
            ax.set_xticks([0, 0.44])
            ax.set_xticklabels(["Reported", "TAMR"], rotation=20, ha="right")
            if row_idx == 0:
                ax.set_title(label, fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(dataset, fontsize=9.5)
            style_axes(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    save_all(fig, output_prefix)


def plot_ranking_radar(summary: list[dict[str, float | str]], output_prefix: Path) -> None:
    summary_by_dataset = {str(row["dataset"]): row for row in summary}
    angles = np.linspace(0, 2 * np.pi, len(DATASETS), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    labels = DATASETS

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for ax, (key, title) in zip(axes, [("auroc", "AUROC"), ("auprc", "AUPRC")]):
        tamr = np.array([float(summary_by_dataset[dataset][f"{key}_mean"]) for dataset in DATASETS])
        best = np.array([max(REPORTED_BASELINES[dataset][method][key][0] for method in METHODS) for dataset in DATASETS])
        tamr_closed = np.concatenate([tamr, [tamr[0]]])
        best_closed = np.concatenate([best, [best[0]]])

        ax.plot(angles_closed, best_closed, color=BASELINE_DARK, linewidth=1.35, marker="o", markersize=3.3, label="Best reported")
        ax.fill(angles_closed, best_closed, color=BASELINE_DARK, alpha=0.07)
        ax.plot(angles_closed, tamr_closed, color=TAMR, linewidth=1.9, marker="s", markersize=3.8, label="TAMR-DTI")
        ax.fill(angles_closed, tamr_closed, color=TAMR, alpha=0.12)
        lower = 0.88 if key == "auroc" else 0.84
        if key == "auprc":
            lower = 0.78
        ax.set_ylim(lower, 1.0)
        ax.set_yticks(np.linspace(lower, 1.0, 4))
        ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(lower, 1.0, 4)], fontsize=7)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, pad=12)
        ax.grid(color=GRID, linewidth=0.75, alpha=0.9)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.08), ncol=2, frameon=False, fontsize=8)
    save_all(fig, output_prefix)


def plot_seed_stability(rows: list[dict[str, float | int | str]], output_prefix: Path) -> None:
    colors = [METRIC_COLORS[label] for _, label in METRICS]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), constrained_layout=True)
    for ax, (metric_idx, (key, label)) in zip(axes.flat, enumerate(METRICS)):
        metric_min = 1.0
        metric_max = 0.0
        for dataset_idx, dataset in enumerate(DATASETS):
            values = np.array(
                [float(row[key]) for row in rows if row["dataset"] == dataset],
                dtype=float,
            )
            metric_min = min(metric_min, float(values.min()))
            metric_max = max(metric_max, float(values.max()))
            jitter = rng.uniform(-0.07, 0.07, size=len(values))
            ax.scatter(
                np.full(len(values), dataset_idx) + jitter,
                values,
                s=24,
                color=colors[metric_idx],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=3,
            )
            mean = values.mean()
            std = values.std(ddof=1)
            ax.errorbar(
                dataset_idx,
                mean,
                yerr=std,
                fmt="D",
                color=TEXT,
                ecolor=TEXT,
                elinewidth=1.0,
                capsize=3,
                markersize=4.2,
                zorder=4,
            )

        pad = max(0.006, (metric_max - metric_min) * 0.25)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(len(DATASETS)))
        ax.set_xticklabels(DATASETS, rotation=20, ha="right")
        ax.set_ylim(max(0.0, metric_min - pad), min(1.005, metric_max + pad))
        style_axes(ax)

    axes[0, 0].set_ylabel("Test score", fontsize=10)
    axes[1, 0].set_ylabel("Test score", fontsize=10)
    save_all(fig, output_prefix)


def plot_ci_intervals(summary: list[dict[str, float | str]], output_prefix: Path) -> None:
    colors = [METRIC_COLORS[label] for _, label in METRICS]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), constrained_layout=True)
    for ax, (metric_idx, (key, label)) in zip(axes.flat, enumerate(METRICS)):
        means = np.array([float(row[f"{key}_mean"]) for row in summary])
        lows = np.array([float(row[f"{key}_ci_low"]) for row in summary])
        highs = np.array([float(row[f"{key}_ci_high"]) for row in summary])
        x = np.arange(len(DATASETS))
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="o",
            color=colors[metric_idx],
            ecolor=colors[metric_idx],
            elinewidth=1.1,
            capsize=3,
            markersize=4.2,
        )
        pad = max(0.006, float(highs.max() - lows.min()) * 0.25)
        ax.set_ylim(max(0.0, float(lows.min()) - pad), min(1.005, float(highs.max()) + pad))
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, rotation=20, ha="right")
        style_axes(ax)

    axes[0, 0].set_ylabel("Mean and 95% CI", fontsize=10)
    axes[1, 0].set_ylabel("Mean and 95% CI", fontsize=10)
    save_all(fig, output_prefix)


def plot_all_method_ci_comparison(rows: list[dict[str, float | str]], output_prefix: Path) -> None:
    methods = METHODS + ["TAMR-DTI"]
    y_positions = np.arange(len(methods))
    fig, axes = plt.subplots(len(DATASETS), len(CI_METRICS), figsize=(8.5, 8.6), constrained_layout=True)

    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, (_, label) in enumerate(CI_METRICS):
            ax = axes[row_idx, col_idx]
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["metric"] == label
            ]
            by_method = {str(row["method"]): row for row in subset}
            lows = []
            highs = []
            for method_idx, method in enumerate(methods):
                row = by_method[method]
                mean = float(row["mean"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])
                lows.append(low)
                highs.append(high)
                color = TAMR if method == "TAMR-DTI" else BASELINE
                marker = "D" if method == "TAMR-DTI" else "o"
                ax.errorbar(
                    mean,
                    method_idx,
                    xerr=[[mean - low], [high - mean]],
                    fmt=marker,
                    color=color,
                    ecolor=color,
                    elinewidth=1.05 if method == "TAMR-DTI" else 0.75,
                    capsize=2.4 if method == "TAMR-DTI" else 1.8,
                    markersize=4.4 if method == "TAMR-DTI" else 2.5,
                    alpha=0.98 if method == "TAMR-DTI" else 0.55,
                )
            low = max(0.70, min(lows) - 0.015)
            high = min(1.005, max(highs) + 0.015)
            ax.set_xlim(low, high)
            ax.set_ylim(-0.75, len(methods) - 0.25)
            ax.invert_yaxis()
            if row_idx == 0:
                ax.set_title(label, fontsize=9.5)
            if col_idx == 0:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(methods, fontsize=7.0)
                ax.set_ylabel(dataset, fontsize=9.0)
            else:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([])
            ax.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.tick_params(axis="x", labelsize=7)

    save_all(fig, output_prefix)


def main() -> None:
    apply_paper_theme(plt)

    rows = load_seed_rows()
    summary = summarize(rows)
    reported_rows = baseline_rows()
    all_rows = combined_rows(summary)
    best_delta_rows = best_reported_delta_rows(summary)
    strongest_delta_rows = strongest_prior_delta_rows(summary)
    ci_rows = ci_comparison_rows(summary)

    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(rows, output_dir / "main_seed_metrics.csv")
        save_csv(summary, output_dir / "main_results_summary.csv")
        save_csv(reported_rows, output_dir / "reported_baseline_metrics.csv")
        save_csv(all_rows, output_dir / "main_all_method_metrics.csv")
        save_csv(best_delta_rows, output_dir / "main_delta_vs_best_reported.csv")
        save_csv(strongest_delta_rows, output_dir / "main_delta_vs_strongest_prior.csv")
        save_csv(ci_rows, output_dir / "reported_and_tamr_ci_metrics.csv")
        plot_main_comparison(summary, output_dir / "main_results_comparison")
        plot_ranking_radar(summary, output_dir / "main_ranking_radar")
        plot_seed_stability(rows, output_dir / "main_seed_stability")
        plot_ci_intervals(summary, output_dir / "main_ci_intervals")
        plot_all_method_ci_comparison(ci_rows, output_dir / "all_method_ci_comparison")

    print("Main-result summary:")
    for row in summary:
        metrics = " ".join(
            f"{label}={float(row[f'{key}_mean']):.4f}±{float(row[f'{key}_std']):.4f}"
            for key, label in METRICS
        )
        print(f"- {row['dataset']}: {metrics}")
    print("Saved figures to:")
    for output_dir in OUTPUT_DIRS:
        print(f"- {output_dir}")


if __name__ == "__main__":
    main()
