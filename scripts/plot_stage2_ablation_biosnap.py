#!/usr/bin/env python3
"""Plot BioSNAP seed42 ablation results for TAMR-DTI."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_theme import (
    ABLATION_COLORS,
    DROP_CMAP,
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

METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("acc_05", "Acc@0.5"),
    ("f1_05", "F1@0.5"),
]

RUNS = [
    {
        "label": "Full",
        "short": "Full",
        "source": REPO_ROOT / "outputs/result/stage2-main-06-full-tamr-dti-n8-biosnap-seed42/best_metrics.txt",
        "kind": "best_metrics",
    },
    {
        "label": "w/o Target-aware Conformer",
        "short": "w/o TCM",
        "source": REPO_ROOT
        / "swanlog/stage2-ablation-final-without-target-aware-conformer-module-biosnap-seed42"
        / "run-20260519_171627-ese1yh2mey1vn6hrhyt57/backup.swanlab",
        "kind": "swanlog",
    },
    {
        "label": "w/o Protein FiLM",
        "short": "w/o FiLM",
        "source": REPO_ROOT
        / "swanlog/stage2-ablation-final-without-ligand-conditioned-protein-film-biosnap-seed42"
        / "run-20260519_154218-p83ay6azqf1m0lfnj1lv1/backup.swanlab",
        "kind": "swanlog",
    },
    {
        "label": "w/o Protein Mamba",
        "short": "w/o Mamba",
        "source": REPO_ROOT
        / "swanlog/stage2-ablation-final-without-protein-mamba-refinement-biosnap-seed42"
        / "run-20260519_161048-vki3l4lxqw6a6k9pq1gvo/backup.swanlab",
        "kind": "swanlog",
    },
    {
        "label": "w/o Bidirectional Modulation",
        "short": "w/o BiMod",
        "source": REPO_ROOT
        / "swanlog/stage2-ablation-final-without-bidirectional-modulation-biosnap-seed42"
        / "run-20260519_151637-wvbneziw3kjzl3486m0i5/backup.swanlab",
        "kind": "swanlog",
    },
]


def load_best_metrics(path: Path) -> dict[str, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_test="):
            return normalize_metric_keys(ast.literal_eval(line.split("=", 1)[1]))
    raise ValueError(f"best_test not found in {path}")


def load_swanlog_best(path: Path) -> dict[str, float]:
    text = path.read_bytes().decode("utf-8", errors="ignore").replace('\\"', '"')
    matches = re.findall(r"best_test: (\{.*?\})\"", text)
    if not matches:
        raise ValueError(f"best_test not found in {path}")
    return normalize_metric_keys(ast.literal_eval(matches[-1]))


def normalize_metric_keys(metrics: dict) -> dict[str, float]:
    normalized = {key: float(value) for key, value in metrics.items()}
    if "auc" in normalized and "auroc" not in normalized:
        normalized["auroc"] = normalized["auc"]
    if "aupr" in normalized and "auprc" not in normalized:
        normalized["auprc"] = normalized["aupr"]
    if "acc" in normalized and "accuracy" not in normalized:
        normalized["accuracy"] = normalized["acc"]
    return normalized


def load_rows() -> list[dict[str, float | str]]:
    rows = []
    existing_rows = load_existing_rows()
    for run in RUNS:
        path = run["source"]
        if Path(path).exists() and run["kind"] == "best_metrics":
            metrics = load_best_metrics(path)
        elif Path(path).exists():
            metrics = load_swanlog_best(path)
        elif str(run["short"]) in existing_rows:
            rows.append(existing_rows[str(run["short"])])
            continue
        else:
            raise FileNotFoundError(f"metrics source not found: {path}")

        row = {
            "label": run["label"],
            "short": run["short"],
            "source": str(path.relative_to(REPO_ROOT)),
        }
        for key, _ in METRICS:
            row[key] = metrics[key]
        rows.append(row)
    return rows


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = ["label", "short", "source"] + [key for key, _ in METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows() -> dict[str, dict[str, float | str]]:
    path = PAPER_DIR / "figures" / "ablation_biosnap_seed42_metrics.csv"
    if not path.exists():
        return {}

    rows: dict[str, dict[str, float | str]] = {}
    with path.open(encoding="utf-8", newline="") as rf:
        for row in csv.DictReader(rf):
            parsed: dict[str, float | str] = {
                "label": row["label"],
                "short": row["short"],
                "source": row["source"],
            }
            for key, _ in METRICS:
                parsed[key] = float(row[key])
            rows[str(parsed["short"])] = parsed
    return rows


def style_axes(ax) -> None:
    theme_style_axes(ax, grid_axis="y", labelsize=9)


def plot_raw_metrics(rows: list[dict[str, float | str]], output_prefix: Path) -> None:
    labels = [str(row["short"]) for row in rows]
    metric_labels = [label for _, label in METRICS]
    values = np.array([[float(row[key]) for key, _ in METRICS] for row in rows])

    colors = ABLATION_COLORS
    hatches = ["", "///", "\\\\\\", "...", "xx"]
    x = np.arange(len(metric_labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    offsets = (np.arange(len(rows)) - (len(rows) - 1) / 2) * width
    for idx, (label, color, hatch) in enumerate(zip(labels, colors, hatches)):
        ax.bar(
            x + offsets[idx],
            values[idx],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.75,
            hatch=hatch,
        )

    ax.set_ylabel("Test score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.83, 0.945)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    style_axes(ax)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_performance_drop(rows: list[dict[str, float | str]], output_prefix: Path) -> None:
    full = rows[0]
    labels = [str(row["short"]) for row in rows]
    metric_labels = [label for _, label in METRICS]
    drops = np.array(
        [[float(full[key]) - float(row[key]) for key, _ in METRICS] for row in rows]
    )

    vmax = max(0.023, float(drops.max())) * 1.03
    fig, ax = plt.subplots(figsize=(6.9, 3.45), constrained_layout=True)
    image = ax.imshow(drops, cmap=DROP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("Ablation setting", fontsize=10)

    ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(len(labels)):
        for metric_idx in range(len(metric_labels)):
            value = drops[row_idx, metric_idx]
            color = "white" if value > vmax * 0.55 else TEXT
            ax.text(
                metric_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Drop vs. Full", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_paper_theme(plt)

    rows = load_rows()
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(rows, output_dir / "ablation_biosnap_seed42_metrics.csv")
        plot_raw_metrics(rows, output_dir / "ablation_biosnap_seed42_metrics")
        plot_performance_drop(rows, output_dir / "ablation_biosnap_seed42_drop")

    print("Ablation metrics:")
    for row in rows:
        metrics = " ".join(f"{label}={float(row[key]):.4f}" for key, label in METRICS)
        print(f"- {row['short']}: {metrics}")
    print("Saved figures to:")
    for output_dir in OUTPUT_DIRS:
        print(f"- {output_dir}")


if __name__ == "__main__":
    main()
