#!/usr/bin/env python3
"""Plot BioSNAP seed42 conformer-count comparison for TAMR-DTI."""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_theme import (
    GRID,
    METRIC_COLORS,
    MUTED_TEXT,
    TEXT,
    apply_paper_theme,
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
        "label": "n=1 target-aware",
        "short": "n=1",
        "effective_conformers": 1,
        "target_aware": True,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-n1-biosnap-seed42",
    },
    {
        "label": "n=2 target-aware",
        "short": "n=2",
        "effective_conformers": 2,
        "target_aware": True,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-n2-biosnap-seed42",
    },
    {
        "label": "n=4 target-aware",
        "short": "n=4",
        "effective_conformers": 4,
        "target_aware": True,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-n4-biosnap-seed42",
    },
    {
        "label": "n=8 target-aware",
        "short": "n=8",
        "effective_conformers": 8,
        "target_aware": True,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-n8-biosnap-seed42",
    },
    {
        "label": "n=16 target-aware",
        "short": "n=16",
        "effective_conformers": 16,
        "target_aware": True,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-n16-biosnap-seed42",
    },
    {
        "label": "n=8 uniform average",
        "short": "avg",
        "effective_conformers": 8,
        "target_aware": False,
        "source": REPO_ROOT / "swanlog/stage2-comparative-conformer-avg-biosnap-seed42",
    },
]


def normalize_metric_keys(metrics: dict) -> dict[str, float]:
    normalized = {key: float(value) for key, value in metrics.items()}
    if "auc" in normalized and "auroc" not in normalized:
        normalized["auroc"] = normalized["auc"]
    if "aupr" in normalized and "auprc" not in normalized:
        normalized["auprc"] = normalized["aupr"]
    if "acc" in normalized and "accuracy" not in normalized:
        normalized["accuracy"] = normalized["acc"]
    return normalized


def find_backup(run_root: Path) -> Path:
    backups = sorted(run_root.glob("run-*/backup.swanlab"))
    if not backups:
        raise FileNotFoundError(f"No backup.swanlab found under {run_root}")
    return backups[-1]


def load_swanlog_best(path: Path) -> tuple[int, int | None, float | None, dict[str, float]]:
    text = path.read_bytes().decode("utf-8", errors="ignore").replace('\\"', '"')
    best_epoch_matches = re.findall(r"best_epoch: ([0-9]+)", text)
    stop_epoch_matches = re.findall(r"early stopping at epoch ([0-9]+)", text)
    runtime_matches = re.findall(r"Total running time: ([0-9.]+)s", text)
    best_test_matches = re.findall(r"best_test: (\{.*?\})\"", text)
    if not best_epoch_matches or not best_test_matches:
        raise ValueError(f"Could not parse best metrics from {path}")
    best_epoch = int(best_epoch_matches[-1])
    stop_epoch = int(stop_epoch_matches[-1]) if stop_epoch_matches else None
    runtime_s = float(runtime_matches[-1]) if runtime_matches else None
    best_test = normalize_metric_keys(ast.literal_eval(best_test_matches[-1]))
    return best_epoch, stop_epoch, runtime_s, best_test


def load_rows() -> list[dict[str, float | int | str | bool | None]]:
    rows = []
    for run in RUNS:
        backup = find_backup(run["source"])
        best_epoch, stop_epoch, runtime_s, metrics = load_swanlog_best(backup)
        row: dict[str, float | int | str | bool | None] = {
            "label": run["label"],
            "short": run["short"],
            "effective_conformers": run["effective_conformers"],
            "target_aware": run["target_aware"],
            "best_epoch": best_epoch,
            "stop_epoch": stop_epoch if stop_epoch is not None else "",
            "runtime_s": runtime_s if runtime_s is not None else "",
            "threshold": metrics["threshold"],
            "source": str(backup.relative_to(REPO_ROOT)),
        }
        for key, _ in METRICS:
            row[key] = metrics[key]
        rows.append(row)
    return rows


def save_csv(rows: list[dict[str, float | int | str | bool | None]], path: Path) -> None:
    fieldnames = [
        "label",
        "short",
        "effective_conformers",
        "target_aware",
        "best_epoch",
        "stop_epoch",
        "runtime_s",
        "threshold",
        "source",
    ] + [key for key, _ in METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows() -> list[dict[str, float | int | str | bool | None]]:
    path = PAPER_DIR / "figures" / "conformer_biosnap_seed42_metrics.csv"
    if not path.exists():
        return []

    rows: list[dict[str, float | int | str | bool | None]] = []
    with path.open(encoding="utf-8", newline="") as rf:
        for row in csv.DictReader(rf):
            parsed: dict[str, float | int | str | bool | None] = {
                "label": row["label"],
                "short": row["short"],
                "effective_conformers": int(row["effective_conformers"]),
                "target_aware": row["target_aware"] == "True",
                "best_epoch": int(row["best_epoch"]),
                "stop_epoch": row["stop_epoch"],
                "runtime_s": float(row["runtime_s"]),
                "threshold": float(row["threshold"]),
                "source": row["source"],
            }
            for key, _ in METRICS:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def save_all(fig, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(
    rows: list[dict[str, float | int | str | bool | None]], output_prefix: Path
) -> None:
    variants = [str(row["short"]) for row in rows]
    metric_labels = [label for _, label in METRICS]
    values = np.array([[float(row[key]) for key, _ in METRICS] for row in rows])

    x = np.arange(len(rows))
    width = 0.18
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * width
    colors = [METRIC_COLORS[label] for _, label in METRICS]

    fig, ax = plt.subplots(figsize=(7.4, 3.55))
    for idx, ((_, metric_label), color) in enumerate(zip(METRICS, colors)):
        metric_values = values[:, idx]
        bars = ax.bar(
            x + offsets[idx],
            metric_values,
            width=width,
            label=metric_label,
            color=color,
            edgecolor="white",
            linewidth=0.55,
        )
        best_index = int(np.argmax(metric_values))
        bars[best_index].set_edgecolor(TEXT)
        bars[best_index].set_linewidth(1.15)

    ax.set_ylim(0.845, 0.94)
    ax.set_yticks(np.arange(0.85, 0.941, 0.01))
    ax.set_ylabel("Absolute test score", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{row['short']}\n{'TA' if row['target_aware'] else 'avg'}"
            for row in rows
        ],
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax.text(
        0.5,
        -0.23,
        "Bars show absolute test scores; y-axis starts at 0.845 to make small gaps visible.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED_TEXT,
    )
    save_all(fig, output_prefix)


def main() -> None:
    apply_paper_theme(plt)
    try:
        rows = load_rows()
    except FileNotFoundError:
        rows = load_existing_rows()
        if not rows:
            raise
    for output_dir in OUTPUT_DIRS:
        save_csv(rows, output_dir / "conformer_biosnap_seed42_metrics.csv")
        plot_grouped_bars(rows, output_dir / "conformer_biosnap_seed42_comparison")


if __name__ == "__main__":
    main()
