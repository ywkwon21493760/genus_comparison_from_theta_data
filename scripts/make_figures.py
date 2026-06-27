#!/usr/bin/env python3
r"""
Generate manuscript-format figures from archived summary CSV files.

This script preserves the figure file names used by the current manuscript:
  figures/baseline_ood_acc_by_preprocessing.pdf
  figures/neural_ood_acc_by_preprocessing.pdf
  figures/baseline_id_acc_by_preprocessing.pdf
  figures/baseline_ood_auroc_by_preprocessing.pdf
  figures/neural_id_acc_by_preprocessing.pdf
  figures/neural_ood_auroc_by_preprocessing.pdf

Each figure is a three-panel plot for raw, log1p, and z-score preprocessing.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

BASELINE_MODELS = ["CosThr", "L1Thr", "L2Thr", "LogReg"]
NEURAL_MODELS = ["DecisionTree", "RandomForest", "MLP", "Transformer"]
NORM_ORDER = ["raw", "log1p", "zscore"]
NORM_TITLE = {"raw": "raw", "log1p": "log1p", "zscore": "z-score"}

BASELINE_MODEL_DISPLAY = {
    "Baseline_CosThresh": "CosThr",
    "Baseline_L1Thresh": "L1Thr",
    "Baseline_L2Thresh": "L2Thr",
    "Baseline_LogReg": "LogReg",
    "CosThresh": "CosThr",
    "CosThr": "CosThr",
    "L1Thresh": "L1Thr",
    "L1Thr": "L1Thr",
    "L2Thresh": "L2Thr",
    "L2Thr": "L2Thr",
    "LogReg": "LogReg",
}

NEURAL_MODEL_DISPLAY = {
    "DecisionTree": "DecisionTree",
    "RandomForest": "RandomForest",
    "MLP": "MLP",
    "Transformer": "Transformer",
    "Neural_DecisionTree": "DecisionTree",
    "Neural_RandomForest": "RandomForest",
    "Neural_MLP": "MLP",
    "Neural_Transformer": "Transformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript-format figures.")
    parser.add_argument(
        "--baseline-summary",
        default="results/baseline/results_baselines_summary_mean_std.csv",
        help="Baseline summary CSV.",
    )
    parser.add_argument(
        "--neural-summary",
        default="results/nonlinear_neural/results_neural_summary_mean_std.csv",
        help="Nonlinear/neural summary CSV.",
    )
    parser.add_argument("--outdir", default="figures", help="Output directory for figures.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png"],
        help="Output formats to write.",
    )
    return parser.parse_args()


def canonical_norm(x: str) -> str:
    x = str(x).strip()
    if x in {"z-score", "z_score"}:
        return "zscore"
    return x


def read_rows(path: str | Path, model_map: Dict[str, str]) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"P", "norm", "model", "id_acc_mean", "ood_acc_mean", "ood_auroc_mean"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            r = dict(row)
            r["P"] = int(float(r["P"]))
            r["norm"] = canonical_norm(r["norm"])
            r["model"] = model_map.get(r["model"].strip(), r["model"].strip())
            for key in ["id_acc_mean", "ood_acc_mean", "ood_auroc_mean"]:
                r[key] = float(r[key])
            rows.append(r)
    return rows


def subset(rows: Iterable[dict], norm: str, model: str) -> List[dict]:
    out = [r for r in rows if r["norm"] == norm and r["model"] == model]
    return sorted(out, key=lambda r: r["P"])


def make_three_panel(
    rows: List[dict],
    models: List[str],
    metric: str,
    ylabel: str,
    outdir: Path,
    basename: str,
    formats: List[str],
    ylim: tuple[float, float] | None = None,
    chance_line: bool = False,
) -> None:
    # Figure size and layout are chosen to match the compact manuscript plots.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), sharey=True)

    handles = []
    labels = []
    for ax, norm in zip(axes, NORM_ORDER):
        for model in models:
            data = subset(rows, norm, model)
            if not data:
                continue
            xs = [r["P"] for r in data]
            ys = [r[metric] for r in data]
            line, = ax.plot(xs, ys, marker="o", markersize=1.8, linewidth=0.85, label=model)
            if norm == NORM_ORDER[0]:
                handles.append(line)
                labels.append(model)
        ax.set_title(NORM_TITLE[norm], fontsize=6.5)
        ax.set_xlabel(r"$P$", fontsize=6.5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.7)
        ax.tick_params(axis="both", labelsize=5.8)
        ax.set_xlim(0, 610)
        if chance_line:
            ax.axhline(0.5, linestyle="--", linewidth=0.8, alpha=0.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel, fontsize=6.5)
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.80, wspace=0.06)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = outdir / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    baseline_rows = read_rows(args.baseline_summary, BASELINE_MODEL_DISPLAY)
    neural_rows = read_rows(args.neural_summary, NEURAL_MODEL_DISPLAY)

    make_three_panel(
        baseline_rows,
        BASELINE_MODELS,
        metric="ood_acc_mean",
        ylabel="OOD accuracy",
        outdir=outdir,
        basename="baseline_ood_acc_by_preprocessing",
        formats=args.formats,
        ylim=(0.45, 0.85),
        chance_line=True,
    )
    make_three_panel(
        neural_rows,
        NEURAL_MODELS,
        metric="ood_acc_mean",
        ylabel="OOD accuracy",
        outdir=outdir,
        basename="neural_ood_acc_by_preprocessing",
        formats=args.formats,
        ylim=(0.48, 1.0),
        chance_line=True,
    )
    make_three_panel(
        baseline_rows,
        BASELINE_MODELS,
        metric="id_acc_mean",
        ylabel="ID accuracy",
        outdir=outdir,
        basename="baseline_id_acc_by_preprocessing",
        formats=args.formats,
        ylim=(0.48, 0.95),
        chance_line=False,
    )
    make_three_panel(
        baseline_rows,
        BASELINE_MODELS,
        metric="ood_auroc_mean",
        ylabel="OOD AUROC",
        outdir=outdir,
        basename="baseline_ood_auroc_by_preprocessing",
        formats=args.formats,
        ylim=(0.0, 1.0),
        chance_line=False,
    )
    make_three_panel(
        neural_rows,
        NEURAL_MODELS,
        metric="id_acc_mean",
        ylabel="ID accuracy",
        outdir=outdir,
        basename="neural_id_acc_by_preprocessing",
        formats=args.formats,
        ylim=(0.55, 1.0),
        chance_line=False,
    )
    make_three_panel(
        neural_rows,
        NEURAL_MODELS,
        metric="ood_auroc_mean",
        ylabel="OOD AUROC",
        outdir=outdir,
        basename="neural_ood_auroc_by_preprocessing",
        formats=args.formats,
        ylim=(0.45, 0.85),
        chance_line=False,
    )
    print(f"Wrote manuscript-format figures to {outdir}")


if __name__ == "__main__":
    main()
