#!/usr/bin/env python3
"""
Regenerate manuscript figures from summary CSV files.

This script generates the figure files used in the manuscript:

  figures/baseline_ood_acc_by_preprocessing.pdf
  figures/neural_ood_acc_by_preprocessing.pdf
  figures/baseline_id_acc_by_preprocessing.pdf
  figures/baseline_ood_auroc_by_preprocessing.pdf
  figures/neural_id_acc_by_preprocessing.pdf
  figures/neural_ood_auroc_by_preprocessing.pdf

The plots use both line styles and markers, in addition to color, so that
curves remain distinguishable in grayscale and for readers with color-vision
deficiencies. Panels are labelled (a), (b), and (c).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl

# Avoid Type 3 fonts in PDF output where possible.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt


P_ORDER = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
           110, 120, 130, 140, 150, 200, 300, 400, 500, 600]

NORM_ORDER = ["raw", "log1p", "zscore"]
PANEL_TITLES = {
    "raw": "(a) raw",
    "log1p": "(b) log1p",
    "zscore": "(c) z-score",
}

BASELINE_MODEL_MAP = {
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

NEURAL_MODEL_MAP = {
    "DecisionTree": "DecisionTree",
    "RandomForest": "RandomForest",
    "MLP": "MLP",
    "Transformer": "Transformer",
    "Neural_DecisionTree": "DecisionTree",
    "Neural_RandomForest": "RandomForest",
    "Neural_MLP": "MLP",
    "Neural_Transformer": "Transformer",
}

NORM_MAP = {
    "raw": "raw",
    "log1p": "log1p",
    "zscore": "zscore",
    "z-score": "zscore",
    "z_score": "zscore",
}

STYLE_MAP = {
    0: {"linestyle": "-",  "marker": "o"},
    1: {"linestyle": "--", "marker": "s"},
    2: {"linestyle": "-.", "marker": "^"},
    3: {"linestyle": ":",  "marker": "D"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate manuscript figures from summary CSV files."
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results/baseline/results_baselines_summary_mean_std.csv"),
        help="Baseline summary CSV.",
    )
    parser.add_argument(
        "--neural-summary",
        type=Path,
        default=Path("results/nonlinear_neural/results_neural_summary_mean_std.csv"),
        help="Nonlinear/neural summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png"],
        help="Output formats.",
    )
    return parser.parse_args()


def read_summary(path: Path, model_map: Dict[str, str]) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "P", "norm", "model",
            "id_acc_mean", "ood_acc_mean", "ood_auroc_mean",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            model = model_map.get(row["model"].strip(), row["model"].strip())
            norm = NORM_MAP.get(row["norm"].strip(), row["norm"].strip())

            rows.append({
                "P": int(float(row["P"])),
                "norm": norm,
                "model": model,
                "id_acc_mean": float(row["id_acc_mean"]),
                "ood_acc_mean": float(row["ood_acc_mean"]),
                "ood_auroc_mean": float(row["ood_auroc_mean"]),
            })

    return rows


def series_for(
    rows: List[dict],
    *,
    norm: str,
    model: str,
    metric: str,
) -> Tuple[List[int], List[float]]:
    selected = [
        row for row in rows
        if row["norm"] == norm and row["model"] == model
    ]
    selected.sort(key=lambda r: P_ORDER.index(r["P"]) if r["P"] in P_ORDER else r["P"])

    xs = [row["P"] for row in selected]
    ys = [row[metric] for row in selected]
    return xs, ys


def plot_by_preprocessing(
    rows: List[dict],
    *,
    model_order: List[str],
    metric: str,
    ylabel: str,
    output_stem: str,
    outdir: Path,
    formats: Iterable[str],
) -> None:
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10.8, 3.2),
        sharey=True,
        constrained_layout=False,
    )

    handles = []
    labels = []

    for ax, norm in zip(axes, NORM_ORDER):
        ax.set_title(PANEL_TITLES[norm], fontsize=10)

        for idx, model in enumerate(model_order):
            xs, ys = series_for(rows, norm=norm, model=model, metric=metric)
            if not xs:
                continue

            style = STYLE_MAP[idx % len(STYLE_MAP)]
            line, = ax.plot(
                xs,
                ys,
                label=model,
                linewidth=1.7,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                **style,
            )

            if norm == NORM_ORDER[0]:
                handles.append(line)
                labels.append(model)

        ax.set_xlabel(r"Prefix length $P$")
        ax.set_xticks([10, 100, 200, 300, 400, 500, 600])
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.45)

    axes[0].set_ylabel(ylabel, fontsize=9)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(labels),
            frameon=False,
            fontsize=8.5,
            handlelength=2.6,
            columnspacing=1.2,
        )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.78, wspace=0.20)

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = outdir / f"{output_stem}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[wrote] {out_path}")

    plt.close(fig)


def main() -> None:
    args = parse_args()

    baseline_rows = read_summary(args.baseline_summary, BASELINE_MODEL_MAP)
    neural_rows = read_summary(args.neural_summary, NEURAL_MODEL_MAP)

    baseline_models = ["CosThr", "L1Thr", "L2Thr", "LogReg"]
    neural_models = ["DecisionTree", "RandomForest", "MLP", "Transformer"]

    plot_by_preprocessing(
        baseline_rows,
        model_order=baseline_models,
        metric="ood_acc_mean",
        ylabel="Out-of-distribution (OOD) accuracy",
        output_stem="baseline_ood_acc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )

    plot_by_preprocessing(
        neural_rows,
        model_order=neural_models,
        metric="ood_acc_mean",
        ylabel="Out-of-distribution (OOD) accuracy",
        output_stem="neural_ood_acc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )

    plot_by_preprocessing(
        baseline_rows,
        model_order=baseline_models,
        metric="id_acc_mean",
        ylabel="In-distribution (ID) accuracy",
        output_stem="baseline_id_acc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )

    plot_by_preprocessing(
        baseline_rows,
        model_order=baseline_models,
        metric="ood_auroc_mean",
        ylabel="OOD AUROC",
        output_stem="baseline_ood_auroc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )

    plot_by_preprocessing(
        neural_rows,
        model_order=neural_models,
        metric="id_acc_mean",
        ylabel="In-distribution (ID) accuracy",
        output_stem="neural_id_acc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )

    plot_by_preprocessing(
        neural_rows,
        model_order=neural_models,
        metric="ood_auroc_mean",
        ylabel="OOD AUROC",
        output_stem="neural_ood_auroc_by_preprocessing",
        outdir=args.outdir,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
