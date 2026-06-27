#!/usr/bin/env python3
r"""
Generate manuscript-format LaTeX tables from archived summary CSV files.

This script is designed to preserve the table file names and LaTeX style used
in the current manuscript.

Inputs:
  --baseline-summary results/baseline/results_baselines_summary_mean_std.csv
  --neural-summary   results/nonlinear_neural/results_neural_summary_mean_std.csv

Outputs:
  tables/baseline_compact_summary_table.tex
  tables/neural_compact_summary_table.tex
  tables/baseline_full_results_longtable.tex
  tables/neural_full_results_longtable_bottomcaption_no_underline.tex

The compact baseline table can either be \input into the manuscript or copied
into the current inline Table 3 location.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SELECTED_P = [10, 40, 100, 200, 300, 600]
BASELINE_ORDER = ["CosThr", "L1Thr", "L2Thr", "LogReg"]
NEURAL_ORDER = ["DecisionTree", "RandomForest", "MLP", "Transformer"]
NORM_ORDER = ["raw", "log1p", "zscore"]
NORM_DISPLAY = {"raw": "raw", "log1p": "log1p", "zscore": "z-score"}

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
    parser = argparse.ArgumentParser(description="Generate manuscript-format LaTeX tables.")
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
    parser.add_argument("--outdir", default="tables", help="Output directory for LaTeX tables.")
    return parser.parse_args()


def read_rows(path: str | Path, model_map: Dict[str, str]) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"P", "norm", "model", "id_acc_mean", "id_acc_std", "ood_acc_mean", "ood_acc_std", "ood_auroc_mean", "ood_auroc_std"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            r = dict(row)
            r["P"] = int(float(r["P"]))
            r["norm"] = canonical_norm(r["norm"])
            r["model"] = model_map.get(r["model"].strip(), r["model"].strip())
            for key in ["id_acc_mean", "id_acc_std", "ood_acc_mean", "ood_acc_std", "ood_auroc_mean", "ood_auroc_std"]:
                r[key] = float(r[key])
            rows.append(r)
    return rows


def canonical_norm(x: str) -> str:
    x = str(x).strip()
    if x in {"z-score", "z_score"}:
        return "zscore"
    return x


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def pm(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def row_key(row: dict) -> Tuple[int, str, str]:
    return int(row["P"]), row["norm"], row["model"]


def index_rows(rows: Iterable[dict]) -> Dict[Tuple[int, str, str], dict]:
    return {row_key(r): r for r in rows}


def best_by_p_model(rows: List[dict], P: int, model: str, metric: str = "ood_acc_mean") -> dict:
    candidates = [r for r in rows if r["P"] == P and r["model"] == model]
    if not candidates:
        raise ValueError(f"No rows for P={P}, model={model}")
    return max(candidates, key=lambda r: r[metric])


def best_by_p(rows: List[dict], P: int, models: List[str], metric: str = "ood_acc_mean") -> dict:
    candidates = [r for r in rows if r["P"] == P and r["model"] in models]
    if not candidates:
        raise ValueError(f"No rows for P={P}, models={models}")
    return max(candidates, key=lambda r: r[metric])


def write_baseline_compact(rows: List[dict], outdir: Path) -> None:
    path = outdir / "baseline_compact_summary_table.tex"
    lines: List[str] = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\hline\hline",
        r"\(P\) & CosThr & L1Thr & L2Thr & LogReg & Best baseline \\",
        r"\hline",
    ]
    for P in SELECTED_P:
        vals = {}
        for model in BASELINE_ORDER:
            vals[model] = best_by_p_model(rows, P, model)["ood_acc_mean"]
        best_model = max(BASELINE_ORDER, key=lambda m: vals[m])
        lines.append(
            f"{P} & {fmt4(vals['CosThr'])} & {fmt4(vals['L1Thr'])} & {fmt4(vals['L2Thr'])} & {fmt4(vals['LogReg'])} & {best_model} \\\\"
        )
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\caption{Compact baseline summary. Entries are best mean OOD accuracies over preprocessing choices for each model and truncation length. Full baseline results over all \(P\), preprocessing choices, and metrics are reported in Appendix~\ref{app:full-baseline-results}.}",
        r"\label{tab:baseline-main}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_neural_compact(baseline_rows: List[dict], neural_rows: List[dict], outdir: Path) -> None:
    path = outdir / "neural_compact_summary_table.tex"
    lines: List[str] = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrr}",
        r"\hline\hline",
        r"\(P\) & Best baseline OOD & Best nonlinear/neural OOD & Model / preproc. & Gap \\",
        r"\hline",
    ]
    for P in SELECTED_P:
        b = best_by_p(baseline_rows, P, BASELINE_ORDER)
        n = best_by_p(neural_rows, P, NEURAL_ORDER)
        gap = n["ood_acc_mean"] - b["ood_acc_mean"]
        model_preproc = f"{n['model']} / {NORM_DISPLAY[n['norm']]}"
        lines.append(
            f"{P} & {fmt4(b['ood_acc_mean'])} & {fmt4(n['ood_acc_mean'])} & {model_preproc} & {gap:+.4f} \\\\"
        )
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\caption{Compact nonlinear/neural summary. Entries are best mean OOD accuracies over preprocessing choices. The gap is relative to the best baseline at the same truncation length. Full nonlinear/neural results are reported in Appendix~\ref{app:full-neural-results}.}",
        r"\label{tab:neural-main}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def longtable_lines(rows: List[dict], models: List[str], caption: str, label: str) -> List[str]:
    sort_index = {model: i for i, model in enumerate(models)}
    rows_sorted = sorted(rows, key=lambda r: (r["P"], NORM_ORDER.index(r["norm"]), sort_index.get(r["model"], 999)))
    lines: List[str] = []
    lines += [
        r"\begin{longtable}{rllccc}",
        r"\hline\hline",
        r"\(P\) & Preproc. & Model & ID Acc. & OOD Acc. & OOD AUROC \\",
        r"\hline",
        r"\endfirsthead",
        r"\multicolumn{6}{c}{\tablename~\thetable{} -- continued from previous page}\\",
        r"\hline\hline",
        r"\(P\) & Preproc. & Model & ID Acc. & OOD Acc. & OOD AUROC \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\multicolumn{6}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\hline\hline",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\endlastfoot",
    ]
    for r in rows_sorted:
        lines.append(
            f"{r['P']} & {NORM_DISPLAY[r['norm']]} & {r['model']} & "
            f"{pm(r['id_acc_mean'], r['id_acc_std'])} & "
            f"{pm(r['ood_acc_mean'], r['ood_acc_std'])} & "
            f"{pm(r['ood_auroc_mean'], r['ood_auroc_std'])} \\\\"
        )
    lines += [r"\end{longtable}", ""]
    return lines


def write_full_tables(baseline_rows: List[dict], neural_rows: List[dict], outdir: Path) -> None:
    baseline_caption = (
        r"Full baseline results over all truncation lengths, preprocessing choices, and baseline models. "
        r"Entries are mean $\pm$ standard deviation over five random seeds."
    )
    neural_caption = (
        r"Full nonlinear and neural model results over all truncation lengths, preprocessing choices, and models. "
        r"Entries are mean $\pm$ standard deviation over five random seeds. DecisionTree and RandomForest are tree-based nonlinear baselines; "
        r"MLP is trained on pair features; Transformer treats each pair as a two-channel length-\(P\) sequence."
    )
    (outdir / "baseline_full_results_longtable.tex").write_text(
        "\n".join(longtable_lines(baseline_rows, BASELINE_ORDER, baseline_caption, "tab:baseline-full")),
        encoding="utf-8",
    )
    (outdir / "neural_full_results_longtable_bottomcaption_no_underline.tex").write_text(
        "\n".join(longtable_lines(neural_rows, NEURAL_ORDER, neural_caption, "tab:neural-full")),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_rows(args.baseline_summary, BASELINE_MODEL_DISPLAY)
    neural_rows = read_rows(args.neural_summary, NEURAL_MODEL_DISPLAY)
    write_baseline_compact(baseline_rows, outdir)
    write_neural_compact(baseline_rows, neural_rows, outdir)
    write_full_tables(baseline_rows, neural_rows, outdir)
    print(f"Wrote manuscript-format tables to {outdir}")


if __name__ == "__main__":
    main()
