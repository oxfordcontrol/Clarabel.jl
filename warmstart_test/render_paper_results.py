"""Render saved three-method results as vector figures and revision-marked tables."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

CASES = {
    "svm_hyperparameter_tuning": ("svm_l1", r"$\lambda$", "Tuning the regularization parameter in SVMs", "table:svm_hyperparameter_tuning"),
    "svm_robust_hyperparameter_tuning": ("svm_l2", r"$\lambda$", "Tuning the regularization parameter in robust SVMs", "table:svm_robust_hyperparameter_tuning"),
    "portfolio": ("portfolio_rebalancing", "Reoptimization index", "Mean-variance portfolio rebalancing", "table:portfolio-comparison"),
    "portfolio_efficientFrontier": ("efficient_frontier", r"$r_0$", "Efficient frontier", "table:efficient-frontier"),
}
METHODS = (("warm", "Smooth", "#2264a5", "o"),
           ("convex", "Convex", "#d97900", "^"),
           ("cold", "Cold", "#444444", "s"))


def revision(text):
    return r"\revision{" + text + "}"


def numeric(value, significant=3):
    return "--" if value is None else f"{value:.{significant}g}"


def write_table(path, data, name, x_label, caption, label):
    frontier = name == "portfolio_efficientFrontier"
    x = data["t"] if frontier else data["varVal"]
    headers = [x_label] + ([r"$f(r_0)$"] if frontier else [])
    nvalues = len(headers)
    short = len(x) <= 10
    columns = "r"*nvalues + "|rrr|rrr}"
    lines = ([r"\begin{table}[htbp]", r"\centering"] if short else [r"\begingroup"])
    lines += [r"\small", r"\setlength{\tabcolsep}{4pt}"]
    if not short:
        lines.append(r"\begin{longtable}{" + columns)
    lines += [r"\caption{" + revision(caption + ": three starting methods") + "}",
              r"\label{" + label + "}" + ("" if short else r"\\")]
    if short:
        lines.append(r"\begin{tabular}{" + columns)
    lines += [" & "*nvalues + r"\multicolumn{3}{c|}{" + revision("Iterations") +
              r"} & \multicolumn{3}{c}{" + revision("Solve time (s)") + r"}\\",
              " & ".join(revision(h) for h in headers + [m[1] for m in METHODS]*2) + r"\\",
              r"\hline"]
    if not short:
        lines.append(r"\endhead")
    for i, value in enumerate(x):
        cells = [numeric(value, 5)]
        if frontier:
            cells.append(numeric(data["ft"][i], 5))
        for method, *_ in METHODS:
            cells.append(str(data[method+"_iterations"][i])
                         if data[method+"_status"][i] == "SOLVED" else "--")
        for method, *_ in METHODS:
            cells.append(numeric(data[method+"_time"][i]))
        lines.append(" & ".join(revision(cell) for cell in cells) + r"\\")
    lines += ([r"\hline", r"\end{tabular}", r"\end{table}", ""] if short else
              [r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines))


def write_plot(directory, slug, data, name, x_label):
    x = np.array(data["t"] if name == "portfolio_efficientFrontier" else data["varVal"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.subplots_adjust(left=0.09,right=0.985,bottom=0.21,top=0.97,wspace=0.34)
    for method, label, color, marker in METHODS:
        valid = np.array([s == "SOLVED" for s in data[method+"_status"]])
        for axis, key in zip(axes, ("iterations", "time")):
            values = np.array([np.nan if v is None else v for v in data[method+"_"+key]], dtype=float)
            values[~valid] = np.nan
            axis.plot(x, values, label=label, color=color, marker=marker,
                      markersize=2.5, markevery=max(1, len(x)//20), linewidth=1.0,
                      linestyle="--" if method == "cold" else "-")
    for axis, ylabel in zip(axes, ("Iterations", "Solve time (s)")):
        axis.set_xlabel(x_label)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    if name == "portfolio_efficientFrontier":
        for axis in axes:
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[1].legend(loc="best", frameon=False, fontsize=9)
    fig.savefig(directory / (slug+"_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results/paper_comparison_results.json")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    figures, tables = args.paper / "figures", args.paper / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "pdf.fonttype": 42, "axes.labelsize": 9})
    summary = {}
    for name, (slug, x_label, caption, label) in CASES.items():
        case = data[name]
        if name == "portfolio":
            write_plot(figures, slug, case, name, x_label)
        else:
            write_table(tables / (slug+"_comparison.tex"), case, name, x_label, caption, label)
        summary[name] = {key: case[key] for key in (
            "warm_geomean_iter", "warm_geomean_time", "convex_geomean_iter",
            "convex_geomean_time", "warm_success_count", "convex_success_count",
            "cold_success_count", "max_objective_spread", "julia_version", "cpu", "blas_threads")}
        summary[name]["n"] = len(case["cold_iterations"])
    output = tables / "comparison_summary.md"
    lines = ["# Section 7.2/7.3 benchmark comparison", "",
             "Ratios are geometric means relative to cold start; smaller is better.", "",
             "| Experiment | Smooth iterations | Convex iterations | Smooth time | Convex time | Solved (Smooth / Convex / Cold) |",
             "|---|---:|---:|---:|---:|---|"]
    for name, (_, _, caption, _) in CASES.items():
        row = summary[name]
        ratios = [row[k] for k in ("warm_geomean_iter", "convex_geomean_iter",
                                  "warm_geomean_time", "convex_geomean_time")]
        counts = " / ".join(str(row[m+"_success_count"])+"/"+str(row["n"])
                            for m in ("warm", "convex", "cold"))
        lines.append("| "+caption+" | "+" | ".join(f"{r:.4f}" if r is not None else "--"
                                                   for r in ratios)+" | "+counts+" |")
    lines += ["", "Convex-combination unit-start weight: t = 0.01.",
              "Julia 1.11.4; one Julia thread and one BLAS thread; Intel Core i9-13900HX.",
              "Portfolio covariance uses the sample normalization 1/(d-1), matching the draft.", ""]
    output.write_text("\n".join(lines))
    print("Rendered the paper's three comparison tables and rebalancing figure; summary:", output)


if __name__ == "__main__":
    main()
