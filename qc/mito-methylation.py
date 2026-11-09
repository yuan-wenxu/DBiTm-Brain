#!/usr/bin/env python3
"""Plot mitochondrial CG/CH or lambda CG methylation from summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from methylation import (
    METHOD_HATCHES,
    SAMPLE_COLORS,
    atomic_write_table,
    configure_plot_style,
    draw_sample_legend,
    plt,
    save_figure_atomic,
)


CONTEXT_FIELDS = {
    "mito": {
        "CG": "host_mito_mean_methylation",
        "CA": "host_mito_mean_ca_methylation",
        "CC": "host_mito_mean_cc_methylation",
        "CT": "host_mito_mean_ct_methylation",
    },
    "lambda": {
        "CG": "lambda_mean_methylation",
    },
}
TARGET_LABELS = {"mito": "Mitochondrial", "lambda": "Lambda CG"}
SUMMARY_FIELDS = [
    "sample", "context", "mean_methylation_percent", "summary_tsv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs=2, action="append", required=True,
        metavar=("SAMPLE", "FILE"),
        help="Sample name and sample_summary.tsv; repeat in plot order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target", choices=tuple(CONTEXT_FIELDS), default="mito",
        help="mito: CG/CA/CC/CT; lambda: CG only (default: mito).",
    )
    return parser.parse_args(argv)


def read_summary_rates(path: Path, target: str) -> dict[str, float]:
    """Read source percentages without multiplying by 100."""
    if not path.is_file():
        raise FileNotFoundError(f"Sample summary not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 1:
        raise ValueError(f"Expected one sample summary row: {path}")
    fields = CONTEXT_FIELDS[target]
    missing = set(fields.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {target} columns in {path}: {sorted(missing)}")
    rates = pd.to_numeric(
        frame.iloc[0][list(fields.values())], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 100)).any():
        raise ValueError(
            f"{target} rates must be finite percentages in [0, 100]: {path}"
        )
    return dict(zip(fields, rates.tolist(), strict=True))


def draw_context_bars(
    axis: plt.Axes, frame: pd.DataFrame, names: list[str], target: str
) -> None:
    contexts = list(CONTEXT_FIELDS[target])
    positions = np.arange(len(contexts))
    width = 0.76 / len(names)
    for index, sample in enumerate(names):
        sample_rates = frame.loc[frame["sample"].eq(sample)].set_index(
            "context"
        )
        values = sample_rates.loc[contexts, "mean_methylation_percent"]
        method = "TAPS-beta" if sample.endswith("-beta") else "TAPS"
        offset = (index - (len(names) - 1) / 2) * width
        bar_positions = [index] if target == "lambda" else positions + offset
        bar_width = 0.65 if target == "lambda" else width * 0.9
        bars = axis.bar(
            bar_positions, values.to_numpy(), width=bar_width,
            color=SAMPLE_COLORS[sample], edgecolor="#333333",
            linewidth=0.4, alpha=0.92, hatch=METHOD_HATCHES[method],
        )
        if target == "lambda":
            axis.bar_label(
                bars, labels=[f"{value:.2f}%" for value in values],
                padding=-3, fontsize=5,
                bbox={
                    "facecolor": "white", "edgecolor": "none",
                    "alpha": 0.85, "pad": 0.5,
                },
            )
    for boundary in np.arange(len(contexts) - 1) + 0.5:
        axis.axvline(boundary, color="#D0D0D0", linewidth=0.7, zorder=0)
    if target == "lambda":
        axis.set_xlim(-0.5, len(names) - 0.5)
        axis.set_xticks([(len(names) - 1) / 2], contexts)
    else:
        axis.set_xlim(-0.5, len(contexts) - 0.5)
        axis.set_xticks(positions, contexts)
    axis.tick_params(axis="both", labelsize=5)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def plot_summary_rates(frame: pd.DataFrame, path: Path, target: str) -> None:
    configure_plot_style()
    names = frame["sample"].drop_duplicates().tolist()
    figure = plt.figure(figsize=(4, 2), layout="tight")
    figure.get_layout_engine().set(pad=0.3, w_pad=0.3)
    try:
        grid = figure.add_gridspec(1, 2, width_ratios=(3, 1.4))
        axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[0, 1])
        draw_context_bars(axis, frame, names, target)
        maximum = float(frame["mean_methylation_percent"].max())
        locator = MaxNLocator(nbins=5)
        upper = (
            100.0 if target == "lambda"
            else locator.tick_values(0, maximum * 1.1 or 1.0)[-1]
        )
        axis.set_ylim(0, upper)
        axis.set_ylabel("Mean methylation (%)")
        axis.yaxis.set_major_locator(locator)
        draw_sample_legend(legend_axis, names)
        figure.suptitle(f"{TARGET_LABELS[target]} methylation", fontsize=7)
        save_figure_atomic(figure, path)
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = []
        names = set()
        for sample, filename in args.input:
            if not sample.strip() or sample in names:
                raise ValueError(f"Blank or duplicate sample name: {sample!r}")
            if sample not in SAMPLE_COLORS:
                raise ValueError(f"SAMPLE_COLORS lacks an entry for: {sample}")
            names.add(sample)
            path = Path(filename).expanduser().resolve()
            rates = read_summary_rates(path, args.target)
            for context, rate in rates.items():
                rows.append({
                    "sample": sample,
                    "context": context,
                    "mean_methylation_percent": rate,
                    "summary_tsv": str(path),
                })
            print(f"{sample}: {args.target} methylation (%)={rates}", flush=True)
        output_dir = args.output_dir.expanduser().resolve()
        output_name = f"{args.target}_methylation_rate_bar.png"
        summary_name = f"{args.target}_methylation_rate_summary.tsv"
        for name in (output_name, summary_name):
            output_path = output_dir / name
            if output_path.exists():
                raise FileExistsError(f"Output already exists: {output_path}")
        frame = pd.DataFrame(rows, columns=SUMMARY_FIELDS)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_summary_rates(frame, output_dir / output_name, args.target)
        atomic_write_table(frame, output_dir / summary_name)
        print(f"[{args.target}-methylation] output-dir={output_dir}")
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"[{args.target}-methylation] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
