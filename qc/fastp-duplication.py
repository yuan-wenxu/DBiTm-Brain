#!/usr/bin/env python3
"""Plot fastp read duplication rates for named methylation samples."""

from __future__ import annotations

import argparse
import json
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


OUTPUT_NAME = "fastp_duplication_rate_bar.png"
SUMMARY_NAME = "fastp_duplication_rate_summary.tsv"
SUMMARY_FIELDS = [
    "sample", "duplication_rate", "duplication_percent", "fastp_json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs=2, action="append", required=True,
        metavar=("SAMPLE", "FILE"),
        help="Sample name and fastp JSON. Repeat in the desired plot order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_duplication_rate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"fastp JSON not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        rate = report["duplication"]["rate"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing duplication.rate in {path}") from error
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not np.isfinite(rate)
        or not 0 <= rate <= 1
    ):
        raise ValueError(
            f"duplication.rate must be a finite number in [0, 1]: {path}"
        )
    return float(rate)


def plot_duplication_rates(frame: pd.DataFrame, path: Path) -> None:
    configure_plot_style()
    names = frame["sample"].tolist()
    percentages = frame["duplication_percent"].to_numpy()
    figure = plt.figure(figsize=(4, 2), layout="tight")
    figure.get_layout_engine().set(pad=0.3, w_pad=0.3)
    try:
        grid = figure.add_gridspec(1, 2, width_ratios=(3, 1.4))
        axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[0, 1])
        bars = axis.bar(
            np.arange(len(names)), percentages, width=0.65,
            color=[SAMPLE_COLORS[sample] for sample in names],
            edgecolor="#333333", linewidth=0.4, alpha=0.92,
        )
        for bar, sample in zip(bars, names, strict=True):
            method = "TAPS-beta" if sample.endswith("-beta") else "TAPS"
            bar.set_hatch(METHOD_HATCHES[method])
        axis.bar_label(
            bars, labels=[f"{value:.2f}%" for value in percentages],
            padding=2, fontsize=5,
        )
        upper = max(5.0, np.ceil(percentages.max() * 1.25 / 5) * 5)
        axis.set_ylim(0, upper)
        axis.set_xlim(-0.5, len(names) - 0.5)
        axis.set_xticks([])
        axis.set_ylabel("Duplication rate (%)")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.tick_params(axis="y", labelsize=5)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        draw_sample_legend(legend_axis, names)
        figure.suptitle("Read duplication rate (fastp)", fontsize=7)
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
            rate = read_duplication_rate(path)
            rows.append({
                "sample": sample,
                "duplication_rate": rate,
                "duplication_percent": rate * 100,
                "fastp_json": str(path),
            })
            print(f"{sample}: fastp duplication rate={rate:.2%}", flush=True)
        output_dir = args.output_dir.expanduser().resolve()
        for name in (OUTPUT_NAME, SUMMARY_NAME):
            output_path = output_dir / name
            if output_path.exists():
                raise FileExistsError(f"Output already exists: {output_path}")
        frame = pd.DataFrame(rows, columns=SUMMARY_FIELDS)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_duplication_rates(frame, output_dir / OUTPUT_NAME)
        atomic_write_table(frame, output_dir / SUMMARY_NAME)
        print(f"[fastp-duplication] output-dir={output_dir}")
        return 0
    except (OSError, ValueError) as error:
        print(f"[fastp-duplication] error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
