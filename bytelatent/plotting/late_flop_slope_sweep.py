from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from bytelatent.plotting.validation_bpb_flops import (
    build_plot_frame,
    find_metrics_files,
)


DEFAULT_K_PCTS = "5,10,15,20,25,33,50"
REQUIRED_COLUMNS = {"series", "train_flops", "validation_bpb"}


def parse_k_pcts(raw: str) -> list[float]:
    values = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        value = float(part)
        if value <= 0 or value >= 100:
            raise ValueError("k percentages must be greater than 0 and less than 100")
        values.append(value)

    if not values:
        raise ValueError("at least one k percentage is required")
    return sorted(set(values))


def validate_plot_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"input data is missing columns: {sorted(missing)}")


def load_csv_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_plot_frame(df)
    return df


def load_metrics_frame(runs_dir: Path, inputs: list[Path]) -> pd.DataFrame:
    metrics_files = find_metrics_files(runs_dir, inputs)
    if not metrics_files:
        raise ValueError("no metrics.jsonl files found")

    frames = [build_plot_frame(path, runs_dir) for path in metrics_files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no validation bpb rows found next to metrics.jsonl files")
    return pd.concat(frames, ignore_index=True)


def load_plot_frame(runs_dir: Path, inputs: list[Path]) -> pd.DataFrame:
    csv_inputs = [path for path in inputs if path.suffix == ".csv"]
    metrics_inputs = [path for path in inputs if path.suffix != ".csv"]

    frames = [load_csv_frame(path) for path in csv_inputs]
    if metrics_inputs or not inputs:
        frames.append(load_metrics_frame(runs_dir, metrics_inputs))

    if not frames:
        raise ValueError("no input data found")

    df = pd.concat(frames, ignore_index=True)
    validate_plot_frame(df)
    return df


def prepare_series_frame(group: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "series",
            "run",
            "validation_source",
            "global_step",
            "created_at",
            "train_flops",
            "validation_bpb",
        ]
        if column in group.columns
    ]
    df = group[columns].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["train_flops", "validation_bpb"]
    )
    df = df[df["train_flops"] > 0].copy()
    if df.empty:
        return df

    if "created_at" in df.columns:
        df["_created_at_sort"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["_created_at_sort"] = pd.NaT

    sort_columns = ["train_flops", "_created_at_sort"]
    if "global_step" in df.columns:
        sort_columns.append("global_step")
    df = df.sort_values(sort_columns)
    df = df.groupby("train_flops", as_index=False, sort=False).tail(1)
    return df.drop(columns=["_created_at_sort"]).sort_values("train_flops")


def slope_rows_for_series(series: str, group: pd.DataFrame, k_pcts: list[float]) -> list[dict]:
    df = prepare_series_frame(group).reset_index(drop=True)
    if len(df) < 3:
        return []

    train_flops = df["train_flops"].to_numpy(dtype=float)
    validation_bpb = df["validation_bpb"].to_numpy(dtype=float)
    final_flops = float(train_flops[-1])
    final_bpb = float(validation_bpb[-1])
    rows = []

    for k_pct in k_pcts:
        cutoff_flops = final_flops * (1 - k_pct / 100)
        window_positions = np.flatnonzero(train_flops >= cutoff_flops)
        if len(window_positions) == 0:
            continue

        first_window_index = int(window_positions[0])
        if first_window_index == 0:
            continue

        pre_window_index = first_window_index - 1
        window_df = df.iloc[first_window_index:].copy()
        tail_df = df.iloc[pre_window_index:].copy()
        tail_flops = tail_df["train_flops"].to_numpy(dtype=float)
        tail_bpb = tail_df["validation_bpb"].to_numpy(dtype=float)
        tail_delta_flops = np.diff(tail_flops)
        valid_tail_segments = tail_delta_flops > 0
        tail_pairwise_slopes = np.diff(tail_bpb)[valid_tail_segments] / tail_delta_flops[
            valid_tail_segments
        ]

        pre_to_final_flops = final_flops - float(train_flops[pre_window_index])
        pre_to_final_delta_bpb = final_bpb - float(validation_bpb[pre_window_index])
        pre_to_final_slope = pre_to_final_delta_bpb / pre_to_final_flops

        row = {
            "series": series,
            "k_pct": k_pct,
            "cutoff_train_flops": cutoff_flops,
            "total_points": len(df),
            "window_points": len(window_df),
            "tail_points": len(tail_df),
            "tail_slope_segments": len(tail_pairwise_slopes),
            "pre_window_global_step": df.iloc[pre_window_index].get("global_step", np.nan),
            "pre_window_train_flops": train_flops[pre_window_index],
            "pre_window_validation_bpb": validation_bpb[pre_window_index],
            "final_global_step": df.iloc[-1].get("global_step", np.nan),
            "final_train_flops": final_flops,
            "final_validation_bpb": final_bpb,
            "pre_to_final_delta_bpb": pre_to_final_delta_bpb,
            "pre_to_final_improvement_bpb": -pre_to_final_delta_bpb,
            "pre_to_final_slope_bpb_per_flop": pre_to_final_slope,
            "pre_to_final_improvement_bpb_per_1e18_flops": -pre_to_final_slope * 1e18,
            "mean_tail_pairwise_slope_bpb_per_flop": np.nan,
            "mean_tail_pairwise_improvement_bpb_per_1e18_flops": np.nan,
        }
        if len(tail_pairwise_slopes) > 0:
            mean_slope = float(np.mean(tail_pairwise_slopes))
            row["mean_tail_pairwise_slope_bpb_per_flop"] = mean_slope
            row["mean_tail_pairwise_improvement_bpb_per_1e18_flops"] = -mean_slope * 1e18
        rows.append(row)

    return rows


def build_sweep_frame(
    df: pd.DataFrame,
    *,
    k_pcts: list[float],
    series_regex: str | None,
) -> pd.DataFrame:
    validate_plot_frame(df)
    if series_regex is not None:
        pattern = re.compile(series_regex)
        df = df[df["series"].map(lambda series: bool(pattern.search(str(series))))]
        if df.empty:
            raise ValueError(f"no series matched {series_regex!r}")

    rows = []
    for series, group in df.groupby("series", sort=True):
        rows.extend(slope_rows_for_series(series, group, k_pcts))

    if not rows:
        raise ValueError("no sweep rows could be computed")
    return pd.DataFrame(rows).sort_values(["k_pct", "series"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sweep late-training validation bpb slopes over final training flops"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="plot csv files, metrics.jsonl files, or run directories",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/late_flop_slope_sweep.csv"),
    )
    parser.add_argument(
        "--k-pcts",
        default=DEFAULT_K_PCTS,
        help=f"comma- or space-separated k percentages. default: {DEFAULT_K_PCTS}",
    )
    parser.add_argument(
        "--series-regex",
        default=None,
        help="optional regex for selecting series to analyze",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_plot_frame(args.runs_dir, args.inputs)
    sweep = build_sweep_frame(
        df,
        k_pcts=parse_k_pcts(args.k_pcts),
        series_regex=args.series_regex,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output, index=False)
    print(f"wrote sweep to {args.output}")


if __name__ == "__main__":
    main()
