from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bytelatent.plotting.fit_gpt2_power_law import (
    FitResult,
    fit_power_law,
    predict_power_law,
    select_series,
    valid_fit_frame,
)
from bytelatent.plotting.validation_bpb_flops import (
    build_plot_frame,
    find_metrics_files,
)


def make_label(series: str) -> str:
    label = series.replace(" / fineweb_edu_10bt.validation.05_06_07.5m.arrow", "")
    label = label.replace(
        "distilgpt2_hf_fineweb_1p7b / validation", "distilgpt2 hf fineweb"
    )
    label = label.replace("gpt2_hf_49m_fineweb_1p7b / validation", "gpt2 hf 49m fineweb")
    return label


def create_matplotlib_plot(
    df: pd.DataFrame,
    output: Path,
    *,
    y_column: str,
    title: str,
    log_x: bool,
    power_law_predictions: pd.DataFrame | None = None,
    power_law_label: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for this script. run with: "
            "uv run --with matplotlib python -m "
            "bytelatent.plotting.validation_bpb_flops_matplotlib ..."
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for series, group in df.sort_values("train_flops").groupby("series"):
        ax.plot(
            group["train_flops"],
            group[y_column],
            marker="o",
            linewidth=2,
            markersize=4,
            label=make_label(series),
        )

    if power_law_predictions is not None and not power_law_predictions.empty:
        prediction_column = f"predicted_{y_column}"
        for series, group in power_law_predictions.groupby("series"):
            label = power_law_label or f"{make_label(series)} power law fit"
            ax.plot(
                group["train_flops"],
                group[prediction_column],
                color="black",
                linestyle="--",
                linewidth=2.4,
                label=label,
            )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("cumulative training flops")
    ax.set_ylabel(
        "validation bpb"
        if y_column == "validation_bpb"
        else "validation bpb change from first eval"
    )
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_power_law_predictions(
    df: pd.DataFrame,
    *,
    y_column: str,
    series_regex: str,
    model: str,
    floor: float | None,
    floor_min: float | None,
    floor_max: float | None,
    floor_grid_size: int,
    x_ref: float | None,
    min_points: int,
    prediction_points: int,
) -> tuple[pd.DataFrame, list[FitResult]]:
    selected = select_series(df, series_regex)
    x_min = float(df["train_flops"].min())
    x_max = float(df["train_flops"].max())
    if x_min <= 0:
        raise ValueError("train_flops must be positive for power-law predictions")

    x_grid = np.geomspace(x_min, x_max, prediction_points)
    predictions = []
    results = []
    for series, group in selected.groupby("series"):
        fit_df = valid_fit_frame(group, y_column)
        result = fit_power_law(
            series,
            fit_df,
            y_column=y_column,
            model=model,
            min_points=min_points,
            floor=floor,
            floor_min=floor_min,
            floor_max=floor_max,
            floor_grid_size=floor_grid_size,
            x_ref=x_ref,
        )
        results.append(result)
        predictions.append(
            pd.DataFrame(
                {
                    "series": series,
                    "train_flops": x_grid,
                    f"predicted_{y_column}": predict_power_law(x_grid, result),
                    "floor": result.floor,
                    "scale": result.scale,
                    "exponent": result.exponent,
                    "x_ref": result.x_ref,
                    "rmse": result.rmse,
                    "r2": result.r2,
                    "n_points": result.n_points,
                }
            )
        )

    return pd.concat(predictions, ignore_index=True), results


def print_power_law_results(results: list[FitResult], y_column: str) -> None:
    for result in results:
        print(
            "power law fit: "
            f"{result.series}: {y_column} = {result.floor:.8g} "
            f"+ {result.scale:.8g} * "
            f"(train_flops / {result.x_ref:.8g}) ** {result.exponent:.8g}; "
            f"rmse={result.rmse:.8g}; r2={result.r2:.8g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="plot validation bpb over cumulative training flops with matplotlib"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="metrics.jsonl files or run directories. defaults to runs/**/metrics.jsonl",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/validation_bpb_flops_matplotlib.png"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="optional csv path for the joined plot data",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="plot validation bpb instead of change from the first eval",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="use a linear x-axis instead of log flops",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="optional plot title",
    )
    parser.add_argument(
        "--power-law-fit-series-regex",
        default=None,
        help="optional series regex to fit and overlay as a dashed power-law baseline",
    )
    parser.add_argument(
        "--power-law-fit-label",
        default=None,
        help="optional label for the dashed power-law baseline",
    )
    parser.add_argument(
        "--power-law-model",
        choices=["offset-power", "power"],
        default="offset-power",
    )
    parser.add_argument("--power-law-floor", type=float, default=None)
    parser.add_argument("--power-law-floor-min", type=float, default=None)
    parser.add_argument("--power-law-floor-max", type=float, default=None)
    parser.add_argument("--power-law-floor-grid-size", type=int, default=1000)
    parser.add_argument("--power-law-x-ref", type=float, default=None)
    parser.add_argument("--power-law-min-points", type=int, default=3)
    parser.add_argument("--power-law-prediction-points", type=int, default=200)
    parser.add_argument(
        "--power-law-csv-output",
        type=Path,
        default=None,
        help="optional csv path for the dashed power-law baseline points",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_files = find_metrics_files(args.runs_dir, args.inputs)
    if not metrics_files:
        raise ValueError("no metrics.jsonl files found")

    frames = [build_plot_frame(path, args.runs_dir) for path in metrics_files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no validation bpb rows found next to metrics.jsonl files")

    df = pd.concat(frames, ignore_index=True)
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_output, index=False)
        print(f"wrote plot data to {args.csv_output}")

    y_column = "validation_bpb" if args.absolute else "delta_validation_bpb"
    title = args.title
    if title is None:
        title = (
            "validation bpb vs cumulative training flops"
            if args.absolute
            else "validation bpb change vs cumulative training flops"
        )

    power_law_predictions = None
    if args.power_law_fit_series_regex is not None:
        power_law_predictions, power_law_results = build_power_law_predictions(
            df,
            y_column=y_column,
            series_regex=args.power_law_fit_series_regex,
            model=args.power_law_model,
            floor=args.power_law_floor,
            floor_min=args.power_law_floor_min,
            floor_max=args.power_law_floor_max,
            floor_grid_size=args.power_law_floor_grid_size,
            x_ref=args.power_law_x_ref,
            min_points=args.power_law_min_points,
            prediction_points=args.power_law_prediction_points,
        )
        print_power_law_results(power_law_results, y_column)
        if args.power_law_csv_output is not None:
            args.power_law_csv_output.parent.mkdir(parents=True, exist_ok=True)
            power_law_predictions.to_csv(args.power_law_csv_output, index=False)
            print(f"wrote power-law baseline to {args.power_law_csv_output}")

    create_matplotlib_plot(
        df,
        args.output,
        y_column=y_column,
        title=title,
        log_x=not args.linear_x,
        power_law_predictions=power_law_predictions,
        power_law_label=args.power_law_fit_label,
    )
    print(f"wrote plot to {args.output}")


if __name__ == "__main__":
    main()
