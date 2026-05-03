from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from bytelatent.plotting.validation_bpb_flops import (
    build_plot_frame,
    find_metrics_files,
)


@dataclass(frozen=True)
class FitResult:
    series: str
    model: str
    n_points: int
    floor: float
    scale: float
    exponent: float
    x_ref: float
    rmse: float
    r2: float


def valid_fit_frame(df: pd.DataFrame, y_column: str) -> pd.DataFrame:
    fit_df = df[["train_flops", y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    fit_df = fit_df[fit_df["train_flops"] > 0]
    return fit_df.sort_values("train_flops")


def predict_power_law(x: np.ndarray, result: FitResult) -> np.ndarray:
    return result.floor + result.scale * (x / result.x_ref) ** result.exponent


def score_fit(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    total = float(np.sum((y - y.mean()) ** 2))
    if total == 0:
        return rmse, float("nan")
    r2 = 1 - float(np.sum(residuals**2)) / total
    return rmse, r2


def fit_with_floor(
    series: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    floor: float,
    x_ref: float,
) -> FitResult:
    shifted_y = y - floor
    if np.any(shifted_y <= 0):
        raise ValueError(
            f"floor {floor:.6g} is not below every y value for series {series!r}"
        )

    log_x = np.log(x / x_ref)
    log_y = np.log(shifted_y)
    exponent, log_scale = np.polyfit(log_x, log_y, deg=1)
    result = FitResult(
        series=series,
        model=model,
        n_points=len(x),
        floor=float(floor),
        scale=float(np.exp(log_scale)),
        exponent=float(exponent),
        x_ref=float(x_ref),
        rmse=float("nan"),
        r2=float("nan"),
    )
    y_pred = predict_power_law(x, result)
    rmse, r2 = score_fit(y, y_pred)
    return FitResult(
        series=result.series,
        model=result.model,
        n_points=result.n_points,
        floor=result.floor,
        scale=result.scale,
        exponent=result.exponent,
        x_ref=result.x_ref,
        rmse=rmse,
        r2=r2,
    )


def default_floor_bounds(y: np.ndarray) -> tuple[float, float]:
    min_y = float(np.min(y))
    max_y = float(np.max(y))
    span = max(max_y - min_y, abs(min_y) * 1e-6, 1e-12)
    margin = max(span * 1e-3, 1e-12)
    lower = min_y - 2 * span
    upper = min_y - margin
    if min_y > 0:
        lower = max(0.0, lower)
    return lower, upper


def fit_power_law(
    series: str,
    fit_df: pd.DataFrame,
    *,
    y_column: str,
    model: str,
    min_points: int,
    floor: float | None,
    floor_min: float | None,
    floor_max: float | None,
    floor_grid_size: int,
    x_ref: float | None,
) -> FitResult:
    if len(fit_df) < min_points:
        raise ValueError(
            f"series {series!r} has {len(fit_df)} usable points; need at least {min_points}"
        )

    x = fit_df["train_flops"].to_numpy(dtype=float)
    y = fit_df[y_column].to_numpy(dtype=float)
    resolved_x_ref = float(np.median(x) if x_ref is None else x_ref)
    if resolved_x_ref <= 0:
        raise ValueError("x_ref must be positive")

    if model == "power":
        return fit_with_floor(
            series,
            x,
            y,
            model=model,
            floor=0.0,
            x_ref=resolved_x_ref,
        )

    if floor is not None:
        return fit_with_floor(
            series,
            x,
            y,
            model=model,
            floor=floor,
            x_ref=resolved_x_ref,
        )

    default_min, default_max = default_floor_bounds(y)
    resolved_floor_min = default_min if floor_min is None else floor_min
    resolved_floor_max = default_max if floor_max is None else floor_max
    if resolved_floor_min >= resolved_floor_max:
        raise ValueError("floor-min must be less than floor-max")
    if floor_grid_size < 2:
        raise ValueError("floor-grid-size must be at least 2")

    best_result: FitResult | None = None
    for candidate_floor in np.linspace(
        resolved_floor_min,
        resolved_floor_max,
        floor_grid_size,
    ):
        result = fit_with_floor(
            series,
            x,
            y,
            model=model,
            floor=float(candidate_floor),
            x_ref=resolved_x_ref,
        )
        if best_result is None or result.rmse < best_result.rmse:
            best_result = result

    if best_result is None:
        raise ValueError(f"could not fit series {series!r}")
    return best_result


def load_plot_frame(args: argparse.Namespace) -> pd.DataFrame:
    metrics_files = find_metrics_files(args.runs_dir, args.inputs)
    if not metrics_files:
        raise ValueError("no metrics.jsonl files found")

    frames = [build_plot_frame(path, args.runs_dir) for path in metrics_files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("no validation bpb rows found next to metrics.jsonl files")
    return pd.concat(frames, ignore_index=True)


def select_series(df: pd.DataFrame, series_regex: str) -> pd.DataFrame:
    pattern = re.compile(series_regex)
    selected = df[df["series"].map(lambda series: bool(pattern.search(series)))]
    if selected.empty:
        available = "\n".join(f"  {series}" for series in sorted(df["series"].unique()))
        raise ValueError(
            f"no series matched {series_regex!r}. available series:\n{available}"
        )
    return selected


def make_prediction_frame(
    result: FitResult,
    observed: pd.DataFrame,
    *,
    y_column: str,
    prediction_points: int,
) -> pd.DataFrame:
    x_min = float(observed["train_flops"].min())
    x_max = float(observed["train_flops"].max())
    x_grid = np.geomspace(x_min, x_max, prediction_points)
    y_pred = predict_power_law(x_grid, result)
    return pd.DataFrame(
        {
            "series": result.series,
            "model": result.model,
            "train_flops": x_grid,
            f"predicted_{y_column}": y_pred,
            "floor": result.floor,
            "scale": result.scale,
            "exponent": result.exponent,
            "x_ref": result.x_ref,
            "rmse": result.rmse,
            "r2": result.r2,
            "n_points": result.n_points,
        }
    )


def plot_fits(
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
    *,
    y_column: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. run with: "
            "uv run --with matplotlib python -m "
            "bytelatent.plotting.fit_gpt2_power_law ..."
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pred_column = f"predicted_{y_column}"
    for series, group in selected.sort_values("train_flops").groupby("series"):
        ax.scatter(group["train_flops"], group[y_column], s=28, label=f"{series} data")
        pred_group = predictions[predictions["series"] == series]
        ax.plot(
            pred_group["train_flops"],
            pred_group[pred_column],
            linewidth=2,
            label=f"{series} fit",
        )

    ax.set_xscale("log")
    ax.set_xlabel("cumulative training flops")
    ax.set_ylabel(y_column.replace("_", " "))
    ax.set_title("gpt2 validation bpb power-law fit")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def result_rows(results: list[FitResult]) -> list[dict[str, float | int | str]]:
    return [
        {
            "series": result.series,
            "model": result.model,
            "n_points": result.n_points,
            "floor": result.floor,
            "scale": result.scale,
            "exponent": result.exponent,
            "x_ref": result.x_ref,
            "rmse": result.rmse,
            "r2": result.r2,
        }
        for result in results
    ]


def print_results(results: list[FitResult], y_column: str) -> None:
    for result in results:
        print(f"series: {result.series}")
        print(
            "equation: "
            f"{y_column} = {result.floor:.8g} + {result.scale:.8g} "
            f"* (train_flops / {result.x_ref:.8g}) ** {result.exponent:.8g}"
        )
        print(f"points: {result.n_points}")
        print(f"rmse: {result.rmse:.8g}")
        print(f"r2: {result.r2:.8g}")
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fit a power law to gpt2 validation bpb over training flops"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="metrics.jsonl files or run directories. defaults to runs/**/metrics.jsonl",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--series-regex",
        default=r"(^|/)gpt2_hf",
        help="regex for series names to fit. use 'distilgpt2' for distilgpt2 runs",
    )
    parser.add_argument(
        "--y-column",
        choices=["validation_bpb", "delta_validation_bpb"],
        default="validation_bpb",
    )
    parser.add_argument(
        "--model",
        choices=["offset-power", "power"],
        default="offset-power",
        help="offset-power fits floor + scale * x^exponent; power fixes floor to 0",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="fixed floor for offset-power. defaults to a grid search below observed y",
    )
    parser.add_argument("--floor-min", type=float, default=None)
    parser.add_argument("--floor-max", type=float, default=None)
    parser.add_argument("--floor-grid-size", type=int, default=1000)
    parser.add_argument(
        "--x-ref",
        type=float,
        default=None,
        help="normalizing flops value. defaults to the median flops for each series",
    )
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--prediction-points", type=int, default=200)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/gpt2_power_law_fit.csv"),
        help="where to write fitted curve predictions",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="optional csv path for fit parameters",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="optional png path for data with fitted curve overlay",
    )
    parser.add_argument(
        "--list-series",
        action="store_true",
        help="print available series and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_plot_frame(args)

    if args.list_series:
        for series in sorted(df["series"].unique()):
            print(series)
        return

    selected = select_series(df, args.series_regex)
    results = []
    prediction_frames = []
    for series, group in selected.groupby("series"):
        fit_df = valid_fit_frame(group, args.y_column)
        result = fit_power_law(
            series,
            fit_df,
            y_column=args.y_column,
            model=args.model,
            min_points=args.min_points,
            floor=args.floor,
            floor_min=args.floor_min,
            floor_max=args.floor_max,
            floor_grid_size=args.floor_grid_size,
            x_ref=args.x_ref,
        )
        results.append(result)
        prediction_frames.append(
            make_prediction_frame(
                result,
                fit_df,
                y_column=args.y_column,
                prediction_points=args.prediction_points,
            )
        )

    print_results(results, args.y_column)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    print(f"wrote fitted curve to {args.output_csv}")

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows(results)).to_csv(args.summary_output, index=False)
        print(f"wrote fit summary to {args.summary_output}")

    if args.plot_output is not None:
        plot_fits(selected, predictions, args.plot_output, y_column=args.y_column)
        print(f"wrote fit plot to {args.plot_output}")


if __name__ == "__main__":
    main()
