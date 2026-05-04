import json
import pathlib
import statistics


ROOT = pathlib.Path("/alloc/blt")
OUT_DIR = ROOT / "plot_data" / "tiny_decoder"
OUTPUT_PATH = OUT_DIR / "val_bpb_vs_training_flops_345-50k_vs_baseline.svg"
MAX_STEP = 15564
TARGET_NAME = "Hash345 50k"
RUNS = {
    TARGET_NAME: ROOT / "runs/h100_fineweb_1p7b_bs8_hash345_50k",
    "Baseline Ngram": ROOT / "runs/h100_fineweb_1p7b_bs8_ngram",
}
COLORS = {
    TARGET_NAME: "#d9480f",
    "Baseline Ngram": "#1c7ed6",
}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def metric_from_validation_row(row: dict) -> dict | None:
    flat_bpb = row.get("validation/bpb")
    flat_ppl = row.get("validation/ppl")
    if flat_bpb is not None:
        return {"bpb": flat_bpb, "ppl": flat_ppl}
    for key, value in row.items():
        if key in {"global_step", "created_at"}:
            continue
        if isinstance(value, dict) and "bpb" in value:
            return value
    return None


def prep_run(run_dir: pathlib.Path, checkpoint_only: bool = False, max_step: int | None = None) -> dict:
    train = load_jsonl(run_dir / "metrics.jsonl")
    val = load_jsonl(run_dir / "metrics.validation.jsonl")
    if not train:
        raise ValueError(f"missing training metrics in {run_dir}")

    flops_per_token = statistics.median(
        row["speed/FLOPS"] / row["speed/wps"] for row in train if row["speed/wps"] > 0
    )
    tokens_per_step = statistics.median(
        row["optim/total_tokens"] / row["global_step"]
        for row in train
        if row["global_step"] > 0
    )
    checkpoint_steps = {
        int(path.name)
        for path in (run_dir / "checkpoints").iterdir()
        if path.is_dir() and path.name.isdigit()
    } if (run_dir / "checkpoints").exists() else set()

    points = []
    for row in val:
        metric = metric_from_validation_row(row)
        if metric is None:
            continue
        step = row["global_step"]
        if max_step is not None and step > max_step:
            continue
        if checkpoint_only and checkpoint_steps and step not in checkpoint_steps:
            continue
        total_tokens = step * tokens_per_step
        total_flops = total_tokens * flops_per_token
        points.append(
            {
                "step": step,
                "tokens": total_tokens,
                "flops": total_flops,
                "bpb": metric["bpb"],
                "ppl": metric.get("ppl"),
            }
        )

    points.sort(key=lambda point: point["step"])
    return {
        "flops_per_token": flops_per_token,
        "tokens_per_step": tokens_per_step,
        "points": points,
        "checkpoint_steps": sorted(checkpoint_steps),
    }


def format_x(value: float, scale: float, decimals: int = 2) -> str:
    return f"{value / scale:.{decimals}f}"


def write_svg(data: dict) -> pathlib.Path:
    baseline_points = data["Baseline Ngram"]["points"]
    target_points = data[TARGET_NAME]["points"]
    plotted_points = [pt for run in data.values() for pt in run["points"]]
    if not plotted_points:
        raise ValueError("no plotted points available")

    min_x = min(pt["flops"] for pt in plotted_points)
    max_x = max(pt["flops"] for pt in plotted_points)
    min_y = min(pt["bpb"] for pt in plotted_points)
    max_y = max(pt["bpb"] for pt in plotted_points)

    xpad = (max_x - min_x) * 0.08 if max_x > min_x else 1.0
    ypad = (max_y - min_y) * 0.20 if max_y > min_y else 0.1
    min_x -= xpad
    max_x += xpad
    min_y -= ypad
    max_y += ypad

    width, height = 1000, 620
    ml, mr, mt, mb = 90, 30, 40, 80
    pw, ph = width - ml - mr, height - mt - mb

    def sx(x):
        return ml + (x - min_x) / (max_x - min_x) * pw

    def sy(y):
        return mt + ph - (y - min_y) / (max_y - min_y) * ph

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.axis{font-size:13px}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}</style>'
    )
    svg.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    svg.append(f'<text x="{ml}" y="28" class="title">Validation BPB vs Total Training FLOPs</text>')
    svg.append(f'<text x="{ml}" y="48" class="subtitle">{TARGET_NAME} checkpoints vs initial BLT baseline ngram run</text>')

    for i in range(6):
        y = min_y + (max_y - min_y) * i / 5
        yy = sy(y)
        svg.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{width - mr}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{ml - 12}" y="{yy + 4:.1f}" text-anchor="end" class="small">{y:.3f}</text>')

    for i in range(6):
        x = min_x + (max_x - min_x) * i / 5
        xx = sx(x)
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{height - mb}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{height - mb + 24}" text-anchor="middle" class="small">{format_x(x, 1e15)}</text>')

    svg.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#111827"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#111827"/>')
    svg.append(f'<text x="{(ml + width - mr) / 2:.1f}" y="{height - 18}" text-anchor="middle" class="axis">Total training FLOPs (x10^15)</text>')
    mid_y = (mt + height - mb) / 2
    svg.append(f'<text x="22" y="{mid_y:.1f}" transform="rotate(-90 22 {mid_y:.1f})" text-anchor="middle" class="axis">Validation bpb</text>')

    for name in [TARGET_NAME, "Baseline Ngram"]:
        points = data[name]["points"]
        if len(points) >= 2:
            path = " ".join(
                (("M" if idx == 0 else "L") + f' {sx(pt["flops"]):.1f} {sy(pt["bpb"]):.1f}')
                for idx, pt in enumerate(points)
            )
            svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[name]}" stroke-width="3"/>')
        for pt in points:
            xx, yy = sx(pt["flops"]), sy(pt["bpb"])
            svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{COLORS[name]}"/>')
            svg.append(f'<text x="{xx + 8:.1f}" y="{yy - 8:.1f}" class="small">step {pt["step"]}</text>')

    lx, ly = width - 240, 70
    svg.append(f'<rect x="{lx}" y="{ly}" width="190" height="58" rx="8" fill="white" stroke="#d1d5db"/>')
    for idx, name in enumerate([TARGET_NAME, "Baseline Ngram"]):
        yy = ly + 18 + idx * 22
        svg.append(f'<line x1="{lx + 12}" y1="{yy}" x2="{lx + 34}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{lx + 23}" cy="{yy}" r="4" fill="{COLORS[name]}"/>')
        svg.append(f'<text x="{lx + 42}" y="{yy + 4}" class="small">{name}</text>')

    baseline_latest = baseline_points[-1]
    svg.append(
        f'<text x="{ml}" y="{height - 50}" class="small">Baseline latest: step {baseline_latest["step"]}, bpb {baseline_latest["bpb"]:.3f}, flops {baseline_latest["flops"] / 1e15:.2f}e15</text>'
    )
    if target_points:
        target_latest = target_points[-1]
        baseline_match = min(baseline_points, key=lambda pt: abs(pt["step"] - target_latest["step"]))
        svg.append(
            f'<text x="{ml}" y="{height - 32}" class="small">{TARGET_NAME} latest: step {target_latest["step"]}, bpb {target_latest["bpb"]:.3f}, flops {target_latest["flops"] / 1e15:.2f}e15. Closest baseline step {baseline_match["step"]}: bpb {baseline_match["bpb"]:.3f}</text>'
        )
    else:
        checkpoint_steps = data[TARGET_NAME]["checkpoint_steps"]
        checkpoint_note = (
            f'No {TARGET_NAME} validation metrics yet. Latest checkpoint step: {checkpoint_steps[-1]}'
            if checkpoint_steps
            else f"No {TARGET_NAME} checkpoints found yet."
        )
        svg.append(f'<text x="{ml}" y="{height - 32}" class="small">{checkpoint_note}</text>')
    svg.append("</svg>")

    OUTPUT_PATH.write_text("\n".join(svg))
    return OUTPUT_PATH


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        TARGET_NAME: prep_run(RUNS[TARGET_NAME], checkpoint_only=True, max_step=MAX_STEP),
        "Baseline Ngram": prep_run(RUNS["Baseline Ngram"], max_step=MAX_STEP),
    }
    out_path = write_svg(data)
    print(out_path)


if __name__ == "__main__":
    main()
