import csv
import json
import pathlib
import statistics


ROOT = pathlib.Path("/alloc/blt")
OUT_DIR = ROOT / "blt_patchsize_v2_graphs"
MAX_STEP = 15564
RUNS = {
    "PatchLen Sin v2": ROOT / "runs/h100_fineweb_1p7b_bs8_patchlen_sin_v2",
    "Baseline Ngram": ROOT / "runs/h100_fineweb_1p7b_bs8_ngram",
}
COLORS = {
    "PatchLen Sin v2": "#d9480f",
    "Baseline Ngram": "#1c7ed6",
}


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prep_run(run_dir: pathlib.Path) -> dict:
    train = load_jsonl(run_dir / "metrics.jsonl")
    val = load_jsonl(run_dir / "metrics.validation.jsonl")
    flops_per_token = statistics.median(
        row["speed/FLOPS"] / row["speed/wps"] for row in train if row["speed/wps"] > 0
    )
    tokens_per_step = statistics.median(
        row["optim/total_tokens"] / row["global_step"]
        for row in train
        if row["global_step"] > 0
    )
    points = []
    for row in val:
        metric = next(v for k, v in row.items() if k.endswith(".arrow"))
        step = row["global_step"]
        if step > MAX_STEP:
            continue
        total_tokens = step * tokens_per_step
        total_flops = total_tokens * flops_per_token
        points.append(
            {
                "step": step,
                "tokens": total_tokens,
                "flops": total_flops,
                "bpb": metric["bpb"],
                "ppl": metric["ppl"],
            }
        )

    return {
        "flops_per_token": flops_per_token,
        "tokens_per_step": tokens_per_step,
        "points": points,
    }


def format_x(value: float, scale: float, decimals: int = 2) -> str:
    return f"{value / scale:.{decimals}f}"


def write_svg(data: dict, x_key: str, title: str, subtitle: str, x_label: str, x_scale: float, out_name: str):
    all_points = [pt for run in data.values() for pt in run["points"]]
    min_x = min(pt[x_key] for pt in all_points)
    max_x = max(pt[x_key] for pt in all_points)
    min_y = min(pt["bpb"] for pt in all_points)
    max_y = max(pt["bpb"] for pt in all_points)

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
    svg.append(f'<text x="{ml}" y="28" class="title">{title}</text>')
    svg.append(f'<text x="{ml}" y="48" class="subtitle">{subtitle}</text>')

    for i in range(6):
        y = min_y + (max_y - min_y) * i / 5
        yy = sy(y)
        svg.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{width - mr}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{ml - 12}" y="{yy + 4:.1f}" text-anchor="end" class="small">{y:.3f}</text>')

    for i in range(6):
        x = min_x + (max_x - min_x) * i / 5
        xx = sx(x)
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{height - mb}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{height - mb + 24}" text-anchor="middle" class="small">{format_x(x, x_scale)}</text>')

    svg.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#111827"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#111827"/>')
    svg.append(f'<text x="{(ml + width - mr) / 2:.1f}" y="{height - 18}" text-anchor="middle" class="axis">{x_label}</text>')
    mid_y = (mt + height - mb) / 2
    svg.append(f'<text x="22" y="{mid_y:.1f}" transform="rotate(-90 22 {mid_y:.1f})" text-anchor="middle" class="axis">Validation bpb</text>')

    for name, run in data.items():
        points = run["points"]
        if len(points) >= 2:
            path = " ".join((("M" if idx == 0 else "L") + f' {sx(pt[x_key]):.1f} {sy(pt["bpb"]):.1f}') for idx, pt in enumerate(points))
            svg.append(f'<path d="{path}" fill="none" stroke="{COLORS[name]}" stroke-width="3"/>')
        for pt in points:
            xx, yy = sx(pt[x_key]), sy(pt["bpb"])
            svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{COLORS[name]}"/>')
            svg.append(f'<text x="{xx + 8:.1f}" y="{yy - 8:.1f}" class="small">step {pt["step"]}</text>')

    lx, ly = width - 240, 70
    svg.append(f'<rect x="{lx}" y="{ly}" width="190" height="58" rx="8" fill="white" stroke="#d1d5db"/>')
    for idx, name in enumerate(["PatchLen Sin v2", "Baseline Ngram"]):
        yy = ly + 18 + idx * 22
        svg.append(f'<line x1="{lx + 12}" y1="{yy}" x2="{lx + 34}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>')
        svg.append(f'<circle cx="{lx + 23}" cy="{yy}" r="4" fill="{COLORS[name]}"/>')
        svg.append(f'<text x="{lx + 42}" y="{yy + 4}" class="small">{name}</text>')

    patch = data["PatchLen Sin v2"]["points"][-1]
    baseline_match = min(data["Baseline Ngram"]["points"], key=lambda pt: abs(pt["step"] - patch["step"]))
    svg.append(f'<text x="{ml}" y="{height - 50}" class="small">v2 latest checkpoint: step {patch["step"]}, bpb {patch["bpb"]:.3f}, tokens {patch["tokens"] / 1e9:.3f}B, flops {patch["flops"] / 1e15:.2f}e15</text>')
    svg.append(f'<text x="{ml}" y="{height - 32}" class="small">Closest baseline step {baseline_match["step"]}: bpb {baseline_match["bpb"]:.3f}, tokens {baseline_match["tokens"] / 1e9:.3f}B, flops {baseline_match["flops"] / 1e15:.2f}e15</text>')
    svg.append("</svg>")

    out_path = OUT_DIR / out_name
    out_path.write_text("\n".join(svg))
    return out_path


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = {
        "PatchLen Sin v2": prep_run(RUNS["PatchLen Sin v2"]),
        "Baseline Ngram": prep_run(RUNS["Baseline Ngram"]),
    }

    outs = [
        write_svg(
            data, "flops",
            "Validation BPB vs Total Training FLOPs",
            "Patch-length sinusoid v2 vs BLT baseline ngram",
            "Total training FLOPs (×10¹⁵)", 1e15,
            "val_bpb_vs_training_flops_patchlen_v2_vs_baseline.svg",
        ),
        write_svg(
            data, "tokens",
            "Validation BPB vs Total Training Tokens",
            "Patch-length sinusoid v2 vs BLT baseline ngram",
            "Total training tokens (billions)", 1e9,
            "val_bpb_vs_training_tokens_patchlen_v2_vs_baseline.svg",
        ),
        write_svg(
            data, "step",
            "Validation BPB vs Global Step",
            "Patch-length sinusoid v2 vs BLT baseline ngram",
            "Global step", 1,
            "val_bpb_vs_global_step_patchlen_v2_vs_baseline.svg",
        ),
    ]

    csv_path = OUT_DIR / "v2_checkpoints_vs_baseline.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run", "global_step", "val_bpb", "val_ppl",
                "total_tokens", "total_tokens_billions",
                "total_flops", "total_flops_e15",
                "flops_per_token", "tokens_per_step",
            ]
        )
        for name, run in data.items():
            for pt in run["points"]:
                writer.writerow(
                    [
                        name, pt["step"], pt["bpb"], pt["ppl"],
                        int(pt["tokens"]), pt["tokens"] / 1e9,
                        pt["flops"], pt["flops"] / 1e15,
                        run["flops_per_token"], run["tokens_per_step"],
                    ]
                )
    outs.append(csv_path)

    summary_path = OUT_DIR / "summary.txt"
    patch = data["PatchLen Sin v2"]["points"][-1]
    baseline_match = min(data["Baseline Ngram"]["points"], key=lambda pt: abs(pt["step"] - patch["step"]))
    summary_path.write_text(
        "\n".join(
            [
                f'PatchLen Sin v2 flops/token ~= {data["PatchLen Sin v2"]["flops_per_token"]:.2f}',
                f'Baseline Ngram flops/token ~= {data["Baseline Ngram"]["flops_per_token"]:.2f}',
                f'PatchLen Sin v2 tokens/step ~= {data["PatchLen Sin v2"]["tokens_per_step"]:.2f}',
                f'Baseline Ngram tokens/step ~= {data["Baseline Ngram"]["tokens_per_step"]:.2f}',
                f'PatchLen Sin v2 latest checkpoint: step {patch["step"]}, tokens {patch["tokens"]:.3e}, flops {patch["flops"]:.3e}, bpb {patch["bpb"]:.6f}',
                f'Closest baseline checkpoint: step {baseline_match["step"]}, tokens {baseline_match["tokens"]:.3e}, flops {baseline_match["flops"]:.3e}, bpb {baseline_match["bpb"]:.6f}',
            ]
        )
    )
    outs.append(summary_path)

    for path in outs:
        print(path)


if __name__ == "__main__":
    main()
