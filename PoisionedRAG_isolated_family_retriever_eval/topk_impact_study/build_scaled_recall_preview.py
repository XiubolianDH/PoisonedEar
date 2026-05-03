from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visualization-only preview with scaled recall.")
    parser.add_argument("--input-json", required=True, help="Path to comparison_summary_topk_impact.json")
    parser.add_argument("--output-dir", required=True, help="Directory for preview outputs")
    parser.add_argument("--scale", type=float, default=10.0, help="Scale factor for recall visualization")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_preview(rows: list[dict], scale: float) -> dict:
    preview_rows = []
    for row in rows:
        new_row = dict(row)
        scaled_recall = row["Recall@k"] * scale
        new_row["Scaled_Recall@k_for_plot"] = scaled_recall
        new_row["Scaled_Recall_scale_factor"] = scale
        preview_rows.append(new_row)

    return {
        "visualization_only": True,
        "warning": "Scaled recall is for figure readability only. Original Recall@k remains the official metric.",
        "scale_factor": scale,
        "rows": preview_rows,
    }


def build_svg(rows: list[dict], scale: float) -> str:
    width, height = 980, 560
    margin_left = 80
    margin_right = 40
    margin_top = 80
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x0 = margin_left
    y0 = margin_top

    k_values = [row["top_k"] for row in rows]

    def x_pos(k: int) -> float:
        if len(k_values) == 1:
            return x0 + plot_w / 2
        return x0 + (k_values.index(k) / (len(k_values) - 1)) * plot_w

    # Three plotted series: Overall_ASR, Precision@k, Scaled Recall@k
    series = [
        ("Overall_ASR", "#c2410c"),
        ("Precision@k", "#ea580c"),
        (f"Scaled Recall@k x{int(scale)}", "#fb923c"),
    ]

    def y_pos(v: float) -> float:
        vmax = 1.0
        return y0 + plot_h - (v / vmax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf5"/>',
        '<text x="36" y="40" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#7c2d12">Top-k Impact Preview (Visualization-Only Scaled Recall)</text>',
        '<text x="36" y="64" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#9a3412">Original metrics unchanged. The recall curve is rescaled only for visual readability.</text>',
        f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#fed7aa" rx="16"/>',
    ]

    # Grid and y-axis labels
    for i in range(6):
        frac = i / 5
        yy = y0 + plot_h - frac * plot_h
        parts.append(f'<line x1="{x0}" y1="{yy}" x2="{x0 + plot_w}" y2="{yy}" stroke="#ffedd5"/>')
        parts.append(
            f'<text x="{x0 - 10}" y="{yy + 4}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#9a3412">{frac:.1f}</text>'
        )

    # X-axis labels
    for k in k_values:
        xx = x_pos(k)
        parts.append(f'<line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y0 + plot_h}" stroke="#fff7ed"/>')
        parts.append(
            f'<text x="{xx}" y="{y0 + plot_h + 24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700" fill="#9a3412">k={k}</text>'
        )

    for label, color in series:
        if label.startswith("Scaled Recall"):
            vals = [min(1.0, row["Recall@k"] * scale) for row in rows]
        else:
            vals = [row[label] for row in rows]
        pts = [(x_pos(k), y_pos(v)) for k, v in zip(k_values, vals)]
        points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3.5" points="{points_str}"/>')
        for (x, y), v in zip(pts, vals):
            parts.append(f'<circle cx="{x}" cy="{y}" r="4.8" fill="{color}"/>')
            parts.append(
                f'<text x="{x}" y="{y - 10}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#7c2d12">{v:.3f}</text>'
            )

    # Legend
    legend_x = 150
    legend_y = height - 34
    spacing = 220
    for idx, (label, color) in enumerate(series):
        lx = legend_x + idx * spacing
        parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 32}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{lx + 40}" y="{legend_y + 4}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">{label}</text>')

    parts.append(
        f'<text x="36" y="{height - 58}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#9a3412">Note: only the recall curve is scaled by x{int(scale)} in this preview. F1 is intentionally left unchanged.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_json)
    preview = build_preview(rows, args.scale)

    preview_json = output_dir / "scaled_recall_preview.json"
    preview_svg = output_dir / "scaled_recall_preview.svg"

    preview_json.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    preview_svg.write_text(build_svg(rows, args.scale), encoding="utf-8")

    print(preview_json)
    print(preview_svg)


if __name__ == "__main__":
    main()
