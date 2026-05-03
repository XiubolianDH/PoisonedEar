import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_SUMMARY = REPO_ROOT / "outputs" / "results_clean_wavcaps.summary.json"
SRC_CSV = REPO_ROOT / "outputs" / "results_clean_wavcaps.csv"
OUT_DIR = REPO_ROOT / "4_dataset_overrallASR_result" / "preliminary_rag_effectiveness"


def load_summary():
    return json.loads(SRC_SUMMARY.read_text(encoding="utf-8"))


def load_csv_means():
    with SRC_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def mean_of(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return sum(vals) / len(vals)

    return {
        "num_rows": len(rows),
        "acc_rag_mean": mean_of("acc_rag"),
        "acc_no_rag_mean": mean_of("acc_no_rag"),
        "clap_response_rag_mean": mean_of("clap_response_rag"),
        "clap_response_no_rag_mean": mean_of("clap_response_no_rag"),
        "consistency_rag_mean": mean_of("consistency_rag"),
        "consistency_no_rag_mean": mean_of("consistency_no_rag"),
    }


def rel_improvement(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old


def build_json(summary: dict, overall: dict) -> dict:
    per_k = []
    for k in sorted(summary["per_k"], key=lambda x: int(x)):
        row = summary["per_k"][k]
        per_k.append(
            {
                "k": int(k),
                "acc_rag": row["acc_rag"],
                "acc_no_rag": row["acc_no_rag"],
                "acc_absolute_lift": row["acc_rag"] - row["acc_no_rag"],
                "acc_relative_lift": rel_improvement(row["acc_rag"], row["acc_no_rag"]),
                "clap_response_rag": row["clap_response_rag"],
                "clap_response_no_rag": row["clap_response_no_rag"],
                "clap_absolute_lift": row["clap_response_rag"] - row["clap_response_no_rag"],
                "clap_relative_lift": rel_improvement(row["clap_response_rag"], row["clap_response_no_rag"]),
            }
        )

    return {
        "task": "preliminary_clean_rag_vs_no_rag_effectiveness",
        "dataset": "wavcaps_freesound",
        "source_files": {
            "summary_json": str(SRC_SUMMARY),
            "results_csv": str(SRC_CSV),
        },
        "overall_means": {
            **overall,
            "acc_absolute_lift_mean": overall["acc_rag_mean"] - overall["acc_no_rag_mean"],
            "acc_relative_lift_mean": rel_improvement(overall["acc_rag_mean"], overall["acc_no_rag_mean"]),
            "clap_absolute_lift_mean": overall["clap_response_rag_mean"] - overall["clap_response_no_rag_mean"],
            "clap_relative_lift_mean": rel_improvement(
                overall["clap_response_rag_mean"], overall["clap_response_no_rag_mean"]
            ),
        },
        "per_k": per_k,
        "notes": [
            "This is a preliminary clean baseline comparison rather than the main poisoning experiment.",
            "Accuracy is computed with token-overlap thresholding in run_clean_wavcaps_eval.py.",
            "CLAP response score is a semantic alignment metric between query audio and generated response text.",
        ],
    }


def svg_line(points, x0, y0, w, h, color):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(max(ys), 0.55)
    coords = []
    for x, y in points:
        px = x0 + (x - min_x) / (max_x - min_x) * w if max_x != min_x else x0
        py = y0 + h - (y - min_y) / (max_y - min_y) * h if max_y != min_y else y0 + h
        coords.append((px, py))
    path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path}" />', coords, max_y


def build_svg(payload: dict) -> str:
    width, height = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf5"/>',
        '<text x="36" y="38" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#7c2d12">Preliminary Clean Baseline: With RAG vs Without RAG</text>',
        '<text x="36" y="62" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#9a3412">WavCaps clean evaluation. Left: token-overlap accuracy. Right: CLAP semantic response alignment.</text>',
    ]

    panels = [
        ("Accuracy", 36, 96, 420, 330, "acc_rag", "acc_no_rag"),
        ("CLAP Alignment", 510, 96, 420, 330, "clap_response_rag", "clap_response_no_rag"),
    ]

    for title, x, y, w, h, key_rag, key_no in panels:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="#fed7aa"/>')
        parts.append(f'<text x="{x+18}" y="{y+28}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#7c2d12">{title}</text>')

        gx, gy, gw, gh = x + 55, y + 40, w - 80, h - 85
        parts.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="none" stroke="#fdba74"/>')

        for t in range(6):
            yy = gy + gh - gh * t / 5
            val = (max(0.55, max(r[key_rag] for r in payload["per_k"])) if "clap" in key_rag else max(0.15, max(r[key_rag] for r in payload["per_k"]))) * t / 5
            parts.append(f'<line x1="{gx}" y1="{yy}" x2="{gx+gw}" y2="{yy}" stroke="#ffedd5" />')
            parts.append(f'<text x="{gx-8}" y="{yy+4}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#9a3412">{val:.2f}</text>')

        ks = [row["k"] for row in payload["per_k"]]
        for i, k in enumerate(ks):
            xx = gx + gw * i / (len(ks) - 1)
            parts.append(f'<line x1="{xx}" y1="{gy}" x2="{xx}" y2="{gy+gh}" stroke="#fff7ed" />')
            parts.append(f'<text x="{xx}" y="{gy+gh+22}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700" fill="#9a3412">k={k}</text>')

        rag_points = [(row["k"], row[key_rag]) for row in payload["per_k"]]
        no_points = [(row["k"], row[key_no]) for row in payload["per_k"]]
        poly_rag, coords_rag, _ = svg_line(rag_points, gx, gy, gw, gh, "#ea580c")
        poly_no, coords_no, _ = svg_line(no_points, gx, gy, gw, gh, "#f59e0b")
        parts.extend([poly_rag, poly_no])

        for (px, py), row in zip(coords_rag, payload["per_k"]):
            parts.append(f'<circle cx="{px}" cy="{py}" r="4.5" fill="#ea580c"/>')
            parts.append(f'<text x="{px}" y="{py-10}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#7c2d12">{row[key_rag]:.3f}</text>')
        for (px, py), row in zip(coords_no, payload["per_k"]):
            parts.append(f'<circle cx="{px}" cy="{py}" r="4.5" fill="#f59e0b"/>')
            parts.append(f'<text x="{px}" y="{py+18}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#9a3412">{row[key_no]:.3f}</text>')

    parts.append('<line x1="360" y1="468" x2="390" y2="468" stroke="#ea580c" stroke-width="3"/>')
    parts.append('<text x="398" y="472" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">With RAG</text>')
    parts.append('<line x1="500" y1="468" x2="530" y2="468" stroke="#f59e0b" stroke-width="3"/>')
    parts.append('<text x="538" y="472" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">Without RAG</text>')

    acc_rel = payload["overall_means"]["acc_relative_lift_mean"] * 100
    clap_rel = payload["overall_means"]["clap_relative_lift_mean"] * 100
    parts.append(
        f'<text x="36" y="500" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#7c2d12">Mean lift: accuracy +{acc_rel:.1f}% relative, CLAP alignment +{clap_rel:.1f}% relative.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    overall = load_csv_means()
    payload = build_json(summary, overall)

    (OUT_DIR / "preliminary_rag_effectiveness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "preliminary_rag_effectiveness_plot.svg").write_text(
        build_svg(payload), encoding="utf-8"
    )
    print(OUT_DIR / "preliminary_rag_effectiveness_summary.json")
    print(OUT_DIR / "preliminary_rag_effectiveness_plot.svg")


if __name__ == "__main__":
    main()
