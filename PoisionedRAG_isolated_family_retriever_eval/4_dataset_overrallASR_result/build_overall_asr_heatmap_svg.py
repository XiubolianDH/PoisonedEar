import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "4_dataset_overrallASR_result" / "plots"

DATASETS = {
    "wavcaps": REPO_ROOT / "malicious_dataset" / "all_classes_attack_metrics_v3" / "all_classes_long_table_v3.json",
    "audioset": REPO_ROOT / "malicious_dataset" / "audioset_all_classes_overall_asr" / "audioset_all_classes_overall_asr_long.json",
    "clotho": REPO_ROOT / "4_dataset_overrallASR_result" / "clotho" / "clotho_all_classes_overall_asr_long_predicted_calibrated.json",
    "esc50": REPO_ROOT / "4_dataset_overrallASR_result" / "esc50" / "esc50_all_classes_overall_asr_long_predicted_calibrated.json",
}

CLASS_ORDER = [
    "music_instrument",
    "human_voice_speech",
    "vehicle_transport",
    "animal_bio",
    "machine_mechanical",
    "water_weather",
    "impact_material",
    "electronic_synthetic",
    "ambience_environment",
]
RATE_ORDER = [1, 5, 10, 15, 20]
SHORT_LABELS = {
    "music_instrument": "music",
    "human_voice_speech": "voice",
    "vehicle_transport": "vehicle",
    "animal_bio": "animal",
    "machine_mechanical": "machine",
    "water_weather": "water",
    "impact_material": "impact",
    "electronic_synthetic": "electronic",
    "ambience_environment": "ambience",
}


def load_rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_lookup(rows):
    lookup = {}
    for row in rows:
        lookup[(row["class_name"], row["poison_rate_pct"])] = row["Overall_ASR"]
    return lookup


def color_for(value: float) -> str:
    value = max(0.0, min(1.0, value))
    # light cream -> bright orange -> deep burnt orange
    if value <= 0.5:
        t = value / 0.5
        r1, g1, b1 = (255, 247, 237)
        r2, g2, b2 = (255, 159, 28)
    else:
        t = (value - 0.5) / 0.5
        r1, g1, b1 = (255, 159, 28)
        r2, g2, b2 = (181, 71, 8)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def text_color(value: float) -> str:
    return "#f8fafc" if value >= 0.58 else "#0f172a"


def make_svg():
    lookups = {name: build_lookup(load_rows(path)) for name, path in DATASETS.items()}

    panel_w = 380
    panel_h = 305
    cell_w = 44
    cell_h = 24
    left_pad = 110
    top_pad = 58
    legend_h = 80
    outer_w = panel_w * 2 + 110
    outer_h = panel_h * 2 + legend_h + 90

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{outer_w}" height="{outer_h}" viewBox="0 0 {outer_w} {outer_h}">',
        '<rect width="100%" height="100%" fill="#fcfbf7"/>',
        '<text x="34" y="36" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#102a43">Overall ASR Heatmap: 4 Datasets x 9 Classes x 5 Poison Rates</text>',
        '<text x="34" y="58" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#486581">Solid datasets: tested. Clotho and ESC-50: calibrated predictions aligned to the dataset-level overall ASR summary.</text>',
    ]

    panel_positions = {
        "wavcaps": (34, 82),
        "audioset": (34 + panel_w + 46, 82),
        "clotho": (34, 82 + panel_h + 26),
        "esc50": (34 + panel_w + 46, 82 + panel_h + 26),
    }

    for dataset, (x0, y0) in panel_positions.items():
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="16" fill="#ffffff" stroke="#d9e2ec"/>')
        subtitle = "tested" if dataset in {"wavcaps", "audioset"} else "predicted"
        parts.append(f'<text x="{x0 + 16}" y="{y0 + 26}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#102a43">{dataset}</text>')
        parts.append(f'<text x="{x0 + 100}" y="{y0 + 26}" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#7b8794">{subtitle}</text>')

        grid_x = x0 + left_pad
        grid_y = y0 + top_pad
        lookup = lookups[dataset]

        for j, rate in enumerate(RATE_ORDER):
            cx = grid_x + j * cell_w + cell_w / 2
            parts.append(f'<text x="{cx}" y="{y0 + 46}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" font-weight="700" fill="#334e68">{rate}%</text>')

        for i, class_name in enumerate(CLASS_ORDER):
            cy = grid_y + i * cell_h + cell_h / 2 + 4
            parts.append(f'<text x="{grid_x - 8}" y="{cy}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#334e68">{SHORT_LABELS[class_name]}</text>')
            for j, rate in enumerate(RATE_ORDER):
                value = lookup[(class_name, rate)]
                fill = color_for(value)
                tx = grid_x + j * cell_w
                ty = grid_y + i * cell_h
                parts.append(f'<rect x="{tx}" y="{ty}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="{fill}"/>')
                parts.append(
                    f'<text x="{tx + (cell_w - 2)/2}" y="{ty + 16}" text-anchor="middle" '
                    f'font-family="Arial, Helvetica, sans-serif" font-size="10" font-weight="700" fill="{text_color(value)}">{value:.2f}</text>'
                )

    legend_x = 210
    legend_y = outer_h - 52
    parts.append(f'<text x="{legend_x}" y="{legend_y - 20}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="700" fill="#102a43">Overall ASR</text>')
    for i in range(101):
        value = i / 100
        fill = color_for(value)
        x = legend_x + i * 5.2
        parts.append(f'<rect x="{x}" y="{legend_y}" width="6" height="16" fill="{fill}"/>')
    for tick, label in [(0.0, "0.0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")]:
        x = legend_x + tick * 100 * 5.2
        parts.append(f'<text x="{x}" y="{legend_y + 34}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#486581">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg = make_svg()
    out_path = PLOTS_DIR / "overall_asr_heatmap_4datasets_9classes_5rates.svg"
    out_path.write_text(svg, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
