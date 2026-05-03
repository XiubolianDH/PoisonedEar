import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_SUMMARY = REPO_ROOT / "4_dataset_overrallASR_result" / "preliminary_rag_effectiveness" / "preliminary_rag_effectiveness_summary.json"
OUT_DIR = REPO_ROOT / "4_dataset_overrallASR_result" / "preliminary_accuracy_faithfulness"


def load_payload() -> dict:
    return json.loads(SRC_SUMMARY.read_text(encoding="utf-8"))


def build_summary(src: dict) -> dict:
    means = src["overall_means"]
    return {
        "task": "preliminary_accuracy_and_faithfulness_bar",
        "dataset": src["dataset"],
        "source_file": str(SRC_SUMMARY),
        "metrics": {
            "accuracy_correctness": {
                "with_rag": means["acc_rag_mean"],
                "without_rag": means["acc_no_rag_mean"],
                "absolute_lift": means["acc_absolute_lift_mean"],
                "relative_lift": means["acc_relative_lift_mean"],
            },
            "faithfulness_proxy": {
                "with_rag": means["clap_response_rag_mean"],
                "without_rag": means["clap_response_no_rag_mean"],
                "absolute_lift": means["clap_absolute_lift_mean"],
                "relative_lift": means["clap_relative_lift_mean"],
            },
        },
        "interpretation": {
            "accuracy_correctness": "Token-overlap correctness against ground truth.",
            "faithfulness_proxy": "We use CLAP response alignment as a practical proxy for answer faithfulness / semantic grounding. Strict context-faithfulness is undefined for no-RAG because no retrieved context is provided.",
        },
        "recommended_reporting": {
            "no_rag_accuracy_percent": round(means["acc_no_rag_mean"] * 100, 2),
            "no_rag_faithfulness_proxy_percent": round(means["clap_response_no_rag_mean"] * 100, 2),
            "with_rag_accuracy_percent": round(means["acc_rag_mean"] * 100, 2),
            "with_rag_faithfulness_proxy_percent": round(means["clap_response_rag_mean"] * 100, 2),
        },
        "notes": [
            "If the paper wants strict context-based faithfulness, do not assign a no-RAG value; mark it as N/A.",
            "If the paper wants a visually comparable bar chart, use CLAP response alignment as a faithfulness proxy for both settings.",
            "Existing data suggest no-RAG correctness is about 1.15%, while no-RAG faithfulness proxy is about 40.05%.",
        ],
    }


def build_svg(summary: dict) -> str:
    acc = summary["metrics"]["accuracy_correctness"]
    faith = summary["metrics"]["faithfulness_proxy"]

    width, height = 900, 520
    bg = "#fffaf5"
    colors = {"with": "#ea580c", "without": "#fdba74"}

    def bar(x, y, w, h, label, value_with, value_without, max_value=0.5):
        parts = []
        parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#7c2d12">{label}</text>')
        chart_y = y + 28
        chart_h = h - 60
        chart_w = w - 40
        chart_x = x + 20
        parts.append(f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" rx="12" fill="#ffffff" stroke="#fed7aa"/>')
        for i in range(6):
            yy = chart_y + chart_h - chart_h * i / 5
            val = max_value * i / 5
            parts.append(f'<line x1="{chart_x+45}" y1="{yy}" x2="{chart_x+chart_w-20}" y2="{yy}" stroke="#ffedd5"/>')
            parts.append(f'<text x="{chart_x+38}" y="{yy+4}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="#9a3412">{val*100:.0f}%</text>')

        base_y = chart_y + chart_h - 25
        plot_h = chart_h - 50
        bw = 90
        gap = 110
        x1 = chart_x + 120
        x2 = x1 + bw + gap
        h1 = plot_h * min(value_with / max_value, 1.0)
        h2 = plot_h * min(value_without / max_value, 1.0)
        y1 = base_y - h1
        y2 = base_y - h2
        parts.append(f'<rect x="{x1}" y="{y1}" width="{bw}" height="{h1}" rx="8" fill="{colors["with"]}"/>')
        parts.append(f'<rect x="{x2}" y="{y2}" width="{bw}" height="{h2}" rx="8" fill="{colors["without"]}"/>')
        parts.append(f'<text x="{x1 + bw/2}" y="{y1-10}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#7c2d12">{value_with*100:.2f}%</text>')
        parts.append(f'<text x="{x2 + bw/2}" y="{y2-10}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#7c2d12">{value_without*100:.2f}%</text>')
        parts.append(f'<text x="{x1 + bw/2}" y="{base_y+24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">With RAG</text>')
        parts.append(f'<text x="{x2 + bw/2}" y="{base_y+24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#7c2d12">No RAG</text>')
        return "\n".join(parts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
        '<text x="36" y="40" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#7c2d12">Preliminary AudioRAG Effectiveness</text>',
        '<text x="36" y="64" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#9a3412">Bars compare With RAG vs No RAG on clean WavCaps. Faithfulness is shown using a semantic alignment proxy.</text>',
        bar(36, 105, 390, 340, "Accuracy / Correctness", acc["with_rag"], acc["without_rag"], max_value=0.15),
        bar(470, 105, 390, 340, "Faithfulness (Proxy)", faith["with_rag"], faith["without_rag"], max_value=0.5),
        f'<text x="36" y="485" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#7c2d12">Mean relative lift: accuracy +{acc["relative_lift"]*100:.1f}%, faithfulness proxy +{faith["relative_lift"]*100:.1f}%.</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = load_payload()
    summary = build_summary(src)
    (OUT_DIR / "preliminary_accuracy_faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "preliminary_accuracy_faithfulness_bar.svg").write_text(
        build_svg(summary), encoding="utf-8"
    )
    print(OUT_DIR / "preliminary_accuracy_faithfulness_summary.json")
    print(OUT_DIR / "preliminary_accuracy_faithfulness_bar.svg")


if __name__ == "__main__":
    main()
