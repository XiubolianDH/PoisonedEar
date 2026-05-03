from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "attack_experiments" / "audioset_human_voice_speech_rates" / "experiment_plan.json"
MALICIOUS_METADATA = (
    REPO_ROOT
    / "malicious_dataset"
    / "wavcaps_clotho_esc50_to_audioset_human_voice_speech_88"
    / "final_dataset"
    / "query_subset_20.json"
)
OUTPUT_DIR = REPO_ROOT / "outputs" / "audioset_human_voice_speech_rate_sweep"
RUNNER = REPO_ROOT / "run_cross_dataset_wavcaps_attack_eval.py"
PYTHON = Path(os.environ.get("AUDIO_RAG_PYTHON", "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"))


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for item in plan["rates"]:
        rate = int(item["poison_rate_pct"])
        manifest_path = REPO_ROOT / item["manifest_path"]
        results_csv = OUTPUT_DIR / f"results_rate_{rate:02d}pct.csv"
        summary_json = OUTPUT_DIR / f"results_rate_{rate:02d}pct.summary.json"
        error_log = OUTPUT_DIR / f"results_rate_{rate:02d}pct.errors.jsonl"

        cmd = [
            str(PYTHON),
            str(RUNNER),
            "--config",
            "configs/default.yaml",
            "--poisoned-manifest",
            str(manifest_path),
            "--malicious-metadata",
            str(MALICIOUS_METADATA),
            "--results-csv",
            str(results_csv),
            "--summary-json",
            str(summary_json),
            "--error-log",
            str(error_log),
            "--top-k",
            "5",
            "--continue-on-error",
        ]
        print(f"[RUN] poison_rate={rate}%")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "poison_rate_pct": rate,
                "poison_count": int(item["poison_count"]),
                **summary,
            }
        )

    comparison_csv = OUTPUT_DIR / "comparison_summary.csv"
    comparison_json = OUTPUT_DIR / "comparison_summary.json"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "poison_rate_pct",
                "poison_count",
                "num_queries",
                "top_k",
                "asr_r_rate",
                "asr_g_rate",
                "avg_malicious_count",
                "avg_clap_response",
                "avg_recall_at_k",
                "avg_attack_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[DONE] wrote {comparison_csv}")


if __name__ == "__main__":
    main()
