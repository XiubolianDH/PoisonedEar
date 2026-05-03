from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_cross_dataset_wavcaps_attack_eval.py"
PYTHON = Path(os.environ.get("AUDIO_RAG_PYTHON", "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a top-k impact sweep on a fixed poisoned manifest.")
    parser.add_argument("--poisoned-manifest", required=True, help="Path to the fixed poisoned manifest (e.g. 15%% rate manifest).")
    parser.add_argument("--malicious-metadata", required=True, help="Path to metadata/query subset JSON.")
    parser.add_argument("--output-dir", required=True, help="Independent output directory for this top-k sweep.")
    parser.add_argument("--poison-rate-pct", required=True, type=int, help="Fixed poison rate for labeling outputs.")
    parser.add_argument("--poison-count", required=True, type=int, help="Poison count at the fixed poison rate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 15, 20])
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for k in args.k_values:
        results_csv = output_dir / f"results_topk_{k:02d}.csv"
        summary_json = output_dir / f"results_topk_{k:02d}.summary.json"
        error_log = output_dir / f"results_topk_{k:02d}.errors.jsonl"

        cmd = [
            str(PYTHON),
            str(RUNNER),
            "--config",
            args.config,
            "--poisoned-manifest",
            str(Path(args.poisoned_manifest)),
            "--malicious-metadata",
            str(Path(args.malicious_metadata)),
            "--results-csv",
            str(results_csv),
            "--summary-json",
            str(summary_json),
            "--error-log",
            str(error_log),
            "--top-k",
            str(k),
        ]
        if args.continue_on_error:
            cmd.append("--continue-on-error")

        print(f"[RUN] poison_rate={args.poison_rate_pct}% top_k={k}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "poison_rate_pct": args.poison_rate_pct,
                "poison_count": args.poison_count,
                "top_k": k,
                **summary,
            }
        )

    comparison_csv = output_dir / "comparison_summary_topk.csv"
    comparison_json = output_dir / "comparison_summary_topk.json"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "poison_rate_pct",
                "poison_count",
                "top_k",
                "num_queries",
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

    comparison_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {comparison_csv}")


if __name__ == "__main__":
    main()
