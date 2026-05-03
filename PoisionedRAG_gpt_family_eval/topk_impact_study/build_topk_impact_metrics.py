from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize top-k impact results into Overall_ASR / Precision / Recall metrics.")
    parser.add_argument("--input-dir", required=True, help="Directory produced by run_topk_impact_eval.py")
    parser.add_argument("--poison-rate-pct", required=True, type=int)
    parser.add_argument("--poison-count", required=True, type=int)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 15, 20])
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def summarize_k(k: int, csv_path: Path, poison_rate_pct: int, poison_count: int) -> dict[str, float | int]:
    rows = load_rows(csv_path)

    retrieval_hits = [row for row in rows if to_int(row, "asr_r") == 1]
    asr_r = len(retrieval_hits) / len(rows) if rows else 0.0

    if retrieval_hits:
        asr_g_given_r = sum(to_int(row, "asr_g") for row in retrieval_hits) / len(retrieval_hits)
        avg_attack_margin_given_r = sum(to_float(row, "attack_margin") for row in retrieval_hits) / len(retrieval_hits)
    else:
        asr_g_given_r = 0.0
        avg_attack_margin_given_r = 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for row in rows:
        malicious_count = to_int(row, "malicious_count")
        precision = malicious_count / k if k > 0 else 0.0
        recall = malicious_count / poison_count if poison_count > 0 else 0.0
        f1 = safe_f1(precision, recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    overall_asr = asr_r * asr_g_given_r
    return {
        "poison_rate_pct": poison_rate_pct,
        "poison_count": poison_count,
        "num_queries": len(rows),
        "top_k": k,
        "ASR-R": asr_r,
        "ASR-G|R": asr_g_given_r,
        "avg_attack_margin|R": avg_attack_margin_given_r,
        "Precision@k": sum(precisions) / len(precisions) if precisions else 0.0,
        "Recall@k": sum(recalls) / len(recalls) if recalls else 0.0,
        "F1@k": sum(f1s) / len(f1s) if f1s else 0.0,
        "Overall_ASR": overall_asr,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = input_dir / "metrics_topk_impact"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int]] = []
    for k in args.k_values:
        csv_path = input_dir / f"results_topk_{k:02d}.csv"
        summary = summarize_k(k, csv_path, args.poison_rate_pct, args.poison_count)
        summaries.append(summary)
        per_k_json = output_dir / f"results_topk_{k:02d}.summary.v3.json"
        per_k_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "poison_rate_pct",
        "poison_count",
        "num_queries",
        "top_k",
        "ASR-R",
        "ASR-G|R",
        "avg_attack_margin|R",
        "Precision@k",
        "Recall@k",
        "F1@k",
        "Overall_ASR",
    ]
    comparison_csv = output_dir / "comparison_summary_topk_impact.csv"
    comparison_json = output_dir / "comparison_summary_topk_impact.json"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    comparison_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(comparison_json)


if __name__ == "__main__":
    main()
