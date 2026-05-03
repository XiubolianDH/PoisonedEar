from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify whether a top-k impact run is clean and consistent.")
    parser.add_argument("--input-dir", required=True, help="Top-k impact output directory")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10, 15, 20])
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    report: list[dict[str, object]] = []
    counts: list[int] = []

    for k in args.k_values:
        csv_path = input_dir / f"results_topk_{k:02d}.csv"
        summary_path = input_dir / f"results_topk_{k:02d}.summary.json"

        row_count = count_rows(csv_path) if csv_path.exists() else None
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        summary_queries = summary.get("num_queries") if summary else None
        consistent = row_count == summary_queries if row_count is not None and summary_queries is not None else False

        if row_count is not None:
            counts.append(row_count)

        report.append(
            {
                "top_k": k,
                "csv_exists": csv_path.exists(),
                "summary_exists": summary_path.exists(),
                "csv_row_count": row_count,
                "summary_num_queries": summary_queries,
                "row_count_matches_summary": consistent,
            }
        )

    unique_counts = sorted(set(counts))
    payload = {
        "all_row_counts_equal": len(unique_counts) == 1,
        "unique_num_queries": unique_counts,
        "per_k": report,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
