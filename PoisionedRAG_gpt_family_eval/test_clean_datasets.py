from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    python_bin = "/home/shuhaoz/miniconda3/envs/audio_rag/bin/python"
    for dataset_name in ["wavcaps", "clotho", "audioset", "esc50"]:
        manifest_path = root / "data" / "manifests" / f"{dataset_name}_clean.jsonl"
        rows = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8")]
        sample = min(rows, key=lambda row: row["stats"]["duration_seconds"])
        cmd = [
            python_bin,
            "run_attack.py",
            "--config",
            "configs/default.yaml",
            "--dataset-name",
            dataset_name,
            "--mode",
            "clean",
            "--query-audio",
            sample["audio_path"],
            "--query-text",
            "What is this sound?",
            "--expected-answer",
            sample["text"],
            "--run-name",
            f"test_{dataset_name}_clean",
        ]
        print(f"\n=== {dataset_name} ===")
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
