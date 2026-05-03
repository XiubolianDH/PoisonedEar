from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm.auto import tqdm

from dataset.loaders import load_dataset
from retriever.base import AudioClassifier
from retriever.encoders import build_encoder
from retriever.index import FaissAudioRetriever
from run_clean_wavcaps_eval import compute_kl_metrics_for_sample
from utils.config import load_yaml_config
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill KL columns for clean WavCaps evaluation CSV.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--results-csv", default="outputs/results_clean_wavcaps.csv", help="Existing evaluation CSV.")
    return parser.parse_args()


def load_wavcaps_samples(config: dict) -> dict[str, object]:
    settings = dict(config["data"]["datasets"]["wavcaps"])
    settings["enabled"] = True
    samples = list(load_dataset(name="wavcaps", settings=settings))
    return {sample.sample_id: sample for sample in samples}


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    config = load_yaml_config(args.config)
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        raise SystemExit(f"Results CSV not found: {csv_path}")

    logger.info("Loading WavCaps samples and building spectral retriever for KL backfill.")
    sample_by_id = load_wavcaps_samples(config)
    samples = list(sample_by_id.values())
    retriever = FaissAudioRetriever(
        encoder=build_encoder(config["retrieval"]["encoder"], config["retrieval"].get("retriever_checkpoint_overrides")),
        similarity=config["retrieval"]["similarity"],
    )
    retriever.build(samples, show_progress=True, progress_desc="Encoding retrieval corpus for KL backfill")

    logger.info("Initializing PANNs classifier for KL backfill.")
    classifier = build_encoder("panns")
    if not isinstance(classifier, AudioClassifier):
        raise SystemExit("Configured PANNs backend does not expose AudioClassifier.")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    updated = 0
    for row in tqdm(rows, desc="Backfilling KL", unit="row", dynamic_ncols=True):
        if row.get("kl_rag") not in ("", None) and row.get("kl_no_rag") not in ("", None):
            continue
        sample = sample_by_id.get(row["sample_id"])
        if sample is None:
            continue
        k_value = int(row["k"])
        hits = retriever.query(audio_path=sample.audio_path, top_k=k_value)
        kl_rag, kl_no_rag = compute_kl_metrics_for_sample(
            classifier=classifier,
            clean_audio_path=sample.audio_path,
            retrieved_audio_paths=[hit.sample.audio_path for hit in hits],
        )
        row["kl_rag"] = "" if kl_rag is None else f"{kl_rag:.6f}"
        row["kl_no_rag"] = "" if kl_no_rag is None else f"{kl_no_rag:.6f}"
        updated += 1

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Backfilled KL columns for %s rows in %s", updated, csv_path)


if __name__ == "__main__":
    main()
