from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from attack.anchor_selection import AnchorSelector
from attack.cdab import CDABPipeline
from attack_generation.schemas import CDABAttackRecord
from dataset.manager import DatasetManager
from retriever.encoders import build_encoder
from utils.config import ensure_dir, load_yaml_config
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated CDAB attack plans without modifying the main pipeline.")
    parser.add_argument("--config", required=True, help="Path to attack-generation YAML config.")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    logger = setup_logger("attack_generation")

    seed = int(config["project"].get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)

    root_config = load_yaml_config("configs/default.yaml")
    manager = DatasetManager(root_config["data"])
    samples = manager.load_manifest(mode="clean", dataset_name=config["input"]["source_dataset_name"])
    if not samples:
        raise SystemExit(
            "Clean manifest not found for attack planning. Run dataset preparation first so outputs/manifests exist."
        )

    sample_by_id = {sample.sample_id: sample for sample in samples}
    selected_ids = config["attack"].get("source_sample_ids") or list(sample_by_id.keys())[:1]
    target_texts = list(config["attack"].get("target_texts", []))
    if not target_texts:
        raise SystemExit("No target_texts provided in attack-generation config.")

    encoder = build_encoder(root_config["retrieval"]["encoder"], root_config["retrieval"].get("retriever_checkpoint_overrides"))
    embeddings = encoder.encode_audio([sample.audio_path for sample in samples]).astype("float32")

    selector = AnchorSelector(num_clusters=int(root_config["attack"]["clustering"]["num_clusters"]))
    cdab = CDABPipeline()

    rows: list[dict] = []
    for source_sample_id in selected_ids:
        sample = sample_by_id.get(source_sample_id)
        if sample is None:
            logger.warning("Skipping unknown source sample id: %s", source_sample_id)
            continue
        sample_index = next(index for index, item in enumerate(samples) if item.sample_id == source_sample_id)
        selection = selector.select(
            samples=samples,
            embeddings=embeddings,
            query_index=sample_index,
            count=int(config["attack"]["anchor_count"]),
        )
        anchor_texts = [sample_by_id[anchor_id].text for anchor_id in selection.anchor_sample_ids if anchor_id in sample_by_id]
        for target_text in target_texts:
            plan = cdab.build_plan(source_text=sample.text, target_text=target_text)
            record = CDABAttackRecord(
                source_sample_id=source_sample_id,
                source_text=sample.text,
                target_text=target_text,
                anchor_sample_ids=selection.anchor_sample_ids,
                anchor_texts=anchor_texts,
                cluster_id=selection.cluster_id,
                metadata={
                    "source_dataset": sample.dataset_name,
                    "selection_mode": config["attack"].get("selection_mode", "nearest_neighbors"),
                    "anchor_count": int(config["attack"]["anchor_count"]),
                    "cdab_plan": {
                        "de_semanticized_text": plan.de_semanticized_text,
                        "physicalized_text": plan.physicalized_text,
                        "reframed_text": plan.reframed_text,
                    },
                },
            )
            rows.append(record.to_json())

    output_path = Path(config["output"]["output_jsonl"])
    write_jsonl(output_path, rows)
    logger.info("Wrote %s CDAB attack plan records to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
