from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Audio RAG ablation sweeps.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--query-audio", required=True, help="Path to evaluation query audio.")
    parser.add_argument("--query-text", default="What is this sound?", help="User text query.")
    parser.add_argument("--expected-answer", default=None, help="Expected clean answer.")
    parser.add_argument("--attack-target", default=None, help="Attack target answer.")
    parser.add_argument("--mode", choices=["clean", "poisoned"], default="clean", help="Dataset mode.")
    parser.add_argument("--run-name", default="ablation_run", help="Logger run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.config import ensure_dir, load_yaml_config
    from utils.logging import ExperimentLogger, setup_logger
    from dataset.manager import DatasetManager
    from evaluation.metrics import summarize_metrics
    from models.wrappers import build_victim_model
    from rag.engine import AudioRAGEngine
    from retriever.encoders import build_encoder
    from retriever.index import FaissAudioRetriever

    config = load_yaml_config(args.config)
    logger = setup_logger()
    experiment_logger = ExperimentLogger(output_dir=ensure_dir(config["project"]["output_dir"]), run_name=args.run_name)
    manager = DatasetManager(config["data"])
    samples = manager.load_manifest(mode=args.mode)
    if not samples:
        raise SystemExit("Dataset manifest not found. Prepare datasets first with run_attack.py --prepare-datasets.")

    victim_model = build_victim_model(config["models"])
    bundles = []
    sweep_space = itertools.product(
        config["experiments"]["retrievers"],
        config["experiments"]["retrieval_k_values"],
        config["experiments"]["poison_count_values"],
        config["experiments"]["noise_levels"],
        config["experiments"]["compression_bitrates"],
    )
    for retriever_name, top_k, poison_count, noise_level, bitrate in sweep_space:
        sweep_config = deepcopy(config)
        sweep_config["retrieval"]["encoder"] = retriever_name
        sweep_config["rag"]["top_k"] = top_k
        sweep_config["attack"]["poison_count"] = poison_count
        sweep_config["runtime_perturbation"] = {"noise_level": noise_level, "compression_bitrate": bitrate}
        try:
            encoder = build_encoder(retriever_name, sweep_config["retrieval"].get("retriever_checkpoint_overrides"))
            retriever = FaissAudioRetriever(encoder=encoder, similarity=sweep_config["retrieval"]["similarity"])
            retriever.build(samples)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping retriever=%s because initialization failed: %s", retriever_name, exc)
            continue
        engine = AudioRAGEngine(
            retriever=retriever,
            encoder=encoder,
            victim_model=victim_model,
            rag_config=sweep_config["rag"],
            evaluation_config=sweep_config["evaluation"],
        )
        result = engine.run_query(
            audio_path=args.query_audio,
            user_query=args.query_text,
            expected_answer=args.expected_answer,
            attack_target=args.attack_target,
            poisoned_ids={sample.sample_id for sample in samples if sample.is_poisoned},
            clean_reference_text=args.expected_answer,
            clean_reference_audio_path=args.query_audio,
        )
        bundles.append(result.metrics)
        payload = {
            "retriever": retriever_name,
            "top_k": top_k,
            "poison_count": poison_count,
            "noise_level": noise_level,
            "compression_bitrate": bitrate,
            "metrics": result.metrics.to_json(),
        }
        experiment_logger.log("ablation_result", payload)
        logger.info("Ablation result: %s", json.dumps(payload))

    summary = summarize_metrics(bundles)
    experiment_logger.log("ablation_summary", summary)
    logger.info("Ablation summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
