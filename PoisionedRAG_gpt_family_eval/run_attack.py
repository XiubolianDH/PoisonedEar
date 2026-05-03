from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Audio RAG poisoning evaluation experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--prepare-datasets", action="store_true", help="Build manifests and dataset statistics.")
    parser.add_argument("--query-audio", help="Path to query audio file.")
    parser.add_argument("--query-text", default="What is this sound?", help="User text query.")
    parser.add_argument("--expected-answer", default=None, help="Expected clean answer for ACC/Recall calculations.")
    parser.add_argument("--attack-target", default=None, help="Target answer for ASR-G calculations.")
    parser.add_argument("--mode", choices=["clean", "poisoned"], default=None, help="Dataset mode override.")
    parser.add_argument("--dataset-name", choices=["wavcaps", "clotho", "audioset", "esc50"], default=None, help="Optional dataset-specific manifest.")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval top-k override.")
    parser.add_argument("--run-name", default="attack_run", help="Logger run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.config import ensure_dir, load_yaml_config
    from utils.logging import ExperimentLogger, setup_logger
    from attack.anchor_selection import AnchorSelector
    from attack.cdab import CDABPipeline
    from attack.injection import InjectionSimulator
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
    if args.prepare_datasets:
        clean_samples, poisoned_samples = manager.prepare()
        logger.info("Prepared clean=%s poisoned=%s samples", len(clean_samples), len(poisoned_samples))
        experiment_logger.log("dataset_prepare", {"clean_samples": len(clean_samples), "poisoned_samples": len(poisoned_samples)})

    mode = args.mode or config["rag"]["mode"]
    samples = manager.load_manifest(mode=mode, dataset_name=args.dataset_name)
    if not samples:
        raise SystemExit(
            "No dataset manifest available for the selected mode. "
            "Set dataset metadata paths in configs/default.yaml and run with --prepare-datasets first."
        )
    encoder = build_encoder(config["retrieval"]["encoder"], config["retrieval"].get("retriever_checkpoint_overrides"))
    retriever = FaissAudioRetriever(encoder=encoder, similarity=config["retrieval"]["similarity"])
    retrieval_index = retriever.build(samples)
    logger.info("Built FAISS index with %s samples using %s", len(retrieval_index.samples), retrieval_index.encoder_name)

    anchor_selector = AnchorSelector(num_clusters=int(config["attack"]["clustering"]["num_clusters"]))
    injection_simulator = InjectionSimulator(
        deduplication_threshold=float(config["attack"]["injection_simulation"]["deduplication_threshold"]),
        filtering_drop_rate=float(config["attack"]["injection_simulation"]["filtering_drop_rate"]),
        ranking_noise_std=float(config["attack"]["injection_simulation"]["ranking_noise_std"]),
    )
    cdab = CDABPipeline()

    experiment_logger.log(
        "attack_structure",
        {
            "anchor_selector": anchor_selector.__class__.__name__,
            "injection_simulator": injection_simulator.__class__.__name__,
            "cdab_pipeline": cdab.__class__.__name__,
        },
    )

    if not args.query_audio:
        logger.info("Framework initialized successfully. Provide --query-audio to run end-to-end inference.")
        return

    rag_config = dict(config["rag"])
    if args.top_k is not None:
        rag_config["top_k"] = args.top_k
    victim_model = build_victim_model(config["models"])
    engine = AudioRAGEngine(
        retriever=retriever,
        encoder=encoder,
        victim_model=victim_model,
        rag_config=rag_config,
        evaluation_config=config["evaluation"],
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
    logger.info("Generated response: %s", result.generated_response)
    logger.info("Metrics: %s", json.dumps(result.metrics.to_json(), indent=2))
    experiment_logger.log(
        "query_result",
        {
            "response": result.generated_response,
            "context": result.context,
            "retrieved_ids": [hit.sample.sample_id for hit in result.retrieved_hits],
            "metrics": result.metrics.to_json(),
            "summary": summarize_metrics([result.metrics]),
        },
    )


if __name__ == "__main__":
    main()
