from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from attack_generation.schemas import CDABGenerationResult
from dataset.loaders import load_dataset
from dataset.types import AudioSample
from models.base import ModelInput, ModelOutput
from models.wrappers import build_victim_model
from utils.config import ensure_dir, load_yaml_config
from utils.logging import setup_logger

CSV_FIELDS = [
    "sample_id",
    "dataset",
    "audio_path",
    "clean_caption",
    "acoustic_description",
    "adversarial_caption",
    "acoustic_consistency",
    "semantic_distance",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CDAB adversarial captions in an isolated workspace.")
    parser.add_argument("--config", required=True, help="Path to attack-generation YAML config.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def token_set(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {token for token in normalized.split() if token and token not in STOPWORDS}


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.dot(vector_a, vector_b) / ((np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) + 1e-12))


def parse_generation_output(text: str) -> tuple[str | None, str | None]:
    acoustic_description = None
    adversarial_caption = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("acoustic_description:"):
            acoustic_description = line.split(":", 1)[1].strip()
        elif lowered.startswith("adversarial_caption:"):
            adversarial_caption = line.split(":", 1)[1].strip()
    if acoustic_description is None:
        acoustic_match = re.search(
            r"acoustic_description:\s*(.+?)(?:\n|adversarial_caption:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if acoustic_match:
            acoustic_description = acoustic_match.group(1).strip()
    if adversarial_caption is None:
        adversarial_match = re.search(
            r"adversarial_caption:\s*(.+)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if adversarial_match:
            adversarial_caption = adversarial_match.group(1).strip()
    return acoustic_description, adversarial_caption


def build_dataset_samples(root_config: dict, generation_config: dict) -> list[AudioSample]:
    dataset_names = list(generation_config["input"]["datasets"])
    sample_ids_filter = set(generation_config["input"].get("source_sample_ids") or [])
    max_samples_per_dataset = generation_config["input"].get("max_samples_per_dataset")

    samples: list[AudioSample] = []
    for dataset_name in dataset_names:
        settings = dict(root_config["data"]["datasets"][dataset_name])
        settings["enabled"] = True
        loaded = [sample for sample in load_dataset(dataset_name, settings) if Path(sample.audio_path).exists()]
        if sample_ids_filter:
            loaded = [sample for sample in loaded if sample.sample_id in sample_ids_filter]
        if max_samples_per_dataset is not None:
            loaded = loaded[: int(max_samples_per_dataset)]
        samples.extend(loaded)

    num_shards = int(generation_config["input"].get("num_shards", 1))
    shard_index = int(generation_config["input"].get("shard_index", 0))
    if num_shards > 1:
        samples = [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]
    return samples


def load_completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def ensure_csv_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def append_error(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def seed_domain_tracker_from_csv(path: Path, tracker: DomainDiversityTracker | None) -> None:
    if tracker is None or not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            caption = (row.get("adversarial_caption") or "").strip()
            if caption:
                tracker.record_caption(caption)


class MetricComputer:
    def __init__(self, metrics_config: dict, root_config: dict, logger: Any) -> None:
        self.enabled = bool(metrics_config.get("enabled", True))
        self.logger = logger
        self.text_model = None
        self.text_cache: dict[str, np.ndarray] = {}
        if not self.enabled:
            return
        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer(metrics_config["text_embedding_model"])

    def encode_text(self, text: str) -> np.ndarray:
        if text not in self.text_cache:
            assert self.text_model is not None
            embedding = self.text_model.encode([text], normalize_embeddings=True)[0]
            self.text_cache[text] = np.asarray(embedding, dtype=np.float32)
        return self.text_cache[text]

    def compute(self, clean_caption: str, acoustic_description: str, adversarial_caption: str) -> tuple[float | None, float | None]:
        if not self.enabled:
            return None, None
        try:
            acoustic_description_embedding = self.encode_text(acoustic_description)
            clean_text_embedding = self.encode_text(clean_caption)
            adversarial_text_embedding = self.encode_text(adversarial_caption)
            acoustic_consistency = cosine_similarity(acoustic_description_embedding, adversarial_text_embedding)
            semantic_similarity = cosine_similarity(clean_text_embedding, adversarial_text_embedding)
            semantic_distance = 1.0 - semantic_similarity
            return float(acoustic_consistency), float(semantic_distance)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Metric computation failed: %s", exc)
            return None, None


@dataclass
class CandidateRecord:
    acoustic_description: str
    adversarial_caption: str
    acoustic_consistency: float
    semantic_distance: float
    interpretation_style: str
    domain_labels: list[str]
    domain_penalty: float
    score: float


class DomainDiversityTracker:
    def __init__(
        self,
        domain_families: dict[str, list[str]],
        recent_window_size: int,
        max_recent_domain_usage: int,
        preferred_domain_count: int,
        domain_penalty_weight: float,
    ) -> None:
        self.domain_families = {
            name: [normalize_text(keyword) for keyword in keywords if normalize_text(keyword)]
            for name, keywords in domain_families.items()
        }
        self.recent_domains: deque[str] = deque(maxlen=max(1, recent_window_size))
        self.max_recent_domain_usage = max(1, max_recent_domain_usage)
        self.preferred_domain_count = max(1, preferred_domain_count)
        self.domain_penalty_weight = max(0.0, domain_penalty_weight)
        self.total_counts: Counter[str] = Counter()

    def infer_domains(self, text: str) -> list[str]:
        lowered = normalize_text(text)
        matched: list[str] = []
        for name, keywords in self.domain_families.items():
            if any(keyword in lowered for keyword in keywords):
                matched.append(name)
        return matched or ["unclassified"]

    def recent_counts(self) -> Counter[str]:
        return Counter(self.recent_domains)

    def overused_domains(self) -> list[str]:
        return sorted(
            [name for name, count in self.recent_counts().items() if count >= self.max_recent_domain_usage and name != "unclassified"]
        )

    def preferred_domains(self) -> list[str]:
        candidates = [
            name
            for name in self.domain_families
            if name not in self.overused_domains()
        ]
        ranked = sorted(candidates, key=lambda name: (self.total_counts.get(name, 0), name))
        return ranked[: self.preferred_domain_count]

    def penalty(self, domain_labels: list[str]) -> float:
        recent_counts = self.recent_counts()
        penalty = 0.0
        for label in domain_labels:
            if label == "unclassified":
                continue
            penalty += self.domain_penalty_weight * (
                recent_counts.get(label, 0) + (0.15 * self.total_counts.get(label, 0))
            )
        return float(penalty)

    def is_overused(self, domain_labels: list[str]) -> bool:
        recent_counts = self.recent_counts()
        return any(
            label != "unclassified" and recent_counts.get(label, 0) >= self.max_recent_domain_usage
            for label in domain_labels
        )

    def record_caption(self, caption: str) -> None:
        for label in self.infer_domains(caption):
            self.total_counts[label] += 1
            self.recent_domains.append(label)


def extract_forbidden_words(clean_caption: str, config_words: Iterable[str]) -> set[str]:
    caption_words = {token for token in token_set(clean_caption) if len(token) >= 4}
    return caption_words.union({normalize_text(word) for word in config_words if normalize_text(word)})


def is_low_level_description(text: str, forbidden_words: set[str]) -> bool:
    lowered = normalize_text(text)
    return not any(word in lowered.split() for word in forbidden_words)


def differs_from_clean_caption(clean_caption: str, adversarial_caption: str) -> bool:
    return normalize_text(clean_caption) != normalize_text(adversarial_caption)


def is_consistent(acoustic_description: str, adversarial_caption: str, cue_words: list[str]) -> bool:
    acoustic_tokens = token_set(acoustic_description)
    adversarial_tokens = token_set(adversarial_caption)
    shared_cues = [word for word in cue_words if word in acoustic_tokens and word in adversarial_tokens]
    return bool(shared_cues) or len(acoustic_tokens.intersection(adversarial_tokens)) >= 2


def build_prompt(
    template: str,
    interpretation_style: str,
    avoid_domains: list[str],
    preferred_domains: list[str],
) -> str:
    avoid_domains_text = ", ".join(avoid_domains) if avoid_domains else "none"
    preferred_domains_text = ", ".join(preferred_domains) if preferred_domains else "any underused acoustically plausible domain"
    return template.format(
        interpretation_style=interpretation_style,
        avoid_domains=avoid_domains_text,
        preferred_domains=preferred_domains_text,
    )


def candidate_text_distance(metric_computer: MetricComputer, text_a: str, text_b: str) -> float:
    embedding_a = metric_computer.encode_text(text_a)
    embedding_b = metric_computer.encode_text(text_b)
    return 1.0 - cosine_similarity(embedding_a, embedding_b)


def contains_anti_collapse_phrase(text: str, phrases: list[str]) -> bool:
    lowered = normalize_text(text)
    return any(normalize_text(phrase) in lowered for phrase in phrases)


def select_diverse_styles(all_styles: list[str], count: int) -> list[str]:
    if count <= len(all_styles):
        return random.sample(all_styles, count)
    styles = list(all_styles)
    while len(styles) < count:
        styles.append(all_styles[len(styles) % len(all_styles)])
    random.shuffle(styles)
    return styles[:count]


def score_candidate(acoustic_consistency: float, semantic_distance: float, domain_penalty: float) -> float:
    return (1.4 * semantic_distance) + acoustic_consistency - domain_penalty


def validate_generation(
    clean_caption: str,
    acoustic_description: str | None,
    adversarial_caption: str | None,
    forbidden_words: set[str],
    cue_words: list[str],
    require_caption_difference: bool,
    acoustic_consistency: float | None,
    semantic_distance: float | None,
    min_acoustic_consistency: float,
    min_semantic_distance: float,
    anti_collapse_phrases: list[str],
    domain_tracker: DomainDiversityTracker | None,
    domain_labels: list[str],
) -> tuple[bool, str | None]:
    if not acoustic_description or not adversarial_caption:
        return False, "parsed_fields_empty"
    if not is_low_level_description(acoustic_description, forbidden_words):
        return False, "acoustic_description_contains_semantic_words"
    if require_caption_difference and not differs_from_clean_caption(clean_caption, adversarial_caption):
        return False, "adversarial_caption_matches_clean_caption"
    if contains_anti_collapse_phrase(adversarial_caption, anti_collapse_phrases):
        return False, "adversarial_caption_collapsed_to_generic_phrase"
    if not is_consistent(acoustic_description, adversarial_caption, cue_words):
        return False, "acoustic_description_adversarial_caption_inconsistent"
    if acoustic_consistency is None:
        return False, "acoustic_consistency_missing"
    if acoustic_consistency < min_acoustic_consistency:
        return False, "acoustic_consistency_too_low"
    if semantic_distance is None:
        return False, "semantic_distance_missing"
    if semantic_distance < min_semantic_distance:
        return False, "semantic_distance_too_low"
    if domain_tracker is not None and domain_tracker.is_overused(domain_labels):
        return False, "domain_family_overused"
    return True, None


def select_best_candidate(
    candidates: list[CandidateRecord],
    metric_computer: MetricComputer,
    min_candidate_diversity_distance: float,
) -> CandidateRecord | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    diverse_candidates: list[CandidateRecord] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if not diverse_candidates:
            diverse_candidates.append(candidate)
            continue
        distances = [
            candidate_text_distance(metric_computer, candidate.adversarial_caption, existing.adversarial_caption)
            for existing in diverse_candidates
        ]
        if all(distance >= min_candidate_diversity_distance for distance in distances):
            diverse_candidates.append(candidate)

    pool = diverse_candidates or sorted(candidates, key=lambda item: item.score, reverse=True)[:1]
    return sorted(pool, key=lambda item: item.score, reverse=True)[0]


def generate_with_retry(
    victim_model: Any,
    model_input: ModelInput,
    max_attempts: int,
    retry_backoff_seconds: float,
    logger: Any,
    request_label: str,
) -> ModelOutput:
    last_exc: Exception | None = None
    for attempt in range(max(max_attempts, 1)):
        started_at = time.time()
        logger.info("Starting CDAB generation: %s (attempt %s/%s)", request_label, attempt + 1, max_attempts)
        try:
            output = victim_model.generate(model_input)
            logger.info("Finished CDAB generation: %s in %.2fs", request_label, time.time() - started_at)
            return output
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "CDAB generation failed: %s in %.2fs with %s: %s",
                request_label,
                time.time() - started_at,
                exc.__class__.__name__,
                exc,
            )
            if attempt + 1 >= max_attempts:
                break
            time.sleep(retry_backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    root_config = load_yaml_config("configs/default.yaml")
    logger = setup_logger("attack_generation")

    seed = int(config["project"].get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)

    model_config = dict(root_config["models"]) if config["model"].get("use_root_model_config", True) else {}
    model_config.update({k: v for k, v in config["model"].items() if k != "use_root_model_config"})
    victim_model = build_victim_model(model_config)
    metric_computer = MetricComputer(config.get("metrics", {}), root_config=root_config, logger=logger)

    samples = build_dataset_samples(root_config=root_config, generation_config=config)
    if not samples:
        raise SystemExit("No samples found for CDAB generation.")

    prompt_template = load_prompt_template(config["attack"]["prompt_template_path"])
    output_csv = Path(config["output"]["output_csv"])
    completed_ids = load_completed_sample_ids(output_csv)
    domain_tracker = DomainDiversityTracker(
        domain_families=dict(config["attack"].get("domain_families", {})),
        recent_window_size=int(config["attack"].get("recent_domain_window_size", 12)),
        max_recent_domain_usage=int(config["attack"].get("max_recent_domain_usage", 3)),
        preferred_domain_count=int(config["attack"].get("preferred_domain_count", 3)),
        domain_penalty_weight=float(config["attack"].get("domain_penalty_weight", 0.18)),
    )
    seed_domain_tracker_from_csv(output_csv, domain_tracker)
    error_log_path = Path(config["output"]["error_jsonl"])
    csv_handle, csv_writer = ensure_csv_writer(output_csv)

    show_progress = bool(config["runtime"].get("show_progress", True))
    progress = tqdm(
        total=len([sample for sample in samples if sample.sample_id not in completed_ids]),
        desc="CDAB Generation",
        unit="sample",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for sample in samples:
            if sample.sample_id in completed_ids:
                continue

            forbidden_words = extract_forbidden_words(
                sample.text,
                config["attack"].get("low_level_forbidden_words", []),
            )
            cue_words = list(config["attack"].get("consistency_cue_words", []))
            max_attempts = int(config["attack"].get("max_attempts_per_sample", 3))
            candidates_per_sample = int(config["attack"].get("candidates_per_sample", 4))
            retry_backoff_seconds = float(config["runtime"].get("retry_backoff_seconds", 3.0))
            require_caption_difference = bool(config["attack"].get("require_caption_difference", True))
            min_acoustic_consistency = float(config["attack"].get("min_acoustic_consistency", 0.45))
            min_semantic_distance = float(config["attack"].get("min_semantic_distance", 0.35))
            min_candidate_diversity_distance = float(config["attack"].get("min_candidate_diversity_distance", 0.18))
            interpretation_styles = list(config["attack"].get("interpretation_styles", []))
            anti_collapse_phrases = list(config["attack"].get("anti_collapse_phrases", []))
            avoid_domains = domain_tracker.overused_domains()
            preferred_domains = domain_tracker.preferred_domains()

            generation_result: CDABGenerationResult | None = None
            last_validation_error: str | None = None

            for attempt in range(max_attempts):
                try:
                    if show_progress:
                        progress.set_postfix(sample=sample.sample_id, stage=f"attempt_{attempt + 1}")
                    candidate_records: list[CandidateRecord] = []
                    chosen_styles = select_diverse_styles(
                        interpretation_styles
                        or [
                            "dynamic physical process",
                            "signal or emission pattern",
                            "system behavior or interaction",
                            "abstract non-obvious mechanism",
                        ],
                        candidates_per_sample,
                    )
                    for candidate_index, interpretation_style in enumerate(chosen_styles, start=1):
                        prompt = build_prompt(
                            prompt_template,
                            interpretation_style=interpretation_style,
                            avoid_domains=avoid_domains,
                            preferred_domains=preferred_domains,
                        )
                        model_input = ModelInput(
                            audio_path=sample.audio_path,
                            user_query="Generate a CDAB acoustic description and adversarial caption.",
                            retrieved_context="",
                            prompt=prompt,
                        )
                        output = generate_with_retry(
                            victim_model=victim_model,
                            model_input=model_input,
                            max_attempts=1,
                            retry_backoff_seconds=retry_backoff_seconds,
                            logger=logger,
                            request_label=f"sample={sample.sample_id} candidate={candidate_index} style={interpretation_style}",
                        )
                        acoustic_description, adversarial_caption = parse_generation_output(output.text)
                        acoustic_consistency, semantic_distance = (None, None)
                        if acoustic_description and adversarial_caption:
                            acoustic_consistency, semantic_distance = metric_computer.compute(
                                clean_caption=sample.text,
                                acoustic_description=acoustic_description,
                                adversarial_caption=adversarial_caption,
                            )
                        domain_labels = domain_tracker.infer_domains(adversarial_caption or "")
                        domain_penalty = domain_tracker.penalty(domain_labels)
                        valid, reason = validate_generation(
                            clean_caption=sample.text,
                            acoustic_description=acoustic_description,
                            adversarial_caption=adversarial_caption,
                            forbidden_words=forbidden_words,
                            cue_words=cue_words,
                            require_caption_difference=require_caption_difference,
                            acoustic_consistency=acoustic_consistency,
                            semantic_distance=semantic_distance,
                            min_acoustic_consistency=min_acoustic_consistency,
                            min_semantic_distance=min_semantic_distance,
                            anti_collapse_phrases=anti_collapse_phrases,
                            domain_tracker=domain_tracker,
                            domain_labels=domain_labels,
                        )
                        if valid:
                            assert acoustic_description is not None
                            assert adversarial_caption is not None
                            assert acoustic_consistency is not None
                            assert semantic_distance is not None
                            candidate_records.append(
                                CandidateRecord(
                                    acoustic_description=acoustic_description,
                                    adversarial_caption=adversarial_caption,
                                    acoustic_consistency=acoustic_consistency,
                                    semantic_distance=semantic_distance,
                                    interpretation_style=interpretation_style,
                                    domain_labels=domain_labels,
                                    domain_penalty=domain_penalty,
                                    score=score_candidate(acoustic_consistency, semantic_distance, domain_penalty),
                                )
                            )
                        else:
                            last_validation_error = reason
                            logger.warning(
                                "Candidate validation failed for sample=%s style=%s: %s",
                                sample.sample_id,
                                interpretation_style,
                                reason,
                            )

                    best_candidate = select_best_candidate(
                        candidate_records,
                        metric_computer=metric_computer,
                        min_candidate_diversity_distance=min_candidate_diversity_distance,
                    )
                    if best_candidate is not None:
                        generation_result = CDABGenerationResult(
                            sample_id=sample.sample_id,
                            dataset=sample.dataset_name,
                            audio_path=sample.audio_path,
                            clean_caption=sample.text,
                            acoustic_description=best_candidate.acoustic_description,
                            adversarial_caption=best_candidate.adversarial_caption,
                            acoustic_consistency=best_candidate.acoustic_consistency,
                            semantic_distance=best_candidate.semantic_distance,
                        )
                        domain_tracker.record_caption(best_candidate.adversarial_caption)
                        break
                    logger.warning("No valid diverse candidate survived for sample=%s on attempt=%s", sample.sample_id, attempt + 1)
                except Exception as exc:  # noqa: BLE001
                    last_validation_error = f"{exc.__class__.__name__}: {exc}"
                    logger.warning("Generation attempt failed for sample=%s: %s", sample.sample_id, last_validation_error)
                    if attempt + 1 < max_attempts:
                        time.sleep(retry_backoff_seconds * (2**attempt))

            if generation_result is None:
                append_error(
                    error_log_path,
                    {
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset_name,
                        "audio_path": sample.audio_path,
                        "clean_caption": sample.text,
                        "error": last_validation_error or "unknown_generation_error",
                        "raw_output_preview": output.text[:500] if 'output' in locals() and getattr(output, 'text', None) else None,
                    },
                )
                if not bool(config["runtime"].get("continue_on_error", True)):
                    raise RuntimeError(f"CDAB generation failed for sample {sample.sample_id}: {last_validation_error}")
                progress.update(1)
                continue

            csv_writer.writerow(generation_result.to_csv_row())
            csv_handle.flush()
            completed_ids.add(sample.sample_id)
            progress.update(1)
            if show_progress:
                progress.set_postfix(sample=sample.sample_id, stage="saved")
    finally:
        progress.close()
        csv_handle.close()

    logger.info("CDAB generation finished. Output CSV: %s", output_csv)


if __name__ == "__main__":
    main()
