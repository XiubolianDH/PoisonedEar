from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model_name: str
    base_config: str
    api_base_env: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    slug: str
    poisoned_manifest: str
    malicious_metadata: str
    poison_count: int


@dataclass(frozen=True)
class DefenseSpec:
    key: str
    kind: str
    description: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt_4o_audio", "openai", "gpt-4o-audio-preview", "configs/openai_gpt4o_audio.yaml"),
    ModelSpec("gpt_4o_mini_audio", "openai", "gpt-4o-mini-audio-preview", "configs/openai_gpt4o_mini_audio.yaml"),
    ModelSpec("gpt_audio", "openai", "gpt-audio", "configs/openai_gpt4o_audio.yaml"),
    ModelSpec("gpt_audio_mini", "openai", "gpt-audio-mini", "configs/openai_gpt4o_mini_audio.yaml"),
    ModelSpec("gemini_2_0_flash", "gemini", "gemini-2.0-flash", "configs/gemini_2_5_flash.yaml"),
    ModelSpec("gemini_2_0_flash_lite", "gemini", "gemini-2.0-flash-lite", "configs/gemini_2_5_flash.yaml"),
    ModelSpec("gemini_2_5_flash", "gemini", "gemini-2.5-flash", "configs/gemini_2_5_flash.yaml"),
    ModelSpec("gemini_2_5_pro", "gemini", "gemini-2.5-pro", "configs/gemini_2_5_pro.yaml"),
    ModelSpec("gemini_2_5_flash_lite", "gemini", "gemini-2.5-flash-lite", "configs/gemini_2_5_flash.yaml"),
    ModelSpec("gemini_3_pro_preview", "gemini", "gemini-3-pro-preview", "configs/gemini_2_5_pro.yaml"),
    ModelSpec("gemini_3_flash_preview", "gemini", "gemini-3-flash-preview", "configs/gemini_2_5_flash.yaml"),
    ModelSpec(
        "anygpt",
        "openai_compatible",
        "AnyGPT",
        "configs/openai_gpt4o_audio.yaml",
        api_base_env="ANYGPT_API_BASE",
        api_key_env="ANYGPT_API_KEY",
    ),
    ModelSpec(
        "qwen3_omni_flash",
        "qwen",
        "qwen3-omni-flash",
        "configs/openai_gpt4o_mini_audio.yaml",
        api_base_env="QWEN_API_BASE",
        api_key_env="QWEN_API_KEY",
    ),
)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "wavcaps",
        "wavcaps_electronic_synthetic_15pct",
        "attack_experiments/wavcaps_electronic_synthetic_rates/rate_15pct/wavcaps_poisoned_manifest.jsonl",
        "malicious_dataset/unified_electronic_synthetic_35/final_dataset/query_subset_20.json",
        poison_count=24,
    ),
    DatasetSpec(
        "clotho",
        "clotho_vehicle_transport_15pct",
        "attack_experiments/clotho_vehicle_transport_rates/rate_15pct/clotho_poisoned_manifest.jsonl",
        "malicious_dataset/wavcaps_clotho_esc50_to_clotho_vehicle_transport_556/final_dataset/query_subset_20.json",
        poison_count=556,
    ),
    DatasetSpec(
        "audioset",
        "audioset_music_instrument_15pct",
        "attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl",
        "malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json",
        poison_count=116,
    ),
    DatasetSpec(
        "esc50",
        "esc50_animal_bio_15pct",
        "attack_experiments/esc50_animal_bio_rates/rate_15pct/esc50_poisoned_manifest.jsonl",
        "malicious_dataset/wavcaps_clotho_esc50_to_esc50_animal_bio_72/final_dataset/query_subset_20.json",
        poison_count=72,
    ),
)


DEFENSE_SPECS: tuple[DefenseSpec, ...] = (
    DefenseSpec("no_defense", "none", "Use the retrieved captions directly."),
    DefenseSpec("paraphrasing", "query_rewrite", "Rewrite the user query before generation while keeping retrieval fixed."),
    DefenseSpec("perplexity_filtering", "document_filter", "Remove high-perplexity captions after retrieval."),
)


def model_keys() -> list[str]:
    return [spec.key for spec in MODEL_SPECS]


def dataset_keys() -> list[str]:
    return [spec.key for spec in DATASET_SPECS]


def defense_keys() -> list[str]:
    return [spec.key for spec in DEFENSE_SPECS]


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")


def get_dataset_spec(key: str) -> DatasetSpec:
    for spec in DATASET_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown dataset key: {key}")


def get_defense_spec(key: str) -> DefenseSpec:
    for spec in DEFENSE_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown defense key: {key}")
