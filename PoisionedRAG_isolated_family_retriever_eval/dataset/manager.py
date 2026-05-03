from __future__ import annotations

import json
from pathlib import Path

from dataset.audio_stats import attach_stats, stats_to_json
from dataset.loaders import load_configured_datasets
from dataset.types import AudioSample
from utils.config import ensure_dir


class DatasetManager:
    def __init__(self, data_config: dict) -> None:
        self.data_config = data_config
        self.clean_dir = ensure_dir(data_config["clean_dir"])
        self.poisoned_dir = ensure_dir(data_config["poisoned_dir"])
        self.stats_json = Path(data_config["stats_json"])
        self.clean_manifest = Path(data_config["manifest_jsonl"])
        self.poisoned_manifest = Path(data_config["poisoned_manifest_jsonl"])
        self.dataset_manifests_dir = ensure_dir(data_config.get("dataset_manifests_dir", "data/manifests"))

    def prepare(self) -> tuple[list[AudioSample], list[AudioSample]]:
        clean_samples = attach_stats(load_configured_datasets(self.data_config))
        poisoned_samples = [self._clone_sample(sample) for sample in clean_samples]
        self._write_manifest(self.clean_manifest, clean_samples)
        self._write_manifest(self.poisoned_manifest, poisoned_samples)
        self._write_per_dataset_manifests(clean_samples, poisoned_samples)
        self._write_stats(clean_samples, poisoned_samples)
        return clean_samples, poisoned_samples

    def load_manifest(self, mode: str, dataset_name: str | None = None) -> list[AudioSample]:
        if dataset_name:
            suffix = "clean" if mode == "clean" else "poisoned"
            manifest = self.dataset_manifests_dir / f"{dataset_name}_{suffix}.jsonl"
        else:
            manifest = self.clean_manifest if mode == "clean" else self.poisoned_manifest
        if not manifest.exists():
            return []
        samples: list[AudioSample] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                stats_raw = raw.pop("stats", None)
                sample = AudioSample(**raw)
                if stats_raw:
                    from dataset.types import AudioStats

                    sample.stats = AudioStats(**stats_raw)
                samples.append(sample)
        return samples

    def _write_manifest(self, path: Path, samples: list[AudioSample]) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample.to_json(), ensure_ascii=True) + "\n")

    def _write_stats(self, clean_samples: list[AudioSample], poisoned_samples: list[AudioSample]) -> None:
        ensure_dir(self.stats_json.parent)
        payload = {
            "clean_dataset": stats_to_json(clean_samples),
            "poisoned_dataset": stats_to_json(poisoned_samples),
        }
        with self.stats_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    @staticmethod
    def _clone_sample(sample: AudioSample) -> AudioSample:
        payload = sample.to_json()
        from dataset.types import AudioStats

        stats_payload = payload.pop("stats", None)
        clone = AudioSample(**payload)
        clone.is_poisoned = False
        if stats_payload:
            clone.stats = AudioStats(**stats_payload)
        return clone

    def _write_per_dataset_manifests(self, clean_samples: list[AudioSample], poisoned_samples: list[AudioSample]) -> None:
        clean_groups: dict[str, list[AudioSample]] = {}
        poisoned_groups: dict[str, list[AudioSample]] = {}
        for sample in clean_samples:
            clean_groups.setdefault(sample.dataset_name, []).append(sample)
        for sample in poisoned_samples:
            poisoned_groups.setdefault(sample.dataset_name, []).append(sample)
        for dataset_name, samples in clean_groups.items():
            self._write_manifest(self.dataset_manifests_dir / f"{dataset_name}_clean.jsonl", samples)
        for dataset_name, samples in poisoned_groups.items():
            self._write_manifest(self.dataset_manifests_dir / f"{dataset_name}_poisoned.jsonl", samples)
