# PoisionedRAG

This folder contains the main PoisonedEar attack-generation and attack-evaluation pipeline for Audio RAG systems.

It is the core workspace for:

- dataset preparation
- clean and poisoned manifest construction
- CDAB-style adversarial caption generation
- malicious dataset packaging
- poison-rate sweeps
- cross-dataset attack evaluation
- top-k sensitivity studies

## Start Here

For the full project overview and the end-to-end reproduction order, see the repository-level [README](../README.md).

## Main Entry Points

### Prepare datasets

```bash
python bootstrap_datasets.py --config configs/default.yaml
python run_attack.py --config configs/default.yaml --prepare-datasets
python test_clean_datasets.py
```

### Download the full local datasets

```bash
python download_full_datasets.py \
  --datasets wavcaps_freesound clotho_full esc50 audioset \
  --root data/full_datasets
```

### Generate adversarial captions

```bash
python attack_generation/generate_cdab_captions.py --config attack_generation/config.yaml
```

### Build a malicious dataset package

```bash
python attack_generation/build_malicious_dataset.py \
  --input-csv attack_generation/generated/cdab_attack_samples_wavcaps_500.csv \
  --output-root malicious_dataset/wavcaps_cdab_500
```

### Build mixed poisoned manifests

```bash
python attack_generation/build_mixed_poisoned_dataset.py \
  --clean-manifest outputs/clean_dataset/manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_cdab_500/metadata.json \
  --poisoned-manifest outputs/poisoned_dataset/manifest.jsonl \
  --wavcaps-poisoned-manifest outputs/manifests/wavcaps_poisoned_manifest.jsonl
```

### Run a representative cross-dataset attack evaluation

```bash
python run_cross_dataset_wavcaps_attack_eval.py \
  --config configs/default.yaml \
  --poisoned-manifest attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json \
  --results-csv outputs/manual_cross_dataset/results.csv \
  --summary-json outputs/manual_cross_dataset/summary.json \
  --error-log outputs/manual_cross_dataset/errors.jsonl \
  --top-k 5 \
  --continue-on-error
```

### Run a representative top-k study

```bash
python topk_impact_study/run_topk_impact_eval.py \
  --poisoned-manifest attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json \
  --output-dir outputs/topk_impact_audioset_music_instrument_15pct \
  --poison-rate-pct 15 \
  --poison-count 116 \
  --k-values 1 5 10 15 20 \
  --continue-on-error
```

## Important Directories

- `attack_generation/`: adversarial caption generation and poisoning-data builders
- `attack_experiments/`: rate-specific poisoned manifests and experiment plans
- `malicious_dataset/`: finalized malicious metadata packages
- `outputs/`: clean, poisoned, sweep, and summary outputs
- `topk_impact_study/`: fixed poison-rate top-k experiments

## Notes

- The project path is intentionally spelled `PoisionedRAG` to match the current repository layout.
- Several sweep scripts support `AUDIO_RAG_PYTHON` for selecting a non-default interpreter.
- Some experiments assume the full dataset tree already exists in `data/full_datasets/`.
