# PoisionedRAG Multimodel ASR Eval

This workspace is for broader attack-success benchmarking across a larger model pool than the base pipeline.

It mirrors the main `PoisionedRAG/` structure, but keeps multimodel attack-success evaluation outputs isolated.

For the full project overview and reproduction order, see the repository-level [README](../README.md).

## Typical Workflow

```bash
cp .env.example .env
python bootstrap_datasets.py --config configs/default.yaml
python run_attack.py --config configs/default.yaml --prepare-datasets
python run_cross_dataset_wavcaps_attack_eval.py \
  --config configs/default.yaml \
  --poisoned-manifest attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json \
  --results-csv outputs/manual_multimodel_asr/results.csv \
  --summary-json outputs/manual_multimodel_asr/summary.json \
  --error-log outputs/manual_multimodel_asr/errors.jsonl \
  --top-k 5 \
  --continue-on-error
```

## Typical Use Cases

- compare attack success across more models
- keep multi-model evaluation outputs separate from the main workspace
- iterate on wrappers, logging, or result aggregation without modifying the base pipeline
