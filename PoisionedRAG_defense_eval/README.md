# PoisionedRAG Defense Eval

This folder is a self-contained defense workspace for evaluating mitigation methods against PoisonedEar-style retrieval poisoning.

Currently implemented defenses:

- `no_defense`
- `paraphrasing`
- `perplexity_filtering`

Supported evaluation axes in the current runner:

- model
- dataset
- defense method

For the full project overview and reproduction order, see the repository-level [README](../README.md).

## Quick Start

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Print the planned matrix

```bash
python scripts/print_experiment_plan.py
```

### Run a preflight check

```bash
python scripts/preflight_check.py --config configs/default.yaml
```

### Run a smoke test without remote generation APIs

```bash
python run_defense_eval.py \
  --config configs/default.yaml \
  --generator-backend local \
  --text-similarity-backend lexical \
  --max-samples 2 \
  --skip-generation-errors \
  --write-run-manifest
```

### Run the full defense matrix

```bash
python run_defense_eval.py \
  --config configs/default.yaml \
  --output-root outputs/full_defense_eval \
  --write-run-manifest
```

## Important Outputs

- per-run `sample_results.jsonl`
- per-run `summary.csv`
- root-level `global_summary.csv`
- root-level `leaderboard.csv`
- root-level `run_manifest.json`
- root-level `progress.json`

## Notes

- This workspace intentionally keeps defense logic isolated from the attack-generation pipeline.
- It reuses datasets and poisoned manifests by path, but stores configs, outputs, and defense implementations locally inside this folder.
