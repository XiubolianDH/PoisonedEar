# PoisonedEar
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Research-orange)
![Conference](https://img.shields.io/badge/CCS-2026-red)
![Task](https://img.shields.io/badge/Task-AudioRAG-purple)
![Attack](https://img.shields.io/badge/Type-Retrieval%20Poisoning%20Attack-critical)

PoisonedEar is a research workspace for studying retrieval poisoning attacks against audio retrieval-augmented generation (Audio RAG) systems, together with follow-up analyses on model families, retriever families, top-k sensitivity, multi-model ASR, and defensive filtering.

This repository is designed as an experiment suite rather than a single script. It includes the main attack pipeline, rate-sweep builders, top-k studies, model-family comparisons, retriever-family comparisons, multi-model attack-success evaluation, and a standalone defense workspace.

# Framework Description

<img width="1364" alt="AudioRAG" src="https://github.com/user-attachments/assets/c1e2299f-ea6d-4f75-9d4f-914d458db5a1" />

Figure above illustrates the overall pipeline of our proposed retrieval poisoning attack against Audio Retrieval-Augmented Generation (AudioRAG) systems.

The attack is initiated by an adversary who constructs malicious audio–text pairs. Specifically, the attacker first selects acoustically matched target audio samples and then generates target-aligned causal descriptions that are semantically shifted toward a desired malicious concept. These descriptions are carefully designed to preserve acoustic plausibility while introducing controlled semantic drift. The resulting audio–text pairs are then injected into the external knowledge database, forming a poisoned retrieval corpus.

At inference time, a user submits an audio query (e.g., “What is this sound?”). The retriever retrieves top-k audio descriptions from the knowledge database based on acoustic similarity. Due to the poisoning, the retrieved context may contain adversarial descriptions that are semantically misleading yet acoustically consistent with the query.

The multimodal large language model (MLLM) conditions its generation on both the input audio and the retrieved textual context. As a result, the poisoned retrieval context can bias the model’s reasoning process, leading to incorrect or malicious responses.

This pipeline highlights a critical vulnerability of AudioRAG systems: although retrieval is grounded in acoustic similarity, the generation stage heavily relies on retrieved textual descriptions, making the system susceptible to cross-modal semantic manipulation.


## At a Glance

- Problem: retrieval poisoning against Audio RAG systems
- Modalities: audio queries, retrieved audio-text context, multimodal generation
- Datasets: WavCaps, Clotho, AudioSet, ESC-50
- Retrievers: CLAP, AudioCLIP, PANNs, wav2vec2, and a local `spectral` smoke-test retriever
- Model backends: OpenAI, Gemini, OpenAI-compatible endpoints, and local smoke-test backends
- Main outputs: poisoned manifests, per-query attack metrics, rate-sweep summaries, top-k summaries, family-matrix summaries, and defense leaderboards

The repository is organized as several closely related experiment workspaces:

- `PoisionedRAG/`: main attack-generation and attack-evaluation pipeline
- `PoisionedRAG_gpt_family_eval/`: fixed-matrix evaluation for GPT and Gemini model families
- `PoisionedRAG_isolated_family_retriever_eval/`: isolated retriever-family and top-k matrix studies
- `PoisionedRAG_multimodel_asr_eval/`: larger multi-model attack-success evaluation
- `PoisionedRAG_defense_eval/`: defense evaluation under retrieval poisoning

The directory names intentionally follow the current repository layout, including the existing `PoisionedRAG` spelling in paths.

## Quick Start

If you want the shortest path to a working local setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./PoisionedRAG
cp PoisionedRAG/.env.example PoisionedRAG/.env
cd PoisionedRAG
python bootstrap_datasets.py --config configs/default.yaml --samples-per-dataset 12
python run_attack.py --config configs/default.yaml --prepare-datasets
python test_clean_datasets.py
```

If you are preparing results for a paper-style run, skip the bootstrap path and go directly to the full-data workflow in the sections below.

## What This Repository Contains

At a high level, the project supports the following workflow:

1. Download or bootstrap clean audio-caption datasets.
2. Build clean manifests and retriever-ready metadata.
3. Generate adversarial captions with a CDAB-style attack-generation workspace.
4. Materialize malicious audio-text datasets and merge them into mixed poisoned manifests.
5. Sweep poison rates or fixed `top-k` retrieval settings.
6. Evaluate attack success across datasets, retrievers, and model families.
7. Evaluate defenses such as paraphrasing and perplexity filtering.

Primary datasets used in the codebase:

- WavCaps
- Clotho
- AudioSet
- ESC-50

Primary retrievers exposed in the codebase:

- CLAP
- AudioCLIP
- PANNs
- wav2vec2
- a lightweight local `spectral` retriever for smoke tests

Primary model backends exposed in the codebase:

- OpenAI audio-capable models
- Gemini audio-capable models
- OpenAI-compatible endpoints such as Qwen or AnyGPT
- local heuristic or smoke-test backends in some subprojects

## Repository Layout

```text
PoisonedEar/
├── README.md
├── PoisionedRAG/
│   ├── attack_generation/          # adversarial caption planning and generation
│   ├── attack_experiments/         # generated poison-rate manifests
│   ├── classified_dataset/         # class-conditioned metadata views
│   ├── data/                       # clean/full datasets and unified bundles
│   ├── malicious_dataset/          # finalized malicious metadata and summaries
│   ├── outputs/                    # evaluation outputs
│   ├── topk_impact_study/          # fixed poison-rate top-k sweeps
│   ├── run_attack.py
│   ├── run_ablation.py
│   ├── run_cross_dataset_wavcaps_attack_eval.py
│   ├── run_clean_wavcaps_eval.py
│   └── download_full_datasets.py
├── PoisionedRAG_gpt_family_eval/
├── PoisionedRAG_isolated_family_retriever_eval/
├── PoisionedRAG_multimodel_asr_eval/
└── PoisionedRAG_defense_eval/
```

## Environment Setup

This project targets Python 3.10+.

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the main package dependencies:

```bash
python -m pip install -e ./PoisionedRAG
```

Install defense-specific extras when running the defense workspace:

```bash
python -m pip install -r PoisionedRAG_defense_eval/requirements.txt
```

### 2. Configure API credentials

Each main experiment workspace provides an `.env.example`. For the main pipeline:

```bash
cp PoisionedRAG/.env.example PoisionedRAG/.env
```

Fill in the values you actually use:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
QWEN_API_BASE=...
QWEN_API_KEY=...
```

The main codebase auto-loads a local `.env` from the corresponding repo root when config loading happens.

### 3. Choose a Python entrypoint

All example commands below use plain `python`. If you keep a dedicated interpreter elsewhere, set:

```bash
export AUDIO_RAG_PYTHON=/absolute/path/to/python
```

Several sweep scripts already read `AUDIO_RAG_PYTHON` and fall back to a hard-coded local interpreter if it is unset.

## Data Preparation

You can prepare data in two ways, depending on whether you want a small smoke-test setup or the full dataset materialization.

### Option A. Bootstrap a small local subset

This is the fastest way to validate the pipeline.

```bash
cd PoisionedRAG
python bootstrap_datasets.py --config configs/default.yaml --samples-per-dataset 12
python run_attack.py --config configs/default.yaml --prepare-datasets
python test_clean_datasets.py
```

What this does:

1. Downloads small subsets from the upstream Hugging Face datasets.
2. Writes local audio files and metadata into `data/bootstrap/`.
3. Builds clean and poisoned manifest scaffolding in `outputs/`.
4. Runs a basic dataset-integrity check.

### Option B. Materialize the full local datasets

Use this path before running the main reported experiments.

```bash
cd PoisionedRAG
python download_full_datasets.py \
  --datasets wavcaps_freesound clotho_full esc50 audioset \
  --root data/full_datasets \
  --cache-root /scratch/$USER/tmp_cache/audio_rag_hf
```

Recommended dataset access check:

```bash
python verify_dataset_access.py --metadata data/full_datasets/wavcaps_freesound/metadata.jsonl --limit 5
python verify_dataset_access.py --metadata data/full_datasets/clotho_full/metadata.csv --limit 5
python verify_dataset_access.py --metadata data/full_datasets/esc50/metadata.csv --limit 5
python verify_dataset_access.py --metadata data/full_datasets/audioset/metadata.csv --limit 5
```

Then build manifests:

```bash
python run_attack.py --config configs/default.yaml --prepare-datasets
```

Expected manifest outputs:

- `PoisionedRAG/outputs/clean_dataset/manifest.jsonl`
- `PoisionedRAG/outputs/poisoned_dataset/manifest.jsonl`
- `PoisionedRAG/outputs/manifests/*.jsonl`

## Main Experimental Workflow

The main project is easiest to understand as seven stages.

### Stage 1. Prepare the clean base corpus

Use either the bootstrap or full-data path above, then confirm that `outputs/manifests/` contains clean per-dataset manifests such as:

- `wavcaps_clean.jsonl`
- `clotho_clean.jsonl`
- `audioset_clean.jsonl`
- `esc50_clean.jsonl`

### Stage 2. Generate adversarial captions in the isolated attack workspace

The attack-generation workspace keeps prompt-based caption synthesis separate from the main RAG runtime.

```bash
cd PoisionedRAG
python attack_generation/generate_cdab_captions.py --config attack_generation/config.yaml
```

Key outputs:

- `attack_generation/generated/*.csv`
- `attack_generation/logs/*.jsonl`

Each generated CSV contains fields such as:

- `sample_id`
- `dataset`
- `audio_path`
- `clean_caption`
- `acoustic_description`
- `adversarial_caption`
- `acoustic_consistency`
- `semantic_distance`

### Stage 3. Build a malicious dataset from generated captions

Convert the generated CSV into a reusable malicious dataset package:

```bash
cd PoisionedRAG
python attack_generation/build_malicious_dataset.py \
  --input-csv attack_generation/generated/cdab_attack_samples_wavcaps_500.csv \
  --output-root malicious_dataset/wavcaps_cdab_500
```

Expected outputs:

- `malicious_dataset/<dataset_name>/audio/`
- `malicious_dataset/<dataset_name>/metadata.json`

### Stage 4. Merge clean manifests with malicious metadata

Build a mixed poisoned manifest that can be indexed and retrieved like the clean corpus:

```bash
cd PoisionedRAG
python attack_generation/build_mixed_poisoned_dataset.py \
  --clean-manifest outputs/clean_dataset/manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_cdab_500/metadata.json \
  --poisoned-manifest outputs/poisoned_dataset/manifest.jsonl \
  --wavcaps-poisoned-manifest outputs/manifests/wavcaps_poisoned_manifest.jsonl
```

This step appends poisoned WavCaps-derived samples to:

- the global mixed manifest
- a WavCaps-only poisoned manifest

### Stage 5. Build poison-rate experiment plans

The repository already includes class- and dataset-specific rate builders. A typical builder:

```bash
cd PoisionedRAG
python attack_generation/build_audioset_ambience_environment_rate_experiments.py
```

Equivalent builders exist for:

- AudioSet classes:
  `ambience_environment`, `animal_bio`, `electronic_synthetic`, `human_voice_speech`, `impact_material`, `machine_mechanical`, `music_instrument`, `vehicle_transport`, `water_weather`
- WavCaps classes:
  `ambience_environment`, `animal_bio`, `electronic_synthetic`, `human_voice_speech`, `impact_material`, `machine_mechanical`, `music_instrument`, `vehicle_transport`, `water_weather`
- Clotho classes:
  `music_instrument`, `vehicle_transport`
- ESC-50 classes:
  `animal_bio`

Each builder writes a directory under `attack_experiments/` with:

- per-rate poisoned manifests
- per-rate malicious subsets
- `experiment_plan.json`

Default poison rates in the builder scripts are:

- `1%`
- `5%`
- `10%`
- `15%`
- `20%`

### Stage 6. Run poison-rate sweeps

Use the matching sweep launcher after the plan is built. Representative example:

```bash
cd PoisionedRAG
python attack_generation/run_audioset_ambience_environment_rate_sweep.py
```

That script will:

1. Read `attack_experiments/audioset_ambience_environment_rates/experiment_plan.json`.
2. Evaluate each poisoned manifest with `run_cross_dataset_wavcaps_attack_eval.py`.
3. Write per-rate `results_rate_XXpct.csv`.
4. Write per-rate `results_rate_XXpct.summary.json`.
5. Aggregate everything into `comparison_summary.csv` and `comparison_summary.json`.

Representative output directory:

- `PoisionedRAG/outputs/audioset_ambience_environment_rate_sweep/`

### Stage 7. Run single-query or ablation experiments

Smoke-test the full Audio RAG stack on one query:

```bash
cd PoisionedRAG
python run_attack.py \
  --config configs/default.yaml \
  --query-audio path/to/query.wav \
  --query-text "What is this sound?" \
  --expected-answer "clean caption or expected answer" \
  --attack-target "target adversarial answer" \
  --mode poisoned \
  --dataset-name wavcaps \
  --top-k 5 \
  --run-name manual_attack_check
```

Run ablations over retriever, `top-k`, poison count, noise level, and compression bitrate:

```bash
cd PoisionedRAG
python run_ablation.py \
  --config configs/default.yaml \
  --query-audio path/to/query.wav \
  --query-text "What is this sound?" \
  --expected-answer "clean caption or expected answer" \
  --attack-target "target adversarial answer" \
  --mode poisoned \
  --run-name ablation_run
```

## Main Published Evaluation Entry Points

The following scripts are the main experiment entry points beyond the basic pipeline:

### 1. Clean evaluation

```bash
cd PoisionedRAG
python run_clean_wavcaps_eval.py
```

Purpose:

- evaluate retrieval/generation behavior on clean WavCaps-only conditions

### 2. Cross-dataset attack evaluation

```bash
cd PoisionedRAG
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

Primary metrics written by this runner:

- `ASR-R`
- `ASR-G`
- `recall@k`
- `CLAP` response similarity
- response similarity to clean vs adversarial captions
- attack margin

### 3. Mixed-WavCaps attack evaluation

```bash
cd PoisionedRAG
python run_mixed_wavcaps_attack_eval.py
python run_mixed_wavcaps_attack_eval_wav2vec2.py
```

Purpose:

- compare mixed-manifest attacks with different retriever backends

## Top-k Impact Study

The fixed poison-rate top-k study is isolated in `PoisionedRAG/topk_impact_study/`.

Recommended representative setting already documented in the codebase:

- dataset: AudioSet
- class: `music_instrument`
- poison rate: `15%`
- poison count: `116`

Run the sweep:

```bash
cd PoisionedRAG
python topk_impact_study/run_topk_impact_eval.py \
  --poisoned-manifest attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json \
  --output-dir outputs/topk_impact_audioset_music_instrument_15pct \
  --poison-rate-pct 15 \
  --poison-count 116 \
  --k-values 1 5 10 15 20 \
  --continue-on-error
```

Build summary tables:

```bash
python topk_impact_study/build_topk_impact_metrics.py \
  --input-dir outputs/topk_impact_audioset_music_instrument_15pct \
  --poison-rate-pct 15 \
  --poison-count 116 \
  --k-values 1 5 10 15 20
```

Important outputs:

- `comparison_summary_topk.json`
- `comparison_summary_topk.csv`

## GPT and Gemini Family Evaluation

`PoisionedRAG_gpt_family_eval/` fixes the poison rate at `15%` and the retrieval depth at `k=5`, then sweeps multiple model families and retrievers over four benchmark datasets.

### GPT-family matrix

```bash
cd PoisionedRAG_gpt_family_eval
python attack_generation/run_gpt_family_fixed_k5_matrix.py \
  --output-root outputs/gpt_family_fixed_k5 \
  --top-k 5 \
  --continue-on-error
```

Default GPT-family models in the script:

- `gpt-4o-audio-preview`
- `gpt-4o-mini-audio-preview`

### Gemini-family matrix

```bash
cd PoisionedRAG_gpt_family_eval
python attack_generation/run_gemini_family_fixed_k5_matrix.py \
  --output-root outputs/gemini_family_fixed_k5 \
  --index-cache-dir outputs/shared_retriever_index_cache \
  --top-k 5 \
  --continue-on-error
```

Default Gemini-family models in the script:

- `gemini-2.5-flash`
- `gemini-2.5-pro`

Both matrix runners evaluate:

- datasets: `wavcaps`, `audioset`, `clotho`, `esc50`
- retrievers: `clap`, `audioclip`, `panns`, `wav2vec2`
- fixed poison rate: `15%`
- fixed `top-k`: `5`

Typical aggregate outputs:

- `gpt_family_overall_asr_recall_k5.csv`
- `gpt_family_overall_asr_recall_k5.json`
- `gemini_family_overall_asr_recall_k5.csv`
- `gemini_family_overall_asr_recall_k5.json`

## Isolated Retriever-Family Evaluation

`PoisionedRAG_isolated_family_retriever_eval/` focuses on the effect of retriever choice while sweeping top-k values for GPT-like and Gemini-like families.

Run the matrix:

```bash
cd PoisionedRAG_isolated_family_retriever_eval
python topk_impact_study/run_retriever_family_matrix.py \
  --output-root outputs/retriever_family_matrix \
  --families gpt gemini \
  --datasets wavcaps audioset clotho esc50 \
  --retrievers clap audioclip panns wav2vec2 \
  --k-values 1 5 10 15 20 \
  --continue-on-error
```

This orchestrates:

1. `topk_impact_study/run_topk_impact_eval.py`
2. `topk_impact_study/build_topk_impact_metrics.py`

for every family-dataset-retriever combination.

## Multi-Model ASR Evaluation

`PoisionedRAG_multimodel_asr_eval/` mirrors the main pipeline but is intended for broader attack-success benchmarking across a larger model pool.

Typical workflow:

```bash
cd PoisionedRAG_multimodel_asr_eval
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

Use this workspace when you want:

- a broader multi-model attack success benchmark
- separate outputs from the main `PoisionedRAG/` workspace
- independent iteration on model wrappers or result aggregation

## Defense Evaluation

`PoisionedRAG_defense_eval/` is a self-contained workspace for testing defenses against PoisonedEar-style retrieval poisoning.

Current defense methods in the folder:

- `no_defense`
- `paraphrasing`
- `perplexity_filtering`

### Step 1. Install defense dependencies

```bash
cd PoisionedRAG_defense_eval
python -m pip install -r requirements.txt
```

### Step 2. Print the planned matrix and run a preflight check

```bash
python scripts/print_experiment_plan.py
python scripts/preflight_check.py --config configs/default.yaml
```

### Step 3. Run a smoke test without remote generation APIs

```bash
python run_defense_eval.py \
  --config configs/default.yaml \
  --generator-backend local \
  --text-similarity-backend lexical \
  --max-samples 2 \
  --skip-generation-errors \
  --write-run-manifest
```

### Step 4. Run the full defense matrix

```bash
python run_defense_eval.py \
  --config configs/default.yaml \
  --output-root outputs/full_defense_eval \
  --write-run-manifest
```

Optional filtering flags:

- `--models ...`
- `--datasets ...`
- `--defenses ...`
- `--retriever ...`
- `--top-k ...`
- `--max-samples ...`

Key outputs:

- per-run `sample_results.jsonl`
- per-run `summary.csv`
- root-level `global_summary.csv`
- root-level `leaderboard.csv`
- root-level `run_manifest.json`
- root-level `progress.json`

## Recommended Reproduction Order

If you are reproducing the project from scratch, the most stable order is:

1. Set up the Python environment and `.env` files.
2. Materialize or bootstrap the clean datasets in `PoisionedRAG/`.
3. Run `run_attack.py --prepare-datasets`.
4. Generate CDAB captions in `attack_generation/`.
5. Build malicious dataset packages.
6. Build mixed poisoned manifests.
7. Build poison-rate experiment plans.
8. Run dataset/class-specific poison-rate sweeps.
9. Run the fixed-rate top-k study.
10. Run GPT/Gemini family matrices.
11. Run isolated retriever-family matrices.
12. Run the defense evaluation matrix.

## Outputs and Artifacts

You will mainly interact with these artifact types:

- clean manifests in `outputs/clean_dataset/` and `outputs/manifests/`
- poisoned manifests in `outputs/poisoned_dataset/` and `attack_experiments/*/rate_*pct/`
- malicious metadata in `malicious_dataset/*/metadata.json`
- per-query results in `results*.csv`
- aggregate summaries in `summary*.json`, `comparison_summary*.csv`, and `comparison_summary*.json`
- matrix-level summaries in `leaderboard.csv`, `global_summary.csv`, and family-level aggregate CSV/JSON files

## Notes and Practical Tips

- Some scripts assume the full datasets already exist under `PoisionedRAG/data/full_datasets/`.
- Several orchestration scripts cache retriever indices to reduce repeated indexing work.
- Many folders already contain prior outputs; keep new experiment outputs in a new subdirectory if you want a clean rerun.
- Some local scripts in the repository still contain machine-specific default paths or output names. Overriding them from the CLI is recommended when preparing a clean release run.
- The default configs are useful for smoke testing, but the published-style evaluations usually rely on the class-specific manifests under `attack_experiments/` and malicious query sets under `malicious_dataset/`.

## Citation

If you use this repository in academic work, please cite the associated paper or project report once the public citation information is available.
