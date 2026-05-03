# Attack Generation Workspace

This folder is a dedicated, non-invasive workspace for constructing attack samples.
It does not modify the current clean-evaluation pipeline or the main `attack/` package.

## Purpose

Use this workspace to:

- build CDAB attack plans
- generate attack-sample metadata
- compute post-generation attack metrics
- optimize for high acoustic consistency and high semantic distance
- diversify adversarial captions across the dataset rather than collapsing into one repeated semantic family
- store intermediate attack artifacts
- keep attack-generation outputs separate from clean RAG outputs

## Layout

- `config.example.yaml`: example configuration for attack generation
- `schemas.py`: attack-generation dataclasses
- `build_cdab_plans.py`: CLI for creating CDAB plan JSONL files
- `generate_cdab_captions.py`: MLLM-driven acoustic description + adversarial caption generator
- `prompts/`: prompt templates for future LLM-assisted CDAB generation
- `workspace/`: generated attack plans and intermediate files
- `generated/`: finalized attack-sample artifacts
- `logs/`: attack-generation logs

## Quick Start

```bash
cd /home/shuhaoz/Desktop/shuhaoz/CU_Project/PoisionedRAG
cp attack_generation/config.example.yaml attack_generation/config.yaml
/home/shuhaoz/miniconda3/envs/audio_rag/bin/python attack_generation/generate_cdab_captions.py --config attack_generation/config.yaml
```

The generated caption CSV is written to `attack_generation/generated/`.
