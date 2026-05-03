# PoisionedRAG Isolated Family Retriever Eval

This workspace isolates the effect of retriever choice by running model-family and top-k matrix experiments on fixed poisoned datasets.

The main use case is to compare:

- GPT-like vs Gemini-like model families
- CLAP vs AudioCLIP vs PANNs vs wav2vec2 retrievers
- `top-k` values such as `1`, `5`, `10`, `15`, and `20`

For the full project overview and reproduction order, see the repository-level [README](../README.md).

## Main Entry Point

```bash
python topk_impact_study/run_retriever_family_matrix.py \
  --output-root outputs/retriever_family_matrix \
  --families gpt gemini \
  --datasets wavcaps audioset clotho esc50 \
  --retrievers clap audioclip panns wav2vec2 \
  --k-values 1 5 10 15 20 \
  --continue-on-error
```

This orchestrates:

1. fixed-rate top-k sweeps
2. summary-metric building
3. family-by-dataset-by-retriever output aggregation

## Key Scripts

- `topk_impact_study/run_retriever_family_matrix.py`
- `topk_impact_study/run_topk_impact_eval.py`
- `topk_impact_study/build_topk_impact_metrics.py`

## Notes

- This folder is useful when you want to study retrieval behavior independently from broader model-matrix reporting.
- The experiment manifests and malicious query sets are expected to already exist in this workspace.
