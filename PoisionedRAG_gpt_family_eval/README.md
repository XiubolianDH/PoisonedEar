# PoisionedRAG GPT Family Eval

This workspace runs fixed-matrix attack evaluations for GPT-family and Gemini-family audio models.

The default design is:

- fixed poison rate: `15%`
- fixed retrieval depth: `k = 5`
- datasets: `wavcaps`, `audioset`, `clotho`, `esc50`
- retrievers: `clap`, `audioclip`, `panns`, `wav2vec2`

For the full project overview and reproduction order, see the repository-level [README](../README.md).

## Main Entry Points

### GPT-family matrix

```bash
python attack_generation/run_gpt_family_fixed_k5_matrix.py \
  --output-root outputs/gpt_family_fixed_k5 \
  --top-k 5 \
  --continue-on-error
```

Default GPT-family models:

- `gpt-4o-audio-preview`
- `gpt-4o-mini-audio-preview`

### Gemini-family matrix

```bash
python attack_generation/run_gemini_family_fixed_k5_matrix.py \
  --output-root outputs/gemini_family_fixed_k5 \
  --index-cache-dir outputs/shared_retriever_index_cache \
  --top-k 5 \
  --continue-on-error
```

Default Gemini-family models:

- `gemini-2.5-flash`
- `gemini-2.5-pro`

## Main Outputs

- `outputs/gpt_family_fixed_k5/...`
- `outputs/gemini_family_fixed_k5/...`
- `gpt_family_overall_asr_recall_k5.csv`
- `gpt_family_overall_asr_recall_k5.json`
- `gemini_family_overall_asr_recall_k5.csv`
- `gemini_family_overall_asr_recall_k5.json`

## Notes

- This folder mirrors the main pipeline structure, but keeps model-family comparisons isolated from the base `PoisionedRAG/` workspace.
- Make sure the required `.env` file and model configs are set before launching the matrix runs.
