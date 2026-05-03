Top-k impact study at a fixed poison rate of 15%
================================================

This directory is intentionally separate from the existing rate-sweep outputs.
It does not overwrite any previous experiment results.

Recommended representative setup
--------------------------------
Dataset/class:
  AudioSet / music_instrument

Fixed poison rate:
  15%

Fixed poison count:
  116

Fixed poisoned manifest:
  attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl

Fixed malicious metadata / query set:
  malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json

Suggested output directory:
  outputs/topk_impact_audioset_music_instrument_15pct


Step 1: run the top-k sweep
---------------------------
cd /home/shuhaoz/Desktop/shuhaoz/CU_Project/PoisionedRAG
mkdir -p outputs/topk_impact_audioset_music_instrument_15pct
/home/shuhaoz/miniconda3/envs/audio_rag/bin/python -u topk_impact_study/run_topk_impact_eval.py \
  --poisoned-manifest attack_experiments/audioset_music_instrument_rates/rate_15pct/audioset_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_audioset_music_instrument_155/final_dataset/query_subset_20.json \
  --output-dir outputs/topk_impact_audioset_music_instrument_15pct \
  --poison-rate-pct 15 \
  --poison-count 116 \
  --k-values 1 5 10 15 20 \
  --continue-on-error 2>&1 | tee outputs/topk_impact_audioset_music_instrument_15pct/manual_run.log


Step 2: build the summary metrics
---------------------------------
cd /home/shuhaoz/Desktop/shuhaoz/CU_Project/PoisionedRAG
python topk_impact_study/build_topk_impact_metrics.py \
  --input-dir outputs/topk_impact_audioset_music_instrument_15pct \
  --poison-rate-pct 15 \
  --poison-count 116 \
  --k-values 1 5 10 15 20


Final output files
------------------
Raw sweep outputs:
  outputs/topk_impact_audioset_music_instrument_15pct/

Metric summaries:
  outputs/topk_impact_audioset_music_instrument_15pct/metrics_topk_impact/comparison_summary_topk_impact.json
  outputs/topk_impact_audioset_music_instrument_15pct/metrics_topk_impact/comparison_summary_topk_impact.csv
