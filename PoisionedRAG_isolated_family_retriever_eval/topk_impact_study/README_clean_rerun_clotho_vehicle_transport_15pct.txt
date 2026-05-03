Clean rerun for top-k impact study on Clotho
============================================

Goal
----
Run a fresh, publication-safe top-k impact experiment for:

  Dataset/class: Clotho / vehicle_transport
  Fixed poison rate: 15%
  Fixed poison count: 556
  Top-k values: 1, 5, 10, 15, 20

This setup uses the completed large malicious pool prepared specifically for
Clotho / vehicle_transport / 15%.


Step 0: build the poison-rate experiment files
----------------------------------------------
cd /scratch/shuhaoz/CU_Project/PoisionedRAG
python attack_generation/build_clotho_vehicle_transport_rate_experiments.py


Fresh output directory
----------------------
outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun


Step 1: run the clean top-k sweep
---------------------------------
cd /scratch/shuhaoz/CU_Project/PoisionedRAG
mkdir -p outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun

/home/shuhaoz/miniconda3/envs/audio_rag/bin/python -u topk_impact_study/run_topk_impact_eval.py \
  --poisoned-manifest attack_experiments/clotho_vehicle_transport_rates/rate_15pct/clotho_poisoned_manifest.jsonl \
  --malicious-metadata malicious_dataset/wavcaps_clotho_esc50_to_clotho_vehicle_transport_556/final_dataset/query_subset_20.json \
  --output-dir outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun \
  --poison-rate-pct 15 \
  --poison-count 556 \
  --k-values 1 5 10 15 20 \
  --continue-on-error 2>&1 | tee outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun/manual_run.log


Step 2: verify the run is clean
-------------------------------
cd /scratch/shuhaoz/CU_Project/PoisionedRAG
python topk_impact_study/verify_topk_impact_run.py \
  --input-dir outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun \
  --k-values 1 5 10 15 20

What you want to see:
  "all_row_counts_equal": true


Step 3: build the final metrics
-------------------------------
cd /scratch/shuhaoz/CU_Project/PoisionedRAG
python topk_impact_study/build_topk_impact_metrics.py \
  --input-dir outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun \
  --poison-rate-pct 15 \
  --poison-count 556 \
  --k-values 1 5 10 15 20


Final result files
------------------
outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun/metrics_topk_impact/comparison_summary_topk_impact.json
outputs/topk_impact_clotho_vehicle_transport_15pct_cleanrerun/metrics_topk_impact/comparison_summary_topk_impact.csv
