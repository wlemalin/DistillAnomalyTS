#!/bin/bash

# F1 insample VL:
python src/aggregate_eval_reports.py \
  --evaluator src/evaluation/eval_affil.py \
  --csv results/F1_vl_models_insample.csv \
  src/annotations/jsonl/annot_openai_4o_vl_ts3.jsonl \
  src/train_VL/post_training_eval/evaluation_apres_ts_vl.jsonl \
  src/train_VL/post_training_eval/evaluation_apres_ts3_vl.jsonl \
  src/train_VL/post_training_eval/evaluation_apres_ts4_vl.jsonl 


# F1 insample txt:
python src/aggregate_eval_reports.py \
  --evaluator src/evaluation/eval_affil.py \
  --csv results/F1_txt_models_insample.csv \
  src/annotations/jsonl/annot_openai_4o_txt_ts.jsonl \
  src/train_text/post_training_eval/jsonl/evaluation_apres_txt_ts.jsonl  \
  src/train_text/post_training_eval/jsonl/evaluation_apres_txt_ts3.jsonl  \
  src/train_text/post_training_eval/jsonl/evaluation_apres_txt_ts4.jsonl  \

# F1 outsample:
python src/aggregate_eval_reports.py \
  --evaluator src/evaluation/eval_affil.py \
  --csv results/F1_outsample.csv \
  src/evaluation/outsample/outsample_eval_txt_ts3.jsonl \
  src/evaluation/outsample/outsample_eval_txt_ts4.jsonl \
  src/evaluation/outsample/outsample_eval_vl_ts3.jsonl \
  src/evaluation/outsample/outsample_eval_vl_ts4.jsonl 

# In sample description quality:
sh src/evaluation/insample_description_quality/make_table.sh


# Out sample description quality:
python src/aggregate_outsample.py \
  --dir src/evaluation/outsample_description_quality/ \
  --csv results/outsample_description_metrics.csv
