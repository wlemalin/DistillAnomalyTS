#! /bin/bash

#SBATCH --job-name=eval_vl
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpuh100
#SBATCH --partition=publicgpu
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=07:00:00

module purge                             # clear any inherited modules :contentReference[oaicite:8]{index=8}
module load python/python-3.11.4         # load the Python you need :contentReference[oaicite:9]{index=9}

echo "Running on $(hostname), Python: $(python --version)"


srun --exclusive -N1 -n1 python eval_lora_vl.py \
  --base_model_dir  "$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir     "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-4o_ts/checkpoint-1236" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ./post_training_evals/evaluation_apres_ts_vl.jsonl

srun --exclusive -N1 -n1 python eval_lora_vl.py \
  --base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir     "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-4o_ts3/checkpoint-1440" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts3.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ./post_training_evals/evaluation_apres_ts3_vl.jsonl

srun --exclusive -N1 -n1 python eval_lora_vl.py \
  --base_model_dir	"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --adapter_dir     "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-4o_ts4/checkpoint-1500" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts4.jsonl" \
  --val_ratio 0.10 \
  --seed 42 \
  --out_json ./post_training_evals/evaluation_apres_ts4_vl.jsonl

