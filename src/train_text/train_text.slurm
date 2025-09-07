#! /bin/bash

#SBATCH --job-name=train_txt
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpuh100
#SBATCH --partition=publicgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=35G
#SBATCH --time=03:55:00

module purge
module load python/python-3.11.4
echo "Running on $(hostname), Python: $(python --version)"

# Train with raw signal only:
srun --exclusive -N1 -n1 python train_text.py \
  --dataset_path "$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_text_n1.jsonl" \
  --output_dir "$HOME/repo/src/train_text/qwen2.5-lora-output-ts-txt/" \
  --out_json txt_ts

# Train with raw signal + moving average + moving std:
srun --exclusive -N1 -n1 python train_text.py \
  --dataset_path "$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_text_n3.jsonl" \
  --output_dir "$HOME/repo/src/train_text/qwen2.5-lora-output-ts3-txt/" \
  --out_json txt_ts3

# Train with raw signal + moving average + moving std + local frequency:
srun --exclusive -N1 -n1 python train_text.py \
  --dataset_path "$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_text_n4.jsonl" \
  --output_dir "$HOME/repo/src/train_text/qwen2.5-lora-output-ts4-txt/" \
  --out_json txt_ts4

# Turn json to jsonl format:
mkdir -p post_training_eval/jsonl
for f in post_training_eval/evaluation_apres_*.json; do
  jq -c '.[]' "$f" > "post_training_eval/jsonl/$(basename "${f%.json}").jsonl"
  done

