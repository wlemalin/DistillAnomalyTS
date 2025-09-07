#! /bin/bash

#SBATCH --job-name=train_vl
#SBATCH --output=qwen_%j.out
#SBATCH --error=qwen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpuh100
#SBATCH --partition=publicgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH --time=12:00:00

# 1) Load your modules (must come before sourcing your venv) 
module purge                             # clear any inherited modules :contentReference[oaicite:8]{index=8}
module load python/python-3.11.4         # load the Python you need :contentReference[oaicite:9]{index=9}

echo "Running on $(hostname), Python: $(python --version)"
python cuda_check.py

srun --exclusive -N1 -n1 python train_lora_vl.py \
  --model_dir		"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts.jsonl" \
  --out_dir         "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts" \
  --val_ratio 0.10 \
  --seed 42 \
  --epochs 12 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --debug

srun --exclusive -N1 -n1 python train_lora_vl.py \
  --model_dir		"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts3.jsonl" \
  --out_dir         "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts3" \
  --val_ratio 0.10 \
  --seed 42 \
  --epochs 14 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --debug

srun --exclusive -N1 -n1 python train_lora_vl.py \
  --model_dir		"$HOME/hf_models/Qwen2.5-VL-3B-Instruct_clean" \
  --data_json		"$HOME/repo/src/annotations/jsonl/filtered_annot_openai_4o_ts4.jsonl" \
  --out_dir         "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts4" \
  --val_ratio 0.10 \
  --seed 42 \
  --epochs 15 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 1e-5 \
  --debug

sh "$HOME/repo/src/train_VL/fix_checkpoint.sh" qwen2.5-vl-3b-ft-lora-vl_ts/
sh "$HOME/repo/src/train_VL/fix_checkpoint.sh" qwen2.5-vl-3b-ft-lora-vl_ts3/
sh "$HOME/repo/src/train_VL/fix_checkpoint.sh" qwen2.5-vl-3b-ft-lora-vl_ts4/
