#! /bin/bash

#SBATCH --job-name=eval_outsample
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpuh100
#SBATCH --partition=publicgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=25G
#SBATCH --time=04:44:00

module purge
module load python/python-3.11.4
echo "Running on $(hostname), Python: $(python --version)"

srun --exclusive -N1 -n1 python eval_outsample_vl.py \
  --mode ts \
  --lora-adapter-path "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts/checkpoint-1236/" \
  --output-jsonl outsample_eval_vl_ts.jsonl \
  --ban-tool-words 

srun --exclusive -N1 -n1 python eval_outsample_vl.py \
  --mode ts3 \
  --lora-adapter-path "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts3/checkpoint-1400/" \
  --output-jsonl outsample_eval_vl_ts3.jsonl \
  --ban-tool-words 

srun --exclusive -N1 -n1 python eval_outsample_vl.py \
  --mode all \
  --lora-adapter-path "$HOME/repo/src/train_VL/qwen2.5-vl-3b-ft-lora-vl-ts4/checkpoint-1440/" \
  --output-jsonl outsample_eval_vl_ts4.jsonl \
  --ban-tool-words
