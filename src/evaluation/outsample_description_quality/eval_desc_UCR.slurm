#! /bin/bash

#SBATCH --job-name=eval_desc_outsample
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

srun --exclusive -N1 -n1 python eval_desc_UCR.py \
  --gt UCR_gt_desc.csv \
  ../outsample/outsample_eval_vl_ts.jsonl \
  eval_desc_vl_ts.json

srun --exclusive -N1 -n1 python eval_desc_UCR.py \
  --gt UCR_gt_desc.csv \
  ../outsample/outsample_eval_vl_ts3.jsonl \
  eval_desc_vl_ts3.json

srun --exclusive -N1 -n1 python eval_desc_UCR.py \
  --gt UCR_gt_desc.csv \
  ../outsample/outsample_eval_vl_ts4.jsonl \
  eval_desc_vl_ts4.json
