#!/usr/bin/env bash

# run training for txt model:
sbatch "$HOME/src/train_text_only/train_text.slurm"
echo "Submitted train_text.slurm"

# and for vl model:
jid=$(sbatch --parsable "$HOME/src/train_VL/train_vl.slurm")
echo "Submitted train_vl.slurm as ${jid}"

sbatch --dependency=afterok:${jid} "$HOME/src/train_VL/eval_vl.slurm"
echo "Submitted eval_vl.slurm with dependency afterok:${jid}"

