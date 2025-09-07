#!/bin/bash

# VL outsample run
jid_outsample=$(sbatch --parsable src/evaluation/outsample/eval.slurm)

# VL outsample desc quality:
sbatch --dependency=afterok:${jid_outsample} src/evaluation/outsample_description_quality/eval_desc_UCR.slurm

