#!/bin/bash
#SBATCH -t 00:05:00
#SBATCH --job-name="gierer"
#SBATCH —gres=gpu:0

cd $SLURM_SUBMIT_DIR
./gierer 128 1 100 0.5 1 6 0.003 100000 12345