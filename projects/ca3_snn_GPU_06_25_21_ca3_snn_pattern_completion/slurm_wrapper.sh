#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="patcomp"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc_02_26_21/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/06_25_21_pattern_completion.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=10G
srun ./ca3_snn_GPU
