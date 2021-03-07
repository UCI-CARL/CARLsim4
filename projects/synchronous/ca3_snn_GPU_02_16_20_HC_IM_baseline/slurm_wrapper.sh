#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
#SBATCH --job-name="sync_baseline"
#SBATCH --output /scratch/username/CARLsim4_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/HC_IM_02_16_ca3_snn_sync_baseline.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=username@gmu.edu
#SBATCH --mem=100G
srun ./ca3_snn_GPU
