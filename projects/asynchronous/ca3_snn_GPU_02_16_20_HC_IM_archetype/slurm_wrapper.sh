#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
#SBATCH --job-name="async_archetype"
#SBATCH --output /scratch/username/CARLsim4_hc/projects/asynchronous/ca3_snn_GPU_02_16_20_HC_IM_archetype/HC_IM_02_16_ca3_snn_async_archetype.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=username@gmu.edu
#SBATCH --mem=100G
srun ./ca3_snn_GPU
