#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="best_ivy_variant_delayed_activation_1000_1"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc_11_05_20/projects/11_30_20_pyr_tests/delayed_activation/ca3_snn_GPU_11_30_20_HC_IM_best_ivy_variant_delayed_activation_1000_1/HC_IM_11_30_best_ivy_variant_delayed_activation_1000_1.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=100G
srun ./ca3_snn_GPU
