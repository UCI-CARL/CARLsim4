#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:3
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="all_pt05"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc/projects/IM_refrac_tests/resting_state_tests/100_tests/ca3_snn_GPU_08_21_20_HC_IM_refrac_all_pt05/HC_IM_08_06_refrac_all_pt05.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=200G
srun ./ca3_snn_GPU
