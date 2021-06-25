#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="save_test"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc_02_26_21/projects/afferent_tests/scaled_down_save_load_tests/ca3_snn_GPU_04_13_21_HC_IM_afferents_scaled_ee_save_test/HC_IM_04_13_21_save_test.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=10G
srun ./ca3_snn_GPU
