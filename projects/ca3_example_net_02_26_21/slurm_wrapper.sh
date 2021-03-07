#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
#SBATCH --job-name="ca3_ex_net"
#SBATCH --output /scratch/username/CARLsim4_hc/projects/ca3_example_net_02_26_21/HC_IM_02_26_ca3_example_net_results.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=username@gmu.edu
#SBATCH --mem=10G
srun ./ca3_snn_GPU
