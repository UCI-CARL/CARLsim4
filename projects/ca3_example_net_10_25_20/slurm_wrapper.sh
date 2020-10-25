#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="ca3_ex_net"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc/projects/ca3_example_net_10_25_20/HC_IM_10_25_ca3_example_net_results.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=10G
srun ./ca3_snn_GPU
