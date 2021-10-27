#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:2
## SBATCH --exclude=NODE0[40,50,56,76,78,80,81]
#SBATCH --exclude=NODE0[40,50,56,77,79]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="random_spnet"
#SBATCH --time=0-12:00:00
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc_08_10_21/projects/lif_izhi_random_spnet/10_18_21_stdp_by_type_test_save_test_nNeur_5k_external_spikegenerator_2GPU.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=50G
srun ./random_spnet
