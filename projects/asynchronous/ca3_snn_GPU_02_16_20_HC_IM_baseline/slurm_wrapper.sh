#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos gaqos
#SBATCH --gres=gpu:1
#SBATCH --exclude=NODE0[40,50,56]
## SBATCH --cpus-per-task=1
## SBATCH --nodelist=NODE077
#SBATCH --job-name="100_2_ivy_ee_vii15_pt55_vei19_ie_1pt45_vei19_ie_U_0pt55_ee_vii15_U_1pt45_conn_stype_v13_pyrdelay_12_varied_count"
#SBATCH --output /scratch/jkopsick/CARLsim4STP_hc_11_05_20/projects/11_17_20_pyr_tests/ivy_tests_v3/best_ivy_tests/100_tests/ca3_snn_GPU_11_17_20_HC_IM_peri_pyr_0pt15_ee_vii15_pt55_vei19_ie_1pt45_vei19_ie_U_0pt55_ee_vii15_U_1pt45_conn_stype_v13_pyrdelay_12_varied_count_100_2/HC_IM_11_17_peri_pyr_0pt15_ee_vii15_pt55_vei19_ie_1pt45_vei19_ie_U_0pt55_ee_vii15_U_1pt45_conn_stype_v13_pyrdelay_12_varied_count_100_2.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jkopsick@gmu.edu
#SBATCH --mem=100G
srun ./ca3_snn_GPU
