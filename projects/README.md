# Simulation of Data-Driven, Neuron-Type Specific CA3 SNNs 
This repository includes information as to how to run an example and full-scale spiking neural network (SNN) model of hippocampal subregion CA3. The following instructions assume that the user has Ubuntu installed or is using a supercomputing cluster that is Linux-based. Additionally, installation instructions of the CARLsim software can be found in the parent directory's [README](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc), but will also be provided here.

## Module Dependencies for the Software:
Beyond the dependencies of CARLsim4 described at the link above, to generate the syntax necessary to run the example and full-scale SNNs one will need to install Python 3 as well as the package dependencies included in the table below. Additionally, one can install the following [Anaconda distribution](https://docs.anaconda.com/anaconda/install/), which includes Python 3 and pandas, but the xlrd function will still need to be downloaded, as it is an optional dependency of pandas.

|module|tested version|
|---|---|
|Anaconda|02.2020|
|Python|3.7.4, 3.7.6|
|pandas|0.25.3, 1.2.3|
|numpy|1.18.1, 1.20.1|
|xlrd|1.2.0|
|boost|1.67.0|

## Choosing a network to run
There are three directories from which SNNs can be simulated: [ca3_example_net_02_26_21](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21), where a scaled-down version of the model can be simulated, [synchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/synchronous) where full-scale versions activated by synchronous stimulation can be simulated, and the [asynchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/asynchronous) where full-scale versions activated by asynchronous stimulation can be simulated. Within both the synchronous and asynchronous directories, three full-scale model versions can be simulated -- the baseline, class, and archetype SNNs. The network features are broadly as follows: the baseline SNN maintains neuron and connection-type specificity; the class SNN maintains neuron-type specificity while removing connection-type specificity; and the archetype SNN maintains connection-type specificity while removing neuron-type specificity.

## Installation and Simulation of the SNNs

### Users with access to GMU ARGO Cluster
For users with an ARGO account at GMU, the following steps will need to be taken:
  
1. Update the bashrc from your home directory (/home/username) with the following settings, which will load all modules necessary to compile and install CARLsim, along with compiling and running the simulations:

  ```
  nano ~/.bashrc
  ```

  ```
  # User specific aliases and functions
  module load gcc/7.3.1
  module load cuda/10.1
  module load boost/1.67.0

  # CARLsim4 related
  export PATH=/cm/shared/apps/cuda/10.1/bin:$PATH
  export LD_LIBRARY_PATH=/cm/shared/apps/cuda/10.1/lib64:$LD_LIBRARY_PATH

  # CARLsim4 mandatory variables
  export CARLSIM4_INSTALL_DIR=/home/username/CARL_hc
  export CUDA_PATH=/cm/shared/apps/cuda/10.1
  export CARLSIM_CUDAVER=10
  export CUDA_MAJOR_NUM=7
  export CUDA_MINOR_NUM=0

  # CARLsim4 optional variables
  export CARLSIM_FASTMATH=0
  export CARLSIM_CUOPTLEVEL=3
  ```
  
2. Create a Python Virtual Environment using virtualenv, along with installation of packages necessary for syntax generation from an input XL file.

  ```
  # Load the Python module
  module load python/3.7.4

  # Create the Python virtualenv
  python -m virtualenv test-site-virtualenv-3.7.4-no-sys-pack -p /usr/bin/python3

  # Unload the Python module now that the virtualenv has been created
  module unload python/3.7.4

  # Switch to the Python virtualenv that has been created
  source test-site-virtualenv-3.7.4-no-sys-pack/bin/activate
  
  # Install necessary packages to run syntax generation code
  pip install numpy
  pip install pandas
  pip install xlrd==1.2.0
  ```
3. Switch to the scratch directory for your account and download the repository from GitHub into a folder name of your choice (CARLsim4_hc used in the example below):
  ```
  cd /scratch/username
  git clone https://github.com/UCI-CARL/CARLsim4.git -b feat/meansdSTPPost_hc CARLsim4_hc
  ```
  
4. Switch to the newly created CARLsim4 folder
  ```
  cd CARLsim4_hc
  ```
  
5. Make and install the CARLsim4 software:
  ```
  make distclean && make
  make install
  ```
  
6. Switch to the directory of the network that you would like to simulate (the code below uses the example network), and run the following commands:

  ```
  # Switch directory
  cd /scratch/username/CARLsim4_hc/projects/ca3_example_net_02_26_21

  # Create the syntax for the SNN to simulate
  python generateSNNSyntax.py

  # Clear the contents of the results directory and any previous version of executables, and then compile the SNN to create a new executable
  make distclean && make
  ```
  
7. Update the SLURM submission script (slurm_wrapper.sh) so that the output goes to the directory of your choice (example used is in the current folder you are in):

  ```
  nano slurm_wrapper.sh
  ```

  ```
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
  ```
  
8. Submit the SLURM script to the ARGO Supercomputing Cluster:

  ```
  sbatch slurm_wrapper.sh
  ```
  
9. Verify that a SLURM job was created after running your SLURM script

  ```
  squeue -u username
  ```
  
10. Once the simulation has been finished (either by checking the squeue command to see if the simulation is still running and/or checking the email provided that will update you when the simulation has finished), view the simulation summary:

  ```
  cat HC_IM_02_26_ca3_example_net_results.txt
  ```

