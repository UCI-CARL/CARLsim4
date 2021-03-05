# Simulation of Data-Driven, Neuron-Type Specific CA3 SNNs 
This repository includes information as to how to run an example and full-scale spiking neural network (SNN) model of hippocampal subregion CA3. Before following the guidelines, please follow the instructions provided by CARLsim provided in their [README](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc).

## Installation
Beyond the dependencies of CARLsim4 described at the link above, to generate the syntax necessary to run the example and full-scale SNNs one will need to install Python 3 as well as the package dependencies included in the table below. Additionally, one can install the following [Anaconda distribution](https://docs.anaconda.com/anaconda/install/), which includes Python 3 and pandas, but the xlrd function will still need to be downloaded, as it is an optional dependency of pandas.

## Module Dependencies:
|module|tested version|
|---|---|
|Anaconda|02.2020|
|Python|3.7.6|
|pandas|0.25.3|
|numpy|1.18.1|
|xlrd|1.2.0|
|boost|1.67.0|

## Choosing a network to run
There are three directories from which SNNs can be simulated: [ca3_example_net_02_26_21](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21), where a scaled-down version of the model can be simulated, [synchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/synchronous) where full-scale versions activated by synchronous stimulation can be simulated, and the [asynchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/asynchronous) where full-scale versions activated by asynchronous stimulation can be simulated. Within both the synchronous and asynchronous directories, three full-scale model versions can be simulated -- the baseline, class, and archetype SNNs. The network features are broadly as follows: the baseline SNN maintains neuron and connection-type specificity; the class SNN maintains neuron-type specificity while removing connection-type specificity; and the archetype SNN maintains connection-type specificity while removing neuron-type specificity.

## Running the networks

### Users with access to GMU ARGO Cluster
For users with an ARGO account at GMU, first, one should update their bashrc with the following settings, which will load all modules necessary to compile and install CARLsim, along with compiling and running the simulations:

# User specific aliases and functions
module load gcc/7.3.1
module load cuda/10.1
module load boost/1.67.0

# CARLsim4 related
export PATH=/cm/shared/apps/cuda/10.1/bin:$PATH
export LD_LIBRARY_PATH=/cm/shared/apps/cuda/10.1/lib64:$LD_LIBRARY_PATH

# CARLsim4 mandatory variables
export CARLSIM4_INSTALL_DIR=/home/jkopsick/CARL_hc_02_26_21
export CUDA_PATH=/cm/shared/apps/cuda/10.1
export CARLSIM_CUDAVER=10
export CUDA_MAJOR_NUM=7
export CUDA_MINOR_NUM=0

# CARLsim4 optional variables
export CARLSIM_FASTMATH=0
export CARLSIM_CUOPTLEVEL=3
