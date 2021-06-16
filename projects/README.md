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

## Creation of a network
The creation of a cell-type and connection-type specific network in CARLsim4 relies on the following critical components, which will each be described in their own section:

### Neuron Type Components
The two necessary components to defining a neuron type in CARLsim are the population size (i.e., the number of neurons) and the input-output relationship of the neuron type to the current it receives. The current release of Hippocampome includes parameter estimates for the [latter](http://hippocampome.org/php/Izhikevich_model.php) but not the former (although they will soon be publicly available). To create an entity that can define both the population size and input-output relationship for a neuron type, one must define a CARLsim group as follows (illustrated with both the excitatory Pyramidal and inhibitory Axo-Axonic cell types):

```
  // Define a group for the excitatory neuron type CA3 Pyramidal
  int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 74366,
                                EXCITATORY_NEURON, 0, GPU_CORES);
                                
  // Define the input-output relationship for CA3 Pyramidal,
  // based on the RASP.NASP firing pattern phenotype
  sim.setNeuronParameters(CA3_Pyramidal, 366.0, 0.0, 0.792338703789581, 0.0,
                          -63.2044008171655, 0.0, -33.6041733124267, 0.0, 0.00838350334098279,
                          0.0, -42.5524776883928, 0.0, 35.8614648558726,
                          0.0, -38.8680990294091, 0.0,
                          588.0, 0.0, 1);
                          
  // Define a group for the inhibitory neuron type CA3 Axo-Axonic
  int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 1909,
                              INHIBITORY_NEURON, 0, GPU_CORES);
  
  // Define the input-output relationship for CA3 Axo-Axonic,
  // based on the ASP firing pattern phenotype
  sim.setNeuronParameters(CA3_Axo_Axonic, 165.0, 0.0, 3.96146287759279, 0.0,
                          -57.099782869594, 0.0, -51.7187562820223, 0.0, 0.00463860807187154,
                          0.0, 8.68364493653417, 0.0, 27.7986355932787,
                          0.0, -73.9685042125372, 0.0,
                          15.0, 0.0, 1);                            
  ```

### Connection Type Components
The three necessary components to defining a connection type in CARLsim are the probability of connection, the short-term synaptic signals, and the conductance delay between two neuron types. In Hippocampome, this can only be performed for neuron types that have either a [known or potential connection](http://hippocampome.org/php/connectivity.php). The current release of Hippocampome includes parameter estimates for the [connection probabilities](http://hippocampome.org/php/synapse_probabilities_sypr.php) and somatic distances of dendrites and axons that can be used for defining a [conductance delay](http://hippocampome.org/php/synapse_probabilities_sd.php). Parameter estimates for the short-term synaptic signals will be made publicly available soon. To create a connection between two neuron types that can define these three parameter types, one must define a connection in CARLsim as follows (illustrated for the CA3 Axo-Axonic to CA3 Pyramidal connection type):

```
  // Define a connection between Axo-Axonic and CA3 Pyramidal neuron types
  sim.connect(CA3_Axo_Axonic, // pre-synaptic neuron type 
              CA3_Pyramidal, // post-synaptic neuron type
              "random", // create connections randomly between the two neuron types
              RangeWeight(0.0f, 1.45f, 2.45f), // define the lower, initial, and upper weight bounds for the connection type
              0.15f, // define the connection probability
              RangeDelay(1), // define the conductance delay 
              RadiusRF(-1.0), // define that no receptive field should be formed
              SYN_PLASTIC, // indicate that the connection type's weight can be modified 
              1.869561088f, // indicate the conductance of the fast currents of the synapse 
              0.0f // indicate the conductance of the slow currents of the synapse);
  ```

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
  make distclean && make -j32
  make install
  ```
  
6. Switch to the directory of the network that you would like to simulate (the code below uses the example network), and run the following commands:

  ```
  # Switch directory
  cd /scratch/username/CARLsim4_hc/projects/ca3_example_net_02_26_21

  # Create the syntax for the SNN to simulate
  python generateSNNSyntax.py
  ```
  
7. The example network uses an order of magnitude less neurons that the full-scale network, so the generateCONFIGStateSTP.h needs to now be updated:

  ```
  nano generateCONFIGStateSTP.h
  ```
  
  ```
  int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 328,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 190,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_Basket = sim.createGroup("CA3_Basket", 51,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 66,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 463,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_Ivy = sim.createGroup("CA3_Ivy", 233,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 152,
                                INHIBITORY_NEURON, 0, GPU_CORES);

  int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 7436,
                                EXCITATORY_NEURON, 0, GPU_CORES);
  ```

8. Compile the SNN:

  ```
  # Clear the contents of the results directory and any previous version of executables, and then compile the SNN to create a new executable
  make distclean && make -j32
  ```

9. Update the SLURM submission script (slurm_wrapper.sh) so that the output goes to the directory of your choice (example used is in the current folder you are in):

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
  
10. Submit the SLURM script to the ARGO Supercomputing Cluster:

  ```
  sbatch slurm_wrapper.sh
  ```
  
11. Verify that a SLURM job was created after running your SLURM script

  ```
  squeue -u username
  ```
  
12. Once the simulation has been finished (either by checking the squeue command to see if the simulation is still running and/or checking the email provided that will update you when the simulation has finished), view the simulation summary:

  ```
  cat HC_IM_02_26_ca3_example_net_results.txt
  ```

