# A Simulation Framework for Testing Data-Driven, Neuron-Type Specific CA3 SNNs 
This repository includes information as to how to run an example and full-scale spiking neural network (SNN) model of hippocampal subregion CA3. Additionally, it describes a framework for building full-scale SNNs to test hypotheses regarding the hippocampal formation. The following instructions assume that the user has Ubuntu installed or is using a supercomputing cluster that is Linux-based. Furthermore, installation instructions of the CARLsim software can be found in the parent directory's [README](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc), but will also be provided here.

## Table of Contents:
* [Language and Module Dependencies](#language-and-module-dependencies-for-the-software)
	* [Programming Languages and Proficiency](#programming-languages-and-proficiency)
	* [Ubuntu Users](#ubuntu)
	* [GMU ARGO Users](#GMU-ARGO-cluster)

* [Creation of a network simulation](#creation-of-a-network-simulation)
	* [Neuron Type Components](#neuron-type-components)
	* [Connection Type Components](#connection-type-components)
	* [Monitoring of specific neuron types](#monitoring-of-specific-neuron-types)
	* [Running a Simulation](#running-a-simulation)
	* [The Simulation Summary](#the-simulation-summary)

* [Choosing a Network to Run](#choosing-a-network-to-run)

* [Installation and Simulation of SNNs](#installation-and-simulation-of-the-SNNs)
	* [Ubuntu Users](#ubuntu-users)
	* [GMU ARGO Users](#users-with-access-to-GMU-ARGO-cluster)

* [Framework to Test Hippocampal Hypotheses](#A-framework-to-test-hypotheses-of-the-hippocampal-formation) 

## Language and Module Dependencies for the Software:
Beyond the dependencies of CARLsim4 described at the link above, to generate the syntax necessary to run the example and full-scale SNNs one first must be familiar with C++ and Python, with recommended proficiency provided below. One will need to install Python 3 as well as the package dependencies included in the tables below. Additionally, one can install the following [Anaconda distribution](https://docs.anaconda.com/anaconda/install/), which includes Python 3 and pandas, but the xlrd function will still need to be downloaded, as it is an optional dependency of pandas.

### Programming Languages and Proficiency
|Language|Proficiency|
|---|---|
|C++|basic to intermediate|
|Python|basic|

### Ubuntu 
|module|tested version(s)|
|---|---|
|Anaconda|02.2020|
|Python|3.7.4, 3.7.6, 3.8.5|
|pandas|0.25.3, 1.2.3|
|numpy|1.18.1, 1.20.1|
|xlrd|1.2.0|
|boost|1.67.0|
|make|4.2.1|
|gcc|7.3.3, 7.5.0|
|nohup|8.30|

### GMU ARGO Cluster
|module|tested version(s)|
|---|---|
|Anaconda|02.2020|
|Python|3.7.4, 3.7.6, 3.8.5|
|pandas|0.25.3, 1.2.3|
|numpy|1.18.1, 1.20.1|
|xlrd|1.2.0|
|boost|1.67.0|

## Creation of a network simulation
The creation of a cell-type and connection-type specific network and the subsequent simulation of it in CARLsim4 relies on the following critical components, which will each be described in their own section:

### Neuron Type Components
The two necessary components to defining a neuron type in CARLsim are the population size (i.e., the number of neurons) and the input-output relationship of the neuron type to the current it receives (i.e. parameters that govern the neuron type's firing patterns). The current release of [Hippocampome.org](http://hippocampome.org/php/index.php) includes parameter estimates for the [latter](http://hippocampome.org/php/Izhikevich_model.php) but not the former (although they will soon be publicly available). Using the examples of the excitatory CA3 Pyramidal and inhibitory CA3 Axo-Axonic neuron types, the parameter estimates for their input-output relationships as defined by the Izhikevich model can be found [here](http://hippocampome.org/php/neuron_page.php?id=2000) and [here](http://hippocampome.org/php/neuron_page.php?id=2028) under the Izhikevich Model section of the pages (subtype 1 parameter sets are used for both neuron types). To create an entity that can define both the population size and input-output relationship for a neuron type, one must define a CARLsim group as follows (more details can be found at the top of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/generateCONFIGStateSTP.h)):

```
  // Define a group for the excitatory neuron type CA3 Pyramidal
  int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", // name of the neuron type
                                      74366, // population size of the neuron type
                                      EXCITATORY_NEURON, // define whether neuron type is excitatory/inhibitory 
                                      0, // define the processor (CPU or GPU) to create the group on (e.g., GPU = 0)
                                      GPU_CORES // define whether the group will be created on CPU/GPU
				      );
                                
  // Define the input-output relationship for CA3 Pyramidal,
  // based on the RASP.NASP firing pattern phenotype
  sim.setNeuronParameters(CA3_Pyramidal, // neuron type
                          366.0, // mean C parameter
                          0.0, // standard deviation C parameter
                          0.792338703789581, // mean k parameter
                          0.0, // standard deviation k parameter
                          -63.2044008171655, // mean resting membrane potential parameter
                          0.0, // standard deviation resting membrane potential parameter
                          -33.6041733124267, // mean spike threshold membrane potential parameter
                          0.0, // standard deviation spike threshold membrane potential parameter
                          0.00838350334098279, // mean a parameter
                          0.0, // standard deviation a parameter
                          -42.5524776883928, // mean b parameter
                          0.0, // standard deviation b parameter
                          35.8614648558726, // mean spike peak membrane potential parameter
                          0.0, // standard deviation spike peak membrane potential parameter
                          -38.8680990294091, // mean c parameter
                          0.0, // standard deviation c parameter
                          588.0, // mean d parameter
                          0.0, // standard deviation d parameter
                          1 // mean refractory period parameter
			  );
                          
  // Define a group for the inhibitory neuron type CA3 Axo-Axonic
  int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 
                                       1909,
                                       INHIBITORY_NEURON, 
                                       0, 
                                       GPU_CORES
				       );
  
  // Define the input-output relationship for CA3 Axo-Axonic,
  // based on the ASP firing pattern phenotype
  sim.setNeuronParameters(CA3_Axo_Axonic, 
                          165.0, 
                          0.0, 
                          3.96146287759279, 
                          0.0,
                          -57.099782869594,
                          0.0, 
                          -51.7187562820223, 
                          0.0, 
                          0.00463860807187154,
                          0.0, 
                          8.68364493653417, 
                          0.0, 
                          27.7986355932787,
                          0.0, 
                          -73.9685042125372, 
                          0.0,
                          15.0, 
                          0.0, 
                          1
			  );                            
  ```

### Connection Type Components
The three necessary components to defining a connection type in CARLsim are the probability of connection, the short-term synaptic signals, and the conductance delay between two neuron types. In Hippocampome, this can only be performed for neuron types that have either a [known or potential connection](http://hippocampome.org/php/connectivity.php). The current release of Hippocampome includes parameter estimates for the [connection probabilities](http://hippocampome.org/php/synapse_probabilities_sypr.php) and somatic distances of dendrites and axons that can be used for defining a [conductance delay](http://hippocampome.org/php/synapse_probabilities_sd.php). Parameter estimates for the short-term synaptic signals will be made publicly available soon. To create a connection between two neuron types that can define these three parameter types, one must define a connection in CARLsim as follows (illustrated for the CA3 Axo-Axonic to CA3 Pyramidal connection type; more details can be found near the top of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/generateCONFIGStateSTP.h)):

```
  // Define a connection between Axo-Axonic and Pyramidal neuron types
  sim.connect(CA3_Axo_Axonic, // presynaptic neuron type 
              CA3_Pyramidal, // postsynaptic neuron type
              "random", // create connections randomly between the two neuron types
              RangeWeight(0.0f, 1.45f, 2.45f), // define the lower, initial, and upper weight bounds for the connection type
              0.15f, // define the connection probability
              RangeDelay(1), // define the conductance delay 
              RadiusRF(-1.0), // define that no receptive field should be formed
              SYN_PLASTIC, // indicate that the connection type's weight can be modified 
              1.869561088f, // indicate the conductance of the fast currents of the synapse 
              0.0f // indicate the conductance of the slow currents of the synapse
	      );
              
   // Define short-term plasticity parameters between the Axo-Axonic and Pyramidal neuron types
   sim.setSTP(CA3_Axo_Axonic, // presynaptic neuron type 
              CA3_Pyramidal, // postsynaptic neuron type
              true, // define short-term plasticity for the connection type
              STPu(0.1302436377f, 0.0f), // define mean and standard deviation of the U parameter
              STPtauU(12.93029701f, 0.0f), // define mean and standard deviation of the tauU parameter; tau_f on Hippocampome
              STPtauX(361.0287219f, 0.0f), // define mean and standard deviation of the tauX parameter; tau_r on Hippocampome
              STPtdAMPA(5.0f, 0.0f), // define mean and standard deviation of the AMPA receptor current decay time constant
              STPtdNMDA(150.0f, 0.0f), // define mean and standard deviation of the NMDA receptor current decay time constant
              STPtdGABAa(7.623472774f, 0.0f), // define mean and standard deviation of the GABAA receptor current decay time constant
              STPtdGABAb(150.0f, 0.0f), // define mean and standard deviation of the GABAB receptor current decay time constant
              STPtrNMDA(0.0f, 0.0f), // define mean and standard deviation of the NMDA receptor current rise time constant; disabled by default but can be enabled with non-zero input
              STPtrGABAb(0.0f, 0.0f) // define mean and standard deviation of the GABAB receptor current rise time constant; disabled by default but can be enabled with non-zero input
	      );
  ```

### Monitoring of specific neuron types
The membrane potential (intracellular recording) and spikes (extracellular recording) of each neuron of a neuron type can be monitored and stored using the NeuronMonitor and SpikeMonitor CARLsim classes, respectively. To instantiate either monitor in a network simulation, it can be done so as follows (more details can be found at the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/generateCONFIGStateSTP.h) and the top of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/generateSETUPStateSTP.h)):

```
  // Define a NeuronMonitor for the Pyramidal neuron type
  sim.setNeuronMonitor(CA3_Pyramidal, // neuron type to be monitored
                       "DEFAULT" // directory location of the file containing the intracellular recording of membrane potential
		       );
 
   // Define a SpikeMonitor for the Pyramidal neuron type
  sim.setSpikeMonitor(CA3_Pyramidal, // neuron type to be monitored
                       "DEFAULT" // directory location of the file containing the extracellular recording of spikes
		       );
  ```

### Running a simulation
Once the neuron type and connection type properties have been defined through CARLsim, along with the specific monitors needed to record intracellular and extracellular activity, the stimulation protocol and simulation duration are defined. For this particular example, we use the synchronous stimulation paradigm that will activate a random subset of 100 Pyramidal cells in the network (more details can be found [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/src/main_ca3_snn_GPU.cpp)):

```
  // Declare variables that will store the start and end ID for the neurons
	// in the pyramidal group
	int pyr_start = sim.getGroupStartNeuronId(CA3_Pyramidal);
	std::cout << "Beginning neuron ID for Pyramidal Cells is : " << pyr_start;
	int pyr_end = sim.getGroupEndNeuronId(CA3_Pyramidal);
	std::cout << "Ending neuron ID for Pyramidal Cells is : " << pyr_end;
	int pyr_range = (pyr_end - pyr_start) + 1;
	std::cout << "The range for Pyramidal Cells is : " << pyr_range;

	// Create vectors that are the length of the number of neurons in the pyramidal
	// group, and another that will store the current at the position for the
  // random pyramidal cells that will be selected
	std::vector<int> pyr_vec( boost::counting_iterator<int>( 0 ),
							              boost::counting_iterator<int>( pyr_range ));
  std::vector<float> current(pyr_range, 0.0f);
  
  // Define the number of neurons to receive input from the external current
  int numPyramidalFire = 100;
  
  // Set the seed of the pseudo-random number generator based on the current system time,
  // which will allow for a random subset of 100 Pyramidal cells to be chosen to receive
  // transient input.
	std::srand(std::time(nullptr));
  
  // run for a total of 9 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int i=0; i<20; i++) 
	{
    	if (i == 0)
        {
            // Set external current for a fraction of pyramidal cells based on the random
            // seed
            for (int i = 0; i < numPyramidalFire; i++)
            {
                int randPyrCell = pyr_vec.front() + ( std::rand() % ( pyr_vec.back() - pyr_vec.front() ) );
                current.at(randPyrCell) = 45000.0f;
            }
            
            // Set the external current for all Pyramidal cells and run the network
            // for one ms
            sim.setExternalCurrent(CA3_Pyramidal, current);
            sim.runNetwork(0,1);
        }
        
		if (i == 1)
		{
    		// Set the external current for all Pyramidal cells to zero and run the 
    		// network for one ms
    		sim.setExternalCurrent(CA3_Pyramidal, 0.0f);
    		sim.runNetwork(0,1);
        }
        
        if (i >= 2 && i < 19)
		{
    		// Run the network for 500 ms
    		sim.runNetwork(0,500);
		}
		
        if (i == 19)
        {
            // Run the network for 498 ms
            sim.runNetwork(0,498);
        }
	}
  ```

### The Simulation Summary
After each simulation is executed, the output of the network to the terminal can be saved to a text file. This allows for a user to get a quick and informative summary of what the network activity looked like during the simulation. An example simulation summary is shown below for the synchronous network with 100 random Pyramidal cells transiently activated (more details can be found [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/HC_IM_02_16_ca3_snn_sync_baseline.txt)):

```
--------------------------------------------------------------------------------
| Stopwatch                                                                    |
|------------------------------------------------------------------------------|
|                  Tag         Start          Stop           Lap         Total |
|                start  00:00:00.000  00:00:00.057  00:00:00.057  00:00:00.057 |
|         setupNetwork  00:00:00.057  00:13:42.719  00:13:42.662  00:13:42.719 |
|           runNetwork  00:13:42.719  01:35:09.850  01:21:27.131  01:35:09.850 |
--------------------------------------------------------------------------------


********************    Simulation Summary      ***************************
Network Parameters: 	numNeurons = 89226 (numNExcReg:numNInhReg = 83.3:16.7)
			numSynapses = 250078223
			maxDelay = 2
Simulation Mode:	COBA
Random Seed:		10
Timing:			Model Simulation Time = 9 sec
			Actual Execution Time = 4887.13 sec
Average Firing Rate:	2+ms delay = 1.975 Hz
			1ms delay = 5.155 Hz
			Overall = 2.505 Hz
Overall Spike Count Transferred:
			2+ms delay = 0
			1ms delay = 0
Overall Spike Count:	2+ms delay = 1321930
			1ms delay = 689466
			Total = 2011396
*********************************************************************************
```

We can view details such as the duration necessary for the network structure to be created (roughly 14 minutes to create a network of ~90K neurons and ~250M synapses) and the duration of time to run a nine second simulation (~1.5 hours). Additionally, we can view the mean firing rate of the network and how many total spikes were generated. Further information regarding how each neuron type fired in 500 ms intervals can be found near the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_baseline/HC_IM_02_16_ca3_snn_sync_baseline.txt).

## Choosing a network to run
There are three directories from which SNNs can be simulated: [ca3_example_net_02_26_21](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21), where a scaled-down version of the model can be simulated, [synchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/synchronous) where full-scale versions activated by synchronous stimulation can be simulated, and the [asynchronous](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/asynchronous) where full-scale versions activated by asynchronous stimulation can be simulated. Within both the synchronous and asynchronous directories, three full-scale model versions can be simulated -- the baseline, class, and archetype SNNs. The network features are broadly as follows: the baseline SNN maintains neuron and connection-type specificity; the class SNN maintains neuron-type specificity while removing connection-type specificity; and the archetype SNN maintains connection-type specificity while removing neuron-type specificity.

## Installation and Simulation of the SNNs

### Ubuntu Users
For users compiling and running simulations with Ubuntu, the following steps will need to be taken:
  
1. Update the bashrc from your home directory (/home/username) with the following settings, which will load all modules necessary to compile and install CARLsim, along with compiling and running the simulations:

  ```
  nano ~/.bashrc
  ```

  ```
  # CARLsim4 related
  export PATH=/path/to/cuda/bin:$PATH # path to CUDA bin
  export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH # path to CUDA library
  
  # CARLsim4 mandatory variables
  export CARLSIM4_INSTALL_DIR=/home/username/CARL_hc # path to install CARL directory
  export CUDA_PATH=/path/to/cuda # path to CUDA samples necessary for compilation
  export CARLSIM_CUDAVER=10 # cuda version installed; 10 used in example
  export CUDA_MAJOR_NUM=7 # compute capability major revision number; 7 used in example
  export CUDA_MINOR_NUM=0 # compute capability major revision number; ; 0 used in example
  
  # CARLsim4 optional variables
  export CARLSIM_FASTMATH=0
  export CARLSIM_CUOPTLEVEL=3
  ```
  
2. Switch to your home directory and download the repository from GitHub into a folder name of your choice (CARLsim4_hc used in the example below):

  ```
  cd /home/username
  git clone https://github.com/UCI-CARL/CARLsim4.git -b feat/meansdSTPPost_hc CARLsim4_hc
  
  ```
  
3. Switch to the newly created CARLsim4 folder

  ```
  cd CARLsim4_hc
  ```
  
4. Make and install the CARLsim4 software:

  ```
  make distclean && make -j32
  make install
  ```
  
5. Switch to the directory of the network that you would like to simulate (the code below uses the [example network](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21)), and run the following commands:

  ```
  # Switch directory
  cd /home/username/CARLsim4_hc/projects/ca3_example_net_02_26_21

  # Create the syntax for the SNN to simulate
  python3 generateSNNSyntax.py
  ```
  
6. The example network uses an order of magnitude less neurons than the full-scale network, and doesn't involve the monitoring of the membrane potential due to memory constraints, so the generateCONFIGStateSTP.h [header file](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21/generateCONFIGStateSTP.h) needs to now be updated:

  ```
  nano generateCONFIGStateSTP.h
  ```
  
  ```
  // These variable declarations are at the beginning of the header file 
  int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 328,
                                INHIBITORY_NEURON, 0, GPU_CORES
				    );

  int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 190,
                                INHIBITORY_NEURON, 0, GPU_CORES
				      );

  int CA3_Basket = sim.createGroup("CA3_Basket", 51,
                                INHIBITORY_NEURON, 0, GPU_CORES
				  );

  int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 66,
                                INHIBITORY_NEURON, 0, GPU_CORES
				  );

  int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 463,
                                INHIBITORY_NEURON, 0, GPU_CORES
					);

  int CA3_Ivy = sim.createGroup("CA3_Ivy", 233,
                                INHIBITORY_NEURON, 0, GPU_CORES
			       );

  int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 152,
                                INHIBITORY_NEURON, 0, GPU_CORES
				     );

  int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 7436,
                                EXCITATORY_NEURON, 0, GPU_CORES
				     );

  // These commands are at the end of the header file 
  
  // sim.setNeuronMonitor(CA3_QuadD_LM, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_Axo_Axonic, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_Basket, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_BC_CCK, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_Bistratified, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_Ivy, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
  // sim.setNeuronMonitor(CA3_Pyramidal, "DEFAULT");
  ```

8. Compile the SNN:

  ```
  # Clear the contents of the results directory and any previous version of executables, and then compile the SNN to create a new executable
  make distclean && make -j32
  ```
  
9. Run the compiled simulation in the background using nohup and output the simulation summary to a text file

  ```
  nohup ./ca3_snn_GPU > HC_IM_02_26_ca3_example_net_results.txt &
  ```
  
10. Once the simulation has been finished (either by checking the squeue command to see if the simulation is still running and/or checking the email provided that will update you when the simulation has finished), view the simulation summary:

  ```
  cat HC_IM_02_26_ca3_example_net_results.txt
  ```



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
  
6. Switch to the directory of the network that you would like to simulate (the code below uses the [example network](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21)), and run the following commands:

  ```
  # Switch directory
  cd /scratch/username/CARLsim4_hc/projects/ca3_example_net_02_26_21

  # Create the syntax for the SNN to simulate
  python generateSNNSyntax.py
  ```
  
7. The example network uses an order of magnitude less neurons that the full-scale network, so the generateCONFIGStateSTP.h [header file](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_example_net_02_26_21/generateCONFIGStateSTP.h) needs to now be updated:

  ```
  nano generateCONFIGStateSTP.h
  ```
  
  ```
  int CA3_QuadD_LM = sim.createGroup("CA3_QuadD_LM", 328,
                                INHIBITORY_NEURON, 0, GPU_CORES
				    );

  int CA3_Axo_Axonic = sim.createGroup("CA3_Axo_Axonic", 190,
                                INHIBITORY_NEURON, 0, GPU_CORES
				      );

  int CA3_Basket = sim.createGroup("CA3_Basket", 51,
                                INHIBITORY_NEURON, 0, GPU_CORES
				  );

  int CA3_BC_CCK = sim.createGroup("CA3_BC_CCK", 66,
                                INHIBITORY_NEURON, 0, GPU_CORES
				  );

  int CA3_Bistratified = sim.createGroup("CA3_Bistratified", 463,
                                INHIBITORY_NEURON, 0, GPU_CORES
					);

  int CA3_Ivy = sim.createGroup("CA3_Ivy", 233,
                                INHIBITORY_NEURON, 0, GPU_CORES
			       );

  int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 152,
                                INHIBITORY_NEURON, 0, GPU_CORES
				     );

  int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 7436,
                                EXCITATORY_NEURON, 0, GPU_CORES
				     );
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

## A framework to test hypotheses of the hippocampal formation
The examples shown above describe how a network model of CA3 consisting of eight neuron types and fifty-one connection types can be utilized to test the network's stability and robustness to a transient, synchronous or asynchronous stimulation protocol. What if we wanted to test different hypotheses regarding CA3, or other subregions of the hippocampal formation? Hippocampal network models created in CARLsim are flexible enough to test additional hypotheses, which we provide an example of below.

Suppose we wanted to understand how different representative cell types in area CA3 were involved in pattern storage and completion. One way we could approach this is by using the cell types involved in the [archetype network](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/synchronous/ca3_snn_GPU_02_16_20_HC_IM_archetype) with a population of dentate gyrus granule cells as external input to the network. The following code will walk through how to create and simulate such a network and the scenario of pattern storage and completion. A directory containing the code to run this example can be found [here](https://github.com/UCI-CARL/CARLsim4/tree/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion).

1. Declare groups for each representative neuron type, along with Izhikevich parameter sets, how they connect to other representative neuron types, and their short-term plasticity rules. Additionally we set the max synaptic weight to 5 nS for each connection type (more details can be found [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/generateCONFIGStateSTP.h)):

```
int CA3_Basket = sim.createGroup("CA3_Basket", 3089,
                              INHIBITORY_NEURON, 0, GPU_CORES
			      	);
                              
int CA3_MFA_ORDEN = sim.createGroup("CA3_MFA_ORDEN", 11771,
                              INHIBITORY_NEURON, 0, GPU_CORES
			      	   );
                              
int CA3_Pyramidal = sim.createGroup("CA3_Pyramidal", 74366,
                              EXCITATORY_NEURON, 0, GPU_CORES
			      	   );

int DG_Granule = sim.createSpikeGeneratorGroup("DG_Granule", 394502,
                              EXCITATORY_NEURON, 0, GPU_CORES
			      		      );
                              
sim.setNeuronParameters(CA3_Basket, 45.0, 0.0, 0.9951729, 0.0,
                                                -57.506126, 0.0, -23.378766, 0.0, 0.003846186,
                                                0.0, 9.2642765, 0.0, 18.454934,
                                                0.0, -47.555661, 0.0,
                                                -6.0, 0.0, 1
		       );
                     
sim.setNeuronParameters(CA3_MFA_ORDEN, 209.0, 0.0, 1.37980713457205, 0.0,
                                                -57.076423571379, 0.0, -39.1020427841762, 0.0, 0.00783805979364104,
                                                0.0, 12.9332855397722, 0.0, 16.3132681887705,
                                                0.0, -40.6806648852695, 0.0,
                                                0.0, 0.0, 1
		       );
                     
sim.setNeuronParameters(CA3_Pyramidal, 366.0, 0.0, 0.792338703789581, 0.0,
                                                -63.2044008171655, 0.0, -33.6041733124267, 0.0, 0.00838350334098279,
                                                0.0, -42.5524776883928, 0.0, 35.8614648558726,
                                                0.0, -38.8680990294091, 0.0,
                                                588.0, 0.0, 1
		       );
                     
sim.connect(CA3_Basket, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 5.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 3.281611994f, 0.0f
	   );
                                       
sim.connect(CA3_Basket, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 5.0f), 0.005f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.808726221f, 0.0f
	   );
                                       
sim.connect(CA3_Basket, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 5.0f), 0.15f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.572405696f, 0.0f
	   );
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Basket, "random", RangeWeight(0.0f, 0.55f, 5.0f), 0.0072882240621001f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.972333716f, 0.0f
	   );
                                       
sim.connect(CA3_MFA_ORDEN, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 5.0f), 0.00210548528014741f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.552656079f, 0.0f
	   );
                                       
sim.connect(CA3_MFA_ORDEN, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 5.0f), 0.0417555599977689f,
                                          RangeDelay(1), RadiusRF(-1.0), SYN_PLASTIC, 1.360315289f, 0.0f
	   );
                                       
sim.connect(CA3_Pyramidal, CA3_Basket, "random", RangeWeight(0.0f, 1.45f, 5.0f), 0.0197417562762975f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 1.172460639f, 0.0f
	   );
                                   
sim.connect(CA3_Pyramidal, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 1.25f, 5.0f), 0.0209934225689348f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.88025265f, 0.0f
	   );
                                   
sim.connect(CA3_Pyramidal, CA3_Pyramidal, "random", RangeWeight(0.0f, 0.55f, 5.0f), 0.0250664662231983f,
                                      RangeDelay(1,2), RadiusRF(-1.0), SYN_PLASTIC, 0.553062478f, 0.0f
	   );

sim.connect(DG_Granule, CA3_Basket, "random", RangeWeight(0.0f, 0.65f, 2.0f), 0.001f,
                                          RangeDelay(1,10), RadiusRF(-1.0), SYN_PLASTIC, 1.4977493f, 0.0f
	   );
                                       
sim.connect(DG_Granule, CA3_MFA_ORDEN, "random", RangeWeight(0.0f, 0.75f, 2.0f), 0.001f,
                                          RangeDelay(1,10), RadiusRF(-1.0), SYN_PLASTIC, 1.35876774f, 0.0f
	   );
                                       
sim.connect(DG_Granule, CA3_Pyramidal, "random", RangeWeight(0.0f, 1.45f, 2.0f), 0.002f,
                                          RangeDelay(1,10), RadiusRF(-1.0), SYN_PLASTIC, 1.262911855f, 0.0f
	   );
                                   
sim.setSTP(CA3_Basket, CA3_Basket, true, STPu(0.38950627465000004f, 0.0f),
                                         STPtauU(11.19042564f, 0.0f),
                                         STPtauX(689.5059466f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.007016545f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_Basket, CA3_MFA_ORDEN, true, STPu(0.301856475f, 0.0f),
                                         STPtauU(19.60369075f, 0.0f),
                                         STPtauX(581.9355018f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.230610278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_Basket, CA3_Pyramidal, true, STPu(0.12521945645000002f, 0.0f),
                                         STPtauU(16.73589406f, 0.0f),
                                         STPtauX(384.3363321f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.63862234f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Basket, true, STPu(0.36184299919999996f, 0.0f),
                                         STPtauU(15.70448009f, 0.0f),
                                         STPtauX(759.1190877f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(3.896195604f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_MFA_ORDEN, true, STPu(0.2855712375f, 0.0f),
                                         STPtauU(22.52027885f, 0.0f),
                                         STPtauX(642.0975453f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.533747322f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_MFA_ORDEN, CA3_Pyramidal, true, STPu(0.11893441670000002f, 0.0f),
                                         STPtauU(20.61711347f, 0.0f),
                                         STPtauX(496.0484093f, 0.0f),
                                         STPtdAMPA(5.0f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(7.149050278f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(CA3_Pyramidal, CA3_Basket, true, STPu(0.12174287290000001f, 0.0f),
                                     STPtauU(21.16086172f, 0.0f),
                                     STPtauX(691.4177768f, 0.0f),
                                     STPtdAMPA(3.97130389f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f)
	  );
                                 
sim.setSTP(CA3_Pyramidal, CA3_MFA_ORDEN, true, STPu(0.14716404225000002f, 0.0f),
                                     STPtauU(29.01335489f, 0.0f),
                                     STPtauX(444.9925289f, 0.0f),
                                     STPtdAMPA(5.948303553f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f)
	  );
                                 
sim.setSTP(CA3_Pyramidal, CA3_Pyramidal, true, STPu(0.27922089865f, 0.0f),
                                     STPtauU(21.44820657f, 0.0f),
                                     STPtauX(318.510891f, 0.0f),
                                     STPtdAMPA(10.21893984f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f)
	  );

sim.setSTP(DG_Granule, CA3_Basket, true, STPu(0.187709502f, 0.0f),
                                         STPtauU(30.28628071f, 0.0f),
                                         STPtauX(744.6556525f, 0.0f),
                                         STPtdAMPA(3.582783578f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.0f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
					 
sim.setSTP(DG_Granule, CA3_MFA_ORDEN, true, STPu(0.194481964f, 0.0f),
                                         STPtauU(48.64778619f, 0.0f),
                                         STPtauX(453.6458777f, 0.0f),
                                         STPtdAMPA(4.86667462f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.0f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
                                     
sim.setSTP(DG_Granule, CA3_Pyramidal, true, STPu(0.156887286f, 0.0f),
                                         STPtauU(42.00785645f, 0.0f),
                                         STPtauX(347.4434166f, 0.0f),
                                         STPtdAMPA(7.425713188f, 0.0f),
                                         STPtdNMDA(150.0f, 0.0f),
                                         STPtdGABAa(5.0f, 0.0f),
                                         STPtdGABAb(150.0f, 0.0f),
                                         STPtrNMDA(0.0f, 0.0f),
                                         STPtrGABAb(0.0f, 0.0f)
	  );
```

2. We set excitatory and inhibitory spike-time dependent plasticity for each neuron type, using default parameters (more details can be found near the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/generateCONFIGStateSTP.h)):

```
sim.setESTDP(CA3_Basket, true, STANDARD, ExpCurve(0.1f, 20.0f, -0.1f, 20.0f));

sim.setISTDP(CA3_Basket, true, STANDARD, ExpCurve(-0.1f, 20.0f, 0.1f, 20.0f));

sim.setESTDP(CA3_MFA_ORDEN, true, STANDARD, ExpCurve(0.1f, 20.0f, -0.1f, 20.0f));

sim.setISTDP(CA3_MFA_ORDEN, true, STANDARD, ExpCurve(-0.1f, 20.0f, 0.1f, 20.0f));

sim.setESTDP(CA3_Pyramidal, true, STANDARD, ExpCurve(0.1f, 20.0f, -0.1f, 20.0f));

sim.setISTDP(CA3_Pyramidal, true, STANDARD, ExpCurve(-0.1f, 20.0f, 0.1f, 20.0f));
```

3. Create SpikeMonitors and NeuronMonitors for each neuron type (more details can be found at the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/generateCONFIGStateSTP.h) and at the top of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/generateSETUPStateSTP.h)):

```
sim.setNeuronMonitor(CA3_Basket, "DEFAULT");

sim.setNeuronMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
sim.setNeuronMonitor(CA3_Pyramidal, "DEFAULT");

sim.setSpikeMonitor(CA3_Basket, "DEFAULT");

sim.setSpikeMonitor(CA3_MFA_ORDEN, "DEFAULT");
                                 
sim.setSpikeMonitor(CA3_Pyramidal, "DEFAULT");
```

4. Create a [PoissonRate object](http://uci-carl.github.io/CARLsim4/ch6_input.html#ch6s1s1_poisson_rate) for the Granule cell population and set the mean firing rate of all neurons to 0.4 Hz. This provides a constant source of random input to the network exhibited during baseline network activity (other forms of network stimulation are described [here](http://uci-carl.github.io/CARLsim4/ch6_input.html); more details can also be found [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/generateSETUPStateSTP.h)):

```
int DG_Granule_frate = 100.0f;

PoissonRate DG_Granule_rate(394502, true); // create PoissonRate object for all Granule cells
DG_Granule_rate.setRates(0.4f); // set all mean firing rates for the object to 0.4 Hz
sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // link the object with defined Granule cell group, with refractory period 1 ms
```

5. In the main simulation script file, we now declare variables and vectors that will be used to select a subset of the granule cell population to increase their firing rates. Ten granule cells are chosen from the assigned set of {0,5,10,...,45} (more details can be found near the middle of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
// Declare variables that will store the start and end ID for the neurons
// in the granule group
int DG_start = sim.getGroupStartNeuronId(DG_Granule);
int DG_end = sim.getGroupEndNeuronId(DG_Granule);
int DG_range = (DG_end - DG_start) + 1;

// Create a vector that is the length of the number of neurons in the granule population
std::vector<int> DG_vec( boost::counting_iterator<int>( 0 ),
                         boost::counting_iterator<int>( DG_range ));
			 
// Define the number of granule cells to fire
int numGranuleFire = 10;

std::vector<int> DG_vec_A;

// Define the location of those granule cells so that we choose the same granule cells each time we call setRates
for (int i = 0; i < numGranuleFire; i++)
{
    DG_vec_A.push_back(5*(i+1));
}
```
6. Before the first simulation begins, the newly created network structure be saved by calling the saveSimulation function (more details can be found near the middle of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
 sim.saveSimulation("ca3SNN1.dat", true); // define where to save the network structure to and save synapse info
```

7. A simulation protocol is now defined which runs the simulation for 10 seconds, where halfway through the simulation the ten granule cells selected have their firing rates elevated to the defined firing rate of 100 Hz within two 25 ms time windows (corresponding to gamma cycles; more details can be found near the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
// run for a total of 10 seconds
// at the end of each runNetwork call, SpikeMonitor stats will be printed
for (int i=0; i<20; i++)
{
	if (i >= 0 && i < 10) 
	{
		sim.runNetwork(0,500); // run network for 500 ms
	}
	
	if ( i == 10)
	{
		for (int j = 0; j < numGranuleFire; j++)
		{
			int randGranCell = DG_vec.front() + DG_vec_A[j]; // choose the jth random granule cell
			DG_Granule_rate.setRate(DG_vec.at(randGranCell), DG_Granule_frate); // set the firing rate for the jth random granule cell
		}
		sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation           
		sim.runNetwork(0,25); // run network for 25 ms
	}
	
	if (i == 11)
	{
		DG_Granule_rate.setRates(0.4f); // set the firing rates for all granule cells back to baseline firing rate
		sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation          
        sim.runNetwork(0,75); // run network for 75 ms
	}

	if ( i == 12)
	{
		for (int j = 0; j < numGranuleFire; j++)
		{
			int randGranCell = DG_vec.front() + DG_vec_A[j]; // choose the jth random granule cell
			DG_Granule_rate.setRate(DG_vec.at(randGranCell), DG_Granule_frate); // set the firing rate for the jth random granule cell
		}
		sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation           
		sim.runNetwork(0,25); // run network for 25 ms
        }
		
	if (i == 13)
	{
		DG_Granule_rate.setRates(0.4f); // set the firing rates for all granule cells back to baseline firing rate
		sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation
		sim.runNetwork(0,75); // run network for 75 ms
	}
        
	if (i >=14 && i < 20)
	{
		sim.runNetwork(0,500); // run network for 500 ms
	}
}
```

8. The outcome of the simulation can now be saved by calling the saveSimulation function, which will save the network structure (more details can be found near the bottom of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
 sim.saveSimulation("ca3SNN2.dat", true); // define where to save the network structure to and save synapse info
```

9. The saved network can now be loaded in additional simulation runs, by calling the loadSimulation function before the setupNetwork is called (invokes the SETUP simulation state; more details can be found near the middle of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
// before calling setupNetwork, call loadSimulation
FILE* fId = NULL;
fId = fopen("ca3SNN2.dat", "rb");
sim.loadSimulation(fId);
```

10. Now the setupNetwork function can be called and after its completion the connection to the network structure file can be closed (more details can be found near the middle of the page [here](https://github.com/UCI-CARL/CARLsim4/blob/feat/meansdSTPPost_hc/projects/ca3_snn_GPU_06_25_21_ca3_snn_pattern_completion/src/main_ca3_snn_GPU.cpp)):

```
sim.setupNetwork();

// ... wait until after setupNetwork is called
fclose(fId);
```

11. The same stimulation protocol from steps 5-10 can be called in additional runs to save and load the network structure to make the pattern stored more robustly. Once the user believes that the pattern has been robustly stored within the network structure, the stimulation protocol can be changed to test for pattern completion by selecting fewer neurons than the original set of granule cells. In the example below, only seven ({0,5,10,...,30}) of the original ten granule cells are selected for testing pattern completion:

```
// Declare variables that will store the start and end ID for the neurons
// in the granule group
int DG_start = sim.getGroupStartNeuronId(DG_Granule);
int DG_end = sim.getGroupEndNeuronId(DG_Granule);
int DG_range = (DG_end - DG_start) + 1;

// Create a vector that is the length of the number of neurons in the granule population
std::vector<int> DG_vec( boost::counting_iterator<int>( 0 ),
                         boost::counting_iterator<int>( DG_range ));
			 
// Define the number of granule cells to fire
int numGranuleFire = 7;

std::vector<int> DG_vec_A;

// Define the location of those granule cells so that we choose the same granule cells each time we call setRates
for (int i = 0; i < numGranuleFire; i++)
{
    DG_vec_A.push_back(5*(i+1));
}
```

12. Steps 6-10 can be followed again to see if the stored pattern was successfully recalled, which can be observed in MATLAB using the [CARLsim MATLAB Analysis Toolbox](http://uci-carl.github.io/CARLsim4/ch9_matlab_oat.html) and custom built functions and scripts used to interact with the Toolbox [here](https://github.com/Hippocampome-Org/snn_analysis/).
