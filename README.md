README for CARLsim SNN simulator
-------------------------------------------------------------------------------

CARLsim is available from: http://socsci.uci.edu/~jkrichma/CARLsim/.

For a list of all people who have contributed to this project, please refer to 
AUTHORS.

For more detailed installation instructions, please refer to INSTALL.

For a log of source code changes, refer to CHANGELOG.

For a description of added features please refer to the RELEASE_NOTES.

### REQUIREMENTS
**Operating System**: Windows 7 or Linux (for a complete list of supported OS’s
see the INSTALL file)

**Software**: NVIDIA CUDA drivers/toolkits/SDKs version 5.0 and up shold be installed and
configured correctly.

**Hardware**: An NVIDIA CUDA-capable GPU device with a compute capability of 2.0
or higher is required.

### QUICKSTART INSTALLATION
**For Linux**:
Set the required environment variables in your .bashrc file. They are described
below:

`CARLSIM_LIB_DIR`: desired installation location of the CARLsim library
`CUDA_INSTALL_PATH`:  location of CUDA installation
`CARLSIM_CUDAVER`: CUDA driver/toolkit version number (major only e.g. 6.5 is 6.0)
`CUDA_MAJOR_NUM`: Major number of CUDA GPU device
`CUDA_MINOR_NUM`: Minor number of CUDA GPU device

You can copy and paste the below code into your .bashrc file replacing their
values with values appropriate for your system.

```
# set CARLsim variables
export CARLSIM_LIB_DIR=/opt/CARL/CARLsim
export CUDA_INSTALL_PATH=/usr/local/cuda
export CARLSIM_CUDAVER=6
export CUDA_MAJOR_NUM=3
export CUDA_MINOR_NUM=5
```

Enter the unzipped CARLsim directory (same directory in which this README is
located) and run the following:

```
make && sudo make install
```

This will compile and install the CARLsim library.

Note: ‘sudo’ is only necessary when installing CARLsim in a location other than
your home directory.

**For Windows**:


TODO: TS


### SOURCE CODE DIRECTORY DESCRIPTION
Below is the directory layout of the CARLsim source code.

<pre>
├── carlsim                       # CARLsim source code folder
│   ├── connection_monitor          # Utility to record synaptic data
│   ├── group_monitor               # Utility to record neuron group data
│   ├── interface                   # CARLsim interface (public user interface)
│   ├── kernel                      # CARLsim core functionality
│   ├── server                      # Utility to implement real-time server functionality
│   ├── spike_monitor               # Utility to record neuron spike data
│   └── test                        # Google test regression suite tests
├── doc                           # CARLsim documentation generation folder
│   └── html                        # Generated documentation in html
├── projects                      # User projects directory
│   └── hello_world                 # Project template for new users
├── tools                         # CARLsim tools that are not built-in
│   ├── ecj_pti                     # Automated parameter-tuning interface using ECJ
│   ├── eo_pti                      # Automated parameter-tuning interface using EO (deprecated)
│   ├── offline_analysis_toolbox    # Collection of MATLAB scripts for data analysis
│   ├── simple_weight_tuner         # Simple weight parameter-tuning tool
│   ├── spike_generators            # Collection of input spike generation tools
│   └── visual_stimulus             # Collection of MATLAB/CARLsim tools for visual stimuli
├── tutorial                      # Source code for CARLsim tutorials
│   ├── colorblind
│   ├── colorcycle
│   ├── common
│   ├── dastdp
│   ├── istdp
│   ├── multiGPU
│   ├── orientation
│   ├── random
│   ├── spikeInfo
│   ├── tuneFiringRatesECJ
│   └── v1MTLIP
└── user-guide                    # Source code for user-guide documentation
    ├── 10_ecj                      # Chapter 10: Paramter-tuning with ECJ
    ├── 11_regression_suite         # Chapter 11: Regression Suite Testing with Google Tests
    ├── 12_advanced_topics          # Chapter 12: Advanced Topics
    ├── 13_example_networks         # Chapter 13: Example SNNs
    ├── 1_getting_started           # Chapter 01: Getting Started with CARLsim
    ├── 2_basics                    # Chapter 02: Basics
    ├── 3_neurons_synapses_groups   # Chapter 03: Neurons, Synapses, and Groups
    ├── 4_connections               # Chapter 04: Connections
    ├── 5_synaptic_plasticity       # Chapter 05: Synaptic Plasticity and Learning
    ├── 6_input                     # Chapter 06: Getting Input into CARLsim
    ├── 7_monitoring                # Chapter 07: Data Monitoring
    ├── 8_saving_and_loading        # Chapter 08: Saving and Loading CARLsim Simulations
    └── 9_visualization             # Chapter 09: Visualization with the Offline Analysis Toolbox (OAT)
</pre>


* Main directory - contains the Makefile, documentation files, and other
directories.

* carlsim - contains the source code of CARLsim.

* doc - location of generated documentation.

* projects - meant to contain user’s CARLsim projects, all it is not
necessary to place project code here as long as the user.mk is configured
correctly and the CARLsim library is installed.

* tools - contains additional tools to be used with CARLsim.

* tutorial - a collection of CARLsim example SNNs with explanation and
documentation.

### MAKEFILE STRUCTURE DESCRIPTION

The Makefile is composed of individual include files (.mk).  The user must
configure their user.mk include file to tell CARLsim about the CUDA version,
CUDA capability of the GPU, and the desired installation of the CARLsim
library.

Each example directory has it’s own Makefile that will build the example
program after the CARLsim library has been installed. To make an example
program, change directories to the name of your example directory
(e.g. examples/random) and type ‘make program_name’ where the ‘program_name’
is the name of the directory (e.g. random).

Example programs were written to be executed in the same directory as their
source code.

### CARLSIM DEVELOPMENT

To compile an example or project from the CARLsim source code, you can refer
to the src.mk files found in each example subdirectory. Essentially, you have
to set the CARLSIM_SRC_PATH and USER_MK_PATH variables in the src.mk files. To
run make using these files simply type:

‘make -f src.mk’

and make will run the default targets for that src.mk file. If you want to
run a specific target from src.mk (let’s call it ‘random’), then you would
run the command:

‘make -f src.mk random’

This is mainly used by the devs who write CARLsim.

### Generating Documentation
To generate documentation, doxygen must be installed. In Linux, documentation
can be generated by entering the ‘doc’ directory and typing `./makedoc`. The
documentation will be generated in the ‘tutorials’ directory.

### FAQ

1. Why do I need to have CUDA installed for CARLsim’s CPU mode?

CARLsim was written primarily as a GPU SNN simulator with the CPU mode being
used mainly for validation and prototyping. Because of this, code for the CPU
and GPU mode is intermingled. CARLsim v4.0 will separate these code-bases
completely.

2. How do I install the parameter-tuning tools?

Instructions for installing the parameter-tuning tools can be found in their
respective directories in the tools directory. For example, if you want to
install the automated parameter-tuning interface using ECJ,the installation
instructions are located in tools/ecj_pti.

### PTI to be moved somewhere else
Parameter Tuning Interface (PTI): Currently the parameter tuning library
that uses Evolving Objects (EO) is deprecated. The new version of the
parameter tuning library will uses Evolutionary Computations in Java (ECJ)
and can be installed as follows. Assuming ECJ version 22 or greater is
installed:

1) Set the ECJ_DIR and ECJ_PTI_DIR variables in user.mk.

2) Change the current directory to ’tools/carlsim_addons/ecj_pti’.

3) Type ‘make && sudo make install’

4) Refer to the tuneFiringRatesECJ example found in examples/tuneFiringRatesECJ
   for how to use CARLsim and ECJ to tune SNNs.

TO UNINSTALL:
CARLsim: Remove the folder where you installed the CARLsim library. This
folder is located in $(CARLSIM_LIB_DIR).

Type ‘make help’ for additional information.

