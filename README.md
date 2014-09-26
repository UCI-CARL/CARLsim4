
README for CARLsim SNN simulator
-------------------------------------------------------------------------------

CARLsim is available from: http://socsci.uci.edu/~jkrichma/CARLsim/.

For a list of all people who have contributed to this project, please refer to 
AUTHORS.

For installation instructions, please refer to INSTALL.

For a log of source code changes, refer to CHANGELOG.

For a description of added features please refer to the RELEASE_NOTES.

### QUICKSTART INSTALLATION

If the NVIDIA CUDA drivers/toolkits/SDKs are installed and configured 
correctly:

Open the user.mk file and set the following entries to correct values.
They come with default values that might not be correct for the user:

<pre>
# desired installation path of libcarlsim and headers
CARLSIM_LIB_INSTALL_DIR ?= /opt/CARL/CARLsim

# cuda capability major version number for GPU device
CUDA_MAJOR_NUM ?= 1
# cuda capability minor version number for GPU device
CUDA_MINOR_NUM ?= 3

# $(OPT_LEVEL): set to 1, 2, or 3 if you want to use optimization.  Default: 0.
# $(DEBUG_INFO): set to 1 to include debug info, set to 0 to not include
# debugging info.  Default: 0.
CARLSIM_CUDAVER ?= 3
CARLSIM_FASTMATH ?= 0
CARLSIM_CUOPTLEVEL ?= 0
CARLSIM_DEBUG ?= 0
</pre>

CARLsim: Type ‘make && sudo make install’. This will compile and install the
CARLsim library.

PTI: Currently the parameter tuning library that uses Evolving Objects (EO) is
deprecated. The new version of the parameter tuning library will use 
Evolutionary Computations in Java (ECJ) but is not yet ready for release.

TO UNINSTALL:
CARLsim: Remove the folder where you installed the CARLsim library. This
folder is located in $(CARLSIM_LIB_INSTALL_DIR).

Type ‘make help’ for additional information.

### SOURCE CODE DIRECTORY DESCRIPTION

<pre>
.
├── carlsim                    # CARLsim source code folder
│   ├── interface              # CARLsim interface (public user interface)
│   ├── kernel                 # CARLsim core functionality
│   ├── server                 # CARLsim server (Windows only)
│   ├── spike_monitor          # CARLsim utility to record spiking activity
│   └── test                   # regression suite (requires google test)
├── doxygen                    # Doxygen documentation generation
├── examples                   # example SNNs
│   ├── colorblind             # V1-V4 color opponency model applied to colorblind test
│   ├── colorcycle             # V1-V4 color opponency model, spectrum of color responses
│   ├── common                 # miscellaneous utilities, such as stimulus generator and motion energy model
│   ├── dastdp                 # dopamine modulated STDP model
│   ├── orientation            # V1-V4 orientation-selective cells (using motion energy model)
│   ├── random                 # an 80-20 RAIN network with STDP
│   ├── simpleEA               # simple EA example using EO library (deprecated)
│   ├── SORFs                  # network that uses EA for self-organizing receptive fields
│   ├── spikeInfo              # example network showing off Spike Monitor functionality
│   ├── tuneFiringRatesEO      # simple network on how to use EA to tune firing rates using EO (deprecated)
│   └── v1MTLIP                # V1-MT-LIP motion energy model for direction/speed tuning and decision-making
├── projects                   # projects folder (put your project here)
└── tools                      # collection of CARLsim tools
    ├── carlsim_addons         # tools for extending CARLsim functionality
    │   ├── eo_pti             # paramter-tuning interface using EO (deprecated)
    │   ├── input_stimulus     # matlab scripts/C++ code for input generation
    │   └── spike_generators   # C++ code to generate advanced spike pattern inputs
    └── matlab_scripts         # scripts used to analyze CARLsim SNN output
</pre>

* Main directory - contains the Makefile, documentation files, and other
directories.

* carlsim - contains the source code of CARLsim

* doxygen -  contains doxygen code to generate documentation.

* examples - contains all the examples for CARLsim and CARLsim PTI. Each
example is in its own subfolder (along with input videos, scripts, and
results). examples/common contains support files that multiple examples use.

* projects - meant to contain user’s CARLsim projects, all it is not
necessary to place project code here as long as the user.mk is configured
correctly and the CARLsim library is installed.

* tools - contains additional tools to be used with CARLsim.


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
