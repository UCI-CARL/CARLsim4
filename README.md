
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

Open the user.mk file and edit the following entries:

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

Then type ‘make && sudo make install’. This will compile and install the
CARLsim library.

TODO: fix the PTI installation instructions.

If BOTH the NVIDIA CUDA drivers/toolkits/SDKs and the Evolving Objects 
libraries are installed and configured correctly:

Type 'make pti' to make the pti library.

Type 'make install' to install the pti library.

Type 'make pti_examples' to compile the pti examples.

TO UNINSTALL:
CARLsim: Remove the folder where you installed the CARLsim library. This
folder is located in $(CARLSIM_LIB_INSTALL_DIR).

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

* examples - contains all the examples for CARLsim and CARLsim PTI. Each
example is in its own subfolder (along with input videos, scripts, and
results). examples/common contains support files that multiple examples use.

* include -  contains all the header files for the PTI.

* libpti - contains the source code and Makefile includes for the PTI.

* test - contains a regression suite


### MAKEFILE STRUCTURE DESCRIPTION


Variable naming conventions - Environment variables that could be defined 
outside the Makefile are all caps.  Variables that are used only in the 
Makefile are not capitalized.

The Makefile is composed of individual include files.  makefile.mk contains
additional makefile definitions. carlsim.mk and libpti.mk contain specific
build rules and definitions for CARLsim and the PTI library. Each example has
a corresponding include file.
