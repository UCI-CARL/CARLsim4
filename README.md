<div align="center">
	<img src="http://socsci.uci.edu/~jkrichma/CARL-Logo-small.jpg" width="300"/>
</div>

# CARLsim 4

[![Build Status](https://travis-ci.org/UCI-CARL/CARLsim4.svg?branch=master)](https://travis-ci.org/UCI-CARL/CARLsim4)
[![Coverage Status](https://coveralls.io/repos/github/UCI-CARL/CARLsim4/badge.svg?branch=master)](https://coveralls.io/github/UCI-CARL/CARLsim4?branch=master)
[![Docs](https://img.shields.io/badge/docs-v4.0.0-blue.svg)](http://uci-carl.github.io/CARLsim4)
[![Google group](https://img.shields.io/badge/Google-Discussion%20group-blue.svg)](https://groups.google.com/forum/#!forum/carlsim-snn-simulator)

CARLsim is an efficient, easy-to-use, GPU-accelerated library for simulating large-scale spiking neural network (SNN) models with a high degree of biological detail. CARLsim allows execution of networks of Izhikevich spiking neurons with realistic synaptic dynamics on both generic x86 CPUs and standard off-the-shelf GPUs. The simulator provides a PyNN-like programming interface in C/C++, which allows for details and parameters to be specified at the synapse, neuron, and network level.

New features in CARLsim 4 include:
- Multi-GPU support
- Hybrid mode
- Multi-compartment and LIF point neurons
- etc.


## Installation

Detailed instructions for installing the latest stable release of CARLsim on Mac OS X / Linux / Windows
can be found in our [User Guide](http://uci-carl.github.io/CARLsim4/ch1_getting_started.html).

### Linux/MacOS

#### For Beginner

1. Download CARLsim 4 zip file by clicking on the `Clone or download` box in the top-right corner.

2. Unzip the source code.

3. Go into `CARLsim4` folder
   ```
   $ cd CARLsim4
   ```

4. Make and install
   ```
   $ make
   $ make install
   ```

5. Verify installation
   ```
   $ cd ~
   $ ls
   ```
   You will see `CARL` folder

6. Go back to `CARLsim4` folder and start your own project! The "Hello World" project is a goot starting point for this.
   Make sure it runs:
   ```
   $ cd CARLsim4
   $ cd projects/hello_world
   $ make
   $ ./hello_world
   ```

#### For Advanced User and Developer

1. Fork CARLsim 4 by clicking on the `Fork` box in the top-right corner.

2. Clone the repo, where `YourUsername` is your actual GitHub user name:
   ```
   $ git clone --recursive https://github.com/YourUsername/CARLsim4
   $ cd CARLsim4
   ```
   Note the `--recursive` option: It will make sure Google Test gets installed.

3. Choose between stable release and latest development version:
   - For the stable release, use the `stable` branch:
     ```
     $ git checkout stable
     ```
   - For the latest development branch, you are already on the right branch (`master`).

4. Choose the installation directory: By default, the CARLsim library lives in `~/CARL/lib`, and CARLsim include files live in `~/CARL/include`.
    You can overwrite these by exporting an evironment variable called `CARLSIM4_INSTALL_DIR`:
    ```
    $ export CARLSIM4_INSTALL_DIR=/path/to/your/preferred/dir
    ```
    or
    ```
    $ export CARLSIM4_INSTALL_DIR=/usr/local
    ```
    if you want to install CARLsim library for all users

5. Make and install:
   ```
   $ make -j4
   $ sudo -E make install
   ```
   Note the `-E` flag, which will cause `sudo` to remember the `CARLSIM4_INSTALL_DIR`.

7. In order to make sure the installation was successful, you can run the regression suite:

   ```
   $ make test
   $ ./carlsim/test/carlsim_tests
   ```
   
8. Start your own project! The "Hello World" project is a goot starting point for this.
   Make sure it runs:

   ```
   $ cd projects/hello_world
   $ make
   $ ./hello_world
   ```

   You can easily create your own project based on this template using the `init.sh` script:

   ```
   $ cd projects
   $ ./init.sh project_name
   ```
   where `project_name` is the name of your new project.
   The script will copy all files from `hello_world/` to `project_name/`, make all required
   file changes to compile the new project, and add all new files to git.

#### Using CMake

1. Obtatin `CARLsim4`'s source code.

2. Create a build directory (you can make it anywhere)

   ```
   $ mkdir .build
   ```

3. Proceed into build directory and do configuration:

   ```
   $ cd .build
   $ cmake \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/usr/local/carlsim \
       -DCARLSIM_NO_CUDA=OFF \
       <path-to-carlsim>
   ```

   As you can see `cmake` accepts several options `-D<name>=<value>`: they define cmake variables.
   `CMAKE_BUILD_TYPE=Release` means that we are going to build release version of the library.
   If you need debug version then pass `Debug`.
   `CMAKE_INSTALL_PREFIX` specifies a directory which we are going to install the library into.
   `CARLSIM_NO_CUDA` switches on/off support of CUDA inside the library.
   `<path-to-carlsim>` must be replaced with the path to the CARLsim4's source directory.

4. Build:

   ```
   make -j <jobs-num>
   ```
   
   Set `<jobs-num>` to the number of logical processors your computer has plus one,
   this will employ parallel building.

5. Install:

   ```
   make install
   ```

### Windows

Simply download the code. Open and build `CARLsim.sln`. Run the "Hello World" project file
`projects\hello_world\hello_world.vcxproj`.


## Prerequisites

CARLsim 4 comes with the following requirements:
- (Windows) Microsoft Visual Studio 2015 or higher.
- (optional) CMake 3.0 or higher in case you want to build it using CMake.
- (optional) CUDA Toolkit 6.0 or higher. For platform-specific CUDA installation instructions, please navigate to 
  the [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-zone).
  This is only required if you want to run CARLsim in `GPU_MODE`. Make sure to install the 
  CUDA samples, too, as CARLsim relies on the file `helper_cuda.h`.
- (optional) A GPU with compute capability 2.0 or higher. To find the compute capability of your device please 
  refer to the [CUDA article on Wikipedia](http://en.wikipedia.org/wiki/CUDA).
  This is only required if you want to run CARLsim in `GPU_MODE`.
- (optional) MATLAB R2014a or higher. This is only required if you want to use the Offline Analysis Toolbox (OAT).

As of CARLsim 3.1 it is no longer necessary to have the CUDA framework installed. However, CARLsim development 
will continue to focus on the GPU implementation.

The current release has been tested on the following platforms:
- Ubuntu 16.04
- Mac OS X 10.11 (El Capitan)
- Windows 7/10
