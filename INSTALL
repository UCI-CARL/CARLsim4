This code has been predominately developed under Linux and the build instructions reflect that.  
While the code should function under MS Windows these instructions are not applicable.  We will
hopefully release instructions in the future on how to setup the code base under Visual Studio 
on Windows based systems.

To run the code the NVIDIA Cuda Driver and SDK must be installed and functional.  You must first
compile the Cuda SDK examples (at least one) in order to have cutil compiled (libcutil).  Once 
that has been accomplished, you need to setup an environment variable that points to where the 
nVidia SDK is located.  So, under most linux systems do something like the following:

export NVIDIA_SDK=(SOME PATH)/NVIDIA_GPU_Computing_SDK

Where (SOME PATH) needs to be filled in with where the SDK is installed.

To then run the simulator you have several choices: random, orientation, colorblind, colorcycle, 
and rdk.

To run them first compile the one you desire by typing make followed by the name, such as:

make random

Then type ./a.out and the simulation will start.

To run Matlab from the simulators directory (or change Matlab's current directory to here).

The rdk and orientation examples can only be run on a system with a compatible GPU (compute 
capability 1.3 or higher).  The colorblind example most likely must be run on CPU because at 
high spatial resolution it requires more memory than most GPUs have available.
