/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#ifndef _CARLSIM_H_
#define _CARLSIM_H_

#include <stdint.h>		// uint64_t, uint32_t, etc.
#include <string>		// std::string
#include <vector>		// std::vector
#include <algorithm>

#include <carlsim_definitions.h>
#include <carlsim_datastructures.h>

// include the following core functionalities instead of forward-declaring, so that the user only needs to include
// carlsim.h
#include <callback.h>
#include <poisson_rate.h>
#include <spike_monitor.h>
#include <neuron_monitor.h>
#include <connection_monitor.h>
#include <group_monitor.h>
#include <linear_algebra.h>

class GroupMonitor;
class ConnectionMonitor;
class SpikeMonitor;
class SpikeGenerator;



// Cross-platform definition (Linux, Windows)
#if defined(WIN32) || defined(WIN64)
#include <Windows.h>

#include <float.h>
#include <time.h>

#ifndef isnan
#define isnan(x) _isnan(x)
#endif

#ifndef isinf
#define isinf(x) (!_finite(x))
#endif

#ifndef srand48
#define srand48(x) srand(x)
#endif

#ifndef drand48
#define drand48() (double(rand())/RAND_MAX)
#endif

#else

#include <pthread.h> // pthread
#include <sys/stat.h> // mkdir
#include <unistd.h> //unix thread affinity macros


#endif



/*!
 * \brief CARLsim User Interface
 * This class provides a user interface to the public sections of CARLsimCore source code. Example networks that use
 * this methodology can be found in the examples/ directory. Documentation is available on our website.
 *
 * This class provides a user interface to the public sections of CARLsimCore source code. Example networks that use
 * this methodology can be found in the examples/ directory. Documentation is available on our website.
 * This file is organized into different sections in the following way:
 * \verbatim
 *  ├── Public section
 *  │     ├── Public methods
 *  │     │     ├── Constructor / destructor
 *  │     │     ├── Setting up a simulation
 *  │     │     ├── Running a simulation
 *  │     │     ├── Plotting / logging
 *  │     │     ├── Interacting with a simulation
 *  │     │     ├── Getters / setters
 *  │     │     └── Set defaults
 *  │     └── Public properties
 *  └── Private section
 *        ├── Private methods
 *        └── Private properties
 * \endverbatim
 * Within these sections, methods and properties are ordered alphabetically.
 *
 */
class CARLsim {
public:
	// +++++ PUBLIC METHODS: CONSTRUCTOR / DESTRUCTOR +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief CARLsim constructor.
	 * Creates a new instance of class CARLsim. All input arguments are optional, but if specified will be constant
	 * throughout the lifetime of the CARLsim object.
	 *
	 * CARLsim allows execution on both generic x86 CPUs and standard off-the-shelf GPUs by specifying the simulation
	 * mode (CPU_MODE and GPU_MODE, respectively). When using the latter in a multi-GPU system, the user can also
	 * specify which CUDA device to use (param numGPUs, 0-indexed).
	 *
	 * The logger mode defines where to print all status, error, and debug messages. Logger mode can either be USER (for
	 * experiment-oriented simulations), DEVELOPER (for developing and debugging code), SHOWTIME (where only warnings
	 * and errors are printed to console), SILENT (e.g., for benchmarking, where no output is generated at all), or
	 * CUSTOM (where the user can specify the file pointers of all log files).
	 * In summary, messages are printed to the following locations, depending on the logger mode:
	 * \verbatim
	 *                 |    USER    | DEVELOPER  |  SHOWTIME  |   SILENT   |  CUSTOM
	 * ----------------|------------|------------|------------|------------|---------
	 * Status msgs     |   stdout   |   stdout   | /dev/null  | /dev/null  |    ?
	 * Errors/warnings |   stderr   |   stderr   |   stderr   | /dev/null  |    ?
	 * Debug msgs      | /dev/null  |   stdout   | /dev/null  | /dev/null  |    ?
	 * All msgs        | debug.log  | debug.log  |  debug.log | debug.log  |    ?
	 * \endverbatim
	 * Location of the CARLsim log file can be set in any mode using setLogFile.
	 * In mode CUSTOM, the other file pointers can be set using setLogsFpCustom.
	 *
	 * \param[in] netName network name
	 * \param[in] preferredSimMode CPU_MODE, GPU_MODE, or HYBRID_MODE
	 * \param[in] loggerMode USER, DEVELOPER, SILENT, or CUSTOM
	 * \param[in] ithGPUs on which GPU to establish a context (deprecated parameter)
	 * \param[in] randSeed random number generator seed
	 * \see setLogFile
	 * \see setLogsFpCustom
	 */
	CARLsim(const std::string& netName = "SNN", SimMode preferredSimMode = CPU_MODE, LoggerMode loggerMode = USER, int ithGPUs = 0, int randSeed = -1);
	~CARLsim();



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Connects a presynaptic to a postsynaptic group using fixed/plastic weights and a range of delay values
	 *
	 * This function is a shortcut to create synaptic connections from a pre-synaptic group grpId1 to a post-synaptic
	 * group grpId2 using a pre-defined primitive type (such as "full", "one-to-one", or "random"). Synapse weights
	 * will stay the same throughout the simulation (SYN_FIXED, no plasticity). All synapses will have the same delay.
	 * For more flexibility, see the other connect() calls.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId1     ID of the pre-synaptic group
	 * \param[in] grpId2     ID of the post-synaptic group
	 * \param[in] connType   connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in
	 *                       pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post.
	 *                       "full-no-direct": same as "full", but i-th neuron of grpId1 will not be connected to the
	 *                       i-th neuron of grpId2. "gaussian": distance-dependent weights depending on the RadiusRF
	 *                       struct, where neurons coding for the same location have weight initWt, and neurons lying
	 *                       on the border of the RF have weight 0.1*initWt.
	 * \param[in] wt         a struct specifying the range of weight magnitudes (initial value and max value). Weights
	 *                       range from 0 to maxWt, and are initialized with initWt. All weight values should be
	 *                       non-negative (equivalent to weight *magnitudes*), even for inhibitory connections.
	 *                       Examples:
	 *                         RangeWeight(0.1)         => all weights will be 0.1 (wt.min=0.1, wt.max=0.1, wt.init=0.1)
	 *                         RangeWeight(0.0,0.2)     => If pre is excitatory: all weights will be in range [0.0,0.2],
	 *                                                     and wt.init=0.0. If pre is inhibitory: all weights will be in
	 *                                                     range [-0.2,0.0], and wt.init=0.0.
	 *                         RangeWeight(0.0,0.1,0.2) => If pre is excitatory: all weights will be in range [0.0,0.2],
	 *                                                     and wt.init=0.1. If pre is inhibitory: all weights will be in
	 *                                                     range [-0.2,0.0], and wt.init=0.0.
	 * \param[in] connProb   connection probability
	 * \param[in] delay      A struct specifying the range of delay values (ms). Synaptic delays must be greater than or
	 *                       equal to 1 ms.
	 *                       Examples:
	 *                         RangeDelay(2) => all delays will be 2 (delay.min=2, delay.max=2)
	 *                         RangeDelay(1,10) => delays will be in range [1,10]
	 * \param[in] radRF      A struct specifying the radius of the receptive field (RF). A radius can be specified in 3
	 *                       dimensions x, y, and z (following the topographic organization of neurons as specified by
	 *                       Grid3D).
	 *                       Receptive fields will be circular with radius as specified. The 3 dimensions follow the
	 *                       ones defined by Grid3D.
	 *                       If the radius in one dimension is 0, no connections will be made in this dimension.
	 *                       If the radius in one dimension is -1, then all possible connections will be made in this
	 *                       dimension (effectively making RF of infinite size).
	 *                       Otherwise, if the radius is a positive real number, the RF radius will be exactly this
	 *                       number. Call RadiusRF with only one argument to make that radius apply to all 3 dimensions.
	 *                       Examples:
	 *                         * Create a 2D Gaussian RF of radius 10 in z-plane: RadiusRF(10, 10, 0)
	 *                           Neuron pre will be connected to neuron post iff (pre.x-post.x)^2+(pre.y-post.y)^2<=100
	 *                           and pre.z==post.z.
	 *                         * Create a 2D heterogeneous Gaussian RF (an ellipse) with semi-axes 10 and 5:
	 *                           RadiusRF(10, 5, 0)
	 *                           Neuron pre will be connected to neuron post iff
	 *                           (pre.x-post.x)/100 + (pre.y-post.y)^2/25 <= 1 and pre.z==post.z.
	 *                         * Connect all, no matter the RF (default): RadiusRF(-1, -1, -1)
	 *                         * Connect one-to-one: RadiusRF(0, 0, 0)
	 *                           Neuron pre will be connected to neuron post iff pre.x==post.x, pre.y==post.y,
	 *                           pre.z==post.z.
	 *                           Note: Use CARLsim::connect with type "one-to-one" instead.
	 * \param[in] synWtType  specifies whether the synapse should be of fixed value (SYN_FIXED) or plastic (SYN_PLASTIC)
	 * \param[in] mulSynFast a multiplication factor to be applied to the fast synaptic current (AMPA in the case of
	 *                       excitatory, and GABAa in the case of inhibitory connections). Default: 1.0
	 * \param[in] mulSynSlow a multiplication factor to be applied to the slow synaptic current (NMDA in the case of
	 *                       excitatory, and GABAb in the case of inhibitory connections). Default: 1.0
	 * \returns a unique ID associated with the newly created connection
	 * \see ch4s1_primitive_types
	 */
	short int connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
		const RangeDelay& delay=RangeDelay(1), const RadiusRF& radRF=RadiusRF(-1.0), bool synWtType=SYN_FIXED,
		float mulSynFast=1.0f, float mulSynSlow=1.0f);

	/*!
	 * \brief Shortcut to make connections with custom connectivity profile but omit scaling factors for synaptic
	 * conductances (default is 1.0 for both)
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 * \see ch4s3_user_defined
	 */
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED);

	/*!
	 * \brief make connections with custom connectivity profile
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 * \see ch4s3_user_defined
	 */
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType=SYN_FIXED);

	/*!
	* \brief make a compartmental connection between two compartmentally enabled groups
	* Note: all compartmentally connected groups must be located on the same partition.
	*
	* first group is in the lower layer; second group is in the upper layer
	* \TODO finish docu
	* \STATE CONFIG
	*/
	short int connectCompartments(int grpIdLower, int grpIdUpper);


	/*!
	 * \brief creates a group of Izhikevich spiking neurons
	 * \TODO finish doc
	 * \STATE ::CONFIG_STATE
	 */
	int createGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	/*!
	 * \brief creates a group of Leaky-Integrate-and-Fire (LIF) spiking neurons
	 * \TODO finish doc
	 * \STATE ::CONFIG_STATE
	 */
	int createGroupLIF(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	/*!
	 * \brief Create a group of Izhikevich spiking neurons on a 3D grid (a primitive cubic Bravais lattice with cubic
	 * side length 1)
	 *
	 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid.
	 * Connections can then be specified depending on the relative placement of neurons via CARLsim::connect. This allows
	 * for the creation of networks with complex spatial structure.
	 *
	 * Each neuron in the group gets assigned a (x,y,z) location on a 3D grid centered around the origin, so that calling
	 * Grid3D(Nx,Ny,Nz) creates coordinates that fall in the range [-(Nx-1)/2, (Nx-1)/2], [-(Ny-1)/2, (Ny-1)/2], and
	 * [-(Nz-1)/2, (Nz-1)/2].
	 * The resulting grid is a primitive cubic Bravais lattice with cubic side length 1 (arbitrary units).
	 * The primitive (or simple) cubic crystal system consists of one lattice point (neuron) on each corner of the cube.
	 * Each neuron at a lattice point is then shared equally between eight adjacent cubes.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpName    the group name
	 * \param[in] grid       a Grid3D struct specifying the dimensions of the 3D lattice
	 * \param[in] neurType   either EXCITATORY_NEURON, INHIBITORY_NEURON or DOPAMINERGIC_NEURON
	 * \since v3.0
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	/*!
	 * \brief Create a group of LIF spiking neurons on a 3D grid (a primitive cubic Bravais lattice with cubic
	 * side length 1)
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpName    the group name
	 * \param[in] grid       a Grid3D struct specifying the dimensions of the 3D lattice
	 * \param[in] neurType   either EXCITATORY_NEURON, INHIBITORY_NEURON or DOPAMINERGIC_NEURON
	 * \since v4.0
	 */
	int createGroupLIF(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	/*!
	 * \brief  creates a spike generator group
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	/*!
	 * \brief create a group of spike generators on a 3D grid
	 *
	 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid. This simplifies
	 * the creation of topographic connections in the network.
	 * Each neuron thus gets assigned a (x,y,z) location on a 3D grid (integer coordinates). Neuron numbers will be
	 * assigned to location in order; where the first dimension specifies the width, the second dimension is height,
	 * and the third dimension is depth. Grid3D(2,2,2) would thus assign neurId 0 to location (0,0,0), neurId 1
	 * to (1,0,0), neurId 3 to (0,1,0), neurId 6 to (2,2,1), and so on.
	 * The third dimension can be thought of as a depth (z-coordinate in 3D), a cortical column (each of which consists
	 * of a 2D arrangement of neurons on a plane), or a channel (such as RGB channels, each of which consists of a 2D
	 * arrangements of neurons coding for (x,y) coordinates of an image). For the user's convenience, the struct thus
	 * provides members Grid3D::depth, Grid3D::column, and Grid3D::channels, which differ only semantically.
	 * \STATE ::CONFIG_STATE
	 * \TODO finish doc
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);


	/*!
	 * \brief Sets default values for conduction decay and rise times or disables COBA alltogether
	 *
	 * This function sets the time constants for the decay of AMPA, NMDA, GABA, and GABAb, and the rise times for
	 * NMDA and GABAb. These constants will be applied to all connections in the network. Set isSet to false to run
	 * your simulation in CUBA mode.
	 * Use setDefaultConductanceTimeConstants to set default values for all time constants.
	 * If you call this function without setting your own defaults, then the following defaults will be used:
	 * tdAMPA=5ms, trNMDA=0, tdNMDA=150ms, tdGABAa=6ms, trGABAb=0, tdGABAb=150ms (instantaneous rise time).
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 */
	void setConductances(bool isSet);

	/*!
	 * \brief Sets custom values for conduction decay times (instantaneous rise time) or disables COBA alltogether
	 *
	 * This function sets the time constants for the decay of AMPA, NMDA, GABAa, and GABAb. The decay constants will be
	 * applied to all connections in the network. Set isSet to false to run your simulation in CUBA mode.
	 * The NMDA current is voltage dependent (see Izhikevich et al., 2004).
	 * Use setConductances(true) to use default decay values.
	 * Use the other setConductances to enable non-zero rise times for NMDA and GABAb.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 * \param[in] tdAMPA  time constant for AMPA decay (ms)
	 * \param[in] tdNMDA  time constant for NMDA decay (ms)
	 * \param[in] tdGABAa time constant for GABAa decay (ms)
	 * \param[in] tdGABAb time constant for GABAb decay (ms)
	 */
	void setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb);

	/*!
	 * \brief Sets custom values for conduction rise and decay times or disables COBA alltogether
	 *
	 * This function sets the time constants for the rise and decay time of AMPA, NMDA, GABAa, and GABAb. AMPA and GABAa
	 * will always have instantaneous rise time. The rise times of NMDA and GABAb can be set manually. They need to be
	 * strictly smaller than the decay time. Set isSet to false to run your simulation in CUBA mode.
	 * We do not provide non-zero rise times for AMPA and GABAa, because these rise times are typically on the order of
	 * 1 ms, which is equal to the simulation time step.
	 * The NMDA current is voltage dependent (see Izhikevich et al., 2004).
	 * Use setConductances(true) to use default decay values.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 * \param[in] tdAMPA  time constant for AMPA decay (ms)
	 * \param[in] trNMDA  time constant for NMDA rise (ms), must be smaller than tdNMDA
	 * \param[in] tdNMDA  time constant for NMDA decay (ms)
	 * \param[in] tdGABAa time constant for GABAa decay (ms)
	 * \param[in] trGABAb time constant for GABAb rise (ms), must be smaller than tdGABAb
	 * \param[in] tdGABAb time constant for GABAb decay (ms)
	 */
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);

	/*!
	 * \brief Sets custom values for implementation of homeostatic synaptic scaling
	 *
	 * This function allows the user to set homeostasis for a particular neuron group. All the neurons
	 * in this group scale their weights in an attempt to attain a target base firing rate set with the
	 * setHomeoBaseFiringRate function. Each neuron keeps track of their own average firing rate. The
	 * time over which this average is computed should be on the scale of seconds to minutes to hours as
	 * opposed to ms if one is to claim it is biologically realistic. The homeoScale sets the scaling
	 * factor applied to the weight change of the synapse due to homeostasis. If you want to make
	 * homeostasis stronger than STDP, you increase this value. Scaling of the synaptic weights is
	 * multiplicative. For more information on this implementation please see:
	 * Carlson, et al. (2013). Proc. of IJCNN 2013.
	 *
	 * Reasonable values to start homeostasis with are:
	 * homeoScale   = 0.1-1.0
	 * avgTimeScale = 5-10 seconds
	 *
	 * Default values are:
	 * homeoScale   = 0.1
	 * avgTimeScale = 10 seconds
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId        the group ID of group to which homeostasis is applied
	 * \param[in] isSet        a boolean, setting it to true/false enables/disables homeostasis
	 * \param[in] homeoScale   scaling factor multiplied to weight change due to homeostasis
	 * \param[in] avgTimeScale time in seconds over which average firing rate for neurons in this group is
	 *                         averaged
	 */
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale);

	 /*!
	 * \brief Sets default values for implementation of homeostatic synaptic scaling
	 *
	 * This function allows the user to set homeostasis with default values for a particular neuron
	 * group. For more information, read the setHomeostasis function description above.
	 *
	 * Default values are:
	 * homeoScale   = 0.1
	 * avgTimeScale = 10 seconds
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId        the group ID of group to which homeostasis is applied
	 * \param[in] isSet        a boolean, setting it to true/false enables/disables homeostasis
	 */
	void setHomeostasis(int grpId, bool isSet);

	 /*!
	 * \brief Sets the homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	 *
	 * This function allows the user to set the homeostatic target firing with or without a standard
	 * deviation. All neurons in the group will use homeostatic synaptic scaling to attain the target
	 * firing rate. You can have a standard deviation to the base firing rate or you can leave this
	 * argument blank, which will set the standard deviation to 0.
	 * It should be noted that the baseFiringSD only sets the base firing rate
	 * to a single value within that standard deviation. It does not vary the value of the base firing
	 * rate from this value or within a particular range. For more information on this implementation
	 * please see: Carlson, et al. (2013). Proc. of IJCNN 2013.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId        the group ID of group for which these settings are applied
	 * \param[in] baseFiring   target firing rate of every neuron in this group
	 * \param[in] baseFiringSD standard deviation of target firing rate of every neuron in this group
	 */
	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD=0.0f);

	/*
	* \brief Sets the integration method for the simulation
	*
	* This function specifies the integration method for the simulation. Currently, the chosen integration method
	* will apply to all neurons in the network.
	*
	* The basic simulation time step is 1ms, meaning that spike times cannot be retrieved with sub-millisecond
	* precision. However, the integration time step can be lower than 1ms, which is specified by numStepsPerMs.
	* A numStepsPerMs set to 10 would take 10 integration steps per 1ms simulation time step.
	*
	* By default, the simulation will use Forward-Euler with an integration step of 0.5ms (i.e.,
	* <tt>numStepsPerMs</tt>=2).
	*
	* Currently CARLsim supports the following integration methods:
	* - FORWARD_EULER: The most basic, forward-Euler method. Suggested value for <tt>numStepsPerMs</tt>: >= 2.
	* - RUNGE_KUTTA4:  Fourth-order Runge-Kutta (aka classical Runge-Kutta, aka RK4).
	*                  Suggested value for <tt>numStepsPerMs</tt>: >= 10.
	*
	* \STATE ::CONFIG_STATE
	* \param[in] method the integration method to use
	* \param[in] numStepsPerMs the number of integration steps per 1ms simulation time step
	*
	* \note Note that the higher numStepsPerMs the slower the simulation may be, due to increased computational load.
	* \since v3.1
	*/
	void setIntegrationMethod(integrationMethod_t method, int numStepsPerMs);

	/*!
	 * \brief Sets Izhikevich params a, b, c, and d with as mean +- standard deviation
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd);

	/*!
	 * \brief Sets Izhikevich params a, b, c, and d of a neuron group.
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d);

	/*!
	* \brief Sets Izhikevich params C, k, vr, vt, a, b, vpeak, c, and d of a neuron group
	* C must be positive. There are no limits imposed on other parameters
	* This is a nine parameter Izhikevich simple spiking model
	*
	* \STATE CONFIG
	* \param[in] grpId			the group ID of a group for which these settings are applied
	* \param[in] izh_C			Membrane capacitance parameter
	* \param[in] izh_k			Coefficient present in equation for voltage
	* \param[in] izh_vr		Resting membrane potential parameter
	* \param[in] izh_vt		Instantaneous threshold potential parameter
	* \param[in] izh_a			Recovery time constant
	* \param[in] izh_b			Coefficient present in equation for voltage
	* \param[in] izh_vpeak		The spike cutoff value parameter
	* \param[in] izh_c			The voltage reset value parameter
	* \param[in] izh_d			Parameter describing the total amount of outward minus inward currents activated
	*                          during the spike and affecting the after spike behavior
	* \since v3.1
	*/
	void setNeuronParameters(int grpId, float izh_C, float izh_k, float izh_vr, float izh_vt,
		float izh_a, float izh_b, float izh_vpeak, float izh_c, float izh_d);

	/*!
	* \brief Sets Izhikevich params C, k, vr, vt, a, b, vpeak, c, and d with as mean +- standard deviation
	* C must be positive. There are no limits imposed on other parameters
	* This is a nine parameter Izhikevich simple spiking model
	*
	* \STATE CONFIG
	* \param[in] grpId			the group ID of a group for which these settings are applied
	* \param[in] izh_C			Membrane capacitance parameter
	* \param[in] izh_C_sd		Standard deviation for membrane capacitance parameter
	* \param[in] izh_k			Coefficient present in equation for voltage
	* \param[in] izh_k_sd		Standard deviation for coefficient present in equation for voltage
	* \param[in] izh_vr		Resting membrane potential parameter
	* \param[in] izh_vr_sd		Standard deviation for resting membrane potential parameter
	* \param[in] izh_vt		Instantaneous threshold potential parameter
	* \param[in] izh_vt_sd		Standard deviation for instantaneous threshold potential parameter
	* \param[in] izh_a			Recovery time constant
	* \param[in] izh_a_sd		Standard deviation for recovery time constant
	* \param[in] izh_b			Coefficient present in equation for voltage
	* \param[in] izh_b_sd		Standard deviation for coefficient present in equation for voltage
	* \param[in] izh_vpeak		The spike cutoff value parameter
	* \param[in] izh_vpeak_sd	Standard deviation for the spike cutoff value parameter
	* \param[in] izh_c			The voltage reset value parameter
	* \param[in] izh_c_sd		Standard deviation for the voltage reset value parameter
	* \param[in] izh_d			Parameter describing the total amount of outward minus inward currents activated
	*                          during the spike and affecting the after spike behavior
	* \param[in] izh_d_sd		Standard deviation for the parameter describing the total amount of outward minus
	*                          inward currents activated during the spike and affecting the after spike behavior
	* \since v3.1
	*/
	void setNeuronParameters(int grpId, float izh_C, float izh_C_sd, float izh_k, float izh_k_sd,
		float izh_vr, float izh_vr_sd, float izh_vt, float izh_vt_sd,
		float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_vpeak, float izh_vpeak_sd, float izh_c, float izh_c_sd,
		float izh_d, float izh_d_sd);

	/*!
	 * \brief Sets neuron parameters for a group of LIF spiking neurons
	 *
	 * \param[in] grpId group ID
	 * \param[in] tau_m Membrane time constant in ms (controls decay/leak)
	 * \param[in] tau_ref absolute refractory period in ms
	 * \param[in] vTh Threshold voltage for firing (must be > vReset)
	 * \param[in] vReset Membrane potential resets to this value immediately after spike
	 * \param[in] rMem Range of total membrane resistance of the neuron group, uniformly distributed or fixed for the whole group
	 * \STATE ::CONFIG_STATE
	 */
	void setNeuronParametersLIF(int grpId, int tau_m, int tau_ref=0, float vTh=1.0f, float vReset=0.0f, const RangeRmem& rMem = RangeRmem(1.0f));
    
   /*!
	* \brief Sets coupling constants G_u and G_d for the compartment.
	*
	* \TODO finish docu
	* \STATE ::CONFIG_STATE
	* \param[in] couplingUp		Coupling constant for "up" compartmental connection
	* \param[in] couplingDown	Coupling constant for "down" compartmental connection
	*/

	void setCompartmentParameters(int grpId, float couplingUp, float couplingDown);

	/*!
	 * \brief Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron
	 * group.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId the symbolic name of a group
	 * \param[in] baseDP  the baseline concentration of Dopamine
	 * \param[in] tauDP the decay time constant of Dopamine
	 * \param[in] base5HT  the baseline concentration of Serotonin
	 * \param[in] tau5HT the decay time constant of Serotonin
	 * \param[in] baseACh  the baseline concentration of Acetylcholine
	 * \param[in] tauACh the decay time constant of Acetylcholine
	 * \param[in] baseNE  the baseline concentration of Noradrenaline
	 * \param[in] tauNE the decay time constant of Noradrenaline
	 */
	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							float baseACh, float tauACh, float baseNE, float tauNE);

	/*!
	 * \brief Sets default neuromodulators
	 *
	 * \TODO: this should be implemented via default arguments as members of the class, so that the user can call
	 * setDefaultNeuromodulators()
	 *
	 * \STATE ::CONFIG_STATE
	 */
	void setNeuromodulator(int grpId, float tauDP = 100.0f, float tau5HT = 100.0f,
							float tauACh = 100.0f, float tauNE = 100.0f);

	/*!
	 * \brief Sets default STDP mode and params
	 *
	 * Set STDP parameters. Do not use this function, it is deprecated.
	 *
	 * \sa setESTDP
	 * \deprecated For clearness, do not use default STDP settings.
	 * \since v2.1
	 */
	void setSTDP(int grpId, bool isSet);

	/*!
	 * \brief Sets STDP params for a group, custom
	 *
	 * Set STDP parameters. Do not use this function, it is deprecated.
	 *
	 * \sa setESTDP
	 * \deprecated For clearness, please use CARLsim::setESTDP() with E-STDP curve struct.
	 * \since v2.1
	 */
	void setSTDP(int grpId, bool isSet, STDPType type, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus);

	/*!
	 * \brief Sets default E-STDP mode and parameters
	 *
	 * Set E-STDP parameters using default settings. Do not use this function, it is deprecated.
	 *
	 * \STATE ::CONFIG_STATE
	 * \deprecated For clearness, please do not use default STDP settings.
	 * \since v3.0
	 */
	void setESTDP(int grpId, bool isSet);

	/*!
	 * \brief Sets E-STDP with the exponential curve
	 *
	 * \param[in] grpId the group ID of group for which these settings are applied
	 * \param[in] isSet the flag indicating if E-STDP is enabled
	 * \param[in] type the flag indicating if E-STDP is modulated by dopamine (i.e., DA-STDP)
	 * \param[in] curve the struct defining the exponential curve
	 *
	 * \STATE ::CONFIG_STATE
	 * \sa STDPType
	 * \sa ExpCurve
	 * \since v3.0
	 */
	void setESTDP(int grpId, bool isSet, STDPType type, ExpCurve curve);

	/*!
	 * \brief Sets E-STDP with the timing-based curve
	 *
	 * \param[in] grpId the group ID of group for which these settings are applied
	 * \param[in] isSet the flag indicating if E-STDP is enabled
	 * \param[in] type the flag indicating if E-STDP is modulated by dopamine (i.e., DA-STDP)
	 * \param[in] curve the struct defining the timing-based curve
	 *
	 * \STATE ::CONFIG_STATE
	 * \sa STDPType
	 * \sa TimingBasedCurve
	 * \since v3.0
	 */
	void setESTDP(int grpId, bool isSet, STDPType type, TimingBasedCurve curve);

	/*!
	 * \brief Sets default I-STDP mode and parameters
	 *
	 * Set I-STDP parameters using default settings. Do not use this function, it is deprecated.
	 *
	 * \STATE ::CONFIG_STATE
	 * \deprecated For clearness, please do not use default STDP settings.
	 * \since v3.0
	 */
	void setISTDP(int grpId, bool isSet);

	/*!
	 * \brief Sets I-STDP with the exponential curve
	 *
	 * \param[in] grpId the group ID of group for which these settings are applied
	 * \param[in] isSet the flag indicating if I-STDP is enabled
	 * \param[in] type the flag indicating if I-STDP is modulated by dopamine (i.e., DA-STDP)
	 * \param[in] curve the struct defining the exponential curve
	 *
	 * \STATE ::CONFIG_STATE
	 * \sa STDPType
	 * \sa ExpCurve
	 * \since v3.0
	 */
	void setISTDP(int grpId, bool isSet, STDPType type, ExpCurve curve);

	/*!
	 * \brief Sets I-STDP with the pulse curve
	 *
	 * \param[in] grpId the group ID of group for which these settings are applied
	 * \param[in] isSet the flag indicating if I-STDP is enabled
	 * \param[in] type the flag indicating if I-STDP is modulated by dopamine (i.e., DA-STDP)
	 * \param[in] curve the struct defining the pulse curve
	 *
	 * \STATE ::CONFIG_STATE
	 * \sa STDPType
	 * \sa PulseCurve
	 * \since v3.0
	 */
	void setISTDP(int grpId, bool isSet, STDPType type, PulseCurve curve);

	/*!
	 * \brief Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
	 *
	 * CARLsim implements the short-term plasticity model of (Tsodyks & Markram, 1998; Mongillo, Barak, & Tsodyks, 2008)
	 * \f{eqnarray}
	 * \frac{du}{dt} & = & \frac{-u}{STP\_tau\_u} + STP\_U  (1-u^-)  \delta(t-t_{spk}) \\
	 * \frac{dx}{dt} & = & \frac{1-x}{STP\_tau\_x} - u^+  x^-  \delta(t-t_{spk}) \\
	 * \frac{dI}{dt} & = & \frac{-I}{\tau_S} + A  u^+  x-  \delta(t-t_{spk}) \f}
	 * where u- means value of variable u right before spike update, and x+ means value of variable x right after
	 * the spike update, and A is the synaptic weight.
	 * The STD effect is modeled by a normalized variable (0<=x<=1), denoting the fraction of resources that remain
	 * available after neurotransmitter depletion.
	 * The STF effect is modeled by a utilization parameter u, representing the fraction of available resources ready for
	 * use (release probability). Following a spike, (i) u increases due to spike-induced calcium influx to the
	 * presynaptic terminal, after which (ii) a fraction u of available resources is consumed to produce the post-synaptic
	 * current. Between spikes, u decays back to zero with time constant STP_tau_u (tau_F), and x recovers to value one
	 * with time constant STP_tau_x (tau_D).
	 *
	 * Source: Misha Tsodyks and Si Wu (2013) Short-term synaptic plasticity. Scholarpedia, 8(10):3153., rev #136920
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId       pre-synaptic group ID
	 * \param[in] isSet       a flag whether to enable/disable STP
	 * \param[in] STP_U       increment of u induced by a spike
	 * \param[in] STP_tau_u   decay constant of u (tau_F)
	 * \param[in] STP_tau_x   decay constant of x (tau_D)
	 * \note STP will be applied to all outgoing synapses of all neurons in this group.
	 * \note All outgoing synapses of a certain (pre-synaptic) neuron share the resources of that same neuron.
	 */
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x);

	/*!
	 * \brief Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically) using default values
	 *
	 * This function enables/disables STP on a specific pre-synaptic group and assign default values to all STP
	 * parameters.
	 * The default parameters for an excitatory neuron are U=0.45, tau_u=50.0, tau_f=750.0 (depressive).
	 * The default parameters for an inhibitory neuron are U=0.15, tau_u=750.0, tau_f=50.0 (facilitative).
	 *
	 * Source: Misha Tsodyks and Si Wu (2013) Short-term synaptic plasticity. Scholarpedia, 8(10):3153., rev #136920
	 *
	 * These default values can be overridden using setDefaultSTPparams.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId   pre-synaptic group ID
	 * \param[in] isSet   a flag whether to enable/disable STP
	 * \note STP will be applied to all outgoing synapses of all neurons in this group.
	 * \note All outgoing synapses of a certain (pre-synaptic) neuron share the resources of that same neuron.
	 * \see setDefaultSTPparams
	 * \see setSTP(int, bool, float, float, float)
	 * \since v3.0
	 */
	void setSTP(int grpId, bool isSet);

	/*!
	 * \brief Sets the weight and weight change update parameters
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] wtANDwtChangeUpdateInterval the interval between two wt (weight) and wtChange (weight change) update.
	 * \param[in] enableWtChangeDecay enable weight change decay
	 * \param[in] wtChangeDecay the decay ratio of weight change (wtChange)
	 */
	void setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay=0.9f);


	// +++++ PUBLIC METHODS: RUNNING A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for time=(nSec*seconds + nMsec*milliseconds)
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE. First call to runNetwork will make CARLsim state switch from ::SETUP_STATE to ::RUN_STATE.
	 * \param[in] nSec 			  number of seconds to run the network
	 * \param[in] nMsec 		  number of milliseconds to run the network
	 * \param[in] printRunSummary enable the printing of a summary at the end of this run
	 */
	int runNetwork(int nSec, int nMsec=0, bool printRunSummary=true);

	/*!
	 * \brief build the network
	 *
	 * \STATE ::CONFIG_STATE. Will make CARLsim state switch from ::CONFIG_STATE to ::SETUP_STATE.
	 */
	void setupNetwork();

	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const FILE* getLogFpInf();	//!< returns file pointer to info log
	const FILE* getLogFpErr();	//!< returns file pointer to error log
	const FILE* getLogFpDeb();	//!< returns file pointer to debug log
	const FILE* getLogFpLog();	//!< returns file pointer to log file

	/*!
	 * \brief Saves important simulation and network infos to file.
	 *
	 * The network state consists of all
	 * the synaptic connections, weights, delays, and whether the connections are plastic or fixed. As an
	 * option, the user can choose whether or not to save the synaptic weight information (which could be
	 * a large amount of data) with the saveSynapseInfo argument. The value of this argument is true by
	 * default which causes the synaptic weight values to be output by default along with the rest of the
	 * network information.
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] fileName          string of filename of saved simulation data.
	 * \param[in] saveSynapseInfo   boolean value that determines if the weight values are written to
	 *                              the data file or not. The weight values are written if the boolean value is true.
	 * \see CARLsim::loadSimulation
	 * \since v2.0
	 */
	void saveSimulation(const std::string& fileName, bool saveSynapseInfo=true);

	/*!
	 * \brief Sets the name of the log file
	 *
	 * This function sets a new path/name for the CARLsim log file. By default, the log file name is given depending on
	 * the LoggerMode specified in #CARLsim.
	 * However, it can be manually overridden using this function.
	 * In order to disable the log file, pass string "NULL".
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param fileName the name of the log file
	 * \note This function cannot be called in LoggerMode CUSTOM. In this case, use setLogsFpCustom instead
	 * \attention Make sure the directory exists!
	 * \see setLogsFpCustom
	 */
	void setLogFile(const std::string& fileName);

	/*!
	 * \brief Sets the file pointers for all log files in CUSTOM mode
	 *
	 * In LoggerMode CUSTOM, custom file pointers can be used for the info, error, and debug log streams.
	 * In this case, CARLsim does not take ownership of the file pointers; that is, the user should fclose them.
	 * Setting a file pointer to NULL will not change the currently assigned file pointer (default value points to the
	 * bit bucket).
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param[in] fpInf file pointer for status info
	 * \param[in] fpErr file pointer for errors/warnings
	 * \param[in] fpDeb file pointer for debug info
	 * \param[in] fpLog file pointer for debug log file that contains all the above info
	 * \note This function can be called only in LoggerMode CUSTOM.
	 * \note Use NULL in order not to change current file pointers.
	 * \attention Make sure to fclose the file pointers. But, do not fclose stdout or stderr, or they will remain
	 * closed for the remainder of the process.
	 */
	void setLogsFpCustom(FILE* fpInf=NULL, FILE* fpErr=NULL, FILE* fpDeb=NULL, FILE* fpLog=NULL);



	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Adds a constant bias to the weight of every synapse in the connection
	 *
	 * This method adds a constant bias to the weight of every synapse in the connection specified by connId. The bias
	 * can be positive or negative.
	 * If a bias is specified that makes any weight+bias lie outside the range [minWt,maxWt] of this connection, the
	 * range will be updated accordingly if the flag updateWeightRange is set to true.
	 * If the flag is set to false, then the specified weight value will be corrected to lie on the boundary (either
	 * minWt or maxWt).
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId            the connection ID to manipulate
	 * \param[in] bias              the bias value to add to every synapse
	 * \param[in] updateWeightRange a flag specifying what to do when the specified weight+bias lies outside the range
	 *                              [minWt,maxWt]. Set to true to update the range accordingly. Set to false to adjust
	 *                              the weight to be either minWt or maxWt. Default: false.
	 *
	 * \note A weight cannot drop below zero, no matter what.
	 * \see setWeight
	 * \see scaleWeights
	 * \since v3.0
	 */
	void biasWeights(short int connId, float bias, bool updateWeightRange=false);

	/*!
	 * \brief Loads a simulation (and network state) from file. The file pointer fid must point to a
	 * valid CARLsim network save file (created with CARLsim::saveSimulation).
	 *
	 * Past CARLsim networks can be loaded from file by setting up the same number of groups, connections, and neurons
	 * as was used to store the network via CARLsim::saveSimulation, and then calling CARLsim::loadSimulation to
	 * overwrite all corresponding synaptic weight and delay values from file.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] fid       file pointer to a save file created with CARLsim::saveSimulation
	 *
	 *\note In order for CARLsim::loadSimulation to work, the configured network must have the identical number of
	 * groups, connections, and neurons as the one stored with CARLsim::saveSimulation.
	 * \note In addition, CARLsim::saveSimulation must have been called with flag <tt>saveSynapseInfo</tt> set to
	 * <tt>true</tt>.
	 * \attention Wait with calling fclose on the file pointer until ::SETUP_STATE!
	 * \see CARLsim::saveSimulation
	 * \since v2.0
	 */
	void loadSimulation(FILE* fid);

	/*!
	 * \brief reset Spike Counter to zero
	 *
	 * Manually resets the spike buffers of a Spike Counter to zero (for a specific group).
	 * Buffers get reset to zero automatically after <tt>recordDur</tt> (see CARLsim::setSpikeCounter).
	 * However, the buffer can be manually reset at any point in time (during ::SETUP_STATE and ::RUN_STATE).
	 *
	 * At any point in time (during ::SETUP_STATE or ::RUN_STATE), all SpikeCounters can be reset via:
	 * \code
	 * sim.resetSpikeCounters(-1); // reset for all groups, -1==ALL
	 * \endcode
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId the group for which to reset the spike counts. Set to ALL if you want to reset all SpikeCounters.
	 * \see CARLsim::setSpikeCounter
	 * \see CARLsim::getSpikeCounter
	 */
	//void resetSpikeCounter(int grpId);

	/*!
	 * \brief Multiplies the weight of every synapse in the connection with a scaling factor
	 *
	 * This method scales the weight of every synapse in the connection specified by connId with a scaling factor.
	 * The scaling factor cannot be negative.
	 * If a scaling factor is specified that makes any weight*scale lie outside the range [minWt,maxWt] of this
	 * connection, the range will be updated accordingly if the flag updateWeightRange is set to true.
	 * If the flag is set to false, then the specified weight value will be corrected to lie on the boundary (either
	 * minWt or maxWt).
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId            the connection ID to manipulate
	 * \param[in] scale             the scaling factor to apply to every synapse (cannot be negative)
	 * \param[in] updateWeightRange a flag specifying what to do when the specified weight*scale lies outside the range
	 *                              [minWt,maxWt]. Set to true to update the range accordingly. Set to false to adjust
	 *                              the weight to be either minWt or maxWt. Default: false.
	 *
	 * \note A weight cannot drop below zero, no matter what.
	 * \see setWeight
	 * \see biasWeights
	 * \since v3.0
	 */
	void scaleWeights(short int connId, float scale, bool updateWeightRange=false);

	/*!
	 * \brief Sets a connection monitor for a group, custom ConnectionMonitor class
	 *
	 * To retrieve connection status, a connection-monitoring callback mechanism is used. This mechanism allows the user
	 * to monitor connection status between groups. Connection monitors are registered for two groups (i.e., pre- and
	 * post- synaptic groups) and are called automatically by the simulator every second.
	 *
	 * CARLsim supports two different recording mechanisms: Recording to a weight file (binary) and recording to a
	 * ConnectionMonitor object. The former is useful for off-line analysis of synaptic weights (e.g., using
	 * \ref ch9_matlab_oat).
	 * The latter is useful to calculate different weight metrics and statistics on-line, such as the percentage of
	 * weight values that fall in a certain weight range, or the number of weights that have been changed since the
	 * last snapshot.
	 *
	 * A file name can be specified via variable fileName (make sure the specified directory exists). The easiest way
	 * is to set fileName to string "DEFAULT", in which case a default file name will be created in the results
	 * directory: "results/conn_{preGrpName}_{postGrpName}.dat", where preGrpName is the name assigned to the
	 * pre-synaptic group at initialization, and postGrpName is the name assigned to the post-synaptic group at
	 * initialization.
	 * If no binary file shall be created, set fileName equal to the string "NULL".
	 *
	 * The function returns a pointer to a ConnectionMonitor object, which can be used to calculate weight changes
	 * and other connection stats.
	 * See \ref ch7s2_connection_monitor of the User Guide for more information on how to use ConnectionMonitor.
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE
	 * \param[in] grpIdPre 		the pre-synaptic group ID
	 * \param[in] grpIdPost 	the post-synaptic group ID
	 * \param[in] fname         file name of the binary to be created
	 * \see ch7s2_connection_monitor
	 * \see ch9s1_matlab_oat
	 */
	ConnectionMonitor* setConnectionMonitor(int grpIdPre, int grpIdPost, const std::string& fname);

	/*!
	 * \brief Sets the amount of current (mA) to inject into a group
	 *
	 * This method injects current, specified on a per-neuron basis, into the soma of each neuron in the group, at
	 * each timestep of the simulation. current is a float vector of current amounts (mA), one element per neuron in
	 * the group.
	 *
	 * To input different currents into a neuron over time, the idea is to run short periods of runNetwork and
	 * subsequently calling setExternalCurrent again with updated current values.
	 *
	 * For example: Inject 5mA for 50 ms, then 0mA for 10 sec
	 * \code
	 * // 5mA for 50 ms, 10 neurons in the group
	 * std::vector<float> current(10, 5.0f);
	 * snn.setExternalCurrent(g0, current);
	 * snn.runNetwork(0,50);
	 *
	 * // 0mA for 10 sec, use convenience function for reset
	 * snn.setExternalCurrent(g0, 0.0f);
	 * snn.runNetwork(10,0);
	 * \endcode
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] grpId    the group ID
	 * \param[in] current  a float vector of current amounts (mA), one value per neuron in the group
	 *
	 * \note This method cannot be applied to SpikeGenerator groups.
	 * \note If all neurons in the group should receive the same amount of current, you can use the convenience
	 * function setExternalCurrent(int grpId, float current).
	 *
	 * \attention Make sure to reset current after use (i.e., for the next call to runNetwork), otherwise
	 * the current will keep getting applied to the group.
	 * \see setExternalCurrent(int grpId, float current)
	 * \see setSpikeRate
	 * \see setSpikeGenerator
	 * \see \ref ch6s2_generating_current
	 */
	void setExternalCurrent(int grpId, const std::vector<float>& current);

	/*!
	 * \brief Sets the amount of current (mA) to inject to each neuron in a group
	 *
	 * This methods injects a specific amount of current into the soma of every neuron in the group. current is a float
	 * value in mA. The same current is applied to every neuron in the group.
	 *
	 * For example: inject 4.5 mA into every neuron in the group, for 500 ms
	 * \code
	 * // 4.5 mA for 500 ms
	 * snn.setExternalCurrent(g0, 4.5f);
	 * snn.runNetwork(0,500);
	 *
	 * // don't forget to reset current afterwards if necessary
	 * snn.setExternalCurrent(g0, 0.0f);
	 * snn.runNetwork(10,0); // zero external current
	 * \endcode
	 *
	 * \note This method cannot be applied to SpikeGenerator groups.
	 * \note If each neuron in the group should receive a different amount of current, you can use the method
	 * setExternalCurrent(int grpId, const std::vector<float>& current) instead.
	 *
	 * \attention Make sure to reset current after use (i.e., for the next call to runNetwork), otherwise
	 * the current will keep getting applied to the group.
	 * \see setExternalCurrent(int grpId, const std::vector<float>& current)
	 * \see setSpikeRate
	 * \see setSpikeGenerator
	 * \see \ref ch6s2_generating_current
	 */
	void setExternalCurrent(int grpId, float current);

	/*!
	 * \brief Sets a group monitor for a group, custom GroupMonitor class
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE
	 */
	GroupMonitor* setGroupMonitor(int grpId, const std::string& fname);

	/*!
	 * \brief A SpikeCounter keeps track of the number of spikes per neuron in a group.
	 *
	 * A SpikeCounter keeps track of all spikes per neuron for a certain time period (recordDur).
	 * After that, the spike buffers get reset to zero number of spikes.
	 *
	 * This function works for Izhikevich neurons as well as Spike Generators.
	 *
	 * The recording time can be set to any x number of ms, so that after x ms the spike counts will be reset
	 * to zero. If x==-1, then the spike counts will never be reset (should only overflow after 97 days of sim).
	 * Also, spike counts can be manually reset at any time by calling CARLsim::resetSpikeCounter(grpId);
	 *
	 * At any point in time (during ::RUN_STATE), CARLsim::getSpikeCounter can be called to get an integer array
	 * that contains the number of spikes for each neuron in the group.
	 *
	 * There can be only SpikeCounter per group. However, a group can have both a SpikeMonitor and a SpikeCounter.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId the group for which you want to enable a SpikeCounter
	 * \param[in] recordDur number of ms for which to record spike numbers. Spike numbers will be reset to zero after
	 * this. Set frameDur to -1 to never reset spike counts. Default: -1.
	 * \see CARLsim::getSpikeCounter
	 * \see CARLsim::resetSpikeCounter
	 */
	//void setSpikeCounter(int grpId, int recordDur=-1);

	/*!
	 * \brief Associates a SpikeGenerator object with a group
	 *
	 * A custom SpikeGenerator object can be used to allow for more fine-grained control overs spike generation by
	 * specifying individual spike times for each neuron in a group.
	 *
	 * In order to specify spike times, a new class must be defined first that derives from the SpikeGenerator class
	 * and implements the virtual method SpikeGenerator::nextSpikeTime.
	 * Then, in order for a custom SpikeGenerator to be associated with a SpikeGenerator group,
	 * CARLsim::setSpikeGenerator must be called on the group in ::CONFIG_STATE:.
	 *
	 * A number of interesting Spike Generators is provided in the <tt>tools/spike_generators</tt> directory, such
	 * as PeriodicSpikeGenerator, SpikeGeneratorFromVector, and SpikeGeneratorFromFile.
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] grpId           the group with which to associate a SpikeGenerator object
	 * \param[in] spikeGenFunc pointer to a custom SpikeGenerator object
	 * \see \ref ch6s1_generating_spikes
	 */
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGenFunc);

	/*!
	 * \brief Sets a Spike Monitor for a groups, prints spikes to binary file
	 *
	 * To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
	 * basic statistics, store spike trains, or perform more complicated output monitoring. Spike monitors are
	 * registered for a group and are called automatically by the simulator every second. Similar to an address event
	 * representation (AER), the spike monitor indicates which neurons spiked by using the neuron ID within a group
	 * (0-indexed) and the time of the spike. Only one spike monitor is allowed per group.
	 *
	 * CARLsim supports two different recording mechanisms: Recording to a spike file (binary) and recording to a
	 * SpikeMonitor object. The former is useful for off-line analysis of activity (e.g., using \ref ch9_matlab_oat).
	 * The latter is useful to calculate different spike metrics and statistics on-line, such as mean firing rate and
	 * standard deviation, or the number of neurons whose firing rate lies in a certain interval.
	 *
	 * A file name can be specified via variable fileName (make sure the specified directory exists). The easiest way
	 * is to set fileName to string "DEFAULT", in which case a default file name will be created in the results
	 * directory: "results/spk_{group name}.dat", where group name is the name assigned to the group at initialization
	 * (can be retrieved via getGroupName).
	 * If no binary file shall be created, set fileName equal to the string "NULL".
	 *
	 * The function returns a pointer to a SpikeMonitor object, which can be used to calculate spike statistics (such
	 * group firing rate, number of silent neurons, etc.) or retrieve all spikes from a particular time window.
	 * See \ref ch7s1_spike_monitor of the User Guide for more information on how to use SpikeMonitor.
	 *
	 * If you call setSpikeMonitor twice on the same group, the same SpikeMonitor pointer will be returned, and the
	 * name of the spike file will be updated. This is the same as calling SpikeMonitor::setLogFile directly, and
	 * allows you to redirect the spike file stream mid-simulation (see \ref ch7s1s3_redirecting_file_streams).
	 *
	 * \STATE ::SETUP_STATE
	 * \param[in] grpId 		the group ID
	 * \param[in] fileName 		name of the binary file to be created. Leave empty for default name
	 *                      	"results/spk_{grpName}.dat". Set to string "NULL" to suppress file creation. Default: ""
	 * \returns   SpikeMonitor*	pointer to a SpikeMonitor object, which can be used to calculate spike statistics
	 *                          (such as group firing rate, number of silent neurons, etc.) or retrieve all spikes in
	 * 							AER format
	 *
	 * \note Only one SpikeMonitor is allowed per group.
	 * \attention Using SpikeMonitor::startRecording and SpikeMonitor::stopRecording might significantly slow down the
	 * simulation. It is unwise to use this mechanism to record a large number of spikes over a long period of time.
	 * \see ch7s1_spike_monitor
	 * \see ch9s1_matlab_oat
	 */
	SpikeMonitor* setSpikeMonitor(int grpId, const std::string& fileName);

	/*!
	* \brief Sets a Neuron Monitor for a groups, print voltage, recovery, and total current values to binary file
	*
	* To retrieve outputs, a neuron-monitoring callback mechanism is used. This mechanism allows the user to calculate
	* basic statistics, store voltage/recovery/current values, or perform more complicated output monitoring. Neuron monitors are
	* registered for a group and are called automatically by the simulator every second. Similar to an address event
	* representation (AER), the neuron monitor indicates neuron's state by using the neuron ID within a group
	* (0-indexed). Only one neuron monitor is allowed per group.
	*
	* CARLsim supports two different recording mechanisms: Recording to a neuron state (voltage, recovery, and current)
	* file (binary) and recording to a
	* NeuronMonitor object. The former is useful for off-line analysis of activity (e.g., using \ref ch9_matlab_oat).
	* The latter is useful to calculate different neuron state metrics and statistics on-line.
	*
	* A file name can be specified via variable fileName (make sure the specified directory exists). The easiest way
	* is to set fileName to string "DEFAULT", in which case a default file name will be created in the results
	* directory: "results/nrnstate_{group name}.dat", where group name is the name assigned to the group at initialization
	* (can be retrieved via getGroupName).
	* If no binary file shall be created, set fileName equal to the string "NULL".
	*
	* The function returns a pointer to a NeuronMonitor object, which can be used to calculate neuron statistics
	* or retrieve all neuron state values from a particular time window.
	* See \ref ??? of the User Guide for more information on how to use NeuronMonitor.
	*
	* If you call setNeuronMonitor twice on the same group, the same NeuronMonitor pointer will be returned, and the
	* name of the neuron state file will be updated. This is the same as calling NeuronMonitor::setLogFile directly, and
	* allows you to redirect the spike file stream mid-simulation (see \ref ch7s1s3_redirecting_file_streams).
	*
	* \STATE ::SETUP_STATE
	* \param[in] grpId 		the group ID
	* \param[in] fileName 		name of the binary file to be created. Leave empty for default name
	*                      	"results/nrnstate_{grpName}.dat". Set to string "NULL" to suppress file creation. Default: ""
	* \returns   NeuronMonitor*	pointer to a NeuronMonitor object, which can be used to calculate neuron state statistics
	*                           or retrieve all spikes in AER format
	*
	* \note Only one NeuronMonitor is allowed per group. NeuronMonitor cannot be placed on groups with >100 (LARGE_NEURON_MON_GRP_SIZE) neurons
	* \attention Using NeuronMonitor::startRecording and NeuronMonitor::stopRecording might significantly slow down the
	* simulation. It is unwise to use this mechanism to record a large number of neuron state values (voltage, recovery, 
	* and total current values) over a long period of time.
	* \see ???
	* \see ch9s1_matlab_oat
	*/
	NeuronMonitor* setNeuronMonitor(int grpId, const std::string& fileName);

	/*!
	 * \brief Sets a spike rate
	 * \TODO finish docu
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] grpId      group ID
	 * \param[in] spikeRate  pointer to PoissonRate object
	 * \param[in] refPeriod  refactory period (ms). Default: 1ms.
	 *
	 * \note This method can only be applied to SpikeGenerator groups.
	 * \note setSpikeRate will *not* take over ownership of PoissonRate. In other words, if you allocate the
	 * PoissonRate object on the heap, you are responsible for correctly deallocating it.
	 * \attention Make sure to reset spike rate after use (i.e., for the next call to runNetwork), otherwise
	 * the rate will keep getting applied to the group.
	 * \see setExternalCurrent
	 * \see setSpikeGenerator
	 */
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1);

	/*!
	 * \brief Sets the weight value of a specific synapse
	 *
	 * This method sets the weight value of the synapse that belongs to connection connId and connects pre-synaptic
	 * neuron neurIdPre to post-synaptic neuron neurIdPost. Neuron IDs should be zero-indexed, so that the first
	 * neuron in the group has ID 0.
	 *
	 * If a weight value is specified that lies outside the range [minWt,maxWt] of this connection, the range will be
	 * updated accordingly if the flag updateWeightRange is set to true. If the flag is set to false, then the
	 * specified weight value will be corrected to lie on the boundary (either minWt or maxWt).
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId            the connection ID to manipulate
	 * \param[in] neurIdPre         pre-synaptic neuron ID (zero-indexed)
	 * \param[in] neurIdPost        post-synaptic neuron ID (zero-indexed)
	 * \param[in] weight            the weight value to set for this synapse
	 * \param[in] updateWeightRange a flag specifying what to do when the specified weight lies outside the range
	 *                              [minWt,maxWt]. Set to true to update the range accordingly. Set to false to adjust
	 *                              the weight to be either minWt or maxWt. Default: false.
	 *
	 * \note Neuron IDs should be zero-indexed (first neuron in the group should have ID 0).
	 * \note A weight cannot drop below zero, no matter what.
	 * \attention Make sure this function is called on a synapse that actually exists!
	 * \see biasWeights
	 * \see scaleWeights
	 * \since v3.0
	 */
	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange=false);

	/*!
	 * \brief Enters a testing phase in which all weight changes are disabled
	 *
	 * This function can be used to temporarily disable all weight updates (such as from STDP or homeostasis)
	 * in the network. This can be useful in an experimental setting that consists of 1) a training phase, where
	 * STDP or other plasticity mechanisms learn some input stimulus set, and 2) a testing phase, where the
	 * learned synaptic weights are evaluated (without making any further weight modifications) by presenting
	 * some test stimuli.
	 *
	 * An optional parameter specifies whether the accumulated weight updates so far should be applied to the weights
	 * before entering the testing phase. Recall that although weight changes are accumulated every millisecond,
	 * they are only applied to the weights every so often (see CARLsim::setWeightAndWeightChangeUpdate).
	 * If updateWeights is set to true, then the accumulated weight changes will be applied to the weights, even if
	 * CARLsim::startTesting is called off the weight update grid.
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] updateWeights   whether to apply the accumulated weight changes before entering the testing phase
	 * \note Calling this function on a simulation with no plastic synapses will have no effect.
	 * \see CARLsim::stopTesting
	 * \since v3.1
	 */
	void startTesting(bool updateWeights=true);

	/*!
	 * \brief Exits a testing phase, making weight changes possible again
	 *
	 * This function can be used to exit a testing phase (in which all weight changes were disabled), after which
	 * weight modifications are possible again. This can be useful in an experimental setting with multiple training
	 * phases followed by testing phases.
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \note Calling this function on a simulation with no plastic synapses will have no effect.
	 * \see CARLsim::startTesting
	 * \since v3.1
	 */
	void stopTesting();

	/*!
	 * \brief Writes population weights from gIDpre to gIDpost to file fname in binary.
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	//void writePopWeights(std::string fname, int gIDpre, int gIDpost);



	// +++++ PUBLIC METHODS: GETTERS / SETTERS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Returns the current CARLsim state
	 *
	 * A CARLsim simulation goes through the following states:
	 * - ::CONFIG_STATE 	configuration state, where the neural network is configured
	 * - ::SETUP_STATE 		setup state, where the neural network is prepared for execution
	 * - ::RUN_STATE 	execution state, where the simulation is executed
	 *
	 * Certain methods can only be called in certain states. Check their documentation to see which method can be called
	 * in which state.
	 *
	 * Certain methods perform state transitions. setupNetwork will change the state from ::CONFIG_STATE to ::SETUP_STATE. The
	 * first call to runNetwork will change the state from ::SETUP_STATE to ::RUN_STATE.
	 * \returns current CARLsim state
	 */
	CARLsimState getCARLsimState();

	/*!
	 * \brief gets AMPA vector of a group
	 *
	 * \TODO finish docu
	 * \STATE ::RUN_STATE
	 * \deprecated This function is deprecated. It will be replaced by NeuronMonitor.
	 */
	std::vector<float> getConductanceAMPA(int grpId);

	/*!
	 * \brief gets NMDA vector of a group
	 *
	 * \TODO finish docu
	 * \STATE ::RUN_STATE
	 * \deprecated This function is deprecated. It will be replaced by NeuronMonitor.
	 */
	std::vector<float> getConductanceNMDA(int grpId);

	/*!
	 * \brief gets GABAa vector of a group
	 *
	 * \TODO finish docu
	 * \STATE ::RUN_STATE
	 * \deprecated This function is deprecated. It will be replaced by NeuronMonitor.
	 */
	std::vector<float> getConductanceGABAa(int grpId);

	/*!
	 * \brief gets GABAb vector of a group
	 *
	 * \TODO finish docu
	 * \STATE ::RUN_STATE
	 * \deprecated This function is deprecated. It will be replaced by NeuronMonitor.
	 */
	std::vector<float> getConductanceGABAb(int grpId);

	/*!
	 * \brief returns the RangeDelay struct for a specific connection ID
	 *
	 * This function returns the RangeDelay struct for a specific connection ID. The RangeDelay struct contains
	 * fields for the minimum and maximum synaptic delay in the connection.
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId connection ID
	 * \returns RangeDelay struct
	 * \since v3.0
	 */
	RangeDelay getDelayRange(short int connId);

	/*!
	 * \brief gets delays
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost);

	// FIXME: This function is called in SNN::connect() at CONFIG_STATE, which violate the restriction
	/*!
	 * \brief returns the 3D grid struct of a group
	 *
	 * This function returns the Grid3D struct of a particular neuron group.
	 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid. This simplifies
	 * the creation of topographic connections in the network. The dimensions of the grid can thus be retrieved by
	 * calling Grid3D.width, Grid3D.height, and Grid3D.depth. The total number of neurons is given by Grid3D.N.
	 * See createGroup and Grid3D for more information.
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] grpId the group ID for which to get the Grid3D struct
	 * \returns the 3D grid struct of a group
	 */
	Grid3D getGroupGrid3D(int grpId);

	/*!
	 * \brief finds the ID of the group with name grpName
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getGroupId(std::string grpName);

	/*!
	 * \brief gets group name
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	std::string getGroupName(int grpId);

	/*!
	 * \brief returns the 3D location a neuron codes for
	 *
	 * This function returns the (x,y,z) location that a neuron with ID neurID (global) codes for.
	 * Note that neurID is global; that is, the first neuron in the group does not necessarily have ID 0.
	 *
	 * The location is determined by the actual neuron ID (the first neuron in the group coding for the origin (0,0,0),
	 * and by the dimensions of the 3D grid the group is allocated on (integer coordinates). Neuron numbers are
	 * assigned to location in order; where the first dimension specifies the width, the second dimension is height,
	 * and the third dimension is depth.
	 *
	 * For more information see createGroup and the Grid3D struct.
 	 * See also getNeuronLocation3D(int grpId, int relNeurId).
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::EXE_STATE
	 * \param[in] neurId the neuron ID for which the 3D location should be returned
	 * \returns the 3D location a neuron codes for as a Point3D struct
	 */
	Point3D getNeuronLocation3D(int neurId);

	/*!
	* \brief Returns the maximum number of allowed compartmental connections per group.
	*
	* A compartmentally enabled neuron group cannot have more than this number of compartmental connections.
	* This value is controlled by MAX_NUM_COMP_CONN in carlsim_definitions.h.
	*/
	int getMaxNumCompConnections();

	/*!
	 * \brief returns the 3D location a neuron codes for
	 *
	 * This function returns the (x,y,z) location that a neuron with ID  relNeurId (relative to the group) codes for.
	 * Note that neurID is relative to the ID of the first neuron in the group; that is, the first neuron in the group
	 * has relNeurId 0, the second one has relNeurId 1, etc.
	 * In other words: relNeurId = neurId - sim.getGroupStartNeuronId();
	 *
	 * The location is determined by the actual neuron ID (the first neuron in the group coding for the origin (0,0,0),
	 * and by the dimensions of the 3D grid the group is allocated on (integer coordinates). Neuron numbers are
	 * assigned to location in order; where the first dimension specifies the width, the second dimension is height,
	 * and the third dimension is depth.
	 *
	 * For more information see createGroup and the Grid3D struct.
	 * See also getNeuronLocation3D(int neurId).
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::EXE_STATE
	 * \param[in] grpId       the group ID
	 * \param[in] relNeurId   the neuron ID (relative to the group) for which the 3D location should be returned
	 * \returns the 3D location a neuron codes for as a Point3D struct
	 */
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	/*!
	 * \brief Returns the number of connections (pairs of pre-post groups) in the network
	 *
	 * This function returns the number of connections (pairs of pre-post groups) in the network. Each pre-post
	 * pair of neuronal groups has its own connection ID, which is returned by a call to connect.
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \returns the number of connections (pairs of pre-post groups) in the network
	 */
	int getNumConnections();

	/*!
	 * \brief returns the number of connections associated with a connection ID
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumSynapticConnections(short int connectionId);

	/*!
	 * \brief returns the number of groups in the network
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumGroups();

	/*!
	 * \brief returns the total number of allocated neurons in the network
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeurons();

	/*!
	 * \brief returns the total number of regular (Izhikevich) neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsReg();

	/*!
	 * \brief returns the total number of regular (Izhikevich) excitatory neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsRegExc();

	/*!
	 * \brief returns the total number of regular (Izhikevich) inhibitory neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsRegInh();

	/*!
	 * \brief returns the total number of spike generator neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsGen();

	/*!
	 * \brief returns the total number of excitatory spike generator neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsGenExc();

	/*!
	 * \brief returns the total number of inhibitory spike generator neurons
	 *
	 * \note This number might change throughout CARLsim state ::CONFIG_STATE, up to calling setupNetwork).
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumNeuronsGenInh();

	/*!
	 * \brief returns the total number of allocated synaptic connections in the network
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getNumSynapses();

	/*!
	 * \brief returns the first neuron id of a groupd specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getGroupStartNeuronId(int grpId);

	/*!
	 * \brief returns the last neuron id of a groupd specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getGroupEndNeuronId(int grpId);

	/*!
	 * \brief returns the number of neurons of a group specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 */
	int getGroupNumNeurons(int grpId);

	/*!
	 * \brief returns the stdp information of a group specified by grpId
	 *
	 * This function returns the current STDP setting of a group.
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \sa GroupSTDPInfo
	 */
	GroupSTDPInfo getGroupSTDPInfo(int grpId);

	/*!
	 * \brief returns the neuromodulator information of a group specified by grpId
	 *
	 * This function returns the current setting for neuromodulators.
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \sa GroupNeuromodulatorInfo
	 */
	GroupNeuromodulatorInfo getGroupNeuromodulatorInfo(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getSimTime();

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getSimTimeSec();

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getSimTimeMsec();

	/*!
	 * \brief Returns the number of spikes per neuron for a certain group
	 *
	 * A SpikeCounter keeps track of all spikes per neuron binned into a certain time period (recordDur).
	 * This function allows to query the spike array at any point in time.
	 * It will return a pointer to an int array if the group has a valid SpikeCounter, or NULL otherwise. The number
	 * of elements in the int array is the number of neurons in the group.
	 *
	 * Before this function can be used, a SpikeCounter must be set up for the group via CARLsim::setSpikeCounter.
	 *
	 * Usage example:
	 * \code
	 * // During CONFIG state, create a neuron group with 10 neurons.
	 * int g0 = sim.createGroup("group0", 10, EXCITATORY_NEURON);
	 *
	 * // Set up a SpikeCounter on group g0 that counts the number
	 * // of spikes in bins of 50 ms
	 * sim.setSpikeCounter(g0, 50);
	 *
	 * // move to setup state
	 * sim.setupNetwork();
	 *
	 * // run the network for a bit, say 32 ms
	 * sim.runNetwork(0,32);
	 *
	 * // get the number of spikes in these 32 ms
	 * int* spkArr = sim.getSpikeCounter(g0);
	 *
	 * // print number of spikes for each neuron in the group
	 * for (int i=0; i<10; i++)
	 *    printf("Neuron %d has %d spikes\n",i,spkArr[i]);
	 * \endcode
	 *
	 * \STATE ::RUN_STATE
	 * \param[in] grpId	   the group for which you want the spikes (cannot be ALL)
	 * \returns pointer to array of ints if SpikeCounter exists, else NULL. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (int) since the last reset.
	 * \see CARLsim::setSpikeCounter
	 * \see CARLsim::resetSpikeCounter
	 */
	//int* getSpikeCounter(int grpId);

	/*!
	 * \brief returns pointer to previously allocated SpikeMonitor object, NULL else
	 *
	 * This function returns a pointer to a SpikeMonitor object that has previously been created using the method
	 * setSpikeMonitor. If the group does not have a SpikeMonitor, NULL is returned.
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param[in] grpId the group ID
	 * \returns pointer to SpikeMonitor object if exists, NULL else
	 * \since v3.0
	 */
	SpikeMonitor* getSpikeMonitor(int grpId);

	/*!
	 * \brief returns the RangeWeight struct for a specific connection ID
	 *
	 * This function returns the RangeWeight struct for a specific connection ID. The RangeWeight struct contains
	 * fields for the minimum, initial, and maximum weight in the connection.
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId connection ID
	 * \returns RangeWeight struct
	 * \since v3.0
	 */
	RangeWeight getWeightRange(short int connId);

	/*!
	 * \brief Returns whether a connection is fixed or plastic
	 *
	 * This function returns whether the synapses in a certain connection ID are fixed (false) or plastic (true).
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param[in] connId connection ID
	 * \since v3.1
	 */
	bool isConnectionPlastic(short int connId);

	/*!
	 * \brief Returns whether a group has homeostasis enabled
	 *
	 * This functions returns whether a group has homeostasis enabled (true) or not (false).
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param[in] grpId group ID
	 * \since v3.1
	 */
	bool isGroupWithHomeostasis(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	bool isExcitatoryGroup(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	bool isInhibitoryGroup(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	bool isPoissonGroup(int grpId);

	// +++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Sets default values for conductance time constants
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] tdAMPA   time constant for AMPA decay (ms)
	 * \param[in] trNMDA   time constant for NMDA rise (ms)
	 * \param[in] tdNMDA   time constant for NMDA decay (ms)
	 * \param[in] tdGABAa  time constant for GABAa decay (ms)
	 * \param[in] trGABAb  time constant for GABAb rise (ms)
	 * \param[in] tdGABAb  time constant for GABAb decay (ms)
	 */
	void setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);

	/*!
	 * \brief Sets default homeostasis params
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 */
	void setDefaultHomeostasisParams(float homeoScale, float avgTimeScale);

	/*!
	 * \brief Sets default options for save file
	 *
	 * \TODO finish docu
	 * \STATE ::CONFIG_STATE
	 */
	void setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo);

	/*!
	* \brief sets default STDP params
	*
	* Sets default STDP parameters. Do not use this function, it is deprecated.
	*
	* \STATE ::CONFIG_STATE
	* \deprecated For clearness, setting STDP parameters using setESTDP and setISTDP is strongly recommended.
	*/
	void setDefaultSTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType);

	/*!
	* \brief sets default values for E-STDP params
	*
	* Sets default E-STDP parameters. Do not use this function, it is deprecated.
	*
	* \STATE ::CONFIG_STATE
	* \deprecated For clearness, setting STDP parameters using setESTDP and setISTDP is strongly recommended.
	*/
	void setDefaultESTDPparams(float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, STDPType stdpType);

	/*!
	* \brief sets default values for I-STDP params
	*
	* Sets default I-STDP parameters. Do not use this function, it is deprecated.
	*
	* \STATE ::CONFIG_STATE
	* \deprecated For clearness, setting STDP parameters using setESTDP and setISTDP is strongly recommended.
	*/
	void setDefaultISTDPparams(float betaLTP, float betaLTD, float lambda, float delta, STDPType stdpType);

	/*!
	 * \brief Sets default values for STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
	 *
	 * This function sets the default values for STP parameters U tau_u, and tau_x.
	 * These values will then apply to all subsequent calls to setSTP(int, bool).
	 *
	 * CARLsim will automatically assign the following values, which can be changed at any time during ::CONFIG_STATE:
	 * The default parameters for an excitatory neuron are U=0.45, tau_u=50.0, tau_f=750.0 (depressive).
	 * The default parameters for an inhibitory neuron are U=0.15, tau_u=750.0, tau_f=50.0 (facilitative).
	 *
	 * Source: Misha Tsodyks and Si Wu (2013) Short-term synaptic plasticity. Scholarpedia, 8(10):3153., rev #136920
	 *
	 * \STATE ::CONFIG_STATE
	 * \param[in] neurType   either EXCITATORY_NEURON or INHIBITORY_NEURON
	 * \param[in] STP_U      default value for increment of u induced by a spike
	 * \param[in] STP_tau_u  default value for decay constant of u
	 * \param[in] STP_tau_x  default value for decay constant of x
	 * \see setSTP(int, bool)
	 * \see setSTP(int, bool, float, float, float)
	 * \since v3.0
	 */
	void setDefaultSTPparams(int neurType, float STP_U, float STP_tau_u, float STP_tau_x);


private:
	// This class provides a pImpl for the CARLsim User API.
	// \see https://marcmutz.wordpress.com/translated-articles/pimp-my-pimpl/
	class Impl;
	Impl* _impl;
};
#endif
