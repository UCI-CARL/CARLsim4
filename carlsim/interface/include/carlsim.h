/*
 * Copyright (c) 2014 Regents of the University of California. All rights reserved.
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
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */

#ifndef _CARLSIM_H_
#define _CARLSIM_H_

#include <stdint.h>		// uint64_t, uint32_t, etc.
#include <string>		// std::string
#include <vector>		// std::vector

#include <callback.h>
#include <carlsim_definitions.h>
#include <carlsim_datastructures.h>

// include the following core functionalities instead of forward-declaring, so that the user only needs to include
// carlsim.h
#include <poisson_rate.h>
#include <spike_monitor.h>

#include <linear_algebra.h>

// \TODO: complete documentation




class CpuSNN; // forward-declaration of implementation
class GroupMonitorCore;
class ConnectionMonitorCore;
class ConnectionGeneratorCore;
class SpikeGeneratorCore;

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
	 * specify which CUDA device to use (param ithGPU, 0-indexed).
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
	 * Location of the debug log file can be set in any mode using CARLsim::setLogDebugFp.
	 * In mode CUSTOM, the other file pointers can be set using CARLsim::setLogsFp.
	 *
	 * \param[in] netName 		network name
	 * \param[in] simMode		either CPU_MODE or GPU_MODE
	 * \param[in] loggerMode    either USER, DEVELOPER, SILENT, or CUSTOM
	 * \param[in] ithGPU 		on which GPU to establish a context (only relevant in GPU_MODE)
	 * \param[in] randSeed 		random number generator seed
	 */
	CARLsim(const std::string& netName="SNN", simMode_t simMode=CPU_MODE, loggerMode_t loggerMode=USER, int ithGPU=0,
				int randSeed=-1);
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
	 * \STATE CONFIG
	 * \param[in] grpId1     ID of the pre-synaptic group
	 * \param[in] grpId2     ID of the post-synaptic group
	 * \param[in] connType   connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in
	 *                       pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post.
	 *                       "full-no-direct": same as "full", but i-th neuron of grpId1 will not be connected to the
	 *                       i-th neuron of grpId2.
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
	 *                         * Create a 2D Gaussian RF of radius 10: RadiusRF(10, 10, 0)
	 *                         * Create a 2D heterogeneous Gaussian RF (an ellipse) with semi-axes 10 and 5:
	 *                           RadiusRF(10, 5, 0)
	 *                         * Connect only the third dimension: RadiusRF(0, 0, 1)
	 *                         * Connect all, no matter the RF (default): RadiusRF(-1, -1, -1)
	 *                         * Don't connect anything (silly, not allowed): RadiusRF(0, 0, 0)
	 * \param[in] synWtType  specifies whether the synapse should be of fixed value (SYN_FIXED) or plastic (SYN_PLASTIC)
	 * \param[in] mulSynFast a multiplication factor to be applied to the fast synaptic current (AMPA in the case of
	 *                       excitatory, and GABAa in the case of inhibitory connections). Default: 1.0
	 * \param[in] mulSynSlow a multiplication factor to be applied to the slow synaptic current (NMDA in the case of
	 *                       excitatory, and GABAb in the case of inhibitory connections). Default: 1.0
	 * \returns a unique ID associated with the newly created connection
	 */
	short int connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
		const RangeDelay& delay=RangeDelay(1), const RadiusRF& radRF=RadiusRF(-1), bool synWtType=SYN_FIXED,
		float mulSynFast=1.0f, float mulSynSlow=1.0f);

	/*!
	 * \brief Shortcut to make connections with custom connectivity profile but omit scaling factors for synaptic
	 * conductances (default is 1.0 for both)
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED, int maxM=0,
						int maxPreM=0);

	/*!
	 * \brief make connections with custom connectivity profile
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType=SYN_FIXED, int maxM=0,int maxPreM=0);


	/*!
	 * \brief creates a group of Izhikevich spiking neurons
	 * \TODO finish doc
	 * \STATE CONFIG
	 */
	int createGroup(const std::string& grpName, int nNeur, int neurType);
	
	/*!
	 * \brief create a group of neurons on a 3D grid
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
	 * \STATE CONFIG
	 * \TODO finish doc
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType);

	/*!
	 * \brief  creates a spike generator group
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType);
	
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
	 * \STATE CONFIG
	 * \TODO finish doc
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType);


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
	 * \STATE CONFIG
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
	 * \STATE CONFIG
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
	 * \STATE CONFIG
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
	 * time over which this average is computed should be on the scale of seconds to minutes hours as
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
	 * \STATE CONFIG
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
	 * \STATE CONFIG
	 * \param[in] grpId        the group ID of group to which homeostasis is applied
	 * \param[in] isSet        a boolean, setting it to true/false enables/disables homeostasis
	 */
	void setHomeostasis(int grpId, bool isSet);

	 /*!
	 * \brief Sets the homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	 *
	 * This function allows the user to set the homeostatic target firing with or without a standard
	 * deviation. All neurons in the group will use homeostatic synaptic scaling to attain the target
	 * firing rate. You can have a standard deviation to the base firing rate or you can disable it
	 * by setting it to 0. It should be noted that the baseFiringSD only sets the base firing rate
	 * to a single value within that standard deviation. It does not vary the value of the base firing
	 * rate from this value or within a particular range. For more information on this implementation
	 * please see: Carlson, et al. (2013). Proc. of IJCNN 2013.
	 *
	 * \STATE CONFIG
	 * \param[in] grpId        the group ID of group for which these settings are applied
	 * \param[in] baseFiring   target firing rate of every neuron in this group
	 * \param[in] baseFiringSD standard deviation of target firing rate of every neuron in this group
	 */
	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD);

	/*!
	 * \brief Sets Izhikevich params a, b, c, and d with as mean +- standard deviation
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd);

	/*!
	 * \brief Sets Izhikevich params a, b, c, and d of a neuron group.
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d);

	/*!
	 * \brief Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron
	 * group.
	 *
	 * \STATE CONFIG
	 * \param groupId the symbolic name of a group
	 * \param baseDP  the baseline concentration of Dopamine
	 * \param tauDP the decay time constant of Dopamine
	 * \param base5HT  the baseline concentration of Serotonin
	 * \param tau5HT the decay time constant of Serotonin
	 * \param baseACh  the baseline concentration of Acetylcholine
	 * \param tauACh the decay time constant of Acetylcholine
	 * \param baseNE  the baseline concentration of Noradrenaline
	 * \param tauNE the decay time constant of Noradrenaline
	 */
	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							float baseACh, float tauACh, float baseNE, float tauNE);

	/*!
	 * \brief Sets default neuromodulators
	 *
	 * \TODO: this should be implemented via default arguments as members of the class, so that the user can call
	 * setDefaultNeuromodulators()
	 *
	 * \STATE CONFIG
	 */
	void setNeuromodulator(int grpId, float tauDP = 100.0f, float tau5HT = 100.0f,
							float tauACh = 100.0f, float tauNE = 100.0f);

	/*!
	 * \brief Sets default STDP mode and params
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setSTDP(int grpId, bool isSet);

	/*!
	 * \brief Sets STDP params for a group, custom
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setSTDP(int grpId, bool isSet, stdpType_t type, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD);

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
	 * current. Between spikes, u decays back to zero with time constant STP_tau_u (\tau_F), and x recovers to value one
	 * with time constant STP_tau_x (\tau_D).
	 *
	 * \STATE CONFIG
	 * \param[in] grpId       pre-synaptic group id. STP will apply to all neurons of that group!
	 * \param[in] isSet       a flag whether to enable/disable STP
	 * \param[in] STP_U       increment of u induced by a spike
	 * \param[in] STP_tau_u   decay constant of u (\tau_F)
	 * \param[in] STP_tau_x   decay constant of x (\tau_D)
	 */
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x);

	/*!
	 * \brief Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically) using default values
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setSTP(int grpId, bool isSet);

	/*!
	 * \brief Sets the weight and weight change update parameters
	 *
	 * \STATE CONFIG
	 * \param[in] wtANDwtChangeUpdateInterval the interval between two wt (weight) and wtChange (weight change) update.
	 * \param[in] enableWtChangeDecay enable weight change decay
	 * \param[in] wtChangeDecay the decay ratio of weight change (wtChange)
	 */
	void setWeightAndWeightChangeUpdate(updateInterval_t wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay=0.9f);


	// +++++ PUBLIC METHODS: RUNNING A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for time=(nSec*seconds + nMsec*milliseconds)
	 *
	 * \STATE SETUP, EXECUTION. First call to runNetwork will make CARLsim state switch from SETUP to EXECUTION.
	 * \param[in] nSec 			  number of seconds to run the network
	 * \param[in] nMsec 		  number of milliseconds to run the network
	 * \param[in] printRunSummary enable the printing of a summary at the end of this run
	 * \param[in] copyState 	  enable copying of data from device to host
	 */
	int runNetwork(int nSec, int nMsec=0, bool printRunSummary=true, bool copyState=false);

	/*!
	 * \brief build the network
	 *
	 * \STATE CONFIG. Will make CARLsim state switch from CONFIG to SETUP.
	 * \param[in] removeTempMemory 	remove temp memory after building network
	 */
	void setupNetwork(bool removeTempMemory = true);

	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const FILE* getLogFpInf();	//!< returns file pointer to info log
	const FILE* getLogFpErr();	//!< returns file pointer to error log
	const FILE* getLogFpDeb();	//!< returns file pointer to debug log
	const FILE* getLogFpLog();	//!< returns file pointer to log file

	/*!
	 * \brief Saves important simulation and network infos to file
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	void saveSimulation(const std::string& fileName, bool saveSynapseInfo=true);

	/*!
	 * \brief Sets the file pointer of the debug log file
	 *
	 * \param[in] fpLog file pointer to new log file
	 */
	void setLogDebugFp(FILE* fpLog);

	/*!
	 * \brief Sets the file pointers for all log files
	 *
	 * \param[in] fpInf file pointer for status info
	 * \param[in] fpErr file pointer for errors/warnings
	 * \param[in] fpDeb file pointer for debug info
	 * \param[in] fpLog file pointer for debug log file that contains all the above info
	 */
	void setLogsFp(FILE* fpInf, FILE* fpErr=NULL, FILE* fpDeb=NULL, FILE* fpLog=NULL);



	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief loads a simulation (and network state) from file
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void loadSimulation(FILE* fid);

	/*!
	 * \brief Reassigns fixed weights to values passed into the function in a single 1D float matrix (weightMatrix)
	 *
	 * The user passes the connection ID (connectID), the weightMatrix, and the matrixSize.
	 * This function only works for fixed synapses and for connections of type CONN_USER_DEFINED. Only the weights are 
	 * changed, not the maxWts, delays, or connected values
	 * \TODO finish docu
	 *
	 * \STATE SETUP
	 */
	void reassignFixedWeights(short int connectId, float weightMatrix[], int matrixSize);

	//! \deprecated Deprecated
	void resetSpikeCntUtil(int grpId=ALL); //!< resets spike count for particular neuron group

	/*!
	 * \brief reset Spike Counter to zero
	 *
	 * Manually resets the spike buffers of a Spike Counter to zero (for a specific group).
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time, as long as you call CARLsim::setupNetwork() first.
	 *
	 * \STATE EXECUTION
	 * \param grpId the group for which to reset the spike counts. Set to ALL if you want to reset all Spike Counters.
	 */
	void resetSpikeCounter(int grpId);

	/*!
	 * \brief Sets a connection monitor for a group, custom ConnectionMonitor class
	 *
	 * To retrieve connection status, a connection-monitoring callback mechanism is used. This mechanism allows the user
	 * to monitor connection status between groups. Connection monitors are registered for two groups (i.e., pre- and
	 * post- synaptic groups) and are called automatically by the simulator every second.
	 *
	 * Use setConnectionMonitor(grpIdPre,grpIdPost) to use a ConnectionMonitor with default settings.
	 *
	 * \STATE CONFIG, SETUP
	 * \param[in] grpIdPre 		the pre-synaptic group ID
	 * \param[in] grpIdPost 	the post-synaptic group ID
	 * \param[in] connectionMon an instance of class ConnectionMonitor (see callback.h)
	 */
	void setConnectionMonitor(int grpIdPre, int grpIdPost, ConnectionMonitor* connectionMon=NULL);

	/*!
	 * \brief Sets the amount of current (mA) to inject into a group
	 *
	 * This method injects current, specified on a per-neuron basis, into the soma of each neuron in the group, at 
	 * each timestep of the simulation. current is a float vector of current amounts (mA), one element per neuron in 
	 * the group.
	 *
	 * To input different currents into a neuron over time, the idea is to run short periods of CARLsim::runNetwork
	 * and subsequently calling CARLsim::setExternalCurrent again with updated current values.
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
	 * \STATE SETUP, EXECUTION
	 * \param[in] grpId    the group ID
	 * \param[in] current  a float vector of current amounts (mA), one value per neuron in the group
	 *
	 * \note This method cannot be applied to SpikeGenerator groups.
	 * \note If all neurons in the group should receive the same amount of current, you can use the convenience 
	 * function CARLsim::setExternalCurrent(int grpId, float current).
	 *
	 * \attention Make sure to reset current after use (i.e., for the next call to CARLsim::runNetwork), otherwise
	 * the current will keep getting applied to the group.
	 * \see CARLsim::setExternalCurrent(int grpId, float current)
	 * \see CARLsim::setSpikeRate
	 * \see CARLsim::setSpikeGenerator
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
	 * CARLsim::setExternalCurrent(int grpId, const std::vector<float>& current) instead.
	 *
	 * \attention Make sure to reset current after use (i.e., for the next call to CARLsim::runNetwork), otherwise
	 * the current will keep getting applied to the group.
	 * \see CARLsim::setExternalCurrent(int grpId, const std::vector<float>& current)
	 * \see CARLsim::setSpikeRate
	 * \see CARLsim::setSpikeGenerator
	 */
	void setExternalCurrent(int grpId, float current);

	/*!
	 * \brief Sets a group monitor for a group, custom GroupMonitor class
	 *
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP
	 */
	void setGroupMonitor(int grpId, GroupMonitor* groupMon=NULL);

	/*!
	 * \brief A Spike Counter keeps track of the number of spikes per neuron in a group.
	 *
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur).
	 * After that, the spike buffers get reset to zero number of spikes.
	 * Works for excitatory/inhibitory neurons.
	 * The recording time can be set to any x number of ms, so that after x ms the spike counts will be reset
	 * to zero. If x==-1, then the spike counts will never be reset (should only overflow after 97 days of sim).
	 * Also, spike counts can be manually reset at any time by calling snn->resetSpikeCounter(group);
	 * At any time, you can call getSpikeCounter to get the spiking information out.
	 * You can have only one spike counter per group. However, a group can have both a SpikeMonitor and a SpikeCounter.
	 *
	 * \STATE CONFIG, SETUP
	 * \param[in] grpId the group for which you want to enable a SpikeCounter
	 * \param[in] recordDur number of ms for which to record spike numbers. Spike numbers will be reset to zero after
	 * this. Set frameDur to -1 to never reset spike counts. Default: -1.
	 */
	void setSpikeCounter(int grpId, int recordDur=-1);

	/*!
	 * \brief Sets up a spike generator
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen);

	/*!
	 * \brief Sets a Spike Monitor for a groups, prints spikes to binary file
	 *
	 * To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
	 * basic statistics, store spike trains, or perform more complicated output monitoring. Spike monitors are
	 * registered for a group and are called automatically by the simulator every second. Similar to an address event
	 * representation (AER), the spike monitor indicates which neurons spiked by using the neuron ID within a group
	 * (0-indexed) and the time of the spike. Only one spike monitor is allowed per group.
	 *
	 * Every second, the SpikeMonitor will print to console the total and average number of spikes in the group.
	 *
	 * In addition, all spikes will be stored in a binary file (in AER format). This file can be read off-line in Matlab
	 * by using readSpikes.m from /util/scripts. A file name can be specified via variable fname (specified directory
	 * must exist). If no file name is specified, a default one will be created in the results directory:
	 * "results/spk{group name}.dat", where group name is the name assigned to the group at initialization (can be
	 * retrieved via CARLsim::getGroupName).
	 * If no binary file shall be created, set fname equal to the string "NULL".
	 *
	 * The function returns a pointer to a SpikeMonitor object, which can be used to calculate spike statistics (such
	 * group firing rate, number of silent neurons, etc.) or retrieve all spikes from a particular time window. See
	 * /util/spike_monitor/spike_monitor.h for more information on how to interact with the SpikeMonitor object.
	 *
	 * \STATE CONFIG, SETUP
	 * \param[in] grpId 		the group ID
	 * \param[in] fname 		name of the binary file to be created. Leave empty for default name
	 *                      	"results/spk{grpName}.dat". Set to string "NULL" to suppress file creation. Default: ""
	 * \returns   SpikeMonitor*	pointer to a SpikeMonitor object, which can be used to calculate spike statistics
	 *                          (such as group firing rate, number of silent neurons, etc.) or retrieve all spikes in
	 * 							AER format
	 *
	 * \note Only one SpikeMonitor is allowed per group.
	 *
	 * \attention Using SpikeMonitor::startRecording and SpikeMonitor::stopRecording might significantly slow down the
	 * simulation. It is unwise to use this mechanism to record a large number of spikes over a long period of time.
	 */
	SpikeMonitor* setSpikeMonitor(int grpId, const std::string& fname="");

	/*!
	 * \brief Sets a spike rate
	 * \TODO finish docu
	 *
	 * \STATE SETUP, EXECUTION
	 * \param[in] grpId      group ID
	 * \param[in] spikeRate  pointer to PoissonRate object
	 * \param[in] refPeriod  refactory period (ms). Default: 1ms.
	 *
	 * \note This method can only be applied to SpikeGenerator groups.
	 *
	 * \attention Make sure to reset spike rate after use (i.e., for the next call to CARLsim::runNetwork), otherwise
	 * the rate will keep getting applied to the group.
	 * \see CARLsim::setExternalCurrent
	 * \see CARLsim::setSpikeGenerator
	 */
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1);

	/*!
	 * \brief Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	 * weight values back to their default values by setting resetWeights = true.
	 *
	 * \STATE EXECUTION
	 * \deprecated This function is deprecated.
	 */
	void updateNetwork(bool resetFiringInfo, bool resetWeights);

	/*!
	 * \brief function writes population weights from gIDpre to gIDpost to file fname in binary.
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	void writePopWeights(std::string fname, int gIDpre, int gIDpost);



	// +++++ PUBLIC METHODS: GETTER / SETTERS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Returns the current CARLsim state
	 *
	 * A CARLsim simulation goes through the following states:
	 * - CONFIG 	configuration state, where the neural network is configured
	 * - SETUP 		setup state, where the neural network is prepared for execution
	 * - EXECUTION 	execution state, where the simulation is executed
	 *
	 * Certain methods can only be called in certain states. Check their documentation to see which method can be called
	 * in which state.
	 *
	 * Certain methods perform state transitions. CARLsim::setupNetwork will change the state from CONFIG to SETUP. The
	 * first call to CARLsim::runNetwork will change the state from SETUP to EXECUTION.
	 * \returns current CARLsim state
	 */
	carlsimState_t getCarlsimState() { return carlsimState_; }

	/*!
	 * \brief gets delays
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays=NULL);

	/*!
	 * \brief returns the 3D grid struct of a group
	 *
	 * This function returns the Grid3D struct of a particular neuron group.
	 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid. This simplifies
	 * the creation of topographic connections in the network. The dimensions of the grid can thus be retrieved by
	 * calling Grid3D.width, Grid3D.height, and Grid3D.depth. The total number of neurons is given by Grid3D.N.
	 * See CARLsim::createGroup and Grid3D for more information.
	 * \STATE SETUP, EXECUTION
	 * \param[in] grpId the group ID for which to get the Grid3D struct
	 * \returns the 3D grid struct of a group
	 */
	Grid3D getGroupGrid3D(int grpId);

	/*!
	 * \brief finds the ID of the group with name grpName
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	int getGroupId(std::string grpName);

	/*!
	 * \brief gets group name
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
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
	 * For more information see CARLsim::createGroup and the Grid3D struct.
 	 * See also CARLsim::getNeuronLocation3D(int grpId, int relNeurId).
	 *
	 * \STATE CONFIG, SETUP, EXE
	 * \param[in] neurId the neuron ID for which the 3D location should be returned
	 * \returns the 3D location a neuron codes for as a Point3D struct
	 */
	Point3D getNeuronLocation3D(int neurId);

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
	 * For more information see CARLsim::createGroup and the Grid3D struct.
	 * See also CARLsim::getNeuronLocation3D(int neurId).
	 *
	 * \STATE CONFIG, SETUP, EXE
	 * \param[in] neurId the neuron ID for which the 3D location should be returned
	 * \returns the 3D location a neuron codes for as a Point3D struct
	 */
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	/*!
	 * \brief Returns the number of connections (pairs of pre-post groups) in the network
	 *
	 * This function returns the number of connections (pairs of pre-post groups) in the network. Each pre-post
	 * pair of neuronal groups has its own connection ID, which is returned by a call to CARLsim::connect.
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \STATE CONFIG, SETUP, EXECUTION
	 * \returns the number of connections (pairs of pre-post groups) in the network
	 */
	int getNumConnections();

	/*!
	 * \brief returns the number of connections associated with a connection ID
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	int getNumSynapticConnections(short int connectionId);

	/*!
	 * \brief returns the number of groups in the network
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumGroups();

	/*!
	 * \brief returns the total number of allocated neurons in the network
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeurons();

	/*!
	 * \brief returns the total number of regular (Izhikevich) neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsReg();

	/*!
	 * \brief returns the total number of regular (Izhikevich) excitatory neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsRegExc();

	/*!
	 * \brief returns the total number of regular (Izhikevich) inhibitory neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsRegInh();

	/*!
	 * \brief returns the total number of spike generator neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsGen();

	/*!
	 * \brief returns the total number of excitatory spike generator neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsGenExc();

	/*!
	 * \brief returns the total number of inhibitory spike generator neurons
	 *
	 * \Note This number might change throughout CARLsim state CONFIG, up to calling CARLsim::setupNetwork).
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumNeuronsGenInh();

	/*!
	 * \brief returns the total number of allocated pre-synaptic connections in the network
	 *
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumPreSynapses();

	/*!
	 * \brief returns the total number of allocated post-synaptic connections in the network
	 *
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getNumPostSynapses();

	/*!
	 * \brief returns the first neuron id of a groupd specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	int getGroupStartNeuronId(int grpId);

	/*!
	 * \brief returns the last neuron id of a groupd specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	int getGroupEndNeuronId(int grpId);

	/*!
	 * \brief returns the number of neurons of a group specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE CONFIG, SETUP, EXECUTION
	 */
	int getGroupNumNeurons(int grpId);

	/*!
	 * \brief returns the stdp information of a group specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	GroupSTDPInfo_t getGroupSTDPInfo(int grpId);

	/*!
	 * \brief returns the neuromodulator information of a group specified by grpId
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	GroupNeuromodulatorInfo_t getGroupNeuromodulatorInfo(int grpId);

	/*!
	 * \brief Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
	 *
	 * and the size of the 1D array in size.  gIDpre(post) is the group ID for the pre(post)synaptic group,
	 * weights is a pointer to a single dimensional array of floats, size is the size of that array which is
	 * returned to the user.  NOTE: user must free memory from weights to avoid a memory leak.
	 * \TODO why not readPopWeight()?
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	void getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	uint64_t getSimTime();

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	uint32_t getSimTimeSec();

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	uint32_t getSimTimeMsec();

	/*!
	 * \brief returns pointer to 1D array of the number of spikes every neuron in the group has fired
	 *
	 * \deprecated deprecated
	 * \STATE EXECUTION
	 */
	int* getSpikeCntPtr(int grpId);

	/*!
	 * \brief return the number of spikes per neuron for a certain group
	 *
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur) at any point in time.
	 *
	 * \STATE EXECUTION
	 * \param[in] grpId	   the group for which you want the spikes (cannot be ALL)
	 * \returns pointer to array of ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (int) since the last reset.
	 */
	int* getSpikeCounter(int grpId);

	/*!
	 * \brief returns pointer to weight array
	 *
	 * \STATE EXECUTION
	 * \TODO: maybe consider renaming getPopWeightChanges
	 * \FIXME fix this
	 * \deprecated deprecated
	 */
	float* getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges=NULL);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	bool isExcitatoryGroup(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	bool isInhibitoryGroup(int grpId);

	/*!
	 * \brief returns
	 *
	 * \TODO finish docu
	 * \STATE SETUP, EXECUTION
	 */
	bool isPoissonGroup(int grpId);

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 *
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 * \deprecated deprecated
	 * \STATE EXECUTION
	 */
	void setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr);


	// +++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Sets default values for conductance time constants
	 *
	 * \STATE CONFIG
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
	 * \STATE CONFIG
	 */
	void setDefaultHomeostasisParams(float homeoScale, float avgTimeScale);

	/*!
	 * \brief Sets default options for save file
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo);

	/*!
	* \brief sets default values for STDP params
	*
	* \TODO finish docu
	* \STATE CONFIG
	*/
	void setDefaultSTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD);

	/*!
	 * \brief sets default values for STP params (neurType either EXCITATORY_NEURON or INHIBITORY_NEURON)
	 *
	 * \TODO finish docu
	 * \STATE CONFIG
	 */
	void setDefaultSTPparams(int neurType, float STP_U, float STP_tau_U, float STP_tau_x);


private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	void CARLsimInit();					//!< init function, unsafe computations that would usually go in constructor

	bool existsGrpId(int grpId);		//!< checks whether a certain grpId exists in grpIds_

	void handleUserWarnings(); 			//!< print all user warnings, continue only after user input

	void printSimulationSpecs();

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	CpuSNN* snn_;					//!< an instance of CARLsim core class
	std::string netName_;			//!< network name
	int randSeed_;					//!< RNG seed
	simMode_t simMode_;				//!< CPU_MODE or GPU_MODE
	loggerMode_t loggerMode_;		//!< logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	int ithGPU_;					//!< on which device to establish a context
	bool enablePrint_;
	bool copyState_;

	unsigned int numConnections_;	//!< keep track of number of allocated connections
	std::vector<std::string> userWarnings_; // !< an accumulated list of user warnings

	std::vector<int> grpIds_;		//!< a list of all created group IDs
	std::vector<ConnectionMonitorCore*> connMon_; //!< a list of all created connection monitors
	std::vector<GroupMonitorCore*> groupMon_; //!< a list of all created group monitors
	std::vector<SpikeGeneratorCore*> spkGen_; //!< a list of all created spike generators
	std::vector<ConnectionGeneratorCore*> connGen_; //!< a list of all created connection generators

	bool hasSetHomeoALL_;			//!< informs that homeostasis have been set for ALL groups (can't add more groups)
	bool hasSetHomeoBaseFiringALL_;	//!< informs that base firing has been set for ALL groups (can't add more groups)
	bool hasSetSTDPALL_; 			//!< informs that STDP have been set for ALL groups (can't add more groups)
	bool hasSetSTPALL_; 			//!< informs that STP have been set for ALL groups (can't add more groups)
	bool hasSetConductances_;		//!< informs that setConductances has been called
	carlsimState_t carlsimState_;	//!< the current state of carlsim

	int def_tdAMPA_;				//!< default value for AMPA decay (ms)
	int def_trNMDA_;				//!< default value for NMDA rise (ms)
	int def_tdNMDA_;				//!< default value for NMDA decay (ms)
	int def_tdGABAa_;				//!< default value for GABAa decay (ms)
	int def_trGABAb_;				//!< default value for GABAb rise (ms)
	int def_tdGABAb_;				//!< default value for GABAb decay (ms)

	// all default values for STDP
	stdpType_t def_STDP_type_;		//!< default mode for STDP
	float def_STDP_alphaLTP_;		//!< default value for LTP amplitude
	float def_STDP_tauLTP_;			//!< default value for LTP decay (ms)
	float def_STDP_alphaLTD_;		//!< default value for LTD amplitude
	float def_STDP_tauLTD_;			//!< default value for LTD decay (ms)

	// all default values for STP
	float def_STP_U_exc_;			//!< default value for STP U excitatory
	float def_STP_tau_u_exc_;		//!< default value for STP u decay (\tau_F) excitatory (ms)
	float def_STP_tau_x_exc_;		//!< default value for STP x decay (\tau_D) excitatory (ms)
	float def_STP_U_inh_;			//!< default value for STP U inhibitory
	float def_STP_tau_u_inh_;		//!< default value for STP u decay (\tau_F) inhibitory (ms)
	float def_STP_tau_x_inh_;		//!< default value for STP x decay (\tau_D) inhibitory (ms)

	// all default values for homeostasis
	float def_homeo_scale_;			//!< default homeoScale
	float def_homeo_avgTimeScale_;	//!< default avgTimeScale

	// all default values for save file
	std::string def_save_fileName_;	//!< file name for saving network info
	bool def_save_synapseInfo_;		//!< flag to inform whether to include synapse info in fpSave_
};
#endif
