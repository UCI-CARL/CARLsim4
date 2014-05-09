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

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <string>		// std::string
#include <vector>		// std::vector

#include <callback.h>
#include <carlsim_definitions.h>
#include <carlsim_datastructures.h>

#include <poisson_rate.h>

// TODO: complete documentation




class CpuSNN; // forward-declaration of private implementation

/*!
 * \brief CARLsim User Interface.
 * This class provides a user interface to the public sections of CARLsimCore source code. Example networks that use
 * this methodology can be found in the examples/ directory. Documentation is available on our website.
 *
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
	 * \param[in] nConfig 		number of network configurations 									// TODO: explain
	 * \param[in] randSeed 		random number generator seed
	 */
	CARLsim(std::string netName="SNN", simMode_t simMode=CPU_MODE, loggerMode_t loggerMode=USER, int ithGPU=0,
				int nConfig=1, int randSeed=-1);
	~CARLsim();



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Connects a presynaptic to a postsynaptic group using fixed/plastic weights and a range of delay values
	 *
	 * This function is a shortcut to create synaptic connections from a pre-synaptic group grpId1 to a post-synaptic
	 * group grpId2 using a pre-defined primitive type (such as "full", "one-to-one", or "random"). Synapse weights 
	 * will stay the same throughout the simulation (SYN_FIXED, no plasticity). All synapses will have the same delay.
	 * For more flexibility, see the other connect() calls.
	 * \param[in] grpId1	 ID of the pre-synaptic group
	 * \param[in] grpId2 	 ID of the post-synaptic group
	 * \param[in] connType 	 connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in 
	 *						 pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post.
	 * 						 "full-no-direct": same as "full", but i-th neuron of grpId1 will not be connected to the
	 *                       i-th neuron of grpId2.
	 * \param[in] wt 		 a struct specifying the range of weight magnitudes (initial value and max value). Weights
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
	 * \param[in] connProb	 connection probability
	 * \param[in] delay 	 a struct specifying the range of delay values (ms). Synaptic delays must be greater than or
	 *                       equal to 1 ms.
	 *                       Examples:
	 *                         RangeDelay(2) => all delays will be 2 (delay.min=2, delay.max=2)
	 *                         RangeDelay(1,10) => delays will be in range [1,10]
	 * \param[in] synWtType  specifies whether the synapse should be of fixed value (SYN_FIXED) or plastic (SYN_PLASTIC)
	 * \param[in] mulSynFast a multiplication factor to be applied to the fast synaptic current (AMPA in the case of
	 *                       excitatory, and GABAa in the case of inhibitory connections). Default: 1.0
	 * \param[in] mulSynSlow a multiplication factor to be applied to the slow synaptic current (NMDA in the case of
	 *                       excitatory, and GABAb in the case of inhibitory connections). Default: 1.0
	 * \returns a unique ID associated with the newly created connection
	 * \note The method can be called at configuration state only
	 */
	short int connect(int grpId1, int grpId2, const std::string& connType, RangeWeight wt, float connProb,
		RangeDelay delay, bool synWtType=SYN_FIXED, float mulSynFast=1.0f, float mulSynSlow=1.0f);

	//! shortcut to make connections with custom connectivity profile but omit scaling factors for synaptic
	//! conductances (default is 1.0 for both)
	//! \note The method can be called at configuration state only
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED, int maxM=0, 
						int maxPreM=0);

	//! make connections with custom connectivity profile
	//! \note The method can be called at configuration state only
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType=SYN_FIXED, int maxM=0,int maxPreM=0);


	//! creates a group of Izhikevich spiking neurons
	//! \note The method can be called at configuration state only
	int createGroup(const std::string grpName, int nNeur, int neurType, int configId=ALL);

	//! creates a spike generator group
	//! \note The method can be called at configuration state only
	int createSpikeGeneratorGroup(const std::string grpName, int nNeur, int neurType, int configId=ALL);


	//! Sets default values for conduction decays or disables COBA if isSet==false
	/*!
	 * \brief Sets default values for conduction decay and rise times or disables COBA alltogether
	 *
	 * This function sets the time constants for the decay of AMPA, NMDA, GABA, and GABAb, and the rise times for
	 * NMDA and GABAb. These constants will be applied to all connections in the network. Set isSet to false to run
	 * your simulation in CUBA mode.
	 * Use setDefaultConductanceTimeConstants to set default values for all time constants.
	 * If you call this function without setting your own defaults, then the following defaults will be used:
	 * tdAMPA=5ms, trNMDA=0, tdNMDA=150ms, tdGABAa=6ms, trGABAb=0, tdGABAb=150ms (instantaneous rise time).
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 * \note The method can be called at configuration state only
	 */
	void setConductances(bool isSet, int configId=ALL);

	/*!
	 * \brief Sets custom values for conduction decay times (instantaneous rise time) or disables COBA alltogether
	 *
	 * This function sets the time constants for the decay of AMPA, NMDA, GABAa, and GABAb. The decay constants will be
	 * applied to all connections in the network. Set isSet to false to run your simulation in CUBA mode.
	 * The NMDA current is voltage dependent (see Izhikevich et al., 2004).
	 * Use setConductances(true) to use default decay values.
	 * Use the other setConductances to enable non-zero rise times for NMDA and GABAb.
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 * \param[in] tdAMPA  time constant for AMPA decay (ms)
	 * \param[in] tdNMDA  time constant for NMDA decay (ms)
	 * \param[in] tdGABAa time constant for GABAa decay (ms)
	 * \param[in] tdGABAb time constant for GABAb decay (ms)
	 * \note The method can be called at configuration state only
	 */
	void setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb, int configId=ALL);

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
	 * \param[in] isSet   a flag to inform whether to run simulation in COBA mode (true) or CUBA mode (false)
	 * \param[in] tdAMPA  time constant for AMPA decay (ms)
	 * \param[in] trNMDA  time constant for NMDA rise (ms), must be smaller than tdNMDA
	 * \param[in] tdNMDA  time constant for NMDA decay (ms)
	 * \param[in] tdGABAa time constant for GABAa decay (ms)
	 * \param[in] trGABAb time constant for GABAb rise (ms), must be smaller than tdGABAb
	 * \param[in] tdGABAb time constant for GABAb decay (ms)
	 * \note The method can be called at configuration state only
	 */
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb,
		int configId=ALL);

	//! Sets default homeostasis params for group
	//! \note The method can be called at configuration state only
	void setHomeostasis(int grpId, bool isSet, int configId=ALL);

	//! Sets custom homeostasis params for group
	//! \note The method can be called at configuration state only
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId=ALL);

	//! Sets homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	//! \note The method can be called at configuration state only
	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId=ALL);

	//! Sets Izhikevich params a, b, c, and d with as mean +- standard deviation
	//! \note The method can be called at configuration state only
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId=ALL);

	//! Sets Izhikevich params a, b, c, and d of a neuron group.
	//! \note The method can be called at configuration state only
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId=ALL);

	/*!
	 * \brief Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron
	 * group.
	 *
	 * \param groupId the symbolic name of a group
	 * \param baseDP  the baseline concentration of Dopamine
	 * \param tauDP the decay time constant of Dopamine
	 * \param base5HT  the baseline concentration of Serotonin
	 * \param tau5HT the decay time constant of Serotonin
	 * \param baseACh  the baseline concentration of Acetylcholine
	 * \param tauACh the decay time constant of Acetylcholine
	 * \param baseNE  the baseline concentration of Noradrenaline 
	 * \param tauNE the decay time constant of Noradrenaline 
	 * \param configId (optional, deprecated) configuration id
	 * \note The method can be called at configuration state only
	 */
	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							float baseACh, float tauACh, float baseNE, float tauNE, int configId = ALL);

	// TODO: this should be implemented via default arguments as members of the class, so that the user can call
	// setDefaultNeuromodulators()
	//! \note The method can be called at configuration state only
	void setNeuromodulator(int grpId, float tauDP = 100.0f, float tau5HT = 100.0f,
							float tauACh = 100.0f, float tauNE = 100.0f, int configId = ALL);

	//! Sets default STDP mode and params
	//! \note The method can be called at configuration state only
	void setSTDP(int grpId, bool isSet, int configId=ALL);

	//! Sets STDP params for a group, custom
	//! \note The method can be called at configuration state only
	void setSTDP(int grpId, bool isSet, stdpType_t type, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD,
		int configId=ALL);

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
	 * \param[in] grpId       pre-synaptic group id. STP will apply to all neurons of that group!
	 * \param[in] isSet       a flag whether to enable/disable STP
	 * \param[in] STP_U       increment of u induced by a spike
	 * \param[in] STP_tau_u   decay constant of u (\tau_F)
	 * \param[in] STP_tau_x   decay constant of x (\tau_D)
	 * \param[in] configId    configuration ID of group
	 * \note The method can be called at configuration state only
	 */
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x, int configId=ALL);

	//! Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically) using default values
	//! \note The method can be called at configuration state only
	void setSTP(int grpId, bool isSet, int configId=ALL);

	//! Sets the weight and weight change update parameters
	/*!
	 * \param[in] updateWeightInterval the interval between two weight update.
	 * \param[in] updateWeightChangeInterval the interval between two weight update.
	 * \param[in] tauWeightChange the decay time constant of weight change (wtChange)
	 * \note The method can be called at configuration state only
	 */
	void setWeightAndWeightChangeUpdate(updateInterval_t updateWeightInterval = INTERVAL_1000MS,
		updateInterval_t updateWeightChangeInterval = INTERVAL_1000MS, int tauWeightChange = 10);


	// +++++ PUBLIC METHODS: RUNNING A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for time=(nSec*seconds + nMsec*milliseconds)
	 * \param[in] nSec 			number of seconds to run the network
	 * \param[in] nMsec 		number of milliseconds to run the network
	 * \param[in] copyState 	enable copying of data from device to host
	 * \note The method can be called at setup state and execution state. If Carlsim is at setup state, it will
	 * change to execution state
	 */
	int runNetwork(int nSec, int nMsec, bool copyState=false);

	/*!
	 * \brief build the network 
	 * \param[in] removeTempMemory 	remove temp memory after building network
	 * \note The method can be called at configuration state only. Carlsim will change to setup state
	 */
	void setupNetwork(bool removeTempMemory = true);

	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! saves important simulation and network infos to file
	//! \note The method can be called at setup state and execution state
	void saveSimulation(std::string fileName, bool saveSynapseInfo=true);

	// FIXME: needs overhaul
	//! Sets update cycle for log messages
	/*!
	 * \brief Sets update cycle for printing the network status (seconds)
	 *
	 * Network status includes includes spiking and plasticity information (SpikeMonitor updates, weight changes, etc.).
	 * Set cycle to -1 to disable.
	 * \param[in] showStatusCycle how often to print network state (seconds)
	 */
	void setLogCycle(int showStatusCycle);

	/*!
	 * \brief Sets the file pointer of the debug log file
	 * \param[in] fpLog file pointer to new log file
	 */
	void setLogDebugFp(FILE* fpLog);

	/*!
	 * \brief Sets the file pointers for all log files
	 * \param[in] fpOut file pointer for status info
	 * \param[in] fpErr file pointer for errors/warnings
	 * \param[in] fpDeb file pointer for debug info
	 * \param[in] fpLog file pointer for debug log file that contains all the above info
	 */
	void setLogsFp(FILE* fpOut, FILE* fpErr=NULL, FILE* fpDeb=NULL, FILE* fpLog=NULL);



	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! reads the network state from file
	//! \note The method can be called at configuration state only
	void readNetwork(FILE* fid);

	/*!
	 * \brief Reassigns fixed weights to values passed into the function in a single 1D float matrix (weightMatrix)
	 *
	 * The user passes the connection ID (connectID), the weightMatrix, the matrixSize, and 
	 * configuration ID (configID).  This function only works for fixed synapses and for connections of type
	 * CONN_USER_DEFINED. Only the weights are changed, not the maxWts, delays, or connected values
	 * \note The method can be called at setup state only
	 */
	void reassignFixedWeights(short int connectId, float weightMatrix[], int matrixSize, int configId=ALL);

	// Deprecated
	void resetSpikeCntUtil(int grpId=ALL); //!< resets spike count for particular neuron group

	/*!
	 * \brief reset Spike Counter to zero
	 *
	 * Manually resets the spike buffers of a Spike Counter to zero (for a specific group).
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time.
	 * \param grpId the group for which to reset the spike counts. Set to ALL if you want to reset all Spike Counters.
	 * \param configId the config id for which to reset the spike counts. Set to ALL if you want to reset all configIds
	 * \note The method can be called at execution state only
	 */
	void resetSpikeCounter(int grpId, int configId=ALL);

	/*!
	 * \brief Sets a connection monitor for a group, custom ConnectionMonitor class
	 *
	 * To retrieve connection status, a connection-monitoring callback mechanism is used. This mechanism allows the user
	 * to monitor connection status between groups. Connection monitors are registered for two groups (i.e., pre- and
	 * post- synaptic groups) and are called automatically by the simulator every second.
	 *
	 * Use setConnectionMonitor(grpIdPre,grpIdPost) to use a ConnectionMonitor with default settings.
	 *
	 * \param[in] grpIdPre 		the pre-synaptic group ID
	 * \param[in] grpIdPost 	the post-synaptic group ID
	 * \param[in] connectionMon an instance of class ConnectionMonitor (see callback.h)
	 * \note The method can be called at configuration state and setup state
	 */
	void setConnectionMonitor(int grpIdPre, int grpIdPost, ConnectionMonitor* connectionMon=NULL, int configId=ALL);

	/*!
	 * \brief Sets a group monitor for a group, custom GroupMonitor class
	 * \note The method can be called at configuration state and setup state
	 */
	void setGroupMonitor(int grpId, GroupMonitor* groupMon=NULL, int configId=ALL);

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
	 * \param grpId the group for which you want to enable a SpikeCounter
	 * \param recordDur number of ms for which to record spike numbers. Spike numbers will be reset to zero after
	 * this. Set frameDur to -1 to never reset spike counts. Default: -1.
	 * \note The method can be called at configuration state and setup state
	 */
	void setSpikeCounter(int grpId, int recordDur=-1, int configId=ALL);

	//! Sets up a spike generator
	//! \note The method can be called at configuration state and setup state
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId=ALL);

	/*!
	 * \brief Sets a spike monitor for a group, custom SpikeMonitor class
	 *
	 * You can either write your own class that derives from SpikeMonitor, and directly access the neuron IDs and
	 * spike times in 1000 ms bins, or you can set spikeMon=NULL, in which case the spike counts will simply be
	 * output to console every 1000 ms.
	 * If you want to dump spiking information to file, use the other SpikeMonitor.
	 * If you need spiking information in smaller bins, use a SpikeCounter.
	 * \note The method can be called at configuration state and setup state
	 */
	void setSpikeMonitor(int gid, SpikeMonitor* spikeMon=NULL, int configId=ALL);

	//! Sets a spike monitor for a group, prints spikes to binary file
	//! \note The method can be called at configuration state and setup state
	void setSpikeMonitor(int grpId, const std::string& fname, int configId=0);

	//! \note The method can be called at setup state and execution state
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1, int configId=ALL);

	//! Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	//! weight values back to their default values by setting resetWeights = true.
	//! \note The method can be called at execution state only
	// Deprecated
	void updateNetwork(bool resetFiringInfo, bool resetWeights);

	//! function writes population weights from gIDpre to gIDpost to file fname in binary.
	//! \note The method can be called at setup state and execution state
	void writePopWeights(std::string fname, int gIDpre, int gIDpost, int configId=0);



	// +++++ PUBLIC METHODS: GETTER / SETTERS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	carlsimState_t getCarlsimState() { return carlsimState_; }
// FIXME
//	grpConnectInfo_t* getConnectInfo(short int connectId, int configId=0); //!< gets connection info struct
	//! \note The method can be called at setup state and execution state
	int  getConnectionId(short int connId, int configId);

	//! \note The method can be called at setup state and execution state
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays=NULL);

	//! \note The method can be called at setup state and execution state
	int getGroupId(int grpId, int configId=0);
	//group_info_t getGroupInfo(int grpId, int configId=0); //!< gets group info struct
	std::string getGroupName(int grpId, int configId=0);

	//! \note The method can be called at setup state and execution state
	int getNumConfigurations();							//!< gets number of network configurations
	int getNumConnections(short int connectionId);		//!< gets number of connections associated with a connection ID
	int getNumGroups();									//!< gets number of groups in the network
	int getNumNeurons(); //!< returns the total number of allocated neurons in the network
	int getNumPreSynapses(); //!< returns the total number of allocated pre-synaptic connections in the network
	int getNumPostSynapses(); //!< returns the total number of allocated post-synaptic connections in the network

	int getGroupStartNeuronId(int grpId); //!< get the first neuron id of a groupd specified by grpId
	int getGroupEndNeuronId(int grpId); //!< get the last neuron id of a groupd specified by grpId
	int getGroupNumNeurons(int grpId); //!< get the number of neurons of a groupd specified by grpId

	/*!
	 * \brief Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
	 *
	 * and the size of the 1D array in size.  gIDpre(post) is the group ID for the pre(post)synaptic group, 
	 * weights is a pointer to a single dimensional array of floats, size is the size of that array which is 
	 * returned to the user, and configID is the configuration ID of the SNN.  NOTE: user must free memory from
	 * weights to avoid a memory leak.
	 * \note The method can be called at setup state and execution state
	 */
	// why not readPopWeight()?
	void getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId=0);

	uint64_t getSimTime();
	uint32_t getSimTimeSec();
	uint32_t getSimTimeMsec();

	//! Returns pointer to 1D array of the number of spikes every neuron in the group has fired
	//! \note The method can be called at execution state, Deprecated
	unsigned int* getSpikeCntPtr(int grpId);

	/*!
	 * \brief return the number of spikes per neuron for a certain group
	 *
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur) at any point in time.
	 * \param[in] grpId	   the group for which you want the spikes (cannot be ALL)
	 * \param[in] configId the configuration ID (cannot be ALL)
	 * \returns pointer to array of ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (int) since the last reset.
	 * \note The method can be called at execution state
	 */
	int* getSpikeCounter(int grpId, int configId=0);

	// FIXME: fix this
	// TODO: maybe consider renaming getPopWeightChanges
	//! \note The method can be called at execution state, Deprecated
	float* getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges=NULL);

	//! \note The method can be called at setup state and execution state
	bool isExcitatoryGroup(int grpId);
	bool isInhibitoryGroup(int grpId);
	bool isPoissonGroup(int grpId);

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 *
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 * \note The method can be called at execution state only
	 */
	//Deprecated
	void setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr);


	// +++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Sets default values for conductance time constants
	 *
	 * \param[in] tdAMPA   time constant for AMPA decay (ms)
	 * \param[in] trNMDA   time constant for NMDA rise (ms)
	 * \param[in] tdNMDA   time constant for NMDA decay (ms)
	 * \param[in] tdGABAa  time constant for GABAa decay (ms)
	 * \param[in] trGABAb  time constant for GABAb rise (ms)
	 * \param[in] tdGABAb  time constant for GABAb decay (ms)
	 * \note The method can be called at configuration state only
	 */
	void setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);

	//! sets default homeostasis params
	//! \note The method can be called at configuration state only
	void setDefaultHomeostasisParams(float homeoScale, float avgTimeScale);

	//! sets default options for save file
	//! \note The method can be called at configuration state only
	void setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo);

	//! sets default values for STDP params
	//! \note The method can be called at configuration state only
	void setDefaultSTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD);

	//! sets default values for STP params (neurType either EXCITATORY_NEURON or INHIBITORY_NEURON)
	//! \note The method can be called at configuration state only
	void setDefaultSTPparams(int neurType, float STP_U, float STP_tau_U, float STP_tau_x);


private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	void CARLsimInit();					//!< init function, unsafe computations that would usually go in constructor

	bool existsGrpId(int grpId);		//!< checks whether a certain grpId exists in grpIds_

	void handleUserWarnings(); 			//!< print all user warnings, continue only after user input


	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	CpuSNN* snn_;					//!< an instance of CARLsim core class
	std::string netName_;			//!< network name
	int nConfig_;					//!< number of configurations
	int randSeed_;					//!< RNG seed
	simMode_t simMode_;				//!< CPU_MODE or GPU_MODE
	loggerMode_t loggerMode_;		//!< logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	int ithGPU_;					//!< on which device to establish a context
	bool enablePrint_;
	bool copyState_;

	unsigned int numConnections_;	//!< keep track of number of allocated connections
	std::vector<std::string> userWarnings_; // !< an accumulated list of user warnings

	std::vector<int> grpIds_;		//!< a list of all created group IDs

	bool hasSetHomeoALL_;			//!< informs that homeostasis have been set for ALL groups (can't add more groups)
	bool hasSetHomeoBaseFiringALL_;	//!< informs that base firing has been set for ALL groups (can't add more groups)
	bool hasSetSTDPALL_; 			//!< informs that STDP have been set for ALL groups (can't add more groups)
	bool hasSetSTPALL_; 			//!< informsthat STP have been set for ALL groups (can't add more groups)
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