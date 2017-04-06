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

/*
 * Common Abbreviations used in CARLsim
 * snn/SNN: spiking neural network
 * stp/STP: short-term plasticity
 * stdp/STDP: spike-timing dependent plasticity
 * syn/SYN: synapse
 * wt/WT: weight
 * exc/Exc: excitatory
 * inh/Inh: inhibitory
 * grp: group
 * nid/NId/NID: neuron id
 * gid/GId/GID: group id
 *
 * conn: connection
 * min: minimum
 * max: maximum
 * num: number
 * mem: memory
 * id/Id/ID: identification
 * info: information
 * cnt: count
 * curr: current
 * mon: monitor
 * reg/Reg: regurlar
 * pois/Pois: poisson
 */
#ifndef _SNN_GOLD_H_
#define _SNN_GOLD_H_

#include <map>
#include <list>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <climits>

#include <carlsim.h>
#include <callback_core.h>

#include <snn_definitions.h>
#include <snn_datastructures.h>

// #include <spike_buffer.h>
#include <poisson_rate.h>

class SpikeMonitor;
class SpikeMonitorCore;
class ConnectionMonitorCore;
class ConnectionMonitor;

class SpikeBuffer;


// **************************************************************************************************************** //
// CPUSNN CORE CLASS
// **************************************************************************************************************** //

/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
class SNN {

	// **************************************************************************************************************** //
	// PUBLIC METHODS
	// **************************************************************************************************************** //
public:
	/** SNN Constructor
	 * 
	 * \brief SNN Constructor
	 * \param name the symbolic name of a spiking neural network
	 * \param preferredSimMode preferred simulation platform (CPU/GPU/hybrid)
	 * \param loggerMode log mode to control verbosity
	 * \param randSeed randomize seed of the random number generator
	 */
	SNN(const std::string& name, SimMode preferredSimMode, LoggerMode loggerMode, int randSeed);

	//! SNN Destructor
	/*!
	 * \brief clean up all allocated resource
	 */
	~SNN();

	// +++++ PUBLIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const static unsigned int MAJOR_VERSION = 4; //!< major release version, as in CARLsim X
	const static unsigned int MINOR_VERSION = 0; //!< minor release version, as in CARLsim 2.X



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// NOTE: there should be no default argument values in here, this should be handled by the user interface

	/** Creates synaptic projections from a pre-synaptic group to a post-synaptic group using a pre-defined primitive
	 * type.
	 *
	 * \brief make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	 *
	 * \param grpIdPre ID of the pre-synaptic group
	 * \param grpIdPost ID of the post-synaptic group
	 * \param _type connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post (no self-connections).
	 * \param initWt initial weight strength (arbitrary units); should be negative for inhibitory connections
	 * \param maxWt upper bound on weight strength (arbitrary units); should be negative for inhibitory connections
	 * \param prob connection probability
	 * \param minDelay the minimum delay allowed (ms)
	 * \param maxdelay the maximum delay allowed (ms)
	 * \param radius A struct of type RadiusRF to specify the receptive field radius in 3 dimensions
	 * \param mulSynFast a multiplication factor to be applied to the fast synaptic current (AMPA in the case of excitatory, and GABAa in the case of inhibitory connections)
	 * \param mulSynSlow a multiplication factor to be applied to the slow synaptic current (NMDA in the case of excitatory, and GABAb in the case of inhibitory connections)
	 * \param synWtType connection type, either SYN_FIXED or SYN_PLASTIC
	 * \return number of created synaptic projections
	 */
	short int connect(int gIDpre, int gIDpost, const std::string& _type, float initWt, float maxWt, float prob,
		uint8_t minDelay, uint8_t maxDelay, RadiusRF radius,
		float mulSynFast, float mulSynSlow, bool synWtType);

	/** Creates manually defined synaptic projections using a callback mechanism to an instance of class ConnectionGenerator.
	 *
	 * \brief Creates synaptic projections from group gIdPre to group gIdPost using a callback mechanism.
	 *
	 * \param gIdPre ID of the pre-synaptic group
	 * \param gIdPost ID of the post-synaptic group
	 * \param conn pointer to an instance of class ConnectionGenerator
	 * \param mulSynFast a multiplication factor to be applied to the fast synaptic current (AMPA in the case of excitatory, and GABAa in the case of inhibitory connections)
	 * \param mulSynSlow a multiplication factor to be applied to the slow synaptic current (NMDA in the case of excitatory, and GABAb in the case of inhibitory connections)
	 * \param synWtType connection type, either SYN_FIXED or SYN_PLASTIC
	 * \return number of created synaptic projections
	 */
	short int connect(int gIDpre, int gIDpost, ConnectionGeneratorCore* conn, float mulSynFast, float mulSynSlow,
		bool synWtType);

	/** Creates a group of Izhikevich spiking neurons
	 *
	 * \brief Creates a group of Izhikevich spiking neurons and defines preferred runtime implementaion
	 *
	 * \param grpName The symbolic name of a group
	 * \param grid  Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param neurType The type of neuron, such as EXCITATORY_NEURON
	 * \param preferredPartition The preferred runtime partition for the group
	 * \param preferredBackend The prefrerred computing device (CPU_CORES or GPU_CORES) for this group
	 * \return The global group-ID of the neuron group
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend);

	/** Creates a spike generator group (dummy-neurons that can generate poisson spike-trains, not Izhikevich spiking neurons)
	 *
	 * \brief Creates a group of spike generators and sets preferred runtime implementaion parameters
	 *
	 * \param grpName the symbolic name of a group
	 * \param grid Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param neurType the type of neuron, currently only support EXCITATORY NEURON
	 * \param preferredPartition The preferred runtime partition for the group
	 * \param preferredBackend The prefrerred computing device (CPU_CORES or GPU_CORES) for this group
	 * \return The global group-ID of the spike generator group
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend);


	/** Sets custom values for conductance decay (\f$tau_{deacy}\f$) or disables conductances alltogether
	 *  These will be applied to all connections in a network.
	 *  For details on the ODE that is implemented refer to (Izhikevich et al, 2004), and for suitable values see (Dayan & Abbott, 2001).
	 *
	 * \brief Sets custom values for conduction rise and decay times or disables COBA alltogether
	 *
	 * \param isSet enables the use of COBA mode
	 * \param tdAMPA decay time constant of AMPA (ms)
	 * \param trNMDA rise time constant of NMDA (ms)
	 * \param tdNMDA deacy time constant of NMDA (ms)
	 * \param tdGABAa deacy time constant of GABAa (ms)
	 * \param trGABAb rise time constant of GABAb (ms)
	 * \param tdGABAb decay time constant of GABAb (ms)
	 * \return void
	 */
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);


	/**
	 * Sets the homeostasis parameters.
	 * homeostasisScale is strength of
	 * homeostasis compared to the strength of normal LTP/LTD from STDP (which is 1), and avgTimeScale is the time
	 * frame over which the average firing rate is averaged (it should be larger in scale than STDP timescales).
	 *
	 * \brief Sets custom values for implementation of homeostatic synaptic scaling
	 *
	 * \param grpId        the group ID of group to which homeostasis is applied
	 * \param isSet        a boolean, setting it to true/false enables/disables homeostasis
	 * \param homeoScale   scaling factor multiplied to weight change due to homeostasis
	 * \param avgTimeScale time in seconds over which average firing rate for neurons in this group is
	 *                         averaged
	 * \return void
	 */
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale);

	/** Sets homeostatic target firing rate (enforced through homeostatic synaptic scaling). For more information on this implementation
	 *  please see: Carlson, et al. (2013). Proc. of IJCNN 2013.
	 *
	 * \brief Sets the homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	 *
	 * \param grpId        the ID of group to which homeostasis is applied
	 * \param baseFiring target firing rate of every neuron in this group
	 * \param baseFiringSD standard deviation of target firing rate of every neuron in this group
	 * \return void
	 */
	void setHomeoBaseFiringRate(int groupId, float baseFiring, float baseFiringSD);


	/** Sets the Izhikevich parameters a, b, c, and d of a neuron group. Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d and standard deviation _a_sd, _b_sd, _c_sd, and _d_sd, respectively.
	 *
	 * \brief Sets the Izhikevich parameters a, b, c, and d of a neuron group with mean +- standard deviation.
	 *
	 * \param grpId the ID of the group whose izhikevich neuron parameters are set
	 * \param izh_a  the mean value of izhikevich parameter a
	 * \param izh_a_sd the standard deviation value of izhikevich parameter a
	 * \param izh_b  the mean value of izhikevich parameter b
	 * \param izh_b_sd the standard deviation value of izhikevich parameter b
	 * \param izh_c  the mean value of izhikevich parameter c
	 * \param izh_c_sd the standard deviation value of izhikevich parameter c
	 * \param izh_d  the mean value of izhikevich parameter d
	 * \param izh_d_sd the standard deviation value of izhikevich parameter d
	 * \return void
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd);

	/** Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron group.
	 *
	 * \brief Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron group.
	 *
	 * \param grpId the symbolic name of a group
	 * \param baseDP  the baseline concentration of Dopamine
	 * \param tauDP the decay time constant of Dopamine
	 * \param base5HT  the baseline concentration of Serotonin
	 * \param tau5HT the decay time constant of Serotonin
	 * \param baseACh  the baseline concentration of Acetylcholine
	 * \param tauACh the decay time constant of Acetylcholine
	 * \param baseNE  the baseline concentration of Noradrenaline
	 * \param tauNE the decay time constant of Noradrenaline
	 * \return void
	 */
	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
		float baseACh, float tauACh, float baseNE, float tauNE);

	/** Set the spike-timing-dependent plasticity (STDP) parameters for a post-synaptic neuron group on excitatory synapses.
	 * STDP must be defined post-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1,
	 * call setSTP on group 1. Fore details on the phenomeon, see (for example) (Bi and Poo, 2001).
	 *
	 * \brief Set the spike-timing-dependent plasticity (STDP) parameters for a post-synaptic neuron group.
	 *
	 * \param grpId ID of the post-synaptic neuron group
	 * \param isSet set to true to enable STDP for this group
	 * \param type STDP type (STANDARD, DA_MOD)
	 * \param curve A STDPCurve enumeration value defining the curve of ESTDP, such as EXP_CURVE and TIMING_BASED_CURVE
	 * \param alphaPlus max magnitude for LTP change
	 * \param tauPlus decay time constant for LTP
	 * \param alphaMinus max magnitude for LTD change (leave positive)
	 * \param tauMinus decay time constant for LTD
	 * \param gamma parameter for timing-based STDP in exciatory synapses to decide between LTP and LTD; STDP parameters KAPPA and OMEGA are derived from this
	 * \return void
	 */
	void setESTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma);

	/** Set the inhibitory spike-timing-dependent plasticity (STDP) with anti-hebbian curve for a post-synaptic neuron group on inhibitory synapses.
	 *  Set the inhibitory spike-timing-dependent plasticity (STDP) with anti-hebbian curve for a post-synaptic neuron group. STDP must be defined post-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1,
	 *  call setSTP on group 1. Fore details on the phenomeon, see (for example) (Bi and Poo, 2001).
	 * 
	 * \brief Set the inhibitory spike-timing-dependent plasticity (STDP) with anti-hebbian curve for a post-synaptic neuron group on inhibitory synapses.
	 *
	 * \param grpId ID of the post-synaptic neuron group
	 * \param isSet set to true to enable STDP for this group
	 * \param type STDP type (STANDARD, DA_MOD)
	 * \param curve A STDPCurve enumeration value defining the curve of ISTDP, such as EXP_CURVE and PULSE_CURVE
	 * \param ab1 magnitude for LTP change
	 * \param ab2 magnitude for LTD change (leave positive)
	 * \param tau1 the interval for LTP
	 * \param tau2 the interval for LTD
	 * \return void
	 */
	void setISTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float ab1, float ab2, float tau1, float tau2);

	/** CARLsim implements the short-term plasticity model of (Tsodyks & Markram, 1998; Mongillo, Barak, & Tsodyks, 2008).\n
	 * \f$du/dt = -u/STP_{tau_u} + STP_U * (1-u-) * \delta(t-t_{spk})\f$\n
	 * \f$dx/dt = (1-x)/STP_{tau_x} - u+ * x- * \delta(t-t_{spk})\f$\n
	 * \f$dI/dt = -I/tau_S + A * u+ * x- * \delta(t-t_{spk})\f$\n
	 * where u- means value of variable u right before spike update, and x+ means value of variable x right after
	 * the spike update, and A is the synaptic weight.
	 * The STD effect is modeled by a normalized variable (0<=x<=1), denoting the fraction of resources that remain
	 * available after neurotransmitter depletion.
	 * The STF effect is modeled by a utilization parameter u, representing the fraction of available resources ready for
	 * use (release probability). Following a spike, (i) u increases due to spike-induced calcium influx to the
	 * presynaptic terminal, after which (ii) a fraction u of available resources is consumed to produce the post-synaptic
	 * current. Between spikes, u decays back to zero with time constant \f$STP_{tau_u} (\tau_F)\f$, and x recovers to value one
	 * with time constant \f$STP_{tau_x} (\tau_D)\f$.
	 *
	 * \brief Sets STP params \f$U\f$, \f$tau_u\f$, and \f$tau_x\f$ of a neuron group (pre-synaptically)
	 *
	 * \param grpId       pre-synaptic group id. STP will apply to all neurons of that group!
	 * \param isSet       a flag whether to enable/disable STP
	 * \param STP_U 	  increment of u induced by a spike
	 * \param STP_tau_u   decay constant of \f$u (\tau_F)\f$
	 * \param STP_tau_x   decay constant of \f$x (\tau_D)\f$
	 * \return void
	 * \note STP will be applied to all outgoing synapses of all neurons in this group.
	 * \note All outgoing synapses of a certain (pre-synaptic) neuron share the resources of that same neuron.
	 */
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x);

	/** Sets the STDP weight and weight change update interval and scale parameters. Also sets up stdp scale factor according to update interval.
	 * \brief Sets the STDP weight and weight change update interval and scale parameters.
	 *
	 * \param wtANDwtChangeUpdateInterval the interval between two wt (weight) and wtChange (weight change) update.
	 * \param enableWtChangeDecay enable weight change decay
	 * \param wtChangeDecay the decay ratio of weight change (wtChange)
	 */
	void setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay);

	// +++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/** Run the simulation for timesteps = (_nsec*1000 + _nmsec). Each timestep is 1 ms.
	 * \brief run the simulation for timesteps = (_nsec*1000 + _nmsec)
	 *
	 * \param _nsec 		  number of seconds to run the network
	 * \param _nmsec 		  number of milliseconds to run the network
	 * \param printRunSummary whether to print a basic summary of the run at the end
	 * \return 0 on successfull completion of simulation
	 */
	int runNetwork(int _nsec, int _nmsec, bool printRunSummary);

	/** Builds the network. Will make CARLsim state switch from ::CONFIG_STATE to ::SETUP_STATE. reorganize the network and do the necessary allocation
	 *  of all variable for carrying out the simulation. This code is run only one time during network initialization.
	 * \brief Builds the network. Will make CARLsim state switch from ::CONFIG_STATE to ::SETUP_STATE.
	 */
	void setupNetwork();

	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/** This method adds a constant bias to the weight of every synapse in the connection specified by connId. The bias
	 * can be positive or negative.
	 * If a bias is specified that makes any weight+bias lie outside the range [minWt,maxWt] of this connection, the
	 * range will be updated accordingly if the flag updateWeightRange is set to true.
	 * If the flag is set to false, then the specified weight value will be corrected to lie on the boundary (either
	 * minWt or maxWt).
	 *
	 * \brief Adds a constant bias to the weight of every synapse in the connection
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param connId            the connection ID to manipulate
	 * \param bias              the bias value to add to every synapse
	 * \param updateWeightRange a flag specifying what to do when the specified weight+bias lies outside the range
	 *                              [minWt,maxWt]. Set to true to update the range accordingly. Set to false to adjust
	 *                              the weight to be either minWt or maxWt. Default: false.
	 *
	 * \see setWeight
	 * \see scaleWeights
	 * \note A weight cannot drop below zero, no matter what.
	 */
	void biasWeights(short int connId, float bias, bool updateWeightRange = false);

	/** Deallocates all dynamical structures on both CPU and GPU sides and exits. Deallocates monitors
	 *  , config data, and runtime data. Closes all file streams other than stderr and stdout.
	 * 
	 * \brief Deallocates all dynamical structures on both CPU and GPU sides and exits. 
	 * \param val Exit status
	 * \return void
	 */
	void exitSimulation(int val = 1);

	/** Reads a CARLsim network state previously saved in a file. Such a file can be created using SNN::saveSimulation().
	 *  After calling SNN::loadSimulation, you should run SNN::runNetwork before calling fclose(fp).
	 *
	 * \brief Reads a CARLsim network state previously saved in a file
	 * \param fid: file pointer to handle the file where the network state was previously saved using using SNN::saveSimulation()
	 * \see SNN::saveSimulation()
	 */
	void loadSimulation(FILE* fid);

	/** This method scales the weight of every synapse in the connection specified by connId with a scaling factor.
	 *  The scaling factor cannot be negative.
	 *  If a scaling factor is specified that makes any weight*scale lie outside the range [minWt,maxWt] of this
	 *  connection, the range will be updated accordingly if the flag updateWeightRange is set to true.
	 *  If the flag is set to false, then the specified weight value will be corrected to lie on the boundary (either
	 *  minWt or maxWt).
	 *
	 * \brief Multiplies the weight of every synapse in the connection with a scaling factor
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param connId            the connection ID to manipulate
	 * \param scale             the scaling factor to apply to every synapse (cannot be negative)
	 * \param updateWeightRange a flag specifying what to do when the specified weight*scale lies outside the range
	 *                              [minWt,maxWt]. Set to true to update the range accordingly. Set to false to adjust
	 *                              the weight to be either minWt or maxWt. Default: false.
	 *
	 * \note A weight cannot drop below zero, no matter what.
	 * \see setWeight
	 * \see biasWeights
	 */
	void scaleWeights(short int connId, float scale, bool updateWeightRange = false);

	/** Sets up a group monitor registered with a callback to process the spikes.
	 *  The method first checks if the group already has a monitor and if it does not, then
	 *  creates a new GroupMonitorCore object and initialize analysis components. If file pointer exists, it has already been fopened
	 *  this will also write the header section of the group status file. It also 
	 *  creates a new GroupMonitor object for the user-interface, which later will be deallocated by
	 *  SNN::deleteObjects.
	 *
	 * \param grpId ID of the neuron group
	 * \param fid file pointer for recording group status (neuromodulators)
	 * \return A GroupMonitor object for the group
	 */
	GroupMonitor* setGroupMonitor(int grpId, FILE* fid);

	/** To retrieve connection status, a connection-monitoring callback mechanism is used. This mechanism allows the user
	 *  to monitor connection status between groups. Connection monitors are registered for two groups (i.e., pre- and
	 *  post- synaptic groups) and are called automatically by the simulator every second.
	 *
	 *  CARLsim supports two different recording mechanisms: Recording to a weight file (binary) and recording to a
	 *  ConnectionMonitor object. The former is useful for off-line analysis of synaptic weights (e.g., using
	 *  \ref ch9_matlab_oat).
	 *  The latter is useful to calculate different weight metrics and statistics on-line, such as the percentage of
	 *  weight values that fall in a certain weight range, or the number of weights that have been changed since the
	 *  last snapshot.
	 *
	 *  The function returns a pointer to a ConnectionMonitor object, which can be used to calculate weight changes
	 *  and other connection stats.
	 *  
	 * \brief Sets a connection monitor for connections between two groups, custom ConnectionMonitor class
	 *
	 * \param grpIdPre 		the pre-synaptic group ID
	 * \param grpIdPost 	the post-synaptic group ID
	 * \param fname         file pointer of the binary to be created
	 * \return a new ConnectionMonitor object for the user-interface
	 */

	ConnectionMonitor* setConnectionMonitor(int grpIdPre, int grpIdPost, FILE* fid);

	/** This method injects current, specified on a per-neuron basis, into the soma of each neuron in the group, at
	 * each timestep of the simulation. current is a float vector of current amounts (mA), one element per neuron in
	 * the group.
	 *
	 * To input different currents into a neuron over time, the idea is to run short periods of runNetwork and
	 * subsequently calling setExternalCurrent again with updated current values.
	 *
	 * \brief Sets the amount of current (mA) to inject into a group
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId    the group ID
	 * \param current  a float vector of current amounts (mA), one value per neuron in the group
	 *
	 * \note This method cannot be applied to SpikeGenerator groups.
	 *
	 * \see setSpikeRate
	 * \see setSpikeGenerator
	 */
	void setExternalCurrent(int grpId, const std::vector<float>& current);

	/** A custom SpikeGenerator object can be used to allow for more fine-grained control overs spike generation by
	 * specifying individual spike times for each neuron in a group.
	 *
	 * In order to specify spike times, a new class must be defined first that derives from the SpikeGenerator class
	 * and implements the virtual method SpikeGenerator::nextSpikeTime.
	 * Then, in order for a custom SpikeGenerator to be associated with a SpikeGenerator group,
	 *
	 * \brief Associates a SpikeGenerator object with a group for fine grained specificaion of spike times for individual neurons
	 *
	 * \param grpId           the group with which to associate a SpikeGenerator object
	 * \param spikeGenFunc pointer to a custom SpikeGenerator object
	 */
	void setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGenFunc);

	/** To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
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
	 * The function returns a pointer to a SpikeMonitor object, which can be used to calculate spike statistics (such
	 * group firing rate, number of silent neurons, etc.) or retrieve all spikes from a particular time window.
	 *
	 * If you call setSpikeMonitor twice on the same group, the same SpikeMonitor pointer will be returned, and the
	 * name of the spike file will be updated. This is the same as calling SpikeMonitor::setLogFile directly, and
	 * allows you to redirect the spike file stream mid-simulation (see \ref ch7s1s3_redirecting_file_streams).
	 *
	 * \brief Sets a Spike Monitor for a groups, prints spikes to binary file
	 *
	 * \param gid 		the group ID
	 * \param fid 		file pointer to the binary to be created
	 * \return   SpikeMonitor*	pointer to a SpikeMonitor object, which can be used to calculate spike statistics
	 *                          (such as group firing rate, number of silent neurons, etc.) or retrieve all spikes in
	 * 							AER format
	 *
	 * \note Only one SpikeMonitor is allowed per group.
	 */
	SpikeMonitor* setSpikeMonitor(int gid, FILE* fid);

	/** Sets firing rate and refractory period for a SpikeGenerator group
	 *
	 * \brief Sets a spike rate of a SpikeGenerator group
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId      SpikeGenerator group ID
	 * \param spikeRate  pointer to PoissonRate object
	 * \param refPeriod  refactory period (ms)
	 *
	 * \note If you allocate the PoissonRate object on the heap, you are responsible for correctly deallocating it.
	 * \see setSpikeGenerator
	 */
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod);

	/** This method sets the weight value of the synapse that belongs to connection connId and connects pre-synaptic
	 * neuron neurIdPre to post-synaptic neuron neurIdPost. Neuron IDs should be zero-indexed, so that the first
	 * neuron in the group has ID 0.
	 *
	 * If a weight value is specified that lies outside the range [minWt,maxWt] of this connection, the range will be
	 * updated accordingly if the flag updateWeightRange is set to true. If the flag is set to false, then the
	 * specified weight value will be corrected to lie on the boundary (either minWt or maxWt).
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param connId            the connection ID to manipulate
	 * \param neurIdPre         pre-synaptic neuron ID (zero-indexed)
	 * \param neurIdPost        post-synaptic neuron ID (zero-indexed)
	 * \param weight            the weight value to set for this synapse
	 * \param updateWeightRange a flag specifying what to do when the specified weight lies outside the range
	 *                              [minWt,maxWt].
	 */
	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange = false);

	/** This function can be used to temporarily disable all weight updates (such as from STDP or homeostasis)
	 *  in the network. An optional parameter specifies whether the accumulated weight updates so far should be applied to the weights
	 *  before entering the testing phase.
	 *
	 *  \STATE ::SETUP_STATE, ::RUN_STATE
	 *
	 * \param shallUpdateWeights   whether to apply the accumulated weight changes before entering the testing phase
	 * \note Calling this function on a simulation with no plastic synapses will have no effect.
	 */
	void startTesting(bool shallUpdateWeights = true);

	/** Exits a testing phase, making weight updates possible again
	 * \brief Exits a testing phase, making weight updates possible again
	 */
	void stopTesting();

	/** Polls the specified ConnectionMonitor to record the connection weights it is monitoring
	 * \brief Polls ConnectionMonitor to record connection weights
	 * 
	 * \param connId ConnectionMonitor ID
	 */
	void updateConnectionMonitor(short int connId = ALL);

	/** Upadates the group status to the groupMonitor object from the runtime buffer since the last
	 * update time. Usually done in every second, but not mandatory. Currently the concentration of neuromodulator.
	 *
	 * \brief Upadates the group status to the groupMonitor object 
	 * \param grpId Group ID for which status update is sought
	 *
	 */
	void updateGroupMonitor(int grpId = ALL);

	/** This function is public in SNN, but it should probably not be a public user function in CARLsim.
	 * It is usually called once every 1000ms by the core to update spike binaries and SpikeMonitor objects. In GPU
	 * mode, it will first copy the firing info to the host. The input argument can either be a specific group ID or
	 * keyword ALL (for all groups).
	 * Core and utility functions can call updateSpikeMonitor at any point in time. The function will automatically
	 * determine the last time it was called, and update SpikeMonitor information only if necessary.
	 *
	 * \brief copy required spikes from firing buffer to spike buffer
	 * \param grpId Group ID for which firing statistic update is sought
	 */
	void updateSpikeMonitor(int grpId = ALL);

	/** The network state consists of all
	 * the synaptic connections, weights, delays, and whether the connections are plastic or fixed. As an
	 * option, the user can choose whether or not to save the synaptic weight information (which could be
	 * a large amount of data) with the saveSynapseInfo argument.
	 *
	 * \brief Saves network configuration and simulation information to a file
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param fid          file pointer to use for saving simulation data.
	 * \param saveSynapseInfo   boolean value that determines if the weight values are written to
	 *                              the data file or not. The weight values are written if the boolean value is true.
	 * \see SNN::loadSimulation
	 */
	void saveSimulation(FILE* fid, bool saveSynapseInfo = false);

	//! function writes population weights from gIDpre to gIDpost to file fname in binary.
	//void writePopWeights(std::string fname, int gIDpre, int gIDpost);


	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/** \brief returns file pointer to info log*/
	const FILE* getLogFpInf() { return fpInf_; }
	/** returns file pointer to error log \brief returns file pointer to error log*/
	const FILE* getLogFpErr() { return fpErr_; }
	/** returns file pointer to debug log \brief returns file pointer to debug log*/
	const FILE* getLogFpDeb() { return fpDeb_; }
	/** returns file pointer to log file \brief returns file pointer to log file*/
	const FILE* getLogFpLog() { return fpLog_; }

	/** Sets the file pointers for all log files. File pointer NULL means don't change it.
	 *
	 * \brief Sets the file pointers for all log files
	 * \param fpInf file pointer to info log
	 * \param fpErr file pointer to error log
	 * \param fpDeb file pointer to debug log
	 * \param fpLog file pointer to Log file  
	 */
	void setLogsFp(FILE* fpInf = NULL, FILE* fpErr = NULL, FILE* fpDeb = NULL, FILE* fpLog = NULL);


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/** find connection ID based on pre-post group pair, complexity O(N)
	 *
	 * \brief find connection ID based on pre-post group pair
	 * \param grpIdPre pre-synatic group ID
	 * \param grpIdPost  post-synaptic group ID
	 * \return connection ID between pre-post groups
	 */
	short int getConnectId(int grpIdPre, int grpIdPost);

	/** Returns connection configuration data for a specified connection. Required for homeostasis.
	 *
	 * \brief Returns connection configuration data for a specified connection
	 * \param connectId connection ID
	 * \return ConnectConfigMap entry for the connection
	 */
	ConnectConfig getConnectConfig(short int connectId);

	/** Returns the RangeDelay struct of a connection\brief returns the RangeDelay struct of a connection \param connId Connction ID*/
	RangeDelay getDelayRange(short int connId);

	/** Returns the delay information for all synaptic connections between a pre-synaptic and a post-synaptic neuron group
	 *
	 * \brief Returns the delay for all synaptic connections between pre and post
	 * \param gGrpIdPre ID of pre-synaptic group
	 * \param gGrpIdPost ID of post-synaptic group
	 * \param numPreN the number of pre-synaptic neurons
	 * \param numPostN the number of post-synaptic neurons
	 * 
	 * \return delay information for all synapses
	 */
	uint8_t* getDelays(int gGrpIdPre, int gGrpIdPost, int& numPreN, int& numPostN);


	/** This function returns the Grid3D struct of a particular neuron group.
	 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid. This simplifies
	 * the creation of topographic connections in the network. The dimensions of the grid can thus be retrieved by
	 * calling Grid3D.width, Grid3D.height, and Grid3D.depth. The total number of neurons is given by Grid3D.N.
	 * See createGroup and Grid3D for more information.
	 *
	 * \brief returns the 3D grid struct of a group
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId the group ID for which to get the Grid3D struct
	 * \return the 3D grid struct of a group
	 */
	Grid3D getGroupGrid3D(int grpId);

	/** Finds the integer ID of the group with name given as a string
	 *
	 * \brief finds the ID of the group with name grpName
	 * \param grpName Name of the group as a string
	 * \return integer ID of the group
	 * 
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	int getGroupId(std::string grpName);

	/** Finds the name of the group given its integer group ID 
	 * \brief gets group name from ID
	 * \param integer ID of the group
	 * \return grpName Name of the group as a string
	 *
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 */
	std::string getGroupName(int grpId);

	/** This function returns the current STDP setting of a group.
	 *
	 * \brief Returns the stdp information of a group specified by grpId
	 * \param integer ID of the group
	 * \return GroupSTDPInfo struct for the group
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \see GroupSTDPInfo
	 */
	GroupSTDPInfo getGroupSTDPInfo(int grpId);

	/** This function returns the current setting for neuromodulators.
	 * \brief returns the neuromodulator information of a group specified by grpId
	 * \param integer ID of the group
	 * \return GroupNeuromodulatorInfo struct for the group
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \sa GroupNeuromodulatorInfo
	 */
	GroupNeuromodulatorInfo getGroupNeuromodulatorInfo(int grpId);


	LoggerMode getLoggerMode() { return loggerMode_; }

	// get functions for GroupInfo
	/** Returns the neuron ID of the first neuron in the given group \brief Returns the neuron ID of the first neuron in the given group*/
	int getGroupStartNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gStartN; }
	/** Returns the neuron ID of the last neuron in the given group \brief Returns the neuron ID of the last neuron in the given group*/
	int getGroupEndNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gEndN; }
	/** Returns number of neurons in the given group \brief Returns number of neurons in the given group*/
	int getGroupNumNeurons(int grpId) { return groupConfigMap[grpId].numN; }
	/** Returns the name of the CARLsim network \brief Returns the name of the CARLsim network*/
	std::string getNetworkName() { return networkName_; }

	/** This function returns the (x,y,z) location that a neuron with ID neurID (global) codes for.
	 * Note that neurID is global; that is, the first neuron in the group does not necessarily have ID 0.
	 *
	 * The location is determined by the actual neuron ID (the first neuron in the group coding for the origin (0,0,0),
	 * and by the dimensions of the 3D grid the group is allocated on (integer coordinates). Neuron numbers are
	 * assigned to location in order; where the first dimension specifies the width, the second dimension is height,
	 * and the third dimension is depth.
	 *
	 * For more information see SNN::createGroup() and the Grid3D struct.
	 *
	 * \brief returns the 3D location a neuron codes for
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::RUN_STATE
	 * \param neurId the neuron ID for which the 3D location should be returned
	 * \return the 3D location a neuron codes for as a Point3D struct
	 * \see getNeuronLocation3D(int grpId, int relNeurId)
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
	 * For more information see SNN::createGroup() and the Grid3D struct.
	 *
	 * \STATE ::CONFIG_STATE, ::SETUP_STATE, ::EXE_STATE
	 * \param grpId       the group ID
	 * \param relNeurId   the neuron ID (relative to the group) for which the 3D location should be returned
	 * \return the 3D location a neuron codes for as a Point3D struct
	 * \see getNeuronLocation3D(int neurId)
	 */
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	/** Returns total number of connections \brief Returns total number of connections*/
	int getNumConnections() { return numConnections; }
	/** Returns number of connections associated with a connection ID \brief Returns number of connections associated with a connection ID*/
	int getNumSynapticConnections(short int connectionId);
	/** Returns total number of groups \brief Returns total number of groups*/
	int getNumGroups() { return numGroups; }
	/** Returns total number of neurons in the network \brief Returns total number of neurons in the network*/
	int getNumNeurons() { return glbNetworkConfig.numN; }
	/** Returns total number of Izhikevich neurons in the network \brief Returns total number of Izhikevich neurons in the network*/
	int getNumNeuronsReg() { return glbNetworkConfig.numNReg; }
	/** Returns total number of excitatory Izhikevich neurons in the network \brief Returns total number of excitatory Izhikevich neurons in the network*/
	int getNumNeuronsRegExc() { return glbNetworkConfig.numNExcReg; }
	/** Returns total number of inhibitory Izhikevich neurons in the network \brief Returns total number of inhibitory Izhikevich neurons in the network*/
	int getNumNeuronsRegInh() { return glbNetworkConfig.numNInhReg; }
	/** Returns total number of poisson spiking neurons in the network \brief Returns total number of poisson spiking neurons in the network*/
	int getNumNeuronsGen() { return glbNetworkConfig.numNPois; }
	/** Returns total number of excitatory poisson spiking neurons in the network \brief Returns total number of excitatory poisson spiking neurons in the network*/
	int getNumNeuronsGenExc() { return glbNetworkConfig.numNExcPois; }
	/** Returns total number of inhibitory poisson spiking neurons in the network \brief Returns total number of inhibitory poisson spiking neurons in the network*/
	int getNumNeuronsGenInh() { return glbNetworkConfig.numNInhPois; }
	/** Returns total number of synapses in the network \brief Returns total number of synapses in the network*/
	int getNumSynapses() { return glbNetworkConfig.numSynNet; }
	/** Returns random seed used in the network \brief Returns randon seed used in the network*/
	int getRandSeed() { return randSeed_; }
	/** Returns current simulation time of the network \brief Returns current simulation time of the network*/
	int getSimTime() { return simTime; }
	/** Returns seconds part of the current simulation time of the network \brief Returns seconds part of the current simulation time of the network*/
	int getSimTimeSec() { return simTimeSec; }
	/** Returns ms part of the current simulation time of the network \brief Returns ms part of the current simulation time of the network*/
	int getSimTimeMs() { return simTimeMs; }

	/** This function returns a pointer to a SpikeMonitor object that has previously been created using the method
	 * setSpikeMonitor. If the group does not have a SpikeMonitor, NULL is returned.
	 *
	 * \brief returns pointer to previously allocated SpikeMonitor object, NULL else
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId the group ID
	 * \return pointer to SpikeMonitor object if exists, NULL else
	 */
	SpikeMonitor* getSpikeMonitor(int grpId);

	/** This function returns a pointer to existing SpikeMonitorCore object.
	 * If the group does not have a SpikeMonitorCore, NULL is returned. Should not be exposed to user interface.
	 *
	 * \brief Returns pointer to existing SpikeMonitorCore object, NULL else
	 * \STATE ::SETUP_STATE, ::RUN_STATE
	 * \param grpId the group ID
	 * \return pointer to SpikeMonitorCore object if exists, NULL else
	 */
	SpikeMonitorCore* getSpikeMonitorCore(int grpId);

	/** Temporary getter to return pointer to current[] array.
	 * \brief Getter to return pointer to current[] array
	 * TODO: replace with NeuronMonitor
	 */
	float* getCurrent() { return managerRuntimeData.current; }

	/** Returns connection weights from pre group neurons to post group neurons in a 2D matrix 
	 * \brief Returns connection weights from pre to post group neurons
	 * \param connId connection ID
	 * \return 2D matrix of weights
	 */
	std::vector< std::vector<float> > getWeightMatrix2D(short int connId);

	/** Returns AMPA value for all neurons in the specified group \brief Returns AMPA value for all neurons in the specified group*/
	std::vector<float> getConductanceAMPA(int grpId);
	/** Returns NMDA value for all neurons in the specified group \brief Returns NMDA value for all neurons in the specified group*/
	std::vector<float> getConductanceNMDA(int grpId);
	/** Returns GABAa value for all neurons in the specified group \brief Returns GABAa value for all neurons in the specified group*/
	std::vector<float> getConductanceGABAa(int grpId);
	/** Returns GABAb value for all neurons in the specified group \brief Returns GABAb value for all neurons in the specified group*/
	std::vector<float> getConductanceGABAb(int grpId);

	/** temporary getter to return pointer to stpu[] 
	 * \brief temporary getter to return pointer to stpu[] 
	 * TODO: replace with NeuronMonitor or ConnectionMonitor
	 */
	 float* getSTPu() { return managerRuntimeData.stpu; }

	/** temporary getter to return pointer to stpx[] 
	 * \brief temporary getter to return pointer to stpx[] 
	 * TODO: replace with NeuronMonitor or ConnectionMonitor
	 */
	float* getSTPx() { return managerRuntimeData.stpx; }

	/** returns whether synapses in the connection are fixed (false) or plastic (true) \brief returns whether synapses in the connection are plastic*/
	bool isConnectionPlastic(short int connId);

	/** returns RangeWeight struct of a connection \brief returns RangeWeight struct of a connection*/
	RangeWeight getWeightRange(short int connId);

	/** Returns if a given group is excitatory \brief Returns if a given group is excitatory*/
	bool isExcitatoryGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_AMPA) || (groupConfigMap[gGrpId].type & TARGET_NMDA); }
	/** Returns if a given group is inhibitory \brief Returns if a given group is inhibitory*/
	bool isInhibitoryGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_GABAa) || (groupConfigMap[gGrpId].type & TARGET_GABAb); }
	/** Returns if a given group is a poisson spike generator group \brief Returns if a given group is a poisson spike generator group*/
	bool isPoissonGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & POISSON_NEURON); }
	/** Returns if a given group is a dopaminergic group \brief Returns if a given group is a dopaminergic group*/
	bool isDopaminergicGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_DA); }

	/** Returns whether group has homeostasis enabled (true) or not (false) \brief Returns whether group has homeostasis enabled*/
	bool isGroupWithHomeostasis(int grpId);

	/** Returns 3D distance from pre to post neuron with repect to the given radius. 
	 *  For given, \f$x = pre.x-post.x, y = pre.y-post.y, z = pre.z-post.z\f$
	 *  The method returns \f$x^2/rad_x^2 + y^2/rad_y^2 + z^2/rad_z^2\f$
	 *  \brief Returns 3D distance from pre to post neuron with repect to the given radius
	 *  \param radius a RadiusRF struct, the referrence radius in 3D
	 *  \param pre Grid3D struct for the pre neuron
	 *  \param post Grid3D struct for the post neuron
	 *  \return real valued 3D distance between pre and post with respect to the given radius
	 */
	double getRFDist3D(const RadiusRF& radius, const Point3D& pre, const Point3D& post);
	/** Checks whether a point pre lies in the receptive field for point post.
	 *  For given, \f$x = pre.x-post.x, y = pre.y-post.y, z = pre.z-post.z.\f$
	 *  The method returns: \f$0.0 <= x^2/rad_x^2 + y^2/rad_y^2 + z^2/rad_z^2 <= 1.0\f$
	 *  Calls SNN::getRFDist3D(...).
	 *  \brief Checks whether a point pre lies in the receptive field for point post
	 *  \param radius a RadiusRF struct, the referrence radius in 3D
	 *  \param pre Grid3D struct for the pre neuron
	 *  \param post Grid3D struct for the post neuron
	 *  \return boolean whether \f$0.0 <= x^2/rad_x^2 + y^2/rad_y^2 + z^2/rad_z^2 <= 1.0\f$ (true)
	 */
	bool isPoint3DinRF(const RadiusRF& radius, const Point3D& pre, const Point3D& post);

	/** Returns true if the network synapses are conductance based \brief Returns if the network synapses are conductance based*/
	bool isSimulationWithCOBA() { return sim_with_conductances; }
	/** Returns true if the network synapses are current based \brief Returns if the network synapses are current based*/
	bool isSimulationWithCUBA() { return !sim_with_conductances; }
	/** Returns true if simulation with rise time for NMDA \brief Returns if simulation with rise time for NMDA*/
	bool isSimulationWithNMDARise() { return sim_with_NMDA_rise; }
	/** Returns true if simulation with rise time for GABAb \brief Returns if simulation with rise time for GABAb*/
	bool isSimulationWithGABAbRise() { return sim_with_GABAb_rise; }
	/** Returns true if synapses are fixed during simulation \brief Returns if synapses are fixed during simulation*/
	bool isSimulationWithFixedWeightsOnly() { return sim_with_fixedwts; }
	/** Returns true if simulation with Homeostasis\brief Returns true if simulation with Homeostasis*/
	bool isSimulationWithHomeostasis() { return sim_with_homeostasis; }
	/** Returns true if synapses are plastic during simulation \brief Returns if synapses are plastic during simulation*/
	bool isSimulationWithPlasticWeights() { return !sim_with_fixedwts; }
	/** Returns true if synapse weights are updated using STDP learning rule \brief Returns if synapse weights are updated using STDP learning rule*/
	bool isSimulationWithSTDP() { return sim_with_stdp; }
	/** Returns true if synapse weights are updated using STP learning rule \brief Returns if synapse weights are updated using STP learning rule*/
	bool isSimulationWithSTP() { return sim_with_stp; }

	// **************************************************************************************************************** //
	// PRIVATE METHODS
	// **************************************************************************************************************** //

private:
	//! all unsafe operations of constructor
	void SNNinit();

	//! advance time step in a simulation
	void advSimStep();

	//! allocates and initializes all core datastructures
	void allocateManagerRuntimeData();

	int assignGroup(int gGrpId, int availableNeuronId);
	int assignGroup(std::list<GroupConfigMD>::iterator grpIt, int localGroupId, int availableNeuronId);
	void generateGroupRuntime(int netId, int lGrpId);
	void generatePoissonGroupRuntime(int netId, int lGrpId);
	void generateConnectionRuntime(int netId);

	/*!
	 * \brief scan all GroupConfigs and ConnectConfigs for generating the configuration of a local network
	 */
	void generateRuntimeNetworkConfigs();
	void generateRuntimeGroupConfigs();
	void generateRuntimeConnectConfigs();

	/*!
	 * \brief scan all group configs and connection configs for generating the configuration of a global network
	 */
	void collectGlobalNetworkConfigC();
	void compileConnectConfig(); //!< for future use
	void compileGroupConfig();

	void collectGlobalNetworkConfigP();

	/*!
	 * \brief generate connections among groups according to connect configuration
	 */
	void connectNetwork();
	inline void connectNeurons(int netId, int srcGrp, int destGrp, int srcN, int destN, short int connId, int externalNetId);
	inline void connectNeurons(int netId, int _grpSrc, int _grpDest, int _nSrc, int _nDest, short int _connId, float initWt, float maxWt, uint8_t delay, int externalNetId);
	void connectFull(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectOneToOne(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectRandom(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectGaussian(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectUserDefined(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);

	void deleteObjects();			//!< deallocates all used data structures in snn_cpu.cpp

	void findMaxNumSynapsesGroups(int* _maxNumPostSynGrp, int* _maxNumPreSynGrp);
	void findMaxNumSynapsesNeurons(int _netId, int& _maxNumPostSynN, int& _maxNumPreSynN);
	void findMaxSpikesD1D2(int netId, unsigned int& _maxSpikesD1, unsigned int& _maxSpikesD2);
	void findNumSynapsesNetwork(int netId, int& _numPostSynNet, int& _numPreSynNet); //!< find the total number of synapses in the network
	void findNumN(int _netId, int& _numN, int& _nunNExternal, int& numNAssigned,
		int& _numNReg, int& _numNExcReg, int& _numNInhReg,
		int& _numNPois, int& _numNExcPois, int& _numNInhPois);
	void findNumNSpikeGenAndOffset(int _netId);

	void generatePostSynapticSpike(int preNId, int postNId, int synId, int tD, int netId);
	void fillSpikeGenBits(int netId);
	void userDefinedSpikeGenerator(int gGrpId);

	float generateWeight(int connProp, float initWt, float maxWt, int nid, int grpId);

	//! performs various verification checkups before building the network
	void verifyNetwork();

	//! make sure STDP post-group has some incoming plastic connections
	void verifySTDP();

	//! make sure every group with homeostasis also has STDP
	void verifyHomeostasis();

	//! performs a consistency check to see whether numN* class members have been accumulated correctly
	//void verifyNumNeurons();

	void compileSNN();

	void partitionSNN();

	void generateRuntimeSNN();

	/*!
	 * \brief generates spike times according to a Poisson process
	 *
	 * This function generates the next spike time for a Poisson spike generator, given the current spike time
	 * <tt>currTime</tt>. The function is used in generateSpikesFromRate.
	 * A Poisson process will always generate inter-spike-interval (ISI) values from an exponential distribution.
	 * The time between each pair of consecutive events has an exponential distribution with parameter \lambda and
	 * each of these ISI values is assumed to be independent of other ISI values.
	 * What follows a Poisson distribution is the actual number of spikes sent during a certain interval.
	 *
	 * The refractory period (in ms) will assert that all spike pairs are at least <tt>refractPeriod</tt> milliseconds
	 * apart (inclusive). That is, the ISI generated from an exponential distribution will be discarded if it turns
	 * out that ISI < refractPeriod
	 * \param[in] currTime       time of current (or "last") spike
	 * \param[in] frate          mean firing rate to be achieved (same as \lambda of the exponential distribution)
	 * \param[in] refractPeriod  refractory period to be honored (in ms)
	 * \returns next spike time (current time plus generated ISI)
	 */
	 //int poissonSpike(int currTime, float frate, int refractPeriod);

	void printConnectionInfo(short int connId);
	void printConnectionInfo(int netId, std::list<ConnectConfig>::iterator connIt);
	void printGroupInfo(int grpId);
	void printGroupInfo(int netId, std::list<GroupConfigMD>::iterator grpIt);
	void printSimSummary(); //!< prints a simulation summary at the end of sim
	void printStatusConnectionMonitor(int connId = ALL);
	void printStatusGroupMonitor(int gGrpId = ALL);
	void printStatusSpikeMonitor(int gGrpId = ALL);
	void printSikeRoutingInfo();

	int loadSimulation_internal(bool onlyPlastic);

	void resetConductances(int netId);
	void resetCurrent(int netId);
	void resetFiringInformation(); //!< resets the firing information when updateNetwork is called
	void resetGroupConfigs(bool deallocate = false);
	void resetNeuromodulator(int netId, int lGrpId);
	void resetNeuron(int netId, int lGrpId, int lNId);
	void resetMonitors(bool deallocate = false);
	void resetConnectionConfigs(bool deallocate = false);
	void deleteManagerRuntimeData();
	void resetPoissonNeuron(int netId, int lGrpId, int lNId); //!< use local ids
	void resetPropogationBuffer();
	void resetSynapse(int netId, bool changeWeights = false);
	void resetTimeTable();
	void resetFiringTable();
	void routeSpikes();
	void transferSpikes(void* dest, int destNetId, void* src, int srcNetId, int size);
	void resetTiming();

	inline SynInfo SET_CONN_ID(int nid, int sid, int grpId);

	void setGrpTimeSlice(int grpId, int timeSlice); //!< used for the Poisson generator. TODO: further optimize
	int setRandSeed(int seed);	//!< setter function for const member randSeed_

	void startTiming();
	void stopTiming();

	void generateUserDefinedSpikes();

	void allocateManagerSpikeTables();

	bool updateTime(); //!< updates simTime, returns true when a new second is started

	// Abstract layer for setupNetwork() and runNetwork()
	void allocateSNN(int netId);
	void clearExtFiringTable();
	void convertExtSpikesD1(int netId, int startIdx, int endIdx, int GtoLOffset);
	void convertExtSpikesD2(int netId, int startIdx, int endIdx, int GtoLOffset);
	void doCurrentUpdate();
	void doSTPUpdateAndDecayCond();
	void deleteRuntimeData();
	void findFiring();
	void globalStateUpdate();
	void resetSpikeCnt(int gGrpId);
	void shiftSpikeTables();
	void spikeGeneratorUpdate();
	void updateTimingTable();
	void updateWeights();
	void updateNetworkConfig(int netId);

	// Abstract layer for trasferring data (local-to-global copy)
	void fetchConductanceAMPA(int gGrpId);
	void fetchConductanceNMDA(int gGrpId);
	void fetchConductanceGABAa(int gGrpId);
	void fetchConductanceGABAb(int gGrpId);
	void fetchNetworkSpikeCount();
	void fetchNeuronSpikeCount(int gGrpId);
	void fetchSTPState(int gGrpId);

	// Abstract layer for trasferring data (local-to-local copy)
	void fetchSpikeTables(int netId);
	void fetchGroupState(int netId, int lGrpId);
	void fetchWeightState(int netId, int lGrpId);
	void fetchGrpIdsLookupArray(int netId);
	void fetchConnIdsLookupArray(int netId);
	void fetchLastSpikeTime(int netId);
	void fetchPreConnectionInfo(int netId);
	void fetchPostConnectionInfo(int netId);
	void fetchSynapseState(int netId);
	void fetchExtFiringTable(int netId);
	void fetchTimeTable(int netId);
	void writeBackTimeTable(int netId);

#ifndef __NO_CUDA__
	// GPU implementation for setupNetwork() and runNetwork()
	void allocateSNN_GPU(int netId); //!< allocates runtime data on GPU memory and initialize GPU
	void assignPoissonFiringRate_GPU(int netId);
	void clearExtFiringTable_GPU(int netId);
	void convertExtSpikesD1_GPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void convertExtSpikesD2_GPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void doCurrentUpdateD1_GPU(int netId);
	void doCurrentUpdateD2_GPU(int netId);
	void doSTPUpdateAndDecayCond_GPU(int netId);
	void deleteRuntimeData_GPU(int netId);		//!< deallocates all used data structures in snn_gpu.cu
	void findFiring_GPU(int netId);
	void globalStateUpdate_C_GPU(int netId);
	void globalStateUpdate_N_GPU(int netId);
	void globalStateUpdate_G_GPU(int netId);
	void resetSpikeCnt_GPU(int netId, int lGrpId); //!< Utility function to clear spike counts in the GPU code.
	void shiftSpikeTables_F_GPU(int netId);
	void shiftSpikeTables_T_GPU(int netId);
	void spikeGeneratorUpdate_GPU(int netId);
	void updateTimingTable_GPU(int netId);
	void updateWeights_GPU(int netId);
#else
	void allocateSNN_GPU(int netId) { assert(false); } //!< allocates runtime data on GPU memory and initialize GPU
	void assignPoissonFiringRate_GPU(int netId) { assert(false); }
	void clearExtFiringTable_GPU(int netId) { assert(false); }
	void convertExtSpikesD1_GPU(int netId, int startIdx, int endIdx, int GtoLOffset) { assert(false); }
	void convertExtSpikesD2_GPU(int netId, int startIdx, int endIdx, int GtoLOffset) { assert(false); }
	void doCurrentUpdateD1_GPU(int netId) { assert(false); }
	void doCurrentUpdateD2_GPU(int netId) { assert(false); }
	void doSTPUpdateAndDecayCond_GPU(int netId) { assert(false); }
	void deleteRuntimeData_GPU(int netId) { assert(false); }		//!< deallocates all used data structures in snn_gpu.cu
	void findFiring_GPU(int netId) { assert(false); }
	void globalStateUpdate_C_GPU(int netId) { assert(false); }
	void globalStateUpdate_N_GPU(int netId) { assert(false); }
	void globalStateUpdate_G_GPU(int netId) { assert(false); }
	void resetSpikeCnt_GPU(int netId, int lGrpId) { assert(false); } //!< Utility function to clear spike counts in the GPU code.
	void shiftSpikeTables_F_GPU(int netId) { assert(false); }
	void shiftSpikeTables_T_GPU(int netId) { assert(false); }
	void spikeGeneratorUpdate_GPU(int netId) { assert(false); }
	void updateTimingTable_GPU(int netId) { assert(false); }
	void updateWeights_GPU(int netId) { assert(false); }
#endif

#ifndef __NO_CUDA__
	// GPU backend: utility function
	void allocateGroupId(int netId);
	int  allocateStaticLoad(int netId, int bufSize);
	void checkAndSetGPUDevice(int netId);
	void checkDestSrcPtrs(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId, int destOffset);
	int configGPUDevice();
	void initGPU(int netId);
#else
	int configGPUDevice() { return 0; }
#endif

#ifndef __NO_CUDA__
	// GPU backend: data transfer functions
	void copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyNeuronState(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copySynapseState(int netId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyWeightState(int netId, int lGrpId, cudaMemcpyKind kind);
	void copyNetworkConfig(int netId, cudaMemcpyKind kind);
	void copyGroupConfigs(int netId);
	void copyGrpIdsLookupArray(int netId, cudaMemcpyKind kind);
	void copyConnIdsLookupArray(int netId, cudaMemcpyKind kind);
	void copyLastSpikeTime(int netId, cudaMemcpyKind kind);
	void copyNetworkSpikeCount(int netId, cudaMemcpyKind kind,
		unsigned int* spikeCountD1, unsigned int* spikeCountD2,
		unsigned int* spikeCountExtD1, unsigned int* spikeCountExtD2);
	void copySpikeTables(int netId, cudaMemcpyKind kind);
	void copyTimeTable(int netId, cudaMemcpyKind kind);
	void copyExtFiringTable(int netId, cudaMemcpyKind kind);
#else
	#define cudaMemcpyKind int
	#define cudaMemcpyHostToHost 0
	#define cudaMemcpyHostToDevice 0
	#define cudaMemcpyDeviceToHost 0
	#define cudaMemcpyDeviceToDevice 0
	#define cudaMemcpyDefault 0

	void copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) { assert(false); }
	void copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) { assert(false); }
	void copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) { assert(false); }
	void copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) { assert(false); }
	void copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyNeuronState(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) { assert(false); }
	void copySynapseState(int netId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
	void copyWeightState(int netId, int lGrpId, cudaMemcpyKind kind) { assert(false); }
	void copyNetworkConfig(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyGroupConfigs(int netId) { assert(false); }
	void copyGrpIdsLookupArray(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyConnIdsLookupArray(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyLastSpikeTime(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyNetworkSpikeCount(int netId, cudaMemcpyKind kind,
	unsigned int* spikeCountD1, unsigned int* spikeCountD2,
	unsigned int* spikeCountExtD1, unsigned int* spikeCountExtD2) { assert(false); }
	void copySpikeTables(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyTimeTable(int netId, cudaMemcpyKind kind) { assert(false); }
	void copyExtFiringTable(int netId, cudaMemcpyKind kind) { assert(false); }
#endif

	// CPU implementation for setupNetwork() and runNetwork()

	//allocates runtime data on CPU memory
	void allocateSNN_CPU(int netId); 

	// runNetwork functions - multithreaded in POSIX using pthreads
#if defined(WIN32) || defined(WIN64)
	void assignPoissonFiringRate_CPU(int netId);
	void clearExtFiringTable_CPU(int netId);
	void convertExtSpikesD2_CPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void convertExtSpikesD1_CPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void doCurrentUpdateD2_CPU(int netId);
	void doCurrentUpdateD1_CPU(int netId);
	void doSTPUpdateAndDecayCond_CPU(int netId);
	void deleteRuntimeData_CPU(int netId);
	void findFiring_CPU(int netId);
	void globalStateUpdate_CPU(int netId);
	void resetSpikeCnt_CPU(int netId, int lGrpId); //!< Resets the spike count for a particular group.
	void shiftSpikeTables_CPU(int netId);
	void spikeGeneratorUpdate_CPU(int netId);
	void updateTimingTable_CPU(int netId);
	void updateWeights_CPU(int netId);
#else // for POSIX systems - returns a void* to pthread_create - only differ in the return type compared to the counterparts above
	void* assignPoissonFiringRate_CPU(int netId);
	void* clearExtFiringTable_CPU(int netId);
	void* convertExtSpikesD2_CPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void* convertExtSpikesD1_CPU(int netId, int startIdx, int endIdx, int GtoLOffset);
	void* doCurrentUpdateD2_CPU(int netId);
	void* doCurrentUpdateD1_CPU(int netId);
	void* doSTPUpdateAndDecayCond_CPU(int netId);
	void* deleteRuntimeData_CPU(int netId);
	void* findFiring_CPU(int netId);
	void* globalStateUpdate_CPU(int netId);
	void* resetSpikeCnt_CPU(int netId, int lGrpId); //!< Resets the spike count for a particular group.
	void* shiftSpikeTables_CPU(int netId);
	void* spikeGeneratorUpdate_CPU(int netId);
	void* updateTimingTable_CPU(int netId);
	void* updateWeights_CPU(int netId);

	// static multithreading helper methods for the above CPU runNetwork() methods
	static void* helperAssignPoissonFiringRate_CPU(void*);
	static void* helperClearExtFiringTable_CPU(void*);
	static void* helperConvertExtSpikesD2_CPU(void*);
	static void* helperConvertExtSpikesD1_CPU(void*);
	static void* helperDoCurrentUpdateD2_CPU(void*);
	static void* helperDoCurrentUpdateD1_CPU(void*);
	static void* helperDoSTPUpdateAndDecayCond_CPU(void*);
	static void* helperDeleteRuntimeData_CPU(void*);
	static void* helperFindFiring_CPU(void*);
	static void* helperGlobalStateUpdate_CPU(void*);
	static void* helperResetSpikeCnt_CPU(void*);
	static void* helperShiftSpikeTables_CPU(void*);
	static void* helperSpikeGeneratorUpdate_CPU(void*);
	static void* helperUpdateTimingTable_CPU(void*);
	static void* helperUpdateWeights_CPU(void*);
#endif

	// CPU computing backend: data transfer function
	void copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);
	void copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	void copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	void copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);
	void copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);	
	void copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);	
	void copyNeuronState(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);	
	void copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);	
	void copySynapseState(int netId, RuntimeData* dest, RuntimeData* src, bool allocateMem);	
	void copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);	
	void copyWeightState(int netId, int lGrpId);
	void copyNetworkConfig(int netId);
	void copyGrpIdsLookupArray(int netId);
	void copyConnIdsLookupArray(int netId);
	void copyLastSpikeTime(int netId);
	void copyNetworkSpikeCount(int netId,
	unsigned int* spikeCountD1, unsigned int* spikeCountD2,
	unsigned int* spikeCountExtD1, unsigned int* spikeCountExtD2);
	void copySpikeTables(int netId);
	void copyTimeTable(int netId, bool toManager);
	void copyExtFiringTable(int netId);
	
	// CPU backend: utility function
	void firingUpdateSTP(int lNId, int lGrpId, int netId);
	void updateLTP(int lNId, int lGrpId, int netId);
	void resetFiredNeuron(int lNId, short int lGrpId, int netId);
	bool getPoissonSpike(int lNId, int netId);
	bool getSpikeGenBit(unsigned int nIdPos, int netId);

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	SNNState snnState; //!< state of the network
	FILE* loadSimFID;

	const std::string networkName_;	//!< network name
	const LoggerMode loggerMode_;	//!< current logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	const SimMode preferredSimMode_;//!< preferred simulation mode
	const int randSeed_;			//!< random number seed to use

	int numGPUs;    //!< number of GPU(s) is used in the simulation
	int numCores;   //!< number of CPU Core(s) is used in the simulation

	int numAvailableGPUs; //!< number of available GPU(s) in the machine

	bool simulatorDeleted;
	bool spikeRateUpdated;

	//! switch to make all weights fixed (such as in testing phase) or not
	bool sim_in_testing;

	int numGroups;      //!< the number of groups (as in snn.createGroup, snn.createSpikeGeneratorGroup)
	int numConnections; //!< the number of connections (as in snn.connect(...))

	std::map<int, GroupConfig> groupConfigMap;   //!< the hash table storing group configs created at CONFIG_STATE
	std::map<int, GroupConfigMD> groupConfigMDMap; //!< the hash table storing group configs meta data generated at SETUP_STATE
	std::map<int, ConnectConfig> connectConfigMap; //!< the hash table storing connection configs created at CONFIG_STATE

	// data structure assisting network partitioning
	std::list<GroupConfigMD> groupPartitionLists[MAX_NET_PER_SNN];
	std::list<ConnectConfig> localConnectLists[MAX_NET_PER_SNN];
	std::list<ConnectConfig> externalConnectLists[MAX_NET_PER_SNN];

	std::list<ConnectionInfo> connectionLists[MAX_NET_PER_SNN];

	std::list<RoutingTableEntry> spikeRoutingTable;

	float 		*mulSynFast;	//!< scaling factor for fast synaptic currents, per connection
	float 		*mulSynSlow;	//!< scaling factor for slow synaptic currents, per connection

	//! Buffer to store spikes
	SpikeBuffer* spikeBuf;

	bool sim_with_conductances; //!< flag to inform whether we run in COBA mode (true) or CUBA mode (false)
	bool sim_with_NMDA_rise;    //!< a flag to inform whether to compute NMDA rise time
	bool sim_with_GABAb_rise;   //!< a flag to inform whether to compute GABAb rise time
	double dAMPA;               //!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	double rNMDA;               //!< multiplication factor for rise time of NMDA
	double dNMDA;               //!< multiplication factor for decay time of NMDA
	double sNMDA;               //!< scaling factor for NMDA amplitude
	double dGABAa;              //!< multiplication factor for decay time of GABAa
	double rGABAb;              //!< multiplication factor for rise time of GABAb
	double dGABAb;              //!< multiplication factor for decay time of GABAb
	double sGABAb;              //!< scaling factor for GABAb amplitude

	bool sim_with_fixedwts;
	bool sim_with_stdp;
	bool sim_with_modulated_stdp;
	bool sim_with_homeostasis;
	bool sim_with_stp;
	bool sim_with_spikecounters; //!< flag will be true if there are any spike counters around

	GlobalNetworkConfig glbNetworkConfig; //!< the global network related information, including neurons, synapses and network configs

	//time and timestep
	int simTimeRunStart; //!< the start time of current/last runNetwork call
	int simTimeRunStop;  //!< the end time of current/last runNetwork call
	int simTimeLastRunSummary; //!< the time at which the last run summary was printed
	int simTimeMs;      //!< The simulation time showing milliseconds within a second
	int simTimeSec;     //!< The simulation time showing seconds in a simulation
	int simTime;        //!< The absolute simulation time. The unit is millisecond. this value is not reset but keeps increasing to its max value.

	//! vairables for tracking performance
#ifndef __NO_CUDA__
	StopWatchInterface* timer;
#endif
	float cumExecutionTime;
	float lastExecutionTime;
	float prevExecutionTime;
	float executionTime;

	FILE*	fpInf_; //!< fp of where to write all simulation output (status info) if not in silent mode
	FILE*	fpErr_; //!< fp of where to write all errors if not in silent mode
	FILE*	fpDeb_; //!< fp of where to write all debug info if not in silent mode
	FILE*	fpLog_;

	// keep track of number of SpikeMonitor/SpikeMonitorCore objects
	int numSpikeMonitor;
	SpikeMonitorCore* spikeMonCoreList[MAX_GRP_PER_SNN];
	SpikeMonitor*     spikeMonList[MAX_GRP_PER_SNN];

	// \FIXME \DEPRECATED this one moved to group-based
	long int    simTimeLastUpdSpkMon_; //!< last time we ran updateSpikeMonitor

	int numSpikeGenGrps;

	// keep track of number of GroupMonitor/GroupMonitorCore objects
	int numGroupMonitor;
	GroupMonitorCore*	groupMonCoreList[MAX_GRP_PER_SNN];
	GroupMonitor*		groupMonList[MAX_GRP_PER_SNN];

	// neuron monitor variables
	//NeuronMonitorCore* neurBufferCallback[MAX_]
	int numNeuronMonitor;

	// connection monitor variables
	int numConnectionMonitor;
	ConnectionMonitorCore* connMonCoreList[MAX_CONN_PER_SNN];
	ConnectionMonitor*     connMonList[MAX_CONN_PER_SNN];

	RuntimeData runtimeData[MAX_NET_PER_SNN];
	RuntimeData managerRuntimeData;

	typedef struct ManagerRuntimeDataSize_s {
		unsigned int maxMaxSpikeD1;
		unsigned int maxMaxSpikeD2;
		int maxNumN;
		int maxNumNReg;
		int maxNumNSpikeGen;
		int maxNumNAssigned;
		int maxNumGroups;
		int maxNumConnections;
		int maxNumPostSynNet;
		int maxNumPreSynNet;
		int maxNumNPerGroup;
		int glbNumN;
		int glbNumNReg;
	} ManagerRuntimeDataSize;

	ManagerRuntimeDataSize managerRTDSize;

	// runtime configurations
	NetworkConfigRT networkConfigs[MAX_NET_PER_SNN]; //!< the network configs used on GPU(s);
	GroupConfigRT	groupConfigs[MAX_NET_PER_SNN][MAX_GRP_PER_SNN];
	ConnectConfigRT connectConfigs[MAX_NET_PER_SNN][MAX_CONN_PER_SNN]; //!< for future use

	// weight update parameter
	int wtANDwtChangeUpdateInterval_;
	int wtANDwtChangeUpdateIntervalCnt_;
	float stdpScaleFactor_;
	float wtChangeDecay_; //!< the wtChange decay
};

#endif
