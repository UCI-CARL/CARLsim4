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
class NeuronMonitor;
class NeuronMonitorCore;
class ConnectionMonitorCore;
class ConnectionMonitor;

class SpikeBuffer;


/// **************************************************************************************************************** ///
/// CPUSNN CORE CLASS
/// **************************************************************************************************************** ///

/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
class SNN {

	/// **************************************************************************************************************** ///
	/// PUBLIC METHODS
	/// **************************************************************************************************************** ///
public:
	//! SNN Constructor
	/*!
	 * \brief
	 * \param name the symbolic name of a spiking neural network
	 * \param loggerMode log mode
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

	//! Creates synaptic projections from a pre-synaptic group to a post-synaptic group using a pre-defined primitive
	//! type.
	/*!
	 * \brief make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	 *
	 * \param grpIdPre ID of the pre-synaptic group
	 * \param grpIdPost ID of the post-synaptic group
	 * \param connType connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post (no self-connections).
	 * \param initWt initial weight strength (arbitrary units); should be negative for inhibitory connections
	 * \param maxWt upper bound on weight strength (arbitrary units); should be negative for inhibitory connections
	 * \param connProb connection probability
	 * \param minDelay the minimum delay allowed (ms)
	 * \param maxdelay: the maximum delay allowed (ms)
	 * \param synWtType: (optional) connection type, either SYN_FIXED or SYN_PLASTIC, default = SYN_FIXED.
	 * \param wtType: (optional) DEPRECATED
	 * \return number of created synaptic projections
	 */
	short int connect(int gIDpre, int gIDpost, const std::string& _type, float initWt, float maxWt, float prob,
		uint8_t minDelay, uint8_t maxDelay, RadiusRF radius,
		float mulSynFast, float mulSynSlow, bool synWtType);

	/* Creates synaptic projections using a callback mechanism.
	 *
	 * \param _grpIdPre:ID of the pre-synaptic group
	 * \param _grpIdPost ID of the post-synaptic group
	 * \param _conn: pointer to an instance of class ConnectionGenerator
	 * \param _synWtType: (optional) connection type, either SYN_FIXED or SYN_PLASTIC, default = SYN_FIXED
	 * \return number of created synaptic projections
	 */
	short int connect(int gIDpre, int gIDpost, ConnectionGeneratorCore* conn, float mulSynFast, float mulSynSlow,
		bool synWtType);

	/* Creates synaptic projections using a callback mechanism.
	*
	* \param _grpId1:ID lower layer group
	* \param _grpId2 ID upper level group
	*/
	short int connectCompartments(int grpIdLower, int grpIdUpper);

	//! Creates a group of Izhikevich spiking neurons
	/*!
	 * \param name the symbolic name of a group
	 * \param grid  Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend);

	//! Creates a group of LIF spiking neurons
	/*!
	 * \param grpName the symbolic name of a group
	 * \param grid Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param neurType the type of neuron
	 * \param preferredPartition defines the desired runtime partition for the group
	 * \param preferredBackend defines whether the group will be placed on CPU or GPU
	 */
	int createGroupLIF(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend);

	//! Creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons)
	/*!
	 * \param name the symbolic name of a group
	 * \param grid Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron, currently only support EXCITATORY NEURON
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend);

	/*!
	* \brief Coupling constants for the compartment are set using this method.
	* \param grpId  		the symbolic name of a group
	* \param couplingUp   	the coupling constant for upper connections
	* \param couplingDown	the coupling constant for lower connections
	*/
	void setCompartmentParameters(int grpId, float couplingUp, float couplingDown);

	/*!
	 * \brief Sets custom values for conductance decay (\tau_decay) or disables conductances alltogether
	 * These will be applied to all connections in a network
	 * For details on the ODE that is implemented refer to (Izhikevich et al, 2004), and for suitable values see (Dayan & Abbott, 2001).
	 *
	 * \param isSet: enables the use of COBA mode
	 * \param tAMPA: time _constant of AMPA decay (ms); for example, 5.0
	 * \param tNMDA: time constant of NMDA decay (ms); for example, 150.0
	 * \param tGABAa: time constant of GABAa decay (ms); for example, 6.0
	 * \param tGABAb: time constant of GABAb decay (ms); for example, 150.0
	 */
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);


	/*!
	 * \brief Sets the homeostasis parameters. g is the grpID, enable=true(false) enables(disables) homeostasis,
	 * homeostasisScale is strength of
	 * homeostasis compared to the strength of normal LTP/LTD from STDP (which is 1), and avgTimeScale is the time
	 * frame over which the average firing rate is averaged (it should be larger in scale than STDP timescales).
	 */
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale);

	//! Sets homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	void setHomeoBaseFiringRate(int groupId, float baseFiring, float baseFiringSD);

	//! Sets the integration method and the number of integration steps per 1ms simulation time step
	void setIntegrationMethod(integrationMethod_t method, int numStepsPerMs);

	//! Sets the Izhikevich parameters a, b, c, and d of a neuron group.
	/*!
	 * \brief Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d and standard deviation _a_sd, _b_sd, _c_sd, and _d_sd, respectively
	 * \param _groupId the symbolic name of a group
	 * \param _a  the mean value of izhikevich parameter a
	 * \param _a_sd the standard deviation value of izhikevich parameter a
	 * \param _b  the mean value of izhikevich parameter b
	 * \param _b_sd the standard deviation value of izhikevich parameter b
	 * \param _c  the mean value of izhikevich parameter c
	 * \param _c_sd the standard deviation value of izhikevich parameter c
	 * \param _d  the mean value of izhikevich parameter d
	 * \param _d_sd the standard deviation value of izhikevich parameter d
	 */
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd);

	/*!
	 * \brief Sets neuron parameters for a group of LIF spiking neurons
	 *
	 * \param[in] grpId group ID
	 * \param[in] tau_m Membrane time constant in ms (controls decay/leak)
	 * \param[in] tau_ref absolute refractory period in ms
	 * \param[in] vTh Threshold voltage for firing (must be > vReset)
	 * \param[in] vReset Membrane potential resets to this value immediately after spike
	 * \param[in] minRmem minimum membrane resistance
	 * \param[in] maxRmem maximum membrane resistance
	 * 
	 */
	void setNeuronParametersLIF(int grpId, int tau_m, int tau_ref, float vTh, float vReset, double minRmem, double maxRmem);

	//! Sets the Izhikevich parameters C, k, vr, vt, a, b, vpeak, c, and d of a neuron group.
	/*!
	* \brief Parameter values for each neuron are given by a normal distribution with mean _C, _k, _vr, _vt, _a, _b, _vpeak, _c, and _d
	* and standard deviation _C_sd, _k_sd, _vr_sd, _vt_sd, _a_sd, _b_sd, _vpeak_sd, _c_sd, and _d_sd, respectively
	* \param _groupId the symbolic name of a group
	* \param _C  the mean value of izhikevich parameter C
	* \param _C_sd the standart deviation value of izhikevich parameter C
	* \param _k  the mean value of izhikevich parameter k
	* \param _k_sd the standart deviation value of izhikevich parameter k
	* \param _vr  the mean value of izhikevich parameter vr
	* \param _vr_sd the standart deviation value of izhikevich parameter vr
	* \param _vt  the mean value of izhikevich parameter vt
	* \param _vt_sd the standart deviation value of izhikevich parameter vt
	* \param _a  the mean value of izhikevich parameter a
	* \param _a_sd the standard deviation value of izhikevich parameter a
	* \param _b  the mean value of izhikevich parameter b
	* \param _b_sd the standard deviation value of izhikevich parameter b
	* \param _vpeak  the mean value of izhikevich parameter vpeak
	* \param _vpeak_sd the standart deviation value of izhikevich parameter vpeak
	* \param _c  the mean value of izhikevich parameter c
	* \param _c_sd the standard deviation value of izhikevich parameter c
	* \param _d  the mean value of izhikevich parameter d
	* \param _d_sd the standard deviation value of izhikevich parameter d
	*/
	void setNeuronParameters(int grpId, float izh_C, float izh_C_sd, float izh_k, float izh_k_sd,
		float izh_vr, float izh_vr_sd, float izh_vt, float izh_vt_sd,
		float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
		float izh_vpeak, float izh_vpeak_sd, float izh_c, float izh_c_sd,
		float izh_d, float izh_d_sd);

	//! Sets baseline concentration and decay time constant of neuromodulators (DP, 5HT, ACh, NE) for a neuron group.
	/*!
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

	//! Set the spike-timing-dependent plasticity (STDP) for a neuron group.
	/*
	 * \brief STDP must be defined post-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1,
	 * call setSTP on group 1. Fore details on the phenomeon, see (for example) (Bi & Poo, 2001).
	 * \param[in] grpId ID of the neuron group
	 * \param[in] isSet_enable set to true to enable STDP for this group
	 * \param[in] type STDP type (STANDARD, DA_MOD)
	 * \param[in] alphaPlus max magnitude for LTP change
	 * \param[in] tauPlus decay time constant for LTP
	 * \param[in] alphaMinus max magnitude for LTD change (leave positive)
	 * \param[in] tauMinus decay time constant for LTD
	 */
	void setESTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma);

	//! Set the inhibitory spike-timing-dependent plasticity (STDP) with anti-hebbian curve for a neuron group
	/*
	 * \brief STDP must be defined post-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1,
	 * call setSTP on group 1. Fore details on the phenomeon, see (for example) (Bi & Poo, 2001).
	 * \param[in] grpId ID of the neuron group
	 * \param[in] isSet_enable set to true to enable STDP for this group
	 * \param[in] type STDP type (STANDARD, DA_MOD)
	 * \param[in] curve STDP curve
	 * \param[in] ab1 magnitude for LTP change
	 * \param[in] ab2 magnitude for LTD change (leave positive)
	 * \param[in] tau1, the interval for LTP
	 * \param[in] tau2, the interval for LTD
	 */
	void setISTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float ab1, float ab2, float tau1, float tau2);

	/*!
	 * \brief Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
	 * CARLsim implements the short-term plasticity model of (Tsodyks & Markram, 1998; Mongillo, Barak, & Tsodyks, 2008)
	 * du/dt = -u/STP_tau_u + STP_U * (1-u-) * \delta(t-t_spk)
	 * dx/dt = (1-x)/STP_tau_x - u+ * x- * \delta(t-t_spk)
	 * dI/dt = -I/tau_S + A * u+ * x- * \delta(t-t_spk)
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
	 * \param[in] STP_tau_u   decay constant of u (\tau_F)
	 * \param[in] STP_tau_x   decay constant of x (\tau_D)
	 */
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x);

	//! Sets the weight and weight change update parameters
	/*!
	 * \param[in] wtANDwtChangeUpdateInterval the interval between two wt (weight) and wtChange (weight change) update.
	 * \param[in] enableWtChangeDecay enable weight change decay
	 * \param[in] wtChangeDecay the decay ratio of weight change (wtChange)
	 */
	void setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay);

	// +++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for n sec
	 *
	 * \param[in] printRunSummary whether to print a basic summary of the run at the end
	 */
	int runNetwork(int _nsec, int _nmsec, bool printRunSummary);

	/*!
	 * \brief build the network
	 * \param[in] removeTempMemory 	remove temp memory after building network
	 */
	void setupNetwork();

	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// adds a bias to every weight in the connection
	void biasWeights(short int connId, float bias, bool updateWeightRange = false);

	//! deallocates all dynamical structures and exits
	void exitSimulation(int val = 1);

	//! reads the network state from file
	//! Reads a CARLsim network file. Such a file can be created using SNN:writeNetwork.
	/*
	 * \brief After calling SNN::loadSimulation, you should run SNN::runNetwork before calling fclose(fp).
	 * \param fid: file pointer
	 * \sa SNN::saveSimulation()
	 */
	void loadSimulation(FILE* fid);

	// multiplies every weight with a scaling factor
	void scaleWeights(short int connId, float scale, bool updateWeightRange = false);

	//! sets up a group monitor registered with a callback to process the spikes.
	/*!
	 * \param[in] grpId ID of the neuron group
	 * \param[in] fid file pointer for recording group status (neuromodulators)
	 */
	GroupMonitor* setGroupMonitor(int grpId, FILE* fid);

	//! sets up a network monitor registered with a callback to process the spikes.
	/*!
	 * \param[in] grpIdPre ID of the pre-synaptic neuron group
	 * \param[in] grpIdPost ID of the post-synaptic neuron group
	 * \param[in] connectionMon ConnectionMonitorCore class
	 */
	ConnectionMonitor* setConnectionMonitor(int grpIdPre, int grpIdPost, FILE* fid);

	//! injects current (mA) into the soma of every neuron in the group
	void setExternalCurrent(int grpId, const std::vector<float>& current);

	//! sets up a spike generator
	void setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGenFunc);

	//! sets up a spike monitor registered with a callback to process the spikes, there can only be one SpikeMonitor per group
	/*!
	 * \param grpId ID of the neuron group
	 * \param spikeMon (optional) spikeMonitor class
	 * \return SpikeMonitor* pointer to a SpikeMonitor object
	 */
	SpikeMonitor* setSpikeMonitor(int gid, FILE* fid);

	//! sets up a neuron monitor registered with a callback to process the neuron state values, there can only be one NeuronMonitor per group
	/*!
	* \param grpId ID of the neuron group
	* \param neuronMon (optional) neuronMonitor class
	* \return NeuronMonitor* pointer to a NeuronMonitor object
	*/
	NeuronMonitor* setNeuronMonitor(int gid, FILE* fid);

	//!Sets the Poisson spike rate for a group. For information on how to set up spikeRate, see Section Poisson spike generators in the Tutorial.
	/*!Input arguments:
	 * \param grpId ID of the neuron group
	 * \param spikeRate pointer to a PoissonRate instance
	 * \param refPeriod (optional) refractive period,  default = 1
	 */
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod);

	//! sets the weight value of a specific synapse
	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange = false);

	//! enters a testing phase, where all weight updates are disabled
	void startTesting(bool shallUpdateWeights = true);

	//! exits a testing phase, making weight updates possible again
	void stopTesting();

	//! polls connection weights
	void updateConnectionMonitor(short int connId = ALL);

	//! access group status (currently the concentration of neuromodulator)
	void updateGroupMonitor(int grpId = ALL);

	/*!
	 * \brief copy required spikes from firing buffer to spike buffer
	 *
	 * This function is public in SNN, but it should probably not be a public user function in CARLsim.
	 * It is usually called once every 1000ms by the core to update spike binaries and SpikeMonitor objects. In GPU
	 * mode, it will first copy the firing info to the host. The input argument can either be a specific group ID or
	 * keyword ALL (for all groups).
	 * Core and utility functions can call updateSpikeMonitor at any point in time. The function will automatically
	 * determine the last time it was called, and update SpikeMonitor information only if necessary.
	 */
	void updateSpikeMonitor(int grpId = ALL);

	/*!
	* \brief copy required neuron state values from ??? buffer to ??? buffer
	*
	* This function is public in SNN, but it should probably not be a public user function in CARLsim.
	* It is usually called once every 1000ms by the core to update neuron state value binaries and NeuronMonitor objects. In GPU
	* mode, it will first copy the neuron state info to the host. The input argument can either be a specific group ID or
	* keyword ALL (for all groups).
	* Core and utility functions can call updateNeuronMonitor at any point in time. The function will automatically
	* determine the last time it was called, and update SpikeMonitor information only if necessary.
	*/
	void updateNeuronMonitor(int grpId = ALL);

	//! stores the pre and post synaptic neuron ids with the weight and delay
	/*
	 * \param fid file pointer
	 */
	void saveSimulation(FILE* fid, bool saveSynapseInfo = false);

	//! function writes population weights from gIDpre to gIDpost to file fname in binary.
	//void writePopWeights(std::string fname, int gIDpre, int gIDpost);


	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! returns file pointer to info log
	const FILE* getLogFpInf() { return fpInf_; }
	//! returns file pointer to error log
	const FILE* getLogFpErr() { return fpErr_; }
	//! returns file pointer to debug log
	const FILE* getLogFpDeb() { return fpDeb_; }
	//! returns file pointer to log file
	const FILE* getLogFpLog() { return fpLog_; }

	/*!
	 * \brief Sets the file pointers for all log files
	 * file pointer NULL means don't change it.
	 */
	void setLogsFp(FILE* fpInf = NULL, FILE* fpErr = NULL, FILE* fpDeb = NULL, FILE* fpLog = NULL);


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	short int getConnectId(int grpIdPre, int grpIdPost); //!< find connection ID based on pre-post group pair, O(N)
	ConnectConfig getConnectConfig(short int connectId); //!< required for homeostasis

	//! returns the RangeDelay struct of a connection
	RangeDelay getDelayRange(short int connId);

	//! Returns the delay information for all synaptic connections between a pre-synaptic and a post-synaptic neuron group
	/*!
	 * \param gGrpIdPre ID of pre-synaptic group
	 * \param gGrpIdPost ID of post-synaptic group
	 * \param numPreN return the number of pre-synaptic neurons
	 * \param numPostN retrun the number of post-synaptic neurons
	 * \param delays (optional) return the delay information for all synapses, default = NULL
	 * \return delays information for all synapses
	 */
	uint8_t* getDelays(int gGrpIdPre, int gGrpIdPost, int& numPreN, int& numPostN);

	Grid3D getGroupGrid3D(int grpId);
	int getGroupId(std::string grpName);
	std::string getGroupName(int grpId);
	GroupSTDPInfo getGroupSTDPInfo(int grpId);
	GroupNeuromodulatorInfo getGroupNeuromodulatorInfo(int grpId);

	LoggerMode getLoggerMode() { return loggerMode_; }

	// get functions for GroupInfo
	int getGroupStartNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gStartN; }
	int getGroupEndNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gEndN; }
	int getGroupNumNeurons(int gGrpId) { return groupConfigMap[gGrpId].numN; }

	std::string getNetworkName() { return networkName_; }

	Point3D getNeuronLocation3D(int neurId);
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	int getNumConnections() { return numConnections; }
	int getNumSynapticConnections(short int connectionId);		//!< gets number of connections associated with a connection ID
	int getNumCompartmentConnections() { return numCompartmentConnections; }
	int getNumGroups() { return numGroups; }
	int getNumNeurons() { return glbNetworkConfig.numN; }
	int getNumNeuronsReg() { return glbNetworkConfig.numNReg; }
	int getNumNeuronsRegExc() { return glbNetworkConfig.numNExcReg; }
	int getNumNeuronsRegInh() { return glbNetworkConfig.numNInhReg; }
	int getNumNeuronsGen() { return glbNetworkConfig.numNPois; }
	int getNumNeuronsGenExc() { return glbNetworkConfig.numNExcPois; }
	int getNumNeuronsGenInh() { return glbNetworkConfig.numNInhPois; }
	int getNumSynapses() { return glbNetworkConfig.numSynNet; }

	int getRandSeed() { return randSeed_; }

	int getSimTime() { return simTime; }
	int getSimTimeSec() { return simTimeSec; }
	int getSimTimeMs() { return simTimeMs; }

	//! Returns pointer to existing SpikeMonitor object, NULL else
	SpikeMonitor* getSpikeMonitor(int grpId);

	//! Returns pointer to existing SpikeMonitorCore object, NULL else.
	//! Should not be exposed to user interface
	SpikeMonitorCore* getSpikeMonitorCore(int grpId);

	//! Returns pointer to existing NeuronMonitor object, NULL else
	NeuronMonitor* getNeuronMonitor(int grpId);

	//! Returns pointer to existing NeuronMonitorCore object, NULL else.
	//! Should not be exposed to user interface
	NeuronMonitorCore* getNeuronMonitorCore(int grpId);

	//! temporary getter to return pointer to current[] \TODO replace with NeuronMonitor
	float* getCurrent() { return managerRuntimeData.current; }

	std::vector< std::vector<float> > getWeightMatrix2D(short int connId);

	std::vector<float> getConductanceAMPA(int grpId);
	std::vector<float> getConductanceNMDA(int grpId);
	std::vector<float> getConductanceGABAa(int grpId);
	std::vector<float> getConductanceGABAb(int grpId);

	//! temporary getter to return pointer to stpu[] \TODO replace with NeuronMonitor or ConnectionMonitor
	float* getSTPu() { return managerRuntimeData.stpu; }

	//! temporary getter to return pointer to stpx[] \TODO replace with NeuronMonitor or ConnectionMonitor
	float* getSTPx() { return managerRuntimeData.stpx; }

	//! returns whether synapses in connection are fixed (false) or plastic (true)
	bool isConnectionPlastic(short int connId);

	//! returns RangeWeight struct of a connection
	RangeWeight getWeightRange(short int connId);

	bool isExcitatoryGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_AMPA) || (groupConfigMap[gGrpId].type & TARGET_NMDA); }
	bool isInhibitoryGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_GABAa) || (groupConfigMap[gGrpId].type & TARGET_GABAb); }
	bool isPoissonGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & POISSON_NEURON); }
	bool isDopaminergicGroup(int gGrpId) { return (groupConfigMap[gGrpId].type & TARGET_DA); }

	//! returns whether group has homeostasis enabled (true) or not (false)
	bool isGroupWithHomeostasis(int grpId);

	//! checks whether a point pre lies in the receptive field for point post
	double getRFDist3D(const RadiusRF& radius, const Point3D& pre, const Point3D& post);
	bool isPoint3DinRF(const RadiusRF& radius, const Point3D& pre, const Point3D& post);

	bool isSimulationWithCompartments() { return sim_with_compartments; }
	bool isSimulationWithCOBA() { return sim_with_conductances; }
	bool isSimulationWithCUBA() { return !sim_with_conductances; }
	bool isSimulationWithNMDARise() { return sim_with_NMDA_rise; }
	bool isSimulationWithGABAbRise() { return sim_with_GABAb_rise; }
	bool isSimulationWithFixedWeightsOnly() { return sim_with_fixedwts; }
	bool isSimulationWithHomeostasis() { return sim_with_homeostasis; }
	bool isSimulationWithPlasticWeights() { return !sim_with_fixedwts; }
	bool isSimulationWithSTDP() { return sim_with_stdp; }
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
	void generateCompConnectionRuntime(int netId);

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

	//! performs consistency checks for compartmentally enabled neurons
	void verifyCompartments();

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

	float getCompCurrent(int netid, int lGrpId, int lneurId, float const0 = 0.0f, float const1 = 0.0f);

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
	void fetchNeuronStateBuffer(int netId, int lGrpId);
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
	void copyNeuronStateBuffer(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
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
	void copyNeuronStateBuffer(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) { assert(false); }
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

	// runNetwork functions - multithreaded in LINUX using pthreads
#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
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
#else // for APPLE and Win systems - returns a void* to pthread_create - only differ in the return type compared to the counterparts above
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
	void copyNeuronStateBuffer(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
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
	int numCompartmentConnections; //!< number of connectCompartment calls

	std::map<int, GroupConfig> groupConfigMap;   //!< the hash table storing group configs created at CONFIG_STATE
	std::map<int, GroupConfigMD> groupConfigMDMap; //!< the hash table storing group configs meta data generated at SETUP_STATE
	std::map<int, ConnectConfig> connectConfigMap; //!< the hash table storing connection configs created at CONFIG_STATE
	std::map<int, compConnectConfig> compConnectConfigMap; //!< the hash table storing compConnection configs created at CONFIG_STATE

	// data structure assisting network partitioning
	std::list<GroupConfigMD> groupPartitionLists[MAX_NET_PER_SNN];
	std::list<ConnectConfig> localConnectLists[MAX_NET_PER_SNN];
	std::list<ConnectConfig> externalConnectLists[MAX_NET_PER_SNN];
	std::list<compConnectConfig> localCompConnectLists[MAX_NET_PER_SNN];

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

	bool sim_with_compartments;
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
	SpikeMonitorCore*  spikeMonCoreList[MAX_GRP_PER_SNN];
	SpikeMonitor*      spikeMonList[MAX_GRP_PER_SNN];

	// neuron monitor variables
	int numNeuronMonitor;
	NeuronMonitor*     neuronMonList[MAX_GRP_PER_SNN];
	NeuronMonitorCore* neuronMonCoreList[MAX_GRP_PER_SNN];

	// \FIXME \DEPRECATED this one moved to group-based
	long int    simTimeLastUpdSpkMon_; //!< last time we ran updateSpikeMonitor

	int numSpikeGenGrps;

	// keep track of number of GroupMonitor/GroupMonitorCore objects
	int numGroupMonitor;
	GroupMonitorCore*	groupMonCoreList[MAX_GRP_PER_SNN];
	GroupMonitor*		groupMonList[MAX_GRP_PER_SNN];

	// neuron monitor variables
	//NeuronMonitorCore* neurBufferCallback[MAX_]
	//int numNeuronMonitor;

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
