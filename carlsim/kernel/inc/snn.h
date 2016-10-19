/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 5/22/2015
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
	 * \param simMode simulation mode, CPU_MODE: running simluation on CPUs, GPU_MODE: running simulation with GPU
	 * acceleration, default = CPU_MODE
	 * \param loggerMode log mode
	 * \param numGPUs
	 * \param randSeed randomize seed of the random number generator
	 */
	SNN(const std::string& name, SimMode simMode, LoggerMode loggerMode, int numGPUs, int randSeed);

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
		uint8_t minDelay, uint8_t maxDelay, float radX, float radY, float radZ,
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

	//! Creates a group of Izhikevich spiking neurons
	/*!
	 * \param name the symbolic name of a group
	 * \param grid  Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferedGPU);

	//! Creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons)
	/*!
	 * \param name the symbolic name of a group
	 * \param grid Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron, currently only support EXCITATORY NEURON
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferedGPU);


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
	void biasWeights(short int connId, float bias, bool updateWeightRange=false);

	//! deallocates all dynamical structures and exits
	void exitSimulation(int val=1);

	//! reads the network state from file
	//! Reads a CARLsim network file. Such a file can be created using SNN:writeNetwork.
	/*
	 * \brief After calling SNN::loadSimulation, you should run SNN::runNetwork before calling fclose(fp).
	 * \param fid: file pointer
	 * \sa SNN::saveSimulation()
	 */
	 void loadSimulation(FILE* fid);

	/*!
	 * \brief reset Spike Counter to zero
	 * Manually resets the spike buffers of a Spike Counter to zero (for a specific group).
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time.
	 * \param grpId the group for which to reset the spike counts. Set to ALL if you want to reset all Spike Counters.
	 */
	void resetSpikeCounter(int grpId);

	// multiplies every weight with a scaling factor
	void scaleWeights(short int connId, float scale, bool updateWeightRange=false);

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

	/*!
	 * \brief A Spike Counter keeps track of the number of spikes per neuron in a group.
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
	 */
	void setSpikeCounter(int grpId, int recordDur);

	//! sets up a spike generator
	void setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGenFunc);

	//! sets up a spike monitor registered with a callback to process the spikes, there can only be one SpikeMonitor per group
	/*!
	 * \param grpId ID of the neuron group
	 * \param spikeMon (optional) spikeMonitor class
	 * \return SpikeMonitor* pointer to a SpikeMonitor object
	 */
	SpikeMonitor* setSpikeMonitor(int gid, FILE* fid);

	//!Sets the Poisson spike rate for a group. For information on how to set up spikeRate, see Section Poisson spike generators in the Tutorial.
	/*!Input arguments:
	 * \param grpId ID of the neuron group
	 * \param spikeRate pointer to a PoissonRate instance
	 * \param refPeriod (optional) refractive period,  default = 1
	 */
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod);

	//! sets the weight value of a specific synapse
	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange=false);

	//! enters a testing phase, where all weight updates are disabled
	void startTesting(bool shallUpdateWeights=true);

	//! exits a testing phase, making weight updates possible again
	void stopTesting();

	//! polls connection weights
	void updateConnectionMonitor(short int connId=ALL);

	//! access group status (currently the concentration of neuromodulator)
	void updateGroupMonitor(int grpId=ALL);

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
	void updateSpikeMonitor(int grpId=ALL);

	//! Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	//! weight values back to their default values by setting resetWeights = true.
	void updateNetwork(bool resetFiringInfo, bool resetWeights);

	//! stores the pre and post synaptic neuron ids with the weight and delay
	/*
	 * \param fid file pointer
	 */
	void saveSimulation(FILE* fid, bool saveSynapseInfo=false);

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
	void setLogsFp(FILE* fpInf=NULL, FILE* fpErr=NULL, FILE* fpDeb=NULL, FILE* fpLog=NULL);


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	short int getConnectId(int grpIdPre, int grpIdPost); //!< find connection ID based on pre-post group pair, O(N)
	ConnectConfig getConnectConfig(short int connectId); //!< required for homeostasis

	//! returns the RangeDelay struct of a connection
	RangeDelay getDelayRange(short int connId);

	//! Returns the delay information for all synaptic connections between a pre-synaptic and a post-synaptic neuron group
	/*!
	 * \param _grpIdPre ID of pre-synaptic group
	 * \param _grpIdPost ID of post-synaptic group
	 * \param _nPre return the number of pre-synaptic neurons
	 * \param _nPost retrun the number of post-synaptic neurons
	 * \param _delays (optional) return the delay information for all synapses, default = NULL
	 * \return delay information for all synapses
	 */
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays);

	Grid3D getGroupGrid3D(int grpId);
	int getGroupId(std::string grpName);
	//GroupConfigRT getGroupConfig(int grpId);
	std::string getGroupName(int grpId);
	GroupSTDPInfo getGroupSTDPInfo(int grpId);
	GroupNeuromodulatorInfo getGroupNeuromodulatorInfo(int grpId);

	LoggerMode getLoggerMode() { return loggerMode_; }

	// get functions for GroupInfo
	int getGroupStartNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gStartN; }
	int getGroupEndNeuronId(int gGrpId) { return groupConfigMDMap[gGrpId].gEndN; }
	int getGroupNumNeurons(int grpId) { return groupConfigMap[grpId].numN; }

	std::string getNetworkName() { return networkName_; }

	Point3D getNeuronLocation3D(int neurId);
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	int getNumConnections() { return numConnections; }
	int getNumSynapticConnections(short int connectionId);		//!< gets number of connections associated with a connection ID
	int getNumGroups() { return numGroups; }
	int getNumNeurons() { return glbNetworkConfig.numN; }
	int getNumNeuronsReg() { return glbNetworkConfig.numNReg; }
	int getNumNeuronsRegExc() { return glbNetworkConfig.numNExcReg; }
	int getNumNeuronsRegInh() { return glbNetworkConfig.numNInhReg; }
	int getNumNeuronsGen() { return glbNetworkConfig.numNPois; }
	int getNumNeuronsGenExc() { return glbNetworkConfig.numNExcPois; }
	int getNumNeuronsGenInh() { return glbNetworkConfig.numNInhPois; }
	int getNumPreSynapses() { return networkConfigs[0].numPreSynNet; }
	int getNumPostSynapses() { return networkConfigs[0].numPostSynNet; }

	int getRandSeed() { return randSeed_; }

	SimMode getSimMode() { return simMode_; }
	int getSimTime() { return simTime; }
	int getSimTimeSec() { return simTimeSec; }
	int getSimTimeMs() { return simTimeMs; }

	//! Returns pointer to existing SpikeMonitor object, NULL else
	SpikeMonitor* getSpikeMonitor(int grpId);

	//! Returns pointer to existing SpikeMonitorCore object, NULL else.
	//! Should not be exposed to user interface
	SpikeMonitorCore* getSpikeMonitorCore(int grpId);

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

	bool isSimulationWithCOBA() { return sim_with_conductances; }
	bool isSimulationWithCUBA() { return !sim_with_conductances; }
	bool isSimulationWithNMDARise() { return sim_with_NMDA_rise; }
	bool isSimulationWithGABAbRise() { return sim_with_GABAb_rise; }
	bool isSimulationWithFixedWeightsOnly() { return sim_with_fixedwts; }
	bool isSimulationWithHomeostasis() { return sim_with_homeostasis; }
	bool isSimulationWithPlasticWeights() { return !sim_with_fixedwts; }
	bool isSimulationWithSTDP() { return sim_with_stdp; }
	bool isSimulationWithSTP() { return sim_with_stp; }

/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

private:
	// +++++ CPU MODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	
	//! all unsafe operations of constructor
	void SNNinit();

	//! allocates and initializes all core datastructures
	void allocateManagerRuntimeData();

	//! add the entry that the current neuron has spiked
	int  addSpikeToTable(int nId, int grpId);

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
	void collectGlobalNetworkConfig();
	void compileConnectConfig(); //!< for future use
	void compileGroupConfig();

	/*!
	 * \brief generate connections among groups according to connect configuration
	 */
	void connectNetwork();
	inline void connectNeurons(int netId, int srcGrp,  int destGrp, int srcN, int destN, short int connId, int externalNetId);
	void connectFull(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectOneToOne(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectRandom(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectGaussian(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void connectUserDefined(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal);
	void clearExtFiringTable();
	void clearExtFiringTable_GPU();

	void deleteObjects();			//!< deallocates all used data structures in snn_cpu.cpp

	void doCurrentUpdateD1(int netId);
	void doCurrentUpdateD2(int netId);
	void doGPUSim();
	void doCPUSim();
	void doSTPUpdateAndDecayCond();

	void findFiring();
	int findGrpId(int nid);//!< For the given neuron nid, find the group id

	void findMaxDelay(int* _maxDelay);
	//! find the maximum post-synaptic and pre-synaptic length
	//! this used to be in updateParameters
	void findMaxNumSynapsesGroups(int* _maxNumPostSynGrp, int* _maxNumPreSynGrp);
	void findMaxNumSynapsesNeurons(int _netId, int& _maxNumPostSynN, int& _maxNumPreSynN);
	void findMaxSpikesD1D2(int netId, unsigned int& _maxSpikesD1, unsigned int& _maxSpikesD2);
	void findNumSynapsesNetwork(int netId, int& _numPostSynNet, int& _numPreSynNet); //!< find the total number of synapses in the network
	void findNumN(int _netId, int& _numN, int& _nunNExternal, int& numNAssigned,
                  int& _numNReg, int& _numNExcReg, int& _numNInhReg,
                  int& _numNPois, int& _numNExcPois, int& _numNInhPois);
	void findNumNSpikeGen(int _netId, int& _numNSpikeGen);

	void generatePostSynapticSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, int tD);
	void fillSpikeGenBits(int netId);
	void userDefinedSpikeGenerator(int gGrpId);
	//void generateSpikesFromRate(int grpId);

	float generateWeight(int connProp, float initWt, float maxWt, int nid, int grpId);

	void globalStateUpdate();

	//! initialize GroupConfigRT structure
	void initGroupConfig(GroupConfigRT* groupConfig);

	//! performs various verification checkups before building the network
	void verifyNetwork();

	//! make sure STDP post-group has some incoming plastic connections
	void verifySTDP();

	//! make sure every group with homeostasis also has STDP
	void verifyHomeostasis();

	//! performs a consistency check to see whether numN* class members have been accumulated correctly
	void verifyNumNeurons();

	void compileSNN();
	
	void partitionSNN();

	void generateRuntimeSNN();

	void allocateSNN(int netId);


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

	// NOTE: all these printer functions should be in printSNNInfo.cpp
	// FIXME: are any of these actually supposed to be public?? they are not yet in carlsim.h
	void printConnection(const std::string& fname);
	void printConnection(FILE* fp);
	void printConnection(int grpId, FILE* fp); //!< print the connection info of grpId
	void printConnectionInfo(short int connId);
	void printConnectionInfo(int netId, std::list<ConnectConfig>::iterator connIt);
	void printConnectionInfo(FILE* fp);
	void printConnectionInfo2(FILE *fpg);
	void printCurrentInfo(FILE* fp); //!< for GPU debugging
	//void printFiringRate(char *fname=NULL);
	void printGroupInfo(int grpId);	//!< CARLSIM_INFO prints group info
	void printGroupInfo(int netId, std::list<GroupConfigMD>::iterator grpIt);
	void printGroupInfo2(FILE* fpg);
	void printMemoryInfo(FILE* fp); //!< prints memory info to file
	void printNetworkInfo(FILE* fp);
	//void printNeuronState(int grpId, FILE* fp);
	void printPostConnection(FILE* fp); //!< print all post connections
	void printPostConnection(int grpId, FILE* fp);
	int  printPostConnection2(int grpId, FILE* fpg);
	void printPreConnection(FILE* fp); //!< print all pre connections
	void printPreConnection(int grpId, FILE* fp);
	int  printPreConnection2(int grpId, FILE* fpg);
	void printSimSummary(); 	//!< prints a simulation summary at the end of sim
	//void printState(FILE* fp);
	void printStatusConnectionMonitor(int connId=ALL);
	void printStatusGroupMonitor(int grpId=ALL);
	void printStatusSpikeMonitor(int grpId=ALL);
	void printWeights(int preGrpId, int postGrpId=-1);

	int loadSimulation_internal(bool onlyPlastic);

	void resetConductances(int netId);
	void resetCPUTiming();
	void resetCurrent(int netId);
	void resetFiringInformation(); //!< resets the firing information when updateNetwork is called
	void resetGroupConfigs(bool deallocate = false);
	void resetNeuromodulator(int netId, int lGrpId);
	void resetNeuron(int netId, int lGrpId, int lNId);
	//void resetPointers(bool deallocate=false);
	void resetMonitors(bool deallocate=false);
	void resetConnectionConfigs(bool deallocate=false);
	void deleteManagerRuntimeData();
	void resetPoissonNeuron(int netId, int lGrpId, int lNId); //!< use local ids
	void resetPropogationBuffer();
	void resetSpikeCnt();					//!< Resets the spike count for a particular group.
	void resetSynapse(int netId, bool changeWeights=false);
	void resetTimeTable();
	void resetFiringTable();
	void routeSpikes();
	void routeSpikes_GPU();

	inline SynInfo SET_CONN_ID(int nid, int sid, int grpId);

	void setGrpTimeSlice(int grpId, int timeSlice); //!< used for the Poisson generator. TODO: further optimize
	int setRandSeed(int seed);	//!< setter function for const member randSeed_

	void startCPUTiming();
	void stopCPUTiming();

	void generateUserDefinedSpikes();

	void allocateManagerSpikeTables();
	//void updateStateAndFiringTable();
	bool updateTime(); //!< updates simTime, returns true when a new second is started
	// +++++ GPU MODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	// TODO: consider moving to snn_gpu.h

	void allocateGroupId(int netId);
	void allocateSNN_GPU(int netId); //!< allocates runtime data on GPU memory and initialize GPU
	void allocateSNN_CPU(int netId); //!< allocates runtime data on CPU memory
	int  allocateStaticLoad(int netId, int bufSize);

	void assignPoissonFiringRate();
	void assignPoissonFiringRate_GPU();

	void checkAndSetGPUDevice(int netId);
	void checkDestSrcPtrs(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId, int destOffset);
	void checkInitialization(char* testString=NULL);
	void checkInitialization2(char* testString=NULL);
	void configGPUDevice();

	void copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);
	void copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	void copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	void copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);

	/*!
	 * \brief Copy neuron parameters (Izhikevich params, baseFiring) from host to device pointer
	 *
	 * This function copies the neuron parameters (Izh_a, Izh_b, Izh_c, Izh_d, baseFiring, baseFiringInv) from the
	 * host variables to a device pointer.
	 *
	 * In contrast to other copy methods, this one does not need the source pointer or a cudaMemcpyKind, because the
	 * copy process can only go from host to device. Thus, the source is always Izh_a, Izh_b, etc. variables, and
	 * cudaMemcpyKind is always cudaMemcpyHostToDevice.
	 *
	 * If allocateMem is set to true, then cudaMalloc is called first on the data structures, before contents are
	 * copied. If allocateMem is false, only the contents will be copied.
	 *
	 * If grpId is set to -1, then the information of all groups will be copied. Otherwise the information of only
	 * a single group will be copied. If allocateMem is set to true, grpId must be set to -1 (must allocate all groups
	 * at the same time in order to avoid memory fragmentation).
	 */
	void copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);

	void copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	void copyNeuronState(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copyNeuronState(int netId, int lGrpId, RuntimeData* dest, bool allocateMem);
	void copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset);
	void copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem, int destOffset);
	void copySynapseState(int netId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem);
	void copySynapseState(int netId, RuntimeData* dest, bool allocateMem);
	void copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem);
	void copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, bool allocateMem);
	//void copyWeightsGPU(int nid, int src_grp);
	void copyWeightState(int netId, int lGrpId);
	void copyNetworkConfig(int netId);
	void copyGroupConfigs(int netId);
	void copyGrpIdsLookupArray(int netId);
	void copyConnIdsLookupArray(int netId);
	void copyLastSpikeTime(int netId);

	void deleteObjects_CPU();
	void deleteObjects_GPU();		//!< deallocates all used data structures in snn_gpu.cu
	void doCurrentUpdate();
	void doCurrentUpdate_GPU();
	void doSTPUpdateAndDecayCond_GPU();
	void dumpSpikeBuffToFile_GPU(int gid);
	void findFiring_GPU();

	// fetch functions supporting local-to-global copy
	void fetchConductanceAMPA(int gGrpId);
	void fetchConductanceNMDA(int gGrpId);
	void fetchConductanceGABAa(int gGrpId);
	void fetchConductanceGABAb(int gGrpId);
	void fetchNetworkSpikeCount();
	void fetchNeuronSpikeCount(int gGrpId);
	void fetchSTPState(int gGrpId);
	
	// fetch functions supporting local-to-local copy
	void fetchSpikeTables(int netId);
	void fetchGroupState(int netId, int lGrpId);
	void fetchWeightState(int netId, int lGrpId);
	void fetchGrpIdsLookupArray(int netId);
	void fetchConnIdsLookupArray(int netId);
	void fetchLastSpikeTime(int netId);

	void globalStateUpdate_GPU();
	void initGPU(int netId);

	//void resetFiringInformation_GPU(); //!< resets the firing information in GPU_MODE when updateNetwork is called
	void resetGPUTiming();
	void resetSpikeCnt_GPU(int gGrpId); //!< Utility function to clear spike counts in the GPU code.

	/*!
	 * \brief reset spike counter to zero in GPU mode
	 *
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time through calling the public equivalent. This one gets called in
	 * SNN::resetSpikeCounter if we're running GPU mode.
	 * \param grpId	the group for which you want to reset the spikes
	 */
	void resetSpikeCounter_GPU(int grpId);

	void spikeGeneratorUpdate();
	void spikeGeneratorUpdate_GPU();
	void startGPUTiming();
	void stopGPUTiming();
	void shiftSpikeTables();
	void shiftSpikeTables_GPU();
	void updateNetwork_GPU(bool resetFiringInfo); //!< Allows parameters to be reset in the middle of the simulation
	void updateWeights();
	void updateWeights_GPU();
	void updateTimingTable();
	void updateTimingTable_GPU();
	
	// Utility functions
	void firingUpdateSTP(int lNId, int lGrpId, int netId);
	void updateLTP(int lNId, int lGrpId, int netId);
	void resetFiredNeuron(int lNId, short int lGrpId, int netId);
	bool getPoissonSpike(int lNId, int netId);
	bool getSpikeGenBit(unsigned int nIdPos, int netId);

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	SNNState snnState; //!< state of the network
	FILE* loadSimFID;

	const std::string networkName_;	//!< network name
	const SimMode simMode_;		//!< current simulation mode (CPU_MODE or GPU_MODE) FIXME: give better name
	const LoggerMode loggerMode_;	//!< current logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	const int randSeed_;			//!< random number seed to use
	int numGPUs_;				//!< on which CUDA device to establish a context (only in GPU_MODE)

	bool simulatorDeleted;
	bool spikeRateUpdated;

	//! vairables for tracking performance
	float prevCpuExecutionTime;
	float cpuExecutionTime;
	float prevGpuExecutionTime;
	float gpuExecutionTime;

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

	// spiking neural network related information, including neurons, synapses and network parameters
	//int maxDelay_;        //!< maximum axonal delay in the global network

	//int numN;             //!< number of neurons in the spiking neural network
	//int numNReg;          //!< number of regular (spking) neurons
	//int numNExcReg;       //!< number of regular excitatory neurons
	//int numNInhReg;       //!< number of regular inhibitory neurons
	//int numNExcPois;      //!< number of excitatory poisson neurons
	//int numNInhPois;      //!< number of inhibitory poisson neurons
	//int numNPois;         //!< number of poisson neurons
	GlobalNetworkConfig glbNetworkConfig;

	//time and timestep
	int simTimeRunStart; //!< the start time of current/last runNetwork call
	int simTimeRunStop;  //!< the end time of current/last runNetwork call
	int simTimeLastRunSummary; //!< the time at which the last run summary was printed
	int simTimeMs;      //!< The simulation time showing milliseconds within a second
	int simTimeSec;     //!< The simulation time showing seconds in a simulation
	int simTime;        //!< The absolute simulation time. The unit is millisecond. this value is not reset but keeps increasing to its max value.

	// cuda keep track of performance...
	StopWatchInterface* timer;
	float cumExecutionTime;
	float lastExecutionTime;

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

	RuntimeData gpuRuntimeData[MAX_NET_PER_SNN];
	RuntimeData cpuRuntimeData[MAX_NET_PER_SNN];
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
		//int glbNumGroups;
		//int glbNumConnections;
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
