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

#include <carlsim.h>
#include <callback_core.h>

#include <snn_definitions.h>
#include <snn_datastructures.h>

#include <propagated_spike_buffer.h>
#include <poisson_rate.h>
#include <gpu_random.h>

class SpikeMonitor;
class SpikeMonitorCore;
class ConnectionMonitorCore;
class ConnectionMonitor;


/// **************************************************************************************************************** ///
/// CPUSNN CORE CLASS
/// **************************************************************************************************************** ///

/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
class CpuSNN {

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
	 * \param ithGPU
	 * \param randSeed randomize seed of the random number generator
	 */
	CpuSNN(const std::string& name, simMode_t simMode, loggerMode_t loggerMode, int ithGPU, int randSeed);

	//! SNN Destructor
	/*!
	 * \brief clean up all allocated resource
	 */
	~CpuSNN();

	// +++++ PUBLIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const static unsigned int MAJOR_VERSION = 3; //!< major release version, as in CARLsim X
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
	 * \param _maxPostM: (optional) maximum number of post-synaptic connections (per neuron), Set to 0 for no limit, default = 0
	 * \param _maxPreM: (optional) maximum number of pre-synaptic connections (per neuron), Set to 0 for no limit, default = 0.
	 * \return number of created synaptic projections
	 */
	short int connect(int gIDpre, int gIDpost, ConnectionGeneratorCore* conn, float mulSynFast, float mulSynSlow,
		bool synWtType,	int maxM, int maxPreM);

	//! Creates a group of Izhikevich spiking neurons
	/*!
	 * \param name the symbolic name of a group
	 * \param grid  Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron
	 */
	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType);

	//! Creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons)
	/*!
	 * \param name the symbolic name of a group
	 * \param grid Grid3D struct to create neurons on a 3D grid (x,y,z)
	 * \param nType the type of neuron, currently only support EXCITATORY NEURON
	 */
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType);


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
	void setESTDP(int grpId, bool isSet, stdpType_t type, stdpCurve_t curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma);

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
	void setISTDP(int grpId, bool isSet, stdpType_t type, stdpCurve_t curve, float ab1, float ab2, float tau1, float tau2);

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
	void setWeightAndWeightChangeUpdate(updateInterval_t wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay);

	// +++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for n sec
	 *
	 * \param[in] printRunSummary whether to print a basic summary of the run at the end
	 * \param[in] copyState 	enable copying of data from device to host
	 */
	int runNetwork(int _nsec, int _nmsec, bool printRunSummary, bool copyState);

	/*!
	 * \brief build the network
	 * \param[in] removeTempMemory 	remove temp memory after building network
	 */
	void setupNetwork(bool removeTempMemory);

	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// adds a bias to every weight in the connection
	void biasWeights(short int connId, float bias, bool updateWeightRange=false);

	//! deallocates all dynamical structures and exits
	void exitSimulation(int val=1);

	//! reads the network state from file
	//! Reads a CARLsim network file. Such a file can be created using CpuSNN:writeNetwork.
	/*
	 * \brief After calling CpuSNN::loadSimulation, you should run CpuSNN::runNetwork before calling fclose(fp).
	 * \param fid: file pointer
	 * \sa CpuSNN::saveSimulation()
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
//	void setConnectionMonitor(int grpIdPre, int grpIdPost, ConnectionMonitorCore* connectionMon);
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
	void setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGen);

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

	// sets the weight value of a specific synapse
	void setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange=false);

	//! polls connection weights
	void updateConnectionMonitor(short int connId=ALL);

	//! access group status (currently the concentration of neuromodulator)
	void updateGroupMonitor(int grpId=ALL);

	/*!
	 * \brief copy required spikes from firing buffer to spike buffer
	 *
	 * This function is public in CpuSNN, but it should probably not be a public user function in CARLsim.
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
	void writePopWeights(std::string fname, int gIDpre, int gIDpost);


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
	grpConnectInfo_t* getConnectInfo(short int connectId); //!< required for homeostasis

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
	group_info_t getGroupInfo(int groupId);
	std::string getGroupName(int grpId);
	GroupSTDPInfo_t getGroupSTDPInfo(int grpId);
	GroupNeuromodulatorInfo_t getGroupNeuromodulatorInfo(int grpId);

	loggerMode_t getLoggerMode() { return loggerMode_; }

	// get functions for GroupInfo
	int getGroupStartNeuronId(int grpId)  { return grp_Info[grpId].StartN; }
	int getGroupEndNeuronId(int grpId)    { return grp_Info[grpId].EndN; }
	int getGroupNumNeurons(int grpId)     { return grp_Info[grpId].SizeN; }

	std::string getNetworkName() { return networkName_; }

	Point3D getNeuronLocation3D(int neurId);
	Point3D getNeuronLocation3D(int grpId, int relNeurId);

	int getNumConnections() { return numConnections; }
	int getNumSynapticConnections(short int connectionId);		//!< gets number of connections associated with a connection ID
	int getNumGroups() { return numGrp; }
	int getNumNeurons() { return numN; }
	int getNumNeuronsReg() { return numNReg; }
	int getNumNeuronsRegExc() { return numNExcReg; }
	int getNumNeuronsRegInh() { return numNInhReg; }
	int getNumNeuronsGen() { return numNPois; }
	int getNumNeuronsGenExc() { return numNExcPois; }
	int getNumNeuronsGenInh() { return numNInhPois; }
	int getNumPreSynapses() { return preSynCnt; }
	int getNumPostSynapses() { return postSynCnt; }

	int getRandSeed() { return randSeed_; }

	simMode_t getSimMode()		{ return simMode_; }
	unsigned int getSimTime()		{ return simTime; }
	unsigned int getSimTimeSec()	{ return simTimeSec; }
	unsigned int getSimTimeMs()		{ return simTimeMs; }

	//! returns pointer to existing SpikeMonitor object, NULL else
	SpikeMonitor* getSpikeMonitor(int grpId);

	/*!
	 * \brief return the number of spikes per neuron for a certain group
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur) at any point in time.
	 * \param grpId	the group for which you want the spikes
	 * \return pointer to array of ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (int) since the last reset.
	 */
	int* getSpikeCounter(int grpId);

	//! temporary getter to return pointer to current[] \TODO replace with NeuronMonitor
	float* getCurrent() { return current; }

	std::vector< std::vector<float> > getWeightMatrix2D(short int connId);

	std::vector<float> getConductanceAMPA(int grpId);
	std::vector<float> getConductanceNMDA(int grpId);
	std::vector<float> getConductanceGABAa(int grpId);
	std::vector<float> getConductanceGABAb(int grpId);

	//! temporary getter to return pointer to stpu[] \TODO replace with NeuronMonitor or ConnectionMonitor
	float* getSTPu() { return stpu; }

	//! temporary getter to return pointer to stpx[] \TODO replace with NeuronMonitor or ConnectionMonitor
	float* getSTPx() { return stpx; }

    bool isConnectionPlastic(short int connId);

	//! returns RangeWeight struct of a connection
	RangeWeight getWeightRange(short int connId);

	bool isExcitatoryGroup(int g) { return (grp_Info[g].Type&TARGET_AMPA) || (grp_Info[g].Type&TARGET_NMDA); }
	bool isInhibitoryGroup(int g) { return (grp_Info[g].Type&TARGET_GABAa) || (grp_Info[g].Type&TARGET_GABAb); }
	bool isPoissonGroup(int g) { return (grp_Info[g].Type&POISSON_NEURON); }
	bool isDopaminergicGroup(int g) { return (grp_Info[g].Type&TARGET_DA); }

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
	void CpuSNNinit();	//!< all unsafe operations of constructor

	void buildNetworkInit(unsigned int nNeur, unsigned int nPostSyn, unsigned int nPreSyn, unsigned int maxDelay);

	int  addSpikeToTable(int id, int g); //!< add the entry that the current neuron has spiked

	void buildGroup(int groupId);
	void buildNetwork();
	void buildPoissonGroup(int groupId);

	/*!
	 * \brief reset Spike Counters to zero if simTime % recordDur == 0
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur)
	 * after this period of time, the spike buffers need to be reset
	 * this function checks simTime vs. recordDur and resets the spike buffers if necessary
	 */
	void checkSpikeCounterRecordDur();

	void compactConnections(); //!< minimize any other wastage in that array by compacting the store
	void connectFull(grpConnectInfo_t* info);
	void connectOneToOne(grpConnectInfo_t* info);
	void connectRandom(grpConnectInfo_t* info);
	void connectGaussian(grpConnectInfo_t* info);
	void connectUserDefined(grpConnectInfo_t* info);

	void deleteObjects();			//!< deallocates all used data structures in snn_cpu.cpp

	void doD1CurrentUpdate();
	void doD2CurrentUpdate();
	void doGPUSim();
	void doSnnSim();
	void doSTPUpdateAndDecayCond();

	void findFiring();
	int findGrpId(int nid);//!< For the given neuron nid, find the group id

	//! finds the maximum post-synaptic and pre-synaptic length
	//! this used to be in updateParameters
	void findMaxNumSynapses(int* numPostSynapses, int* numPreSynapses);

	void generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD);
	void generateSpikes();
	void generateSpikes(int grpId);
	void generateSpikesFromFuncPtr(int grpId);
	void generateSpikesFromRate(int grpId);

	int getPoissNeuronPos(int nid);
	float getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId);

	void globalStateUpdate();

	void initSynapticWeights(); //!< initialize all the synaptic weights to appropriate values. total size of the synaptic connection is 'length'

	//! performs a consistency check to see whether numN* class members have been accumulated correctly
	bool isNumNeuronsConsistent();

	void makePtrInfo();				//!< creates CPU net ptrs

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
	unsigned int poissonSpike(unsigned int currTime, float frate, int refractPeriod);

	// NOTE: all these printer functions should be in printSNNInfo.cpp
	// FIXME: are any of these actually supposed to be public?? they are not yet in carlsim.h
	void printConnection(const std::string& fname);
	void printConnection(FILE* fp);
	void printConnection(int grpId, FILE* fp); //!< print the connection info of grpId
	void printConnectionInfo(short int connId);
	void printConnectionInfo(FILE* fp);
	void printConnectionInfo2(FILE *fpg);
	void printCurrentInfo(FILE* fp); //!< for GPU debugging
	void printFiringRate(char *fname=NULL);
	void printGroupInfo(int grpId);	//!< CARLSIM_INFO prints group info
	void printGroupInfo2(FILE* fpg);
	void printMemoryInfo(FILE* fp); //!< prints memory info to file
	void printNetworkInfo(FILE* fp);
	void printNeuronState(int grpId, FILE* fp);
	void printParameters(FILE *fp);
	void printPostConnection(FILE* fp); //!< print all post connections
	void printPostConnection(int grpId, FILE* fp);
	int  printPostConnection2(int grpId, FILE* fpg);
	void printPreConnection(FILE* fp); //!< print all pre connections
	void printPreConnection(int grpId, FILE* fp);
	int  printPreConnection2(int grpId, FILE* fpg);
	void printSimSummary(); 	//!< prints a simulation summary at the end of sim
	void printState(FILE* fp);
	void printStatusConnectionMonitor(int connId=ALL);
	void printStatusGroupMonitor(int grpId=ALL);
	void printStatusSpikeMonitor(int grpId=ALL);
	void printTuningLog(FILE* fp);
	void printWeights(int preGrpId, int postGrpId=-1);

	int loadSimulation_internal(bool onlyPlastic);

	void reorganizeDelay();
	void reorganizeNetwork(bool removeTempMemory);

	void resetConductances();
	void resetCounters();
	void resetCPUTiming();
	void resetCurrent();
	void resetFiringInformation(); //!< resets the firing information when updateNetwork is called
	void resetGroups();
	void resetNeuromodulator(int grpId);
	void resetNeuron(unsigned int nid, int grpId);
	void resetPointers(bool deallocate=false);
	void resetPoissonNeuron(unsigned int nid, int grpId);
	void resetPropogationBuffer();
	void resetSpikeCnt(int grpId=ALL);					//!< Resets the spike count for a particular group.
	void resetSynapticConnections(bool changeWeights=false);
	void resetTimingTable();

	post_info_t SET_CONN_ID(int nid, int sid, int grpId);

	inline void setConnection(int srcGrpId, int destGrpId, unsigned int src, unsigned int dest, float synWt,
		float maxWt, uint8_t dVal, int connProp, short int connId);

	void setGrpTimeSlice(int grpId, int timeSlice); //!< used for the Poisson generator. TODO: further optimize
	int setRandSeed(int seed);	//!< setter function for const member randSeed_

	void startCPUTiming();
	void stopCPUTiming();

	void swapConnections(int nid, int oldPos, int newPos);

	void updateAfterMaxTime();
	void updateSpikesFromGrp(int grpId);
	void updateSpikeGenerators();
	void updateSpikeGeneratorsInit();

	int  updateSpikeTables();
	//void updateStateAndFiringTable();
	bool updateTime(); //!< updates simTime, returns true when a new second is started

	// +++++ GPU MODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	// TODO: consider moving to snn_gpu.h
	void CpuSNNinit_GPU();	//!< initializes params needed in snn_gpu.cu (gets called in CpuSNN constructor)

	void allocateGroupId();
	void allocateNetworkParameters();
	void allocateSNN_GPU(); //!< allocates required memory and then initialize the GPU
	int  allocateStaticLoad(int bufSize);

	void assignPoissonFiringRate_GPU();

	void checkAndSetGPUDevice();
	void checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId);
	void checkInitialization(char* testString=NULL);
	void checkInitialization2(char* testString=NULL);
	void configGPUDevice();

	void copyConductanceAMPA(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyConductanceNMDA(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyConductanceGABAa(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyConductanceGABAb(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyConductanceState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyConnections(network_ptr_t* dest, int kind, bool allocateMem);
	void copyExternalCurrent(network_ptr_t* dest, network_ptr_t* src, bool allocateMem, int grpId=-1);
	void copyFiringInfo_GPU();
	void copyFiringStateFromGPU (int grpId = -1);
	void copyGroupState(network_ptr_t* dest, network_ptr_t* src,  cudaMemcpyKind kind, bool allocateMem, int grpId=-1);

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
	void copyNeuronParametersFromHostToDevice(network_ptr_t* dest, bool allocateMem, int grpId=-1);

	void copyNeuronState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem, int grpId=-1);
	void copyParameters();
	void copyPostConnectionInfo(network_ptr_t* dest, bool allocateMem);
	void copyState(network_ptr_t* dest, bool allocateMem);
	void copySTPState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, bool allocateMem);
	void copyWeightsGPU(unsigned int nid, int src_grp);
	void copyWeightState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, //!< copy presynaptic info
		bool allocateMem, int grpId=-1);

	void deleteObjects_GPU();		//!< deallocates all used data structures in snn_gpu.cu
	void doCurrentUpdate_GPU();
	void doSTPUpdateAndDecayCond_GPU(int gridSize=64, int blkSize=128);
	void dumpSpikeBuffToFile_GPU(int gid);
	void findFiring_GPU(int gridSize=64, int blkSize=128);

	/*!
	 * \brief return the number of spikes per neuron for a certain group in GPU mode
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur)
	 * at any point in time.
	 * \param grpId	the group for which you want the spikes
	 * \return pointer to array of unsigned ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (unsigned int) since the last reset.
	 */
	int* getSpikeCounter_GPU(int grpId);

	void globalStateUpdate_GPU();
	void initGPU(int gridSize, int blkSize);

	void resetFiringInformation_GPU(); //!< resets the firing information in GPU_MODE when updateNetwork is called
	void resetGPUTiming();
	void resetSpikeCnt_GPU(int _startGrp, int _endGrp); //!< Utility function to clear spike counts in the GPU code.

	/*!
	 * \brief reset spike counter to zero in GPU mode
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time through calling the public equivalent. This one gets called in
	 * CpuSNN::resetSpikeCounter if we're running GPU mode.
	 * \param grpId	the group for which you want to reset the spikes
	 */
	void resetSpikeCounter_GPU(int grpId);

	void setSpikeGenBit_GPU(unsigned int nid, int grp);
	void spikeGeneratorUpdate_GPU();
	void startGPUTiming();
	void stopGPUTiming();
	void updateFiringTable();
	void updateFiringTable_GPU();
	void updateNetwork_GPU(bool resetFiringInfo); //!< Allows parameters to be reset in the middle of the simulation
	void updateWeights();
	void updateWeights_GPU();
	//void updateStateAndFiringTable_GPU();
	void updateTimingTable_GPU();


	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	FILE* loadSimFID;

	const std::string networkName_;	//!< network name
	const simMode_t simMode_;		//!< current simulation mode (CPU_MODE or GPU_MODE) FIXME: give better name
	const loggerMode_t loggerMode_;	//!< current logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	const int ithGPU_;				//!< on which CUDA device to establish a context (only in GPU_MODE)
	const int randSeed_;			//!< random number seed to use


	//! temporary variables created and deleted by network after initialization
	uint8_t			*tmp_SynapticDelay;

	bool simulatorDeleted;
	bool spikeRateUpdated;

	float prevCpuExecutionTime;
	float cpuExecutionTime;
	float prevGpuExecutionTime;
	float gpuExecutionTime;


	//! properties of the network (number of groups, network name, allocated neurons etc..)
	bool			doneReorganization;
	bool			memoryOptimized;

	int				numGrp;
	int				numConnections;		//!< number of connection calls (as in snn.connect(...))
	//! keeps track of total neurons/presynapses/postsynapses currently allocated
	unsigned int	allocatedN;
	unsigned int	allocatedPre;
	unsigned int	allocatedPost;

	grpConnectInfo_t* connectBegin;
	short int 	*cumConnIdPre;		//!< connId, per synapse, presynaptic cumulative indexing
	float 		*mulSynFast;	//!< scaling factor for fast synaptic currents, per connection
	float 		*mulSynSlow;	//!< scaling factor for slow synaptic currents, per connection

	short int *grpIds;

	//! Buffer to store spikes
	PropagatedSpikeBuffer* pbuf;

	bool sim_with_conductances;		//!< flag to inform whether we run in COBA mode (true) or CUBA mode (false)
	bool sim_with_NMDA_rise;	//!< a flag to inform whether to compute NMDA rise time
	bool sim_with_GABAb_rise;	//!< a flag to inform whether to compute GABAb rise time
	double dAMPA;				//!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	double rNMDA;				//!< multiplication factor for rise time of NMDA
	double dNMDA;				//!< multiplication factor for decay time of NMDA
	double sNMDA;				//!< scaling factor for NMDA amplitude
	double dGABAa;				//!< multiplication factor for decay time of GABAa
	double rGABAb;				//!< multiplication factor for rise time of GABAb
	double dGABAb;				//!< multiplication factor for decay time of GABAb
	double sGABAb;				//!< scaling factor for GABAb amplitude

	bool sim_with_fixedwts;
	bool sim_with_stdp;
	bool sim_with_modulated_stdp;
	bool sim_with_homeostasis;
	bool sim_with_stp;
	bool sim_with_spikecounters; //!< flag will be true if there are any spike counters around

	// spiking neural network related information, including neurons, synapses and network parameters
	int	        	numN;				//!< number of neurons in the spiking neural network
	int				numPostSynapses;	//!< maximum number of post-synaptic connections in groups
	int				numPreSynapses;		//!< maximum number of pre-syanptic connections in groups
	int				maxDelay_;					//!< maximum axonal delay in groups
	int				numNReg;			//!< number of regular (spking) neurons
	int				numNExcReg;			//!< number of regular excitatory neurons
	int				numNInhReg;			//!< number of regular inhibitory neurons
	int   			numNExcPois;		//!< number of excitatory poisson neurons
	int				numNInhPois;		//!< number of inhibitory poisson neurons
	int				numNPois;			//!< number of poisson neurons
	float       	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current, *extCurrent;
	bool			*curSpike;
	int         	*nSpikeCnt;     //!< spike counts per neuron
	unsigned short       	*Npre;			//!< stores the number of input connections to the neuron
	unsigned short			*Npre_plastic;	//!< stores the number of excitatory input connection to the input
	unsigned short       	*Npost;			//!< stores the number of output connections from a neuron.
	uint32_t    	*lastSpikeTime;	//!< stores the most recent spike time of the neuron
	float			*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
	float	 		*maxSynWt;		//!< maximum synaptic weight for given connection..
	uint32_t    	*synSpikeTime;	//!< stores the spike time of each synapse
	unsigned int		postSynCnt; //!< stores the total number of post-synaptic connections in the network
	unsigned int		preSynCnt; //!< stores the total number of pre-synaptic connections in the network
	#ifdef NEURON_NOISE
	float			*intrinsicWeight;
	#endif
	//added to include homeostasis. -- KDC
	float					*baseFiring;
	float                 *avgFiring;
	unsigned int		*cumulativePost;
	unsigned int		*cumulativePre;
	post_info_t		*preSynapticIds;
	post_info_t		*postSynapticIds;		//!< 10 bit syn id, 22 bit neuron id, ordered based on delay
	delay_info_t    *postDelayInfo;      	//!< delay information

	//! size of memory used for different parts of the network
	typedef struct snnSize_s {
		unsigned int		neuronInfoSize;
		unsigned int		synapticInfoSize;
		unsigned int		networkInfoSize;
		unsigned int		spikingInfoSize;
		unsigned int		debugInfoSize;
		unsigned int		addInfoSize;	//!< includes random number generator etc.
		unsigned int		blkInfoSize;
		unsigned int		monitorInfoSize;
	} snnSize_t;

	snnSize_t cpuSnnSz;
	snnSize_t gpuSnnSz;
	unsigned int 	postConnCnt;
	unsigned int	preConnCnt;

	//! firing info
	unsigned int		*timeTableD2;
	unsigned int		*timeTableD1;
	unsigned int		*firingTableD2;
	unsigned int		*firingTableD1;
	unsigned int		maxSpikesD1;
	unsigned int		maxSpikesD2;

	//time and timestep

	unsigned int    simTimeRunStart; //!< the start time of current/last runNetwork call
	unsigned int    simTimeRunStop;  //!< the end time of current/last runNetwork call
	unsigned int    simTimeLastRunSummary; //!< the time at which the last run summary was printed

	unsigned int	simTimeMs;
	uint64_t        simTimeSec;		//!< this is used to store the seconds.
	unsigned int	simTime;		//!< The absolute simulation time. The unit is millisecond. this value is not reset but keeps increasing to its max value.
	unsigned int	spikeCountAll1secHost;
	unsigned int	secD1fireCntHost;
	unsigned int	secD2fireCntHost;	//!< firing counts for each second
	unsigned int	spikeCountAllHost;
	unsigned int	spikeCountD1Host;
	unsigned int	spikeCountD2Host;	//!< overall firing counts values
	unsigned int	nPoissonSpikes;

		//cuda keep track of performance...
#if __CUDA3__
		unsigned int    timer;
#else
		StopWatchInterface* timer;
#endif
		float		cumExecutionTime;
		float		lastExecutionTime;

	FILE*	fpInf_;			//!< fp of where to write all simulation output (status info) if not in silent mode
	FILE*	fpErr_;			//!< fp of where to write all errors if not in silent mode
	FILE*	fpDeb_;			//!< fp of where to write all debug info if not in silent mode
	FILE*	fpLog_;

	// keep track of number of SpikeMonitor/SpikeMonitorCore objects
	unsigned int numSpikeMonitor;
	SpikeMonitorCore* spikeMonCoreList[MAX_GRP_PER_SNN];
	SpikeMonitor*     spikeMonList[MAX_GRP_PER_SNN];

	// \FIXME \DEPRECATED this one moved to group-based
	long int    simTimeLastUpdSpkMon_; //!< last time we ran updateSpikeMonitor



	unsigned int	numSpikeGenGrps;

	int numSpkCnt; //!< number of real-time spike monitors in the network
	int* spkCntBuf[MAX_GRP_PER_SNN]; //!< the actual buffer of spike counts (per group, per neuron)


	unsigned int		numGroupMonitor;
	GroupMonitorCore*	groupMonCoreList[MAX_GRP_PER_SNN];
	GroupMonitor*		groupMonList[MAX_GRP_PER_SNN];

	// group monitor assistive buffers
	float*			grpDABuffer[MAX_GRP_PER_SNN];
	float*			grp5HTBuffer[MAX_GRP_PER_SNN];
	float*			grpAChBuffer[MAX_GRP_PER_SNN];
	float*			grpNEBuffer[MAX_GRP_PER_SNN];

	// neuron monitor variables
//	NeuronMonitorCore* neurBufferCallback[MAX_]
	int numNeuronMonitor;

	// connection monitor variables
	int numConnectionMonitor;
	ConnectionMonitorCore* connMonCoreList[MAX_nConnections];
	ConnectionMonitor*     connMonList[MAX_nConnections];


	/* Tsodyks & Markram (1998), where the short-term dynamics of synapses is characterized by three parameters:
	   U (which roughly models the release probability of a synaptic vesicle for the first spike in a train of spikes),
	   maxDelay_ (time constant for recovery from depression), and F (time constant for recovery from facilitation). */
	   float *stpu;
	   float *stpx;

	   float *gAMPA;
	   float *gNMDA;
	   float *gNMDA_r;
	   float *gNMDA_d;
	   float *gGABAa;
	   float *gGABAb;
	   float *gGABAb_r;
	   float *gGABAb_d;

	// concentration of neuromodulators for each group
	float*	grpDA;
	float*	grp5HT;
	float*	grpACh;
	float*	grpNE;

	network_info_t 	net_Info;

	network_ptr_t  		cpu_gpuNetPtrs;
	network_ptr_t   	cpuNetPtrs;

	//int   Noffset;
	int	  NgenFunc;					//!< this counts the spike generator offsets...

	bool finishedPoissonGroup;		//!< This variable is set after we have finished
	//!< creating the poisson group...

	bool showGrpFiringInfo;

	// gpu related info...
	// information about various data allocated at GPU side...
	unsigned int	gpu_tStep, gpu_simSec;		//!< this is used to store the seconds.
	unsigned int	gpu_simTime;				//!< this value is not reset but keeps increasing to its max value.

	group_info_t	  	grp_Info[MAX_GRP_PER_SNN];
	group_info2_t		grp_Info2[MAX_GRP_PER_SNN];
	uint32_t*	spikeGenBits;

	// weight update parameter
	int wtANDwtChangeUpdateInterval_;
	int wtANDwtChangeUpdateIntervalCnt_;
	float stdpScaleFactor_;
	float wtChangeDecay_; //!< the wtChange decay

	RNG_rand48* gpuPoissonRand;
};

#endif
