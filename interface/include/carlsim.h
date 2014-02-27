#ifndef _CARLSIM_H_
#define _CARLSIM_H_

#include <snn.h>	// FIXME: remove snn.h dependency
#include <string>
#include <map>

class CpuSNN;

class CARLsim {
public:
	CARLsim(std::string netName="SNN", int numConfig=1, int randSeed=42, int simType=CPU_MODE, int ithGPU=0,
				bool enableSilentMode=false);
	~CARLsim();



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! shortcut to create SYN_FIXED connections with just one weight and one delay value
	// returns connection id
	uint16_t connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay);

	//! shortcut to create SYN_FIXED connections with one weight / delay and two scaling factors for synaptic currents
	// returns connection id
	uint16_t connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay,
						float mulSynFast, float mulSynSlow);

	//! shortcut to create SYN_FIXED/SYN_PLASTIC connections with initWt/maxWt, minDelay/maxDelay, but to omit
	//! scaling factors for synaptic conductances (default is 1.0 for both)
	// returns connection id
	uint16_t connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt, float connProb,
						uint8_t minDelay, uint8_t maxDelay, bool synWtType);

	//! make connection from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	// returns connection id
	uint16_t connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt, float connProb,
						uint8_t minDelay, uint8_t maxDelay, float mulSynFast, float mulSynSlow, bool synWtType);

	//! shortcut to make connections with custom connectivity profile but omit scaling factors for synaptic
	//! conductances (default is 1.0 for both)
	uint16_t connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED, int maxM=0, 
						int maxPreM=0);

	//! make connections with custom connectivity profile
	uint16_t connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType=SYN_FIXED, int maxM=0,int maxPreM=0);


	//! creates a group of Izhikevich spiking neurons
	int createGroup(const std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);

	//! creates a spike generator group
	int createSpikeGeneratorGroup(const std::string grpName, unsigned int nNeur, int neurType, int configId=ALL);


	//! Sets default values for conduction decays or disables COBA if enable==false
	void setConductances(int grpId, bool isSet, int configId=ALL);

	//! Sets custom values for conduction decays or disables COBA if enable==false
	void setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb,
							int configId=ALL);

	//! Sets default homeostasis params for group
	void setHomeostasis(int grpId, bool isSet, int configId=ALL);

	//! Sets custom homeostasis params for group
	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId=ALL);

	//! Sets homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId=ALL);

	//! Sets Izhikevich params a, b, c, and d with as mean +- standard deviation
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId=ALL);

	//! Sets Izhikevich params a, b, c, and d of a neuron group. 
	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId=ALL);

	//! Sets default STDP params
	void setSTDP(int grpId, bool isSet, int configId=ALL);

	//! Sets STDP params for a group, custom
	void setSTDP(int grpId, bool isSet, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD, int configId=ALL);

	void setSTP(int grpId, bool isSet, int configId=ALL);

	void setSTP(int grpId, bool isSet, float STP_U, float STP_tD, float STP_tF, int configId=ALL);



	// +++++ PUBLIC METHODS: RUNNING A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! run network using default simulation mode
	int runNetwork(int nSec, int nMsec);

	//! run network with custom simulation mode and options
	int runNetwork(int nSec, int nMsec, int simType, int ithGPU=0, bool enablePrint=false, bool copyState=false);



	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// FIXME: needs overhaul
	//! Sets update cycle for log messages
	void setLogCycle(unsigned int _cnt, int mode=0, FILE *fp=NULL);


	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! reads the network state from file
	void readNetwork(FILE* fid);

	/*!
	 * \brief Reassigns fixed weights to values passed into the function in a single 1D float matrix (weightMatrix)
	 * The user passes the connection ID (connectID), the weightMatrix, the matrixSize, and 
	 * configuration ID (configID).  This function only works for fixed synapses and for connections of type
	 * CONN_USER_DEFINED. Only the weights are changed, not the maxWts, delays, or connected values
	 */
	void reassignFixedWeights(uint16_t connectId, float weightMatrix[], int matrixSize, int configId=ALL);

	void resetSpikeCntUtil(int grpId=ALL); //!< resets spike count for particular neuron group

	//! Sets up a spike generator
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId=ALL);

	//! Sets a spike monitor for a group, custom SpikeMonitor class
	void setSpikeMonitor(int gid, SpikeMonitor* spikeMon=NULL, int configId=ALL);

	//! Sets a spike monitor for a group, prints spikes to binary file
	void setSpikeMonitor(int grpId, const std::string& fname, int configId=0);

	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1, int configId=ALL);

	//! Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	//! weight values back to their default values by setting resetWeights = true.
	void updateNetwork(bool resetFiringInfo, bool resetWeights);

	//!< writes the network state to file
	void writeNetwork(FILE* fid);

	//! function writes population weights from gIDpre to gIDpost to file fname in binary.
	void writePopWeights(std::string fname, int gIDpre, int gIDpost, int configId=0);



	// +++++ PUBLIC METHODS: GETTER / SETTERS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	grpConnectInfo_t* getConnectInfo(uint16_t connectId, int configId=0); //!< gets connection info struct
	int  getConnectionId(uint16_t connId, int configId);

	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays=NULL);

	int getGroupId(int grpId, int configId=0);
	group_info_t getGroupInfo(int grpId, int configId=0); //!< gets group info struct
	std::string getGroupName(int grpId, int configId=0);

	int getNumConfigurations() { return numConfig_; }	//!< gets number of network configurations
	int getNumConnections(uint16_t connectionId);		//!< gets number of connections associated with a connection ID
	int getNumGroups();									//!< gets number of groups in the network

	/*!
	 * \brief Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
	 * and the size of the 1D array in size.  gIDpre(post) is the group ID for the pre(post)synaptic group, 
	 * weights is a pointer to a single dimensional array of floats, size is the size of that array which is 
	 * returned to the user, and configID is the configuration ID of the SNN.  NOTE: user must free memory from
	 * weights to avoid a memory leak.  
	 */
	void getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId=0);

	uint64_t getSimTime();
	uint32_t getSimTimeSec();
	uint32_t getSimTimeMsec();

	//! Returns pointer to 1D array of the number of spikes every neuron in the group has fired
	unsigned int* getSpikeCntPtr(int grpId, int simType);

	//! use default simulation mode
	unsigned int* getSpikeCntPtr(int grpId);

	// FIXME: fix this
	// TODO: maybe consider renaming getPopWeightChanges
	float* getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges=NULL);

	int grpStartNeuronId(int grpId);
	int grpEndNeuronId(int grpId);
	int grpNumNeurons(int grpId);

	bool isExcitatoryGroup(int grpId);
	bool isInhibitoryGroup(int grpId);
	bool isPoissonGroup(int grpId);

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 */
	void setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr);

	void setGroupInfo(int grpId, group_info_t info, int configId=ALL);
	void setPrintState(int grpId, bool status);
	void setSimLogs(bool isSet, std::string logDirName="");
	void setTuningLog(std::string fname);


	// +++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! sets default values for conductance decays
	void setDefaultConductanceDecay(float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb);

	//! sets default homeostasis params
	void setDefaultHomeostasisParams(float homeoScale, float avgTimeScale);

	//! sets default values for STDP params
	void setDefaultSTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD);

	//! sets default values for STP params (neurType either EXCITATORY_NEURON or INHIBITORY_NEURON)
	void setDefaultSTPparams(int neurType, float STP_U, float STP_tD, float STP_tF);


	// +++++ PUBLIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief a struct that contains important info about each allocated group
	 * Can be used (for example) to determine whether all (or none of the) groups have conductances enabled.
	 */
	struct grpInfo_s {
		int grpId;			// the grp id returned from CARLsimCore
		bool hasSetCond;	// whether conductances are set
	};

private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	void checkConductances(); 			//!< all or none of the groups must enable conductances

	void handleUserWarnings(); 			//!< print all user warnings, continue only after user input
	void handleNetworkConsistency();	//!< do all setupNetwork error checks

	void printSimulationSpecs();

	grpInfo_s makeGrpInfo(int grpId, bool hasSetCond); //!< factory function for grpInfo_s


	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	CpuSNN* snn_;			//!< an instance of CARLsim core class
	int numConfig_;			//!< number of configurations
	int randSeed_;			//!< RNG seed
	int simMode_;			//!< CPU_MODE or GPU_MODE
	int ithGPU_;			//!< on which device to establish a context
	bool enablePrint_;
	bool copyState_;

	unsigned int numConnections_;	//!< keep track of number of allocated connections

	/*!
	 * \brief a map that stores the grpId and some associated grpInfo (struct)
	 * grpInfo can be used (for example) to detect whether all allocated groups have conductances enabled (disabled).
	 * A new entry can be added to the map by using the factory function makeGrpInfo.
	 */
	std::map<int, grpInfo_s> grpInfo_;

	bool hasRunNetwork_;			//!< flag to inform that network has been run

	bool hasSetHomeoALL_;			//!< informs that homeostasis have been set for ALL groups (can't add more groups)
	bool hasSetHomeoBaseFiringALL_;	//!< informs that base firing has been set for ALL groups (can't add more groups)
	bool hasSetSTDPALL_; 			//!< informs that STDP have been set for ALL groups (can't add more groups)
	bool hasSetSTPALL_; 			//!< informsthat STP have been set for ALL groups (can't add more groups)

	std::vector<std::string> userWarnings_; // !< an accumulated list of user warnings
	float def_tdAMPA_;				//!< default value for AMPA decay (ms)
	float def_tdNMDA_;				//!< default value for NMDA decay (ms)
	float def_tdGABAa_;				//!< default value for GABAa decay (ms)
	float def_tdGABAb_;				//!< default value for GABAb decay (ms)

	float def_STDP_alphaLTP_;		//!< default value for LTP amplitude
	float def_STDP_tauLTP_;			//!< default value for LTP decay (ms)
	float def_STDP_alphaLTD_;		//!< default value for LTD amplitude
	float def_STDP_tauLTD_;			//!< default value for LTD decay (ms)

	float def_STP_U_exc_;			//!< default value for STP U excitatory
	float def_STP_tD_exc_;			//!< default value for STP tD excitatory (ms)
	float def_STP_tF_exc_;			//!< default value for STP tF excitatory (ms)
	float def_STP_U_inh_;			//!< default value for STP U inhibitory
	float def_STP_tD_inh_;			//!< default value for STP tD inhibitory (ms)
	float def_STP_tF_inh_;			//!< default value for STP tF inhibitory (ms)

	float def_homeo_scale_;			//!< default homeoScale
	float def_homeo_avgTimeScale_;	//!< default avgTimeScale
};
#endif