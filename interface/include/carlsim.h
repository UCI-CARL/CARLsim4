#ifndef _CARLSIM_H_
#define _CARLSIM_H_

#include <snn.h>
#include <string>
#include <map>


class CARLsim {
public:
	CARLsim(std::string netName="SNN", int numConfig=1, int randSeed=42, int simType=CPU_MODE, int ithGPU=0,
				bool enablePrint=false, bool copyState=false);
	~CARLsim();



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! shortcut to create SYN_FIXED connections with just one weight and one delay value
	// returns connection id
	int connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay);

	//! make connection from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	// returns connection id
	int connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt, float connProb,
					uint8_t minDelay, uint8_t maxDelay, bool synWtType);

	//! make connections with custom connectivity profile
	int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED, int maxM=0,int maxPreM=0);


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

	// FIXME: needs overhaul
	//! Sets update cycle for log messages
	void setLogCycle(unsigned int _cnt, int mode=0, FILE *fp=NULL);




	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! reads the network state from file
	void readNetwork(FILE* fid);

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

	grpConnectInfo_t* getConnectInfo(int connectId, int configId=0); //!< gets connection info struct

	int getGroupId(int grpId, int configId=0);
	group_info_t getGroupInfo(int grpId, int configId=0); //!< gets group info struct
	std::string getGroupName(int grpId, int configId=0);

	//! Returns pointer to 1D array of the number of spikes every neuron in the group has fired
	unsigned int* getSpikeCntPtr(int grpId, int simType);

	//! use default simulation mode
	unsigned int* getSpikeCntPtr(int grpId);

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 */
	void setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr);


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

	std::map<int, grpInfo_s> grpInfo_;

	bool hasRunNetwork_;				//!< flag to inform that network has been run

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