#include <carlsim.h>
#include <user_errors.h>

//#include <snn.h>		// FIXME: move snn.h dependency out of carlsim.h
#include <string>		// std::string
#include <iostream>		// std::cout, std::endl
#include <sstream>		// std::stringstream
#include <algorithm>	// std::find


// includes for mkdir
#if CREATE_SPIKEDIR_IF_NOT_EXISTS
	#include <sys/stat.h>
	#include <errno.h>
	#include <libgen.h>
#endif

// NOTE: Conceptual code documentation should go in carlsim.h. Do not include extensive high-level documentation here,
// but do document your code.


// FIXME: consider moving this... but it doesn't belong in the core code. Also, see class SpikeMonitor in snn.h

/*!
 * \brief class to automatically write spike times collected with SpikeMonitor to binary file
 * This class derives from class SpikeMonitor and implements its virtual member update() to automatically store
 * collected spike times (and neuron IDs) to binary file.
 * Note that this function will only be called every 1000 ms.
 */
class WriteSpikesToFile: public SpikeMonitor {
public:
	WriteSpikesToFile(FILE* fid) {
		fileId_ = fid;
	}
	~WriteSpikesToFile() {}; // TODO: where does fileId_ get closed?

	/*
	 * \brief update method that gets called every 1000 ms by CARLsimCore
	 * This is an implementation of virtual void SpikeMonitor::update. It gets called every 1000 ms with a pointer to
	 * all the neurons (neurIds) that have spiked during the last 1000 ms (timeCnts).
	 * This implementation will iterate over all neuron IDs and spike times, and print them to file (binary).
	 * To save space, neuron IDs are stored in a continuous (flattened) list, whereas timeCnts holds the number of
	 * neurons that have spiked at each time step (reduced AER).
	 * Example: There are 3 neurons, where neuron with ID 0 spikes at time 1, neurons with ID 1 and 2 both spike at
	 *  		time 3. Then neurIds = {0,1,2} and timeCnts = {0,1,0,2,0,...,0}. Note that neurIds could also be {0,2,1}
	 *
	 * \param[in] snn 		pointer to an instance of CARLsimCore
	 * \param[in] grpId 	the group ID from which to record spikes
	 * \param[in] neurIds	pointer to a flattened list that contains all the IDs of neurons that have spiked within
	 *                      the last 1000 ms.
	 * \param[in] timeCnts 	pointer to a data structures that holds the number of spikes at each time step during the
	 *  					last 1000 ms. timeCnts[i] will hold the number of spikes in the i-th millisecond.
	 */
	void update(CpuSNN* snn, int grpId, unsigned int* neurIds, unsigned int* timeCnts) {
		int pos    = 0; // keep track of position in flattened list of neuron IDs

		for (int t=0; t < 1000; t++) {
			for(int i=0; i<timeCnts[t];i++,pos++) {
				int time = t + snn->getSimTime() - 1000;
				int id   = neurIds[pos];
				int cnt = fwrite(&time,sizeof(int),1,fileId_);
				assert(cnt != 0);
				cnt = fwrite(&id,sizeof(int),1,fileId_);
				assert(cnt != 0);
			}
		}

		fflush(fileId_);
	}

private:
	FILE* fileId_;
};



/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///

CARLsim::CARLsim(std::string netName, int nConfig, int randSeed, int simMode, int ithGPU, loggerMode_t loggerMode) {
	netName_ 					= netName;
	nConfig_ 					= nConfig;
	randSeed_					= randSeed;
	simMode_ 					= simMode;
	ithGPU_ 					= ithGPU;
	loggerMode_ 				= loggerMode;
	enablePrint_ = false;
	copyState_ = false;

	numConnections_				= 0;

	hasRunNetwork_  			= false;
	hasSetHomeoALL_ 			= false;
	hasSetHomeoBaseFiringALL_ 	= false;
	hasSetSTDPALL_ 				= false;
	hasSetSTPALL_ 				= false;

	CARLsimInit(); // move everything else out of constructor
}

CARLsim::~CARLsim() {
	// deallocate all dynamically allocated structures
	if (snn_!=NULL)
		delete snn_;
}

// unsafe computations that would otherwise go in constructor
void CARLsim::CARLsimInit() {
	snn_ = new CpuSNN(netName_, nConfig_, randSeed_, simMode_, loggerMode_);

	// set default time constants for synaptic current decay
	// TODO: add ref
	def_tdAMPA_  = 5.0f;	// default decay time for AMPA (ms)
	def_tdNMDA_  = 150.0f;	// default decay time for NMDA (ms)
	def_tdGABAa_ = 6.0f;	// default decay time for GABAa (ms)
	def_tdGABAb_ = 150.0f;	// default decay time for GABAb (ms)

	// set default values for STDP params
	// TODO: add ref
	def_STDP_alphaLTP_ = 0.001f;
	def_STDP_tauLTP_   = 20.0f;
	def_STDP_alphaLTD_ = 0.0012f;
	def_STDP_tauLTD_   = 20.0f;

	// set default values for STP params
	// TODO: add ref
	def_STP_U_exc_  = 0.2f;
	def_STP_tD_exc_ = 700.0f;
	def_STP_tF_exc_ = 20.0f;
	def_STP_U_inh_  = 0.5f;
	def_STP_tD_inh_ = 800.0f;
	def_STP_tF_inh_ = 1000.0f;

	// set default homeostasis params
	// TODO: add ref
	def_homeo_scale_ = 0.1f;
	def_homeo_avgTimeScale_ = 10.f;
}



/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// +++++++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// Connects a presynaptic to a postsynaptic group using fixed weights and a single delay value
uint16_t CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << "Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << "Group Id " << grpId2;
	UserErrors::userAssert(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::userAssert(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::userAssert(!isExcitatoryGroup(grpId1) || wt>0, UserErrors::MUST_BE_POSITIVE, funcName, "wt");
	UserErrors::userAssert(!isInhibitoryGroup(grpId1) || wt<0, UserErrors::MUST_BE_NEGATIVE, funcName, "wt");
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	return snn_->connect(grpId1, grpId2, connType, wt, wt, connProb, delay, delay, 1.0f, 1.0f, SYN_FIXED);
}

// shortcut to create SYN_FIXED connections with one weight / delay and two scaling factors for synaptic currents
uint16_t CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay,
							float mulSynFast, float mulSynSlow) {
	assert(!hasRunNetwork_); // TODO: make nice
	assert(++numConnections_ <= MAX_numConnections);

	return snn_->connect(grpId1,grpId2,connType,wt,wt,connProb,delay,delay,mulSynFast,mulSynSlow,SYN_FIXED);
}

// shortcut to create SYN_FIXED/SYN_PLASTIC connections with initWt/maxWt, minDelay/maxDelay, but to omit
// scaling factors for synaptic conductances (default is 1.0 for both)
uint16_t CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt,
							float connProb,	uint8_t minDelay, uint8_t maxDelay, bool synWtType)
{
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::userAssert(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::userAssert(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::userAssert(!isExcitatoryGroup(grpId1) || maxWt>0, UserErrors::MUST_BE_POSITIVE, funcName, "maxWt");
	UserErrors::userAssert(!isInhibitoryGroup(grpId1) || maxWt<0, UserErrors::MUST_BE_NEGATIVE, funcName, "maxWt");
	UserErrors::userAssert(initWt*maxWt>=0, UserErrors::MUST_HAVE_SAME_SIGN, funcName, "initWt and maxWt");
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	return snn_->connect(grpId1,grpId2,connType,initWt,maxWt,connProb,minDelay,maxDelay,1.0f,1.0f,synWtType);
}

// custom connectivity profile
uint16_t CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// TODO: check for sign of weights
	return snn_->connect(grpId1, grpId2, conn, 1.0f, 1.0f, synWtType, maxM, maxPreM);
}

// custom connectivity profile
uint16_t CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	assert(++numConnections_ <= MAX_numConnections);

	return snn_->connect(grpId1, grpId2, conn, mulSynFast, mulSynSlow, synWtType, maxM, maxPreM);
}


// create group of Izhikevich spiking neurons
int CARLsim::createGroup(std::string grpName, int nNeur, int neurType, int configId) {
	std::string funcName = "createGroup(\""+grpName+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// if user has called any set functions with grpId=ALL, and is now adding another group, previously set properties
	// will not apply to newly added group
	if (hasSetSTPALL_)
		userWarnings_.push_back("USER WARNING: Make sure to call setSTP on group "+grpName);
	if (hasSetSTDPALL_)
		userWarnings_.push_back("USER WARNING: Make sure to call setSTDP on group "+grpName);
	if (hasSetHomeoALL_)
		userWarnings_.push_back("USER WARNING: Make sure to call setHomeostasis on group "+grpName);
	if (hasSetHomeoBaseFiringALL_)
		userWarnings_.push_back("USER WARNING: Make sure to call setHomeoBaseFiringRate on group "+grpName);

	int grpId = snn_->createGroup(grpName.c_str(),nNeur,neurType,configId);
	grpIds_.push_back(grpId); // keep track of all groups

	return grpId;
}

// create group of spike generators
int CARLsim::createSpikeGeneratorGroup(std::string grpName, int nNeur, int neurType, int configId) {
	std::string funcName = "createSpikeGeneratorGroup(\""+grpName+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	int grpId = snn_->createSpikeGeneratorGroup(grpName.c_str(),nNeur,neurType,configId);
	grpIds_.push_back(grpId); // keep track of all groups

	return grpId;
}


// set conductance values, use defaults
void CARLsim::setConductances(int grpId, bool isSet, int configId) {
	std::string funcName = "setConductances(\""+getGroupName(grpId,configId)+"\")";
	std::stringstream grpIdStr; grpIdStr << "setConductances(" << grpId << "). Group Id " << grpId;
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	if (isSet) { // enable conductances, use default values
		snn_->setConductances(grpId,true,def_tdAMPA_,def_tdNMDA_,def_tdGABAa_,def_tdGABAb_,configId);
	} else { // disable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}

}

// set conductances values, custom
void CARLsim::setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb,
								int configId)
{
	std::string funcName = "setConductances(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(grpId,true,tdAMPA,tdNMDA,tdGABAa,tdGABAb,configId);
	} else { // disable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}
}

// set default homeostasis params
void CARLsim::setHomeostasis(int grpId, bool isSet, int configId) {
	std::string funcName = "setHomeostasis(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,def_homeo_scale_,def_homeo_avgTimeScale_,configId);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("USER WARNING: Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId,configId));
	} else { // disable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f,configId);
	}
}

// set custom homeostasis params for group
void CARLsim::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId) {
	std::string funcName = "setHomeostasis(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,homeoScale,avgTimeScale,configId);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("USER WARNING: Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId,configId));
	} else { // disable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f,configId);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void CARLsim::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId) {
	std::string funcName = "setHomeoBaseFiringRate(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	hasSetHomeoBaseFiringALL_ = grpId=ALL; // adding groups after this will not have base firing set

	snn_->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD, configId);
}

// set neuron parameters for Izhikevich neuron, with standard deviations
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId)
{
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// wrapper identical to core func
	snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
}

// set neuron parameters for Izhikevich neuron
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId) {
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// set standard deviations of Izzy params to zero
	snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f, configId);
}

// set STDP, default
void CARLsim::setSTDP(int grpId, bool isSet, int configId) {
	std::string funcName = "setSTDP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		snn_->setSTDP(grpId,true,def_STDP_alphaLTP_,def_STDP_tauLTP_,def_STDP_alphaLTD_,def_STDP_tauLTD_,configId);
	} else { // disable STDP
		snn_->setSTDP(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}	
}

// set STDP, custom
void CARLsim::setSTDP(int grpId, bool isSet, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD, int configId) {
	std::string funcName = "setSTDP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		assert(tauLTP>0); // TODO make nice
		assert(tauLTD>0);
		snn_->setSTDP(grpId,true,alphaLTP,tauLTP,alphaLTD,tauLTD,configId);
	} else { // disable STDP
		snn_->setSTDP(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}
}

// set STP, default
void CARLsim::setSTP(int grpId, bool isSet, int configId) {
	std::string funcName = "setSTP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::userAssert(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName);

		if (isExcitatoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_exc_,def_STP_tD_exc_,def_STP_tF_exc_,configId);
		else if (isInhibitoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_inh_,def_STP_tD_inh_,def_STP_tF_inh_,configId);
		else {
			// some error message
		}
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f,configId);
	}		
}

// set STP, custom
void CARLsim::setSTP(int grpId, bool isSet, float STP_U, float STP_tD, float STP_tF, int configId) {
	std::string funcName = "setSTP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::userAssert(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName);

		snn_->setSTP(grpId,true,STP_U,STP_tD,STP_tF,configId);
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f,configId);
	}		
}


// +++++++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// run network
int CARLsim::runNetwork(int nSec, int nMsec) {
	if (!hasRunNetwork_) {
		handleNetworkConsistency();	// before running network, make sure it's consistent
		handleUserWarnings();		// before running network, make sure user didn't provoque any user warnings
//		printSimulationSpecs();		// first time around, show simMode etc.
	}

	hasRunNetwork_ = true;

	return snn_->runNetwork(nSec, nMsec, simMode_, ithGPU_, enablePrint_, copyState_);
}

// run network with custom options
int CARLsim::runNetwork(int nSec, int nMsec, int simType, int ithGPU, bool enablePrint, bool copyState) {
	if (!hasRunNetwork_) {
		handleNetworkConsistency();	// before running network, make sure it's consistent
		handleUserWarnings();	// before running network, make sure user didn't provoque any user warnings
//		printSimulationSpecs(); // first time around, show simMode etc.
	}

	hasRunNetwork_ = true;

	return snn_->runNetwork(nSec, nMsec, simType, ithGPU, enablePrint, copyState);	
}

// +++++++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// sets update cycle for log messages
void CARLsim::setLogCycle(unsigned int cnt, int mode, FILE *fp) {
	std::string funcName = "setLogCycle()";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	snn_->setLogCycle(cnt, mode, fp);
}


// +++++++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// reads network state from file
void CARLsim::readNetwork(FILE* fid) {
	std::string funcName = "readNetwork()";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	snn_->readNetwork(fid);
}

void CARLsim::reassignFixedWeights(uint16_t connectId, float weightMatrix[], int matrixSize, int configId) {
	snn_->reassignFixedWeights(connectId,weightMatrix,matrixSize,configId);
}


// resets spike count for particular neuron group
void CARLsim::resetSpikeCntUtil(int grpId) {
	snn_->resetSpikeCntUtil(grpId);
}

// sets up a spike generator
void CARLsim::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId) {
	std::string funcName = "setSpikeGenerator(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::userAssert(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	snn_->setSpikeGenerator(grpId,spikeGen,configId);
}

// set spike monitor for a group
void CARLsim::setSpikeMonitor(int grpId, SpikeMonitor* spikeMon, int configId) {
	std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId,configId)+"\",SpikeMonitor*)";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::userAssert(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	snn_->setSpikeMonitor(grpId,spikeMon,configId);
}


// set spike monitor for group and write spikes to file
void CARLsim::setSpikeMonitor(int grpId, const std::string& fname, int configId) {
	std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId,configId)+"\",\""+fname+"\")";
	UserErrors::userAssert(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::userAssert(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "configId");	// configId can't be ALL
	UserErrors::userAssert(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	// try to open spike file
	FILE* fid = fopen(fname.c_str(),"wb"); // FIXME: where does fid get closed?
	if (fid==NULL) {
		// file could not be opened

		#if CREATE_SPIKEDIR_IF_NOT_EXISTS
			// if option set, attempt to create directory
			int status;

			// this is annoying...for dirname we need to convert from const string to char*
	    	char fchar[200];
	    	strcpy(fchar,fname.c_str());

			#if defined(_WIN32) || defined(_WIN64) // TODO: test it
				status = _mkdir(dirname(fchar); // Windows platform
			#else
			    status = mkdir(dirname(fchar), 0777); // Unix
			#endif

			std::string fileError = "%%CARLSIM_ROOT%%/results/ does not exist. Thus file " + fname;
			UserErrors::userAssert(status!=-1 || errno==EEXIST, UserErrors::FILE_CANNOT_CREATE, funcName, fileError);

			// now that the directory is created, fopen file
			fid = fopen(fname.c_str(),"wb");
		#else
		    // default case: print error and exit
		    std::string fileError = ". Enable option CREATE_SPIKEDIR_IF_NOT_EXISTS in config.h to attempt "
		    							"creating the specified subdirectory automatically. File " + fname;
		    UserErrors::userAssert(false, UserErrors::FILE_CANNOT_OPEN, fileName, fileError);
		#endif
	}

	setSpikeMonitor(grpId, new WriteSpikesToFile(fid), configId);
}

// assign spike rate to poisson group
void CARLsim::setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod, int configId) {
	snn_->setSpikeRate(grpId, spikeRate, refPeriod, configId);
}

// Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
// weight values back to their default values by setting resetWeights = true.
void CARLsim::updateNetwork(bool resetFiringInfo, bool resetWeights) {
	snn_->updateNetwork(resetFiringInfo,resetWeights);
}

// writes network state to file
void CARLsim::writeNetwork(FILE* fid) {
	snn_->writeNetwork(fid);
}

// function writes population weights from gIDpre to gIDpost to file fname in binary.
void CARLsim::writePopWeights(std::string fname, int gIDpre, int gIDpost, int configId) {
	std::string funcName = "writePopWeights("+fname+")";
	UserErrors::userAssert(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "configId");	// configId can't be ALL
	snn_->writePopWeights(fname,gIDpre,gIDpost,configId);
}



// +++++++++ PUBLIC METHODS: SETTERS / GETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// get connection info struct
grpConnectInfo_t* CARLsim::getConnectInfo(uint16_t connectId, int configId) {
	std::stringstream funcName;	funcName << "getConnectInfo(" << connectId << "," << configId << ")";
	UserErrors::userAssert(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
	return snn_->getConnectInfo(connectId,configId);
}

int CARLsim::getConnectionId(uint16_t connectId, int configId) {
	return snn_->getConnectionId(connectId,configId);
}

uint8_t* CARLsim::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	return snn_->getDelays(gIDpre,gIDpost,Npre,Npost,delays);
}

int CARLsim::getGroupId(int grpId, int configId) {
	std::stringstream funcName;	funcName << "getConnectInfo(" << grpId << "," << configId << ")";
	UserErrors::userAssert(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
	return snn_->getGroupId(grpId,configId);
}
// get group info struct
group_info_t CARLsim::getGroupInfo(int grpId, int configId) {
	std::stringstream funcName;	funcName << "getConnectInfo(" << grpId << "," << configId << ")";
	UserErrors::userAssert(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
	return snn_->getGroupInfo(grpId, configId);
}

// get group name
std::string CARLsim::getGroupName(int grpId, int configId) {
	return snn_->getGroupName(grpId, configId);
}

int CARLsim::getNumConnections(uint16_t connectionId) {
	return snn_->getNumConnections(connectionId);
}

int CARLsim::getNumGroups() { return snn_->getNumGroups(); }
uint64_t CARLsim::getSimTime() { return snn_->getSimTime(); }
uint32_t CARLsim::getSimTimeSec() { return snn_->getSimTimeSec(); }
uint32_t CARLsim::getSimTimeMsec() { return snn_->getSimTimeMs(); }

// Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
// and the size of the 1D array in size.
void CARLsim::getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId) {
	snn_->getPopWeights(gIDpre,gIDpost,weights,size,configId);
}

unsigned int* CARLsim::getSpikeCntPtr(int grpId) {
	return snn_->getSpikeCntPtr(grpId,simMode_); // use default sim mode
}

unsigned int* CARLsim::getSpikeCntPtr(int grpId, int simType) {
	assert(simType==CPU_MODE || simType==GPU_MODE);
	return snn_->getSpikeCntPtr(grpId,simType);
}

float* CARLsim::getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges) {
	return snn_->getWeightChanges(gIDpre,gIDpost,Npre,Npost,weightChanges);
}

int CARLsim::grpStartNeuronId(int grpId) { return snn_->grpStartNeuronId(grpId); }
int CARLsim::grpEndNeuronId(int grpId) { return snn_->grpEndNeuronId(grpId); }
int CARLsim::grpNumNeurons(int grpId) { return snn_->grpNumNeurons(grpId); }

bool CARLsim::isExcitatoryGroup(int grpId) { return snn_->isExcitatoryGroup(grpId); }
bool CARLsim::isInhibitoryGroup(int grpId) { return snn_->isInhibitoryGroup(grpId); }
bool CARLsim::isPoissonGroup(int grpId) { return snn_->isPoissonGroup(grpId); }

// Sets enableGpuSpikeCntPtr to true or false.
void CARLsim::setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr) {
	snn_->setCopyFiringStateFromGPU(enableGPUSpikeCntPtr);
}

void CARLsim::setGroupInfo(int grpId, group_info_t info, int configId) { snn_->setGroupInfo(grpId,info,configId); }
void CARLsim::setPrintState(int grpId, bool status) { snn_->setPrintState(grpId,status); }
void CARLsim::setSimLogs(bool isSet, std::string logDirName) { snn_->setSimLogs(isSet,logDirName); }
void CARLsim::setTuningLog(std::string fname) { snn_->setTuningLog(fname); }




// +++++++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// set default values for conductance decay times
void CARLsim::setDefaultConductanceDecay(float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb) {
	assert(tdAMPA>0); // TODO make nice
	assert(tdNMDA>0);
	assert(tdGABAa>0);
	assert(tdGABAb>0);

	def_tdAMPA_  = tdAMPA;
	def_tdNMDA_  = tdNMDA;
	def_tdGABAa_ = tdGABAa;
	def_tdGABAb_ = tdGABAb;
}

void CARLsim::setDefaultHomeostasisParams(float homeoScale, float avgTimeScale) {
	assert(avgTimeScale>0); // TODO make nice

	def_homeo_scale_ = homeoScale;
	def_homeo_avgTimeScale_ = avgTimeScale;
}

// set default values for STDP params
void CARLsim::setDefaultSTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD) {
	assert(tauLTP>0); // TODO make nice
	assert(tauLTD>0);
	def_STDP_alphaLTP_ = alphaLTP;
	def_STDP_tauLTP_ = tauLTP;
	def_STDP_alphaLTD_ = alphaLTD;
	def_STDP_tauLTD_ = tauLTD;
}

// set default STP values for an EXCITATORY_NEURON or INHIBITORY_NEURON
void CARLsim::setDefaultSTPparams(int neurType, float STP_U, float STP_tD, float STP_tF) {
	assert(neurType==EXCITATORY_NEURON || neurType==INHIBITORY_NEURON); // TODO make nice
	assert(STP_tD>0);
	assert(STP_tF>0);

	switch (neurType) {
		case EXCITATORY_NEURON:
			def_STP_U_exc_ = STP_U;
			def_STP_tD_exc_ = STP_tD;
			def_STP_tF_exc_ = STP_tF;
			break;
		case INHIBITORY_NEURON:
			def_STP_U_inh_ = STP_U;
			def_STP_tD_inh_ = STP_tD;
			def_STP_tF_inh_ = STP_tF;
			break;
		default:
			// some error message instead of assert
			break;
	}
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// check whether all or none of the groups have conductances enabled
void CARLsim::checkConductances() {
	bool allSame;
	for (std::vector<int>::const_iterator it = grpIds_.begin(); it!=grpIds_.end(); ++it) {
		for (int c=0; c<nConfig_; c++) {
			group_info_t grpInfo = getGroupInfo(*it,c);
			if (grpInfo.isSpikeGenerator)	// NOTE: skipping spike generator might not be required, but it's cleaner
				continue;
			allSame = (it==grpIds_.begin() && c==0) ? grpInfo.WithConductances : allSame==grpInfo.WithConductances;
		}
	}

	std::string errorMsg = "If one group enables conductances, then all groups (except for generators) must enable "
							"conductances. All conductances";
	UserErrors::userAssert(allSame, UserErrors::MUST_HAVE_SAME_SIGN, "", errorMsg);
}

// check whether grpId exists in grpIds_
bool CARLsim::existsGrpId(int grpId) {
	return std::find(grpIds_.begin(), grpIds_.end(), grpId)!=grpIds_.end();
}

// check for setupNetwork user errors
void CARLsim::handleNetworkConsistency() {
	checkConductances();	// conductances have to be set either for all groups or for none

}

// print all user warnings, continue only after user input
void CARLsim::handleUserWarnings() {
	if (userWarnings_.size()) {
		for (int i=0; i<userWarnings_.size(); i++)
			fprintf(stdout,"%s\n",userWarnings_[i].c_str()); // print all user warnings

		fprintf(stdout,"Ignore warnings and continue? Y/n ");
		char ignoreWarn = std::cin.get();
		if (std::cin.fail() || ignoreWarn!='y' && ignoreWarn!='Y') {
			fprintf(stdout,"exiting...\n");
			exit(1);
		}
	}
}

// print all simulation specs
void CARLsim::printSimulationSpecs() {
	if (simMode_==CPU_MODE) {
		fprintf(stdout,"CPU_MODE, enablePrint=%s, copyState=%s\n\n",enablePrint_?"on":"off",copyState_?"on":"off");
	} else {
		fprintf(stdout,"GPU_MODE, GPUid=%d, enablePrint=%s, copyState=%s\n\n",ithGPU_,enablePrint_?"on":"off",
					copyState_?"on":"off");
	}
}