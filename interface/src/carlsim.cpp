#include <carlsim.h>

#include <snn.h>
#include <string>
#include <iostream>

// includes for mkdir
#if defined(CREATE_SPIKEDIR_IF_NOT_EXISTS)
	#include <sys/stat.h>
	#include <errno.h>
	#include <libgen.h>
#endif


// FIXME: consider moving this... also, see class SpikeMonitor in snn.h
class WriteSpikesToFile: public SpikeMonitor {
public:
	WriteSpikesToFile(FILE* fid) {
		fileId_ = fid;
	}
	~WriteSpikesToFile() {}; // TODO: where does fileId_ get closed?

	void update(CpuSNN* snn, int grpId, unsigned int* neuronIds, unsigned int* timeCnts) {
		int pos    = 0;

		for (int t=0; t < 1000; t++) {
			for(int i=0; i<timeCnts[t];i++,pos++) {
				int time = t + snn->getSimTime() - 1000;
				int id   = neuronIds[pos];
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

CARLsim::CARLsim(std::string netName, int numConfig, int randSeed, int simType, int ithGPU, bool enablePrint,
					bool copyState) {
	snn_ = new CpuSNN(netName.c_str(), numConfig, randSeed, simType);
	numConfig_ 					= numConfig;
	randSeed_					= randSeed;
	simMode_ 					= simType;
	ithGPU_ 					= ithGPU;
	enablePrint_ 				= enablePrint;
	copyState_ 					= copyState;

	hasRunNetwork_  			= false;

	hasSetHomeoALL_ 			= false;
	hasSetHomeoBaseFiringALL_ 	= false;
	hasSetSTDPALL_ 				= false;
	hasSetSTPALL_ 				= false;

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

CARLsim::~CARLsim() {
	// deallocate all dynamically allocated structures
	if (snn_!=NULL)
		delete snn_;
}


/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// +++++++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// shortcut to make SYN_FIXED with one weight and one delay value
int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay) {
	assert(!hasRunNetwork_); // TODO: make nice

	return snn_->connect(grpId1, grpId2, connType, wt, wt, connProb, delay, delay, SYN_FIXED);
}

// basic connection function, from each neuron in grpId1 to neurons in grpId2
int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt, float connProb,
					uint8_t minDelay, uint8_t maxDelay, bool synWtType) {
	assert(!hasRunNetwork_); // TODO: make nice

	return snn_->connect(grpId1, grpId2, connType, initWt, maxWt, connProb, minDelay, maxDelay, synWtType);
}

// custom connectivity profile
int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType, int maxM, int maxPreM) {
	assert(!hasRunNetwork_); // TODO: make nice

	return snn_->connect(grpId1, grpId2, conn, synWtType, maxM, maxPreM);
}


// create group of Izhikevich spiking neurons
int CARLsim::createGroup(std::string grpName, unsigned int nNeur, int neurType, int configId) {
	assert(!hasRunNetwork_); // TODO: make nice error message

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

	// keep track of group info
	grpInfo_[grpId] = makeGrpInfo(grpId,false);

	return grpId;
}

// create group of spike generators
int CARLsim::createSpikeGeneratorGroup(std::string grpName, unsigned int nNeur, int neurType, int configId) {
	assert(!hasRunNetwork_); // TODO: make nicer
	return snn_->createSpikeGeneratorGroup(grpName.c_str(),nNeur,neurType,configId);
}


// set conductance values, use defaults
void CARLsim::setConductances(int grpId, bool isSet, int configId) {
	assert(!hasRunNetwork_); // TODO make nice

	// keep track of groups with conductances set
	if (grpId==ALL) {
		for (std::map<int,grpInfo_s>::iterator iter = grpInfo_.begin(); iter!=grpInfo_.end(); ++iter)
			grpInfo_[iter->first].hasSetCond = isSet;
	} else {
		grpInfo_[grpId].hasSetCond = isSet;
	}

	if (isSet) { // enable conductances, use default values
		snn_->setConductances(grpId,true,def_tdAMPA_,def_tdNMDA_,def_tdGABAa_,def_tdGABAb_,configId);
	} else { // discable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}

}

// set conductances values, custom
void CARLsim::setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb,
								int configId) {
	assert(!hasRunNetwork_); // TODO make nice

	// keep track of groups with conductances set
	if (grpId==ALL) {
		for (std::map<int,grpInfo_s>::iterator iter = grpInfo_.begin(); iter!=grpInfo_.end(); ++iter)
			grpInfo_[iter->first].hasSetCond = isSet;
	} else {
		grpInfo_[grpId].hasSetCond = isSet;
	}

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(grpId,true,tdAMPA,tdNMDA,tdGABAa,tdGABAb,configId);
	} else { // discable conductances
		snn_->setConductances(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}
}

// set default homeostasis params
void CARLsim::setHomeostasis(int grpId, bool isSet, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,def_homeo_scale_,def_homeo_avgTimeScale_,configId);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("USER WARNING: Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId));
	} else { // discable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f,configId);
	}
}

// set custom homeostasis params for group
void CARLsim::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,homeoScale,avgTimeScale,configId);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("USER WARNING: Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId));
	} else { // discable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f,configId);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void CARLsim::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	hasSetHomeoBaseFiringALL_ = grpId=ALL; // adding groups after this will not have base firing set

	snn_->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD, configId);
}

// set neuron parameters for Izhikevich neuron, with standard deviations
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId) {
	assert(!hasRunNetwork_); // TODO: make nice

	// wrapper identical to core func
	snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
}

// set neuron parameters for Izhikevich neuron
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId) {
	assert(!hasRunNetwork_); // TODO: make nice

	// set standard deviations of Izzy params to zero
	snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f, configId);
}

// set STDP, default
void CARLsim::setSTDP(int grpId, bool isSet, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		snn_->setSTDP(grpId,true,def_STDP_alphaLTP_,def_STDP_tauLTP_,def_STDP_alphaLTD_,def_STDP_tauLTD_,configId);
	} else { // disable STDP
		snn_->setSTDP(grpId,false,0.0f,0.0f,0.0f,0.0f,configId);
	}	
}

// set STDP, custom
void CARLsim::setSTDP(int grpId, bool isSet, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
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
	assert(!hasRunNetwork_); // TODO make nice
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		assert(snn_->isExcitatoryGroup(grpId) || snn_->isInhibitoryGroup(grpId)); // TODO make nice

		if (snn_->isExcitatoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_exc_,def_STP_tD_exc_,def_STP_tF_exc_,configId);
		else if (snn_->isInhibitoryGroup(grpId))
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
	assert(!hasRunNetwork_); // TODO make nice
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		assert(snn_->isExcitatoryGroup(grpId) || snn_->isInhibitoryGroup(grpId)); // TODO make nice

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
		printSimulationSpecs();		// first time around, show simMode etc.
	}

	hasRunNetwork_ = true;

	return snn_->runNetwork(nSec, nMsec, simMode_, ithGPU_, enablePrint_, copyState_);
}

// run network with custom options
int CARLsim::runNetwork(int nSec, int nMsec, int simType, int ithGPU, bool enablePrint, bool copyState) {
	if (!hasRunNetwork_) {
		handleUserWarnings();	// before running network, make sure user didn't provoque any user warnings
		printSimulationSpecs(); // first time around, show simMode etc.
	}

	hasRunNetwork_ = true;

	return snn_->runNetwork(nSec, nMsec, simType, ithGPU, enablePrint, copyState);	
}


// sets update cycle for log messages
void CARLsim::setLogCycle(unsigned int cnt, int mode, FILE *fp) {
	assert(!hasRunNetwork_); // TODO make nice
	snn_->setLogCycle(cnt, mode, fp);
}


// +++++++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// reads network state from file
void CARLsim::readNetwork(FILE* fid) {
	assert(!hasRunNetwork_); // TODO make nice
	snn_->readNetwork(fid);
}

// resets spike count for particular neuron group
void CARLsim::resetSpikeCntUtil(int grpId) {
	snn_->resetSpikeCntUtil(grpId);
}

// sets up a spike generator
void CARLsim::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	assert(grpId!=ALL);

	snn_->setSpikeGenerator(grpId,spikeGen,configId);
}

// set spike monitor for a group
void CARLsim::setSpikeMonitor(int grpId, SpikeMonitor* spikeMon, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	assert(grpId!=ALL);
	snn_->setSpikeMonitor(grpId,spikeMon,configId);
}


// set spike monitor for group and write spikes to file
void CARLsim::setSpikeMonitor(int grpId, const std::string& fname, int configId) {
	assert(!hasRunNetwork_); // TODO make nice
	assert(configId!=ALL);
	assert(grpId!=ALL);

	// try to open spike file
	FILE* fid = fopen(fname.c_str(),"wb"); // FIXME: where does fid get closed?
	if (fid==NULL) {
		// file could not be opened

		#if defined(CREATE_SPIKEDIR_IF_NOT_EXISTS)
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
			if (status==-1 && errno!=EEXIST) {
				fprintf(stderr,"ERROR %d: could not create spike file '%s', directory '%%CARLSIM_ROOT%%/results/' does not exist\n",errno,fname.c_str());
				exit(1);
		    }

			// now that the directory is created, fopen file
			fid = fopen(fname.c_str(),"wb");
		#else
		    // default case: print error and exit
		    fprintf(stderr,"ERROR: File \"%s\" could not be opened, please check if it exists.\n",fname.c_str());
		    fprintf(stderr,"       Enable option CREATE_SPIKEDIR_IF_NOT_EXISTS in config.h to attempt creating the "
			    "specified subdirectory automatically.\n");
		    exit(1);
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
	assert(configId!=ALL); // TODO make nice
	snn_->writePopWeights(fname,gIDpre,gIDpost,configId);
}



// +++++++++ PUBLIC METHODS: SETTERS / GETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// get connection info struct
grpConnectInfo_t* CARLsim::getConnectInfo(int connectId, int configId) {
	assert(configId!=ALL); // TODO make nice
	return snn_->getConnectInfo(connectId,configId);
}

int CARLsim::getGroupId(int grpId, int configId) {
	assert(configId!=ALL); // TODO make nice
	return snn_->getGroupId(grpId,configId);
}
// get group info struct
group_info_t CARLsim::getGroupInfo(int grpId, int configId) {
	assert(configId!=ALL); // TODO make nice
	return snn_->getGroupInfo(grpId, configId);
}

// get group info struct
std::string CARLsim::getGroupName(int grpId, int configId) {
	assert(configId!=ALL); // TODO make nice
	return snn_->getGroupName(grpId, configId);
}

unsigned int* CARLsim::getSpikeCntPtr(int grpId) {
	return snn_->getSpikeCntPtr(grpId,simMode_); // use default sim mode
}

unsigned int* CARLsim::getSpikeCntPtr(int grpId, int simType) {
	assert(simType==CPU_MODE || simType==GPU_MODE);
	return snn_->getSpikeCntPtr(grpId,simType);
}


// Sets enableGpuSpikeCntPtr to true or false.
void CARLsim::setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr) {
	snn_->setCopyFiringStateFromGPU(enableGPUSpikeCntPtr);
}



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
	for (std::map<int, grpInfo_s>::iterator iter = grpInfo_.begin(); iter != grpInfo_.end(); ++iter) {
		allSame = (iter==grpInfo_.begin()) ? iter->second.hasSetCond : allSame==iter->second.hasSetCond;
	}

	if (!allSame) {
		// TODO: make nice
		printf("USER ERROR: If one group enables conductances, then all groups (except for generators) must enable conductances.\n");
		exit(1);
	}
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

// factory function for making grpInfo_s
CARLsim::grpInfo_s CARLsim::makeGrpInfo(int grpId, bool hasSetCond) {
	grpInfo_s grpInfo = {grpId, hasSetCond};
	return grpInfo;
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