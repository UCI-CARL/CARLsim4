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

#include <carlsim.h>
#include <user_errors.h>
#include <callback_core.h>

#include <iostream>		// std::cout, std::endl
#include <sstream>		// std::stringstream
#include <algorithm>	// std::find

#include <connection_monitor_core.h>



#include <snn.h>

// includes for mkdir
#if CREATE_SPIKEDIR_IF_NOT_EXISTS
	#if (WIN32 || WIN64)
	#else
		#include <sys/stat.h>
		#include <errno.h>
		#include <libgen.h>
	#endif
#endif

// NOTE: Conceptual code documentation should go in carlsim.h. Do not include extensive high-level documentation here,
// but do document your code.
/// **************************************************************************************************************** ///
/// INIT STATIC PROPERTIES
/// **************************************************************************************************************** ///
bool CARLsim::gpuAllocation[MAX_NUM_CUDA_DEVICES] = {false};
std::string CARLsim::gpuOccupiedBy[MAX_NUM_CUDA_DEVICES];

#if (WIN32 || WIN64)
HANDLE CARLsim::gpuAllocationLock = CreateMutex(NULL, FALSE, NULL);
#else
pthread_mutex_t CARLsim::gpuAllocationLock = PTHREAD_MUTEX_INITIALIZER;
#endif
/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///

// constructor
CARLsim::CARLsim(const std::string& netName, simMode_t simMode, loggerMode_t loggerMode, int ithGPU,
						int randSeed)
{
	netName_ 					= netName;
	simMode_ 					= simMode;
	loggerMode_ 				= loggerMode;
	ithGPU_ 					= ithGPU;
	randSeed_					= randSeed;
	enablePrint_ = false;
	copyState_ = false;

	numConnections_				= 0;

	hasSetHomeoALL_ 			= false;
	hasSetHomeoBaseFiringALL_ 	= false;
	hasSetSTDPALL_ 				= false;
	hasSetSTPALL_ 				= false;
	hasSetConductances_			= false;
	carlsimState_				= CONFIG_STATE;

	snn_ = NULL;

	CARLsimInit(); // move everything else out of constructor
}

CARLsim::~CARLsim() {
	// save simulation
	if (carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE)
		saveSimulation(def_save_fileName_,def_save_synapseInfo_);

	// deallocate all dynamically allocated structures
	for (int i=0; i<groupMon_.size(); i++) {
		if (groupMon_[i]!=NULL)
			delete groupMon_[i];
		groupMon_[i]=NULL;
	}
	for (int i=0; i<connMon_.size(); i++) {
		if (connMon_[i]!=NULL)
			delete connMon_[i];
		connMon_[i]=NULL;
	}
	for (int i=0; i<spkGen_.size(); i++) {
		if (spkGen_[i]!=NULL)
			delete spkGen_[i];
		spkGen_[i]=NULL;
	}
	for (int i=0; i<connGen_.size(); i++) {
		if (connGen_[i]!=NULL)
			delete connGen_[i];
		connGen_[i]=NULL;
	}
	if (snn_!=NULL)
		delete snn_;
	snn_=NULL;

#if (WIN32 || WIN64)
	if (simMode_ == GPU_MODE) {
		WaitForSingleObject(gpuAllocationLock, INFINITE);
		gpuAllocation[ithGPU_] = false;
		ReleaseMutex(gpuAllocationLock);
	}
#else // linux
	if (simMode_ == GPU_MODE) {
		pthread_mutex_lock(&gpuAllocationLock);
		gpuAllocation[ithGPU_] = false;
		pthread_mutex_unlock(&gpuAllocationLock);
	}
#endif
}

// unsafe computations that would otherwise go in constructor
void CARLsim::CARLsimInit() {
	bool gpuAllocationResult = false;
	std::string funcName = "CARLsimInit()";

	UserErrors::assertTrue(simMode_!=UNKNOWN_SIM,UserErrors::CANNOT_BE_UNKNOWN,"CARLsim()","Simulation mode");
	UserErrors::assertTrue(loggerMode_!=UNKNOWN_LOGGER,UserErrors::CANNOT_BE_UNKNOWN,"CARLsim()","Logger mode");

	// Allocate GPU
	if (simMode_ == GPU_MODE) {
		if (ithGPU_ >= MAX_NUM_CUDA_DEVICES || ithGPU_ < 0) {
			CARLSIM_ERROR(funcName.c_str(), "Maximum number of GPUs supported by CARLsim is 8");
			exit(EXIT_FAILURE); // abort
		}
#if (WIN32 || WIN64)
		WaitForSingleObject(gpuAllocationLock, INFINITE);
		if (!gpuAllocation[ithGPU_]) {
			gpuAllocation[ithGPU_] = true;
			gpuAllocationResult = true;
			gpuOccupiedBy[ithGPU_] = netName_;
		}
		ReleaseMutex(gpuAllocationLock);
#else
		pthread_mutex_lock(&gpuAllocationLock);
		if (!gpuAllocation[ithGPU_]) {
			gpuAllocation[ithGPU_] = true;
			gpuAllocationResult = true;
			gpuOccupiedBy[ithGPU_] = netName_;
		}
		pthread_mutex_unlock(&gpuAllocationLock);
#endif
		if (!gpuAllocationResult) {
			std::string errorMsg = "GPU Allocation Conflict, GPU has been occupied by CARLsim object " + gpuOccupiedBy[ithGPU_];
			CARLSIM_ERROR(funcName.c_str(), errorMsg.c_str());
			exit(EXIT_FAILURE); // abort
		}
	}

	snn_ = new CpuSNN(netName_, simMode_, loggerMode_, ithGPU_, randSeed_);

	// set default time constants for synaptic current decay
	// TODO: add ref
	setDefaultConductanceTimeConstants(5, 0, 150, 6, 0, 150);

	// set default values for STDP params
	// TODO: add ref
	// TODO: make STDP type part of default func
	def_STDP_type_      = STANDARD;
	setDefaultESTDPparams(0.001f, 20.0f, 0.0012f, 20.0f);
	setDefaultISTDPparams(0.001f, 0.0012f, 12.0f, 40.0f);

	// set default values for STP params
	// TODO: add ref
	setDefaultSTPparams(EXCITATORY_NEURON, 0.2f, 20.0f, 700.0f);
	setDefaultSTPparams(INHIBITORY_NEURON, 0.5f, 1000.0f, 800.0f);

	// set default homeostasis params
	// Ref: Carlson, et al. (2013). Proc. of IJCNN 2013.
	setDefaultHomeostasisParams(0.1f, 10.0f);

	// set default save sim params
	// TODO: when we run executable from local dir, put save file in results/
	setDefaultSaveOptions("results/sim_"+netName_+".dat",false);
}



/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///

// +++++++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// Connects a presynaptic to a postsynaptic group using one of the primitive types
short int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
		const RangeDelay& delay, const RadiusRF& radRF, bool synWtType, float mulSynFast, float mulSynSlow) {
	std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
	std::stringstream grpId1str; grpId1str << "Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << "Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
		" is PoissonGroup, connect");
	UserErrors::assertTrue(wt.max>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.max");
	UserErrors::assertTrue(wt.min>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.min");
	UserErrors::assertTrue(wt.init>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "wt.init");
	UserErrors::assertTrue(synWtType==SYN_PLASTIC || synWtType==SYN_FIXED && wt.init==wt.max,
		UserErrors::MUST_BE_IDENTICAL, funcName, "For fixed synapses, initWt and maxWt");
	UserErrors::assertTrue(delay.min>0, UserErrors::MUST_BE_POSITIVE, funcName, "delay.min");
	UserErrors::assertTrue(radRF.radX!=0 || radRF.radY!=0 || radRF.radZ!=0, UserErrors::CANNOT_BE_ZERO, funcName,
		"Receptive field radius");
	UserErrors::assertTrue(connType.compare("one-to-one")!=0
		|| connType.compare("one-to-one")==0 && radRF.radX<=0 && radRF.radY<=0 && radRF.radZ<=0,
		UserErrors::CANNOT_BE_LARGER, funcName, "Receptive field radius", "zero for type \"one-to-one\".");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	// TODO: enable support for non-zero min
	if (fabs(wt.min)>1e-15) {
		std::cerr << funcName << ": " << wt << ". Non-zero minimum weights are not yet supported.\n" << std::endl;
		assert(false);
	}

	return snn_->connect(grpId1, grpId2, connType, wt.init, wt.max, connProb, delay.min, delay.max,
		radRF.radX, radRF.radY, radRF.radZ, mulSynFast,	mulSynSlow, synWtType);
}

// custom connectivity profile
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
		" is PoissonGroup, connect");
	UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	// TODO: check for sign of weights
	ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(this, conn);
	connGen_.push_back(CGC);
	return snn_->connect(grpId1, grpId2, CGC, 1.0f, 1.0f, synWtType, maxM, maxPreM);
}

// custom connectivity profile
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1)+"\",\""+getGroupName(grpId2)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() +
		" is PoissonGroup, connect");
	UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	assert(++numConnections_ <= MAX_nConnections);

	ConnectionGeneratorCore* CGC = new ConnectionGeneratorCore(this, conn);
	connGen_.push_back(CGC);
	return snn_->connect(grpId1, grpId2, CGC, mulSynFast, mulSynSlow, synWtType,
		maxM, maxPreM);
}

// create group of Izhikevich spiking neurons on 1D grid
int CARLsim::createGroup(const std::string& grpName, int nNeur, int neurType) {
	return createGroup(grpName, Grid3D(nNeur,1,1), neurType);
}

// create group of Izhikevich spiking neurons on 3D grid
int CARLsim::createGroup(const std::string& grpName, const Grid3D& grid, int neurType) {
	std::string funcName = "createGroup(\""+grpName+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	UserErrors::assertTrue(grid.x>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.x");
	UserErrors::assertTrue(grid.y>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.y");
	UserErrors::assertTrue(grid.z>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.z");

	// if user has called any set functions with grpId=ALL, and is now adding another group, previously set properties
	// will not apply to newly added group
	if (hasSetSTPALL_)
		userWarnings_.push_back("Make sure to call setSTP on group "+grpName);
	if (hasSetSTDPALL_)
		userWarnings_.push_back("Make sure to call setSTDP on group "+grpName);
	if (hasSetHomeoALL_)
		userWarnings_.push_back("Make sure to call setHomeostasis on group "+grpName);
	if (hasSetHomeoBaseFiringALL_)
		userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "+grpName);

	int grpId = snn_->createGroup(grpName.c_str(),grid,neurType);
	grpIds_.push_back(grpId); // keep track of all groups

	return grpId;
}

// create group of spike generators on 1D grid
int CARLsim::createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType) {
	return createSpikeGeneratorGroup(grpName, Grid3D(nNeur,1,1), neurType);
}

// create group of spike generators on 3D grid
int CARLsim::createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType) {
	std::string funcName = "createSpikeGeneratorGroup(\""+grpName+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	UserErrors::assertTrue(grid.x>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.x");
	UserErrors::assertTrue(grid.y>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.y");
	UserErrors::assertTrue(grid.z>0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grid.z");

	int grpId = snn_->createSpikeGeneratorGroup(grpName.c_str(),grid,neurType);
	grpIds_.push_back(grpId); // keep track of all groups

	return grpId;
}


// set conductance values, use defaults
void CARLsim::setConductances(bool isSet) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << ")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
		funcName.str(), "CONFIG.");
	hasSetConductances_ = true;

	if (isSet) { // enable conductances, use default values
		snn_->setConductances(true,def_tdAMPA_,0,def_tdNMDA_,def_tdGABAa_,0,def_tdGABAb_);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0);
	}

}

// set conductances values, custom
void CARLsim::setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << tdNMDA << ","
		<< tdGABAa << "," << tdGABAb << ")";
	UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(),
		funcName.str(),"CONFIG.");
	hasSetConductances_ = true;

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(true,tdAMPA,0,tdNMDA,tdGABAa,0,tdGABAb);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0);
	}
}

// set conductances values, custom
void CARLsim::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << trNMDA << "," <<
		tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << ")";
	UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(!isSet||trNMDA>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trNMDA");
	UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(!isSet||trGABAb>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
	UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trGABAb and tdGABAb");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), 
		funcName.str(), "CONFIG.");
	hasSetConductances_ = true;

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0);
	}
}

// set default homeostasis params
void CARLsim::setHomeostasis(int grpId, bool isSet) {
	std::string funcName = "setHomeostasis(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,def_homeo_scale_,def_homeo_avgTimeScale_);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId));
	} else { // disable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f);
	}
}

// set custom homeostasis params for group
void CARLsim::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale) {
	std::string funcName = "setHomeostasis(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetHomeoALL_ = grpId==ALL; // adding groups after this will not have homeostasis set

	if (isSet) { // enable homeostasis, use default values
		snn_->setHomeostasis(grpId,true,homeoScale,avgTimeScale);
		if (grpId!=ALL && hasSetHomeoBaseFiringALL_)
			userWarnings_.push_back("Make sure to call setHomeoBaseFiringRate on group "
										+ getGroupName(grpId));
	} else { // disable conductances
		snn_->setHomeostasis(grpId,false,0.0f,0.0f);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void CARLsim::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD) {
	std::string funcName = "setHomeoBaseFiringRate(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetHomeoBaseFiringALL_ = grpId; // adding groups after this will not have base firing set

	snn_->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD);
}

// set neuron parameters for Izhikevich neuron, with standard deviations
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd) {
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	// wrapper identical to core func
	snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
}

// set neuron parameters for Izhikevich neuron
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d) {
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	// set standard deviations of Izzy params to zero
	snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f);
}

// set parameters for each neuronmodulator
void CARLsim::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							   float baseACh, float tauACh, float baseNE, float tauNE) {
	std::string funcName = "setNeuromodulator(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(baseDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(base5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(baseACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(baseNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	snn_->setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE);
}

void CARLsim::setNeuromodulator(int grpId,float tauDP, float tau5HT, float tauACh, float tauNE) {
	std::string funcName = "setNeuromodulator(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	snn_->setNeuromodulator(grpId, 1.0f, tauDP, 1.0f, tau5HT, 1.0f, tauACh, 1.0f, tauNE);
}

// set STDP, default, wrapper function
void CARLsim::setSTDP(int grpId, bool isSet) {
	setESTDP(grpId, isSet);
}

// set STDP, custom, wrapper function
void CARLsim::setSTDP(int grpId, bool isSet, stdpType_t type, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD) {
		setESTDP(grpId, isSet, type, HebbianCurve(alphaLTP, tauLTP, alphaLTD, tauLTD));
}

// set ESTDP, default
void CARLsim::setESTDP(int grpId, bool isSet) {
	std::string funcName = "setESTDP(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values and type
		snn_->setESTDP(grpId, true, def_STDP_type_, HEBBIAN, def_STDP_alphaLTP_, def_STDP_tauLTP_, def_STDP_alphaLTD_, def_STDP_tauLTD_, 0.0f);
	} else { // disable STDP
		snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	}
}

// set ESTDP by stdp curve
void CARLsim::setESTDP(int grpId, bool isSet, stdpType_t type, HebbianCurve curve) {
	std::string funcName = "setESTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		snn_->setESTDP(grpId, true, type, curve.stdpCurve, curve.alphaLTP, curve.tauLTP, curve.alphaLTD, curve.tauLTD, 0.0f);
	} else { // disable STDP and DA-STDP as well
		snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	}
}

// set ESTDP by stdp curve
void CARLsim::setESTDP(int grpId, bool isSet, stdpType_t type, HalfHebbianCurve curve) {
	std::string funcName = "setESTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		snn_->setESTDP(grpId, true, type, curve.stdpCurve, curve.alphaLTP, curve.tauLTP, curve.alphaLTD, curve.tauLTD, curve.gama);
	} else { // disable STDP and DA-STDP as well
		snn_->setESTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f);
	}
}

// set ISTDP, default
void CARLsim::setISTDP(int grpId, bool isSet) {
	std::string funcName = "setISTDP(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values and type
		snn_->setISTDP(grpId, true, def_STDP_type_, CONSTANT_SYMMETRIC, def_STDP_betaLTP_, def_STDP_betaLTD_, def_STDP_lamda_, def_STDP_delta_);
	} else { // disable STDP
		snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
	}
}

// set ISTDP by stdp curve
void CARLsim::setISTDP(int grpId, bool isSet, stdpType_t type, AntiHebbianCurve curve) {
	std::string funcName = "setISTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		snn_->setISTDP(grpId, true, type, curve.stdpCurve, curve.alphaLTP, curve.alphaLTD, curve.tauLTP, curve.tauLTD);
	} else { // disable STDP and DA-STDP as well
		snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
	}
}

// set ISTDP by stdp curve
void CARLsim::setISTDP(int grpId, bool isSet, stdpType_t type, ConstantSymmetricCurve curve) {
	std::string funcName = "setISTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		snn_->setISTDP(grpId, true, type, curve.stdpCurve, curve.betaLTP, curve.betaLTD, curve.lamda, curve.delta);
	} else { // disable STDP and DA-STDP as well
		snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
	}
}

// set ISTDP by stdp curve
void CARLsim::setISTDP(int grpId, bool isSet, stdpType_t type, LinearSymmetricCurve curve) {
	std::string funcName = "setISTDP(\""+getGroupName(grpId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		snn_->setISTDP(grpId, true, type, curve.stdpCurve, curve.betaLTP, curve.betaLTD, curve.lamda, curve.delta);
	} else { // disable STDP and DA-STDP as well
		snn_->setISTDP(grpId, false, UNKNOWN_STDP, UNKNOWN_CURVE, 0.0f, 0.0f, 1.0f, 1.0f);
	}
}

// set STP, default
void CARLsim::setSTP(int grpId, bool isSet) {
	std::string funcName = "setSTP(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName, "setSTP");

		if (isExcitatoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_exc_,def_STP_tau_u_exc_,def_STP_tau_x_exc_);
		else if (isInhibitoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_inh_,def_STP_tau_u_inh_,def_STP_tau_x_inh_);
		else {
			// some error message
		}
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f);
	}
}

// set STP, custom
void CARLsim::setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x) {
	std::string funcName = "setSTP(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName,"setSTP");

		snn_->setSTP(grpId,true,STP_U,STP_tau_u,STP_tau_x);
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f);
	}
}

void CARLsim::setWeightAndWeightChangeUpdate(updateInterval_t wtANDwtChangeUpdateInterval, bool enableWtChangeDecay,
											 float wtChangeDecay) {
	std::string funcName = "setWeightAndWeightChangeUpdate()";
	UserErrors::assertTrue(wtChangeDecay > 0.0f, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(wtChangeDecay < 1.0f, UserErrors::CANNOT_BE_LARGER, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	snn_->setWeightAndWeightChangeUpdate(wtANDwtChangeUpdateInterval, enableWtChangeDecay, wtChangeDecay);
}


// +++++++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// run network with custom options
int CARLsim::runNetwork(int nSec, int nMsec, bool printRunSummary, bool copyState) {
	std::string funcName = "runNetwork()";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
				UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	// run some checks before running network for the first time
	if (carlsimState_ != EXE_STATE) {
		// if user hasn't called setConductances, set to false and disp warning
		if (!hasSetConductances_) {
			userWarnings_.push_back("CARLsim::setConductances has not been called. Setting simulation mode to CUBA.");
		}

		// make sure user didn't provoque any user warnings
		handleUserWarnings();
	}

	carlsimState_ = EXE_STATE;

	return snn_->runNetwork(nSec, nMsec, printRunSummary, copyState);	
}

// setup network with custom options
void CARLsim::setupNetwork(bool removeTempMemory) {
	std::string funcName = "setupNetwork()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	carlsimState_ = SETUP_STATE;

	snn_->setupNetwork(removeTempMemory);
}

// +++++++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

const FILE* CARLsim::getLogFpInf() { return snn_->getLogFpInf(); }
const FILE* CARLsim::getLogFpErr() { return snn_->getLogFpErr(); }
const FILE* CARLsim::getLogFpDeb() { return snn_->getLogFpDeb(); }
const FILE* CARLsim::getLogFpLog() { return snn_->getLogFpLog(); }

void CARLsim::saveSimulation(const std::string& fileName, bool saveSynapseInfo) {
	FILE* fpSave = fopen(fileName.c_str(),"wb");
	std::string funcName = "saveSimulation()";
	UserErrors::assertTrue(fpSave!=NULL,UserErrors::FILE_CANNOT_OPEN,fileName);
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	snn_->saveSimulation(fpSave,saveSynapseInfo);

	fclose(fpSave);
}

// set new file pointer for debug log file
void CARLsim::setLogDebugFp(FILE* fpLog) {
	UserErrors::assertTrue(fpLog!=NULL,UserErrors::CANNOT_BE_NULL,"setLogDebugFp","fpLog");

	snn_->setLogDebugFp(fpLog);
}

// set new file pointer for all files
void CARLsim::setLogsFp(FILE* fpInf, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
	UserErrors::assertTrue(loggerMode_==CUSTOM,UserErrors::MUST_BE_LOGGER_CUSTOM,"setLogsFp","Logger mode");

	snn_->setLogsFp(fpInf,fpErr,fpDeb,fpLog);
}


// +++++++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// reads network state from file
void CARLsim::loadSimulation(FILE* fid) {
	std::string funcName = "loadSimulation()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	snn_->loadSimulation(fid);
}

void CARLsim::reassignFixedWeights(short int connectId, float weightMatrix[], int matrixSize) {
	std::string funcName = "reassignFixedWeights()";
	UserErrors::assertTrue(loggerMode_==CUSTOM,UserErrors::MUST_BE_LOGGER_CUSTOM,"setLogsFp","Logger mode");
	UserErrors::assertTrue(carlsimState_==SETUP_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP.");

	snn_->reassignFixedWeights(connectId,weightMatrix,matrixSize);
}


// resets spike count for particular neuron group
void CARLsim::resetSpikeCntUtil(int grpId) {
	std::string funcName = "resetSpikeCntUtil()";
	UserErrors::assertTrue(false, UserErrors::IS_DEPRECATED, funcName);

	UserErrors::assertTrue(carlsimState_==SETUP_STATE||carlsimState_==EXE_STATE,
		UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	snn_->resetSpikeCntUtil(grpId);
}

// resets spike counters
void CARLsim::resetSpikeCounter(int grpId) {
	std::string funcName = "resetSpikeCounter()";
	UserErrors::assertTrue(carlsimState_==SETUP_STATE||carlsimState_==EXE_STATE,
		UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	snn_->resetSpikeCounter(grpId);
}

// set spike monitor for group and write spikes to file
ConnectionMonitor* CARLsim::setConnectionMonitor(int grpIdPre, int grpIdPost, const std::string& fname) {
	std::string funcName = "setConnectionMonitor(\"" + getGroupName(grpIdPre) + "\",\"" + getGroupName(grpIdPost)
		+ "\",\"" + fname + "\")";
	UserErrors::assertTrue(grpIdPre!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPre"); // grpId can't be ALL
	UserErrors::assertTrue(grpIdPost!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPost");
	UserErrors::assertTrue(grpIdPre>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpIdPre");// grpId can't be negative
	UserErrors::assertTrue(grpIdPost>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpIdPost");
	UserErrors::assertTrue(carlsimState_==SETUP_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
		funcName, "SETUP.");

	// empty string: use default name for binary file
#if (WIN32 || WIN64)
	// \TODO make default path for Windows platform
	std::string fileName = fname.empty() ? "NULL" : fname;
#else
	std::string fileName = fname.empty() ? "results/conn_" + snn_->getGroupName(grpIdPre) + "_"
		+ snn_->getGroupName(grpIdPost) + ".dat" : fname;
#endif

	FILE* fid;
	if (fileName=="NULL") {
		// user does not want a binary file created
		fid = NULL;
	} else {
		// try to open spike file
		fid = fopen(fileName.c_str(),"wb");
		if (fid==NULL) {
			// file could not be opened

			// default case: print error and exit
			std::string fileError = " Double-check file permissions and make sure directory exists.";
			UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
		}
	}

	// return SpikeMonitor object
	return snn_->setConnectionMonitor(grpIdPre, grpIdPost, fid);
}

void CARLsim::setExternalCurrent(int grpId, const std::vector<float>& current) {
	std::string funcName = "setExternalCurrent(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(current.size()==getGroupNumNeurons(grpId), UserErrors::MUST_BE_IDENTICAL, funcName,
		"current.size()", "number of neurons in the group.");
	UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
	UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==EXE_STATE, 
		UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");
	
	snn_->setExternalCurrent(grpId, current);
}

void CARLsim::setExternalCurrent(int grpId, float current) {
	std::string funcName = "setExternalCurrent(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
	UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==EXE_STATE, 
		UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	std::vector<float> vecCurrent(getGroupNumNeurons(grpId), current);
	snn_->setExternalCurrent(grpId, vecCurrent);
}

// set group monitor for a group
void CARLsim::setGroupMonitor(int grpId, GroupMonitor* groupMon) {
	//std::string funcName = "setGroupMonitor(\""+getGroupName(grpId)+"\",GroupMonitor*)";
	//UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL
	//UserErrors::assertTrue(carlsimState_==CONFIG_STATE || carlsimState_==SETUP_STATE,
	//				UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG or SETUP.");

	//GroupMonitorCore* GMC = new GroupMonitorCore(this, groupMon);
	//groupMon_.push_back(GMC);
	//snn_->setGroupMonitor(grpId, GMC);
}

// sets a spike counter for a group
void CARLsim::setSpikeCounter(int grpId, int recordDur) {
	std::stringstream funcName;	funcName << "setSpikeCounter(" << grpId << "," << recordDur << ")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE || carlsimState_==SETUP_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), "CONFIG or SETUP.");

	snn_->setSpikeCounter(grpId,recordDur);
}

// sets up a spike generator
void CARLsim::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen) {
	std::string funcName = "setSpikeGenerator(\""+getGroupName(grpId)+"\")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL
	UserErrors::assertTrue(isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
	UserErrors::assertTrue(spikeGen!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE,	UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	SpikeGeneratorCore* SGC = new SpikeGeneratorCore(this, spikeGen);
	spkGen_.push_back(SGC);
	snn_->setSpikeGenerator(grpId, SGC);
}

// set spike monitor for group and write spikes to file
SpikeMonitor* CARLsim::setSpikeMonitor(int grpId, const std::string& fname) {
	std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId)+"\",\""+fname+"\")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// grpId can't be ALL
	UserErrors::assertTrue(grpId>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "grpId"); // grpId can't be negative
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE || carlsimState_==SETUP_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG or SETUP.");

	// empty string: use default name for binary file
#if (WIN32 || WIN64)
	// \TODO make default path for Windows platform
	std::string fileName = fname.empty() ? "NULL" : fname;
#else
	std::string fileName = fname.empty() ? "results/spk"+snn_->getGroupName(grpId)+".dat" : fname;
#endif

	FILE* fid;
	if (fileName=="NULL") {
		// user does not want a binary file created
		fid = NULL;
	} else {
		// try to open spike file
		fid = fopen(fileName.c_str(),"wb");
		if (fid==NULL) {
			// file could not be opened

			// default case: print error and exit
			std::string fileError = " Double-check file permissions and make sure directory exists.";
			UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
		}
	}

	// return SpikeMonitor object
	return snn_->setSpikeMonitor(grpId, fid);
}

// assign spike rate to poisson group
void CARLsim::setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod) {
	std::string funcName = "setSpikeRate()";
	UserErrors::assertTrue(carlsimState_==SETUP_STATE || carlsimState_==EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");
	UserErrors::assertTrue(spikeRate->getNumNeurons()==getGroupNumNeurons(grpId), UserErrors::MUST_BE_IDENTICAL,
		funcName, "PoissonRate length", "the number of neurons in the group.");
	UserErrors::assertTrue(!spikeRate->isOnGPU() || spikeRate->isOnGPU()&&getSimMode()==GPU_MODE,
		UserErrors::CAN_ONLY_BE_CALLED_IN_MODE, funcName, "PoissonRate on GPU", "GPU_MODE.");

	snn_->setSpikeRate(grpId, spikeRate, refPeriod);
}

// Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
// weight values back to their default values by setting resetWeights = true.
void CARLsim::updateNetwork(bool resetFiringInfo, bool resetWeights) {
	std::string funcName = "updateNetwork()";
	UserErrors::assertTrue(false, UserErrors::IS_DEPRECATED, funcName);

	UserErrors::assertTrue(carlsimState_==EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "EXECUTION.");

	snn_->updateNetwork(resetFiringInfo,resetWeights);
}

// function writes population weights from gIDpre to gIDpost to file fname in binary.
void CARLsim::writePopWeights(std::string fname, int gIDpre, int gIDpost) {
	std::string funcName = "writePopWeights("+fname+")";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	snn_->writePopWeights(fname,gIDpre,gIDpost);
}



// +++++++++ PUBLIC METHODS: SETTERS / GETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

std::vector<float> CARLsim::getConductanceAMPA(int grpId) {
	std::string funcName = "getConductanceAMPA()";
	UserErrors::assertTrue(carlsimState_ == EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, 
		funcName, funcName, "EXECUTION.");
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
		"[0,getNumGroups()]");

	return snn_->getConductanceAMPA(grpId);
}

std::vector<float> CARLsim::getConductanceNMDA(int grpId) {
	std::string funcName = "getConductanceNMDA()";
	UserErrors::assertTrue(carlsimState_ == EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, 
		funcName, funcName, "EXECUTION.");
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
		"[0,getNumGroups()]");

	return snn_->getConductanceNMDA(grpId);
}

std::vector<float> CARLsim::getConductanceGABAa(int grpId) {
	std::string funcName = "getConductanceGABAa()";
	UserErrors::assertTrue(carlsimState_ == EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, 
		funcName, funcName, "EXECUTION.");
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
		"[0,getNumGroups()]");

	return snn_->getConductanceGABAa(grpId);
}

std::vector<float> CARLsim::getConductanceGABAb(int grpId) {
	std::string funcName = "getConductanceGABAb()";
	UserErrors::assertTrue(carlsimState_ == EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, 
		funcName, funcName, "EXECUTION.");
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "grpId",
		"[0,getNumGroups()]");

	return snn_->getConductanceGABAb(grpId);
}

uint8_t* CARLsim::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	std::string funcName = "getDelays()";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");
	UserErrors::assertTrue(gIDpre>=0 && gIDpre<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "gIDpre",
		"[0,getNumGroups()]");
	UserErrors::assertTrue(gIDpost>=0 && gIDpost<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName, "gIDpre",
		"[0,getNumGroups()]");

	return snn_->getDelays(gIDpre,gIDpost,Npre,Npost,delays);
}

Grid3D CARLsim::getGroupGrid3D(int grpId) {
	std::stringstream funcName;	funcName << "getConnectInfo(" << grpId << ")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or EXECUTION.");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), 
		"grpId", "[0,getNumGroups()]");

	return snn_->getGroupGrid3D(grpId);
}

int CARLsim::getGroupId(std::string grpName) {
	std::string funcName = "getGroupId("+grpName+")";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	return snn_->getGroupId(grpName);
}

std::string CARLsim::getGroupName(int grpId) {
	std::stringstream funcName; funcName << "getGroupName(" << grpId << ")";
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
		"[0,getNumGroups()]");

	return snn_->getGroupName(grpId);
}

int CARLsim::getGroupStartNeuronId(int grpId) {
	std::stringstream funcName; funcName << "getGroupStartNeuronId(" << grpId << ")";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or EXECUTION.");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
		"[0,getNumGroups()]");

	return snn_->getGroupStartNeuronId(grpId);
}

int CARLsim::getGroupEndNeuronId(int grpId) {
	std::stringstream funcName; funcName << "getGroupEndNeuronId(" << grpId << ")";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), funcName.str(), "SETUP or EXECUTION.");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
		"[0,getNumGroups()]");

	return snn_->getGroupEndNeuronId(grpId);
}

int CARLsim::getGroupNumNeurons(int grpId) {
	std::stringstream funcName; funcName << "getGroupNumNeurons(" << grpId << ")";
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), "grpId",
		"[0,getNumGroups()]");

	return snn_->getGroupNumNeurons(grpId);
}

Point3D CARLsim::getNeuronLocation3D(int neurId) {
	std::stringstream funcName;	funcName << "getNeuronLocation3D(" << neurId << ")";
	UserErrors::assertTrue(neurId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "neurId");
	UserErrors::assertTrue(neurId>=0 && neurId<getNumNeurons(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), 
		"neurId", "[0,getNumNeurons()]");

	return snn_->getNeuronLocation3D(neurId);
}

Point3D CARLsim::getNeuronLocation3D(int grpId, int relNeurId) {
	std::stringstream funcName;	funcName << "getNeuronLocation3D(" << grpId << "," << relNeurId << ")";
	UserErrors::assertTrue(relNeurId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "neurId");
	UserErrors::assertTrue(grpId>=0 && grpId<getNumGroups(), UserErrors::MUST_BE_IN_RANGE, funcName.str(), 
		"grpId", "[0,getNumGroups()]");
	UserErrors::assertTrue(relNeurId>=0 && relNeurId<getGroupNumNeurons(grpId), UserErrors::MUST_BE_IN_RANGE,
		funcName.str(), "relNeurId", "[0,getGroupNumNeurons()]");

	return snn_->getNeuronLocation3D(grpId, relNeurId);
}

int CARLsim::getNumConnections() { return snn_->getNumConnections(); }

int CARLsim::getNumGroups() { return snn_->getNumGroups(); }
int CARLsim::getNumNeurons() { return snn_->getNumNeurons(); }
int CARLsim::getNumNeuronsReg() { return snn_->getNumNeuronsReg(); }
int CARLsim::getNumNeuronsRegExc() { return snn_->getNumNeuronsRegExc(); }
int CARLsim::getNumNeuronsRegInh() { return snn_->getNumNeuronsRegInh(); }
int CARLsim::getNumNeuronsGen() { return snn_->getNumNeuronsGen(); }
int CARLsim::getNumNeuronsGenExc() { return snn_->getNumNeuronsGenExc(); }
int CARLsim::getNumNeuronsGenInh() { return snn_->getNumNeuronsGenInh(); }

int CARLsim::getNumPreSynapses() {
	std::string funcName = "getNumPreSynapses()";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	return snn_->getNumPreSynapses();
}

int CARLsim::getNumSynapticConnections(short int connectionId) {
	std::stringstream funcName;	funcName << "getNumConnections(" << connectionId << ")";
	UserErrors::assertTrue(connectionId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "connectionId");
	UserErrors::assertTrue(connectionId>=0 && connectionId<getNumConnections(), UserErrors::MUST_BE_IN_RANGE, 
		funcName.str(), "connectionId", "[0,getNumSynapticConnections()]");
	return snn_->getNumSynapticConnections(connectionId);
}
int CARLsim::getNumPostSynapses() {
	std::string funcName = "getNumPostSynapses()";
	UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
					UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "SETUP or EXECUTION.");

	return snn_->getNumPostSynapses(); }


GroupSTDPInfo_t CARLsim::getGroupSTDPInfo(int grpId) {
	std::string funcName = "getGroupSTDPInfo()";
	//UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
	//				UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, "SETUP or EXECUTION.");

	return snn_->getGroupSTDPInfo(grpId);
}

GroupNeuromodulatorInfo_t CARLsim::getGroupNeuromodulatorInfo(int grpId) {
	std::string funcName = "getGroupNeuromodulatorInfo()";
	//UserErrors::assertTrue(carlsimState_ == SETUP_STATE || carlsimState_ == EXE_STATE,
	//				UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, "SETUP or EXECUTION.");

	return snn_->getGroupNeuromodulatorInfo(grpId);
}

simMode_t CARLsim::getSimMode() { return simMode_; }

uint64_t CARLsim::getSimTime() { return snn_->getSimTime(); }
uint32_t CARLsim::getSimTimeSec() { return snn_->getSimTimeSec(); }
uint32_t CARLsim::getSimTimeMsec() { return snn_->getSimTimeMs(); }

int* CARLsim::getSpikeCntPtr(int grpId) {
	std::string funcName = "getSpikeCntPtr()";
	UserErrors::assertTrue(false, UserErrors::IS_DEPRECATED, funcName);

	UserErrors::assertTrue(carlsimState_==EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "EXECUTION.");

	return snn_->getSpikeCntPtr(grpId);
}

// get spiking information out for a given group
int* CARLsim::getSpikeCounter(int grpId) {
	std::stringstream funcName;	funcName << "getSpikeCounter(" << grpId << ")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");
	UserErrors::assertTrue(carlsimState_==EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), 
		"EXECUTION.");

	return snn_->getSpikeCounter(grpId);
}

bool CARLsim::isExcitatoryGroup(int grpId) { return snn_->isExcitatoryGroup(grpId); }
bool CARLsim::isInhibitoryGroup(int grpId) { return snn_->isInhibitoryGroup(grpId); }
bool CARLsim::isPoissonGroup(int grpId) { return snn_->isPoissonGroup(grpId); }

// Sets enableGpuSpikeCntPtr to true or false.
void CARLsim::setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr) {
	std::string funcName = "setCopyFiringStateFromGPU()";
	UserErrors::assertTrue(false, UserErrors::IS_DEPRECATED, funcName);

	UserErrors::assertTrue(carlsimState_==EXE_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "EXECUTION.");

	snn_->setCopyFiringStateFromGPU(enableGPUSpikeCntPtr);
}



// +++++++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// set default values for conductance decay times
void CARLsim::setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb,
int tdGABAb) {
	std::stringstream funcName;	funcName << "setDefaultConductanceTimeConstants(" << tdAMPA << "," << trNMDA <<
		"," << tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << ")";
	UserErrors::assertTrue(tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(trNMDA>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trNMDA");
	UserErrors::assertTrue(tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(trGABAb>=0, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAb");
	UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
	UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trGABAb and tdGABAb");
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName.str(), 
		"CONFIG.");

	def_tdAMPA_  = tdAMPA;
	def_trNMDA_  = trNMDA;
	def_tdNMDA_  = tdNMDA;
	def_tdGABAa_ = tdGABAa;
	def_trGABAb_ = trGABAb;
	def_tdGABAb_ = tdGABAb;
}

void CARLsim::setDefaultHomeostasisParams(float homeoScale, float avgTimeScale) {
	std::string funcName = "setDefaultHomeostasisparams()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	assert(avgTimeScale>0); // TODO make nice

	def_homeo_scale_ = homeoScale;
	def_homeo_avgTimeScale_ = avgTimeScale;
}

void CARLsim::setDefaultSaveOptions(std::string fileName, bool saveSynapseInfo) {
	std::string funcName = "setDefaultSaveOptions()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	def_save_fileName_ = fileName;
	def_save_synapseInfo_ = saveSynapseInfo;

	// try to open save file to make sure we have permission (so the user immediately knows about the error and doesn't
	// have to wait until their simulation run has ended)
	FILE* fpTry = fopen(def_save_fileName_.c_str(),"wb");
	UserErrors::assertTrue(fpTry!=NULL,UserErrors::FILE_CANNOT_OPEN,"Default save file",def_save_fileName_);
	fclose(fpTry);
}

// wrapper function, set default values for E-STDP params
void CARLsim::setDefaultSTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD) {
	setDefaultESTDPparams(alphaLTP, tauLTP, alphaLTD, tauLTD);
}

// set default values for E-STDP params
void CARLsim::setDefaultESTDPparams(float alphaLTP, float tauLTP, float alphaLTD, float tauLTD) {
	std::string funcName = "setDefaultESTDPparams()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	UserErrors::assertTrue(alphaLTP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(alphaLTD > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauLTP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauLTD > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	def_STDP_alphaLTP_ = alphaLTP;
	def_STDP_tauLTP_ = tauLTP;
	def_STDP_alphaLTD_ = alphaLTD;
	def_STDP_tauLTD_ = tauLTD;
}

// set default values for I-STDP params
void CARLsim::setDefaultISTDPparams(float betaLTP, float betaLTD, float lamda, float delta) {
	std::string funcName = "setDefaultISTDPparams()";
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");
	UserErrors::assertTrue(betaLTP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(betaLTD > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(lamda > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(delta > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	def_STDP_betaLTP_ = betaLTP;
	def_STDP_betaLTD_ = betaLTD;
	def_STDP_lamda_ = lamda;
	def_STDP_delta_ = delta;
}

// set default STP values for an EXCITATORY_NEURON or INHIBITORY_NEURON
void CARLsim::setDefaultSTPparams(int neurType, float STP_U, float STP_tau_u, float STP_tau_x) {
	std::string funcName = "setDefaultSTPparams()";
	UserErrors::assertTrue(neurType==EXCITATORY_NEURON || neurType==INHIBITORY_NEURON, UserErrors::WRONG_NEURON_TYPE,
									funcName);
	UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, funcName, "CONFIG.");

	assert(STP_tau_u>0.0f);
	assert(STP_tau_x>0.0f);

	switch (neurType) {
		case EXCITATORY_NEURON:
			def_STP_U_exc_ = STP_U;
			def_STP_tau_u_exc_ = STP_tau_u;
			def_STP_tau_x_exc_ = STP_tau_x;
			break;
		case INHIBITORY_NEURON:
			def_STP_U_inh_ = STP_U;
			def_STP_tau_u_inh_ = STP_tau_u;
			def_STP_tau_x_inh_ = STP_tau_x;
			break;
		default:
			// some error message instead of assert
			break;
	}
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// check whether grpId exists in grpIds_
bool CARLsim::existsGrpId(int grpId) {
	return std::find(grpIds_.begin(), grpIds_.end(), grpId)!=grpIds_.end();
}

// print all user warnings, continue only after user input
void CARLsim::handleUserWarnings() {
	if (userWarnings_.size()) {
		for (int i=0; i<userWarnings_.size(); i++)
			CARLSIM_WARN("runNetwork()",userWarnings_[i].c_str());

		fprintf(stdout,"Ignore warnings and continue? Y/n ");
		char ignoreWarn = std::cin.get();
		if (std::cin.fail() || ignoreWarn!='y' && ignoreWarn!='Y') {
			fprintf(stdout,"Exiting...\n");
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
