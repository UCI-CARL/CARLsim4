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

#include <string>		// std::string
#include <iostream>		// std::cout, std::endl
#include <sstream>		// std::stringstream
#include <algorithm>	// std::find

#include <carlsim.h>
#include <user_errors.h>
#include <callback_core.h>

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
	 * all the neurons (neurIds) that have spiked during the last timeInterval ms (usually 1000).
	 * It can be called for less than 1000 ms at the end of a simulation.
	 * This implementation will iterate over all neuron IDs and spike times, and print them to file (binary).
	 * To save space, neuron IDs are stored in a continuous (flattened) list, whereas timeCnts holds the number of
	 * neurons that have spiked at each time step (reduced AER).
	 * Example: There are 3 neurons, where neuron with ID 0 spikes at time 1, neurons with ID 1 and 2 both spike at
	 *  		time 3. Then neurIds = {0,1,2} and timeCnts = {0,1,0,2,0,...,0}. Note that neurIds could also be {0,2,1}
	 *
	 * \param[in] snn 		   pointer to an instance of CARLsimCore
	 * \param[in] grpId 	   the group ID from which to record spikes
	 * \param[in] neurIds	   pointer to a flattened list that contains all the IDs of neurons that have spiked within
	 *                         the last 1000 ms.
	 * \param[in] timeCnts 	   pointer to a data structures that holds the number of spikes at each time step during the
	 *  					   last 1000 ms. timeCnts[i] will hold the number of spikes in the i-th millisecond.
	 * \param[in] timeInterval the time interval to parse (usually 1000ms)
	 */
	void update(CARLsim* s, int grpId, unsigned int* neurIds, unsigned int* timeCnts, int timeInterval) {
		int pos    = 0; // keep track of position in flattened list of neuron IDs

		for (int t=0; t < timeInterval; t++) {
			for(int i=0; i<timeCnts[t];i++,pos++) {
				// timeInterval might be < 1000 at the end of a simulation
				int time = t + s->getSimTime() - timeInterval;
				assert(time>=0);
				
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

// constructor
CARLsim::CARLsim(std::string netName, simMode_t simMode, loggerMode_t loggerMode, int ithGPU, int nConfig,
						int randSeed)
{
	netName_ 					= netName;
	simMode_ 					= simMode;
	loggerMode_ 				= loggerMode;
	ithGPU_ 					= ithGPU;
	nConfig_ 					= nConfig;
	randSeed_					= randSeed;
	enablePrint_ = false;
	copyState_ = false;

	numConnections_				= 0;

	hasRunNetwork_  			= false;
	hasSetHomeoALL_ 			= false;
	hasSetHomeoBaseFiringALL_ 	= false;
	hasSetSTDPALL_ 				= false;
	hasSetSTPALL_ 				= false;

	snn_ = NULL;

	CARLsimInit(); // move everything else out of constructor
}

CARLsim::~CARLsim() {
	// deallocate all dynamically allocated structures
	if (snn_!=NULL)
		delete snn_;
}

// unsafe computations that would otherwise go in constructor
void CARLsim::CARLsimInit() {
	UserErrors::assertTrue(simMode_!=UNKNOWN_SIM,UserErrors::CANNOT_BE_UNKNOWN,"CARLsim()","Simulation mode");
	UserErrors::assertTrue(loggerMode_!=UNKNOWN_LOGGER,UserErrors::CANNOT_BE_UNKNOWN,"CARLsim()","Logger mode");
	snn_ = new CpuSNN(netName_, simMode_, loggerMode_, ithGPU_, nConfig_, randSeed_);

	// set default time constants for synaptic current decay
	// TODO: add ref
	def_tdAMPA_  = 5;	// default decay time for AMPA (ms)
	def_trNMDA_  = 0;	// default rise time for NMDA (ms)
	def_tdNMDA_  = 150;	// default decay time for NMDA (ms)
	def_tdGABAa_ = 6;	// default decay time for GABAa (ms)
	def_trGABAb_ = 0;   // default rise time for GABAb (ms)
	def_tdGABAb_ = 150;	// default decay time for GABAb (ms)

	// set default values for STDP params
	// TODO: add ref
	def_STDP_type_      = STANDARD;
	def_STDP_alphaLTP_ = 0.001f;
	def_STDP_tauLTP_   = 20.0f;
	def_STDP_alphaLTD_ = 0.0012f;
	def_STDP_tauLTD_   = 20.0f;

	// set default values for STP params
	// TODO: add ref
	def_STP_U_exc_  = 0.2f;
	def_STP_tau_u_exc_ = 20.0f;
	def_STP_tau_x_exc_ = 700.0f;
	def_STP_U_inh_  = 0.5f;
	def_STP_tau_u_inh_ = 1000.0f;
	def_STP_tau_x_inh_ = 800.0f;

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
short int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << "Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << "Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() + 
		" is PoissonGroup, connect");
	UserErrors::assertTrue(!isExcitatoryGroup(grpId1) || wt>0, UserErrors::MUST_BE_POSITIVE, funcName, "wt");
	UserErrors::assertTrue(!isInhibitoryGroup(grpId1) || wt<0, UserErrors::MUST_BE_NEGATIVE, funcName, "wt");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	return snn_->connect(grpId1, grpId2, connType, wt, wt, connProb, delay, delay, 1.0f, 1.0f, SYN_FIXED);
}

// shortcut to create SYN_FIXED connections with one weight / delay and two scaling factors for synaptic currents
short int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float wt, float connProb, uint8_t delay,
							float mulSynFast, float mulSynSlow) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << "Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << "Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() + 
		" is PoissonGroup, connect");
	UserErrors::assertTrue(!isExcitatoryGroup(grpId1) || wt>0, UserErrors::MUST_BE_POSITIVE, funcName, "wt");
	UserErrors::assertTrue(!isInhibitoryGroup(grpId1) || wt<0, UserErrors::MUST_BE_NEGATIVE, funcName, "wt");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	return snn_->connect(grpId1,grpId2,connType,wt,wt,connProb,delay,delay,mulSynFast,mulSynSlow,SYN_FIXED);
}

// shortcut to create SYN_FIXED/SYN_PLASTIC connections with initWt/maxWt, minDelay/maxDelay, but to omit
// scaling factors for synaptic conductances (default is 1.0 for both)
short int CARLsim::connect(int grpId1, int grpId2, const std::string& connType, float initWt, float maxWt,
							float connProb,	uint8_t minDelay, uint8_t maxDelay, bool synWtType)
{
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() + 
		" is PoissonGroup, connect");
	UserErrors::assertTrue(!isExcitatoryGroup(grpId1) || maxWt>0, UserErrors::MUST_BE_POSITIVE, funcName, "maxWt");
	UserErrors::assertTrue(!isInhibitoryGroup(grpId1) || maxWt<0, UserErrors::MUST_BE_NEGATIVE, funcName, "maxWt");
	UserErrors::assertTrue(initWt*maxWt>=0, UserErrors::MUST_HAVE_SAME_SIGN, funcName, "initWt and maxWt");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	return snn_->connect(grpId1,grpId2,connType,initWt,maxWt,connProb,minDelay,maxDelay,1.0f,1.0f,synWtType);
}

// custom connectivity profile
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() + 
		" is PoissonGroup, connect");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName);

	// TODO: check for sign of weights
	return snn_->connect(grpId1, grpId2, new ConnectionGeneratorCore(this, conn), 1.0f, 1.0f, synWtType, maxM, maxPreM);
}

// custom connectivity profile
short int CARLsim::connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	std::string funcName = "connect(\""+getGroupName(grpId1,0)+"\",\""+getGroupName(grpId2,0)+"\")";
	std::stringstream grpId1str; grpId1str << ". Group Id " << grpId1;
	std::stringstream grpId2str; grpId2str << ". Group Id " << grpId2;
	UserErrors::assertTrue(grpId1!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId1str.str()); // grpId can't be ALL
	UserErrors::assertTrue(grpId2!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, grpId2str.str());
	UserErrors::assertTrue(!isPoissonGroup(grpId2), UserErrors::WRONG_NEURON_TYPE, funcName, grpId2str.str() + 
		" is PoissonGroup, connect");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(conn!=NULL, UserErrors::CANNOT_BE_NULL, funcName);
	assert(++numConnections_ <= MAX_nConnections);

	return snn_->connect(grpId1, grpId2, new ConnectionGeneratorCore(this, conn), mulSynFast, mulSynSlow, synWtType, maxM, maxPreM);
}


// create group of Izhikevich spiking neurons
int CARLsim::createGroup(std::string grpName, int nNeur, int neurType, int configId) {
	std::string funcName = "createGroup(\""+grpName+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

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
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	int grpId = snn_->createSpikeGeneratorGroup(grpName.c_str(),nNeur,neurType,configId);
	grpIds_.push_back(grpId); // keep track of all groups

	return grpId;
}


// set conductance values, use defaults
void CARLsim::setConductances(bool isSet, int configId) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << "," << configId << ")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName.str()); // can't change setup after run

	if (isSet) { // enable conductances, use default values
		snn_->setConductances(true,def_tdAMPA_,0,def_tdNMDA_,def_tdGABAa_,0,def_tdGABAb_,configId);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0,configId);
	}

}

// set conductances values, custom
void CARLsim::setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb, int configId) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << tdNMDA << ","
		<< tdGABAa << "," << tdGABAb << "," << configId << ")";
	UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName.str()); // can't change setup after run

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(true,tdAMPA,0,tdNMDA,tdGABAa,0,tdGABAb,configId);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0,configId);
	}
}

// set conductances values, custom
void CARLsim::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb,
int configId) {
	std::stringstream funcName; funcName << "setConductances(" << isSet << "," << tdAMPA << "," << trNMDA << "," <<
		tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << "," << configId << ")";
	UserErrors::assertTrue(!isSet||tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(!isSet||trNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trNMDA");
	UserErrors::assertTrue(!isSet||tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(!isSet||tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(!isSet||trGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(!isSet||tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
	UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trGABAb and tdGABAb");
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName.str()); // can't change setup after run

	if (isSet) { // enable conductances, use custom values
		snn_->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb,configId);
	} else { // disable conductances
		snn_->setConductances(false,0,0,0,0,0,0,configId);
	}
}

// set default homeostasis params
void CARLsim::setHomeostasis(int grpId, bool isSet, int configId) {
	std::string funcName = "setHomeostasis(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

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
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

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
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	hasSetHomeoBaseFiringALL_ = grpId=ALL; // adding groups after this will not have base firing set

	snn_->setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD, configId);
}

// set neuron parameters for Izhikevich neuron, with standard deviations
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
							 		float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId)
{
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// wrapper identical to core func
	snn_->setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
}

// set neuron parameters for Izhikevich neuron
void CARLsim::setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d, int configId) {
	std::string funcName = "setNeuronParameters(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	// set standard deviations of Izzy params to zero
	snn_->setNeuronParameters(grpId, izh_a, 0.0f, izh_b, 0.0f, izh_c, 0.0f, izh_d, 0.0f, configId);
}

// set parameters for each neuronmodulator
void CARLsim::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							   float baseACh, float tauACh, float baseNE, float tauNE, int configId) {
	std::string funcName = "setNeuromodulator(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(baseDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(base5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(baseACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(baseNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);

	snn_->setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE, configId);
}

void CARLsim::setNeuromodulator(int grpId,float tauDP, float tau5HT, float tauACh, float tauNE, int configId) {
	std::string funcName = "setNeuromodulator(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(tauDP > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tau5HT > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauACh > 0, UserErrors::MUST_BE_POSITIVE, funcName);
	UserErrors::assertTrue(tauNE > 0, UserErrors::MUST_BE_POSITIVE, funcName);

	snn_->setNeuromodulator(grpId, 1.0f, tauDP, 1.0f, tau5HT, 1.0f, tauACh, 1.0f, tauNE, configId);
}

// set STDP, default
void CARLsim::setSTDP(int grpId, bool isSet, int configId) {
	std::string funcName = "setSTDP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values and type
		snn_->setSTDP(grpId, true, def_STDP_type_, def_STDP_alphaLTP_, def_STDP_tauLTP_, def_STDP_alphaLTD_,
			def_STDP_tauLTD_, configId);
	} else { // disable STDP
		snn_->setSTDP(grpId, false, UNKNOWN_STDP, 0.0f, 0.0f, 0.0f, 0.0f, configId);
	}	
}

// set STDP, custom
void CARLsim::setSTDP(int grpId, bool isSet, stdpType_t type, float alphaLTP, float tauLTP, float alphaLTD,
		float tauLTD, int configId) {
	std::string funcName = "setSTDP(\""+getGroupName(grpId,configId)+","+stdpType_string[type]+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(type!=UNKNOWN_STDP, UserErrors::CANNOT_BE_UNKNOWN, funcName, "Mode");
	hasSetSTDPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use custom values
		assert(tauLTP>0); // TODO make nice
		assert(tauLTD>0);
		snn_->setSTDP(grpId, true, type, alphaLTP, tauLTP, alphaLTD, tauLTD, configId);
	} else { // disable STDP and DA-STDP as well
		snn_->setSTDP(grpId, false, UNKNOWN_STDP, 0.0f, 0.0f, 0.0f, 0.0f, configId);
	}
}

// set STP, default
void CARLsim::setSTP(int grpId, bool isSet, int configId) {
	std::string funcName = "setSTP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName, "setSTP");

		if (isExcitatoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_exc_,def_STP_tau_u_exc_,def_STP_tau_x_exc_,configId);
		else if (isInhibitoryGroup(grpId))
			snn_->setSTP(grpId,true,def_STP_U_inh_,def_STP_tau_u_inh_,def_STP_tau_x_inh_,configId);
		else {
			// some error message
		}
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f,configId);
	}		
}

// set STP, custom
void CARLsim::setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x, int configId) {
	std::string funcName = "setSTP(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	hasSetSTPALL_ = grpId==ALL; // adding groups after this will not have conductances set

	if (isSet) { // enable STDP, use default values
		UserErrors::assertTrue(isExcitatoryGroup(grpId) || isInhibitoryGroup(grpId), UserErrors::WRONG_NEURON_TYPE,
									funcName,"setSTP");

		snn_->setSTP(grpId,true,STP_U,STP_tau_u,STP_tau_x,configId);
	} else { // disable STDP
		snn_->setSTP(grpId,false,0.0f,0.0f,0.0f,configId);
	}		
}

void CARLsim::setWeightAndWeightChangeUpdate(updateIterval_t updateWtInterval, updateIterval_t updateWtChangeInterval,
											 int tauWeightChange) {
	std::string funcName = "setWeightAndWeightChangeUpdate()";
	UserErrors::assertTrue(updateWtChangeInterval <= updateWtInterval, UserErrors::CANNOT_BE_LARGER, funcName);
	UserErrors::assertTrue(tauWeightChange > 0, UserErrors::MUST_BE_POSITIVE, funcName);

	snn_->setWeightAndWeightChangeUpdate(updateWtInterval, updateWtChangeInterval, tauWeightChange);
}


// +++++++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// run network with custom options
int CARLsim::runNetwork(int nSec, int nMsec, bool copyState) {
	if (!hasRunNetwork_) {
		handleUserWarnings();	// before running network, make sure user didn't provoque any user warnings
//		printSimulationSpecs(); // first time around, show simMode etc.
	}

	hasRunNetwork_ = true;

	return snn_->runNetwork(nSec, nMsec, copyState);	
}

// +++++++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// sets update cycle for showing network status
void CARLsim::setLogCycle(int showStatusCycle) {
	snn_->setLogCycle(showStatusCycle);
}

// set new file pointer for debug log file
void CARLsim::setLogDebugFp(FILE* fpLog) {
	UserErrors::assertTrue(fpLog!=NULL,UserErrors::CANNOT_BE_NULL,"setLogDebugFp","fpLog");

	snn_->setLogDebugFp(fpLog);
}

// set new file pointer for all files
void CARLsim::setLogsFp(FILE* fpOut, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
	UserErrors::assertTrue(loggerMode_==CUSTOM,UserErrors::MUST_BE_LOGGER_CUSTOM,"setLogsFp","Logger mode");

	snn_->setLogsFp(fpOut,fpErr,fpDeb,fpLog);
}


// +++++++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// reads network state from file
void CARLsim::readNetwork(FILE* fid) {
	std::string funcName = "readNetwork()";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run

	snn_->readNetwork(fid);
}

void CARLsim::reassignFixedWeights(short int connectId, float weightMatrix[], int matrixSize, int configId) {
	snn_->reassignFixedWeights(connectId,weightMatrix,matrixSize,configId);
}


// resets spike count for particular neuron group
void CARLsim::resetSpikeCntUtil(int grpId) {
	snn_->resetSpikeCntUtil(grpId);
}

// resets spike counters
void CARLsim::resetSpikeCounter(int grpId, int configId) {
	snn_->resetSpikeCounter(grpId,configId);
}

// set network monitor for a group
void CARLsim::setConnectionMonitor(int grpIdPre, int grpIdPost, ConnectionMonitor* connectionMon, int configId) {
	std::string funcName = "setConnectionMonitor(\""+getGroupName(grpIdPre,configId)+"\",ConnectionMonitor*)";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(grpIdPre!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPre");		// groupId can't be ALL
	UserErrors::assertTrue(grpIdPost!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpIdPost");		// groupId can't be ALL

	snn_->setConnectionMonitor(grpIdPre, grpIdPost, new ConnectionMonitorCore(this, connectionMon),configId);
}

// set group monitor for a group
void CARLsim::setGroupMonitor(int grpId, GroupMonitor* groupMon, int configId) {
	std::string funcName = "setGroupMonitor(\""+getGroupName(grpId,configId)+"\",GroupMonitor*)";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	snn_->setGroupMonitor(grpId, new GroupMonitorCore(this, groupMon),configId);
}

// sets a spike counter for a group
void CARLsim::setSpikeCounter(int grpId, int recordDur, int configId) {
	std::stringstream funcName;	funcName << "setSpikeCounter(" << grpId << "," << recordDur << "," << configId << ")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");

	snn_->setSpikeCounter(grpId,recordDur,configId);
}

// sets up a spike generator
void CARLsim::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId) {
	std::string funcName = "setSpikeGenerator(\""+getGroupName(grpId,configId)+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL
	UserErrors::assertTrue(spikeGen!=NULL, UserErrors::CANNOT_BE_NULL, funcName);

	snn_->setSpikeGenerator(grpId, new SpikeGeneratorCore(this, spikeGen),configId);
}

// set spike monitor for a group
void CARLsim::setSpikeMonitor(int grpId, SpikeMonitor* spikeMon, int configId) {
	std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId,configId)+"\",SpikeMonitor*)";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	snn_->setSpikeMonitor(grpId, new SpikeMonitorCore(this, spikeMon),configId);
}


// set spike monitor for group and write spikes to file
void CARLsim::setSpikeMonitor(int grpId, const std::string& fname, int configId) {
	std::string funcName = "setSpikeMonitor(\""+getGroupName(grpId,configId)+"\",\""+fname+"\")";
	UserErrors::assertTrue(!hasRunNetwork_, UserErrors::NETWORK_ALREADY_RUN, funcName); // can't change setup after run
	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "configId");	// configId can't be ALL
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "grpId");		// groupId can't be ALL

	// try to open spike file
	FILE* fid = fopen(fname.c_str(),"wb"); // FIXME: where does fid get closed?
	if (fid==NULL) {
		// file could not be opened

		// default case: print error and exit
		std::string fileError = "Make sure directory exists: "+fname;
		UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, fname, fileError);
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
	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName, "configId");	// configId can't be ALL
	snn_->writePopWeights(fname,gIDpre,gIDpost,configId);
}



// +++++++++ PUBLIC METHODS: SETTERS / GETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// FIXME
// get connection info struct
//grpConnectInfo_t* CARLsim::getConnectInfo(short int connectId, int configId) {
//	std::stringstream funcName;	funcName << "getConnectInfo(" << connectId << "," << configId << ")";
//	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
//	return snn_->getConnectInfo(connectId,configId);
//}

int CARLsim::getConnectionId(short int connectId, int configId) {
	return snn_->getConnectionId(connectId,configId);
}

uint8_t* CARLsim::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	return snn_->getDelays(gIDpre,gIDpost,Npre,Npost,delays);
}

int CARLsim::getGroupId(int grpId, int configId) {
	std::stringstream funcName;	funcName << "getConnectInfo(" << grpId << "," << configId << ")";
	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
	return snn_->getGroupId(grpId,configId);
}
// get group info struct
//group_info_t CARLsim::getGroupInfo(int grpId, int configId) {
//	std::stringstream funcName;	funcName << "getConnectInfo(" << grpId << "," << configId << ")";
//	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");			// configId can't be ALL
//	return snn_->getGroupInfo(grpId, configId);
//}

std::string CARLsim::getGroupName(int grpId, int configId) { return snn_->getGroupName(grpId, configId); }
int CARLsim::getNumConnections(short int connectionId) { return snn_->getNumConnections(connectionId); }
int CARLsim::getNumGroups() { return snn_->getNumGroups(); }
int CARLsim::getNumNeurons() { return snn_->getNumNeurons(); }
int CARLsim::getNumPreSynapses() { return snn_->getNumPreSynapses(); }
int CARLsim::getNumPostSynapses() { return snn_->getNumPostSynapses(); } 

int CARLsim::getGroupStartNeuronId(int grpId) { return snn_->getGroupStartNeuronId(grpId); }
int CARLsim::getGroupEndNeuronId(int grpId) { return snn_->getGroupEndNeuronId(grpId); }
int CARLsim::getGroupNumNeurons(int grpId) { return snn_->getGroupNumNeurons(grpId); }

uint64_t CARLsim::getSimTime() { return snn_->getSimTime(); }
uint32_t CARLsim::getSimTimeSec() { return snn_->getSimTimeSec(); }
uint32_t CARLsim::getSimTimeMsec() { return snn_->getSimTimeMs(); }

// Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
// and the size of the 1D array in size.
void CARLsim::getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId) {
	snn_->getPopWeights(gIDpre,gIDpost,weights,size,configId);
}

unsigned int* CARLsim::getSpikeCntPtr(int grpId) {
	return snn_->getSpikeCntPtr(grpId);
}

// get spiking information out for a given group
int* CARLsim::getSpikeCounter(int grpId, int configId) {
	std::stringstream funcName;	funcName << "getSpikeCounter(" << grpId << "," << configId << ")";
	UserErrors::assertTrue(grpId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "grpId");
	UserErrors::assertTrue(configId!=ALL, UserErrors::ALL_NOT_ALLOWED, funcName.str(), "configId");

	return snn_->getSpikeCounter(grpId,configId);
}

float* CARLsim::getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges) {
	return snn_->getWeightChanges(gIDpre,gIDpost,Npre,Npost,weightChanges);
}

bool CARLsim::isExcitatoryGroup(int grpId) { return snn_->isExcitatoryGroup(grpId); }
bool CARLsim::isInhibitoryGroup(int grpId) { return snn_->isInhibitoryGroup(grpId); }
bool CARLsim::isPoissonGroup(int grpId) { return snn_->isPoissonGroup(grpId); }

// Sets enableGpuSpikeCntPtr to true or false.
void CARLsim::setCopyFiringStateFromGPU(bool enableGPUSpikeCntPtr) {
	snn_->setCopyFiringStateFromGPU(enableGPUSpikeCntPtr);
}



// +++++++++ PUBLIC METHODS: SET DEFAULTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// set default values for conductance decay times
void CARLsim::setDefaultConductanceTimeConstants(int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, 
int tdGABAb) {
	std::stringstream funcName;	funcName << "setDefaultConductanceTimeConstants(" << tdAMPA << "," << trNMDA << 
		"," << tdNMDA << "," << tdGABAa << "," << trGABAb << "," << tdGABAb << ")";
	UserErrors::assertTrue(tdAMPA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdAMPA");
	UserErrors::assertTrue(trNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trNMDA");
	UserErrors::assertTrue(tdNMDA>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdNMDA");
	UserErrors::assertTrue(tdGABAa>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAa");
	UserErrors::assertTrue(trGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "trGABAb");
	UserErrors::assertTrue(tdGABAb>0, UserErrors::MUST_BE_POSITIVE, funcName.str(), "tdGABAb");
	UserErrors::assertTrue(trNMDA!=tdNMDA, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trNMDA and tdNMDA");
	UserErrors::assertTrue(trGABAb!=tdGABAb, UserErrors::CANNOT_BE_IDENTICAL, funcName.str(), "trGABAb and tdGABAb");

	def_tdAMPA_  = tdAMPA;
	def_trNMDA_  = trNMDA;
	def_tdNMDA_  = tdNMDA;
	def_tdGABAa_ = tdGABAa;
	def_trGABAb_ = trGABAb;
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
void CARLsim::setDefaultSTPparams(int neurType, float STP_U, float STP_tau_u, float STP_tau_x) {
	std::string funcName = "setDefaultSTPparams()";
	UserErrors::assertTrue(neurType==EXCITATORY_NEURON || neurType==INHIBITORY_NEURON, UserErrors::WRONG_NEURON_TYPE,
									funcName);
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
