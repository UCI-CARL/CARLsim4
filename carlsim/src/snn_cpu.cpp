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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */

#include <snn.h>
#include <sstream>

#if (WIN32 || WIN64)
	#include <float.h>
	#include <time.h>

	#ifndef isnan
	#define isnan(x) _isnan(x)
	#endif

	#ifndef isinf
	#define isinf(x) (!_finite(x))
	#endif

	#ifndef srand48
	#define srand48(x) srand(x)
	#endif

	#ifndef drand48
	#define drand48() (double(rand())/RAND_MAX)
	#endif
#else
	#include <string.h>
	#define strcmpi(s1,s2) strcasecmp(s1,s2)
#endif

MTRand_closed getRandClosed;
MTRand	      getRand;

RNG_rand48* gpuRand48 = NULL;


// \FIXME what are the following for? why were they all the way at the bottom of this file?

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

#define SETPOST_INFO(name, nid, sid, val) name[cumulativePost[nid]+sid]=val;

#define SETPRE_INFO(name, nid, sid, val)  name[cumulativePre[nid]+sid]=val;



/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///


// TODO: consider moving unsafe computations out of constructor
CpuSNN::CpuSNN(std::string& name, simMode_t simMode, loggerMode_t loggerMode, int ithGPU, int nConfig, int randSeed)
					: networkName_(name), simMode_(simMode), loggerMode_(loggerMode), ithGPU_(ithGPU),
					  nConfig_(nConfig), randSeed_(CpuSNN::setRandSeed(randSeed)) // all of these are const
{
	// move all unsafe operations out of constructor
	CpuSNNinit();
}

// destructor
CpuSNN::~CpuSNN() {
	if (!simulatorDeleted)
		deleteObjects();
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: SETTING UP A SIMULATION
/// ************************************************************************************************************ ///

// make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
short int CpuSNN::connect(int grpId1, int grpId2, const std::string& _type, float initWt, float maxWt, float prob,
						uint8_t minDelay, uint8_t maxDelay, float _mulSynFast, float _mulSynSlow, bool synWtType) {
						//const std::string& wtType
	int retId=-1;
	for(int c=0; c < nConfig_; c++, grpId1++, grpId2++) {
		assert(grpId1 < numGrp);
		assert(grpId2 < numGrp);
		assert(minDelay <= maxDelay);
		assert(!isPoissonGroup(grpId2));

    //* \deprecated Do these ramp thingies still work?
//    bool useRandWts = (wtType.find("random") != std::string::npos);
//    bool useRampDownWts = (wtType.find("ramp-down") != std::string::npos);
//    bool useRampUpWts = (wtType.find("ramp-up") != std::string::npos);
//    uint32_t connProp = SET_INITWTS_RANDOM(useRandWts)
//      | SET_CONN_PRESENT(1)
//      | SET_FIXED_PLASTIC(synWtType)
//      | SET_INITWTS_RAMPUP(useRampUpWts)
//      | SET_INITWTS_RAMPDOWN(useRampDownWts);
		uint32_t connProp = SET_CONN_PRESENT(1) | SET_FIXED_PLASTIC(synWtType);

		grpConnectInfo_t* newInfo 	= (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));
		newInfo->grpSrc   			= grpId1;
		newInfo->grpDest  			= grpId2;
		newInfo->initWt	  			= initWt;
		newInfo->maxWt	  			= maxWt;
		newInfo->maxDelay 			= maxDelay;
		newInfo->minDelay 			= minDelay;
		newInfo->mulSynFast			= _mulSynFast;
		newInfo->mulSynSlow			= _mulSynSlow;
		newInfo->connProp 			= connProp;
		newInfo->p 					= prob;
		newInfo->type	  			= CONN_UNKNOWN;
		newInfo->numPostSynapses 	= 1;

		newInfo->next 				= connectBegin; //linked list of connection..
		connectBegin 				= newInfo;

		if ( _type.find("random") != std::string::npos) {
			newInfo->type 	= CONN_RANDOM;
			newInfo->numPostSynapses	= MIN(grp_Info[grpId2].SizeN,((int) (prob*grp_Info[grpId2].SizeN +5*sqrt(prob*(1-prob)*grp_Info[grpId2].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 5 stds.
			newInfo->numPreSynapses   = MIN(grp_Info[grpId1].SizeN,((int) (prob*grp_Info[grpId1].SizeN +5*sqrt(prob*(1-prob)*grp_Info[grpId1].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 5 stds.
		}
		//so you're setting the size to be prob*Number of synapses in group info + some standard deviation ...
		else if ( _type.find("full-no-direct") != std::string::npos) {
			newInfo->type 	= CONN_FULL_NO_DIRECT;
			newInfo->numPostSynapses	= grp_Info[grpId2].SizeN-1;
			newInfo->numPreSynapses	= grp_Info[grpId1].SizeN-1;
		}
		else if ( _type.find("full") != std::string::npos) {
			newInfo->type 	= CONN_FULL;
			newInfo->numPostSynapses	= grp_Info[grpId2].SizeN;
			newInfo->numPreSynapses   = grp_Info[grpId1].SizeN;
		}
		else if ( _type.find("one-to-one") != std::string::npos) {
			newInfo->type 	= CONN_ONE_TO_ONE;
			newInfo->numPostSynapses	= 1;
			newInfo->numPreSynapses	= 1;
		}
		else {
			CARLSIM_ERROR("Invalid connection type (should be 'random', 'full', 'one-to-one', or 'full-no-direct')");
			exitSimulation(-1);
		}

		if (newInfo->numPostSynapses > MAX_nPostSynapses) {
			CARLSIM_ERROR("Connection exceeded the maximum number of output synapses (%d), has %d.",
						MAX_nPostSynapses,newInfo->numPostSynapses);
			assert(newInfo->numPostSynapses <= MAX_nPostSynapses);
		}

		if (newInfo->numPreSynapses > MAX_nPreSynapses) {
			CARLSIM_ERROR("Connection exceeded the maximum number of input synapses (%d), has %d.",
						MAX_nPreSynapses,newInfo->numPreSynapses);
			assert(newInfo->numPreSynapses <= MAX_nPreSynapses);
		}

		// update the pre and post size...
		// Subtlety: each group has numPost/PreSynapses from multiple connections.
		// The newInfo->numPost/PreSynapses are just for this specific connection.
		// We are adding the synapses counted in this specific connection to the totals for both groups.
		grp_Info[grpId1].numPostSynapses 	+= newInfo->numPostSynapses;
		grp_Info[grpId2].numPreSynapses 	+= newInfo->numPreSynapses;

		CARLSIM_DEBUG("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d",
						grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,
						grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

		newInfo->connId	= numConnections++;
		assert(numConnections <= MAX_nConnections);	// make sure we don't overflow connId

		if(c==0)
			retId = newInfo->connId;

		CARLSIM_DEBUG("CONNECT SETUP: connId=%d, mulFast=%f, mulSlow=%f",newInfo->connId,newInfo->mulSynFast,
							newInfo->mulSynSlow);
	}
	assert(retId != -1);
	return retId;
}

// make custom connections from grpId1 to grpId2
short int CpuSNN::connect(int grpId1, int grpId2, ConnectionGeneratorCore* conn, float _mulSynFast, float _mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	int retId=-1;

	for(int c=0; c < nConfig_; c++, grpId1++, grpId2++) {
		assert(grpId1 < numGrp);
		assert(grpId2 < numGrp);

		if (maxM == 0)
			maxM = grp_Info[grpId2].SizeN;

		if (maxPreM == 0)
			maxPreM = grp_Info[grpId1].SizeN;

		if (maxM > MAX_nPostSynapses) {
			CARLSIM_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of output synapses (%d), "
								"has %d.", grp_Info2[grpId1].Name.c_str(),grpId1,grp_Info2[grpId2].Name.c_str(),
								grpId2,	MAX_nPostSynapses,maxM);
			assert(maxM <= MAX_nPostSynapses);
		}

		if (maxPreM > MAX_nPreSynapses) {
			CARLSIM_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of input synapses (%d), "
								"has %d.\n", grp_Info2[grpId1].Name.c_str(), grpId1,grp_Info2[grpId2].Name.c_str(),
								grpId2, MAX_nPreSynapses,maxPreM);
			assert(maxPreM <= MAX_nPreSynapses);
		}

		grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));

		newInfo->grpSrc   = grpId1;
		newInfo->grpDest  = grpId2;
		newInfo->initWt	  = 1;
		newInfo->maxWt	  = 1;
		newInfo->maxDelay = 1;
		newInfo->minDelay = 1;
		newInfo->mulSynFast = _mulSynFast;
		newInfo->mulSynSlow = _mulSynSlow;
		newInfo->connProp = SET_CONN_PRESENT(1) | SET_FIXED_PLASTIC(synWtType);
		newInfo->type	  = CONN_USER_DEFINED;
		newInfo->numPostSynapses	  	  = maxM;
		newInfo->numPreSynapses	  = maxPreM;
		newInfo->conn	= conn;

		newInfo->next	= connectBegin;  // build a linked list
		connectBegin      = newInfo;

		// update the pre and post size...
		grp_Info[grpId1].numPostSynapses    += newInfo->numPostSynapses;
		grp_Info[grpId2].numPreSynapses += newInfo->numPreSynapses;

		CARLSIM_DEBUG("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d",
						grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,
						grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

		newInfo->connId	= numConnections++;
		assert(numConnections <= MAX_nConnections);	// make sure we don't overflow connId

		if(c==0)
			retId = newInfo->connId;
	}
	assert(retId != -1);
	return retId;
}


// create group of Izhikevich neurons
// use int for nNeur to avoid arithmetic underflow
int CpuSNN::createGroup(const std::string& grpName, Grid3D& grid, int neurType, int configId) {
	assert(grid.x*grid.y*grid.z>0);
	assert(neurType>=0); assert(configId>=-1);	assert(configId<nConfig_);
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			createGroup(grpName, grid, neurType, c);
		return (numGrp-nConfig_);
	} else {
		assert(numGrp < MAX_GRP_PER_SNN);

		if ( (!(neurType&TARGET_AMPA) && !(neurType&TARGET_NMDA) &&
			  !(neurType&TARGET_GABAa) && !(neurType&TARGET_GABAb)) || (neurType&POISSON_NEURON)) {
			CARLSIM_ERROR("Invalid type using createGroup... Cannot create poisson generators here.");
			exitSimulation(1);
		}

		// We don't store the Grid3D struct in grp_Info so we don't have to deal with allocating structs on the GPU
		grp_Info[numGrp].SizeN  			= grid.x * grid.y * grid.z; // number of neurons in the group
        grp_Info[numGrp].SizeX              = grid.x; // number of neurons in first dim of Grid3D
        grp_Info[numGrp].SizeY              = grid.y; // number of neurons in second dim of Grid3D
        grp_Info[numGrp].SizeZ              = grid.z; // number of neurons in third dim of Grid3D

		grp_Info[numGrp].Type   			= neurType;
		grp_Info[numGrp].WithSTP			= false;
		grp_Info[numGrp].WithSTDP			= false;
		grp_Info[numGrp].WithSTDPtype       = UNKNOWN_STDP;
		grp_Info[numGrp].WithHomeostasis	= false;

		if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb)) {
			grp_Info[numGrp].MaxFiringRate 	= INHIBITORY_NEURON_MAX_FIRING_RATE;
		} else {
			grp_Info[numGrp].MaxFiringRate 	= EXCITATORY_NEURON_MAX_FIRING_RATE;
		}

		grp_Info2[numGrp].ConfigId			= configId;
		grp_Info2[numGrp].Name  			= grpName;
		grp_Info[numGrp].isSpikeGenerator	= false;
		grp_Info[numGrp].MaxDelay			= 1;

		grp_Info2[numGrp].Izh_a 			= -1; // \FIXME ???

		std::stringstream outStr;
		outStr << configId;
		grp_Info2[numGrp].Name 				= (configId==0)?grpName:grpName+"_"+outStr.str();
		finishedPoissonGroup				= true;

		numGrp++;
		return (numGrp-1);
	}
}

// create spike generator group
// use int for nNeur to avoid arithmetic underflow
int CpuSNN::createSpikeGeneratorGroup(const std::string& grpName, Grid3D& grid, int neurType, int configId) {
		assert(grid.x*grid.y*grid.z>0);
		assert(neurType>=0); assert(configId>=-1);	assert(configId<nConfig_);
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			createSpikeGeneratorGroup(grpName, grid, neurType, c);
		return (numGrp-nConfig_);
	} else {
		grp_Info[numGrp].SizeN   		= grid.x * grid.y * grid.z; // number of neurons in the group
        grp_Info[numGrp].SizeX          = grid.x; // number of neurons in first dim of Grid3D
        grp_Info[numGrp].SizeY          = grid.y; // number of neurons in second dim of Grid3D
        grp_Info[numGrp].SizeZ          = grid.z; // number of neurons in third dim of Grid3D
		grp_Info[numGrp].Type    		= neurType | POISSON_NEURON;
		grp_Info[numGrp].WithSTP		= false;
		grp_Info[numGrp].WithSTDP		= false;
		grp_Info[numGrp].WithSTDPtype   = UNKNOWN_STDP;
		grp_Info[numGrp].WithHomeostasis	= false;
		grp_Info[numGrp].isSpikeGenerator	= true;		// these belong to the spike generator class...
		grp_Info2[numGrp].ConfigId		= configId;
		grp_Info2[numGrp].Name    		= grpName;
		grp_Info[numGrp].MaxFiringRate 	= POISSON_MAX_FIRING_RATE;

		std::stringstream outStr;
		outStr << configId;
		if (configId != 0)
			grp_Info2[numGrp].Name = grpName + "_" + outStr.str();

		numGrp++;
		numSpikeGenGrps++;

		return (numGrp-1);
	}
}

// set conductance values for a simulation (custom values or disable conductances alltogether)
void CpuSNN::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa,
int trGABAb, int tdGABAb, int configId) {
	if (configId!=ALL) {
		CARLSIM_ERROR("Using setConductances with configId!=ALL is deprecated"); // \deprecated
		assert(configId==ALL);
	}

	if (isSet) {
		assert(tdAMPA>0); assert(tdNMDA>0); assert(tdGABAa>0); assert(tdGABAb>0);
		assert(trNMDA>=0); assert(trGABAb>=0); // 0 to disable rise times
		assert(trNMDA!=tdNMDA); assert(trGABAb!=tdGABAb); // singularity
	}

	// we do not care about configId anymore
	// set conductances globally for all connections
	sim_with_conductances  |= isSet;
	dAMPA  = 1.0-1.0/tdAMPA;
	dNMDA  = 1.0-1.0/tdNMDA;
	dGABAa = 1.0-1.0/tdGABAa;
	dGABAb = 1.0-1.0/tdGABAb;

	if (trNMDA>0) {
		// use rise time for NMDA
		sim_with_NMDA_rise = true;
		rNMDA = 1.0-1.0/trNMDA;

		// compute max conductance under this model to scale it back to 1
		// otherwise the peak conductance will not be equal to the weight
		double tmax = (-tdNMDA*trNMDA*log(1.0*trNMDA/tdNMDA))/(tdNMDA-trNMDA); // t at which cond will be max
		sNMDA = 1.0/(exp(-tmax/tdNMDA)-exp(-tmax/trNMDA)); // scaling factor, 1 over max amplitude
		assert(!isinf(tmax) && !isnan(tmax) && tmax>=0);
		assert(!isinf(sNMDA) && !isnan(sNMDA) && sNMDA>0);
	}

	if (trGABAb>0) {
		// use rise time for GABAb
		sim_with_GABAb_rise = true;
		rGABAb = 1.0-1.0/trGABAb;

		// compute max conductance under this model to scale it back to 1
		// otherwise the peak conductance will not be equal to the weight
		double tmax = (-tdGABAb*trGABAb*log(1.0*trGABAb/tdGABAb))/(tdGABAb-trGABAb); // t at which cond will be max
		sGABAb = 1.0/(exp(-tmax/tdGABAb)-exp(-tmax/trGABAb)); // scaling factor, 1 over max amplitude
		assert(!isinf(tmax) && !isnan(tmax)); assert(!isinf(sGABAb) && !isnan(sGABAb) && sGABAb>0);
	}
//		grp_Info[cGrpId].newUpdates 		= true; // \deprecated

	if (sim_with_conductances) {
		CARLSIM_INFO("Running COBA mode:");
		CARLSIM_INFO("  - AMPA decay time            = %5d ms", tdAMPA);
		CARLSIM_INFO("  - NMDA rise time %s  = %5d ms", sim_with_NMDA_rise?"          ":"(disabled)", trNMDA);
		CARLSIM_INFO("  - GABAa decay time           = %5d ms", tdGABAa);
		CARLSIM_INFO("  - GABAb rise time %s = %5d ms", sim_with_GABAb_rise?"          ":"(disabled)",trGABAb);
		CARLSIM_INFO("  - GABAb decay time           = %5d ms", tdGABAb);
	} else {
		CARLSIM_INFO("Running CUBA mode (all synaptic conductances disabled)");
	}
}

// set homeostasis for group
void CpuSNN::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setHomeostasis(g, isSet, homeoScale, avgTimeScale, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setHomeostasis(g, isSet, homeoScale, avgTimeScale, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setHomeostasis(grpId, isSet, homeoScale, avgTimeScale, c);
	} else {
		// set conductances for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_homeostasis 			   |= isSet;
		grp_Info[cGrpId].WithHomeostasis    = isSet;
		grp_Info[cGrpId].homeostasisScale   = homeoScale;
		grp_Info[cGrpId].avgTimeScale       = avgTimeScale;
		grp_Info[cGrpId].avgTimeScaleInv    = 1.0f/avgTimeScale;
		grp_Info[cGrpId].avgTimeScale_decay = (avgTimeScale*1000.0f-1.0f)/(avgTimeScale*1000.0f);
		grp_Info[cGrpId].newUpdates 		= true; // \FIXME: what's this?

		CARLSIM_INFO("Homeostasis parameters %s for %d (%s):\thomeoScale: %f, avgTimeScale: %f",
					isSet?"enabled":"disabled",cGrpId,grp_Info2[cGrpId].Name.c_str(),homeoScale,avgTimeScale);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void CpuSNN::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setHomeoBaseFiringRate(g, baseFiring, baseFiringSD, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setHomeoBaseFiringRate(g, baseFiring, baseFiringSD, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD, c);
	} else {
		// set conductances for a given group and configId
		int cGrpId 						= getGroupId(grpId, configId);
		assert(grp_Info[cGrpId].WithHomeostasis);

		grp_Info2[cGrpId].baseFiring 	= baseFiring;
		grp_Info2[cGrpId].baseFiringSD 	= baseFiringSD;
		grp_Info[cGrpId].newUpdates 	= true; //TODO: I have to see how this is handled.  -- KDC

		CARLSIM_INFO("Homeostatic base firing rate set for %d (%s):\tbaseFiring: %3.3f, baseFiringStd: %3.3f",
							cGrpId,grp_Info2[cGrpId].Name.c_str(),baseFiring,baseFiringSD);
	}
}


// set Izhikevich parameters for group
void CpuSNN::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
								float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId)
{
	assert(grpId>=-1); assert(izh_a>0); assert(izh_a_sd>=0); assert(izh_b>0); assert(izh_b_sd>=0); assert(izh_c_sd>=0);
	assert(izh_d>0); assert(izh_d_sd>=0); assert(configId>=-1);

	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setNeuronParameters(g, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setNeuronParameters(g, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, c);
	} else {
		int cGrpId = getGroupId(grpId, configId);
		grp_Info2[cGrpId].Izh_a	  	=   izh_a;
		grp_Info2[cGrpId].Izh_a_sd  =   izh_a_sd;
		grp_Info2[cGrpId].Izh_b	  	=   izh_b;
		grp_Info2[cGrpId].Izh_b_sd  =   izh_b_sd;
		grp_Info2[cGrpId].Izh_c		=   izh_c;
		grp_Info2[cGrpId].Izh_c_sd	=   izh_c_sd;
		grp_Info2[cGrpId].Izh_d		=   izh_d;
		grp_Info2[cGrpId].Izh_d_sd	=   izh_d_sd;
	}
}

void CpuSNN::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh, float tauACh, float baseNE, float tauNE, int configId) {
	if (configId == ALL) {
		for (int c = 0; c < nConfig_; c++)
			setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE, c);
	} else {
		int cGrpId = getGroupId(grpId, configId);
		grp_Info[cGrpId].baseDP	= baseDP;
		grp_Info[cGrpId].decayDP = 1.0 - (1.0 / tauDP);
		grp_Info[cGrpId].base5HT = base5HT;
		grp_Info[cGrpId].decay5HT = 1.0 - (1.0 / tau5HT);
		grp_Info[cGrpId].baseACh = baseACh;
		grp_Info[cGrpId].decayACh = 1.0 - (1.0 / tauACh);
		grp_Info[cGrpId].baseNE	= baseNE;
		grp_Info[cGrpId].decayNE = 1.0 - (1.0 / tauNE);
	}
}

// set STDP params
void CpuSNN::setSTDP(int grpId, bool isSet, stdpType_t type, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD,
	int configId) {
	assert(grpId>=-1); assert(configId>=-1);
	if (isSet) {
		assert(type!=UNKNOWN_STDP);
		assert(alphaLTP>=0); assert(tauLTP>=0); assert(alphaLTD>=0); assert(tauLTD>=0);
	}

	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setSTDP(g, isSet, type, alphaLTP, tauLTP, alphaLTD, tauLTD, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setSTDP(g, isSet, type, alphaLTP, tauLTP, alphaLTD, tauLTD, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setSTDP(grpId, isSet, type, alphaLTP, tauLTP, alphaLTD, tauLTD, c);
	} else {
		// set STDP for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_stdp 				   |= isSet;
		grp_Info[cGrpId].WithSTDP 		= isSet;
		grp_Info[cGrpId].WithSTDPtype	= type;
		grp_Info[cGrpId].ALPHA_LTP 		= alphaLTP;
		grp_Info[cGrpId].ALPHA_LTD 		= alphaLTD;
		grp_Info[cGrpId].TAU_LTP_INV 	= 1.0f/tauLTP;
		grp_Info[cGrpId].TAU_LTD_INV	= 1.0f/tauLTD;
		grp_Info[cGrpId].newUpdates 	= true; // \FIXME whatsathiis?

		CARLSIM_INFO("STDP %s for %s(%d)", isSet?"enabled":"disabled", grp_Info2[cGrpId].Name.c_str(), cGrpId);
	}
}


// set STP params
void CpuSNN::setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x, int configId) {
	assert(grpId>=-1); assert(configId>=-1);
	if (isSet) {
		assert(STP_U>0 && STP_U<=1); assert(STP_tau_u>0); assert(STP_tau_x>0);
	}

	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setSTP(g, isSet, STP_U, STP_tau_u, STP_tau_x, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setSTP(g, isSet, STP_U, STP_tau_u, STP_tau_x, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setSTP(grpId, isSet, STP_U, STP_tau_u, STP_tau_x, c);
	} else {
		// set STDP for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_stp 				   |= isSet;
		grp_Info[cGrpId].WithSTP 		= isSet;
		grp_Info[cGrpId].STP_A 			= (STP_U>0.0f) ? 1.0/STP_U : 1.0f; // scaling factor
		grp_Info[cGrpId].STP_U 			= STP_U;
		grp_Info[cGrpId].STP_tau_u_inv	= 1.0f/STP_tau_u; // facilitatory
		grp_Info[cGrpId].STP_tau_x_inv	= 1.0f/STP_tau_x; // depressive
		grp_Info[cGrpId].newUpdates = true;

		CARLSIM_INFO("STP %s for %d (%s):\tA: %1.4f, U: %1.4f, tau_u: %4.0f, tau_x: %4.0f", isSet?"enabled":"disabled",
					cGrpId, grp_Info2[cGrpId].Name.c_str(), grp_Info[cGrpId].STP_A, STP_U, STP_tau_u, STP_tau_x);
	}
}

void CpuSNN::setWeightAndWeightChangeUpdate(updateInterval_t wtUpdateInterval, updateInterval_t wtChangeUpdateInterval,
											int tauWeightChange) {
	switch (wtUpdateInterval) {
		case INTERVAL_10MS:
			wtUpdateInterval_ = 10;
			stdpScaleFactor_ = 0.005f;
			break;
		case INTERVAL_100MS:
			wtUpdateInterval_ = 100;
			stdpScaleFactor_ = 0.05f;
			break;
		case INTERVAL_1000MS:
		default:
			wtUpdateInterval_ = 1000;
			stdpScaleFactor_ = 0.5f;
			break;
	}

	switch (wtChangeUpdateInterval) {
	case INTERVAL_10MS:
		wtChangeUpdateInterval_ = 10;
		break;
	case INTERVAL_100MS:
		wtChangeUpdateInterval_ = 100;
		break;
	case INTERVAL_1000MS:
	default:
		wtChangeUpdateInterval_ = 1000;
		break;
	}

	wtChangeDecay_ = 1.0 - (1.0 / tauWeightChange);

	CARLSIM_INFO("Update weight every %d ms, stdpScaleFactor = %1.3f", wtUpdateInterval_, stdpScaleFactor_);
	CARLSIM_INFO("Update weight change every %d ms, wtChangeDecay = %1.3f", wtChangeUpdateInterval_, wtChangeDecay_);
}


/// ************************************************************************************************************ ///
/// PUBLIC METHODS: RUNNING A SIMULATION
/// ************************************************************************************************************ ///

// if 
int CpuSNN::runNetwork(int _nsec, int _nmsec, bool printRunSummary, bool copyState) {
	assert(_nmsec >= 0 && _nmsec < 1000);
	assert(_nsec  >= 0);
	int runDuration = _nsec*1000 + _nmsec;

	// setupNetwork() must have already been called
	assert(doneReorganization);

	// first-time run: inform the user the simulation is running now
	if (simTime==0) {
		CARLSIM_INFO("");
		CARLSIM_INFO("*******************      Running %s Simulation      ****************************\n",
			simMode_==GPU_MODE?"GPU":"CPU");
	}

	// reset all spike counters
	if (simMode_==GPU_MODE)
		resetSpikeCnt_GPU(0,numGrp);
	else
		resetSpikeCnt(ALL);

	// store current start time for future reference
	simTimeRunStart = simTime;
	simTimeRunStop  = simTime+runDuration;
	assert(simTimeRunStop>=simTimeRunStart); // check for arithmetic underflow

	// set the Poisson generation time slice to be at the run duration up to PROPOGATED_BUFFER_SIZE ms.
	// \TODO: should it be PROPAGATED_BUFFER_SIZE-1 or PROPAGATED_BUFFER_SIZE ? 
	setGrpTimeSlice(ALL, MAX(1,MIN(runDuration,PROPAGATED_BUFFER_SIZE-1)));

	CUDA_RESET_TIMER(timer);
	CUDA_START_TIMER(timer);

	// if nsec=0, simTimeMs=10, we need to run the simulator for 10 timeStep;
	// if nsec=1, simTimeMs=10, we need to run the simulator for 1*1000+10, time Step;
	for(int i=0; i<runDuration; i++) {
		if(simMode_ == CPU_MODE)
			doSnnSim();
		else
			doGPUSim();

		// update weight every updateInterval ms if plastic synapses present
		if (!sim_with_fixedwts && (wtUpdateInterval_==++wtUpdateIntervalCnt_)) {
			wtUpdateIntervalCnt_ = 0; // reset counter

			if (simMode_ == CPU_MODE) {
				updateWeights();
			} else{
				updateWeights_GPU();

				if (copyState) {
					// TODO: build DA buffer in GPU memory so that we can retrieve data every one second instead of 10ms
					// Log dopamine concentration
					copyGroupState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, 0);
					for (int i = 0; i < numGrp; i++) {
						int monitorId = grp_Info[i].GroupMonitorId;
						if (monitorId != -1)
							grpDABuffer[monitorId][simTimeMs / wtUpdateInterval_] = cpuNetPtrs.grpDA[i];
					}
				}
			}
		}

		if (updateTime()) {
			// finished one sec of simulation...
			if (numSpikeMonitor) {
				updateSpikeMonitor();
			}
			if (numGroupMonitor) {
				updateGroupMonitor();
			}
			if (numConnectionMonitor) {
				updateConnectionMonitor();
			}

			if(simMode_ == CPU_MODE)
				updateFiringTable();
			else
				updateFiringTable_GPU();
		}

		// \deprecated remove this
		if(enableGPUSpikeCntPtr==true && simMode_ == CPU_MODE){
			CARLSIM_ERROR("Error: the enableGPUSpikeCntPtr flag cannot be set in CPU_MODE");
			assert(simMode_==GPU_MODE);
		}
		if(enableGPUSpikeCntPtr==true && simMode_ == GPU_MODE){
			copyFiringStateFromGPU();
		}
	}

	// in GPU mode, copy info from device to host
	if (simMode_==GPU_MODE) {
		if(copyState) {
			copyNeuronState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, ALL);

			if (sim_with_stp) {
				copySTPState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false);
			}
		}
	}

	// user can opt to display some runNetwork summary
	if (printRunSummary) {
		showStatus();

		if (numSpikeMonitor) {
			// if there are SpikeMonitors available and it's time to show the log, print basic spike stats
			// for each group with SpikeMon on
			for (int grpId=0; grpId<numGrp; grpId++) {
				int monitorId = grp_Info[grpId].SpikeMonitorId;
				if (monitorId==-1)
					continue;

				// in GPU mode, need to get data from device first
				if (simMode_==GPU_MODE)
					copyFiringStateFromGPU(grpId);

				// \TODO nSpikeCnt should really be a member of the SpikeMonitor object that gets populated if
				// printRunSummary is true or mode==COUNT.....
				// so then we can use spkMonObj->print(false); // showSpikeTimes==false
				int grpSpk = 0;
				for (int neurId=grp_Info[grpId].StartN; neurId<=grp_Info[grpId].EndN; neurId++)
					grpSpk += nSpikeCnt[neurId]; // add up all neuronal spike counts

				float meanRate = grpSpk*1000.0/runDuration/grp_Info[grpId].SizeN;
				float std = 0.0f;
				if (grp_Info[grpId].SizeN > 1) {
					for (int neurId=grp_Info[grpId].StartN; neurId<=grp_Info[grpId].EndN; neurId++)
						std += (nSpikeCnt[neurId]-meanRate)*(nSpikeCnt[neurId]-meanRate);

					std = sqrt(std/(grp_Info[grpId].SizeN-1.0));
				}


				CARLSIM_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %dms (%.2f +/- %.2f Hz)",
					(float)(simTime/1000.0),
					grp_Info2[grpId].Name.c_str(),
					grpId,
					grpSpk,
					runDuration,
					meanRate,
					std);
			}
		}
	}

	// call updateSpikeMonitor again to fetch all the missing spikes
	updateSpikeMonitor();

	// keep track of simulation time...
	CUDA_STOP_TIMER(timer);
	lastExecutionTime = CUDA_GET_TIMER_VALUE(timer);
	cumExecutionTime += lastExecutionTime;
	return 0;
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: INTERACTING WITH A SIMULATION
/// ************************************************************************************************************ ///

// deallocates dynamical structures and exits
void CpuSNN::exitSimulation(int val) {
	deleteObjects();
	exit(val);
}

// reads network state from file
void CpuSNN::readNetwork(FILE* fid) {
	readNetworkFID = fid;
}

// reassigns weights from the input weightMatrix to the weights between two
// specified neuron groups.
// TODO: figure out scope; is this a user function?
void CpuSNN::reassignFixedWeights(short int connectId, float weightMatrix[], int sizeMatrix, int configId) {
	// handle the config == ALL recursive call contigency.
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			reassignFixedWeights(connectId, weightMatrix, sizeMatrix, c);
	} else {
		int j;
		//first find the correct connection
		grpConnectInfo_t* connInfo; //connInfo = current connection information.
		connInfo = getConnectInfo(connectId,configId);
		//make sure that it is for fixed connections.
		bool synWtType = GET_FIXED_PLASTIC(connInfo->connProp);
		if(synWtType == SYN_PLASTIC){
			CARLSIM_ERROR("The synapses in this connection must be SYN_FIXED in order to use this function.");
			exitSimulation(1);
		}
		//make sure that the user passes the correctly sized matrix
		if(connInfo->numberOfConnections != sizeMatrix){
			CARLSIM_ERROR("The size of the input weight matrix and the number of synaptic connections in this "
							"connection do not match.");
			exitSimulation(1);
		}
		//We have to iterate over all the presynaptic connections of each postsynaptic neuron
		//and see if they are part of our srcGrp.  If they are,
		int destGrp = connInfo->grpDest;
		int srcGrp  = connInfo->grpSrc;
		//iterate over all neurons in the destination group.
		for(int postId=grp_Info[destGrp].StartN; postId <= grp_Info[destGrp].EndN; postId++) {
			int offset            = cumulativePre[postId];
			float* synWtPtr       = &wt[cumulativePre[postId]];
			post_info_t *preIdPtr = &preSynapticIds[offset];
			//iterate over all presynaptic connections in current postsynaptic neuron.
			for (j=0; j < Npre[postId]; j++,preIdPtr++, synWtPtr++) {
				int preId       = GET_CONN_NEURON_ID((*preIdPtr));
				assert(preId < numN);
				short int currentSrcId = grpIds[preId];
				//if the neuron is part of the source group, assign it a value
				//from the reassignment matrix.
				if(currentSrcId == srcGrp){
					//assign wt to reassignment matrix value
					*synWtPtr = (*weightMatrix);
					//iterate reassignment matrix
					weightMatrix++;
				}
			}
		}
	}
	//after all configurations and weights have been set, copy them back to the GPU if
	//necessary:
	if(simMode_ == GPU_MODE)
		copyUpdateVariables_GPU();
}

//! \deprecated right?
//! but we do need resetSpikeCnt and resetSpikeCnt_GPU
void CpuSNN::resetSpikeCntUtil(int my_grpId ) {
  int startGrp, endGrp;

  if(!doneReorganization)
    return;

  if(simMode_ == GPU_MODE){
    //call analogous function, return, else do CPU stuff
    if (my_grpId == ALL) {
      startGrp = 0;
      endGrp   = numGrp;
    }
    else {
      startGrp = my_grpId;
      endGrp   = my_grpId+nConfig_;
    }
    resetSpikeCnt_GPU(startGrp, endGrp);
    return;
  }

  if (my_grpId == -1) {
    startGrp = 0;
    endGrp   = numGrp;
  }
  else {
    startGrp = my_grpId;
    endGrp   = my_grpId+nConfig_;
  }

  resetSpikeCnt(ALL);
}

// reset spike counter to zero
void CpuSNN::resetSpikeCounter(int grpId, int configId) {
	if (!sim_with_spikecounters)
		return;

	assert(grpId>=-1); assert(grpId<numGrp); assert(configId>=-1); assert(configId<nConfig_);

	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g<numGrp; g++)
			resetSpikeCounter(g,0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			resetSpikeCounter(g,configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			resetSpikeCounter(grpId,c);
	} else {
		int cGrpId = getGroupId(grpId,configId);

		// only update if SpikeMonRT is set for this group
		if (!grp_Info[cGrpId].withSpikeCounter)
			return;

		grp_Info[cGrpId].spkCntRecordDurHelper = 0;

		if (simMode_==GPU_MODE) {
			// grpId and configId can no longer be ALL
			resetSpikeCounter_GPU(grpId,configId);
		}
		else {
			int bufPos = grp_Info[cGrpId].spkCntBufPos; // retrieve buf pos
			memset(spkCntBuf[bufPos],0,grp_Info[cGrpId].SizeN*sizeof(int)); // set all to 0
		}
	}
}

void CpuSNN::setGroupMonitor(int grpId, GroupMonitorCore* groupMon, int configId) {
	if (configId == ALL) {
		for(int c = 0; c < nConfig_; c++)
			setGroupMonitor(grpId, groupMon, c);
	} else {
		int cGrpId = getGroupId(grpId, configId);

		// store the grpId for further reference
		groupMonitorGrpId[numGroupMonitor] = cGrpId;

		// also inform the grp that it is being monitored...
		grp_Info[cGrpId].GroupMonitorId = numGroupMonitor;

		grpBufferCallback[numGroupMonitor] = groupMon;

		// create the new buffer for keeping track of group status in the system
		grpDABuffer[numGroupMonitor] = new float[100]; // maximum resolution 10 ms
		grp5HTBuffer[numGroupMonitor] = new float[100]; // maximum resolution 10 ms
		grpAChBuffer[numGroupMonitor] = new float[100]; // maximum resolution 10 ms
		grpNEBuffer[numGroupMonitor] = new float[100]; // maximum resolution 10 ms

		memset(grpDABuffer[numGroupMonitor], 0, sizeof(float) * 100);

		numGroupMonitor++;

		// Finally update the size info that will be useful to see
		// how much memory are we eating...
		// \FIXME: when running on GPU mode??
		cpuSnnSz.monitorInfoSize += sizeof(float) * 100 * 4;
	}
}

void CpuSNN::setConnectionMonitor(int grpIdPre, int grpIdPost, ConnectionMonitorCore* connectionMon, int configId) {
	if (configId == ALL) {
		for(int c = 0; c < nConfig_; c++)
			setConnectionMonitor(grpIdPre, grpIdPost, connectionMon, c);
	} else {
		int cGrpIdPre = getGroupId(grpIdPre, configId);
		int cGrpIdPost = getGroupId(grpIdPost, configId);

		// store the grpId for further reference
		connMonGrpIdPre[numConnectionMonitor] = cGrpIdPre;
		connMonGrpIdPost[numConnectionMonitor] = cGrpIdPost;

		// also inform the grp that it is being monitored...
		grp_Info[cGrpIdPre].ConnectionMonitorId = numConnectionMonitor;

		connBufferCallback[numConnectionMonitor] = connectionMon; // Default value of _netMon is NULL

		numConnectionMonitor++;
	}
}

// sets up a spike generator
void CpuSNN::setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGen, int configId) {
	assert(!doneReorganization); // must be called before setupNetwork to work on GPU
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			setSpikeGenerator(grpId, spikeGen,c);
	} else {
		int cGrpId = getGroupId(grpId, configId);

		assert(spikeGen);
		assert (grp_Info[cGrpId].isSpikeGenerator);
		grp_Info[cGrpId].spikeGen = spikeGen;
	}
}

// A Spike Counter keeps track of the number of spikes per neuron in a group.
void CpuSNN::setSpikeCounter(int grpId, int recordDur, int configId) {
	assert(grpId>=0); assert(grpId<numGrp); assert(configId>=-1); assert(configId<nConfig_);

	// the following does currently not make sense because SpikeGenerators are not supported
	/*
	assert(grpId>=-1); assert(grpId<numGrp); assert(configId>=-1); assert(configId<nConfig_);
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setSpikeCounter(g, recordDur, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += nConfig_) {
			int g = getGroupId(grpId1, configId);
			setSpikeCounter(g, recordDur, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setSpikeCounter(grpId, recordDur, c);
	} else {
		*/

	if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < nConfig_; c++)
			setSpikeCounter(grpId, recordDur, c);
	} else {

		int cGrpId = getGroupId(grpId, configId);

		// TODO: implement same for spike generators on GPU side (see CpuSNN::generateSpikes)
		if (grp_Info[cGrpId].isSpikeGenerator) {
			CARLSIM_ERROR("ERROR: Spike Counters for Spike Generators are currently not supported.");
			exit(1);
			return;
		}

		sim_with_spikecounters = true; // inform simulation
		grp_Info[cGrpId].withSpikeCounter = true; // inform the group
		grp_Info[cGrpId].spkCntRecordDur = (recordDur>0)?recordDur:-1; // set record duration, after which spike buf will be reset
		grp_Info[cGrpId].spkCntRecordDurHelper = 0; // counter to help make fast modulo
		grp_Info[cGrpId].spkCntBufPos = numSpkCnt; // inform group which pos it has in spike buf
		spkCntBuf[numSpkCnt] = new int[grp_Info[cGrpId].SizeN]; // create spike buf
		memset(spkCntBuf[numSpkCnt],0,(grp_Info[cGrpId].SizeN)*sizeof(int)); // set all to 0

		numSpkCnt++;

		CARLSIM_INFO("SpikeCounter set for Group %d (%s): %d ms recording window",cGrpId,
			grp_Info2[cGrpId].Name.c_str(),recordDur);
	}
}

// record spike information, return a SpikeInfo object
SpikeMonitor* CpuSNN::setSpikeMonitor(int grpId, FILE* fid, int configId) {
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			setSpikeMonitor(grpId, fid ,c);
	} else {
		int cGrpId = getGroupId(grpId, configId);

		// create new SpikeMonitorCore object in any case and initialize analysis components
		// spkMonObj destructor (see below) will deallocate it
		SpikeMonitorCore* spkMonCoreObj = new SpikeMonitorCore(this, numSpikeMonitor, cGrpId);
		spikeMonCoreList[numSpikeMonitor] = spkMonCoreObj;

		// assign spike file ID if we selected to write to a file, else it's NULL
		// if file pointer exists, it has already been fopened
		// spkMonCoreObj destructor will fclose it
		spkMonCoreObj->setSpikeFileId(fid);

		// create a new SpikeMonitor object for the user-interface
		// CpuSNN::deleteObjects will deallocate it
		SpikeMonitor* spkMonObj = new SpikeMonitor(spkMonCoreObj);
		spikeMonList[numSpikeMonitor] = spkMonObj;

		// also inform the grp that it is being monitored...
		grp_Info[cGrpId].SpikeMonitorId	= numSpikeMonitor;

	    // not eating much memory anymore, got rid of all buffers
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitor*);
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitorCore*);

		numSpikeMonitor++;
		CARLSIM_INFO("SpikeMonitor set for group %d (%s)",cGrpId,grp_Info2[grpId].Name.c_str());

		return spkMonObj;
	}
}

// assigns spike rate to group
void CpuSNN::setSpikeRate(int grpId, PoissonRate* ratePtr, int refPeriod, int configId) {
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			setSpikeRate(grpId, ratePtr, refPeriod,c);
	} else {
		int cGrpId = getGroupId(grpId, configId);

		assert(ratePtr);
		if (ratePtr->len != grp_Info[cGrpId].SizeN) {
			CARLSIM_ERROR("The PoissonRate length did not match the number of neurons in group %s(%d).",
						grp_Info2[cGrpId].Name.c_str(),grpId);
			exitSimulation(1);
		}

		assert (grp_Info[cGrpId].isSpikeGenerator);
		grp_Info[cGrpId].RatePtr = ratePtr;
		grp_Info[cGrpId].RefractPeriod   = refPeriod;
		spikeRateUpdated = true;
	}
}


// function used for parameter tuning interface
void CpuSNN::updateNetwork(bool resetFiringInfo, bool resetWeights) {
	if(!doneReorganization){
		CARLSIM_ERROR("UpdateNetwork function was called but nothing was done because reorganizeNetwork must be "
						"called first.");
		return;
	}

	//change weights back to the default level for all the connections...
	if(resetWeights)
		resetSynapticConnections(true);
	else
		resetSynapticConnections(false);

	// Reset v,u,firing time values to default values...
	resetGroups();

	if(resetFiringInfo)
		resetFiringInformation();

	if(simMode_==GPU_MODE){
		//copyGrpInfo_GPU();
		//do a call to updateNetwork_GPU()
		updateNetwork_GPU(resetFiringInfo);
	}

	printTuningLog(fpDeb_);
}

// writes network state to file
// handling of file pointer should be handled externally: as far as this function is concerned, it is simply
// trying to write to file
void CpuSNN::saveSimulation(FILE* fid, bool saveSynapseInfo) {
	int tmpInt;
	float tmpFloat;

	// +++++ WRITE HEADER SECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// write file signature
	tmpInt = 294338571;
	if (!fwrite(&tmpInt,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

	// write version number
	tmpFloat = 1.0f;
	if (!fwrite(&tmpFloat,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

	// write simulation time so far (in seconds)
	tmpFloat = ((float)simTimeSec) + ((float)simTimeMs)/1000.0f;
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

	// write execution time so far (in seconds)
	if(simMode_ == GPU_MODE) {
		stopGPUTiming();
		tmpFloat = gpuExecutionTime/1000.0f;
	} else {
		stopCPUTiming();
		tmpFloat = cpuExecutionTime/1000.0f;
	}
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

	// TODO: add more params of interest

	// write network info
	if (!fwrite(&numN,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
	if (!fwrite(&preSynCnt,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
	if (!fwrite(&postSynCnt,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
	if (!fwrite(&numGrp,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

	// write group info
	char name[100];
	for (int g=0;g<numGrp;g++) {
		if (!fwrite(&grp_Info[g].StartN,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
		if (!fwrite(&grp_Info[g].EndN,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

		strncpy(name,grp_Info2[g].Name.c_str(),100);
		if (!fwrite(name,1,100,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
	}


	// +++++ WRITE SYNAPSE INFO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// \FIXME: replace with faster version
	if (saveSynapseInfo) {
		for (unsigned int i=0;i<numN;i++) {
			unsigned int offset = cumulativePost[i];

			unsigned int count = 0;
			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++)
					count++;
			}

			if (!fwrite(&count,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");

			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
					// get synaptic info...
					post_info_t post_info = postSynapticIds[offset + idx_d];

					// get neuron id
					//int p_i = (post_info&POST_SYN_NEURON_MASK);
					unsigned int p_i = GET_CONN_NEURON_ID(post_info);
					assert(p_i<numN);

					// get syn id
					unsigned int s_i = GET_CONN_SYN_ID(post_info);
					//>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;
					assert(s_i<(Npre[p_i]));

					// get the cumulative position for quick access...
					unsigned int pos_i = cumulativePre[p_i] + s_i;

					uint8_t delay = t+1;
					uint8_t plastic = s_i < Npre_plastic[p_i]; // plastic or fixed.

					if (!fwrite(&i,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&p_i,sizeof(int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(wt[pos_i]),sizeof(float),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(maxSynWt[pos_i]),sizeof(float),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&delay,sizeof(uint8_t),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&plastic,sizeof(uint8_t),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(cumConnIdPre[pos_i]),sizeof(short int),1,fid)) CARLSIM_ERROR("saveSimulation fwrite error");
				}
			}
		}
	}
}

// writes population weights from gIDpre to gIDpost to file fname in binary
void CpuSNN::writePopWeights(std::string fname, int grpPreId, int grpPostId, int configId){
	assert(grpPreId>=0); assert(grpPostId>=0);
	assert(configId>=0); // ALL not allowed

	float* weights;
	int matrixSize;
	FILE* fid;
	int numPre, numPost;
	fid = fopen(fname.c_str(), "wb");
	assert(fid != NULL);

	if(!doneReorganization){
		CARLSIM_ERROR("Simulation has not been run yet, cannot output weights.");
		exitSimulation(1);
	}

	post_info_t* preId;
	int pre_nid, pos_ij;

	int cGrpIdPre = getGroupId(grpPreId, configId);
	int cGrpIdPost = getGroupId(grpPostId, configId);

	//population sizes
	numPre = grp_Info[cGrpIdPre].SizeN;
	numPost = grp_Info[cGrpIdPost].SizeN;

	//first iteration gets the number of synaptic weights to place in our
	//weight matrix.
	matrixSize=0;
	//iterate over all neurons in the post group
	for (int i=grp_Info[cGrpIdPost].StartN; i<=grp_Info[cGrpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cumulativePre[i]; // i-th neuron, j=0th synapse
		//iterate over all presynaptic synapses
		for(int j=0; j<Npre[i]; pos_ij++,j++) {
			preId = &preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[cGrpIdPre].StartN || pre_nid>grp_Info[cGrpIdPre].EndN)
				continue; // connection does not belong to group cGrpIdPre
			matrixSize++;
		}
	}

	//now we have the correct size
	weights = new float[matrixSize];
	//second iteration assigns the weights
	int curr = 0; // iterator for return array
	//iterate over all neurons in the post group
	for (int i=grp_Info[cGrpIdPost].StartN; i<=grp_Info[cGrpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cumulativePre[i]; // i-th neuron, j=0th synapse
		//do the GPU copy here.  Copy the current weights from GPU to CPU.
		if(simMode_==GPU_MODE){
			copyWeightsGPU(i,cGrpIdPre);
		}
		//iterate over all presynaptic synapses
		for(int j=0; j<Npre[i]; pos_ij++,j++) {
			preId = &preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[cGrpIdPre].StartN || pre_nid>grp_Info[cGrpIdPre].EndN)
				continue; // connection does not belong to group cGrpIdPre
			weights[curr] = wt[pos_ij];
			curr++;
		}
	}

	fwrite(weights,sizeof(float),matrixSize,fid);
	fclose(fid);
	//Let my memory FREE!!!
	delete [] weights;
}


/// ************************************************************************************************************ ///
/// PUBLIC METHODS: PLOTTING / LOGGING
/// ************************************************************************************************************ ///

// set new file pointer for debug log file
void CpuSNN::setLogDebugFp(FILE* fpLog) {
	assert(fpLog!=NULL);

	if (fpLog_!=NULL && fpLog!=stdout && fpLog!=stderr)
		fclose(fpLog_);

	fpLog_ = fpLog;
}

// set new file pointer for all files
void CpuSNN::setLogsFp(FILE* fpInf, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
	assert(loggerMode_==CUSTOM); // only valid in custom mode
	assert(fpInf!=NULL); // at least one of the must be non-NULL

	if (fpInf_!=NULL && fpInf_!=stdout && fpInf_!=stderr)
		fclose(fpInf_);
	fpInf_ = fpInf;

	if (fpErr!=NULL) {
		if (fpErr_!=NULL && fpErr_!=stdout && fpErr_!=stderr)
			fclose(fpErr_);
		fpErr_ = fpErr;
	}

	if (fpDeb!=NULL) {
		if (fpDeb_!=NULL && fpDeb_!=stdout && fpDeb_!=stderr)
			fclose(fpDeb_);
		fpDeb_ = fpDeb;
	}

	if (fpLog!=NULL) {
		if (fpLog_!=NULL && fpLog_!=stdout && fpLog_!=stderr)
			fclose(fpLog_);
		fpLog_ = fpLog;
	}
}


/// **************************************************************************************************************** ///
/// GETTERS / SETTERS
/// **************************************************************************************************************** ///

//! used for parameter tuning functionality
grpConnectInfo_t* CpuSNN::getConnectInfo(short int connectId, int configId) {
	grpConnectInfo_t* nextConn = connectBegin;
	connectId = getConnectionId (connectId, configId);
	CHECK_CONNECTION_ID(connectId, numConnections);

	// clear all existing connection info...
	while (nextConn) {
		if (nextConn->connId == connectId) {
			nextConn->newUpdates = true;		// \FIXME: this is a Jay hack
			return nextConn;
		}
		nextConn = nextConn->next;
	}

	CARLSIM_DEBUG("Total Connections = %d", numConnections);
	CARLSIM_DEBUG("ConnectId (%d) cannot be recognized", connectId);
	return NULL;
}

int  CpuSNN::getConnectionId(short int connId, int configId) {
	assert(configId>=0); assert(configId<nConfig_);

	connId = connId+configId;
	assert(connId>=0); assert(connId<numConnections);

	return connId;
}

std::vector<float> CpuSNN::getConductanceAMPA() {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceAMPA(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, ALL);

	std::vector<float> gAMPAvec;
	for (int i=0; i<numNReg; i++)
		gAMPAvec.push_back(gAMPA[i]);
	return gAMPAvec;
}

std::vector<float> CpuSNN::getConductanceNMDA() {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceNMDA(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, ALL);

	std::vector<float> gNMDAvec;
	if (isSimulationWithNMDARise()) {
		// need to construct conductance from rise and decay parts
		for (int i=0; i<numNReg; i++) {
			gNMDAvec.push_back(gNMDA_d[i]-gNMDA_r[i]);
		}
	} else {
		for (int i=0; i<numNReg; i++)
			gNMDAvec.push_back(gNMDA[i]);
	}
	return gNMDAvec;
}

std::vector<float> CpuSNN::getConductanceGABAa() {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceGABAa(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, ALL);

	std::vector<float> gGABAaVec;
	for (int i=0; i<numNReg; i++)
		gGABAaVec.push_back(gGABAa[i]);
	return gGABAaVec;
}

std::vector<float> CpuSNN::getConductanceGABAb() {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceGABAb(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, ALL);

	std::vector<float> gGABAbVec;
	if (isSimulationWithNMDARise()) {
		// need to construct conductance from rise and decay parts
		for (int i=0; i<numNReg; i++) {
			gGABAbVec.push_back(gGABAb_d[i]-gGABAb_r[i]);
		}
	} else {
		for (int i=0; i<numNReg; i++)
			gGABAbVec.push_back(gGABAb[i]);
	}
	return gGABAbVec;
}

// this is a user function
// \FIXME: fix this
uint8_t* CpuSNN::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	Npre = grp_Info[gIDpre].SizeN;
	Npost = grp_Info[gIDpost].SizeN;

	if (delays == NULL) delays = new uint8_t[Npre*Npost];
	memset(delays,0,Npre*Npost);

	for (int i=grp_Info[gIDpre].StartN;i<grp_Info[gIDpre].EndN;i++) {
		unsigned int offset = cumulativePost[i];

		for (int t=0;t<maxDelay_;t++) {
			delay_info_t dPar = postDelayInfo[i*(maxDelay_+1)+t];

			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
				// get synaptic info...
				post_info_t post_info = postSynapticIds[offset + idx_d];

				// get neuron id
				//int p_i = (post_info&POST_SYN_NEURON_MASK);
				int p_i = GET_CONN_NEURON_ID(post_info);
				assert(p_i<numN);

				if (p_i >= grp_Info[gIDpost].StartN && p_i <= grp_Info[gIDpost].EndN) {
					// get syn id
					int s_i = GET_CONN_SYN_ID(post_info);

					// get the cumulative position for quick access...
					unsigned int pos_i = cumulativePre[p_i] + s_i;

					delays[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = t+1;
				}
			}
		}
	}
	return delays;
}

Grid3D CpuSNN::getGroupGrid3D(int grpId) {
	assert(grpId>=0 && grpId<numGrp);
	return Grid3D(grp_Info[grpId].SizeX, grp_Info[grpId].SizeY, grp_Info[grpId].SizeZ);	
}

int CpuSNN::getGroupId(int grpId, int configId) {
	assert(grpId>=0 && grpId<numGrp);
	assert(configId>=0 && configId<nConfig_);

	int cGrpId = (grpId+configId);
	assert(cGrpId  < numGrp);
	return cGrpId;
}

group_info_t CpuSNN::getGroupInfo(int grpId, int configId) {
	assert(grpId>=-1 && grpId<numGrp);
	assert(configId>=-1 && configId<nConfig_);

	int cGrpId = getGroupId(grpId, configId);
	return grp_Info[cGrpId];
}

std::string CpuSNN::getGroupName(int grpId, int configId) {
	assert(grpId>=-1 && grpId<numGrp);
	assert(configId>=-1 && configId<nConfig_);

	if (grpId==ALL)
		return "ALL";

	if (configId==ALL) {
        std::string name = grp_Info2[grpId].Name;
		return name+",ALL";
    }

	int cGrpId = getGroupId(grpId, configId);
	return grp_Info2[cGrpId].Name;
}

Point3D CpuSNN::getNeuronLocation3D(int neurId) {
	assert(neurId>=0 && neurId<numN);
	int grpId = grpIds[neurId];
	assert(neurId>=grp_Info[grpId].StartN && neurId<=grp_Info[grpId].EndN);

	// adjust neurId for neuron ID of first neuron in the group
	neurId -= grp_Info[grpId].StartN;

	int coord_x = neurId % grp_Info[grpId].SizeX;
	int coord_y = (neurId/grp_Info[grpId].SizeX)%grp_Info[grpId].SizeY;
	int coord_z = neurId/(grp_Info[grpId].SizeX*grp_Info[grpId].SizeY);
	return Point3D(coord_x, coord_y, coord_z);
}

// returns the number of synaptic connections associated with this connection.
int CpuSNN::getNumSynapticConnections(short int connectionId) {
  grpConnectInfo_t* connInfo;
  grpConnectInfo_t* connIterator = connectBegin;
  while(connIterator){
    if(connIterator->connId == connectionId){
      //found the corresponding connection
      return connIterator->numberOfConnections;
    }
    //move to the next grpConnectInfo_t
    connIterator=connIterator->next;
  }
  //we didn't find the connection.
  CARLSIM_ERROR("Connection ID was not found.  Quitting.");
  exitSimulation(1);
}

// gets weights from synaptic connections from gIDpre to gIDpost
void CpuSNN::getPopWeights(int grpPreId, int grpPostId, float*& weights, int& matrixSize, int configId) {
	assert(configId>=0); assert(configId<nConfig_);
	assert(grpPreId>=0); assert(grpPreId<numGrp); assert(grpPostId>=0); assert(grpPostId<numGrp);
	post_info_t* preId;
	int pre_nid, pos_ij;
	int numPre, numPost;

	int cGrpIdPre = getGroupId(grpPreId, configId);
	int cGrpIdPost = getGroupId(grpPostId, configId);

	//population sizes
	numPre = grp_Info[cGrpIdPre].SizeN;
	numPost = grp_Info[cGrpIdPost].SizeN;

	//first iteration gets the number of synaptic weights to place in our
	//weight matrix.
	matrixSize=0;
	//iterate over all neurons in the post group
	for (int i=grp_Info[cGrpIdPost].StartN; i<=grp_Info[cGrpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cumulativePre[i]; // i-th post neuron, jth pre neuron
		//iterate over all presynaptic synapses of the current postsynaptic neuron
		for(int j=0; j<Npre[i]; pos_ij++,j++) {
			preId = &preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[cGrpIdPre].StartN || pre_nid>grp_Info[cGrpIdPre].EndN)
				continue; // connection does not belong to group cGrpIdPre
			matrixSize++;
		}
	}
	//now we have the correct size matrix
	weights = new float[matrixSize];

	//second iteration assigns the weights
	int curr = 0; // iterator for return array

	//iterate over all neurons in the post group
	for (int i=grp_Info[cGrpIdPost].StartN; i<=grp_Info[cGrpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cumulativePre[i]; // i-th neuron, j=0th synapse
		//do the GPU copy here.  Copy the current weights from GPU to CPU.
		if(simMode_==GPU_MODE){
			copyWeightsGPU(i,cGrpIdPre);
		}
		//iterate over all presynaptic synapses
		for(int j=0; j<Npre[i]; pos_ij++,j++) {
			//TAGS:TODO: We have to double check we have access to preSynapticIds in GPU_MODE.
			//We can check where they were allocated and make sure that this occurs in
			//both the CPU and GPU modes.
			preId = &preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[cGrpIdPre].StartN || pre_nid>grp_Info[cGrpIdPre].EndN)
				continue; // connection does not belong to group cGrpIdPre
			//the weights stored in wt were copied from the GPU in the above block
			weights[curr] = wt[pos_ij];
			curr++;
		}
	}
}

// Returns pointer to nSpikeCnt, which is a 1D array of the number of spikes every neuron in the group
int* CpuSNN::getSpikeCntPtr(int grpId) {
	//! do check to make sure appropriate flag is set
	if(simMode_ == GPU_MODE && enableGPUSpikeCntPtr == false){
		CARLSIM_ERROR("Error: the enableGPUSpikeCntPtr flag must be set to true to use this function in GPU_MODE.");
		assert(enableGPUSpikeCntPtr);
	}
    
	if(simMode_ == GPU_MODE){
		assert(enableGPUSpikeCntPtr);
	}
    
	return ((grpId == -1) ? nSpikeCnt : &nSpikeCnt[grp_Info[grpId].StartN]);
}

// return spike buffer, which contains #spikes per neuron in the group
int* CpuSNN::getSpikeCounter(int grpId, int configId) {
	assert(grpId>=0); assert(grpId<numGrp); assert(configId>=0); assert(configId<nConfig_);

	int cGrpId = getGroupId(grpId, configId);
	if (!grp_Info[cGrpId].withSpikeCounter)
		return NULL;

	if (simMode_==GPU_MODE)
		return getSpikeCounter_GPU(grpId,configId);
	else {
		int bufPos = grp_Info[cGrpId].spkCntBufPos; // retrieve buf pos
		return spkCntBuf[bufPos]; // return pointer to buffer
	}
}

// this is a user function
// TODO: fix this
float* CpuSNN::getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges) {
	 Npre = grp_Info[gIDpre].SizeN;
	 Npost = grp_Info[gIDpost].SizeN;

	if (weightChanges==NULL) weightChanges = new float[Npre*Npost];
		memset(weightChanges,0,Npre*Npost*sizeof(float));

	// copy the pre synaptic data from GPU, if needed
	// note: this will not include wtChange[] and synSpikeTime[] if sim_with_fixedwts
	if (simMode_ == GPU_MODE)
    	copyWeightState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, gIDpost);

	for (int i=grp_Info[gIDpre].StartN;i<grp_Info[gIDpre].EndN;i++) {
		unsigned int offset = cumulativePost[i];

		for (int t=0;t<maxDelay_;t++) {
			delay_info_t dPar = postDelayInfo[i*(maxDelay_+1)+t];

			for(int idx_d = dPar.delay_index_start; idx_d < (dPar.delay_index_start + dPar.delay_length); idx_d = idx_d+1) {

				// get synaptic info...
				post_info_t post_info = postSynapticIds[offset + idx_d];

				// get neuron id
				//int p_i = (post_info&POST_SYN_NEURON_MASK);
				int p_i = GET_CONN_NEURON_ID(post_info);
				assert(p_i<numN);

				if (p_i >= grp_Info[gIDpost].StartN && p_i <= grp_Info[gIDpost].EndN) {
					// get syn id
					int s_i = GET_CONN_SYN_ID(post_info);

					// get the cumulative position for quick access...
					unsigned int pos_i = cumulativePre[p_i] + s_i;

					// if a group has fixed input weights, it will not have wtChange[] on the GPU side
					if (grp_Info[gIDpost].FixedInputWts)
						weightChanges[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = 0.0f;
					else
						weightChanges[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = wtChange[pos_i];
				}
			}
		}
	}

	return weightChanges;
}


// True allows getSpikeCntPtr_GPU to copy firing state information from GPU kernel to cpuNetPtrs
// Warning: setting this flag to true will slow down the simulation significantly.
void CpuSNN::setCopyFiringStateFromGPU(bool _enableGPUSpikeCntPtr) {
	enableGPUSpikeCntPtr=_enableGPUSpikeCntPtr;
}

/*
 * \deprecated this is not a good idea...
void CpuSNN::setGroupInfo(int grpId, group_info_t info, int configId) {
	if (configId == ALL) {
		for(int c=0; c < nConfig_; c++)
			setGroupInfo(grpId, info, c);
	} else {
		int cGrpId = getGroupId(grpId, configId);
		grp_Info[cGrpId] = info;
	}
}
*/


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// all unsafe operations of CpuSNN constructor
void CpuSNN::CpuSNNinit() {
	assert(nConfig_>0 && nConfig_<=MAX_nConfig); assert(ithGPU_>=0);
	assert(simMode_!=UNKNOWN_SIM); assert(loggerMode_!=UNKNOWN_LOGGER);

	// set logger mode (defines where to print all status, error, and debug messages)
	switch (loggerMode_) {
	case USER:
		fpInf_ = stdout;
		fpErr_ = stderr;
		#if (WIN32 || WIN64)
			fpDeb_ = fopen("nul","w");
		#else
			fpDeb_ = fopen("/dev/null","w");
		#endif
		break;
	case DEVELOPER:
		fpInf_ = stdout;
		fpErr_ = stderr;
		fpDeb_ = stdout;
		break;
	case SHOWTIME:
		#if (WIN32 || WIN64)
			fpInf_ = fopen("nul","w");
		#else
			fpInf_ = fopen("/dev/null","w");
		#endif
		fpErr_ = stderr;
		#if (WIN32 || WIN64)
			fpDeb_ = fopen("nul","w");
		#else
			fpDeb_ = fopen("/dev/null","w");
		#endif
		break;
	case SILENT:
	case CUSTOM:
		#if (WIN32 || WIN64)
			fpInf_ = fopen("nul","w");
			fpErr_ = fopen("nul","w");
			fpDeb_ = fopen("nul","w");
		#else
			fpInf_ = fopen("/dev/null","w");
			fpErr_ = fopen("/dev/null","w");
			fpDeb_ = fopen("/dev/null","w");
		#endif
	break;
	default:
		CARLSIM_ERROR("Unknown logger mode");
		exit(1);
	}
	fpLog_ = fopen("debug.log","w");

	#ifdef __REGRESSION_TESTING__
	#if (WIN32 || WIN64)
		fpInf_ = fopen("nul","w");
		fpErr_ = fopen("nul","w");
		fpDeb_ = fopen("nul","w");
	#else
		fpInf_ = fopen("/dev/null","w");
		fpErr_ = fopen("/dev/null","w");
		fpDeb_ = fopen("/dev/null","w");
	#endif
	#endif

	CARLSIM_INFO("*********************************************************************************");
	CARLSIM_INFO("********************      Welcome to CARLsim %d.%d      ***************************",
				MAJOR_VERSION,MINOR_VERSION);
	CARLSIM_INFO("*********************************************************************************\n");

	CARLSIM_INFO("***************************** Configuring Network ********************************");
	CARLSIM_INFO("Starting CARLsim simulation \"%s\" in %s mode",networkName_.c_str(),
		loggerMode_string[loggerMode_]);
	CARLSIM_INFO("nConfig: %d, randSeed: %d",nConfig_,randSeed_);

	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	CARLSIM_DEBUG("Current local time and date: %s", asctime(timeinfo));

	// init random seed
	srand48(randSeed_);
	getRand.seed(randSeed_*2);
	getRandClosed.seed(randSeed_*3);

	finishedPoissonGroup  = false;
	connectBegin = NULL;

	simTimeLastUpdSpkMon_ = 0;
	simTimeRunStart     = 0;    simTimeRunStop      = 0;
	simTimeMs	 		= 0;    simTimeSec          = 0;    simTime = 0;
	spikeCountAll1sec	= 0;    secD1fireCntHost    = 0;    secD2fireCntHost  = 0;
	spikeCountAll 		= 0;    spikeCountD2Host    = 0;    spikeCountD1Host = 0;
	nPoissonSpikes 		= 0;

	numGrp   = 0;
	numConnections = 0;
	numSpikeGenGrps  = 0;
	NgenFunc = 0;
	simulatorDeleted = false;
	enableGPUSpikeCntPtr = false;

	allocatedN      = 0;
	allocatedPre    = 0;
	allocatedPost   = 0;
	doneReorganization = false;
	memoryOptimized	   = false;

	cumExecutionTime = 0.0;
	cpuExecutionTime = 0.0;
	gpuExecutionTime = 0.0;

	spikeRateUpdated = false;
	numSpikeMonitor = 0;
	numGroupMonitor = 0;
	numConnectionMonitor = 0;
	numSpkCnt = 0;

	sim_with_fixedwts = true; // default is true, will be set to false if there are any plastic synapses
	sim_with_conductances = false; // default is false
	sim_with_stdp = false;
	sim_with_modulated_stdp = false;
	sim_with_homeostasis = false;
	sim_with_stp = false;

	maxSpikesD2 = maxSpikesD1 = 0;
	readNetworkFID = NULL;

	numN = 0;
	numPostSynapses = 0;
	maxDelay_ = 0;

	// conductance info struct for simulation
	sim_with_NMDA_rise = false;
	sim_with_GABAb_rise = false;
	dAMPA  = 1.0-1.0/5.0;		// some default decay and rise times
	rNMDA  = 1.0-1.0/10.0;
	dNMDA  = 1.0-1.0/150.0;
	sNMDA  = 1.0;
	dGABAa = 1.0-1.0/6.0;
	rGABAb = 1.0-1.0/100.0;
	dGABAb = 1.0-1.0/150.0;
	sGABAb = 1.0;

	// reset all pointers, don't deallocate (false)
	resetPointers(false);


	memset(&cpuSnnSz, 0, sizeof(cpuSnnSz));

	showGrpFiringInfo = true;

	// initialize propogated spike buffers.....
	pbuf = new PropagatedSpikeBuffer(0, PROPAGATED_BUFFER_SIZE);

	memset(&cpu_gpuNetPtrs, 0, sizeof(network_ptr_t));
	memset(&net_Info, 0, sizeof(network_info_t));
	cpu_gpuNetPtrs.allocated = false;

	memset(&cpuNetPtrs, 0, sizeof(network_ptr_t));
	cpuNetPtrs.allocated = false;

	for (int i=0; i < MAX_GRP_PER_SNN; i++) {
		grp_Info[i].Type = UNKNOWN_NEURON;
		grp_Info[i].MaxFiringRate = UNKNOWN_NEURON_MAX_FIRING_RATE;
		grp_Info[i].SpikeMonitorId = -1;
		grp_Info[i].GroupMonitorId = -1;
		grp_Info[i].ConnectionMonitorId = -1;
		grp_Info[i].FiringCount1sec=0;
		grp_Info[i].numPostSynapses 		= 0;	// default value
		grp_Info[i].numPreSynapses 	= 0;	// default value
		grp_Info[i].WithSTP = false;
		grp_Info[i].WithSTDP = false;
		grp_Info[i].WithSTDPtype = UNKNOWN_STDP;
		grp_Info[i].FixedInputWts = true; // Default is true. This value changed to false
		// if any incoming  connections are plastic
		grp_Info[i].isSpikeGenerator = false;
		grp_Info[i].RatePtr = NULL;

		grp_Info[i].homeoId = -1;
		grp_Info[i].avgTimeScale  = 10000.0;

		grp_Info[i].baseDP = 1.0f;
		grp_Info[i].base5HT = 1.0f;
		grp_Info[i].baseACh = 1.0f;
		grp_Info[i].baseNE = 1.0f;
		grp_Info[i].decayDP = 1 - (1.0f / 100);
		grp_Info[i].decay5HT = 1 - (1.0f / 100);
		grp_Info[i].decayACh = 1 - (1.0f / 100);
		grp_Info[i].decayNE = 1 - (1.0f / 100);

		grp_Info[i].spikeGen = NULL;

		grp_Info[i].withSpikeCounter = false;
		grp_Info[i].spkCntRecordDur = -1;
		grp_Info[i].spkCntRecordDurHelper = 0;
		grp_Info[i].spkCntBufPos = -1;

		grp_Info[i].StartN       = -1;
		grp_Info[i].EndN       	 = -1;

		grp_Info[i].CurrTimeSlice = 0;
		grp_Info[i].NewTimeSlice = 0;
		grp_Info[i].SliceUpdateTime = 0;

		grp_Info2[i].numPostConn = 0;
		grp_Info2[i].numPreConn  = 0;
		grp_Info2[i].maxPostConn = 0;
		grp_Info2[i].maxPreConn  = 0;
		grp_Info2[i].sumPostConn = 0;
		grp_Info2[i].sumPreConn  = 0;

	}

	CUDA_CREATE_TIMER(timer);
	CUDA_RESET_TIMER(timer);

	// default weight update parameter
	wtUpdateInterval_ = 1000; // update weights every 1000 ms (default)
	wtUpdateIntervalCnt_ = 0; // helper var to implement fast modulo
	stdpScaleFactor_ = 1.0f;
	wtChangeUpdateInterval_ = 1000; // update weight change every 1000 ms (default)
	wtChangeDecay_ = 0.0f;

	// initialize parameters needed in snn_gpu.cu
	// \FIXME: naming is terrible... so it's a CPU SNN on GPU...
	CpuSNNinit_GPU();
}

//! update (initialize) numN, numPostSynapses, numPreSynapses, maxDelay_, postSynCnt, preSynCnt
//! allocate space for voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, gAMPA, gNMDA, gGABAa, gGABAb
//! lastSpikeTime, curSpike, nSpikeCnt, intrinsicWeight, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre
//! postSynapticIds, tmp_SynapticDely, postDelayInfo, wt, maxSynWt, preSynapticIds, timeTableD2, timeTableD1
void CpuSNN::buildNetworkInit(unsigned int nNeur, unsigned int nPostSyn, unsigned int nPreSyn, unsigned int maxDelay) {
	numN = nNeur;
	numPostSynapses = nPostSyn;
	maxDelay_ = maxDelay;
	numPreSynapses = nPreSyn;

	// \FIXME: need to figure out STP buffer for delays > 1
	if (sim_with_stp && maxDelay>1) {
		CARLSIM_ERROR("STP with delays > 1 ms is currently not supported.");
		exitSimulation(1);
	}

	voltage	 = new float[numNReg];
	recovery = new float[numNReg];
	Izh_a	 = new float[numNReg];
	Izh_b    = new float[numNReg];
	Izh_c	 = new float[numNReg];
	Izh_d	 = new float[numNReg];
	current	 = new float[numNReg];
	cpuSnnSz.neuronInfoSize += (sizeof(float)*numNReg*7);

	if (sim_with_conductances) {
		gAMPA  = new float[numNReg];
		gGABAa = new float[numNReg];
		cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;

		if (sim_with_NMDA_rise) {
			// If NMDA rise time is enabled, we'll have to compute NMDA conductance in two steps (using an exponential
			// for the rise time and one for the decay time)
			gNMDA_r = new float[numNReg];
			gNMDA_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			gNMDA = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}

		if (sim_with_GABAb_rise) {
			gGABAb_r = new float[numNReg];
			gGABAb_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			gGABAb = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}
	}

	grpDA = new float[numGrp];
	grp5HT = new float[numGrp];
	grpACh = new float[numGrp];
	grpNE = new float[numGrp];

	resetCurrent();
	resetConductances();

	lastSpikeTime	= new uint32_t[numN];
	cpuSnnSz.neuronInfoSize += sizeof(int) * numN;
	memset(lastSpikeTime, 0, sizeof(lastSpikeTime[0]) * numN);

	curSpike   = new bool[numN];
	nSpikeCnt  = new int[numN];
	CARLSIM_INFO("allocated nSpikeCnt");

	//! homeostasis variables
	if (sim_with_homeostasis) {
		avgFiring  = new float[numN];
		baseFiring = new float[numN];
	}

	intrinsicWeight  = new float[numN];
	memset(intrinsicWeight,0,sizeof(float)*numN);
	cpuSnnSz.neuronInfoSize += (sizeof(int)*numN*2+sizeof(bool)*numN);

	// STP can be applied to spike generators, too -> numN
	if (sim_with_stp) {
		// \TODO: The size of these data structures could be reduced to the max synaptic delay of all
		// connections with STP. That number might not be the same as maxDelay_.
		stpu = new float[numN*(maxDelay_+1)];
		stpx = new float[numN*(maxDelay_+1)];
		memset(stpu, 0, sizeof(float)*numN*(maxDelay_+1)); // memset works for 0.0
		for (int i=0; i < numN*(maxDelay_+1); i++)
			stpx[i] = 1.0f; // but memset doesn't work for 1.0
		cpuSnnSz.synapticInfoSize += (2*sizeof(float)*numN*(maxDelay_+1));
	}

	Npre 		   = new unsigned short[numN];
	Npre_plastic   = new unsigned short[numN];
	Npost 		   = new unsigned short[numN];
	cumulativePost = new unsigned int[numN];
	cumulativePre  = new unsigned int[numN];
	cpuSnnSz.networkInfoSize += (int)(sizeof(int) * numN * 3.5);

	postSynCnt = 0;
	preSynCnt  = 0;
	for(int g = 0; g < numGrp; g++) {
		// check for INT overflow: postSynCnt is O(numNeurons*numSynapses), must be able to fit within u int limit
		assert(postSynCnt < UINT_MAX - (grp_Info[g].SizeN * grp_Info[g].numPostSynapses));
		assert(preSynCnt < UINT_MAX - (grp_Info[g].SizeN * grp_Info[g].numPreSynapses));
		postSynCnt += (grp_Info[g].SizeN * grp_Info[g].numPostSynapses);
		preSynCnt  += (grp_Info[g].SizeN * grp_Info[g].numPreSynapses);
	}
	assert(postSynCnt/numN <= numPostSynapses); // divide by numN to prevent INT overflow
	postSynapticIds		= new post_info_t[postSynCnt+100];
	tmp_SynapticDelay	= new uint8_t[postSynCnt+100];	//!< Temporary array to store the delays of each connection
	postDelayInfo		= new delay_info_t[numN*(maxDelay_+1)];	//!< Possible delay values are 0....maxDelay_ (inclusive of maxDelay_)
	cpuSnnSz.networkInfoSize += ((sizeof(post_info_t)+sizeof(uint8_t))*postSynCnt+100)+(sizeof(delay_info_t)*numN*(maxDelay_+1));
	assert(preSynCnt/numN <= numPreSynapses); // divide by numN to prevent INT overflow

	wt  			= new float[preSynCnt+100];
	maxSynWt     	= new float[preSynCnt+100];

	mulSynFast 		= new float[MAX_nConnections];
	mulSynSlow 		= new float[MAX_nConnections];
	cumConnIdPre	= new short int[preSynCnt+100];

	//! Temporary array to hold pre-syn connections. will be deleted later if necessary
	preSynapticIds	= new post_info_t[preSynCnt + 100];
	// size due to weights and maximum weights
	cpuSnnSz.synapticInfoSize += ((sizeof(int) + 2 * sizeof(float) + sizeof(post_info_t)) * (preSynCnt + 100));

	timeTableD2  = new unsigned int[1000 + maxDelay_ + 1];
	timeTableD1  = new unsigned int[1000 + maxDelay_ + 1];
	resetTimingTable();
	cpuSnnSz.spikingInfoSize += sizeof(int) * 2 * (1000 + maxDelay_ + 1);

	// poisson Firing Rate
	cpuSnnSz.neuronInfoSize += (sizeof(int) * numNPois);
}


int CpuSNN::addSpikeToTable(int nid, int g) {
	int spikeBufferFull = 0;
	lastSpikeTime[nid] = simTime;
	curSpike[nid] = true;
	nSpikeCnt[nid]++;
	if (sim_with_homeostasis)
		avgFiring[nid] += 1000/(grp_Info[g].avgTimeScale*1000);

	if (simMode_ == GPU_MODE) {
		assert(grp_Info[g].isSpikeGenerator == true);
		setSpikeGenBit_GPU(nid, g);
		return 0;
	}

	if (grp_Info[g].WithSTP) {
		// update the spike-dependent part of du/dt and dx/dt
		// we need to retrieve the STP values from the right buffer position (right before vs. right after the spike)
		int ind_plus = STP_BUF_POS(nid,simTime); // index of right after the spike, such as in u^+
	    int ind_minus = STP_BUF_POS(nid,(simTime-1)); // index of right before the spike, such as in u^-

		// du/dt = -u/tau_F + U * (1-u^-) * \delta(t-t_{spk})
		stpu[ind_plus] += grp_Info[g].STP_U*(1.0-stpu[ind_minus]);

		// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
		stpx[ind_plus] -= stpu[ind_plus]*stpx[ind_minus];
	}

	if (grp_Info[g].MaxDelay == 1) {
		assert(nid < numN);
		firingTableD1[secD1fireCntHost] = nid;
		secD1fireCntHost++;
		grp_Info[g].FiringCount1sec++;
		if (secD1fireCntHost >= maxSpikesD1) {
			spikeBufferFull = 2;
			secD1fireCntHost = maxSpikesD1-1;
		}
	} else {
		assert(nid < numN);
		firingTableD2[secD2fireCntHost] = nid;
		grp_Info[g].FiringCount1sec++;
		secD2fireCntHost++;
		if (secD2fireCntHost >= maxSpikesD2) {
			spikeBufferFull = 1;
			secD2fireCntHost = maxSpikesD2-1;
		}
	}
	return spikeBufferFull;
}


void CpuSNN::buildGroup(int grpId) {
	assert(grp_Info[grpId].StartN == -1);
	grp_Info[grpId].StartN = allocatedN;
	grp_Info[grpId].EndN   = allocatedN + grp_Info[grpId].SizeN - 1;

	CARLSIM_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);

	resetNeuromodulator(grpId);

	allocatedN = allocatedN + grp_Info[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		resetNeuron(i, grpId);
		Npre_plastic[i]	= 0;
		Npre[i]		  	= 0;
		Npost[i]	  	= 0;
		cumulativePost[i] = allocatedPost;
		cumulativePre[i]  = allocatedPre;
		allocatedPost    += grp_Info[grpId].numPostSynapses;
		allocatedPre     += grp_Info[grpId].numPreSynapses;
	}

	assert(allocatedPost <= postSynCnt);
	assert(allocatedPre  <= preSynCnt);
}

//! build the network based on the current setting (e.g., group, connection)
/*!
 * \sa createGroup(), connect()
 */
void CpuSNN::buildNetwork() {
	grpConnectInfo_t* newInfo = connectBegin;
	int curN = 0, curD = 0, numPostSynapses = 0, numPreSynapses = 0;

	assert(nConfig_ > 0);

	//update main set of parameters
	// update (initialize) numNExcPois, numNInhPois, numNExcReg, numNInhReg, numNPois, numNReg
	updateParameters(&curN, &numPostSynapses, &numPreSynapses, nConfig_);

	// update (initialize) maxSpikesD1, maxSpikesD2 and allocate sapce for firingTableD1 and firingTableD2
	curD = updateSpikeTables();

	assert((curN > 0) && (curN == numNExcReg + numNInhReg + numNPois));
	assert(numPostSynapses > 0);
	assert(numPreSynapses > 0);

	// display the evaluated network and delay length....
	CARLSIM_INFO("\n***************************** Setting up Network **********************************");
	CARLSIM_INFO("numN = %d, numPostSynapses = %d, numPreSynapses = %d, maxDelay = %d", curN, numPostSynapses,
					numPreSynapses, curD);

	assert(curD != 0);
	assert(numPostSynapses != 0);
	assert(curN != 0);
	assert(numPreSynapses != 0);

	if (numPostSynapses > MAX_nPostSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPostSynapses>MAX_nPostSynapses)
				CARLSIM_ERROR("Grp: %s(%d) has too many output synapses (%d), max %d.",grp_Info2[g].Name.c_str(),g,
							grp_Info[g].numPostSynapses,MAX_nPostSynapses);
		}
		assert(numPostSynapses <= MAX_nPostSynapses);
	}
	if (numPreSynapses > MAX_nPreSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPreSynapses>MAX_nPreSynapses)
				CARLSIM_ERROR("Grp: %s(%d) has too many input synapses (%d), max %d.",grp_Info2[g].Name.c_str(),g,
 							grp_Info[g].numPreSynapses,MAX_nPreSynapses);
		}
		assert(numPreSynapses <= MAX_nPreSynapses);
	}
	assert(curD <= MAX_SynapticDelay); assert(curN <= 1000000);

	// initialize all the parameters....
	//! update (initialize) numN, numPostSynapses, numPreSynapses, maxDelay_, postSynCnt, preSynCnt
	//! allocate space for voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, gAMPA, gNMDA, gGABAa, gGABAb
	//! lastSpikeTime, curSpike, nSpikeCnt, intrinsicWeight, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre
	//! postSynapticIds, tmp_SynapticDely, postDelayInfo, wt, maxSynWt, preSynapticIds, timeTableD2, timeTableD1, grpDA, grp5HT, grpACh, grpNE
	buildNetworkInit(curN, numPostSynapses, numPreSynapses, curD);

	// we build network in the order...
	/////    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
	////     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
	///      Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
	int allocatedGrp = 0;
	for(int order = 0; order < 4; order++) {
		for(int configId = 0; configId < nConfig_; configId++) {
			for(int g = 0; g < numGrp; g++) {
				if (grp_Info2[g].ConfigId == configId) {
					if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && (grp_Info[g].Type & POISSON_NEURON) && order == 3) {
						buildPoissonGroup(g);
						allocatedGrp++;
					} else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type & POISSON_NEURON) && order == 2) {
						buildPoissonGroup(g);
						allocatedGrp++;
					} else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON) && order == 0) {
						buildGroup(g);
						allocatedGrp++;
					} else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON) && order == 1) {
						buildGroup(g);
						allocatedGrp++;
					}
				}
			}
		}
	}
	assert(allocatedGrp == numGrp);

	// print group overview
	for (int g=0;g<numGrp;g++) {
		printGroupInfo(g);
	}


	grpIds = new short int[numN];
	for (int nid=0; nid<numN; nid++) {
		grpIds[nid] = -1;
		for (int g=0; g<numGrp; g++) {
			if (nid>=grp_Info[g].StartN && nid<=grp_Info[g].EndN) {
				grpIds[nid] = (short int)g;
//				printf("grpIds[%d] = %d\n",nid,g);
				break;
			}
		}
		assert(grpIds[nid]!=-1);
	}

	if (readNetworkFID != NULL) {
		// we the user specified readNetwork the synaptic weights will be restored here...
		#if READNETWORK_ADD_SYNAPSES_FROM_FILE
			assert(readNetwork_internal(true) >= 0); // read the plastic synapses first
			assert(readNetwork_internal(false) >= 0); // read the fixed synapses second
		#endif
	} else {
		// build all the connections here...
		// we run over the linked list two times...
		// first time, we make all plastic connections...
		// second time, we make all fixed connections...
		// this ensures that all the initial pre and post-synaptic
		// connections are of fixed type and later if of plastic type
		for(int con = 0; con < 2; con++) {
			newInfo = connectBegin;
			while(newInfo) {
				bool synWtType = GET_FIXED_PLASTIC(newInfo->connProp);
				if (synWtType == SYN_PLASTIC) {
					// given group has plastic connection, and we need to apply STDP rule...
					grp_Info[newInfo->grpDest].FixedInputWts = false;
				}

				// store scaling factors for synaptic currents in connection-centric array
				mulSynFast[newInfo->connId] = newInfo->mulSynFast;
				mulSynSlow[newInfo->connId] = newInfo->mulSynSlow;


				if( ((con == 0) && (synWtType == SYN_PLASTIC)) || ((con == 1) && (synWtType == SYN_FIXED))) {
					switch(newInfo->type) {
						case CONN_RANDOM:
							connectRandom(newInfo);
							break;
						case CONN_FULL:
							connectFull(newInfo);
							break;
						case CONN_FULL_NO_DIRECT:
							connectFull(newInfo);
							break;
						case CONN_ONE_TO_ONE:
							connectOneToOne(newInfo);
							break;
						case CONN_USER_DEFINED:
							connectUserDefined(newInfo);
							break;
						default:
							CARLSIM_ERROR("Invalid connection type( should be 'random', or 'full')");
							exitSimulation(-1);
					}

					float avgPostM = newInfo->numberOfConnections/grp_Info[newInfo->grpSrc].SizeN;
					float avgPreM  = newInfo->numberOfConnections/grp_Info[newInfo->grpDest].SizeN;

					CARLSIM_INFO("connect(%s(%d) => %s(%d), iWt=%1.4f, mWt=%1.4f, numPostSynapses=%d, numPreSynapses=%d, "
									"minD=%d, maxD=%d, %s)", grp_Info2[newInfo->grpSrc].Name.c_str(), newInfo->grpSrc,
									grp_Info2[newInfo->grpDest].Name.c_str(), newInfo->grpDest, newInfo->initWt,
									newInfo->maxWt, (int)avgPostM, (int)avgPreM, newInfo->minDelay, newInfo->maxDelay,
									synWtType?"Plastic":"Fixed");
				}
				newInfo = newInfo->next;
			}
		}
	}
}

void CpuSNN::buildPoissonGroup(int grpId) {
	assert(grp_Info[grpId].StartN == -1);
	grp_Info[grpId].StartN 	= allocatedN;
	grp_Info[grpId].EndN   	= allocatedN + grp_Info[grpId].SizeN - 1;

	CARLSIM_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);

	allocatedN = allocatedN + grp_Info[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		resetPoissonNeuron(i, grpId);
		Npre_plastic[i]	  = 0;
		Npre[i]		  	  = 0;
		Npost[i]	      = 0;
		cumulativePost[i] = allocatedPost;
		cumulativePre[i]  = allocatedPre;
		allocatedPost    += grp_Info[grpId].numPostSynapses;
		allocatedPre     += grp_Info[grpId].numPreSynapses;
	}
	assert(allocatedPost <= postSynCnt);
	assert(allocatedPre  <= preSynCnt);
}

/*!
 * \brief check whether Spike Counters need to be reset
 *
 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur)
 * After this period of time, the spike buffers need to be reset. The trick is to reset it in the very next
 * millisecond, before continuing. For example, if recordDur=1000ms, we want to reset it right before we start
 * executing the 1001st millisecond, so that at t=1000ms the user is able to access non-zero data.
 */
void CpuSNN::checkSpikeCounterRecordDur() {
	for (int g=0; g<numGrp; g++) {
		// skip groups w/o spkMonRT or non-real record durations
		if (!grp_Info[g].withSpikeCounter || grp_Info[g].spkCntRecordDur<=0)
			continue;

		// skip if simTime doesn't need udpating
		// we want to update in spkCntRecordDur + 1, because this function is called rigth at the beginning
		// of each millisecond
		if ( (simTime % ++grp_Info[g].spkCntRecordDurHelper) != 1)
			continue;

 		if (simMode_==GPU_MODE)
			resetSpikeCounter_GPU(g,0);
		else
			resetSpikeCounter(g,0);
	}
}

// We parallelly cleanup the postSynapticIds array to minimize any other wastage in that array by compacting the store
// Appropriate alignment specified by ALIGN_COMPACTION macro is used to ensure some level of alignment (if necessary)
void CpuSNN::compactConnections() {
	unsigned int* tmp_cumulativePost = new unsigned int[numN];
	unsigned int* tmp_cumulativePre  = new unsigned int[numN];
	unsigned int lastCnt_pre         = 0;
	unsigned int lastCnt_post        = 0;

	tmp_cumulativePost[0]   = 0;
	tmp_cumulativePre[0]    = 0;

	for(int i=1; i < numN; i++) {
		lastCnt_post = tmp_cumulativePost[i-1]+Npost[i-1]; //position of last pointer
		lastCnt_pre  = tmp_cumulativePre[i-1]+Npre[i-1]; //position of last pointer
		#if COMPACTION_ALIGNMENT_POST
			lastCnt_post= lastCnt_post + COMPACTION_ALIGNMENT_POST-lastCnt_post%COMPACTION_ALIGNMENT_POST;
			lastCnt_pre = lastCnt_pre  + COMPACTION_ALIGNMENT_PRE- lastCnt_pre%COMPACTION_ALIGNMENT_PRE;
		#endif
		tmp_cumulativePost[i] = lastCnt_post;
		tmp_cumulativePre[i]  = lastCnt_pre;
		assert(tmp_cumulativePost[i] <= cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= cumulativePre[i]);
	}

	// compress the post_synaptic array according to the new values of the tmp_cumulative counts....
	unsigned int tmp_postSynCnt = tmp_cumulativePost[numN-1]+Npost[numN-1];
	unsigned int tmp_preSynCnt  = tmp_cumulativePre[numN-1]+Npre[numN-1];
	assert(tmp_postSynCnt <= allocatedPost);
	assert(tmp_preSynCnt  <= allocatedPre);
	assert(tmp_postSynCnt <= postSynCnt);
	assert(tmp_preSynCnt  <= preSynCnt);
	CARLSIM_DEBUG("******************");
	CARLSIM_DEBUG("CompactConnection: ");
	CARLSIM_DEBUG("******************");
	CARLSIM_DEBUG("old_postCnt = %d, new_postCnt = %d", postSynCnt, tmp_postSynCnt);
	CARLSIM_DEBUG("old_preCnt = %d,  new_postCnt = %d", preSynCnt,  tmp_preSynCnt);

	// new buffer with required size + 100 bytes of additional space just to provide limited overflow
	post_info_t* tmp_postSynapticIds   = new post_info_t[tmp_postSynCnt+100];

	// new buffer with required size + 100 bytes of additional space just to provide limited overflow
	post_info_t* tmp_preSynapticIds	= new post_info_t[tmp_preSynCnt+100];
	float* tmp_wt	    	  		= new float[tmp_preSynCnt+100];
	float* tmp_maxSynWt   	  		= new float[tmp_preSynCnt+100];
	short int *tmp_cumConnIdPre 		= new short int[tmp_preSynCnt+100];
	float *tmp_mulSynFast 			= new float[numConnections];
	float *tmp_mulSynSlow  			= new float[numConnections];

	// compact synaptic information
	for(int i=0; i<numN; i++) {
		assert(tmp_cumulativePost[i] <= cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= cumulativePre[i]);
		for( int j=0; j<Npost[i]; j++) {
			unsigned int tmpPos = tmp_cumulativePost[i]+j;
			unsigned int oldPos = cumulativePost[i]+j;
			tmp_postSynapticIds[tmpPos] = postSynapticIds[oldPos];
			tmp_SynapticDelay[tmpPos]   = tmp_SynapticDelay[oldPos];
		}
		for( int j=0; j<Npre[i]; j++) {
			unsigned int tmpPos =  tmp_cumulativePre[i]+j;
			unsigned int oldPos =  cumulativePre[i]+j;
			tmp_preSynapticIds[tmpPos]  = preSynapticIds[oldPos];
			tmp_maxSynWt[tmpPos] 	    = maxSynWt[oldPos];
			tmp_wt[tmpPos]              = wt[oldPos];
			tmp_cumConnIdPre[tmpPos]	= cumConnIdPre[oldPos];
		}
	}

	// delete old buffer space
	delete[] postSynapticIds;
	postSynapticIds = tmp_postSynapticIds;
	cpuSnnSz.networkInfoSize -= (sizeof(post_info_t)*postSynCnt);
	cpuSnnSz.networkInfoSize += (sizeof(post_info_t)*(tmp_postSynCnt+100));

	delete[] cumulativePost;
	cumulativePost  = tmp_cumulativePost;

	delete[] cumulativePre;
	cumulativePre   = tmp_cumulativePre;

	delete[] maxSynWt;
	maxSynWt = tmp_maxSynWt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] wt;
	wt = tmp_wt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] cumConnIdPre;
	cumConnIdPre = tmp_cumConnIdPre;
	cpuSnnSz.synapticInfoSize -= (sizeof(short int)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(short int)*(tmp_preSynCnt+100));

	// compact connection-centric information
	for (int i=0; i<numConnections; i++) {
		tmp_mulSynFast[i] = mulSynFast[i];
		tmp_mulSynSlow[i] = mulSynSlow[i];
	}
	delete[] mulSynFast;
	delete[] mulSynSlow;
	mulSynFast = tmp_mulSynFast;
	mulSynSlow = tmp_mulSynSlow;
	cpuSnnSz.networkInfoSize -= (2*sizeof(uint8_t)*preSynCnt);
	cpuSnnSz.networkInfoSize += (2*sizeof(uint8_t)*(tmp_preSynCnt+100));


	delete[] preSynapticIds;
	preSynapticIds  = tmp_preSynapticIds;
	cpuSnnSz.synapticInfoSize -= (sizeof(post_info_t)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(post_info_t)*(tmp_preSynCnt+100));

	preSynCnt	= tmp_preSynCnt;
	postSynCnt	= tmp_postSynCnt;
}


// make 'C' full connections from grpSrc to grpDest
void CpuSNN::connectFull(grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	bool noDirect = (info->type == CONN_FULL_NO_DIRECT);

	for(int nid = grp_Info[grpSrc].StartN; nid <= grp_Info[grpSrc].EndN; nid++)  {
		for(int j = grp_Info[grpDest].StartN; j <= grp_Info[grpDest].EndN; j++) { // j: the temp neuron id
			if((noDirect) && (nid - grp_Info[grpSrc].StartN) == (j - grp_Info[grpDest].StartN))
				continue;
			uint8_t dVal = info->minDelay + (int)(0.5 + (getRandClosed() * (info->maxDelay - info->minDelay)));
			assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
			float synWt = getWeights(info->connProp, info->initWt, info->maxWt, nid, grpSrc);

			setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp, info->connId);
			info->numberOfConnections++;
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}

void CpuSNN::connectOneToOne (grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	assert( grp_Info[grpDest].SizeN == grp_Info[grpSrc].SizeN );

	for(int nid=grp_Info[grpSrc].StartN,j=grp_Info[grpDest].StartN; nid<=grp_Info[grpSrc].EndN; nid++, j++)  {
		uint8_t dVal = info->minDelay + (int)(0.5+(getRandClosed()*(info->maxDelay-info->minDelay)));
		assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
		float synWt = getWeights(info->connProp, info->initWt, info->maxWt, nid, grpSrc);
		setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp, info->connId);
		info->numberOfConnections++;
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;

}

// make 'C' random connections from grpSrc to grpDest
void CpuSNN::connectRandom (grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	for(int pre_nid=grp_Info[grpSrc].StartN; pre_nid<=grp_Info[grpSrc].EndN; pre_nid++) {
		for(int post_nid=grp_Info[grpDest].StartN; post_nid<=grp_Info[grpDest].EndN; post_nid++) {
			if (getRand() < info->p) {
				uint8_t dVal = info->minDelay + (int)(0.5+(getRandClosed()*(info->maxDelay-info->minDelay)));
				assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
				float synWt = getWeights(info->connProp, info->initWt, info->maxWt, pre_nid, grpSrc);
				setConnection(grpSrc, grpDest, pre_nid, post_nid, synWt, info->maxWt, dVal, info->connProp, info->connId);
				info->numberOfConnections++;
			}
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}

// user-defined functions called here...
// This is where we define our user-defined call-back function.  -- KDC
void CpuSNN::connectUserDefined (grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	info->maxDelay = 0;
	for(int nid=grp_Info[grpSrc].StartN; nid<=grp_Info[grpSrc].EndN; nid++) {
		for(int nid2=grp_Info[grpDest].StartN; nid2 <= grp_Info[grpDest].EndN; nid2++) {
			int srcId  = nid  - grp_Info[grpSrc].StartN;
			int destId = nid2 - grp_Info[grpDest].StartN;
			float weight, maxWt, delay;
			bool connected;

			info->conn->connect(this, grpSrc, srcId, grpDest, destId, weight, maxWt, delay, connected);
			if(connected)  {
				if (GET_FIXED_PLASTIC(info->connProp) == SYN_FIXED)
					maxWt = weight;

				assert(delay>=1);
				assert(delay<=MAX_SynapticDelay);
				assert(weight<=maxWt);

				setConnection(grpSrc, grpDest, nid, nid2, weight, maxWt, delay, info->connProp, info->connId);
				info->numberOfConnections++;
				if(delay > info->maxDelay)
				info->maxDelay = delay;
			}
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}


// delete all objects (CPU and GPU side)
void CpuSNN::deleteObjects() {
	if (simulatorDeleted)
		return;

	printSimSummary();

		// don't fclose if it's stdout or stderr, otherwise they're gonna stay closed for the rest of the process
	if (fpInf_!=NULL && fpInf_!=stdout && fpInf_!=stderr)
		fclose(fpInf_);
	if (fpErr_!=NULL && fpErr_!=stdout && fpErr_!=stderr)
		fclose(fpErr_);
	if (fpDeb_!=NULL && fpDeb_!=stdout && fpDeb_!=stderr)
		fclose(fpDeb_);
	if (fpLog_!=NULL && fpLog_!=stdout && fpLog_!=stderr)
		fclose(fpLog_);

	resetPointers(true); // deallocate pointers
		
	// do the same as above, but for snn_gpu.cu
	deleteObjects_GPU();
	simulatorDeleted = true;
}



// This method loops through all spikes that are generated by neurons with a delay of 1ms
// and delivers the spikes to the appropriate post-synaptic neuron
void CpuSNN::doD1CurrentUpdate() {
	int k     = secD1fireCntHost-1;
	int k_end = timeTableD1[simTimeMs+maxDelay_];

	while((k>=k_end) && (k>=0)) {

		int neuron_id      = firingTableD1[k];
		assert(neuron_id<numN);

		delay_info_t dPar = postDelayInfo[neuron_id*(maxDelay_+1)];

		unsigned int  offset = cumulativePost[neuron_id];

		for(int idx_d = dPar.delay_index_start;
			idx_d < (dPar.delay_index_start + dPar.delay_length);
			idx_d = idx_d+1) {
				generatePostSpike( neuron_id, idx_d, offset, 0);
		}
		k=k-1;
	}
}

// This method loops through all spikes that are generated by neurons with a delay of 2+ms
// and delivers the spikes to the appropriate post-synaptic neuron
void CpuSNN::doD2CurrentUpdate() {
	int k = secD2fireCntHost-1;
	int k_end = timeTableD2[simTimeMs+1];
	int t_pos = simTimeMs;

	while((k>=k_end)&& (k >=0)) {

		// get the neuron id from the index k
		int i  = firingTableD2[k];

		// find the time of firing from the timeTable using index k
		while (!((k >= timeTableD2[t_pos+maxDelay_])&&(k < timeTableD2[t_pos+maxDelay_+1]))) {
			t_pos = t_pos - 1;
			assert((t_pos+maxDelay_-1)>=0);
		}

		// \TODO: Instead of using the complex timeTable, can neuronFiringTime value...???
		// Calculate the time difference between time of firing of neuron and the current time...
		int tD = simTimeMs - t_pos;

		assert((tD<maxDelay_)&&(tD>=0));
		assert(i<numN);

		delay_info_t dPar = postDelayInfo[i*(maxDelay_+1)+tD];

		unsigned int offset = cumulativePost[i];

		// for each delay variables
		for(int idx_d = dPar.delay_index_start;
			idx_d < (dPar.delay_index_start + dPar.delay_length);
			idx_d = idx_d+1) {
			generatePostSpike( i, idx_d, offset, tD);
		}

		k=k-1;
	}
}

void CpuSNN::doSnnSim() {
	// for all Spike Counters, reset their spike counts to zero if simTime % recordDur == 0
	if (sim_with_spikecounters) {
		checkSpikeCounterRecordDur();
	}

	// decay STP vars and conductances
	doSTPUpdateAndDecayCond();

	updateSpikeGenerators();

	//generate all the scheduled spikes from the spikeBuffer..
	generateSpikes();

	// find the neurons that has fired..
	findFiring();

	timeTableD2[simTimeMs+maxDelay_+1] = secD2fireCntHost;
	timeTableD1[simTimeMs+maxDelay_+1] = secD1fireCntHost;

	doD2CurrentUpdate();
	doD1CurrentUpdate();
	globalStateUpdate();

	return;
}

void CpuSNN::doSTPUpdateAndDecayCond() {
	int spikeBufferFull = 0;

	//decay the STP variables before adding new spikes.
	for(int g=0; (g < numGrp) & !spikeBufferFull; g++) {
		for(int i=grp_Info[g].StartN; i<=grp_Info[g].EndN; i++) {
	   		//decay the STP variables before adding new spikes.
			if (grp_Info[g].WithSTP) {
				int ind_plus  = STP_BUF_POS(i,simTime);
				int ind_minus = STP_BUF_POS(i,(simTime-1));
				stpu[ind_plus] = stpu[ind_minus]*(1.0-grp_Info[g].STP_tau_u_inv);
				stpx[ind_plus] = stpx[ind_minus] + (1.0-stpx[ind_minus])*grp_Info[g].STP_tau_x_inv;
			}

			if (grp_Info[g].Type&POISSON_NEURON)
				continue;

			// decay conductances
			if (sim_with_conductances) {
				gAMPA[i]  *= dAMPA;
				gGABAa[i] *= dGABAa;

				if (sim_with_NMDA_rise) {
					gNMDA_r[i] *= rNMDA;	// rise
					gNMDA_d[i] *= dNMDA;	// decay
				} else {
					gNMDA[i]   *= dNMDA;	// instantaneous rise
				}

				if (sim_with_GABAb_rise) {
					gGABAb_r[i] *= rGABAb;	// rise
					gGABAb_d[i] *= dGABAb;	// decay
				} else {
					gGABAb[i] *= dGABAb;	// instantaneous rise
				}
			}
			else {
				current[i] = 0.0f; // in CUBA mode, reset current to 0 at each time step and sum up all wts
			}
		}
	}
}


void CpuSNN::findFiring() {
	int spikeBufferFull = 0;

	for(int g=0; (g < numGrp) & !spikeBufferFull; g++) {
		// given group of neurons belong to the poisson group....
		if (grp_Info[g].Type&POISSON_NEURON)
			continue;

		// his flag is set if with_stdp is set and also grpType is set to have GROUP_SYN_FIXED
		for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {

			assert(i < numNReg);

			if (voltage[i] >= 30.0) {
				voltage[i] = Izh_c[i];
				recovery[i] += Izh_d[i];

				// if flag hasSpkMonRT is set, we want to keep track of how many spikes per neuron in the group
				if (grp_Info[g].withSpikeCounter) {
					int bufPos = grp_Info[g].spkCntBufPos; // retrieve buf pos
					int bufNeur = i-grp_Info[g].StartN;
					spkCntBuf[bufPos][bufNeur]++;
//					printf("%d: %s[%d], nid=%d, %u spikes\n",simTimeMs,grp_Info2[g].Name.c_str(),g,i,spkMonRTbuf[bufPos][buffNeur]);
				}
				spikeBufferFull = addSpikeToTable(i, g);

				if (spikeBufferFull)
					break;

				if (grp_Info[g].WithSTDP) {
					unsigned int pos_ij = cumulativePre[i];
					for(int j=0; j < Npre_plastic[i]; pos_ij++, j++) {
						int stdp_tDiff = (simTime-synSpikeTime[pos_ij]);
						assert(!((stdp_tDiff < 0) && (synSpikeTime[pos_ij] != MAX_SIMULATION_TIME)));

						if (stdp_tDiff > 0) {
							#ifdef INHIBITORY_STDP
							// if this is an excitatory or inhibitory synapse
							if (maxSynWt[pos_ij] >= 0)
							#endif
							if ((stdp_tDiff*grp_Info[g].TAU_LTP_INV)<25)
								wtChange[pos_ij] += STDP(stdp_tDiff, grp_Info[g].ALPHA_LTP, grp_Info[g].TAU_LTP_INV);
							#ifdef INHIBITORY_STDP
							else
								if ((stdp_tDiff > 0) && ((stdp_tDiff*grp_Info[g].TAU_LTD_INV)<25)) {
									wtChange[pos_ij] -= (STDP(stdp_tDiff, grp_Info[g].ALPHA_LTP, grp_Info[g].TAU_LTP_INV)
										- STDP(stdp_tDiff, grp_Info[g].ALPHA_LTD*1.5, grp_Info[g].TAU_LTD_INV));
								}
							#endif
						}
					}
				}
				spikeCountAll1sec++;
			}
		}
	}
}

int CpuSNN::findGrpId(int nid) {
	CARLSIM_WARN("Using findGrpId is deprecated, use array grpIds[] instead...");
	for(int g=0; g < numGrp; g++) {
		if(nid >=grp_Info[g].StartN && (nid <=grp_Info[g].EndN)) {
			return g;
		}
	}
	CARLSIM_ERROR("findGrp(): cannot find the group for neuron %d", nid);
	exitSimulation(1);
}


void CpuSNN::generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD) {
	// get synaptic info...
	post_info_t post_info = postSynapticIds[offset + idx_d];

	// get post-neuron id
	unsigned int post_i = GET_CONN_NEURON_ID(post_info);
	assert(post_i<numN);

	// get syn id
	int s_i = GET_CONN_SYN_ID(post_info);
	assert(s_i<(Npre[post_i]));

	// get the cumulative position for quick access
	unsigned int pos_i = cumulativePre[post_i] + s_i;
	assert(post_i < numNReg); // \FIXME is this assert supposed to be for pos_i?

	// get group id of pre- / post-neuron
	short int post_grpId = grpIds[post_i];
	short int pre_grpId = grpIds[pre_i];

	unsigned int pre_type = grp_Info[pre_grpId].Type;

	// get connect info from the cumulative synapse index for mulSynFast/mulSynSlow (requires less memory than storing
	// mulSynFast/Slow per synapse or storing a pointer to grpConnectInfo_s)
	// mulSynFast will be applied to fast currents (either AMPA or GABAa)
	// mulSynSlow will be applied to slow currents (either NMDA or GABAb)
	short int mulIndex = cumConnIdPre[pos_i];
	assert(mulIndex>=0 && mulIndex<numConnections);


	// for each presynaptic spike, postsynaptic (synaptic) current is going to increase by some amplitude (change)
	// generally speaking, this amplitude is the weight; but it can be modulated by STP
	float change = wt[pos_i];

	if (grp_Info[pre_grpId].WithSTP) {
		// if pre-group has STP enabled, we need to modulate the weight
		// NOTE: Order is important! (Tsodyks & Markram, 1998; Mongillo, Barak, & Tsodyks, 2008)
		// use u^+ (value right after spike-update) but x^- (value right before spike-update)

		// dI/dt = -I/tau_S + A * u^+ * x^- * \delta(t-t_{spk})
		// I noticed that for connect(.., RangeDelay(1), ..) tD will be 0
		int ind_minus = STP_BUF_POS(pre_i,(simTime-tD-1));
		int ind_plus  = STP_BUF_POS(pre_i,(simTime-tD));

		change *= grp_Info[pre_grpId].STP_A*stpu[ind_plus]*stpx[ind_minus];

//		fprintf(stderr,"%d: %d[%d], numN=%d, td=%d, maxDelay_=%d, ind-=%d, ind+=%d, stpu=[%f,%f], stpx=[%f,%f], change=%f, wt=%f\n", 
//			simTime, pre_grpId, pre_i,
//					numN, tD, maxDelay_, ind_minus, ind_plus,
//					stpu[ind_minus], stpu[ind_plus], stpx[ind_minus], stpx[ind_plus], change, wt[pos_i]);
	}

	// update currents
	// NOTE: it's faster to += 0.0 rather than checking for zero and not updating
	if (sim_with_conductances) {
		if (pre_type & TARGET_AMPA) // if post_i expresses AMPAR
			gAMPA [post_i] += change*mulSynFast[mulIndex]; // scale by some factor
		if (pre_type & TARGET_NMDA) {
			if (sim_with_NMDA_rise) {
				gNMDA_r[post_i] += change*sNMDA*mulSynSlow[mulIndex];
				gNMDA_d[post_i] += change*sNMDA*mulSynSlow[mulIndex];
			} else {
				gNMDA [post_i] += change*mulSynSlow[mulIndex];
			}
		}
		if (pre_type & TARGET_GABAa)
			gGABAa[post_i] -= change*mulSynFast[mulIndex]; // wt should be negative for GABAa and GABAb
		if (pre_type & TARGET_GABAb) {
			if (sim_with_GABAb_rise) {
				gGABAb_r[post_i] -= change*sGABAb*mulSynSlow[mulIndex];
				gGABAb_d[post_i] -= change*sGABAb*mulSynSlow[mulIndex];
			} else {
				gGABAb[post_i] -= change*mulSynSlow[mulIndex];
			}
		}
	} else {
		current[post_i] += change;
	}

	synSpikeTime[pos_i] = simTime;

	// Got one spike from dopaminergic neuron, increase dopamine concentration in the target area
	if (pre_type & TARGET_DA) {
		cpuNetPtrs.grpDA[post_grpId] += 0.02;
	}

	// STDP calculation....
	if (grp_Info[post_grpId].WithSTDP) {
		int stdp_tDiff = (simTime-lastSpikeTime[post_i]);

		if (stdp_tDiff >= 0) {
			#ifdef INHIBITORY_STDP
			if ((pre_type & TARGET_GABAa) || (pre_type & TARGET_GABAb))
			{
				if ((stdp_tDiff*grp_Info[post_grpId].TAU_LTD_INV)<25)
					wtChange[pos_i] -= (STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTP, grp_Info[post_grpId].TAU_LTP_INV)
			    					 - STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTD*1.5, grp_Info[post_grpId].TAU_LTD_INV));
				}
				else
			#endif
			{
				if ((stdp_tDiff*grp_Info[post_grpId].TAU_LTD_INV)<25)
					wtChange[pos_i] -= STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTD, grp_Info[post_grpId].TAU_LTD_INV);
			}
		}
		assert(!((stdp_tDiff < 0) && (lastSpikeTime[post_i] != MAX_SIMULATION_TIME)));
	}
}

void CpuSNN::generateSpikes() {
	PropagatedSpikeBuffer::const_iterator srg_iter;
	PropagatedSpikeBuffer::const_iterator srg_iter_end = pbuf->endSpikeTargetGroups();

	for( srg_iter = pbuf->beginSpikeTargetGroups(); srg_iter != srg_iter_end; ++srg_iter )  {
		// Get the target neurons for the given groupId
		int nid	 = srg_iter->stg;
		//delaystep_t del = srg_iter->delay;
		//generate a spike to all the target neurons from source neuron nid with a delay of del
		short int g = grpIds[nid];

/*
// MB: Uncomment this if you want to activate real-time spike monitors for SpikeGenerators
// However, the GPU version of this is not implemented... Need to implement it for the case 1) GPU mode
// and generators on CPU side, 2) GPU mode and generators on GPU side
			// if flag hasSpkCnt is set, we want to keep track of how many spikes per neuron in the group
			if (grp_Info[g].withSpikeCounter) {
				int bufPos = grp_Info[g].spkCntBufPos; // retrieve buf pos
				int bufNeur = nid-grp_Info[g].StartN;
				spkCntBuf[bufPos][bufNeur]++;
				printf("%d: %s[%d], nid=%d, %u spikes\n",simTimeMs,grp_Info2[g].Name.c_str(),g,nid,spkCntBuf[bufPos][bufNeur]);
			}
*/
		addSpikeToTable (nid, g);
		spikeCountAll1sec++;
		nPoissonSpikes++;
	}

	// advance the time step to the next phase...
	pbuf->nextTimeStep();
}

void CpuSNN::generateSpikesFromFuncPtr(int grpId) {
	bool done;
	SpikeGeneratorCore* spikeGen = grp_Info[grpId].spikeGen;
	int timeSlice = grp_Info[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;
	for(int i = grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = lastSpikeTime[i];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		// the end of the valid time window is either the length of the scheduling time slice from now (because that
		// is the max of the allowed propagated buffer size) or simply the end of the simulation
		unsigned int endOfTimeWindow = MIN(currTime+timeSlice,simTimeRunStop);

		done = false;
		while (!done) {

			nextTime = spikeGen->nextSpikeTime(this, grpId, i - grp_Info[grpId].StartN, currTime, nextTime);

			// found a valid time window
			if (nextTime < endOfTimeWindow) {
				if (nextTime >= currTime) {
//					fprintf(stderr,"%u: spike scheduled for %d at %u\n",currTime, i-grp_Info[grpId].StartN,nextTime);
					// scheduled spike...
					// \TODO CPU mode does not check whether the same AER event has been scheduled before (bug #212)
					// check how GPU mode does it, then do the same here.
					pbuf->scheduleSpikeTargetGroup(i, nextTime - currTime);
					spikeCnt++;
				}
			} else {
				done = true;
			}
		}
	}
}

void CpuSNN::generateSpikesFromRate(int grpId) {
	bool done;
	PoissonRate* rate = grp_Info[grpId].RatePtr;
	float refPeriod = grp_Info[grpId].RefractPeriod;
	int timeSlice   = grp_Info[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;

	if (rate == NULL) return;

	if (rate->onGPU) {
		CARLSIM_ERROR("Specifying rates on the GPU but using the CPU SNN is not supported.");
		exitSimulation(1);
	}

	const float* ptr = rate->rates;
	for (int cnt=0;cnt<rate->len;cnt++,ptr++) {
		float frate = *ptr;

		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = lastSpikeTime[grp_Info[grpId].StartN+cnt];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		done = false;
		while (!done && frate>0) {
			nextTime = poissonSpike(nextTime, frate/1000.0, refPeriod);
			// found a valid timeSlice
			if (nextTime < (currTime+timeSlice)) {
				if (nextTime >= currTime) {
					int nid = grp_Info[grpId].StartN+cnt;
					pbuf->scheduleSpikeTargetGroup(nid, nextTime-currTime);
					spikeCnt++;
				}
			}
			else {
				done=true;
			}
		}
	}
}


// initialize all the synaptic weights to appropriate values..
// total size of the synaptic connection is 'length' ...
void CpuSNN::initSynapticWeights() {
	// Initialize the network wtChange, wt, synaptic firing time
	wtChange         = new float[preSynCnt];
	synSpikeTime     = new uint32_t[preSynCnt];
	cpuSnnSz.synapticInfoSize = sizeof(float)*(preSynCnt*2);

	resetSynapticConnections(false);
}


inline int CpuSNN::getPoissNeuronPos(int nid) {
	int nPos = nid-numNReg;
	assert(nid >= numNReg);
	assert(nid < numN);
	assert((nid-numNReg) < numNPois);
	return nPos;
}

//We need pass the neuron id (nid) and the grpId just for the case when we want to
//ramp up/down the weights.  In that case we need to set the weights of each synapse
//depending on their nid (their position with respect to one another). -- KDC
float CpuSNN::getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId) {
	float actWts;
	// \FIXME: are these ramping thingies still supported?
	bool setRandomWeights   = GET_INITWTS_RANDOM(connProp);
	bool setRampDownWeights = GET_INITWTS_RAMPDOWN(connProp);
	bool setRampUpWeights   = GET_INITWTS_RAMPUP(connProp);

	if (setRandomWeights)
		actWts = initWt * drand48();
	else if (setRampUpWeights)
		actWts = (initWt + ((nid - grp_Info[grpId].StartN) * (maxWt - initWt) / grp_Info[grpId].SizeN));
	else if (setRampDownWeights)
		actWts = (maxWt - ((nid - grp_Info[grpId].StartN) * (maxWt - initWt) / grp_Info[grpId].SizeN));
	else
		actWts = initWt;

	return actWts;
}


void  CpuSNN::globalStateUpdate() {
	double tmp_iNMDA, tmp_I;
	double tmp_gNMDA, tmp_gGABAb;

	for(int g=0; g < numGrp; g++) {
		if (grp_Info[g].Type&POISSON_NEURON) {
			if (grp_Info[g].WithHomeostasis) {
				for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++)
					avgFiring[i] *= grp_Info[g].avgTimeScale_decay;
			}
			continue;
		}

		// decay dopamine concentration
		if (cpuNetPtrs.grpDA[g] > grp_Info[g].baseDP)
			cpuNetPtrs.grpDA[g] *= grp_Info[g].decayDP;

		for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
			assert(i < numNReg);
			if (grp_Info[g].WithHomeostasis)
				avgFiring[i] *= grp_Info[g].avgTimeScale_decay;

			if (sim_with_conductances) {
				// COBA model

				// all the tmpIs will be summed into current[i] in the following loop
				current[i] = 0.0f;

				// \FIXME: these tmp vars cause a lot of rounding errors... consider rewriting
				for (int j=0; j<COND_INTEGRATION_SCALE; j++) {
					tmp_iNMDA = (voltage[i]+80.0)*(voltage[i]+80.0)/60.0/60.0;

					tmp_gNMDA = sim_with_NMDA_rise ? gNMDA_d[i]-gNMDA_r[i] : gNMDA[i];
					tmp_gGABAb = sim_with_GABAb_rise ? gGABAb_d[i]-gGABAb_r[i] : gGABAb[i];

					tmp_I = -(   gAMPA[i]*(voltage[i]-0)
									 + tmp_gNMDA*tmp_iNMDA/(1+tmp_iNMDA)*(voltage[i]-0)
									 + gGABAa[i]*(voltage[i]+70)
									 + tmp_gGABAb*(voltage[i]+90)
								   );

					#ifdef NEURON_NOISE
						double noiseI = -intrinsicWeight[i]*log(getRand());
						if (isnan(noiseI) || isinf(noiseI))
							noiseI = 0;
						tmp_I += noiseI;
					#endif

					voltage[i]+=((0.04*voltage[i]+5.0)*voltage[i]+140.0-recovery[i]+tmp_I)/COND_INTEGRATION_SCALE;
					assert(!isnan(voltage[i]) && !isinf(voltage[i]));

					// keep track of total current
					current[i] += tmp_I;

					if (voltage[i] > 30) {
						voltage[i] = 30;
						j=COND_INTEGRATION_SCALE; // break the loop but evaluate u[i]
//						if (gNMDA[i]>=10.0f) CARLSIM_WARN("High NMDA conductance (gNMDA>=10.0) may cause instability");
//						if (gGABAb[i]>=2.0f) CARLSIM_WARN("High GABAb conductance (gGABAb>=2.0) may cause instability");
					}
					if (voltage[i] < -90)
						voltage[i] = -90;
					recovery[i]+=Izh_a[i]*(Izh_b[i]*voltage[i]-recovery[i])/COND_INTEGRATION_SCALE;
				} // end COND_INTEGRATION_SCALE loop
			} else {
				// CUBA model
				voltage[i]+=0.5*((0.04*voltage[i]+5.0)*voltage[i]+140.0-recovery[i]+current[i]); //for numerical stability
				voltage[i]+=0.5*((0.04*voltage[i]+5.0)*voltage[i]+140.0-recovery[i]+current[i]); //time step is 0.5 ms
				if (voltage[i] > 30)
					voltage[i] = 30;
				if (voltage[i] < -90)
					voltage[i] = -90;
				recovery[i]+=Izh_a[i]*(Izh_b[i]*voltage[i]-recovery[i]);
			} // end COBA/CUBA
		} // end StartN...EndN
	} // end numGrp
}


// creates the CPU net pointers
// don't forget to cudaFree the device pointers if you make cpu_gpuNetPtrs
void CpuSNN::makePtrInfo() {
	cpuNetPtrs.voltage			= voltage;
	cpuNetPtrs.recovery			= recovery;
	cpuNetPtrs.current			= current;
	cpuNetPtrs.Npre				= Npre;
	cpuNetPtrs.Npost			= Npost;
	cpuNetPtrs.cumulativePost 	= cumulativePost;
	cpuNetPtrs.cumulativePre  	= cumulativePre;
	cpuNetPtrs.synSpikeTime		= synSpikeTime;
	cpuNetPtrs.wt				= wt;
	cpuNetPtrs.wtChange			= wtChange;
	cpuNetPtrs.mulSynFast 		= mulSynFast;
	cpuNetPtrs.mulSynSlow 		= mulSynSlow;
	cpuNetPtrs.cumConnIdPre 	= cumConnIdPre;
	cpuNetPtrs.nSpikeCnt		= nSpikeCnt;
	cpuNetPtrs.curSpike 		= curSpike;
	cpuNetPtrs.firingTableD2 	= firingTableD2;
	cpuNetPtrs.firingTableD1 	= firingTableD1;
	cpuNetPtrs.grpIds 			= grpIds;

	// homeostasis variables
	cpuNetPtrs.avgFiring    	= avgFiring;
	cpuNetPtrs.baseFiring   	= baseFiring;

	cpuNetPtrs.gAMPA        	= gAMPA;
	cpuNetPtrs.gGABAa       	= gGABAa;
	if (sim_with_NMDA_rise) {
		cpuNetPtrs.gNMDA 		= NULL;
		cpuNetPtrs.gNMDA_r		= gNMDA_r;
		cpuNetPtrs.gNMDA_d		= gNMDA_d;
	} else {
		cpuNetPtrs.gNMDA		= gNMDA;
		cpuNetPtrs.gNMDA_r 		= NULL;
		cpuNetPtrs.gNMDA_d 		= NULL;
	}
	if (sim_with_GABAb_rise) {
		cpuNetPtrs.gGABAb		= NULL;
		cpuNetPtrs.gGABAb_r		= gGABAb_r;
		cpuNetPtrs.gGABAb_d		= gGABAb_d;
	} else {
		cpuNetPtrs.gGABAb		= gGABAb;
		cpuNetPtrs.gGABAb_r 	= NULL;
		cpuNetPtrs.gGABAb_d 	= NULL;
	}
	cpuNetPtrs.grpDA			= grpDA;
	cpuNetPtrs.grp5HT			= grp5HT;
	cpuNetPtrs.grpACh			= grpACh;
	cpuNetPtrs.grpNE			= grpNE;
	cpuNetPtrs.allocated    	= true;
	cpuNetPtrs.memType      	= CPU_MODE;
	cpuNetPtrs.stpu 			= stpu;
	cpuNetPtrs.stpx				= stpx;
}

// will be used in generateSpikesFromRate
unsigned int CpuSNN::poissonSpike(unsigned int currTime, float frate, int refractPeriod) {
	bool done = false;
	unsigned int nextTime = 0;

	// refractory period must be 1 or greater, 0 means could have multiple spikes specified at the same time.
	assert(refractPeriod>0);
	static int cnt = 0;
	while(!done) {
		float randVal = drand48();
		unsigned int tmpVal  = -log(randVal)/frate;
		nextTime = currTime + tmpVal;
		if ((nextTime - currTime) >= (unsigned) refractPeriod)
			done = true;
	}

	assert(nextTime != 0);
	return nextTime;
}

// \FIXME: this guy is a mess
#if READNETWORK_ADD_SYNAPSES_FROM_FILE
int CpuSNN::readNetwork_internal(bool onlyPlastic)
#else
int CpuSNN::readNetwork_internal()
#endif
{
	long file_position = ftell(readNetworkFID); // so that we can restore the file position later...
	int tmpInt;
	float tmpFloat;

	// read file signature
	if (!fread(&tmpInt,sizeof(int),1,readNetworkFID)) return -11;
	if (tmpInt != 294338571) return -10;

	// read version number
	if (!fread(&tmpFloat,sizeof(float),1,readNetworkFID)) return -11;
	if (tmpFloat > 1.0) return -10;

	// read simulation and execution time
	if (!fread(&tmpFloat,sizeof(float),2,readNetworkFID)) return -11;

	// read number of neurons
	if (!fread(&tmpInt,sizeof(int),1,readNetworkFID)) return -11;
	int nrCells = tmpInt;
	if (nrCells != numN) return -5;

	// read total synapse counts
	if (!fread(&tmpInt,sizeof(int),2,readNetworkFID)) return -11;

	// read number of groups
	if (!fread(&tmpInt,sizeof(int),1,readNetworkFID)) return -11;
	if (numGrp != tmpInt) return -1;

	char name[100];
	int startN, endN;

	for (int g=0;g<numGrp;g++) {
		if (!fread(&startN,sizeof(int),1,readNetworkFID)) return -11;
		if (!fread(&endN,sizeof(int),1,readNetworkFID)) return -11;
		if (startN != grp_Info[g].StartN) return -2;
		if (endN != grp_Info[g].EndN) return -3;
		if (!fread(name,1,100,readNetworkFID)) return -11;
		if (strcmp(name,grp_Info2[g].Name.c_str()) != 0) return -4;
	}

	for (unsigned int i=0;i<nrCells;i++) {
		unsigned int nrSynapses = 0;
		if (!fread(&nrSynapses,sizeof(int),1,readNetworkFID)) return -11;

		for (int j=0;j<nrSynapses;j++) {
			unsigned int nIDpre;
			unsigned int nIDpost;
			float weight, maxWeight;
			uint8_t delay;
			uint8_t plastic;
			short int connId;

			if (!fread(&nIDpre,sizeof(int),1,readNetworkFID)) return -11;
			if (nIDpre != i) return -6;
			if (!fread(&nIDpost,sizeof(int),1,readNetworkFID)) return -11;
			if (nIDpost >= nrCells) return -7;
			if (!fread(&weight,sizeof(float),1,readNetworkFID)) return -11;

			short int gIDpre = grpIds[nIDpre];
			if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight>0)
					|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight<0)) {
				return -8;
			}

			if (!fread(&maxWeight,sizeof(float),1,readNetworkFID)) return -11;
			if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight>=0)
					|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight<=0)) {
				return -8;
			}

			if (!fread(&delay,sizeof(uint8_t),1,readNetworkFID)) return -11;
			if (delay > MAX_SynapticDelay) return -9;
			if (!fread(&plastic,sizeof(uint8_t),1,readNetworkFID)) return -11;
			if (!fread(&connId,sizeof(short int),1,readNetworkFID)) return -11;

			#if READNETWORK_ADD_SYNAPSES_FROM_FILE
				if ((plastic && onlyPlastic) || (!plastic && !onlyPlastic)) {
					int gIDpost = grpIds[nIDpost];
					int connProp = SET_FIXED_PLASTIC(plastic?SYN_PLASTIC:SYN_FIXED);

					setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp, connId);
					grp_Info2[gIDpre].sumPostConn++;
					grp_Info2[gIDpost].sumPreConn++;

					if (delay > grp_Info[gIDpre].MaxDelay)
						grp_Info[gIDpre].MaxDelay = delay;
				}
			#endif
		}
	}
	#if READNETWORK_ADD_SYNAPSES_FROM_FILE
		fseek(readNetworkFID,file_position,SEEK_SET);
	#endif
	return 0;
}


// The post synaptic connections are sorted based on delay here so that we can reduce storage requirement
// and generation of spike at the post-synaptic side.
// We also create the delay_info array has the delay_start and delay_length parameter
void CpuSNN::reorganizeDelay()
{
	for(int grpId=0; grpId < numGrp; grpId++) {
		for(int nid=grp_Info[grpId].StartN; nid <= grp_Info[grpId].EndN; nid++) {
			unsigned int jPos=0;					// this points to the top of the delay queue
			unsigned int cumN=cumulativePost[nid];	// cumulativePost[] is unsigned int
			unsigned int cumDelayStart=0; 			// Npost[] is unsigned short
			for(int td = 0; td < maxDelay_; td++) {
				unsigned int j=jPos;				// start searching from top of the queue until the end
				unsigned int cnt=0;					// store the number of nodes with a delay of td;
				while(j < Npost[nid]) {
					// found a node j with delay=td and we put
					// the delay value = 1 at array location td=0;
					if(td==(tmp_SynapticDelay[cumN+j]-1)) {
						assert(jPos<Npost[nid]);
						swapConnections(nid, j, jPos);

						jPos=jPos+1;
						cnt=cnt+1;
					}
					j=j+1;
				}

				// update the delay_length and start values...
				postDelayInfo[nid*(maxDelay_+1)+td].delay_length	     = cnt;
				postDelayInfo[nid*(maxDelay_+1)+td].delay_index_start  = cumDelayStart;
				cumDelayStart += cnt;

				assert(cumDelayStart <= Npost[nid]);
			}

			// total cumulative delay should be equal to number of post-synaptic connections at the end of the loop
			assert(cumDelayStart == Npost[nid]);
			for(unsigned int j=1; j < Npost[nid]; j++) {
				unsigned int cumN=cumulativePost[nid]; // cumulativePost[] is unsigned int
				if( tmp_SynapticDelay[cumN+j] < tmp_SynapticDelay[cumN+j-1]) {
	  				CARLSIM_ERROR("Post-synaptic delays not sorted correctly... id=%d, delay[%d]=%d, delay[%d]=%d",
						nid, j, tmp_SynapticDelay[cumN+j], j-1, tmp_SynapticDelay[cumN+j-1]);
					assert( tmp_SynapticDelay[cumN+j] >= tmp_SynapticDelay[cumN+j-1]);
				}
			}
		}
	}
}

// after all the initalization. Its time to create the synaptic weights, weight change and also
// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
void CpuSNN::reorganizeNetwork(bool removeTempMemory) {
	//Double check...sometimes by mistake we might call reorganize network again...
	if(doneReorganization)
		return;

	CARLSIM_DEBUG("Beginning reorganization of network....");

	// time to build the complete network with relevant parameters..
	buildNetwork();

	//..minimize any other wastage in that array by compacting the store
	compactConnections();

	// The post synaptic connections are sorted based on delay here
	reorganizeDelay();

	// Print the statistics again but dump the results to a file
	printMemoryInfo(fpDeb_);

	// initialize the synaptic weights accordingly..
	initSynapticWeights();

	updateSpikeGeneratorsInit();

	//ensure that we dont do all the above optimizations again
	doneReorganization = true;

	// reset all spike cnt
	resetSpikeCnt(ALL);

	printTuningLog(fpDeb_);

	makePtrInfo();

	CARLSIM_INFO("");
	CARLSIM_INFO("*****************      Initializing %s Simulation      *************************",
		simMode_==GPU_MODE?"GPU":"CPU");

	if(removeTempMemory) {
		memoryOptimized = true;
		delete[] tmp_SynapticDelay;
		tmp_SynapticDelay = NULL;
	}
}


void CpuSNN::resetConductances() {
	if (sim_with_conductances) {
		memset(gAMPA, 0, sizeof(float)*numNReg);
		if (sim_with_NMDA_rise) {
			memset(gNMDA_r, 0, sizeof(float)*numNReg);
			memset(gNMDA_d, 0, sizeof(float)*numNReg);
		} else {
			memset(gNMDA, 0, sizeof(float)*numNReg);
		}
		memset(gGABAa, 0, sizeof(float)*numNReg);
		if (sim_with_GABAb_rise) {
			memset(gGABAb_r, 0, sizeof(float)*numNReg);
			memset(gGABAb_d, 0, sizeof(float)*numNReg);
		} else {
			memset(gGABAb, 0, sizeof(float)*numNReg);
		}
	}
}

void CpuSNN::resetCounters() {
	assert(numNReg <= numN);
	memset(curSpike, 0, sizeof(bool) * numN);
}

void CpuSNN::resetCPUTiming() {
	prevCpuExecutionTime = cumExecutionTime;
	cpuExecutionTime     = 0.0;
}

void CpuSNN::resetCurrent() {
	assert(current != NULL);
	memset(current, 0, sizeof(float) * numNReg);
}

void CpuSNN::resetFiringInformation() {
	// Reset firing tables and time tables to default values..

	// reset Various Times..
	spikeCountAll	  = 0;
	spikeCountAll1sec = 0;
	spikeCountD2Host = 0;
	spikeCountD1Host = 0;
	secD1fireCntHost  = 0;
	secD2fireCntHost  = 0;

	for(int i=0; i < numGrp; i++) {
		grp_Info[i].FiringCount1sec = 0;
	}

	// reset various times...
	simTimeMs  = 0;
	simTimeSec = 0;
	simTime    = 0;

	// reset the propogation Buffer.
	resetPropogationBuffer();
	// reset Timing  Table..
	resetTimingTable();
}

void CpuSNN::resetGPUTiming() {
	prevGpuExecutionTime = cumExecutionTime;
	gpuExecutionTime     = 0.0;
}

void CpuSNN::resetGroups() {
	for(int g=0; (g < numGrp); g++) {
		// reset spike generator group...
		if (grp_Info[g].isSpikeGenerator) {
			grp_Info[g].CurrTimeSlice = grp_Info[g].NewTimeSlice;
			grp_Info[g].SliceUpdateTime  = 0;
			for(int nid=grp_Info[g].StartN; nid <= grp_Info[g].EndN; nid++)
				resetPoissonNeuron(nid, g);
		}
		// reset regular neuron group...
		else {
			for(int nid=grp_Info[g].StartN; nid <= grp_Info[g].EndN; nid++)
				resetNeuron(nid, g);
		}
	}

	// reset the currents for each neuron
	resetCurrent();

	// reset the conductances...
	resetConductances();

	//  reset various counters in the group...
	resetCounters();
}

void CpuSNN::resetNeuromodulator(int grpId) {
	grpDA[grpId] = grp_Info[grpId].baseDP;
	grp5HT[grpId] = grp_Info[grpId].base5HT;
	grpACh[grpId] = grp_Info[grpId].baseACh;
	grpNE[grpId] = grp_Info[grpId].baseNE;
}

void CpuSNN::resetNeuron(unsigned int neurId, int grpId) {
	assert(neurId < numNReg);
    if (grp_Info2[grpId].Izh_a == -1) {
		CARLSIM_ERROR("setNeuronParameters must be called for group %s (%d)",grp_Info2[grpId].Name.c_str(),grpId);
		exitSimulation(1);
	}

	Izh_a[neurId] = grp_Info2[grpId].Izh_a + grp_Info2[grpId].Izh_a_sd*(float)getRandClosed();
	Izh_b[neurId] = grp_Info2[grpId].Izh_b + grp_Info2[grpId].Izh_b_sd*(float)getRandClosed();
	Izh_c[neurId] = grp_Info2[grpId].Izh_c + grp_Info2[grpId].Izh_c_sd*(float)getRandClosed();
	Izh_d[neurId] = grp_Info2[grpId].Izh_d + grp_Info2[grpId].Izh_d_sd*(float)getRandClosed();

	voltage[neurId] = Izh_c[neurId];	// initial values for new_v
	recovery[neurId] = Izh_b[neurId]*voltage[neurId]; // initial values for u


 	if (grp_Info[grpId].WithHomeostasis) {
		// set the baseFiring with some standard deviation.
		if(drand48()>0.5)   {
			baseFiring[neurId] = grp_Info2[grpId].baseFiring + grp_Info2[grpId].baseFiringSD*-log(drand48());
		} else  {
			baseFiring[neurId] = grp_Info2[grpId].baseFiring - grp_Info2[grpId].baseFiringSD*-log(drand48());
			if(baseFiring[neurId] < 0.1) baseFiring[neurId] = 0.1;
		}

		if( grp_Info2[grpId].baseFiring != 0.0) {
			avgFiring[neurId]  = baseFiring[neurId];
		} else {
			baseFiring[neurId] = 0.0;
			avgFiring[neurId]  = 0;
		}
	}

	lastSpikeTime[neurId]  = MAX_SIMULATION_TIME;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(neurId,j);
			stpu[ind] = 0.0f;
			stpx[ind] = 1.0f;
		}
	}
}

void CpuSNN::resetPointers(bool deallocate) {
	if (voltage!=NULL && deallocate) delete[] voltage;
	if (recovery!=NULL && deallocate) delete[] recovery;
	if (current!=NULL && deallocate) delete[] current;
	voltage=NULL; recovery=NULL; current=NULL;

	if (Izh_a!=NULL && deallocate) delete[] Izh_a;
	if (Izh_b!=NULL && deallocate) delete[] Izh_b;
	if (Izh_c!=NULL && deallocate) delete[] Izh_c;
	if (Izh_d!=NULL && deallocate) delete[] Izh_d;
	Izh_a=NULL; Izh_b=NULL; Izh_c=NULL; Izh_d=NULL;

	if (Npre!=NULL && deallocate) delete[] Npre;
	if (Npre_plastic!=NULL && deallocate) delete[] Npre_plastic;
	if (Npost!=NULL && deallocate) delete[] Npost;
	Npre=NULL; Npre_plastic=NULL; Npost=NULL;

	if (cumulativePre!=NULL && deallocate) delete[] cumulativePre;
	if (cumulativePost!=NULL && deallocate) delete[] cumulativePost;
	cumulativePre=NULL; cumulativePost=NULL;

	if (gAMPA!=NULL && deallocate) delete[] gAMPA;
	if (gNMDA!=NULL && deallocate) delete[] gNMDA;
	if (gNMDA_r!=NULL && deallocate) delete[] gNMDA_r;
	if (gNMDA_d!=NULL && deallocate) delete[] gNMDA_d;
	if (gGABAa!=NULL && deallocate) delete[] gGABAa;
	if (gGABAb!=NULL && deallocate) delete[] gGABAb;
	if (gGABAb_r!=NULL && deallocate) delete[] gGABAb_r;
	if (gGABAb_d!=NULL && deallocate) delete[] gGABAb_d;
	gAMPA=NULL; gNMDA=NULL; gNMDA_r=NULL; gNMDA_d=NULL; gGABAa=NULL; gGABAb=NULL; gGABAb_r=NULL; gGABAb_d=NULL;

	if (stpu!=NULL && deallocate) delete[] stpu;
	if (stpx!=NULL && deallocate) delete[] stpx;
	stpu=NULL; stpx=NULL;

	if (avgFiring!=NULL && deallocate) delete[] avgFiring;
	if (baseFiring!=NULL && deallocate) delete[] baseFiring;
	avgFiring=NULL; baseFiring=NULL;

	if (lastSpikeTime!=NULL && deallocate) delete[] lastSpikeTime;
	if (synSpikeTime !=NULL && deallocate) delete[] synSpikeTime;
	if (curSpike!=NULL && deallocate) delete[] curSpike;
	if (nSpikeCnt!=NULL && deallocate) delete[] nSpikeCnt;
	lastSpikeTime=NULL; synSpikeTime=NULL; curSpike=NULL; nSpikeCnt=NULL;

	if (postDelayInfo!=NULL && deallocate) delete[] postDelayInfo;
	if (preSynapticIds!=NULL && deallocate) delete[] preSynapticIds;
	if (postSynapticIds!=NULL && deallocate) delete[] postSynapticIds;
	postDelayInfo=NULL; preSynapticIds=NULL; postSynapticIds=NULL;

	if (wt!=NULL && deallocate) delete[] wt;
	if (maxSynWt!=NULL && deallocate) delete[] maxSynWt;
	if (wtChange !=NULL && deallocate) delete[] wtChange;
	wt=NULL; maxSynWt=NULL; wtChange=NULL;

	if (mulSynFast!=NULL && deallocate) delete[] mulSynFast;
	if (mulSynSlow!=NULL && deallocate) delete[] mulSynSlow;
	if (cumConnIdPre!=NULL && deallocate) delete[] cumConnIdPre;
	mulSynFast=NULL; mulSynSlow=NULL; cumConnIdPre=NULL;

	if (grpIds!=NULL && deallocate) delete[] grpIds;
	grpIds=NULL;

	#ifdef NEURON_NOISE
	if (intrinsicWeight!=NULL && deallocate) delete[] intrinsicWeight;
	#endif

	if (firingTableD2!=NULL && deallocate) delete[] firingTableD2;
	if (firingTableD1!=NULL && deallocate) delete[] firingTableD1;
	if (timeTableD2!=NULL && deallocate) delete[] timeTableD2;
	if (timeTableD1!=NULL && deallocate) delete[] timeTableD1;
	firingTableD2=NULL; firingTableD1=NULL; timeTableD2=NULL; timeTableD1=NULL;

	// delete all SpikeMonitor objects
	// don't kill SpikeMonitorCore objects, they will get killed automatically
	for (int i=0; i<numSpikeMonitor; i++) {
		if (spikeMonList[i]!=NULL && deallocate) delete spikeMonList[i];
		spikeMonList[i]=NULL;
	}
	// delete all Spike Counters
	for (int i=0; i<numSpkCnt; i++) {
		if (spkCntBuf[i]!=NULL && deallocate)
			delete[] spkCntBuf[i];
		spkCntBuf[i]=NULL;
	}

	if (pbuf!=NULL && deallocate) delete pbuf;
	if (spikeGenBits!=NULL && deallocate) delete[] spikeGenBits;
	pbuf=NULL; spikeGenBits=NULL;

	// clear all existing connection info
	if (deallocate) {
		while (connectBegin) {
			grpConnectInfo_t* nextConn = connectBegin->next;
			if (connectBegin!=NULL && deallocate) {
				free(connectBegin);
				connectBegin = nextConn;
			}
		}
	}
	connectBegin=NULL;

	// clear data (i.e., concentration of neuromodulator) of groups
	if (grpDA != NULL && deallocate) delete [] grpDA;
	if (grp5HT != NULL && deallocate) delete [] grp5HT;
	if (grpACh != NULL && deallocate) delete [] grpACh;
	if (grpNE != NULL && deallocate) delete [] grpNE;
	grpDA = NULL;
	grp5HT = NULL;
	grpACh = NULL;
	grpNE = NULL;

	// clear data buffer for group monitor
	for (int i = 0; i < numGroupMonitor; i++) {
		if (grpDABuffer != NULL && deallocate) delete [] grpDABuffer[i];
		if (grp5HTBuffer != NULL && deallocate) delete [] grp5HTBuffer[i];
		if (grpAChBuffer != NULL && deallocate) delete [] grpAChBuffer[i];
		if (grpNEBuffer != NULL && deallocate) delete [] grpNEBuffer[i];
		grpDABuffer[i] = NULL;
		grp5HTBuffer[i] = NULL;
		grpAChBuffer[i] = NULL;
		grpNEBuffer[i] = NULL;
	}
}


void CpuSNN::resetPoissonNeuron(unsigned int nid, int grpId) {
	assert(nid < numN);
	lastSpikeTime[nid]  = MAX_SIMULATION_TIME;
	if (grp_Info[grpId].WithHomeostasis)
		avgFiring[nid]      = 0.0;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(nid,j);
			stpu[nid] = 0.0f;
			stpx[nid] = 1.0f;
		}
	}
}

void CpuSNN::resetPropogationBuffer() {
	pbuf->reset(0, 1023);
}

// resets nSpikeCnt[]
void CpuSNN::resetSpikeCnt(int grpId) {
	int startGrp, endGrp;

	if (!doneReorganization)
		return;

	if (grpId == -1) {
		startGrp = 0;
		endGrp = numGrp;
	} else {
		 startGrp = grpId;
		 endGrp = grpId + nConfig_;
	}

	for (int g = startGrp; g<endGrp; g++) {
		int startN = grp_Info[g].StartN;
		int endN   = grp_Info[g].EndN;
		for (int i=startN; i<=endN; i++)
			nSpikeCnt[i] = 0;
	}
}

//Reset wt, wtChange, pre-firing time values to default values, rewritten to
//integrate changes between JMN and MDR -- KDC
//if changeWeights is false, we should keep the values of the weights as they currently
//are but we should be able to change them to plastic or fixed synapses. -- KDC
void CpuSNN::resetSynapticConnections(bool changeWeights) {
	int j;
	// Reset wt,wtChange,pre-firingtime values to default values...
	for(int destGrp=0; destGrp < numGrp; destGrp++) {
		const char* updateStr = (grp_Info[destGrp].newUpdates == true)?"(**)":"";
		CARLSIM_DEBUG("Grp: %d:%s s=%d e=%d %s", destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
					grp_Info[destGrp].EndN,  updateStr);
		CARLSIM_DEBUG("Grp: %d:%s s=%d e=%d  %s",  destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
					grp_Info[destGrp].EndN, updateStr);

		for(int nid=grp_Info[destGrp].StartN; nid <= grp_Info[destGrp].EndN; nid++) {
			unsigned int offset = cumulativePre[nid];
			for (j=0;j<Npre[nid]; j++) {
				wtChange[offset+j] = 0.0;						// synaptic derivatives is reset
				synSpikeTime[offset+j] = MAX_SIMULATION_TIME;	// some large negative value..
			}
			post_info_t *preIdPtr = &preSynapticIds[cumulativePre[nid]];
			float* synWtPtr       = &wt[cumulativePre[nid]];
			float* maxWtPtr       = &maxSynWt[cumulativePre[nid]];
			int prevPreGrp  = -1;

			for (j=0; j < Npre[nid]; j++,preIdPtr++, synWtPtr++, maxWtPtr++) {
				int preId    = GET_CONN_NEURON_ID((*preIdPtr));
				assert(preId < numN);
				int srcGrp = grpIds[preId];
				grpConnectInfo_t* connInfo;
				grpConnectInfo_t* connIterator = connectBegin;
				while(connIterator) {
					if(connIterator->grpSrc == srcGrp && connIterator->grpDest == destGrp) {
						//we found the corresponding connection
						connInfo=connIterator;
						break;
					}
					//move to the next grpConnectInfo_t
					connIterator=connIterator->next;
				}
				assert(connInfo != NULL);
				int connProp   = connInfo->connProp;
				bool   synWtType = GET_FIXED_PLASTIC(connProp);
				// print debug information...
				if( prevPreGrp != srcGrp) {
					if(nid==grp_Info[destGrp].StartN) {
						const char* updateStr = (connInfo->newUpdates==true)? "(**)":"";
						CARLSIM_DEBUG("\t%d (%s) start=%d, type=%s maxWts = %f %s", srcGrp,
										grp_Info2[srcGrp].Name.c_str(), j, (j<Npre_plastic[nid]?"P":"F"),
										connInfo->maxWt, updateStr);
					}
					prevPreGrp = srcGrp;
				}

				if(!changeWeights)
					continue;

				// if connection was plastic or if the connection weights were updated we need to reset the weights
				// TODO: How to account for user-defined connection reset
				if ((synWtType == SYN_PLASTIC) || connInfo->newUpdates) {
					*synWtPtr = getWeights(connInfo->connProp, connInfo->initWt, connInfo->maxWt, nid, srcGrp);
					*maxWtPtr = connInfo->maxWt;
				}
			}
		}
		grp_Info[destGrp].newUpdates = false;
	}

	grpConnectInfo_t* connInfo = connectBegin;
	// clear all existing connection info...
	while (connInfo) {
		connInfo->newUpdates = false;
		connInfo = connInfo->next;
	}
}

void CpuSNN::resetTimingTable() {
		memset(timeTableD2, 0, sizeof(int) * (1000 + maxDelay_ + 1));
		memset(timeTableD1, 0, sizeof(int) * (1000 + maxDelay_ + 1));
}



//! set one specific connection from neuron id 'src' to neuron id 'dest'
inline void CpuSNN::setConnection(int srcGrp,  int destGrp,  unsigned int src, unsigned int dest, float synWt,
									float maxWt, uint8_t dVal, int connProp, short int connId) {
	assert(dest<=CONN_SYN_NEURON_MASK);			// total number of neurons is less than 1 million within a GPU
	assert((dVal >=1) && (dVal <= maxDelay_));

	// we have exceeded the number of possible connection for one neuron
	if(Npost[src] >= grp_Info[srcGrp].numPostSynapses)	{
		CARLSIM_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		CARLSIM_ERROR("Large number of postsynaptic connections established");
		CARLSIM_ERROR("Increase maxM param in connect(%s,%s)",grp_Info2[srcGrp].Name.c_str(),grp_Info2[destGrp].Name.c_str());
		exitSimulation(1);
	}

	if(Npre[dest] >= grp_Info[destGrp].numPreSynapses) {
		CARLSIM_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		CARLSIM_ERROR("Large number of presynaptic connections established");
		CARLSIM_ERROR("Increase maxPreM param in connect(%s,%s)", grp_Info2[srcGrp].Name.c_str(), grp_Info2[destGrp].Name.c_str());
		exitSimulation(1);
	}

	int p = Npost[src];

	assert(Npost[src] >= 0);
	assert(Npre[dest] >= 0);
	assert((src * numPostSynapses + p) / numN < numPostSynapses); // divide by numN to prevent INT overflow

	unsigned int post_pos = cumulativePost[src] + Npost[src];
	unsigned int pre_pos  = cumulativePre[dest] + Npre[dest];

	assert(post_pos < postSynCnt);
	assert(pre_pos  < preSynCnt);

	//generate a new postSynapticIds id for the current connection
	postSynapticIds[post_pos]   = SET_CONN_ID(dest, Npre[dest], destGrp);
	tmp_SynapticDelay[post_pos] = dVal;

	preSynapticIds[pre_pos] = SET_CONN_ID(src, Npost[src], srcGrp);
	wt[pre_pos] 	  = synWt;
	maxSynWt[pre_pos] = maxWt;
	cumConnIdPre[pre_pos] = connId;

	bool synWtType = GET_FIXED_PLASTIC(connProp);

	if (synWtType == SYN_PLASTIC) {
		sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
		Npre_plastic[dest]++;
		// homeostasis
		if (grp_Info[destGrp].WithHomeostasis && grp_Info[destGrp].homeoId ==-1)
			grp_Info[destGrp].homeoId = dest; // this neuron info will be printed
	}

	Npre[dest] += 1;
	Npost[src] += 1;

	grp_Info2[srcGrp].numPostConn++;
	grp_Info2[destGrp].numPreConn++;

	if (Npost[src] > grp_Info2[srcGrp].maxPostConn)
		grp_Info2[srcGrp].maxPostConn = Npost[src];
	if (Npre[dest] > grp_Info2[destGrp].maxPreConn)
	grp_Info2[destGrp].maxPreConn = Npre[src];
}

void CpuSNN::setGrpTimeSlice(int grpId, int timeSlice) {
	if (grpId == ALL) {
		for(int g=0; (g < numGrp); g++) {
			if (grp_Info[g].isSpikeGenerator)
				setGrpTimeSlice(g, timeSlice);
		}
	} else {
		assert((timeSlice > 0 ) && (timeSlice <  PROPAGATED_BUFFER_SIZE));
		// the group should be poisson spike generator group
		grp_Info[grpId].NewTimeSlice = timeSlice;
		grp_Info[grpId].CurrTimeSlice = timeSlice;
	}
}

// method to set const member randSeed_
int CpuSNN::setRandSeed(int seed) {
	if (seed<0)
		return time(NULL);
	else if(seed==0)
		return 123;
	else
		return seed;
}

// reorganize the network and do the necessary allocation
// of all variable for carrying out the simulation..
// this code is run only one time during network initialization
void CpuSNN::setupNetwork(bool removeTempMem) {
	if(!doneReorganization)
		reorganizeNetwork(removeTempMem);

	if((simMode_ == GPU_MODE) && (cpu_gpuNetPtrs.allocated == false))
		allocateSNN_GPU();
}


void CpuSNN::startCPUTiming() { prevCpuExecutionTime = cumExecutionTime; }
void CpuSNN::startGPUTiming() { prevGpuExecutionTime = cumExecutionTime; }
void CpuSNN::stopCPUTiming() {
	cpuExecutionTime += (cumExecutionTime - prevCpuExecutionTime);
	prevCpuExecutionTime = cumExecutionTime;
}
void CpuSNN::stopGPUTiming() {
	gpuExecutionTime += (cumExecutionTime - prevGpuExecutionTime);
	prevGpuExecutionTime = cumExecutionTime;
}


void CpuSNN::swapConnections(int nid, int oldPos, int newPos) {
	unsigned int cumN=cumulativePost[nid];

	// Put the node oldPos to the top of the delay queue
	post_info_t tmp = postSynapticIds[cumN+oldPos];
	postSynapticIds[cumN+oldPos]= postSynapticIds[cumN+newPos];
	postSynapticIds[cumN+newPos]= tmp;

	// Ensure that you have shifted the delay accordingly....
	uint8_t tmp_delay = tmp_SynapticDelay[cumN+oldPos];
	tmp_SynapticDelay[cumN+oldPos] = tmp_SynapticDelay[cumN+newPos];
	tmp_SynapticDelay[cumN+newPos] = tmp_delay;

	// update the pre-information for the postsynaptic neuron at the position oldPos.
	post_info_t  postInfo = postSynapticIds[cumN+oldPos];
	int  post_nid = GET_CONN_NEURON_ID(postInfo);
	int  post_sid = GET_CONN_SYN_ID(postInfo);

	post_info_t* preId    = &preSynapticIds[cumulativePre[post_nid]+post_sid];
	int  pre_nid  = GET_CONN_NEURON_ID((*preId));
	int  pre_sid  = GET_CONN_SYN_ID((*preId));
	int  pre_gid  = GET_CONN_GRP_ID((*preId));
	assert (pre_nid == nid);
	assert (pre_sid == newPos);
	*preId = SET_CONN_ID( pre_nid, oldPos, pre_gid);

	// update the pre-information for the postsynaptic neuron at the position newPos
	postInfo = postSynapticIds[cumN+newPos];
	post_nid = GET_CONN_NEURON_ID(postInfo);
	post_sid = GET_CONN_SYN_ID(postInfo);

	preId    = &preSynapticIds[cumulativePre[post_nid]+post_sid];
	pre_nid  = GET_CONN_NEURON_ID((*preId));
	pre_sid  = GET_CONN_SYN_ID((*preId));
	pre_gid  = GET_CONN_GRP_ID((*preId));
	assert (pre_nid == nid);
	assert (pre_sid == oldPos);
	*preId = SET_CONN_ID( pre_nid, newPos, pre_gid);
}


void CpuSNN::updateAfterMaxTime() {
  CARLSIM_WARN("Maximum Simulation Time Reached...Resetting simulation time");

	// This will be our cut of time. All other time values
	// that are less than cutOffTime will be set to zero
	unsigned int cutOffTime = (MAX_SIMULATION_TIME - 10*1000);

	for(int g=0; g < numGrp; g++) {

	if (grp_Info[g].isSpikeGenerator) {
			int diffTime = (grp_Info[g].SliceUpdateTime - cutOffTime);
			grp_Info[g].SliceUpdateTime = (diffTime < 0) ? 0 : diffTime;
		}

		// no STDP then continue...
		if(!grp_Info[g].FixedInputWts) {
			continue;
		}

		for(int k=0, nid = grp_Info[g].StartN; nid <= grp_Info[g].EndN; nid++,k++) {
		assert(nid < numNReg);
			// calculate the difference in time
			signed diffTime = (lastSpikeTime[nid] - cutOffTime);
			lastSpikeTime[nid] = (diffTime < 0) ? 0 : diffTime;

			// do the same thing with all synaptic connections..
			unsigned* synTime = &synSpikeTime[cumulativePre[nid]];
			for(int i=0; i < Npre[nid]; i++, synTime++) {
				// calculate the difference in time
				signed diffTime = (synTime[0] - cutOffTime);
				synTime[0]      = (diffTime < 0) ? 0 : diffTime;
			}
		}
	}

	simTime = MAX_SIMULATION_TIME - cutOffTime;
	resetPropogationBuffer();
}

void CpuSNN::updateConnectionMonitor() {
	for (int grpId = 0; grpId < numGrp; grpId++) {
		int monitorId = grp_Info[grpId].ConnectionMonitorId;

		if(monitorId != -1) {
			int grpIdPre = connMonGrpIdPre[monitorId];
			int grpIdPost = connMonGrpIdPost[monitorId];
			float* weights = NULL;
			float avgWeight = 0.0f;
			int weightSzie;
			getPopWeights(grpIdPre, grpIdPost, weights, weightSzie, 0);

			for (int i = 0; i < weightSzie; i++)
				avgWeight += weights[i];
			avgWeight /= weightSzie;

			CARLSIM_INFO("");
			CARLSIM_INFO("(t=%.3fs) Connection Monitor for Group %s to Group %s has average weight %f",
				(float)(simTime/1000.0),
				grp_Info2[grpIdPre].Name.c_str(), grp_Info2[grpIdPost].Name.c_str(), avgWeight);

			printWeights(grpIdPre,grpIdPost);

			// call the callback function
			if (connBufferCallback[monitorId])
				connBufferCallback[monitorId]->update(this, grpIdPre, grpIdPost, weights, weightSzie);

			if (weights != NULL)
				delete [] weights;
		}
	}
}

void CpuSNN::updateGroupMonitor() {
	// TODO: build DA, 5HT, ACh, NE buffer in GPU memory and retrieve data every one second
	// Currently, there is no buffer in GPU side. data are retrieved at every 10 ms simulation time

	for (int grpId = 0; grpId < numGrp; grpId++) {
		int monitorId = grp_Info[grpId].GroupMonitorId;

		if(monitorId != -1) {
			CARLSIM_INFO("Group Monitor for Group %s has DA(%f)", grp_Info2[grpId].Name.c_str(), grpDABuffer[monitorId][0]);

			// call the callback function
			if (grpBufferCallback[monitorId])
				grpBufferCallback[monitorId]->update(this, grpId, grpDABuffer[monitorId], 100);
		}
	}
}


//! update CpuSNN::numNExcPois, CpuSNN::numNInhPois, CpuSNN::numNExcReg, CpuSNN::numNInhReg, CpuSNN::numNPois, CpuSNN::numNReg
/*
 * \param _numN [out] current number of neurons
 * \param _numMaxPostSynapses [out] number of maximum post synapses in groups
 * \param _numMaxPreSynapses [out] number of maximum pre synapses in groups
 * \param _numConfig [in] (deprecated) number of configuration
 */
void CpuSNN::updateParameters(int* curN, int* numPostSynapses, int* numPreSynapses, int nConfig) {
	assert(nConfig > 0);
	numNExcPois = 0;
	numNInhPois = 0;
	numNExcReg = 0;
	numNInhReg = 0;
	*numPostSynapses = 0;
	*numPreSynapses = 0;

	//  scan all the groups and find the required information
	//  about the group (numN, numPostSynapses, numPreSynapses and others).
	for(int g=0; g < numGrp; g++)  {
		if (grp_Info[g].Type==UNKNOWN_NEURON) {
			CARLSIM_ERROR("Unknown group for %d (%s)", g, grp_Info2[g].Name.c_str());
			exitSimulation(1);
		}

		if (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON))
			numNInhReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON))
			numNExcReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type & POISSON_NEURON))
			numNExcPois += grp_Info[g].SizeN;
		else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type & POISSON_NEURON))
			numNInhPois += grp_Info[g].SizeN;

		// find the values for maximum postsynaptic length
		// and maximum pre-synaptic length
		if (grp_Info[g].numPostSynapses >= *numPostSynapses)
			*numPostSynapses = grp_Info[g].numPostSynapses;
		if (grp_Info[g].numPreSynapses >= *numPreSynapses)
			*numPreSynapses = grp_Info[g].numPreSynapses;
	}

	*curN  = numNExcReg + numNInhReg + numNExcPois + numNInhPois;
	numNPois = numNExcPois + numNInhPois;
	numNReg = numNExcReg +numNInhReg;
}

void CpuSNN::updateSpikesFromGrp(int grpId) {
	assert(grp_Info[grpId].isSpikeGenerator==true);

	bool done;
	//static FILE* _fp = fopen("spikes.txt", "w");
	unsigned int currTime = simTime;

	int timeSlice = grp_Info[grpId].CurrTimeSlice;
	grp_Info[grpId].SliceUpdateTime  = simTime;

	// we dont generate any poisson spike if during the
	// current call we might exceed the maximum 32 bit integer value
	if (((uint64_t) currTime + timeSlice) >= MAX_SIMULATION_TIME)
		return;

	if (grp_Info[grpId].spikeGen) {
		generateSpikesFromFuncPtr(grpId);
	} else {
		// current mode is GPU, and GPU would take care of poisson generators
		// and other information about refractor period etc. So no need to continue further...
#if !TESTING_CPU_GPU_POISSON
    if(simMode_ == GPU_MODE)
      return;
#endif

		generateSpikesFromRate(grpId);
	}
}

void CpuSNN::updateSpikeGenerators() {
	for(int g=0; g<numGrp; g++) {
		if (grp_Info[g].isSpikeGenerator) {
			// This evaluation is done to check if its time to get new set of spikes..
			// check whether simTime has advance more than the current time slice, in which case we need to schedule
			// spikes for the next time slice
			// we always have to run this the first millisecond of a new runNetwork call; that is,
			// when simTime==simTimeRunStart
			if(((simTime-grp_Info[g].SliceUpdateTime) >= (unsigned) grp_Info[g].CurrTimeSlice || simTime == simTimeRunStart)) {
				updateSpikesFromGrp(g);
			}
		}
	}
}

void CpuSNN::updateSpikeGeneratorsInit() {
	int cnt=0;
	for(int g=0; (g < numGrp); g++) {
		if (grp_Info[g].isSpikeGenerator) {
			// This is done only during initialization
			grp_Info[g].CurrTimeSlice = grp_Info[g].NewTimeSlice;

			// we only need NgenFunc for spike generator callbacks that need to transfer their spikes to the GPU
			if (grp_Info[g].spikeGen) {
				grp_Info[g].Noffset = NgenFunc;
				NgenFunc += grp_Info[g].SizeN;
			}
			//Note: updateSpikeFromGrp() will be called first time in updateSpikeGenerators()
			//updateSpikesFromGrp(g);
			cnt++;
			assert(cnt <= numSpikeGenGrps);
		}
	}

	// spikeGenBits can be set only once..
	assert(spikeGenBits == NULL);

	if (NgenFunc) {
		spikeGenBits = new uint32_t[NgenFunc/32+1];
		cpuNetPtrs.spikeGenBits = spikeGenBits;
		// increase the total memory size used by the routine...
		cpuSnnSz.addInfoSize += sizeof(spikeGenBits[0])*(NgenFunc/32+1);
	}
}

//! update CpuSNN::maxSpikesD1, CpuSNN::maxSpikesD2 and allocate sapce for CpuSNN::firingTableD1 and CpuSNN::firingTableD2
/*!
 * \return maximum delay in groups
 */
int CpuSNN::updateSpikeTables() {
	int curD = 0;
	int grpSrc;
	// find the maximum delay in the given network
	// and also the maximum delay for each group.
	grpConnectInfo_t* newInfo = connectBegin;
	while(newInfo) {
		grpSrc = newInfo->grpSrc;
		if (newInfo->maxDelay > curD)
			curD = newInfo->maxDelay;

		// check if the current connection's delay meaning grp1's delay
		// is greater than the MaxDelay for grp1. We find the maximum
		// delay for the grp1 by this scheme.
		if (newInfo->maxDelay > grp_Info[grpSrc].MaxDelay)
		 	grp_Info[grpSrc].MaxDelay = newInfo->maxDelay;

		newInfo = newInfo->next;
	}

	for(int g = 0; g < numGrp; g++) {
		if (grp_Info[g].MaxDelay == 1)
			maxSpikesD1 += (grp_Info[g].SizeN * grp_Info[g].MaxFiringRate);
		else
			maxSpikesD2 += (grp_Info[g].SizeN * grp_Info[g].MaxFiringRate);
	}

	if ((maxSpikesD1 + maxSpikesD2) < (numNExcReg + numNInhReg + numNPois) * UNKNOWN_NEURON_MAX_FIRING_RATE) {
		CARLSIM_ERROR("Insufficient amount of buffer allocated...");
		exitSimulation(1);
	}

	firingTableD2 = new unsigned int[maxSpikesD2];
	firingTableD1 = new unsigned int[maxSpikesD1];
	cpuSnnSz.spikingInfoSize += sizeof(int) * ((maxSpikesD2 + maxSpikesD1) + 2* (1000 + maxDelay_ + 1));

	return curD;
}

// This function is called every second by simulator...
// This function updates the firingTable by removing older firing values...
void CpuSNN::updateFiringTable() {
	// Read the neuron ids that fired in the last maxDelay_ seconds
	// and put it to the beginning of the firing table...
	for(int p=timeTableD2[999],k=0;p<timeTableD2[999+maxDelay_+1];p++,k++) {
		firingTableD2[k]=firingTableD2[p];
	}

	for(int i=0; i < maxDelay_; i++) {
		timeTableD2[i+1] = timeTableD2[1000+i+1]-timeTableD2[1000];
	}

	timeTableD1[maxDelay_] = 0;

	/* the code of weight update has been moved to CpuSNN::updateWeights() */

	spikeCountAll	+= spikeCountAll1sec;
	spikeCountD2Host += (secD2fireCntHost-timeTableD2[maxDelay_]);
	spikeCountD1Host += secD1fireCntHost;

	secD1fireCntHost  = 0;
	spikeCountAll1sec = 0;
	secD2fireCntHost = timeTableD2[maxDelay_];

	for (int i=0; i < numGrp; i++) {
		grp_Info[i].FiringCount1sec=0;
	}
}

// updates simTime, returns true when new second started
bool CpuSNN::updateTime() {
	bool finishedOneSec = false;

	// done one second worth of simulation
	// update relevant parameters...now
	if(++simTimeMs == 1000) {
		simTimeMs = 0;
		simTimeSec++;
		finishedOneSec = true;
	}

	simTime++;
	if(simTime >= MAX_SIMULATION_TIME){
		// reached the maximum limit of the simulation time using 32 bit value...
		updateAfterMaxTime();
	}

	return finishedOneSec;
}


void CpuSNN::updateSpikeMonitor(int grpId) {
	// don't continue if no spike monitors in the network
	if (!numSpikeMonitor)
		return;

	if (grpId==ALL) {
		for (int g=0; g<numGrp; g++)
			updateSpikeMonitor(g);
	} else {
		// update spike monitor of a specific group

		// find index in spike monitor arrays
		int monitorId = grp_Info[grpId].SpikeMonitorId;

		// don't continue if no spike monitor enabled for this group
		if (monitorId<0)
			return;

		// find last update time for this group
		SpikeMonitorCore* spkMonObj = spikeMonCoreList[monitorId];
		long int lastUpdate = spkMonObj->getLastUpdated();

		// don't continue if time interval is zero (nothing to update)
		if ( ((long int)getSimTime()) - lastUpdate <=0)
			return;

		if ( ((long int)getSimTime()) - lastUpdate > 1000)
			CARLSIM_ERROR("updateSpikeMonitor(grpId=%d) must be called at least once every second",grpId);

		if (simMode_ == GPU_MODE) {
			// copy the neuron firing information from the GPU to the CPU..
			copyFiringInfo_GPU();
		}

		// find the time interval in which to update spikes
		// usually, we call updateSpikeMonitor once every second, so the time interval is [0,1000)
		// however, updateSpikeMonitor can be called at any time t \in [0,1000)... so we can have the cases
		// [0,t), [t,1000), and even [t1, t2)
		int numMsMin = lastUpdate%1000; // lower bound is given by last time we called update
		int numMsMax = getSimTimeMs(); // upper bound is given by current time
		if (numMsMax==0)
			numMsMax = 1000; // special case: full second
		assert(numMsMin<numMsMax);

		// current time is last completed second in milliseconds (plus t to be added below)
		// special case is after each completed second where !getSimTimeMs(): here we look 1s back
		int currentTimeSec = getSimTimeSec();
		if (!getSimTimeMs())
			currentTimeSec--;

		// save current time as last update time
		spkMonObj->setLastUpdated( (long int)getSimTime() );

		// prepare fast access
		FILE* spkFileId = spikeMonCoreList[monitorId]->getSpikeFileId();
		bool writeSpikesToFile = spkFileId!=NULL;
		bool writeSpikesToArray = spkMonObj->getMode()==AER && spkMonObj->isRecording();

		// Read one spike at a time from the buffer and put the spikes to an appopriate monitor buffer. Later the user
		// may need need to dump these spikes to an output file
		for (int k=0; k < 2; k++) {
			unsigned int* timeTablePtr = (k==0)?timeTableD2:timeTableD1;
			unsigned int* fireTablePtr = (k==0)?firingTableD2:firingTableD1;
			for(int t=numMsMin; t<numMsMax; t++) {
				for(int i=timeTablePtr[t+maxDelay_]; i<timeTablePtr[t+maxDelay_+1];i++) {
					// retrieve the neuron id
					int nid   = fireTablePtr[i];
					if (simMode_ == GPU_MODE)
						nid = GET_FIRING_TABLE_NID(nid);
					assert(nid < numN);

					// make sure neuron belongs to currently relevant group
					int this_grpId = grpIds[nid];
					if (this_grpId != grpId)
						continue;

					// adjust nid to be 0-indexed for each group
					// this way, if a group has 10 neurons, their IDs in the spike file and spike monitor will be
					// indexed from 0..9, no matter what their real nid is
					nid -= grp_Info[grpId].StartN;
					assert(nid>=0);

					// current time is last completed second plus whatever is leftover in t
					int time = currentTimeSec*1000 + t;

					if (writeSpikesToFile) {
						int cnt;
						cnt = fwrite(&time, sizeof(int), 1, spkFileId); assert(cnt==1);
						cnt = fwrite(&nid,  sizeof(int), 1, spkFileId); assert(cnt==1);
					}

					if (writeSpikesToArray) {
						spkMonObj->pushAER(time,nid);
					}
				}
			}
		}

		if (spkFileId!=NULL) // flush spike file
			fflush(spkFileId);
	}
}

// This function updates the synaptic weights from its derivatives..
void CpuSNN::updateWeights() {
	// update synaptic weights here for all the neurons..
	for(int g = 0; g < numGrp; g++) {
		// no changable weights so continue without changing..
		if(grp_Info[g].FixedInputWts || !(grp_Info[g].WithSTDP))
			continue;

		for(int i = grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
			assert(i < numNReg);
			unsigned int offset = cumulativePre[i];
			float diff_firing = 0.0;
			float homeostasisScale = 1.0;

			if(grp_Info[g].WithHomeostasis) {
				assert(baseFiring[i]>0);
				diff_firing = 1-avgFiring[i]/baseFiring[i];
				homeostasisScale = grp_Info[g].homeostasisScale;
			}

			if (i==grp_Info[g].StartN)
				CARLSIM_DEBUG("Weights, Change at %lu (diff_firing: %f)", simTimeSec, diff_firing);

			for(int j = 0; j < Npre_plastic[i]; j++) {
				//	if (i==grp_Info[g].StartN)
				//		CARLSIM_DEBUG("%1.2f %1.2f \t", wt[offset+j]*10, wtChange[offset+j]*10);
				float effectiveWtChange = stdpScaleFactor_ * wtChange[offset + j];

				// homeostatic weight update
				switch (grp_Info[g].WithSTDPtype) {
				case STANDARD:
					if (grp_Info[g].WithHomeostasis) {
						wt[offset+j] += (diff_firing*wt[offset+j]*homeostasisScale + wtChange[offset+j])*baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						// just STDP weight update
						wt[offset+j] += effectiveWtChange;
					}
					wtChange[offset+j] = 0.0f;
					break;
				case DA_MOD:
					if (grp_Info[g].WithHomeostasis) {
						effectiveWtChange = cpuNetPtrs.grpDA[g] * effectiveWtChange;
						wt[offset+j] += (diff_firing*wt[offset+j]*homeostasisScale + effectiveWtChange)*baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						wt[offset+j] += cpuNetPtrs.grpDA[g] * effectiveWtChange;
					}
					wtChange[offset+j] *= wtChangeDecay_;
					break;
				case UNKNOWN_STDP:
				default:
					// we shouldn't even be in here if !WithSTDP
					break;
				}

				// if this is an excitatory or inhibitory synapse
				if (maxSynWt[offset + j] >= 0) {
					if (wt[offset + j] >= maxSynWt[offset + j])
						wt[offset + j] = maxSynWt[offset + j];
					if (wt[offset + j] < 0)
						wt[offset + j] = 0.0;
				} else {
					if (wt[offset + j] <= maxSynWt[offset + j])
						wt[offset + j] = maxSynWt[offset + j];
					if (wt[offset+j] > 0)
						wt[offset+j] = 0.0;
				}
			}
		}
	}
}
