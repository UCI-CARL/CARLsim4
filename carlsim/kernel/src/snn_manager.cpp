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

#include <snn.h>
#include <sstream>

#include <connection_monitor.h>
#include <connection_monitor_core.h>
#include <spike_monitor.h>
#include <spike_monitor_core.h>
#include <group_monitor.h>
#include <group_monitor_core.h>

// \FIXME what are the following for? why were they all the way at the bottom of this file?

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///


// TODO: consider moving unsafe computations out of constructor
SNN::SNN(const std::string& name, SimMode simMode, LoggerMode loggerMode, int ithGPU, int randSeed)
					: networkName_(name), simMode_(simMode), loggerMode_(loggerMode), ithGPU_(ithGPU),
					  randSeed_(SNN::setRandSeed(randSeed)) // all of these are const
{
	// move all unsafe operations out of constructor
	SNNinit();
}

// destructor
SNN::~SNN() {
	if (!simulatorDeleted)
		deleteObjects();
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: SETTING UP A SIMULATION
/// ************************************************************************************************************ ///

// make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
short int SNN::connect(int grpId1, int grpId2, const std::string& _type, float initWt, float maxWt, float prob,
						uint8_t minDelay, uint8_t maxDelay, float radX, float radY, float radZ,
						float _mulSynFast, float _mulSynSlow, bool synWtType) {
						//const std::string& wtType
	int retId=-1;
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

	Grid3D szPre = getGroupGrid3D(grpId1);
	Grid3D szPost = getGroupGrid3D(grpId2);

	// initialize configuration of a connection
	ConnectConfig connConfig;
	//ConnectConfig* newInfo = (ConnectConfig*) calloc(1, sizeof(ConnectConfig));
	connConfig.grpSrc   		  = grpId1;
	connConfig.grpDest  		  = grpId2;
	connConfig.initWt	  		  = initWt;
	connConfig.maxWt	  		  = maxWt;
	connConfig.maxDelay 		  = maxDelay;
	connConfig.minDelay 		  = minDelay;
//		newInfo->radX             = (radX<0) ? MAX(szPre.x,szPost.x) : radX; // <0 means full connectivity, so the
//		newInfo->radY             = (radY<0) ? MAX(szPre.y,szPost.y) : radY; // effective group size is Grid3D.x. Grab
//		newInfo->radZ             = (radZ<0) ? MAX(szPre.z,szPost.z) : radZ; // the larger of pre / post to connect all
	connConfig.radX             = radX;
	connConfig.radY             = radY;
	connConfig.radZ             = radZ;
	connConfig.mulSynFast       = _mulSynFast;
	connConfig.mulSynSlow       = _mulSynSlow;
	connConfig.connProp         = connProp;
	connConfig.p                = prob;
	connConfig.type             = CONN_UNKNOWN;
	connConfig.numPostSynapses  = 1;
	connConfig.numPreSynapses   = 1;
	connConfig.connectionMonitorId = -1;
	connConfig.connId = -1;

	//newInfo->next 				= connectBegin; //linked list of connection..
	//connectBegin 				= newInfo;

	if ( _type.find("random") != std::string::npos) {
		connConfig.type = CONN_RANDOM;
		connConfig.numPostSynapses = MIN(groupConfig[grpId2].SizeN,((int) (prob*groupConfig[grpId2].SizeN +6.5*sqrt(prob*(1-prob)*groupConfig[grpId2].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 6.5 stds.
		connConfig.numPreSynapses = MIN(groupConfig[grpId1].SizeN,((int) (prob*groupConfig[grpId1].SizeN +6.5*sqrt(prob*(1-prob)*groupConfig[grpId1].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 6.5 stds.
	}
	//so you're setting the size to be prob*Number of synapses in group info + some standard deviation ...
	else if ( _type.find("full-no-direct") != std::string::npos) {
		connConfig.type	= CONN_FULL_NO_DIRECT;
		connConfig.numPostSynapses = groupConfig[grpId2].SizeN-1;
		connConfig.numPreSynapses = groupConfig[grpId1].SizeN-1;
	}
	else if ( _type.find("full") != std::string::npos) {
		connConfig.type = CONN_FULL;
		connConfig.numPostSynapses = groupConfig[grpId2].SizeN;
		connConfig.numPreSynapses = groupConfig[grpId1].SizeN;
	}
	else if ( _type.find("one-to-one") != std::string::npos) {
		connConfig.type = CONN_ONE_TO_ONE;
		connConfig.numPostSynapses = 1;
		connConfig.numPreSynapses = 1;
	} else if ( _type.find("gaussian") != std::string::npos) {
		connConfig.type   = CONN_GAUSSIAN;
		// the following will soon go away, just assume the worst case for now
		connConfig.numPostSynapses = groupConfig[grpId2].SizeN;
		connConfig.numPreSynapses = groupConfig[grpId1].SizeN;
	} else {
		KERNEL_ERROR("Invalid connection type (should be 'random', 'full', 'one-to-one', 'full-no-direct', or 'gaussian')");
		exitSimulation(-1);
	}

	// assign connection id
	assert(connConfig.connId == -1);
	connConfig.connId = numConnections;

	if (connConfig.numPostSynapses > MAX_NUM_POST_SYN) {
		KERNEL_ERROR("ConnID %d exceeded the maximum number of output synapses (%d), has %d.",
			connConfig.connId, MAX_NUM_POST_SYN, connConfig.numPostSynapses);
		assert(connConfig.numPostSynapses <= MAX_NUM_POST_SYN);
	}

	if (connConfig.numPreSynapses > MAX_NUM_PRE_SYN) {
		KERNEL_ERROR("ConnID %d exceeded the maximum number of input synapses (%d), has %d.",
			connConfig.connId, MAX_NUM_PRE_SYN, connConfig.numPreSynapses);
		assert(connConfig.numPreSynapses <= MAX_NUM_PRE_SYN);
	}

	// update the pre and post size...
	// Subtlety: each group has numPost/PreSynapses from multiple connections.
	// The newInfo->numPost/PreSynapses are just for this specific connection.
	// We are adding the synapses counted in this specific connection to the totals for both groups.
	groupConfig[grpId1].numPostSynapses	+= connConfig.numPostSynapses;
	groupConfig[grpId2].numPreSynapses  += connConfig.numPreSynapses;

	KERNEL_DEBUG("groupConfig[%d, %s].numPostSynapses = %d, groupConfig[%d, %s].numPreSynapses = %d",
					grpId1,groupInfo[grpId1].Name.c_str(),groupConfig[grpId1].numPostSynapses,grpId2,
					groupInfo[grpId2].Name.c_str(),groupConfig[grpId2].numPreSynapses);

	KERNEL_DEBUG("CONNECT SETUP: connId=%d, mulFast=%f, mulSlow=%f", connConfig.connId, connConfig.mulSynFast, connConfig.mulSynSlow);

	connectConfigMap[numConnections] = connConfig; // connConfig.connId == numConnections
	
	assert(numConnections < MAX_CONN_PER_SNN);	// make sure we don't overflow connId
	numConnections++;
	
	return (numConnections-1);
}

// make custom connections from grpId1 to grpId2
short int SNN::connect(int grpId1, int grpId2, ConnectionGeneratorCore* conn, float _mulSynFast, float _mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	int retId=-1;

	assert(grpId1 < numGrp);
	assert(grpId2 < numGrp);

	if (maxM == 0)
		maxM = groupConfig[grpId2].SizeN;

	if (maxPreM == 0)
		maxPreM = groupConfig[grpId1].SizeN;

	if (maxM > MAX_NUM_POST_SYN) {
		KERNEL_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of output synapses (%d), "
							"has %d.", groupInfo[grpId1].Name.c_str(),grpId1,groupInfo[grpId2].Name.c_str(),
							grpId2,	MAX_NUM_POST_SYN,maxM);
		assert(maxM <= MAX_NUM_POST_SYN);
	}

	if (maxPreM > MAX_NUM_PRE_SYN) {
		KERNEL_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of input synapses (%d), "
							"has %d.\n", groupInfo[grpId1].Name.c_str(), grpId1,groupInfo[grpId2].Name.c_str(),
							grpId2, MAX_NUM_PRE_SYN,maxPreM);
		assert(maxPreM <= MAX_NUM_PRE_SYN);
	}

	// initialize the configuration of a connection
	ConnectConfig connConfig;
	//ConnectConfig* newInfo = (ConnectConfig*) calloc(1, sizeof(ConnectConfig));

	connConfig.grpSrc   = grpId1;
	connConfig.grpDest  = grpId2;
	connConfig.initWt	  = 1;
	connConfig.maxWt	  = 1;
	connConfig.maxDelay = MAX_SYN_DELAY;
	connConfig.minDelay = 1;
	connConfig.mulSynFast = _mulSynFast;
	connConfig.mulSynSlow = _mulSynSlow;
	connConfig.connProp = SET_CONN_PRESENT(1) | SET_FIXED_PLASTIC(synWtType);
	connConfig.type = CONN_USER_DEFINED;
	connConfig.numPostSynapses = maxM;
	connConfig.numPreSynapses = maxPreM;
	connConfig.conn = conn;
	connConfig.connectionMonitorId = -1;
	connConfig.connId = -1;

	//newInfo->next	= connectBegin;  // build a linked list
	//connectBegin      = newInfo;

	// update the pre and post size...
	groupConfig[grpId1].numPostSynapses += connConfig.numPostSynapses;
	groupConfig[grpId2].numPreSynapses += connConfig.numPreSynapses;

	KERNEL_DEBUG("groupConfig[%d, %s].numPostSynapses = %d, groupConfig[%d, %s].numPreSynapses = %d",
					grpId1,groupInfo[grpId1].Name.c_str(),groupConfig[grpId1].numPostSynapses,grpId2,
					groupInfo[grpId2].Name.c_str(),groupConfig[grpId2].numPreSynapses);

	// assign a connection id
	assert(connConfig.connId == -1);
	connConfig.connId = numConnections;

	connectConfigMap[numConnections] = connConfig; // connConfig.connId == numConnections

	assert(numConnections < MAX_CONN_PER_SNN);	// make sure we don't overflow connId
	numConnections++;

	return (numConnections-1);
}


// create group of Izhikevich neurons
// use int for nNeur to avoid arithmetic underflow
int SNN::createGroup(const std::string& grpName, const Grid3D& grid, int neurType) {
	assert(grid.numX * grid.numY * grid.numZ > 0);
	assert(neurType >= 0);
	assert(numGrp < MAX_GRP_PER_SNN);

	if ( (!(neurType & TARGET_AMPA) && !(neurType & TARGET_NMDA) &&
		  !(neurType & TARGET_GABAa) && !(neurType & TARGET_GABAb)) || (neurType & POISSON_NEURON)) {
		KERNEL_ERROR("Invalid type using createGroup... Cannot create poisson generators here.");
		exitSimulation(1);
	}

	//// initialize group configuration
	//GroupConfig grpConfig;
	//
	//// init parameters of neural group size and location
	//grpConfig.Name = grpName;
	//grpConfig.type = neurType;
	//grpConfig.numN = grid.N;;
	//grpConfig.sizeX = grid.numX;
	//grpConfig.sizeY = grid.numY;
	//grpConfig.sizeZ = grid.numZ;
	//grpConfig.distX = grid.distX;
	//grpConfig.distY = grid.distY;
	//grpConfig.distZ = grid.distZ;
	//grpConfig.offsetX = grid.offsetX;
	//grpConfig.offsetY = grid.offsetY;
	//grpConfig.offsetZ = grid.offsetZ;

	//// init parameters of neural group dynamics
	//grpConfig.Izh_a = -1.0f;
	//grpConfig.Izh_a_sd = -1.0f;
	//grpConfig.Izh_b = -1.0f;
	//grpConfig.Izh_b_sd = -1.0f;
	//grpConfig.Izh_c = -1.0f;
	//grpConfig.Izh_c_sd = -1.0f;
	//grpConfig.Izh_d = -1.0f;
	//grpConfig.Izh_d_sd = -1.0f;

	//grpConfig.isSpikeGenerator = false;

	//// init homeostatic plasticity configs
	//grpConfig.baseFiring = -1.0f;
	//grpConfig.baseFiringSD = -1.0f;
	//grpConfig.avgTimeScale = -1.0f;
	//grpConfig.avgTimeScaleDecay = -1.0f;
	//grpConfig.homeostasisScale = -1.0f;

	//// init parameters of neuromodulator
	//grpConfig.baseDP = -1.0f;
	//grpConfig.base5HT = -1.0f;
	//grpConfig.baseACh = -1.0f;
	//grpConfig.baseNE = -1.0f;
	//grpConfig.decayDP = -1.0f;
	//grpConfig.decay5HT = -1.0f;
	//grpConfig.decayACh = -1.0f;
	//grpConfig.decayNE = -1.0f;

	//groupConfigMap[numGrp] = grpConfig;

	// We don't store the Grid3D struct in groupConfig so we don't have to deal with allocating structs on the GPU
	groupConfig[numGrp].SizeN = grid.N; // number of neurons in the group
	groupConfig[numGrp].SizeX = grid.numX; // number of neurons in first dim of Grid3D
	groupConfig[numGrp].SizeY = grid.numY; // number of neurons in second dim of Grid3D
	groupConfig[numGrp].SizeZ = grid.numZ; // number of neurons in third dim of Grid3D

	groupConfig[numGrp].Type   			= neurType;
	groupConfig[numGrp].WithSTP			= false;
	groupConfig[numGrp].WithSTDP			= false;
	groupConfig[numGrp].WithESTDPtype      = UNKNOWN_STDP;
	groupConfig[numGrp].WithISTDPtype		= UNKNOWN_STDP;
	groupConfig[numGrp].WithHomeostasis	= false;

	if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb)) {
		groupConfig[numGrp].MaxFiringRate 	= INHIBITORY_NEURON_MAX_FIRING_RATE;
	} else {
		groupConfig[numGrp].MaxFiringRate 	= EXCITATORY_NEURON_MAX_FIRING_RATE;
	}

	groupConfig[numGrp].isSpikeGenerator	= false;
	groupConfig[numGrp].MaxDelay			= 1;
	
	groupInfo[numGrp].Name  			= grpName;
	groupInfo[numGrp].Izh_a 			= -1; // \FIXME ???

	// init homeostasis params even though not used
	groupInfo[numGrp].baseFiring        = 10.0f;
	groupInfo[numGrp].baseFiringSD      = 0.0f;

	groupInfo[numGrp].Name              = grpName;

	// update number of neuron counters
	if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb))
		numNInhReg += grid.N; // regular inhibitory neuron
	else
		numNExcReg += grid.N; // regular excitatory neuron
	numNReg += grid.N;
	numN += grid.N;

	numGrp++;
	return (numGrp-1);
}

// create spike generator group
// use int for nNeur to avoid arithmetic underflow
int SNN::createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType) {
	assert(grid.numX * grid.numY * grid.numZ > 0);
	assert(neurType >= 0);
	assert(numGrp < MAX_GRP_PER_SNN);

	//// initialize group configuration
	//GroupConfig grpConfig;
	//
	//// init parameters of neural group size and location
	//grpConfig.Name = grpName;
	//grpConfig.type = neurType | POISSON_NEURON;
	//grpConfig.numN = grid.N;;
	//grpConfig.sizeX = grid.numX;
	//grpConfig.sizeY = grid.numY;
	//grpConfig.sizeZ = grid.numZ;
	//grpConfig.distX = grid.distX;
	//grpConfig.distY = grid.distY;
	//grpConfig.distZ = grid.distZ;
	//grpConfig.offsetX = grid.offsetX;
	//grpConfig.offsetY = grid.offsetY;
	//grpConfig.offsetZ = grid.offsetZ;

	//// init parameters of neural group dynamics
	//grpConfig.Izh_a = -1.0f;
	//grpConfig.Izh_a_sd = -1.0f;
	//grpConfig.Izh_b = -1.0f;
	//grpConfig.Izh_b_sd = -1.0f;
	//grpConfig.Izh_c = -1.0f;
	//grpConfig.Izh_c_sd = -1.0f;
	//grpConfig.Izh_d = -1.0f;
	//grpConfig.Izh_d_sd = -1.0f;

	//grpConfig.isSpikeGenerator = true;

	//// init homeostatic plasticity configs
	//grpConfig.baseFiring = -1.0f;
	//grpConfig.baseFiringSD = -1.0f;
	//grpConfig.avgTimeScale = -1.0f;
	//grpConfig.avgTimeScaleDecay = -1.0f;
	//grpConfig.homeostasisScale = -1.0f;

	//// init parameters of neuromodulator
	//grpConfig.baseDP = -1.0f;
	//grpConfig.base5HT = -1.0f;
	//grpConfig.baseACh = -1.0f;
	//grpConfig.baseNE = -1.0f;
	//grpConfig.decayDP = -1.0f;
	//grpConfig.decay5HT = -1.0f;
	//grpConfig.decayACh = -1.0f;
	//grpConfig.decayNE = -1.0f;

	//groupConfigMap[numGrp] = grpConfig;

	groupConfig[numGrp].SizeN = grid.N; // number of neurons in the group
	groupConfig[numGrp].SizeX = grid.numX; // number of neurons in first dim of Grid3D
	groupConfig[numGrp].SizeY = grid.numY; // number of neurons in second dim of Grid3D
	groupConfig[numGrp].SizeZ = grid.numZ; // number of neurons in third dim of Grid3D
	groupConfig[numGrp].Type    		= neurType | POISSON_NEURON;
	groupConfig[numGrp].WithSTP		= false;
	groupConfig[numGrp].WithSTDP		= false;
	groupConfig[numGrp].WithESTDPtype  = UNKNOWN_STDP;
	groupConfig[numGrp].WithISTDPtype	= UNKNOWN_STDP;
	groupConfig[numGrp].WithHomeostasis	= false;
	groupConfig[numGrp].isSpikeGenerator	= true;		// these belong to the spike generator class...
	groupInfo[numGrp].Name    		= grpName;
	groupConfig[numGrp].MaxFiringRate 	= POISSON_MAX_FIRING_RATE;

	groupInfo[numGrp].Name          = grpName;

	if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb))
		numNInhPois += grid.N; // inh poisson group
	else
		numNExcPois += grid.N; // exc poisson group
	numNPois += grid.N;
	numN += grid.N;

	numGrp++;
	numSpikeGenGrps++;

	return (numGrp-1);
}

// set conductance values for a simulation (custom values or disable conductances alltogether)
void SNN::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa,
int trGABAb, int tdGABAb) {
	if (isSet) {
		assert(tdAMPA>0); assert(tdNMDA>0); assert(tdGABAa>0); assert(tdGABAb>0);
		assert(trNMDA>=0); assert(trGABAb>=0); // 0 to disable rise times
		assert(trNMDA!=tdNMDA); assert(trGABAb!=tdGABAb); // singularity
	}

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

	if (sim_with_conductances) {
		KERNEL_INFO("Running COBA mode:");
		KERNEL_INFO("  - AMPA decay time            = %5d ms", tdAMPA);
		KERNEL_INFO("  - NMDA rise time %s  = %5d ms", sim_with_NMDA_rise?"          ":"(disabled)", trNMDA);
		KERNEL_INFO("  - GABAa decay time           = %5d ms", tdGABAa);
		KERNEL_INFO("  - GABAb rise time %s = %5d ms", sim_with_GABAb_rise?"          ":"(disabled)",trGABAb);
		KERNEL_INFO("  - GABAb decay time           = %5d ms", tdGABAb);
	} else {
		KERNEL_INFO("Running CUBA mode (all synaptic conductances disabled)");
	}
}

// set homeostasis for group
void SNN::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale) {
	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1++) {
			setHomeostasis(grpId1, isSet, homeoScale, avgTimeScale);
		}
	} else {
		// set conductances for a given group
		sim_with_homeostasis 			   |= isSet;
		groupConfig[grpId].WithHomeostasis    = isSet;
		groupConfig[grpId].homeostasisScale   = homeoScale;
		groupConfig[grpId].avgTimeScale       = avgTimeScale;
		groupConfig[grpId].avgTimeScaleInv    = 1.0f/avgTimeScale;
		groupConfig[grpId].avgTimeScale_decay = (avgTimeScale*1000.0f-1.0f)/(avgTimeScale*1000.0f);
		groupConfig[grpId].newUpdates 		= true; // \FIXME: what's this?

		KERNEL_INFO("Homeostasis parameters %s for %d (%s):\thomeoScale: %f, avgTimeScale: %f",
					isSet?"enabled":"disabled",grpId,groupInfo[grpId].Name.c_str(),homeoScale,avgTimeScale);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void SNN::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD) {
	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1++) {
			setHomeoBaseFiringRate(grpId1, baseFiring, baseFiringSD);
		}
	} else {
		// set conductances for a given group
		assert(groupConfig[grpId].WithHomeostasis);

		groupInfo[grpId].baseFiring 	= baseFiring;
		groupInfo[grpId].baseFiringSD 	= baseFiringSD;
		groupConfig[grpId].newUpdates 	= true; //TODO: I have to see how this is handled.  -- KDC

		KERNEL_INFO("Homeostatic base firing rate set for %d (%s):\tbaseFiring: %3.3f, baseFiringStd: %3.3f",
							grpId,groupInfo[grpId].Name.c_str(),baseFiring,baseFiringSD);
	}
}


// set Izhikevich parameters for group
void SNN::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
								float izh_c, float izh_c_sd, float izh_d, float izh_d_sd)
{
	assert(grpId>=-1); assert(izh_a_sd>=0); assert(izh_b_sd>=0); assert(izh_c_sd>=0);
	assert(izh_d_sd>=0);

	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1++) {
			setNeuronParameters(grpId1, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
		}
	} else {
		groupInfo[grpId].Izh_a	  	=   izh_a;
		groupInfo[grpId].Izh_a_sd  =   izh_a_sd;
		groupInfo[grpId].Izh_b	  	=   izh_b;
		groupInfo[grpId].Izh_b_sd  =   izh_b_sd;
		groupInfo[grpId].Izh_c		=   izh_c;
		groupInfo[grpId].Izh_c_sd	=   izh_c_sd;
		groupInfo[grpId].Izh_d		=   izh_d;
		groupInfo[grpId].Izh_d_sd	=   izh_d_sd;
	}
}

void SNN::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh,
	float tauACh, float baseNE, float tauNE) {

	groupConfig[grpId].baseDP	= baseDP;
	groupConfig[grpId].decayDP = 1.0 - (1.0 / tauDP);
	groupConfig[grpId].base5HT = base5HT;
	groupConfig[grpId].decay5HT = 1.0 - (1.0 / tau5HT);
	groupConfig[grpId].baseACh = baseACh;
	groupConfig[grpId].decayACh = 1.0 - (1.0 / tauACh);
	groupConfig[grpId].baseNE	= baseNE;
	groupConfig[grpId].decayNE = 1.0 - (1.0 / tauNE);
}

// set ESTDP params
void SNN::setESTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma) {
	assert(grpId>=-1);
	if (isSet) {
		assert(type!=UNKNOWN_STDP);
		assert(tauPlus>0.0f); assert(tauMinus>0.0f); assert(gamma>=0.0f);
	}

	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1++) {
			setESTDP(grpId1, isSet, type, curve, alphaPlus, tauPlus, alphaMinus, tauMinus, gamma);
		}
	} else {
		// set STDP for a given group
		// set params for STDP curve
		groupConfig[grpId].ALPHA_PLUS_EXC 		= alphaPlus;
		groupConfig[grpId].ALPHA_MINUS_EXC 	= alphaMinus;
		groupConfig[grpId].TAU_PLUS_INV_EXC 	= 1.0f/tauPlus;
		groupConfig[grpId].TAU_MINUS_INV_EXC	= 1.0f/tauMinus;
		groupConfig[grpId].GAMMA				= gamma;
		groupConfig[grpId].KAPPA				= (1 + exp(-gamma/tauPlus))/(1 - exp(-gamma/tauPlus));
		groupConfig[grpId].OMEGA				= alphaPlus * (1 - groupConfig[grpId].KAPPA);
		// set flags for STDP function
		groupConfig[grpId].WithESTDPtype	= type;
		groupConfig[grpId].WithESTDPcurve = curve;
		groupConfig[grpId].WithESTDP		= isSet;
		groupConfig[grpId].WithSTDP		|= groupConfig[grpId].WithESTDP;
		sim_with_stdp					|= groupConfig[grpId].WithSTDP;

		KERNEL_INFO("E-STDP %s for %s(%d)", isSet?"enabled":"disabled", groupInfo[grpId].Name.c_str(), grpId);
	}
}

// set ISTDP params
void SNN::setISTDP(int grpId, bool isSet, STDPType type, STDPCurve curve, float ab1, float ab2, float tau1, float tau2) {
	assert(grpId>=-1);
	if (isSet) {
		assert(type!=UNKNOWN_STDP);
		assert(tau1>0); assert(tau2>0);
	}

	if (grpId==ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1++) {
			setISTDP(grpId1, isSet, type, curve, ab1, ab2, tau1, tau2);
		}
	} else {
		// set STDP for a given group
		// set params for STDP curve
		if (curve == EXP_CURVE) {
			groupConfig[grpId].ALPHA_PLUS_INB = ab1;
			groupConfig[grpId].ALPHA_MINUS_INB = ab2;
			groupConfig[grpId].TAU_PLUS_INV_INB = 1.0f / tau1;
			groupConfig[grpId].TAU_MINUS_INV_INB = 1.0f / tau2;
			groupConfig[grpId].BETA_LTP 		= 0.0f;
			groupConfig[grpId].BETA_LTD 		= 0.0f;
			groupConfig[grpId].LAMBDA			= 1.0f;
			groupConfig[grpId].DELTA			= 1.0f;
		} else {
			groupConfig[grpId].ALPHA_PLUS_INB = 0.0f;
			groupConfig[grpId].ALPHA_MINUS_INB = 0.0f;
			groupConfig[grpId].TAU_PLUS_INV_INB = 1.0f;
			groupConfig[grpId].TAU_MINUS_INV_INB = 1.0f;
			groupConfig[grpId].BETA_LTP 		= ab1;
			groupConfig[grpId].BETA_LTD 		= ab2;
			groupConfig[grpId].LAMBDA			= tau1;
			groupConfig[grpId].DELTA			= tau2;
		}
		// set flags for STDP function
		//FIXME: separate STDPType to ESTDPType and ISTDPType
		groupConfig[grpId].WithISTDPtype	= type;
		groupConfig[grpId].WithISTDPcurve = curve;
		groupConfig[grpId].WithISTDP		= isSet;
		groupConfig[grpId].WithSTDP		|= groupConfig[grpId].WithISTDP;
		sim_with_stdp					|= groupConfig[grpId].WithSTDP;

		KERNEL_INFO("I-STDP %s for %s(%d)", isSet?"enabled":"disabled", groupInfo[grpId].Name.c_str(), grpId);
	}
}

// set STP params
void SNN::setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x) {
	assert(grpId>=-1);
	if (isSet) {
		assert(STP_U>0 && STP_U<=1); assert(STP_tau_u>0); assert(STP_tau_x>0);
	}

	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1++) {
			setSTP(grpId1, isSet, STP_U, STP_tau_u, STP_tau_x);
		}
	} else {
		// set STDP for a given group
		sim_with_stp 				   |= isSet;
		groupConfig[grpId].WithSTP 		= isSet;
		groupConfig[grpId].STP_A 			= (STP_U>0.0f) ? 1.0/STP_U : 1.0f; // scaling factor
		groupConfig[grpId].STP_U 			= STP_U;
		groupConfig[grpId].STP_tau_u_inv	= 1.0f/STP_tau_u; // facilitatory
		groupConfig[grpId].STP_tau_x_inv	= 1.0f/STP_tau_x; // depressive
		groupConfig[grpId].newUpdates = true;

		KERNEL_INFO("STP %s for %d (%s):\tA: %1.4f, U: %1.4f, tau_u: %4.0f, tau_x: %4.0f", isSet?"enabled":"disabled",
					grpId, groupInfo[grpId].Name.c_str(), groupConfig[grpId].STP_A, STP_U, STP_tau_u, STP_tau_x);
	}
}

void SNN::setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay) {
	assert(wtChangeDecay > 0.0f && wtChangeDecay < 1.0f);

	switch (wtANDwtChangeUpdateInterval) {
		case INTERVAL_10MS:
			wtANDwtChangeUpdateInterval_ = 10;
			break;
		case INTERVAL_100MS:
			wtANDwtChangeUpdateInterval_ = 100;
			break;
		case INTERVAL_1000MS:
		default:
			wtANDwtChangeUpdateInterval_ = 1000;
			break;
	}

	if (enableWtChangeDecay) {
		// set up stdp factor according to update interval
		switch (wtANDwtChangeUpdateInterval) {
		case INTERVAL_10MS:
			stdpScaleFactor_ = 0.005f;
			break;
		case INTERVAL_100MS:
			stdpScaleFactor_ = 0.05f;
			break;
		case INTERVAL_1000MS:
		default:
			stdpScaleFactor_ = 0.5f;
			break;
		}
		// set up weight decay
		wtChangeDecay_ = wtChangeDecay;
	} else {
		stdpScaleFactor_ = 1.0f;
		wtChangeDecay_ = 0.0f;
	}

	KERNEL_INFO("Update weight and weight change every %d ms", wtANDwtChangeUpdateInterval_);
	KERNEL_INFO("Weight Change Decay is %s", enableWtChangeDecay? "enabled" : "disable");
	KERNEL_INFO("STDP scale factor = %1.3f, wtChangeDecay = %1.3f", stdpScaleFactor_, wtChangeDecay_);
}

/// ************************************************************************************************************ ///
/// PUBLIC METHODS: GENERATE A SIMULATION
/// ************************************************************************************************************ ///

// reorganize the network and do the necessary allocation
// of all variable for carrying out the simulation..
// this code is run only one time during network initialization
void SNN::setupNetwork(bool removeTempMem) {
	switch (snnState) {
	case CONFIG_SNN:
		compileSNN(removeTempMem);
	case COMPILED_SNN:
		linkSNN();
	case LINKED_SNN:
		optimizeAndPartitionSNN();
	case OPTIMIZED_PARTITIONED_SNN:
		allocateSNN();
		break;
	case EXECUTABLE_SNN:
		break;
	default:
		KERNEL_ERROR("Unknown SNN state");
		break;
	}
}

/// ************************************************************************************************************ ///
/// PUBLIC METHODS: RUNNING A SIMULATION
/// ************************************************************************************************************ ///

int SNN::runNetwork(int _nsec, int _nmsec, bool printRunSummary, bool copyState) {
	assert(_nmsec >= 0 && _nmsec < 1000);
	assert(_nsec  >= 0);
	int runDurationMs = _nsec*1000 + _nmsec;
	KERNEL_DEBUG("runNetwork: runDur=%dms, printRunSummary=%s, copyState=%s", runDurationMs, printRunSummary?"y":"n",
		copyState?"y":"n");

	// setupNetwork() must have already been called
	assert(snnState == EXECUTABLE_SNN);

	// don't bother printing if logger mode is SILENT
	printRunSummary = (loggerMode_==SILENT) ? false : printRunSummary;

	// first-time run: inform the user the simulation is running now
	if (simTime==0 && printRunSummary) {
		KERNEL_INFO("");
		if (simMode_==GPU_MODE) {
			KERNEL_INFO("******************** Running GPU Simulation on GPU %d ***************************",
			ithGPU_);
		} else {
			KERNEL_INFO("********************      Running CPU Simulation      ***************************");
		}
		KERNEL_INFO("");
	}

	// reset all spike counters
	if (simMode_==GPU_MODE)
		resetSpikeCnt_GPU(0,numGrp);
	else
		resetSpikeCnt(ALL);

	// store current start time for future reference
	simTimeRunStart = simTime;
	simTimeRunStop  = simTime+runDurationMs;
	assert(simTimeRunStop>=simTimeRunStart); // check for arithmetic underflow

	// ConnectionMonitor is a special case: we might want the first snapshot at t=0 in the binary
	// but updateTime() is false for simTime==0.
	// And we cannot put this code in ConnectionMonitorCore::init, because then the user would have no
	// way to call ConnectionMonitor::setUpdateTimeIntervalSec before...
	if (simTime==0 && numConnectionMonitor) {
		updateConnectionMonitor();
	}

	// set the Poisson generation time slice to be at the run duration up to PROPOGATED_BUFFER_SIZE ms.
	// \TODO: should it be PROPAGATED_BUFFER_SIZE-1 or PROPAGATED_BUFFER_SIZE ?
	setGrpTimeSlice(ALL, MAX(1,MIN(runDurationMs,PROPAGATED_BUFFER_SIZE-1)));

	CUDA_RESET_TIMER(timer);
	CUDA_START_TIMER(timer);

	// if nsec=0, simTimeMs=10, we need to run the simulator for 10 timeStep;
	// if nsec=1, simTimeMs=10, we need to run the simulator for 1*1000+10, time Step;
	for(int i=0; i<runDurationMs; i++) {
		if(simMode_ == CPU_MODE)
			doSnnSim();
		else
			doGPUSim();

		// update weight every updateInterval ms if plastic synapses present
		if (!sim_with_fixedwts && wtANDwtChangeUpdateInterval_ == ++wtANDwtChangeUpdateIntervalCnt_) {
			wtANDwtChangeUpdateIntervalCnt_ = 0; // reset counter
			if (!sim_in_testing) {
				// keep this if statement separate from the above, so that the counter is updated correctly
				if (simMode_ == CPU_MODE) {
					updateWeights();
				} else{
					updateWeights_GPU();
				}
			}
		}

		// Note: updateTime() advance simTime, simTimeMs, and simTimeSec accordingly
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

		if(simMode_ == GPU_MODE){
			copyFiringStateFromGPU();
		}
	}

	// in GPU mode, copy info from device to host
	if (simMode_==GPU_MODE) {
		if(copyState) {
			copyNeuronState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, ALL);

			if (sim_with_stp) {
				copySTPState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
			}
		}
	}

	// user can opt to display some runNetwork summary
	if (printRunSummary) {

		// if there are Monitors available and it's time to show the log, print status for each group
		if (numSpikeMonitor) {
			printStatusSpikeMonitor(ALL);
		}
		if (numConnectionMonitor) {
			printStatusConnectionMonitor(ALL);
		}
		if (numGroupMonitor) {
			printStatusGroupMonitor(ALL);
		}

		// record time of run summary print
		simTimeLastRunSummary = simTime;
	}

	// call updateSpike(Group)Monitor again to fetch all the left-over spikes and group status (neuromodulator)
	updateSpikeMonitor();
	updateGroupMonitor();

	// keep track of simulation time...
	CUDA_STOP_TIMER(timer);
	lastExecutionTime = CUDA_GET_TIMER_VALUE(timer);
	cumExecutionTime += lastExecutionTime;
	return 0;
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: INTERACTING WITH A SIMULATION
/// ************************************************************************************************************ ///

// adds a bias to every weight in the connection
void SNN::biasWeights(short int connId, float bias, bool updateWeightRange) {
	assert(connId>=0 && connId<numConnections);

	// iterate over all postsynaptic neurons
	for (int i=groupConfig[connectConfigMap[connId].grpDest].StartN; i<=groupConfig[connectConfigMap[connId].grpDest].EndN; i++) {
		unsigned int cumIdx = snnRuntimeData.cumulativePre[i];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j=0; j<snnRuntimeData.Npre[i]; pos_ij++, j++) {
			if (snnRuntimeData.cumConnIdPre[pos_ij]==connId) {
				// apply bias to weight
				float weight = snnRuntimeData.wt[pos_ij] + bias;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight+bias>connInfo->maxWt || weight+bias<connInfo->minWt);
				bool needToPrintDebug = (weight > connectConfigMap[connId].maxWt || weight < 0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connectConfigMap[connId].maxWt = fmax(connectConfigMap[connId].maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), 0.0f, connectConfigMap[connId].maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = fmin(weight, connectConfigMap[connId].maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = fmax(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), weight, 0.0f, connectConfigMap[connId].maxWt);
					}
				}

				// update datastructures
				snnRuntimeData.wt[pos_ij] = weight;
				snnRuntimeData.maxSynWt[pos_ij] = connectConfigMap[connId].maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (simMode_==GPU_MODE) {
			CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[cumIdx]), &(snnRuntimeData.wt[cumIdx]), sizeof(float)*snnRuntimeData.Npre[i],
				cudaMemcpyHostToDevice) );

			if (gpuRuntimeData.maxSynWt!=NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[cumIdx]), &(snnRuntimeData.maxSynWt[cumIdx]),
					sizeof(float)*snnRuntimeData.Npre[i], cudaMemcpyHostToDevice) );
			}
		}
	}
}

// deallocates dynamical structures and exits
void SNN::exitSimulation(int val) {
	deleteObjects();
	exit(val);
}

// reads network state from file
void SNN::loadSimulation(FILE* fid) {
	loadSimFID = fid;
}

// reset spike counter to zero
void SNN::resetSpikeCounter(int grpId) {
	if (!sim_with_spikecounters)
		return;

	assert(grpId>=-1); assert(grpId<numGrp);

	if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1<numGrp; grpId1 ++) {
			resetSpikeCounter(grpId1);
		}
	} else {
		// only update if SpikeMonRT is set for this group
		if (!groupConfig[grpId].withSpikeCounter)
			return;

		groupConfig[grpId].spkCntRecordDurHelper = 0;

		if (simMode_==GPU_MODE) {
			resetSpikeCounter_GPU(grpId);
		}
		else {
			int bufPos = groupConfig[grpId].spkCntBufPos; // retrieve buf pos
			memset(spkCntBuf[bufPos],0,groupConfig[grpId].SizeN*sizeof(int)); // set all to 0
		}
	}
}

// multiplies every weight with a scaling factor
void SNN::scaleWeights(short int connId, float scale, bool updateWeightRange) {
	assert(connId>=0 && connId<numConnections);
	assert(scale>=0.0f);

	//ConnectConfig* connInfo = getConnectInfo(connId);

	// iterate over all postsynaptic neurons
	for (int i=groupConfig[connectConfigMap[connId].grpDest].StartN; i<=groupConfig[connectConfigMap[connId].grpDest].EndN; i++) {
		unsigned int cumIdx = snnRuntimeData.cumulativePre[i];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j=0; j<snnRuntimeData.Npre[i]; pos_ij++, j++) {
			if (snnRuntimeData.cumConnIdPre[pos_ij]==connId) {
				// apply bias to weight
				float weight = snnRuntimeData.wt[pos_ij]*scale;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight>connInfo->maxWt || weight<connInfo->minWt);
				bool needToPrintDebug = (weight > connectConfigMap[connId].maxWt || weight<0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connectConfigMap[connId].maxWt = fmax(connectConfigMap[connId].maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), 0.0f, connectConfigMap[connId].maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = fmin(weight, connectConfigMap[connId].maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = fmax(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), weight, 0.0f, connectConfigMap[connId].maxWt);
					}
				}

				// update datastructures
				snnRuntimeData.wt[pos_ij] = weight;
				snnRuntimeData.maxSynWt[pos_ij] = connectConfigMap[connId].maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (simMode_==GPU_MODE) {
			CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[cumIdx]), &(snnRuntimeData.wt[cumIdx]), sizeof(float)*snnRuntimeData.Npre[i],
				cudaMemcpyHostToDevice) );

			if (gpuRuntimeData.maxSynWt!=NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[cumIdx]), &(snnRuntimeData.maxSynWt[cumIdx]),
					sizeof(float)*snnRuntimeData.Npre[i], cudaMemcpyHostToDevice));
			}
		}
	}
}

GroupMonitor* SNN::setGroupMonitor(int grpId, FILE* fid) {
	// check whether group already has a GroupMonitor
	if (groupConfig[grpId].GroupMonitorId >= 0) {
		KERNEL_ERROR("setGroupMonitor has already been called on Group %d (%s).",
			grpId, groupInfo[grpId].Name.c_str());
		exitSimulation(1);
	}

	// create new GroupMonitorCore object in any case and initialize analysis components
	// grpMonObj destructor (see below) will deallocate it
	GroupMonitorCore* grpMonCoreObj = new GroupMonitorCore(this, numGroupMonitor, grpId);
	groupMonCoreList[numGroupMonitor] = grpMonCoreObj;

	// assign group status file ID if we selected to write to a file, else it's NULL
	// if file pointer exists, it has already been fopened
	// this will also write the header section of the group status file
	// grpMonCoreObj destructor will fclose it
	grpMonCoreObj->setGroupFileId(fid);

	// create a new GroupMonitor object for the user-interface
	// SNN::deleteObjects will deallocate it
	GroupMonitor* grpMonObj = new GroupMonitor(grpMonCoreObj);
	groupMonList[numGroupMonitor] = grpMonObj;

	// also inform the group that it is being monitored...
	groupConfig[grpId].GroupMonitorId = numGroupMonitor;

    // not eating much memory anymore, got rid of all buffers
	cpuSnnSz.monitorInfoSize += sizeof(GroupMonitor*);
	cpuSnnSz.monitorInfoSize += sizeof(GroupMonitorCore*);

	numGroupMonitor++;
	KERNEL_INFO("GroupMonitor set for group %d (%s)",grpId,groupInfo[grpId].Name.c_str());

	return grpMonObj;
}

ConnectionMonitor* SNN::setConnectionMonitor(int grpIdPre, int grpIdPost, FILE* fid) {
	// find connection based on pre-post pair
	short int connId = getConnectId(grpIdPre,grpIdPost);
	if (connId<0) {
		KERNEL_ERROR("No connection found from group %d(%s) to group %d(%s)", grpIdPre, getGroupName(grpIdPre).c_str(),
			grpIdPost, getGroupName(grpIdPost).c_str());
		exitSimulation(1);
	}

	// check whether connection already has a connection monitor
	//ConnectConfig* connInfo = getConnectInfo(connId);
	if (connectConfigMap[connId].connectionMonitorId >= 0) {
		KERNEL_ERROR("setConnectionMonitor has already been called on Connection %d (MonitorId=%d)", connId, connectConfigMap[connId].connectionMonitorId);
		exitSimulation(1);
	}

	// inform the connection that it is being monitored...
	// this needs to be called before new ConnectionMonitorCore
	connectConfigMap[connId].connectionMonitorId = numConnectionMonitor;

	// create new ConnectionMonitorCore object in any case and initialize
	// connMonObj destructor (see below) will deallocate it
	ConnectionMonitorCore* connMonCoreObj = new ConnectionMonitorCore(this, numConnectionMonitor, connId,
		grpIdPre, grpIdPost);
	connMonCoreList[numConnectionMonitor] = connMonCoreObj;

	// assign conn file ID if we selected to write to a file, else it's NULL
	// if file pointer exists, it has already been fopened
	// this will also write the header section of the conn file
	// connMonCoreObj destructor will fclose it
	connMonCoreObj->setConnectFileId(fid);

	// create a new ConnectionMonitor object for the user-interface
	// SNN::deleteObjects will deallocate it
	ConnectionMonitor* connMonObj = new ConnectionMonitor(connMonCoreObj);
	connMonList[numConnectionMonitor] = connMonObj;

	// now init core object (depends on several datastructures allocated above)
	connMonCoreObj->init();

    // not eating much memory anymore, got rid of all buffers
	cpuSnnSz.monitorInfoSize += sizeof(ConnectionMonitor*);
	cpuSnnSz.monitorInfoSize += sizeof(ConnectionMonitorCore*);

	numConnectionMonitor++;
	KERNEL_INFO("ConnectionMonitor %d set for Connection %d: %d(%s) => %d(%s)", connectConfigMap[connId].connectionMonitorId, connId, grpIdPre, getGroupName(grpIdPre).c_str(),
		grpIdPost, getGroupName(grpIdPost).c_str());

	return connMonObj;
}

void SNN::setExternalCurrent(int grpId, const std::vector<float>& current) {
	assert(grpId>=0); assert(grpId<numGrp);
	assert(!isPoissonGroup(grpId));
	assert(current.size() == getGroupNumNeurons(grpId));

	// // update flag for faster handling at run-time
	// if (count_if(current.begin(), current.end(), isGreaterThanZero)) {
	// 	groupConfig[grpId].WithCurrentInjection = true;
	// } else {
	// 	groupConfig[grpId].WithCurrentInjection = false;
	// }

	// store external current in array
	for (int i=groupConfig[grpId].StartN, j=0; i<=groupConfig[grpId].EndN; i++, j++) {
		snnRuntimeData.extCurrent[i] = current[j];
	}

	// copy to GPU if necessary
	// don't allocate; allocation done in buildNetwork
	if (simMode_==GPU_MODE) {
		copyExternalCurrent(&gpuRuntimeData, &snnRuntimeData, false, grpId);
	}
}

// sets up a spike generator
void SNN::setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGen) {
	assert(snnState == CONFIG_SNN); // must be called before setupNetwork() to work on GPU
	assert(spikeGen);
	assert (groupConfig[grpId].isSpikeGenerator);
	groupConfig[grpId].spikeGen = spikeGen;
}

// A Spike Counter keeps track of the number of spikes per neuron in a group.
void SNN::setSpikeCounter(int grpId, int recordDur) {
	assert(grpId>=0); assert(grpId<numGrp);

	sim_with_spikecounters = true; // inform simulation
	groupConfig[grpId].withSpikeCounter = true; // inform the group
	groupConfig[grpId].spkCntRecordDur = (recordDur>0)?recordDur:-1; // set record duration, after which spike buf will be reset
	groupConfig[grpId].spkCntRecordDurHelper = 0; // counter to help make fast modulo
	groupConfig[grpId].spkCntBufPos = numSpkCnt; // inform group which pos it has in spike buf
	spkCntBuf[numSpkCnt] = new int[groupConfig[grpId].SizeN]; // create spike buf
	memset(spkCntBuf[numSpkCnt],0,(groupConfig[grpId].SizeN)*sizeof(int)); // set all to 0

	numSpkCnt++;

	KERNEL_INFO("SpikeCounter set for Group %d (%s): %d ms recording window", grpId, groupInfo[grpId].Name.c_str(),
		recordDur);
}

// record spike information, return a SpikeInfo object
SpikeMonitor* SNN::setSpikeMonitor(int grpId, FILE* fid) {
	// check whether group already has a SpikeMonitor
	if (groupConfig[grpId].SpikeMonitorId >= 0) {
		// in this case, return the current object and update fid
		SpikeMonitor* spkMonObj = getSpikeMonitor(grpId);

		// update spike file ID
		SpikeMonitorCore* spkMonCoreObj = getSpikeMonitorCore(grpId);
		spkMonCoreObj->setSpikeFileId(fid);

		KERNEL_INFO("SpikeMonitor updated for group %d (%s)",grpId,groupInfo[grpId].Name.c_str());
		return spkMonObj;
	} else {
		// create new SpikeMonitorCore object in any case and initialize analysis components
		// spkMonObj destructor (see below) will deallocate it
		SpikeMonitorCore* spkMonCoreObj = new SpikeMonitorCore(this, numSpikeMonitor, grpId);
		spikeMonCoreList[numSpikeMonitor] = spkMonCoreObj;

		// assign spike file ID if we selected to write to a file, else it's NULL
		// if file pointer exists, it has already been fopened
		// this will also write the header section of the spike file
		// spkMonCoreObj destructor will fclose it
		spkMonCoreObj->setSpikeFileId(fid);

		// create a new SpikeMonitor object for the user-interface
		// SNN::deleteObjects will deallocate it
		SpikeMonitor* spkMonObj = new SpikeMonitor(spkMonCoreObj);
		spikeMonList[numSpikeMonitor] = spkMonObj;

		// also inform the grp that it is being monitored...
		groupConfig[grpId].SpikeMonitorId	= numSpikeMonitor;

    	// not eating much memory anymore, got rid of all buffers
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitor*);
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitorCore*);

		numSpikeMonitor++;
		KERNEL_INFO("SpikeMonitor set for group %d (%s)",grpId,groupInfo[grpId].Name.c_str());

		return spkMonObj;
	}
}

// assigns spike rate to group
void SNN::setSpikeRate(int grpId, PoissonRate* ratePtr, int refPeriod) {
	assert(grpId>=0 && grpId<numGrp);
	assert(ratePtr);
	assert(groupConfig[grpId].isSpikeGenerator);
	assert(ratePtr->getNumNeurons()==groupConfig[grpId].SizeN);
	assert(refPeriod>=1);

	groupConfig[grpId].RatePtr = ratePtr;
	groupConfig[grpId].RefractPeriod   = refPeriod;
	spikeRateUpdated = true;
}

// sets the weight value of a specific synapse
void SNN::setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange) {
	assert(connId>=0 && connId<getNumConnections());
	assert(weight>=0.0f);

	//ConnectConfig* connInfo = getConnectInfo(connId);
	assert(neurIdPre>=0  && neurIdPre<getGroupNumNeurons(connectConfigMap[connId].grpSrc));
	assert(neurIdPost>=0 && neurIdPost<getGroupNumNeurons(connectConfigMap[connId].grpDest));

	float maxWt = fabs(connectConfigMap[connId].maxWt);
	float minWt = 0.0f;

	// inform user of acton taken if weight is out of bounds
	bool needToPrintDebug = (weight>maxWt || weight<minWt);

	if (updateWeightRange) {
		// if this flag is set, we need to update minWt,maxWt accordingly
		// will be saving new maxSynWt and copying to GPU below
//		connInfo->minWt = fmin(connInfo->minWt, weight);
		maxWt = fmax(maxWt, weight);
		if (needToPrintDebug) {
			KERNEL_DEBUG("setWeight(%d,%d,%d,%f,%s): updated weight ranges to [%f,%f]", connId, neurIdPre, neurIdPost,
				weight, (updateWeightRange?"true":"false"), minWt, maxWt);
		}
	} else {
		// constrain weight to boundary values
		// compared to above, we swap minWt/maxWt logic
		weight = fmin(weight, maxWt);
		weight = fmax(weight, minWt);
		if (needToPrintDebug) {
			KERNEL_DEBUG("setWeight(%d,%d,%d,%f,%s): constrained weight %f to [%f,%f]", connId, neurIdPre, neurIdPost,
				weight, (updateWeightRange?"true":"false"), weight, minWt, maxWt);
		}
	}

	// find real ID of pre- and post-neuron
	int neurIdPreReal = groupConfig[connectConfigMap[connId].grpSrc].StartN+neurIdPre;
	int neurIdPostReal = groupConfig[connectConfigMap[connId].grpDest].StartN+neurIdPost;

	// iterate over all presynaptic synapses until right one is found
	bool synapseFound = false;
	int pos_ij = snnRuntimeData.cumulativePre[neurIdPostReal];
	for (int j=0; j<snnRuntimeData.Npre[neurIdPostReal]; pos_ij++, j++) {
		post_info_t* preId = &(snnRuntimeData.preSynapticIds[pos_ij]);
		int pre_nid = GET_CONN_NEURON_ID((*preId));
		if (GET_CONN_NEURON_ID((*preId))==neurIdPreReal) {
			assert(snnRuntimeData.cumConnIdPre[pos_ij]==connId); // make sure we've got the right connection ID

			snnRuntimeData.wt[pos_ij] = isExcitatoryGroup(connectConfigMap[connId].grpSrc) ? weight : -1.0*weight;
			snnRuntimeData.maxSynWt[pos_ij] = isExcitatoryGroup(connectConfigMap[connId].grpSrc) ? maxWt : -1.0*maxWt;

			if (simMode_==GPU_MODE) {
				// need to update datastructures on GPU
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[pos_ij]), &(snnRuntimeData.wt[pos_ij]), sizeof(float), cudaMemcpyHostToDevice));
				if (gpuRuntimeData.maxSynWt!=NULL) {
					// only copy maxSynWt if datastructure actually exists on the GPU
					// (that logic should be done elsewhere though)
					CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[pos_ij]), &(snnRuntimeData.maxSynWt[pos_ij]), sizeof(float), cudaMemcpyHostToDevice));
				}
			}

			// synapse found and updated: we're done!
			synapseFound = true;
			break;
		}
	}

	if (!synapseFound) {
		KERNEL_WARN("setWeight(%d,%d,%d,%f,%s): Synapse does not exist, not updated.", connId, neurIdPre, neurIdPost,
			weight, (updateWeightRange?"true":"false"));
	}
}


// writes network state to file
// handling of file pointer should be handled externally: as far as this function is concerned, it is simply
// trying to write to file
void SNN::saveSimulation(FILE* fid, bool saveSynapseInfo) {
	int tmpInt;
	float tmpFloat;

	// +++++ WRITE HEADER SECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// write file signature
	tmpInt = 294338571; // some int used to identify saveSimulation files
	if (!fwrite(&tmpInt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	// write version number
	tmpFloat = 0.2f;
	if (!fwrite(&tmpFloat,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	// write simulation time so far (in seconds)
	tmpFloat = ((float)simTimeSec) + ((float)simTimeMs)/1000.0f;
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	// write execution time so far (in seconds)
	if(simMode_ == GPU_MODE) {
		stopGPUTiming();
		tmpFloat = gpuExecutionTime/1000.0f;
	} else {
		stopCPUTiming();
		tmpFloat = cpuExecutionTime/1000.0f;
	}
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	// TODO: add more params of interest

	// write network info
	if (!fwrite(&numN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&preSynCnt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&postSynCnt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&numGrp,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	// write group info
	char name[100];
	for (int g=0;g<numGrp;g++) {
		if (!fwrite(&groupConfig[g].StartN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfig[g].EndN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		if (!fwrite(&groupConfig[g].SizeX,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfig[g].SizeY,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfig[g].SizeZ,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		strncpy(name,groupInfo[g].Name.c_str(),100);
		if (!fwrite(name,1,100,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	}

	// +++++ Fetch WEIGHT DATA (GPU Mode only) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	if (simMode_ == GPU_MODE)
		copyWeightState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
	// +++++ WRITE SYNAPSE INFO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// \FIXME: replace with faster version
	if (saveSynapseInfo) {
		for (unsigned int i=0;i<numN;i++) {
			unsigned int offset = snnRuntimeData.cumulativePost[i];

			unsigned int count = 0;
			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++)
					count++;
			}

			if (!fwrite(&count,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
					// get synaptic info...
					post_info_t post_info = snnRuntimeData.postSynapticIds[offset + idx_d];

					// get neuron id
					//int p_i = (post_info&POST_SYN_NEURON_MASK);
					unsigned int p_i = GET_CONN_NEURON_ID(post_info);
					assert(p_i<numN);

					// get syn id
					unsigned int s_i = GET_CONN_SYN_ID(post_info);
					//>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;
					assert(s_i<(snnRuntimeData.Npre[p_i]));

					// get the cumulative position for quick access...
					unsigned int pos_i = snnRuntimeData.cumulativePre[p_i] + s_i;

					uint8_t delay = t+1;
					uint8_t plastic = s_i < snnRuntimeData.Npre_plastic[p_i]; // plastic or fixed.

					if (!fwrite(&i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&p_i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(snnRuntimeData.wt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(snnRuntimeData.maxSynWt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&delay,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&plastic,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(snnRuntimeData.cumConnIdPre[pos_i]),sizeof(short int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
				}
			}
		}
	}
}

// writes population weights from gIDpre to gIDpost to file fname in binary
void SNN::writePopWeights(std::string fname, int grpIdPre, int grpIdPost) {
	assert(grpIdPre>=0); assert(grpIdPost>=0);

	float* weights;
	int matrixSize;
	FILE* fid;
	int numPre, numPost;
	fid = fopen(fname.c_str(), "wb");
	assert(fid != NULL);

	if(snnState == CONFIG_SNN || snnState == COMPILED_SNN || snnState == LINKED_SNN){
		KERNEL_ERROR("Simulation has not been run yet, cannot output weights.");
		exitSimulation(1);
	}

	post_info_t* preId;
	int pre_nid, pos_ij;

	//population sizes
	numPre = groupConfig[grpIdPre].SizeN;
	numPost = groupConfig[grpIdPost].SizeN;

	//first iteration gets the number of synaptic weights to place in our
	//weight matrix.
	matrixSize=0;
	//iterate over all neurons in the post group
	for (int i=groupConfig[grpIdPost].StartN; i<=groupConfig[grpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = snnRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
		//iterate over all presynaptic synapses
		for(int j=0; j<snnRuntimeData.Npre[i]; pos_ij++,j++) {
			preId = &snnRuntimeData.preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<groupConfig[grpIdPre].StartN || pre_nid>groupConfig[grpIdPre].EndN)
				continue; // connection does not belong to group grpIdPre
			matrixSize++;
		}
	}

	//now we have the correct size
	weights = new float[matrixSize];
	//second iteration assigns the weights
	int curr = 0; // iterator for return array
	//iterate over all neurons in the post group
	for (int i=groupConfig[grpIdPost].StartN; i<=groupConfig[grpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = snnRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
		//do the GPU copy here.  Copy the current weights from GPU to CPU.
		if(simMode_==GPU_MODE){
			copyWeightsGPU(i,grpIdPre);
		}
		//iterate over all presynaptic synapses
		for(int j=0; j<snnRuntimeData.Npre[i]; pos_ij++,j++) {
			preId = &(snnRuntimeData.preSynapticIds[pos_ij]);
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<groupConfig[grpIdPre].StartN || pre_nid>groupConfig[grpIdPre].EndN)
				continue; // connection does not belong to group grpIdPre
			weights[curr] = snnRuntimeData.wt[pos_ij];
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

// set new file pointer for all files
// fp==NULL is code for don't change it
// can be called in all logger modes; however, the analogous interface function can only be called in CUSTOM
void SNN::setLogsFp(FILE* fpInf, FILE* fpErr, FILE* fpDeb, FILE* fpLog) {
	if (fpInf!=NULL) {
		if (fpInf_!=NULL && fpInf_!=stdout && fpInf_!=stderr)
			fclose(fpInf_);
		fpInf_ = fpInf;
	}

	if (fpErr!=NULL) {
		if (fpErr_ != NULL && fpErr_!=stdout && fpErr_!=stderr)
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

// loop over linked list entries to find a connection with the right pre-post pair, O(N)
short int SNN::getConnectId(int grpIdPre, int grpIdPost) {
	//ConnectConfig* connInfo = connectBegin;

	short int connId = -1;
	//while (connInfo) {
	//	// check whether pre and post match
	//	if (connInfo->grpSrc == grpIdPre && connInfo->grpDest == grpIdPost) {
	//		connId = connInfo->connId;
	//		break;
	//	}

	//	// otherwise, keep looking
	//	connInfo = connInfo->next;
	//}

	for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
		if (it->second.grpSrc == grpIdPre && it->second.grpDest == grpIdPost) {
			connId = it->second.connId;
			break;
		}
	}

	return connId;
}

//! used for parameter tuning functionality
//ConnectConfig* SNN::getConnectInfo(short int connectId) {
//	ConnectConfig* nextConn = connectBegin;
//	CHECK_CONNECTION_ID(connectId, numConnections);
//
//	// clear all existing connection info...
//	while (nextConn) {
//		if (nextConn->connId == connectId) {
//			nextConn->newUpdates = true;		// \FIXME: this is a Jay hack
//			return nextConn;
//		}
//		nextConn = nextConn->next;
//	}
//
//	KERNEL_DEBUG("Total Connections = %d", numConnections);
//	KERNEL_DEBUG("ConnectId (%d) cannot be recognized", connectId);
//	return NULL;
//}

ConnectConfig SNN::getConnectConfig(short int connId) {
	//ConnectConfig* nextConn = connectBegin;
	CHECK_CONNECTION_ID(connId, numConnections);

	if (connectConfigMap.find(connId) == connectConfigMap.end()) {
		KERNEL_ERROR("Total Connections = %d", numConnections);
		KERNEL_ERROR("ConnectId (%d) cannot be recognized", connId);
	}
	
	connectConfigMap[connId].newUpdates = true;// \FIXME: this is a Jay hack

	return connectConfigMap[connId];
}

std::vector<float> SNN::getConductanceAMPA(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE) {
		copyConductanceAMPA(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);
	}

	std::vector<float> gAMPAvec;
	for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
		gAMPAvec.push_back(snnRuntimeData.gAMPA[i]);
	}
	return gAMPAvec;
}

std::vector<float> SNN::getConductanceNMDA(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceNMDA(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);

	std::vector<float> gNMDAvec;
	if (isSimulationWithNMDARise()) {
		// need to construct conductance from rise and decay parts
		for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
			gNMDAvec.push_back(snnRuntimeData.gNMDA_d[i]-snnRuntimeData.gNMDA_r[i]);
		}
	} else {
		for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
			gNMDAvec.push_back(snnRuntimeData.gNMDA[i]);
		}
	}
	return gNMDAvec;
}

std::vector<float> SNN::getConductanceGABAa(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE) {
		copyConductanceGABAa(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);
	}

	std::vector<float> gGABAaVec;
	for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
		gGABAaVec.push_back(snnRuntimeData.gGABAa[i]);
	}
	return gGABAaVec;
}

std::vector<float> SNN::getConductanceGABAb(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceGABAb(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);

	std::vector<float> gGABAbVec;
	if (isSimulationWithGABAbRise()) {
		// need to construct conductance from rise and decay parts
		for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
			gGABAbVec.push_back(snnRuntimeData.gGABAb_d[i]-snnRuntimeData.gGABAb_r[i]);
		}
	} else {
		for (int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
			gGABAbVec.push_back(snnRuntimeData.gGABAb[i]);
		}
	}
	return gGABAbVec;
}

// returns RangeDelay struct of a connection
RangeDelay SNN::getDelayRange(short int connId) {
	assert(connId>=0 && connId<numConnections);
	//ConnectConfig* connInfo = getConnectInfo(connId);
	return RangeDelay(connectConfigMap[connId].minDelay, connectConfigMap[connId].maxDelay);
}


// this is a user function
// \FIXME: fix this
uint8_t* SNN::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	Npre = groupConfig[gIDpre].SizeN;
	Npost = groupConfig[gIDpost].SizeN;

	if (delays == NULL) delays = new uint8_t[Npre*Npost];
	memset(delays,0,Npre*Npost);

	for (int i=groupConfig[gIDpre].StartN;i<groupConfig[gIDpre].EndN;i++) {
		unsigned int offset = snnRuntimeData.cumulativePost[i];

		for (int t=0;t<maxDelay_;t++) {
			delay_info_t dPar = snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
				// get synaptic info...
				post_info_t post_info = snnRuntimeData.postSynapticIds[offset + idx_d];

				// get neuron id
				//int p_i = (post_info&POST_SYN_NEURON_MASK);
				int p_i = GET_CONN_NEURON_ID(post_info);
				assert(p_i<numN);

				if (p_i >= groupConfig[gIDpost].StartN && p_i <= groupConfig[gIDpost].EndN) {
					// get syn id
					int s_i = GET_CONN_SYN_ID(post_info);

					// get the cumulative position for quick access...
					unsigned int pos_i = snnRuntimeData.cumulativePre[p_i] + s_i;

					delays[i+Npre*(p_i-groupConfig[gIDpost].StartN)] = t+1;
				}
			}
		}
	}
	return delays;
}

Grid3D SNN::getGroupGrid3D(int grpId) {
	assert(grpId>=0 && grpId<numGrp);
	return Grid3D(groupConfig[grpId].SizeX, groupConfig[grpId].SizeY, groupConfig[grpId].SizeZ);
}

// find ID of group with name grpName
int SNN::getGroupId(std::string grpName) {
	for (int grpId=0; grpId<numGrp; grpId++) {
		if (groupInfo[grpId].Name.compare(grpName)==0)
			return grpId;
	}

	// group not found
	return -1;
}

GroupConfigRT SNN::getGroupConfig(int grpId) {
	assert(grpId>=-1 && grpId<numGrp);
	return groupConfig[grpId];
}

std::string SNN::getGroupName(int grpId) {
	assert(grpId>=-1 && grpId<numGrp);

	if (grpId==ALL)
		return "ALL";

	return groupInfo[grpId].Name;
}

GroupSTDPInfo SNN::getGroupSTDPInfo(int grpId) {
	GroupSTDPInfo gInfo;

	gInfo.WithSTDP = groupConfig[grpId].WithSTDP;
	gInfo.WithESTDP = groupConfig[grpId].WithESTDP;
	gInfo.WithISTDP = groupConfig[grpId].WithISTDP;
	gInfo.WithESTDPtype = groupConfig[grpId].WithESTDPtype;
	gInfo.WithISTDPtype = groupConfig[grpId].WithISTDPtype;
	gInfo.WithESTDPcurve = groupConfig[grpId].WithESTDPcurve;
	gInfo.WithISTDPcurve = groupConfig[grpId].WithISTDPcurve;
	gInfo.ALPHA_MINUS_EXC = groupConfig[grpId].ALPHA_MINUS_EXC;
	gInfo.ALPHA_PLUS_EXC = groupConfig[grpId].ALPHA_PLUS_EXC;
	gInfo.TAU_MINUS_INV_EXC = groupConfig[grpId].TAU_MINUS_INV_EXC;
	gInfo.TAU_PLUS_INV_EXC = groupConfig[grpId].TAU_PLUS_INV_EXC;
	gInfo.ALPHA_MINUS_INB = groupConfig[grpId].ALPHA_MINUS_INB;
	gInfo.ALPHA_PLUS_INB = groupConfig[grpId].ALPHA_PLUS_INB;
	gInfo.TAU_MINUS_INV_INB = groupConfig[grpId].TAU_MINUS_INV_INB;
	gInfo.TAU_PLUS_INV_INB = groupConfig[grpId].TAU_PLUS_INV_INB;
	gInfo.GAMMA = groupConfig[grpId].GAMMA;
	gInfo.BETA_LTP = groupConfig[grpId].BETA_LTP;
	gInfo.BETA_LTD = groupConfig[grpId].BETA_LTD;
	gInfo.LAMBDA = groupConfig[grpId].LAMBDA;
	gInfo.DELTA = groupConfig[grpId].DELTA;

	return gInfo;
}

GroupNeuromodulatorInfo SNN::getGroupNeuromodulatorInfo(int grpId) {
	GroupNeuromodulatorInfo gInfo;

	gInfo.baseDP = groupConfig[grpId].baseDP;
	gInfo.base5HT = groupConfig[grpId].base5HT;
	gInfo.baseACh = groupConfig[grpId].baseACh;
	gInfo.baseNE = groupConfig[grpId].baseNE;
	gInfo.decayDP = groupConfig[grpId].decayDP;
	gInfo.decay5HT = groupConfig[grpId].decay5HT;
	gInfo.decayACh = groupConfig[grpId].decayACh;
	gInfo.decayNE = groupConfig[grpId].decayNE;

	return gInfo;
}

Point3D SNN::getNeuronLocation3D(int neurId) {
	assert(neurId>=0 && neurId<numN);
	int grpId = snnRuntimeData.grpIds[neurId];
	assert(neurId>=groupConfig[grpId].StartN && neurId<=groupConfig[grpId].EndN);

	// adjust neurId for neuron ID of first neuron in the group
	neurId -= groupConfig[grpId].StartN;

	return getNeuronLocation3D(grpId, neurId);
}

Point3D SNN::getNeuronLocation3D(int grpId, int relNeurId) {
	assert(grpId>=0 && grpId<numGrp);
	assert(relNeurId>=0 && relNeurId<getGroupNumNeurons(grpId));

	// coordinates are in x e[-SizeX/2,SizeX/2], y e[-SizeY/2,SizeY/2], z e[-SizeZ/2,SizeZ/2]
	// instead of x e[0,SizeX], etc.
	int intX = relNeurId % groupConfig[grpId].SizeX;
	int intY = (relNeurId/groupConfig[grpId].SizeX)%groupConfig[grpId].SizeY;
	int intZ = relNeurId/(groupConfig[grpId].SizeX*groupConfig[grpId].SizeY);

	// so subtract SizeX/2, etc. to get coordinates center around origin
	double coordX = 1.0*intX - (groupConfig[grpId].SizeX-1)/2.0;
	double coordY = 1.0*intY - (groupConfig[grpId].SizeY-1)/2.0;
	double coordZ = 1.0*intZ - (groupConfig[grpId].SizeZ-1)/2.0;
	return Point3D(coordX, coordY, coordZ);
}

// returns the number of synaptic connections associated with this connection.
int SNN::getNumSynapticConnections(short int connId) {
	//ConnectConfig* connInfo;
	//ConnectConfig* connIterator = connectBegin;
	//while(connIterator){
	//  if(connIterator->connId == connectionId){
	//    //found the corresponding connection
	//    return connIterator->numberOfConnections;
	//  }
	//  //move to the next ConnectConfig
	//  connIterator=connIterator->next;
	//}


	//we didn't find the connection.
	if (connectConfigMap.find(connId) == connectConfigMap.end()) {
		KERNEL_ERROR("Connection ID was not found.  Quitting.");
		exitSimulation(1);
	}

	return connectConfigMap[connId].numberOfConnections;
}

// return spike buffer, which contains #spikes per neuron in the group
int* SNN::getSpikeCounter(int grpId) {
	assert(grpId>=0); assert(grpId<numGrp);

	if (!groupConfig[grpId].withSpikeCounter)
		return NULL;

	// determine whether spike counts are currently stored on CPU or GPU side
	bool retrieveSpikesFromGPU = simMode_==GPU_MODE;
	if (groupConfig[grpId].isSpikeGenerator) {
		// this flag should be set if group was created via CARLsim::createSpikeGeneratorGroup
		// could be SpikeGen callback or PoissonRate
		if (groupConfig[grpId].RatePtr != NULL) {
			// group is Poisson group
			// even though mean rates might be on either CPU or GPU (RatePtr->isOnGPU()), in GPU mode the
			// actual random numbers will always be generated on the GPU
//			retrieveSpikesFromGPU = simMode_==GPU_MODE;
		} else {
			// group is generator with callback, CPU only
			retrieveSpikesFromGPU = false;
		}
	}

	// retrieve spikes from either CPU or GPU
	if (retrieveSpikesFromGPU) {
		return getSpikeCounter_GPU(grpId);
	} else {
		int bufPos = groupConfig[grpId].spkCntBufPos; // retrieve buf pos
		return spkCntBuf[bufPos]; // return pointer to buffer
	}
}

// returns pointer to existing SpikeMonitor object, NULL else
SpikeMonitor* SNN::getSpikeMonitor(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	if (groupConfig[grpId].SpikeMonitorId>=0) {
		return spikeMonList[(groupConfig[grpId].SpikeMonitorId)];
	} else {
		return NULL;
	}
}

SpikeMonitorCore* SNN::getSpikeMonitorCore(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	if (groupConfig[grpId].SpikeMonitorId>=0) {
		return spikeMonCoreList[(groupConfig[grpId].SpikeMonitorId)];
	} else {
		return NULL;
	}
}

// returns RangeWeight struct of a connection
RangeWeight SNN::getWeightRange(short int connId) {
	assert(connId>=0 && connId<numConnections);
	//ConnectConfig* connInfo = getConnectInfo(connId);
	return RangeWeight(0.0f, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt);
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// all unsafe operations of SNN constructor
void SNN::SNNinit() {
	assert(ithGPU_>=0);

	// initialize snnState
	snnState = CONFIG_SNN;
	
	// set logger mode (defines where to print all status, error, and debug messages)
	switch (loggerMode_) {
	case USER:
		fpInf_ = stdout;
		fpErr_ = stderr;
		#if defined(WIN32) || defined(WIN64)
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
		#if defined(WIN32) || defined(WIN64)
			fpInf_ = fopen("nul","w");
		#else
			fpInf_ = fopen("/dev/null","w");
		#endif
		fpErr_ = stderr;
		#if defined(WIN32) || defined(WIN64)
			fpDeb_ = fopen("nul","w");
		#else
			fpDeb_ = fopen("/dev/null","w");
		#endif
		break;
	case SILENT:
	case CUSTOM:
		#if defined(WIN32) || defined(WIN64)
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
		fpErr_ = stderr; // need to open file stream first
		KERNEL_ERROR("Unknown logger mode");
		exit(1);
	}
	#if defined(WIN32) || defined(WIN64)
		fpLog_= fopen("results\\carlsim.log","w");
	#else
		fpLog_ = fopen("results/carlsim.log","w");
	#endif

	#ifdef __REGRESSION_TESTING__
	#if defined(WIN32) || defined(WIN64)
		fpInf_ = fopen("nul","w");
		fpErr_ = fopen("nul","w");
		fpDeb_ = fopen("nul","w");
	#else
		fpInf_ = fopen("/dev/null","w");
		fpErr_ = fopen("/dev/null","w");
		fpDeb_ = fopen("/dev/null","w");
	#endif
	#endif

	KERNEL_INFO("*********************************************************************************");
	KERNEL_INFO("********************      Welcome to CARLsim %d.%d      ***************************",
				MAJOR_VERSION,MINOR_VERSION);
	KERNEL_INFO("*********************************************************************************\n");

	KERNEL_INFO("***************************** Configuring Network ********************************");
	KERNEL_INFO("Starting CARLsim simulation \"%s\" in %s mode",networkName_.c_str(),
		loggerMode_string[loggerMode_]);
	KERNEL_INFO("Random number seed: %d",randSeed_);

	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	KERNEL_DEBUG("Current local time and date: %s", asctime(timeinfo));

	// init random seed
	srand48(randSeed_);
	//getRand.seed(randSeed_*2);
	//getRandClosed.seed(randSeed_*3);

	//connectBegin = NULL;

	simTimeRunStart     = 0;    simTimeRunStop      = 0;
	simTimeLastRunSummary = 0;
	simTimeMs	 		= 0;    simTimeSec          = 0;    simTime = 0;
	spikeCountAll1secHost	= 0;    secD1fireCntHost    = 0;    secD2fireCntHost  = 0;
	spikeCountAllHost 		= 0;    spikeCountD2Host    = 0;    spikeCountD1Host = 0;
	nPoissonSpikes 		= 0;

	numGrp   = 0;
	numConnections = 0;
	numSpikeGenGrps  = 0;
	NgenFunc = 0;
	simulatorDeleted = false;

	allocatedN      = 0;
	allocatedPre    = 0;
	allocatedPost   = 0;

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
	sim_in_testing = false;

	maxSpikesD1 = 0;
	maxSpikesD2 = 0;

	loadSimFID = NULL;

	numN = 0;
	numNPois = 0;
	numNExcPois = 0;
	numNInhPois = 0;
	numNReg = 0;
	numNExcReg = 0;
	numNInhReg = 0;

	numPostSynapses_ = 0;
	numPreSynapses_ = 0;
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

	// each SNN object hold its own random number object
	gpuPoissonRand = NULL;

	// reset all monitors, don't deallocate (false)
	resetMonitors(false);

	resetConnectionConfigs(false);

	// reset all runtime data, don't deallocate (false)
	resetRuntimeData(false);

	memset(&cpuSnnSz, 0, sizeof(cpuSnnSz));

	showGrpFiringInfo = true;

	// initialize propogated spike buffers.....
	pbuf = new PropagatedSpikeBuffer(0, PROPAGATED_BUFFER_SIZE);

	memset(&gpuRuntimeData, 0, sizeof(RuntimeData));
	memset(&networkConfig, 0, sizeof(NetworkConfigRT));
	gpuRuntimeData.allocated = false;

	memset(&snnRuntimeData, 0, sizeof(RuntimeData));
	snnRuntimeData.allocated = false;

	for (int i=0; i < MAX_GRP_PER_SNN; i++) {
		groupConfig[i].Type = UNKNOWN_NEURON;
		groupConfig[i].MaxFiringRate = UNKNOWN_NEURON_MAX_FIRING_RATE;
		groupConfig[i].SpikeMonitorId = -1;
		groupConfig[i].GroupMonitorId = -1;
//		groupConfig[i].ConnectionMonitorId = -1;
		groupConfig[i].FiringCount1sec=0;
		groupConfig[i].numPostSynapses 		= 0;	// default value
		groupConfig[i].numPreSynapses 	= 0;	// default value
		groupConfig[i].WithSTP = false;
		groupConfig[i].WithSTDP = false;
		groupConfig[i].WithESTDP = false;
		groupConfig[i].WithISTDP = false;
		groupConfig[i].WithESTDPtype = UNKNOWN_STDP;
		groupConfig[i].WithISTDPtype = UNKNOWN_STDP;
		groupConfig[i].WithESTDPcurve = UNKNOWN_CURVE;
		groupConfig[i].WithISTDPcurve = UNKNOWN_CURVE;
		groupConfig[i].FixedInputWts = true; // Default is true. This value changed to false
		// if any incoming  connections are plastic
		groupConfig[i].isSpikeGenerator = false;
		groupConfig[i].RatePtr = NULL;

		groupConfig[i].homeoId = -1;
		groupConfig[i].avgTimeScale  = 10000.0;

		groupConfig[i].baseDP = 1.0f;
		groupConfig[i].base5HT = 1.0f;
		groupConfig[i].baseACh = 1.0f;
		groupConfig[i].baseNE = 1.0f;
		groupConfig[i].decayDP = 1 - (1.0f / 100);
		groupConfig[i].decay5HT = 1 - (1.0f / 100);
		groupConfig[i].decayACh = 1 - (1.0f / 100);
		groupConfig[i].decayNE = 1 - (1.0f / 100);

		groupConfig[i].spikeGen = NULL;

		groupConfig[i].withSpikeCounter = false;
		groupConfig[i].spkCntRecordDur = -1;
		groupConfig[i].spkCntRecordDurHelper = 0;
		groupConfig[i].spkCntBufPos = -1;

		groupConfig[i].StartN       = -1;
		groupConfig[i].EndN       	 = -1;

		groupConfig[i].CurrTimeSlice = 0;
		groupConfig[i].NewTimeSlice = 0;
		groupConfig[i].SliceUpdateTime = 0;

		groupInfo[i].numPostConn = 0;
		groupInfo[i].numPreConn  = 0;
		groupInfo[i].maxPostConn = 0;
		groupInfo[i].maxPreConn  = 0;
		groupInfo[i].sumPostConn = 0;
		groupInfo[i].sumPreConn  = 0;

	}

	CUDA_CREATE_TIMER(timer);
	CUDA_RESET_TIMER(timer);

	// default weight update parameter
	wtANDwtChangeUpdateInterval_ = 1000; // update weights every 1000 ms (default)
	wtANDwtChangeUpdateIntervalCnt_ = 0; // helper var to implement fast modulo
	stdpScaleFactor_ = 1.0f;
	wtChangeDecay_ = 0.0f;

	if (simMode_ == GPU_MODE)
		configGPUDevice();
}

void SNN::allocateSNN() {
	// Confirm allocation of SNN runtime data in main memory
	snnRuntimeData.allocated = true;
	snnRuntimeData.memType = CPU_MODE;

	switch (simMode_) {
	case GPU_MODE:
		allocateSNN_GPU();
		break;
	case CPU_MODE:
		allocateSNN_CPU();
		break;
	default:
		KERNEL_ERROR("Unknown simMode_");
		break;
	}
}

void SNN::allocateSNN_CPU() {
	snnRuntimeData.allocated = true;
	snnRuntimeData.memType = CPU_MODE;

	snnState = EXECUTABLE_SNN;
}

//! update (initialize) numN, numPostSynapses, numPreSynapses, maxDelay_, postSynCnt, preSynCnt
//! allocate space for voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, gAMPA, gNMDA, gGABAa, gGABAb
//! lastSpikeTime, curSpike, nSpikeCnt, intrinsicWeight, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre
//! postSynapticIds, tmp_SynapticDely, postDelayInfo, wt, maxSynWt, preSynapticIds, timeTableD2, timeTableD1
void SNN::buildNetworkInit() {
	// \FIXME: need to figure out STP buffer for delays > 1
	if (sim_with_stp && maxDelay_>1) {
		KERNEL_ERROR("STP with delays > 1 ms is currently not supported.");
		exitSimulation(1);
	}

	snnRuntimeData.voltage	   = new float[numNReg];
	snnRuntimeData.recovery   = new float[numNReg];
	snnRuntimeData.Izh_a	   = new float[numNReg];
	snnRuntimeData.Izh_b      = new float[numNReg];
	snnRuntimeData.Izh_c	   = new float[numNReg];
	snnRuntimeData.Izh_d	   = new float[numNReg];
	snnRuntimeData.current	   = new float[numNReg];
	snnRuntimeData.extCurrent = new float[numNReg];
	memset(snnRuntimeData.extCurrent, 0, sizeof(snnRuntimeData.extCurrent[0])*numNReg);

	cpuSnnSz.neuronInfoSize += (sizeof(float)*numNReg*8);

	if (sim_with_conductances) {
		snnRuntimeData.gAMPA  = new float[numNReg];
		snnRuntimeData.gGABAa = new float[numNReg];
		cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;

		if (sim_with_NMDA_rise) {
			// If NMDA rise time is enabled, we'll have to compute NMDA conductance in two steps (using an exponential
			// for the rise time and one for the decay time)
			snnRuntimeData.gNMDA_r = new float[numNReg];
			snnRuntimeData.gNMDA_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			snnRuntimeData.gNMDA = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}

		if (sim_with_GABAb_rise) {
			snnRuntimeData.gGABAb_r = new float[numNReg];
			snnRuntimeData.gGABAb_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			snnRuntimeData.gGABAb = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}
	}

	snnRuntimeData.grpDA = new float[numGrp];
	snnRuntimeData.grp5HT = new float[numGrp];
	snnRuntimeData.grpACh = new float[numGrp];
	snnRuntimeData.grpNE = new float[numGrp];

	// init neuromodulators and their assistive buffers
	for (int i = 0; i < numGrp; i++) {
		snnRuntimeData.grpDABuffer[i] = new float[1000]; // 1 second DA buffer
		snnRuntimeData.grp5HTBuffer[i] = new float[1000];
		snnRuntimeData.grpAChBuffer[i] = new float[1000];
		snnRuntimeData.grpNEBuffer[i] = new float[1000];
	}

	resetCurrent();
	resetConductances();

	snnRuntimeData.lastSpikeTime	= new uint32_t[numN];
	cpuSnnSz.neuronInfoSize += sizeof(int) * numN;
	memset(snnRuntimeData.lastSpikeTime, 0, sizeof(snnRuntimeData.lastSpikeTime[0]) * numN);

	snnRuntimeData.curSpike   = new bool[numN];
	snnRuntimeData.nSpikeCnt  = new int[numN];
	KERNEL_INFO("allocated nSpikeCnt");

	//! homeostasis variables
	if (sim_with_homeostasis) {
		snnRuntimeData.avgFiring  = new float[numN];
		snnRuntimeData.baseFiring = new float[numN];
	}

	// STP can be applied to spike generators, too -> numN
	if (sim_with_stp) {
		// \TODO: The size of these data structures could be reduced to the max synaptic delay of all
		// connections with STP. That number might not be the same as maxDelay_.
		snnRuntimeData.stpu = new float[numN*(maxDelay_+1)];
		snnRuntimeData.stpx = new float[numN*(maxDelay_+1)];
		memset(snnRuntimeData.stpu, 0, sizeof(float)*numN*(maxDelay_+1)); // memset works for 0.0
		for (int i=0; i < numN*(maxDelay_+1); i++)
			snnRuntimeData.stpx[i] = 1.0f; // but memset doesn't work for 1.0
		cpuSnnSz.synapticInfoSize += (2*sizeof(float)*numN*(maxDelay_+1));
	}

	snnRuntimeData.Npre 		   = new unsigned short[numN];
	snnRuntimeData.Npre_plastic   = new unsigned short[numN];
	snnRuntimeData.Npost 		   = new unsigned short[numN];
	snnRuntimeData.cumulativePost = new unsigned int[numN];
	snnRuntimeData.cumulativePre  = new unsigned int[numN];
	cpuSnnSz.networkInfoSize += (int)(sizeof(int) * numN * 3.5);

	postSynCnt = 0;
	preSynCnt  = 0;
	for(int g=0; g<numGrp; g++) {
		// check for INT overflow: postSynCnt is O(numNeurons*numSynapses), must be able to fit within u int limit
		assert(postSynCnt < UINT_MAX - (groupConfig[g].SizeN * groupConfig[g].numPostSynapses));
		assert(preSynCnt < UINT_MAX - (groupConfig[g].SizeN * groupConfig[g].numPreSynapses));
		postSynCnt += (groupConfig[g].SizeN * groupConfig[g].numPostSynapses);
		preSynCnt  += (groupConfig[g].SizeN * groupConfig[g].numPreSynapses);
	}
	assert(postSynCnt/numN <= numPostSynapses_); // divide by numN to prevent INT overflow
	snnRuntimeData.postSynapticIds		= new post_info_t[postSynCnt+100];
	tmp_SynapticDelay	= new uint8_t[postSynCnt+100];	//!< Temporary array to store the delays of each connection
	snnRuntimeData.postDelayInfo		= new delay_info_t[numN*(maxDelay_+1)];	//!< Possible delay values are 0....maxDelay_ (inclusive of maxDelay_)
	cpuSnnSz.networkInfoSize += ((sizeof(post_info_t)+sizeof(uint8_t))*postSynCnt+100)+(sizeof(delay_info_t)*numN*(maxDelay_+1));
	assert(preSynCnt/numN <= numPreSynapses_); // divide by numN to prevent INT overflow

	snnRuntimeData.wt  			= new float[preSynCnt+100];
	snnRuntimeData.maxSynWt     	= new float[preSynCnt+100];

	mulSynFast 		= new float[MAX_CONN_PER_SNN];
	mulSynSlow 		= new float[MAX_CONN_PER_SNN];
	snnRuntimeData.cumConnIdPre	= new short int[preSynCnt+100];

	//! Temporary array to hold pre-syn connections. will be deleted later if necessary
	snnRuntimeData.preSynapticIds	= new post_info_t[preSynCnt + 100];
	// size due to weights and maximum weights
	cpuSnnSz.synapticInfoSize += ((sizeof(int) + 2 * sizeof(float) + sizeof(post_info_t)) * (preSynCnt + 100));

	timeTableD2  = new unsigned int[1000 + maxDelay_ + 1];
	timeTableD1  = new unsigned int[1000 + maxDelay_ + 1];
	resetTimingTable();
	cpuSnnSz.spikingInfoSize += sizeof(int) * 2 * (1000 + maxDelay_ + 1);

	// poisson Firing Rate
	cpuSnnSz.neuronInfoSize += (sizeof(int) * numNPois);
}


int SNN::addSpikeToTable(int nid, int g) {
	int spikeBufferFull = 0;
	snnRuntimeData.lastSpikeTime[nid] = simTime;
	snnRuntimeData.curSpike[nid] = true;
	snnRuntimeData.nSpikeCnt[nid]++;
	if (sim_with_homeostasis)
		snnRuntimeData.avgFiring[nid] += 1000/(groupConfig[g].avgTimeScale*1000);

	if (simMode_ == GPU_MODE) {
		assert(groupConfig[g].isSpikeGenerator == true);
		setSpikeGenBit_GPU(nid, g);
		return 0;
	}

	if (groupConfig[g].WithSTP) {
		// update the spike-dependent part of du/dt and dx/dt
		// we need to retrieve the STP values from the right buffer position (right before vs. right after the spike)
		int ind_plus = STP_BUF_POS(nid,simTime); // index of right after the spike, such as in u^+
	    int ind_minus = STP_BUF_POS(nid,(simTime-1)); // index of right before the spike, such as in u^-

		// du/dt = -u/tau_F + U * (1-u^-) * \delta(t-t_{spk})
		snnRuntimeData.stpu[ind_plus] += groupConfig[g].STP_U*(1.0-snnRuntimeData.stpu[ind_minus]);

		// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
		snnRuntimeData.stpx[ind_plus] -= snnRuntimeData.stpu[ind_plus]*snnRuntimeData.stpx[ind_minus];
	}

	if (groupConfig[g].MaxDelay == 1) {
		assert(nid < numN);
		snnRuntimeData.firingTableD1[secD1fireCntHost] = nid;
		secD1fireCntHost++;
		groupConfig[g].FiringCount1sec++;
		if (secD1fireCntHost >= maxSpikesD1) {
			spikeBufferFull = 2;
			secD1fireCntHost = maxSpikesD1-1;
		}
	} else {
		assert(nid < numN);
		snnRuntimeData.firingTableD2[secD2fireCntHost] = nid;
		groupConfig[g].FiringCount1sec++;
		secD2fireCntHost++;
		if (secD2fireCntHost >= maxSpikesD2) {
			spikeBufferFull = 1;
			secD2fireCntHost = maxSpikesD2-1;
		}
	}
	return spikeBufferFull;
}


void SNN::buildGroup(int grpId) {
	assert(groupConfig[grpId].StartN == -1);
	groupConfig[grpId].StartN = allocatedN;
	groupConfig[grpId].EndN   = allocatedN + groupConfig[grpId].SizeN - 1;

	KERNEL_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, groupInfo[grpId].Name.c_str(), groupConfig[grpId].StartN, groupConfig[grpId].EndN);

	resetNeuromodulator(grpId);

	allocatedN = allocatedN + groupConfig[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=groupConfig[grpId].StartN; i <= groupConfig[grpId].EndN; i++) {
		resetNeuron(i, grpId);
		snnRuntimeData.Npre_plastic[i]	= 0;
		snnRuntimeData.Npre[i]		  	= 0;
		snnRuntimeData.Npost[i]	  	= 0;
		snnRuntimeData.cumulativePost[i] = allocatedPost;
		snnRuntimeData.cumulativePre[i]  = allocatedPre;
		allocatedPost    += groupConfig[grpId].numPostSynapses;
		allocatedPre     += groupConfig[grpId].numPreSynapses;
	}

	assert(allocatedPost <= postSynCnt);
	assert(allocatedPre  <= preSynCnt);
}

//! build the network based on the current setting (e.g., group, connection)
/*!
 * \sa createGroup(), connect()
 */
void SNN::buildNetwork() {
	//ConnectConfig* newInfo = connectBegin;

	// find the maximum values for number of pre- and post-synaptic neurons
	findMaxNumSynapses(&numPostSynapses_, &numPreSynapses_);

	// update (initialize) maxSpikesD1, maxSpikesD2 and allocate space for firingTableD1 and firingTableD2
	maxDelay_ = updateSpikeTables();

	// make sure number of neurons and max delay are within bounds
	assert(maxDelay_ <= MAX_SYN_DELAY); 
	assert(numN <= 1000000);
	assert((numN > 0) && (numN == numNExcReg + numNInhReg + numNPois));

	// display the evaluated network and delay length....
	KERNEL_INFO("\n");
	KERNEL_INFO("***************************** Setting up Network **********************************");
	KERNEL_INFO("numN = %d, numPostSynapses = %d, numPreSynapses = %d, maxDelay = %d", numN, numPostSynapses_,
					numPreSynapses_, maxDelay_);

	if (numPostSynapses_ > MAX_NUM_POST_SYN) {
		for (int g=0;g<numGrp;g++) {
			if (groupConfig[g].numPostSynapses>MAX_NUM_POST_SYN)
				KERNEL_ERROR("Grp: %s(%d) has too many output synapses (%d), max %d.",groupInfo[g].Name.c_str(),g,
							groupConfig[g].numPostSynapses,MAX_NUM_POST_SYN);
		}
		assert(numPostSynapses_ <= MAX_NUM_POST_SYN);
	}
	if (numPreSynapses_ > MAX_NUM_PRE_SYN) {
		for (int g=0;g<numGrp;g++) {
			if (groupConfig[g].numPreSynapses>MAX_NUM_PRE_SYN)
				KERNEL_ERROR("Grp: %s(%d) has too many input synapses (%d), max %d.",groupInfo[g].Name.c_str(),g,
 							groupConfig[g].numPreSynapses,MAX_NUM_PRE_SYN);
		}
		assert(numPreSynapses_ <= MAX_NUM_PRE_SYN);
	}

	// initialize all the parameters....
	//! update (initialize) numN, numPostSynapses, numPreSynapses, maxDelay_, postSynCnt, preSynCnt
	//! allocate space for voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, gAMPA, gNMDA, gGABAa, gGABAb
	//! lastSpikeTime, curSpike, nSpikeCnt, intrinsicWeight, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre
	//! postSynapticIds, tmp_SynapticDely, postDelayInfo, wt, maxSynWt, preSynapticIds, timeTableD2, timeTableD1, grpDA, grp5HT, grpACh, grpNE
	buildNetworkInit();

	// we build network in the order...
	//    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
	//     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
	//     Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
	int allocatedGrp = 0;
	for(int order = 0; order < 4; order++) {
		for(int g = 0; g < numGrp; g++) {
			if (IS_EXCITATORY_TYPE(groupConfig[g].Type) && (groupConfig[g].Type & POISSON_NEURON) && order == 3) {
				buildPoissonGroup(g);
				allocatedGrp++;
			} else if (IS_INHIBITORY_TYPE(groupConfig[g].Type) &&  (groupConfig[g].Type & POISSON_NEURON) && order == 2) {
				buildPoissonGroup(g);
				allocatedGrp++;
			} else if (IS_EXCITATORY_TYPE(groupConfig[g].Type) && !(groupConfig[g].Type & POISSON_NEURON) && order == 0) {
				buildGroup(g);
				allocatedGrp++;
			} else if (IS_INHIBITORY_TYPE(groupConfig[g].Type) && !(groupConfig[g].Type & POISSON_NEURON) && order == 1) {
				buildGroup(g);
				allocatedGrp++;
			}
		}
	}
	assert(allocatedGrp == numGrp);

	// print group overview
	for (int g=0;g<numGrp;g++) {
		printGroupInfo(g);
	}


	snnRuntimeData.grpIds = new short int[numN];
	for (int nid=0; nid<numN; nid++) {
		snnRuntimeData.grpIds[nid] = -1;
		for (int g=0; g<numGrp; g++) {
			if (nid>=groupConfig[g].StartN && nid<=groupConfig[g].EndN) {
				snnRuntimeData.grpIds[nid] = (short int)g;
				//printf("grpIds[%d] = %d\n",nid,g);
				break;
			}
		}
		assert(snnRuntimeData.grpIds[nid]!=-1);
	}

	if (loadSimFID != NULL) {
		int loadError;
		// we the user specified loadSimulation the synaptic weights will be restored here...
		KERNEL_DEBUG("Start to load simulation");
		loadError = loadSimulation_internal(true); // read the plastic synapses first
		KERNEL_DEBUG("loadSimulation_internal() error number:%d", loadError);
		loadError = loadSimulation_internal(false); // read the fixed synapses second
		KERNEL_DEBUG("loadSimulation_internal() error number:%d", loadError);
		for(int con = 0; con < 2; con++) {
			//newInfo = connectBegin;
			//while(newInfo) {
			for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
				bool synWtType = GET_FIXED_PLASTIC(it->second.connProp);
				if (synWtType == SYN_PLASTIC) {
					// given group has plastic connection, and we need to apply STDP rule...
					groupConfig[it->second.grpDest].FixedInputWts = false;
				}

				// store scaling factors for synaptic currents in connection-centric array
				mulSynFast[it->second.connId] = it->second.mulSynFast;
				mulSynSlow[it->second.connId] = it->second.mulSynSlow;

				if( ((con == 0) && (synWtType == SYN_PLASTIC)) || ((con == 1) && (synWtType == SYN_FIXED))) {
					printConnectionInfo(it->second.connId);
				}
				//newInfo = newInfo->next;
			}
		}
	} else {
		// build all the connections here...
		// we run over the linked list two times...
		// first time, we make all plastic connections...
		// second time, we make all fixed connections...
		// this ensures that all the initial pre and post-synaptic
		// connections are of fixed type and later if of plastic type
		for(int con = 0; con < 2; con++) {
			//newInfo = connectBegin;
			//while(newInfo) {
			for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
				bool synWtType = GET_FIXED_PLASTIC(it->second.connProp);
				if (synWtType == SYN_PLASTIC) {
					// given group has plastic connection, and we need to apply STDP rule...
					groupConfig[it->second.grpDest].FixedInputWts = false;
				}

				// store scaling factors for synaptic currents in connection-centric array
				mulSynFast[it->second.connId] = it->second.mulSynFast;
				mulSynSlow[it->second.connId] = it->second.mulSynSlow;


				if( ((con == 0) && (synWtType == SYN_PLASTIC)) || ((con == 1) && (synWtType == SYN_FIXED))) {
					switch(it->second.type) {
						case CONN_RANDOM:
							connectRandom(it->second.connId);
							break;
						case CONN_FULL:
							connectFull(it->second.connId);
							break;
						case CONN_FULL_NO_DIRECT:
							connectFull(it->second.connId);
							break;
						case CONN_ONE_TO_ONE:
							connectOneToOne(it->second.connId);
							break;
						case CONN_GAUSSIAN:
							connectGaussian(it->second.connId);
							break;
						case CONN_USER_DEFINED:
							connectUserDefined(it->second.connId);
							break;
						default:
							KERNEL_ERROR("Invalid connection type( should be 'random', 'full', 'full-no-direct', or 'one-to-one')");
							exitSimulation(-1);
					}

					printConnectionInfo(it->second.connId);
				}
				//newInfo = newInfo->next;
			}
		}
	}
}

void SNN::buildPoissonGroup(int grpId) {
	assert(groupConfig[grpId].StartN == -1);
	groupConfig[grpId].StartN 	= allocatedN;
	groupConfig[grpId].EndN   	= allocatedN + groupConfig[grpId].SizeN - 1;

	KERNEL_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, groupInfo[grpId].Name.c_str(), groupConfig[grpId].StartN, groupConfig[grpId].EndN);

	allocatedN = allocatedN + groupConfig[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=groupConfig[grpId].StartN; i <= groupConfig[grpId].EndN; i++) {
		resetPoissonNeuron(i, grpId);
		snnRuntimeData.Npre_plastic[i]	  = 0;
		snnRuntimeData.Npre[i]		  	  = 0;
		snnRuntimeData.Npost[i]	      = 0;
		snnRuntimeData.cumulativePost[i] = allocatedPost;
		snnRuntimeData.cumulativePre[i]  = allocatedPre;
		allocatedPost    += groupConfig[grpId].numPostSynapses;
		allocatedPre     += groupConfig[grpId].numPreSynapses;
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
void SNN::checkSpikeCounterRecordDur() {
	for (int g=0; g<numGrp; g++) {
		// skip groups w/o spkMonRT or non-real record durations
		if (!groupConfig[g].withSpikeCounter || groupConfig[g].spkCntRecordDur<=0)
			continue;

		// skip if simTime doesn't need udpating
		// we want to update in spkCntRecordDur + 1, because this function is called rigth at the beginning
		// of each millisecond
		if ( (simTime % ++groupConfig[g].spkCntRecordDurHelper) != 1)
			continue;

 		if (simMode_==GPU_MODE)
			resetSpikeCounter_GPU(g);
		else
			resetSpikeCounter(g);
	}
}

// We parallelly cleanup the postSynapticIds array to minimize any other wastage in that array by compacting the store
// Appropriate alignment specified by ALIGN_COMPACTION macro is used to ensure some level of alignment (if necessary)
void SNN::compactConnections() {
	unsigned int* tmp_cumulativePost = new unsigned int[numN];
	unsigned int* tmp_cumulativePre  = new unsigned int[numN];
	unsigned int lastCnt_pre         = 0;
	unsigned int lastCnt_post        = 0;

	tmp_cumulativePost[0]   = 0;
	tmp_cumulativePre[0]    = 0;

	for(int i=1; i < numN; i++) {
		lastCnt_post = tmp_cumulativePost[i-1]+snnRuntimeData.Npost[i-1]; //position of last pointer
		lastCnt_pre  = tmp_cumulativePre[i-1]+snnRuntimeData.Npre[i-1]; //position of last pointer
		#if COMPACTION_ALIGNMENT_POST
			lastCnt_post= lastCnt_post + COMPACTION_ALIGNMENT_POST-lastCnt_post%COMPACTION_ALIGNMENT_POST;
			lastCnt_pre = lastCnt_pre  + COMPACTION_ALIGNMENT_PRE- lastCnt_pre%COMPACTION_ALIGNMENT_PRE;
		#endif
		tmp_cumulativePost[i] = lastCnt_post;
		tmp_cumulativePre[i]  = lastCnt_pre;
		assert(tmp_cumulativePost[i] <= snnRuntimeData.cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= snnRuntimeData.cumulativePre[i]);
	}

	// compress the post_synaptic array according to the new values of the tmp_cumulative counts....
	unsigned int tmp_postSynCnt = tmp_cumulativePost[numN-1]+snnRuntimeData.Npost[numN-1];
	unsigned int tmp_preSynCnt  = tmp_cumulativePre[numN-1]+snnRuntimeData.Npre[numN-1];
	assert(tmp_postSynCnt <= allocatedPost);
	assert(tmp_preSynCnt  <= allocatedPre);
	assert(tmp_postSynCnt <= postSynCnt);
	assert(tmp_preSynCnt  <= preSynCnt);
	KERNEL_DEBUG("******************");
	KERNEL_DEBUG("CompactConnection: ");
	KERNEL_DEBUG("******************");
	KERNEL_DEBUG("old_postCnt = %d, new_postCnt = %d", postSynCnt, tmp_postSynCnt);
	KERNEL_DEBUG("old_preCnt = %d,  new_postCnt = %d", preSynCnt,  tmp_preSynCnt);

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
		assert(tmp_cumulativePost[i] <= snnRuntimeData.cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= snnRuntimeData.cumulativePre[i]);
		for( int j=0; j<snnRuntimeData.Npost[i]; j++) {
			unsigned int tmpPos = tmp_cumulativePost[i]+j;
			unsigned int oldPos = snnRuntimeData.cumulativePost[i]+j;
			tmp_postSynapticIds[tmpPos] = snnRuntimeData.postSynapticIds[oldPos];
			tmp_SynapticDelay[tmpPos]   = tmp_SynapticDelay[oldPos];
		}
		for( int j=0; j<snnRuntimeData.Npre[i]; j++) {
			unsigned int tmpPos =  tmp_cumulativePre[i]+j;
			unsigned int oldPos =  snnRuntimeData.cumulativePre[i]+j;
			tmp_preSynapticIds[tmpPos]  = snnRuntimeData.preSynapticIds[oldPos];
			tmp_maxSynWt[tmpPos] 	    = snnRuntimeData.maxSynWt[oldPos];
			tmp_wt[tmpPos]              = snnRuntimeData.wt[oldPos];
			tmp_cumConnIdPre[tmpPos]	= snnRuntimeData.cumConnIdPre[oldPos];
		}
	}

	// delete old buffer space
	delete[] snnRuntimeData.postSynapticIds;
	snnRuntimeData.postSynapticIds = tmp_postSynapticIds;
	cpuSnnSz.networkInfoSize -= (sizeof(post_info_t)*postSynCnt);
	cpuSnnSz.networkInfoSize += (sizeof(post_info_t)*(tmp_postSynCnt+100));

	delete[] snnRuntimeData.cumulativePost;
	snnRuntimeData.cumulativePost  = tmp_cumulativePost;

	delete[] snnRuntimeData.cumulativePre;
	snnRuntimeData.cumulativePre   = tmp_cumulativePre;

	delete[] snnRuntimeData.maxSynWt;
	snnRuntimeData.maxSynWt = tmp_maxSynWt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] snnRuntimeData.wt;
	snnRuntimeData.wt = tmp_wt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] snnRuntimeData.cumConnIdPre;
	snnRuntimeData.cumConnIdPre = tmp_cumConnIdPre;
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


	delete[] snnRuntimeData.preSynapticIds;
	snnRuntimeData.preSynapticIds  = tmp_preSynapticIds;
	cpuSnnSz.synapticInfoSize -= (sizeof(post_info_t)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(post_info_t)*(tmp_preSynCnt+100));

	preSynCnt	= tmp_preSynCnt;
	postSynCnt	= tmp_postSynCnt;
}

// make 'C' full connections from grpSrc to grpDest
void SNN::connectFull(short int connId) {
	int grpSrc = connectConfigMap[connId].grpSrc;
	int grpDest = connectConfigMap[connId].grpDest;
	bool noDirect = (connectConfigMap[connId].type == CONN_FULL_NO_DIRECT);

	// rebuild struct for easier handling
	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);

	for(int i = groupConfig[grpSrc].StartN; i <= groupConfig[grpSrc].EndN; i++)  {
		Point3D loc_i = getNeuronLocation3D(i); // 3D coordinates of i
		for(int j = groupConfig[grpDest].StartN; j <= groupConfig[grpDest].EndN; j++) { // j: the temp neuron id
			// if flag is set, don't connect direct connections
			if((noDirect) && (i - groupConfig[grpSrc].StartN) == (j - groupConfig[grpDest].StartN))
				continue;

			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j
			if (!isPoint3DinRF(radius, loc_i, loc_j))
				continue;

			//uint8_t dVal = info->minDelay + (int)(0.5 + (drand48() * (info->maxDelay - info->minDelay)));
			uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
			assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
			float synWt = getWeights(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, i, grpSrc);

			setConnection(grpSrc, grpDest, i, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);// info->connId);
			connectConfigMap[connId].numberOfConnections++;
		}
	}

	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
}

// after all the initalization. Its time to create the synaptic weights, weight change and also
// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
void SNN::compileSNN(bool removeTempMemory) {
	KERNEL_DEBUG("Beginning compilation of network....");

	// perform various consistency checks:
	// - numNeurons vs. sum of all neurons
	// - STDP set on a post-group with incoming plastic connections
	// - etc.
	verifyNetwork();

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

	// reset all spike cnt
	resetSpikeCnt(ALL);

	printTuningLog(fpDeb_);

	KERNEL_INFO("");
	KERNEL_INFO("*****************      Initializing %s Simulation      *************************",
		simMode_==GPU_MODE?"GPU":"CPU");

	delete[] tmp_SynapticDelay;
	tmp_SynapticDelay = NULL;

	//ensure that we dont compile the network again
	snnState = COMPILED_SNN;
}



void SNN::connectGaussian(short int connId) {
	// rebuild struct for easier handling
	// adjust with sqrt(2) in order to make the Gaussian kernel depend on 2*sigma^2
	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);

	// in case pre and post have different Grid3D sizes: scale pre to the grid size of post
	int grpSrc = connectConfigMap[connId].grpSrc;
	int grpDest = connectConfigMap[connId].grpDest;
	Grid3D grid_i = getGroupGrid3D(grpSrc);
	Grid3D grid_j = getGroupGrid3D(grpDest);
	Point3D scalePre = Point3D(grid_j.numX, grid_j.numY, grid_j.numZ) / Point3D(grid_i.numX, grid_i.numY, grid_i.numZ);

	for(int i = groupConfig[grpSrc].StartN; i <= groupConfig[grpSrc].EndN; i++)  {
		Point3D loc_i = getNeuronLocation3D(i)*scalePre; // i: adjusted 3D coordinates

		for(int j = groupConfig[grpDest].StartN; j <= groupConfig[grpDest].EndN; j++) { // j: the temp neuron id
			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j

			// make sure point is in RF
			double rfDist = getRFDist3D(radius,loc_i,loc_j);
			if (rfDist < 0.0 || rfDist > 1.0)
				continue;

			// if rfDist is valid, it returns a number between 0 and 1
			// we want these numbers to fit to Gaussian weigths, so that rfDist=0 corresponds to max Gaussian weight
			// and rfDist=1 corresponds to 0.1 times max Gaussian weight
			// so we're looking at gauss = exp(-a*rfDist), where a such that exp(-a)=0.1
			// solving for a, we find that a = 2.3026
			double gauss = exp(-2.3026*rfDist);
			if (gauss < 0.1)
				continue;

			if (drand48() < connectConfigMap[connId].p) {
				uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
				assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
				float synWt = gauss * connectConfigMap[connId].initWt; // scale weight according to gauss distance
				setConnection(grpSrc, grpDest, i, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);//info->connId);
				connectConfigMap[connId].numberOfConnections++;
			}
		}
	}

	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
}

void SNN::connectOneToOne(short int connId) {
	int grpSrc = connectConfigMap[connId].grpSrc;
	int grpDest = connectConfigMap[connId].grpDest;
	assert( groupConfig[grpDest].SizeN == groupConfig[grpSrc].SizeN );

	// NOTE: RadiusRF does not make a difference here: ignore
	for(int nid=groupConfig[grpSrc].StartN,j=groupConfig[grpDest].StartN; nid<=groupConfig[grpSrc].EndN; nid++, j++)  {
		uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
		assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
		float synWt = getWeights(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, nid, grpSrc);
		setConnection(grpSrc, grpDest, nid, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);//info->connId);
		connectConfigMap[connId].numberOfConnections++;
	}

	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
}

// make 'C' random connections from grpSrc to grpDest
void SNN::connectRandom(short int connId) {
	int grpSrc = connectConfigMap[connId].grpSrc;
	int grpDest = connectConfigMap[connId].grpDest;

	// rebuild struct for easier handling
	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);

	for(int pre_nid=groupConfig[grpSrc].StartN; pre_nid<=groupConfig[grpSrc].EndN; pre_nid++) {
		Point3D loc_pre = getNeuronLocation3D(pre_nid); // 3D coordinates of i
		for(int post_nid=groupConfig[grpDest].StartN; post_nid<=groupConfig[grpDest].EndN; post_nid++) {
			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_post = getNeuronLocation3D(post_nid); // 3D coordinates of j
			if (!isPoint3DinRF(radius, loc_pre, loc_post))
				continue;

			if (drand48() < connectConfigMap[connId].p) {
				//uint8_t dVal = info->minDelay + (int)(0.5+(drand48()*(info->maxDelay-info->minDelay)));
				uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
				assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
				float synWt = getWeights(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, pre_nid, grpSrc);
				setConnection(grpSrc, grpDest, pre_nid, post_nid, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId); //info->connId);
				connectConfigMap[connId].numberOfConnections++;
			}
		}
	}

	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
}

// user-defined functions called here...
// This is where we define our user-defined call-back function.  -- KDC
void SNN::connectUserDefined(short int connId) {
	int grpSrc = connectConfigMap[connId].grpSrc;
	int grpDest = connectConfigMap[connId].grpDest;
	connectConfigMap[connId].maxDelay = 0;
	for(int nid=groupConfig[grpSrc].StartN; nid<=groupConfig[grpSrc].EndN; nid++) {
		for(int nid2=groupConfig[grpDest].StartN; nid2 <= groupConfig[grpDest].EndN; nid2++) {
			int srcId  = nid  - groupConfig[grpSrc].StartN;
			int destId = nid2 - groupConfig[grpDest].StartN;
			float weight, maxWt, delay;
			bool connected;

			connectConfigMap[connId].conn->connect(this, grpSrc, srcId, grpDest, destId, weight, maxWt, delay, connected);
			if(connected)  {
				if (GET_FIXED_PLASTIC(connectConfigMap[connId].connProp) == SYN_FIXED)
					maxWt = weight;

				connectConfigMap[connId].maxWt = maxWt;

				assert(delay >= 1);
				assert(delay <= MAX_SYN_DELAY);
				assert(abs(weight) <= abs(maxWt));

				// adjust the sign of the weight based on inh/exc connection
				weight = isExcitatoryGroup(grpSrc) ? fabs(weight) : -1.0*fabs(weight);
				maxWt  = isExcitatoryGroup(grpSrc) ? fabs(maxWt)  : -1.0*fabs(maxWt);

				setConnection(grpSrc, grpDest, nid, nid2, weight, maxWt, delay, connectConfigMap[connId].connProp, connId);// info->connId);
				connectConfigMap[connId].numberOfConnections++;
				if(delay > connectConfigMap[connId].maxDelay) {
					connectConfigMap[connId].maxDelay = delay;
				}
			}
		}
	}

	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
}


// delete all objects (CPU and GPU side)
void SNN::deleteObjects() {
	if (simulatorDeleted)
		return;

	printSimSummary();

	// fclose file streams, unless in custom mode
	if (loggerMode_ != CUSTOM) {
		// don't fclose if it's stdout or stderr, otherwise they're gonna stay closed for the rest of the process
		if (fpInf_!=NULL && fpInf_!=stdout && fpInf_!=stderr)
			fclose(fpInf_);
		if (fpErr_!=NULL && fpErr_!=stdout && fpErr_!=stderr)
			fclose(fpErr_);
		if (fpDeb_!=NULL && fpDeb_!=stdout && fpDeb_!=stderr)
			fclose(fpDeb_);
		if (fpLog_!=NULL && fpLog_!=stdout && fpLog_!=stderr)
			fclose(fpLog_);
	}

	// deallocate objects
	resetMonitors(true);
	resetConnectionConfigs(true);
	resetRuntimeData(true);

	// do the same as above, but for snn_gpu.cu
	deleteObjects_GPU();
	simulatorDeleted = true;
}

void SNN::doSnnSim() {
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

int SNN::findGrpId(int nid) {
	KERNEL_WARN("Using findGrpId is deprecated, use array grpIds[] instead...");
	for(int g=0; g < numGrp; g++) {
		if(nid >=groupConfig[g].StartN && (nid <=groupConfig[g].EndN)) {
			return g;
		}
	}
	KERNEL_ERROR("findGrp(): cannot find the group for neuron %d", nid);
	exitSimulation(1);
}

void SNN::findMaxNumSynapses(int* numPostSynapses, int* numPreSynapses) {
	*numPostSynapses = 0;
	*numPreSynapses = 0;

	//  scan all the groups and find the required information
	for (int g=0; g<numGrp; g++) {
		// find the values for maximum postsynaptic length
		// and maximum pre-synaptic length
		if (groupConfig[g].numPostSynapses >= *numPostSynapses)
			*numPostSynapses = groupConfig[g].numPostSynapses;
		if (groupConfig[g].numPreSynapses >= *numPreSynapses)
			*numPreSynapses = groupConfig[g].numPreSynapses;
	}
}

void SNN::generateSpikes() {
	PropagatedSpikeBuffer::const_iterator srg_iter;
	PropagatedSpikeBuffer::const_iterator srg_iter_end = pbuf->endSpikeTargetGroups();

	for( srg_iter = pbuf->beginSpikeTargetGroups(); srg_iter != srg_iter_end; ++srg_iter )  {
		// Get the target neurons for the given groupId
		int nid	 = srg_iter->stg;
		//delaystep_t del = srg_iter->delay;
		//generate a spike to all the target neurons from source neuron nid with a delay of del
		short int g = snnRuntimeData.grpIds[nid];

		addSpikeToTable (nid, g);
		spikeCountAll1secHost++;
		nPoissonSpikes++;
	}

	// advance the time step to the next phase...
	pbuf->nextTimeStep();
}

void SNN::generateSpikesFromFuncPtr(int grpId) {
	// \FIXME this function is a mess
	bool done;
	SpikeGeneratorCore* spikeGen = groupConfig[grpId].spikeGen;
	int timeSlice = groupConfig[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;
	for(int i = groupConfig[grpId].StartN; i <= groupConfig[grpId].EndN; i++) {
		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = snnRuntimeData.lastSpikeTime[i];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		// the end of the valid time window is either the length of the scheduling time slice from now (because that
		// is the max of the allowed propagated buffer size) or simply the end of the simulation
		unsigned int endOfTimeWindow = MIN(currTime+timeSlice,simTimeRunStop);

		done = false;
		while (!done) {
			// generate the next spike time (nextSchedTime) from the nextSpikeTime callback
			unsigned int nextSchedTime = spikeGen->nextSpikeTime(this, grpId, i - groupConfig[grpId].StartN, currTime, 
				nextTime, endOfTimeWindow);

			// the generated spike time is valid only if:
			// - it has not been scheduled before (nextSchedTime > nextTime)
			//    - but careful: we would drop spikes at t=0, because we cannot initialize nextTime to -1...
			// - it is within the scheduling time slice (nextSchedTime < endOfTimeWindow)
			// - it is not in the past (nextSchedTime >= currTime)
			if ((nextSchedTime==0 || nextSchedTime>nextTime) && nextSchedTime<endOfTimeWindow && nextSchedTime>=currTime) {
//				fprintf(stderr,"%u: spike scheduled for %d at %u\n",currTime, i-groupConfig[grpId].StartN,nextSchedTime);
				// scheduled spike...
				// \TODO CPU mode does not check whether the same AER event has been scheduled before (bug #212)
				// check how GPU mode does it, then do the same here.
				nextTime = nextSchedTime;
				pbuf->scheduleSpikeTargetGroup(i, nextTime - currTime);
				spikeCnt++;

				// update number of spikes if SpikeCounter set
				if (groupConfig[grpId].withSpikeCounter) {
					int bufPos = groupConfig[grpId].spkCntBufPos; // retrieve buf pos
					int bufNeur = i-groupConfig[grpId].StartN;
					spkCntBuf[bufPos][bufNeur]++;
				}
			} else {
				done = true;
			}
		}
	}
}

void SNN::generateSpikesFromRate(int grpId) {
	bool done;
	PoissonRate* rate = groupConfig[grpId].RatePtr;
	float refPeriod = groupConfig[grpId].RefractPeriod;
	int timeSlice   = groupConfig[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;

	if (rate == NULL)
		return;

	if (rate->isOnGPU()) {
		KERNEL_ERROR("Specifying rates on the GPU but using the CPU SNN is not supported.");
		exitSimulation(1);
	}

	const int nNeur = rate->getNumNeurons();
	if (nNeur != groupConfig[grpId].SizeN) {
		KERNEL_ERROR("Length of PoissonRate array (%d) did not match number of neurons (%d) for group %d(%s).",
			nNeur, groupConfig[grpId].SizeN, grpId, getGroupName(grpId).c_str());
		exitSimulation(1);
	}

	for (int neurId=0; neurId<nNeur; neurId++) {
		float frate = rate->getRate(neurId);

		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = snnRuntimeData.lastSpikeTime[groupConfig[grpId].StartN + neurId];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		done = false;
		while (!done && frate>0) {
			nextTime = poissonSpike(nextTime, frate/1000.0, refPeriod);
			// found a valid timeSlice
			if (nextTime < (currTime+timeSlice)) {
				if (nextTime >= currTime) {
//					int nid = groupConfig[grpId].StartN+cnt;
					pbuf->scheduleSpikeTargetGroup(groupConfig[grpId].StartN + neurId, nextTime-currTime);
					spikeCnt++;

					// update number of spikes if SpikeCounter set
					if (groupConfig[grpId].withSpikeCounter) {
						int bufPos = groupConfig[grpId].spkCntBufPos; // retrieve buf pos
						spkCntBuf[bufPos][neurId]++;
					}
				}
			}
			else {
				done=true;
			}
		}
	}
}

inline int SNN::getPoissNeuronPos(int nid) {
	int nPos = nid-numNReg;
	assert(nid >= numNReg);
	assert(nid < numN);
	assert((nid-numNReg) < numNPois);
	return nPos;
}

//We need pass the neuron id (nid) and the grpId just for the case when we want to
//ramp up/down the weights.  In that case we need to set the weights of each synapse
//depending on their nid (their position with respect to one another). -- KDC
float SNN::getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId) {
	float actWts;
	// \FIXME: are these ramping thingies still supported?
	bool setRandomWeights   = GET_INITWTS_RANDOM(connProp);
	bool setRampDownWeights = GET_INITWTS_RAMPDOWN(connProp);
	bool setRampUpWeights   = GET_INITWTS_RAMPUP(connProp);

	if (setRandomWeights)
		actWts = initWt * drand48();
	else if (setRampUpWeights)
		actWts = (initWt + ((nid - groupConfig[grpId].StartN) * (maxWt - initWt) / groupConfig[grpId].SizeN));
	else if (setRampDownWeights)
		actWts = (maxWt - ((nid - groupConfig[grpId].StartN) * (maxWt - initWt) / groupConfig[grpId].SizeN));
	else
		actWts = initWt;

	return actWts;
}

// initialize all the synaptic weights to appropriate values..
// total size of the synaptic connection is 'length' ...
void SNN::initSynapticWeights() {
	// Initialize the network wtChange, wt, synaptic firing time
	snnRuntimeData.wtChange         = new float[preSynCnt];
	snnRuntimeData.synSpikeTime     = new uint32_t[preSynCnt];
	cpuSnnSz.synapticInfoSize = sizeof(float)*(preSynCnt*2);

	resetSynapticConnections(false);
}

// checks whether a connection ID contains plastic synapses O(#connections)
bool SNN::isConnectionPlastic(short int connId) {
	assert(connId != ALL);
	assert(connId < numConnections);

	//// search linked list for right connection ID
	//ConnectConfig* connInfo = connectBegin;
	//bool isPlastic = false;
	//while (connInfo) {
	//	if (connId == connInfo->connId) {
	//		// get syn wt type from connection property
	//		isPlastic = GET_FIXED_PLASTIC(connInfo->connProp);
	//		break;
	//	}

	//	connInfo = connInfo->next;
	//}
	
	return GET_FIXED_PLASTIC(connectConfigMap[connId].connProp);

	//return isPlastic;
}

// returns whether group has homeostasis enabled
bool SNN::isGroupWithHomeostasis(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	return (groupConfig[grpId].WithHomeostasis);
}

// performs various verification checkups before building the network
void SNN::verifyNetwork() {
	// make sure number of neuron parameters have been accumulated correctly
	// NOTE: this used to be updateParameters
	verifyNumNeurons();

	// make sure STDP post-group has some incoming plastic connections
	verifySTDP();

	// make sure every group with homeostasis also has STDP
	verifyHomeostasis();
}

// checks whether STDP is set on a post-group with incoming plastic connections
void SNN::verifySTDP() {
	for (int grpId=0; grpId<getNumGroups(); grpId++) {
		if (groupConfig[grpId].WithSTDP) {
			// for each post-group, check if any of the incoming connections are plastic
			//ConnectConfig* connInfo = connectBegin;
			bool isAnyPlastic = false;
			//while (connInfo) {
			for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
				if (it->second.grpDest == grpId) {
					// get syn wt type from connection property
					isAnyPlastic |= GET_FIXED_PLASTIC(it->second.connProp);
					if (isAnyPlastic) {
						// at least one plastic connection found: break while
						break;
					}
				}
				//connInfo = connInfo->next;
			}
			if (!isAnyPlastic) {
				KERNEL_ERROR("If STDP on group %d (%s) is set, group must have some incoming plastic connections.",
					grpId, groupInfo[grpId].Name.c_str());
				exitSimulation(1);
			}
		}
	}
}

// checks whether every group with Homeostasis also has STDP
void SNN::verifyHomeostasis() {
	for (int grpId=0; grpId<getNumGroups(); grpId++) {
		if (groupConfig[grpId].WithHomeostasis) {
			if (!groupConfig[grpId].WithSTDP) {
				KERNEL_ERROR("If homeostasis is enabled on group %d (%s), then STDP must be enabled, too.",
					grpId, groupInfo[grpId].Name.c_str());
				exitSimulation(1);
			}
		}
	}
}

// checks whether the numN* class members are consistent and complete
void SNN::verifyNumNeurons() {
	int nExcPois = 0;
	int nInhPois = 0;
	int nExcReg = 0;
	int nInhReg = 0;

	//  scan all the groups and find the required information
	//  about the group (numN, numPostSynapses, numPreSynapses and others).
	for(int g=0; g<numGrp; g++)  {
		if (groupConfig[g].Type==UNKNOWN_NEURON) {
			KERNEL_ERROR("Unknown group for %d (%s)", g, groupInfo[g].Name.c_str());
			exitSimulation(1);
		}

		if (IS_INHIBITORY_TYPE(groupConfig[g].Type) && !(groupConfig[g].Type & POISSON_NEURON))
			nInhReg += groupConfig[g].SizeN;
		else if (IS_EXCITATORY_TYPE(groupConfig[g].Type) && !(groupConfig[g].Type & POISSON_NEURON))
			nExcReg += groupConfig[g].SizeN;
		else if (IS_EXCITATORY_TYPE(groupConfig[g].Type) &&  (groupConfig[g].Type & POISSON_NEURON))
			nExcPois += groupConfig[g].SizeN;
		else if (IS_INHIBITORY_TYPE(groupConfig[g].Type) &&  (groupConfig[g].Type & POISSON_NEURON))
			nInhPois += groupConfig[g].SizeN;
	}

	// check the newly gathered information with class members
	if (numN != nExcReg+nInhReg+nExcPois+nInhPois) {
		KERNEL_ERROR("nExcReg+nInhReg+nExcPois+nInhPois=%d does not add up to numN=%d",
			nExcReg+nInhReg+nExcPois+nInhPois, numN);
		exitSimulation(1);
	}
	if (numNReg != nExcReg+nInhReg) {
		KERNEL_ERROR("nExcReg+nInhReg=%d does not add up to numNReg=%d", nExcReg+nInhReg, numNReg);
		exitSimulation(1);
	}
	if (numNPois != nExcPois+nInhPois) {
		KERNEL_ERROR("nExcPois+nInhPois=%d does not add up to numNPois=%d", nExcPois+nInhPois, numNPois);
		exitSimulation(1);
	}
//	printf("numN=%d == %d\n",numN,nExcReg+nInhReg+nExcPois+nInhPois);
//	printf("numNReg=%d == %d\n",numNReg, nExcReg+nInhReg);
//	printf("numNPois=%d == %d\n",numNPois, nExcPois+nInhPois);
}

// \FIXME: not sure where this should go... maybe create some helper file?
bool SNN::isPoint3DinRF(const RadiusRF& radius, const Point3D& pre, const Point3D& post) {
	// Note: RadiusRF rad is assumed to be the fanning in to the post neuron. So if the radius is 10 pixels, it means
	// that if you look at the post neuron, it will receive input from neurons that code for locations no more than
	// 10 pixels away. (The opposite is called a response/stimulus field.)

	double rfDist = getRFDist3D(radius, pre, post);
	return (rfDist >= 0.0 && rfDist <= 1.0);
}

double SNN::getRFDist3D(const RadiusRF& radius, const Point3D& pre, const Point3D& post) {
	// Note: RadiusRF rad is assumed to be the fanning in to the post neuron. So if the radius is 10 pixels, it means
	// that if you look at the post neuron, it will receive input from neurons that code for locations no more than
	// 10 pixels away.

	// ready output argument
	// SNN::isPoint3DinRF() will return true (connected) if rfDist e[0.0, 1.0]
	double rfDist = -1.0;

	// pre and post are connected in a generic 3D ellipsoid RF if x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1.0, where
	// x = pre.x-post.x, y = pre.y-post.y, z = pre.z-post.z
	// x < 0 means:  connect if y and z satisfy some constraints, but ignore x
	// x == 0 means: connect if y and z satisfy some constraints, and enforce pre.x == post.x
	if (radius.radX==0 && pre.x!=post.x || radius.radY==0 && pre.y!=post.y || radius.radZ==0 && pre.z!=post.z) {
		rfDist = -1.0;
	} else {
		// 3D ellipsoid: x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1.0
		double xTerm = (radius.radX<=0) ? 0.0 : pow(pre.x-post.x,2)/pow(radius.radX,2);
		double yTerm = (radius.radY<=0) ? 0.0 : pow(pre.y-post.y,2)/pow(radius.radY,2);
		double zTerm = (radius.radZ<=0) ? 0.0 : pow(pre.z-post.z,2)/pow(radius.radZ,2);
		rfDist = xTerm + yTerm + zTerm;
	}

	return rfDist;
}

// will be used in generateSpikesFromRate
// The time between each pair of consecutive events has an exponential distribution with parameter \lambda and
// each of these ISI values is assumed to be independent of other ISI values.
// What follows a Poisson distribution is the actual number of spikes sent during a certain interval.
unsigned int SNN::poissonSpike(unsigned int currTime, float frate, int refractPeriod) {
	// refractory period must be 1 or greater, 0 means could have multiple spikes specified at the same time.
	assert(refractPeriod>0);
	assert(frate>=0.0f);

	bool done = false;
	unsigned int nextTime = 0;
	while (!done) {
		// A Poisson process will always generate inter-spike-interval (ISI) values from an exponential distribution.
		float randVal = drand48();
		unsigned int tmpVal  = -log(randVal)/frate;

		// add new ISI to current time
		// this might be faster than keeping currTime fixed until drand48() returns a large enough value for the ISI
		nextTime = currTime + tmpVal;

		// reject new firing time if ISI is smaller than refractory period
		if ((nextTime - currTime) >= (unsigned) refractPeriod)
			done = true;
	}

	assert(nextTime != 0);
	return nextTime;
}

void SNN::linkSNN() {
	snnState = LINKED_SNN;
}

int SNN::loadSimulation_internal(bool onlyPlastic) {
	// TSC: so that we can restore the file position later...
	// MB: not sure why though...
	long file_position = ftell(loadSimFID);
	
	int tmpInt;
	float tmpFloat;

	bool readErr = false; // keep track of reading errors
	size_t result;


	// ------- read header ----------------

	fseek(loadSimFID, 0, SEEK_SET);

	// read file signature
	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	readErr |= (result!=1);
	if (tmpInt != 294338571) {
		KERNEL_ERROR("loadSimulation: Unknown file signature. This does not seem to be a "
			"simulation file created with CARLsim::saveSimulation.");
		exitSimulation(-1);
	}

	// read file version number
	result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	readErr |= (result!=1);
	if (tmpFloat > 0.2f) {
		KERNEL_ERROR("loadSimulation: Unsupported version number (%f)",tmpFloat);
		exitSimulation(-1);
	}

	// read simulation time
	result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	readErr |= (result!=1);

	// read execution time
	result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	readErr |= (result!=1);

	// read number of neurons
	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	readErr |= (result!=1);
	if (tmpInt != numN) {
		KERNEL_ERROR("loadSimulation: Number of neurons in file (%d) and simulation (%d) don't match.",
			tmpInt, numN);
		exitSimulation(-1);
	}

	// read number of pre-synapses
	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	readErr |= (result!=1);
	if (preSynCnt != tmpInt) {
		KERNEL_ERROR("loadSimulation: preSynCnt in file (%d) and simulation (%d) don't match.",
			tmpInt, preSynCnt);
		exitSimulation(-1);
	}

	// read number of post-synapses
	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	readErr |= (result!=1);
	if (postSynCnt != tmpInt) {
		KERNEL_ERROR("loadSimulation: postSynCnt in file (%d) and simulation (%d) don't match.",
			tmpInt, postSynCnt);
		exitSimulation(-1);
	}

	// read number of groups
	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	readErr |= (result!=1);
	if (tmpInt != numGrp) {
		KERNEL_ERROR("loadSimulation: Number of groups in file (%d) and simulation (%d) don't match.",
			tmpInt, numGrp);
		exitSimulation(-1);
	}

	// throw reading error instead of proceeding
	if (readErr) {
		fprintf(stderr,"loadSimulation: Error while reading file header");
		exitSimulation(-1);
	}


	// ------- read group information ----------------

	for (int g=0; g<numGrp; g++) {
		// read StartN
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);
		if (tmpInt != groupConfig[g].StartN) {
			KERNEL_ERROR("loadSimulation: StartN in file (%d) and grpInfo (%d) for group %d don't match.",
				tmpInt, groupConfig[g].StartN, g);
			exitSimulation(-1);
		}

		// read EndN
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);
		if (tmpInt != groupConfig[g].EndN) {
			KERNEL_ERROR("loadSimulation: EndN in file (%d) and grpInfo (%d) for group %d don't match.",
				tmpInt, groupConfig[g].EndN, g);
			exitSimulation(-1);
		}

		// read SizeX
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);

		// read SizeY
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);

		// read SizeZ
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);

		// read group name
		char name[100];
		result = fread(name, sizeof(char), 100, loadSimFID);
		readErr |= (result!=100);
		if (strcmp(name,groupInfo[g].Name.c_str()) != 0) {
			KERNEL_ERROR("loadSimulation: Group names in file (%s) and grpInfo (%s) don't match.", name,
				groupInfo[g].Name.c_str());
			exitSimulation(-1);
		}
	}

	if (readErr) {
		KERNEL_ERROR("loadSimulation: Error while reading group info");
		exitSimulation(-1);
	}


	// ------- read synapse information ----------------

	for (unsigned int i=0; i<numN; i++) {
		int nrSynapses = 0;

		// read number of synapses
		result = fread(&nrSynapses, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);

		for (int j=0; j<nrSynapses; j++) {
			unsigned int nIDpre;
			unsigned int nIDpost;
			float weight, maxWeight;
			uint8_t delay;
			uint8_t plastic;
			short int connId;

			// read nIDpre
			result = fread(&nIDpre, sizeof(int), 1, loadSimFID);
			readErr |= (result!=1);
			if (nIDpre != i) {
				KERNEL_ERROR("loadSimulation: nIDpre in file (%u) and simulation (%u) don't match.", nIDpre, i);
				exitSimulation(-1);
			}

			// read nIDpost
			result = fread(&nIDpost, sizeof(int), 1, loadSimFID);
			readErr |= (result!=1);
			if (nIDpost >= numN) {
				KERNEL_ERROR("loadSimulation: nIDpre in file (%u) is larger than in simulation (%u).", nIDpost, numN);
				exitSimulation(-1);
			}

			// read weight
			result = fread(&weight, sizeof(float), 1, loadSimFID);
			readErr |= (result!=1);

			short int gIDpre = snnRuntimeData.grpIds[nIDpre];
			if (IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type) && (weight>0)
					|| !IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type) && (weight<0)) {
				KERNEL_ERROR("loadSimulation: Sign of weight value (%s) does not match neuron type (%s)",
					((weight>=0.0f)?"plus":"minus"), 
					(IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type)?"inhibitory":"excitatory"));
				exitSimulation(-1);
			}

			// read max weight
			result = fread(&maxWeight, sizeof(float), 1, loadSimFID);
			readErr |= (result!=1);
			if (IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type) && (maxWeight>=0)
					|| !IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type) && (maxWeight<=0)) {
				KERNEL_ERROR("loadSimulation: Sign of maxWeight value (%s) does not match neuron type (%s)",
					((maxWeight>=0.0f)?"plus":"minus"), 
					(IS_INHIBITORY_TYPE(groupConfig[gIDpre].Type)?"inhibitory":"excitatory"));
				exitSimulation(-1);
			}

			// read delay
			result = fread(&delay, sizeof(uint8_t), 1, loadSimFID);
			readErr |= (result!=1);
			if (delay > MAX_SYN_DELAY) {
				KERNEL_ERROR("loadSimulation: delay in file (%d) is larger than MAX_SYN_DELAY (%d)",
					(int)delay, (int)MAX_SYN_DELAY);
				exitSimulation(-1);
			}

			assert(!isnan(weight));
			// read plastic/fixed
			result = fread(&plastic, sizeof(uint8_t), 1, loadSimFID);
			readErr |= (result!=1);

			// read connection ID
			result = fread(&connId, sizeof(short int), 1, loadSimFID);
			readErr |= (result!=1);

			if ((plastic && onlyPlastic) || (!plastic && !onlyPlastic)) {
				int gIDpost = snnRuntimeData.grpIds[nIDpost];
				int connProp = SET_FIXED_PLASTIC(plastic?SYN_PLASTIC:SYN_FIXED);

				setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp, connId);
				groupInfo[gIDpre].sumPostConn++;
				groupInfo[gIDpost].sumPreConn++;

				if (delay > groupConfig[gIDpre].MaxDelay)
					groupConfig[gIDpre].MaxDelay = delay;
			}
		}
	}

	fseek(loadSimFID,file_position,SEEK_SET);

	return 0;
}


// The post synaptic connections are sorted based on delay here so that we can reduce storage requirement
// and generation of spike at the post-synaptic side.
// We also create the delay_info array has the delay_start and delay_length parameter
void SNN::reorganizeDelay()
{
	for(int grpId=0; grpId < numGrp; grpId++) {
		for(int nid=groupConfig[grpId].StartN; nid <= groupConfig[grpId].EndN; nid++) {
			unsigned int jPos=0;					// this points to the top of the delay queue
			unsigned int cumN=snnRuntimeData.cumulativePost[nid];	// cumulativePost[] is unsigned int
			unsigned int cumDelayStart=0; 			// Npost[] is unsigned short
			for(int td = 0; td < maxDelay_; td++) {
				unsigned int j=jPos;				// start searching from top of the queue until the end
				unsigned int cnt=0;					// store the number of nodes with a delay of td;
				while(j < snnRuntimeData.Npost[nid]) {
					// found a node j with delay=td and we put
					// the delay value = 1 at array location td=0;
					if(td==(tmp_SynapticDelay[cumN+j]-1)) {
						assert(jPos<snnRuntimeData.Npost[nid]);
						swapConnections(nid, j, jPos);

						jPos=jPos+1;
						cnt=cnt+1;
					}
					j=j+1;
				}

				// update the delay_length and start values...
				snnRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_length	     = cnt;
				snnRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_index_start  = cumDelayStart;
				cumDelayStart += cnt;

				assert(cumDelayStart <= snnRuntimeData.Npost[nid]);
			}

			// total cumulative delay should be equal to number of post-synaptic connections at the end of the loop
			assert(cumDelayStart == snnRuntimeData.Npost[nid]);
			for(unsigned int j=1; j < snnRuntimeData.Npost[nid]; j++) {
				unsigned int cumN=snnRuntimeData.cumulativePost[nid]; // cumulativePost[] is unsigned int
				if( tmp_SynapticDelay[cumN+j] < tmp_SynapticDelay[cumN+j-1]) {
	  				KERNEL_ERROR("Post-synaptic delays not sorted correctly... id=%d, delay[%d]=%d, delay[%d]=%d",
						nid, j, tmp_SynapticDelay[cumN+j], j-1, tmp_SynapticDelay[cumN+j-1]);
					assert( tmp_SynapticDelay[cumN+j] >= tmp_SynapticDelay[cumN+j-1]);
				}
			}
		}
	}
}

void SNN::optimizeAndPartitionSNN() {
	snnState = OPTIMIZED_PARTITIONED_SNN;
}

void SNN::resetConductances() {
	if (sim_with_conductances) {
		memset(snnRuntimeData.gAMPA, 0, sizeof(float)*numNReg);
		if (sim_with_NMDA_rise) {
			memset(snnRuntimeData.gNMDA_r, 0, sizeof(float)*numNReg);
			memset(snnRuntimeData.gNMDA_d, 0, sizeof(float)*numNReg);
		} else {
			memset(snnRuntimeData.gNMDA, 0, sizeof(float)*numNReg);
		}
		memset(snnRuntimeData.gGABAa, 0, sizeof(float)*numNReg);
		if (sim_with_GABAb_rise) {
			memset(snnRuntimeData.gGABAb_r, 0, sizeof(float)*numNReg);
			memset(snnRuntimeData.gGABAb_d, 0, sizeof(float)*numNReg);
		} else {
			memset(snnRuntimeData.gGABAb, 0, sizeof(float)*numNReg);
		}
	}
}

void SNN::resetCounters() {
	assert(numNReg <= numN);
	memset(snnRuntimeData.curSpike, 0, sizeof(bool) * numN);
}

void SNN::resetCPUTiming() {
	prevCpuExecutionTime = cumExecutionTime;
	cpuExecutionTime     = 0.0;
}

void SNN::resetCurrent() {
	assert(snnRuntimeData.current != NULL);
	memset(snnRuntimeData.current, 0, sizeof(float) * numNReg);
}

void SNN::resetFiringInformation() {
	// Reset firing tables and time tables to default values..

	// reset Various Times..
	spikeCountAllHost	  = 0;
	spikeCountAll1secHost = 0;
	spikeCountD2Host = 0;
	spikeCountD1Host = 0;
	secD1fireCntHost  = 0;
	secD2fireCntHost  = 0;

	for(int i=0; i < numGrp; i++) {
		groupConfig[i].FiringCount1sec = 0;
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

void SNN::resetGPUTiming() {
	prevGpuExecutionTime = cumExecutionTime;
	gpuExecutionTime     = 0.0;
}

void SNN::resetGroups() {
	for(int g=0; (g < numGrp); g++) {
		// reset spike generator group...
		if (groupConfig[g].isSpikeGenerator) {
			groupConfig[g].CurrTimeSlice = groupConfig[g].NewTimeSlice;
			groupConfig[g].SliceUpdateTime  = 0;
			for(int nid=groupConfig[g].StartN; nid <= groupConfig[g].EndN; nid++)
				resetPoissonNeuron(nid, g);
		}
		// reset regular neuron group...
		else {
			for(int nid=groupConfig[g].StartN; nid <= groupConfig[g].EndN; nid++)
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

void SNN::resetNeuromodulator(int grpId) {
	snnRuntimeData.grpDA[grpId] = groupConfig[grpId].baseDP;
	snnRuntimeData.grp5HT[grpId] = groupConfig[grpId].base5HT;
	snnRuntimeData.grpACh[grpId] = groupConfig[grpId].baseACh;
	snnRuntimeData.grpNE[grpId] = groupConfig[grpId].baseNE;
}

void SNN::resetNeuron(unsigned int neurId, int grpId) {
	assert(neurId < numNReg);
    if (groupInfo[grpId].Izh_a == -1) {
		KERNEL_ERROR("setNeuronParameters must be called for group %s (%d)",groupInfo[grpId].Name.c_str(),grpId);
		exitSimulation(1);
	}

	snnRuntimeData.Izh_a[neurId] = groupInfo[grpId].Izh_a + groupInfo[grpId].Izh_a_sd*(float)drand48();
	snnRuntimeData.Izh_b[neurId] = groupInfo[grpId].Izh_b + groupInfo[grpId].Izh_b_sd*(float)drand48();
	snnRuntimeData.Izh_c[neurId] = groupInfo[grpId].Izh_c + groupInfo[grpId].Izh_c_sd*(float)drand48();
	snnRuntimeData.Izh_d[neurId] = groupInfo[grpId].Izh_d + groupInfo[grpId].Izh_d_sd*(float)drand48();

	snnRuntimeData.voltage[neurId] = snnRuntimeData.Izh_c[neurId];	// initial values for new_v
	snnRuntimeData.recovery[neurId] = snnRuntimeData.Izh_b[neurId]*snnRuntimeData.voltage[neurId]; // initial values for u


 	if (groupConfig[grpId].WithHomeostasis) {
		// set the baseFiring with some standard deviation.
		if(drand48()>0.5)   {
			snnRuntimeData.baseFiring[neurId] = groupInfo[grpId].baseFiring + groupInfo[grpId].baseFiringSD*-log(drand48());
		} else  {
			snnRuntimeData.baseFiring[neurId] = groupInfo[grpId].baseFiring - groupInfo[grpId].baseFiringSD*-log(drand48());
			if(snnRuntimeData.baseFiring[neurId] < 0.1) snnRuntimeData.baseFiring[neurId] = 0.1;
		}

		if( groupInfo[grpId].baseFiring != 0.0) {
			snnRuntimeData.avgFiring[neurId]  = snnRuntimeData.baseFiring[neurId];
		} else {
			snnRuntimeData.baseFiring[neurId] = 0.0;
			snnRuntimeData.avgFiring[neurId]  = 0;
		}
	}

	snnRuntimeData.lastSpikeTime[neurId]  = MAX_SIMULATION_TIME;

	if(groupConfig[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(neurId,j);
			snnRuntimeData.stpu[ind] = 0.0f;
			snnRuntimeData.stpx[ind] = 1.0f;
		}
	}
}

void SNN::resetMonitors(bool deallocate) {
	// order is important! monitor objects might point to SNN or CARLsim,
	// need to deallocate them first


	// -------------- DEALLOCATE MONITOR OBJECTS ---------------------- //

	// delete all SpikeMonitor objects
	// don't kill SpikeMonitorCore objects, they will get killed automatically
	for (int i=0; i<numSpikeMonitor; i++) {
		if (spikeMonList[i]!=NULL && deallocate) delete spikeMonList[i];
		spikeMonList[i]=NULL;
	}

	// delete all GroupMonitor objects
	// don't kill GroupMonitorCore objects, they will get killed automatically
	for (int i=0; i<numGroupMonitor; i++) {
		if (groupMonList[i]!=NULL && deallocate) delete groupMonList[i];
		groupMonList[i]=NULL;
	}

	// delete all ConnectionMonitor objects
	// don't kill ConnectionMonitorCore objects, they will get killed automatically
	for (int i=0; i<numConnectionMonitor; i++) {
		if (connMonList[i]!=NULL && deallocate) delete connMonList[i];
		connMonList[i]=NULL;
	}
}

void SNN::resetConnectionConfigs(bool deallocate) {
	// clear all existing connection info
	if (deallocate) {
		//while (connectBegin) {
		//	ConnectConfig* nextConn = connectBegin->next;
		//	if (connectBegin!=NULL && deallocate) {
		//		free(connectBegin);
		//		connectBegin = nextConn;
		//	}
		//}
		connectConfigMap.clear();
	}
	//connectBegin=NULL;
}

void SNN::resetRuntimeData(bool deallocate) {
	// delete all Spike Counters
	for (int i=0; i<numSpkCnt; i++) {
		if (spkCntBuf[i]!=NULL && deallocate)
			delete[] spkCntBuf[i];
		spkCntBuf[i]=NULL;
	}

	if (pbuf!=NULL && deallocate) delete pbuf;
	if (snnRuntimeData.spikeGenBits!=NULL && deallocate) delete[] snnRuntimeData.spikeGenBits;
	pbuf=NULL; snnRuntimeData.spikeGenBits=NULL;

	// clear data (i.e., concentration of neuromodulator) of groups
	if (snnRuntimeData.grpDA != NULL && deallocate) delete [] snnRuntimeData.grpDA;
	if (snnRuntimeData.grp5HT != NULL && deallocate) delete [] snnRuntimeData.grp5HT;
	if (snnRuntimeData.grpACh != NULL && deallocate) delete [] snnRuntimeData.grpACh;
	if (snnRuntimeData.grpNE != NULL && deallocate) delete [] snnRuntimeData.grpNE;
	snnRuntimeData.grpDA = NULL;
	snnRuntimeData.grp5HT = NULL;
	snnRuntimeData.grpACh = NULL;
	snnRuntimeData.grpNE = NULL;

	// clear assistive data buffer for group monitor
	if (deallocate) {
		for (int i = 0; i < numGrp; i++) {
			if (snnRuntimeData.grpDABuffer[i] != NULL) delete [] snnRuntimeData.grpDABuffer[i];
			if (snnRuntimeData.grp5HTBuffer[i] != NULL) delete [] snnRuntimeData.grp5HTBuffer[i];
			if (snnRuntimeData.grpAChBuffer[i] != NULL) delete [] snnRuntimeData.grpAChBuffer[i];
			if (snnRuntimeData.grpNEBuffer[i] != NULL) delete [] snnRuntimeData.grpNEBuffer[i];
			snnRuntimeData.grpDABuffer[i] = NULL;
			snnRuntimeData.grp5HTBuffer[i] = NULL;
			snnRuntimeData.grpAChBuffer[i] = NULL;
			snnRuntimeData.grpNEBuffer[i] = NULL;
		}
	} else {
		memset(snnRuntimeData.grpDABuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(snnRuntimeData.grp5HTBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(snnRuntimeData.grpAChBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(snnRuntimeData.grpNEBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
	}


	// -------------- DEALLOCATE CORE OBJECTS ---------------------- //

	if (snnRuntimeData.voltage!=NULL && deallocate) delete[] snnRuntimeData.voltage;
	if (snnRuntimeData.recovery!=NULL && deallocate) delete[] snnRuntimeData.recovery;
	if (snnRuntimeData.current!=NULL && deallocate) delete[] snnRuntimeData.current;
	if (snnRuntimeData.extCurrent!=NULL && deallocate) delete[] snnRuntimeData.extCurrent;
	snnRuntimeData.voltage=NULL; snnRuntimeData.recovery=NULL; snnRuntimeData.current=NULL; snnRuntimeData.extCurrent=NULL;

	if (snnRuntimeData.Izh_a!=NULL && deallocate) delete[] snnRuntimeData.Izh_a;
	if (snnRuntimeData.Izh_b!=NULL && deallocate) delete[] snnRuntimeData.Izh_b;
	if (snnRuntimeData.Izh_c!=NULL && deallocate) delete[] snnRuntimeData.Izh_c;
	if (snnRuntimeData.Izh_d!=NULL && deallocate) delete[] snnRuntimeData.Izh_d;
	snnRuntimeData.Izh_a=NULL; snnRuntimeData.Izh_b=NULL; snnRuntimeData.Izh_c=NULL; snnRuntimeData.Izh_d=NULL;

	if (snnRuntimeData.Npre!=NULL && deallocate) delete[] snnRuntimeData.Npre;
	if (snnRuntimeData.Npre_plastic!=NULL && deallocate) delete[] snnRuntimeData.Npre_plastic;
	if (snnRuntimeData.Npost!=NULL && deallocate) delete[] snnRuntimeData.Npost;
	snnRuntimeData.Npre=NULL; snnRuntimeData.Npre_plastic=NULL; snnRuntimeData.Npost=NULL;

	if (snnRuntimeData.cumulativePre!=NULL && deallocate) delete[] snnRuntimeData.cumulativePre;
	if (snnRuntimeData.cumulativePost!=NULL && deallocate) delete[] snnRuntimeData.cumulativePost;
	snnRuntimeData.cumulativePre=NULL; snnRuntimeData.cumulativePost=NULL;

	if (snnRuntimeData.gAMPA!=NULL && deallocate) delete[] snnRuntimeData.gAMPA;
	if (snnRuntimeData.gNMDA!=NULL && deallocate) delete[] snnRuntimeData.gNMDA;
	if (snnRuntimeData.gNMDA_r!=NULL && deallocate) delete[] snnRuntimeData.gNMDA_r;
	if (snnRuntimeData.gNMDA_d!=NULL && deallocate) delete[] snnRuntimeData.gNMDA_d;
	if (snnRuntimeData.gGABAa!=NULL && deallocate) delete[] snnRuntimeData.gGABAa;
	if (snnRuntimeData.gGABAb!=NULL && deallocate) delete[] snnRuntimeData.gGABAb;
	if (snnRuntimeData.gGABAb_r!=NULL && deallocate) delete[] snnRuntimeData.gGABAb_r;
	if (snnRuntimeData.gGABAb_d!=NULL && deallocate) delete[] snnRuntimeData.gGABAb_d;
	snnRuntimeData.gAMPA=NULL; snnRuntimeData.gNMDA=NULL; snnRuntimeData.gNMDA_r=NULL; snnRuntimeData.gNMDA_d=NULL;
	snnRuntimeData.gGABAa=NULL; snnRuntimeData.gGABAb=NULL; snnRuntimeData.gGABAb_r=NULL; snnRuntimeData.gGABAb_d=NULL;

	if (snnRuntimeData.stpu!=NULL && deallocate) delete[] snnRuntimeData.stpu;
	if (snnRuntimeData.stpx!=NULL && deallocate) delete[] snnRuntimeData.stpx;
	snnRuntimeData.stpu=NULL; snnRuntimeData.stpx=NULL;

	if (snnRuntimeData.avgFiring!=NULL && deallocate) delete[] snnRuntimeData.avgFiring;
	if (snnRuntimeData.baseFiring!=NULL && deallocate) delete[] snnRuntimeData.baseFiring;
	snnRuntimeData.avgFiring=NULL; snnRuntimeData.baseFiring=NULL;

	if (snnRuntimeData.lastSpikeTime!=NULL && deallocate) delete[] snnRuntimeData.lastSpikeTime;
	if (snnRuntimeData.synSpikeTime !=NULL && deallocate) delete[] snnRuntimeData.synSpikeTime;
	if (snnRuntimeData.curSpike!=NULL && deallocate) delete[] snnRuntimeData.curSpike;
	if (snnRuntimeData.nSpikeCnt!=NULL && deallocate) delete[] snnRuntimeData.nSpikeCnt;
	snnRuntimeData.lastSpikeTime=NULL; snnRuntimeData.synSpikeTime=NULL; snnRuntimeData.curSpike=NULL; snnRuntimeData.nSpikeCnt=NULL;

	if (snnRuntimeData.postDelayInfo!=NULL && deallocate) delete[] snnRuntimeData.postDelayInfo;
	if (snnRuntimeData.preSynapticIds!=NULL && deallocate) delete[] snnRuntimeData.preSynapticIds;
	if (snnRuntimeData.postSynapticIds!=NULL && deallocate) delete[] snnRuntimeData.postSynapticIds;
	snnRuntimeData.postDelayInfo=NULL; snnRuntimeData.preSynapticIds=NULL; snnRuntimeData.postSynapticIds=NULL;

	if (snnRuntimeData.wt!=NULL && deallocate) delete[] snnRuntimeData.wt;
	if (snnRuntimeData.maxSynWt!=NULL && deallocate) delete[] snnRuntimeData.maxSynWt;
	if (snnRuntimeData.wtChange !=NULL && deallocate) delete[] snnRuntimeData.wtChange;
	snnRuntimeData.wt=NULL; snnRuntimeData.maxSynWt=NULL; snnRuntimeData.wtChange=NULL;

	if (mulSynFast!=NULL && deallocate) delete[] mulSynFast;
	if (mulSynSlow!=NULL && deallocate) delete[] mulSynSlow;
	if (snnRuntimeData.cumConnIdPre!=NULL && deallocate) delete[] snnRuntimeData.cumConnIdPre;
	mulSynFast=NULL; mulSynSlow=NULL; snnRuntimeData.cumConnIdPre=NULL;

	if (snnRuntimeData.grpIds!=NULL && deallocate) delete[] snnRuntimeData.grpIds;
	snnRuntimeData.grpIds=NULL;

	if (snnRuntimeData.firingTableD2!=NULL && deallocate) delete[] snnRuntimeData.firingTableD2;
	if (snnRuntimeData.firingTableD1!=NULL && deallocate) delete[] snnRuntimeData.firingTableD1;
	if (timeTableD2!=NULL && deallocate) delete[] timeTableD2;
	if (timeTableD1!=NULL && deallocate) delete[] timeTableD1;
	snnRuntimeData.firingTableD2=NULL; snnRuntimeData.firingTableD1=NULL; timeTableD2=NULL; timeTableD1=NULL;

	// clear poisson generator
	if (gpuPoissonRand != NULL) delete gpuPoissonRand;
	gpuPoissonRand = NULL;
}


void SNN::resetPoissonNeuron(unsigned int nid, int grpId) {
	assert(nid < numN);
	snnRuntimeData.lastSpikeTime[nid]  = MAX_SIMULATION_TIME;
	if (groupConfig[grpId].WithHomeostasis)
		snnRuntimeData.avgFiring[nid] = 0.0;

	if(groupConfig[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(nid,j);
			snnRuntimeData.stpu[nid] = 0.0f;
			snnRuntimeData.stpx[nid] = 1.0f;
		}
	}
}

void SNN::resetPropogationBuffer() {
	pbuf->reset(0, 1023);
}

// resets nSpikeCnt[]
// used for CPU mode
void SNN::resetSpikeCnt(int grpId) {
	int startGrp, endGrp;

	if (grpId == -1) {
		startGrp = 0;
		endGrp = numGrp;
	} else {
		 startGrp = grpId;
		 endGrp = grpId;
	}

	for (int g = startGrp; g<endGrp; g++) {
		int startN = groupConfig[g].StartN;
		int endN   = groupConfig[g].EndN;
		for (int i=startN; i<=endN; i++)
			snnRuntimeData.nSpikeCnt[i] = 0;
	}
}

//Reset wt, wtChange, pre-firing time values to default values, rewritten to
//integrate changes between JMN and MDR -- KDC
//if changeWeights is false, we should keep the values of the weights as they currently
//are but we should be able to change them to plastic or fixed synapses. -- KDC
void SNN::resetSynapticConnections(bool changeWeights) {
	int j;
	// Reset wt,wtChange,pre-firingtime values to default values...
	for(int destGrp=0; destGrp < numGrp; destGrp++) {
		const char* updateStr = (groupConfig[destGrp].newUpdates == true)?"(**)":"";
		KERNEL_DEBUG("Grp: %d:%s s=%d e=%d %s", destGrp, groupInfo[destGrp].Name.c_str(), groupConfig[destGrp].StartN,
					groupConfig[destGrp].EndN,  updateStr);
		KERNEL_DEBUG("Grp: %d:%s s=%d e=%d  %s",  destGrp, groupInfo[destGrp].Name.c_str(), groupConfig[destGrp].StartN,
					groupConfig[destGrp].EndN, updateStr);

		for(int nid=groupConfig[destGrp].StartN; nid <= groupConfig[destGrp].EndN; nid++) {
			unsigned int offset = snnRuntimeData.cumulativePre[nid];
			for (j=0;j<snnRuntimeData.Npre[nid]; j++) {
				snnRuntimeData.wtChange[offset+j] = 0.0;						// synaptic derivatives is reset
				snnRuntimeData.synSpikeTime[offset+j] = MAX_SIMULATION_TIME;	// some large negative value..
			}
			post_info_t *preIdPtr = &(snnRuntimeData.preSynapticIds[snnRuntimeData.cumulativePre[nid]]);
			float* synWtPtr       = &(snnRuntimeData.wt[snnRuntimeData.cumulativePre[nid]]);
			float* maxWtPtr       = &(snnRuntimeData.maxSynWt[snnRuntimeData.cumulativePre[nid]]);
			int prevPreGrp  = -1;

			for (j=0; j < snnRuntimeData.Npre[nid]; j++,preIdPtr++, synWtPtr++, maxWtPtr++) {
				int preId    = GET_CONN_NEURON_ID((*preIdPtr));
				assert(preId < numN);
				int srcGrp = snnRuntimeData.grpIds[preId];
				short int connId = getConnectId(srcGrp, destGrp);
				//ConnectConfig* connInfo;
				//ConnectConfig* connIterator = connectBegin;
				//while(connIterator) {
				//	if(connIterator->grpSrc == srcGrp && connIterator->grpDest == destGrp) {
				//		//we found the corresponding connection
				//		connInfo=connIterator;
				//		break;
				//	}
				//	//move to the next ConnectConfig
				//	connIterator=connIterator->next;
				//}
				//assert(connInfo != NULL);
				assert(connId != -1);
				//int connProp   = connInfo->connProp;
				bool   synWtType = GET_FIXED_PLASTIC(connectConfigMap[connId].connProp);
				// print debug information...
				if( prevPreGrp != srcGrp) {
					if(nid==groupConfig[destGrp].StartN) {
						const char* updateStr = (connectConfigMap[connId].newUpdates==true)? "(**)":"";
						KERNEL_DEBUG("\t%d (%s) start=%d, type=%s maxWts = %f %s", srcGrp,
										groupInfo[srcGrp].Name.c_str(), j, (j<snnRuntimeData.Npre_plastic[nid]?"P":"F"),
										connectConfigMap[connId].maxWt, updateStr);
					}
					prevPreGrp = srcGrp;
				}

				if(!changeWeights)
					continue;

				// if connection was plastic or if the connection weights were updated we need to reset the weights
				// TODO: How to account for user-defined connection reset
				if ((synWtType == SYN_PLASTIC) ||  connectConfigMap[connId].newUpdates) {
					*synWtPtr = getWeights(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, nid, srcGrp);
					*maxWtPtr = connectConfigMap[connId].maxWt;
				}
			}
		}
		groupConfig[destGrp].newUpdates = false;
	}

	//ConnectConfig* connInfo = connectBegin;
	// clear all existing connection info...
	//while (connInfo) {
	for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
		it->second.newUpdates = false;
		//connInfo = connInfo->next;
	}
}

void SNN::resetTimingTable() {
		memset(timeTableD2, 0, sizeof(int) * (1000 + maxDelay_ + 1));
		memset(timeTableD1, 0, sizeof(int) * (1000 + maxDelay_ + 1));
}


//! nid=neuron id, sid=synapse id, grpId=group id.
inline post_info_t SNN::SET_CONN_ID(int nid, int sid, int grpId) {
	if (sid > CONN_SYN_MASK) {
		KERNEL_ERROR("Error: Syn Id (%d) exceeds maximum limit (%d) for neuron %d (group %d)", sid, CONN_SYN_MASK, nid,
			grpId);
		exitSimulation(1);
	}
	post_info_t p;
	p.postId = (((sid)<<CONN_SYN_NEURON_BITS)+((nid)&CONN_SYN_NEURON_MASK));
	p.grpId  = grpId;
	return p;
}

//! set one specific connection from neuron id 'src' to neuron id 'dest'
inline void SNN::setConnection(int srcGrp,  int destGrp,  unsigned int src, unsigned int dest, float synWt,
									float maxWt, uint8_t dVal, int connProp, short int connId) {
	assert(dest<=CONN_SYN_NEURON_MASK);			// total number of neurons is less than 1 million within a GPU
	assert((dVal >=1) && (dVal <= maxDelay_));

	// adjust sign of weight based on pre-group (negative if pre is inhibitory)
	synWt = isExcitatoryGroup(srcGrp) ? fabs(synWt) : -1.0*fabs(synWt);
	maxWt = isExcitatoryGroup(srcGrp) ? fabs(maxWt) : -1.0*fabs(maxWt);

	// we have exceeded the number of possible connection for one neuron
	if(snnRuntimeData.Npost[src] >= groupConfig[srcGrp].numPostSynapses)	{
		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, groupInfo[srcGrp].Name.c_str(),
					dest, groupInfo[destGrp].Name.c_str(), synWt, dVal);
		KERNEL_ERROR("Large number of postsynaptic connections established (%d), max for this group %d.", snnRuntimeData.Npost[src], groupConfig[srcGrp].numPostSynapses);
		exitSimulation(1);
	}

	if(snnRuntimeData.Npre[dest] >= groupConfig[destGrp].numPreSynapses) {
		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, groupInfo[srcGrp].Name.c_str(),
					dest, groupInfo[destGrp].Name.c_str(), synWt, dVal);
		KERNEL_ERROR("Large number of presynaptic connections established (%d), max for this group %d.", snnRuntimeData.Npre[dest], groupConfig[destGrp].numPreSynapses);
		exitSimulation(1);
	}

	int p = snnRuntimeData.Npost[src];

	assert(snnRuntimeData.Npost[src] >= 0);
	assert(snnRuntimeData.Npre[dest] >= 0);
	assert((src * numPostSynapses_ + p) / numN < numPostSynapses_); // divide by numN to prevent INT overflow

	unsigned int post_pos = snnRuntimeData.cumulativePost[src] + snnRuntimeData.Npost[src];
	unsigned int pre_pos  = snnRuntimeData.cumulativePre[dest] + snnRuntimeData.Npre[dest];

	assert(post_pos < postSynCnt);
	assert(pre_pos  < preSynCnt);

	//generate a new postSynapticIds id for the current connection
	snnRuntimeData.postSynapticIds[post_pos]   = SET_CONN_ID(dest, snnRuntimeData.Npre[dest], destGrp);
	tmp_SynapticDelay[post_pos] = dVal;

	snnRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID(src, snnRuntimeData.Npost[src], srcGrp);
	snnRuntimeData.wt[pre_pos] 	  = synWt;
	snnRuntimeData.maxSynWt[pre_pos] = maxWt;
	snnRuntimeData.cumConnIdPre[pre_pos] = connId;

	bool synWtType = GET_FIXED_PLASTIC(connProp);

	if (synWtType == SYN_PLASTIC) {
		sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
		snnRuntimeData.Npre_plastic[dest]++;
		// homeostasis
		if (groupConfig[destGrp].WithHomeostasis && groupConfig[destGrp].homeoId ==-1)
			groupConfig[destGrp].homeoId = dest; // this neuron info will be printed
	}

	snnRuntimeData.Npre[dest] += 1;
	snnRuntimeData.Npost[src] += 1;

	groupInfo[srcGrp].numPostConn++;
	groupInfo[destGrp].numPreConn++;

	if (snnRuntimeData.Npost[src] > groupInfo[srcGrp].maxPostConn)
		groupInfo[srcGrp].maxPostConn = snnRuntimeData.Npost[src];
	if (snnRuntimeData.Npre[dest] > groupInfo[destGrp].maxPreConn)
	groupInfo[destGrp].maxPreConn = snnRuntimeData.Npre[src];
}

void SNN::setGrpTimeSlice(int grpId, int timeSlice) {
	if (grpId == ALL) {
		for(int g=0; (g < numGrp); g++) {
			if (groupConfig[g].isSpikeGenerator)
				setGrpTimeSlice(g, timeSlice);
		}
	} else {
		assert((timeSlice > 0 ) && (timeSlice <  PROPAGATED_BUFFER_SIZE));
		// the group should be poisson spike generator group
		groupConfig[grpId].NewTimeSlice = timeSlice;
		groupConfig[grpId].CurrTimeSlice = timeSlice;
	}
}

// method to set const member randSeed_
int SNN::setRandSeed(int seed) {
	if (seed<0)
		return time(NULL);
	else if(seed==0)
		return 123;
	else
		return seed;
}

void SNN::startCPUTiming() { prevCpuExecutionTime = cumExecutionTime; }
void SNN::startGPUTiming() { prevGpuExecutionTime = cumExecutionTime; }
void SNN::stopCPUTiming() {
	cpuExecutionTime += (cumExecutionTime - prevCpuExecutionTime);
	prevCpuExecutionTime = cumExecutionTime;
}
void SNN::stopGPUTiming() {
	gpuExecutionTime += (cumExecutionTime - prevGpuExecutionTime);
	prevGpuExecutionTime = cumExecutionTime;
}

// enters testing phase
// in testing, no weight changes can be made, allowing you to evaluate learned weights, etc.
void SNN::startTesting(bool shallUpdateWeights) {
	// because this can be called at any point in time, if we're off the 1-second grid, we want to make
	// sure to apply the accumulated weight changes to the weight matrix
	// but we don't reset the wt update interval counter
	if (shallUpdateWeights && !sim_in_testing) {
		// careful: need to temporarily adjust stdpScaleFactor to make this right
		if (wtANDwtChangeUpdateIntervalCnt_) {
			float storeScaleSTDP = stdpScaleFactor_;
			stdpScaleFactor_ = 1.0f/wtANDwtChangeUpdateIntervalCnt_;

			if (simMode_ == CPU_MODE) {
				updateWeights();
			} else{
				updateWeights_GPU();
			}
			stdpScaleFactor_ = storeScaleSTDP;
		}
	}

	sim_in_testing = true;
	networkConfig.sim_in_testing = true;

	if (simMode_ == GPU_MODE) {
		// copy new network info struct to GPU (|TODO copy only a single boolean)
		copyNetworkConfig();
	}
}

// exits testing phase
void SNN::stopTesting() {
	sim_in_testing = false;
	networkConfig.sim_in_testing = false;

	if (simMode_ == GPU_MODE) {
		// copy new network_info struct to GPU (|TODO copy only a single boolean)
		copyNetworkConfig();
	}
}


void SNN::swapConnections(int nid, int oldPos, int newPos) {
	unsigned int cumN=snnRuntimeData.cumulativePost[nid];

	// Put the node oldPos to the top of the delay queue
	post_info_t tmp = snnRuntimeData.postSynapticIds[cumN+oldPos];
	snnRuntimeData.postSynapticIds[cumN+oldPos]= snnRuntimeData.postSynapticIds[cumN+newPos];
	snnRuntimeData.postSynapticIds[cumN+newPos]= tmp;

	// Ensure that you have shifted the delay accordingly....
	uint8_t tmp_delay = tmp_SynapticDelay[cumN+oldPos];
	tmp_SynapticDelay[cumN+oldPos] = tmp_SynapticDelay[cumN+newPos];
	tmp_SynapticDelay[cumN+newPos] = tmp_delay;

	// update the pre-information for the postsynaptic neuron at the position oldPos.
	post_info_t  postInfo = snnRuntimeData.postSynapticIds[cumN+oldPos];
	int  post_nid = GET_CONN_NEURON_ID(postInfo);
	int  post_sid = GET_CONN_SYN_ID(postInfo);

	post_info_t* preId    = &(snnRuntimeData.preSynapticIds[snnRuntimeData.cumulativePre[post_nid]+post_sid]);
	int  pre_nid  = GET_CONN_NEURON_ID((*preId));
	int  pre_sid  = GET_CONN_SYN_ID((*preId));
	int  pre_gid  = GET_CONN_GRP_ID((*preId));
	assert (pre_nid == nid);
	assert (pre_sid == newPos);
	*preId = SET_CONN_ID( pre_nid, oldPos, pre_gid);

	// update the pre-information for the postsynaptic neuron at the position newPos
	postInfo = snnRuntimeData.postSynapticIds[cumN+newPos];
	post_nid = GET_CONN_NEURON_ID(postInfo);
	post_sid = GET_CONN_SYN_ID(postInfo);

	preId    = &(snnRuntimeData.preSynapticIds[snnRuntimeData.cumulativePre[post_nid]+post_sid]);
	pre_nid  = GET_CONN_NEURON_ID((*preId));
	pre_sid  = GET_CONN_SYN_ID((*preId));
	pre_gid  = GET_CONN_GRP_ID((*preId));
	assert (pre_nid == nid);
	assert (pre_sid == oldPos);
	*preId = SET_CONN_ID( pre_nid, newPos, pre_gid);
}

void SNN::updateConnectionMonitor(short int connId) {
	for (int monId=0; monId<numConnectionMonitor; monId++) {
		if (connId==ALL || connMonCoreList[monId]->getConnectId()==connId) {
			int timeInterval = connMonCoreList[monId]->getUpdateTimeIntervalSec();
			if (timeInterval==1 || timeInterval>1 && (getSimTime()%timeInterval)==0) {
				// this ConnectionMonitor wants periodic recording
				connMonCoreList[monId]->writeConnectFileSnapshot(simTime,
					getWeightMatrix2D(connMonCoreList[monId]->getConnectId()));
			}
		}
	}
}


std::vector< std::vector<float> > SNN::getWeightMatrix2D(short int connId) {
	assert(connId!=ALL);
	//ConnectConfig* connInfo = connectBegin;
	std::vector< std::vector<float> > wtConnId;

	// loop over all connections and find the ones with Connection Monitors
	//while (connInfo) {
	//	if (connInfo->connId==connId) {
			int grpIdPre = connectConfigMap[connId].grpSrc;
			int grpIdPost = connectConfigMap[connId].grpDest;

			// init weight matrix with right dimensions
			for (int i=0; i<groupConfig[grpIdPre].SizeN; i++) {
				std::vector<float> wtSlice;
				for (int j=0; j<groupConfig[grpIdPost].SizeN; j++) {
					wtSlice.push_back(NAN);
				}
				wtConnId.push_back(wtSlice);
			}

			// copy the weights for a given post-group from device
			// \TODO: check if the weights for this grpIdPost have already been copied
			// \TODO: even better, but tricky because of ordering, make copyWeightState connection-based
			if (simMode_==GPU_MODE) {
				copyWeightState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpIdPost);
			}

			for (int postId=groupConfig[grpIdPost].StartN; postId<=groupConfig[grpIdPost].EndN; postId++) {
				unsigned int pos_ij = snnRuntimeData.cumulativePre[postId];
				for (int i=0; i<snnRuntimeData.Npre[postId]; i++, pos_ij++) {
					// skip synapses that belong to a different connection ID
					if (snnRuntimeData.cumConnIdPre[pos_ij] != connId) //connInfo->connId)
						continue;

					// find pre-neuron ID and update ConnectionMonitor container
					int preId = GET_CONN_NEURON_ID(snnRuntimeData.preSynapticIds[pos_ij]);
					wtConnId[preId-getGroupStartNeuronId(grpIdPre)][postId-getGroupStartNeuronId(grpIdPost)] =
						fabs(snnRuntimeData.wt[pos_ij]);
				}
			}
			//break;
	//	}
	//	connInfo = connInfo->next;
	//}

	return wtConnId;
}

void SNN::updateGroupMonitor(int grpId) {
	// don't continue if no group monitors in the network
	if (!numGroupMonitor)
		return;

	if (grpId == ALL) {
		for (int g = 0; g < numGrp; g++)
			updateGroupMonitor(g);
	} else {
		// update group monitor of a specific group

		// find index in group monitor arrays
		int monitorId = groupConfig[grpId].GroupMonitorId;

		// don't continue if no group monitor enabled for this group
		if (monitorId < 0)
			return;

		// find last update time for this group
		GroupMonitorCore* grpMonObj = groupMonCoreList[monitorId];
		int lastUpdate = grpMonObj->getLastUpdated();

		// don't continue if time interval is zero (nothing to update)
		if (getSimTime() - lastUpdate <=0)
			return;

		if (getSimTime() - lastUpdate > 1000)
			KERNEL_ERROR("updateGroupMonitor(grpId=%d) must be called at least once every second",grpId);

		if (simMode_ == GPU_MODE) {
			// copy the group status (neuromodulators) from the GPU to the CPU..
			copyGroupState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
		}

		// find the time interval in which to update group status
		// usually, we call updateGroupMonitor once every second, so the time interval is [0,1000)
		// however, updateGroupMonitor can be called at any time t \in [0,1000)... so we can have the cases
		// [0,t), [t,1000), and even [t1, t2)
		int numMsMin = lastUpdate%1000; // lower bound is given by last time we called update
		int numMsMax = getSimTimeMs(); // upper bound is given by current time
		if (numMsMax == 0)
			numMsMax = 1000; // special case: full second
		assert(numMsMin < numMsMax);

		// current time is last completed second in milliseconds (plus t to be added below)
		// special case is after each completed second where !getSimTimeMs(): here we look 1s back
		int currentTimeSec = getSimTimeSec();
		if (!getSimTimeMs())
			currentTimeSec--;

		// save current time as last update time
		grpMonObj->setLastUpdated(getSimTime());

		// prepare fast access
		FILE* grpFileId = groupMonCoreList[monitorId]->getGroupFileId();
		bool writeGroupToFile = grpFileId != NULL;
		bool writeGroupToArray = grpMonObj->isRecording();
		float data;

		// Read one peice of data at a time from the buffer and put the data to an appopriate monitor buffer. Later the user
		// may need need to dump these group status data to an output file
		for(int t = numMsMin; t < numMsMax; t++) {
			// fetch group status data, support dopamine concentration currently
			data = snnRuntimeData.grpDABuffer[grpId][t];

			// current time is last completed second plus whatever is leftover in t
			int time = currentTimeSec*1000 + t;

			if (writeGroupToFile) {
				// TODO: write to group status file
			}

			if (writeGroupToArray) {
				grpMonObj->pushData(time, data);
			}
		}

		if (grpFileId!=NULL) // flush group status file
			fflush(grpFileId);
	}
}

void SNN::updateSpikesFromGrp(int grpId) {
	assert(groupConfig[grpId].isSpikeGenerator==true);

	bool done;
	//static FILE* _fp = fopen("spikes.txt", "w");
	unsigned int currTime = simTime;

	int timeSlice = groupConfig[grpId].CurrTimeSlice;
	groupConfig[grpId].SliceUpdateTime  = simTime;

	// we dont generate any poisson spike if during the
	// current call we might exceed the maximum 32 bit integer value
	if (((uint64_t) currTime + timeSlice) >= MAX_SIMULATION_TIME)
		return;

	if (groupConfig[grpId].spikeGen) {
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

void SNN::updateSpikeGenerators() {
	for(int g=0; g<numGrp; g++) {
		if (groupConfig[g].isSpikeGenerator) {
			// This evaluation is done to check if its time to get new set of spikes..
			// check whether simTime has advance more than the current time slice, in which case we need to schedule
			// spikes for the next time slice
			// we always have to run this the first millisecond of a new runNetwork call; that is,
			// when simTime==simTimeRunStart
			if(((simTime-groupConfig[g].SliceUpdateTime) >= (unsigned) groupConfig[g].CurrTimeSlice || simTime == simTimeRunStart)) {
				updateSpikesFromGrp(g);
			}
		}
	}
}

void SNN::updateSpikeGeneratorsInit() {
	int cnt=0;
	for(int g=0; (g < numGrp); g++) {
		if (groupConfig[g].isSpikeGenerator) {
			// This is done only during initialization
			groupConfig[g].CurrTimeSlice = groupConfig[g].NewTimeSlice;

			// we only need NgenFunc for spike generator callbacks that need to transfer their spikes to the GPU
			if (groupConfig[g].spikeGen) {
				groupConfig[g].Noffset = NgenFunc;
				NgenFunc += groupConfig[g].SizeN;
			}
			//Note: updateSpikeFromGrp() will be called first time in updateSpikeGenerators()
			//updateSpikesFromGrp(g);
			cnt++;
			assert(cnt <= numSpikeGenGrps);
		}
	}

	// spikeGenBits can be set only once..
	assert(snnRuntimeData.spikeGenBits == NULL);

	if (NgenFunc) {
		snnRuntimeData.spikeGenBits = new uint32_t[NgenFunc/32+1];
		// increase the total memory size used by the routine...
		cpuSnnSz.addInfoSize += sizeof(snnRuntimeData.spikeGenBits[0])*(NgenFunc/32+1);
	}
}

//! update SNN::maxSpikesD1, SNN::maxSpikesD2 and allocate sapce for SNN::firingTableD1 and SNN::firingTableD2
/*!
 * \return maximum delay in groups
 */
int SNN::updateSpikeTables() {
	int curD = 0;
	int grpSrc;
	// find the maximum delay in the given network
	// and also the maximum delay for each group.
	//ConnectConfig* newInfo = connectBegin;
	//while(newInfo) {
	for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
		grpSrc = it->second.grpSrc;
		if (it->second.maxDelay > curD)
			curD = it->second.maxDelay;

		// check if the current connection's delay meaning grp1's delay
		// is greater than the MaxDelay for grp1. We find the maximum
		// delay for the grp1 by this scheme.
		if (it->second.maxDelay > groupConfig[grpSrc].MaxDelay)
		 	groupConfig[grpSrc].MaxDelay = it->second.maxDelay;

		//newInfo = newInfo->next;
	}

	for(int g = 0; g < numGrp; g++) {
		if (groupConfig[g].MaxDelay == 1)
			maxSpikesD1 += (groupConfig[g].SizeN * groupConfig[g].MaxFiringRate);
		else
			maxSpikesD2 += (groupConfig[g].SizeN * groupConfig[g].MaxFiringRate);
	}

	if ((maxSpikesD1 + maxSpikesD2) < (numNExcReg + numNInhReg + numNPois) * UNKNOWN_NEURON_MAX_FIRING_RATE) {
		KERNEL_ERROR("Insufficient amount of buffer allocated...");
		exitSimulation(1);
	}

	snnRuntimeData.firingTableD2 = new unsigned int[maxSpikesD2];
	snnRuntimeData.firingTableD1 = new unsigned int[maxSpikesD1];
	cpuSnnSz.spikingInfoSize += sizeof(int) * ((maxSpikesD2 + maxSpikesD1) + 2* (1000 + maxDelay_ + 1));

	return curD;
}

// updates simTime, returns true when new second started
bool SNN::updateTime() {
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
        KERNEL_WARN("Maximum Simulation Time Reached...Resetting simulation time");
	}

	return finishedOneSec;
}


void SNN::updateSpikeMonitor(int grpId) {
	// don't continue if no spike monitors in the network
	if (!numSpikeMonitor)
		return;

	if (grpId==ALL) {
		for (int g=0; g<numGrp; g++)
			updateSpikeMonitor(g);
	} else {
		// update spike monitor of a specific group

		// find index in spike monitor arrays
		int monitorId = groupConfig[grpId].SpikeMonitorId;

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
			KERNEL_ERROR("updateSpikeMonitor(grpId=%d) must be called at least once every second",grpId);

        // AER buffer max size warning here.
        // Because of C++ short-circuit evaluation, the last condition should not be evaluated
        // if the previous conditions are false.
        if (spkMonObj->getAccumTime() > LONG_SPIKE_MON_DURATION \
                && this->getGroupNumNeurons(grpId) > LARGE_SPIKE_MON_GRP_SIZE \
                && spkMonObj->isBufferBig()){
            // change this warning message to correct message
            KERNEL_WARN("updateSpikeMonitor(grpId=%d) is becoming very large. (>%lu MB)",grpId,(long int) MAX_SPIKE_MON_BUFFER_SIZE/1024 );// make this better
            KERNEL_WARN("Reduce the cumulative recording time (currently %lu minutes) or the group size (currently %d) to avoid this.",spkMonObj->getAccumTime()/(1000*60),this->getGroupNumNeurons(grpId));
       }
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
			unsigned int* fireTablePtr = (k==0)?snnRuntimeData.firingTableD2:snnRuntimeData.firingTableD1;
			for(int t=numMsMin; t<numMsMax; t++) {
				for(int i=timeTablePtr[t+maxDelay_]; i<timeTablePtr[t+maxDelay_+1];i++) {
					// retrieve the neuron id
					int nid   = fireTablePtr[i];
					if (simMode_ == GPU_MODE)
						nid = GET_FIRING_TABLE_NID(nid);
					assert(nid < numN);

					// make sure neuron belongs to currently relevant group
					int this_grpId = snnRuntimeData.grpIds[nid];
					if (this_grpId != grpId)
						continue;

					// adjust nid to be 0-indexed for each group
					// this way, if a group has 10 neurons, their IDs in the spike file and spike monitor will be
					// indexed from 0..9, no matter what their real nid is
					nid -= groupConfig[grpId].StartN;
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
