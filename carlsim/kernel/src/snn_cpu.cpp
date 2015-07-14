/* * Copyright (c) 2014 Regents of the University of California. All rights reserved.
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

#include <connection_monitor.h>
#include <connection_monitor_core.h>
#include <spike_monitor.h>
#include <spike_monitor_core.h>
#include <group_monitor.h>
#include <group_monitor_core.h>

// \FIXME what are the following for? why were they all the way at the bottom of this file?

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

#define SETPOST_INFO(name, nid, sid, val) name[cumulativePost[nid]+sid]=val;

#define SETPRE_INFO(name, nid, sid, val)  name[cumulativePre[nid]+sid]=val;



/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///


// TODO: consider moving unsafe computations out of constructor
SNN::SNN(const std::string& name, simMode_t simMode, loggerMode_t loggerMode, int ithGPU, int randSeed)
					: networkName_(name), simMode_(simMode), loggerMode_(loggerMode), ithGPU_(ithGPU),
					  randSeed_(SNN::setRandSeed(randSeed)) // all of these are const
{
	// move all unsafe operations out of constructor
	CpuSNNinit();
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

	grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));
	newInfo->grpSrc   		  = grpId1;
	newInfo->grpDest  		  = grpId2;
	newInfo->initWt	  		  = initWt;
	newInfo->maxWt	  		  = maxWt;
	newInfo->maxDelay 		  = maxDelay;
	newInfo->minDelay 		  = minDelay;
//		newInfo->radX             = (radX<0) ? MAX(szPre.x,szPost.x) : radX; // <0 means full connectivity, so the
//		newInfo->radY             = (radY<0) ? MAX(szPre.y,szPost.y) : radY; // effective group size is Grid3D.x. Grab
//		newInfo->radZ             = (radZ<0) ? MAX(szPre.z,szPost.z) : radZ; // the larger of pre / post to connect all
	newInfo->radX             = radX;
	newInfo->radY             = radY;
	newInfo->radZ             = radZ;
	newInfo->mulSynFast       = _mulSynFast;
	newInfo->mulSynSlow       = _mulSynSlow;
	newInfo->connProp         = connProp;
	newInfo->p                = prob;
	newInfo->type             = CONN_UNKNOWN;
	newInfo->numPostSynapses  = 1;
	newInfo->connectionMonitorId = -1;

	newInfo->next 				= connectBegin; //linked list of connection..
	connectBegin 				= newInfo;

	if ( _type.find("random") != std::string::npos) {
		newInfo->type 	= CONN_RANDOM;
		newInfo->numPostSynapses	= MIN(grp_Info[grpId2].SizeN,((int) (prob*grp_Info[grpId2].SizeN +6.5*sqrt(prob*(1-prob)*grp_Info[grpId2].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 6.5 stds.
		newInfo->numPreSynapses   = MIN(grp_Info[grpId1].SizeN,((int) (prob*grp_Info[grpId1].SizeN +6.5*sqrt(prob*(1-prob)*grp_Info[grpId1].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 6.5 stds.
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
	} else if ( _type.find("gaussian") != std::string::npos) {
		newInfo->type   = CONN_GAUSSIAN;
		// the following will soon go away, just assume the worst case for now
		newInfo->numPostSynapses	= grp_Info[grpId2].SizeN;
		newInfo->numPreSynapses   = grp_Info[grpId1].SizeN;
	} else {
		KERNEL_ERROR("Invalid connection type (should be 'random', 'full', 'one-to-one', 'full-no-direct', or 'gaussian')");
		exitSimulation(-1);
	}

	if (newInfo->numPostSynapses > MAX_nPostSynapses) {
		KERNEL_ERROR("ConnID %d exceeded the maximum number of output synapses (%d), has %d.",
			newInfo->connId,
			MAX_nPostSynapses, newInfo->numPostSynapses);
		assert(newInfo->numPostSynapses <= MAX_nPostSynapses);
	}

	if (newInfo->numPreSynapses > MAX_nPreSynapses) {
		KERNEL_ERROR("ConnID %d exceeded the maximum number of input synapses (%d), has %d.",
			newInfo->connId,
			MAX_nPreSynapses, newInfo->numPreSynapses);
		assert(newInfo->numPreSynapses <= MAX_nPreSynapses);
	}

	// update the pre and post size...
	// Subtlety: each group has numPost/PreSynapses from multiple connections.
	// The newInfo->numPost/PreSynapses are just for this specific connection.
	// We are adding the synapses counted in this specific connection to the totals for both groups.
	grp_Info[grpId1].numPostSynapses 	+= newInfo->numPostSynapses;
	grp_Info[grpId2].numPreSynapses 	+= newInfo->numPreSynapses;

	KERNEL_DEBUG("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d",
					grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,
					grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

	newInfo->connId	= numConnections++;
	assert(numConnections <= MAX_nConnections);	// make sure we don't overflow connId

	retId = newInfo->connId;

	KERNEL_DEBUG("CONNECT SETUP: connId=%d, mulFast=%f, mulSlow=%f",newInfo->connId,newInfo->mulSynFast,
						newInfo->mulSynSlow);
	assert(retId != -1);
	return retId;
}

// make custom connections from grpId1 to grpId2
short int SNN::connect(int grpId1, int grpId2, ConnectionGeneratorCore* conn, float _mulSynFast, float _mulSynSlow,
						bool synWtType, int maxM, int maxPreM) {
	int retId=-1;

	assert(grpId1 < numGrp);
	assert(grpId2 < numGrp);

	if (maxM == 0)
		maxM = grp_Info[grpId2].SizeN;

	if (maxPreM == 0)
		maxPreM = grp_Info[grpId1].SizeN;

	if (maxM > MAX_nPostSynapses) {
		KERNEL_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of output synapses (%d), "
							"has %d.", grp_Info2[grpId1].Name.c_str(),grpId1,grp_Info2[grpId2].Name.c_str(),
							grpId2,	MAX_nPostSynapses,maxM);
		assert(maxM <= MAX_nPostSynapses);
	}

	if (maxPreM > MAX_nPreSynapses) {
		KERNEL_ERROR("Connection from %s (%d) to %s (%d) exceeded the maximum number of input synapses (%d), "
							"has %d.\n", grp_Info2[grpId1].Name.c_str(), grpId1,grp_Info2[grpId2].Name.c_str(),
							grpId2, MAX_nPreSynapses,maxPreM);
		assert(maxPreM <= MAX_nPreSynapses);
	}

	grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));

	newInfo->grpSrc   = grpId1;
	newInfo->grpDest  = grpId2;
	newInfo->initWt	  = 1;
	newInfo->maxWt	  = 1;
	newInfo->maxDelay = MAX_SynapticDelay;
	newInfo->minDelay = 1;
	newInfo->mulSynFast = _mulSynFast;
	newInfo->mulSynSlow = _mulSynSlow;
	newInfo->connProp = SET_CONN_PRESENT(1) | SET_FIXED_PLASTIC(synWtType);
	newInfo->type	  = CONN_USER_DEFINED;
	newInfo->numPostSynapses	  	  = maxM;
	newInfo->numPreSynapses	  = maxPreM;
	newInfo->conn	= conn;
	newInfo->connectionMonitorId = -1;

	newInfo->next	= connectBegin;  // build a linked list
	connectBegin      = newInfo;

	// update the pre and post size...
	grp_Info[grpId1].numPostSynapses    += newInfo->numPostSynapses;
	grp_Info[grpId2].numPreSynapses += newInfo->numPreSynapses;

	KERNEL_DEBUG("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d",
					grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,
					grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

	newInfo->connId	= numConnections++;
	assert(numConnections <= MAX_nConnections);	// make sure we don't overflow connId

	retId = newInfo->connId;
	assert(retId != -1);
	return retId;
}


// create group of Izhikevich neurons
// use int for nNeur to avoid arithmetic underflow
int SNN::createGroup(const std::string& grpName, const Grid3D& grid, int neurType) {
	assert(grid.x*grid.y*grid.z>0);
	assert(neurType>=0);
	assert(numGrp < MAX_GRP_PER_SNN);

	if ( (!(neurType&TARGET_AMPA) && !(neurType&TARGET_NMDA) &&
		  !(neurType&TARGET_GABAa) && !(neurType&TARGET_GABAb)) || (neurType&POISSON_NEURON)) {
		KERNEL_ERROR("Invalid type using createGroup... Cannot create poisson generators here.");
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
	grp_Info[numGrp].WithESTDPtype      = UNKNOWN_STDP;
	grp_Info[numGrp].WithISTDPtype		= UNKNOWN_STDP;
	grp_Info[numGrp].WithHomeostasis	= false;

	if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb)) {
		grp_Info[numGrp].MaxFiringRate 	= INHIBITORY_NEURON_MAX_FIRING_RATE;
	} else {
		grp_Info[numGrp].MaxFiringRate 	= EXCITATORY_NEURON_MAX_FIRING_RATE;
	}

	grp_Info2[numGrp].Name  			= grpName;
	grp_Info[numGrp].isSpikeGenerator	= false;
	grp_Info[numGrp].MaxDelay			= 1;

	grp_Info2[numGrp].Izh_a 			= -1; // \FIXME ???

	// init homeostasis params even though not used
	grp_Info2[numGrp].baseFiring        = 10.0f;
	grp_Info2[numGrp].baseFiringSD      = 0.0f;

	grp_Info2[numGrp].Name              = grpName;
	finishedPoissonGroup				= true;

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
	assert(grid.x*grid.y*grid.z>0);
	assert(neurType>=0);
	grp_Info[numGrp].SizeN   		= grid.x * grid.y * grid.z; // number of neurons in the group
	grp_Info[numGrp].SizeX          = grid.x; // number of neurons in first dim of Grid3D
	grp_Info[numGrp].SizeY          = grid.y; // number of neurons in second dim of Grid3D
	grp_Info[numGrp].SizeZ          = grid.z; // number of neurons in third dim of Grid3D
	grp_Info[numGrp].Type    		= neurType | POISSON_NEURON;
	grp_Info[numGrp].WithSTP		= false;
	grp_Info[numGrp].WithSTDP		= false;
	grp_Info[numGrp].WithESTDPtype  = UNKNOWN_STDP;
	grp_Info[numGrp].WithISTDPtype	= UNKNOWN_STDP;
	grp_Info[numGrp].WithHomeostasis	= false;
	grp_Info[numGrp].isSpikeGenerator	= true;		// these belong to the spike generator class...
	grp_Info2[numGrp].Name    		= grpName;
	grp_Info[numGrp].MaxFiringRate 	= POISSON_MAX_FIRING_RATE;

	grp_Info2[numGrp].Name          = grpName;

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
		grp_Info[grpId].WithHomeostasis    = isSet;
		grp_Info[grpId].homeostasisScale   = homeoScale;
		grp_Info[grpId].avgTimeScale       = avgTimeScale;
		grp_Info[grpId].avgTimeScaleInv    = 1.0f/avgTimeScale;
		grp_Info[grpId].avgTimeScale_decay = (avgTimeScale*1000.0f-1.0f)/(avgTimeScale*1000.0f);
		grp_Info[grpId].newUpdates 		= true; // \FIXME: what's this?

		KERNEL_INFO("Homeostasis parameters %s for %d (%s):\thomeoScale: %f, avgTimeScale: %f",
					isSet?"enabled":"disabled",grpId,grp_Info2[grpId].Name.c_str(),homeoScale,avgTimeScale);
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
		assert(grp_Info[grpId].WithHomeostasis);

		grp_Info2[grpId].baseFiring 	= baseFiring;
		grp_Info2[grpId].baseFiringSD 	= baseFiringSD;
		grp_Info[grpId].newUpdates 	= true; //TODO: I have to see how this is handled.  -- KDC

		KERNEL_INFO("Homeostatic base firing rate set for %d (%s):\tbaseFiring: %3.3f, baseFiringStd: %3.3f",
							grpId,grp_Info2[grpId].Name.c_str(),baseFiring,baseFiringSD);
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
		grp_Info2[grpId].Izh_a	  	=   izh_a;
		grp_Info2[grpId].Izh_a_sd  =   izh_a_sd;
		grp_Info2[grpId].Izh_b	  	=   izh_b;
		grp_Info2[grpId].Izh_b_sd  =   izh_b_sd;
		grp_Info2[grpId].Izh_c		=   izh_c;
		grp_Info2[grpId].Izh_c_sd	=   izh_c_sd;
		grp_Info2[grpId].Izh_d		=   izh_d;
		grp_Info2[grpId].Izh_d_sd	=   izh_d_sd;
	}
}

void SNN::setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh,
	float tauACh, float baseNE, float tauNE) {

	grp_Info[grpId].baseDP	= baseDP;
	grp_Info[grpId].decayDP = 1.0 - (1.0 / tauDP);
	grp_Info[grpId].base5HT = base5HT;
	grp_Info[grpId].decay5HT = 1.0 - (1.0 / tau5HT);
	grp_Info[grpId].baseACh = baseACh;
	grp_Info[grpId].decayACh = 1.0 - (1.0 / tauACh);
	grp_Info[grpId].baseNE	= baseNE;
	grp_Info[grpId].decayNE = 1.0 - (1.0 / tauNE);
}

// set ESTDP params
void SNN::setESTDP(int grpId, bool isSet, stdpType_t type, stdpCurve_t curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma) {
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
		grp_Info[grpId].ALPHA_PLUS_EXC 		= alphaPlus;
		grp_Info[grpId].ALPHA_MINUS_EXC 	= alphaMinus;
		grp_Info[grpId].TAU_PLUS_INV_EXC 	= 1.0f/tauPlus;
		grp_Info[grpId].TAU_MINUS_INV_EXC	= 1.0f/tauMinus;
		grp_Info[grpId].GAMMA				= gamma;
		grp_Info[grpId].KAPPA				= (1 + exp(-gamma/tauPlus))/(1 - exp(-gamma/tauPlus));
		grp_Info[grpId].OMEGA				= alphaPlus * (1 - grp_Info[grpId].KAPPA);
		// set flags for STDP function
		grp_Info[grpId].WithESTDPtype	= type;
		grp_Info[grpId].WithESTDPcurve = curve;
		grp_Info[grpId].WithESTDP		= isSet;
		grp_Info[grpId].WithSTDP		|= grp_Info[grpId].WithESTDP;
		sim_with_stdp					|= grp_Info[grpId].WithSTDP;

		KERNEL_INFO("E-STDP %s for %s(%d)", isSet?"enabled":"disabled", grp_Info2[grpId].Name.c_str(), grpId);
	}
}

// set ISTDP params
void SNN::setISTDP(int grpId, bool isSet, stdpType_t type, stdpCurve_t curve, float ab1, float ab2, float tau1, float tau2) {
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
			grp_Info[grpId].ALPHA_PLUS_INB = ab1;
			grp_Info[grpId].ALPHA_MINUS_INB = ab2;
			grp_Info[grpId].TAU_PLUS_INV_INB = 1.0f / tau1;
			grp_Info[grpId].TAU_MINUS_INV_INB = 1.0f / tau2;
			grp_Info[grpId].BETA_LTP 		= 0.0f;
			grp_Info[grpId].BETA_LTD 		= 0.0f;
			grp_Info[grpId].LAMBDA			= 1.0f;
			grp_Info[grpId].DELTA			= 1.0f;
		} else {
			grp_Info[grpId].ALPHA_PLUS_INB = 0.0f;
			grp_Info[grpId].ALPHA_MINUS_INB = 0.0f;
			grp_Info[grpId].TAU_PLUS_INV_INB = 1.0f;
			grp_Info[grpId].TAU_MINUS_INV_INB = 1.0f;
			grp_Info[grpId].BETA_LTP 		= ab1;
			grp_Info[grpId].BETA_LTD 		= ab2;
			grp_Info[grpId].LAMBDA			= tau1;
			grp_Info[grpId].DELTA			= tau2;
		}
		// set flags for STDP function
		//FIXME: separate STDPType to ESTDPType and ISTDPType
		grp_Info[grpId].WithISTDPtype	= type;
		grp_Info[grpId].WithISTDPcurve = curve;
		grp_Info[grpId].WithISTDP		= isSet;
		grp_Info[grpId].WithSTDP		|= grp_Info[grpId].WithISTDP;
		sim_with_stdp					|= grp_Info[grpId].WithSTDP;

		KERNEL_INFO("I-STDP %s for %s(%d)", isSet?"enabled":"disabled", grp_Info2[grpId].Name.c_str(), grpId);
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
		grp_Info[grpId].WithSTP 		= isSet;
		grp_Info[grpId].STP_A 			= (STP_U>0.0f) ? 1.0/STP_U : 1.0f; // scaling factor
		grp_Info[grpId].STP_U 			= STP_U;
		grp_Info[grpId].STP_tau_u_inv	= 1.0f/STP_tau_u; // facilitatory
		grp_Info[grpId].STP_tau_x_inv	= 1.0f/STP_tau_x; // depressive
		grp_Info[grpId].newUpdates = true;

		KERNEL_INFO("STP %s for %d (%s):\tA: %1.4f, U: %1.4f, tau_u: %4.0f, tau_x: %4.0f", isSet?"enabled":"disabled",
					grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].STP_A, STP_U, STP_tau_u, STP_tau_x);
	}
}

void SNN::setWeightAndWeightChangeUpdate(updateInterval_t wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay) {
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
/// PUBLIC METHODS: RUNNING A SIMULATION
/// ************************************************************************************************************ ///

int SNN::runNetwork(int _nsec, int _nmsec, bool printRunSummary, bool copyState) {
	assert(_nmsec >= 0 && _nmsec < 1000);
	assert(_nsec  >= 0);
	int runDurationMs = _nsec*1000 + _nmsec;
	KERNEL_DEBUG("runNetwork: runDur=%dms, printRunSummary=%s, copyState=%s", runDurationMs, printRunSummary?"y":"n",
		copyState?"y":"n");

	// setupNetwork() must have already been called
	assert(doneReorganization);

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
			copyNeuronState(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, ALL);

			if (sim_with_stp) {
				copySTPState(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
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

	grpConnectInfo_t* connInfo = getConnectInfo(connId);

	// iterate over all postsynaptic neurons
	for (int i=grp_Info[connInfo->grpDest].StartN; i<=grp_Info[connInfo->grpDest].EndN; i++) {
		unsigned int cumIdx = cpuRuntimeData.cumulativePre[i];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j=0; j<cpuRuntimeData.Npre[i]; pos_ij++, j++) {
			if (cpuRuntimeData.cumConnIdPre[pos_ij]==connId) {
				// apply bias to weight
				float weight = cpuRuntimeData.wt[pos_ij] + bias;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight+bias>connInfo->maxWt || weight+bias<connInfo->minWt);
				bool needToPrintDebug = (weight>connInfo->maxWt || weight<0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connInfo->maxWt = fmax(connInfo->maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), 0.0f, connInfo->maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = fmin(weight, connInfo->maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = fmax(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), weight, 0.0f, connInfo->maxWt);
					}
				}

				// update datastructures
				cpuRuntimeData.wt[pos_ij] = weight;
				cpuRuntimeData.maxSynWt[pos_ij] = connInfo->maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (simMode_==GPU_MODE) {
			CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[cumIdx]), &(cpuRuntimeData.wt[cumIdx]), sizeof(float)*cpuRuntimeData.Npre[i],
				cudaMemcpyHostToDevice) );

			if (gpuRuntimeData.maxSynWt!=NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[cumIdx]), &(cpuRuntimeData.maxSynWt[cumIdx]),
					sizeof(float)*cpuRuntimeData.Npre[i], cudaMemcpyHostToDevice) );
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
		if (!grp_Info[grpId].withSpikeCounter)
			return;

		grp_Info[grpId].spkCntRecordDurHelper = 0;

		if (simMode_==GPU_MODE) {
			resetSpikeCounter_GPU(grpId);
		}
		else {
			int bufPos = grp_Info[grpId].spkCntBufPos; // retrieve buf pos
			memset(spkCntBuf[bufPos],0,grp_Info[grpId].SizeN*sizeof(int)); // set all to 0
		}
	}
}

// multiplies every weight with a scaling factor
void SNN::scaleWeights(short int connId, float scale, bool updateWeightRange) {
	assert(connId>=0 && connId<numConnections);
	assert(scale>=0.0f);

	grpConnectInfo_t* connInfo = getConnectInfo(connId);

	// iterate over all postsynaptic neurons
	for (int i=grp_Info[connInfo->grpDest].StartN; i<=grp_Info[connInfo->grpDest].EndN; i++) {
		unsigned int cumIdx = cpuRuntimeData.cumulativePre[i];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j=0; j<cpuRuntimeData.Npre[i]; pos_ij++, j++) {
			if (cpuRuntimeData.cumConnIdPre[pos_ij]==connId) {
				// apply bias to weight
				float weight = cpuRuntimeData.wt[pos_ij]*scale;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight>connInfo->maxWt || weight<connInfo->minWt);
				bool needToPrintDebug = (weight>connInfo->maxWt || weight<0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connInfo->maxWt = fmax(connInfo->maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), 0.0f, connInfo->maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = fmin(weight, connInfo->maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = fmax(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), weight, 0.0f, connInfo->maxWt);
					}
				}

				// update datastructures
				cpuRuntimeData.wt[pos_ij] = weight;
				cpuRuntimeData.maxSynWt[pos_ij] = connInfo->maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (simMode_==GPU_MODE) {
			CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[cumIdx]), &(cpuRuntimeData.wt[cumIdx]), sizeof(float)*cpuRuntimeData.Npre[i],
				cudaMemcpyHostToDevice) );

			if (gpuRuntimeData.maxSynWt!=NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[cumIdx]), &(cpuRuntimeData.maxSynWt[cumIdx]),
					sizeof(float)*cpuRuntimeData.Npre[i], cudaMemcpyHostToDevice));
			}
		}
	}
}

GroupMonitor* SNN::setGroupMonitor(int grpId, FILE* fid) {
	// check whether group already has a GroupMonitor
	if (grp_Info[grpId].GroupMonitorId >= 0) {
		KERNEL_ERROR("setGroupMonitor has already been called on Group %d (%s).",
			grpId, grp_Info2[grpId].Name.c_str());
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
	grp_Info[grpId].GroupMonitorId = numGroupMonitor;

    // not eating much memory anymore, got rid of all buffers
	cpuSnnSz.monitorInfoSize += sizeof(GroupMonitor*);
	cpuSnnSz.monitorInfoSize += sizeof(GroupMonitorCore*);

	numGroupMonitor++;
	KERNEL_INFO("GroupMonitor set for group %d (%s)",grpId,grp_Info2[grpId].Name.c_str());

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
	grpConnectInfo_t* connInfo = getConnectInfo(connId);
	if (connInfo->connectionMonitorId >= 0) {
		KERNEL_ERROR("setConnectionMonitor has already been called on Connection %d (MonitorId=%d)", connId, connInfo->connectionMonitorId);
		exitSimulation(1);
	}

	// inform the connection that it is being monitored...
	// this needs to be called before new ConnectionMonitorCore
	connInfo->connectionMonitorId = numConnectionMonitor;

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
	KERNEL_INFO("ConnectionMonitor %d set for Connection %d: %d(%s) => %d(%s)", connInfo->connectionMonitorId, connId, grpIdPre, getGroupName(grpIdPre).c_str(),
		grpIdPost, getGroupName(grpIdPost).c_str());

	return connMonObj;
}

void SNN::setExternalCurrent(int grpId, const std::vector<float>& current) {
	assert(grpId>=0); assert(grpId<numGrp);
	assert(!isPoissonGroup(grpId));
	assert(current.size() == getGroupNumNeurons(grpId));

	// // update flag for faster handling at run-time
	// if (count_if(current.begin(), current.end(), isGreaterThanZero)) {
	// 	grp_Info[grpId].WithCurrentInjection = true;
	// } else {
	// 	grp_Info[grpId].WithCurrentInjection = false;
	// }

	// store external current in array
	for (int i=grp_Info[grpId].StartN, j=0; i<=grp_Info[grpId].EndN; i++, j++) {
		cpuRuntimeData.extCurrent[i] = current[j];
	}

	// copy to GPU if necessary
	// don't allocate; allocation done in buildNetwork
	if (simMode_==GPU_MODE) {
		copyExternalCurrent(&gpuRuntimeData, &cpuRuntimeData, false, grpId);
	}
}

// sets up a spike generator
void SNN::setSpikeGenerator(int grpId, SpikeGeneratorCore* spikeGen) {
	assert(!doneReorganization); // must be called before setupNetwork to work on GPU
	assert(spikeGen);
	assert (grp_Info[grpId].isSpikeGenerator);
	grp_Info[grpId].spikeGen = spikeGen;
}

// A Spike Counter keeps track of the number of spikes per neuron in a group.
void SNN::setSpikeCounter(int grpId, int recordDur) {
	assert(grpId>=0); assert(grpId<numGrp);

	sim_with_spikecounters = true; // inform simulation
	grp_Info[grpId].withSpikeCounter = true; // inform the group
	grp_Info[grpId].spkCntRecordDur = (recordDur>0)?recordDur:-1; // set record duration, after which spike buf will be reset
	grp_Info[grpId].spkCntRecordDurHelper = 0; // counter to help make fast modulo
	grp_Info[grpId].spkCntBufPos = numSpkCnt; // inform group which pos it has in spike buf
	spkCntBuf[numSpkCnt] = new int[grp_Info[grpId].SizeN]; // create spike buf
	memset(spkCntBuf[numSpkCnt],0,(grp_Info[grpId].SizeN)*sizeof(int)); // set all to 0

	numSpkCnt++;

	KERNEL_INFO("SpikeCounter set for Group %d (%s): %d ms recording window", grpId, grp_Info2[grpId].Name.c_str(),
		recordDur);
}

// record spike information, return a SpikeInfo object
SpikeMonitor* SNN::setSpikeMonitor(int grpId, FILE* fid) {
	// check whether group already has a SpikeMonitor
	if (grp_Info[grpId].SpikeMonitorId >= 0) {
		// in this case, return the current object and update fid
		SpikeMonitor* spkMonObj = getSpikeMonitor(grpId);

		// update spike file ID
		SpikeMonitorCore* spkMonCoreObj = getSpikeMonitorCore(grpId);
		spkMonCoreObj->setSpikeFileId(fid);

		KERNEL_INFO("SpikeMonitor updated for group %d (%s)",grpId,grp_Info2[grpId].Name.c_str());
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
		grp_Info[grpId].SpikeMonitorId	= numSpikeMonitor;

    	// not eating much memory anymore, got rid of all buffers
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitor*);
		cpuSnnSz.monitorInfoSize += sizeof(SpikeMonitorCore*);

		numSpikeMonitor++;
		KERNEL_INFO("SpikeMonitor set for group %d (%s)",grpId,grp_Info2[grpId].Name.c_str());

		return spkMonObj;
	}
}

// assigns spike rate to group
void SNN::setSpikeRate(int grpId, PoissonRate* ratePtr, int refPeriod) {
	assert(grpId>=0 && grpId<numGrp);
	assert(ratePtr);
	assert(grp_Info[grpId].isSpikeGenerator);
	assert(ratePtr->getNumNeurons()==grp_Info[grpId].SizeN);
	assert(refPeriod>=1);

	grp_Info[grpId].RatePtr = ratePtr;
	grp_Info[grpId].RefractPeriod   = refPeriod;
	spikeRateUpdated = true;
}

// sets the weight value of a specific synapse
void SNN::setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange) {
	assert(connId>=0 && connId<getNumConnections());
	assert(weight>=0.0f);

	grpConnectInfo_t* connInfo = getConnectInfo(connId);
	assert(neurIdPre>=0  && neurIdPre<getGroupNumNeurons(connInfo->grpSrc));
	assert(neurIdPost>=0 && neurIdPost<getGroupNumNeurons(connInfo->grpDest));

	float maxWt = fabs(connInfo->maxWt);
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
	int neurIdPreReal = grp_Info[connInfo->grpSrc].StartN+neurIdPre;
	int neurIdPostReal = grp_Info[connInfo->grpDest].StartN+neurIdPost;

	// iterate over all presynaptic synapses until right one is found
	bool synapseFound = false;
	int pos_ij = cpuRuntimeData.cumulativePre[neurIdPostReal];
	for (int j=0; j<cpuRuntimeData.Npre[neurIdPostReal]; pos_ij++, j++) {
		post_info_t* preId = &(cpuRuntimeData.preSynapticIds[pos_ij]);
		int pre_nid = GET_CONN_NEURON_ID((*preId));
		if (GET_CONN_NEURON_ID((*preId))==neurIdPreReal) {
			assert(cpuRuntimeData.cumConnIdPre[pos_ij]==connId); // make sure we've got the right connection ID

			cpuRuntimeData.wt[pos_ij] = isExcitatoryGroup(connInfo->grpSrc) ? weight : -1.0*weight;
			cpuRuntimeData.maxSynWt[pos_ij] = isExcitatoryGroup(connInfo->grpSrc) ? maxWt : -1.0*maxWt;

			if (simMode_==GPU_MODE) {
				// need to update datastructures on GPU
				CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.wt[pos_ij]), &(cpuRuntimeData.wt[pos_ij]), sizeof(float), cudaMemcpyHostToDevice));
				if (gpuRuntimeData.maxSynWt!=NULL) {
					// only copy maxSynWt if datastructure actually exists on the GPU
					// (that logic should be done elsewhere though)
					CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.maxSynWt[pos_ij]), &(cpuRuntimeData.maxSynWt[pos_ij]), sizeof(float), cudaMemcpyHostToDevice));
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
		if (!fwrite(&grp_Info[g].StartN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&grp_Info[g].EndN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		if (!fwrite(&grp_Info[g].SizeX,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&grp_Info[g].SizeY,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&grp_Info[g].SizeZ,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		strncpy(name,grp_Info2[g].Name.c_str(),100);
		if (!fwrite(name,1,100,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	}

	// +++++ Fetch WEIGHT DATA (GPU Mode only) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	if (simMode_ == GPU_MODE)
		copyWeightState(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
	// +++++ WRITE SYNAPSE INFO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// \FIXME: replace with faster version
	if (saveSynapseInfo) {
		for (unsigned int i=0;i<numN;i++) {
			unsigned int offset = cpuRuntimeData.cumulativePost[i];

			unsigned int count = 0;
			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = cpuRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++)
					count++;
			}

			if (!fwrite(&count,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

			for (int t=0;t<maxDelay_;t++) {
				delay_info_t dPar = cpuRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

				for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
					// get synaptic info...
					post_info_t post_info = cpuRuntimeData.postSynapticIds[offset + idx_d];

					// get neuron id
					//int p_i = (post_info&POST_SYN_NEURON_MASK);
					unsigned int p_i = GET_CONN_NEURON_ID(post_info);
					assert(p_i<numN);

					// get syn id
					unsigned int s_i = GET_CONN_SYN_ID(post_info);
					//>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;
					assert(s_i<(cpuRuntimeData.Npre[p_i]));

					// get the cumulative position for quick access...
					unsigned int pos_i = cpuRuntimeData.cumulativePre[p_i] + s_i;

					uint8_t delay = t+1;
					uint8_t plastic = s_i < cpuRuntimeData.Npre_plastic[p_i]; // plastic or fixed.

					if (!fwrite(&i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&p_i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(cpuRuntimeData.wt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(cpuRuntimeData.maxSynWt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&delay,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&plastic,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
					if (!fwrite(&(cpuRuntimeData.cumConnIdPre[pos_i]),sizeof(short int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
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

	if(!doneReorganization){
		KERNEL_ERROR("Simulation has not been run yet, cannot output weights.");
		exitSimulation(1);
	}

	post_info_t* preId;
	int pre_nid, pos_ij;

	//population sizes
	numPre = grp_Info[grpIdPre].SizeN;
	numPost = grp_Info[grpIdPost].SizeN;

	//first iteration gets the number of synaptic weights to place in our
	//weight matrix.
	matrixSize=0;
	//iterate over all neurons in the post group
	for (int i=grp_Info[grpIdPost].StartN; i<=grp_Info[grpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cpuRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
		//iterate over all presynaptic synapses
		for(int j=0; j<cpuRuntimeData.Npre[i]; pos_ij++,j++) {
			preId = &cpuRuntimeData.preSynapticIds[pos_ij];
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[grpIdPre].StartN || pre_nid>grp_Info[grpIdPre].EndN)
				continue; // connection does not belong to group grpIdPre
			matrixSize++;
		}
	}

	//now we have the correct size
	weights = new float[matrixSize];
	//second iteration assigns the weights
	int curr = 0; // iterator for return array
	//iterate over all neurons in the post group
	for (int i=grp_Info[grpIdPost].StartN; i<=grp_Info[grpIdPost].EndN; i++) {
		// for every post-neuron, find all pre
		pos_ij = cpuRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
		//do the GPU copy here.  Copy the current weights from GPU to CPU.
		if(simMode_==GPU_MODE){
			copyWeightsGPU(i,grpIdPre);
		}
		//iterate over all presynaptic synapses
		for(int j=0; j<cpuRuntimeData.Npre[i]; pos_ij++,j++) {
			preId = &(cpuRuntimeData.preSynapticIds[pos_ij]);
			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
			if (pre_nid<grp_Info[grpIdPre].StartN || pre_nid>grp_Info[grpIdPre].EndN)
				continue; // connection does not belong to group grpIdPre
			weights[curr] = cpuRuntimeData.wt[pos_ij];
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
	grpConnectInfo_t* connInfo = connectBegin;

	short int connId = -1;
	while (connInfo) {
		// check whether pre and post match
		if (connInfo->grpSrc == grpIdPre && connInfo->grpDest == grpIdPost) {
			connId = connInfo->connId;
			break;
		}

		// otherwise, keep looking
		connInfo = connInfo->next;
	}

	return connId;
}

//! used for parameter tuning functionality
grpConnectInfo_t* SNN::getConnectInfo(short int connectId) {
	grpConnectInfo_t* nextConn = connectBegin;
	CHECK_CONNECTION_ID(connectId, numConnections);

	// clear all existing connection info...
	while (nextConn) {
		if (nextConn->connId == connectId) {
			nextConn->newUpdates = true;		// \FIXME: this is a Jay hack
			return nextConn;
		}
		nextConn = nextConn->next;
	}

	KERNEL_DEBUG("Total Connections = %d", numConnections);
	KERNEL_DEBUG("ConnectId (%d) cannot be recognized", connectId);
	return NULL;
}

std::vector<float> SNN::getConductanceAMPA(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE) {
		copyConductanceAMPA(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);
	}

	std::vector<float> gAMPAvec;
	for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
		gAMPAvec.push_back(cpuRuntimeData.gAMPA[i]);
	}
	return gAMPAvec;
}

std::vector<float> SNN::getConductanceNMDA(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceNMDA(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);

	std::vector<float> gNMDAvec;
	if (isSimulationWithNMDARise()) {
		// need to construct conductance from rise and decay parts
		for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
			gNMDAvec.push_back(cpuRuntimeData.gNMDA_d[i]-cpuRuntimeData.gNMDA_r[i]);
		}
	} else {
		for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
			gNMDAvec.push_back(cpuRuntimeData.gNMDA[i]);
		}
	}
	return gNMDAvec;
}

std::vector<float> SNN::getConductanceGABAa(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE) {
		copyConductanceGABAa(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);
	}

	std::vector<float> gGABAaVec;
	for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
		gGABAaVec.push_back(cpuRuntimeData.gGABAa[i]);
	}
	return gGABAaVec;
}

std::vector<float> SNN::getConductanceGABAb(int grpId) {
	assert(isSimulationWithCOBA());

	// need to copy data from GPU first
	if (getSimMode()==GPU_MODE)
		copyConductanceGABAb(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);

	std::vector<float> gGABAbVec;
	if (isSimulationWithGABAbRise()) {
		// need to construct conductance from rise and decay parts
		for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
			gGABAbVec.push_back(cpuRuntimeData.gGABAb_d[i]-cpuRuntimeData.gGABAb_r[i]);
		}
	} else {
		for (int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
			gGABAbVec.push_back(cpuRuntimeData.gGABAb[i]);
		}
	}
	return gGABAbVec;
}

// returns RangeDelay struct of a connection
RangeDelay SNN::getDelayRange(short int connId) {
	assert(connId>=0 && connId<numConnections);
	grpConnectInfo_t* connInfo = getConnectInfo(connId);
	return RangeDelay(connInfo->minDelay, connInfo->maxDelay);
}


// this is a user function
// \FIXME: fix this
uint8_t* SNN::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	Npre = grp_Info[gIDpre].SizeN;
	Npost = grp_Info[gIDpost].SizeN;

	if (delays == NULL) delays = new uint8_t[Npre*Npost];
	memset(delays,0,Npre*Npost);

	for (int i=grp_Info[gIDpre].StartN;i<grp_Info[gIDpre].EndN;i++) {
		unsigned int offset = cpuRuntimeData.cumulativePost[i];

		for (int t=0;t<maxDelay_;t++) {
			delay_info_t dPar = cpuRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
				// get synaptic info...
				post_info_t post_info = cpuRuntimeData.postSynapticIds[offset + idx_d];

				// get neuron id
				//int p_i = (post_info&POST_SYN_NEURON_MASK);
				int p_i = GET_CONN_NEURON_ID(post_info);
				assert(p_i<numN);

				if (p_i >= grp_Info[gIDpost].StartN && p_i <= grp_Info[gIDpost].EndN) {
					// get syn id
					int s_i = GET_CONN_SYN_ID(post_info);

					// get the cumulative position for quick access...
					unsigned int pos_i = cpuRuntimeData.cumulativePre[p_i] + s_i;

					delays[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = t+1;
				}
			}
		}
	}
	return delays;
}

Grid3D SNN::getGroupGrid3D(int grpId) {
	assert(grpId>=0 && grpId<numGrp);
	return Grid3D(grp_Info[grpId].SizeX, grp_Info[grpId].SizeY, grp_Info[grpId].SizeZ);
}

// find ID of group with name grpName
int SNN::getGroupId(std::string grpName) {
	for (int grpId=0; grpId<numGrp; grpId++) {
		if (grp_Info2[grpId].Name.compare(grpName)==0)
			return grpId;
	}

	// group not found
	return -1;
}

GroupConfig SNN::getGroupInfo(int grpId) {
	assert(grpId>=-1 && grpId<numGrp);
	return grp_Info[grpId];
}

std::string SNN::getGroupName(int grpId) {
	assert(grpId>=-1 && grpId<numGrp);

	if (grpId==ALL)
		return "ALL";

	return grp_Info2[grpId].Name;
}

GroupSTDPInfo_t SNN::getGroupSTDPInfo(int grpId) {
	GroupSTDPInfo_t gInfo;

	gInfo.WithSTDP = grp_Info[grpId].WithSTDP;
	gInfo.WithESTDP = grp_Info[grpId].WithESTDP;
	gInfo.WithISTDP = grp_Info[grpId].WithISTDP;
	gInfo.WithESTDPtype = grp_Info[grpId].WithESTDPtype;
	gInfo.WithISTDPtype = grp_Info[grpId].WithISTDPtype;
	gInfo.WithESTDPcurve = grp_Info[grpId].WithESTDPcurve;
	gInfo.WithISTDPcurve = grp_Info[grpId].WithISTDPcurve;
	gInfo.ALPHA_MINUS_EXC = grp_Info[grpId].ALPHA_MINUS_EXC;
	gInfo.ALPHA_PLUS_EXC = grp_Info[grpId].ALPHA_PLUS_EXC;
	gInfo.TAU_MINUS_INV_EXC = grp_Info[grpId].TAU_MINUS_INV_EXC;
	gInfo.TAU_PLUS_INV_EXC = grp_Info[grpId].TAU_PLUS_INV_EXC;
	gInfo.ALPHA_MINUS_INB = grp_Info[grpId].ALPHA_MINUS_INB;
	gInfo.ALPHA_PLUS_INB = grp_Info[grpId].ALPHA_PLUS_INB;
	gInfo.TAU_MINUS_INV_INB = grp_Info[grpId].TAU_MINUS_INV_INB;
	gInfo.TAU_PLUS_INV_INB = grp_Info[grpId].TAU_PLUS_INV_INB;
	gInfo.GAMMA = grp_Info[grpId].GAMMA;
	gInfo.BETA_LTP = grp_Info[grpId].BETA_LTP;
	gInfo.BETA_LTD = grp_Info[grpId].BETA_LTD;
	gInfo.LAMBDA = grp_Info[grpId].LAMBDA;
	gInfo.DELTA = grp_Info[grpId].DELTA;

	return gInfo;
}

GroupNeuromodulatorInfo_t SNN::getGroupNeuromodulatorInfo(int grpId) {
	GroupNeuromodulatorInfo_t gInfo;

	gInfo.baseDP = grp_Info[grpId].baseDP;
	gInfo.base5HT = grp_Info[grpId].base5HT;
	gInfo.baseACh = grp_Info[grpId].baseACh;
	gInfo.baseNE = grp_Info[grpId].baseNE;
	gInfo.decayDP = grp_Info[grpId].decayDP;
	gInfo.decay5HT = grp_Info[grpId].decay5HT;
	gInfo.decayACh = grp_Info[grpId].decayACh;
	gInfo.decayNE = grp_Info[grpId].decayNE;

	return gInfo;
}

Point3D SNN::getNeuronLocation3D(int neurId) {
	assert(neurId>=0 && neurId<numN);
	int grpId = cpuRuntimeData.grpIds[neurId];
	assert(neurId>=grp_Info[grpId].StartN && neurId<=grp_Info[grpId].EndN);

	// adjust neurId for neuron ID of first neuron in the group
	neurId -= grp_Info[grpId].StartN;

	return getNeuronLocation3D(grpId, neurId);
}

Point3D SNN::getNeuronLocation3D(int grpId, int relNeurId) {
	assert(grpId>=0 && grpId<numGrp);
	assert(relNeurId>=0 && relNeurId<getGroupNumNeurons(grpId));

	// coordinates are in x e[-SizeX/2,SizeX/2], y e[-SizeY/2,SizeY/2], z e[-SizeZ/2,SizeZ/2]
	// instead of x e[0,SizeX], etc.
	int intX = relNeurId % grp_Info[grpId].SizeX;
	int intY = (relNeurId/grp_Info[grpId].SizeX)%grp_Info[grpId].SizeY;
	int intZ = relNeurId/(grp_Info[grpId].SizeX*grp_Info[grpId].SizeY);

	// so subtract SizeX/2, etc. to get coordinates center around origin
	double coordX = 1.0*intX - (grp_Info[grpId].SizeX-1)/2.0;
	double coordY = 1.0*intY - (grp_Info[grpId].SizeY-1)/2.0;
	double coordZ = 1.0*intZ - (grp_Info[grpId].SizeZ-1)/2.0;
	return Point3D(coordX, coordY, coordZ);
}

// returns the number of synaptic connections associated with this connection.
int SNN::getNumSynapticConnections(short int connectionId) {
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
  KERNEL_ERROR("Connection ID was not found.  Quitting.");
  exitSimulation(1);
}

// return spike buffer, which contains #spikes per neuron in the group
int* SNN::getSpikeCounter(int grpId) {
	assert(grpId>=0); assert(grpId<numGrp);

	if (!grp_Info[grpId].withSpikeCounter)
		return NULL;

	// determine whether spike counts are currently stored on CPU or GPU side
	bool retrieveSpikesFromGPU = simMode_==GPU_MODE;
	if (grp_Info[grpId].isSpikeGenerator) {
		// this flag should be set if group was created via CARLsim::createSpikeGeneratorGroup
		// could be SpikeGen callback or PoissonRate
		if (grp_Info[grpId].RatePtr != NULL) {
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
		int bufPos = grp_Info[grpId].spkCntBufPos; // retrieve buf pos
		return spkCntBuf[bufPos]; // return pointer to buffer
	}
}

// returns pointer to existing SpikeMonitor object, NULL else
SpikeMonitor* SNN::getSpikeMonitor(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	if (grp_Info[grpId].SpikeMonitorId>=0) {
		return spikeMonList[(grp_Info[grpId].SpikeMonitorId)];
	} else {
		return NULL;
	}
}

SpikeMonitorCore* SNN::getSpikeMonitorCore(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	if (grp_Info[grpId].SpikeMonitorId>=0) {
		return spikeMonCoreList[(grp_Info[grpId].SpikeMonitorId)];
	} else {
		return NULL;
	}
}

// returns RangeWeight struct of a connection
RangeWeight SNN::getWeightRange(short int connId) {
	assert(connId>=0 && connId<numConnections);
	grpConnectInfo_t* connInfo = getConnectInfo(connId);
	return RangeWeight(0.0f, connInfo->initWt, connInfo->maxWt);
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// all unsafe operations of SNN constructor
void SNN::CpuSNNinit() {
	assert(ithGPU_>=0);

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

	finishedPoissonGroup  = false;
	connectBegin = NULL;

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
	sim_in_testing = false;

	maxSpikesD2 = maxSpikesD1 = 0;
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

	// reset all pointers, don't deallocate (false)
	resetPointers(false);

	memset(&cpuSnnSz, 0, sizeof(cpuSnnSz));

	showGrpFiringInfo = true;

	// initialize propogated spike buffers.....
	pbuf = new PropagatedSpikeBuffer(0, PROPAGATED_BUFFER_SIZE);

	memset(&gpuRuntimeData, 0, sizeof(RuntimeData));
	memset(&net_Info, 0, sizeof(network_info_t));
	gpuRuntimeData.allocated = false;

	memset(&cpuRuntimeData, 0, sizeof(RuntimeData));
	cpuRuntimeData.allocated = false;

	for (int i=0; i < MAX_GRP_PER_SNN; i++) {
		grp_Info[i].Type = UNKNOWN_NEURON;
		grp_Info[i].MaxFiringRate = UNKNOWN_NEURON_MAX_FIRING_RATE;
		grp_Info[i].SpikeMonitorId = -1;
		grp_Info[i].GroupMonitorId = -1;
//		grp_Info[i].ConnectionMonitorId = -1;
		grp_Info[i].FiringCount1sec=0;
		grp_Info[i].numPostSynapses 		= 0;	// default value
		grp_Info[i].numPreSynapses 	= 0;	// default value
		grp_Info[i].WithSTP = false;
		grp_Info[i].WithSTDP = false;
		grp_Info[i].WithESTDP = false;
		grp_Info[i].WithISTDP = false;
		grp_Info[i].WithESTDPtype = UNKNOWN_STDP;
		grp_Info[i].WithISTDPtype = UNKNOWN_STDP;
		grp_Info[i].WithESTDPcurve = UNKNOWN_CURVE;
		grp_Info[i].WithISTDPcurve = UNKNOWN_CURVE;
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
	wtANDwtChangeUpdateInterval_ = 1000; // update weights every 1000 ms (default)
	wtANDwtChangeUpdateIntervalCnt_ = 0; // helper var to implement fast modulo
	stdpScaleFactor_ = 1.0f;
	wtChangeDecay_ = 0.0f;

	if (simMode_ == GPU_MODE)
		configGPUDevice();
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

	cpuRuntimeData.voltage	   = new float[numNReg];
	cpuRuntimeData.recovery   = new float[numNReg];
	cpuRuntimeData.Izh_a	   = new float[numNReg];
	cpuRuntimeData.Izh_b      = new float[numNReg];
	cpuRuntimeData.Izh_c	   = new float[numNReg];
	cpuRuntimeData.Izh_d	   = new float[numNReg];
	cpuRuntimeData.current	   = new float[numNReg];
	cpuRuntimeData.extCurrent = new float[numNReg];
	memset(cpuRuntimeData.extCurrent, 0, sizeof(cpuRuntimeData.extCurrent[0])*numNReg);

	cpuSnnSz.neuronInfoSize += (sizeof(float)*numNReg*8);

	if (sim_with_conductances) {
		cpuRuntimeData.gAMPA  = new float[numNReg];
		cpuRuntimeData.gGABAa = new float[numNReg];
		cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;

		if (sim_with_NMDA_rise) {
			// If NMDA rise time is enabled, we'll have to compute NMDA conductance in two steps (using an exponential
			// for the rise time and one for the decay time)
			cpuRuntimeData.gNMDA_r = new float[numNReg];
			cpuRuntimeData.gNMDA_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			cpuRuntimeData.gNMDA = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}

		if (sim_with_GABAb_rise) {
			cpuRuntimeData.gGABAb_r = new float[numNReg];
			cpuRuntimeData.gGABAb_d = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg*2;
		} else {
			cpuRuntimeData.gGABAb = new float[numNReg];
			cpuSnnSz.neuronInfoSize += sizeof(float)*numNReg;
		}
	}

	cpuRuntimeData.grpDA = new float[numGrp];
	cpuRuntimeData.grp5HT = new float[numGrp];
	cpuRuntimeData.grpACh = new float[numGrp];
	cpuRuntimeData.grpNE = new float[numGrp];

	// init neuromodulators and their assistive buffers
	for (int i = 0; i < numGrp; i++) {
		cpuRuntimeData.grpDABuffer[i] = new float[1000]; // 1 second DA buffer
		cpuRuntimeData.grp5HTBuffer[i] = new float[1000];
		cpuRuntimeData.grpAChBuffer[i] = new float[1000];
		cpuRuntimeData.grpNEBuffer[i] = new float[1000];
	}

	resetCurrent();
	resetConductances();

	cpuRuntimeData.lastSpikeTime	= new uint32_t[numN];
	cpuSnnSz.neuronInfoSize += sizeof(int) * numN;
	memset(cpuRuntimeData.lastSpikeTime, 0, sizeof(cpuRuntimeData.lastSpikeTime[0]) * numN);

	cpuRuntimeData.curSpike   = new bool[numN];
	cpuRuntimeData.nSpikeCnt  = new int[numN];
	KERNEL_INFO("allocated nSpikeCnt");

	//! homeostasis variables
	if (sim_with_homeostasis) {
		cpuRuntimeData.avgFiring  = new float[numN];
		cpuRuntimeData.baseFiring = new float[numN];
	}

	#ifdef NEURON_NOISE
	intrinsicWeight  = new float[numN];
	memset(intrinsicWeight,0,sizeof(float)*numN);
	cpuSnnSz.neuronInfoSize += (sizeof(int)*numN*2+sizeof(bool)*numN);
	#endif

	// STP can be applied to spike generators, too -> numN
	if (sim_with_stp) {
		// \TODO: The size of these data structures could be reduced to the max synaptic delay of all
		// connections with STP. That number might not be the same as maxDelay_.
		cpuRuntimeData.stpu = new float[numN*(maxDelay_+1)];
		cpuRuntimeData.stpx = new float[numN*(maxDelay_+1)];
		memset(cpuRuntimeData.stpu, 0, sizeof(float)*numN*(maxDelay_+1)); // memset works for 0.0
		for (int i=0; i < numN*(maxDelay_+1); i++)
			cpuRuntimeData.stpx[i] = 1.0f; // but memset doesn't work for 1.0
		cpuSnnSz.synapticInfoSize += (2*sizeof(float)*numN*(maxDelay_+1));
	}

	cpuRuntimeData.Npre 		   = new unsigned short[numN];
	cpuRuntimeData.Npre_plastic   = new unsigned short[numN];
	cpuRuntimeData.Npost 		   = new unsigned short[numN];
	cpuRuntimeData.cumulativePost = new unsigned int[numN];
	cpuRuntimeData.cumulativePre  = new unsigned int[numN];
	cpuSnnSz.networkInfoSize += (int)(sizeof(int) * numN * 3.5);

	postSynCnt = 0;
	preSynCnt  = 0;
	for(int g=0; g<numGrp; g++) {
		// check for INT overflow: postSynCnt is O(numNeurons*numSynapses), must be able to fit within u int limit
		assert(postSynCnt < UINT_MAX - (grp_Info[g].SizeN * grp_Info[g].numPostSynapses));
		assert(preSynCnt < UINT_MAX - (grp_Info[g].SizeN * grp_Info[g].numPreSynapses));
		postSynCnt += (grp_Info[g].SizeN * grp_Info[g].numPostSynapses);
		preSynCnt  += (grp_Info[g].SizeN * grp_Info[g].numPreSynapses);
	}
	assert(postSynCnt/numN <= numPostSynapses_); // divide by numN to prevent INT overflow
	cpuRuntimeData.postSynapticIds		= new post_info_t[postSynCnt+100];
	tmp_SynapticDelay	= new uint8_t[postSynCnt+100];	//!< Temporary array to store the delays of each connection
	cpuRuntimeData.postDelayInfo		= new delay_info_t[numN*(maxDelay_+1)];	//!< Possible delay values are 0....maxDelay_ (inclusive of maxDelay_)
	cpuSnnSz.networkInfoSize += ((sizeof(post_info_t)+sizeof(uint8_t))*postSynCnt+100)+(sizeof(delay_info_t)*numN*(maxDelay_+1));
	assert(preSynCnt/numN <= numPreSynapses_); // divide by numN to prevent INT overflow

	cpuRuntimeData.wt  			= new float[preSynCnt+100];
	cpuRuntimeData.maxSynWt     	= new float[preSynCnt+100];

	mulSynFast 		= new float[MAX_nConnections];
	mulSynSlow 		= new float[MAX_nConnections];
	cpuRuntimeData.cumConnIdPre	= new short int[preSynCnt+100];

	//! Temporary array to hold pre-syn connections. will be deleted later if necessary
	cpuRuntimeData.preSynapticIds	= new post_info_t[preSynCnt + 100];
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
	cpuRuntimeData.lastSpikeTime[nid] = simTime;
	cpuRuntimeData.curSpike[nid] = true;
	cpuRuntimeData.nSpikeCnt[nid]++;
	if (sim_with_homeostasis)
		cpuRuntimeData.avgFiring[nid] += 1000/(grp_Info[g].avgTimeScale*1000);

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
		cpuRuntimeData.stpu[ind_plus] += grp_Info[g].STP_U*(1.0-cpuRuntimeData.stpu[ind_minus]);

		// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
		cpuRuntimeData.stpx[ind_plus] -= cpuRuntimeData.stpu[ind_plus]*cpuRuntimeData.stpx[ind_minus];
	}

	if (grp_Info[g].MaxDelay == 1) {
		assert(nid < numN);
		cpuRuntimeData.firingTableD1[secD1fireCntHost] = nid;
		secD1fireCntHost++;
		grp_Info[g].FiringCount1sec++;
		if (secD1fireCntHost >= maxSpikesD1) {
			spikeBufferFull = 2;
			secD1fireCntHost = maxSpikesD1-1;
		}
	} else {
		assert(nid < numN);
		cpuRuntimeData.firingTableD2[secD2fireCntHost] = nid;
		grp_Info[g].FiringCount1sec++;
		secD2fireCntHost++;
		if (secD2fireCntHost >= maxSpikesD2) {
			spikeBufferFull = 1;
			secD2fireCntHost = maxSpikesD2-1;
		}
	}
	return spikeBufferFull;
}


void SNN::buildGroup(int grpId) {
	assert(grp_Info[grpId].StartN == -1);
	grp_Info[grpId].StartN = allocatedN;
	grp_Info[grpId].EndN   = allocatedN + grp_Info[grpId].SizeN - 1;

	KERNEL_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);

	resetNeuromodulator(grpId);

	allocatedN = allocatedN + grp_Info[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		resetNeuron(i, grpId);
		cpuRuntimeData.Npre_plastic[i]	= 0;
		cpuRuntimeData.Npre[i]		  	= 0;
		cpuRuntimeData.Npost[i]	  	= 0;
		cpuRuntimeData.cumulativePost[i] = allocatedPost;
		cpuRuntimeData.cumulativePre[i]  = allocatedPre;
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
void SNN::buildNetwork() {
	grpConnectInfo_t* newInfo = connectBegin;

	// find the maximum values for number of pre- and post-synaptic neurons
	findMaxNumSynapses(&numPostSynapses_, &numPreSynapses_);

	// update (initialize) maxSpikesD1, maxSpikesD2 and allocate space for firingTableD1 and firingTableD2
	maxDelay_ = updateSpikeTables();

	// make sure number of neurons and max delay are within bounds
	assert(maxDelay_ <= MAX_SynapticDelay); 
	assert(numN <= 1000000);
	assert((numN > 0) && (numN == numNExcReg + numNInhReg + numNPois));

	// display the evaluated network and delay length....
	KERNEL_INFO("\n");
	KERNEL_INFO("***************************** Setting up Network **********************************");
	KERNEL_INFO("numN = %d, numPostSynapses = %d, numPreSynapses = %d, maxDelay = %d", numN, numPostSynapses_,
					numPreSynapses_, maxDelay_);

	if (numPostSynapses_ > MAX_nPostSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPostSynapses>MAX_nPostSynapses)
				KERNEL_ERROR("Grp: %s(%d) has too many output synapses (%d), max %d.",grp_Info2[g].Name.c_str(),g,
							grp_Info[g].numPostSynapses,MAX_nPostSynapses);
		}
		assert(numPostSynapses_ <= MAX_nPostSynapses);
	}
	if (numPreSynapses_ > MAX_nPreSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPreSynapses>MAX_nPreSynapses)
				KERNEL_ERROR("Grp: %s(%d) has too many input synapses (%d), max %d.",grp_Info2[g].Name.c_str(),g,
 							grp_Info[g].numPreSynapses,MAX_nPreSynapses);
		}
		assert(numPreSynapses_ <= MAX_nPreSynapses);
	}

	// initialize all the parameters....
	//! update (initialize) numN, numPostSynapses, numPreSynapses, maxDelay_, postSynCnt, preSynCnt
	//! allocate space for voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, gAMPA, gNMDA, gGABAa, gGABAb
	//! lastSpikeTime, curSpike, nSpikeCnt, intrinsicWeight, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre
	//! postSynapticIds, tmp_SynapticDely, postDelayInfo, wt, maxSynWt, preSynapticIds, timeTableD2, timeTableD1, grpDA, grp5HT, grpACh, grpNE
	buildNetworkInit();

	// we build network in the order...
	/////    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
	////     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
	///      Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
	int allocatedGrp = 0;
	for(int order = 0; order < 4; order++) {
		for(int g = 0; g < numGrp; g++) {
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
	assert(allocatedGrp == numGrp);

	// print group overview
	for (int g=0;g<numGrp;g++) {
		printGroupInfo(g);
	}


	cpuRuntimeData.grpIds = new short int[numN];
	for (int nid=0; nid<numN; nid++) {
		cpuRuntimeData.grpIds[nid] = -1;
		for (int g=0; g<numGrp; g++) {
			if (nid>=grp_Info[g].StartN && nid<=grp_Info[g].EndN) {
				cpuRuntimeData.grpIds[nid] = (short int)g;
//				printf("grpIds[%d] = %d\n",nid,g);
				break;
			}
		}
		assert(cpuRuntimeData.grpIds[nid]!=-1);
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
					printConnectionInfo(newInfo->connId);
				}
				newInfo = newInfo->next;
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
						case CONN_GAUSSIAN:
							connectGaussian(newInfo);
							break;
						case CONN_USER_DEFINED:
							connectUserDefined(newInfo);
							break;
						default:
							KERNEL_ERROR("Invalid connection type( should be 'random', 'full', 'full-no-direct', or 'one-to-one')");
							exitSimulation(-1);
					}

					printConnectionInfo(newInfo->connId);
				}
				newInfo = newInfo->next;
			}
		}
	}
}

void SNN::buildPoissonGroup(int grpId) {
	assert(grp_Info[grpId].StartN == -1);
	grp_Info[grpId].StartN 	= allocatedN;
	grp_Info[grpId].EndN   	= allocatedN + grp_Info[grpId].SizeN - 1;

	KERNEL_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);

	allocatedN = allocatedN + grp_Info[grpId].SizeN;
	assert(allocatedN <= numN);

	for(int i=grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		resetPoissonNeuron(i, grpId);
		cpuRuntimeData.Npre_plastic[i]	  = 0;
		cpuRuntimeData.Npre[i]		  	  = 0;
		cpuRuntimeData.Npost[i]	      = 0;
		cpuRuntimeData.cumulativePost[i] = allocatedPost;
		cpuRuntimeData.cumulativePre[i]  = allocatedPre;
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
void SNN::checkSpikeCounterRecordDur() {
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
		lastCnt_post = tmp_cumulativePost[i-1]+cpuRuntimeData.Npost[i-1]; //position of last pointer
		lastCnt_pre  = tmp_cumulativePre[i-1]+cpuRuntimeData.Npre[i-1]; //position of last pointer
		#if COMPACTION_ALIGNMENT_POST
			lastCnt_post= lastCnt_post + COMPACTION_ALIGNMENT_POST-lastCnt_post%COMPACTION_ALIGNMENT_POST;
			lastCnt_pre = lastCnt_pre  + COMPACTION_ALIGNMENT_PRE- lastCnt_pre%COMPACTION_ALIGNMENT_PRE;
		#endif
		tmp_cumulativePost[i] = lastCnt_post;
		tmp_cumulativePre[i]  = lastCnt_pre;
		assert(tmp_cumulativePost[i] <= cpuRuntimeData.cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= cpuRuntimeData.cumulativePre[i]);
	}

	// compress the post_synaptic array according to the new values of the tmp_cumulative counts....
	unsigned int tmp_postSynCnt = tmp_cumulativePost[numN-1]+cpuRuntimeData.Npost[numN-1];
	unsigned int tmp_preSynCnt  = tmp_cumulativePre[numN-1]+cpuRuntimeData.Npre[numN-1];
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
		assert(tmp_cumulativePost[i] <= cpuRuntimeData.cumulativePost[i]);
		assert(tmp_cumulativePre[i]  <= cpuRuntimeData.cumulativePre[i]);
		for( int j=0; j<cpuRuntimeData.Npost[i]; j++) {
			unsigned int tmpPos = tmp_cumulativePost[i]+j;
			unsigned int oldPos = cpuRuntimeData.cumulativePost[i]+j;
			tmp_postSynapticIds[tmpPos] = cpuRuntimeData.postSynapticIds[oldPos];
			tmp_SynapticDelay[tmpPos]   = tmp_SynapticDelay[oldPos];
		}
		for( int j=0; j<cpuRuntimeData.Npre[i]; j++) {
			unsigned int tmpPos =  tmp_cumulativePre[i]+j;
			unsigned int oldPos =  cpuRuntimeData.cumulativePre[i]+j;
			tmp_preSynapticIds[tmpPos]  = cpuRuntimeData.preSynapticIds[oldPos];
			tmp_maxSynWt[tmpPos] 	    = cpuRuntimeData.maxSynWt[oldPos];
			tmp_wt[tmpPos]              = cpuRuntimeData.wt[oldPos];
			tmp_cumConnIdPre[tmpPos]	= cpuRuntimeData.cumConnIdPre[oldPos];
		}
	}

	// delete old buffer space
	delete[] cpuRuntimeData.postSynapticIds;
	cpuRuntimeData.postSynapticIds = tmp_postSynapticIds;
	cpuSnnSz.networkInfoSize -= (sizeof(post_info_t)*postSynCnt);
	cpuSnnSz.networkInfoSize += (sizeof(post_info_t)*(tmp_postSynCnt+100));

	delete[] cpuRuntimeData.cumulativePost;
	cpuRuntimeData.cumulativePost  = tmp_cumulativePost;

	delete[] cpuRuntimeData.cumulativePre;
	cpuRuntimeData.cumulativePre   = tmp_cumulativePre;

	delete[] cpuRuntimeData.maxSynWt;
	cpuRuntimeData.maxSynWt = tmp_maxSynWt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] cpuRuntimeData.wt;
	cpuRuntimeData.wt = tmp_wt;
	cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_preSynCnt+100));

	delete[] cpuRuntimeData.cumConnIdPre;
	cpuRuntimeData.cumConnIdPre = tmp_cumConnIdPre;
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


	delete[] cpuRuntimeData.preSynapticIds;
	cpuRuntimeData.preSynapticIds  = tmp_preSynapticIds;
	cpuSnnSz.synapticInfoSize -= (sizeof(post_info_t)*preSynCnt);
	cpuSnnSz.synapticInfoSize += (sizeof(post_info_t)*(tmp_preSynCnt+100));

	preSynCnt	= tmp_preSynCnt;
	postSynCnt	= tmp_postSynCnt;
}

// make 'C' full connections from grpSrc to grpDest
void SNN::connectFull(grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	bool noDirect = (info->type == CONN_FULL_NO_DIRECT);

	// rebuild struct for easier handling
	RadiusRF radius(info->radX, info->radY, info->radZ);

	for(int i = grp_Info[grpSrc].StartN; i <= grp_Info[grpSrc].EndN; i++)  {
		Point3D loc_i = getNeuronLocation3D(i); // 3D coordinates of i
		for(int j = grp_Info[grpDest].StartN; j <= grp_Info[grpDest].EndN; j++) { // j: the temp neuron id
			// if flag is set, don't connect direct connections
			if((noDirect) && (i - grp_Info[grpSrc].StartN) == (j - grp_Info[grpDest].StartN))
				continue;

			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j
			if (!isPoint3DinRF(radius, loc_i, loc_j))
				continue;

			//uint8_t dVal = info->minDelay + (int)(0.5 + (drand48() * (info->maxDelay - info->minDelay)));
			uint8_t dVal = info->minDelay + rand() % (info->maxDelay - info->minDelay + 1);
			assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
			float synWt = getWeights(info->connProp, info->initWt, info->maxWt, i, grpSrc);

			setConnection(grpSrc, grpDest, i, j, synWt, info->maxWt, dVal, info->connProp, info->connId);
			info->numberOfConnections++;
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}

void SNN::connectGaussian(grpConnectInfo_t* info) {
	// rebuild struct for easier handling
	// adjust with sqrt(2) in order to make the Gaussian kernel depend on 2*sigma^2
	RadiusRF radius(info->radX, info->radY, info->radZ);

	// in case pre and post have different Grid3D sizes: scale pre to the grid size of post
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	Grid3D grid_i = getGroupGrid3D(grpSrc);
	Grid3D grid_j = getGroupGrid3D(grpDest);
	Point3D scalePre = Point3D(grid_j.x, grid_j.y, grid_j.z) / Point3D(grid_i.x, grid_i.y, grid_i.z);

	for(int i = grp_Info[grpSrc].StartN; i <= grp_Info[grpSrc].EndN; i++)  {
		Point3D loc_i = getNeuronLocation3D(i)*scalePre; // i: adjusted 3D coordinates

		for(int j = grp_Info[grpDest].StartN; j <= grp_Info[grpDest].EndN; j++) { // j: the temp neuron id
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

			if (drand48() < info->p) {
				uint8_t dVal = info->minDelay + rand() % (info->maxDelay - info->minDelay + 1);
				assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
				float synWt = gauss * info->initWt; // scale weight according to gauss distance
				setConnection(grpSrc, grpDest, i, j, synWt, info->maxWt, dVal, info->connProp, info->connId);
				info->numberOfConnections++;
			}
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}

void SNN::connectOneToOne (grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;
	assert( grp_Info[grpDest].SizeN == grp_Info[grpSrc].SizeN );

	// NOTE: RadiusRF does not make a difference here: ignore
	for(int nid=grp_Info[grpSrc].StartN,j=grp_Info[grpDest].StartN; nid<=grp_Info[grpSrc].EndN; nid++, j++)  {
		uint8_t dVal = info->minDelay + rand() % (info->maxDelay - info->minDelay + 1);
		assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
		float synWt = getWeights(info->connProp, info->initWt, info->maxWt, nid, grpSrc);
		setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp, info->connId);
		info->numberOfConnections++;
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
}

// make 'C' random connections from grpSrc to grpDest
void SNN::connectRandom (grpConnectInfo_t* info) {
	int grpSrc = info->grpSrc;
	int grpDest = info->grpDest;

	// rebuild struct for easier handling
	RadiusRF radius(info->radX, info->radY, info->radZ);

	for(int pre_nid=grp_Info[grpSrc].StartN; pre_nid<=grp_Info[grpSrc].EndN; pre_nid++) {
		Point3D loc_pre = getNeuronLocation3D(pre_nid); // 3D coordinates of i
		for(int post_nid=grp_Info[grpDest].StartN; post_nid<=grp_Info[grpDest].EndN; post_nid++) {
			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_post = getNeuronLocation3D(post_nid); // 3D coordinates of j
			if (!isPoint3DinRF(radius, loc_pre, loc_post))
				continue;

			if (drand48() < info->p) {
				//uint8_t dVal = info->minDelay + (int)(0.5+(drand48()*(info->maxDelay-info->minDelay)));
				uint8_t dVal = info->minDelay + rand() % (info->maxDelay - info->minDelay + 1);
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
void SNN::connectUserDefined (grpConnectInfo_t* info) {
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

				info->maxWt = maxWt;

				assert(delay >= 1);
				assert(delay <= MAX_SynapticDelay);
				assert(abs(weight) <= abs(maxWt));

				// adjust the sign of the weight based on inh/exc connection
				weight = isExcitatoryGroup(grpSrc) ? fabs(weight) : -1.0*fabs(weight);
				maxWt  = isExcitatoryGroup(grpSrc) ? fabs(maxWt)  : -1.0*fabs(maxWt);

				setConnection(grpSrc, grpDest, nid, nid2, weight, maxWt, delay, info->connProp, info->connId);
				info->numberOfConnections++;
				if(delay > info->maxDelay) {
					info->maxDelay = delay;
				}
			}
		}
	}

	grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
	grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
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

	resetPointers(true); // deallocate pointers

	// do the same as above, but for snn_gpu.cu
	deleteObjects_GPU();
	simulatorDeleted = true;
}



// This method loops through all spikes that are generated by neurons with a delay of 1ms
// and delivers the spikes to the appropriate post-synaptic neuron
void SNN::doD1CurrentUpdate() {
	int k     = secD1fireCntHost-1;
	int k_end = timeTableD1[simTimeMs+maxDelay_];

	while((k>=k_end) && (k>=0)) {

		int neuron_id      = cpuRuntimeData.firingTableD1[k];
		assert(neuron_id<numN);

		delay_info_t dPar = cpuRuntimeData.postDelayInfo[neuron_id*(maxDelay_+1)];

		unsigned int  offset = cpuRuntimeData.cumulativePost[neuron_id];

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
void SNN::doD2CurrentUpdate() {
	int k = secD2fireCntHost-1;
	int k_end = timeTableD2[simTimeMs+1];
	int t_pos = simTimeMs;

	while((k>=k_end)&& (k >=0)) {

		// get the neuron id from the index k
		int i  = cpuRuntimeData.firingTableD2[k];

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

		delay_info_t dPar = cpuRuntimeData.postDelayInfo[i*(maxDelay_+1)+tD];

		unsigned int offset = cpuRuntimeData.cumulativePost[i];

		// for each delay variables
		for(int idx_d = dPar.delay_index_start;
			idx_d < (dPar.delay_index_start + dPar.delay_length);
			idx_d = idx_d+1) {
			generatePostSpike( i, idx_d, offset, tD);
		}

		k=k-1;
	}
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

void SNN::doSTPUpdateAndDecayCond() {
	int spikeBufferFull = 0;

	//decay the STP variables before adding new spikes.
	for(int g=0; (g < numGrp) & !spikeBufferFull; g++) {
		for(int i=grp_Info[g].StartN; i<=grp_Info[g].EndN; i++) {
	   		//decay the STP variables before adding new spikes.
			if (grp_Info[g].WithSTP) {
				int ind_plus  = STP_BUF_POS(i,simTime);
				int ind_minus = STP_BUF_POS(i,(simTime-1));
				cpuRuntimeData.stpu[ind_plus] = cpuRuntimeData.stpu[ind_minus]*(1.0-grp_Info[g].STP_tau_u_inv);
				cpuRuntimeData.stpx[ind_plus] = cpuRuntimeData.stpx[ind_minus] + (1.0-cpuRuntimeData.stpx[ind_minus])*grp_Info[g].STP_tau_x_inv;
			}

			if (grp_Info[g].Type&POISSON_NEURON)
				continue;

			// decay conductances
			if (sim_with_conductances) {
				cpuRuntimeData.gAMPA[i]  *= dAMPA;
				cpuRuntimeData.gGABAa[i] *= dGABAa;

				if (sim_with_NMDA_rise) {
					cpuRuntimeData.gNMDA_r[i] *= rNMDA;	// rise
					cpuRuntimeData.gNMDA_d[i] *= dNMDA;	// decay
				} else {
					cpuRuntimeData.gNMDA[i]   *= dNMDA;	// instantaneous rise
				}

				if (sim_with_GABAb_rise) {
					cpuRuntimeData.gGABAb_r[i] *= rGABAb;	// rise
					cpuRuntimeData.gGABAb_d[i] *= dGABAb;	// decay
				} else {
					cpuRuntimeData.gGABAb[i] *= dGABAb;	// instantaneous rise
				}
			}
			else {
				cpuRuntimeData.current[i] = 0.0f; // in CUBA mode, reset current to 0 at each time step and sum up all wts
			}
		}
	}
}

void SNN::findFiring() {
	int spikeBufferFull = 0;

	for(int g=0; (g < numGrp) & !spikeBufferFull; g++) {
		// given group of neurons belong to the poisson group....
		if (grp_Info[g].Type&POISSON_NEURON)
			continue;

		// his flag is set if with_stdp is set and also grpType is set to have GROUP_SYN_FIXED
		for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {

			assert(i < numNReg);

			if (cpuRuntimeData.voltage[i] >= 30.0) {
				cpuRuntimeData.voltage[i] = cpuRuntimeData.Izh_c[i];
				cpuRuntimeData.recovery[i] += cpuRuntimeData.Izh_d[i];

				// if flag hasSpkMonRT is set, we want to keep track of how many spikes per neuron in the group
				if (grp_Info[g].withSpikeCounter) {// put the condition for runNetwork
					int bufPos = grp_Info[g].spkCntBufPos; // retrieve buf pos
					int bufNeur = i-grp_Info[g].StartN;
					spkCntBuf[bufPos][bufNeur]++;
				}
				spikeBufferFull = addSpikeToTable(i, g);

				if (spikeBufferFull)
					break;

				// STDP calculation: the post-synaptic neuron fires after the arrival of a pre-synaptic spike
				if (!sim_in_testing && grp_Info[g].WithSTDP) {
					unsigned int pos_ij = cpuRuntimeData.cumulativePre[i]; // the index of pre-synaptic neuron
					for(int j=0; j < cpuRuntimeData.Npre_plastic[i]; pos_ij++, j++) {
						int stdp_tDiff = (simTime-cpuRuntimeData.synSpikeTime[pos_ij]);
						assert(!((stdp_tDiff < 0) && (cpuRuntimeData.synSpikeTime[pos_ij] != MAX_SIMULATION_TIME)));

						if (stdp_tDiff > 0) {
							// check this is an excitatory or inhibitory synapse
							if (grp_Info[g].WithESTDP && cpuRuntimeData.maxSynWt[pos_ij] >= 0) { // excitatory synapse
								// Handle E-STDP curve
								switch (grp_Info[g].WithESTDPcurve) {
								case EXP_CURVE: // exponential curve
									if (stdp_tDiff * grp_Info[g].TAU_PLUS_INV_EXC < 25)
										cpuRuntimeData.wtChange[pos_ij] += STDP(stdp_tDiff, grp_Info[g].ALPHA_PLUS_EXC, grp_Info[g].TAU_PLUS_INV_EXC);
									break;
								case TIMING_BASED_CURVE: // sc curve
									if (stdp_tDiff * grp_Info[g].TAU_PLUS_INV_EXC < 25) {
										if (stdp_tDiff <= grp_Info[g].GAMMA)
											cpuRuntimeData.wtChange[pos_ij] += grp_Info[g].OMEGA + grp_Info[g].KAPPA * STDP(stdp_tDiff, grp_Info[g].ALPHA_PLUS_EXC, grp_Info[g].TAU_PLUS_INV_EXC);
										else // stdp_tDiff > GAMMA
											cpuRuntimeData.wtChange[pos_ij] -= STDP(stdp_tDiff, grp_Info[g].ALPHA_PLUS_EXC, grp_Info[g].TAU_PLUS_INV_EXC);
									}
									break;
								default:
									KERNEL_ERROR("Invalid E-STDP curve!");
									break;
								}
							} else if (grp_Info[g].WithISTDP && cpuRuntimeData.maxSynWt[pos_ij] < 0) { // inhibitory synapse
								// Handle I-STDP curve
								switch (grp_Info[g].WithISTDPcurve) {
								case EXP_CURVE: // exponential curve
									if (stdp_tDiff * grp_Info[g].TAU_PLUS_INV_INB < 25) { // LTP of inhibitory synapse, which decreases synapse weight
										cpuRuntimeData.wtChange[pos_ij] -= STDP(stdp_tDiff, grp_Info[g].ALPHA_PLUS_INB, grp_Info[g].TAU_PLUS_INV_INB);
									}
									break;
								case PULSE_CURVE: // pulse curve
									if (stdp_tDiff <= grp_Info[g].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
										cpuRuntimeData.wtChange[pos_ij] -= grp_Info[g].BETA_LTP;
										//printf("I-STDP LTP\n");
									} else if (stdp_tDiff <= grp_Info[g].DELTA) { // LTD of inhibitory syanpse, which increase sysnapse weight
										cpuRuntimeData.wtChange[pos_ij] -= grp_Info[g].BETA_LTD;
										//printf("I-STDP LTD\n");
									} else { /*do nothing*/}
									break;
								default:
									KERNEL_ERROR("Invalid I-STDP curve!");
									break;
								}
							}
						}
					}
				}
				spikeCountAll1secHost++;
			}
		}
	}
}

int SNN::findGrpId(int nid) {
	KERNEL_WARN("Using findGrpId is deprecated, use array grpIds[] instead...");
	for(int g=0; g < numGrp; g++) {
		if(nid >=grp_Info[g].StartN && (nid <=grp_Info[g].EndN)) {
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
		if (grp_Info[g].numPostSynapses >= *numPostSynapses)
			*numPostSynapses = grp_Info[g].numPostSynapses;
		if (grp_Info[g].numPreSynapses >= *numPreSynapses)
			*numPreSynapses = grp_Info[g].numPreSynapses;
	}
}

void SNN::generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD) {
	// get synaptic info...
	post_info_t post_info = cpuRuntimeData.postSynapticIds[offset + idx_d];

	// get post-neuron id
	unsigned int post_i = GET_CONN_NEURON_ID(post_info);
	assert(post_i<numN);

	// get syn id
	int s_i = GET_CONN_SYN_ID(post_info);
	assert(s_i<(cpuRuntimeData.Npre[post_i]));

	// get the cumulative position for quick access
	unsigned int pos_i = cpuRuntimeData.cumulativePre[post_i] + s_i;
	assert(post_i < numNReg); // \FIXME is this assert supposed to be for pos_i?

	// get group id of pre- / post-neuron
	short int post_grpId = cpuRuntimeData.grpIds[post_i];
	short int pre_grpId = cpuRuntimeData.grpIds[pre_i];

	unsigned int pre_type = grp_Info[pre_grpId].Type;

	// get connect info from the cumulative synapse index for mulSynFast/mulSynSlow (requires less memory than storing
	// mulSynFast/Slow per synapse or storing a pointer to grpConnectInfo_s)
	// mulSynFast will be applied to fast currents (either AMPA or GABAa)
	// mulSynSlow will be applied to slow currents (either NMDA or GABAb)
	short int mulIndex = cpuRuntimeData.cumConnIdPre[pos_i];
	assert(mulIndex>=0 && mulIndex<numConnections);


	// for each presynaptic spike, postsynaptic (synaptic) current is going to increase by some amplitude (change)
	// generally speaking, this amplitude is the weight; but it can be modulated by STP
	float change = cpuRuntimeData.wt[pos_i];

	if (grp_Info[pre_grpId].WithSTP) {
		// if pre-group has STP enabled, we need to modulate the weight
		// NOTE: Order is important! (Tsodyks & Markram, 1998; Mongillo, Barak, & Tsodyks, 2008)
		// use u^+ (value right after spike-update) but x^- (value right before spike-update)

		// dI/dt = -I/tau_S + A * u^+ * x^- * \delta(t-t_{spk})
		// I noticed that for connect(.., RangeDelay(1), ..) tD will be 0
		int ind_minus = STP_BUF_POS(pre_i,(simTime-tD-1));
		int ind_plus  = STP_BUF_POS(pre_i,(simTime-tD));

		change *= grp_Info[pre_grpId].STP_A*cpuRuntimeData.stpu[ind_plus]*cpuRuntimeData.stpx[ind_minus];

//		fprintf(stderr,"%d: %d[%d], numN=%d, td=%d, maxDelay_=%d, ind-=%d, ind+=%d, stpu=[%f,%f], stpx=[%f,%f], change=%f, wt=%f\n",
//			simTime, pre_grpId, pre_i,
//					numN, tD, maxDelay_, ind_minus, ind_plus,
//					stpu[ind_minus], stpu[ind_plus], stpx[ind_minus], stpx[ind_plus], change, wt[pos_i]);
	}

	// update currents
	// NOTE: it's faster to += 0.0 rather than checking for zero and not updating
	if (sim_with_conductances) {
		if (pre_type & TARGET_AMPA) // if post_i expresses AMPAR
			cpuRuntimeData.gAMPA [post_i] += change*mulSynFast[mulIndex]; // scale by some factor
		if (pre_type & TARGET_NMDA) {
			if (sim_with_NMDA_rise) {
				cpuRuntimeData.gNMDA_r[post_i] += change*sNMDA*mulSynSlow[mulIndex];
				cpuRuntimeData.gNMDA_d[post_i] += change*sNMDA*mulSynSlow[mulIndex];
			} else {
				cpuRuntimeData.gNMDA [post_i] += change*mulSynSlow[mulIndex];
			}
		}
		if (pre_type & TARGET_GABAa)
			cpuRuntimeData.gGABAa[post_i] -= change*mulSynFast[mulIndex]; // wt should be negative for GABAa and GABAb
		if (pre_type & TARGET_GABAb) {
			if (sim_with_GABAb_rise) {
				cpuRuntimeData.gGABAb_r[post_i] -= change*sGABAb*mulSynSlow[mulIndex];
				cpuRuntimeData.gGABAb_d[post_i] -= change*sGABAb*mulSynSlow[mulIndex];
			} else {
				cpuRuntimeData.gGABAb[post_i] -= change*mulSynSlow[mulIndex];
			}
		}
	} else {
		cpuRuntimeData.current[post_i] += change;
	}

	cpuRuntimeData.synSpikeTime[pos_i] = simTime;

	// Got one spike from dopaminergic neuron, increase dopamine concentration in the target area
	if (pre_type & TARGET_DA) {
		cpuRuntimeData.grpDA[post_grpId] += 0.04;
	}

	// STDP calculation: the post-synaptic neuron fires before the arrival of a pre-synaptic spike
	if (!sim_in_testing && grp_Info[post_grpId].WithSTDP) {
		int stdp_tDiff = (simTime-cpuRuntimeData.lastSpikeTime[post_i]);

		if (stdp_tDiff >= 0) {
			if (grp_Info[post_grpId].WithISTDP && ((pre_type & TARGET_GABAa) || (pre_type & TARGET_GABAb))) { // inhibitory syanpse
				// Handle I-STDP curve
				switch (grp_Info[post_grpId].WithISTDPcurve) {
				case EXP_CURVE: // exponential curve
					if ((stdp_tDiff*grp_Info[post_grpId].TAU_MINUS_INV_INB)<25) { // LTD of inhibitory syanpse, which increase synapse weight
						cpuRuntimeData.wtChange[pos_i] -= STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_MINUS_INB, grp_Info[post_grpId].TAU_MINUS_INV_INB);
					}
					break;
				case PULSE_CURVE: // pulse curve
					if (stdp_tDiff <= grp_Info[post_grpId].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
						cpuRuntimeData.wtChange[pos_i] -= grp_Info[post_grpId].BETA_LTP;
					} else if (stdp_tDiff <= grp_Info[post_grpId].DELTA) { // LTD of inhibitory syanpse, which increase synapse weight
						cpuRuntimeData.wtChange[pos_i] -= grp_Info[post_grpId].BETA_LTD;
					} else { /*do nothing*/ }
					break;
				default:
					KERNEL_ERROR("Invalid I-STDP curve");
					break;
				}
			} else if (grp_Info[post_grpId].WithESTDP && ((pre_type & TARGET_AMPA) || (pre_type & TARGET_NMDA))) { // excitatory synapse
				// Handle E-STDP curve
				switch (grp_Info[post_grpId].WithESTDPcurve) {
				case EXP_CURVE: // exponential curve
				case TIMING_BASED_CURVE: // sc curve
					if (stdp_tDiff * grp_Info[post_grpId].TAU_MINUS_INV_EXC < 25)
						cpuRuntimeData.wtChange[pos_i] += STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_MINUS_EXC, grp_Info[post_grpId].TAU_MINUS_INV_EXC);
					break;
				default:
					KERNEL_ERROR("Invalid E-STDP curve");
					break;
				}
			} else { /*do nothing*/ }
		}
		assert(!((stdp_tDiff < 0) && (cpuRuntimeData.lastSpikeTime[post_i] != MAX_SIMULATION_TIME)));
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
		short int g = cpuRuntimeData.grpIds[nid];

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
	SpikeGeneratorCore* spikeGen = grp_Info[grpId].spikeGen;
	int timeSlice = grp_Info[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;
	for(int i = grp_Info[grpId].StartN; i <= grp_Info[grpId].EndN; i++) {
		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = cpuRuntimeData.lastSpikeTime[i];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		// the end of the valid time window is either the length of the scheduling time slice from now (because that
		// is the max of the allowed propagated buffer size) or simply the end of the simulation
		unsigned int endOfTimeWindow = MIN(currTime+timeSlice,simTimeRunStop);

		done = false;
		while (!done) {
			// generate the next spike time (nextSchedTime) from the nextSpikeTime callback
			unsigned int nextSchedTime = spikeGen->nextSpikeTime(this, grpId, i - grp_Info[grpId].StartN, currTime, 
				nextTime, endOfTimeWindow);

			// the generated spike time is valid only if:
			// - it has not been scheduled before (nextSchedTime > nextTime)
			//    - but careful: we would drop spikes at t=0, because we cannot initialize nextTime to -1...
			// - it is within the scheduling time slice (nextSchedTime < endOfTimeWindow)
			// - it is not in the past (nextSchedTime >= currTime)
			if ((nextSchedTime==0 || nextSchedTime>nextTime) && nextSchedTime<endOfTimeWindow && nextSchedTime>=currTime) {
//				fprintf(stderr,"%u: spike scheduled for %d at %u\n",currTime, i-grp_Info[grpId].StartN,nextSchedTime);
				// scheduled spike...
				// \TODO CPU mode does not check whether the same AER event has been scheduled before (bug #212)
				// check how GPU mode does it, then do the same here.
				nextTime = nextSchedTime;
				pbuf->scheduleSpikeTargetGroup(i, nextTime - currTime);
				spikeCnt++;

				// update number of spikes if SpikeCounter set
				if (grp_Info[grpId].withSpikeCounter) {
					int bufPos = grp_Info[grpId].spkCntBufPos; // retrieve buf pos
					int bufNeur = i-grp_Info[grpId].StartN;
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
	PoissonRate* rate = grp_Info[grpId].RatePtr;
	float refPeriod = grp_Info[grpId].RefractPeriod;
	int timeSlice   = grp_Info[grpId].CurrTimeSlice;
	unsigned int currTime = simTime;
	int spikeCnt = 0;

	if (rate == NULL)
		return;

	if (rate->isOnGPU()) {
		KERNEL_ERROR("Specifying rates on the GPU but using the CPU SNN is not supported.");
		exitSimulation(1);
	}

	const int nNeur = rate->getNumNeurons();
	if (nNeur != grp_Info[grpId].SizeN) {
		KERNEL_ERROR("Length of PoissonRate array (%d) did not match number of neurons (%d) for group %d(%s).",
			nNeur, grp_Info[grpId].SizeN, grpId, getGroupName(grpId).c_str());
		exitSimulation(1);
	}

	for (int neurId=0; neurId<nNeur; neurId++) {
		float frate = rate->getRate(neurId);

		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		unsigned int nextTime = cpuRuntimeData.lastSpikeTime[grp_Info[grpId].StartN + neurId];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		done = false;
		while (!done && frate>0) {
			nextTime = poissonSpike(nextTime, frate/1000.0, refPeriod);
			// found a valid timeSlice
			if (nextTime < (currTime+timeSlice)) {
				if (nextTime >= currTime) {
//					int nid = grp_Info[grpId].StartN+cnt;
					pbuf->scheduleSpikeTargetGroup(grp_Info[grpId].StartN + neurId, nextTime-currTime);
					spikeCnt++;

					// update number of spikes if SpikeCounter set
					if (grp_Info[grpId].withSpikeCounter) {
						int bufPos = grp_Info[grpId].spkCntBufPos; // retrieve buf pos
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
		actWts = (initWt + ((nid - grp_Info[grpId].StartN) * (maxWt - initWt) / grp_Info[grpId].SizeN));
	else if (setRampDownWeights)
		actWts = (maxWt - ((nid - grp_Info[grpId].StartN) * (maxWt - initWt) / grp_Info[grpId].SizeN));
	else
		actWts = initWt;

	return actWts;
}


void  SNN::globalStateUpdate() {
	double tmp_iNMDA, tmp_I;
	double tmp_gNMDA, tmp_gGABAb;

	for(int g=0; g < numGrp; g++) {
		if (grp_Info[g].Type&POISSON_NEURON) {
			if (grp_Info[g].WithHomeostasis) {
				for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++)
					cpuRuntimeData.avgFiring[i] *= grp_Info[g].avgTimeScale_decay;
			}
			continue;
		}

		// decay dopamine concentration
		if ((grp_Info[g].WithESTDPtype == DA_MOD || grp_Info[g].WithISTDP == DA_MOD) && cpuRuntimeData.grpDA[g] > grp_Info[g].baseDP) {
			cpuRuntimeData.grpDA[g] *= grp_Info[g].decayDP;
		}
		cpuRuntimeData.grpDABuffer[g][simTimeMs] = cpuRuntimeData.grpDA[g];

		for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
			assert(i < numNReg);
			if (grp_Info[g].WithHomeostasis)
				cpuRuntimeData.avgFiring[i] *= grp_Info[g].avgTimeScale_decay;

			if (sim_with_conductances) {
				// COBA model

				// all the tmpIs will be summed into current[i] in the following loop
				cpuRuntimeData.current[i] = 0.0f;

				// \FIXME: these tmp vars cause a lot of rounding errors... consider rewriting
				for (int j=0; j<COND_INTEGRATION_SCALE; j++) {
					tmp_iNMDA = (cpuRuntimeData.voltage[i]+80.0)*(cpuRuntimeData.voltage[i]+80.0)/60.0/60.0;

					tmp_gNMDA = sim_with_NMDA_rise ? cpuRuntimeData.gNMDA_d[i]-cpuRuntimeData.gNMDA_r[i] : cpuRuntimeData.gNMDA[i];
					tmp_gGABAb = sim_with_GABAb_rise ? cpuRuntimeData.gGABAb_d[i]-cpuRuntimeData.gGABAb_r[i] : cpuRuntimeData.gGABAb[i];

					tmp_I = -(   cpuRuntimeData.gAMPA[i]*(cpuRuntimeData.voltage[i]-0)
									 + tmp_gNMDA*tmp_iNMDA/(1+tmp_iNMDA)*(cpuRuntimeData.voltage[i]-0)
									 + cpuRuntimeData.gGABAa[i]*(cpuRuntimeData.voltage[i]+70)
									 + tmp_gGABAb*(cpuRuntimeData.voltage[i]+90)
								   );

					#ifdef NEURON_NOISE
					double noiseI = -intrinsicWeight[i]*log(drand48());
					if (isnan(noiseI) || isinf(noiseI))
						noiseI = 0;
					tmp_I += noiseI;
					#endif

					cpuRuntimeData.voltage[i] += ((0.04*cpuRuntimeData.voltage[i]+5.0)*cpuRuntimeData.voltage[i]+140.0-cpuRuntimeData.recovery[i]+tmp_I+cpuRuntimeData.extCurrent[i])
						/ COND_INTEGRATION_SCALE;
					assert(!isnan(cpuRuntimeData.voltage[i]) && !isinf(cpuRuntimeData.voltage[i]));

					// keep track of total current
					cpuRuntimeData.current[i] += tmp_I;

					if (cpuRuntimeData.voltage[i] > 30) {
						cpuRuntimeData.voltage[i] = 30;
						j=COND_INTEGRATION_SCALE; // break the loop but evaluate u[i]
//						if (gNMDA[i]>=10.0f) KERNEL_WARN("High NMDA conductance (gNMDA>=10.0) may cause instability");
//						if (gGABAb[i]>=2.0f) KERNEL_WARN("High GABAb conductance (gGABAb>=2.0) may cause instability");
					}
					if (cpuRuntimeData.voltage[i] < -90)
						cpuRuntimeData.voltage[i] = -90;
					cpuRuntimeData.recovery[i]+=cpuRuntimeData.Izh_a[i]*(cpuRuntimeData.Izh_b[i]*cpuRuntimeData.voltage[i]-cpuRuntimeData.recovery[i])/COND_INTEGRATION_SCALE;
				} // end COND_INTEGRATION_SCALE loop
			} else {
				// CUBA model
				cpuRuntimeData.voltage[i] += 0.5*((0.04*cpuRuntimeData.voltage[i]+5.0)*cpuRuntimeData.voltage[i] + 140.0 - cpuRuntimeData.recovery[i]
					+ cpuRuntimeData.current[i] + cpuRuntimeData.extCurrent[i]); //for numerical stability
				cpuRuntimeData.voltage[i] += 0.5*((0.04*cpuRuntimeData.voltage[i]+5.0)*cpuRuntimeData.voltage[i] + 140.0 - cpuRuntimeData.recovery[i]
					+ cpuRuntimeData.current[i] + cpuRuntimeData.extCurrent[i]); //time step is 0.5 ms
				if (cpuRuntimeData.voltage[i] > 30)
					cpuRuntimeData.voltage[i] = 30;
				if (cpuRuntimeData.voltage[i] < -90)
					cpuRuntimeData.voltage[i] = -90;
				cpuRuntimeData.recovery[i]+=cpuRuntimeData.Izh_a[i]*(cpuRuntimeData.Izh_b[i]*cpuRuntimeData.voltage[i]-cpuRuntimeData.recovery[i]);
			} // end COBA/CUBA
		} // end StartN...EndN
	} // end numGrp
}

// initialize all the synaptic weights to appropriate values..
// total size of the synaptic connection is 'length' ...
void SNN::initSynapticWeights() {
	// Initialize the network wtChange, wt, synaptic firing time
	cpuRuntimeData.wtChange         = new float[preSynCnt];
	cpuRuntimeData.synSpikeTime     = new uint32_t[preSynCnt];
	cpuSnnSz.synapticInfoSize = sizeof(float)*(preSynCnt*2);

	resetSynapticConnections(false);
}

// checks whether a connection ID contains plastic synapses O(#connections)
bool SNN::isConnectionPlastic(short int connId) {
	assert(connId!=ALL);
	assert(connId<numConnections);

	// search linked list for right connection ID
	grpConnectInfo_t* connInfo = connectBegin;
	bool isPlastic = false;
	while (connInfo) {
		if (connId == connInfo->connId) {
			// get syn wt type from connection property
			isPlastic = GET_FIXED_PLASTIC(connInfo->connProp);
			break;
		}

		connInfo = connInfo->next;
	}

	return isPlastic;
}

// returns whether group has homeostasis enabled
bool SNN::isGroupWithHomeostasis(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	return (grp_Info[grpId].WithHomeostasis);
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
		if (grp_Info[grpId].WithSTDP) {
			// for each post-group, check if any of the incoming connections are plastic
			grpConnectInfo_t* connInfo = connectBegin;
			bool isAnyPlastic = false;
			while (connInfo) {
				if (connInfo->grpDest == grpId) {
					// get syn wt type from connection property
					isAnyPlastic |= GET_FIXED_PLASTIC(connInfo->connProp);
					if (isAnyPlastic) {
						// at least one plastic connection found: break while
						break;
					}
				}
				connInfo = connInfo->next;
			}
			if (!isAnyPlastic) {
				KERNEL_ERROR("If STDP on group %d (%s) is set, group must have some incoming plastic connections.",
					grpId, grp_Info2[grpId].Name.c_str());
				exitSimulation(1);
			}
		}
	}
}

// checks whether every group with Homeostasis also has STDP
void SNN::verifyHomeostasis() {
	for (int grpId=0; grpId<getNumGroups(); grpId++) {
		if (grp_Info[grpId].WithHomeostasis) {
			if (!grp_Info[grpId].WithSTDP) {
				KERNEL_ERROR("If homeostasis is enabled on group %d (%s), then STDP must be enabled, too.",
					grpId, grp_Info2[grpId].Name.c_str());
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
		if (grp_Info[g].Type==UNKNOWN_NEURON) {
			KERNEL_ERROR("Unknown group for %d (%s)", g, grp_Info2[g].Name.c_str());
			exitSimulation(1);
		}

		if (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON))
			nInhReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type & POISSON_NEURON))
			nExcReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type & POISSON_NEURON))
			nExcPois += grp_Info[g].SizeN;
		else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type & POISSON_NEURON))
			nInhPois += grp_Info[g].SizeN;
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

// creates the CPU net pointers
// don't forget to cudaFree the device pointers if you make gpuRuntimeData
void SNN::makePtrInfo() {
	if (sim_with_NMDA_rise) {
		cpuRuntimeData.gNMDA 		= NULL;

	} else {
		cpuRuntimeData.gNMDA_r 		= NULL;
		cpuRuntimeData.gNMDA_d 		= NULL;
	}
	if (sim_with_GABAb_rise) {
		cpuRuntimeData.gGABAb		= NULL;

	} else {
		cpuRuntimeData.gGABAb_r 	= NULL;
		cpuRuntimeData.gGABAb_d 	= NULL;
	}

	cpuRuntimeData.allocated    	= true;
	cpuRuntimeData.memType      	= CPU_MODE;
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
		if (tmpInt != grp_Info[g].StartN) {
			KERNEL_ERROR("loadSimulation: StartN in file (%d) and grpInfo (%d) for group %d don't match.",
				tmpInt, grp_Info[g].StartN, g);
			exitSimulation(-1);
		}

		// read EndN
		result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
		readErr |= (result!=1);
		if (tmpInt != grp_Info[g].EndN) {
			KERNEL_ERROR("loadSimulation: EndN in file (%d) and grpInfo (%d) for group %d don't match.",
				tmpInt, grp_Info[g].EndN, g);
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
		if (strcmp(name,grp_Info2[g].Name.c_str()) != 0) {
			KERNEL_ERROR("loadSimulation: Group names in file (%s) and grpInfo (%s) don't match.", name,
				grp_Info2[g].Name.c_str());
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

			short int gIDpre = cpuRuntimeData.grpIds[nIDpre];
			if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight>0)
					|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight<0)) {
				KERNEL_ERROR("loadSimulation: Sign of weight value (%s) does not match neuron type (%s)",
					((weight>=0.0f)?"plus":"minus"), 
					(IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type)?"inhibitory":"excitatory"));
				exitSimulation(-1);
			}

			// read max weight
			result = fread(&maxWeight, sizeof(float), 1, loadSimFID);
			readErr |= (result!=1);
			if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight>=0)
					|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight<=0)) {
				KERNEL_ERROR("loadSimulation: Sign of maxWeight value (%s) does not match neuron type (%s)",
					((maxWeight>=0.0f)?"plus":"minus"), 
					(IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type)?"inhibitory":"excitatory"));
				exitSimulation(-1);
			}

			// read delay
			result = fread(&delay, sizeof(uint8_t), 1, loadSimFID);
			readErr |= (result!=1);
			if (delay > MAX_SynapticDelay) {
				KERNEL_ERROR("loadSimulation: delay in file (%d) is larger than MAX_SynapticDelay (%d)",
					(int)delay, (int)MAX_SynapticDelay);
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
				int gIDpost = cpuRuntimeData.grpIds[nIDpost];
				int connProp = SET_FIXED_PLASTIC(plastic?SYN_PLASTIC:SYN_FIXED);

				setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp, connId);
				grp_Info2[gIDpre].sumPostConn++;
				grp_Info2[gIDpost].sumPreConn++;

				if (delay > grp_Info[gIDpre].MaxDelay)
					grp_Info[gIDpre].MaxDelay = delay;
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
		for(int nid=grp_Info[grpId].StartN; nid <= grp_Info[grpId].EndN; nid++) {
			unsigned int jPos=0;					// this points to the top of the delay queue
			unsigned int cumN=cpuRuntimeData.cumulativePost[nid];	// cumulativePost[] is unsigned int
			unsigned int cumDelayStart=0; 			// Npost[] is unsigned short
			for(int td = 0; td < maxDelay_; td++) {
				unsigned int j=jPos;				// start searching from top of the queue until the end
				unsigned int cnt=0;					// store the number of nodes with a delay of td;
				while(j < cpuRuntimeData.Npost[nid]) {
					// found a node j with delay=td and we put
					// the delay value = 1 at array location td=0;
					if(td==(tmp_SynapticDelay[cumN+j]-1)) {
						assert(jPos<cpuRuntimeData.Npost[nid]);
						swapConnections(nid, j, jPos);

						jPos=jPos+1;
						cnt=cnt+1;
					}
					j=j+1;
				}

				// update the delay_length and start values...
				cpuRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_length	     = cnt;
				cpuRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_index_start  = cumDelayStart;
				cumDelayStart += cnt;

				assert(cumDelayStart <= cpuRuntimeData.Npost[nid]);
			}

			// total cumulative delay should be equal to number of post-synaptic connections at the end of the loop
			assert(cumDelayStart == cpuRuntimeData.Npost[nid]);
			for(unsigned int j=1; j < cpuRuntimeData.Npost[nid]; j++) {
				unsigned int cumN=cpuRuntimeData.cumulativePost[nid]; // cumulativePost[] is unsigned int
				if( tmp_SynapticDelay[cumN+j] < tmp_SynapticDelay[cumN+j-1]) {
	  				KERNEL_ERROR("Post-synaptic delays not sorted correctly... id=%d, delay[%d]=%d, delay[%d]=%d",
						nid, j, tmp_SynapticDelay[cumN+j], j-1, tmp_SynapticDelay[cumN+j-1]);
					assert( tmp_SynapticDelay[cumN+j] >= tmp_SynapticDelay[cumN+j-1]);
				}
			}
		}
	}
}

// after all the initalization. Its time to create the synaptic weights, weight change and also
// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
void SNN::reorganizeNetwork(bool removeTempMemory) {
	//Double check...sometimes by mistake we might call reorganize network again...
	if(doneReorganization)
		return;

	KERNEL_DEBUG("Beginning reorganization of network....");

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

	//ensure that we dont do all the above optimizations again
	doneReorganization = true;

	// reset all spike cnt
	resetSpikeCnt(ALL);

	printTuningLog(fpDeb_);

	makePtrInfo();

	KERNEL_INFO("");
	KERNEL_INFO("*****************      Initializing %s Simulation      *************************",
		simMode_==GPU_MODE?"GPU":"CPU");

	if(removeTempMemory) {
		memoryOptimized = true;
		delete[] tmp_SynapticDelay;
		tmp_SynapticDelay = NULL;
	}
}


void SNN::resetConductances() {
	if (sim_with_conductances) {
		memset(cpuRuntimeData.gAMPA, 0, sizeof(float)*numNReg);
		if (sim_with_NMDA_rise) {
			memset(cpuRuntimeData.gNMDA_r, 0, sizeof(float)*numNReg);
			memset(cpuRuntimeData.gNMDA_d, 0, sizeof(float)*numNReg);
		} else {
			memset(cpuRuntimeData.gNMDA, 0, sizeof(float)*numNReg);
		}
		memset(cpuRuntimeData.gGABAa, 0, sizeof(float)*numNReg);
		if (sim_with_GABAb_rise) {
			memset(cpuRuntimeData.gGABAb_r, 0, sizeof(float)*numNReg);
			memset(cpuRuntimeData.gGABAb_d, 0, sizeof(float)*numNReg);
		} else {
			memset(cpuRuntimeData.gGABAb, 0, sizeof(float)*numNReg);
		}
	}
}

void SNN::resetCounters() {
	assert(numNReg <= numN);
	memset(cpuRuntimeData.curSpike, 0, sizeof(bool) * numN);
}

void SNN::resetCPUTiming() {
	prevCpuExecutionTime = cumExecutionTime;
	cpuExecutionTime     = 0.0;
}

void SNN::resetCurrent() {
	assert(cpuRuntimeData.current != NULL);
	memset(cpuRuntimeData.current, 0, sizeof(float) * numNReg);
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

void SNN::resetGPUTiming() {
	prevGpuExecutionTime = cumExecutionTime;
	gpuExecutionTime     = 0.0;
}

void SNN::resetGroups() {
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

void SNN::resetNeuromodulator(int grpId) {
	cpuRuntimeData.grpDA[grpId] = grp_Info[grpId].baseDP;
	cpuRuntimeData.grp5HT[grpId] = grp_Info[grpId].base5HT;
	cpuRuntimeData.grpACh[grpId] = grp_Info[grpId].baseACh;
	cpuRuntimeData.grpNE[grpId] = grp_Info[grpId].baseNE;
}

void SNN::resetNeuron(unsigned int neurId, int grpId) {
	assert(neurId < numNReg);
    if (grp_Info2[grpId].Izh_a == -1) {
		KERNEL_ERROR("setNeuronParameters must be called for group %s (%d)",grp_Info2[grpId].Name.c_str(),grpId);
		exitSimulation(1);
	}

	cpuRuntimeData.Izh_a[neurId] = grp_Info2[grpId].Izh_a + grp_Info2[grpId].Izh_a_sd*(float)drand48();
	cpuRuntimeData.Izh_b[neurId] = grp_Info2[grpId].Izh_b + grp_Info2[grpId].Izh_b_sd*(float)drand48();
	cpuRuntimeData.Izh_c[neurId] = grp_Info2[grpId].Izh_c + grp_Info2[grpId].Izh_c_sd*(float)drand48();
	cpuRuntimeData.Izh_d[neurId] = grp_Info2[grpId].Izh_d + grp_Info2[grpId].Izh_d_sd*(float)drand48();

	cpuRuntimeData.voltage[neurId] = cpuRuntimeData.Izh_c[neurId];	// initial values for new_v
	cpuRuntimeData.recovery[neurId] = cpuRuntimeData.Izh_b[neurId]*cpuRuntimeData.voltage[neurId]; // initial values for u


 	if (grp_Info[grpId].WithHomeostasis) {
		// set the baseFiring with some standard deviation.
		if(drand48()>0.5)   {
			cpuRuntimeData.baseFiring[neurId] = grp_Info2[grpId].baseFiring + grp_Info2[grpId].baseFiringSD*-log(drand48());
		} else  {
			cpuRuntimeData.baseFiring[neurId] = grp_Info2[grpId].baseFiring - grp_Info2[grpId].baseFiringSD*-log(drand48());
			if(cpuRuntimeData.baseFiring[neurId] < 0.1) cpuRuntimeData.baseFiring[neurId] = 0.1;
		}

		if( grp_Info2[grpId].baseFiring != 0.0) {
			cpuRuntimeData.avgFiring[neurId]  = cpuRuntimeData.baseFiring[neurId];
		} else {
			cpuRuntimeData.baseFiring[neurId] = 0.0;
			cpuRuntimeData.avgFiring[neurId]  = 0;
		}
	}

	cpuRuntimeData.lastSpikeTime[neurId]  = MAX_SIMULATION_TIME;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(neurId,j);
			cpuRuntimeData.stpu[ind] = 0.0f;
			cpuRuntimeData.stpx[ind] = 1.0f;
		}
	}
}

void SNN::resetPointers(bool deallocate) {
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

	// delete all Spike Counters
	for (int i=0; i<numSpkCnt; i++) {
		if (spkCntBuf[i]!=NULL && deallocate)
			delete[] spkCntBuf[i];
		spkCntBuf[i]=NULL;
	}

	if (pbuf!=NULL && deallocate) delete pbuf;
	if (cpuRuntimeData.spikeGenBits!=NULL && deallocate) delete[] cpuRuntimeData.spikeGenBits;
	pbuf=NULL; cpuRuntimeData.spikeGenBits=NULL;

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
	if (cpuRuntimeData.grpDA != NULL && deallocate) delete [] cpuRuntimeData.grpDA;
	if (cpuRuntimeData.grp5HT != NULL && deallocate) delete [] cpuRuntimeData.grp5HT;
	if (cpuRuntimeData.grpACh != NULL && deallocate) delete [] cpuRuntimeData.grpACh;
	if (cpuRuntimeData.grpNE != NULL && deallocate) delete [] cpuRuntimeData.grpNE;
	cpuRuntimeData.grpDA = NULL;
	cpuRuntimeData.grp5HT = NULL;
	cpuRuntimeData.grpACh = NULL;
	cpuRuntimeData.grpNE = NULL;

	// clear assistive data buffer for group monitor
	if (deallocate) {
		for (int i = 0; i < numGrp; i++) {
			if (cpuRuntimeData.grpDABuffer[i] != NULL) delete [] cpuRuntimeData.grpDABuffer[i];
			if (cpuRuntimeData.grp5HTBuffer[i] != NULL) delete [] cpuRuntimeData.grp5HTBuffer[i];
			if (cpuRuntimeData.grpAChBuffer[i] != NULL) delete [] cpuRuntimeData.grpAChBuffer[i];
			if (cpuRuntimeData.grpNEBuffer[i] != NULL) delete [] cpuRuntimeData.grpNEBuffer[i];
			cpuRuntimeData.grpDABuffer[i] = NULL;
			cpuRuntimeData.grp5HTBuffer[i] = NULL;
			cpuRuntimeData.grpAChBuffer[i] = NULL;
			cpuRuntimeData.grpNEBuffer[i] = NULL;
		}
	} else {
		memset(cpuRuntimeData.grpDABuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(cpuRuntimeData.grp5HTBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(cpuRuntimeData.grpAChBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
		memset(cpuRuntimeData.grpNEBuffer, 0, sizeof(float*) * MAX_GRP_PER_SNN);
	}


	// -------------- DEALLOCATE CORE OBJECTS ---------------------- //

	if (cpuRuntimeData.voltage!=NULL && deallocate) delete[] cpuRuntimeData.voltage;
	if (cpuRuntimeData.recovery!=NULL && deallocate) delete[] cpuRuntimeData.recovery;
	if (cpuRuntimeData.current!=NULL && deallocate) delete[] cpuRuntimeData.current;
	if (cpuRuntimeData.extCurrent!=NULL && deallocate) delete[] cpuRuntimeData.extCurrent;
	cpuRuntimeData.voltage=NULL; cpuRuntimeData.recovery=NULL; cpuRuntimeData.current=NULL; cpuRuntimeData.extCurrent=NULL;

	if (cpuRuntimeData.Izh_a!=NULL && deallocate) delete[] cpuRuntimeData.Izh_a;
	if (cpuRuntimeData.Izh_b!=NULL && deallocate) delete[] cpuRuntimeData.Izh_b;
	if (cpuRuntimeData.Izh_c!=NULL && deallocate) delete[] cpuRuntimeData.Izh_c;
	if (cpuRuntimeData.Izh_d!=NULL && deallocate) delete[] cpuRuntimeData.Izh_d;
	cpuRuntimeData.Izh_a=NULL; cpuRuntimeData.Izh_b=NULL; cpuRuntimeData.Izh_c=NULL; cpuRuntimeData.Izh_d=NULL;

	if (cpuRuntimeData.Npre!=NULL && deallocate) delete[] cpuRuntimeData.Npre;
	if (cpuRuntimeData.Npre_plastic!=NULL && deallocate) delete[] cpuRuntimeData.Npre_plastic;
	if (cpuRuntimeData.Npost!=NULL && deallocate) delete[] cpuRuntimeData.Npost;
	cpuRuntimeData.Npre=NULL; cpuRuntimeData.Npre_plastic=NULL; cpuRuntimeData.Npost=NULL;

	if (cpuRuntimeData.cumulativePre!=NULL && deallocate) delete[] cpuRuntimeData.cumulativePre;
	if (cpuRuntimeData.cumulativePost!=NULL && deallocate) delete[] cpuRuntimeData.cumulativePost;
	cpuRuntimeData.cumulativePre=NULL; cpuRuntimeData.cumulativePost=NULL;

	if (cpuRuntimeData.gAMPA!=NULL && deallocate) delete[] cpuRuntimeData.gAMPA;
	if (cpuRuntimeData.gNMDA!=NULL && deallocate) delete[] cpuRuntimeData.gNMDA;
	if (cpuRuntimeData.gNMDA_r!=NULL && deallocate) delete[] cpuRuntimeData.gNMDA_r;
	if (cpuRuntimeData.gNMDA_d!=NULL && deallocate) delete[] cpuRuntimeData.gNMDA_d;
	if (cpuRuntimeData.gGABAa!=NULL && deallocate) delete[] cpuRuntimeData.gGABAa;
	if (cpuRuntimeData.gGABAb!=NULL && deallocate) delete[] cpuRuntimeData.gGABAb;
	if (cpuRuntimeData.gGABAb_r!=NULL && deallocate) delete[] cpuRuntimeData.gGABAb_r;
	if (cpuRuntimeData.gGABAb_d!=NULL && deallocate) delete[] cpuRuntimeData.gGABAb_d;
	cpuRuntimeData.gAMPA=NULL; cpuRuntimeData.gNMDA=NULL; cpuRuntimeData.gNMDA_r=NULL; cpuRuntimeData.gNMDA_d=NULL;
	cpuRuntimeData.gGABAa=NULL; cpuRuntimeData.gGABAb=NULL; cpuRuntimeData.gGABAb_r=NULL; cpuRuntimeData.gGABAb_d=NULL;

	if (cpuRuntimeData.stpu!=NULL && deallocate) delete[] cpuRuntimeData.stpu;
	if (cpuRuntimeData.stpx!=NULL && deallocate) delete[] cpuRuntimeData.stpx;
	cpuRuntimeData.stpu=NULL; cpuRuntimeData.stpx=NULL;

	if (cpuRuntimeData.avgFiring!=NULL && deallocate) delete[] cpuRuntimeData.avgFiring;
	if (cpuRuntimeData.baseFiring!=NULL && deallocate) delete[] cpuRuntimeData.baseFiring;
	cpuRuntimeData.avgFiring=NULL; cpuRuntimeData.baseFiring=NULL;

	if (cpuRuntimeData.lastSpikeTime!=NULL && deallocate) delete[] cpuRuntimeData.lastSpikeTime;
	if (cpuRuntimeData.synSpikeTime !=NULL && deallocate) delete[] cpuRuntimeData.synSpikeTime;
	if (cpuRuntimeData.curSpike!=NULL && deallocate) delete[] cpuRuntimeData.curSpike;
	if (cpuRuntimeData.nSpikeCnt!=NULL && deallocate) delete[] cpuRuntimeData.nSpikeCnt;
	cpuRuntimeData.lastSpikeTime=NULL; cpuRuntimeData.synSpikeTime=NULL; cpuRuntimeData.curSpike=NULL; cpuRuntimeData.nSpikeCnt=NULL;

	if (cpuRuntimeData.postDelayInfo!=NULL && deallocate) delete[] cpuRuntimeData.postDelayInfo;
	if (cpuRuntimeData.preSynapticIds!=NULL && deallocate) delete[] cpuRuntimeData.preSynapticIds;
	if (cpuRuntimeData.postSynapticIds!=NULL && deallocate) delete[] cpuRuntimeData.postSynapticIds;
	cpuRuntimeData.postDelayInfo=NULL; cpuRuntimeData.preSynapticIds=NULL; cpuRuntimeData.postSynapticIds=NULL;

	if (cpuRuntimeData.wt!=NULL && deallocate) delete[] cpuRuntimeData.wt;
	if (cpuRuntimeData.maxSynWt!=NULL && deallocate) delete[] cpuRuntimeData.maxSynWt;
	if (cpuRuntimeData.wtChange !=NULL && deallocate) delete[] cpuRuntimeData.wtChange;
	cpuRuntimeData.wt=NULL; cpuRuntimeData.maxSynWt=NULL; cpuRuntimeData.wtChange=NULL;

	if (mulSynFast!=NULL && deallocate) delete[] mulSynFast;
	if (mulSynSlow!=NULL && deallocate) delete[] mulSynSlow;
	if (cpuRuntimeData.cumConnIdPre!=NULL && deallocate) delete[] cpuRuntimeData.cumConnIdPre;
	mulSynFast=NULL; mulSynSlow=NULL; cpuRuntimeData.cumConnIdPre=NULL;

	if (cpuRuntimeData.grpIds!=NULL && deallocate) delete[] cpuRuntimeData.grpIds;
	cpuRuntimeData.grpIds=NULL;

	if (cpuRuntimeData.firingTableD2!=NULL && deallocate) delete[] cpuRuntimeData.firingTableD2;
	if (cpuRuntimeData.firingTableD1!=NULL && deallocate) delete[] cpuRuntimeData.firingTableD1;
	if (timeTableD2!=NULL && deallocate) delete[] timeTableD2;
	if (timeTableD1!=NULL && deallocate) delete[] timeTableD1;
	cpuRuntimeData.firingTableD2=NULL; cpuRuntimeData.firingTableD1=NULL; timeTableD2=NULL; timeTableD1=NULL;

	// clear poisson generator
	if (gpuPoissonRand != NULL) delete gpuPoissonRand;
	gpuPoissonRand = NULL;
}


void SNN::resetPoissonNeuron(unsigned int nid, int grpId) {
	assert(nid < numN);
	cpuRuntimeData.lastSpikeTime[nid]  = MAX_SIMULATION_TIME;
	if (grp_Info[grpId].WithHomeostasis)
		cpuRuntimeData.avgFiring[nid] = 0.0;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j<=maxDelay_; j++) { // is of size maxDelay_+1
			int ind = STP_BUF_POS(nid,j);
			cpuRuntimeData.stpu[nid] = 0.0f;
			cpuRuntimeData.stpx[nid] = 1.0f;
		}
	}
}

void SNN::resetPropogationBuffer() {
	pbuf->reset(0, 1023);
}

// resets nSpikeCnt[]
void SNN::resetSpikeCnt(int grpId) {
	int startGrp, endGrp;

	if (!doneReorganization)
		return;

	if (grpId == -1) {
		startGrp = 0;
		endGrp = numGrp;
	} else {
		 startGrp = grpId;
		 endGrp = grpId;
	}

	for (int g = startGrp; g<endGrp; g++) {
		int startN = grp_Info[g].StartN;
		int endN   = grp_Info[g].EndN;
		for (int i=startN; i<=endN; i++)
			cpuRuntimeData.nSpikeCnt[i] = 0;
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
		const char* updateStr = (grp_Info[destGrp].newUpdates == true)?"(**)":"";
		KERNEL_DEBUG("Grp: %d:%s s=%d e=%d %s", destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
					grp_Info[destGrp].EndN,  updateStr);
		KERNEL_DEBUG("Grp: %d:%s s=%d e=%d  %s",  destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
					grp_Info[destGrp].EndN, updateStr);

		for(int nid=grp_Info[destGrp].StartN; nid <= grp_Info[destGrp].EndN; nid++) {
			unsigned int offset = cpuRuntimeData.cumulativePre[nid];
			for (j=0;j<cpuRuntimeData.Npre[nid]; j++) {
				cpuRuntimeData.wtChange[offset+j] = 0.0;						// synaptic derivatives is reset
				cpuRuntimeData.synSpikeTime[offset+j] = MAX_SIMULATION_TIME;	// some large negative value..
			}
			post_info_t *preIdPtr = &(cpuRuntimeData.preSynapticIds[cpuRuntimeData.cumulativePre[nid]]);
			float* synWtPtr       = &(cpuRuntimeData.wt[cpuRuntimeData.cumulativePre[nid]]);
			float* maxWtPtr       = &(cpuRuntimeData.maxSynWt[cpuRuntimeData.cumulativePre[nid]]);
			int prevPreGrp  = -1;

			for (j=0; j < cpuRuntimeData.Npre[nid]; j++,preIdPtr++, synWtPtr++, maxWtPtr++) {
				int preId    = GET_CONN_NEURON_ID((*preIdPtr));
				assert(preId < numN);
				int srcGrp = cpuRuntimeData.grpIds[preId];
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
						KERNEL_DEBUG("\t%d (%s) start=%d, type=%s maxWts = %f %s", srcGrp,
										grp_Info2[srcGrp].Name.c_str(), j, (j<cpuRuntimeData.Npre_plastic[nid]?"P":"F"),
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
	if(cpuRuntimeData.Npost[src] >= grp_Info[srcGrp].numPostSynapses)	{
		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		KERNEL_ERROR("Large number of postsynaptic connections established (%d), max for this group %d.", cpuRuntimeData.Npost[src], grp_Info[srcGrp].numPostSynapses);
		exitSimulation(1);
	}

	if(cpuRuntimeData.Npre[dest] >= grp_Info[destGrp].numPreSynapses) {
		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		KERNEL_ERROR("Large number of presynaptic connections established (%d), max for this group %d.", cpuRuntimeData.Npre[dest], grp_Info[destGrp].numPreSynapses);
		exitSimulation(1);
	}

	int p = cpuRuntimeData.Npost[src];

	assert(cpuRuntimeData.Npost[src] >= 0);
	assert(cpuRuntimeData.Npre[dest] >= 0);
	assert((src * numPostSynapses_ + p) / numN < numPostSynapses_); // divide by numN to prevent INT overflow

	unsigned int post_pos = cpuRuntimeData.cumulativePost[src] + cpuRuntimeData.Npost[src];
	unsigned int pre_pos  = cpuRuntimeData.cumulativePre[dest] + cpuRuntimeData.Npre[dest];

	assert(post_pos < postSynCnt);
	assert(pre_pos  < preSynCnt);

	//generate a new postSynapticIds id for the current connection
	cpuRuntimeData.postSynapticIds[post_pos]   = SET_CONN_ID(dest, cpuRuntimeData.Npre[dest], destGrp);
	tmp_SynapticDelay[post_pos] = dVal;

	cpuRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID(src, cpuRuntimeData.Npost[src], srcGrp);
	cpuRuntimeData.wt[pre_pos] 	  = synWt;
	cpuRuntimeData.maxSynWt[pre_pos] = maxWt;
	cpuRuntimeData.cumConnIdPre[pre_pos] = connId;

	bool synWtType = GET_FIXED_PLASTIC(connProp);

	if (synWtType == SYN_PLASTIC) {
		sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
		cpuRuntimeData.Npre_plastic[dest]++;
		// homeostasis
		if (grp_Info[destGrp].WithHomeostasis && grp_Info[destGrp].homeoId ==-1)
			grp_Info[destGrp].homeoId = dest; // this neuron info will be printed
	}

	cpuRuntimeData.Npre[dest] += 1;
	cpuRuntimeData.Npost[src] += 1;

	grp_Info2[srcGrp].numPostConn++;
	grp_Info2[destGrp].numPreConn++;

	if (cpuRuntimeData.Npost[src] > grp_Info2[srcGrp].maxPostConn)
		grp_Info2[srcGrp].maxPostConn = cpuRuntimeData.Npost[src];
	if (cpuRuntimeData.Npre[dest] > grp_Info2[destGrp].maxPreConn)
	grp_Info2[destGrp].maxPreConn = cpuRuntimeData.Npre[src];
}

void SNN::setGrpTimeSlice(int grpId, int timeSlice) {
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
int SNN::setRandSeed(int seed) {
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
void SNN::setupNetwork(bool removeTempMem) {
	if(!doneReorganization)
		reorganizeNetwork(removeTempMem);

	if((simMode_ == GPU_MODE) && (gpuRuntimeData.allocated == false))
		allocateSNN_GPU();
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
	net_Info.sim_in_testing = true;

	if (simMode_ == GPU_MODE) {
		// copy new network info struct to GPU (|TODO copy only a single boolean)
		copyNetworkInfo();
	}
}

// exits testing phase
void SNN::stopTesting() {
	sim_in_testing = false;
	net_Info.sim_in_testing = false;

	if (simMode_ == GPU_MODE) {
		// copy new network_info struct to GPU (|TODO copy only a single boolean)
		copyNetworkInfo();
	}
}


void SNN::swapConnections(int nid, int oldPos, int newPos) {
	unsigned int cumN=cpuRuntimeData.cumulativePost[nid];

	// Put the node oldPos to the top of the delay queue
	post_info_t tmp = cpuRuntimeData.postSynapticIds[cumN+oldPos];
	cpuRuntimeData.postSynapticIds[cumN+oldPos]= cpuRuntimeData.postSynapticIds[cumN+newPos];
	cpuRuntimeData.postSynapticIds[cumN+newPos]= tmp;

	// Ensure that you have shifted the delay accordingly....
	uint8_t tmp_delay = tmp_SynapticDelay[cumN+oldPos];
	tmp_SynapticDelay[cumN+oldPos] = tmp_SynapticDelay[cumN+newPos];
	tmp_SynapticDelay[cumN+newPos] = tmp_delay;

	// update the pre-information for the postsynaptic neuron at the position oldPos.
	post_info_t  postInfo = cpuRuntimeData.postSynapticIds[cumN+oldPos];
	int  post_nid = GET_CONN_NEURON_ID(postInfo);
	int  post_sid = GET_CONN_SYN_ID(postInfo);

	post_info_t* preId    = &(cpuRuntimeData.preSynapticIds[cpuRuntimeData.cumulativePre[post_nid]+post_sid]);
	int  pre_nid  = GET_CONN_NEURON_ID((*preId));
	int  pre_sid  = GET_CONN_SYN_ID((*preId));
	int  pre_gid  = GET_CONN_GRP_ID((*preId));
	assert (pre_nid == nid);
	assert (pre_sid == newPos);
	*preId = SET_CONN_ID( pre_nid, oldPos, pre_gid);

	// update the pre-information for the postsynaptic neuron at the position newPos
	postInfo = cpuRuntimeData.postSynapticIds[cumN+newPos];
	post_nid = GET_CONN_NEURON_ID(postInfo);
	post_sid = GET_CONN_SYN_ID(postInfo);

	preId    = &(cpuRuntimeData.preSynapticIds[cpuRuntimeData.cumulativePre[post_nid]+post_sid]);
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
	grpConnectInfo_t* connInfo = connectBegin;
	std::vector< std::vector<float> > wtConnId;

	// loop over all connections and find the ones with Connection Monitors
	while (connInfo) {
		if (connInfo->connId==connId) {
			int grpIdPre = connInfo->grpSrc;
			int grpIdPost = connInfo->grpDest;

			// init weight matrix with right dimensions
			for (int i=0; i<grp_Info[grpIdPre].SizeN; i++) {
				std::vector<float> wtSlice;
				for (int j=0; j<grp_Info[grpIdPost].SizeN; j++) {
					wtSlice.push_back(NAN);
				}
				wtConnId.push_back(wtSlice);
			}

			// copy the weights for a given post-group from device
			// \TODO: check if the weights for this grpIdPost have already been copied
			// \TODO: even better, but tricky because of ordering, make copyWeightState connection-based
			if (simMode_==GPU_MODE) {
				copyWeightState(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpIdPost);
			}

			for (int postId=grp_Info[grpIdPost].StartN; postId<=grp_Info[grpIdPost].EndN; postId++) {
				unsigned int pos_ij = cpuRuntimeData.cumulativePre[postId];
				for (int i=0; i<cpuRuntimeData.Npre[postId]; i++, pos_ij++) {
					// skip synapses that belong to a different connection ID
					if (cpuRuntimeData.cumConnIdPre[pos_ij]!=connInfo->connId)
						continue;

					// find pre-neuron ID and update ConnectionMonitor container
					int preId = GET_CONN_NEURON_ID(cpuRuntimeData.preSynapticIds[pos_ij]);
					wtConnId[preId-getGroupStartNeuronId(grpIdPre)][postId-getGroupStartNeuronId(grpIdPost)] =
						fabs(cpuRuntimeData.wt[pos_ij]);
				}
			}
			break;
		}
		connInfo = connInfo->next;
	}

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
		int monitorId = grp_Info[grpId].GroupMonitorId;

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
			copyGroupState(&cpuRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false);
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
			data = cpuRuntimeData.grpDABuffer[grpId][t];

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

void SNN::updateSpikeGenerators() {
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

void SNN::updateSpikeGeneratorsInit() {
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
	assert(cpuRuntimeData.spikeGenBits == NULL);

	if (NgenFunc) {
		cpuRuntimeData.spikeGenBits = new uint32_t[NgenFunc/32+1];
		// increase the total memory size used by the routine...
		cpuSnnSz.addInfoSize += sizeof(cpuRuntimeData.spikeGenBits[0])*(NgenFunc/32+1);
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
		KERNEL_ERROR("Insufficient amount of buffer allocated...");
		exitSimulation(1);
	}

	cpuRuntimeData.firingTableD2 = new unsigned int[maxSpikesD2];
	cpuRuntimeData.firingTableD1 = new unsigned int[maxSpikesD1];
	cpuSnnSz.spikingInfoSize += sizeof(int) * ((maxSpikesD2 + maxSpikesD1) + 2* (1000 + maxDelay_ + 1));

	return curD;
}

// This function is called every second by simulator...
// This function updates the firingTable by removing older firing values...
void SNN::updateFiringTable() {
	// Read the neuron ids that fired in the last maxDelay_ seconds
	// and put it to the beginning of the firing table...
	for(int p=timeTableD2[999],k=0;p<timeTableD2[999+maxDelay_+1];p++,k++) {
		cpuRuntimeData.firingTableD2[k]=cpuRuntimeData.firingTableD2[p];
	}

	for(int i=0; i < maxDelay_; i++) {
		timeTableD2[i+1] = timeTableD2[1000+i+1]-timeTableD2[1000];
	}

	timeTableD1[maxDelay_] = 0;

	/* the code of weight update has been moved to SNN::updateWeights() */

	spikeCountAllHost	+= spikeCountAll1secHost;
	spikeCountD2Host += (secD2fireCntHost-timeTableD2[maxDelay_]);
	spikeCountD1Host += secD1fireCntHost;

	secD1fireCntHost  = 0;
	spikeCountAll1secHost = 0;
	secD2fireCntHost = timeTableD2[maxDelay_];

	for (int i=0; i < numGrp; i++) {
		grp_Info[i].FiringCount1sec=0;
	}
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
			unsigned int* fireTablePtr = (k==0)?cpuRuntimeData.firingTableD2:cpuRuntimeData.firingTableD1;
			for(int t=numMsMin; t<numMsMax; t++) {
				for(int i=timeTablePtr[t+maxDelay_]; i<timeTablePtr[t+maxDelay_+1];i++) {
					// retrieve the neuron id
					int nid   = fireTablePtr[i];
					if (simMode_ == GPU_MODE)
						nid = GET_FIRING_TABLE_NID(nid);
					assert(nid < numN);

					// make sure neuron belongs to currently relevant group
					int this_grpId = cpuRuntimeData.grpIds[nid];
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
void SNN::updateWeights() {
	// at this point we have already checked for sim_in_testing and sim_with_fixedwts
	assert(sim_in_testing==false);
	assert(sim_with_fixedwts==false);

	// update synaptic weights here for all the neurons..
	for(int g = 0; g < numGrp; g++) {
		// no changable weights so continue without changing..
		if(grp_Info[g].FixedInputWts || !(grp_Info[g].WithSTDP))
			continue;

		for(int i = grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
			assert(i < numNReg);
			unsigned int offset = cpuRuntimeData.cumulativePre[i];
			float diff_firing = 0.0;
			float homeostasisScale = 1.0;

			if(grp_Info[g].WithHomeostasis) {
				assert(cpuRuntimeData.baseFiring[i]>0);
				diff_firing = 1-cpuRuntimeData.avgFiring[i]/cpuRuntimeData.baseFiring[i];
				homeostasisScale = grp_Info[g].homeostasisScale;
			}

			if (i==grp_Info[g].StartN)
				KERNEL_DEBUG("Weights, Change at %lu (diff_firing: %f)", simTimeSec, diff_firing);

			for(int j = 0; j < cpuRuntimeData.Npre_plastic[i]; j++) {
				//	if (i==grp_Info[g].StartN)
				//		KERNEL_DEBUG("%1.2f %1.2f \t", wt[offset+j]*10, wtChange[offset+j]*10);
				float effectiveWtChange = stdpScaleFactor_ * cpuRuntimeData.wtChange[offset + j];
//				if (wtChange[offset+j])
//					printf("connId=%d, wtChange[%d]=%f\n",cumConnIdPre[offset+j],offset+j,wtChange[offset+j]);

				// homeostatic weight update
				// FIXME: check WithESTDPtype and WithISTDPtype first and then do weight change update
				switch (grp_Info[g].WithESTDPtype) {
				case STANDARD:
					if (grp_Info[g].WithHomeostasis) {
						cpuRuntimeData.wt[offset+j] += (diff_firing*cpuRuntimeData.wt[offset+j]*homeostasisScale + cpuRuntimeData.wtChange[offset+j])*cpuRuntimeData.baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						// just STDP weight update
						cpuRuntimeData.wt[offset+j] += effectiveWtChange;
					}
					break;
				case DA_MOD:
					if (grp_Info[g].WithHomeostasis) {
						effectiveWtChange = cpuRuntimeData.grpDA[g] * effectiveWtChange;
						cpuRuntimeData.wt[offset+j] += (diff_firing*cpuRuntimeData.wt[offset+j]*homeostasisScale + effectiveWtChange)*cpuRuntimeData.baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						cpuRuntimeData.wt[offset+j] += cpuRuntimeData.grpDA[g] * effectiveWtChange;
					}
					break;
				case UNKNOWN_STDP:
				default:
					// we shouldn't even be in here if !WithSTDP
					break;
				}

				switch (grp_Info[g].WithISTDPtype) {
				case STANDARD:
					if (grp_Info[g].WithHomeostasis) {
						cpuRuntimeData.wt[offset+j] += (diff_firing*cpuRuntimeData.wt[offset+j]*homeostasisScale + cpuRuntimeData.wtChange[offset+j])*cpuRuntimeData.baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						// just STDP weight update
						cpuRuntimeData.wt[offset+j] += effectiveWtChange;
					}
					break;
				case DA_MOD:
					if (grp_Info[g].WithHomeostasis) {
						effectiveWtChange = cpuRuntimeData.grpDA[g] * effectiveWtChange;
						cpuRuntimeData.wt[offset+j] += (diff_firing*cpuRuntimeData.wt[offset+j]*homeostasisScale + effectiveWtChange)*cpuRuntimeData.baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
					} else {
						cpuRuntimeData.wt[offset+j] += cpuRuntimeData.grpDA[g] * effectiveWtChange;
					}
					break;
				case UNKNOWN_STDP:
				default:
					// we shouldn't even be in here if !WithSTDP
					break;
				}

				// It is users' choice to decay weight change or not
				// see setWeightAndWeightChangeUpdate()
				cpuRuntimeData.wtChange[offset+j] *= wtChangeDecay_;

				// if this is an excitatory or inhibitory synapse
				if (cpuRuntimeData.maxSynWt[offset + j] >= 0) {
					if (cpuRuntimeData.wt[offset + j] >= cpuRuntimeData.maxSynWt[offset + j])
						cpuRuntimeData.wt[offset + j] = cpuRuntimeData.maxSynWt[offset + j];
					if (cpuRuntimeData.wt[offset + j] < 0)
						cpuRuntimeData.wt[offset + j] = 0.0;
				} else {
					if (cpuRuntimeData.wt[offset + j] <= cpuRuntimeData.maxSynWt[offset + j])
						cpuRuntimeData.wt[offset + j] = cpuRuntimeData.maxSynWt[offset + j];
					if (cpuRuntimeData.wt[offset+j] > 0)
						cpuRuntimeData.wt[offset+j] = 0.0;
				}
			}
		}
	}
}
