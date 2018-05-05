/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
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
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#include <snn.h>
#include <sstream>
#include <algorithm>

#include <connection_monitor.h>
#include <connection_monitor_core.h>
#include <spike_monitor.h>
#include <spike_monitor_core.h>
#include <group_monitor.h>
#include <group_monitor_core.h>
#include <neuron_monitor.h>
#include <neuron_monitor_core.h>

#include <spike_buffer.h>
#include <error_code.h>

// \FIXME what are the following for? why were they all the way at the bottom of this file?

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///


// TODO: consider moving unsafe computations out of constructor
SNN::SNN(const std::string& name, SimMode preferredSimMode, LoggerMode loggerMode, int randSeed)
					: networkName_(name), preferredSimMode_(preferredSimMode), loggerMode_(loggerMode),
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
						uint8_t minDelay, uint8_t maxDelay, RadiusRF radius,
						float _mulSynFast, float _mulSynSlow, bool synWtType) {
						//const std::string& wtType
	int retId=-1;
	assert(grpId1 < numGroups);
	assert(grpId2 < numGroups);
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

	connConfig.grpSrc   = grpId1;
	connConfig.grpDest  = grpId2;
	connConfig.initWt   = initWt;
	connConfig.maxWt    = maxWt;
	connConfig.maxDelay = maxDelay;
	connConfig.minDelay = minDelay;
//		newInfo->radX             = (radX<0) ? MAX(szPre.x,szPost.x) : radX; // <0 means full connectivity, so the
//		newInfo->radY             = (radY<0) ? MAX(szPre.y,szPost.y) : radY; // effective group size is Grid3D.x. Grab
//		newInfo->radZ             = (radZ<0) ? MAX(szPre.z,szPost.z) : radZ; // the larger of pre / post to connect all
	connConfig.connRadius = radius;
	connConfig.mulSynFast      = _mulSynFast;
	connConfig.mulSynSlow      = _mulSynSlow;
	connConfig.connProp        = connProp;
	connConfig.connProbability = prob;
	connConfig.type            = CONN_UNKNOWN;
	connConfig.connectionMonitorId = -1;
	connConfig.connId = -1;
	connConfig.conn = NULL;
	connConfig.numberOfConnections = 0;

	if ( _type.find("random") != std::string::npos) {
		connConfig.type = CONN_RANDOM;
	}
	//so you're setting the size to be prob*Number of synapses in group info + some standard deviation ...
	else if ( _type.find("full-no-direct") != std::string::npos) {
		connConfig.type	= CONN_FULL_NO_DIRECT;
	}
	else if ( _type.find("full") != std::string::npos) {
		connConfig.type = CONN_FULL;
	}
	else if ( _type.find("one-to-one") != std::string::npos) {
		connConfig.type = CONN_ONE_TO_ONE;
	} else if ( _type.find("gaussian") != std::string::npos) {
		connConfig.type   = CONN_GAUSSIAN;
	} else {
		KERNEL_ERROR("Invalid connection type (should be 'random', 'full', 'one-to-one', 'full-no-direct', or 'gaussian')");
		exitSimulation(-1);
	}

	// assign connection id
	assert(connConfig.connId == -1);
	connConfig.connId = numConnections;

	KERNEL_DEBUG("CONNECT SETUP: connId=%d, mulFast=%f, mulSlow=%f", connConfig.connId, connConfig.mulSynFast, connConfig.mulSynSlow);

	// store the configuration of a connection
	connectConfigMap[numConnections] = connConfig; // connConfig.connId == numConnections
	
	assert(numConnections < MAX_CONN_PER_SNN);	// make sure we don't overflow connId
	numConnections++;
	
	return (numConnections - 1);
}

// make custom connections from grpId1 to grpId2
short int SNN::connect(int grpId1, int grpId2, ConnectionGeneratorCore* conn, float _mulSynFast, float _mulSynSlow,
						bool synWtType) {
	int retId=-1;

	assert(grpId1 < numGroups);
	assert(grpId2 < numGroups);

	// initialize the configuration of a connection
	ConnectConfig connConfig;

	connConfig.grpSrc   = grpId1;
	connConfig.grpDest  = grpId2;
	connConfig.initWt	  = 0.0f;
	connConfig.maxWt	  = 0.0f;
	connConfig.maxDelay = MAX_SYN_DELAY;
	connConfig.minDelay = 1;
	connConfig.mulSynFast = _mulSynFast;
	connConfig.mulSynSlow = _mulSynSlow;
	connConfig.connProp = SET_CONN_PRESENT(1) | SET_FIXED_PLASTIC(synWtType);
	connConfig.type = CONN_USER_DEFINED;
	connConfig.conn = conn;
	connConfig.connectionMonitorId = -1;
	connConfig.connId = -1;
	connConfig.numberOfConnections = 0;

	// assign a connection id
	assert(connConfig.connId == -1);
	connConfig.connId = numConnections;

	// store the configuration of a connection
	connectConfigMap[numConnections] = connConfig; // connConfig.connId == numConnections

	assert(numConnections < MAX_CONN_PER_SNN);	// make sure we don't overflow connId
	numConnections++;

	return (numConnections - 1);
}

// make a compartmental connection between two groups
short int SNN::connectCompartments(int grpIdLower, int grpIdUpper) {
	assert(grpIdLower >= 0 && grpIdLower < numGroups);
	assert(grpIdUpper >= 0 && grpIdUpper < numGroups);
	assert(grpIdLower != grpIdUpper);
	assert(!isPoissonGroup(grpIdLower));
	assert(!isPoissonGroup(grpIdUpper));

	// the two groups must be located on the same partition
	assert(groupConfigMap[grpIdLower].preferredNetId == groupConfigMap[grpIdUpper].preferredNetId);

	// this flag must be set if any compartmental connections exist
	// note that grpId.withCompartments is not necessarily set just yet, this will be done in
	// CpuSNN::setCompartmentParameters
	sim_with_compartments = true;

	compConnectConfig compConnConfig;

	compConnConfig.grpSrc = grpIdLower;
	compConnConfig.grpDest = grpIdUpper;
	compConnConfig.connId = -1;

	// assign a connection id
	assert(compConnConfig.connId == -1);
	compConnConfig.connId = numCompartmentConnections;

	// store the configuration of a connection
	compConnectConfigMap[numCompartmentConnections] = compConnConfig;

	numCompartmentConnections++;

	return (numCompartmentConnections - 1);
}

// create group of Izhikevich neurons
// use int for nNeur to avoid arithmetic underflow
int SNN::createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	assert(grid.numX * grid.numY * grid.numZ > 0);
	assert(neurType >= 0);
	assert(numGroups < MAX_GRP_PER_SNN);

	if ( (!(neurType & TARGET_AMPA) && !(neurType & TARGET_NMDA) &&
		  !(neurType & TARGET_GABAa) && !(neurType & TARGET_GABAb)) || (neurType & POISSON_NEURON)) {
		KERNEL_ERROR("Invalid type using createGroup... Cannot create poisson generators here.");
		exitSimulation(1);
	}

	// initialize group configuration
	GroupConfig grpConfig;
	GroupConfigMD grpConfigMD;

	//All groups are non-compartmental by default
	grpConfig.withCompartments = false;
	
	// init parameters of neural group size and location
	grpConfig.grpName = grpName;
	grpConfig.type = neurType;
	grpConfig.numN = grid.N;
	
	grpConfig.isSpikeGenerator = false;
	grpConfig.grid = grid;
	grpConfig.isLIF = false;

	if (preferredPartition == ANY) {
		grpConfig.preferredNetId = ANY;
	} else if (preferredBackend == CPU_CORES) {
		grpConfig.preferredNetId = preferredPartition + CPU_RUNTIME_BASE;
	} else {
		grpConfig.preferredNetId = preferredPartition + GPU_RUNTIME_BASE;
	}

	// assign a global group id
	grpConfigMD.gGrpId = numGroups;

	// store the configuration of a group
	groupConfigMap[numGroups] = grpConfig; // numGroups == grpId
	groupConfigMDMap[numGroups] = grpConfigMD;

	assert(numGroups < MAX_GRP_PER_SNN); // make sure we don't overflow connId
	numGroups++;

	return grpConfigMD.gGrpId;
}

// create group of LIF neurons
// use int for nNeur to avoid arithmetic underflow
int SNN::createGroupLIF(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	assert(grid.numX * grid.numY * grid.numZ > 0);
	assert(neurType >= 0);
	assert(numGroups < MAX_GRP_PER_SNN);

	if ( (!(neurType & TARGET_AMPA) && !(neurType & TARGET_NMDA) &&
		  !(neurType & TARGET_GABAa) && !(neurType & TARGET_GABAb)) || (neurType & POISSON_NEURON)) {
		KERNEL_ERROR("Invalid type using createGroup... Cannot create poisson generators here.");
		exitSimulation(1);
	}

	// initialize group configuration
	GroupConfig grpConfig;
	GroupConfigMD grpConfigMD;
	
	// init parameters of neural group size and location
	grpConfig.grpName = grpName;
	grpConfig.type = neurType;
	grpConfig.numN = grid.N;
	
	grpConfig.isLIF = true;
	grpConfig.isSpikeGenerator = false;
	grpConfig.grid = grid;

	if (preferredPartition == ANY) {
		grpConfig.preferredNetId = ANY;
	} else if (preferredBackend == CPU_CORES) {
		grpConfig.preferredNetId = preferredPartition + CPU_RUNTIME_BASE;
	} else {
		grpConfig.preferredNetId = preferredPartition + GPU_RUNTIME_BASE;
	}

	// assign a global group id
	grpConfigMD.gGrpId = numGroups;

	// store the configuration of a group
	groupConfigMap[numGroups] = grpConfig; // numGroups == grpId
	groupConfigMDMap[numGroups] = grpConfigMD;

	assert(numGroups < MAX_GRP_PER_SNN); // make sure we don't overflow connId
	numGroups++;

	return grpConfigMD.gGrpId;
}

// create spike generator group
// use int for nNeur to avoid arithmetic underflow
int SNN::createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition, ComputingBackend preferredBackend) {
	assert(grid.numX * grid.numY * grid.numZ > 0);
	assert(neurType >= 0);
	assert(numGroups < MAX_GRP_PER_SNN);

	// initialize group configuration
	GroupConfig grpConfig;
	GroupConfigMD grpConfigMD;

	//All groups are non-compartmental by default  FIXME:IS THIS NECESSARY?
	grpConfig.withCompartments = false;

	// init parameters of neural group size and location
	grpConfig.grpName = grpName;
	grpConfig.type = neurType | POISSON_NEURON;
	grpConfig.numN = grid.N;
	grpConfig.isSpikeGenerator = true;
	grpConfig.grid = grid;
	grpConfig.isLIF = false;

	if (preferredPartition == ANY) {
		grpConfig.preferredNetId = ANY;
	}
	else if (preferredBackend == CPU_CORES) {
		grpConfig.preferredNetId = preferredPartition + CPU_RUNTIME_BASE;
	}
	else {
		grpConfig.preferredNetId = preferredPartition + GPU_RUNTIME_BASE;
	}

	// assign a global group id
	grpConfigMD.gGrpId = numGroups;

	// store the configuration of a group
	groupConfigMap[numGroups] = grpConfig;
	groupConfigMDMap[numGroups] = grpConfigMD;

	assert(numGroups < MAX_GRP_PER_SNN); // make sure we don't overflow connId
	numGroups++;
	numSpikeGenGrps++;

	return grpConfigMD.gGrpId;
}

void SNN::setCompartmentParameters(int gGrpId, float couplingUp, float couplingDown) {
	if (gGrpId == ALL) { 
		for (int grpId = 0; grpId<numGroups; grpId++) {
			setCompartmentParameters(grpId, couplingUp, couplingDown);
		}
	}
	else {
		groupConfigMap[gGrpId].withCompartments = true;
		groupConfigMap[gGrpId].compCouplingUp = couplingUp;
		groupConfigMap[gGrpId].compCouplingDown = couplingDown;
		glbNetworkConfig.numComp += groupConfigMap[gGrpId].numN;
	}
}


// set conductance values for a simulation (custom values or disable conductances alltogether)
void SNN::setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb) {
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
void SNN::setHomeostasis(int gGrpId, bool isSet, float homeoScale, float avgTimeScale) {
	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setHomeostasis(grpId, isSet, homeoScale, avgTimeScale);
		}
	} else {
		// set conductances for a given group
		sim_with_homeostasis					  |= isSet;
		groupConfigMap[gGrpId].homeoConfig.WithHomeostasis    = isSet;
		groupConfigMap[gGrpId].homeoConfig.homeostasisScale   = homeoScale;
		groupConfigMap[gGrpId].homeoConfig.avgTimeScale       = avgTimeScale;
		groupConfigMap[gGrpId].homeoConfig.avgTimeScaleInv    = 1.0f / avgTimeScale;
		groupConfigMap[gGrpId].homeoConfig.avgTimeScaleDecay = (avgTimeScale * 1000.0f - 1.0f) / (avgTimeScale * 1000.0f);

		KERNEL_INFO("Homeostasis parameters %s for %d (%s):\thomeoScale: %f, avgTimeScale: %f",
					isSet?"enabled":"disabled", gGrpId, groupConfigMap[gGrpId].grpName.c_str(), homeoScale, avgTimeScale);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void SNN::setHomeoBaseFiringRate(int gGrpId, float baseFiring, float baseFiringSD) {
	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD);
		}
	} else {
		// set homeostatsis for a given group
		groupConfigMap[gGrpId].homeoConfig.baseFiring = baseFiring;
		groupConfigMap[gGrpId].homeoConfig.baseFiringSD = baseFiringSD;

		KERNEL_INFO("Homeostatic base firing rate set for %d (%s):\tbaseFiring: %3.3f, baseFiringStd: %3.3f",
							gGrpId, groupConfigMap[gGrpId].grpName.c_str(), baseFiring, baseFiringSD);
	}
}


void SNN::setIntegrationMethod(integrationMethod_t method, int numStepsPerMs) {
	assert(numStepsPerMs >= 1 && numStepsPerMs <= 100);
	glbNetworkConfig.simIntegrationMethod = method;
	glbNetworkConfig.simNumStepsPerMs = numStepsPerMs;
	glbNetworkConfig.timeStep = 1.0f / numStepsPerMs;
}

// set Izhikevich parameters for group
void SNN::setNeuronParameters(int gGrpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
								float izh_c, float izh_c_sd, float izh_d, float izh_d_sd)
{
	assert(gGrpId >= -1);
	assert(izh_a_sd >= 0); assert(izh_b_sd >= 0); assert(izh_c_sd >= 0); assert(izh_d_sd >= 0);

	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setNeuronParameters(grpId, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd);
		}
	} else {
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a = izh_a;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a_sd = izh_a_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b = izh_b;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b_sd = izh_b_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c = izh_c;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c_sd = izh_c_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d = izh_d;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d_sd = izh_d_sd;
		groupConfigMap[gGrpId].withParamModel_9 = 0;
		groupConfigMap[gGrpId].isLIF = 0;
	}
}

// set (9) Izhikevich parameters for group
void SNN::setNeuronParameters(int gGrpId, float izh_C, float izh_C_sd, float izh_k, float izh_k_sd,
	float izh_vr, float izh_vr_sd, float izh_vt, float izh_vt_sd,
	float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
	float izh_vpeak, float izh_vpeak_sd, float izh_c, float izh_c_sd,
	float izh_d, float izh_d_sd)
{
	assert(gGrpId >= -1);
	assert(izh_C_sd >= 0); assert(izh_k_sd >= 0); assert(izh_vr_sd >= 0);
	assert(izh_vt_sd >= 0); assert(izh_a_sd >= 0); assert(izh_b_sd >= 0); assert(izh_vpeak_sd >= 0);
	assert(izh_c_sd >= 0); assert(izh_d_sd >= 0);

	if (gGrpId == ALL) { // shortcut for all groups
		for (int grpId = 0; grpId<numGroups; grpId++) {
			setNeuronParameters(grpId, izh_C, izh_C_sd, izh_k, izh_k_sd, izh_vr, izh_vr_sd, izh_vt, izh_vt_sd,
				izh_a, izh_a_sd, izh_b, izh_b_sd, izh_vpeak, izh_vpeak_sd, izh_c, izh_c_sd,
				izh_d, izh_d_sd);
		}
	}
	else {
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a = izh_a;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a_sd = izh_a_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b = izh_b;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b_sd = izh_b_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c = izh_c;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c_sd = izh_c_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d = izh_d;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d_sd = izh_d_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_C = izh_C;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_C_sd = izh_C_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_k = izh_k;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_k_sd = izh_k_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vr = izh_vr;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vr_sd = izh_vr_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vt = izh_vt;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vt_sd = izh_vt_sd;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vpeak = izh_vpeak;
		groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vpeak_sd = izh_vpeak_sd;
		groupConfigMap[gGrpId].withParamModel_9 = 1;
		groupConfigMap[gGrpId].isLIF = 0;
		KERNEL_INFO("Set a nine parameter group!");
	}
}


// set LIF parameters for the group
void SNN::setNeuronParametersLIF(int gGrpId, int tau_m, int tau_ref, float vTh, float vReset, double minRmem, double maxRmem)
{
	assert(gGrpId >= -1);
	assert(tau_m >= 0); assert(tau_ref >= 0); assert(vReset < vTh);
	assert(minRmem >= 0.0f); assert(minRmem <= maxRmem);

	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setNeuronParametersLIF(grpId, tau_m, tau_ref, vTh, vReset, minRmem, maxRmem);
		}
	} else {
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_tau_m = tau_m;
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_tau_ref = tau_ref;
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_vTh = vTh;
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_vReset = vReset;
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_minRmem = minRmem;
		groupConfigMap[gGrpId].neuralDynamicsConfig.lif_maxRmem = maxRmem;
		groupConfigMap[gGrpId].withParamModel_9 = 0;
		groupConfigMap[gGrpId].isLIF = 1;
	}
}

void SNN::setNeuromodulator(int gGrpId, float baseDP, float tauDP, float base5HT, float tau5HT, float baseACh,
	float tauACh, float baseNE, float tauNE) {

	assert(gGrpId >= -1);
	assert(baseDP > 0.0f); assert(base5HT > 0.0f); assert(baseACh > 0.0f); assert(baseNE > 0.0f);
	assert(tauDP > 0); assert(tau5HT > 0); assert(tauACh > 0); assert(tauNE > 0);

	if (gGrpId == ALL) { // shortcut for all groups
		for (int grpId = 0; grpId < numGroups; grpId++) {
			setNeuromodulator(grpId, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE);
		}
	} else {
		groupConfigMap[gGrpId].neuromodulatorConfig.baseDP = baseDP;
		groupConfigMap[gGrpId].neuromodulatorConfig.decayDP = 1.0f - (1.0f / tauDP);
		groupConfigMap[gGrpId].neuromodulatorConfig.base5HT = base5HT;
		groupConfigMap[gGrpId].neuromodulatorConfig.decay5HT = 1.0f - (1.0f / tau5HT);
		groupConfigMap[gGrpId].neuromodulatorConfig.baseACh = baseACh;
		groupConfigMap[gGrpId].neuromodulatorConfig.decayACh = 1.0f - (1.0f / tauACh);
		groupConfigMap[gGrpId].neuromodulatorConfig.baseNE = baseNE;
		groupConfigMap[gGrpId].neuromodulatorConfig.decayNE = 1.0f - (1.0f / tauNE);
	}
}

// set ESTDP params
void SNN::setESTDP(int gGrpId, bool isSet, STDPType type, STDPCurve curve, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus, float gamma) {
	assert(gGrpId >= -1);
	if (isSet) {
		assert(type!=UNKNOWN_STDP);
		assert(tauPlus > 0.0f); assert(tauMinus > 0.0f); assert(gamma >= 0.0f);
	}

	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setESTDP(grpId, isSet, type, curve, alphaPlus, tauPlus, alphaMinus, tauMinus, gamma);
		}
	} else {
		// set STDP for a given group
		// set params for STDP curve
		groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_EXC 	= alphaPlus;
		groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_EXC 	= alphaMinus;
		groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_EXC 	= 1.0f / tauPlus;
		groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_EXC	= 1.0f / tauMinus;
		groupConfigMap[gGrpId].stdpConfig.GAMMA				= gamma;
		groupConfigMap[gGrpId].stdpConfig.KAPPA				= (1 + exp(-gamma / tauPlus)) / (1 - exp(-gamma / tauPlus));
		groupConfigMap[gGrpId].stdpConfig.OMEGA				= alphaPlus * (1 - groupConfigMap[gGrpId].stdpConfig.KAPPA);
		// set flags for STDP function
		groupConfigMap[gGrpId].stdpConfig.WithESTDPtype	= type;
		groupConfigMap[gGrpId].stdpConfig.WithESTDPcurve = curve;
		groupConfigMap[gGrpId].stdpConfig.WithESTDP		= isSet;
		groupConfigMap[gGrpId].stdpConfig.WithSTDP		|= groupConfigMap[gGrpId].stdpConfig.WithESTDP;
		sim_with_stdp									|= groupConfigMap[gGrpId].stdpConfig.WithSTDP;

		KERNEL_INFO("E-STDP %s for %s(%d)", isSet?"enabled":"disabled", groupConfigMap[gGrpId].grpName.c_str(), gGrpId);
	}
}

// set ISTDP params
void SNN::setISTDP(int gGrpId, bool isSet, STDPType type, STDPCurve curve, float ab1, float ab2, float tau1, float tau2) {
	assert(gGrpId >= -1);
	if (isSet) {
		assert(type != UNKNOWN_STDP);
		assert(tau1 > 0); assert(tau2 > 0);
	}

	if (gGrpId==ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setISTDP(grpId, isSet, type, curve, ab1, ab2, tau1, tau2);
		}
	} else {
		// set STDP for a given group
		// set params for STDP curve
		if (curve == EXP_CURVE) {
			groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_INB = ab1;
			groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_INB = ab2;
			groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_INB = 1.0f / tau1;
			groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_INB = 1.0f / tau2;
			groupConfigMap[gGrpId].stdpConfig.BETA_LTP 		= 0.0f;
			groupConfigMap[gGrpId].stdpConfig.BETA_LTD 		= 0.0f;
			groupConfigMap[gGrpId].stdpConfig.LAMBDA			= 1.0f;
			groupConfigMap[gGrpId].stdpConfig.DELTA			= 1.0f;
		} else {
			groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_INB = 0.0f;
			groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_INB = 0.0f;
			groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_INB = 1.0f;
			groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_INB = 1.0f;
			groupConfigMap[gGrpId].stdpConfig.BETA_LTP 		= ab1;
			groupConfigMap[gGrpId].stdpConfig.BETA_LTD 		= ab2;
			groupConfigMap[gGrpId].stdpConfig.LAMBDA			= tau1;
			groupConfigMap[gGrpId].stdpConfig.DELTA			= tau2;
		}
		// set flags for STDP function
		//FIXME: separate STDPType to ESTDPType and ISTDPType
		groupConfigMap[gGrpId].stdpConfig.WithISTDPtype	= type;
		groupConfigMap[gGrpId].stdpConfig.WithISTDPcurve = curve;
		groupConfigMap[gGrpId].stdpConfig.WithISTDP		= isSet;
		groupConfigMap[gGrpId].stdpConfig.WithSTDP		|= groupConfigMap[gGrpId].stdpConfig.WithISTDP;
		sim_with_stdp					|= groupConfigMap[gGrpId].stdpConfig.WithSTDP;

		KERNEL_INFO("I-STDP %s for %s(%d)", isSet?"enabled":"disabled", groupConfigMap[gGrpId].grpName.c_str(), gGrpId);
	}
}

// set STP params
void SNN::setSTP(int gGrpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x) {
	assert(gGrpId >= -1);
	if (isSet) {
		assert(STP_U > 0 && STP_U <= 1); assert(STP_tau_u > 0); assert(STP_tau_x > 0);
	}

	if (gGrpId == ALL) { // shortcut for all groups
		for(int grpId = 0; grpId < numGroups; grpId++) {
			setSTP(grpId, isSet, STP_U, STP_tau_u, STP_tau_x);
		}
	} else {
		// set STDP for a given group
		sim_with_stp									|= isSet;
		groupConfigMap[gGrpId].stpConfig.WithSTP		= isSet;
		groupConfigMap[gGrpId].stpConfig.STP_A			= (STP_U > 0.0f) ? 1.0 / STP_U : 1.0f; // scaling factor
		groupConfigMap[gGrpId].stpConfig.STP_U 			= STP_U;
		groupConfigMap[gGrpId].stpConfig.STP_tau_u_inv	= 1.0f / STP_tau_u; // facilitatory
		groupConfigMap[gGrpId].stpConfig.STP_tau_x_inv	= 1.0f / STP_tau_x; // depressive

		KERNEL_INFO("STP %s for %d (%s):\tA: %1.4f, U: %1.4f, tau_u: %4.0f, tau_x: %4.0f", isSet?"enabled":"disabled",
					gGrpId, groupConfigMap[gGrpId].grpName.c_str(), groupConfigMap[gGrpId].stpConfig.STP_A, STP_U, STP_tau_u, STP_tau_x);
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
void SNN::setupNetwork() {
	switch (snnState) {
	case CONFIG_SNN:
		compileSNN();
	case COMPILED_SNN:
		partitionSNN();
	case PARTITIONED_SNN:
		generateRuntimeSNN();
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

int SNN::runNetwork(int _nsec, int _nmsec, bool printRunSummary) {
	assert(_nmsec >= 0 && _nmsec < 1000);
	assert(_nsec  >= 0);
	int runDurationMs = _nsec*1000 + _nmsec;
	KERNEL_DEBUG("runNetwork: runDur=%dms, printRunSummary=%s", runDurationMs, printRunSummary?"y":"n");

	// setupNetwork() must have already been called
	assert(snnState == EXECUTABLE_SNN);

	// don't bother printing if logger mode is SILENT
	printRunSummary = (loggerMode_==SILENT) ? false : printRunSummary;

	// first-time run: inform the user the simulation is running now
	if (simTime==0 && printRunSummary) {
		KERNEL_INFO("");
		KERNEL_INFO("******************** Running the simulation on %d GPU(s) and %d CPU(s) ***************************", numGPUs, numCores);
		KERNEL_INFO("");
	}

	// reset all spike counters
	resetSpikeCnt(ALL);

	// store current start time for future reference
	simTimeRunStart = simTime;
	simTimeRunStop  = simTime + runDurationMs;
	assert(simTimeRunStop >= simTimeRunStart); // check for arithmetic underflow

	// ConnectionMonitor is a special case: we might want the first snapshot at t=0 in the binary
	// but updateTime() is false for simTime==0.
	// And we cannot put this code in ConnectionMonitorCore::init, because then the user would have no
	// way to call ConnectionMonitor::setUpdateTimeIntervalSec before...
	if (simTime == 0 && numConnectionMonitor) {
		updateConnectionMonitor();
	}

	// set the Poisson generation time slice to be at the run duration up to MAX_TIME_SLICE
	setGrpTimeSlice(ALL, std::max(1, std::min(runDurationMs, MAX_TIME_SLICE)));

#ifndef __NO_CUDA__
	CUDA_RESET_TIMER(timer);
	CUDA_START_TIMER(timer);
#endif

	//KERNEL_INFO("Reached the advSimStep loop!");

	// if nsec=0, simTimeMs=10, we need to run the simulator for 10 timeStep;
	// if nsec=1, simTimeMs=10, we need to run the simulator for 1*1000+10, time Step;
	for(int i = 0; i < runDurationMs; i++) {
		advSimStep();
		//KERNEL_INFO("Executed an advSimStep!");

		// update weight every updateInterval ms if plastic synapses present
		if (!sim_with_fixedwts && wtANDwtChangeUpdateInterval_ == ++wtANDwtChangeUpdateIntervalCnt_) {
			wtANDwtChangeUpdateIntervalCnt_ = 0; // reset counter
			if (!sim_in_testing) {
				// keep this if statement separate from the above, so that the counter is updated correctly
				updateWeights();
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
			if (numNeuronMonitor) {
				updateNeuronMonitor();
			}
			
			shiftSpikeTables();
		}

		fetchNeuronSpikeCount(ALL);
	}

	//KERNEL_INFO("Updated monitors!");

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
#ifndef __NO_CUDA__
	CUDA_STOP_TIMER(timer);
	lastExecutionTime = CUDA_GET_TIMER_VALUE(timer);
	cumExecutionTime += lastExecutionTime;
#endif
	return 0;
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: INTERACTING WITH A SIMULATION
/// ************************************************************************************************************ ///

// adds a bias to every weight in the connection
void SNN::biasWeights(short int connId, float bias, bool updateWeightRange) {
	assert(connId>=0 && connId<numConnections);

	int netId = groupConfigMDMap[connectConfigMap[connId].grpDest].netId;
	int lGrpId = groupConfigMDMap[connectConfigMap[connId].grpDest].lGrpId;

	fetchPreConnectionInfo(netId);
	fetchConnIdsLookupArray(netId);
	fetchSynapseState(netId);
	// iterate over all postsynaptic neurons
	for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++) {
		unsigned int cumIdx = managerRuntimeData.cumulativePre[lNId];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j = 0; j < managerRuntimeData.Npre[lNId]; pos_ij++, j++) {
			if (managerRuntimeData.connIdsPreIdx[pos_ij] == connId) {
				// apply bias to weight
				float weight = managerRuntimeData.wt[pos_ij] + bias;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight+bias>connInfo->maxWt || weight+bias<connInfo->minWt);
				bool needToPrintDebug = (weight > connectConfigMap[connId].maxWt || weight < 0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connectConfigMap[connId].maxWt = std::max(connectConfigMap[connId].maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), 0.0f, connectConfigMap[connId].maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = std::min(weight, connectConfigMap[connId].maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = std::max(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("biasWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, bias,
							(updateWeightRange?"true":"false"), weight, 0.0f, connectConfigMap[connId].maxWt);
					}
				}

				// update datastructures
				managerRuntimeData.wt[pos_ij] = weight;
				managerRuntimeData.maxSynWt[pos_ij] = connectConfigMap[connId].maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (netId < CPU_RUNTIME_BASE) {
#ifndef __NO_CUDA__
			CUDA_CHECK_ERRORS( cudaMemcpy(&(runtimeData[netId].wt[cumIdx]), &(managerRuntimeData.wt[cumIdx]), sizeof(float)*managerRuntimeData.Npre[lNId],
				cudaMemcpyHostToDevice) );

			if (runtimeData[netId].maxSynWt != NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU runtime
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(runtimeData[netId].maxSynWt[cumIdx]), &(managerRuntimeData.maxSynWt[cumIdx]),
					sizeof(float) * managerRuntimeData.Npre[lNId], cudaMemcpyHostToDevice) );
			}
#else
			assert(false);
#endif
		} else {
			memcpy(&runtimeData[netId].wt[cumIdx], &managerRuntimeData.wt[cumIdx], sizeof(float) * managerRuntimeData.Npre[lNId]);

			if (runtimeData[netId].maxSynWt != NULL) {
				// only copy maxSynWt if datastructure actually exists on the CPU runtime
				// (that logic should be done elsewhere though)
				memcpy(&runtimeData[netId].maxSynWt[cumIdx], &managerRuntimeData.maxSynWt[cumIdx], sizeof(float) * managerRuntimeData.Npre[lNId]);
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

// multiplies every weight with a scaling factor
void SNN::scaleWeights(short int connId, float scale, bool updateWeightRange) {
	assert(connId>=0 && connId<numConnections);
	assert(scale>=0.0f);

	int netId = groupConfigMDMap[connectConfigMap[connId].grpDest].netId;
	int lGrpId = groupConfigMDMap[connectConfigMap[connId].grpDest].lGrpId;

	fetchPreConnectionInfo(netId);
	fetchConnIdsLookupArray(netId);
	fetchSynapseState(netId);

	// iterate over all postsynaptic neurons
	for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++) {
		unsigned int cumIdx = managerRuntimeData.cumulativePre[lNId];

		// iterate over all presynaptic neurons
		unsigned int pos_ij = cumIdx;
		for (int j = 0; j < managerRuntimeData.Npre[lNId]; pos_ij++, j++) {
			if (managerRuntimeData.connIdsPreIdx[pos_ij]==connId) {
				// apply bias to weight
				float weight = managerRuntimeData.wt[pos_ij] * scale;

				// inform user of acton taken if weight is out of bounds
//				bool needToPrintDebug = (weight>connInfo->maxWt || weight<connInfo->minWt);
				bool needToPrintDebug = (weight > connectConfigMap[connId].maxWt || weight < 0.0f);

				if (updateWeightRange) {
					// if this flag is set, we need to update minWt,maxWt accordingly
					// will be saving new maxSynWt and copying to GPU below
//					connInfo->minWt = fmin(connInfo->minWt, weight);
					connectConfigMap[connId].maxWt = std::max(connectConfigMap[connId].maxWt, weight);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): updated weight ranges to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), 0.0f, connectConfigMap[connId].maxWt);
					}
				} else {
					// constrain weight to boundary values
					// compared to above, we swap minWt/maxWt logic
					weight = std::min(weight, connectConfigMap[connId].maxWt);
//					weight = fmax(weight, connInfo->minWt);
					weight = std::max(weight, 0.0f);
					if (needToPrintDebug) {
						KERNEL_DEBUG("scaleWeights(%d,%f,%s): constrained weight %f to [%f,%f]", connId, scale,
							(updateWeightRange?"true":"false"), weight, 0.0f, connectConfigMap[connId].maxWt);
					}
				}

				// update datastructures
				managerRuntimeData.wt[pos_ij] = weight;
				managerRuntimeData.maxSynWt[pos_ij] = connectConfigMap[connId].maxWt; // it's easier to just update, even if it hasn't changed
			}
		}

		// update GPU datastructures in batches, grouped by post-neuron
		if (netId < CPU_RUNTIME_BASE) {
#ifndef __NO_CUDA__
			CUDA_CHECK_ERRORS(cudaMemcpy(&runtimeData[netId].wt[cumIdx], &managerRuntimeData.wt[cumIdx], sizeof(float)*managerRuntimeData.Npre[lNId],
				cudaMemcpyHostToDevice));

			if (runtimeData[netId].maxSynWt != NULL) {
				// only copy maxSynWt if datastructure actually exists on the GPU runtime
				// (that logic should be done elsewhere though)
				CUDA_CHECK_ERRORS(cudaMemcpy(&runtimeData[netId].maxSynWt[cumIdx], &managerRuntimeData.maxSynWt[cumIdx],
					sizeof(float) * managerRuntimeData.Npre[lNId], cudaMemcpyHostToDevice));
			}
#else
			assert(false);
#endif
		} else {
			memcpy(&runtimeData[netId].wt[cumIdx], &managerRuntimeData.wt[cumIdx], sizeof(float) * managerRuntimeData.Npre[lNId]);

			if (runtimeData[netId].maxSynWt != NULL) {
				// only copy maxSynWt if datastructure actually exists on the CPU runtime
				// (that logic should be done elsewhere though)
				memcpy(&runtimeData[netId].maxSynWt[cumIdx], &managerRuntimeData.maxSynWt[cumIdx], sizeof(float) * managerRuntimeData.Npre[lNId]);
			}
		}
	}
}

// FIXME: distinguish the function call at CONFIG_STATE and SETUP_STATE, where groupConfigs[0][] might not be available
// or groupConfigMap is not sync with groupConfigs[0][]
GroupMonitor* SNN::setGroupMonitor(int gGrpId, FILE* fid) {
	int netId = groupConfigMDMap[gGrpId].netId;
	int lGrpId = groupConfigMDMap[gGrpId].lGrpId;

	// check whether group already has a GroupMonitor
	if (groupConfigMDMap[gGrpId].groupMonitorId >= 0) {
		KERNEL_ERROR("setGroupMonitor has already been called on Group %d (%s).", gGrpId, groupConfigMap[gGrpId].grpName.c_str());
		exitSimulation(1);
	}

	// create new GroupMonitorCore object in any case and initialize analysis components
	// grpMonObj destructor (see below) will deallocate it
	GroupMonitorCore* grpMonCoreObj = new GroupMonitorCore(this, numGroupMonitor, gGrpId);
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
	groupConfigMDMap[gGrpId].groupMonitorId = numGroupMonitor;

	numGroupMonitor++;
	KERNEL_INFO("GroupMonitor set for group %d (%s)", gGrpId, groupConfigMap[gGrpId].grpName.c_str());

	return grpMonObj;
}

// FIXME: distinguish the function call at CONFIG_STATE and SETUP_STATE, where group(connect)Config[] might not be available
// or group(connect)ConfigMap is not sync with group(connect)Config[]
ConnectionMonitor* SNN::setConnectionMonitor(int grpIdPre, int grpIdPost, FILE* fid) {
	// find connection based on pre-post pair
	short int connId = getConnectId(grpIdPre, grpIdPost);
	if (connId<0) {
		KERNEL_ERROR("No connection found from group %d(%s) to group %d(%s)", grpIdPre, getGroupName(grpIdPre).c_str(),
			grpIdPost, getGroupName(grpIdPost).c_str());
		exitSimulation(1);
	}

	// check whether connection already has a connection monitor
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

	numConnectionMonitor++;
	KERNEL_INFO("ConnectionMonitor %d set for Connection %d: %d(%s) => %d(%s)", connectConfigMap[connId].connectionMonitorId, connId, grpIdPre, getGroupName(grpIdPre).c_str(),
		grpIdPost, getGroupName(grpIdPost).c_str());

	return connMonObj;
}

// FIXME: distinguish the function call at CONFIG_STATE and SETUP_STATE, where groupConfigs[0][] might not be available
// or groupConfigMap is not sync with groupConfigs[0][]
// sets up a spike generator
void SNN::setSpikeGenerator(int gGrpId, SpikeGeneratorCore* spikeGenFunc) {
	assert(snnState == CONFIG_SNN); // must be called before setupNetwork() to work on GPU
	assert(spikeGenFunc);
	assert(groupConfigMap[gGrpId].isSpikeGenerator);
	groupConfigMap[gGrpId].spikeGenFunc = spikeGenFunc;
}

// record spike information, return a SpikeInfo object
SpikeMonitor* SNN::setSpikeMonitor(int gGrpId, FILE* fid) {
	// check whether group already has a SpikeMonitor
	if (groupConfigMDMap[gGrpId].spikeMonitorId >= 0) {
		// in this case, return the current object and update fid
		SpikeMonitor* spkMonObj = getSpikeMonitor(gGrpId);

		// update spike file ID
		SpikeMonitorCore* spkMonCoreObj = getSpikeMonitorCore(gGrpId);
		spkMonCoreObj->setSpikeFileId(fid);

		KERNEL_INFO("SpikeMonitor updated for group %d (%s)", gGrpId, groupConfigMap[gGrpId].grpName.c_str());
		return spkMonObj;
	} else {
		// create new SpikeMonitorCore object in any case and initialize analysis components
		// spkMonObj destructor (see below) will deallocate it
		SpikeMonitorCore* spkMonCoreObj = new SpikeMonitorCore(this, numSpikeMonitor, gGrpId);
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
		groupConfigMDMap[gGrpId].spikeMonitorId = numSpikeMonitor;

		numSpikeMonitor++;
		KERNEL_INFO("SpikeMonitor set for group %d (%s)", gGrpId, groupConfigMap[gGrpId].grpName.c_str());

		return spkMonObj;
	}
}

// record neuron state information, return a NeuronInfo object
NeuronMonitor* SNN::setNeuronMonitor(int gGrpId, FILE* fid) {
	int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
	int netId = groupConfigMDMap[gGrpId].netId;

	if (getGroupNumNeurons(gGrpId) > 128) {
		KERNEL_WARN("Due to limited memory space, only the first 128 neurons can be monitored by NeuronMonitor");
	}

	// check whether group already has a SpikeMonitor
	if (groupConfigMDMap[gGrpId].neuronMonitorId >= 0) {
		// in this case, return the current object and update fid
		NeuronMonitor* nrnMonObj = getNeuronMonitor(gGrpId);

		// update spike file ID
		NeuronMonitorCore* nrnMonCoreObj = getNeuronMonitorCore(gGrpId);
		nrnMonCoreObj->setNeuronFileId(fid);

		KERNEL_INFO("NeuronMonitor updated for group %d (%s)", gGrpId, groupConfigMap[gGrpId].grpName.c_str());
		return nrnMonObj;
	} else {
		// create new NeuronMonitorCore object in any case and initialize analysis components
		// nrnMonObj destructor (see below) will deallocate it
		NeuronMonitorCore* nrnMonCoreObj = new NeuronMonitorCore(this, numNeuronMonitor, gGrpId);
		neuronMonCoreList[numNeuronMonitor] = nrnMonCoreObj;

		// assign neuron state file ID if we selected to write to a file, else it's NULL
		// if file pointer exists, it has already been fopened
		// this will also write the header section of the spike file
		// spkMonCoreObj destructor will fclose it
		nrnMonCoreObj->setNeuronFileId(fid);

		// create a new NeuronMonitor object for the user-interface
		// SNN::deleteObjects will deallocate it
		NeuronMonitor* nrnMonObj = new NeuronMonitor(nrnMonCoreObj);
		neuronMonList[numNeuronMonitor] = nrnMonObj;

		// also inform the grp that it is being monitored...
		groupConfigMDMap[gGrpId].neuronMonitorId = numNeuronMonitor;

		numNeuronMonitor++;
		KERNEL_INFO("NeuronMonitor set for group %d (%s)", gGrpId, groupConfigMap[gGrpId].grpName.c_str());

		return nrnMonObj;
	}
}

// FIXME: distinguish the function call at CONFIG_STATE and RUN_STATE, where groupConfigs[0][] might not be available
// or groupConfigMap is not sync with groupConfigs[0][]
// assigns spike rate to group
void SNN::setSpikeRate(int gGrpId, PoissonRate* ratePtr, int refPeriod) {
	int netId = groupConfigMDMap[gGrpId].netId;
	int lGrpId = groupConfigMDMap[gGrpId].lGrpId;

	assert(gGrpId >= 0 && lGrpId < networkConfigs[netId].numGroups);
	assert(ratePtr);
	assert(groupConfigMap[gGrpId].isSpikeGenerator);
	assert(ratePtr->getNumNeurons() == groupConfigMap[gGrpId].numN);
	assert(refPeriod >= 1);

	groupConfigMDMap[gGrpId].ratePtr = ratePtr;
	groupConfigMDMap[gGrpId].refractPeriod = refPeriod;
	spikeRateUpdated = true;
}

// sets the weight value of a specific synapse
void SNN::setWeight(short int connId, int neurIdPre, int neurIdPost, float weight, bool updateWeightRange) {
	assert(connId>=0 && connId<getNumConnections());
	assert(weight>=0.0f);

	assert(neurIdPre >= 0  && neurIdPre < getGroupNumNeurons(connectConfigMap[connId].grpSrc));
	assert(neurIdPost >= 0 && neurIdPost < getGroupNumNeurons(connectConfigMap[connId].grpDest));

	float maxWt = fabs(connectConfigMap[connId].maxWt);
	float minWt = 0.0f;

	// inform user of acton taken if weight is out of bounds
	bool needToPrintDebug = (weight>maxWt || weight<minWt);

	int netId = groupConfigMDMap[connectConfigMap[connId].grpDest].netId;
	int lGrpId = groupConfigMDMap[connectConfigMap[connId].grpDest].lGrpId;

	fetchPreConnectionInfo(netId);
	fetchConnIdsLookupArray(netId);
	fetchSynapseState(netId);

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
	int neurIdPreReal = groupConfigs[netId][lGrpId].lStartN + neurIdPre;
	int neurIdPostReal = groupConfigs[netId][lGrpId].lStartN + neurIdPost;

	// iterate over all presynaptic synapses until right one is found
	bool synapseFound = false;
	int pos_ij = managerRuntimeData.cumulativePre[neurIdPostReal];
	for (int j = 0; j < managerRuntimeData.Npre[neurIdPostReal]; pos_ij++, j++) {
		SynInfo* preId = &(managerRuntimeData.preSynapticIds[pos_ij]);
		int pre_nid = GET_CONN_NEURON_ID((*preId));
		if (GET_CONN_NEURON_ID((*preId)) == neurIdPreReal) {
			assert(managerRuntimeData.connIdsPreIdx[pos_ij] == connId); // make sure we've got the right connection ID

			managerRuntimeData.wt[pos_ij] = isExcitatoryGroup(connectConfigMap[connId].grpSrc) ? weight : -1.0 * weight;
			managerRuntimeData.maxSynWt[pos_ij] = isExcitatoryGroup(connectConfigMap[connId].grpSrc) ? maxWt : -1.0 * maxWt;

			if (netId < CPU_RUNTIME_BASE) {
#ifndef __NO_CUDA__
				// need to update datastructures on GPU runtime
				CUDA_CHECK_ERRORS(cudaMemcpy(&runtimeData[netId].wt[pos_ij], &managerRuntimeData.wt[pos_ij], sizeof(float), cudaMemcpyHostToDevice));
				if (runtimeData[netId].maxSynWt != NULL) {
					// only copy maxSynWt if datastructure actually exists on the GPU runtime
					// (that logic should be done elsewhere though)
					CUDA_CHECK_ERRORS(cudaMemcpy(&runtimeData[netId].maxSynWt[pos_ij], &managerRuntimeData.maxSynWt[pos_ij], sizeof(float), cudaMemcpyHostToDevice));
				}
#else
				assert(false);
#endif
			} else {
				// need to update datastructures on CPU runtime
				memcpy(&runtimeData[netId].wt[pos_ij], &managerRuntimeData.wt[pos_ij], sizeof(float));
				if (runtimeData[netId].maxSynWt != NULL) {
					// only copy maxSynWt if datastructure actually exists on the CPU runtime
					// (that logic should be done elsewhere though)
					memcpy(&runtimeData[netId].maxSynWt[pos_ij], &managerRuntimeData.maxSynWt[pos_ij], sizeof(float));
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

void SNN::setExternalCurrent(int grpId, const std::vector<float>& current) {
	assert(grpId >= 0); assert(grpId < numGroups);
	assert(!isPoissonGroup(grpId));
	assert(current.size() == getGroupNumNeurons(grpId));

	int netId = groupConfigMDMap[grpId].netId;
	int lGrpId = groupConfigMDMap[grpId].lGrpId;

	// // update flag for faster handling at run-time
	// if (count_if(current.begin(), current.end(), isGreaterThanZero)) {
	// 	groupConfigs[0][grpId].WithCurrentInjection = true;
	// } else {
	// 	groupConfigs[0][grpId].WithCurrentInjection = false;
	// }

	// store external current in array
	for (int lNId = groupConfigs[netId][lGrpId].lStartN, j = 0; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++, j++) {
		managerRuntimeData.extCurrent[lNId] = current[j];
	}

	// copy to GPU if necessary
	// don't allocate; allocation done in generateRuntimeData
	if (netId < CPU_RUNTIME_BASE) {
		copyExternalCurrent(netId, lGrpId, &runtimeData[netId], cudaMemcpyHostToDevice, false);
	}
	else {
		copyExternalCurrent(netId, lGrpId, &runtimeData[netId], false);
	}
}

// writes network state to file
// handling of file pointer should be handled externally: as far as this function is concerned, it is simply
// trying to write to file
void SNN::saveSimulation(FILE* fid, bool saveSynapseInfo) {
	int tmpInt;
	float tmpFloat;

	//// +++++ WRITE HEADER SECTION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//// write file signature
	tmpInt = 294338571; // some int used to identify saveSimulation files
	if (!fwrite(&tmpInt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	//// write version number
	tmpFloat = 0.2f;
	if (!fwrite(&tmpFloat,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	//// write simulation time so far (in seconds)
	tmpFloat = ((float)simTimeSec) + ((float)simTimeMs)/1000.0f;
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	//// write execution time so far (in seconds)
	stopTiming();
	tmpFloat = executionTime/1000.0f;
	if (!fwrite(&tmpFloat,sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	//// TODO: add more params of interest

	//// write network info
	if (!fwrite(&glbNetworkConfig.numN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	int dummyInt = 0;
	//if (!fwrite(&numPreSynNet,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&dummyInt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//if (!fwrite(&numPostSynNet,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&dummyInt,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	if (!fwrite(&numGroups,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	
	//// write group info
	char name[100];
	for (int gGrpId=0;gGrpId<numGroups;gGrpId++) {
		if (!fwrite(&groupConfigMDMap[gGrpId].gStartN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfigMDMap[gGrpId].gEndN,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		if (!fwrite(&groupConfigMap[gGrpId].grid.numX,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfigMap[gGrpId].grid.numY,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
		if (!fwrite(&groupConfigMap[gGrpId].grid.numZ,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

		strncpy(name,groupConfigMap[gGrpId].grpName.c_str(),100);
		if (!fwrite(name,1,100,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	}

	//// +++++ Fetch WEIGHT DATA (GPU Mode only) ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	//if (simMode_ == GPU_MODE)
	//	copyWeightState(&managerRuntimeData, &runtimeData[0], cudaMemcpyDeviceToHost, false);
	//// +++++ WRITE SYNAPSE INFO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//// \FIXME: replace with faster version
	//if (saveSynapseInfo) {
	//	for (int i = 0; i < numN; i++) {
	//		unsigned int offset = managerRuntimeData.cumulativePost[i];

	//		unsigned int count = 0;
	//		for (int t=0;t<maxDelay_;t++) {
	//			DelayInfo dPar = managerRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

	//			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++)
	//				count++;
	//		}

	//		if (!fwrite(&count,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");

	//		for (int t=0;t<maxDelay_;t++) {
	//			DelayInfo dPar = managerRuntimeData.postDelayInfo[i*(maxDelay_+1)+t];

	//			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
	//				// get synaptic info...
	//				SynInfo post_info = managerRuntimeData.postSynapticIds[offset + idx_d];

	//				// get neuron id
	//				//int p_i = (post_info&POST_SYN_NEURON_MASK);
	//				unsigned int p_i = GET_CONN_NEURON_ID(post_info);
	//				assert(p_i<numN);

	//				// get syn id
	//				unsigned int s_i = GET_CONN_SYN_ID(post_info);
	//				//>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;
	//				assert(s_i<(managerRuntimeData.Npre[p_i]));

	//				// get the cumulative position for quick access...
	//				unsigned int pos_i = managerRuntimeData.cumulativePre[p_i] + s_i;

	//				uint8_t delay = t+1;
	//				uint8_t plastic = s_i < managerRuntimeData.Npre_plastic[p_i]; // plastic or fixed.

	//				if (!fwrite(&i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&p_i,sizeof(int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&(managerRuntimeData.wt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&(managerRuntimeData.maxSynWt[pos_i]),sizeof(float),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&delay,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&plastic,sizeof(uint8_t),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//				if (!fwrite(&(managerRuntimeData.connIdsPreIdx[pos_i]),sizeof(short int),1,fid)) KERNEL_ERROR("saveSimulation fwrite error");
	//			}
	//		}
	//	}
	//}
}

// writes population weights from gIDpre to gIDpost to file fname in binary
//void SNN::writePopWeights(std::string fname, int grpIdPre, int grpIdPost) {
//	assert(grpIdPre>=0); assert(grpIdPost>=0);
//
//	float* weights;
//	int matrixSize;
//	FILE* fid;
//	int numPre, numPost;
//	fid = fopen(fname.c_str(), "wb");
//	assert(fid != NULL);
//
//	if(snnState == CONFIG_SNN || snnState == COMPILED_SNN || snnState == PARTITIONED_SNN){
//		KERNEL_ERROR("Simulation has not been run yet, cannot output weights.");
//		exitSimulation(1);
//	}
//
//	SynInfo* preId;
//	int pre_nid, pos_ij;
//
//	//population sizes
//	numPre = groupConfigs[0][grpIdPre].SizeN;
//	numPost = groupConfigs[0][grpIdPost].SizeN;
//
//	//first iteration gets the number of synaptic weights to place in our
//	//weight matrix.
//	matrixSize=0;
//	//iterate over all neurons in the post group
//	for (int i=groupConfigs[0][grpIdPost].StartN; i<=groupConfigs[0][grpIdPost].EndN; i++) {
//		// for every post-neuron, find all pre
//		pos_ij = managerRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
//		//iterate over all presynaptic synapses
//		for(int j=0; j<managerRuntimeData.Npre[i]; pos_ij++,j++) {
//			preId = &managerRuntimeData.preSynapticIds[pos_ij];
//			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
//			if (pre_nid<groupConfigs[0][grpIdPre].StartN || pre_nid>groupConfigs[0][grpIdPre].EndN)
//				continue; // connection does not belong to group grpIdPre
//			matrixSize++;
//		}
//	}
//
//	//now we have the correct size
//	weights = new float[matrixSize];
//	//second iteration assigns the weights
//	int curr = 0; // iterator for return array
//	//iterate over all neurons in the post group
//	for (int i=groupConfigs[0][grpIdPost].StartN; i<=groupConfigs[0][grpIdPost].EndN; i++) {
//		// for every post-neuron, find all pre
//		pos_ij = managerRuntimeData.cumulativePre[i]; // i-th neuron, j=0th synapse
//		//do the GPU copy here.  Copy the current weights from GPU to CPU.
//		if(simMode_==GPU_MODE){
//			copyWeightsGPU(i,grpIdPre);
//		}
//		//iterate over all presynaptic synapses
//		for(int j=0; j<managerRuntimeData.Npre[i]; pos_ij++,j++) {
//			preId = &(managerRuntimeData.preSynapticIds[pos_ij]);
//			pre_nid = GET_CONN_NEURON_ID((*preId)); // neuron id of pre
//			if (pre_nid<groupConfigs[0][grpIdPre].StartN || pre_nid>groupConfigs[0][grpIdPre].EndN)
//				continue; // connection does not belong to group grpIdPre
//			weights[curr] = managerRuntimeData.wt[pos_ij];
//			curr++;
//		}
//	}
//
//	fwrite(weights,sizeof(float),matrixSize,fid);
//	fclose(fid);
//	//Let my memory FREE!!!
//	delete [] weights;
//}


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
	short int connId = -1;

	for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
		if (it->second.grpSrc == grpIdPre && it->second.grpDest == grpIdPost) {
			connId = it->second.connId;
			break;
		}
	}

	return connId;
}

ConnectConfig SNN::getConnectConfig(short int connId) {
	CHECK_CONNECTION_ID(connId, numConnections);

	if (connectConfigMap.find(connId) == connectConfigMap.end()) {
		KERNEL_ERROR("Total Connections = %d", numConnections);
		KERNEL_ERROR("ConnectId (%d) cannot be recognized", connId);
	}

	return connectConfigMap[connId];
}

std::vector<float> SNN::getConductanceAMPA(int gGrpId) {
	assert(isSimulationWithCOBA());

	// copy data to the manager runtime
	fetchConductanceAMPA(gGrpId);

	std::vector<float> gAMPAvec;
	for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
		gAMPAvec.push_back(managerRuntimeData.gAMPA[gNId]);
	}
	return gAMPAvec;
}

std::vector<float> SNN::getConductanceNMDA(int gGrpId) {
	assert(isSimulationWithCOBA());

	// copy data to the manager runtime
	fetchConductanceNMDA(gGrpId);

	std::vector<float> gNMDAvec;
	if (isSimulationWithNMDARise()) {
		// need to construct conductance from rise and decay parts
		for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
			gNMDAvec.push_back(managerRuntimeData.gNMDA_d[gNId] - managerRuntimeData.gNMDA_r[gNId]);
		}
	} else {
		for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
			gNMDAvec.push_back(managerRuntimeData.gNMDA[gNId]);
		}
	}
	return gNMDAvec;
}

std::vector<float> SNN::getConductanceGABAa(int gGrpId) {
	assert(isSimulationWithCOBA());

	// copy data to the manager runtime
	fetchConductanceGABAa(gGrpId);

	std::vector<float> gGABAaVec;
	for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
		gGABAaVec.push_back(managerRuntimeData.gGABAa[gNId]);
	}
	return gGABAaVec;
}

std::vector<float> SNN::getConductanceGABAb(int gGrpId) {
	assert(isSimulationWithCOBA());

	// copy data to the manager runtime
	fetchConductanceGABAb(gGrpId);

	std::vector<float> gGABAbVec;
	if (isSimulationWithGABAbRise()) {
		// need to construct conductance from rise and decay parts
		for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
			gGABAbVec.push_back(managerRuntimeData.gGABAb_d[gNId] - managerRuntimeData.gGABAb_r[gNId]);
		}
	} else {
		for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
			gGABAbVec.push_back(managerRuntimeData.gGABAb[gNId]);
		}
	}
	return gGABAbVec;
}

// returns RangeDelay struct of a connection
RangeDelay SNN::getDelayRange(short int connId) {
	assert(connId>=0 && connId<numConnections);

	return RangeDelay(connectConfigMap[connId].minDelay, connectConfigMap[connId].maxDelay);
}

// \TODO: bad API design (return allocated memory to user), consider to move this function to connection monitor
uint8_t* SNN::getDelays(int gGrpIdPre, int gGrpIdPost, int& numPreN, int& numPostN) {
	int netIdPost = groupConfigMDMap[gGrpIdPost].netId;
	int lGrpIdPost = groupConfigMDMap[gGrpIdPost].lGrpId;
	int lGrpIdPre = -1;
	uint8_t* delays;

	for (int lGrpId = 0; lGrpId < networkConfigs[netIdPost].numGroupsAssigned; lGrpId++)
		if (groupConfigs[netIdPost][lGrpId].gGrpId == gGrpIdPre) {
			lGrpIdPre = lGrpId;
			break;
		}
	assert(lGrpIdPre != -1);
	
	numPreN = groupConfigMap[gGrpIdPre].numN;
	numPostN = groupConfigMap[gGrpIdPost].numN;

	delays = new uint8_t[numPreN * numPostN];
	memset(delays, 0, numPreN * numPostN);

	fetchPostConnectionInfo(netIdPost);

	for (int lNIdPre = groupConfigs[netIdPost][lGrpIdPre].lStartN; lNIdPre < groupConfigs[netIdPost][lGrpIdPre].lEndN; lNIdPre++) {
		unsigned int offset = managerRuntimeData.cumulativePost[lNIdPre];

		for (int t = 0; t < glbNetworkConfig.maxDelay; t++) {
			DelayInfo dPar = managerRuntimeData.postDelayInfo[lNIdPre * (glbNetworkConfig.maxDelay + 1) + t];

			for(int idx_d = dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++) {
				// get synaptic info...
				SynInfo postSynInfo = managerRuntimeData.postSynapticIds[offset + idx_d];

				// get local post neuron id
				int lNIdPost = GET_CONN_NEURON_ID(postSynInfo);
				assert(lNIdPost < glbNetworkConfig.numN);

				if (lNIdPost >= groupConfigs[netIdPost][lGrpIdPost].lStartN && lNIdPost <= groupConfigs[netIdPost][lGrpIdPost].lEndN) {
					delays[(lNIdPre - groupConfigs[netIdPost][lGrpIdPre].lStartN) + numPreN * (lNIdPost - groupConfigs[netIdPost][lGrpIdPost].lStartN)] = t + 1;
				}
			}
		}
	}
	return delays;
}

Grid3D SNN::getGroupGrid3D(int gGrpId) {
	assert(gGrpId >= 0 && gGrpId < numGroups);

	return groupConfigMap[gGrpId].grid;
}

// find ID of group with name grpName
int SNN::getGroupId(std::string grpName) {
	int grpId = -1;
	for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
		if (groupConfigMap[gGrpId].grpName.compare(grpName) == 0) {
			grpId = gGrpId;
			break;
		}
	}

	return grpId;
}

std::string SNN::getGroupName(int gGrpId) {
	assert(gGrpId >= -1 && gGrpId < numGroups);

	if (gGrpId == ALL)
		return "ALL";

	return groupConfigMap[gGrpId].grpName;
}

GroupSTDPInfo SNN::getGroupSTDPInfo(int gGrpId) {
	GroupSTDPInfo gInfo;

	gInfo.WithSTDP = groupConfigMap[gGrpId].stdpConfig.WithSTDP;
	gInfo.WithESTDP = groupConfigMap[gGrpId].stdpConfig.WithESTDP;
	gInfo.WithISTDP = groupConfigMap[gGrpId].stdpConfig.WithISTDP;
	gInfo.WithESTDPtype = groupConfigMap[gGrpId].stdpConfig.WithESTDPtype;
	gInfo.WithISTDPtype = groupConfigMap[gGrpId].stdpConfig.WithISTDPtype;
	gInfo.WithESTDPcurve = groupConfigMap[gGrpId].stdpConfig.WithESTDPcurve;
	gInfo.WithISTDPcurve = groupConfigMap[gGrpId].stdpConfig.WithISTDPcurve;
	gInfo.ALPHA_MINUS_EXC = groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_EXC;
	gInfo.ALPHA_PLUS_EXC = groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_EXC;
	gInfo.TAU_MINUS_INV_EXC = groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_EXC;
	gInfo.TAU_PLUS_INV_EXC = groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_EXC;
	gInfo.ALPHA_MINUS_INB = groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_INB;
	gInfo.ALPHA_PLUS_INB = groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_INB;
	gInfo.TAU_MINUS_INV_INB = groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_INB;
	gInfo.TAU_PLUS_INV_INB = groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_INB;
	gInfo.GAMMA = groupConfigMap[gGrpId].stdpConfig.GAMMA;
	gInfo.BETA_LTP = groupConfigMap[gGrpId].stdpConfig.BETA_LTP;
	gInfo.BETA_LTD = groupConfigMap[gGrpId].stdpConfig.BETA_LTD;
	gInfo.LAMBDA = groupConfigMap[gGrpId].stdpConfig.LAMBDA;
	gInfo.DELTA = groupConfigMap[gGrpId].stdpConfig.DELTA;

	return gInfo;
}

GroupNeuromodulatorInfo SNN::getGroupNeuromodulatorInfo(int gGrpId) {
	GroupNeuromodulatorInfo gInfo;

	gInfo.baseDP = groupConfigMap[gGrpId].neuromodulatorConfig.baseDP;
	gInfo.base5HT = groupConfigMap[gGrpId].neuromodulatorConfig.base5HT;
	gInfo.baseACh = groupConfigMap[gGrpId].neuromodulatorConfig.baseACh;
	gInfo.baseNE = groupConfigMap[gGrpId].neuromodulatorConfig.baseNE;
	gInfo.decayDP = groupConfigMap[gGrpId].neuromodulatorConfig.decayDP;
	gInfo.decay5HT = groupConfigMap[gGrpId].neuromodulatorConfig.decay5HT;
	gInfo.decayACh = groupConfigMap[gGrpId].neuromodulatorConfig.decayACh;
	gInfo.decayNE = groupConfigMap[gGrpId].neuromodulatorConfig.decayNE;

	return gInfo;
}

Point3D SNN::getNeuronLocation3D(int gNId) {
	int gGrpId = -1;
	assert(gNId >= 0 && gNId < glbNetworkConfig.numN);
	
	// search for global group id
	for (std::map<int, GroupConfigMD>::iterator grpIt = groupConfigMDMap.begin(); grpIt != groupConfigMDMap.end(); grpIt++) {
		if (gNId >= grpIt->second.gStartN && gNId <= grpIt->second.gEndN)
			gGrpId = grpIt->second.gGrpId;
	}

	// adjust neurId for neuron ID of first neuron in the group
	int neurId = gNId - groupConfigMDMap[gGrpId].gStartN;

	return getNeuronLocation3D(gGrpId, neurId);
}

Point3D SNN::getNeuronLocation3D(int gGrpId, int relNeurId) {
	Grid3D grid = groupConfigMap[gGrpId].grid;
	assert(gGrpId >= 0 && gGrpId < numGroups);
	assert(relNeurId >= 0 && relNeurId < getGroupNumNeurons(gGrpId));

	int intX = relNeurId % grid.numX;
	int intY = (relNeurId / grid.numX) % grid.numY;
	int intZ = relNeurId / (grid.numX * grid.numY);

	// get coordinates center around origin
	double coordX = grid.distX * intX + grid.offsetX;
	double coordY = grid.distY * intY + grid.offsetY;
	double coordZ = grid.distZ * intZ + grid.offsetZ;
	return Point3D(coordX, coordY, coordZ);
}

// returns the number of synaptic connections associated with this connection.
int SNN::getNumSynapticConnections(short int connId) {
	//we didn't find the connection.
	if (connectConfigMap.find(connId) == connectConfigMap.end()) {
		KERNEL_ERROR("Connection ID was not found.  Quitting.");
		exitSimulation(1);
	}

	return connectConfigMap[connId].numberOfConnections;
}

// returns pointer to existing SpikeMonitor object, NULL else
SpikeMonitor* SNN::getSpikeMonitor(int gGrpId) {
	assert(gGrpId >= 0 && gGrpId < getNumGroups());

	if (groupConfigMDMap[gGrpId].spikeMonitorId >= 0) {
		return spikeMonList[(groupConfigMDMap[gGrpId].spikeMonitorId)];
	} else {
		return NULL;
	}
}

SpikeMonitorCore* SNN::getSpikeMonitorCore(int gGrpId) {
	assert(gGrpId >= 0 && gGrpId < getNumGroups());

	if (groupConfigMDMap[gGrpId].spikeMonitorId >= 0) {
		return spikeMonCoreList[(groupConfigMDMap[gGrpId].spikeMonitorId)];
	} else {
		return NULL;
	}
}

// returns pointer to existing NeuronMonitor object, NULL else
NeuronMonitor* SNN::getNeuronMonitor(int gGrpId) {
	assert(gGrpId >= 0 && gGrpId < getNumGroups());

	if (groupConfigMDMap[gGrpId].neuronMonitorId >= 0) {
		return neuronMonList[(groupConfigMDMap[gGrpId].neuronMonitorId)];
	}
	else {
		return NULL;
	}
}

NeuronMonitorCore* SNN::getNeuronMonitorCore(int gGrpId) {
	assert(gGrpId >= 0 && gGrpId < getNumGroups());

	if (groupConfigMDMap[gGrpId].neuronMonitorId >= 0) {
		return neuronMonCoreList[(groupConfigMDMap[gGrpId].neuronMonitorId)];
	}
	else {
		return NULL;
	}
}

RangeWeight SNN::getWeightRange(short int connId) {
	assert(connId>=0 && connId<numConnections);

	return RangeWeight(0.0f, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt);
}


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

// all unsafe operations of SNN constructor
void SNN::SNNinit() {
	// initialize snnState
	snnState = CONFIG_SNN;
	
	// set logger mode (defines where to print all status, error, and debug messages)
	switch (loggerMode_) {
	case USER:
		fpInf_ = stdout;
		fpErr_ = stderr;
		#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
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
		#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
			fpInf_ = fopen("nul","w");
		#else
			fpInf_ = fopen("/dev/null","w");
		#endif
		fpErr_ = stderr;
		#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
			fpDeb_ = fopen("nul","w");
		#else
			fpDeb_ = fopen("/dev/null","w");
		#endif
		break;
	case SILENT:
	case CUSTOM:
		#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
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
		exit(UNKNOWN_LOGGER_ERROR);

	}

	// try to open log file in results folder: create if not exists
#if defined(WIN32) || defined(WIN64)
	CreateDirectory("results", NULL);
	fpLog_ = fopen("results/carlsim.log", "w");
#else
	struct stat sb;
	int createDir = 1;
	if (stat("results", &sb) == -1 || !S_ISDIR(sb.st_mode)) {
		// results dir does not exist, try to create:
		createDir = mkdir("results", 0777);
	}

	if (createDir == -1) {
		// tried to create dir, but failed
		fprintf(stderr, "Could not create directory \"results/\", which is required to "
			"store simulation results. Aborting simulation...\n");
		exit(NO_LOGGER_DIR_ERROR);
	} else {
		// open log file
		fpLog_ = fopen("results/carlsim.log", "w");

		if (createDir == 0) {
			// newly created dir: now that fpLog_/fpInf_ exist, inform user
			KERNEL_INFO("Created results directory \"results/\".");
		}
	}
#endif
	if (fpLog_ == NULL) {
		fprintf(stderr, "Could not create the directory \"results/\" or the log file \"results/carlsim.log\""
			", which is required to store simulation results. Aborting simulation...\n");
		exit(NO_LOGGER_DIR_ERROR);
	}

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

	simTimeRunStart = 0; simTimeRunStop = 0;
	simTimeLastRunSummary = 0;
	simTimeMs = 0; simTimeSec = 0; simTime = 0;

	numGroups = 0;
	numConnections = 0;
	numCompartmentConnections = 0;
	numSpikeGenGrps = 0;
	simulatorDeleted = false;

	cumExecutionTime = 0.0;
	executionTime = 0.0;

	spikeRateUpdated = false;
	numSpikeMonitor = 0;
	numNeuronMonitor = 0;
	numGroupMonitor = 0;
	numConnectionMonitor = 0;

	sim_with_compartments = false;
	sim_with_fixedwts = true; // default is true, will be set to false if there are any plastic synapses
	sim_with_conductances = false; // default is false
	sim_with_stdp = false;
	sim_with_modulated_stdp = false;
	sim_with_homeostasis = false;
	sim_with_stp = false;
	sim_in_testing = false;

	loadSimFID = NULL;

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

	// default integration method: Forward-Euler with 0.5ms integration step
	setIntegrationMethod(FORWARD_EULER, 2);

	mulSynFast = NULL;
	mulSynSlow = NULL;

	// reset all monitors, don't deallocate (false)
	resetMonitors(false);

	resetGroupConfigs(false);

	resetConnectionConfigs(false);

	// initialize spike buffer
	spikeBuf = new SpikeBuffer(0, MAX_TIME_SLICE);

	memset(networkConfigs, 0, sizeof(NetworkConfigRT) * MAX_NET_PER_SNN);
	
	// reset all runtime data
	// GPU/CPU runtime data
	memset(runtimeData, 0, sizeof(RuntimeData) * MAX_NET_PER_SNN);
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) // FIXME: redundant??
		runtimeData[netId].allocated = false;

	// Manager runtime data
	memset(&managerRuntimeData, 0, sizeof(RuntimeData));
	managerRuntimeData.allocated = false; // FIXME: redundant??

	// default weight update parameter
	wtANDwtChangeUpdateInterval_ = 1000; // update weights every 1000 ms (default)
	wtANDwtChangeUpdateIntervalCnt_ = 0; // helper var to implement fast modulo
	stdpScaleFactor_ = 1.0f;
	wtChangeDecay_ = 0.0f;

	// FIXME: use it when necessary
#ifndef __NO_CUDA__
	CUDA_CREATE_TIMER(timer);
	CUDA_RESET_TIMER(timer);
#endif
}

void SNN::advSimStep() {
	doSTPUpdateAndDecayCond();

	//KERNEL_INFO("STPUpdate!");

	spikeGeneratorUpdate();

	//KERNEL_INFO("spikeGeneratorUpdate!");

	findFiring();

	//KERNEL_INFO("Find firing!");

	updateTimingTable();

	routeSpikes();

	doCurrentUpdate();

	//KERNEL_INFO("doCurrentUpdate!");

	globalStateUpdate();

	//KERNEL_INFO("globalStateUpdate!");

	clearExtFiringTable();
}

void SNN::doSTPUpdateAndDecayCond() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			assert(runtimeData[netId].allocated);
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				doSTPUpdateAndDecayCond_GPU(netId);
			else{//CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					doSTPUpdateAndDecayCond_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperDoSTPUpdateAndDecayCond_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif
}

void SNN::spikeGeneratorUpdate() {
	// If poisson rate has been updated, assign new poisson rate
	if (spikeRateUpdated) {
		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
			cpu_set_t cpus;	
			ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
			int threadCount = 0;
		#endif

		for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
			if (!groupPartitionLists[netId].empty()) {
				if (netId < CPU_RUNTIME_BASE) // GPU runtime
					assignPoissonFiringRate_GPU(netId);
				else{ // CPU runtime
					#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
						assignPoissonFiringRate_CPU(netId);
					#else // Linux or MAC
						pthread_attr_t attr;
						pthread_attr_init(&attr);
						CPU_ZERO(&cpus);
						CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
						pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

						argsThreadRoutine[threadCount].snn_pointer = this;
						argsThreadRoutine[threadCount].netId = netId;
						argsThreadRoutine[threadCount].lGrpId = 0;
						argsThreadRoutine[threadCount].startIdx = 0;
						argsThreadRoutine[threadCount].endIdx = 0;
						argsThreadRoutine[threadCount].GtoLOffset = 0;

						pthread_create(&threads[threadCount], &attr, &SNN::helperAssignPoissonFiringRate_CPU, (void*)&argsThreadRoutine[threadCount]);
						pthread_attr_destroy(&attr);
						threadCount++;
					#endif
				}
			}
		}

		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			// join all the threads
			for (int i=0; i<threadCount; i++){
				pthread_join(threads[i], NULL);
			}
		#endif

		spikeRateUpdated = false;
	}

	// If time slice has expired, check if new spikes needs to be generated by user-defined spike generators
	generateUserDefinedSpikes();

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				spikeGeneratorUpdate_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					spikeGeneratorUpdate_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperSpikeGeneratorUpdate_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif

	// tell the spike buffer to advance to the next time step
	spikeBuf->step();
}

void SNN::findFiring() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				findFiring_GPU(netId);
			else {// CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					findFiring_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperFindFiring_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif
}

void SNN::doCurrentUpdate() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				doCurrentUpdateD2_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					doCurrentUpdateD2_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperDoCurrentUpdateD2_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
		threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				doCurrentUpdateD1_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					doCurrentUpdateD1_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperDoCurrentUpdateD1_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif
}

void SNN::updateTimingTable() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				updateTimingTable_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					updateTimingTable_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperUpdateTimingTable_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif
}

void SNN::globalStateUpdate() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				globalStateUpdate_C_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					globalStateUpdate_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperGlobalStateUpdate_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				globalStateUpdate_N_GPU(netId);
		}
	}

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				globalStateUpdate_G_GPU(netId);
		}
	}
}

void SNN::clearExtFiringTable() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				clearExtFiringTable_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					clearExtFiringTable_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperClearExtFiringTable_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif
}

void SNN::updateWeights() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				updateWeights_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					updateWeights_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperUpdateWeights_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif

}

void SNN::updateNetworkConfig(int netId) {
	assert(netId < MAX_NET_PER_SNN);

	if (netId < CPU_RUNTIME_BASE) // GPU runtime
		copyNetworkConfig(netId, cudaMemcpyHostToDevice);
	else
		copyNetworkConfig(netId); // CPU runtime
}

void SNN::shiftSpikeTables() {
	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				shiftSpikeTables_F_GPU(netId);
			else { // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					shiftSpikeTables_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperShiftSpikeTables_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				shiftSpikeTables_T_GPU(netId);
		}
	}
}

void SNN::allocateSNN(int netId) {
	assert(netId > ANY && netId < MAX_NET_PER_SNN);
	
	if (netId < CPU_RUNTIME_BASE)
		allocateSNN_GPU(netId);
	else 
		allocateSNN_CPU(netId);
}

void SNN::allocateManagerRuntimeData() {
	// reset variable related to spike count
	managerRuntimeData.spikeCountSec = 0;
	managerRuntimeData.spikeCountD1Sec = 0;
	managerRuntimeData.spikeCountD2Sec = 0;
	managerRuntimeData.spikeCountLastSecLeftD2 = 0;
	managerRuntimeData.spikeCount = 0;
	managerRuntimeData.spikeCountD1 = 0;
	managerRuntimeData.spikeCountD2 = 0;
	managerRuntimeData.nPoissonSpikes = 0;
	managerRuntimeData.spikeCountExtRxD1 = 0;
	managerRuntimeData.spikeCountExtRxD2 = 0;

	managerRuntimeData.voltage    = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.nextVoltage = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.recovery   = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_a      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_b      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_c      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_d      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_C	  = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_k	  = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_vr	  = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_vt	  = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.Izh_vpeak  = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_tau_m      = new int[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_tau_ref      = new int[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_tau_ref_c      = new int[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_vTh      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_vReset      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_gain      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.lif_bias      = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.current    = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.extCurrent = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.totalCurrent = new float[managerRTDSize.maxNumNReg];
	managerRuntimeData.curSpike   = new bool[managerRTDSize.maxNumNReg];
	memset(managerRuntimeData.voltage, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.nextVoltage, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.recovery, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_a, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_b, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_c, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_d, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_C, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_k, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_vr, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_vt, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.Izh_vpeak, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_tau_m, 0, sizeof(int) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_tau_ref, 0, sizeof(int) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_tau_ref_c, 0, sizeof(int) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_vTh, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_vReset, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_gain, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.lif_bias, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.current, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.extCurrent, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.totalCurrent, 0, sizeof(float) * managerRTDSize.maxNumNReg);
	memset(managerRuntimeData.curSpike, 0, sizeof(bool) * managerRTDSize.maxNumNReg);

	managerRuntimeData.nVBuffer = new float[MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups]; // 1 second v buffer
	managerRuntimeData.nUBuffer = new float[MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups];
	managerRuntimeData.nIBuffer = new float[MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups];
	memset(managerRuntimeData.nVBuffer, 0, sizeof(float) * MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.nUBuffer, 0, sizeof(float) * MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.nIBuffer, 0, sizeof(float) * MAX_NEURON_MON_GRP_SZIE * 1000 * managerRTDSize.maxNumGroups);

	managerRuntimeData.gAMPA  = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gNMDA_r = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gNMDA_d = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gNMDA = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	memset(managerRuntimeData.gAMPA, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gNMDA_r, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gNMDA_d, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gNMDA, 0, sizeof(float) * managerRTDSize.glbNumNReg);

	managerRuntimeData.gGABAa = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gGABAb_r = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gGABAb_d = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	managerRuntimeData.gGABAb = new float[managerRTDSize.glbNumNReg]; // sufficient to hold all regular neurons in the global network
	memset(managerRuntimeData.gGABAa, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gGABAb_r, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gGABAb_d, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	memset(managerRuntimeData.gGABAb, 0, sizeof(float) * managerRTDSize.glbNumNReg);
	
	// allocate neuromodulators and their assistive buffers
	managerRuntimeData.grpDA  = new float[managerRTDSize.maxNumGroups];
	managerRuntimeData.grp5HT = new float[managerRTDSize.maxNumGroups];
	managerRuntimeData.grpACh = new float[managerRTDSize.maxNumGroups];
	managerRuntimeData.grpNE  = new float[managerRTDSize.maxNumGroups];
	memset(managerRuntimeData.grpDA, 0, sizeof(float) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.grp5HT, 0, sizeof(float) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.grpACh, 0, sizeof(float) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.grpNE, 0, sizeof(float) * managerRTDSize.maxNumGroups);


	managerRuntimeData.grpDABuffer  = new float[managerRTDSize.maxNumGroups * 1000]; // 1 second DA buffer
	managerRuntimeData.grp5HTBuffer = new float[managerRTDSize.maxNumGroups * 1000];
	managerRuntimeData.grpAChBuffer = new float[managerRTDSize.maxNumGroups * 1000];
	managerRuntimeData.grpNEBuffer  = new float[managerRTDSize.maxNumGroups * 1000];
	memset(managerRuntimeData.grpDABuffer, 0, managerRTDSize.maxNumGroups * sizeof(float) * 1000);
	memset(managerRuntimeData.grp5HTBuffer, 0, managerRTDSize.maxNumGroups * sizeof(float) * 1000);
	memset(managerRuntimeData.grpAChBuffer, 0, managerRTDSize.maxNumGroups * sizeof(float) * 1000);
	memset(managerRuntimeData.grpNEBuffer, 0, managerRTDSize.maxNumGroups * sizeof(float) * 1000);

	managerRuntimeData.lastSpikeTime = new int[managerRTDSize.maxNumNAssigned];
	memset(managerRuntimeData.lastSpikeTime, 0, sizeof(int) * managerRTDSize.maxNumNAssigned);
	
	managerRuntimeData.nSpikeCnt = new int[managerRTDSize.glbNumN];
	memset(managerRuntimeData.nSpikeCnt, 0, sizeof(int) * managerRTDSize.glbNumN); // sufficient to hold all neurons in the global network

	//! homeostasis variables
	managerRuntimeData.avgFiring  = new float[managerRTDSize.maxNumN];
	managerRuntimeData.baseFiring = new float[managerRTDSize.maxNumN];
	memset(managerRuntimeData.avgFiring, 0, sizeof(float) * managerRTDSize.maxNumN);
	memset(managerRuntimeData.baseFiring, 0, sizeof(float) * managerRTDSize.maxNumN);

	// STP can be applied to spike generators, too -> numN
	// \TODO: The size of these data structures could be reduced to the max synaptic delay of all
	// connections with STP. That number might not be the same as maxDelay_.
	managerRuntimeData.stpu = new float[managerRTDSize.maxNumN * (glbNetworkConfig.maxDelay + 1)];
	managerRuntimeData.stpx = new float[managerRTDSize.maxNumN * (glbNetworkConfig.maxDelay + 1)];
	memset(managerRuntimeData.stpu, 0, sizeof(float) * managerRTDSize.maxNumN * (glbNetworkConfig.maxDelay + 1));
	memset(managerRuntimeData.stpx, 0, sizeof(float) * managerRTDSize.maxNumN * (glbNetworkConfig.maxDelay + 1));

	managerRuntimeData.Npre           = new unsigned short[managerRTDSize.maxNumNAssigned];
	managerRuntimeData.Npre_plastic   = new unsigned short[managerRTDSize.maxNumNAssigned];
	managerRuntimeData.Npost          = new unsigned short[managerRTDSize.maxNumNAssigned];
	managerRuntimeData.cumulativePost = new unsigned int[managerRTDSize.maxNumNAssigned];
	managerRuntimeData.cumulativePre  = new unsigned int[managerRTDSize.maxNumNAssigned];
	memset(managerRuntimeData.Npre, 0, sizeof(short) * managerRTDSize.maxNumNAssigned);
	memset(managerRuntimeData.Npre_plastic, 0, sizeof(short) * managerRTDSize.maxNumNAssigned);
	memset(managerRuntimeData.Npost, 0, sizeof(short) * managerRTDSize.maxNumNAssigned);
	memset(managerRuntimeData.cumulativePost, 0, sizeof(int) * managerRTDSize.maxNumNAssigned);
	memset(managerRuntimeData.cumulativePre, 0, sizeof(int) * managerRTDSize.maxNumNAssigned);

	managerRuntimeData.postSynapticIds = new SynInfo[managerRTDSize.maxNumPostSynNet];
	managerRuntimeData.postDelayInfo   = new DelayInfo[managerRTDSize.maxNumNAssigned * (glbNetworkConfig.maxDelay + 1)];	//!< Possible delay values are 0....maxDelay_ (inclusive of maxDelay_)
	memset(managerRuntimeData.postSynapticIds, 0, sizeof(SynInfo) * managerRTDSize.maxNumPostSynNet);
	memset(managerRuntimeData.postDelayInfo, 0, sizeof(DelayInfo) * managerRTDSize.maxNumNAssigned * (glbNetworkConfig.maxDelay + 1));

	managerRuntimeData.preSynapticIds	= new SynInfo[managerRTDSize.maxNumPreSynNet];
	memset(managerRuntimeData.preSynapticIds, 0, sizeof(SynInfo) * managerRTDSize.maxNumPreSynNet);

	managerRuntimeData.wt           = new float[managerRTDSize.maxNumPreSynNet];
	managerRuntimeData.wtChange     = new float[managerRTDSize.maxNumPreSynNet];
	managerRuntimeData.maxSynWt     = new float[managerRTDSize.maxNumPreSynNet];
	managerRuntimeData.synSpikeTime = new int[managerRTDSize.maxNumPreSynNet];
	memset(managerRuntimeData.wt, 0, sizeof(float) * managerRTDSize.maxNumPreSynNet);
	memset(managerRuntimeData.wtChange, 0, sizeof(float) * managerRTDSize.maxNumPreSynNet);
	memset(managerRuntimeData.maxSynWt, 0, sizeof(float) * managerRTDSize.maxNumPreSynNet);
	memset(managerRuntimeData.synSpikeTime, 0, sizeof(int) * managerRTDSize.maxNumPreSynNet);

	mulSynFast = new float[managerRTDSize.maxNumConnections];
	mulSynSlow = new float[managerRTDSize.maxNumConnections];
	memset(mulSynFast, 0, sizeof(float) * managerRTDSize.maxNumConnections);
	memset(mulSynSlow, 0, sizeof(float) * managerRTDSize.maxNumConnections);

	managerRuntimeData.connIdsPreIdx	= new short int[managerRTDSize.maxNumPreSynNet];
	memset(managerRuntimeData.connIdsPreIdx, 0, sizeof(short int) * managerRTDSize.maxNumPreSynNet);

	managerRuntimeData.grpIds = new short int[managerRTDSize.maxNumNAssigned];
	memset(managerRuntimeData.grpIds, 0, sizeof(short int) * managerRTDSize.maxNumNAssigned);

	managerRuntimeData.spikeGenBits = new unsigned int[managerRTDSize.maxNumNSpikeGen / 32 + 1];

	// Confirm allocation of SNN runtime data in main memory
	managerRuntimeData.allocated = true;
	managerRuntimeData.memType = CPU_MEM;
}

int SNN::assignGroup(int gGrpId, int availableNeuronId) {
	int newAvailableNeuronId;
	assert(groupConfigMDMap[gGrpId].gStartN == -1); // The group has not yet been assigned
	groupConfigMDMap[gGrpId].gStartN = availableNeuronId;
	groupConfigMDMap[gGrpId].gEndN = availableNeuronId + groupConfigMap[gGrpId].numN - 1;

	KERNEL_DEBUG("Allocation for %d(%s), St=%d, End=%d",
				gGrpId, groupConfigMap[gGrpId].grpName.c_str(), groupConfigMDMap[gGrpId].gStartN, groupConfigMDMap[gGrpId].gEndN);

	newAvailableNeuronId = availableNeuronId + groupConfigMap[gGrpId].numN;
	//assert(newAvailableNeuronId <= numN);

	return newAvailableNeuronId;
}

int SNN::assignGroup(std::list<GroupConfigMD>::iterator grpIt, int localGroupId, int availableNeuronId) {
	int newAvailableNeuronId;
	assert(grpIt->lGrpId == -1); // The group has not yet been assigned
	grpIt->lGrpId = localGroupId;
	grpIt->lStartN = availableNeuronId;
	grpIt->lEndN = availableNeuronId + groupConfigMap[grpIt->gGrpId].numN - 1;

	grpIt->LtoGOffset = grpIt->gStartN - grpIt->lStartN;
	grpIt->GtoLOffset = grpIt->lStartN - grpIt->gStartN;

	KERNEL_DEBUG("Allocation for group (%s) [id:%d, local id:%d], St=%d, End=%d", groupConfigMap[grpIt->gGrpId].grpName.c_str(),
		grpIt->gGrpId, grpIt->lGrpId, grpIt->lStartN, grpIt->lEndN);

	newAvailableNeuronId = availableNeuronId + groupConfigMap[grpIt->gGrpId].numN;

	return newAvailableNeuronId;
}

void SNN::generateGroupRuntime(int netId, int lGrpId) {
	resetNeuromodulator(netId, lGrpId);

	for(int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++)
		resetNeuron(netId, lGrpId, lNId);
}

void SNN::generateRuntimeGroupConfigs() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
			// publish the group configs in an array for quick access and accessible on GPUs (cuda doesn't support std::list)
			int gGrpId = grpIt->gGrpId;
			int lGrpId = grpIt->lGrpId;

			// Data published by groupConfigMDMap[] are generated in compileSNN() and are invariant in partitionSNN()
			// Data published by grpIt are generated in partitionSNN() and maybe have duplicated copys
			groupConfigs[netId][lGrpId].netId = grpIt->netId;
			groupConfigs[netId][lGrpId].gGrpId = grpIt->gGrpId;
			groupConfigs[netId][lGrpId].gStartN = grpIt->gStartN;
			groupConfigs[netId][lGrpId].gEndN = grpIt->gEndN;
			groupConfigs[netId][lGrpId].lGrpId = grpIt->lGrpId;
			groupConfigs[netId][lGrpId].lStartN = grpIt->lStartN;
			groupConfigs[netId][lGrpId].lEndN = grpIt->lEndN;
			groupConfigs[netId][lGrpId].LtoGOffset = grpIt->LtoGOffset;
			groupConfigs[netId][lGrpId].GtoLOffset = grpIt->GtoLOffset;
			groupConfigs[netId][lGrpId].Type = groupConfigMap[gGrpId].type;
			groupConfigs[netId][lGrpId].numN = groupConfigMap[gGrpId].numN;
			groupConfigs[netId][lGrpId].numPostSynapses = grpIt->numPostSynapses;
			groupConfigs[netId][lGrpId].numPreSynapses = grpIt->numPreSynapses;
			groupConfigs[netId][lGrpId].isSpikeGenerator = groupConfigMap[gGrpId].isSpikeGenerator;
			groupConfigs[netId][lGrpId].isSpikeGenFunc = groupConfigMap[gGrpId].spikeGenFunc != NULL ? true : false;
			groupConfigs[netId][lGrpId].WithSTP =  groupConfigMap[gGrpId].stpConfig.WithSTP;
			groupConfigs[netId][lGrpId].WithSTDP =  groupConfigMap[gGrpId].stdpConfig.WithSTDP;
			groupConfigs[netId][lGrpId].WithESTDP =  groupConfigMap[gGrpId].stdpConfig.WithESTDP;
			groupConfigs[netId][lGrpId].WithISTDP = groupConfigMap[gGrpId].stdpConfig.WithISTDP;
			groupConfigs[netId][lGrpId].WithESTDPtype = groupConfigMap[gGrpId].stdpConfig.WithESTDPtype;
			groupConfigs[netId][lGrpId].WithISTDPtype =  groupConfigMap[gGrpId].stdpConfig.WithISTDPtype; 
			groupConfigs[netId][lGrpId].WithESTDPcurve =  groupConfigMap[gGrpId].stdpConfig.WithESTDPcurve;
			groupConfigs[netId][lGrpId].WithISTDPcurve =  groupConfigMap[gGrpId].stdpConfig.WithISTDPcurve;
			groupConfigs[netId][lGrpId].WithHomeostasis =  groupConfigMap[gGrpId].homeoConfig.WithHomeostasis;
			groupConfigs[netId][lGrpId].FixedInputWts = grpIt->fixedInputWts;
			groupConfigs[netId][lGrpId].hasExternalConnect = grpIt->hasExternalConnect;
			groupConfigs[netId][lGrpId].Noffset = grpIt->Noffset; // Note: Noffset is not valid at this time
			groupConfigs[netId][lGrpId].MaxDelay = grpIt->maxOutgoingDelay;
			groupConfigs[netId][lGrpId].STP_A = groupConfigMap[gGrpId].stpConfig.STP_A;
			groupConfigs[netId][lGrpId].STP_U = groupConfigMap[gGrpId].stpConfig.STP_U;
			groupConfigs[netId][lGrpId].STP_tau_u_inv = groupConfigMap[gGrpId].stpConfig.STP_tau_u_inv; 
			groupConfigs[netId][lGrpId].STP_tau_x_inv = groupConfigMap[gGrpId].stpConfig.STP_tau_x_inv;
			groupConfigs[netId][lGrpId].TAU_PLUS_INV_EXC = groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_EXC;
			groupConfigs[netId][lGrpId].TAU_MINUS_INV_EXC = groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_EXC;
			groupConfigs[netId][lGrpId].ALPHA_PLUS_EXC = groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_EXC;
			groupConfigs[netId][lGrpId].ALPHA_MINUS_EXC = groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_EXC;
			groupConfigs[netId][lGrpId].GAMMA = groupConfigMap[gGrpId].stdpConfig.GAMMA;
			groupConfigs[netId][lGrpId].KAPPA = groupConfigMap[gGrpId].stdpConfig.KAPPA;
			groupConfigs[netId][lGrpId].OMEGA = groupConfigMap[gGrpId].stdpConfig.OMEGA;
			groupConfigs[netId][lGrpId].TAU_PLUS_INV_INB = groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_INB;
			groupConfigs[netId][lGrpId].TAU_MINUS_INV_INB = groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_INB;
			groupConfigs[netId][lGrpId].ALPHA_PLUS_INB = groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_INB;
			groupConfigs[netId][lGrpId].ALPHA_MINUS_INB = groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_INB;
			groupConfigs[netId][lGrpId].BETA_LTP = groupConfigMap[gGrpId].stdpConfig.BETA_LTP;
			groupConfigs[netId][lGrpId].BETA_LTD = groupConfigMap[gGrpId].stdpConfig.BETA_LTD;
			groupConfigs[netId][lGrpId].LAMBDA = groupConfigMap[gGrpId].stdpConfig.LAMBDA;
			groupConfigs[netId][lGrpId].DELTA = groupConfigMap[gGrpId].stdpConfig.DELTA;

			groupConfigs[netId][lGrpId].numCompNeighbors = 0;
			groupConfigs[netId][lGrpId].withCompartments = groupConfigMap[gGrpId].withCompartments;
			groupConfigs[netId][lGrpId].compCouplingUp = groupConfigMap[gGrpId].compCouplingUp;
			groupConfigs[netId][lGrpId].compCouplingDown = groupConfigMap[gGrpId].compCouplingDown;
			memset(&groupConfigs[netId][lGrpId].compNeighbors, 0, sizeof(groupConfigs[netId][lGrpId].compNeighbors[0])*MAX_NUM_COMP_CONN);
			memset(&groupConfigs[netId][lGrpId].compCoupling, 0, sizeof(groupConfigs[netId][lGrpId].compCoupling[0])*MAX_NUM_COMP_CONN);

			//!< homeostatic plasticity variables
			groupConfigs[netId][lGrpId].avgTimeScale = groupConfigMap[gGrpId].homeoConfig.avgTimeScale;
			groupConfigs[netId][lGrpId].avgTimeScale_decay = groupConfigMap[gGrpId].homeoConfig.avgTimeScaleDecay;
			groupConfigs[netId][lGrpId].avgTimeScaleInv = groupConfigMap[gGrpId].homeoConfig.avgTimeScaleInv;
			groupConfigs[netId][lGrpId].homeostasisScale = groupConfigMap[gGrpId].homeoConfig.homeostasisScale;

			// parameters of neuromodulator
			groupConfigs[netId][lGrpId].baseDP = groupConfigMap[gGrpId].neuromodulatorConfig.baseDP;
			groupConfigs[netId][lGrpId].base5HT = groupConfigMap[gGrpId].neuromodulatorConfig.base5HT;
			groupConfigs[netId][lGrpId].baseACh = groupConfigMap[gGrpId].neuromodulatorConfig.baseACh;
			groupConfigs[netId][lGrpId].baseNE = groupConfigMap[gGrpId].neuromodulatorConfig.baseNE;
			groupConfigs[netId][lGrpId].decayDP = groupConfigMap[gGrpId].neuromodulatorConfig.decayDP;
			groupConfigs[netId][lGrpId].decay5HT = groupConfigMap[gGrpId].neuromodulatorConfig.decay5HT;
			groupConfigs[netId][lGrpId].decayACh = groupConfigMap[gGrpId].neuromodulatorConfig.decayACh;
			groupConfigs[netId][lGrpId].decayNE = groupConfigMap[gGrpId].neuromodulatorConfig.decayNE;

			// sync groupConfigs[][] and groupConfigMDMap[]
			if (netId == grpIt->netId) {
				groupConfigMDMap[gGrpId].netId = grpIt->netId;
				groupConfigMDMap[gGrpId].gGrpId = grpIt->gGrpId;
				groupConfigMDMap[gGrpId].gStartN = grpIt->gStartN;
				groupConfigMDMap[gGrpId].gEndN = grpIt->gEndN;
				groupConfigMDMap[gGrpId].lGrpId = grpIt->lGrpId;
				groupConfigMDMap[gGrpId].lStartN = grpIt->lStartN;
				groupConfigMDMap[gGrpId].lEndN = grpIt->lEndN;
				groupConfigMDMap[gGrpId].numPostSynapses = grpIt->numPostSynapses;
				groupConfigMDMap[gGrpId].numPreSynapses = grpIt->numPreSynapses;
				groupConfigMDMap[gGrpId].LtoGOffset = grpIt->LtoGOffset;
				groupConfigMDMap[gGrpId].GtoLOffset = grpIt->GtoLOffset;
				groupConfigMDMap[gGrpId].fixedInputWts = grpIt->fixedInputWts;
				groupConfigMDMap[gGrpId].hasExternalConnect = grpIt->hasExternalConnect;
				groupConfigMDMap[gGrpId].Noffset = grpIt->Noffset; // Note: Noffset is not valid at this time
				groupConfigMDMap[gGrpId].maxOutgoingDelay = grpIt->maxOutgoingDelay;
			}
			groupConfigs[netId][lGrpId].withParamModel_9 = groupConfigMap[gGrpId].withParamModel_9;
			groupConfigs[netId][lGrpId].isLIF = groupConfigMap[gGrpId].isLIF;

		}

		// FIXME: How does networkConfigs[netId].numGroups be availabe at this time?! Bug?!
		//int numNSpikeGen = 0;
		//for(int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
		//	if (netId == groupConfigs[netId][lGrpId].netId && groupConfigs[netId][lGrpId].isSpikeGenerator && groupConfigs[netId][lGrpId].isSpikeGenFunc) {
		//	// we only need numNSpikeGen for spike generator callbacks that need to transfer their spikes to the GPU
		//		groupConfigs[netId][lGrpId].Noffset = numNSpikeGen; // FIXME, Noffset is updated after publish group configs
		//		numNSpikeGen += groupConfigs[netId][lGrpId].numN;
		//	}
		//}
		//assert(numNSpikeGen <= networkConfigs[netId].numNPois);
	}
}

void SNN::generateRuntimeConnectConfigs() {
	// sync localConnectLists and connectConfigMap
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		for (std::list<ConnectConfig>::iterator connIt = localConnectLists[netId].begin(); connIt != localConnectLists[netId].end(); connIt++) {
			connectConfigMap[connIt->connId] = *connIt;
		}

		for (std::list<ConnectConfig>::iterator connIt = externalConnectLists[netId].begin(); connIt != externalConnectLists[netId].end(); connIt++) {
			connectConfigMap[connIt->connId] = *connIt;
		}
	}
}

void SNN::generateRuntimeNetworkConfigs() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			// copy the global network config to local network configs
			// global configuration for maximum axonal delay
			networkConfigs[netId].maxDelay  = glbNetworkConfig.maxDelay;
	
			// configurations for execution features
			networkConfigs[netId].sim_with_fixedwts = sim_with_fixedwts;
			networkConfigs[netId].sim_with_conductances = sim_with_conductances;
			networkConfigs[netId].sim_with_homeostasis = sim_with_homeostasis;
			networkConfigs[netId].sim_with_stdp = sim_with_stdp;
			networkConfigs[netId].sim_with_stp = sim_with_stp;
			networkConfigs[netId].sim_in_testing = sim_in_testing;

			// search for active neuron monitor
			networkConfigs[netId].sim_with_nm = false;
			for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
				if (grpIt->netId == netId && grpIt->neuronMonitorId >= 0)
					networkConfigs[netId].sim_with_nm = true;
			}

			// stdp, da-stdp configurations
			networkConfigs[netId].stdpScaleFactor = stdpScaleFactor_;
			networkConfigs[netId].wtChangeDecay = wtChangeDecay_;

			// conductance configurations
			networkConfigs[netId].sim_with_NMDA_rise = sim_with_NMDA_rise;
			networkConfigs[netId].sim_with_GABAb_rise = sim_with_GABAb_rise;
			networkConfigs[netId].dAMPA = dAMPA;
			networkConfigs[netId].rNMDA = rNMDA;
			networkConfigs[netId].dNMDA = dNMDA;
			networkConfigs[netId].sNMDA = sNMDA;
			networkConfigs[netId].dGABAa = dGABAa;
			networkConfigs[netId].rGABAb = rGABAb;
			networkConfigs[netId].dGABAb = dGABAb;
			networkConfigs[netId].sGABAb = sGABAb;

			networkConfigs[netId].simIntegrationMethod = glbNetworkConfig.simIntegrationMethod;
			networkConfigs[netId].simNumStepsPerMs = glbNetworkConfig.simNumStepsPerMs;
			networkConfigs[netId].timeStep = glbNetworkConfig.timeStep;

			// configurations for boundries of neural types
			findNumN(netId, networkConfigs[netId].numN, networkConfigs[netId].numNExternal, networkConfigs[netId].numNAssigned,
					 networkConfigs[netId].numNReg, networkConfigs[netId].numNExcReg, networkConfigs[netId].numNInhReg,
					 networkConfigs[netId].numNPois, networkConfigs[netId].numNExcPois, networkConfigs[netId].numNInhPois);

			// configurations for assigned groups and connections
			networkConfigs[netId].numGroups = 0;
			for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
				if (grpIt->netId == netId)
					networkConfigs[netId].numGroups++;
			}
			networkConfigs[netId].numGroupsAssigned = groupPartitionLists[netId].size();
			//networkConfigs[netId].numConnections = localConnectLists[netId].size();
			//networkConfigs[netId].numAssignedConnections = localConnectLists[netId].size() + externalConnectLists[netId].size();
			//networkConfigs[netId].numConnections = localConnectLists[netId].size() + externalConnectLists[netId].size();
			networkConfigs[netId].numConnections = connectConfigMap.size();// temporarily solution: copy all connection info to each GPU

			// find the maximum number of pre- and post-connections among neurons
			// SNN::maxNumPreSynN and SNN::maxNumPostSynN are updated
			findMaxNumSynapsesNeurons(netId, networkConfigs[netId].maxNumPostSynN, networkConfigs[netId].maxNumPreSynN);

			// find the maximum number of spikes in D1 (i.e., maxDelay == 1) and D2 (i.e., maxDelay >= 2) sets
			findMaxSpikesD1D2(netId, networkConfigs[netId].maxSpikesD1, networkConfigs[netId].maxSpikesD2);

			// find the total number of synapses in the network
			findNumSynapsesNetwork(netId, networkConfigs[netId].numPostSynNet, networkConfigs[netId].numPreSynNet);

			// find out number of user-defined spike gen and update Noffset of each group config
			// Note: groupConfigs[][].Noffset is valid at this time 
			findNumNSpikeGenAndOffset(netId);
		}
	}

	// find manager runtime data size, which is sufficient to hold the data of any gpu runtime
	memset(&managerRTDSize, 0, sizeof(ManagerRuntimeDataSize));
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			// find the maximum number of numN, numNReg ,and numNAssigned among local networks
			if (networkConfigs[netId].numNReg > managerRTDSize.maxNumNReg) managerRTDSize.maxNumNReg = networkConfigs[netId].numNReg;
			if (networkConfigs[netId].numN > managerRTDSize.maxNumN) managerRTDSize.maxNumN = networkConfigs[netId].numN;
			if (networkConfigs[netId].numNAssigned > managerRTDSize.maxNumNAssigned) managerRTDSize.maxNumNAssigned = networkConfigs[netId].numNAssigned;

			// find the maximum number of numNSpikeGen among local networks
			if (networkConfigs[netId].numNSpikeGen > managerRTDSize.maxNumNSpikeGen) managerRTDSize.maxNumNSpikeGen = networkConfigs[netId].numNSpikeGen;
			
			// find the maximum number of numGroups and numConnections among local networks
			if (networkConfigs[netId].numGroups > managerRTDSize.maxNumGroups) managerRTDSize.maxNumGroups = networkConfigs[netId].numGroups;
			if (networkConfigs[netId].numConnections > managerRTDSize.maxNumConnections) managerRTDSize.maxNumConnections = networkConfigs[netId].numConnections;
			
			// find the maximum number of neurons in a group among local networks
			for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
				if (groupConfigMap[grpIt->gGrpId].numN > managerRTDSize.maxNumNPerGroup) managerRTDSize.maxNumNPerGroup = groupConfigMap[grpIt->gGrpId].numN;
			}
			
			// find the maximum number of maxSipkesD1(D2) among networks
			if (networkConfigs[netId].maxSpikesD1 > managerRTDSize.maxMaxSpikeD1) managerRTDSize.maxMaxSpikeD1 = networkConfigs[netId].maxSpikesD1;
			if (networkConfigs[netId].maxSpikesD2 > managerRTDSize.maxMaxSpikeD2) managerRTDSize.maxMaxSpikeD2 = networkConfigs[netId].maxSpikesD2;
			
			// find the maximum number of total # of pre- and post-connections among local networks
			if (networkConfigs[netId].numPreSynNet > managerRTDSize.maxNumPreSynNet) managerRTDSize.maxNumPreSynNet = networkConfigs[netId].numPreSynNet;
			if (networkConfigs[netId].numPostSynNet > managerRTDSize.maxNumPostSynNet) managerRTDSize.maxNumPostSynNet = networkConfigs[netId].numPostSynNet;

			// find the number of numN, and numNReg in the global network
			managerRTDSize.glbNumN += networkConfigs[netId].numN;
			managerRTDSize.glbNumNReg += networkConfigs[netId].numNReg;
		}
	}
}

bool compareSrcNeuron(const ConnectionInfo& first, const ConnectionInfo& second) {
	return (first.nSrc + first.srcGLoffset < second.nSrc + second.srcGLoffset);
}

bool compareDelay(const ConnectionInfo& first, const ConnectionInfo& second) {
	return (first.delay < second.delay);
}

// Note: ConnectInfo stored in connectionList use global ids
void SNN::generateConnectionRuntime(int netId) {
	std::map<int, int> GLoffset; // global nId to local nId offset
	std::map<int, int> GLgrpId; // global grpId to local grpId offset

	// load offset between global neuron id and local neuron id 
	for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
		GLoffset[grpIt->gGrpId] = grpIt->GtoLOffset;
		GLgrpId[grpIt->gGrpId] = grpIt->lGrpId;
	}
	// FIXME: connId is global connId, use connectConfigs[netId][local connId] instead,
	// FIXME; but note connectConfigs[netId][] are NOT complete, lack of exeternal incoming connections
	// generate mulSynFast, mulSynSlow in connection-centric array
	for (std::map<int, ConnectConfig>::iterator connIt = connectConfigMap.begin(); connIt != connectConfigMap.end(); connIt++) {
		// store scaling factors for synaptic currents in connection-centric array
		mulSynFast[connIt->second.connId] = connIt->second.mulSynFast;
		mulSynSlow[connIt->second.connId] = connIt->second.mulSynSlow;
	}

	// parse ConnectionInfo stored in connectionLists[0]
	// note: ConnectInfo stored in connectionList use global ids
	// generate Npost, Npre, Npre_plastic
	int parsedConnections = 0;
	memset(managerRuntimeData.Npost, 0, sizeof(short) * networkConfigs[netId].numNAssigned);
	memset(managerRuntimeData.Npre, 0, sizeof(short) * networkConfigs[netId].numNAssigned);
	memset(managerRuntimeData.Npre_plastic, 0, sizeof(short) * networkConfigs[netId].numNAssigned);
	for (std::list<ConnectionInfo>::iterator connIt = connectionLists[netId].begin(); connIt != connectionLists[netId].end(); connIt++) {
		connIt->srcGLoffset = GLoffset[connIt->grpSrc];
		if (managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]] == SYNAPSE_ID_MASK) {
			KERNEL_ERROR("Error: the number of synapses exceeds maximum limit (%d) for neuron %d (group %d)", SYNAPSE_ID_MASK, connIt->nSrc, connIt->grpSrc);
			exitSimulation(ID_OVERFLOW_ERROR);
		}
		if (managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]] == SYNAPSE_ID_MASK) {
			KERNEL_ERROR("Error: the number of synapses exceeds maximum limit (%d) for neuron %d (group %d)", SYNAPSE_ID_MASK, connIt->nDest, connIt->grpDest);
			exitSimulation(ID_OVERFLOW_ERROR);
		}
		managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]]++;
		managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]]++;

		if (GET_FIXED_PLASTIC(connectConfigMap[connIt->connId].connProp) == SYN_PLASTIC) {
			sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
			managerRuntimeData.Npre_plastic[connIt->nDest + GLoffset[connIt->grpDest]]++;

			// homeostasis
			if (groupConfigMap[connIt->grpDest].homeoConfig.WithHomeostasis && groupConfigMDMap[connIt->grpDest].homeoId == -1)
				groupConfigMDMap[connIt->grpDest].homeoId = connIt->nDest + GLoffset[connIt->grpDest]; // this neuron info will be printed

			// old access to homeostasis
			//if (groupConfigs[netId][GLgrpId[it->grpDest]].WithHomeostasis && groupConfigs[netId][GLgrpId[it->grpDest]].homeoId == -1)
			//	groupConfigs[netId][GLgrpId[it->grpDest]].homeoId = it->nDest + GLoffset[it->grpDest]; // this neuron info will be printed
		}

		// generate the delay vaule
		//it->delay = connectConfigMap[it->connId].minDelay + rand() % (connectConfigMap[it->connId].maxDelay - connectConfigMap[it->connId].minDelay + 1);
		//assert((it->delay >= connectConfigMap[it->connId].minDelay) && (it->delay <= connectConfigMap[it->connId].maxDelay));
		// generate the max weight and initial weight
		//float initWt = generateWeight(connectConfigMap[it->connId].connProp, connectConfigMap[it->connId].initWt, connectConfigMap[it->connId].maxWt, it->nSrc, it->grpSrc);
		//float initWt = connectConfigMap[it->connId].initWt;
		//float maxWt = connectConfigMap[it->connId].maxWt;
		// adjust sign of weight based on pre-group (negative if pre is inhibitory)
		// this access is fine, isExcitatoryGroup() use global grpId
		//it->maxWt = isExcitatoryGroup(it->grpSrc) ? fabs(maxWt) : -1.0 * fabs(maxWt);
		//it->initWt = isExcitatoryGroup(it->grpSrc) ? fabs(initWt) : -1.0 * fabs(initWt);

		parsedConnections++;
	}
	assert(parsedConnections == networkConfigs[netId].numPostSynNet && parsedConnections == networkConfigs[netId].numPreSynNet);

	// generate cumulativePost and cumulativePre
	managerRuntimeData.cumulativePost[0] = 0;
	managerRuntimeData.cumulativePre[0] = 0;
	for (int lNId = 1; lNId < networkConfigs[netId].numNAssigned; lNId++) {
		managerRuntimeData.cumulativePost[lNId] = managerRuntimeData.cumulativePost[lNId - 1] + managerRuntimeData.Npost[lNId - 1];
		managerRuntimeData.cumulativePre[lNId] = managerRuntimeData.cumulativePre[lNId - 1] + managerRuntimeData.Npre[lNId - 1];
	}

	// generate preSynapticIds, parse plastic connections first
	memset(managerRuntimeData.Npre, 0, sizeof(short) * networkConfigs[netId].numNAssigned); // reset managerRuntimeData.Npre to zero, so that it can be used as synId
	parsedConnections = 0;
	for (std::list<ConnectionInfo>::iterator connIt = connectionLists[netId].begin(); connIt != connectionLists[netId].end(); connIt++) {
		if (GET_FIXED_PLASTIC(connectConfigMap[connIt->connId].connProp) == SYN_PLASTIC) {
			int pre_pos = managerRuntimeData.cumulativePre[connIt->nDest + GLoffset[connIt->grpDest]] + managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]];
			assert(pre_pos < networkConfigs[netId].numPreSynNet);

			managerRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID((connIt->nSrc + GLoffset[connIt->grpSrc]), 0, (GLgrpId[connIt->grpSrc])); // managerRuntimeData.Npost[it->nSrc] is not availabe at this parse
			connIt->preSynId = managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]]; // save managerRuntimeData.Npre[it->nDest] as synId

			managerRuntimeData.Npre[connIt->nDest+ GLoffset[connIt->grpDest]]++;
			parsedConnections++;

			// update the maximum number of and pre-connections of a neuron in a group
			//if (managerRuntimeData.Npre[it->nDest] > groupInfo[it->grpDest].maxPreConn)
			//	groupInfo[it->grpDest].maxPreConn = managerRuntimeData.Npre[it->nDest];
		}
	}
	// parse fixed connections
	for (std::list<ConnectionInfo>::iterator connIt = connectionLists[netId].begin(); connIt != connectionLists[netId].end(); connIt++) {
		if (GET_FIXED_PLASTIC(connectConfigMap[connIt->connId].connProp) == SYN_FIXED) {
			int pre_pos = managerRuntimeData.cumulativePre[connIt->nDest + GLoffset[connIt->grpDest]] + managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]];
			assert(pre_pos < networkConfigs[netId].numPreSynNet);

			managerRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID((connIt->nSrc + GLoffset[connIt->grpSrc]), 0, (GLgrpId[connIt->grpSrc])); // managerRuntimeData.Npost[it->nSrc] is not availabe at this parse
			connIt->preSynId = managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]]; // save managerRuntimeData.Npre[it->nDest] as synId

			managerRuntimeData.Npre[connIt->nDest + GLoffset[connIt->grpDest]]++;
			parsedConnections++;

			// update the maximum number of and pre-connections of a neuron in a group
			//if (managerRuntimeData.Npre[it->nDest] > groupInfo[it->grpDest].maxPreConn)
			//	groupInfo[it->grpDest].maxPreConn = managerRuntimeData.Npre[it->nDest];
		}
	}
	assert(parsedConnections == networkConfigs[netId].numPreSynNet);
	//printf("parsed pre connections %d\n", parsedConnections);

	// generate postSynapticIds
	connectionLists[netId].sort(compareSrcNeuron); // sort by local nSrc id
	memset(managerRuntimeData.postDelayInfo, 0, sizeof(DelayInfo) * (networkConfigs[netId].numNAssigned * (glbNetworkConfig.maxDelay + 1)));
	for (int lNId = 0; lNId < networkConfigs[netId].numNAssigned; lNId++) { // pre-neuron order, local nId
		if (managerRuntimeData.Npost[lNId] > 0) {
			std::list<ConnectionInfo> postConnectionList;
			ConnectionInfo targetConn;
			targetConn.nSrc = lNId ; // the other fields does not matter, use local nid to search
			
			std::list<ConnectionInfo>::iterator firstPostConn = std::find(connectionLists[netId].begin(), connectionLists[netId].end(), targetConn);
			std::list<ConnectionInfo>::iterator lastPostConn = firstPostConn;
			std::advance(lastPostConn, managerRuntimeData.Npost[lNId]);
			managerRuntimeData.Npost[lNId] = 0; // reset managerRuntimeData.Npost[lNId] to zero, so that it can be used as synId

			postConnectionList.splice(postConnectionList.begin(), connectionLists[netId], firstPostConn, lastPostConn);
			postConnectionList.sort(compareDelay);

			int post_pos, pre_pos, lastDelay = 0;
			parsedConnections = 0;
			//memset(&managerRuntimeData.postDelayInfo[lNId * (glbNetworkConfig.maxDelay + 1)], 0, sizeof(DelayInfo) * (glbNetworkConfig.maxDelay + 1));
			for (std::list<ConnectionInfo>::iterator connIt = postConnectionList.begin(); connIt != postConnectionList.end(); connIt++) {
				assert(connIt->nSrc + GLoffset[connIt->grpSrc] == lNId);
				post_pos = managerRuntimeData.cumulativePost[connIt->nSrc + GLoffset[connIt->grpSrc]] + managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]];
				pre_pos  = managerRuntimeData.cumulativePre[connIt->nDest + GLoffset[connIt->grpDest]] + connIt->preSynId;

				assert(post_pos < networkConfigs[netId].numPostSynNet);
				//assert(pre_pos  < numPreSynNet);

				// generate a post synaptic id for the current connection
				managerRuntimeData.postSynapticIds[post_pos] = SET_CONN_ID((connIt->nDest + GLoffset[connIt->grpDest]), connIt->preSynId, (GLgrpId[connIt->grpDest]));// used stored managerRuntimeData.Npre[it->nDest] in it->preSynId
				// generate a delay look up table by the way
				assert(connIt->delay > 0);
				if (connIt->delay > lastDelay) {
					managerRuntimeData.postDelayInfo[lNId * (glbNetworkConfig.maxDelay + 1) + connIt->delay - 1].delay_index_start = managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]];
					managerRuntimeData.postDelayInfo[lNId * (glbNetworkConfig.maxDelay + 1) + connIt->delay - 1].delay_length++;
				} else if (connIt->delay == lastDelay) {
					managerRuntimeData.postDelayInfo[lNId * (glbNetworkConfig.maxDelay + 1) + connIt->delay - 1].delay_length++;
				} else {
					KERNEL_ERROR("Post-synaptic delays not sorted correctly... pre_id=%d, delay[%d]=%d, delay[%d]=%d",
						lNId, managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]], connIt->delay, managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]] - 1, lastDelay);
				}
				lastDelay = connIt->delay;

				// update the corresponding pre synaptic id
				SynInfo preId = managerRuntimeData.preSynapticIds[pre_pos];
				assert(GET_CONN_NEURON_ID(preId) == connIt->nSrc + GLoffset[connIt->grpSrc]);
				//assert(GET_CONN_GRP_ID(preId) == it->grpSrc);
				managerRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID((connIt->nSrc + GLoffset[connIt->grpSrc]), managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]], (GLgrpId[connIt->grpSrc]));
				managerRuntimeData.wt[pre_pos] = connIt->initWt;
				managerRuntimeData.maxSynWt[pre_pos] = connIt->maxWt;
				managerRuntimeData.connIdsPreIdx[pre_pos] = connIt->connId;

				managerRuntimeData.Npost[connIt->nSrc + GLoffset[connIt->grpSrc]]++;
				parsedConnections++;

				// update the maximum number of and post-connections of a neuron in a group
				//if (managerRuntimeData.Npost[it->nSrc] > groupInfo[it->grpSrc].maxPostConn)
				//	groupInfo[it->grpSrc].maxPostConn = managerRuntimeData.Npost[it->nSrc];
			}
			assert(parsedConnections == managerRuntimeData.Npost[lNId]);
			//printf("parsed post connections %d\n", parsedConnections);
			// note: elements in postConnectionList are deallocated automatically with postConnectionList
			/* for postDelayInfo debugging
			printf("%d ", lNId);
			for (int t = 0; t < maxDelay_ + 1; t ++) {
				printf("[%d,%d]",
					managerRuntimeData.postDelayInfo[lNId * (maxDelay_ + 1) + t].delay_index_start,
					managerRuntimeData.postDelayInfo[lNId * (maxDelay_ + 1) + t].delay_length);
			}
			printf("\n");
			*/
		}
	}
	assert(connectionLists[netId].empty());

	//int p = managerRuntimeData.Npost[src];

	//assert(managerRuntimeData.Npost[src] >= 0);
	//assert(managerRuntimeData.Npre[dest] >= 0);
	//assert((src * maxNumPostSynGrp + p) / numN < maxNumPostSynGrp); // divide by numN to prevent INT overflow

	//unsigned int post_pos = managerRuntimeData.cumulativePost[src] + managerRuntimeData.Npost[src];
	//unsigned int pre_pos  = managerRuntimeData.cumulativePre[dest] + managerRuntimeData.Npre[dest];

	//assert(post_pos < numPostSynNet);
	//assert(pre_pos  < numPreSynNet);

	////generate a new postSynapticIds id for the current connection
	//managerRuntimeData.postSynapticIds[post_pos]   = SET_CONN_ID(dest, managerRuntimeData.Npre[dest], destGrp);
	//tmp_SynapticDelay[post_pos] = dVal;

	//managerRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID(src, managerRuntimeData.Npost[src], srcGrp);
	//managerRuntimeData.wt[pre_pos] 	  = synWt;
	//managerRuntimeData.maxSynWt[pre_pos] = maxWt;
	//managerRuntimeData.connIdsPreIdx[pre_pos] = connId;

	//bool synWtType = GET_FIXED_PLASTIC(connProp);

	//if (synWtType == SYN_PLASTIC) {
	//	sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
	//	managerRuntimeData.Npre_plastic[dest]++;
	//	// homeostasis
	//	if (groupConfigs[0][destGrp].WithHomeostasis && groupConfigs[0][destGrp].homeoId ==-1)
	//		groupConfigs[0][destGrp].homeoId = dest; // this neuron info will be printed
	//}

	//managerRuntimeData.Npre[dest] += 1;
	//managerRuntimeData.Npost[src] += 1;

	//groupInfo[srcGrp].numPostConn++;
	//groupInfo[destGrp].numPreConn++;

	//// update the maximum number of pre- and post-connections of a neuron in a group
	//if (managerRuntimeData.Npost[src] > groupInfo[srcGrp].maxPostConn)
	//	groupInfo[srcGrp].maxPostConn = managerRuntimeData.Npost[src];
	//if (managerRuntimeData.Npre[dest] > groupInfo[destGrp].maxPreConn)
	//	groupInfo[destGrp].maxPreConn = managerRuntimeData.Npre[src];
}

void SNN::generateCompConnectionRuntime(int netId)
{
	std::map<int, int> GLgrpId; // global grpId to local grpId offset

	for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
		GLgrpId[grpIt->gGrpId] = grpIt->lGrpId;
		//printf("Global group id %i; Local group id %i\n", grpIt->gGrpId, grpIt->lGrpId);
	}

	//printf("The current netid is: %i\n", netId);

	for (std::list<compConnectConfig>::iterator connIt = localCompConnectLists[netId].begin(); connIt != localCompConnectLists[netId].end(); connIt++) {
		//printf("The size of localCompConnectLists is: %i\n", localCompConnectLists[netId].size());
		int grpLower = connIt->grpSrc;
		int grpUpper = connIt->grpDest;

		int i = groupConfigs[netId][GLgrpId[grpLower]].numCompNeighbors;
		if (i >= MAX_NUM_COMP_CONN) {
			KERNEL_ERROR("Group %s(%d) exceeds max number of allowed compartmental connections (%d).",
				groupConfigMap[grpLower].grpName.c_str(), grpLower, (int)MAX_NUM_COMP_CONN);
			exitSimulation(1);
		}
		groupConfigs[netId][GLgrpId[grpLower]].compNeighbors[i] = grpUpper;
		groupConfigs[netId][GLgrpId[grpLower]].compCoupling[i] = groupConfigs[netId][GLgrpId[grpUpper]].compCouplingDown; // get down-coupling from upper neighbor
		groupConfigs[netId][GLgrpId[grpLower]].numCompNeighbors++;

		int j = groupConfigs[netId][GLgrpId[grpUpper]].numCompNeighbors;
		if (j >= MAX_NUM_COMP_CONN) {
			KERNEL_ERROR("Group %s(%d) exceeds max number of allowed compartmental connections (%d).",
				groupConfigMap[grpUpper].grpName.c_str(), grpUpper, (int)MAX_NUM_COMP_CONN);
			exitSimulation(1);
		}
		groupConfigs[netId][GLgrpId[grpUpper]].compNeighbors[j] = grpLower;
		groupConfigs[netId][GLgrpId[grpUpper]].compCoupling[j] = groupConfigs[netId][GLgrpId[grpLower]].compCouplingUp; // get up-coupling from lower neighbor
		groupConfigs[netId][GLgrpId[grpUpper]].numCompNeighbors++;

		//printf("Group %i (local group %i) has %i compartmental neighbors!\n", grpUpper, GLgrpId[grpUpper], groupConfigs[netId][GLgrpId[grpUpper]].numCompNeighbors);
	}
}


void SNN::generatePoissonGroupRuntime(int netId, int lGrpId) {
	for(int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++)
		resetPoissonNeuron(netId, lGrpId, lNId);
}


void SNN::collectGlobalNetworkConfigC() {
	// scan all connect configs to find the maximum delay in the global network, update glbNetworkConfig.maxDelay
	for (std::map<int, ConnectConfig>::iterator connIt = connectConfigMap.begin(); connIt != connectConfigMap.end(); connIt++) {
		if (connIt->second.maxDelay > glbNetworkConfig.maxDelay)
			glbNetworkConfig.maxDelay = connIt->second.maxDelay;
	}
	assert(connectConfigMap.size() > 0 || glbNetworkConfig.maxDelay != -1);

	// scan all group configs to find the number of (reg, pois, exc, inh) neuron in the global network
	for(int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
		if (IS_EXCITATORY_TYPE(groupConfigMap[gGrpId].type) && (groupConfigMap[gGrpId].type & POISSON_NEURON)) {
			glbNetworkConfig.numNExcPois += groupConfigMap[gGrpId].numN;
		} else if (IS_INHIBITORY_TYPE(groupConfigMap[gGrpId].type) &&  (groupConfigMap[gGrpId].type & POISSON_NEURON)) {
			glbNetworkConfig.numNInhPois += groupConfigMap[gGrpId].numN;
		} else if (IS_EXCITATORY_TYPE(groupConfigMap[gGrpId].type) && !(groupConfigMap[gGrpId].type & POISSON_NEURON)) {
			glbNetworkConfig.numNExcReg += groupConfigMap[gGrpId].numN;
		} else if (IS_INHIBITORY_TYPE(groupConfigMap[gGrpId].type) && !(groupConfigMap[gGrpId].type & POISSON_NEURON)) {
			glbNetworkConfig.numNInhReg += groupConfigMap[gGrpId].numN;
		}

		if (groupConfigMDMap[gGrpId].maxOutgoingDelay == 1)
			glbNetworkConfig.numN1msDelay += groupConfigMap[gGrpId].numN;
		else if (groupConfigMDMap[gGrpId].maxOutgoingDelay >= 2)
			glbNetworkConfig.numN2msDelay += groupConfigMap[gGrpId].numN;
	}

	glbNetworkConfig.numNReg = glbNetworkConfig.numNExcReg + glbNetworkConfig.numNInhReg;
	glbNetworkConfig.numNPois = glbNetworkConfig.numNExcPois + glbNetworkConfig.numNInhPois;
	glbNetworkConfig.numN = glbNetworkConfig.numNReg + glbNetworkConfig.numNPois;
}


void SNN::collectGlobalNetworkConfigP() {
	// print group and connection overview
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!localConnectLists[netId].empty() || !externalConnectLists[netId].empty()) {
			for (std::list<ConnectConfig>::iterator connIt = localConnectLists[netId].begin(); connIt != localConnectLists[netId].end(); connIt++)
				glbNetworkConfig.numSynNet += connIt->numberOfConnections;

			for (std::list<ConnectConfig>::iterator connIt = externalConnectLists[netId].begin(); connIt != externalConnectLists[netId].end(); connIt++)
				glbNetworkConfig.numSynNet += connIt->numberOfConnections;
		}
	}
}

// after all the initalization. Its time to create the synaptic weights, weight change and also
// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
void SNN::compileSNN() {
	KERNEL_DEBUG("Beginning compilation of the network....");

	// compile (update) group and connection configs according to their mutual information
	// update GroupConfig::MaxDelay GroupConfig::FixedInputWts
	// assign GroupConfig::StartN and GroupConfig::EndN
	// Note: MaxDelay, FixedInputWts, StartN, and EndN are invariant in single-GPU or multi-GPUs mode
	compileGroupConfig();

	compileConnectConfig(); // for future use

	// collect the global network config according to compiled gorup and connection configs
	// collect SNN::maxDelay_
	collectGlobalNetworkConfigC();

	// perform various consistency checks:
	// - numNeurons vs. sum of all neurons
	// - STDP set on a post-group with incoming plastic connections
	// - etc.
	verifyNetwork();

	// display the global network configuration
	KERNEL_INFO("\n");
	KERNEL_INFO("************************** Global Network Configuration *******************************");
	KERNEL_INFO("The number of neurons in the network (numN) = %d", glbNetworkConfig.numN);
	KERNEL_INFO("The number of regular neurons in the network (numNReg:numNExcReg:numNInhReg) = %d:%d:%d", glbNetworkConfig.numNReg, glbNetworkConfig.numNExcReg, glbNetworkConfig.numNInhReg);
	KERNEL_INFO("The number of poisson neurons in the network (numNPois:numNExcPois:numInhPois) = %d:%d:%d", glbNetworkConfig.numNPois, glbNetworkConfig.numNExcPois, glbNetworkConfig.numNInhPois);
	KERNEL_INFO("The maximum axonal delay in the network (maxDelay) = %d", glbNetworkConfig.maxDelay);

	//ensure that we dont compile the network again
	snnState = COMPILED_SNN;
}

void SNN::compileConnectConfig() {
	// for future  use
}

void SNN::compileGroupConfig() {
	int grpSrc;
	bool synWtType;

	// find the maximum delay for each group according to incoming connection
	for (std::map<int, ConnectConfig>::iterator connIt = connectConfigMap.begin(); connIt != connectConfigMap.end(); connIt++) {
		// check if the current connection's delay meaning grpSrc's delay
		// is greater than the MaxDelay for grpSrc. We find the maximum
		// delay for the grpSrc by this scheme.
		grpSrc = connIt->second.grpSrc;
		if (connIt->second.maxDelay > groupConfigMDMap[grpSrc].maxOutgoingDelay)
		 	groupConfigMDMap[grpSrc].maxOutgoingDelay = connIt->second.maxDelay;

		// given group has plastic connection, and we need to apply STDP rule...
		synWtType = GET_FIXED_PLASTIC(connIt->second.connProp);
		if (synWtType == SYN_PLASTIC) {
			groupConfigMDMap[connIt->second.grpDest].fixedInputWts = false;
		}
	}

	// assigned global neruon ids to each group in the order...
	//    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
	//     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
	//     Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
	int assignedGroup = 0;
	int availableNeuronId = 0;
	for(int order = 0; order < 4; order++) {
		for(int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			if (IS_EXCITATORY_TYPE(groupConfigMap[gGrpId].type) && (groupConfigMap[gGrpId].type & POISSON_NEURON) && order == 3) {
				availableNeuronId = assignGroup(gGrpId, availableNeuronId);
				assignedGroup++;
			} else if (IS_INHIBITORY_TYPE(groupConfigMap[gGrpId].type) &&  (groupConfigMap[gGrpId].type & POISSON_NEURON) && order == 2) {
				availableNeuronId = assignGroup(gGrpId, availableNeuronId);
				assignedGroup++;
			} else if (IS_EXCITATORY_TYPE(groupConfigMap[gGrpId].type) && !(groupConfigMap[gGrpId].type & POISSON_NEURON) && order == 0) {
				availableNeuronId = assignGroup(gGrpId, availableNeuronId);
				assignedGroup++;
			} else if (IS_INHIBITORY_TYPE(groupConfigMap[gGrpId].type) && !(groupConfigMap[gGrpId].type & POISSON_NEURON) && order == 1) {
				availableNeuronId = assignGroup(gGrpId, availableNeuronId);
				assignedGroup++;
			}
		}
	}
	//assert(availableNeuronId == numN);
	assert(assignedGroup == numGroups);
}

void SNN::connectNetwork() {
	// this parse generates local connections
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		for (std::list<ConnectConfig>::iterator connIt = localConnectLists[netId].begin(); connIt != localConnectLists[netId].end(); connIt++) {
			switch(connIt->type) {
				case CONN_RANDOM:
					connectRandom(netId, connIt, false);
					break;
				case CONN_FULL:
					connectFull(netId, connIt, false);
					break;
				case CONN_FULL_NO_DIRECT:
					connectFull(netId, connIt, false);
					break;
				case CONN_ONE_TO_ONE:
					connectOneToOne(netId, connIt, false);
					break;
				case CONN_GAUSSIAN:
					connectGaussian(netId, connIt, false);
					break;
				case CONN_USER_DEFINED:
					connectUserDefined(netId, connIt, false);
					break;
				default:
					KERNEL_ERROR("Invalid connection type( should be 'random', 'full', 'full-no-direct', or 'one-to-one')");
					exitSimulation(-1);
			}
		}
	}

	// this parse generates external connections
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		for (std::list<ConnectConfig>::iterator connIt = externalConnectLists[netId].begin(); connIt != externalConnectLists[netId].end(); connIt++) {
			switch(connIt->type) {
				case CONN_RANDOM:
					connectRandom(netId, connIt, true);
					break;
				case CONN_FULL:
					connectFull(netId, connIt, true);
					break;
				case CONN_FULL_NO_DIRECT:
					connectFull(netId, connIt, true);
					break;
				case CONN_ONE_TO_ONE:
					connectOneToOne(netId, connIt, true);
					break;
				case CONN_GAUSSIAN:
					connectGaussian(netId, connIt, true);
					break;
				case CONN_USER_DEFINED:
					connectUserDefined(netId, connIt, true);
					break;
				default:
					KERNEL_ERROR("Invalid connection type( should be 'random', 'full', 'full-no-direct', or 'one-to-one')");
					exitSimulation(-1);
			}
		}
	}
}

//! set one specific connection from neuron id 'src' to neuron id 'dest'
inline void SNN::connectNeurons(int netId, int _grpSrc, int _grpDest, int _nSrc, int _nDest, short int _connId, int externalNetId) {
	//assert(destN <= CONN_SYN_NEURON_MASK); // total number of neurons is less than 1 million within a GPU
	ConnectionInfo connInfo;
	connInfo.grpSrc = _grpSrc;
	connInfo.grpDest = _grpDest;
	connInfo.nSrc = _nSrc;
	connInfo.nDest = _nDest;
	connInfo.srcGLoffset = 0;
	connInfo.connId = _connId;
	connInfo.preSynId = -1;
	connInfo.initWt = 0.0f;
	connInfo.maxWt = 0.0f;
	connInfo.delay = 0;

	// generate the delay vaule
	connInfo.delay = connectConfigMap[_connId].minDelay + rand() % (connectConfigMap[_connId].maxDelay - connectConfigMap[_connId].minDelay + 1);
	assert((connInfo.delay >= connectConfigMap[_connId].minDelay) && (connInfo.delay <= connectConfigMap[_connId].maxDelay));
	// generate the max weight and initial weight
	//float initWt = generateWeight(connectConfigMap[it->connId].connProp, connectConfigMap[it->connId].initWt, connectConfigMap[it->connId].maxWt, it->nSrc, it->grpSrc);
	float initWt = connectConfigMap[_connId].initWt;
	float maxWt = connectConfigMap[_connId].maxWt;
	// adjust sign of weight based on pre-group (negative if pre is inhibitory)
	// this access is fine, isExcitatoryGroup() use global grpId
	connInfo.maxWt = isExcitatoryGroup(_grpSrc) ? fabs(maxWt) : -1.0 * fabs(maxWt);
	connInfo.initWt = isExcitatoryGroup(_grpSrc) ? fabs(initWt) : -1.0 * fabs(initWt);

	connectionLists[netId].push_back(connInfo);

	// If the connection is external, copy the connection info to the external network
	if (externalNetId >= 0)
		connectionLists[externalNetId].push_back(connInfo);
}

//! set one specific connection from neuron id 'src' to neuron id 'dest'
inline void SNN::connectNeurons(int netId, int _grpSrc, int _grpDest, int _nSrc, int _nDest, short int _connId, float initWt, float maxWt, uint8_t delay, int externalNetId) {
	//assert(destN <= CONN_SYN_NEURON_MASK); // total number of neurons is less than 1 million within a GPU
	ConnectionInfo connInfo;
	connInfo.grpSrc = _grpSrc;
	connInfo.grpDest = _grpDest;
	connInfo.nSrc = _nSrc;
	connInfo.nDest = _nDest;
	connInfo.srcGLoffset = 0;
	connInfo.connId = _connId;
	connInfo.preSynId = -1;
	// adjust the sign of the weight based on inh/exc connection
	connInfo.initWt = isExcitatoryGroup(_grpSrc) ? fabs(initWt) : -1.0*fabs(initWt);
	connInfo.maxWt = isExcitatoryGroup(_grpSrc) ? fabs(maxWt) : -1.0*fabs(maxWt);
	connInfo.delay = delay;

	connectionLists[netId].push_back(connInfo);

	// If the connection is external, copy the connection info to the external network
	if (externalNetId >= 0)
		connectionLists[externalNetId].push_back(connInfo);
}

// make 'C' full connections from grpSrc to grpDest
void SNN::connectFull(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal) {
	int grpSrc = connIt->grpSrc;
	int grpDest = connIt->grpDest;
	bool noDirect = (connIt->type == CONN_FULL_NO_DIRECT);
	int externalNetId = -1;

	if (isExternal) {
		externalNetId = groupConfigMDMap[grpDest].netId;
		assert(netId != externalNetId);
	}

	int gPreStart = groupConfigMDMap[grpSrc].gStartN;
	for(int gPreN = groupConfigMDMap[grpSrc].gStartN; gPreN <= groupConfigMDMap[grpSrc].gEndN; gPreN++)  {
		Point3D locPre = getNeuronLocation3D(grpSrc, gPreN - gPreStart); // 3D coordinates of i
		int gPostStart = groupConfigMDMap[grpDest].gStartN;
		for(int gPostN = groupConfigMDMap[grpDest].gStartN; gPostN <= groupConfigMDMap[grpDest].gEndN; gPostN++) { // j: the temp neuron id
			// if flag is set, don't connect direct connections
			if(noDirect && gPreN == gPostN)
				continue;

			// check whether pre-neuron location is in RF of post-neuron
			Point3D locPost = getNeuronLocation3D(grpDest, gPostN - gPostStart); // 3D coordinates of j
			if (!isPoint3DinRF(connIt->connRadius, locPre, locPost))
				continue;

			connectNeurons(netId, grpSrc, grpDest, gPreN, gPostN, connIt->connId, externalNetId);
			connIt->numberOfConnections++;
		}
	}

	std::list<GroupConfigMD>::iterator grpIt;
	GroupConfigMD targetGrp;

	// update numPostSynapses and numPreSynapses of groups in the local network
	targetGrp.gGrpId = grpSrc; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPostSynapses += connIt->numberOfConnections;

	targetGrp.gGrpId = grpDest; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPreSynapses += connIt->numberOfConnections;
	
	// also update numPostSynapses and numPreSynapses of groups in the external network if the connection is external
	if (isExternal) {
		targetGrp.gGrpId = grpSrc; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPostSynapses += connIt->numberOfConnections;

		targetGrp.gGrpId = grpDest; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPreSynapses += connIt->numberOfConnections;
	}
}

void SNN::connectGaussian(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal) {
	// in case pre and post have different Grid3D sizes: scale pre to the grid size of post
	int grpSrc = connIt->grpSrc;
	int grpDest = connIt->grpDest;
	Grid3D grid_i = getGroupGrid3D(grpSrc);
	Grid3D grid_j = getGroupGrid3D(grpDest);
	Point3D scalePre = Point3D(grid_j.numX, grid_j.numY, grid_j.numZ) / Point3D(grid_i.numX, grid_i.numY, grid_i.numZ);
	int externalNetId = -1;

	if (isExternal) {
		externalNetId = groupConfigMDMap[grpDest].netId;
		assert(netId != externalNetId);
	}

	for(int i = groupConfigMDMap[grpSrc].gStartN; i <= groupConfigMDMap[grpSrc].gEndN; i++)  {
		Point3D loc_i = getNeuronLocation3D(i)*scalePre; // i: adjusted 3D coordinates

		for(int j = groupConfigMDMap[grpDest].gStartN; j <= groupConfigMDMap[grpDest].gEndN; j++) { // j: the temp neuron id
			// check whether pre-neuron location is in RF of post-neuron
			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j

			// make sure point is in RF
			double rfDist = getRFDist3D(connIt->connRadius,loc_i,loc_j);
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

			if (drand48() < connIt->connProbability) {
				float initWt = gauss * connIt->initWt; // scale weight according to gauss distance
				float maxWt = connIt->maxWt;
				uint8_t delay = connIt->minDelay + rand() % (connIt->maxDelay - connIt->minDelay + 1);
				assert((delay >= connIt->minDelay) && (delay <= connIt->maxDelay));

				connectNeurons(netId, grpSrc, grpDest, i, j, connIt->connId, initWt, maxWt, delay, externalNetId);
				connIt->numberOfConnections++;
			}
		}
	}

	std::list<GroupConfigMD>::iterator grpIt;
	GroupConfigMD targetGrp;

	// update numPostSynapses and numPreSynapses of groups in the local network
	targetGrp.gGrpId = grpSrc; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPostSynapses += connIt->numberOfConnections;

	targetGrp.gGrpId = grpDest; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPreSynapses += connIt->numberOfConnections;
	
	// also update numPostSynapses and numPreSynapses of groups in the external network if the connection is external
	if (isExternal) {
		targetGrp.gGrpId = grpSrc; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPostSynapses += connIt->numberOfConnections;

		targetGrp.gGrpId = grpDest; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPreSynapses += connIt->numberOfConnections;
	}
}

void SNN::connectOneToOne(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal) {
	int grpSrc = connIt->grpSrc;
	int grpDest = connIt->grpDest;
	int externalNetId = -1;

	if (isExternal) {
		externalNetId = groupConfigMDMap[grpDest].netId;
		assert(netId != externalNetId);
	}

	assert( groupConfigMap[grpDest].numN == groupConfigMap[grpSrc].numN);

	// NOTE: RadiusRF does not make a difference here: ignore
	for(int gPreN = groupConfigMDMap[grpSrc].gStartN, gPostN = groupConfigMDMap[grpDest].gStartN; gPreN <= groupConfigMDMap[grpSrc].gEndN; gPreN++, gPostN++)  {
		connectNeurons(netId, grpSrc, grpDest, gPreN, gPostN, connIt->connId, externalNetId);
		connIt->numberOfConnections++;
	}

	std::list<GroupConfigMD>::iterator grpIt;
	GroupConfigMD targetGrp;

	// update numPostSynapses and numPreSynapses of groups in the local network
	targetGrp.gGrpId = grpSrc; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPostSynapses += connIt->numberOfConnections;

	targetGrp.gGrpId = grpDest; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPreSynapses += connIt->numberOfConnections;
	
	// also update numPostSynapses and numPreSynapses of groups in the external network if the connection is external
	if (isExternal) {
		targetGrp.gGrpId = grpSrc; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPostSynapses += connIt->numberOfConnections;

		targetGrp.gGrpId = grpDest; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPreSynapses += connIt->numberOfConnections;
	}
}

// make 'C' random connections from grpSrc to grpDest
void SNN::connectRandom(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal) {
	int grpSrc = connIt->grpSrc;
	int grpDest = connIt->grpDest;
	int externalNetId = -1;

	if (isExternal) {
		externalNetId = groupConfigMDMap[grpDest].netId;
		assert(netId != externalNetId);
	}

	int gPreStart = groupConfigMDMap[grpSrc].gStartN;
	for(int gPreN = groupConfigMDMap[grpSrc].gStartN; gPreN <= groupConfigMDMap[grpSrc].gEndN; gPreN++) {
		Point3D locPre = getNeuronLocation3D(grpSrc, gPreN - gPreStart); // 3D coordinates of i
		int gPostStart = groupConfigMDMap[grpDest].gStartN;
		for(int gPostN = groupConfigMDMap[grpDest].gStartN; gPostN <= groupConfigMDMap[grpDest].gEndN; gPostN++) {
			// check whether pre-neuron location is in RF of post-neuron
			Point3D locPost = getNeuronLocation3D(grpDest, gPostN - gPostStart); // 3D coordinates of j
			if (!isPoint3DinRF(connIt->connRadius, locPre, locPost))
				continue;

			if (drand48() < connIt->connProbability) {
				connectNeurons(netId, grpSrc, grpDest, gPreN, gPostN, connIt->connId, externalNetId);
				connIt->numberOfConnections++;
			}
		}
	}

	std::list<GroupConfigMD>::iterator grpIt;
	GroupConfigMD targetGrp;

	// update numPostSynapses and numPreSynapses of groups in the local network
	targetGrp.gGrpId = grpSrc; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPostSynapses += connIt->numberOfConnections;

	targetGrp.gGrpId = grpDest; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPreSynapses += connIt->numberOfConnections;
	
	// also update numPostSynapses and numPreSynapses of groups in the external network if the connection is external
	if (isExternal) {
		targetGrp.gGrpId = grpSrc; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPostSynapses += connIt->numberOfConnections;

		targetGrp.gGrpId = grpDest; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPreSynapses += connIt->numberOfConnections;
	}
}

// FIXME: rewrite user-define call-back function
// user-defined functions called here...
// This is where we define our user-defined call-back function.  -- KDC
void SNN::connectUserDefined(int netId, std::list<ConnectConfig>::iterator connIt, bool isExternal) {
	int grpSrc = connIt->grpSrc;
	int grpDest = connIt->grpDest;
	int externalNetId = -1;

	if (isExternal) {
		externalNetId = groupConfigMDMap[grpDest].netId;
		assert(netId != externalNetId);
	}

	connIt->maxDelay = 0;
	int preStartN = groupConfigMDMap[grpSrc].gStartN;
	int postStartN = groupConfigMDMap[grpDest].gStartN;
	for (int pre_nid = groupConfigMDMap[grpSrc].gStartN; pre_nid <= groupConfigMDMap[grpSrc].gEndN; pre_nid++) {
		//Point3D loc_pre = getNeuronLocation3D(pre_nid); // 3D coordinates of i
		for (int post_nid = groupConfigMDMap[grpDest].gStartN; post_nid <= groupConfigMDMap[grpDest].gEndN; post_nid++) {
			float weight, maxWt, delay;
			bool connected;

			connIt->conn->connect(this, grpSrc, pre_nid - preStartN, grpDest, post_nid - postStartN, weight, maxWt, delay, connected);
			if (connected) {
				assert(delay >= 1);
				assert(delay <= MAX_SYN_DELAY);
				assert(abs(weight) <= abs(maxWt));

				if (GET_FIXED_PLASTIC(connIt->connProp) == SYN_FIXED)
					maxWt = weight;

				if (fabs(maxWt) > connIt->maxWt)
					connIt->maxWt = fabs(maxWt);
				
				if (delay > connIt->maxDelay)
					connIt->maxDelay = delay;

				connectNeurons(netId, grpSrc, grpDest, pre_nid, post_nid, connIt->connId, weight, maxWt, delay, externalNetId);
				connIt->numberOfConnections++;
			}
		}
	}

	std::list<GroupConfigMD>::iterator grpIt;
	GroupConfigMD targetGrp;

	// update numPostSynapses and numPreSynapses of groups in the local network
	targetGrp.gGrpId = grpSrc; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPostSynapses += connIt->numberOfConnections;

	targetGrp.gGrpId = grpDest; // the other fields does not matter
	grpIt = std::find(groupPartitionLists[netId].begin(), groupPartitionLists[netId].end(), targetGrp);
	assert(grpIt != groupPartitionLists[netId].end());
	grpIt->numPreSynapses += connIt->numberOfConnections;

	// also update numPostSynapses and numPreSynapses of groups in the external network if the connection is external
	if (isExternal) {
		targetGrp.gGrpId = grpSrc; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPostSynapses += connIt->numberOfConnections;

		targetGrp.gGrpId = grpDest; // the other fields does not matter
		grpIt = std::find(groupPartitionLists[externalNetId].begin(), groupPartitionLists[externalNetId].end(), targetGrp);
		assert(grpIt != groupPartitionLists[externalNetId].end());
		grpIt->numPreSynapses += connIt->numberOfConnections;
	}
}

//// make 'C' full connections from grpSrc to grpDest
//void SNN::connectFull(short int connId) {
//	int grpSrc = connectConfigMap[connId].grpSrc;
//	int grpDest = connectConfigMap[connId].grpDest;
//	bool noDirect = (connectConfigMap[connId].type == CONN_FULL_NO_DIRECT);
//
//	// rebuild struct for easier handling
//	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);
//
//	for(int i = groupConfigMap[grpSrc].StartN; i <= groupConfigMap[grpSrc].EndN; i++)  {
//		Point3D loc_i = getNeuronLocation3D(i); // 3D coordinates of i
//		for(int j = groupConfigMap[grpDest].StartN; j <= groupConfigMap[grpDest].EndN; j++) { // j: the temp neuron id
//			// if flag is set, don't connect direct connections
//			if((noDirect) && (i - groupConfigMap[grpSrc].StartN) == (j - groupConfigMap[grpDest].StartN))
//				continue;
//
//			// check whether pre-neuron location is in RF of post-neuron
//			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j
//			if (!isPoint3DinRF(radius, loc_i, loc_j))
//				continue;
//
//			//uint8_t dVal = info->minDelay + (int)(0.5 + (drand48() * (info->maxDelay - info->minDelay)));
//			uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
//			assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
//			float synWt = generateWeight(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, i, grpSrc);
//
//			setConnection(grpSrc, grpDest, i, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);// info->connId);
//			connectConfigMap[connId].numberOfConnections++;
//		}
//	}
//
//	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
//	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
//}

//void SNN::connectGaussian(short int connId) {
//	// rebuild struct for easier handling
//	// adjust with sqrt(2) in order to make the Gaussian kernel depend on 2*sigma^2
//	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);
//
//	// in case pre and post have different Grid3D sizes: scale pre to the grid size of post
//	int grpSrc = connectConfigMap[connId].grpSrc;
//	int grpDest = connectConfigMap[connId].grpDest;
//	Grid3D grid_i = getGroupGrid3D(grpSrc);
//	Grid3D grid_j = getGroupGrid3D(grpDest);
//	Point3D scalePre = Point3D(grid_j.numX, grid_j.numY, grid_j.numZ) / Point3D(grid_i.numX, grid_i.numY, grid_i.numZ);
//
//	for(int i = groupConfigMap[grpSrc].StartN; i <= groupConfigMap[grpSrc].EndN; i++)  {
//		Point3D loc_i = getNeuronLocation3D(i)*scalePre; // i: adjusted 3D coordinates
//
//		for(int j = groupConfigMap[grpDest].StartN; j <= groupConfigMap[grpDest].EndN; j++) { // j: the temp neuron id
//			// check whether pre-neuron location is in RF of post-neuron
//			Point3D loc_j = getNeuronLocation3D(j); // 3D coordinates of j
//
//			// make sure point is in RF
//			double rfDist = getRFDist3D(radius,loc_i,loc_j);
//			if (rfDist < 0.0 || rfDist > 1.0)
//				continue;
//
//			// if rfDist is valid, it returns a number between 0 and 1
//			// we want these numbers to fit to Gaussian weigths, so that rfDist=0 corresponds to max Gaussian weight
//			// and rfDist=1 corresponds to 0.1 times max Gaussian weight
//			// so we're looking at gauss = exp(-a*rfDist), where a such that exp(-a)=0.1
//			// solving for a, we find that a = 2.3026
//			double gauss = exp(-2.3026*rfDist);
//			if (gauss < 0.1)
//				continue;
//
//			if (drand48() < connectConfigMap[connId].p) {
//				uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
//				assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
//				float synWt = gauss * connectConfigMap[connId].initWt; // scale weight according to gauss distance
//				setConnection(grpSrc, grpDest, i, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);//info->connId);
//				connectConfigMap[connId].numberOfConnections++;
//			}
//		}
//	}
//
//	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
//	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
//}
//
//void SNN::connectOneToOne(short int connId) {
//	int grpSrc = connectConfigMap[connId].grpSrc;
//	int grpDest = connectConfigMap[connId].grpDest;
//	assert( groupConfigMap[grpDest].SizeN == groupConfigMap[grpSrc].SizeN );
//
//	// NOTE: RadiusRF does not make a difference here: ignore
//	for(int nid=groupConfigMap[grpSrc].StartN,j=groupConfigMap[grpDest].StartN; nid<=groupConfigMap[grpSrc].EndN; nid++, j++)  {
//		uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
//		assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
//		float synWt = generateWeight(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, nid, grpSrc);
//		setConnection(grpSrc, grpDest, nid, j, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId);//info->connId);
//		connectConfigMap[connId].numberOfConnections++;
//	}
//
//	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
//	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
//}
//
//// make 'C' random connections from grpSrc to grpDest
//void SNN::connectRandom(short int connId) {
//	int grpSrc = connectConfigMap[connId].grpSrc;
//	int grpDest = connectConfigMap[connId].grpDest;
//
//	// rebuild struct for easier handling
//	RadiusRF radius(connectConfigMap[connId].radX, connectConfigMap[connId].radY, connectConfigMap[connId].radZ);
//
//	for(int pre_nid = groupConfigMap[grpSrc].StartN; pre_nid <= groupConfigMap[grpSrc].EndN; pre_nid++) {
//		Point3D loc_pre = getNeuronLocation3D(pre_nid); // 3D coordinates of i
//		for(int post_nid = groupConfigMap[grpDest].StartN; post_nid <= groupConfigMap[grpDest].EndN; post_nid++) {
//			// check whether pre-neuron location is in RF of post-neuron
//			Point3D loc_post = getNeuronLocation3D(post_nid); // 3D coordinates of j
//			if (!isPoint3DinRF(radius, loc_pre, loc_post))
//				continue;
//
//			if (drand48() < connectConfigMap[connId].p) {
//				//uint8_t dVal = info->minDelay + (int)(0.5+(drand48()*(info->maxDelay-info->minDelay)));
//				uint8_t dVal = connectConfigMap[connId].minDelay + rand() % (connectConfigMap[connId].maxDelay - connectConfigMap[connId].minDelay + 1);
//				assert((dVal >= connectConfigMap[connId].minDelay) && (dVal <= connectConfigMap[connId].maxDelay));
//				float synWt = generateWeight(connectConfigMap[connId].connProp, connectConfigMap[connId].initWt, connectConfigMap[connId].maxWt, pre_nid, grpSrc);
//				setConnection(grpSrc, grpDest, pre_nid, post_nid, synWt, connectConfigMap[connId].maxWt, dVal, connectConfigMap[connId].connProp, connId); //info->connId);
//				connectConfigMap[connId].numberOfConnections++;
//			}
//		}
//	}
//
//	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
//	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
//}
//
//// user-defined functions called here...
//// This is where we define our user-defined call-back function.  -- KDC
//void SNN::connectUserDefined(short int connId) {
//	int grpSrc = connectConfigMap[connId].grpSrc;
//	int grpDest = connectConfigMap[connId].grpDest;
//	connectConfigMap[connId].maxDelay = 0;
//	for(int nid=groupConfigMap[grpSrc].StartN; nid<=groupConfigMap[grpSrc].EndN; nid++) {
//		for(int nid2=groupConfigMap[grpDest].StartN; nid2 <= groupConfigMap[grpDest].EndN; nid2++) {
//			int srcId  = nid  - groupConfigMap[grpSrc].StartN;
//			int destId = nid2 - groupConfigMap[grpDest].StartN;
//			float weight, maxWt, delay;
//			bool connected;
//
//			connectConfigMap[connId].conn->connect(this, grpSrc, srcId, grpDest, destId, weight, maxWt, delay, connected);
//			if(connected)  {
//				if (GET_FIXED_PLASTIC(connectConfigMap[connId].connProp) == SYN_FIXED)
//					maxWt = weight;
//
//				connectConfigMap[connId].maxWt = maxWt;
//
//				assert(delay >= 1);
//				assert(delay <= MAX_SYN_DELAY);
//				assert(abs(weight) <= abs(maxWt));
//
//				// adjust the sign of the weight based on inh/exc connection
//				weight = isExcitatoryGroup(grpSrc) ? fabs(weight) : -1.0*fabs(weight);
//				maxWt  = isExcitatoryGroup(grpSrc) ? fabs(maxWt)  : -1.0*fabs(maxWt);
//
//				setConnection(grpSrc, grpDest, nid, nid2, weight, maxWt, delay, connectConfigMap[connId].connProp, connId);// info->connId);
//				connectConfigMap[connId].numberOfConnections++;
//				if(delay > connectConfigMap[connId].maxDelay) {
//					connectConfigMap[connId].maxDelay = delay;
//				}
//			}
//		}
//	}
//
//	groupInfo[grpSrc].sumPostConn += connectConfigMap[connId].numberOfConnections;
//	groupInfo[grpDest].sumPreConn += connectConfigMap[connId].numberOfConnections;
//}

void SNN::deleteRuntimeData() {
	// FIXME: assert simulation use GPU first
	// wait for kernels to complete
#ifndef __NO_CUDA__
	CUDA_CHECK_ERRORS(cudaThreadSynchronize());
#endif

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
		cpu_set_t cpus;	
		ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
		int threadCount = 0;
	#endif

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			if (netId < CPU_RUNTIME_BASE) // GPU runtime
				deleteRuntimeData_GPU(netId);
			else{ // CPU runtime
				#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
					deleteRuntimeData_CPU(netId);
				#else // Linux or MAC
					pthread_attr_t attr;
					pthread_attr_init(&attr);
					CPU_ZERO(&cpus);
					CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

					argsThreadRoutine[threadCount].snn_pointer = this;
					argsThreadRoutine[threadCount].netId = netId;
					argsThreadRoutine[threadCount].lGrpId = 0;
					argsThreadRoutine[threadCount].startIdx = 0;
					argsThreadRoutine[threadCount].endIdx = 0;
					argsThreadRoutine[threadCount].GtoLOffset = 0;

					pthread_create(&threads[threadCount], &attr, &SNN::helperDeleteRuntimeData_CPU, (void*)&argsThreadRoutine[threadCount]);
					pthread_attr_destroy(&attr);
					threadCount++;
				#endif
			}
		}
	}

	#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
		// join all the threads
		for (int i=0; i<threadCount; i++){
			pthread_join(threads[i], NULL);
		}
	#endif

#ifndef __NO_CUDA__
	CUDA_DELETE_TIMER(timer);
#endif
}

// delete all objects (CPU and GPU side)
void SNN::deleteObjects() {
	if (simulatorDeleted)
		return;

	printSimSummary();

	// deallocate objects
	resetMonitors(true);
	resetConnectionConfigs(true);
	
	// delete manager runtime data
	deleteManagerRuntimeData();

	deleteRuntimeData();

	// fclose file streams, unless in custom mode
	if (loggerMode_ != CUSTOM) {
		// don't fclose if it's stdout or stderr, otherwise they're gonna stay closed for the rest of the process
		if (fpInf_ != NULL && fpInf_ != stdout && fpInf_ != stderr)
			fclose(fpInf_);
		if (fpErr_ != NULL && fpErr_ != stdout && fpErr_ != stderr)
			fclose(fpErr_);
		if (fpDeb_ != NULL && fpDeb_ != stdout && fpDeb_ != stderr)
			fclose(fpDeb_);
		if (fpLog_ != NULL && fpLog_ != stdout && fpLog_ != stderr)
			fclose(fpLog_);
	}

	simulatorDeleted = true;
}

void SNN::findMaxNumSynapsesGroups(int* _maxNumPostSynGrp, int* _maxNumPreSynGrp) {
	*_maxNumPostSynGrp = 0;
	*_maxNumPreSynGrp = 0;

	//  scan all the groups and find the required information
	for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
		// find the values for maximum postsynaptic length
		// and maximum pre-synaptic length
		if (groupConfigMDMap[gGrpId].numPostSynapses > *_maxNumPostSynGrp)
			*_maxNumPostSynGrp = groupConfigMDMap[gGrpId].numPostSynapses;
		if (groupConfigMDMap[gGrpId].numPreSynapses > *_maxNumPreSynGrp)
			*_maxNumPreSynGrp = groupConfigMDMap[gGrpId].numPreSynapses;
	}
}

void SNN::findMaxNumSynapsesNeurons(int _netId, int& _maxNumPostSynN, int& _maxNumPreSynN) {
	int *tempNpre, *tempNpost;
	int nSrc, nDest, numNeurons;
	std::map<int, int> globalToLocalOffset;

	numNeurons = networkConfigs[_netId].numNAssigned;
	tempNpre = new int[numNeurons];
	tempNpost = new int[numNeurons];
	memset(tempNpre, 0, sizeof(int) * numNeurons);
	memset(tempNpost, 0, sizeof(int) * numNeurons);

	// load offset between global neuron id and local neuron id 
	for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[_netId].begin(); grpIt != groupPartitionLists[_netId].end(); grpIt++) {
		globalToLocalOffset[grpIt->gGrpId] = grpIt->GtoLOffset;
	}

	// calculate number of pre- and post- connections of each neuron
	for (std::list<ConnectionInfo>::iterator connIt = connectionLists[_netId].begin(); connIt != connectionLists[_netId].end(); connIt++) {
		nSrc = connIt->nSrc + globalToLocalOffset[connIt->grpSrc];
		nDest = connIt->nDest + globalToLocalOffset[connIt->grpDest];
		assert(nSrc < numNeurons); assert(nDest < numNeurons);
		tempNpost[nSrc]++;
		tempNpre[nDest]++;
	}

	// find out the maximum number of pre- and post- connections among neurons in a local network
	_maxNumPostSynN = 0;
	_maxNumPreSynN = 0;
	for (int nId = 0; nId < networkConfigs[_netId].numN; nId++) {
		if (tempNpost[nId] > _maxNumPostSynN) _maxNumPostSynN = tempNpost[nId];
		if (tempNpre[nId] > _maxNumPreSynN) _maxNumPreSynN = tempNpre[nId];
	}

	delete [] tempNpre;
	delete [] tempNpost;
}

void SNN::findMaxSpikesD1D2(int _netId, unsigned int& _maxSpikesD1, unsigned int& _maxSpikesD2) {
	_maxSpikesD1 = 0; _maxSpikesD2 = 0;
	for(std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[_netId].begin(); grpIt != groupPartitionLists[_netId].end(); grpIt++) {
		if (grpIt->maxOutgoingDelay == 1)
			_maxSpikesD1 += (groupConfigMap[grpIt->gGrpId].numN * NEURON_MAX_FIRING_RATE);
		else
			_maxSpikesD2 += (groupConfigMap[grpIt->gGrpId].numN * NEURON_MAX_FIRING_RATE);
	}
}

void SNN::findNumN(int _netId, int& _numN, int& _numNExternal, int& _numNAssigned,
                   int& _numNReg, int& _numNExcReg, int& _numNInhReg,
                   int& _numNPois, int& _numNExcPois, int& _numNInhPois) {
	_numN = 0; _numNExternal = 0; _numNAssigned = 0;
	_numNReg = 0; _numNExcReg = 0; _numNInhReg = 0;
	_numNPois = 0; _numNExcPois = 0; _numNInhPois = 0;
	for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[_netId].begin(); grpIt != groupPartitionLists[_netId].end(); grpIt++) {
		int sizeN = groupConfigMap[grpIt->gGrpId].numN;
		unsigned int type = groupConfigMap[grpIt->gGrpId].type;
		if (IS_EXCITATORY_TYPE(type) && (type & POISSON_NEURON) && grpIt->netId == _netId) {
			_numN += sizeN; _numNPois += sizeN; _numNExcPois += sizeN;
		} else if (IS_INHIBITORY_TYPE(type) && (type & POISSON_NEURON) && grpIt->netId == _netId) {
			_numN += sizeN; _numNPois += sizeN; _numNInhPois += sizeN;
		} else if (IS_EXCITATORY_TYPE(type) && !(type & POISSON_NEURON) && grpIt->netId == _netId) {
			_numN += sizeN; _numNReg += sizeN; _numNExcReg += sizeN;
		} else if (IS_INHIBITORY_TYPE(type) && !(type & POISSON_NEURON) && grpIt->netId == _netId) {
			_numN += sizeN; _numNReg += sizeN; _numNInhReg += sizeN;
		} else if (grpIt->netId != _netId) {
			_numNExternal += sizeN;
		} else {
			KERNEL_ERROR("Can't find catagory for the group [%d] ", grpIt->gGrpId);
			exitSimulation(-1);
		}
		_numNAssigned += sizeN;
	}

	assert(_numNReg == _numNExcReg + _numNInhReg);
	assert(_numNPois == _numNExcPois + _numNInhPois);
	assert(_numN == _numNReg + _numNPois);
	assert(_numNAssigned == _numN + _numNExternal);
}

void SNN::findNumNSpikeGenAndOffset(int _netId) {
	networkConfigs[_netId].numNSpikeGen = 0;

	for(int lGrpId = 0; lGrpId < networkConfigs[_netId].numGroups; lGrpId++) {
		if (_netId == groupConfigs[_netId][lGrpId].netId && groupConfigs[_netId][lGrpId].isSpikeGenerator && groupConfigs[_netId][lGrpId].isSpikeGenFunc) {
			groupConfigs[_netId][lGrpId].Noffset = networkConfigs[_netId].numNSpikeGen;
			networkConfigs[_netId].numNSpikeGen += groupConfigs[_netId][lGrpId].numN;
		}
	}

	assert(networkConfigs[_netId].numNSpikeGen <= networkConfigs[_netId].numNPois);
}

void SNN::findNumSynapsesNetwork(int _netId, int& _numPostSynNet, int& _numPreSynNet) {
	_numPostSynNet = 0;
	_numPreSynNet  = 0;

	for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[_netId].begin(); grpIt != groupPartitionLists[_netId].end(); grpIt++) {
		_numPostSynNet += grpIt->numPostSynapses;
		_numPreSynNet += grpIt->numPreSynapses;
		assert(_numPostSynNet < INT_MAX);
		assert(_numPreSynNet <  INT_MAX);
	}

	assert(_numPreSynNet == _numPostSynNet);
}

void SNN::fetchGroupState(int netId, int lGrpId) {
	if (netId < CPU_RUNTIME_BASE)
		copyGroupState(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false);
	else
		copyGroupState(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false);
}

void SNN::fetchWeightState(int netId, int lGrpId) {
	if (netId < CPU_RUNTIME_BASE)
		copyWeightState(netId, lGrpId, cudaMemcpyDeviceToHost);
	else
		copyWeightState(netId, lGrpId);
}

/*!
 * \brief This function copies spike count of each neuron from device (GPU) memory to main (CPU) memory
 *
 * \param[in] gGrpId the group id of the global network of which the spike count of each neuron with in the group are copied to manager runtime data
 */
void SNN::fetchNeuronSpikeCount (int gGrpId) {
	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			fetchNeuronSpikeCount(gGrpId);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int LtoGOffset = groupConfigMDMap[gGrpId].LtoGOffset;

		if (netId < CPU_RUNTIME_BASE)
			copyNeuronSpikeCount(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false, LtoGOffset);
		else
			copyNeuronSpikeCount(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false, LtoGOffset);
	}
}

void SNN::fetchSTPState(int gGrpId) {
}

/*!
 * \brief This function copies AMPA conductances from device (GPU) memory to main (CPU) memory
 *
 * \param[in] gGrpId the group id of the global network of which the spike count of each neuron in the group are copied to manager runtime data
 */
void SNN::fetchConductanceAMPA(int gGrpId) {
	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			fetchConductanceAMPA(gGrpId);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int LtoGOffset = groupConfigMDMap[gGrpId].LtoGOffset;

		if (netId < CPU_RUNTIME_BASE)
			copyConductanceAMPA(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false, LtoGOffset);
		else
			copyConductanceAMPA(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false, LtoGOffset);
	}
}

/*!
 * \brief This function copies NMDA conductances from device (GPU) memory to main (CPU) memory
 *
 * \param[in] gGrpId the group id of the global network of which the spike count of each neuron in the group are copied to manager runtime data
 */
void SNN::fetchConductanceNMDA(int gGrpId) {
	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			fetchConductanceNMDA(gGrpId);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int LtoGOffset = groupConfigMDMap[gGrpId].LtoGOffset;

		if (netId < CPU_RUNTIME_BASE)
			copyConductanceNMDA(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false, LtoGOffset);
		else
			copyConductanceNMDA(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false, LtoGOffset);
	}
}

/*!
 * \brief This function copies GABAa conductances from device (GPU) memory to main (CPU) memory
 *
 * \param[in] gGrpId the group id of the global network of which the spike count of each neuron in the group are copied to manager runtime data
 */
void SNN::fetchConductanceGABAa(int gGrpId) {
	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			fetchConductanceGABAa(gGrpId);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int LtoGOffset = groupConfigMDMap[gGrpId].LtoGOffset;

		if (netId < CPU_RUNTIME_BASE)
			copyConductanceGABAa(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false, LtoGOffset);
		else
			copyConductanceGABAa(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false, LtoGOffset);
	}
}

/*!
 * \brief This function copies GABAb conductances from device (GPU) memory to main (CPU) memory
 *
 * \param[in] gGrpId the group id of the global network of which the spike count of each neuron in the group are copied to manager runtime data
 */
void SNN::fetchConductanceGABAb(int gGrpId) {
	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
			fetchConductanceGABAb(gGrpId);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int LtoGOffset = groupConfigMDMap[gGrpId].LtoGOffset;

		if (netId < CPU_RUNTIME_BASE)
			copyConductanceGABAb(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false, LtoGOffset);
		else
			copyConductanceGABAb(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false, LtoGOffset);
	}
}


void SNN::fetchGrpIdsLookupArray(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copyGrpIdsLookupArray(netId, cudaMemcpyDeviceToHost);
	else
		copyGrpIdsLookupArray(netId);
}

void SNN::fetchConnIdsLookupArray(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copyConnIdsLookupArray(netId, cudaMemcpyDeviceToHost);
	else
		copyConnIdsLookupArray(netId);
}

void SNN::fetchLastSpikeTime(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copyLastSpikeTime(netId, cudaMemcpyDeviceToHost);
	else
		copyLastSpikeTime(netId);
}

void SNN::fetchPreConnectionInfo(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copyPreConnectionInfo(netId, ALL, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false);
	else
		copyPreConnectionInfo(netId, ALL, &managerRuntimeData, &runtimeData[netId], false);
}

void SNN::fetchPostConnectionInfo(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copyPostConnectionInfo(netId, ALL, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false);
	else
		copyPostConnectionInfo(netId, ALL, &managerRuntimeData, &runtimeData[netId], false);
}

void SNN::fetchSynapseState(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copySynapseState(netId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false);
	else
		copySynapseState(netId, &managerRuntimeData, &runtimeData[netId], false);
}


/*!
* \brief This function fetch the spike count in all local networks and sum the up
*/
void SNN::fetchNetworkSpikeCount() {
	unsigned int spikeCountD1, spikeCountD2, spikeCountExtD1, spikeCountExtD2;

	managerRuntimeData.spikeCountD1 = 0;
	managerRuntimeData.spikeCountD2 = 0;
	managerRuntimeData.spikeCountExtRxD2 = 0;
	managerRuntimeData.spikeCountExtRxD1 = 0;
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {

			if (netId < CPU_RUNTIME_BASE) {
				copyNetworkSpikeCount(netId, cudaMemcpyDeviceToHost,
					&spikeCountD1, &spikeCountD2,
					&spikeCountExtD1, &spikeCountExtD2);
				//printf("netId:%d, D1:%d/D2:%d, extD1:%d/D2:%d\n", netId, spikeCountD1, spikeCountD2, spikeCountExtD1, spikeCountExtD2);
			} else {
				copyNetworkSpikeCount(netId,
					&spikeCountD1, &spikeCountD2,
					&spikeCountExtD1, &spikeCountExtD2);
				//printf("netId:%d, D1:%d/D2:%d, extD1:%d/D2:%d\n", netId, spikeCountD1, spikeCountD2, spikeCountExtD1, spikeCountExtD2);
			}

			managerRuntimeData.spikeCountD2 += spikeCountD2 - spikeCountExtD2;
			managerRuntimeData.spikeCountD1 += spikeCountD1 - spikeCountExtD1;
			managerRuntimeData.spikeCountExtRxD2 += spikeCountExtD2;
			managerRuntimeData.spikeCountExtRxD1 += spikeCountExtD1;
		}
	}

	managerRuntimeData.spikeCount = managerRuntimeData.spikeCountD1 + managerRuntimeData.spikeCountD2;
}

void SNN::fetchSpikeTables(int netId) {
	if (netId < CPU_RUNTIME_BASE)
		copySpikeTables(netId, cudaMemcpyDeviceToHost);
	else
		copySpikeTables(netId);
}

void SNN::fetchNeuronStateBuffer(int netId, int lGrpId) {
	if (netId < CPU_RUNTIME_BASE)
		copyNeuronStateBuffer(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], cudaMemcpyDeviceToHost, false);
	else
		copyNeuronStateBuffer(netId, lGrpId, &managerRuntimeData, &runtimeData[netId], false);
}

void SNN::fetchExtFiringTable(int netId) {
	assert(netId < MAX_NET_PER_SNN);
	
	if (netId < CPU_RUNTIME_BASE) { // GPU runtime
		copyExtFiringTable(netId, cudaMemcpyDeviceToHost);
	} else { // CPU runtime
		copyExtFiringTable(netId);
	}
}

void SNN::fetchTimeTable(int netId) {
	assert(netId < MAX_NET_PER_SNN);

	if (netId < CPU_RUNTIME_BASE) { // GPU runtime
		copyTimeTable(netId, cudaMemcpyDeviceToHost);
	} else {
		copyTimeTable(netId, true);
	}
}

void SNN::writeBackTimeTable(int netId) {
	assert(netId < MAX_NET_PER_SNN);

	if (netId < CPU_RUNTIME_BASE) { // GPU runtime
		copyTimeTable(netId, cudaMemcpyHostToDevice);
	} else {
		copyTimeTable(netId, false);
	}
}

void SNN::transferSpikes(void* dest, int destNetId, void* src, int srcNetId, int size) {
#ifndef __NO_CUDA__
	if (srcNetId < CPU_RUNTIME_BASE && destNetId < CPU_RUNTIME_BASE) {
		checkAndSetGPUDevice(destNetId);
		CUDA_CHECK_ERRORS(cudaMemcpyPeer(dest, destNetId, src, srcNetId, size));
	} else if (srcNetId >= CPU_RUNTIME_BASE && destNetId < CPU_RUNTIME_BASE) {
		checkAndSetGPUDevice(destNetId);
		CUDA_CHECK_ERRORS(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
	} else if (srcNetId < CPU_RUNTIME_BASE && destNetId >= CPU_RUNTIME_BASE) {
		checkAndSetGPUDevice(srcNetId);
		CUDA_CHECK_ERRORS(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
	} else if(srcNetId >= CPU_RUNTIME_BASE && destNetId >= CPU_RUNTIME_BASE) {
		memcpy(dest, src, size);
	}
#else
	assert(srcNetId >= CPU_RUNTIME_BASE && destNetId >= CPU_RUNTIME_BASE);
	memcpy(dest, src, size);
#endif
}

void SNN::convertExtSpikesD2(int netId, int startIdx, int endIdx, int GtoLOffset) {
	if (netId < CPU_RUNTIME_BASE)
		convertExtSpikesD2_GPU(netId, startIdx, endIdx, GtoLOffset);
	else
		convertExtSpikesD2_CPU(netId, startIdx, endIdx, GtoLOffset);
}

void SNN::convertExtSpikesD1(int netId, int startIdx, int endIdx, int GtoLOffset) {
	if (netId < CPU_RUNTIME_BASE)
		convertExtSpikesD1_GPU(netId, startIdx, endIdx, GtoLOffset);
	else
		convertExtSpikesD1_CPU(netId, startIdx, endIdx, GtoLOffset);
}

void SNN::routeSpikes() {
	int firingTableIdxD2, firingTableIdxD1;
	int GtoLOffset;

	for (std::list<RoutingTableEntry>::iterator rteItr = spikeRoutingTable.begin(); rteItr != spikeRoutingTable.end(); rteItr++) {
		int srcNetId = rteItr->srcNetId;
		int destNetId = rteItr->destNetId;

		fetchExtFiringTable(srcNetId);

		fetchTimeTable(destNetId);
		firingTableIdxD2 = managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1];
		firingTableIdxD1 = managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1];
		//KERNEL_DEBUG("GPU1 D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);
		//printf("srcNetId %d,destNetId %d, D1:%d/D2:%d\n", srcNetId, destNetId, firingTableIdxD1, firingTableIdxD2);

		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			pthread_t threads[(2 * networkConfigs[srcNetId].numGroups) + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
			cpu_set_t cpus;	
			ThreadStruct argsThreadRoutine[(2 * networkConfigs[srcNetId].numGroups) + 1]; // same as above, +1 array size
			int threadCount = 0;
		#endif

		for (int lGrpId = 0; lGrpId < networkConfigs[srcNetId].numGroups; lGrpId++) {
			if (groupConfigs[srcNetId][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD2[lGrpId] > 0) {
				// search GtoLOffset of the neural group at destination local network
				bool isFound = false;
				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[destNetId].begin(); grpIt != groupPartitionLists[destNetId].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[srcNetId][lGrpId].gGrpId) {
						GtoLOffset = grpIt->GtoLOffset;
						isFound = true;
						break;
					}
				}

				if (isFound) {
					transferSpikes(runtimeData[destNetId].firingTableD2 + firingTableIdxD2, destNetId,
						managerRuntimeData.extFiringTableD2[lGrpId], srcNetId,
						sizeof(int) * managerRuntimeData.extFiringTableEndIdxD2[lGrpId]);

					if (destNetId < CPU_RUNTIME_BASE){
						convertExtSpikesD2_GPU(destNetId, firingTableIdxD2,
							firingTableIdxD2 + managerRuntimeData.extFiringTableEndIdxD2[lGrpId],
							GtoLOffset); // [StartIdx, EndIdx)
					}
					else{// CPU runtime
							#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
								convertExtSpikesD2_CPU(destNetId, firingTableIdxD2,
									firingTableIdxD2 + managerRuntimeData.extFiringTableEndIdxD2[lGrpId],
									GtoLOffset); // [StartIdx, EndIdx)
							#else // Linux or MAC
								pthread_attr_t attr;
								pthread_attr_init(&attr);
								CPU_ZERO(&cpus);
								CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
								pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

								argsThreadRoutine[threadCount].snn_pointer = this;
								argsThreadRoutine[threadCount].netId = destNetId;
								argsThreadRoutine[threadCount].lGrpId = 0;
								argsThreadRoutine[threadCount].startIdx = firingTableIdxD2;
								argsThreadRoutine[threadCount].endIdx = firingTableIdxD2 + managerRuntimeData.extFiringTableEndIdxD2[lGrpId];
								argsThreadRoutine[threadCount].GtoLOffset = GtoLOffset;

								pthread_create(&threads[threadCount], &attr, &SNN::helperConvertExtSpikesD2_CPU, (void*)&argsThreadRoutine[threadCount]);
								pthread_attr_destroy(&attr);
								threadCount++;
							#endif
					}

					firingTableIdxD2 += managerRuntimeData.extFiringTableEndIdxD2[lGrpId];
				}
			}

			if (groupConfigs[srcNetId][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD1[lGrpId] > 0) {
				// search GtoLOffset of the neural group at destination local network
				bool isFound = false;
				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[destNetId].begin(); grpIt != groupPartitionLists[destNetId].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[srcNetId][lGrpId].gGrpId) {
						GtoLOffset = grpIt->GtoLOffset;
						isFound = true;
						break;
					}
				}

				if (isFound) {
					transferSpikes(runtimeData[destNetId].firingTableD1 + firingTableIdxD1, destNetId,
						managerRuntimeData.extFiringTableD1[lGrpId], srcNetId,
						sizeof(int) * managerRuntimeData.extFiringTableEndIdxD1[lGrpId]);
					if (destNetId < CPU_RUNTIME_BASE){
						convertExtSpikesD1_GPU(destNetId, firingTableIdxD1,
							firingTableIdxD1 + managerRuntimeData.extFiringTableEndIdxD1[lGrpId],
							GtoLOffset); // [StartIdx, EndIdx)
					}
					else{// CPU runtime
						#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
								convertExtSpikesD1_CPU(destNetId, firingTableIdxD1,
									firingTableIdxD1 + managerRuntimeData.extFiringTableEndIdxD1[lGrpId],
									GtoLOffset); // [StartIdx, EndIdx)
							#else // Linux or MAC
								pthread_attr_t attr;
								pthread_attr_init(&attr);
								CPU_ZERO(&cpus);
								CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
								pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

								argsThreadRoutine[threadCount].snn_pointer = this;
								argsThreadRoutine[threadCount].netId = destNetId;
								argsThreadRoutine[threadCount].lGrpId = 0;
								argsThreadRoutine[threadCount].startIdx = firingTableIdxD1;
								argsThreadRoutine[threadCount].endIdx = firingTableIdxD1 + managerRuntimeData.extFiringTableEndIdxD1[lGrpId];
								argsThreadRoutine[threadCount].GtoLOffset = GtoLOffset;

								pthread_create(&threads[threadCount], &attr, &SNN::helperConvertExtSpikesD1_CPU, (void*)&argsThreadRoutine[threadCount]);
								pthread_attr_destroy(&attr);
								threadCount++;
							#endif
					}
					firingTableIdxD1 += managerRuntimeData.extFiringTableEndIdxD1[lGrpId];
				}
			}
			//KERNEL_DEBUG("GPU1 New D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);
		}

		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			// join all the threads
			for (int i=0; i<threadCount; i++){
				pthread_join(threads[i], NULL);
			}
		#endif

		managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD2;
		managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD1;
		writeBackTimeTable(destNetId);
	}
}

//We need pass the neuron id (nid) and the grpId just for the case when we want to
//ramp up/down the weights.  In that case we need to set the weights of each synapse
//depending on their nid (their position with respect to one another). -- KDC
float SNN::generateWeight(int connProp, float initWt, float maxWt, int nid, int grpId) {
	float actWts;
	//// \FIXME: are these ramping thingies still supported?
	//bool setRandomWeights   = GET_INITWTS_RANDOM(connProp);
	//bool setRampDownWeights = GET_INITWTS_RAMPDOWN(connProp);
	//bool setRampUpWeights   = GET_INITWTS_RAMPUP(connProp);

	//if (setRandomWeights)
	//	actWts = initWt * drand48();
	//else if (setRampUpWeights)
	//	actWts = (initWt + ((nid - groupConfigs[0][grpId].StartN) * (maxWt - initWt) / groupConfigs[0][grpId].SizeN));
	//else if (setRampDownWeights)
	//	actWts = (maxWt - ((nid - groupConfigs[0][grpId].StartN) * (maxWt - initWt) / groupConfigs[0][grpId].SizeN));
	//else
		actWts = initWt;

	return actWts;
}

// checks whether a connection ID contains plastic synapses O(#connections)
bool SNN::isConnectionPlastic(short int connId) {
	assert(connId != ALL);
	assert(connId < numConnections);
	
	return GET_FIXED_PLASTIC(connectConfigMap[connId].connProp);
}

// FIXME: distinguish the function call at CONFIG_STATE and SETUP_STATE, where groupConfigs[0][] might not be available
// or groupConfigMap is not sync with groupConfigs[0][]
// returns whether group has homeostasis enabled
bool SNN::isGroupWithHomeostasis(int grpId) {
	assert(grpId>=0 && grpId<getNumGroups());
	return (groupConfigMap[grpId].homeoConfig.WithHomeostasis);
}

// performs various verification checkups before building the network
void SNN::verifyNetwork() {
	// make sure number of neuron parameters have been accumulated correctly
	// NOTE: this used to be updateParameters
	//verifyNumNeurons();

	// make sure compartment config is valid
	verifyCompartments();

	// make sure STDP post-group has some incoming plastic connections
	verifySTDP();

	// make sure every group with homeostasis also has STDP
	verifyHomeostasis();

	// make sure the max delay is within bound
	assert(glbNetworkConfig.maxDelay <= MAX_SYN_DELAY);

	// make sure there is sufficient buffer
	//if ((networkConfigs[0].maxSpikesD1 + networkConfigs[0].maxSpikesD2) < (numNExcReg + numNInhReg + numNPois) * UNKNOWN_NEURON_MAX_FIRING_RATE) {
	//	KERNEL_ERROR("Insufficient amount of buffer allocated...");
	//	exitSimulation(1);
	//}

	//make sure the number of pre- and post-connection does not exceed the limitation
	//if (maxNumPostSynGrp > MAX_NUM_POST_SYN) {
	//	for (int g = 0; g < numGroups; g++) {
	//		if (groupConfigMap[g].numPostSynapses>MAX_NUM_POST_SYN)
	//			KERNEL_ERROR("Grp: %s(%d) has too many output synapses (%d), max %d.",groupInfo[g].Name.c_str(),g,
	//						groupConfigMap[g].numPostSynapses,MAX_NUM_POST_SYN);
	//	}
	//	assert(maxNumPostSynGrp <= MAX_NUM_POST_SYN);
	//}

	//if (maxNumPreSynGrp > MAX_NUM_PRE_SYN) {
	//	for (int g = 0; g < numGroups; g++) {
	//		if (groupConfigMap[g].numPreSynapses>MAX_NUM_PRE_SYN)
	//			KERNEL_ERROR("Grp: %s(%d) has too many input synapses (%d), max %d.",groupInfo[g].Name.c_str(),g,
 //							groupConfigMap[g].numPreSynapses,MAX_NUM_PRE_SYN);
	//	}
	//	assert(maxNumPreSynGrp <= MAX_NUM_PRE_SYN);
	//}

	// make sure maxDelay == 1 if STP is enableed
	// \FIXME: need to figure out STP buffer for delays > 1
	if (sim_with_stp && glbNetworkConfig.maxDelay > 1) {
		KERNEL_ERROR("STP with delays > 1 ms is currently not supported.");
		exitSimulation(1);
	}

	if (glbNetworkConfig.maxDelay > MAX_SYN_DELAY) {
		KERNEL_ERROR("You are using a synaptic delay (%d) greater than MAX_SYN_DELAY defined in config.h", glbNetworkConfig.maxDelay);
		exitSimulation(1);
	}
}

void SNN::verifyCompartments() {
	for (std::map<int, compConnectConfig>::iterator it = compConnectConfigMap.begin(); it != compConnectConfigMap.end(); it++)
	{
		int grpLower = it->second.grpSrc;
		int grpUpper = it->second.grpDest;

		// make sure groups are compartmentally enabled
		if (!groupConfigMap[grpLower].withCompartments) {
			KERNEL_ERROR("Group %s(%d) is not compartmentally enabled, cannot be part of a compartmental connection.",
				groupConfigMap[grpLower].grpName.c_str(), grpLower);
			exitSimulation(1);
		}
		if (!groupConfigMap[grpUpper].withCompartments) {
			KERNEL_ERROR("Group %s(%d) is not compartmentally enabled, cannot be part of a compartmental connection.",
				groupConfigMap[grpUpper].grpName.c_str(), grpUpper);
			exitSimulation(1);
		}
	}
}

// checks whether STDP is set on a post-group with incoming plastic connections
void SNN::verifySTDP() {
	for (int gGrpId=0; gGrpId<getNumGroups(); gGrpId++) {
		if (groupConfigMap[gGrpId].stdpConfig.WithSTDP) {
			// for each post-group, check if any of the incoming connections are plastic
			bool isAnyPlastic = false;
			for (std::map<int, ConnectConfig>::iterator it = connectConfigMap.begin(); it != connectConfigMap.end(); it++) {
				if (it->second.grpDest == gGrpId) {
					// get syn wt type from connection property
					isAnyPlastic |= GET_FIXED_PLASTIC(it->second.connProp);
					if (isAnyPlastic) {
						// at least one plastic connection found: break while
						break;
					}
				}
			}
			if (!isAnyPlastic) {
				KERNEL_ERROR("If STDP on group %d (%s) is set, group must have some incoming plastic connections.",
					gGrpId, groupConfigMap[gGrpId].grpName.c_str());
				exitSimulation(1);
			}
		}
	}
}

// checks whether every group with Homeostasis also has STDP
void SNN::verifyHomeostasis() {
	for (int gGrpId=0; gGrpId<getNumGroups(); gGrpId++) {
		if (groupConfigMap[gGrpId].homeoConfig.WithHomeostasis) {
			if (!groupConfigMap[gGrpId].stdpConfig.WithSTDP) {
				KERNEL_ERROR("If homeostasis is enabled on group %d (%s), then STDP must be enabled, too.",
					gGrpId, groupConfigMap[gGrpId].grpName.c_str());
				exitSimulation(1);
			}
		}
	}
}

//// checks whether the numN* class members are consistent and complete
//void SNN::verifyNumNeurons() {
//	int nExcPois = 0;
//	int nInhPois = 0;
//	int nExcReg = 0;
//	int nInhReg = 0;
//
//	//  scan all the groups and find the required information
//	//  about the group (numN, numPostSynapses, numPreSynapses and others).
//	for(int g=0; g<numGroups; g++)  {
//		if (groupConfigMap[g].Type==UNKNOWN_NEURON) {
//			KERNEL_ERROR("Unknown group for %d (%s)", g, groupInfo[g].Name.c_str());
//			exitSimulation(1);
//		}
//
//		if (IS_INHIBITORY_TYPE(groupConfigMap[g].Type) && !(groupConfigMap[g].Type & POISSON_NEURON))
//			nInhReg += groupConfigMap[g].SizeN;
//		else if (IS_EXCITATORY_TYPE(groupConfigMap[g].Type) && !(groupConfigMap[g].Type & POISSON_NEURON))
//			nExcReg += groupConfigMap[g].SizeN;
//		else if (IS_EXCITATORY_TYPE(groupConfigMap[g].Type) &&  (groupConfigMap[g].Type & POISSON_NEURON))
//			nExcPois += groupConfigMap[g].SizeN;
//		else if (IS_INHIBITORY_TYPE(groupConfigMap[g].Type) &&  (groupConfigMap[g].Type & POISSON_NEURON))
//			nInhPois += groupConfigMap[g].SizeN;
//	}
//
//	// check the newly gathered information with class members
//	if (numN != nExcReg+nInhReg+nExcPois+nInhPois) {
//		KERNEL_ERROR("nExcReg+nInhReg+nExcPois+nInhPois=%d does not add up to numN=%d",
//			nExcReg+nInhReg+nExcPois+nInhPois, numN);
//		exitSimulation(1);
//	}
//	if (numNReg != nExcReg+nInhReg) {
//		KERNEL_ERROR("nExcReg+nInhReg=%d does not add up to numNReg=%d", nExcReg+nInhReg, numNReg);
//		exitSimulation(1);
//	}
//	if (numNPois != nExcPois+nInhPois) {
//		KERNEL_ERROR("nExcPois+nInhPois=%d does not add up to numNPois=%d", nExcPois+nInhPois, numNPois);
//		exitSimulation(1);
//	}
//
//	//printf("numN=%d == %d\n",numN,nExcReg+nInhReg+nExcPois+nInhPois);
//	//printf("numNReg=%d == %d\n",numNReg, nExcReg+nInhReg);
//	//printf("numNPois=%d == %d\n",numNPois, nExcPois+nInhPois);
//	
//	assert(numN <= 1000000);
//	assert((numN > 0) && (numN == numNExcReg + numNInhReg + numNPois));
//}

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

void SNN::partitionSNN() {
	int numAssignedNeurons[MAX_NET_PER_SNN] = {0};

	// get number of available GPU card(s) in the present machine
	numAvailableGPUs = configGPUDevice();

	for (std::map<int, GroupConfigMD>::iterator grpIt = groupConfigMDMap.begin(); grpIt != groupConfigMDMap.end(); grpIt++) {
		// assign a group to the GPU specified by users
		int gGrpId = grpIt->second.gGrpId;
		int netId = groupConfigMap[gGrpId].preferredNetId;
		if (netId != ANY) {
			assert(netId > ANY && netId < MAX_NET_PER_SNN);
			grpIt->second.netId = netId;
			numAssignedNeurons[netId] += groupConfigMap[gGrpId].numN;
			groupPartitionLists[netId].push_back(grpIt->second); // Copy by value, create a copy
		} else { // netId == ANY
			// TODO: add callback function that allow user to partition network by theirself
			// FIXME: make sure GPU(s) is available first
			// this parse separates groups into each local network and assign each group a netId
			if (preferredSimMode_ == CPU_MODE) {
				grpIt->second.netId = CPU_RUNTIME_BASE; // CPU 0
				numAssignedNeurons[CPU_RUNTIME_BASE] += groupConfigMap[gGrpId].numN;
				groupPartitionLists[CPU_RUNTIME_BASE].push_back(grpIt->second); // Copy by value, create a copy
			} else if (preferredSimMode_ == GPU_MODE) {
				grpIt->second.netId = GPU_RUNTIME_BASE; // GPU 0
				numAssignedNeurons[GPU_RUNTIME_BASE] += groupConfigMap[gGrpId].numN;
				groupPartitionLists[GPU_RUNTIME_BASE].push_back(grpIt->second); // Copy by value, create a copy
			} else  if (preferredSimMode_ == HYBRID_MODE) {
				// TODO: implement partition algorithm, use naive partition for now (allocate to CPU 0)
				grpIt->second.netId = CPU_RUNTIME_BASE; // CPU 0
				numAssignedNeurons[CPU_RUNTIME_BASE] += groupConfigMap[gGrpId].numN;
				groupPartitionLists[CPU_RUNTIME_BASE].push_back(grpIt->second); // Copy by value, create a copy
			} else {
				KERNEL_ERROR("Unkown simulation mode");
				exitSimulation(-1);
			}
		}

		if (grpIt->second.netId == -1) { // the group was not assigned to any computing backend
			KERNEL_ERROR("Can't assign the group [%d] to any partition", grpIt->second.gGrpId);
			exitSimulation(-1);
		}
	}

	// this parse finds local connections (i.e., connection configs that conect local groups)
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			for (std::map<int, ConnectConfig>::iterator connIt = connectConfigMap.begin(); connIt != connectConfigMap.end(); connIt++) {
				if (groupConfigMDMap[connIt->second.grpSrc].netId == netId && groupConfigMDMap[connIt->second.grpDest].netId == netId) {
					localConnectLists[netId].push_back(connectConfigMap[connIt->second.connId]); // Copy by value
				}
			}

			//printf("The size of compConnectConfigMap is: %i\n", compConnectConfigMap.size());
			for (std::map<int, compConnectConfig>::iterator connIt = compConnectConfigMap.begin(); connIt != compConnectConfigMap.end(); connIt++) {
				if (groupConfigMDMap[connIt->second.grpSrc].netId == netId && groupConfigMDMap[connIt->second.grpDest].netId == netId) {
					localCompConnectLists[netId].push_back(compConnectConfigMap[connIt->second.connId]); // Copy by value
				}
			}
		}
	}

	// this parse finds external groups and external connections
	spikeRoutingTable.clear();
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			for (std::map<int, ConnectConfig>::iterator connIt = connectConfigMap.begin(); connIt != connectConfigMap.end(); connIt++) {
				int srcNetId = groupConfigMDMap[connIt->second.grpSrc].netId;
				int destNetId = groupConfigMDMap[connIt->second.grpDest].netId;
				if (srcNetId == netId && destNetId != netId) {
					// search the source group in groupPartitionLists and mark it as having external connections
					GroupConfigMD targetGroup;
					std::list<GroupConfigMD>::iterator srcGrpIt, destGrpIt;
					
					targetGroup.gGrpId = connIt->second.grpSrc;
					srcGrpIt = find(groupPartitionLists[srcNetId].begin(), groupPartitionLists[srcNetId].end(), targetGroup);
					assert(srcGrpIt != groupPartitionLists[srcNetId].end());
					srcGrpIt->hasExternalConnect = true;

					// FIXME: fail to write external group if the only one external link across GPUs is uni directional (GPU0 -> GPU1, no GPU1 -> GPU0)
					targetGroup.gGrpId = connIt->second.grpDest;
					destGrpIt = find(groupPartitionLists[srcNetId].begin(), groupPartitionLists[srcNetId].end(), targetGroup);
					if (destGrpIt == groupPartitionLists[srcNetId].end()) { // the "external" dest group has not yet been copied to te "local" group partition list
						numAssignedNeurons[srcNetId] += groupConfigMap[connIt->second.grpDest].numN;
						groupPartitionLists[srcNetId].push_back(groupConfigMDMap[connIt->second.grpDest]);
					}

					targetGroup.gGrpId = connIt->second.grpSrc;
					srcGrpIt = find(groupPartitionLists[destNetId].begin(), groupPartitionLists[destNetId].end(), targetGroup);
					if (srcGrpIt == groupPartitionLists[destNetId].end()) {
						numAssignedNeurons[destNetId] += groupConfigMap[connIt->second.grpSrc].numN;
						groupPartitionLists[destNetId].push_back(groupConfigMDMap[connIt->second.grpSrc]);
					}

					externalConnectLists[srcNetId].push_back(connectConfigMap[connIt->second.connId]); // Copy by value
					
					// build the spike routing table by the way
					//printf("%d,%d -> %d,%d\n", srcNetId, connIt->second.grpSrc, destNetId, connIt->second.grpDest);
					RoutingTableEntry rte(srcNetId, destNetId);
					spikeRoutingTable.push_back(rte);
				}
			}
		}
	}

	spikeRoutingTable.unique();

	// assign local neuron ids and, local group ids for each local network in the order
	// MPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP
	// <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory --> | <-- External -->
	// Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson | External Neurons
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			int availableNeuronId = 0;
			int localGroupId = 0;
			for (int order = 0; order < 5; order++) {
				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++) {
					unsigned int type = groupConfigMap[grpIt->gGrpId].type;
					if (IS_EXCITATORY_TYPE(type) && (type & POISSON_NEURON) && order == 3 && grpIt->netId == netId) {
						availableNeuronId = assignGroup(grpIt, localGroupId, availableNeuronId);
						localGroupId++;
					} else if (IS_INHIBITORY_TYPE(type) && (type & POISSON_NEURON) && order == 2 && grpIt->netId == netId) {
						availableNeuronId = assignGroup(grpIt, localGroupId, availableNeuronId);
						localGroupId++;
					} else if (IS_EXCITATORY_TYPE(type) && !(type & POISSON_NEURON) && order == 0 && grpIt->netId == netId) {
						availableNeuronId = assignGroup(grpIt, localGroupId, availableNeuronId);
						localGroupId++;
					} else if (IS_INHIBITORY_TYPE(type) && !(type & POISSON_NEURON) && order == 1 && grpIt->netId == netId) {
						availableNeuronId = assignGroup(grpIt, localGroupId, availableNeuronId);
						localGroupId++;
					} else if (order == 4 && grpIt->netId != netId) {
						availableNeuronId = assignGroup(grpIt, localGroupId, availableNeuronId);
						localGroupId++;
					}
				}
			}
			assert(availableNeuronId == numAssignedNeurons[netId]);
			assert(localGroupId == groupPartitionLists[netId].size());
		}
	}


	// generation connections among groups according to group and connect configs
	// update ConnectConfig::numberOfConnections
	// update GroupConfig::numPostSynapses, GroupConfig::numPreSynapses
	connectNetwork();

	collectGlobalNetworkConfigP();

	// print group and connection overview
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			KERNEL_INFO("\n+ Local Network (%d)", netId);
			KERNEL_INFO("|-+ Group List:");
			for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[netId].begin(); grpIt != groupPartitionLists[netId].end(); grpIt++)
				printGroupInfo(netId, grpIt);
		}

		if (!localConnectLists[netId].empty() || !externalConnectLists[netId].empty()) {
			KERNEL_INFO("|-+ Connection List:");
			for (std::list<ConnectConfig>::iterator connIt = localConnectLists[netId].begin(); connIt != localConnectLists[netId].end(); connIt++)
				printConnectionInfo(netId, connIt);

			for (std::list<ConnectConfig>::iterator connIt = externalConnectLists[netId].begin(); connIt != externalConnectLists[netId].end(); connIt++)
				printConnectionInfo(netId, connIt);
		}
	}

	// print spike routing table
	printSikeRoutingInfo();

	snnState = PARTITIONED_SNN;
}

int SNN::loadSimulation_internal(bool onlyPlastic) {
	//// TSC: so that we can restore the file position later...
	//// MB: not sure why though...
	//long file_position = ftell(loadSimFID);
	//
	//int tmpInt;
	//float tmpFloat;

	//bool readErr = false; // keep track of reading errors
	//size_t result;


	//// ------- read header ----------------

	//fseek(loadSimFID, 0, SEEK_SET);

	//// read file signature
	//result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (tmpInt != 294338571) {
	//	KERNEL_ERROR("loadSimulation: Unknown file signature. This does not seem to be a "
	//		"simulation file created with CARLsim::saveSimulation.");
	//	exitSimulation(-1);
	//}

	//// read file version number
	//result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (tmpFloat > 0.2f) {
	//	KERNEL_ERROR("loadSimulation: Unsupported version number (%f)",tmpFloat);
	//	exitSimulation(-1);
	//}

	//// read simulation time
	//result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	//readErr |= (result!=1);

	//// read execution time
	//result = fread(&tmpFloat, sizeof(float), 1, loadSimFID);
	//readErr |= (result!=1);

	//// read number of neurons
	//result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (tmpInt != numN) {
	//	KERNEL_ERROR("loadSimulation: Number of neurons in file (%d) and simulation (%d) don't match.",
	//		tmpInt, numN);
	//	exitSimulation(-1);
	//}

	//// read number of pre-synapses
	//result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (numPreSynNet != tmpInt) {
	//	KERNEL_ERROR("loadSimulation: numPreSynNet in file (%d) and simulation (%d) don't match.",
	//		tmpInt, numPreSynNet);
	//	exitSimulation(-1);
	//}

	//// read number of post-synapses
	//result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (numPostSynNet != tmpInt) {
	//	KERNEL_ERROR("loadSimulation: numPostSynNet in file (%d) and simulation (%d) don't match.",
	//		tmpInt, numPostSynNet);
	//	exitSimulation(-1);
	//}

	//// read number of groups
	//result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//readErr |= (result!=1);
	//if (tmpInt != numGroups) {
	//	KERNEL_ERROR("loadSimulation: Number of groups in file (%d) and simulation (%d) don't match.",
	//		tmpInt, numGroups);
	//	exitSimulation(-1);
	//}

	//// throw reading error instead of proceeding
	//if (readErr) {
	//	fprintf(stderr,"loadSimulation: Error while reading file header");
	//	exitSimulation(-1);
	//}


	//// ------- read group information ----------------

	//for (int g=0; g<numGroups; g++) {
	//	// read StartN
	//	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);
	//	if (tmpInt != groupConfigs[0][g].StartN) {
	//		KERNEL_ERROR("loadSimulation: StartN in file (%d) and grpInfo (%d) for group %d don't match.",
	//			tmpInt, groupConfigs[0][g].StartN, g);
	//		exitSimulation(-1);
	//	}

	//	// read EndN
	//	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);
	//	if (tmpInt != groupConfigs[0][g].EndN) {
	//		KERNEL_ERROR("loadSimulation: EndN in file (%d) and grpInfo (%d) for group %d don't match.",
	//			tmpInt, groupConfigs[0][g].EndN, g);
	//		exitSimulation(-1);
	//	}

	//	// read SizeX
	//	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);

	//	// read SizeY
	//	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);

	//	// read SizeZ
	//	result = fread(&tmpInt, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);

	//	// read group name
	//	char name[100];
	//	result = fread(name, sizeof(char), 100, loadSimFID);
	//	readErr |= (result!=100);
	//	if (strcmp(name,groupInfo[g].Name.c_str()) != 0) {
	//		KERNEL_ERROR("loadSimulation: Group names in file (%s) and grpInfo (%s) don't match.", name,
	//			groupInfo[g].Name.c_str());
	//		exitSimulation(-1);
	//	}
	//}

	//if (readErr) {
	//	KERNEL_ERROR("loadSimulation: Error while reading group info");
	//	exitSimulation(-1);
	//}


	//// ------- read synapse information ----------------

	//for (int i = 0; i < numN; i++) {
	//	int nrSynapses = 0;

	//	// read number of synapses
	//	result = fread(&nrSynapses, sizeof(int), 1, loadSimFID);
	//	readErr |= (result!=1);

	//	for (int j=0; j<nrSynapses; j++) {
	//		int nIDpre;
	//		int nIDpost;
	//		float weight, maxWeight;
	//		uint8_t delay;
	//		uint8_t plastic;
	//		short int connId;

	//		// read nIDpre
	//		result = fread(&nIDpre, sizeof(int), 1, loadSimFID);
	//		readErr |= (result!=1);
	//		if (nIDpre != i) {
	//			KERNEL_ERROR("loadSimulation: nIDpre in file (%u) and simulation (%u) don't match.", nIDpre, i);
	//			exitSimulation(-1);
	//		}

	//		// read nIDpost
	//		result = fread(&nIDpost, sizeof(int), 1, loadSimFID);
	//		readErr |= (result!=1);
	//		if (nIDpost >= numN) {
	//			KERNEL_ERROR("loadSimulation: nIDpre in file (%u) is larger than in simulation (%u).", nIDpost, numN);
	//			exitSimulation(-1);
	//		}

	//		// read weight
	//		result = fread(&weight, sizeof(float), 1, loadSimFID);
	//		readErr |= (result!=1);

	//		short int gIDpre = managerRuntimeData.grpIds[nIDpre];
	//		if (IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type) && (weight>0)
	//				|| !IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type) && (weight<0)) {
	//			KERNEL_ERROR("loadSimulation: Sign of weight value (%s) does not match neuron type (%s)",
	//				((weight>=0.0f)?"plus":"minus"), 
	//				(IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type)?"inhibitory":"excitatory"));
	//			exitSimulation(-1);
	//		}

	//		// read max weight
	//		result = fread(&maxWeight, sizeof(float), 1, loadSimFID);
	//		readErr |= (result!=1);
	//		if (IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type) && (maxWeight>=0)
	//				|| !IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type) && (maxWeight<=0)) {
	//			KERNEL_ERROR("loadSimulation: Sign of maxWeight value (%s) does not match neuron type (%s)",
	//				((maxWeight>=0.0f)?"plus":"minus"), 
	//				(IS_INHIBITORY_TYPE(groupConfigs[0][gIDpre].Type)?"inhibitory":"excitatory"));
	//			exitSimulation(-1);
	//		}

	//		// read delay
	//		result = fread(&delay, sizeof(uint8_t), 1, loadSimFID);
	//		readErr |= (result!=1);
	//		if (delay > MAX_SYN_DELAY) {
	//			KERNEL_ERROR("loadSimulation: delay in file (%d) is larger than MAX_SYN_DELAY (%d)",
	//				(int)delay, (int)MAX_SYN_DELAY);
	//			exitSimulation(-1);
	//		}

	//		assert(!isnan(weight));
	//		// read plastic/fixed
	//		result = fread(&plastic, sizeof(uint8_t), 1, loadSimFID);
	//		readErr |= (result!=1);

	//		// read connection ID
	//		result = fread(&connId, sizeof(short int), 1, loadSimFID);
	//		readErr |= (result!=1);

	//		if ((plastic && onlyPlastic) || (!plastic && !onlyPlastic)) {
	//			int gIDpost = managerRuntimeData.grpIds[nIDpost];
	//			int connProp = SET_FIXED_PLASTIC(plastic?SYN_PLASTIC:SYN_FIXED);

	//			//setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp, connId);
	//			groupInfo[gIDpre].sumPostConn++;
	//			groupInfo[gIDpost].sumPreConn++;

	//			if (delay > groupConfigs[0][gIDpre].MaxDelay)
	//				groupConfigs[0][gIDpre].MaxDelay = delay;
	//		}
	//	}
	//}

	//fseek(loadSimFID,file_position,SEEK_SET);

	return 0;
}

void SNN::generateRuntimeSNN() {
	// 1. genearte configurations for the simulation
	// generate (copy) group configs from groupPartitionLists[]
	generateRuntimeGroupConfigs();

	// generate (copy) connection configs from localConnectLists[] and exeternalConnectLists[]
	generateRuntimeConnectConfigs();

	// generate local network configs and accquire maximum size of rumtime data
	generateRuntimeNetworkConfigs();

	// 2. allocate space of runtime data used by the manager
	// - allocate firingTableD1, firingTableD2, timeTableD1, timeTableD2
	// - reset firingTableD1, firingTableD2, timeTableD1, timeTableD2
	allocateManagerSpikeTables();
	// - allocate voltage, recovery, Izh_a, Izh_b, Izh_c, Izh_d, current, extCurrent, gAMPA, gNMDA, gGABAa, gGABAb
	// lastSpikeTime, nSpikeCnt, stpu, stpx, Npre, Npre_plastic, Npost, cumulativePost, cumulativePre,
	// postSynapticIds, postDelayInfo, wt, wtChange, synSpikeTime, maxSynWt, preSynapticIds, grpIds, connIdsPreIdx,
	// grpDA, grp5HT, grpACh, grpNE, grpDABuffer, grp5HTBuffer, grpAChBuffer, grpNEBuffer, mulSynFast, mulSynSlow
	// - reset all above
	allocateManagerRuntimeData();

	// 3. initialize manager runtime data according to partitions (i.e., local networks)
	// 4a. allocate appropriate memory space (e.g., main memory (CPU) or device memory (GPU)).
	// 4b. load (copy) them to appropriate memory space for execution
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {	
			KERNEL_INFO("");
			if (netId < CPU_RUNTIME_BASE) {
				KERNEL_INFO("*****************      Initializing GPU %d Runtime      *************************", netId);
			} else {
				KERNEL_INFO("*****************      Initializing CPU %d Runtime      *************************", (netId - CPU_RUNTIME_BASE));
			}
			// build the runtime data according to local network, group, connection configuirations

			// generate runtime data for each group
			for(int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
				// local poisson neurons
				if (groupConfigs[netId][lGrpId].netId == netId && (groupConfigs[netId][lGrpId].Type & POISSON_NEURON)) {
					// - init lstSpikeTime
					// - reset avgFiring, stpu, stpx
					// - init stpx
					generatePoissonGroupRuntime(netId, lGrpId);
				}
				// local regular neurons
				if (groupConfigs[netId][lGrpId].netId == netId && !(groupConfigs[netId][lGrpId].Type & POISSON_NEURON)) {
					// - init grpDA, grp5HT, grpACh, grpNE
					// - init Izh_a, Izh_b, Izh_c, Izh_d, voltage, recovery, stpu, stpx
					// - init baseFiring, avgFiring
					// - init lastSpikeTime
					generateGroupRuntime(netId, lGrpId);
				}
			}

			// - init grpIds
			for (int lNId = 0; lNId < networkConfigs[netId].numNAssigned; lNId++) {
				managerRuntimeData.grpIds[lNId] = -1;
				for(int lGrpId = 0; lGrpId < networkConfigs[netId].numGroupsAssigned; lGrpId++) {
					if (lNId >= groupConfigs[netId][lGrpId].lStartN && lNId <= groupConfigs[netId][lGrpId].lEndN) {
						managerRuntimeData.grpIds[lNId] = (short int)lGrpId;
						break;
					}
				}
				assert(managerRuntimeData.grpIds[lNId] != -1);
			}

			// - init mulSynFast, mulSynSlow
			// - init Npre, Npre_plastic, Npost, cumulativePre, cumulativePost, preSynapticIds, postSynapticIds, postDelayInfo
			// - init wt, maxSynWt
			generateConnectionRuntime(netId);

			generateCompConnectionRuntime(netId);

			// - reset current
			resetCurrent(netId);
			// - reset conductance
			resetConductances(netId);

			// - reset wtChange
			// - init synSpikeTime
			resetSynapse(netId, false);

			allocateSNN(netId);
		}
	}

	// count allocated CPU/GPU runtime
	numGPUs = 0; numCores = 0;
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (netId < CPU_RUNTIME_BASE && runtimeData[netId].allocated)
			numGPUs++;
		if (netId >= CPU_RUNTIME_BASE && runtimeData[netId].allocated)
			numCores++;
	}

	// 5. declare the spiking neural network is excutable
	snnState = EXECUTABLE_SNN;
}

void SNN::resetConductances(int netId) {
	if (networkConfigs[netId].sim_with_conductances) {
		memset(managerRuntimeData.gAMPA, 0, sizeof(float) * networkConfigs[netId].numNReg);
		if (networkConfigs[netId].sim_with_NMDA_rise) {
			memset(managerRuntimeData.gNMDA_r, 0, sizeof(float) * networkConfigs[netId].numNReg);
			memset(managerRuntimeData.gNMDA_d, 0, sizeof(float) * networkConfigs[netId].numNReg);
		} else {
			memset(managerRuntimeData.gNMDA, 0, sizeof(float) * networkConfigs[netId].numNReg);
		}
		memset(managerRuntimeData.gGABAa, 0, sizeof(float) * networkConfigs[netId].numNReg);
		if (networkConfigs[netId].sim_with_GABAb_rise) {
			memset(managerRuntimeData.gGABAb_r, 0, sizeof(float) * networkConfigs[netId].numNReg);
			memset(managerRuntimeData.gGABAb_d, 0, sizeof(float) * networkConfigs[netId].numNReg);
		} else {
			memset(managerRuntimeData.gGABAb, 0, sizeof(float) * networkConfigs[netId].numNReg);
		}
	}
}

void SNN::resetCurrent(int netId) {
	assert(managerRuntimeData.current != NULL);
	memset(managerRuntimeData.current, 0, sizeof(float) * networkConfigs[netId].numNReg);
}

// FIXME: unused function
void SNN::resetFiringInformation() {
	// Reset firing tables and time tables to default values..

	// reset various times...
	simTimeMs  = 0;
	simTimeSec = 0;
	simTime    = 0;

	// reset the propogation Buffer.
	resetPropogationBuffer();
	// reset Timing  Table..
	resetTimeTable();
}

void SNN::resetTiming() {
	prevExecutionTime = cumExecutionTime;
	executionTime = 0.0f;
}

void SNN::resetNeuromodulator(int netId, int lGrpId) {
	managerRuntimeData.grpDA[lGrpId] = groupConfigs[netId][lGrpId].baseDP;
	managerRuntimeData.grp5HT[lGrpId] = groupConfigs[netId][lGrpId].base5HT;
	managerRuntimeData.grpACh[lGrpId] = groupConfigs[netId][lGrpId].baseACh;
	managerRuntimeData.grpNE[lGrpId] = groupConfigs[netId][lGrpId].baseNE;
}

/*!
 * \brief reset neurons using local ids
 */
void SNN::resetNeuron(int netId, int lGrpId, int lNId) {
	int gGrpId = groupConfigs[netId][lGrpId].gGrpId; // get global group id
	assert(lNId < networkConfigs[netId].numNReg);

	if (groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a == -1 && groupConfigMap[gGrpId].isLIF == 0) {
		KERNEL_ERROR("setNeuronParameters must be called for group %s (G:%d,L:%d)",groupConfigMap[gGrpId].grpName.c_str(), gGrpId, lGrpId);
		exitSimulation(1);
	}

	if (groupConfigMap[gGrpId].neuralDynamicsConfig.lif_tau_m == -1 && groupConfigMap[gGrpId].isLIF == 1) {
		KERNEL_ERROR("setNeuronParametersLIF must be called for group %s (G:%d,L:%d)",groupConfigMap[gGrpId].grpName.c_str(), gGrpId, lGrpId);
		exitSimulation(1);
	}

	managerRuntimeData.Izh_a[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_a_sd * (float)drand48();
	managerRuntimeData.Izh_b[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_b_sd * (float)drand48();
	managerRuntimeData.Izh_c[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_c_sd * (float)drand48();
	managerRuntimeData.Izh_d[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_d_sd * (float)drand48();
	managerRuntimeData.Izh_C[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_C + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_C_sd * (float)drand48();
	managerRuntimeData.Izh_k[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_k + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_k_sd * (float)drand48();
	managerRuntimeData.Izh_vr[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vr + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vr_sd * (float)drand48();
	managerRuntimeData.Izh_vt[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vt + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vt_sd * (float)drand48();
	managerRuntimeData.Izh_vpeak[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vpeak + groupConfigMap[gGrpId].neuralDynamicsConfig.Izh_vpeak_sd * (float)drand48();
	managerRuntimeData.lif_tau_m[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.lif_tau_m;
	managerRuntimeData.lif_tau_ref[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.lif_tau_ref;
	managerRuntimeData.lif_tau_ref_c[lNId] = 0;
	managerRuntimeData.lif_vTh[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.lif_vTh;
	managerRuntimeData.lif_vReset[lNId] = groupConfigMap[gGrpId].neuralDynamicsConfig.lif_vReset;
	
	// calculate gain and bias for the lif neuron
	if (groupConfigs[netId][lGrpId].isLIF){
		// gain an bias of the LIF neuron is calculated based on Membrane resistance
		float rmRange = (float)(groupConfigMap[gGrpId].neuralDynamicsConfig.lif_maxRmem - groupConfigMap[gGrpId].neuralDynamicsConfig.lif_minRmem);
		float minRmem = (float)groupConfigMap[gGrpId].neuralDynamicsConfig.lif_minRmem;
		managerRuntimeData.lif_bias[lNId] = 0.0f;
		managerRuntimeData.lif_gain[lNId] = minRmem + rmRange * (float)drand48();
	}

	managerRuntimeData.nextVoltage[lNId] = managerRuntimeData.voltage[lNId] = groupConfigs[netId][lGrpId].isLIF ? managerRuntimeData.lif_vReset[lNId] : (groupConfigs[netId][lGrpId].withParamModel_9 ? managerRuntimeData.Izh_vr[lNId] : managerRuntimeData.Izh_c[lNId]);
	managerRuntimeData.recovery[lNId] = groupConfigs[netId][lGrpId].withParamModel_9 ? 0.0f : managerRuntimeData.Izh_b[lNId] * managerRuntimeData.voltage[lNId];

 	if (groupConfigs[netId][lGrpId].WithHomeostasis) {
		// set the baseFiring with some standard deviation.
		if (drand48() > 0.5) {
			managerRuntimeData.baseFiring[lNId] = groupConfigMap[gGrpId].homeoConfig.baseFiring + groupConfigMap[gGrpId].homeoConfig.baseFiringSD * -log(drand48());
		} else {
			managerRuntimeData.baseFiring[lNId] = groupConfigMap[gGrpId].homeoConfig.baseFiring - groupConfigMap[gGrpId].homeoConfig.baseFiringSD * -log(drand48());
			if(managerRuntimeData.baseFiring[lNId] < 0.1f) managerRuntimeData.baseFiring[lNId] = 0.1f;
		}

		if (groupConfigMap[gGrpId].homeoConfig.baseFiring != 0.0f) {
			managerRuntimeData.avgFiring[lNId] = managerRuntimeData.baseFiring[lNId];
		} else {
			managerRuntimeData.baseFiring[lNId] = 0.0f;
			managerRuntimeData.avgFiring[lNId]  = 0.0f;
		}
	}

	managerRuntimeData.lastSpikeTime[lNId] = MAX_SIMULATION_TIME;

	if(groupConfigs[netId][lGrpId].WithSTP) {
		for (int j = 0; j < networkConfigs[netId].maxDelay + 1; j++) { // is of size maxDelay_+1
			int index = STP_BUF_POS(lNId, j, networkConfigs[netId].maxDelay);
			managerRuntimeData.stpu[index] = 0.0f;
			managerRuntimeData.stpx[index] = 1.0f;
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

	// delete all NeuronMonitor objects
	// don't kill NeuronMonitorCore objects, they will get killed automatically
	for (int i = 0; i<numNeuronMonitor; i++) {
		if (neuronMonList[i] != NULL && deallocate) delete neuronMonList[i];
		neuronMonList[i] = NULL;
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

void SNN::resetGroupConfigs(bool deallocate) {
	// clear all existing group configurations
	if (deallocate) groupConfigMap.clear();
}

void SNN::resetConnectionConfigs(bool deallocate) {
	// clear all existing connection configurations
	if (deallocate) connectConfigMap.clear();
}

void SNN::deleteManagerRuntimeData() {
	if (spikeBuf!=NULL) delete spikeBuf;
	if (managerRuntimeData.spikeGenBits!=NULL) delete[] managerRuntimeData.spikeGenBits;
	spikeBuf=NULL; managerRuntimeData.spikeGenBits=NULL;

	// clear data (i.e., concentration of neuromodulator) of groups
	if (managerRuntimeData.grpDA != NULL) delete [] managerRuntimeData.grpDA;
	if (managerRuntimeData.grp5HT != NULL) delete [] managerRuntimeData.grp5HT;
	if (managerRuntimeData.grpACh != NULL) delete [] managerRuntimeData.grpACh;
	if (managerRuntimeData.grpNE != NULL) delete [] managerRuntimeData.grpNE;
	managerRuntimeData.grpDA = NULL;
	managerRuntimeData.grp5HT = NULL;
	managerRuntimeData.grpACh = NULL;
	managerRuntimeData.grpNE = NULL;

	// clear assistive data buffer for group monitor
	if (managerRuntimeData.grpDABuffer != NULL) delete [] managerRuntimeData.grpDABuffer;
	if (managerRuntimeData.grp5HTBuffer != NULL) delete [] managerRuntimeData.grp5HTBuffer;
	if (managerRuntimeData.grpAChBuffer != NULL) delete [] managerRuntimeData.grpAChBuffer;
	if (managerRuntimeData.grpNEBuffer != NULL) delete [] managerRuntimeData.grpNEBuffer;
	managerRuntimeData.grpDABuffer = NULL; managerRuntimeData.grp5HTBuffer = NULL;
	managerRuntimeData.grpAChBuffer = NULL; managerRuntimeData.grpNEBuffer = NULL;

	// -------------- DEALLOCATE CORE OBJECTS ---------------------- //

	if (managerRuntimeData.voltage!=NULL) delete[] managerRuntimeData.voltage;
	if (managerRuntimeData.nextVoltage != NULL) delete[] managerRuntimeData.nextVoltage;
	if (managerRuntimeData.recovery!=NULL) delete[] managerRuntimeData.recovery;
	if (managerRuntimeData.current!=NULL) delete[] managerRuntimeData.current;
	if (managerRuntimeData.extCurrent!=NULL) delete[] managerRuntimeData.extCurrent;
	if (managerRuntimeData.totalCurrent != NULL) delete[] managerRuntimeData.totalCurrent;
	if (managerRuntimeData.curSpike != NULL) delete[] managerRuntimeData.curSpike;
	if (managerRuntimeData.nVBuffer != NULL) delete[] managerRuntimeData.nVBuffer;
	if (managerRuntimeData.nUBuffer != NULL) delete[] managerRuntimeData.nUBuffer;
	if (managerRuntimeData.nIBuffer != NULL) delete[] managerRuntimeData.nIBuffer;
	managerRuntimeData.voltage=NULL; managerRuntimeData.recovery=NULL; managerRuntimeData.current=NULL; managerRuntimeData.extCurrent=NULL;
	managerRuntimeData.nextVoltage = NULL; managerRuntimeData.totalCurrent = NULL; managerRuntimeData.curSpike = NULL;
	managerRuntimeData.nVBuffer = NULL; managerRuntimeData.nUBuffer = NULL; managerRuntimeData.nIBuffer = NULL;

	if (managerRuntimeData.Izh_a!=NULL) delete[] managerRuntimeData.Izh_a;
	if (managerRuntimeData.Izh_b!=NULL) delete[] managerRuntimeData.Izh_b;
	if (managerRuntimeData.Izh_c!=NULL) delete[] managerRuntimeData.Izh_c;
	if (managerRuntimeData.Izh_d!=NULL) delete[] managerRuntimeData.Izh_d;
	if (managerRuntimeData.Izh_C!=NULL) delete[] managerRuntimeData.Izh_C;
	if (managerRuntimeData.Izh_k!=NULL) delete[] managerRuntimeData.Izh_k;
	if (managerRuntimeData.Izh_vr!=NULL) delete[] managerRuntimeData.Izh_vr;
	if (managerRuntimeData.Izh_vt!=NULL) delete[] managerRuntimeData.Izh_vt;
	if (managerRuntimeData.Izh_vpeak!=NULL) delete[] managerRuntimeData.Izh_vpeak;
	managerRuntimeData.Izh_a=NULL; managerRuntimeData.Izh_b=NULL; managerRuntimeData.Izh_c=NULL; managerRuntimeData.Izh_d=NULL;
	managerRuntimeData.Izh_C = NULL; managerRuntimeData.Izh_k = NULL; managerRuntimeData.Izh_vr = NULL; managerRuntimeData.Izh_vt = NULL; managerRuntimeData.Izh_vpeak = NULL;

	if (managerRuntimeData.lif_tau_m!=NULL) delete[] managerRuntimeData.lif_tau_m;
	if (managerRuntimeData.lif_tau_ref!=NULL) delete[] managerRuntimeData.lif_tau_ref;
	if (managerRuntimeData.lif_tau_ref_c!=NULL) delete[] managerRuntimeData.lif_tau_ref_c;
	if (managerRuntimeData.lif_vTh!=NULL) delete[] managerRuntimeData.lif_vTh;
	if (managerRuntimeData.lif_vReset!=NULL) delete[] managerRuntimeData.lif_vReset;
	if (managerRuntimeData.lif_gain!=NULL) delete[] managerRuntimeData.lif_gain;
	if (managerRuntimeData.lif_bias!=NULL) delete[] managerRuntimeData.lif_bias;
	managerRuntimeData.lif_tau_m=NULL; managerRuntimeData.lif_tau_ref=NULL; managerRuntimeData.lif_vTh=NULL;
	managerRuntimeData.lif_vReset=NULL; managerRuntimeData.lif_gain=NULL; managerRuntimeData.lif_bias=NULL;
	managerRuntimeData.lif_tau_ref_c=NULL;
	
	if (managerRuntimeData.Npre!=NULL) delete[] managerRuntimeData.Npre;
	if (managerRuntimeData.Npre_plastic!=NULL) delete[] managerRuntimeData.Npre_plastic;
	if (managerRuntimeData.Npost!=NULL) delete[] managerRuntimeData.Npost;
	managerRuntimeData.Npre=NULL; managerRuntimeData.Npre_plastic=NULL; managerRuntimeData.Npost=NULL;

	if (managerRuntimeData.cumulativePre!=NULL) delete[] managerRuntimeData.cumulativePre;
	if (managerRuntimeData.cumulativePost!=NULL) delete[] managerRuntimeData.cumulativePost;
	managerRuntimeData.cumulativePre=NULL; managerRuntimeData.cumulativePost=NULL;

	if (managerRuntimeData.gAMPA!=NULL) delete[] managerRuntimeData.gAMPA;
	if (managerRuntimeData.gNMDA!=NULL) delete[] managerRuntimeData.gNMDA;
	if (managerRuntimeData.gNMDA_r!=NULL) delete[] managerRuntimeData.gNMDA_r;
	if (managerRuntimeData.gNMDA_d!=NULL) delete[] managerRuntimeData.gNMDA_d;
	if (managerRuntimeData.gGABAa!=NULL) delete[] managerRuntimeData.gGABAa;
	if (managerRuntimeData.gGABAb!=NULL) delete[] managerRuntimeData.gGABAb;
	if (managerRuntimeData.gGABAb_r!=NULL) delete[] managerRuntimeData.gGABAb_r;
	if (managerRuntimeData.gGABAb_d!=NULL) delete[] managerRuntimeData.gGABAb_d;
	managerRuntimeData.gAMPA=NULL; managerRuntimeData.gNMDA=NULL; managerRuntimeData.gNMDA_r=NULL; managerRuntimeData.gNMDA_d=NULL;
	managerRuntimeData.gGABAa=NULL; managerRuntimeData.gGABAb=NULL; managerRuntimeData.gGABAb_r=NULL; managerRuntimeData.gGABAb_d=NULL;

	if (managerRuntimeData.stpu!=NULL) delete[] managerRuntimeData.stpu;
	if (managerRuntimeData.stpx!=NULL) delete[] managerRuntimeData.stpx;
	managerRuntimeData.stpu=NULL; managerRuntimeData.stpx=NULL;

	if (managerRuntimeData.avgFiring!=NULL) delete[] managerRuntimeData.avgFiring;
	if (managerRuntimeData.baseFiring!=NULL) delete[] managerRuntimeData.baseFiring;
	managerRuntimeData.avgFiring=NULL; managerRuntimeData.baseFiring=NULL;

	if (managerRuntimeData.lastSpikeTime!=NULL) delete[] managerRuntimeData.lastSpikeTime;
	if (managerRuntimeData.synSpikeTime !=NULL) delete[] managerRuntimeData.synSpikeTime;
	if (managerRuntimeData.nSpikeCnt!=NULL) delete[] managerRuntimeData.nSpikeCnt;
	managerRuntimeData.lastSpikeTime=NULL; managerRuntimeData.synSpikeTime=NULL; managerRuntimeData.nSpikeCnt=NULL;

	if (managerRuntimeData.postDelayInfo!=NULL) delete[] managerRuntimeData.postDelayInfo;
	if (managerRuntimeData.preSynapticIds!=NULL) delete[] managerRuntimeData.preSynapticIds;
	if (managerRuntimeData.postSynapticIds!=NULL) delete[] managerRuntimeData.postSynapticIds;
	managerRuntimeData.postDelayInfo=NULL; managerRuntimeData.preSynapticIds=NULL; managerRuntimeData.postSynapticIds=NULL;

	if (managerRuntimeData.wt!=NULL) delete[] managerRuntimeData.wt;
	if (managerRuntimeData.maxSynWt!=NULL) delete[] managerRuntimeData.maxSynWt;
	if (managerRuntimeData.wtChange !=NULL) delete[] managerRuntimeData.wtChange;
	managerRuntimeData.wt=NULL; managerRuntimeData.maxSynWt=NULL; managerRuntimeData.wtChange=NULL;

	if (mulSynFast!=NULL) delete[] mulSynFast;
	if (mulSynSlow!=NULL) delete[] mulSynSlow;
	if (managerRuntimeData.connIdsPreIdx!=NULL) delete[] managerRuntimeData.connIdsPreIdx;
	mulSynFast=NULL; mulSynSlow=NULL; managerRuntimeData.connIdsPreIdx=NULL;

	if (managerRuntimeData.grpIds!=NULL) delete[] managerRuntimeData.grpIds;
	managerRuntimeData.grpIds=NULL;

	if (managerRuntimeData.timeTableD2 != NULL) delete [] managerRuntimeData.timeTableD2;
	if (managerRuntimeData.timeTableD1 != NULL) delete [] managerRuntimeData.timeTableD1;
	managerRuntimeData.timeTableD2 = NULL; managerRuntimeData.timeTableD1 = NULL;
	
	if (managerRuntimeData.firingTableD2!=NULL) delete[] managerRuntimeData.firingTableD2;
	if (managerRuntimeData.firingTableD1!=NULL) delete[] managerRuntimeData.firingTableD1;
	//if (managerRuntimeData.firingTableD2!=NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.firingTableD2));
	//if (managerRuntimeData.firingTableD1!=NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.firingTableD1));
	managerRuntimeData.firingTableD2 = NULL; managerRuntimeData.firingTableD1 = NULL;

	if (managerRuntimeData.extFiringTableD2!=NULL) delete[] managerRuntimeData.extFiringTableD2;
	if (managerRuntimeData.extFiringTableD1!=NULL) delete[] managerRuntimeData.extFiringTableD1;
	//if (managerRuntimeData.extFiringTableD2!=NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.extFiringTableD2));
	//if (managerRuntimeData.extFiringTableD1!=NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.extFiringTableD1));
	managerRuntimeData.extFiringTableD2 = NULL; managerRuntimeData.extFiringTableD1 = NULL;

	if (managerRuntimeData.extFiringTableEndIdxD1 != NULL) delete[] managerRuntimeData.extFiringTableEndIdxD1;
	if (managerRuntimeData.extFiringTableEndIdxD2 != NULL) delete[] managerRuntimeData.extFiringTableEndIdxD2;
	//if (managerRuntimeData.extFiringTableEndIdxD1 != NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.extFiringTableEndIdxD1));
	//if (managerRuntimeData.extFiringTableEndIdxD2 != NULL) CUDA_CHECK_ERRORS(cudaFreeHost(managerRuntimeData.extFiringTableEndIdxD2));
	managerRuntimeData.extFiringTableEndIdxD1 = NULL; managerRuntimeData.extFiringTableEndIdxD2 = NULL;
}

/*!
 * \brief reset poisson neurons using local ids
 */
void SNN::resetPoissonNeuron(int netId, int lGrpId, int lNId) {
	assert(lNId < networkConfigs[netId].numN);
	managerRuntimeData.lastSpikeTime[lNId] = MAX_SIMULATION_TIME;
	if (groupConfigs[netId][lGrpId].WithHomeostasis)
		managerRuntimeData.avgFiring[lNId] = 0.0f;

	if (groupConfigs[netId][lGrpId].WithSTP) {
		for (int j = 0; j < networkConfigs[netId].maxDelay + 1; j++) { // is of size maxDelay_+1
			int index = STP_BUF_POS(lNId, j, networkConfigs[netId].maxDelay);
			managerRuntimeData.stpu[index] = 0.0f;
			managerRuntimeData.stpx[index] = 1.0f;
		}
	}
}

void SNN::resetPropogationBuffer() {
	// FIXME: why 1023?
	spikeBuf->reset(0, 1023);
}

//Reset wt, wtChange, pre-firing time values to default values, rewritten to
//integrate changes between JMN and MDR -- KDC
//if changeWeights is false, we should keep the values of the weights as they currently
//are but we should be able to change them to plastic or fixed synapses. -- KDC
// FIXME: imlement option of resetting weights
void SNN::resetSynapse(int netId, bool changeWeights) {
	memset(managerRuntimeData.wtChange, 0, sizeof(float) * networkConfigs[netId].numPreSynNet); // reset the synaptic derivatives

	for (int syn = 0; syn < networkConfigs[netId].numPreSynNet; syn++)
		managerRuntimeData.synSpikeTime[syn] = MAX_SIMULATION_TIME; // reset the spike time of each syanpse
}

void SNN::resetTimeTable() {
	memset(managerRuntimeData.timeTableD2, 0, sizeof(int) * (1000 + glbNetworkConfig.maxDelay + 1));
	memset(managerRuntimeData.timeTableD1, 0, sizeof(int) * (1000 + glbNetworkConfig.maxDelay + 1));
}

void SNN::resetFiringTable() {
	memset(managerRuntimeData.firingTableD2, 0, sizeof(int) * managerRTDSize.maxMaxSpikeD2);
	memset(managerRuntimeData.firingTableD1, 0, sizeof(int) * managerRTDSize.maxMaxSpikeD1);
	memset(managerRuntimeData.extFiringTableEndIdxD2, 0, sizeof(int) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.extFiringTableEndIdxD1, 0, sizeof(int) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.extFiringTableD2, 0, sizeof(int*) * managerRTDSize.maxNumGroups);
	memset(managerRuntimeData.extFiringTableD1, 0, sizeof(int*) * managerRTDSize.maxNumGroups);
}

void SNN::resetSpikeCnt(int gGrpId) {
	assert(gGrpId >= ALL);

	if (gGrpId == ALL) {
		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			pthread_t threads[numCores + 1]; // 1 additional array size if numCores == 0, it may work though bad practice
			cpu_set_t cpus;	
			ThreadStruct argsThreadRoutine[numCores + 1]; // same as above, +1 array size
			int threadCount = 0;
		#endif
		
		for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
			if (!groupPartitionLists[netId].empty()) {
				if (netId < CPU_RUNTIME_BASE) // GPU runtime
					resetSpikeCnt_GPU(netId, ALL);
				else{ // CPU runtime
					#if defined(WIN32) || defined(WIN64) || defined(__APPLE__)
						resetSpikeCnt_CPU(netId, ALL);
					#else // Linux or MAC
						pthread_attr_t attr;
						pthread_attr_init(&attr);
						CPU_ZERO(&cpus);
						CPU_SET(threadCount%NUM_CPU_CORES, &cpus);
						pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

						argsThreadRoutine[threadCount].snn_pointer = this;
						argsThreadRoutine[threadCount].netId = netId;
						argsThreadRoutine[threadCount].lGrpId = ALL;
						argsThreadRoutine[threadCount].startIdx = 0;
						argsThreadRoutine[threadCount].endIdx = 0;
						argsThreadRoutine[threadCount].GtoLOffset = 0;

						pthread_create(&threads[threadCount], &attr, &SNN::helperResetSpikeCnt_CPU, (void*)&argsThreadRoutine[threadCount]);
						pthread_attr_destroy(&attr);
						threadCount++;
					#endif
				}
			}
		}

		#if !defined(WIN32) && !defined(WIN64) && !defined(__APPLE__) // Linux or MAC
			// join all the threads
			for (int i=0; i<threadCount; i++){
				pthread_join(threads[i], NULL);
			}
		#endif
	} 
	else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;

		if (netId < CPU_RUNTIME_BASE) // GPU runtime
			resetSpikeCnt_GPU(netId, lGrpId);
		else // CPU runtime
			resetSpikeCnt_CPU(netId, lGrpId);
	}
}


//! nid=neuron id, sid=synapse id, grpId=group id.
inline SynInfo SNN::SET_CONN_ID(int nId, int sId, int grpId) {
	if (grpId > GROUP_ID_MASK) {
		KERNEL_ERROR("Error: Group Id (%d) exceeds maximum limit (%d)", grpId, GROUP_ID_MASK);
		exitSimulation(ID_OVERFLOW_ERROR);
	}

	SynInfo synInfo;
	//p.postId = (((sid)<<CONN_SYN_NEURON_BITS)+((nid)&CONN_SYN_NEURON_MASK));
	//p.grpId  = grpId;
	synInfo.gsId = ((grpId << NUM_SYNAPSE_BITS) | sId);
	synInfo.nId = nId;

	return synInfo;
}


void SNN::setGrpTimeSlice(int gGrpId, int timeSlice) {
	if (gGrpId == ALL) {
		for(int grpId = 0; grpId < numGroups; grpId++) {
			if (groupConfigMap[grpId].isSpikeGenerator)
				setGrpTimeSlice(grpId, timeSlice);
		}
	} else {
		assert((timeSlice > 0 ) && (timeSlice <= MAX_TIME_SLICE));
		// the group should be poisson spike generator group
		groupConfigMDMap[gGrpId].currTimeSlice = timeSlice;
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

void SNN::fillSpikeGenBits(int netId) {
	SpikeBuffer::SpikeIterator spikeBufIter;
	SpikeBuffer::SpikeIterator spikeBufIterEnd = spikeBuf->back();

	// Covert spikes stored in spikeBuffer to SpikeGenBit
	for (spikeBufIter = spikeBuf->front(); spikeBufIter != spikeBufIterEnd; ++spikeBufIter) {
		// get the global neuron id and group id for this particular spike
		int gGrpId = spikeBufIter->grpId;

		if (groupConfigMDMap[gGrpId].netId == netId) {
			int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
			int lNId = spikeBufIter->neurId /* gNId */ + groupConfigMDMap[gGrpId].GtoLOffset;

			// add spike to spikeGentBit
			assert(groupConfigMap[gGrpId].isSpikeGenerator == true);

			int nIdPos = (lNId - groupConfigs[netId][lGrpId].lStartN + groupConfigs[netId][lGrpId].Noffset);
			int nIdBitPos = nIdPos % 32;
			int nIdIndex = nIdPos / 32;

			assert(nIdIndex < (networkConfigs[netId].numNSpikeGen / 32 + 1));

			managerRuntimeData.spikeGenBits[nIdIndex] |= (1 << nIdBitPos);
		}
	}
}

void SNN::startTiming() { prevExecutionTime = cumExecutionTime; }
void SNN::stopTiming() {
	executionTime += (cumExecutionTime - prevExecutionTime);
	prevExecutionTime = cumExecutionTime;
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

			updateWeights();

			stdpScaleFactor_ = storeScaleSTDP;
		}
	}

	sim_in_testing = true;

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			networkConfigs[netId].sim_in_testing = true;
			updateNetworkConfig(netId); // update networkConfigRT struct (|TODO copy only a single boolean)
		}
	}
}

// exits testing phase
void SNN::stopTesting() {
	sim_in_testing = false;

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			networkConfigs[netId].sim_in_testing = false;
			updateNetworkConfig(netId); // update networkConfigRT struct (|TODO copy only a single boolean)
		}
	}
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

// FIXME: modify this for multi-GPUs
std::vector< std::vector<float> > SNN::getWeightMatrix2D(short int connId) {
	assert(connId > ALL); // ALL == -1
	std::vector< std::vector<float> > wtConnId;

	int grpIdPre = connectConfigMap[connId].grpSrc;
	int grpIdPost = connectConfigMap[connId].grpDest;

	int netIdPost = groupConfigMDMap[grpIdPost].netId;
	int lGrpIdPost = groupConfigMDMap[grpIdPost].lGrpId;

	// init weight matrix with right dimensions
	for (int i = 0; i < groupConfigMap[grpIdPre].numN; i++) {
		std::vector<float> wtSlice;
		for (int j = 0; j < groupConfigMap[grpIdPost].numN; j++) {
			wtSlice.push_back(NAN);
		}
		wtConnId.push_back(wtSlice);
	}

	// copy the weights for a given post-group from device
	// \TODO: check if the weights for this grpIdPost have already been copied
	// \TODO: even better, but tricky because of ordering, make copyWeightState connection-based

	assert(grpIdPost > ALL); // ALL == -1

	// Note, copyWeightState() also copies pre-connections information (e.g., Npre, Npre_plastic, cumulativePre, and preSynapticIds)
	fetchWeightState(netIdPost, lGrpIdPost);
	fetchConnIdsLookupArray(netIdPost);

	for (int lNIdPost = groupConfigs[netIdPost][lGrpIdPost].lStartN; lNIdPost <= groupConfigs[netIdPost][lGrpIdPost].lEndN; lNIdPost++) {
		unsigned int pos_ij = managerRuntimeData.cumulativePre[lNIdPost];
		for (int i = 0; i < managerRuntimeData.Npre[lNIdPost]; i++, pos_ij++) {
			// skip synapses that belong to a different connection ID
			if (managerRuntimeData.connIdsPreIdx[pos_ij] != connId) //connInfo->connId)
				continue;

			// find pre-neuron ID and update ConnectionMonitor container
			int lNIdPre = GET_CONN_NEURON_ID(managerRuntimeData.preSynapticIds[pos_ij]);
			int lGrpIdPre = GET_CONN_GRP_ID(managerRuntimeData.preSynapticIds[pos_ij]);
			wtConnId[lNIdPre - groupConfigs[netIdPost][lGrpIdPre].lStartN][lNIdPost - groupConfigs[netIdPost][lGrpIdPost].lStartN] =
				fabs(managerRuntimeData.wt[pos_ij]);
		}
	}

	return wtConnId;
}

void SNN::updateGroupMonitor(int gGrpId) {
	// don't continue if no group monitors in the network
	if (!numGroupMonitor)
		return;

	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++)
			updateGroupMonitor(gGrpId);
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		// update group monitor of a specific group
		// find index in group monitor arrays
		int monitorId = groupConfigMDMap[gGrpId].groupMonitorId;

		// don't continue if no group monitor enabled for this group
		if (monitorId < 0) return;

		// find last update time for this group
		GroupMonitorCore* grpMonObj = groupMonCoreList[monitorId];
		int lastUpdate = grpMonObj->getLastUpdated();

		// don't continue if time interval is zero (nothing to update)
		if (getSimTime() - lastUpdate <= 0)
			return;

		if (getSimTime() - lastUpdate > 1000)
			KERNEL_ERROR("updateGroupMonitor(grpId=%d) must be called at least once every second", gGrpId);

		// copy the group status (neuromodulators) to the manager runtime
		fetchGroupState(netId, lGrpId);

		// find the time interval in which to update group status
		// usually, we call updateGroupMonitor once every second, so the time interval is [0,1000)
		// however, updateGroupMonitor can be called at any time t \in [0,1000)... so we can have the cases
		// [0,t), [t,1000), and even [t1, t2)
		int numMsMin = lastUpdate % 1000; // lower bound is given by last time we called update
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
			data = managerRuntimeData.grpDABuffer[lGrpId * 1000 + t];

			// current time is last completed second plus whatever is leftover in t
			int time = currentTimeSec * 1000 + t;

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

// FIXME: wrong to use groupConfigs[0]
void SNN::userDefinedSpikeGenerator(int gGrpId) {
	// \FIXME this function is a mess
	SpikeGeneratorCore* spikeGenFunc = groupConfigMap[gGrpId].spikeGenFunc;
	int netId = groupConfigMDMap[gGrpId].netId;
	int timeSlice = groupConfigMDMap[gGrpId].currTimeSlice;
	int currTime = simTime;
	bool done;

	fetchLastSpikeTime(netId);

	for(int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
		// start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
		int lNId = gNId + groupConfigMDMap[gGrpId].GtoLOffset;
		int nextTime = managerRuntimeData.lastSpikeTime[lNId];
		if (nextTime == MAX_SIMULATION_TIME)
			nextTime = 0;

		// the end of the valid time window is either the length of the scheduling time slice from now (because that
		// is the max of the allowed propagated buffer size) or simply the end of the simulation
		int endOfTimeWindow = std::min(currTime+timeSlice, simTimeRunStop);

		done = false;
		while (!done) {
			// generate the next spike time (nextSchedTime) from the nextSpikeTime callback
			int nextSchedTime = spikeGenFunc->nextSpikeTime(this, gGrpId, gNId - groupConfigMDMap[gGrpId].gStartN, currTime, nextTime, endOfTimeWindow);

			// the generated spike time is valid only if:
			// - it has not been scheduled before (nextSchedTime > nextTime)
			//    - but careful: we would drop spikes at t=0, because we cannot initialize nextTime to -1...
			// - it is within the scheduling time slice (nextSchedTime < endOfTimeWindow)
			// - it is not in the past (nextSchedTime >= currTime)
			if ((nextSchedTime==0 || nextSchedTime>nextTime) && nextSchedTime<endOfTimeWindow && nextSchedTime>=currTime) {
//				fprintf(stderr,"%u: spike scheduled for %d at %u\n",currTime, i-groupConfigs[0][grpId].StartN,nextSchedTime);
				// scheduled spike...
				// \TODO CPU mode does not check whether the same AER event has been scheduled before (bug #212)
				// check how GPU mode does it, then do the same here.
				nextTime = nextSchedTime;
				spikeBuf->schedule(gNId, gGrpId, nextTime - currTime);
			} else {
				done = true;
			}
		}
	}
}

void SNN::generateUserDefinedSpikes() {
	for(int gGrpId = 0; gGrpId < numGroups; gGrpId++) {
		if (groupConfigMap[gGrpId].isSpikeGenerator) {
			// This evaluation is done to check if its time to get new set of spikes..
			// check whether simTime has advance more than the current time slice, in which case we need to schedule
			// spikes for the next time slice
			// we always have to run this the first millisecond of a new runNetwork call; that is,
			// when simTime==simTimeRunStart
			if(((simTime - groupConfigMDMap[gGrpId].sliceUpdateTime) >= groupConfigMDMap[gGrpId].currTimeSlice || simTime == simTimeRunStart)) {
				int timeSlice = groupConfigMDMap[gGrpId].currTimeSlice;
				groupConfigMDMap[gGrpId].sliceUpdateTime = simTime;
				
				// we dont generate any poisson spike if during the
				// current call we might exceed the maximum 32 bit integer value
				if ((simTime + timeSlice) == MAX_SIMULATION_TIME || (simTime + timeSlice) < 0)
					return;

				if (groupConfigMap[gGrpId].spikeGenFunc != NULL) {
					userDefinedSpikeGenerator(gGrpId);
				}
			}
		}
	}
}

/*!
 * \brief Allocate and reset SNN::maxSpikesD1, SNN::maxSpikesD2 and allocate sapce for SNN::firingTableD1 and SNN::firingTableD2
 *
 * \note SpikeTables include firingTableD1(D2) and timeTableD1(D2)
 */
void SNN::allocateManagerSpikeTables() {
	managerRuntimeData.firingTableD2 = new int[managerRTDSize.maxMaxSpikeD2];
	managerRuntimeData.firingTableD1 = new int[managerRTDSize.maxMaxSpikeD1];
	managerRuntimeData.extFiringTableEndIdxD2 = new int[managerRTDSize.maxNumGroups];
	managerRuntimeData.extFiringTableEndIdxD1 = new int[managerRTDSize.maxNumGroups];
	managerRuntimeData.extFiringTableD2 = new int*[managerRTDSize.maxNumGroups];
	managerRuntimeData.extFiringTableD1 = new int*[managerRTDSize.maxNumGroups];

	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.firingTableD2, sizeof(int) * managerRTDSize.maxMaxSpikeD2));
	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.firingTableD1, sizeof(int) * managerRTDSize.maxMaxSpikeD1));
	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.extFiringTableEndIdxD2, sizeof(int) * managerRTDSize.maxNumGroups));
	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.extFiringTableEndIdxD1, sizeof(int) * managerRTDSize.maxNumGroups));
	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.extFiringTableD2, sizeof(int*) * managerRTDSize.maxNumGroups));
	//CUDA_CHECK_ERRORS(cudaMallocHost(&managerRuntimeData.extFiringTableD1, sizeof(int*) * managerRTDSize.maxNumGroups));
	resetFiringTable();
	
	managerRuntimeData.timeTableD2 = new unsigned int[TIMING_COUNT];
	managerRuntimeData.timeTableD1 = new unsigned int[TIMING_COUNT];
	resetTimeTable();
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
	if(simTime == MAX_SIMULATION_TIME || simTime < 0){
        // reached the maximum limit of the simulation time using 32 bit value...
        KERNEL_WARN("Maximum Simulation Time Reached...Resetting simulation time");
	}

	return finishedOneSec;
}

// FIXME: modify this for multi-GPUs
void SNN::updateSpikeMonitor(int gGrpId) {
	// don't continue if no spike monitors in the network
	if (!numSpikeMonitor)
		return;

	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++)
			updateSpikeMonitor(gGrpId);
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		// update spike monitor of a specific group
		// find index in spike monitor arrays
		int monitorId = groupConfigMDMap[gGrpId].spikeMonitorId;

		// don't continue if no spike monitor enabled for this group
		if (monitorId < 0) return;

		// find last update time for this group
		SpikeMonitorCore* spkMonObj = spikeMonCoreList[monitorId];
		long int lastUpdate = spkMonObj->getLastUpdated();

		// don't continue if time interval is zero (nothing to update)
		if ( ((long int)getSimTime()) - lastUpdate <= 0)
			return;

		if ( ((long int)getSimTime()) - lastUpdate > 1000)
			KERNEL_ERROR("updateSpikeMonitor(grpId=%d) must be called at least once every second",gGrpId);

        // AER buffer max size warning here.
        // Because of C++ short-circuit evaluation, the last condition should not be evaluated
        // if the previous conditions are false.
        if (spkMonObj->getAccumTime() > LONG_SPIKE_MON_DURATION \
                && this->getGroupNumNeurons(gGrpId) > LARGE_SPIKE_MON_GRP_SIZE \
                && spkMonObj->isBufferBig()){
            // change this warning message to correct message
            KERNEL_WARN("updateSpikeMonitor(grpId=%d) is becoming very large. (>%lu MB)",gGrpId,(long int) MAX_SPIKE_MON_BUFFER_SIZE/1024 );// make this better
            KERNEL_WARN("Reduce the cumulative recording time (currently %lu minutes) or the group size (currently %d) to avoid this.",spkMonObj->getAccumTime()/(1000*60),this->getGroupNumNeurons(gGrpId));
		}

		// copy the neuron firing information to the manager runtime
		fetchSpikeTables(netId);
		fetchGrpIdsLookupArray(netId);

		// find the time interval in which to update spikes
		// usually, we call updateSpikeMonitor once every second, so the time interval is [0,1000)
		// however, updateSpikeMonitor can be called at any time t \in [0,1000)... so we can have the cases
		// [0,t), [t,1000), and even [t1, t2)
		int numMsMin = lastUpdate % 1000; // lower bound is given by last time we called update
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
		spkMonObj->setLastUpdated( (long int)getSimTime() );

		// prepare fast access
		FILE* spkFileId = spikeMonCoreList[monitorId]->getSpikeFileId();
		bool writeSpikesToFile = spkFileId != NULL;
		bool writeSpikesToArray = spkMonObj->getMode()==AER && spkMonObj->isRecording();

		// Read one spike at a time from the buffer and put the spikes to an appopriate monitor buffer. Later the user
		// may need need to dump these spikes to an output file
		for (int k = 0; k < 2; k++) {
			unsigned int* timeTablePtr = (k == 0) ? managerRuntimeData.timeTableD2 : managerRuntimeData.timeTableD1;
			int* fireTablePtr = (k == 0) ? managerRuntimeData.firingTableD2 : managerRuntimeData.firingTableD1;
			for(int t = numMsMin; t < numMsMax; t++) {
				for(int i = timeTablePtr[t + glbNetworkConfig.maxDelay]; i < timeTablePtr[t + glbNetworkConfig.maxDelay + 1]; i++) {
					// retrieve the neuron id
					int lNId = fireTablePtr[i];

					// make sure neuron belongs to currently relevant group
					int this_grpId = managerRuntimeData.grpIds[lNId];
					if (this_grpId != lGrpId)
						continue;

					// adjust nid to be 0-indexed for each group
					// this way, if a group has 10 neurons, their IDs in the spike file and spike monitor will be
					// indexed from 0..9, no matter what their real nid is
					int nId = lNId - groupConfigs[netId][lGrpId].lStartN;
					assert(nId >= 0);

					// current time is last completed second plus whatever is leftover in t
					int time = currentTimeSec * 1000 + t;

					if (writeSpikesToFile) {
						int cnt;
						cnt = fwrite(&time, sizeof(int), 1, spkFileId); assert(cnt==1);
						cnt = fwrite(&nId,  sizeof(int), 1, spkFileId); assert(cnt==1);
					}

					if (writeSpikesToArray) {
						spkMonObj->pushAER(time, nId);
					}
				}
			}
		}

		if (spkFileId!=NULL) // flush spike file
			fflush(spkFileId);
	}
}

// FIXME: modify this for multi-GPUs
void SNN::updateNeuronMonitor(int gGrpId) {
	// don't continue if no neuron monitors in the network
	if (!numNeuronMonitor)
		return;

	//printf("The global group id is: %i\n", gGrpId);

	if (gGrpId == ALL) {
		for (int gGrpId = 0; gGrpId < numGroups; gGrpId++)
			updateNeuronMonitor(gGrpId);
	}
	else {
		//printf("UpdateNeuronMonitor is being executed!\n");
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		// update spike monitor of a specific group
		// find index in spike monitor arrays
		int monitorId = groupConfigMDMap[gGrpId].neuronMonitorId;

		// don't continue if no spike monitor enabled for this group
		if (monitorId < 0) return;

		// find last update time for this group
		NeuronMonitorCore* nrnMonObj = neuronMonCoreList[monitorId];
		long int lastUpdate = nrnMonObj->getLastUpdated();

		// don't continue if time interval is zero (nothing to update)
		if (((long int)getSimTime()) - lastUpdate <= 0)
			return;

		if (((long int)getSimTime()) - lastUpdate > 1000)
			KERNEL_ERROR("updateNeuronMonitor(grpId=%d) must be called at least once every second", gGrpId);

		// AER buffer max size warning here.
		// Because of C++ short-circuit evaluation, the last condition should not be evaluated
		// if the previous conditions are false.
		
		/*if (nrnMonObj->getAccumTime() > LONG_NEURON_MON_DURATION \
			&& this->getGroupNumNeurons(gGrpId) > LARGE_NEURON_MON_GRP_SIZE \
			&& nrnMonObj->isBufferBig()) {
			// change this warning message to correct message
			KERNEL_WARN("updateNeuronMonitor(grpId=%d) is becoming very large. (>%lu MB)", gGrpId, (long int)MAX_NEURON_MON_BUFFER_SIZE / 1024);// make this better
			KERNEL_WARN("Reduce the cumulative recording time (currently %lu minutes) or the group size (currently %d) to avoid this.", nrnMonObj->getAccumTime() / (1000 * 60), this->getGroupNumNeurons(gGrpId));
		}*/

		// copy the neuron information to manager runtime
		fetchNeuronStateBuffer(netId, lGrpId);
		
		// find the time interval in which to update neuron state info
		// usually, we call updateNeuronMonitor once every second, so the time interval is [0,1000)
		// however, updateNeuronMonitor can be called at any time t \in [0,1000)... so we can have the cases
		// [0,t), [t,1000), and even [t1, t2)
		int numMsMin = lastUpdate % 1000; // lower bound is given by last time we called update
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
		nrnMonObj->setLastUpdated((long int)getSimTime());

		// prepare fast access
		FILE* nrnFileId = neuronMonCoreList[monitorId]->getNeuronFileId();
		bool writeNeuronStateToFile = nrnFileId != NULL;
		bool writeNeuronStateToArray = nrnMonObj->isRecording();

		// Read one neuron state value at a time from the buffer and put the neuron state values to an appopriate monitor buffer.
		// Later the user may need need to dump these neuron state values to an output file
		//printf("The numMsMin is: %i; and numMsMax is: %i\n", numMsMin, numMsMax);
		for (int t = numMsMin; t < numMsMax; t++) {
			//printf("The lStartN is: %i; and lEndN is: %i\n", groupConfigs[netId][lGrpId].lStartN, groupConfigs[netId][lGrpId].lEndN);
			for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++) { 
				float v, u, I;

				// make sure neuron belongs to currently relevant group
				int this_grpId = managerRuntimeData.grpIds[lNId];
				if (this_grpId != lGrpId)
					continue;

				// adjust nid to be 0-indexed for each group
				// this way, if a group has 10 neurons, their IDs in the spike file and spike monitor will be
				// indexed from 0..9, no matter what their real nid is
				int nId = lNId - groupConfigs[netId][lGrpId].lStartN;
				assert(nId >= 0);

				int idxBase = networkConfigs[netId].numGroups * MAX_NEURON_MON_GRP_SZIE * t + lGrpId * MAX_NEURON_MON_GRP_SZIE;
				v = managerRuntimeData.nVBuffer[idxBase + nId];
				u = managerRuntimeData.nUBuffer[idxBase + nId];
				I = managerRuntimeData.nIBuffer[idxBase + nId];

				//printf("Voltage recorded is: %f\n", v);

				// current time is last completed second plus whatever is leftover in t
				int time = currentTimeSec * 1000 + t;

				// WRITE TO A TEXT FILE INSTEAD OF BINARY
				if (writeNeuronStateToFile) {
					int cnt;
					cnt = fwrite(&nId, sizeof(int), 1, nrnFileId); assert(cnt == 1);
					cnt = fwrite(&time, sizeof(int), 1, nrnFileId); assert(cnt == 1);
					cnt = fwrite(&v, sizeof(float), 1, nrnFileId); assert(cnt == 1);
					cnt = fwrite(&u, sizeof(float), 1, nrnFileId); assert(cnt == 1);
					cnt = fwrite(&I, sizeof(float), 1, nrnFileId); assert(cnt == 1);
				}

				if (writeNeuronStateToArray) {
					nrnMonObj->pushNeuronState(nId, v, u, I);
				}
			}
		}

		if (nrnFileId != NULL) // flush spike file
			fflush(nrnFileId);
	}
}

// FIXME: update summary format for multiGPUs
void SNN::printSimSummary() {
	float etime;

	// FIXME: measure total execution time, and GPU excution time
	stopTiming();
	etime = executionTime;

	fetchNetworkSpikeCount();

	KERNEL_INFO("\n");
	KERNEL_INFO("********************    Simulation Summary      ***************************");

	KERNEL_INFO("Network Parameters: \tnumNeurons = %d (numNExcReg:numNInhReg = %2.1f:%2.1f)", 
		glbNetworkConfig.numN, 100.0 * glbNetworkConfig.numNExcReg / glbNetworkConfig.numN, 100.0 * glbNetworkConfig.numNInhReg / glbNetworkConfig.numN);
	KERNEL_INFO("\t\t\tnumSynapses = %d", glbNetworkConfig.numSynNet);
	KERNEL_INFO("\t\t\tmaxDelay = %d", glbNetworkConfig.maxDelay);
	KERNEL_INFO("Simulation Mode:\t%s",sim_with_conductances?"COBA":"CUBA");
	KERNEL_INFO("Random Seed:\t\t%d", randSeed_);
	KERNEL_INFO("Timing:\t\t\tModel Simulation Time = %lld sec", (unsigned long long)simTimeSec);
	KERNEL_INFO("\t\t\tActual Execution Time = %4.2f sec", etime/1000.0f);
	KERNEL_INFO("Average Firing Rate:\t2+ms delay = %3.3f Hz",
		glbNetworkConfig.numN2msDelay > 0 ? managerRuntimeData.spikeCountD2 / (1.0 * simTimeSec * glbNetworkConfig.numN2msDelay) : 0.0f);
	KERNEL_INFO("\t\t\t1ms delay = %3.3f Hz",
		glbNetworkConfig.numN1msDelay > 0 ? managerRuntimeData.spikeCountD1 / (1.0 * simTimeSec * glbNetworkConfig.numN1msDelay) : 0.0f);
	KERNEL_INFO("\t\t\tOverall = %3.3f Hz", managerRuntimeData.spikeCount / (1.0 * simTimeSec * glbNetworkConfig.numN));
	KERNEL_INFO("Overall Spike Count Transferred:");
	KERNEL_INFO("\t\t\t2+ms delay = %d", managerRuntimeData.spikeCountExtRxD2);
	KERNEL_INFO("\t\t\t1ms delay = %d", managerRuntimeData.spikeCountExtRxD1);
	KERNEL_INFO("Overall Spike Count:\t2+ms delay = %d", managerRuntimeData.spikeCountD2);
	KERNEL_INFO("\t\t\t1ms delay = %d", managerRuntimeData.spikeCountD1);
	KERNEL_INFO("\t\t\tTotal = %d", managerRuntimeData.spikeCount);
	KERNEL_INFO("*********************************************************************************\n");
}

//------------------------------ legacy code --------------------------------//

// We parallelly cleanup the postSynapticIds array to minimize any other wastage in that array by compacting the store
// Appropriate alignment specified by ALIGN_COMPACTION macro is used to ensure some level of alignment (if necessary)
//void SNN::compactConnections() {
//	unsigned int* tmp_cumulativePost = new unsigned int[numN];
//	unsigned int* tmp_cumulativePre  = new unsigned int[numN];
//	unsigned int lastCnt_pre         = 0;
//	unsigned int lastCnt_post        = 0;
//
//	tmp_cumulativePost[0]   = 0;
//	tmp_cumulativePre[0]    = 0;
//
//	for(int i=1; i < numN; i++) {
//		lastCnt_post = tmp_cumulativePost[i-1]+managerRuntimeData.Npost[i-1]; //position of last pointer
//		lastCnt_pre  = tmp_cumulativePre[i-1]+managerRuntimeData.Npre[i-1]; //position of last pointer
//		#if COMPACTION_ALIGNMENT_POST
//			lastCnt_post= lastCnt_post + COMPACTION_ALIGNMENT_POST-lastCnt_post%COMPACTION_ALIGNMENT_POST;
//			lastCnt_pre = lastCnt_pre  + COMPACTION_ALIGNMENT_PRE- lastCnt_pre%COMPACTION_ALIGNMENT_PRE;
//		#endif
//		tmp_cumulativePost[i] = lastCnt_post;
//		tmp_cumulativePre[i]  = lastCnt_pre;
//		assert(tmp_cumulativePost[i] <= managerRuntimeData.cumulativePost[i]);
//		assert(tmp_cumulativePre[i]  <= managerRuntimeData.cumulativePre[i]);
//	}
//
//	// compress the post_synaptic array according to the new values of the tmp_cumulative counts....
//	unsigned int tmp_numPostSynNet = tmp_cumulativePost[numN-1]+managerRuntimeData.Npost[numN-1];
//	unsigned int tmp_numPreSynNet  = tmp_cumulativePre[numN-1]+managerRuntimeData.Npre[numN-1];
//	assert(tmp_numPostSynNet <= allocatedPost);
//	assert(tmp_numPreSynNet  <= allocatedPre);
//	assert(tmp_numPostSynNet <= numPostSynNet);
//	assert(tmp_numPreSynNet  <= numPreSynNet);
//	KERNEL_DEBUG("******************");
//	KERNEL_DEBUG("CompactConnection: ");
//	KERNEL_DEBUG("******************");
//	KERNEL_DEBUG("old_postCnt = %d, new_postCnt = %d", numPostSynNet, tmp_numPostSynNet);
//	KERNEL_DEBUG("old_preCnt = %d,  new_postCnt = %d", numPreSynNet,  tmp_numPreSynNet);
//
//	// new buffer with required size + 100 bytes of additional space just to provide limited overflow
//	SynInfo* tmp_postSynapticIds   = new SynInfo[tmp_numPostSynNet+100];
//
//	// new buffer with required size + 100 bytes of additional space just to provide limited overflow
//	SynInfo* tmp_preSynapticIds	= new SynInfo[tmp_numPreSynNet+100];
//	float* tmp_wt	    	  		= new float[tmp_numPreSynNet+100];
//	float* tmp_maxSynWt   	  		= new float[tmp_numPreSynNet+100];
//	short int *tmp_cumConnIdPre 		= new short int[tmp_numPreSynNet+100];
//	float *tmp_mulSynFast 			= new float[numConnections];
//	float *tmp_mulSynSlow  			= new float[numConnections];
//
//	// compact synaptic information
//	for(int i=0; i<numN; i++) {
//		assert(tmp_cumulativePost[i] <= managerRuntimeData.cumulativePost[i]);
//		assert(tmp_cumulativePre[i]  <= managerRuntimeData.cumulativePre[i]);
//		for( int j=0; j<managerRuntimeData.Npost[i]; j++) {
//			unsigned int tmpPos = tmp_cumulativePost[i]+j;
//			unsigned int oldPos = managerRuntimeData.cumulativePost[i]+j;
//			tmp_postSynapticIds[tmpPos] = managerRuntimeData.postSynapticIds[oldPos];
//			tmp_SynapticDelay[tmpPos]   = tmp_SynapticDelay[oldPos];
//		}
//		for( int j=0; j<managerRuntimeData.Npre[i]; j++) {
//			unsigned int tmpPos =  tmp_cumulativePre[i]+j;
//			unsigned int oldPos =  managerRuntimeData.cumulativePre[i]+j;
//			tmp_preSynapticIds[tmpPos]  = managerRuntimeData.preSynapticIds[oldPos];
//			tmp_maxSynWt[tmpPos] 	    = managerRuntimeData.maxSynWt[oldPos];
//			tmp_wt[tmpPos]              = managerRuntimeData.wt[oldPos];
//			tmp_cumConnIdPre[tmpPos]	= managerRuntimeData.connIdsPreIdx[oldPos];
//		}
//	}
//
//	// delete old buffer space
//	delete[] managerRuntimeData.postSynapticIds;
//	managerRuntimeData.postSynapticIds = tmp_postSynapticIds;
//	cpuSnnSz.networkInfoSize -= (sizeof(SynInfo)*numPostSynNet);
//	cpuSnnSz.networkInfoSize += (sizeof(SynInfo)*(tmp_numPostSynNet+100));
//
//	delete[] managerRuntimeData.cumulativePost;
//	managerRuntimeData.cumulativePost  = tmp_cumulativePost;
//
//	delete[] managerRuntimeData.cumulativePre;
//	managerRuntimeData.cumulativePre   = tmp_cumulativePre;
//
//	delete[] managerRuntimeData.maxSynWt;
//	managerRuntimeData.maxSynWt = tmp_maxSynWt;
//	cpuSnnSz.synapticInfoSize -= (sizeof(float)*numPreSynNet);
//	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_numPreSynNet+100));
//
//	delete[] managerRuntimeData.wt;
//	managerRuntimeData.wt = tmp_wt;
//	cpuSnnSz.synapticInfoSize -= (sizeof(float)*numPreSynNet);
//	cpuSnnSz.synapticInfoSize += (sizeof(float)*(tmp_numPreSynNet+100));
//
//	delete[] managerRuntimeData.connIdsPreIdx;
//	managerRuntimeData.connIdsPreIdx = tmp_cumConnIdPre;
//	cpuSnnSz.synapticInfoSize -= (sizeof(short int)*numPreSynNet);
//	cpuSnnSz.synapticInfoSize += (sizeof(short int)*(tmp_numPreSynNet+100));
//
//	// compact connection-centric information
//	for (int i=0; i<numConnections; i++) {
//		tmp_mulSynFast[i] = mulSynFast[i];
//		tmp_mulSynSlow[i] = mulSynSlow[i];
//	}
//	delete[] mulSynFast;
//	delete[] mulSynSlow;
//	mulSynFast = tmp_mulSynFast;
//	mulSynSlow = tmp_mulSynSlow;
//	cpuSnnSz.networkInfoSize -= (2*sizeof(uint8_t)*numPreSynNet);
//	cpuSnnSz.networkInfoSize += (2*sizeof(uint8_t)*(tmp_numPreSynNet+100));
//
//
//	delete[] managerRuntimeData.preSynapticIds;
//	managerRuntimeData.preSynapticIds  = tmp_preSynapticIds;
//	cpuSnnSz.synapticInfoSize -= (sizeof(SynInfo)*numPreSynNet);
//	cpuSnnSz.synapticInfoSize += (sizeof(SynInfo)*(tmp_numPreSynNet+100));
//
//	numPreSynNet	= tmp_numPreSynNet;
//	numPostSynNet	= tmp_numPostSynNet;
//}

//The post synaptic connections are sorted based on delay here so that we can reduce storage requirement
//and generation of spike at the post-synaptic side.
//We also create the delay_info array has the delay_start and delay_length parameter
//void SNN::reorganizeDelay()
//{
//	for(int grpId=0; grpId < numGroups; grpId++) {
//		for(int nid=groupConfigs[0][grpId].StartN; nid <= groupConfigs[0][grpId].EndN; nid++) {
//			unsigned int jPos=0;					// this points to the top of the delay queue
//			unsigned int cumN=managerRuntimeData.cumulativePost[nid];	// cumulativePost[] is unsigned int
//			unsigned int cumDelayStart=0; 			// Npost[] is unsigned short
//			for(int td = 0; td < maxDelay_; td++) {
//				unsigned int j=jPos;				// start searching from top of the queue until the end
//				unsigned int cnt=0;					// store the number of nodes with a delay of td;
//				while(j < managerRuntimeData.Npost[nid]) {
//					// found a node j with delay=td and we put
//					// the delay value = 1 at array location td=0;
//					if(td==(tmp_SynapticDelay[cumN+j]-1)) {
//						assert(jPos<managerRuntimeData.Npost[nid]);
//						swapConnections(nid, j, jPos);
//
//						jPos=jPos+1;
//						cnt=cnt+1;
//					}
//					j=j+1;
//				}
//
//				// update the delay_length and start values...
//				managerRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_length	     = cnt;
//				managerRuntimeData.postDelayInfo[nid*(maxDelay_+1)+td].delay_index_start  = cumDelayStart;
//				cumDelayStart += cnt;
//
//				assert(cumDelayStart <= managerRuntimeData.Npost[nid]);
//			}
//
//			// total cumulative delay should be equal to number of post-synaptic connections at the end of the loop
//			assert(cumDelayStart == managerRuntimeData.Npost[nid]);
//			for(unsigned int j=1; j < managerRuntimeData.Npost[nid]; j++) {
//				unsigned int cumN=managerRuntimeData.cumulativePost[nid]; // cumulativePost[] is unsigned int
//				if( tmp_SynapticDelay[cumN+j] < tmp_SynapticDelay[cumN+j-1]) {
//	  				KERNEL_ERROR("Post-synaptic delays not sorted correctly... id=%d, delay[%d]=%d, delay[%d]=%d",
//						nid, j, tmp_SynapticDelay[cumN+j], j-1, tmp_SynapticDelay[cumN+j-1]);
//					assert( tmp_SynapticDelay[cumN+j] >= tmp_SynapticDelay[cumN+j-1]);
//				}
//			}
//		}
//	}
//}

//void SNN::swapConnections(int nid, int oldPos, int newPos) {
//	unsigned int cumN=managerRuntimeData.cumulativePost[nid];
//
//	// Put the node oldPos to the top of the delay queue
//	SynInfo tmp = managerRuntimeData.postSynapticIds[cumN+oldPos];
//	managerRuntimeData.postSynapticIds[cumN+oldPos]= managerRuntimeData.postSynapticIds[cumN+newPos];
//	managerRuntimeData.postSynapticIds[cumN+newPos]= tmp;
//
//	// Ensure that you have shifted the delay accordingly....
//	uint8_t tmp_delay = tmp_SynapticDelay[cumN+oldPos];
//	tmp_SynapticDelay[cumN+oldPos] = tmp_SynapticDelay[cumN+newPos];
//	tmp_SynapticDelay[cumN+newPos] = tmp_delay;
//
//	// update the pre-information for the postsynaptic neuron at the position oldPos.
//	SynInfo  postInfo = managerRuntimeData.postSynapticIds[cumN+oldPos];
//	int  post_nid = GET_CONN_NEURON_ID(postInfo);
//	int  post_sid = GET_CONN_SYN_ID(postInfo);
//
//	SynInfo* preId    = &(managerRuntimeData.preSynapticIds[managerRuntimeData.cumulativePre[post_nid]+post_sid]);
//	int  pre_nid  = GET_CONN_NEURON_ID((*preId));
//	int  pre_sid  = GET_CONN_SYN_ID((*preId));
//	int  pre_gid  = GET_CONN_GRP_ID((*preId));
//	assert (pre_nid == nid);
//	assert (pre_sid == newPos);
//	*preId = SET_CONN_ID( pre_nid, oldPos, pre_gid);
//
//	// update the pre-information for the postsynaptic neuron at the position newPos
//	postInfo = managerRuntimeData.postSynapticIds[cumN+newPos];
//	post_nid = GET_CONN_NEURON_ID(postInfo);
//	post_sid = GET_CONN_SYN_ID(postInfo);
//
//	preId    = &(managerRuntimeData.preSynapticIds[managerRuntimeData.cumulativePre[post_nid]+post_sid]);
//	pre_nid  = GET_CONN_NEURON_ID((*preId));
//	pre_sid  = GET_CONN_SYN_ID((*preId));
//	pre_gid  = GET_CONN_GRP_ID((*preId));
//	assert (pre_nid == nid);
//	assert (pre_sid == oldPos);
//	*preId = SET_CONN_ID( pre_nid, newPos, pre_gid);
//}

// set one specific connection from neuron id 'src' to neuron id 'dest'
//inline void SNN::setConnection(int srcGrp,  int destGrp,  unsigned int src, unsigned int dest, float synWt,
//									float maxWt, uint8_t dVal, int connProp, short int connId) {
//	assert(dest<=CONN_SYN_NEURON_MASK);			// total number of neurons is less than 1 million within a GPU
//	assert((dVal >=1) && (dVal <= maxDelay_));
//
//	// adjust sign of weight based on pre-group (negative if pre is inhibitory)
//	synWt = isExcitatoryGroup(srcGrp) ? fabs(synWt) : -1.0*fabs(synWt);
//	maxWt = isExcitatoryGroup(srcGrp) ? fabs(maxWt) : -1.0*fabs(maxWt);
//
//	// we have exceeded the number of possible connection for one neuron
//	if(managerRuntimeData.Npost[src] >= groupConfigs[0][srcGrp].numPostSynapses)	{
//		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, groupInfo[srcGrp].Name.c_str(),
//					dest, groupInfo[destGrp].Name.c_str(), synWt, dVal);
//		KERNEL_ERROR("Large number of postsynaptic connections established (%d), max for this group %d.", managerRuntimeData.Npost[src], groupConfigs[0][srcGrp].numPostSynapses);
//		exitSimulation(1);
//	}
//
//	if(managerRuntimeData.Npre[dest] >= groupConfigs[0][destGrp].numPreSynapses) {
//		KERNEL_ERROR("setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)", src, groupInfo[srcGrp].Name.c_str(),
//					dest, groupInfo[destGrp].Name.c_str(), synWt, dVal);
//		KERNEL_ERROR("Large number of presynaptic connections established (%d), max for this group %d.", managerRuntimeData.Npre[dest], groupConfigs[0][destGrp].numPreSynapses);
//		exitSimulation(1);
//	}
//
//	int p = managerRuntimeData.Npost[src];
//
//	assert(managerRuntimeData.Npost[src] >= 0);
//	assert(managerRuntimeData.Npre[dest] >= 0);
//	assert((src * maxNumPostSynGrp + p) / numN < maxNumPostSynGrp); // divide by numN to prevent INT overflow
//
//	unsigned int post_pos = managerRuntimeData.cumulativePost[src] + managerRuntimeData.Npost[src];
//	unsigned int pre_pos  = managerRuntimeData.cumulativePre[dest] + managerRuntimeData.Npre[dest];
//
//	assert(post_pos < numPostSynNet);
//	assert(pre_pos  < numPreSynNet);
//
//	//generate a new postSynapticIds id for the current connection
//	managerRuntimeData.postSynapticIds[post_pos]   = SET_CONN_ID(dest, managerRuntimeData.Npre[dest], destGrp);
//	tmp_SynapticDelay[post_pos] = dVal;
//
//	managerRuntimeData.preSynapticIds[pre_pos] = SET_CONN_ID(src, managerRuntimeData.Npost[src], srcGrp);
//	managerRuntimeData.wt[pre_pos] 	  = synWt;
//	managerRuntimeData.maxSynWt[pre_pos] = maxWt;
//	managerRuntimeData.connIdsPreIdx[pre_pos] = connId;
//
//	bool synWtType = GET_FIXED_PLASTIC(connProp);
//
//	if (synWtType == SYN_PLASTIC) {
//		sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
//		managerRuntimeData.Npre_plastic[dest]++;
//		// homeostasis
//		if (groupConfigs[0][destGrp].WithHomeostasis && groupConfigs[0][destGrp].homeoId ==-1)
//			groupConfigs[0][destGrp].homeoId = dest; // this neuron info will be printed
//	}
//
//	managerRuntimeData.Npre[dest] += 1;
//	managerRuntimeData.Npost[src] += 1;
//
//	groupInfo[srcGrp].numPostConn++;
//	groupInfo[destGrp].numPreConn++;
//
//	if (managerRuntimeData.Npost[src] > groupInfo[srcGrp].maxPostConn)
//		groupInfo[srcGrp].maxPostConn = managerRuntimeData.Npost[src];
//	if (managerRuntimeData.Npre[dest] > groupInfo[destGrp].maxPreConn)
//	groupInfo[destGrp].maxPreConn = managerRuntimeData.Npre[src];
//}
