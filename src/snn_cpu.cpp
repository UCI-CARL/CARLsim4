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
 
#include "snn.h"
#include <sstream>

#if (_WIN32 || _WIN64)
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


// FIXME what are the following for? why were they all the way at the bottom of this file?

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

#define SETPOST_INFO(name, nid, sid, val) name[cumulativePost[nid]+sid]=val;

#define SETPRE_INFO(name, nid, sid, val)  name[cumulativePre[nid]+sid]=val;




/// **************************************************************************************************************** ///
/// CONSTRUCTOR / DESTRUCTOR
/// **************************************************************************************************************** ///

CpuSNN::CpuSNN(const std::string& _name, int _numConfig, int _randSeed, int _mode) {
	fprintf(stdout, "*******************************************************************************\n");
	fprintf(stdout, "********************      Welcome to CARLsim %d.%d      *************************\n",
				MAJOR_VERSION,MINOR_VERSION);
	fprintf(stdout, "*******************************************************************************\n");
	fprintf(stdout, "Starting CARLsim simulation \"%s\"\n",_name.c_str());

	// initialize propogated spike buffers.....
	pbuf = new PropagatedSpikeBuffer(0, PROPAGATED_BUFFER_SIZE);

	numConfig 			  = _numConfig;
	finishedPoissonGroup  = false;
	assert(numConfig > 0);
	assert(numConfig < 100);

	resetPointers();
	numN = 0;
	numPostSynapses = 0;
	D = 0; // FIXME name this maxAllowedDelay or something more meaningful
	memset(&cpuSnnSz, 0, sizeof(cpuSnnSz));
	enableSimLogs = false;
	simLogDirName = "logs";

	fpLog=fopen("tmp_debug.log","w");
	fpProgLog = NULL;
	showLog = 0;		// disable showing log..
	showLogMode = 0;	// show only basic logs. if set higher more logs generated
	showGrpFiringInfo = true;

	currentMode = _mode;
	memset(&cpu_gpuNetPtrs,0,sizeof(network_ptr_t));
	memset(&net_Info,0,sizeof(network_info_t));
	cpu_gpuNetPtrs.allocated = false;

	memset(&cpuNetPtrs,0, sizeof(network_ptr_t));
	cpuNetPtrs.allocated = false;

	numSpikeMonitor  = 0;

	for (int i=0; i < MAX_GRP_PER_SNN; i++) {
		grp_Info[i].Type	 = UNKNOWN_NEURON;
		grp_Info[i].MaxFiringRate  = UNKNOWN_NEURON_MAX_FIRING_RATE;
		grp_Info[i].MonitorId		 = -1;
		grp_Info[i].FiringCount1sec=0;
		grp_Info[i].numPostSynapses 		= 0;	// default value
		grp_Info[i].numPreSynapses 	= 0;	// default value
		grp_Info[i].WithSTP = false;
		grp_Info[i].WithSTDP = false;
		grp_Info[i].FixedInputWts = true; // Default is true. This value changed to false
		// if any incoming  connections are plastic
		grp_Info[i].WithConductances = false;
		grp_Info[i].isSpikeGenerator = false;
		grp_Info[i].RatePtr = NULL;

		grp_Info[i].homeoId = -1;
		grp_Info[i].avgTimeScale  = 10000.0;

		grp_Info[i].dAMPA=1-(1.0/5);	// FIXME why default values again!? this should be in interface
		grp_Info[i].dNMDA=1-(1.0/150);
		grp_Info[i].dGABAa=1-(1.0/6);
		grp_Info[i].dGABAb=1-(1.0/150);

		grp_Info[i].spikeGen = NULL;

		grp_Info[i].StartN       = -1;
		grp_Info[i].EndN       	 = -1;

		grp_Info2[i].numPostConn = 0;
		grp_Info2[i].numPreConn  = 0;
		grp_Info2[i].enablePrint = false;
		grp_Info2[i].maxPostConn = 0;
		grp_Info2[i].maxPreConn  = 0;
		grp_Info2[i].sumPostConn = 0;
		grp_Info2[i].sumPreConn  = 0;
	}

	connectBegin = NULL;

	simTimeMs	 		= 0;	simTimeSec			= 0;	simTime = 0;
	spikeCountAll1sec	= 0;	secD1fireCntHost 	= 0;	secD2fireCntHost  = 0;
	spikeCountAll 		= 0;	spikeCountD2Host	= 0;	spikeCountD1Host = 0;
	nPoissonSpikes 		= 0;

	networkName	= _name;
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

	stpu = NULL;
	stpx = NULL;
	gAMPA = NULL;
	gNMDA = NULL;
	gGABAa = NULL;
	gGABAb = NULL;

	if (_randSeed == -1) {
		randSeed = time(NULL);
	}
	else if(_randSeed==0) {
		randSeed=123;
	}
	srand48(randSeed);
	getRand.seed(randSeed*2);
	getRandClosed.seed(randSeed*3);

	fprintf(stdout, "numConfig: %d, randSeed: %d\n",_numConfig,randSeed);

	fpParam = fopen("param.txt", "w");
	if (fpParam==NULL) {
		fprintf(stderr, "WARNING !!! Unable to open/create parameter file 'param.txt'; check if current directory is writable \n");
		exit(1);
		return;
	}
	fprintf(fpParam, "// *****************************************\n");
	time_t rawtime; struct tm * timeinfo;
	time ( &rawtime ); timeinfo = localtime ( &rawtime );
	fprintf ( fpParam,  "// program name : %s \n", _name.c_str());
	fprintf ( fpParam,  "// rand val  : %d \n", randSeed);
	fprintf ( fpParam,  "// Current local time and date: %s\n", asctime (timeinfo));
	fflush(fpParam);

	CUDA_CREATE_TIMER(timer);
	CUDA_RESET_TIMER(timer);
	cumExecutionTime = 0.0;

	spikeRateUpdated = false;

	sim_with_fixedwts = true; // default is true, will be set to false if there are any plastic synapses
	sim_with_conductances = false; // for all others, the default is false
	sim_with_stdp = false;
	sim_with_stp = false;

	maxSpikesD2 = maxSpikesD1 = 0;
	readNetworkFID = NULL;

	// initialize parameters needed in snn_gpu.cu
	CpuSNNinitGPUparams();
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
int CpuSNN::connect(int grpId1, int grpId2, const std::string& _type, float initWt, float maxWt, float prob,
						uint8_t minDelay, uint8_t maxDelay, float mulSynFast, float mulSynSlow, bool synWtType) {
						//const std::string& wtType
	int retId=-1;
	for(int c=0; c < numConfig; c++, grpId1++, grpId2++) {
		assert(grpId1 < numGrp);
		assert(grpId2 < numGrp);
		assert(minDelay <= maxDelay);

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
		newInfo->mulSynFast			= mulSynFast;
		newInfo->mulSynSlow			= mulSynSlow;
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
		else if ( _type.find("full") != std::string::npos) {
			newInfo->type 	= CONN_FULL;
			newInfo->numPostSynapses	= grp_Info[grpId2].SizeN;
			newInfo->numPreSynapses   = grp_Info[grpId1].SizeN;
		}
		else if ( _type.find("full-no-direct") != std::string::npos) {
			newInfo->type 	= CONN_FULL_NO_DIRECT;
			newInfo->numPostSynapses	= grp_Info[grpId2].SizeN-1;
			newInfo->numPreSynapses	= grp_Info[grpId1].SizeN-1;
		}
		else if ( _type.find("one-to-one") != std::string::npos) {
			newInfo->type 	= CONN_ONE_TO_ONE;
			newInfo->numPostSynapses	= 1;
			newInfo->numPreSynapses	= 1;
		}
		else {
			fprintf(stderr, "Invalid connection type (should be 'random', 'full', 'one-to-one', or 'full-no-direct')\n");
			exitSimulation(-1);
		}

		if (newInfo->numPostSynapses > MAX_numPostSynapses) {
			printf("Connection exceeded the maximum number of output synapses (%d), has %d.\n",MAX_numPostSynapses,newInfo->numPostSynapses);
			assert(newInfo->numPostSynapses <= MAX_numPostSynapses);
		}

		if (newInfo->numPreSynapses > MAX_numPreSynapses) {
			printf("Connection exceeded the maximum number of input synapses (%d), has %d.\n",MAX_numPreSynapses,newInfo->numPreSynapses);
			assert(newInfo->numPreSynapses <= MAX_numPreSynapses);
		}

		// update the pre and post size...
		// Subtlety: each group has numPost/PreSynapses from multiple connections.  
		// The newInfo->numPost/PreSynapses are just for this specific connection.  
		// We are adding the synapses counted in this specific connection to the totals for both groups.
		grp_Info[grpId1].numPostSynapses 	+= newInfo->numPostSynapses;
		grp_Info[grpId2].numPreSynapses 	+= newInfo->numPreSynapses;

		if (showLogMode >= 1)
			printf("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d\n",grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

		newInfo->connId	  = numConnections++;
		if(c==0)
			retId = newInfo->connId;
	}
	assert(retId != -1);
	return retId;
}

// make custom connections from grpId1 to grpId2
int CpuSNN::connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow, 
						bool synWtType, int maxM, int maxPreM) {
	int retId=-1;

	for(int c=0; c < numConfig; c++, grpId1++, grpId2++) {
		assert(grpId1 < numGrp);
		assert(grpId2 < numGrp);

		if (maxM == 0)
			maxM = grp_Info[grpId2].SizeN;

		if (maxPreM == 0)
			maxPreM = grp_Info[grpId1].SizeN;

		if (maxM > MAX_numPostSynapses) {
			printf("Connection from %s (%d) to %s (%d) exceeded the maximum number of output synapses (%d), has %d.\n",
						grp_Info2[grpId1].Name.c_str(),grpId1,grp_Info2[grpId2].Name.c_str(), grpId2,
						MAX_numPostSynapses,maxM);
			assert(maxM <= MAX_numPostSynapses);
		}

		if (maxPreM > MAX_numPreSynapses) {
			printf("Connection from %s (%d) to %s (%d) exceeded the maximum number of input synapses (%d), has %d.\n",
						grp_Info2[grpId1].Name.c_str(), grpId1,grp_Info2[grpId2].Name.c_str(), grpId2,
						MAX_numPreSynapses,maxPreM);
			assert(maxPreM <= MAX_numPreSynapses);
		}

		grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));

		newInfo->grpSrc   = grpId1;
		newInfo->grpDest  = grpId2;
		newInfo->initWt	  = 1;
		newInfo->maxWt	  = 1;
		newInfo->maxDelay = 1;
		newInfo->minDelay = 1;
		newInfo->mulSynFast = mulSynFast;
		newInfo->mulSynSlow = mulSynSlow;
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

		if (showLogMode >= 1) {
			printf("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d\n",
						grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,
						grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);
		}

		newInfo->connId	  = numConnections++;
		if(c==0)
			retId = newInfo->connId;
	}
	assert(retId != -1);
	return retId;
}


// create group of Izhikevich neurons
int CpuSNN::createGroup(const std::string& grpName, unsigned int nNeur, int neurType, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			createGroup(grpName, nNeur, neurType, c);
		return (numGrp-numConfig);
	} else {
		assert(numGrp < MAX_GRP_PER_SNN);

		if ( (!(neurType&TARGET_AMPA) && !(neurType&TARGET_NMDA) &&
			  !(neurType&TARGET_GABAa) && !(neurType&TARGET_GABAb)) || (neurType&POISSON_NEURON)) {
			fprintf(stderr, "Invalid type using createGroup...\n");
			fprintf(stderr, "can not create poisson generators here...\n");
			exitSimulation(1);
		}

		grp_Info[numGrp].SizeN  			= nNeur;
		grp_Info[numGrp].Type   			= neurType;
		grp_Info[numGrp].WithConductances	= false;
		grp_Info[numGrp].WithSTP			= false;
		grp_Info[numGrp].WithSTDP			= false;
		grp_Info[numGrp].WithHomeostasis	= false;
    
		if ( (neurType&TARGET_GABAa) || (neurType&TARGET_GABAb)) {
			grp_Info[numGrp].MaxFiringRate 	= INHIBITORY_NEURON_MAX_FIRING_RATE;
		}
		else {
			grp_Info[numGrp].MaxFiringRate 	= EXCITATORY_NEURON_MAX_FIRING_RATE;
		}

		grp_Info2[numGrp].ConfigId			= configId;
		grp_Info2[numGrp].Name  			= grpName;
		grp_Info[numGrp].isSpikeGenerator	= false;
		grp_Info[numGrp].MaxDelay			= 1;

		grp_Info2[numGrp].Izh_a 			= -1; // FIXME ???

		std::stringstream outStr;
		outStr << configId;
		grp_Info2[numGrp].Name 				= (configId==0)?grpName:grpName+"_"+outStr.str();
		finishedPoissonGroup				= true;

		numGrp++;
		return (numGrp-1);
	}
}

// create spike generator group
int CpuSNN::createSpikeGeneratorGroup(const std::string& grpName, unsigned int nNeur, int neurType, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			createSpikeGeneratorGroup(grpName, nNeur, neurType, c);
		return (numGrp-numConfig);
	} else {
		grp_Info[numGrp].SizeN   		= nNeur;
		grp_Info[numGrp].Type    		= neurType | POISSON_NEURON;
		grp_Info[numGrp].WithConductances	= false;
		grp_Info[numGrp].WithSTP		= false;
		grp_Info[numGrp].WithSTDP		= false;
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

// set conductance values for a group (custom values or disable conductances alltogether)
void CpuSNN::setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb,
								int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setConductances(g, isSet, tdAMPA, tdNMDA, tdGABAa, tdGABAb, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setConductances(g, isSet, tdAMPA, tdNMDA, tdGABAa, tdGABAb, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
			setConductances(grpId, isSet, tdAMPA, tdNMDA, tdGABAa, tdGABAb, c);
	} else {
		// set conductances for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_conductances 			   |= isSet;
		grp_Info[cGrpId].WithConductances 	= isSet;
		grp_Info[cGrpId].dAMPA 				= 1-(1.0/tdAMPA);	// factor for numerical integration
		grp_Info[cGrpId].dNMDA 				= 1-(1.0/tdNMDA);	// iAMPA[t+1] = iAMPA[t]*dAMPA
		grp_Info[cGrpId].dGABAa 			= 1-(1.0/tdGABAa);	// => exponential decay
		grp_Info[cGrpId].dGABAb 			= 1-(1.0/tdGABAb);
		grp_Info[cGrpId].newUpdates 		= true; 			// FIXME What is this?

		fprintf(stderr, "Conductances %s for %d (%s):\ttdAMPA: %4.0f, tdNMDA: %4.0f, tdGABAa: %4.0f, tdGABAb: %4.0f\n",
					isSet?"enabled":"disabled",cGrpId, grp_Info2[cGrpId].Name.c_str(),tdAMPA,tdNMDA,tdGABAa,tdGABAb);
	}
}

// set homeostasis for group
void CpuSNN::setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setHomeostasis(g, isSet, homeoScale, avgTimeScale, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setHomeostasis(g, isSet, homeoScale, avgTimeScale, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
			setHomeostasis(grpId, isSet, homeoScale, avgTimeScale, c);
	} else {
		// set conductances for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		grp_Info[cGrpId].WithHomeostasis    = isSet;
		grp_Info[cGrpId].homeostasisScale   = homeoScale;
		grp_Info[cGrpId].avgTimeScale       = avgTimeScale;
		grp_Info[cGrpId].avgTimeScaleInv    = 1.0f/avgTimeScale;
		grp_Info[cGrpId].avgTimeScale_decay = (avgTimeScale*1000.0f-1.0f)/(avgTimeScale*1000.0f);
		grp_Info[cGrpId].newUpdates 		= true; // FIXME: what's this?

		fprintf(stderr, "Homeostasis parameters %s for %d (%s):\thomeoScale: %f, avgTimeScale: %f\n",
					isSet?"enabled":"disabled",cGrpId,grp_Info2[cGrpId].Name.c_str(),homeoScale,avgTimeScale);
	}
}

// set a homeostatic target firing rate (enforced through homeostatic synaptic scaling)
void CpuSNN::setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setHomeoBaseFiringRate(g, baseFiring, baseFiringSD, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setHomeoBaseFiringRate(g, baseFiring, baseFiringSD, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
			setHomeoBaseFiringRate(grpId, baseFiring, baseFiringSD, c);
	} else {
		// set conductances for a given group and configId
		int cGrpId 						= getGroupId(grpId, configId);
		grp_Info2[cGrpId].baseFiring 	= baseFiring;
		grp_Info2[cGrpId].baseFiringSD 	= baseFiringSD;
		grp_Info[cGrpId].newUpdates 	= true; //TODO: I have to see how this is handled.  -- KDC

		fprintf(stderr, "Homeostatic base firing rate set for %d (%s):\tbaseFiring: %3.3f, baseFiringStd: %3.3f\n",
							cGrpId,grp_Info2[cGrpId].Name.c_str(),baseFiring,baseFiringSD);
	}
}


// set Izhikevich parameters for group
void CpuSNN::setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
								float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setNeuronParameters(g, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setNeuronParameters(g, izh_a, izh_a_sd, izh_b, izh_b_sd, izh_c, izh_c_sd, izh_d, izh_d_sd, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
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


// set STDP params
void CpuSNN::setSTDP(int grpId, bool isSet, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setSTDP(g, isSet, alphaLTP, tauLTP, alphaLTD, tauLTD, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setSTDP(g, isSet, alphaLTP, tauLTP, alphaLTD, tauLTD, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
			setSTDP(grpId, isSet, alphaLTP, tauLTP, alphaLTD, tauLTD, c);
	} else {
		// set STDP for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_stdp 				   |= isSet;
		grp_Info[cGrpId].WithSTDP 		= isSet;
		grp_Info[cGrpId].ALPHA_LTP 		= alphaLTP;
		grp_Info[cGrpId].ALPHA_LTD 		= alphaLTD;
		grp_Info[cGrpId].TAU_LTP_INV 	= 1.0/tauLTP;
		grp_Info[cGrpId].TAU_LTD_INV	= 1.0/tauLTD;
		grp_Info[cGrpId].newUpdates 	= true; // FIXME whatsathiis?

		fprintf(stderr, "STDP %s for %d (%s):\talphaLTP: %1.4f, alphaLTD: %1.4f, tauLTP: %4.0f, tauLTD: %4.0f\n",
					isSet?"enabled":"disabled",cGrpId,grp_Info2[cGrpId].Name.c_str(),
					alphaLTP,alphaLTD,tauLTP,tauLTD);
	}
}


// set STP params
void CpuSNN::setSTP(int grpId, bool isSet, float STP_U, float STP_tD, float STP_tF, int configId) {
	if (grpId==ALL && configId==ALL) { // shortcut for all groups & configs
		for(int g=0; g < numGrp; g++)
			setSTP(g, isSet, STP_U, STP_tD, STP_tF, 0);
	} else if (grpId == ALL) { // shortcut for all groups
		for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
			int g = getGroupId(grpId1, configId);
			setSTP(g, isSet, STP_U, STP_tD, STP_tF, configId);
		}
	} else if (configId == ALL) { // shortcut for all configs
		for(int c=0; c < numConfig; c++)
			setSTP(grpId, isSet, STP_U, STP_tD, STP_tF, c);
	} else {
		// set STDP for a given group and configId
		int cGrpId = getGroupId(grpId, configId);
		sim_with_stp 			   |= isSet;
		grp_Info[cGrpId].WithSTP 	= isSet;
		grp_Info[cGrpId].STP_U 		= STP_U;
		grp_Info[cGrpId].STP_tD 	= STP_tD;
		grp_Info[cGrpId].STP_tF 	= STP_tF;
		grp_Info[cGrpId].newUpdates = true;

		fprintf(stderr, "STP %s for %d (%s):\tU: %1.4f, tD: %4.0f, tF: %4.0f\n", isSet?"enabled":"disabled",
					cGrpId, grp_Info2[cGrpId].Name.c_str(), STP_U, STP_tD, STP_tF);
	}
}


/// ************************************************************************************************************ ///
/// PUBLIC METHODS: RUNNING A SIMULATION
/// ************************************************************************************************************ ///

// Run the simulation for n sec
int CpuSNN::runNetwork(int _nsec, int _nmsec, int simType, int ithGPU, bool enablePrint, bool copyState) {
	DBG(2, fpLog, AT, "runNetwork() called");

	assert(_nmsec >= 0);
	assert(_nsec  >= 0);
	assert(simType == CPU_MODE || simType == GPU_MODE);
	int runDuration = _nsec*1000 + _nmsec;

	// set the Poisson generation time slice to be at the run duration up to PROPOGATED_BUFFER_SIZE ms.
	setGrpTimeSlice(ALL, MAX(1,MIN(runDuration,PROPAGATED_BUFFER_SIZE-1))); 

	// First time when the network is run we do various kind of space compression,
	// and data structure optimization to improve performance and save memory.
	setupNetwork(simType,ithGPU);

	currentMode = simType;

	CUDA_RESET_TIMER(timer);
	CUDA_START_TIMER(timer);

	// if nsec=0, simTimeMs=10, we need to run the simulator for 10 timeStep;
	// if nsec=1, simTimeMs=10, we need to run the simulator for 1*1000+10, time Step;
	for(int i=0; i < runDuration; i++) {
		if(simType == CPU_MODE)
			doSnnSim();
		else
			doGPUSim();

		if (enablePrint) {
			printState();
		}

		if (updateTime()) {
			// finished one sec of simulation...
			if(showLog) {
				if(showLogCycle==showLog++) {
					showStatus(currentMode);
					showLog=1;
				}
			}

			updateSpikeMonitor();

			if(simType == CPU_MODE)
				updateStateAndFiringTable();
			else
				updateStateAndFiringTable_GPU();
		}

		if(enableGPUSpikeCntPtr==true && simType == CPU_MODE){
			fprintf(stderr,"Error: the enableGPUSpikeCntPtr flag cannot be set in CPU_MODE\n");
			assert(currentMode==GPU_MODE);
		}
		if(enableGPUSpikeCntPtr==true && simType == GPU_MODE){
			copyFiringStateFromGPU();
		}
	}
	if(copyState) {
		// copy the state from GPU to GPU
		for(int g=0; g < numGrp; g++) {
			if ((!grp_Info[g].isSpikeGenerator) && (currentMode==GPU_MODE)) {
				copyNeuronState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, g);
			}
		}
	}
  
	// keep track of simulation time...
	CUDA_STOP_TIMER(timer);
	lastExecutionTime = CUDA_GET_TIMER_VALUE(timer);
	cumExecutionTime += lastExecutionTime;
	return 0;
}



/// ************************************************************************************************************ ///
/// PUBLIC METHODS: INTERACTING WITH A SIMULATION
/// ************************************************************************************************************ ///

// Returns pointer to nSpikeCnt, which is a 1D array of the number of spikes every neuron in the group
unsigned int* CpuSNN::getSpikeCntPtr(int grpId, int simType) {
	//! do check to make sure appropriate flag is set
	if(simType == GPU_MODE && enableGPUSpikeCntPtr == false){
		fprintf(stderr,"Error: the enableGPUSpikeCntPtr flag must be set to true to use this function in GPU_MODE.\n");
		assert(enableGPUSpikeCntPtr);
	}
    
	if(simType == GPU_MODE){
		assert(enableGPUSpikeCntPtr);
	}
    
	return ((grpId == -1) ? nSpikeCnt : &nSpikeCnt[grp_Info[grpId].StartN]);
}

// reads network state from file
void CpuSNN::readNetwork(FILE* fid) {
	readNetworkFID = fid;
}

// reassigns weights from the input weightMatrix to the weights between two
// specified neuron groups.
// TODO: figure out scope; is this a user function?
void CpuSNN::reassignFixedWeights(int connectId, float weightMatrix[], int sizeMatrix, int configId) {
	// handle the config == ALL recursive call contigency.
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			reassignFixedWeights(connectId, weightMatrix, sizeMatrix, c);
	} else {
		int j;
		//first find the correct connection
		grpConnectInfo_t* connInfo; //connInfo = current connection information.
		connInfo = getConnectInfo(connectId,configId);
		//make sure that it is for fixed connections.
		bool synWtType = GET_FIXED_PLASTIC(connInfo->connProp);
		if(synWtType == SYN_PLASTIC){
			printf("The synapses in this connection must be SYN_FIXED in order to use this function.\n");
			assert(false);
		}
		//make sure that the user passes the correctly sized matrix
		if(connInfo->numberOfConnections != sizeMatrix){
			printf("The size of the input weight matrix and the number of synaptic connections in this connection do not match.\n");
			assert(false);
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
				int currentSrcId = findGrpId(preId);
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
	if(currentMode == GPU_MODE)
		copyUpdateVariables_GPU();  
}

void CpuSNN::resetSpikeCntUtil(int my_grpId ) {
  int startGrp, endGrp;

  if(!doneReorganization)
    return;

  if(currentMode == GPU_MODE){
    //call analogous function, return, else do CPU stuff
    if (my_grpId == ALL) {
      startGrp = 0;
      endGrp   = numGrp;
    }
    else {
      startGrp = my_grpId;
      endGrp   = my_grpId+numConfig;
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
    endGrp   = my_grpId+numConfig;
  }
  
  for( int grpId=startGrp; grpId < endGrp; grpId++) {
    int startN = grp_Info[grpId].StartN;
    int endN   = grp_Info[grpId].EndN+1;
    for (int i=startN; i < endN; i++)
      nSpikeCnt[i] = 0;
  }
}

// sets up a spike generator
void CpuSNN::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			setSpikeGenerator(grpId, spikeGen,c);
	} else {
		int cGrpId = getGroupId(grpId, configId);

		assert(spikeGen);
		assert (grp_Info[cGrpId].isSpikeGenerator);
		grp_Info[cGrpId].spikeGen = spikeGen;
	}
}


// add a SpikeMonitor for group where spikeMon can be custom class or WriteSpikesToFile
void CpuSNN::setSpikeMonitor(int grpId, SpikeMonitor* spikeMon, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
		setSpikeMonitor(grpId, spikeMon,c);
	} else {
	    int cGrpId = getGroupId(grpId, configId);
	    DBG(2, fpLog, AT, "spikeMonitor Added");

	    // store the gid for further reference
	    monGrpId[numSpikeMonitor]	= cGrpId;

	    // also inform the grp that it is being monitored...
	    grp_Info[cGrpId].MonitorId		= numSpikeMonitor;

	    float maxRate	= grp_Info[cGrpId].MaxFiringRate;

	    // count the size of the buffer for storing 1 sec worth of info..
	    // only the last second is stored in this buffer...
	    int buffSize = (int)(maxRate*grp_Info[cGrpId].SizeN);

	    // store the size for future comparison etc.
	    monBufferSize[numSpikeMonitor] = buffSize;

	    // reset the position of the buffer pointer..
	    monBufferPos[numSpikeMonitor]  = 0;

	    monBufferCallback[numSpikeMonitor] = spikeMon;

	    // create the new buffer for keeping track of all the spikes in the system
	    monBufferFiring[numSpikeMonitor] = new unsigned int[buffSize];
	    monBufferTimeCnt[numSpikeMonitor]= new unsigned int[1000];
	    memset(monBufferTimeCnt[numSpikeMonitor],0,sizeof(int)*(1000));

	    numSpikeMonitor++;

	    // oh. finally update the size info that will be useful to see
	    // how much memory are we eating...
	    cpuSnnSz.monitorInfoSize += sizeof(int)*buffSize;
	    cpuSnnSz.monitorInfoSize += sizeof(int)*(1000);

	    fprintf(stderr,"SpikeMonitor set for group %d (%s)\n",grpId,grp_Info2[grpId].Name.c_str());
	}
}

// assigns spike rate to group
void CpuSNN::setSpikeRate(int grpId, PoissonRate* ratePtr, int refPeriod, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			setSpikeRate(grpId, ratePtr, refPeriod,c);
	} else {
		int cGrpId = getGroupId(grpId, configId);
		if(grp_Info[cGrpId].RatePtr==NULL) {
			fprintf(fpParam, " // refPeriod = %d\n", refPeriod);
		}

		assert(ratePtr);
		if (ratePtr->len != grp_Info[cGrpId].SizeN) {
			fprintf(stderr,"The PoissonRate length did not match the number of neurons in group %s(%d).\n",
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
    fprintf(stderr,"UpdateNetwork function was called but nothing was done because reorganizeNetwork must be called first.\n");
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

  if(currentMode==GPU_MODE){
    //copyGrpInfo_GPU();
    //do a call to updateNetwork_GPU()
    updateNetwork_GPU(resetFiringInfo);
  }

  printTuningLog();
}

// writes network state to file
void CpuSNN::writeNetwork(FILE* fid) {
	unsigned int version = 1;
	fwrite(&version,sizeof(int),1,fid);
	fwrite(&numGrp,sizeof(int),1,fid);
	char name[100];

	for (int g=0;g<numGrp;g++) {
		fwrite(&grp_Info[g].StartN,sizeof(int),1,fid);
		fwrite(&grp_Info[g].EndN,sizeof(int),1,fid);

		strncpy(name,grp_Info2[g].Name.c_str(),100);
		fwrite(name,1,100,fid);
	}

	int nrCells = numN;
	fwrite(&nrCells,sizeof(int),1,fid);

	for (unsigned int i=0;i<nrCells;i++) {
		unsigned int offset = cumulativePost[i];

		unsigned int count = 0;
		for (int t=0;t<D;t++) {
			delay_info_t dPar = postDelayInfo[i*(D+1)+t];

			for(int idx_d=dPar.delay_index_start; idx_d<(dPar.delay_index_start+dPar.delay_length); idx_d++)
				count++;
		}

		fwrite(&count,sizeof(int),1,fid);

		for (int t=0;t<D;t++) {
			delay_info_t dPar = postDelayInfo[i*(D+1)+t];

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

				fwrite(&i,sizeof(int),1,fid);
				fwrite(&p_i,sizeof(int),1,fid);
				fwrite(&(wt[pos_i]),sizeof(float),1,fid);
				fwrite(&(maxSynWt[pos_i]),sizeof(float),1,fid);
				fwrite(&delay,sizeof(uint8_t),1,fid);
				fwrite(&plastic,sizeof(uint8_t),1,fid);
			}
		}
	}
}

// writes population weights from gIDpre to gIDpost to file fname in binary
void CpuSNN::writePopWeights(std::string fname, int grpPreId, int grpPostId, int configId){
	float* weights;
	int matrixSize;
	FILE* fid; 
	int numPre, numPost;
	fid = fopen(fname.c_str(), "wb");
	assert(fid != NULL);

	if(!doneReorganization){
		printf("Simulation has not been run yet, cannot output weights.\n");
		exitSimulation(1);
	}

	post_info_t* preId;
	int pre_nid, pos_ij;

	if(configId < 0){
		printf("Invalid configId.  You can not pass the ALL (ALL=-1) argument.\n");
		assert(false);
	}

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
		if(currentMode==GPU_MODE){
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

//! sets the update cycle for log messages
void CpuSNN::setLogCycle(unsigned int _cnt, int mode, FILE *fp) {
	//enable or disable logging...
	showLog = (_cnt == 0)? 0 : 1;

	//set the update cycle...
	showLogCycle = _cnt;

	showLogMode = mode;

	if (fp!=NULL)
		fpProgLog = fp;
}


/// **************************************************************************************************************** ///
/// GETTERS / SETTERS
/// **************************************************************************************************************** ///

//! used for parameter tuning functionality
grpConnectInfo_t* CpuSNN::getConnectInfo(int connectId, int configId) {
	grpConnectInfo_t* nextConn = connectBegin;
	connectId = getConnectionId (connectId, configId);
	CHECK_CONNECTION_ID(connectId, numConnections);

	// clear all existing connection info...
	while (nextConn) {
		if (nextConn->connId == connectId) {
			nextConn->newUpdates = true;
			return nextConn;
		}
		nextConn = nextConn->next;
	}

	fprintf(stderr, "Total Connections = %d\n", numConnections);
	fprintf(stderr, "ConnectId (%d) cannot be recognized\n", connectId);
	return NULL;
}

int  CpuSNN::getConnectionId(int connId, int configId) {
	if(configId >= numConfig) {
		fprintf(stderr, "getConnectionId(int, int): Assertion `configId(%d) < numConfig(%d)' failed\n", 
					configId, numConfig);
		assert(0);
	}
	connId = connId+configId;
	if (connId  >= numConnections) {
		fprintf(stderr, "getConnectionId(int, int): Assertion `connId(%d) < numConnections(%d)' failed\n", 
					connId, numConnections);
		assert(0);
	}
	return connId;
}

// this is a user function
// FIXME: fix this
uint8_t* CpuSNN::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
	Npre = grp_Info[gIDpre].SizeN;
	Npost = grp_Info[gIDpost].SizeN;

	if (delays == NULL) delays = new uint8_t[Npre*Npost];
	memset(delays,0,Npre*Npost);

	for (int i=grp_Info[gIDpre].StartN;i<grp_Info[gIDpre].EndN;i++) {
		unsigned int offset = cumulativePost[i];

		for (int t=0;t<D;t++) {
			delay_info_t dPar = postDelayInfo[i*(D+1)+t];

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


int CpuSNN::getGroupId(int grpId, int configId) {
	assert(configId < numConfig);
	int cGrpId = (grpId+configId);
	assert(cGrpId  < numGrp);
	return cGrpId;
}

group_info_t CpuSNN::getGroupInfo(int grpId, int configId) {
	int cGrpId = getGroupId(grpId, configId);
	return grp_Info[cGrpId];
}

std::string CpuSNN::getGroupName(int grpId, int configId) {
	int cGrpId = getGroupId(grpId, configId);
	return grp_Info2[cGrpId].Name;
}

// returns the number of synaptic connections associated with this connection.
int CpuSNN::getNumConnections(int connectionId) {
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
  printf("Connection ID was not found.  Quitting.\n");
  assert(false);  
}

// gets weights from synaptic connections from gIDpre to gIDpost
void CpuSNN::getPopWeights(int grpPreId, int grpPostId, float*& weights, int& matrixSize, int configId) {
	post_info_t* preId;
	int pre_nid, pos_ij;
	int numPre, numPost;
  
	if(configId < 0){
		printf("Invalid configId.  You can not pass the ALL (ALL=-1) argument.\n");
		exitSimulation(1);
	}

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
		if(currentMode==GPU_MODE){
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

// this is a user function
// TODO: fix this
float* CpuSNN::getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges) {
  Npre = grp_Info[gIDpre].SizeN;
  Npost = grp_Info[gIDpost].SizeN;

  if (weightChanges==NULL) weightChanges = new float[Npre*Npost];
  memset(weightChanges,0,Npre*Npost*sizeof(float));

  // copy the pre synaptic data from GPU, if needed
  // note: this will not include wtChange[] and synSpikeTime[] if sim_with_fixedwts
  if (currentMode == GPU_MODE)
    copyWeightState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, gIDpost);

  for (int i=grp_Info[gIDpre].StartN;i<grp_Info[gIDpre].EndN;i++) {
    unsigned int offset = cumulativePost[i];

    for (int t=0;t<D;t++) {
      delay_info_t dPar = postDelayInfo[i*(D+1)+t];

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

void CpuSNN::setGroupInfo(int grpId, group_info_t info, int configId) {
	if (configId == ALL) {
		for(int c=0; c < numConfig; c++)
			setGroupInfo(grpId, info, c);
	} else {
		int cGrpId = getGroupId(grpId, configId);
		grp_Info[cGrpId] = info;
	}
}

void CpuSNN::setPrintState(int grpId, bool status) {
	grp_Info2[grpId].enablePrint = status;
}

void CpuSNN::setSimLogs(bool isSet, std::string logDirName) {
	enableSimLogs = isSet;
	if (logDirName != "") {
		simLogDirName = logDirName;
	}
}

void CpuSNN::setTuningLog(std::string fname) {
	fpTuningLog = fopen(fname.c_str(),"w");
	assert(fpTuningLog != NULL);
}



/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

void CpuSNN::CpuSNNInit(unsigned int nNeur, unsigned int nPostSyn, unsigned int nPreSyn, unsigned int maxDelay) {
	numN = nNeur;
	numPostSynapses = nPostSyn;
	D = maxDelay; // FIXME
	numPreSynapses = nPreSyn;

	voltage	 = new float[numNReg];
	recovery = new float[numNReg];
	Izh_a	 = new float[numNReg];
	Izh_b    = new float[numNReg];
	Izh_c	 = new float[numNReg];
	Izh_d	 = new float[numNReg];
	current	 = new float[numNReg];
	cpuSnnSz.neuronInfoSize += (sizeof(int)*numNReg*12);

	// all or none of the groups must have conductances enabled
	// user error handling is done in interface
	if (sim_with_conductances) {
		for (int g=0;g<numGrp;g++) {
			assert(grp_Info[g].WithConductances || !grp_Info[g].WithConductances && grp_Info[g].Type&POISSON_NEURON);
//			if (!grp_Info[g].WithConductances && ((grp_Info[g].Type&POISSON_NEURON)==0)) {
//				printf("If one group enables conductances then all groups, except for generators, must enable conductances.  Group '%s' is not enabled.\n",
//							grp_Info2[g].Name.c_str());
//				assert(false);
//			}
		}

		gAMPA  = new float[numNReg];
		gNMDA  = new float[numNReg];
		gGABAa = new float[numNReg];
		gGABAb = new float[numNReg];
		cpuSnnSz.neuronInfoSize += sizeof(int)*numNReg*4;
	}

	resetCurrent();
	resetConductances();

	lastSpikeTime	= new uint32_t[numN];
	cpuSnnSz.neuronInfoSize += sizeof(int)*numN;
	memset(lastSpikeTime,0,sizeof(lastSpikeTime[0]*numN));

	curSpike   = new bool[numN];
	nSpikeCnt  = new unsigned int[numN];

	//! homeostasis variables
	avgFiring  = new float[numN];
	baseFiring = new float[numN];

	intrinsicWeight  = new float[numN];
	memset(intrinsicWeight,0,sizeof(float)*numN);
	cpuSnnSz.neuronInfoSize += (sizeof(int)*numN*2+sizeof(bool)*numN);

	if (sim_with_stp) {
		stpu = new float[numN*STP_BUF_SIZE];
		stpx = new float[numN*STP_BUF_SIZE];
		for (int i=0; i < numN*STP_BUF_SIZE; i++) {
			stpu[i] = 1; // some default value
			stpx[i] = 1;
		}
		cpuSnnSz.synapticInfoSize += (sizeof(stpu[0])*numN*STP_BUF_SIZE);
	}

	Npre 		   = new unsigned short[numN];
	Npre_plastic   = new unsigned short[numN];
	Npost 		   = new unsigned short[numN];
	cumulativePost = new unsigned int[numN];
	cumulativePre  = new unsigned int[numN];
	cpuSnnSz.networkInfoSize += (int)(sizeof(int)*numN*3.5);

	postSynCnt = 0;
	preSynCnt  = 0;
	for(int g=0; g < numGrp; g++) {
		// check for INT overflow: postSynCnt is O(numNeurons*numSynapses), must be able to fit within u int limit
		assert(postSynCnt < UINT_MAX - (grp_Info[g].SizeN*grp_Info[g].numPostSynapses));
		assert(preSynCnt < UINT_MAX - (grp_Info[g].SizeN*grp_Info[g].numPreSynapses));
		postSynCnt += (grp_Info[g].SizeN*grp_Info[g].numPostSynapses);
		preSynCnt  += (grp_Info[g].SizeN*grp_Info[g].numPreSynapses);
	}
	assert(postSynCnt/numN <= numPostSynapses); // divide by numN to prevent INT overflow
	postSynapticIds		= new post_info_t[postSynCnt+100];
	tmp_SynapticDelay	= new uint8_t[postSynCnt+100];	//!< Temporary array to store the delays of each connection
	postDelayInfo		= new delay_info_t[numN*(D+1)];	//!< Possible delay values are 0....D (inclusive of D)
	cpuSnnSz.networkInfoSize += ((sizeof(post_info_t)+sizeof(uint8_t))*postSynCnt+100)+(sizeof(delay_info_t)*numN*(D+1));
	assert(preSynCnt/numN <= numPreSynapses); // divide by numN to prevent INT overflow

	wt  			= new float[preSynCnt+100];
	maxSynWt     	= new float[preSynCnt+100];

	connIdFromSynId = new int[preSynCnt+100];

	//! Temporary array to hold pre-syn connections. will be deleted later if necessary
	preSynapticIds	= new post_info_t[preSynCnt+100];

	// size due to weights and maximum weights
	cpuSnnSz.synapticInfoSize += ((sizeof(int)+2*sizeof(float)+sizeof(post_info_t))*(preSynCnt+100));

	timeTableD2  = new unsigned int[1000+D+1];
	timeTableD1  = new unsigned int[1000+D+1];
	resetTimingTable();
	cpuSnnSz.spikingInfoSize += sizeof(int)*2*(1000+D+1);

	// poisson Firing Rate
	cpuSnnSz.neuronInfoSize += (sizeof(int)*numNPois);

	tmp_SynapseMatrix_fixed = NULL;
	tmp_SynapseMatrix_plastic = NULL;
}


int CpuSNN::addSpikeToTable(int nid, int g) {
	int spikeBufferFull = 0;
	lastSpikeTime[nid] = simTime;
	curSpike[nid] = true;
	nSpikeCnt[nid]++;
	avgFiring[nid] += 1000/(grp_Info[g].avgTimeScale*1000);

	if(showLogMode >= 3)
		if (nid<128) printf("spiked: %d\n",nid);

	if (currentMode == GPU_MODE) {
		assert(grp_Info[g].isSpikeGenerator == true);
		setSpikeGenBit_GPU(nid, g);
		return 0;
	}

	if (grp_Info[g].WithSTP) {
		// implements Mongillo, Barak and Tsodyks model of Short term plasticity
		int ind = STP_BUF_POS(nid,simTime);
		int ind_1 = STP_BUF_POS(nid,(simTime-1)); // MDR -1 is correct, we use the value before the decay has been
												  // applied for the current time step.
		stpx[ind] = stpx[ind] - stpu[ind_1]*stpx[ind_1];
		stpu[ind] = stpu[ind] + grp_Info[g].STP_U*(1-stpu[ind_1]);
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
	}
	else {
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

	fprintf(fpLog, "Allocation for %d(%s), St=%d, End=%d\n",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);

	resetSpikeCnt(grpId);

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

void CpuSNN::buildNetwork() {
	grpConnectInfo_t* newInfo = connectBegin;
	int curN = 0, curD = 0, numPostSynapses = 0, numPreSynapses = 0;

	assert(numConfig > 0);

	//update main set of parameters
	updateParameters(&curN, &numPostSynapses, &numPreSynapses, numConfig);

	curD = updateSpikeTables();

	assert((curN > 0)&& (curN == numNExcReg + numNInhReg + numNPois));
	assert(numPostSynapses > 0);
	assert(numPreSynapses > 0);

	// display the evaluated network and delay length....
	fprintf(stdout, ">>>>>>>>>>>>>> NUM_CONFIGURATIONS = %d <<<<<<<<<<<<<<<<<<\n", numConfig);
	fprintf(stdout, "**********************************\n");
	fprintf(stdout, "numN = %d, numPostSynapses = %d, numPreSynapses = %d, D = %d\n", curN, numPostSynapses,
					numPreSynapses, curD);
	fprintf(stdout, "**********************************\n");

	fprintf(fpLog, "**********************************\n");
	fprintf(fpLog, "numN = %d, numPostSynapses = %d, numPreSynapses = %d, D = %d\n", curN, numPostSynapses,
					numPreSynapses, curD);
	fprintf(fpLog, "**********************************\n");

	assert(curD != 0); 	assert(numPostSynapses != 0);		assert(curN != 0); 		assert(numPreSynapses != 0);

	if (showLogMode >= 1) {
		for (int g=0;g<numGrp;g++)
			printf("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d\n",
						g,grp_Info2[g].Name.c_str(),grp_Info[g].numPostSynapses,g,grp_Info2[g].Name.c_str(),
						grp_Info[g].numPreSynapses);
	}

	if (numPostSynapses > MAX_numPostSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPostSynapses>MAX_numPostSynapses)
				printf("Grp: %s(%d) has too many output synapses (%d), max %d.\n",grp_Info2[g].Name.c_str(),g,
							grp_Info[g].numPostSynapses,MAX_numPostSynapses);
		}
		assert(numPostSynapses <= MAX_numPostSynapses);
	}
	if (numPreSynapses > MAX_numPreSynapses) {
		for (int g=0;g<numGrp;g++) {
			if (grp_Info[g].numPreSynapses>MAX_numPreSynapses)
				printf("Grp: %s(%d) has too many input synapses (%d), max %d.\n",grp_Info2[g].Name.c_str(),g,
 							grp_Info[g].numPreSynapses,MAX_numPreSynapses);
		}
		assert(numPreSynapses <= MAX_numPreSynapses);
	}
	assert(curD <= MAX_SynapticDelay); assert(curN <= 1000000);

	// initialize all the parameters....
	CpuSNNInit(curN, numPostSynapses, numPreSynapses, curD);

	// we build network in the order...
	/////    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
	////     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
	///      Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
	int allocatedGrp = 0;
	for(int order=0; order < 4; order++) {
		for(int configId=0; configId < numConfig; configId++) {
			for(int g=0; g < numGrp; g++) {
				if (grp_Info2[g].ConfigId == configId) {
					if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && (grp_Info[g].Type&POISSON_NEURON) && order==3) {
						buildPoissonGroup(g);
						allocatedGrp++;
					} else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type&POISSON_NEURON) && order==2) {
						buildPoissonGroup(g);
						allocatedGrp++;
					} else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type&POISSON_NEURON) && order==0) {
					  	buildGroup(g);
					    allocatedGrp++;
					} else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type&POISSON_NEURON) && order==1) {
						buildGroup(g);
						allocatedGrp++;
					}
				}
			}
		}
	}
	assert(allocatedGrp == numGrp);

	if (readNetworkFID != NULL) {
		// we the user specified readNetwork the synaptic weights will be restored here...
		#if READNETWORK_ADD_SYNAPSES_FROM_FILE
			assert(readNetwork_internal(true) >= 0); // read the plastic synapses first
			assert(readNetwork_internal(false) >= 0); // read the fixed synapses second
		#else
			assert(readNetwork_internal() >= 0);
			connectFromMatrix(tmp_SynapseMatrix_plastic, SET_FIXED_PLASTIC(SYN_PLASTIC));
			connectFromMatrix(tmp_SynapseMatrix_fixed, SET_FIXED_PLASTIC(SYN_FIXED));
		#endif
	} else {
		// build all the connections here...
		// we run over the linked list two times...
		// first time, we make all plastic connections...
		// second time, we make all fixed connections...
		// this ensures that all the initial pre and post-synaptic
		// connections are of fixed type and later if of plastic type
		for(int con=0; con < 2; con++) {
			newInfo = connectBegin;
			while(newInfo) {
				bool synWtType = GET_FIXED_PLASTIC(newInfo->connProp);
				if (synWtType == SYN_PLASTIC) {
					// given group has plastic connection, and we need to apply STDP rule...
					grp_Info[newInfo->grpDest].FixedInputWts = false;
				}

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
							printf("Invalid connection type( should be 'random', or 'full')\n");
					}

					float avgPostM = newInfo->numberOfConnections/grp_Info[newInfo->grpSrc].SizeN;
					float avgPreM  = newInfo->numberOfConnections/grp_Info[newInfo->grpDest].SizeN;

					fprintf(stderr, "connect(%s(%d) => %s(%d), iWt=%f, mWt=%f, numPostSynapses=%d, numPreSynapses=%d, minD=%d, maxD=%d, %s)\n",
								grp_Info2[newInfo->grpSrc].Name.c_str(), newInfo->grpSrc,
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

	fprintf(fpLog, "Allocation for %d(%s), St=%d, End=%d\n",
				grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].StartN, grp_Info[grpId].EndN);
	resetSpikeCnt(grpId);

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
  fprintf(fpLog, "******************\n");
  fprintf(fpLog, "CompactConnection: \n");
  fprintf(fpLog, "******************\n");
  fprintf(fpLog, "old_postCnt = %d, new_postCnt = %d\n", postSynCnt, tmp_postSynCnt);
  fprintf(fpLog, "old_preCnt = %d,  new_postCnt = %d\n", preSynCnt,  tmp_preSynCnt);

  // new buffer with required size + 100 bytes of additional space just to provide limited overflow
  post_info_t* tmp_postSynapticIds   = new post_info_t[tmp_postSynCnt+100];

  // new buffer with required size + 100 bytes of additional space just to provide limited overflow
  post_info_t*   tmp_preSynapticIds = new post_info_t[tmp_preSynCnt+100];
  float* tmp_wt	    	  = new float[tmp_preSynCnt+100];
  float* tmp_maxSynWt   	  = new float[tmp_preSynCnt+100];

  for(int i=0; i < numN; i++) {
    assert(tmp_cumulativePost[i] <= cumulativePost[i]);
    assert(tmp_cumulativePre[i]  <= cumulativePre[i]);
    for( int j=0; j < Npost[i]; j++) {
      unsigned int tmpPos = tmp_cumulativePost[i]+j;
      unsigned int oldPos = cumulativePost[i]+j;
      tmp_postSynapticIds[tmpPos] = postSynapticIds[oldPos];
      tmp_SynapticDelay[tmpPos]   = tmp_SynapticDelay[oldPos];
    }
    for( int j=0; j < Npre[i]; j++) {
      unsigned int tmpPos =  tmp_cumulativePre[i]+j;
      unsigned int oldPos =  cumulativePre[i]+j;
      tmp_preSynapticIds[tmpPos]  = preSynapticIds[oldPos];
      tmp_maxSynWt[tmpPos] 	    = maxSynWt[oldPos];
      tmp_wt[tmpPos]              = wt[oldPos];
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
  cpuSnnSz.synapticInfoSize += (sizeof(int)*(tmp_preSynCnt+100));

  delete[] wt;
  wt = tmp_wt;
  cpuSnnSz.synapticInfoSize -= (sizeof(float)*preSynCnt);
  cpuSnnSz.synapticInfoSize += (sizeof(int)*(tmp_preSynCnt+100));

  delete[] preSynapticIds;
  preSynapticIds  = tmp_preSynapticIds;
  cpuSnnSz.synapticInfoSize -= (sizeof(post_info_t)*preSynCnt);
  cpuSnnSz.synapticInfoSize += (sizeof(post_info_t)*(tmp_preSynCnt+100));

  preSynCnt	= tmp_preSynCnt;
  postSynCnt	= tmp_postSynCnt;
}

void CpuSNN::connectFromMatrix(SparseWeightDelayMatrix* mat, int connProp) {
  for (int i=0;i<mat->count;i++) {
    int nIDpre = mat->preIds[i];
    int nIDpost = mat->postIds[i];
    float weight = mat->weights[i];
    float maxWeight = mat->maxWeights[i];
    uint8_t delay = mat->delay_opts[i];
    int gIDpre = findGrpId(nIDpre);
    int gIDpost = findGrpId(nIDpost);

    setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp);

    grp_Info2[gIDpre].sumPostConn++;
    grp_Info2[gIDpost].sumPreConn++;

    if (delay > grp_Info[gIDpre].MaxDelay) grp_Info[gIDpre].MaxDelay = delay;
  }
}

// make 'C' full connections from grpSrc to grpDest
void CpuSNN::connectFull(grpConnectInfo_t* info) {
  int grpSrc = info->grpSrc;
  int grpDest = info->grpDest;
  bool noDirect = (info->type == CONN_FULL_NO_DIRECT);

  for(int nid=grp_Info[grpSrc].StartN; nid<=grp_Info[grpSrc].EndN; nid++)  {
    for(int j=grp_Info[grpDest].StartN; j <= grp_Info[grpDest].EndN; j++) {
      if((noDirect) && (nid - grp_Info[grpSrc].StartN) == (j - grp_Info[grpDest].StartN))
	continue;
      uint8_t dVal = info->minDelay + (int)(0.5+(getRandClosed()*(info->maxDelay-info->minDelay)));
      assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
      float synWt = getWeights(info->connProp, info->initWt, info->maxWt, nid, grpSrc);

      setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp);
      info->numberOfConnections++;
      //setConnection(grpSrc, grpDest, nid, j, info->initWt, info->maxWt, dVal, info->connProp);
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
    setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp);
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
	setConnection(grpSrc, grpDest, pre_nid, post_nid, synWt, info->maxWt, dVal, info->connProp);
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
	if (GET_FIXED_PLASTIC(info->connProp) == SYN_FIXED) maxWt = weight;

	assert(delay>=1);
	assert(delay<=MAX_SynapticDelay);
	assert(weight<=maxWt);

	setConnection(grpSrc, grpDest, nid, nid2, weight, maxWt, delay, info->connProp);
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
	try {
		if(simulatorDeleted)
			return;

		if(fpLog) {
			printSimSummary(fpLog); // TODO: can fpLog be stdout? In this case printSimSummary is executed twice
			printSimSummary();
			fclose(fpLog);
		}

		// close param.txt
		if (fpParam) {
			fclose(fpParam);
		}

		if (voltage!=NULL) 	delete[] voltage;
		if (recovery!=NULL) 	delete[] recovery;
		if (Izh_a!=NULL) 	delete[] Izh_a;
		if (Izh_b!=NULL)		delete[] Izh_b;
		if (Izh_c!=NULL)		delete[] Izh_c;
		if (Izh_d!=NULL)		delete[] Izh_d;
		if (current!=NULL)	delete[] current;

		if (Npre!=NULL)	delete[] Npre;
		if (Npre_plastic!=NULL) delete[] Npre_plastic;
		if (Npost!=NULL)	delete[] Npost;

		if (cumulativePre!=NULL) delete[] cumulativePre;
		if (cumulativePost!=NULL) delete[] cumulativePost;

		if (gAMPA!=NULL) delete[] gAMPA;
		if (gNMDA!=NULL) delete[] gNMDA;
		if (gGABAa!=NULL) delete[] gGABAa;
		if (gGABAb!=NULL) delete[] gGABAb;

		if (stpu!=NULL) delete[] stpu;
		if (stpx!=NULL) delete[] stpx;

		if (lastSpikeTime!=NULL)		delete[] lastSpikeTime;
		if (synSpikeTime !=NULL)		delete[] synSpikeTime;
		if (curSpike!=NULL) delete[] curSpike;
		if (nSpikeCnt!=NULL) delete[] nSpikeCnt;
		if (intrinsicWeight!=NULL) delete[] intrinsicWeight;

		if (postDelayInfo!=NULL) delete[] postDelayInfo;
		if (preSynapticIds!=NULL) delete[] preSynapticIds;
		if (postSynapticIds!=NULL) delete[] postSynapticIds;
		if (tmp_SynapticDelay!=NULL) delete[] tmp_SynapticDelay;

		if(wt!=NULL)			delete[] wt;
		if(maxSynWt!=NULL)		delete[] maxSynWt;
		if(wtChange !=NULL)		delete[] wtChange;
		if(connIdFromSynId!=NULL)	delete[] connIdFromSynId;

		if (firingTableD2) delete[] firingTableD2;
		if (firingTableD1) delete[] firingTableD1;
		if (timeTableD2!=NULL) delete[] timeTableD2;
		if (timeTableD1!=NULL) delete[] timeTableD1;

		delete pbuf;

		// clear all existing connection info...
		while (connectBegin) {
			grpConnectInfo_t* nextConn = connectBegin->next;
			free(connectBegin);
			connectBegin = nextConn;
		}

		for (int i = 0; i < numSpikeMonitor; i++) {
			delete[] monBufferFiring[i];
			delete[] monBufferTimeCnt[i];
		}

		if(spikeGenBits) delete[] spikeGenBits;

		// do the same as above, but for snn_gpu.cu
		deleteObjects_GPU();

		CUDA_DELETE_TIMER(timer);

		simulatorDeleted = true;
	}
	catch(...) {
		fprintf(stderr, "Unknown exception ...\n");
	}
}



// This method loops through all spikes that are generated by neurons with a delay of 1ms
// and delivers the spikes to the appropriate post-synaptic neuron
void CpuSNN::doD1CurrentUpdate()
{
  int k     = secD1fireCntHost-1;
  int k_end = timeTableD1[simTimeMs+D];

  while((k>=k_end) && (k>=0)) {

    int neuron_id      = firingTableD1[k];
    assert(neuron_id<numN);

    delay_info_t dPar = postDelayInfo[neuron_id*(D+1)];

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
void CpuSNN::doD2CurrentUpdate()
{
  int k = secD2fireCntHost-1;
  int k_end = timeTableD2[simTimeMs+1];
  int t_pos = simTimeMs;

  while((k>=k_end)&& (k >=0)) {

    // get the neuron id from the index k
    int i  = firingTableD2[k];

    // find the time of firing from the timeTable using index k
    while (!((k >= timeTableD2[t_pos+D])&&(k < timeTableD2[t_pos+D+1]))) {
      t_pos = t_pos - 1;
      assert((t_pos+D-1)>=0);
    }

    // TODO: Instead of using the complex timeTable, can neuronFiringTime value...???
    // Calculate the time difference between time of firing of neuron and the current time...
    int tD = simTimeMs - t_pos;

    assert((tD<D)&&(tD>=0));
    assert(i<numN);

    delay_info_t dPar = postDelayInfo[i*(D+1)+tD];

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
	doSTPUpdates();

	updateSpikeGenerators();

	//generate all the scheduled spikes from the spikeBuffer..
	generateSpikes();

	// find the neurons that has fired..
	findFiring();

	timeTableD2[simTimeMs+D+1] = secD2fireCntHost;
	timeTableD1[simTimeMs+D+1] = secD1fireCntHost;

	doD2CurrentUpdate();
	doD1CurrentUpdate();
	globalStateUpdate();

	return;
}

void CpuSNN::doSTPUpdates() {
	int spikeBufferFull = 0;

	//decay the STP variables before adding new spikes.
	for(int g=0; (g < numGrp) & !spikeBufferFull; g++) {
		if (grp_Info[g].WithSTP) {
			for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
				int ind = 0, ind_1 = 0;
				ind = STP_BUF_POS(i,simTime);
				ind_1 = STP_BUF_POS(i,(simTime-1));
				stpx[ind] = stpx[ind_1] + (1-stpx[ind_1])/grp_Info[g].STP_tD;
				stpu[ind] = stpu[ind_1] + (grp_Info[g].STP_U - stpu[ind_1])/grp_Info[g].STP_tF;
			}
		}
	}
}


// deallocates dynamical structures and exits
void CpuSNN::exitSimulation(int val) {
	deleteObjects();
	exit(val);
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

      if (grp_Info[g].WithConductances) {
	gAMPA[i] *= grp_Info[g].dAMPA;
	gNMDA[i] *= grp_Info[g].dNMDA;
	gGABAa[i] *= grp_Info[g].dGABAa;
	gGABAb[i] *= grp_Info[g].dGABAb;
      }
      else
	current[i] = 0.0f; // in CUBA mode, reset current to 0 at each time step and sum up all wts

      if (voltage[i] >= 30.0) {
	voltage[i] = Izh_c[i];
	recovery[i] += Izh_d[i];

	spikeBufferFull = addSpikeToTable(i, g);

	if (spikeBufferFull)  break;

	if (grp_Info[g].WithSTDP) {
	  unsigned int pos_ij = cumulativePre[i];
	  for(int j=0; j < Npre_plastic[i]; pos_ij++, j++) {
	    //stdpChanged[pos_ij] = true;
	    int stdp_tDiff = (simTime-synSpikeTime[pos_ij]);
	    assert(!((stdp_tDiff < 0) && (synSpikeTime[pos_ij] != MAX_SIMULATION_TIME)));
	    // don't do LTP if time difference is a lot..

	    if (stdp_tDiff > 0)
#ifdef INHIBITORY_STDP
	      // if this is an excitatory or inhibitory synapse
	      if (maxSynWt[pos_ij] >= 0)
#endif
		if ((stdp_tDiff*grp_Info[g].TAU_LTP_INV)<25)
		  wtChange[pos_ij] += STDP(stdp_tDiff, grp_Info[g].ALPHA_LTP, grp_Info[g].TAU_LTP_INV);

#ifdef INHIBITORY_STDP
		else
		  if ((stdp_tDiff > 0) && ((stdp_tDiff*grp_Info[g].TAU_LTD_INV)<25))
		    wtChange[pos_ij] -= (STDP(stdp_tDiff, grp_Info[g].ALPHA_LTP, grp_Info[g].TAU_LTP_INV)
		    					 - STDP(stdp_tDiff, grp_Info[g].ALPHA_LTD*1.5, grp_Info[g].TAU_LTD_INV));
#endif
	  }
	}
	spikeCountAll1sec++;
      }
    }
  }
}

int CpuSNN::findGrpId(int nid) {
	for(int g=0; g < numGrp; g++) {
		if(nid >=grp_Info[g].StartN && (nid <=grp_Info[g].EndN)) {
			return g;
		}
	}
	fprintf(stderr, "findGrp(): cannot find the group for neuron %d\n", nid);
	assert(0);
}


void CpuSNN::generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD) {
  // get synaptic info...
  post_info_t post_info = postSynapticIds[offset + idx_d];

  // get neuron id
  unsigned int p_i = GET_CONN_NEURON_ID(post_info);
  assert(p_i<numN);

  // get syn id
  int s_i = GET_CONN_SYN_ID(post_info);
  assert(s_i<(Npre[p_i]));

  // get the cumulative position for quick access...
  unsigned int pos_i = cumulativePre[p_i] + s_i;

  assert(p_i < numNReg);

  float change;

  int pre_grpId = findGrpId(pre_i);
  char type = grp_Info[pre_grpId].Type;

  // TODO: MNJ TEST THESE CONDITIONS FOR CORRECTNESS...
  int ind = STP_BUF_POS(pre_i,(simTime-tD-1));

  // if the source group STP is disabled. we need to skip it..
  if (grp_Info[pre_grpId].WithSTP) {
    change = wt[pos_i]*stpx[ind]*stpu[ind];
  } else
    change = wt[pos_i];

  if (grp_Info[pre_grpId].WithConductances) {
    if (type & TARGET_AMPA)  gAMPA [p_i] += change;
    if (type & TARGET_NMDA)  gNMDA [p_i] += change;
    if (type & TARGET_GABAa) gGABAa[p_i] -= change; // wt should be negative for GABAa and GABAb
    if (type & TARGET_GABAb) gGABAb[p_i] -= change;
  } else
    current[p_i] += change;

  int post_grpId = findGrpId(p_i);
  if ((showLogMode >= 3) && (p_i==grp_Info[post_grpId].StartN))
    printf("%d => %d (%d) am=%f ga=%f wt=%f stpu=%f stpx=%f td=%d\n",
	   pre_i, p_i, findGrpId(p_i), gAMPA[p_i], gGABAa[p_i],
	   wt[pos_i],(grp_Info[post_grpId].WithSTP?stpx[ind]:1.0),(grp_Info[post_grpId].WithSTP?stpu[ind]:1.0),tD);

  // STDP calculation....
  if (grp_Info[post_grpId].WithSTDP) {
    //stdpChanged[pos_i]=false;
    //assert((simTime-lastSpikeTime[p_i])>=0);
    int stdp_tDiff = (simTime-lastSpikeTime[p_i]);

    if (stdp_tDiff >= 0) {
#ifdef INHIBITORY_STDP
      if ((type & TARGET_GABAa) || (type & TARGET_GABAb))
	{
	  //printf("I");
	  if ((stdp_tDiff*grp_Info[post_grpId].TAU_LTD_INV)<25)
	    wtChange[pos_i] -= (STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTP, grp_Info[post_grpId].TAU_LTP_INV)
	    					 - STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTD*1.5, grp_Info[post_grpId].TAU_LTD_INV));
	}
      else
#endif
	{
	  //printf("E");
	  if ((stdp_tDiff*grp_Info[post_grpId].TAU_LTD_INV)<25)
	    wtChange[pos_i] -= STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTD, grp_Info[post_grpId].TAU_LTD_INV);
	}

    }
    assert(!((stdp_tDiff < 0) && (lastSpikeTime[p_i] != MAX_SIMULATION_TIME)));
  }

  synSpikeTime[pos_i] = simTime;
}

void CpuSNN::generateSpikes() {
	PropagatedSpikeBuffer::const_iterator srg_iter;
	PropagatedSpikeBuffer::const_iterator srg_iter_end = pbuf->endSpikeTargetGroups();

	for( srg_iter = pbuf->beginSpikeTargetGroups(); srg_iter != srg_iter_end; ++srg_iter )  {
		// Get the target neurons for the given groupId
		int nid	 = srg_iter->stg;
		//delaystep_t del = srg_iter->delay;
		//generate a spike to all the target neurons from source neuron nid with a delay of del
		int g = findGrpId(nid);

		addSpikeToTable (nid, g);
		//fprintf(stderr, "nid = %d\t", nid);
		spikeCountAll1sec++;
		nPoissonSpikes++;
	}

	// advance the time step to the next phase...
	pbuf->nextTimeStep();
}

void CpuSNN::generateSpikesFromFuncPtr(int grpId) {
  bool done;
  SpikeGenerator* spikeGen = grp_Info[grpId].spikeGen;
  int timeSlice = grp_Info[grpId].CurrTimeSlice;
  unsigned int currTime = simTime;
  int spikeCnt=0;
  for(int i=grp_Info[grpId].StartN;i<=grp_Info[grpId].EndN;i++) {
    // start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
    unsigned int nextTime = lastSpikeTime[i];
    if (nextTime == MAX_SIMULATION_TIME)
      nextTime = 0;

    done = false;
    while (!done) {

      nextTime = spikeGen->nextSpikeTime(this, grpId, i-grp_Info[grpId].StartN, nextTime);

      // found a valid time window
      if (nextTime < (currTime+timeSlice)) {
	if (nextTime >= currTime) {
	  // scheduled spike...
	  //fprintf(stderr, "scheduled time = %d, nid = %d\n", nextTime, i);
	  pbuf->scheduleSpikeTargetGroup(i, nextTime-currTime);
	  spikeCnt++;
	}
      }
      else {
	done=true;
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
		printf("specifying rates on the GPU but using the CPU SNN is not supported.\n");
		return;
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
  // FIXME: are these ramping thingies still supported?
  bool setRandomWeights   = GET_INITWTS_RANDOM(connProp);
  bool setRampDownWeights = GET_INITWTS_RAMPDOWN(connProp);
  bool setRampUpWeights   = GET_INITWTS_RAMPUP(connProp);

  if ( setRandomWeights  )
    actWts=initWt*drand48();
  else if (setRampUpWeights)
    actWts=(initWt+((nid-grp_Info[grpId].StartN)*(maxWt-initWt)/grp_Info[grpId].SizeN));
  else if (setRampDownWeights)
    actWts=(maxWt-((nid-grp_Info[grpId].StartN)*(maxWt-initWt)/grp_Info[grpId].SizeN));
  else
    actWts=initWt;

  return actWts;
}


void  CpuSNN::globalStateUpdate() {
#define CUR_DEBUG 1
  //fprintf(stdout, "---%d ----\n", simTime);
  // now we update the state of all the neurons
  for(int g=0; g < numGrp; g++) {
    if (grp_Info[g].Type&POISSON_NEURON){ 
      for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++)
	avgFiring[i] *= grp_Info[g].avgTimeScale_decay;
      continue;
    }

    for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
      assert(i < numNReg);
      avgFiring[i] *= grp_Info[g].avgTimeScale_decay;

      if (grp_Info[g].WithConductances) {
	// all the tmpIs will be summed into current[i] in the following loop
	current[i] = 0.0f;

	for (int j=0; j<COND_INTEGRATION_SCALE; j++) {
	  float NMDAtmp = (voltage[i]+80)*(voltage[i]+80)/60/60;
	  // There is an instability issue when dealing with large conductances, which causes the membr.
	  // pot. to plateau just below the spike threshold... We cap the "slow" conductances to prevent
	  // this issue. Note: 8.0 and 2.0 seemed to work in some experiments, but it might not be the
	  // best choice in general... compare updateNeuronState() in snn_gpu.cu
	  float tmpI =  - (  gAMPA[i]*(voltage[i]-0)
			     + MIN(8.0f,gNMDA[i])*NMDAtmp/(1+NMDAtmp)*(voltage[i]-0) // cap gNMDA at 8.0
			     + gGABAa[i]*(voltage[i]+70)
			     + MIN(2.0f,gGABAb[i])*(voltage[i]+90)); // cap gGABAb at 2.0

	  current[i] += tmpI;

#ifdef NEURON_NOISE
	  float noiseI = -intrinsicWeight[i]*log(getRand());
	  if (isnan(noiseI) || isinf(noiseI)) noiseI = 0;
	  tmpI += noiseI;
#endif

	  voltage[i]+=((0.04f*voltage[i]+5)*voltage[i]+140-recovery[i]+tmpI)/COND_INTEGRATION_SCALE;
	  assert(!isnan(voltage[i]) && !isinf(voltage[i]));

	  if (voltage[i] > 30) {
	    voltage[i] = 30;
	    j=COND_INTEGRATION_SCALE; // break the loop but evaluate u[i]
	  }
	  if (voltage[i] < -90) voltage[i] = -90;
	  recovery[i]+=Izh_a[i]*(Izh_b[i]*voltage[i]-recovery[i])/COND_INTEGRATION_SCALE;
	}
      } else {
	voltage[i]+=0.5f*((0.04f*voltage[i]+5)*voltage[i]+140-recovery[i]+current[i]); // for numerical stability
	voltage[i]+=0.5f*((0.04f*voltage[i]+5)*voltage[i]+140-recovery[i]+current[i]); // time step is 0.5 ms
	if (voltage[i] > 30) voltage[i] = 30;
	if (voltage[i] < -90) voltage[i] = -90;
	recovery[i]+=Izh_a[i]*(Izh_b[i]*voltage[i]-recovery[i]);
      }

      if ((showLogMode >= 2) && (i==grp_Info[g].StartN))
	fprintf(stdout, "%d: voltage=%0.3f, recovery=%0.5f, AMPA=%0.5f, NMDA=%0.5f\n",
		i,  voltage[i], recovery[i], gAMPA[i], gNMDA[i]);
    }
  }
}


// creates the CPU net pointers
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
	cpuNetPtrs.connIdFromSynId	= connIdFromSynId;
	cpuNetPtrs.nSpikeCnt		= nSpikeCnt;
	cpuNetPtrs.curSpike 		= curSpike;
	cpuNetPtrs.firingTableD2 	= firingTableD2;
	cpuNetPtrs.firingTableD1 	= firingTableD1;

	// homeostasis variables
	cpuNetPtrs.avgFiring    	= avgFiring;
	cpuNetPtrs.baseFiring   	= baseFiring;

	cpuNetPtrs.gAMPA        	= gAMPA;
	cpuNetPtrs.gNMDA			= gNMDA;
	cpuNetPtrs.gGABAa       	= gGABAa;
	cpuNetPtrs.gGABAb			= gGABAb;
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

// FIXME: this guy is a mess
#if READNETWORK_ADD_SYNAPSES_FROM_FILE
int CpuSNN::readNetwork_internal(bool onlyPlastic)
#else
  int CpuSNN::readNetwork_internal()
#endif
{
  long file_position = ftell(readNetworkFID); // so that we can restore the file position later...
  unsigned int version;

  if (!fread(&version,sizeof(int),1,readNetworkFID)) return -11;

  if (version > 1) return -10;

  int _numGrp;
  if (!fread(&_numGrp,sizeof(int),1,readNetworkFID)) return -11;

  if (numGrp != _numGrp) return -1;

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

  int nrCells;
  if (!fread(&nrCells,sizeof(int),1,readNetworkFID)) return -11;

  if (nrCells != numN) return -5;

  tmp_SynapseMatrix_fixed = new SparseWeightDelayMatrix(nrCells,nrCells,nrCells*10);
  tmp_SynapseMatrix_plastic = new SparseWeightDelayMatrix(nrCells,nrCells,nrCells*10);

  for (unsigned int i=0;i<nrCells;i++) {
    unsigned int nrSynapses = 0;
    if (!fread(&nrSynapses,sizeof(int),1,readNetworkFID)) return -11;

    for (int j=0;j<nrSynapses;j++) {
      unsigned int nIDpre;
      unsigned int nIDpost;
      float weight, maxWeight;
      uint8_t delay;
      uint8_t plastic;

      if (!fread(&nIDpre,sizeof(int),1,readNetworkFID)) return -11;

      if (nIDpre != i) return -6;

      if (!fread(&nIDpost,sizeof(int),1,readNetworkFID)) return -11;

      if (nIDpost >= nrCells) return -7;

      if (!fread(&weight,sizeof(float),1,readNetworkFID)) return -11;

      int gIDpre = findGrpId(nIDpre);
      if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight>0)
      		|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight<0))
      {
      	return -8;
      }

      if (!fread(&maxWeight,sizeof(float),1,readNetworkFID)) return -11;

      if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight>=0)
      		|| !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight<=0))
      {
      	return -8;
      }

      if (!fread(&delay,sizeof(uint8_t),1,readNetworkFID)) return -11;

      if (delay > MAX_SynapticDelay) return -9;

      if (!fread(&plastic,sizeof(uint8_t),1,readNetworkFID)) return -11;

#if READNETWORK_ADD_SYNAPSES_FROM_FILE
      if ((plastic && onlyPlastic) || (!plastic && !onlyPlastic)) {
	int gIDpost = findGrpId(nIDpost);
	int connProp = SET_FIXED_PLASTIC(plastic?SYN_PLASTIC:SYN_FIXED);

	setConnection(gIDpre, gIDpost, nIDpre, nIDpost, weight, maxWeight, delay, connProp);

	grp_Info2[gIDpre].sumPostConn++;
	grp_Info2[gIDpost].sumPreConn++;

	if (delay > grp_Info[gIDpre].MaxDelay) grp_Info[gIDpre].MaxDelay = delay;
      }
#else
      // add the synapse to the temporary Matrix so that it can be used in buildNetwork()
      if (plastic) {
	tmp_SynapseMatrix_plastic->add(nIDpre,nIDpost,weight,maxWeight,delay,plastic);
      } else {
	tmp_SynapseMatrix_fixed->add(nIDpre,nIDpost,weight,maxWeight,delay,plastic);
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
      for(int td = 0; td < D; td++) {
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
	postDelayInfo[nid*(D+1)+td].delay_length	     = cnt;
	postDelayInfo[nid*(D+1)+td].delay_index_start  = cumDelayStart;
	cumDelayStart += cnt;

	assert(cumDelayStart <= Npost[nid]);
      }

      // total cumulative delay should be equal to number of post-synaptic connections at the end of the loop
      assert(cumDelayStart == Npost[nid]);
      for(unsigned int j=1; j < Npost[nid]; j++) {
	unsigned int cumN=cumulativePost[nid]; // cumulativePost[] is unsigned int
	if( tmp_SynapticDelay[cumN+j] < tmp_SynapticDelay[cumN+j-1]) {
	  fprintf(stderr, "Post-synaptic delays not sorted correctly...\n");
	  fprintf(stderr, "id=%d, delay[%d]=%d, delay[%d]=%d\n",
		  nid, j, tmp_SynapticDelay[cumN+j], j-1, tmp_SynapticDelay[cumN+j-1]);
	  assert( tmp_SynapticDelay[cumN+j] >= tmp_SynapticDelay[cumN+j-1]);
	}
      }
    }
  }
}

// after all the initalization. Its time to create the synaptic weights, weight change and also
// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
void CpuSNN::reorganizeNetwork(bool removeTempMemory, int simType) {
	//Double check...sometimes by mistake we might call reorganize network again...
	if(doneReorganization)
		return;

	fprintf(stdout, "Beginning reorganization of network....\n");

	// time to build the complete network with relevant parameters..
	buildNetwork();

	//..minimize any other wastage in that array by compacting the store
	compactConnections();

	// The post synaptic connections are sorted based on delay here
	reorganizeDelay();

	// Print statistics of the memory used to stdout...
	printMemoryInfo();

	// Print the statistics again but dump the results to a file
	printMemoryInfo(fpLog);

	// initialize the synaptic weights accordingly..
	initSynapticWeights();

	updateSpikeGeneratorsInit();

	//ensure that we dont do all the above optimizations again
	doneReorganization = true;

	printParameters(fpLog);
	printTuningLog();

	makePtrInfo();

	if(simType==GPU_MODE)
		fprintf(stdout, "Starting GPU-SNN Simulations ....\n");
	else
		fprintf(stdout, "Starting CPU-SNN Simulations ....\n");


	if(removeTempMemory) {
		memoryOptimized = true;
		delete[] tmp_SynapticDelay;
		tmp_SynapticDelay = NULL;
	}
}


void CpuSNN::resetConductances() {
	if (sim_with_conductances) {
		assert(gAMPA != NULL);
		memset(gAMPA, 0, sizeof(float)*numNReg);
		memset(gNMDA, 0, sizeof(float)*numNReg);
		memset(gGABAa, 0, sizeof(float)*numNReg);
		memset(gGABAb, 0, sizeof(float)*numNReg);
	}
}

void CpuSNN::resetCounters() {
	assert(numNReg <= numN);
	memset( curSpike, 0, sizeof(bool)*numN);
}

void CpuSNN::resetCPUTiming() {
	prevCpuExecutionTime = cumExecutionTime;
	cpuExecutionTime     = 0.0;
}

void CpuSNN::resetCurrent() {
	assert(current != NULL);
	memset(current, 0, sizeof(float)*numNReg);
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

void CpuSNN::resetNeuron(unsigned int neurId, int grpId) {
	assert(neurId < numNReg);
    if (grp_Info2[grpId].Izh_a == -1) {
		printf("setNeuronParameters must be called for group %s (%d)\n",grp_Info2[grpId].Name.c_str(),grpId);
		exit(-1);
	}

	Izh_a[neurId] = grp_Info2[grpId].Izh_a + grp_Info2[grpId].Izh_a_sd*(float)getRandClosed();
	Izh_b[neurId] = grp_Info2[grpId].Izh_b + grp_Info2[grpId].Izh_b_sd*(float)getRandClosed();
	Izh_c[neurId] = grp_Info2[grpId].Izh_c + grp_Info2[grpId].Izh_c_sd*(float)getRandClosed();
	Izh_d[neurId] = grp_Info2[grpId].Izh_d + grp_Info2[grpId].Izh_d_sd*(float)getRandClosed();

	voltage[neurId] = Izh_c[neurId];	// initial values for new_v
	recovery[neurId] = 0.2f*voltage[neurId];   		// initial values for u
  
 
	// set the baseFiring with some standard deviation.
	if(drand48()>0.5)   {
	baseFiring[neurId] = grp_Info2[grpId].baseFiring + grp_Info2[grpId].baseFiringSD*-log(drand48());
	}
	else  {
	baseFiring[neurId] = grp_Info2[grpId].baseFiring - grp_Info2[grpId].baseFiringSD*-log(drand48());
	if(baseFiring[neurId] < 0.1) baseFiring[neurId] = 0.1;
	}

	if( grp_Info2[grpId].baseFiring != 0.0) {
		avgFiring[neurId]  = baseFiring[neurId];
	}
	else {
		baseFiring[neurId] = 0.0;
		avgFiring[neurId]  = 0;
	}
  
	lastSpikeTime[neurId]  = MAX_SIMULATION_TIME;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j < STP_BUF_SIZE; j++) {
			int ind=STP_BUF_POS(neurId,j);
			stpu[ind] = grp_Info[grpId].STP_U;
			stpx[ind] = 1;
		}
	}
}

void CpuSNN::resetPointers() {
	voltage = NULL;
	recovery = NULL;
	Izh_a = NULL;
	Izh_b = NULL;
	Izh_c = NULL;
	Izh_d = NULL;
	current = NULL;
	Npre = NULL;
	Npost = NULL;
	lastSpikeTime = NULL;
	postSynapticIds = NULL;
	postDelayInfo = NULL;
	wt = NULL;
	maxSynWt = NULL;
	wtChange = NULL;
	connIdFromSynId = NULL;
	synSpikeTime = NULL;
	spikeGenBits = NULL;
	firingTableD2 = NULL;
	firingTableD1 = NULL;

	fpParam = NULL;
	fpLog   = NULL;
	fpProgLog = stderr;
	fpTuningLog = NULL;
	cntTuning  = 0;
}

void CpuSNN::resetPoissonNeuron(unsigned int nid, int grpId) {
	assert(nid < numN);
	lastSpikeTime[nid]  = MAX_SIMULATION_TIME;
	avgFiring[nid]      = 0.0;

	if(grp_Info[grpId].WithSTP) {
		for (int j=0; j < STP_BUF_SIZE; j++) {
			int ind=STP_BUF_POS(nid,j);
			stpu[ind] = grp_Info[grpId].STP_U;
			stpx[ind] = 1;
		}
	}
}

void CpuSNN::resetPropogationBuffer() {
	pbuf->reset(0, 1023);
}

// resets nSpikeCnt[]
void CpuSNN::resetSpikeCnt(int my_grpId) {
	int startGrp, endGrp;

	if(!doneReorganization)
		return;

	if (my_grpId == -1) {
		startGrp = 0;
		endGrp   = numGrp;
	}
	else {
		startGrp = my_grpId;
		endGrp   = my_grpId+numConfig;
	}
  
	for( int grpId=startGrp; grpId < endGrp; grpId++) {
		int startN = grp_Info[grpId].StartN;
		int endN   = grp_Info[grpId].EndN+1;
		for (int i=startN; i < endN; i++)
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
		fprintf(stdout, "Grp: %d:%s s=%d e=%d %s\n", destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
					grp_Info[destGrp].EndN,  updateStr);
		fprintf(fpLog,  "Grp: %d:%s s=%d e=%d  %s\n",  destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN,
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
				int srcGrp   = findGrpId(preId);
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
						fprintf(stdout, "\t%d (%s) start=%d, type=%s maxWts = %f %s\n", srcGrp,
						grp_Info2[srcGrp].Name.c_str(), j, (j<Npre_plastic[nid]?"P":"F"), connInfo->maxWt, updateStr);
						fprintf(fpLog, "\t%d (%s) start=%d, type=%s maxWts = %f %s\n", srcGrp,
						grp_Info2[srcGrp].Name.c_str(), j, (j<Npre_plastic[nid]?"P":"F"), connInfo->maxWt, updateStr);
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
	memset(timeTableD2, 0, sizeof(int)*(1000+D+1));
	memset(timeTableD1, 0, sizeof(int)*(1000+D+1));
}



//! set one specific connection from neuron id 'src' to neuron id 'dest'
inline void CpuSNN::setConnection(int srcGrp,  int destGrp,  unsigned int src, unsigned int dest, float synWt,
									float maxWt, uint8_t dVal, int connProp) {
	assert(dest<=CONN_SYN_NEURON_MASK);			// total number of neurons is less than 1 million within a GPU
	assert((dVal >=1) && (dVal <= D));

	// we have exceeded the number of possible connection for one neuron
	if(Npost[src] >= grp_Info[srcGrp].numPostSynapses)	{
		fprintf(stderr, "setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)\n", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		fprintf(stderr, "(Npost[%d] = %d ) >= (numPostSynapses = %d) value given for the network very less\n", src,
					Npost[src], grp_Info[srcGrp].numPostSynapses);
		fprintf(stderr, "Large number of postsynaptic connections is established\n");
		fprintf(stderr, "Increase the numPostSynapses value for the Group = %s \n", grp_Info2[srcGrp].Name.c_str());
		assert(0);
	}

	if(Npre[dest] >= grp_Info[destGrp].numPreSynapses) {
		fprintf(stderr, "setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)\n", src, grp_Info2[srcGrp].Name.c_str(),
					dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
		fprintf(stderr, "(Npre[%d] = %d) >= (numPreSynapses = %d) value given for the network very less\n", dest,
					Npre[dest], grp_Info[destGrp].numPreSynapses);
		fprintf(stderr, "Large number of presynaptic connections established\n");
		fprintf(stderr, "Increase the numPostSynapses for the Grp = %s value \n", grp_Info2[destGrp].Name.c_str());
		assert(0);
	}

	int p = Npost[src];

	assert(Npost[src] >= 0);
	assert(Npre[dest] >= 0);
	assert((src*numPostSynapses+p)/numN < numPostSynapses); // divide by numN to prevent INT overflow

	unsigned int post_pos = cumulativePost[src] + Npost[src];
	unsigned int pre_pos  = cumulativePre[dest] + Npre[dest];

	assert(post_pos < postSynCnt);
	assert(pre_pos  < preSynCnt);

	//generate a new postSynapticIds id for the current connection
	postSynapticIds[post_pos]   = SET_CONN_ID(dest, Npre[dest], destGrp);
	tmp_SynapticDelay[post_pos] = dVal;

	preSynapticIds[pre_pos] 	= SET_CONN_ID(src, Npost[src], srcGrp);
	wt[pre_pos] 	  = synWt;
	maxSynWt[pre_pos] = maxWt;

	bool synWtType = GET_FIXED_PLASTIC(connProp);

	if (synWtType == SYN_PLASTIC) {
		sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
		Npre_plastic[dest]++;
		// homeostasis
		if (grp_Info[destGrp].WithHomeostasis && grp_Info[destGrp].homeoId ==-1)
			grp_Info[destGrp].homeoId = dest; // this neuron info will be printed
	}

	Npre[dest]+=1;
	Npost[src]+=1;

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

// reorganize the network and do the necessary allocation
// of all variable for carrying out the simulation..
// this code is run only one time during network initialization
void CpuSNN::setupNetwork(int simType, int ithGPU, bool removeTempMem) {
	if(!doneReorganization)
		reorganizeNetwork(removeTempMem, simType);

	if((simType == GPU_MODE) && (cpu_gpuNetPtrs.allocated == false))
		allocateSNN_GPU(ithGPU);
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
  fprintf(stderr, "Maximum Simulation Time Reached...Resetting simulation time\n");

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

void CpuSNN::updateParameters(int* curN, int* numPostSynapses, int* numPreSynapses, int nConfig) {
	assert(nConfig > 0);
	numNExcPois = 0; numNInhPois = 0; numNExcReg = 0; numNInhReg = 0;
	*numPostSynapses   = 0; *numPreSynapses = 0;

	//  scan all the groups and find the required information
	//  about the group (numN, numPostSynapses, numPreSynapses and others).
	for(int g=0; g < numGrp; g++)  {
		if (grp_Info[g].Type==UNKNOWN_NEURON) {
			fprintf(stderr, "Unknown group for %d (%s)\n", g, grp_Info2[g].Name.c_str());
			exitSimulation(1);
		}

		if (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type&POISSON_NEURON))
			numNInhReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type&POISSON_NEURON))
			numNExcReg += grp_Info[g].SizeN;
		else if (IS_EXCITATORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type&POISSON_NEURON))
			numNExcPois += grp_Info[g].SizeN;
		else if (IS_INHIBITORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type&POISSON_NEURON))
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
	numNReg   = numNExcReg +numNInhReg;
}

void CpuSNN::updateSpikesFromGrp(int grpId)
{
  assert(grp_Info[grpId].isSpikeGenerator==true);

  bool done;
  //static FILE* fp = fopen("spikes.txt", "w");
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
    if(currentMode == GPU_MODE)
      return;
#endif

    generateSpikesFromRate(grpId);
  }
}

void CpuSNN::updateSpikeGenerators() {
	for(int g=0; (g < numGrp); g++) {
		if (grp_Info[g].isSpikeGenerator) {
			// This evaluation is done to check if its time to get new set of spikes..
			if(((simTime-grp_Info[g].SliceUpdateTime) >= (unsigned) grp_Info[g].CurrTimeSlice))
				updateSpikesFromGrp(g);
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
			updateSpikesFromGrp(g);
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

	for(int g=0; g < numGrp; g++) {
		if ( grp_Info[g].MaxDelay == 1)
			maxSpikesD1 += (grp_Info[g].SizeN*grp_Info[g].MaxFiringRate);
		else
			maxSpikesD2 += (grp_Info[g].SizeN*grp_Info[g].MaxFiringRate);
	}

	if ((maxSpikesD1+maxSpikesD2) < (numNExcReg+numNInhReg+numNPois)*UNKNOWN_NEURON_MAX_FIRING_RATE) {
		fprintf(stderr, "Insufficient amount of buffer allocated...\n");
		exitSimulation(1);
	}

	firingTableD2 	    = new unsigned int[maxSpikesD2];
	firingTableD1 	    = new unsigned int[maxSpikesD1];
	cpuSnnSz.spikingInfoSize    += sizeof(int)*((maxSpikesD2+maxSpikesD1) + 2*(1000+D+1));

	return curD;
}

// This function is called every second by simulator...
// This function updates the firingTable by removing older firing values...
// and also update the synaptic weights from its derivatives..
void CpuSNN::updateStateAndFiringTable()
{
  // Read the neuron ids that fired in the last D seconds
  // and put it to the beginning of the firing table...
  for(int p=timeTableD2[999],k=0;p<timeTableD2[999+D+1];p++,k++) {
    firingTableD2[k]=firingTableD2[p];
  }

  for(int i=0; i < D; i++) {
    timeTableD2[i+1] = timeTableD2[1000+i+1]-timeTableD2[1000];
  }

  timeTableD1[D] = 0;

  // update synaptic weights here for all the neurons..
  for(int g=0; g < numGrp; g++) {
    // no changable weights so continue without changing..
    if(grp_Info[g].FixedInputWts || !(grp_Info[g].WithSTDP)) {
      //				for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++)
      //					nSpikeCnt[i]=0;
      continue;
    }

    for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
      ///nSpikeCnt[i] = 0;
      assert(i < numNReg);
      unsigned int offset = cumulativePre[i];
      float diff_firing  = 0.0;
      float homeostasisScale = 1.0;
      
      if(grp_Info[g].WithHomeostasis) {
		assert(baseFiring[i]>0);
		diff_firing = 1-avgFiring[i]/baseFiring[i];
		homeostasisScale = grp_Info[g].homeostasisScale;
      }

      if ((showLogMode >= 1) && (i==grp_Info[g].StartN))
	fprintf(fpProgLog,"Weights, Change at %lu (diff_firing:%f) \n", simTimeSec, diff_firing);

      for(int j=0; j < Npre_plastic[i]; j++) {

	if ((showLogMode >= 1) && (i==grp_Info[g].StartN))
	  fprintf(fpProgLog,"%1.2f %1.2f \t", wt[offset+j]*10, wtChange[offset+j]*10);
	// homeostatic weight update
	if(grp_Info[g].WithHomeostasis) {
	  if ((showLogMode >= 3) && (i==grp_Info[g].StartN)) 
	    fprintf(fpProgLog,"%f\t", (diff_firing*(0.0+wt[offset+j]) + wtChange[offset+j])/10/(Npre_plastic[i]+10)/(grp_Info[g].avgTimeScale*2/1000.0)*baseFiring[i]/(1+fabs(diff_firing)*50));
	  //need to figure out exactly why we change the weight to this value.  Specifically, what is with the second term?  -- KDC
	  wt[offset+j] += (diff_firing*wt[offset+j]*homeostasisScale + wtChange[offset+j])*baseFiring[i]/grp_Info[g].avgTimeScale/(1+fabs(diff_firing)*50);
	} else{
	  // just STDP weight update
	  wt[offset+j] += wtChange[offset+j]; 
	  // STDP weight update that is biased towards learning
	  //wt[offset+j] += (wtChange[offset+j]+0.1f);
	}

	//MDR - don't decay weights, just set to 0
	//wtChange[offset+j]*=0.99f;
	wtChange[offset+j] = 0;

	// if this is an excitatory or inhibitory synapse
	if (maxSynWt[offset+j] >= 0) {
	  if (wt[offset+j]>=maxSynWt[offset+j])
	    wt[offset+j] = maxSynWt[offset+j];
	  if (wt[offset+j]<0)
	    wt[offset+j] = 0.0;
	} else {
	  if (wt[offset+j]<=maxSynWt[offset+j])
	    wt[offset+j] = maxSynWt[offset+j];
	  if (wt[offset+j]>0)
	    wt[offset+j] = 0.0;
	}
      }

      if ((showLogMode >= 1) && (i==grp_Info[g].StartN))
	fprintf(fpProgLog,"\n");
    }
  }

  spikeCountAll	+= spikeCountAll1sec;
  spikeCountD2Host += (secD2fireCntHost-timeTableD2[D]);
  spikeCountD1Host += secD1fireCntHost;

  secD1fireCntHost  = 0;
  spikeCountAll1sec = 0;
  secD2fireCntHost  = timeTableD2[D];

  for(int i=0; i < numGrp; i++) {
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



void CpuSNN::updateSpikeMonitor()
{
  // don't continue if numSpikeMonitor is zero
  if(numSpikeMonitor==0)
    return;

  bool bufferOverFlow[MAX_GRP_PER_SNN];
  memset(bufferOverFlow,0,sizeof(bufferOverFlow));

  /* Reset buffer time counter */
  for(int i=0; i < numSpikeMonitor; i++)
    memset(monBufferTimeCnt[i],0,sizeof(int)*(1000));

  /* Reset buffer position */
  memset(monBufferPos,0,sizeof(int)*numSpikeMonitor);

  if(currentMode == GPU_MODE) {
    updateSpikeMonitor_GPU();
  }

  /* Read one spike at a time from the buffer and
     put the spikes to an appopriate monitor buffer.
     Later the user may need need to dump these spikes
     to an output file */
  for(int k=0; k < 2; k++) {
    unsigned int* timeTablePtr = (k==0)?timeTableD2:timeTableD1;
    unsigned int* fireTablePtr = (k==0)?firingTableD2:firingTableD1;
    for(int t=0; t < 1000; t++) {
      for(int i=timeTablePtr[t+D]; i<timeTablePtr[t+D+1];i++) {
	/* retrieve the neuron id */
	int nid   = fireTablePtr[i];
	if (currentMode == GPU_MODE)
	  nid = GET_FIRING_TABLE_NID(nid);
	//fprintf(fpLog, "%d %d \n", t, nid);
	assert(nid < numN);
	  
	int grpId = findGrpId(nid);
	int monitorId = grp_Info[grpId].MonitorId;
	if(monitorId!= -1) {
	    assert(nid >= grp_Info[grpId].StartN);
	    assert(nid <= grp_Info[grpId].EndN);
	    int   pos   = monBufferPos[monitorId];
	    if((pos >= monBufferSize[monitorId]))
	      {
		if(!bufferOverFlow[monitorId])
		  fprintf(stderr, "Buffer Monitor size (%d) is small. Increase buffer firing rate for %s\n",
		  			monBufferSize[monitorId], grp_Info2[grpId].Name.c_str());
		bufferOverFlow[monitorId] = true;
	      }
	    else {
	      monBufferPos[monitorId]++;
	      monBufferFiring[monitorId][pos] = nid-grp_Info[grpId].StartN; // store the Neuron ID relative to the start of the group
	      // we store the total firing at time t...
	      monBufferTimeCnt[monitorId][t]++;
	    }
	} /* if monitoring is enabled for this spike */
      } /* for all spikes happening at time t */
    }  /* for all time t */
  }

  for (int grpId=0;grpId<numGrp;grpId++) {
    int monitorId = grp_Info[grpId].MonitorId;
    if(monitorId!= -1) {
      fprintf(stderr, "Spike Monitor for Group %s has %d spikes (%f Hz)\n",grp_Info2[grpId].Name.c_str(),
      			monBufferPos[monitorId],((float)monBufferPos[monitorId])/(grp_Info[grpId].SizeN));

      // call the callback function
      if (monBufferCallback[monitorId])
	monBufferCallback[monitorId]->update(this,grpId,monBufferFiring[monitorId],monBufferTimeCnt[monitorId]);
    }
  }
}