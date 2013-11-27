/*
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
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
 * Ver 07/13/2013
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

	// includes for mkdir
	#if defined(CREATE_SPIKEDIR_IF_NOT_EXISTS)
		#include <sys/stat.h>
		#include <errno.h>
		#include <libgen.h>
	#endif


/*********************************************/

	void CpuSNN::resetPointers()
	{
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
		//stdpChanged = NULL;
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

	void CpuSNN::resetCurrent()
	{
		assert(current != NULL);
		memset(current, 0, sizeof(float)*numNReg);
	}

	void CpuSNN::resetCounters()
	{
		assert(numNReg <= numN);
		memset( curSpike, 0, sizeof(bool)*numN);
	}

	void CpuSNN::resetConductances()
	{
		if (sim_with_conductances) {
			assert(gAMPA != NULL);
			memset(gAMPA, 0, sizeof(float)*numNReg);
			memset(gNMDA, 0, sizeof(float)*numNReg);
			memset(gGABAa, 0, sizeof(float)*numNReg);
			memset(gGABAb, 0, sizeof(float)*numNReg);
		}
	}

	void CpuSNN::resetTimingTable()
	{
		memset(timeTableD2, 0, sizeof(int)*(1000+D+1));
		memset(timeTableD1, 0, sizeof(int)*(1000+D+1));
	}

	void CpuSNN::CpuSNNInit(unsigned int _numN, unsigned int _numPostSynapses, unsigned int _numPreSynapses, unsigned int _D)
	{
		numN = _numN;
		numPostSynapses = _numPostSynapses;
		D = _D;
		numPreSynapses = _numPreSynapses;

		voltage	 = new float[numNReg];
		recovery    = new float[numNReg];
		Izh_a	 = new float[numNReg];
		Izh_b    = new float[numNReg];
		Izh_c	 = new float[numNReg];
		Izh_d	 = new float[numNReg];
		current	 = new float[numNReg];
		cpuSnnSz.neuronInfoSize += (sizeof(int)*numNReg*12);

		if (sim_with_conductances) {
			for (int g=0;g<numGrp;g++)
				if (!grp_Info[g].WithConductances && ((grp_Info[g].Type&POISSON_NEURON)==0)) {
				printf("If one group enables conductances then all groups, except for generators, must enable conductances.  Group '%s' is not enabled.\n", grp_Info2[g].Name.c_str());
				assert(false);
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
		intrinsicWeight  = new float[numN];
		memset(intrinsicWeight,0,sizeof(float)*numN);
		cpuSnnSz.neuronInfoSize += (sizeof(int)*numN*2+sizeof(bool)*numN);

		if (sim_with_stp) {
			stpu = new float[numN*STP_BUF_SIZE];
			stpx = new float[numN*STP_BUF_SIZE];
			for (int i=0; i < numN*STP_BUF_SIZE; i++) {
				//MDR this should be set later.. (adding a default value)
				stpu[i] = 1;
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
		cpuSnnSz.networkInfoSize += ((sizeof(post_info_t)+sizeof(uint8_t))*postSynCnt+100) + (sizeof(delay_info_t)*numN*(D+1));
		assert(preSynCnt/numN <= numPreSynapses); // divide by numN to prevent INT overflow

		wt  			= new float[preSynCnt+100];
		maxSynWt     	= new float[preSynCnt+100];
		//! Temporary array to hold pre-syn connections. will be deleted later if necessary
		preSynapticIds	= new post_info_t[preSynCnt+100];
		// size due to weights and maximum weights
		cpuSnnSz.synapticInfoSize += ((2*sizeof(float)+sizeof(post_info_t))*(preSynCnt+100));

		timeTableD2  = new unsigned int[1000+D+1];
		timeTableD1  = new unsigned int[1000+D+1];
		resetTimingTable();
		cpuSnnSz.spikingInfoSize += sizeof(int)*2*(1000+D+1);

		// random thalamic current included...
//		randNeuronId = new int[numN];
//		cpuSnnSz.addInfoSize += (sizeof(int)*numN);

		// poisson Firing Rate
		cpuSnnSz.neuronInfoSize += (sizeof(int)*numNPois);

		tmp_SynapseMatrix_fixed = NULL;
		tmp_SynapseMatrix_plastic = NULL;
	}

	void CpuSNN::makePtrInfo()
	{
		// create the CPU Net Ptrs..
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
		cpuNetPtrs.nSpikeCnt		= nSpikeCnt;
		cpuNetPtrs.curSpike 		= curSpike;
		cpuNetPtrs.firingTableD2 	= firingTableD2;
		cpuNetPtrs.firingTableD1 	= firingTableD1;
		cpuNetPtrs.gAMPA        	= gAMPA;
		cpuNetPtrs.gNMDA			= gNMDA;
		cpuNetPtrs.gGABAa       	= gGABAa;
		cpuNetPtrs.gGABAb			= gGABAb;
		cpuNetPtrs.allocated    	= true;
		cpuNetPtrs.memType      	= CPU_MODE;
		cpuNetPtrs.stpu 			= stpu;
		cpuNetPtrs.stpx				= stpx;
	}

	void CpuSNN::resetSpikeCnt(int my_grpId )
	{
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

	CpuSNN::CpuSNN(const string& _name, int _numConfig, int _randSeed, int _mode)
	{
		fprintf(stdout, "************************************************\n");
		fprintf(stdout, "***** GPU-SNN Simulation Begins Version %d.%d *** \n",MAJOR_VERSION,MINOR_VERSION);
		fprintf(stdout, "************************************************\n");

		// initialize propogated spike buffers.....
		pbuf = new PropagatedSpikeBuffer(0, PROPOGATED_BUFFER_SIZE);

		numConfig 			  = _numConfig;
		finishedPoissonGroup  = false;
		assert(numConfig > 0);
		assert(numConfig < 100);

		resetPointers();
		numN = 0; numPostSynapses = 0; D = 0;
		memset(&cpuSnnSz, 0, sizeof(cpuSnnSz));
		enableSimLogs = false;
		simLogDirName = "logs";//strdup("logs");

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

#ifndef VIEW_DOTTY
  #define VIEW_DOTTY false
#endif
		showDotty   = VIEW_DOTTY;
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

/*			grp_Info[i].STP_U  = STP_U_Exc;
			grp_Info[i].STP_tD = STP_tD_Exc;
			grp_Info[i].STP_tF = STP_tF_Exc;
*/

			grp_Info[i].dAMPA=1-(1.0/5);
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
		numProbe 	 = 0;
		neuronProbe  = NULL;

		simTimeMs	 = 0;	simTimeSec		 = 0;   simTime = 0;
		spikeCountAll1sec = 0;  secD1fireCntHost  = 0;	secD2fireCntHost  = 0;
		spikeCountAll     = 0;	spikeCountD2Host = 0;	spikeCountD1Host = 0;
		nPoissonSpikes     = 0;

		networkName	= _name; //new char[sizeof(_name)];
		//strcpy(networkName, _name);
		numGrp   = 0;
		numConnections = 0;
		numSpikeGenGrps  = 0;
		NgenFunc = 0;
//		numNoise = 0;
//		numRandNeurons = 0;
		simulatorDeleted = false;

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

	void CpuSNN::deleteObjects()
	{
		try
		{

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

//			if(val==0)
//				saveConnectionWeights();


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
			deleteObjectsGPU();

			CUDA_DELETE_TIMER(timer);

			simulatorDeleted = true;
		}
		catch(...)
		{
			fprintf(stderr, "Unknow exception ...\n");
		}
	}

	void CpuSNN::exitSimulation(int val)
	{
		deleteObjects();
		exit(val);
	}

	CpuSNN::~CpuSNN()
	{
		if (!simulatorDeleted)
			deleteObjects();
	}

	void CpuSNN::setSTDP( int grpId, bool enable, int configId)
	{
		assert(enable==false);
		setSTDP(grpId,false,0,0,0,0,configId);
	}

	void CpuSNN::setSTDP( int grpId, bool enable, float ALPHA_LTP, float TAU_LTP, float ALPHA_LTD, float TAU_LTD, int configId)
	{
		assert(TAU_LTP >= 0.0);
		assert(TAU_LTD >= 0.0);

		if (grpId == ALL && configId == ALL) {
			for(int g=0; g < numGrp; g++)
				setSTDP(g, enable, ALPHA_LTP,TAU_LTP,ALPHA_LTD,TAU_LTD, 0);
		} else if (grpId == ALL) {
			for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
				int g = getGroupId(grpId1, configId);
				setSTDP(g, enable, ALPHA_LTP,TAU_LTP,ALPHA_LTD,TAU_LTD, configId);
			}
		} else if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setSTDP(grpId, enable, ALPHA_LTP,TAU_LTP,ALPHA_LTD,TAU_LTD, c);
		} else {
			int cGrpId = getGroupId(grpId, configId);

			sim_with_stdp |= enable;
			
			grp_Info[cGrpId].WithSTDP      = enable;
			grp_Info[cGrpId].ALPHA_LTP     = ALPHA_LTP;
			grp_Info[cGrpId].ALPHA_LTD     = ALPHA_LTD;
			grp_Info[cGrpId].TAU_LTP_INV   = 1.0/TAU_LTP;
			grp_Info[cGrpId].TAU_LTD_INV   = 1.0/TAU_LTD;

			grp_Info[cGrpId].newUpdates   = true;

			fprintf(stderr, "STDP %s for %d (%s): %f, %f, %f, %f\n", enable?"enabled":"disabled",
					cGrpId, grp_Info2[cGrpId].Name.c_str(), ALPHA_LTP, ALPHA_LTD, TAU_LTP, TAU_LTD);
		}
	}

	void CpuSNN::setSTP( int grpId, bool enable, int configId)
	{
		assert(enable==false);
		setSTP(grpId,false,0,0,0,configId);
	}

	void CpuSNN::setSTP(int grpId, bool enable, float STP_U, float STP_tD, float STP_tF, int configId)
	{
		if (grpId == ALL && configId == ALL) {
			for(int g=0; g < numGrp; g++)
					setSTP(g, enable, STP_U, STP_tD, STP_tF, 0);
		} else if (grpId == ALL) {
			for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
				int g = getGroupId(grpId1, configId);
				setSTP(g, enable, STP_U, STP_tD, STP_tF, configId);
			}
		} else if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setSTP(grpId, enable, STP_U, STP_tD, STP_tF, c);
		} else {
			int cGrpId = getGroupId(grpId, configId);

			sim_with_stp |= enable;
			
			grp_Info[cGrpId].WithSTP     = enable;
			grp_Info[cGrpId].STP_U=STP_U;
			grp_Info[cGrpId].STP_tD=STP_tD;
			grp_Info[cGrpId].STP_tF=STP_tF;

			grp_Info[cGrpId].newUpdates = true;

			fprintf(stderr, "STP %s for %d (%s): %f, %f, %f\n", enable?"enabled":"disabled",
					cGrpId, grp_Info2[cGrpId].Name.c_str(), STP_U, STP_tD, STP_tF);
		}
	}

	void CpuSNN::setConductances( int grpId, bool enable, int configId)
	{
		assert(enable==false);
		setConductances(grpId,false,0,0,0,0,configId);
	}

	void CpuSNN::setConductances(int grpId, bool enable, float tAMPA, float tNMDA, float tGABAa, float tGABAb, int configId)
	{
		if (grpId == ALL && configId == ALL) {
			for(int g=0; g < numGrp; g++)
				setConductances(g, enable, tAMPA, tNMDA, tGABAa, tGABAb, 0);
		} else if (grpId == ALL) {
			for(int grpId1=0; grpId1 < numGrp; grpId1 += numConfig) {
				int g = getGroupId(grpId1, configId);
				setConductances(g, enable, tAMPA, tNMDA, tGABAa, tGABAb, configId);
			}
		} else if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setConductances(grpId, enable, tAMPA, tNMDA, tGABAa, tGABAb, c);
		} else {
			int cGrpId = getGroupId(grpId, configId);

			sim_with_conductances |= enable;

			grp_Info[cGrpId].WithConductances     = enable;

			grp_Info[cGrpId].dAMPA=1-(1.0/tAMPA);
			grp_Info[cGrpId].dNMDA=1-(1.0/tNMDA);
			grp_Info[cGrpId].dGABAa=1-(1.0/tGABAa);
			grp_Info[cGrpId].dGABAb=1-(1.0/tGABAb);

			grp_Info[cGrpId].newUpdates = true;

			fprintf(stderr, "Conductances %s for %d (%s): %f, %f, %f, %f\n", enable?"enabled":"disabled",
					cGrpId, grp_Info2[cGrpId].Name.c_str(), tAMPA, tNMDA, tGABAa, tGABAb);
		}
	}

	int CpuSNN::createGroup(const string& _name, unsigned int _numN, int _nType, int configId)
	{
		if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				createGroup(_name, _numN, _nType, c);
			return (numGrp-numConfig);
		} else {
			assert(numGrp < MAX_GRP_PER_SNN);

			if ( (!(_nType&TARGET_AMPA) && !(_nType&TARGET_NMDA) &&
			      !(_nType&TARGET_GABAa) && !(_nType&TARGET_GABAb)) || (_nType&POISSON_NEURON)) {
				fprintf(stderr, "Invalid type using createGroup...\n");
				fprintf(stderr, "can not create poisson generators here...\n");
				exitSimulation(1);
			}

			grp_Info[numGrp].SizeN  	= _numN;
			grp_Info[numGrp].Type   	= _nType;
			grp_Info[numGrp].WithConductances	= false;
			grp_Info[numGrp].WithSTP		= false;
			grp_Info[numGrp].WithSTDP		= false;
			if ( (_nType&TARGET_GABAa) || (_nType&TARGET_GABAb)) {
				grp_Info[numGrp].MaxFiringRate 	= INHIBITORY_NEURON_MAX_FIRING_RATE;
			}
			else {
				grp_Info[numGrp].MaxFiringRate 	= EXCITATORY_NEURON_MAX_FIRING_RATE;
			}
			grp_Info2[numGrp].ConfigId		= configId;
			grp_Info2[numGrp].Name  		= _name;//new char[strlen(_name)];
			grp_Info[numGrp].isSpikeGenerator	= false;
			grp_Info[numGrp].MaxDelay		= 1;

			grp_Info2[numGrp].IzhGen = NULL;
			grp_Info2[numGrp].Izh_a = -1;

			std::stringstream outStr;
			outStr << configId;

			if (configId == 0)
				grp_Info2[numGrp].Name = _name;
			else
				grp_Info2[numGrp].Name = _name + "_" + outStr.str();

			finishedPoissonGroup = true;

			numGrp++;

			return (numGrp-1);
		}
	}

	int CpuSNN::createSpikeGeneratorGroup(const string& _name, unsigned int size_n, int type, int configId)
	{
		if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				createSpikeGeneratorGroup(_name, size_n, type, c);
			return (numGrp-numConfig);
		} else {
			grp_Info[numGrp].SizeN   		= size_n;
			grp_Info[numGrp].Type    		= type | POISSON_NEURON;
			grp_Info[numGrp].WithConductances	= false;
			grp_Info[numGrp].WithSTP		= false;
			grp_Info[numGrp].WithSTDP		= false;
			grp_Info[numGrp].isSpikeGenerator	= true;		// these belong to the spike generator class...
			grp_Info2[numGrp].ConfigId		= configId;
			grp_Info2[numGrp].Name    		= _name; //new char[strlen(_name)];
			grp_Info[numGrp].MaxFiringRate 	= POISSON_MAX_FIRING_RATE;
			std::stringstream outStr ;
			outStr << configId;

			if (configId != 0)
				grp_Info2[numGrp].Name = _name + outStr.str();

			numGrp++;
			numSpikeGenGrps++;

			return (numGrp-1);
		}
	}

	int  CpuSNN::getGroupId(int groupId, int configId)
	{
		assert(configId < numConfig);
		int cGrpId = (groupId+configId);
		assert(cGrpId  < numGrp);
		return cGrpId;
	}

	int  CpuSNN::getConnectionId(int connId, int configId)
	{
		if(configId >= numConfig) {
			fprintf(stderr, "getConnectionId(int, int): Assertion `configId(%d) < numConfig(%d)' failed\n", configId, numConfig);
			assert(0);
		}
		connId = connId+configId;
		if (connId  >= numConnections) {
			fprintf(stderr, "getConnectionId(int, int): Assertion `connId(%d) < numConnections(%d)' failed\n", connId, numConnections);
			assert(0);
		}
		return connId;
	}

	void CpuSNN::setNeuronParameters(int groupId, float _a, float _b, float _c, float _d, int configId)
	{
		setNeuronParameters(groupId, _a, 0, _b, 0, _c, 0, _d, 0, configId);
	}

	void CpuSNN::setNeuronParameters(int groupId, float _a, float a_sd, float _b, float b_sd, float _c, float c_sd, float _d, float d_sd, int configId)
	{
		if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setNeuronParameters(groupId, _a, a_sd, _b, b_sd, _c, c_sd, _d, d_sd, c);
		} else {
			int cGrpId = getGroupId(groupId, configId);
			grp_Info2[cGrpId].Izh_a	  	=   _a;
			grp_Info2[cGrpId].Izh_a_sd  	=   a_sd;
			grp_Info2[cGrpId].Izh_b	  	=   _b;
			grp_Info2[cGrpId].Izh_b_sd  	=   b_sd;
			grp_Info2[cGrpId].Izh_c		=   _c;
			grp_Info2[cGrpId].Izh_c_sd	=   c_sd;
			grp_Info2[cGrpId].Izh_d		=   _d;
			grp_Info2[cGrpId].Izh_d_sd	=   d_sd;
		}
	}

	void CpuSNN::setNeuronParameters(int groupId, IzhGenerator* IzhGen, int configId)
	{
		if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setNeuronParameters(groupId, IzhGen, c);
		} else {
			int cGrpId = getGroupId(groupId, configId);

			grp_Info2[cGrpId].IzhGen	=   IzhGen;
		}
	}

	void CpuSNN::setGroupInfo(int grpId, group_info_t info, int configId)
	{
		if (configId == ALL) {
			for(int c=0; c < numConfig; c++)
				setGroupInfo(grpId, info, c);
		} else {
			int cGrpId = getGroupId(grpId, configId);
			grp_Info[cGrpId] = info;
		}
	}

	group_info_t CpuSNN::getGroupInfo(int grpId, int configId)
	{
		int cGrpId = getGroupId(grpId, configId);
		return grp_Info[cGrpId];
	}

	void CpuSNN::buildPoissonGroup(int grpId)
	{
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

	void CpuSNN::resetPoissonNeuron(unsigned int nid, int grpId)
	{
		assert(nid < numN);
		lastSpikeTime[nid]  = MAX_SIMULATION_TIME;

		if(grp_Info[grpId].WithSTP) {
			for (int j=0; j < STP_BUF_SIZE; j++) {
				int ind=STP_BUF_POS(nid,j);
				stpu[ind] = grp_Info[grpId].STP_U;
				stpx[ind] = 1;
			}
		}
	}

	void CpuSNN::resetNeuron(unsigned int nid, int grpId)
	{
		assert(nid < numNReg);
		if (grp_Info2[grpId].IzhGen == NULL) {
			if (grp_Info2[grpId].Izh_a == -1) {
				printf("setNeuronParameters much be called for group %s (%d)\n",grp_Info2[grpId].Name.c_str(),grpId);
				exit(-1);
			}

			Izh_a[nid] = grp_Info2[grpId].Izh_a + grp_Info2[grpId].Izh_a_sd*(float)getRandClosed();
			Izh_b[nid] = grp_Info2[grpId].Izh_b + grp_Info2[grpId].Izh_b_sd*(float)getRandClosed();
			Izh_c[nid] = grp_Info2[grpId].Izh_c + grp_Info2[grpId].Izh_c_sd*(float)getRandClosed();
			Izh_d[nid] = grp_Info2[grpId].Izh_d + grp_Info2[grpId].Izh_d_sd*(float)getRandClosed();
		} else {
			grp_Info2[grpId].IzhGen->set(this, grpId, nid, Izh_a[nid], Izh_b[nid], Izh_c[nid], Izh_d[nid]);
		}

		voltage[nid] = Izh_c[nid];	// initial values for new_v
		recovery[nid] = 0.2f*voltage[nid];   		// initial values for u

		lastSpikeTime[nid]  = MAX_SIMULATION_TIME;

		if(grp_Info[grpId].WithSTP) {
			for (int j=0; j < STP_BUF_SIZE; j++) {
				int ind=STP_BUF_POS(nid,j);
				stpu[ind] = grp_Info[grpId].STP_U;
				stpx[ind] = 1;
			}
		}
	}

	void CpuSNN::buildGroup(int grpId)
	{
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

	// set one specific connection from neuron id 'src' to neuron id 'dest'
	inline void CpuSNN::setConnection(int srcGrp,  int destGrp,  unsigned int src, unsigned int dest, float synWt, float maxWt, uint8_t dVal, int connProp)
	{
		assert(dest<=CONN_SYN_NEURON_MASK);			// total number of neurons is less than 1 million within a GPU
		assert((dVal >=1) && (dVal <= D));

		// we have exceeded the number of possible connection for one neuron
		if(Npost[src] >= grp_Info[srcGrp].numPostSynapses)	{
			fprintf(stderr, "setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)\n", src, grp_Info2[srcGrp].Name.c_str(), dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
			fprintf(stderr, "(Npost[%d] = %d ) >= (numPostSynapses = %d) value given for the network very less\n", src, Npost[src], grp_Info[srcGrp].numPostSynapses);
			fprintf(stderr, "Large number of postsynaptic connections is established\n");
			fprintf(stderr, "Increase the numPostSynapses value for the Group = %s \n", grp_Info2[srcGrp].Name.c_str());
			assert(0);
		}

		if(Npre[dest] >= grp_Info[destGrp].numPreSynapses) {
			fprintf(stderr, "setConnection(%d (Grp=%s), %d (Grp=%s), %f, %d)\n", src, grp_Info2[srcGrp].Name.c_str(), dest, grp_Info2[destGrp].Name.c_str(), synWt, dVal);
			fprintf(stderr, "(Npre[%d] = %d) >= (numPreSynapses = %d) value given for the network very less\n", dest, Npre[dest], grp_Info[destGrp].numPreSynapses);
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

		postSynapticIds[post_pos]   = SET_CONN_ID(dest, Npre[dest], destGrp); //generate a new postSynapticIds id for the current connection
		tmp_SynapticDelay[post_pos] = dVal;

		preSynapticIds[pre_pos] 	= SET_CONN_ID(src, Npost[src], srcGrp);
		wt[pre_pos] 	  = synWt;
		maxSynWt[pre_pos] = maxWt;

		bool synWtType = GET_FIXED_PLASTIC(connProp);

		if (synWtType == SYN_PLASTIC) {
			sim_with_fixedwts = false; // if network has any plastic synapses at all, this will be set to true
			Npre_plastic[dest]++;
		}

		Npre[dest]+=1;
		Npost[src]+=1;

		grp_Info2[srcGrp].numPostConn++;
		grp_Info2[destGrp].numPreConn++;

		if (Npost[src] > grp_Info2[srcGrp].maxPostConn)
			grp_Info2[srcGrp].maxPostConn = Npost[src];

		if (Npre[dest] > grp_Info2[destGrp].maxPreConn)
			grp_Info2[destGrp].maxPreConn = Npre[src];

#if _DEBUG_
		//fprintf(fpLog, "setConnection(%d, %d, %f, %d Npost[%d]=%d, Npre[%d]=%d)\n", src, dest, initWt, dVal, src, Npost[src], dest, Npre[dest]);
#endif

	}

	float CpuSNN::getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId)
	{
		float actWts;
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

	// make 'C' random connections from grpSrc to grpDest
	void CpuSNN::connectRandom (grpConnectInfo_t* info)
	{
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

	void CpuSNN::connectOneToOne (grpConnectInfo_t* info)
	{
		int grpSrc = info->grpSrc;
		int grpDest = info->grpDest;
		assert( grp_Info[grpDest].SizeN == grp_Info[grpSrc].SizeN );
		// C = grp_Info[grpDest].SizeN;

		for(int nid=grp_Info[grpSrc].StartN,j=grp_Info[grpDest].StartN; nid<=grp_Info[grpSrc].EndN; nid++, j++)  {
				uint8_t dVal = info->minDelay + (int)(0.5+(getRandClosed()*(info->maxDelay-info->minDelay)));
				assert((dVal >= info->minDelay) && (dVal <= info->maxDelay));
				float synWt = getWeights(info->connProp, info->initWt, info->maxWt, nid, grpSrc);
				setConnection(grpSrc, grpDest, nid, j, synWt, info->maxWt, dVal, info->connProp);
				//setConnection(grpSrc, grpDest, nid, j, info->initWt, info->maxWt, dVal, info->connProp);
				info->numberOfConnections++;
		}

//		//dotty printf output
//		fprintf(fpDotty, "\t\tg%d -> g%d [style=%s, label=\"numPostSynapses=%d, wt=%3.3f , Dm=%d   \"]\n", grpSrc, grpDest, (info->initWt > 0)?"bold":"dotted", info->numPostSynapses, info->maxWt, info->maxDelay);
//		fprintf(stdout, "Creating One-to-One Connection from '%s' to '%s'\n", grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str());
//		fprintf(fpLog, "Creating One-to-One Connection from '%s' to '%s'\n", grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str());

		grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
		grp_Info2[grpDest].sumPreConn += info->numberOfConnections;

	}

	// user-defined functions called here...
	void CpuSNN::connectUserDefined (grpConnectInfo_t* info)
	{
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

//		// dotty printf output
//		fprintf(fpDotty, "\t\tg%d -> g%d [style=%s, label=\"user-defined\"]\n", grpSrc, grpDest, (info->initWt > 0)?"bold":"dotted");
//		fprintf(stdout, "Creating User-defined Connection from '%s' to '%s'\n", grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str());
//		fprintf(fpLog, "Creating User-defined Connection from '%s' to '%s'\n", grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str());

		grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
		grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
	}

	// make 'C' full connections from grpSrc to grpDest
	void CpuSNN::connectFull (grpConnectInfo_t* info)
	{
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

//		//dotty printf output
//		fprintf(fpDotty, "\t\tg%d -> g%d [style=%s, label=\"numPostSynapses=%d, wt=%3.3f, Dm=%d    \"]\n", grpSrc, grpDest, (info->initWt > 0)?"bold":"dotted", info->numPostSynapses, info->maxWt, info->maxDelay);
//		fprintf(stdout, "Creating Full Connection %s from '%s' to '%s' with Probability %f\n",
//			   (noDirect?"no-direct":" "), grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str(), info->numPostSynapses*1.0/grp_Info[grpDest].SizeN);
//		fprintf(fpLog, "Creating Full Connection %s from '%s' to '%s' with Probability %f\n",
//			   (noDirect?"no-direct":" "), grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str(), info->numPostSynapses*1.0/grp_Info[grpDest].SizeN);

		grp_Info2[grpSrc].sumPostConn += info->numberOfConnections;
		grp_Info2[grpDest].sumPreConn += info->numberOfConnections;
	}


	void CpuSNN::connectFromMatrix(SparseWeightDelayMatrix* mat, int connProp)
	{
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

	void CpuSNN::printDotty ()
	{
		string fname(networkName+".dot");
		fpDotty = fopen(fname.c_str(),"w");

		fprintf(fpDotty, "\
digraph G {\n\
\t\tnode [style=filled];\n\
\t\tcolor=blue;\n");

		for(int g=0; g < numGrp; g++) {
			//add a node to the dotty output
			char stype = grp_Info[g].Type;
			assert(grp_Info2[g].numPostConn == grp_Info2[g].sumPostConn);
			assert(grp_Info2[g].numPreConn == grp_Info2[g].sumPreConn);

			// fprintf(stdout, "Creating Spike Generator Group %s(id=%d) with numN=%d\n", grp_Info2[g].Name.c_str(), g, grp_Info[g].SizeN);
			// fprintf(fpLog, "Creating Spike Generator Group %s(id=%d) with numN=%d\n", grp_Info2[g].Name.c_str(), g, grp_Info[g].SizeN);
			// add a node to the dotty output
			fprintf (fpDotty, "\t\tg%d [%s label=\"id=%d:%s \\n numN=%d avgPost=%3.2f avgPre=%3.2f \\n maxPost=%d maxPre=%d\"];\n", g,
					(stype&POISSON_NEURON) ? "shape = box, ": " ", g, grp_Info2[g].Name.c_str(),
					grp_Info[g].SizeN, grp_Info2[g].numPostConn*1.0/grp_Info[g].SizeN,
					grp_Info2[g].numPreConn*1.0/grp_Info[g].SizeN, grp_Info2[g].maxPostConn, grp_Info2[g].maxPreConn);
			// fprintf(stdout, "Creating Group %s(id=%d) with numN=%d\n", grp_Info2[g].Name.c_str(), g, grp_Info[g].SizeN);
			// fprintf(fpLog, "Creating Group %s(id=%d) with numN=%d\n", grp_Info2[g].Name.c_str(), g, grp_Info[g].SizeN);
		}

/*
		for (int noiseId=0; noiseId < numNoise; noiseId++) {
			int groupId 		   = noiseGenGroup[noiseId].groupId;
			float currentStrength  = noiseGenGroup[noiseId].currentStrength;
			float neuronPercentage = noiseGenGroup[noiseId].neuronPercentage;
			int  numNeuron  = ((groupId==-1)? -1: grp_Info[groupId].SizeN);
			if(groupId==-1)
				fprintf(fpDotty, "\t\tr%d [shape=box, label=\"Global Random Noise\\n(I=%2.2fmA, frac=%2.2f%%)\"];\n", noiseId, currentStrength, neuronPercentage);
			else
				fprintf(fpDotty, "\t\tr%d [shape=box, label=\"Random Noise\\n(I=%2.2fmA, frac=%2.2f%%)\"];\n", noiseId, currentStrength, neuronPercentage);

			if (groupId !=- 1)
				fprintf(fpDotty, "\t\tr%d -> g%d [label=\" n=%d \"];\n", noiseId, groupId, (int) (numNeuron*(neuronPercentage/100.0)));
		}
*/

		grpConnectInfo_t* info = connectBegin;
		while(info) {
			int grpSrc = info->grpSrc;
			int grpDest = info->grpDest;

			float avgPostM = info->numberOfConnections/grp_Info[grpSrc].SizeN;
			float avgPreM  = info->numberOfConnections/grp_Info[grpDest].SizeN;
			//dotty printf output
			fprintf(fpDotty, "\t\tg%d -> g%d [style=%s, arrowType=%s, label=\"avgPost=%3.2f, avgPre=%3.2f, wt=%3.3f, maxD=%d \"]\n",
					grpSrc, grpDest, (info->initWt > 0)?"bold":"dotted", (info->initWt > 0)?"normal":"dot", avgPostM, avgPreM, info->maxWt, info->maxDelay);
//			fprintf(stdout, "Creating Connection from '%s' to '%s' with Probability %1.3f\n",
//					grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str(), avgPostM/grp_Info[grpDest].SizeN);
//			fprintf(fpLog, "Creating Connection from '%s' to '%s' with Probability %1.3f\n",
//					grp_Info2[grpSrc].Name.c_str(), grp_Info2[grpDest].Name.c_str(), avgPostM/grp_Info[grpDest].SizeN);
			info = info->next;
		}

		fprintf(fpDotty, "\n}\n");
		fclose(fpDotty);

		//std::stringstream cmd;
		//cmd  << "kgraphviewer " << networkName << ".dot";
		char cmd[100];
		int dbg = sprintf(cmd, "kgraphviewer %s.dot", networkName.c_str());
		int retVal;
		showDotty = false;
		if(showDotty) {
		  retVal = system(cmd);
		  assert(retVal >= 0);
		}

		fprintf(stdout, "\trun cmd to view graph: %s\n", cmd);
	}

	// make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	int CpuSNN::connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType, int maxM, int maxPreM)
	{
		int retId=-1;

		for(int c=0; c < numConfig; c++, grpId1++, grpId2++) {

			assert(grpId1 < numGrp);
			assert(grpId2 < numGrp);

			if (maxM == 0)
			   maxM = grp_Info[grpId2].SizeN;

			if (maxPreM == 0)
			   maxPreM = grp_Info[grpId1].SizeN;

			if (maxM > MAX_numPostSynapses) {
				printf("Connection from %s (%d) to %s (%d) exceeded the maximum number of output synapses (%d), has %d.\n",grp_Info2[grpId1].Name.c_str(),grpId1,grp_Info2[grpId2].Name.c_str(), grpId2, MAX_numPostSynapses,maxM);
				assert(maxM <= MAX_numPostSynapses);
			}

			if (maxPreM > MAX_numPreSynapses) {
				printf("Connection from %s (%d) to %s (%d) exceeded the maximum number of input synapses (%d), has %d.\n",grp_Info2[grpId1].Name.c_str(), grpId1,grp_Info2[grpId2].Name.c_str(), grpId2,MAX_numPreSynapses,maxPreM);
				assert(maxPreM <= MAX_numPreSynapses);
			}

			grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));

			newInfo->grpSrc   = grpId1;
			newInfo->grpDest  = grpId2;
			newInfo->initWt	  = 1;
			newInfo->maxWt	  = 1;
			newInfo->maxDelay = 1;
			newInfo->minDelay = 1;
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

			if (showLogMode >= 1)
				printf("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d\n",grpId1,grp_Info2[grpId1].Name.c_str(),grp_Info[grpId1].numPostSynapses,grpId2,grp_Info2[grpId2].Name.c_str(),grp_Info[grpId2].numPreSynapses);

			newInfo->connId	  = numConnections++;
			if(c==0)
				retId = newInfo->connId;
		}
		assert(retId != -1);
		return retId;
	}

	// make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
	int CpuSNN::connect(int grpId1, int grpId2, const string& _type, float initWt, float maxWt, float p, uint8_t minDelay, uint8_t maxDelay, bool synWtType, const string& wtType)
	{
		int retId=-1;
		for(int c=0; c < numConfig; c++, grpId1++, grpId2++) {
			assert(grpId1 < numGrp);
			assert(grpId2 < numGrp);
			assert(minDelay <= maxDelay);

			bool useRandWts = (wtType.find("random") != string::npos);
			bool useRampDownWts = (wtType.find("ramp-down") != string::npos);
			bool useRampUpWts = (wtType.find("ramp-up") != string::npos);
			uint32_t connProp = SET_INITWTS_RANDOM(useRandWts)
										| SET_CONN_PRESENT(1)
										| SET_FIXED_PLASTIC(synWtType)
										| SET_INITWTS_RAMPUP(useRampUpWts)
										| SET_INITWTS_RAMPDOWN(useRampDownWts);

			grpConnectInfo_t* newInfo = (grpConnectInfo_t*) calloc(1, sizeof(grpConnectInfo_t));
			newInfo->grpSrc   = grpId1;
			newInfo->grpDest  = grpId2;
			newInfo->initWt	  = initWt;
			newInfo->maxWt	  = maxWt;
			newInfo->maxDelay = maxDelay;
			newInfo->minDelay = minDelay;
			newInfo->connProp = connProp;
			newInfo->p = p;
			newInfo->type	  = CONN_UNKNOWN;
			newInfo->numPostSynapses	  	  = 1;

			newInfo->next     = connectBegin; //linked list of connection..
			connectBegin	= newInfo;


			if ( _type.find("random") != string::npos) {
				newInfo->type 	= CONN_RANDOM;
				newInfo->numPostSynapses	= MIN(grp_Info[grpId2].SizeN,((int) (p*grp_Info[grpId2].SizeN +5*sqrt(p*(1-p)*grp_Info[grpId2].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 5 stds.
				newInfo->numPreSynapses   = MIN(grp_Info[grpId1].SizeN,((int) (p*grp_Info[grpId1].SizeN +5*sqrt(p*(1-p)*grp_Info[grpId1].SizeN)+0.5))); // estimate the maximum number of connections we need.  This uses a binomial distribution at 5 stds.
			}
			else if ( _type.find("full") != string::npos) {
				newInfo->type 	= CONN_FULL;
				newInfo->numPostSynapses	= grp_Info[grpId2].SizeN;
				newInfo->numPreSynapses   = grp_Info[grpId1].SizeN;
			}
			else if ( _type.find("full-no-direct") != string::npos) {
				newInfo->type 	= CONN_FULL_NO_DIRECT;
				newInfo->numPostSynapses	= grp_Info[grpId2].SizeN-1;
				newInfo->numPreSynapses	= grp_Info[grpId1].SizeN-1;
			}
			else if ( _type.find("one-to-one") != string::npos) {
				newInfo->type 	= CONN_ONE_TO_ONE;
				newInfo->numPostSynapses	= 1;
				newInfo->numPreSynapses	= 1;
			}
			else {
				fprintf(stderr, "Invalid connection type (should be 'random', or 'full' or 'one-to-one' or 'full-no-direct')\n");
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

	int CpuSNN::updateSpikeTables()
	{
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

//		maxSpikesD1 = (maxSpikesD1 == 0)? 1 : maxSpikesD1;
//		maxSpikesD2 = (maxSpikesD2 == 0)? 1 : maxSpikesD2;

		if ((maxSpikesD1+maxSpikesD2) < (numNExcReg+numNInhReg+numNPois)*UNKNOWN_NEURON_MAX_FIRING_RATE) {
			fprintf(stderr, "Insufficient amount of buffer allocated...\n");
			exitSimulation(1);
		}

//		maxSpikesD2 = (maxSpikesD1 > maxSpikesD2)?maxSpikesD1:maxSpikesD2;
//		maxSpikesD1 = (maxSpikesD1 > maxSpikesD2)?maxSpikesD1:maxSpikesD2;

		firingTableD2 	    = new unsigned int[maxSpikesD2];
		firingTableD1 	    = new unsigned int[maxSpikesD1];
		cpuSnnSz.spikingInfoSize    += sizeof(int)*((maxSpikesD2+maxSpikesD1) + 2*(1000+D+1));

		return curD;
	}

	void CpuSNN::buildNetwork()
	{
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
		fprintf(stdout, "numN = %d, numPostSynapses = %d, numPreSynapses = %d, D = %d\n", curN, numPostSynapses, numPreSynapses, curD);
		fprintf(stdout, "**********************************\n");

		fprintf(fpLog, "**********************************\n");
		fprintf(fpLog, "numN = %d, numPostSynapses = %d, numPreSynapses = %d, D = %d\n", curN, numPostSynapses, numPreSynapses, curD);
		fprintf(fpLog, "**********************************\n");

		assert(curD != 0); 	assert(numPostSynapses != 0);		assert(curN != 0); 		assert(numPreSynapses != 0);

		if (showLogMode >= 1)
			for (int g=0;g<numGrp;g++)
				printf("grp_Info[%d, %s].numPostSynapses = %d, grp_Info[%d, %s].numPreSynapses = %d\n",g,grp_Info2[g].Name.c_str(),grp_Info[g].numPostSynapses,g,grp_Info2[g].Name.c_str(),grp_Info[g].numPreSynapses);

		if (numPostSynapses > MAX_numPostSynapses) {
			for (int g=0;g<numGrp;g++)
				if (grp_Info[g].numPostSynapses>MAX_numPostSynapses) printf("Grp: %s(%d) has too many output synapses (%d), max %d.\n",grp_Info2[g].Name.c_str(),g,grp_Info[g].numPostSynapses,MAX_numPostSynapses);
			assert(numPostSynapses <= MAX_numPostSynapses);
		}
		if (numPreSynapses > MAX_numPreSynapses) {
			for (int g=0;g<numGrp;g++)
				if (grp_Info[g].numPreSynapses>MAX_numPreSynapses) printf("Grp: %s(%d) has too many input synapses (%d), max %d.\n",grp_Info2[g].Name.c_str(),g,grp_Info[g].numPreSynapses,MAX_numPreSynapses);
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
						if        (IS_EXCITATORY_TYPE(grp_Info[g].Type) &&  (grp_Info[g].Type&POISSON_NEURON) && order==3) {
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
			// read the plastic synapses first
			assert(readNetwork_internal(true) >= 0);

			// read the fixed synapses secon
			assert(readNetwork_internal(false) >= 0);
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
					bool    synWtType = GET_FIXED_PLASTIC(newInfo->connProp);
					if (synWtType == SYN_PLASTIC) {
					    grp_Info[newInfo->grpDest].FixedInputWts = false; // given group has plastic connection, and we need to apply STDP rule...
					}

					if( ((con == 0) && (synWtType == SYN_PLASTIC)) ||
						((con == 1) && (synWtType == SYN_FIXED)))
					{
						switch(newInfo->type)
						{
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
								grp_Info2[newInfo->grpSrc].Name.c_str(), newInfo->grpSrc, grp_Info2[newInfo->grpDest].Name.c_str(),
								newInfo->grpDest, newInfo->initWt, newInfo->maxWt, (int)avgPostM, (int)avgPreM,
								newInfo->minDelay, newInfo->maxDelay, synWtType?"Plastic":"Fixed");
					}
					newInfo = newInfo->next;
				}
			}
		}
	}

	int CpuSNN::findGrpId(int nid)
	{
		for(int g=0; g < numGrp; g++) {
			//printf("%d:%s s=%d e=%d\n", g, grp_Info2[g].Name.c_str(), grp_Info[g].StartN, grp_Info[g].EndN);
			if(nid >=grp_Info[g].StartN && (nid <=grp_Info[g].EndN)) {
				return g;
			}
		}
		fprintf(stderr, "findGrp(): cannot find the group for neuron %d\n", nid);
		assert(0);
	}

	void CpuSNN::writeNetwork(FILE* fid)
	{
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

				for(int idx_d = dPar.delay_index_start; idx_d < (dPar.delay_index_start + dPar.delay_length); idx_d = idx_d+1)
					count++;
			}

			fwrite(&count,sizeof(int),1,fid);

			for (int t=0;t<D;t++) {
				delay_info_t dPar = postDelayInfo[i*(D+1)+t];

				for(int idx_d = dPar.delay_index_start; idx_d < (dPar.delay_index_start + dPar.delay_length); idx_d = idx_d+1) {

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

	void CpuSNN::readNetwork(FILE* fid)
	{
		readNetworkFID = fid;
	}


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
				if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight>0) || !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (weight<0)) return -8;

				if (!fread(&maxWeight,sizeof(float),1,readNetworkFID)) return -11;

				if (IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight>=0) || !IS_INHIBITORY_TYPE(grp_Info[gIDpre].Type) && (maxWeight<=0)) return -8;

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


	// this is a user function
	// FIXME is this guy functional? replace it with Kris' version
	float* CpuSNN::getWeights(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weights) {
		Npre = grp_Info[gIDpre].SizeN;
		Npost = grp_Info[gIDpost].SizeN;

		if (weights==NULL) weights = new float[Npre*Npost];
		memset(weights,0,Npre*Npost*sizeof(float));

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

						weights[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = cpuNetPtrs.wt[pos_i];
					}
				}
			}
		}

		return weights;
	}

	// this is a user function
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



	uint8_t* CpuSNN::getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays) {
		Npre = grp_Info[gIDpre].SizeN;
		Npost = grp_Info[gIDpost].SizeN;

		if (delays == NULL) delays = new uint8_t[Npre*Npost];
		memset(delays,0,Npre*Npost);

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

						delays[i+Npre*(p_i-grp_Info[gIDpost].StartN)] = t+1;
					}
				}
			}
		}

		return delays;
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

				if ((showLogMode >= 1) && (i==grp_Info[g].StartN))
					fprintf(fpProgLog,"Weights, Change at %lu (diff_firing:%f) \n", simTimeSec, diff_firing);

				for(int j=0; j < Npre_plastic[i]; j++) {

					if ((showLogMode >= 1) && (i==grp_Info[g].StartN))
						fprintf(fpProgLog,"%1.2f %1.2f \t", wt[offset+j]*10, wtChange[offset+j]*10);

					wt[offset+j] += wtChange[offset+j];

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

	void CpuSNN::generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD)
	{
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
						wtChange[pos_i] -= (STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTP, grp_Info[post_grpId].TAU_LTP_INV) - STDP(stdp_tDiff, grp_Info[post_grpId].ALPHA_LTD*1.5, grp_Info[post_grpId].TAU_LTD_INV));
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

	void  CpuSNN::globalStateUpdate()
	{
#define CUR_DEBUG 1
		//fprintf(stdout, "---%d ----\n", simTime);
		// now we update the state of all the neurons
		for(int g=0; g < numGrp; g++) {
			if (grp_Info[g].Type&POISSON_NEURON) continue;

			for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
				assert(i < numNReg);

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
//if (i==grp_Info[6].StartN) printf("voltage: %f AMPA: %f NMDA: %f GABAa: %f GABAb: %f\n",voltage[i],gAMPA[i],gNMDA[i],gGABAa[i],gGABAb[i]);
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

	void CpuSNN::printWeight(int grpId, const char *str)
	{
		int stg, endg;
		if(grpId == -1) {
			stg  = 0;
			endg = numGrp;
		}
		else {
			stg = grpId;
			endg = grpId+1;
		}

		for(int g=stg; (g < endg) ; g++) {
			fprintf(stderr, "%s", str);
			if (!grp_Info[g].FixedInputWts) {
				//fprintf(stderr, "w=\t");
				if (currentMode == GPU_MODE) {
					copyWeightState (&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, g);
				}
				int i=grp_Info[g].StartN;
				unsigned int offset = cumulativePre[i];
				//fprintf(stderr, "time=%d, Neuron synaptic weights %d:\n", simTime, i);
				for(int j=0; j < Npre[i]; j++) {
					//fprintf(stderr, "w=%f c=%f spt=%d\t", wt[offset+j], wtChange[offset+j], synSpikeTime[offset+j]);
					fprintf(stdout, "%1.3f,%1.3f\t", cpuNetPtrs.wt[offset+j], cpuNetPtrs.wtChange[offset+j]);
				}
				fprintf(stdout, "\n");
			}
		}
	}

	// show the status of the simulator...
	// when onlyCnt is set, we print the actual count instead of frequency of firing
	void CpuSNN::showStatus(int simType)
	{
		DBG(2, fpLog, AT, "showStatus() called");

		printState("showStatus");

		if(simType == GPU_MODE) {
			showStatus_GPU();
			return;
		}

		FILE* fpVal[2];
		fpVal[0] = fpLog;
		fpVal[1] = fpProgLog;

		for(int k=0; k < 2; k++) {

			if(k==0)
				printWeight(-1);

            fprintf(fpVal[k], "(time=%lld) =========\n\n", (unsigned long long) simTimeSec);


#if REG_TESTING
			// if the overall firing rate is very low... then report error...
			if((spikeCountAll1sec*1.0f/numN) < 1.0) {
				fprintf(fpVal[k], " SIMULATION WARNING !!! Very Low Firing happened...\n");
				fflush(fpVal[k]);
			}
#endif

			fflush(fpVal[k]);
		}

#if REG_TESTING
		if(spikeCountAll1sec == 0) {
			fprintf(stderr, " SIMULATION ERROR !!! Very Low or no firing happened...\n");
			//exit(-1);
		}
#endif
	}

	void CpuSNN::startCPUTiming()
	{
		prevCpuExecutionTime = cumExecutionTime;
	}

	void CpuSNN::resetCPUTiming()
	{
		prevCpuExecutionTime = cumExecutionTime;
		cpuExecutionTime     = 0.0;
	}

	void CpuSNN::stopCPUTiming()
	{
		cpuExecutionTime += (cumExecutionTime - prevCpuExecutionTime);
		prevCpuExecutionTime = cumExecutionTime;
	}

	void CpuSNN::startGPUTiming()
	{
		prevGpuExecutionTime = cumExecutionTime;
	}

	void CpuSNN::resetGPUTiming()
	{
		prevGpuExecutionTime = cumExecutionTime;
		gpuExecutionTime     = 0.0;
	}

	void CpuSNN::stopGPUTiming()
	{
		gpuExecutionTime += (cumExecutionTime - prevGpuExecutionTime);
		prevGpuExecutionTime = cumExecutionTime;
	}

	// reorganize the network and do the necessary allocation
	// of all variable for carrying out the simulation..
	// this code is run only one time during network initialization
	void CpuSNN::setupNetwork(int simType, int ithGPU, bool removeTempMem)
	{
		if(!doneReorganization)
			reorganizeNetwork(removeTempMem, simType);

		if((simType == GPU_MODE) && (cpu_gpuNetPtrs.allocated == false))
			allocateSNN_GPU(ithGPU);
	}

	bool CpuSNN::updateTime()
	{
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

	// Run the simulation for n sec
	int CpuSNN::runNetwork(int _nsec, int _nmsec, int simType, int ithGPU, bool enablePrint, int copyState)
	{
		DBG(2, fpLog, AT, "runNetwork() called");

		assert(_nmsec >= 0);
		assert(_nsec  >= 0);

		assert(simType == CPU_MODE || simType == GPU_MODE);

		int runDuration = _nsec*1000 + _nmsec;

		setGrpTimeSlice(ALL, MAX(1,MIN(runDuration,PROPOGATED_BUFFER_SIZE-1)));  // set the Poisson generation time slice to be at the run duration up to PROPOGATED_BUFFER_SIZE ms.

		// First time when the network is run we do various kind of space compression,
		// and data structure optimization to improve performance and save memory.

		setupNetwork(simType,ithGPU);

		currentMode = simType;

		CUDA_RESET_TIMER(timer);

		CUDA_START_TIMER(timer);

		// if nsec=0, simTimeMs=10, we need to run the simulator for 10 timeStep;
		// if nsec=1, simTimeMs=10, we need to run the simulator for 1*1000+10, time Step;
		for(int i=0; i < runDuration; i++) {

//			initThalInput();

			if(simType == CPU_MODE)
				doSnnSim();
			else
				doGPUSim();

			if (enablePrint) {
				printState();
			}

			if (updateTime()) {

				// finished one sec of simulation...
				if(showLog)
					if(showLogCycle==showLog++) {
						showStatus(currentMode);
						showLog=1;
					}

				updateSpikeMonitor();

				if(simType == CPU_MODE)
					updateStateAndFiringTable();
				else
					updateStateAndFiringTable_GPU();
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
		if(0) {
			fprintf(fpLog, "(t%s = %2.2f sec)\n",  (currentMode == GPU_MODE)?"GPU":"CPU", lastExecutionTime/1000);
			fprintf(stdout, "(t%s = %2.2f sec)\n", (currentMode == GPU_MODE)?"GPU":"CPU", lastExecutionTime/1000);
		}
		cumExecutionTime += lastExecutionTime;
		return 0;
	}

	void CpuSNN::updateAfterMaxTime()
	{
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

	void CpuSNN::resetPropogationBuffer()
	{
		pbuf->reset(0, 1023);
	}

	unsigned int poissonSpike(unsigned int currTime, float frate, int refractPeriod)
	{
		bool done = false;
		unsigned int nextTime = 0;
		assert(refractPeriod>0); // refractory period must be 1 or greater, 0 means could have multiple spikes specified at the same time.
		static int cnt = 0;
		while(!done)
		{
			float randVal = drand48();
			unsigned int tmpVal  = -log(randVal)/frate;
			nextTime = currTime + tmpVal;
			//fprintf(stderr, "%d: next random = %f, frate = %f, currTime = %d, nextTime = %d tmpVal = %d\n", cnt++, randVal, frate, currTime, nextTime, tmpVal);
			if ((nextTime - currTime) >= (unsigned) refractPeriod)
				done = true;
		}

		assert(nextTime != 0);
		return nextTime;
	}

	//
	void CpuSNN::updateSpikeGeneratorsInit()
	{
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

	void CpuSNN::updateSpikeGenerators()
	{
		for(int g=0; (g < numGrp); g++) {
			if (grp_Info[g].isSpikeGenerator) {
				// This evaluation is done to check if its time to get new set of spikes..
//				fprintf(stderr, "[MODE=%s] simtime = %d, NumTimeSlice = %d, SliceUpdate = %d, currTime = %d\n",
//						(currentMode==GPU_MODE)?"GPU_MODE":"CPU_MODE",  simTime, grp_Info[g].NumTimeSlice,
//						grp_Info[g].SliceUpdateTime, grp_Info[g].CurrTimeSlice);
				if(((simTime-grp_Info[g].SliceUpdateTime) >= (unsigned) grp_Info[g].CurrTimeSlice))
					updateSpikesFromGrp(g);
			}
		}
	}

	void CpuSNN::generateSpikesFromFuncPtr(int grpId)
	{
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

	void CpuSNN::generateSpikesFromRate(int grpId)
	{
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

			unsigned int nextTime = lastSpikeTime[grp_Info[grpId].StartN+cnt]; // start the time from the last time it spiked, that way we can ensure that the refractory period is maintained
			if (nextTime == MAX_SIMULATION_TIME)
				nextTime = 0;

			done = false;
			while (!done && frate>0) {
				nextTime = poissonSpike (nextTime, frate/1000.0, refPeriod);
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

	inline int CpuSNN::getPoissNeuronPos(int nid)
	{
		int nPos = nid-numNReg;
		assert(nid >= numNReg);
		assert(nid < numN);
		assert((nid-numNReg) < numNPois);
		return nPos;
	}

	void CpuSNN::setSpikeRate(int grpId, PoissonRate* ratePtr, int refPeriod, int configId)
	{
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
				fprintf(stderr,"The PoissonRate length did not match the number of neurons in group %s(%d).\n", grp_Info2[cGrpId].Name.c_str(),grpId);
				assert(0);
			}

			assert (grp_Info[cGrpId].isSpikeGenerator);
			grp_Info[cGrpId].RatePtr = ratePtr;
			grp_Info[cGrpId].RefractPeriod   = refPeriod;
			spikeRateUpdated = true;

		}
	}

	void CpuSNN::setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId)
	{
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

	void CpuSNN::generateSpikes()
	{
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

		//fprintf(stderr, "\n");

		// advance the time step to the next phase...
		pbuf->nextTimeStep();

	}

	void CpuSNN::setGrpTimeSlice(int grpId, int timeSlice)
	{
		if (grpId == ALL) {
			for(int g=0; (g < numGrp); g++) {
				if (grp_Info[g].isSpikeGenerator)
					setGrpTimeSlice(g, timeSlice);
			}
		} else {
			assert((timeSlice > 0 ) && (timeSlice <  PROPOGATED_BUFFER_SIZE));
			// the group should be poisson spike generator group
			grp_Info[grpId].NewTimeSlice = timeSlice;
			grp_Info[grpId].CurrTimeSlice = timeSlice;
		}
	}

	int CpuSNN::addSpikeToTable(int nid, int g)
	{
		int spikeBufferFull = 0;
		lastSpikeTime[nid] = simTime;
		curSpike[nid] = true;
		nSpikeCnt[nid]++;

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
			int ind_1 = STP_BUF_POS(nid,(simTime-1)); // MDR -1 is correct, we use the value before the decay has been applied for the current time step.
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

	void CpuSNN::findFiring()
	{
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
										wtChange[pos_ij] -= (STDP(stdp_tDiff, grp_Info[g].ALPHA_LTP, grp_Info[g].TAU_LTP_INV) - STDP(stdp_tDiff, grp_Info[g].ALPHA_LTD*1.5, grp_Info[g].TAU_LTD_INV));
							#endif
						}
					}
					spikeCountAll1sec++;
				}
			}
		}
	}

	void CpuSNN::doSTPUpdates()
	{
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

	void CpuSNN::doSnnSim()
	{
		doSTPUpdates();

		//return;

		updateSpikeGenerators();

		//generate all the scheduled spikes from the spikeBuffer..
		generateSpikes();

		if(0) fprintf(fpProgLog, "\nLTP time=%d, \n", simTime);

		// find the neurons that has fired..
		findFiring();

		timeTableD2[simTimeMs+D+1] = secD2fireCntHost;
		timeTableD1[simTimeMs+D+1] = secD1fireCntHost;

		if(0) fprintf(fpProgLog, "\nLTD time=%d,\n", simTime);

		doD2CurrentUpdate();

		doD1CurrentUpdate();

		globalStateUpdate();

		updateMonitors();

		return;

	}

	void CpuSNN::swapConnections(int nid, int oldPos, int newPos)
	{
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


	char* extractFileName(char *fname)
	{
		char* varname = strrchr(fname, '\\');
		size_t len1 = strlen(varname+1);
		char* extname = strchr(varname, '.');
		size_t len2 = strlen(extname);
		varname[len1-len2]='\0';
		return (varname+1);
	}

#define COMPACTION_ALIGNMENT_PRE  16
#define COMPACTION_ALIGNMENT_POST 0

#define SETPOST_INFO(name, nid, sid, val) name[cumulativePost[nid]+sid]=val;

#define SETPRE_INFO(name, nid, sid, val)  name[cumulativePre[nid]+sid]=val;


	// initialize all the synaptic weights to appropriate values..
	// total size of the synaptic connection is 'length' ...
	void CpuSNN::initSynapticWeights()
	{
		// Initialize the network wtChange, wt, synaptic firing time
		wtChange         = new float[preSynCnt];
		synSpikeTime     = new uint32_t[preSynCnt];
//		memset(synSpikeTime,0,sizeof(uint32_t)*preSynCnt);
		cpuSnnSz.synapticInfoSize = sizeof(float)*(preSynCnt*2);

		resetSynapticConnections(false);
	}

	// We parallelly cleanup the postSynapticIds array to minimize any other wastage in that array by compacting the store
	// Appropriate alignment specified by ALIGN_COMPACTION macro is used to ensure some level of alignment (if necessary)
	void CpuSNN::compactConnections()
	{
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

	void CpuSNN::updateParameters(int* curN, int* numPostSynapses, int* numPreSynapses, int nConfig)
	{
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

			if      (IS_INHIBITORY_TYPE(grp_Info[g].Type) && !(grp_Info[g].Type&POISSON_NEURON))
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

	void CpuSNN::resetSynapticConnections(bool changeWeights)
	{
		int j;

		// Reset wt,wtChange,pre-firingtime values to default values...
		for(int destGrp=0; destGrp < numGrp; destGrp++) {
			const char* updateStr = (grp_Info[destGrp].newUpdates == true)?"(**)":"";
			fprintf(stdout, "Grp: %d:%s s=%d e=%d %s\n", destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN, grp_Info[destGrp].EndN,  updateStr);
			fprintf(fpLog,  "Grp: %d:%s s=%d e=%d  %s\n",  destGrp, grp_Info2[destGrp].Name.c_str(), grp_Info[destGrp].StartN, grp_Info[destGrp].EndN, updateStr);

			for(int nid=grp_Info[destGrp].StartN; nid <= grp_Info[destGrp].EndN; nid++) {
				unsigned int offset = cumulativePre[nid];
				for (j=0;j<Npre[nid]; j++) wtChange[offset+j] = 0.0;						// synaptic derivatives is reset
				for (j=0;j<Npre[nid]; j++) synSpikeTime[offset+j] = MAX_SIMULATION_TIME;	// some large negative value..
				post_info_t *preIdPtr = &preSynapticIds[cumulativePre[nid]];
				float* synWtPtr       = &wt[cumulativePre[nid]];
				float* maxWtPtr       = &maxSynWt[cumulativePre[nid]];
				int prevPreGrp  = -1;
				/* MDR -- this code no longer works because grpConnInfo is no longer used/defined
				for (j=0; j < Npre[nid]; j++,preIdPtr++, synWtPtr++, maxWtPtr++) {
					int preId    = GET_CONN_NEURON_ID((*preIdPtr));
					assert(preId < numN);
					int srcGrp   = findGrpId(preId);
					grpConnectInfo_t* connInfo = grpConnInfo[srcGrp][destGrp];
					assert(connInfo != NULL);
					int connProp   = connInfo->connProp;
					bool   synWtType = GET_FIXED_PLASTIC(connProp);

					if ( j >= Npre_plastic[nid] ) {
						// if the j is greater than Npre_plastic it means the
						// connection should be fixed type..
						//MDR unfortunately, for user defined connections, this check is not valid if they have a mixture of fixed and plastic
						//MDR assert(synWtType == SYN_FIXED);
					}

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

					// if connection was of plastic type or if the connection weights were updated we need to reset the weights..
					// TODO: How to account for user-defined connection reset...
					if ((synWtType == SYN_PLASTIC) || connInfo->newUpdates) {
						*synWtPtr = getWeights(connInfo->connProp, connInfo->initWt, connInfo->maxWt, nid, srcGrp);
						*maxWtPtr = connInfo->maxWt;
					}
				}
				*/
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

	void CpuSNN::resetGroups()
	{
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

	void CpuSNN::resetFiringInformation()
	{
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

	void CpuSNN::printTuningLog()
	{
		if (fpTuningLog) {
			fprintf(fpTuningLog, "Generating Tuning log %d\n", cntTuning);
			printParameters(fpTuningLog);
			cntTuning++;
		}
	}

	// after all the initalization. Its time to create the synaptic weights, weight change and also
	// time of firing these are the mostly costly arrays so dense packing is essential to minimize wastage of space
	void CpuSNN::reorganizeNetwork(bool removeTempMemory, int simType)
	{
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

		//not of much use currently
//		updateRandomProperty();

		updateSpikeGeneratorsInit();

		//ensure that we dont do all the above optimizations again
		doneReorganization = true;

		//printParameters(fpParam);
		printParameters(fpLog);

		printTuningLog();

		makePtrInfo();

		// if our initial operating mode is GPU_MODE, then it is time to
		// allocate necessary data within the GPU

//		assert(simType != GPU_MODE || cpu_gpuNetPtrs.allocated);
//		if(netInitMode==GPU_MODE)
//			allocateSNN_GPU();

		if(simType==GPU_MODE)
			fprintf(stdout, "Starting GPU-SNN Simulations ....\n");
		else
			fprintf(stdout, "Starting CPU-SNN Simulations ....\n");

		FILE* fconn = NULL;

#if TESTING
		if (simType == GPU_MODE)
			fconn = fopen("gpu_conn.txt", "w");
		else
			fconn = fopen("conn.txt", "w");
		//printCheckDetailedInfo(fconn);
#endif

#if 0
		printPostConnection(fconn);
		printPreConnection(fconn);
#endif

		//printNetworkInfo();

		printDotty();

#if TESTING
		fclose(fconn);
#endif

		if(removeTempMemory) {
			memoryOptimized = true;
			delete[] tmp_SynapticDelay;
			tmp_SynapticDelay = NULL;
		}
	}

	void  CpuSNN::setDefaultParameters(float alpha_ltp, float tau_ltp, float alpha_ltd, float tau_ltd)
	{
		printf("Warning: setDefaultParameters() is deprecated and may be removed in the future.\nIt is recommended that you set the desired parameters explicitly.\n");
		#define DEFAULT_COND_tAMPA  5.0
		#define DEFAULT_COND_tNMDA  150.0
		#define DEFAULT_COND_tGABAa 6.0
		#define DEFAULT_COND_tGABAb 150.0

		#define DEFAULT_STP_U_Inh  0.5
		#define	DEFAULT_STP_tD_Inh 800
		#define	DEFAULT_STP_tF_Inh 1000
		#define DEFAULT_STP_U_Exc  0.2
		#define	DEFAULT_STP_tD_Exc 700
		#define	DEFAULT_STP_tF_Exc 20

		// setup the conductance parameter for the given network
		setConductances(ALL, true, DEFAULT_COND_tAMPA, DEFAULT_COND_tNMDA, DEFAULT_COND_tGABAa, DEFAULT_COND_tGABAb);

		// setting the STP values for different groups...
		for(int g=0; g < getNumGroups(); g++) {

			// default STDP is set to false for all groups..
			setSTDP(g, false);

			// setup the excitatory group properties here..
			if(isExcitatoryGroup(g)) {
				setSTP(g, true, DEFAULT_STP_U_Exc, DEFAULT_STP_tD_Exc, DEFAULT_STP_tF_Exc);
				if ((alpha_ltp!=0.0) && (!isPoissonGroup(g)))
					setSTDP(g, true, alpha_ltp, tau_ltp, alpha_ltd, tau_ltd);
			}
			else {
				setSTP(g, true, DEFAULT_STP_U_Inh, DEFAULT_STP_tD_Inh, DEFAULT_STP_tF_Inh);
			}
		}
	}

#if ! (_WIN32 || _WIN64)
	#include <string.h>
	#define strcmpi(s1,s2) strcasecmp(s1,s2)
#endif

#define PROBE_CURRENT (1 << 1)
#define PROBE_VOLTAGE (1 << 2)
#define PROBE_FIRING_RATE (1 << 3)

	void CpuSNN::setProbe(int g, const string& type, int startId, int cnt, uint32_t _printProbe)
	{
		int endId;
		assert(startId >= 0);
		assert(startId <= grp_Info[g].SizeN);
		//assert(cnt!=0);

		int i=grp_Info[g].StartN+startId;
		if (cnt<=0)
		   endId = grp_Info[g].EndN;
		else
		   endId = i + cnt - 1;

		if(endId > grp_Info[g].EndN)
		   endId = grp_Info[g].EndN;

		for(; i <= endId; i++) {

			probeParam_t* n  = new probeParam_t;
			memset(n, 0, sizeof(probeParam_t));
			n->next = neuronProbe;

			if(type.find("current") != string::npos) {
				n->type |= PROBE_CURRENT;
				n->bufferI = new float[1000];
				cpuSnnSz.probeInfoSize += sizeof(float)*1000;
			}

			if(type.find("voltage") != string::npos) {
				n->type |= PROBE_VOLTAGE;
				n->bufferV = new float[1000];
				cpuSnnSz.probeInfoSize += sizeof(float)*1000;
			}

			if(type.find("firing-rate") != string::npos) {
				n->type |= PROBE_FIRING_RATE;
				n->spikeBins   = new bool[1000];
				n->bufferFRate = new float[1000];
				cpuSnnSz.probeInfoSize += (sizeof(float)+sizeof(bool))*1000;
				n->cumCount   = 0;
			}

			n->vmax = 40.0; n->vmin = -80.0;
			n->imax = 50.0;	n->imin = -50.0;
			n->debugCnt = 0;
			n->printProbe = _printProbe;
			n->nid 	    = i;
			neuronProbe = n;
			numProbe++;
		}
	}

	void CpuSNN::updateMonitors()
	{
		int cnt=0;
		probeParam_t* n = neuronProbe;

		while(n) {
			int nid = n->nid;
			if(n->printProbe) {

				// if there is an input inhspike or excSpike or curSpike then display values..
				//if( I[nid] != 0 ) {

				/*  FIXME
				    MDR I broke this because I removed inhSpikes and excSpikes because they are incorrectly named...

					if (inhSpikes[nid] || excSpikes[nid] || curSpike[nid] || (n->debugCnt++ > n->printProbe)) {
					fprintf(stderr, "[t=%d, n=%d] voltage=%3.3f current=%3.3f ", simTime, nid, voltage[nid], current[nid]);
					//FIXME should check to see if conductances are enabled for this group
					if (true) {
						fprintf(stderr, " ampa=%3.4f nmda=%3.4f gabaA=%3.4f gabaB=%3.4f ",
								gAMPA[nid], gNMDA[nid], gGABAa[nid], gGABAb[nid]);
					}

					fprintf(stderr, " +Spike=%d ", excSpikes[nid]);

					if (inhSpikes[nid])
						fprintf(stderr, " -Spike=%d ", inhSpikes[nid]);

					if (curSpike[nid])
						fprintf(stderr, " | ");

					fprintf(stderr, "\n");

					if(n->debugCnt > n->printProbe) {
						n->debugCnt = 0;
					}
				}
				*/
			}

			if (n->type & PROBE_CURRENT)
				n->bufferI[simTimeMs] = current[nid];

			if (n->type & PROBE_VOLTAGE)
				n->bufferV[simTimeMs] = voltage[nid];

			#define	NUM_BIN  256
			#define BIN_SIZE 0.001 /* 1ms bin size */

			if (n->type & PROBE_FIRING_RATE) {
				int oldSpike = 0;
				if (simTime >= NUM_BIN)
				 oldSpike = n->spikeBins[simTimeMs%NUM_BIN];
				int newSpike = (voltage[nid] >= 30.0 ) ? 1 : 0;
				n->cumCount   = n->cumCount - oldSpike + newSpike;
				float frate   = n->cumCount/(NUM_BIN*BIN_SIZE);
				n->spikeBins[simTimeMs % NUM_BIN] = newSpike;
				if (simTime >= NUM_BIN)
				  n->bufferFRate[(simTime-NUM_BIN)%1000] = frate;
			}

			n=n->next;
			cnt++;
		}

		//fclose(fp);
		// ensure that we checked all the nodes in the list
		assert(cnt==numProbe);
	}

	class WriteSpikeToFile: public SpikeMonitor {
	public:
		WriteSpikeToFile(FILE* _fid) {
			fid = _fid;
		}

		void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts)
		{
			int pos    = 0;

			for (int t=0; t < 1000; t++) {
				for(int i=0; i<timeCnts[t];i++,pos++) {
					int time = t + s->getSimTime() - 1000;
					int id   = Nids[pos];
					int cnt = fwrite(&time,sizeof(int),1,fid);
					assert(cnt != 0);
					cnt = fwrite(&id,sizeof(int),1,fid);
					assert(cnt != 0);
				}
			}

			fflush(fid);
		}

		FILE* fid;
	};

	void CpuSNN::setSpikeMonitor(int gid, const string& fname, int configId) {
		FILE* fid = fopen(fname.c_str(),"wb");
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
					fprintf(stderr,"ERROR %d: could not create directory: %s\n",errno, strerror(errno));
					exit(1);
					return;
				}

				// now that the directory is created, fopen file
				fid = fopen(fname.c_str(),"wb");
			#else
				// default case: print error and exit
				fprintf(stderr,"ERROR: File \"%s\" could not be opened, please check if it exists.\n",fname.c_str());
				fprintf(stderr,"       Enable option CREATE_SPIKEDIR_IF_NOT_EXISTS in config.h to attempt creating the "
					"specified subdirectory automatically.\n");
				exit(1);
				return;
			#endif
		}

		assert(configId != ALL);

		setSpikeMonitor(gid, new WriteSpikeToFile(fid), configId);
	}

	void CpuSNN::setSpikeMonitor(int grpId, SpikeMonitor* spikeMon, int configId)
	{
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
		}
	}

	void CpuSNN::updateSpikeMonitor()
	{
		// dont continue if numSpikeMonitor is zero
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
								fprintf(stderr, "Buffer Monitor size (%d) is small. Increase buffer firing rate for %s\n", monBufferSize[monitorId], grp_Info2[grpId].Name.c_str());
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
				fprintf(stderr, "Spike Monitor for Group %s has %d spikes (%f Hz)\n",grp_Info2[grpId].Name.c_str(),monBufferPos[monitorId],((float)monBufferPos[monitorId])/(grp_Info[grpId].SizeN));

				// call the callback function
				if (monBufferCallback[monitorId])
					monBufferCallback[monitorId]->update(this,grpId,monBufferFiring[monitorId],monBufferTimeCnt[monitorId]);
			}
		}
	}












/* MDR -- DEPRECATED
	void CpuSNN::initThalInput()
	{
		DBG(2, fpLog, AT, "initThalInput()");

		#define I_EXPONENTIAL_DECAY 5.0f

		// reset thalamic currents here..
		//KILLME NOW
		//resetCurrent();

		resetCounters();

		// first we do the initial simulation corresponding to the random number generator
		int randCnt=0;
		//fprintf(stderr, "Thalamic inputs: ");
		for (int i=0; i < numNoise; i++)
		{
			 float  currentStrength  = noiseGenGroup[i].currentStrength;
			 int    ncount = noiseGenGroup[i].ncount;
			 int    nstart = noiseGenGroup[i].nstart;
			 int    nrands = noiseGenGroup[i].rand_ncount;

			assert(ncount > 0);
			assert(nstart >= 0);
			assert(nrands > 0);

			 for(int j=0; j < nrands; j++,randCnt++) {
				 int rneuron = nstart + (int)(ncount*getRand());
				 randNeuronId[randCnt]= rneuron;
				 // assuming there is only one driver at a time.
				 // More than one noise input is not correct...
				 current[rneuron] = currentStrength;
				 //fprintf(stderr, " %d ", rneuron);
			 }
		}
		//fprintf(stderr, "\n");
	}


	// This would supply the required random current to the specific percentage of neurons. If groupId=-1,
	// then the neuron is picked from the full population.
	// If the groupId=n, then only neurons are picked from selected group n.
	void CpuSNN::randomNoiseCurrent(float neuronPercentage, float currentStrength, int groupId)
	{
		DBG(2, fpLog, AT, "randomNoiseCurrent() added");

		int  numNeuron  = ((groupId==-1)? -1: grp_Info[groupId].SizeN);
		int  nstart     = ((groupId==-1)? -1: grp_Info[groupId].StartN);
		noiseGenGroup[numNoise].groupId		 = groupId;
		noiseGenGroup[numNoise].currentStrength  = currentStrength;
		noiseGenGroup[numNoise].neuronPercentage = neuronPercentage;
		noiseGenGroup[numNoise].ncount		 = numNeuron;
		noiseGenGroup[numNoise].nstart		 = nstart;
		noiseGenGroup[numNoise].rand_ncount	 = (int)(numNeuron*(neuronPercentage/100.0));
		if(noiseGenGroup[numNoise].rand_ncount<=0)
			noiseGenGroup[numNoise].rand_ncount = 1;

		numNoise++;

		// i have already reorganized the network. hence it is important to update the random neuron property..
		if (doneReorganization == true)
			updateRandomProperty();
	}


	// we calculate global properties like how many random number need to be generated
	//each cycle, etc
	void CpuSNN::updateRandomProperty()
	{
		numRandNeurons = 0;
		for (int i=0; i < numNoise; i++) {
		     // paramters for few noise generator has not been updated...update now !!!
		     if(noiseGenGroup[i].groupId == -1) {
			 int  numNeuron  = numNReg;
			 int  nstart     = 0;
			 noiseGenGroup[i].groupId = -1;
			 int neuronPercentage     = noiseGenGroup[i].neuronPercentage;
			 noiseGenGroup[i].ncount  = numNeuron;
			 noiseGenGroup[i].nstart  = nstart;
			 noiseGenGroup[i].rand_ncount	 = (int)(numNeuron*(neuronPercentage/100.0));
			 if(noiseGenGroup[i].rand_ncount<=0)
				 noiseGenGroup[i].rand_ncount = 1;
		     }

		     // update the number of random counts...
		     numRandNeurons += noiseGenGroup[i].rand_ncount;
		}

		assert(numRandNeurons>=0);
	}
*/
