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

#include <snn.h>
#include <connection_monitor_core.h>
#include <group_monitor_core.h>

void SNN::printConnection(const std::string& fname) {
	FILE *fp = fopen(fname.c_str(), "w");
	printConnection(fp);
	fclose(fp);
}

void SNN::printConnection(FILE* const fp) {
	printPostConnection(fp);
	printPreConnection(fp);
}

//! print the connection info of grpId
void SNN::printConnection(int grpId, FILE* const fp) {
	printPostConnection(grpId, fp);
	printPreConnection(grpId, fp);
}

void SNN::printMemoryInfo(FILE* const fp) {
	if (snnState == CONFIG_SNN || snnState == COMPILED_SNN || snnState == LINKED_SNN) {
		KERNEL_DEBUG("checkNetworkBuilt()");
		KERNEL_DEBUG("Network not yet elaborated and built...");
	}

	fprintf(fp, "************* Memory Info ***************\n");
	int totMemSize = cpuSnnSz.networkInfoSize+cpuSnnSz.synapticInfoSize+cpuSnnSz.neuronInfoSize+cpuSnnSz.spikingInfoSize;
	fprintf(fp, "Neuron Info Size:\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.neuronInfoSize*100.0/totMemSize,   cpuSnnSz.neuronInfoSize/(1024.0*1024));
	fprintf(fp, "Synaptic Info Size:\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.synapticInfoSize*100.0/totMemSize, cpuSnnSz.synapticInfoSize/(1024.0*1024));
	fprintf(fp, "Network Size:\t\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.networkInfoSize*100.0/totMemSize,  cpuSnnSz.networkInfoSize/(1024.0*1024));
	fprintf(fp, "Firing Info Size:\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.spikingInfoSize*100.0/totMemSize,   cpuSnnSz.spikingInfoSize/(1024.0*1024));
	fprintf(fp, "Additional Info:\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.addInfoSize*100.0/totMemSize,      cpuSnnSz.addInfoSize/(1024.0*1024));
	fprintf(fp, "DebugInfo Info:\t\t%3.2f %%\t(%3.2f MB)\n", cpuSnnSz.debugInfoSize*100.0/totMemSize,    cpuSnnSz.debugInfoSize/(1024.0*1024));
	fprintf(fp, "*****************************************\n\n");

	fprintf(fp, "************* Connection Info *************\n");
	for(int g=0; g < numGrp; g++) {
		int TNpost=0;
		int TNpre=0;
		int TNpre_plastic=0;
		
		for(int i=groupConfig[g].StartN; i <= groupConfig[g].EndN; i++) {
			TNpost += snnRuntimeData.Npost[i];
			TNpre  += snnRuntimeData.Npre[i];
			TNpre_plastic += snnRuntimeData.Npre_plastic[i];
		}
		
	fprintf(fp, "%s Group (num_neurons=%5d): \n\t\tNpost[%2d] = %3d, Npre[%2d]=%3d Npre_plastic[%2d]=%3d \n\t\tcumPre[%5d]=%5d cumPre[%5d]=%5d cumPost[%5d]=%5d cumPost[%5d]=%5d \n",
		groupInfo[g].Name.c_str(), groupConfig[g].SizeN, g, TNpost/groupConfig[g].SizeN, g, TNpre/groupConfig[g].SizeN, g, TNpre_plastic/groupConfig[g].SizeN,
		groupConfig[g].StartN, snnRuntimeData.cumulativePre[groupConfig[g].StartN],  groupConfig[g].EndN, snnRuntimeData.cumulativePre[groupConfig[g].EndN],
		groupConfig[g].StartN, snnRuntimeData.cumulativePost[groupConfig[g].StartN], groupConfig[g].EndN, snnRuntimeData.cumulativePost[groupConfig[g].EndN]);
	}
	fprintf(fp, "**************************************\n\n");
}

void SNN::printStatusConnectionMonitor(int connId) {
	for (int monId=0; monId<numConnectionMonitor; monId++) {
		if (connId==ALL || connMonCoreList[monId]->getConnectId()==connId) {
			// print connection weights (sparse representation: show only actually connected synapses)
			// show the first hundred connections: (pre=>post) weight
			connMonCoreList[monId]->printSparse(ALL, 100, 4, false);
		}
	}
}

void SNN::printStatusSpikeMonitor(int grpId) {
	if (grpId==ALL) {
		for (int grpId1=0; grpId1<numGrp; grpId1++) {
			printStatusSpikeMonitor(grpId1);
		}
	} else {
		int monitorId = groupConfig[grpId].SpikeMonitorId;
		if (monitorId==-1) {
			return;
		}

		// in GPU mode, need to get data from device first
		if (simMode_==GPU_MODE)
			copyFiringStateFromGPU(grpId);

		// \TODO nSpikeCnt should really be a member of the SpikeMonitor object that gets populated if
		// printRunSummary is true or mode==COUNT.....
		// so then we can use spkMonObj->print(false); // showSpikeTimes==false
		int grpSpk = 0;
		for (int neurId=groupConfig[grpId].StartN; neurId<=groupConfig[grpId].EndN; neurId++)
			grpSpk += snnRuntimeData.nSpikeCnt[neurId]; // add up all neuronal spike counts

		// infer run duration by measuring how much time has passed since the last run summary was printed
		unsigned int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %dms (%.2f +/- %.2f Hz)",
				(float)(simTime/1000.0f),
				groupInfo[grpId].Name.c_str(),
				grpId,
				0,
				0,
				0.0f,
				0.0f);
		} else {
			// if some time has passed since last print
			float meanRate = grpSpk*1000.0f/runDurationMs/groupConfig[grpId].SizeN;
			float std = 0.0f;
			if (groupConfig[grpId].SizeN > 1) {
				for (int neurId=groupConfig[grpId].StartN; neurId<=groupConfig[grpId].EndN; neurId++) {
					float neurRate = snnRuntimeData.nSpikeCnt[neurId]*1000.0f/runDurationMs;
					std += (neurRate-meanRate)*(neurRate-meanRate);
				}
				std = sqrt(std/(groupConfig[grpId].SizeN-1.0));
			}
	
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %ums (%.2f +/- %.2f Hz)",
				(float)(simTime/1000.0f),
				groupInfo[grpId].Name.c_str(),
				grpId,
				grpSpk,
				runDurationMs,
				meanRate,
				std);
		}
	}
}

void SNN::printStatusGroupMonitor(int grpId) {
	if (grpId == ALL) {
		for (int g = 0; g < numGrp; g++) {
			printStatusGroupMonitor(g);
		}
	} else {
		int monitorId = groupConfig[grpId].GroupMonitorId;
		if (monitorId == -1)
			return;

		std::vector<int> peakTimeVector = groupMonCoreList[monitorId]->getPeakTimeVector();
		int numPeaks = peakTimeVector.size();

		// infer run duration by measuring how much time has passed since the last run summary was printed
		unsigned int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %dms",
				(float)(simTime/1000.0f),
				groupInfo[grpId].Name.c_str(),
				grpId,
				0,
				0);
		} else {
			// if some time has passed since last print
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %ums",
				(float)(simTime/1000.0f),
				groupInfo[grpId].Name.c_str(),
				grpId,
				numPeaks,
				runDurationMs);
		}
	}
}


// This method allows us to print all information about the neuron.
// If the enablePrint is false for a specific group, we do not print its state.
void SNN::printState(FILE* const fp) {
	for(int g=0; g < numGrp; g++)
		printNeuronState(g, fp);
}

void SNN::printTuningLog(FILE * const fp) {
	if (fp) {
		fprintf(fp, "Generating Tuning log\n");
//		printParameters(fp);
	}
}

void SNN::printConnectionInfo(FILE * const fp)
{
  ConnectConfig* newInfo = connectBegin;

  //    fprintf(fp, "\nGlobal STDP Info: \n");
  //    fprintf(fp, "------------\n");
  //    fprintf(fp, " alpha_ltp: %f\n tau_ltp: %f\n alpha_ldp: %f\n tau_ldp: %f\n", ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

  fprintf(fp, "\nConnections: \n");
  fprintf(fp, "------------\n");
  while(newInfo) {
    bool synWtType  = GET_FIXED_PLASTIC(newInfo->connProp);
    fprintf(fp, " // (%s => %s): numPostSynapses=%d, numPreSynapses=%d, iWt=%3.3f, mWt=%3.3f, ty=%x, maxD=%d, minD=%d %s\n",
      groupInfo[newInfo->grpSrc].Name.c_str(), groupInfo[newInfo->grpDest].Name.c_str(),
      newInfo->numPostSynapses, newInfo->numPreSynapses, newInfo->initWt,   newInfo->maxWt,
      newInfo->connProp, newInfo->maxDelay, newInfo->minDelay, (synWtType == SYN_PLASTIC)?"(*)":"");

    //      weights of input spike generating layers need not be observed...
    //          bool synWtType  = GET_FIXED_PLASTIC(newInfo->connProp);
    //      if ((synWtType == SYN_PLASTIC) && (enableSimLogs))
    //         storeWeights(newInfo->grpDest, newInfo->grpSrc, "logs");
    newInfo = newInfo->next;
  }

  fflush(fp);
}

void SNN::printConnectionInfo2(FILE * const fpg)
{
  ConnectConfig* newInfo = connectBegin;

	fprintf(fpg, "#Connection Information \n");
	fprintf(fpg, "#(e.g. from => to : approx. # of post (numPostSynapses) : approx. # of pre-synaptic (numPreSynapses) : weights.. : type plastic or fixed : max and min axonal delay\n");
	while(newInfo) {
		bool synWtType	= GET_FIXED_PLASTIC(newInfo->connProp);
		fprintf(fpg, " %d => %d : %s => %s : numPostSynapses %d : numPreSynapses %d : initWeight %f : maxWeight %3.3f : type %s : maxDelay %d : minDelay %d\n",
				newInfo->grpSrc, newInfo->grpDest, groupInfo[newInfo->grpSrc].Name.c_str(), groupInfo[newInfo->grpDest].Name.c_str(),
				newInfo->numPostSynapses, newInfo->numPreSynapses, newInfo->initWt,   newInfo->maxWt,
				(synWtType == SYN_PLASTIC)?"plastic":"fixed", newInfo->maxDelay, newInfo->minDelay);
		newInfo = newInfo->next;
	}
	fprintf(fpg, "\n");
	fflush(fpg);
}

// new print connection info, akin to printGroupInfo
void SNN::printConnectionInfo(short int connId) {
	ConnectConfig* connInfo = getConnectInfo(connId);

	KERNEL_INFO("Connection ID %d: %s(%d) => %s(%d)", connId, groupInfo[connInfo->grpSrc].Name.c_str(),
		connInfo->grpSrc, groupInfo[connInfo->grpDest].Name.c_str(), connInfo->grpDest);
	KERNEL_INFO("  - Type                       = %s", GET_FIXED_PLASTIC(connInfo->connProp)==SYN_PLASTIC?" PLASTIC":"   FIXED")
	KERNEL_INFO("  - Min weight                 = %8.5f", 0.0f); // \TODO
	KERNEL_INFO("  - Max weight                 = %8.5f", fabs(connInfo->maxWt));
	KERNEL_INFO("  - Initial weight             = %8.5f", fabs(connInfo->initWt));
	KERNEL_INFO("  - Min delay                  = %8d", connInfo->minDelay);
	KERNEL_INFO("  - Max delay                  = %8d", connInfo->maxDelay);
  KERNEL_INFO("  - Radius X                   = %8.2f", connInfo->radX);
  KERNEL_INFO("  - Radius Y                   = %8.2f", connInfo->radY);
  KERNEL_INFO("  - Radius Z                   = %8.2f", connInfo->radZ);

	float avgPostM = connInfo->numberOfConnections/groupConfig[connInfo->grpSrc].SizeN;
	float avgPreM  = connInfo->numberOfConnections/groupConfig[connInfo->grpDest].SizeN;
	KERNEL_INFO("  - Avg numPreSynapses         = %8d", (int)avgPreM );
	KERNEL_INFO("  - Avg numPostSynapses        = %8d", (int)avgPostM );
}

void SNN::printGroupInfo(int grpId) {
	KERNEL_INFO("Group %s(%d): ", groupInfo[grpId].Name.c_str(), grpId);
	KERNEL_INFO("  - Type                       =  %s", isExcitatoryGroup(grpId) ? "  EXCIT" :
		(isInhibitoryGroup(grpId) ? "  INHIB" : (isPoissonGroup(grpId)?" POISSON" :
		(isDopaminergicGroup(grpId) ? "  DOPAM" : " UNKNOWN"))) );
	KERNEL_INFO("  - Size                       = %8d", groupConfig[grpId].SizeN);
	KERNEL_INFO("  - Start Id                   = %8d", groupConfig[grpId].StartN);
	KERNEL_INFO("  - End Id                     = %8d", groupConfig[grpId].EndN);
	KERNEL_INFO("  - numPostSynapses            = %8d", groupConfig[grpId].numPostSynapses);
	KERNEL_INFO("  - numPreSynapses             = %8d", groupConfig[grpId].numPreSynapses);

	//if (doneReorganization) {
	//	KERNEL_INFO("  - Avg post connections       = %8.5f", 1.0*groupInfo[grpId].numPostConn/groupConfig[grpId].SizeN);
	//	KERNEL_INFO("  - Avg pre connections        = %8.5f",  1.0*groupInfo[grpId].numPreConn/groupConfig[grpId].SizeN);
	//}

	if(groupConfig[grpId].Type&POISSON_NEURON) {
		KERNEL_INFO("  - Refractory period          = %8.5f", groupConfig[grpId].RefractPeriod);
	}

	if (groupConfig[grpId].WithSTP) {
		KERNEL_INFO("  - STP:");
		KERNEL_INFO("      - STP_A                  = %8.5f", groupConfig[grpId].STP_A);
		KERNEL_INFO("      - STP_U                  = %8.5f", groupConfig[grpId].STP_U);
		KERNEL_INFO("      - STP_tau_u              = %8d", (int) (1.0f/groupConfig[grpId].STP_tau_u_inv));
		KERNEL_INFO("      - STP_tau_x              = %8d", (int) (1.0f/groupConfig[grpId].STP_tau_x_inv));
	}

	if(groupConfig[grpId].WithSTDP) {
		KERNEL_INFO("  - STDP:")
		KERNEL_INFO("      - E-STDP TYPE            = %s",     groupConfig[grpId].WithESTDPtype==STANDARD? "STANDARD" :
			(groupConfig[grpId].WithESTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("      - I-STDP TYPE            = %s",     groupConfig[grpId].WithISTDPtype==STANDARD? "STANDARD" :
			(groupConfig[grpId].WithISTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("      - ALPHA_PLUS_EXC         = %8.5f", groupConfig[grpId].ALPHA_PLUS_EXC);
		KERNEL_INFO("      - ALPHA_MINUS_EXC        = %8.5f", groupConfig[grpId].ALPHA_MINUS_EXC);
		KERNEL_INFO("      - TAU_PLUS_INV_EXC       = %8.5f", groupConfig[grpId].TAU_PLUS_INV_EXC);
		KERNEL_INFO("      - TAU_MINUS_INV_EXC      = %8.5f", groupConfig[grpId].TAU_MINUS_INV_EXC);
		KERNEL_INFO("      - BETA_LTP               = %8.5f", groupConfig[grpId].BETA_LTP);
		KERNEL_INFO("      - BETA_LTD               = %8.5f", groupConfig[grpId].BETA_LTD);
		KERNEL_INFO("      - LAMBDA                 = %8.5f", groupConfig[grpId].LAMBDA);
		KERNEL_INFO("      - DELTA                  = %8.5f", groupConfig[grpId].DELTA);
	}
}

void SNN::printGroupInfo2(FILE* const fpg)
{
  fprintf(fpg, "#Group Information\n");
  for(int g=0; g < numGrp; g++) {
    fprintf(fpg, "group %d: name %s : type %s %s %s %s %s: size %d : start %d : end %d \n",
      g, groupInfo[g].Name.c_str(),
      (groupConfig[g].Type&POISSON_NEURON) ? "poisson " : "",
      (groupConfig[g].Type&TARGET_AMPA) ? "AMPA" : "",
      (groupConfig[g].Type&TARGET_NMDA) ? "NMDA" : "",
      (groupConfig[g].Type&TARGET_GABAa) ? "GABAa" : "",
      (groupConfig[g].Type&TARGET_GABAb) ? "GABAb" : "",
      groupConfig[g].SizeN,
      groupConfig[g].StartN,
      groupConfig[g].EndN);
  }
  fprintf(fpg, "\n");
  fflush(fpg);
}

//! \deprecated
void SNN::printParameters(FILE* const fp) {
	KERNEL_WARN("printParameters is deprecated");
/*	assert(fp!=NULL);
	printGroupInfo(fp);
	printConnectionInfo(fp);*/
}

// print all post-connections...
void SNN::printPostConnection(FILE * const fp)
{
  if(fp) fprintf(fp, "PRINTING POST-SYNAPTIC CONNECTION TOPOLOGY\n");
  if(fp) fprintf(fp, "(((((((((((((((((((((())))))))))))))))))))))\n");
  for(int i=0; i < numGrp; i++)
    printPostConnection(i,fp);
}

// print all the pre-connections...
void SNN::printPreConnection(FILE * const fp)
{
  if(fp) fprintf(fp, "PRINTING PRE-SYNAPTIC CONNECTION TOPOLOGY\n");
  if(fp) fprintf(fp, "(((((((((((((((((((((())))))))))))))))))))))\n");
  for(int i=0; i < numGrp; i++)
    printPreConnection(i,fp);
}

// print the connection info of grpId
int SNN::printPostConnection2(int grpId, FILE* const fpg)
{
  int maxLength = -1;
  for(int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
    fprintf(fpg, " id %d : group %d : postlength %d ", i, findGrpId(i), snnRuntimeData.Npost[i]);
    // fetch the starting position
    post_info_t* postIds = &(snnRuntimeData.postSynapticIds[snnRuntimeData.cumulativePost[i]]);
    for(int j=0; j <= maxDelay_; j++) {
      int len   = snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+j].delay_length;
      int start = snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+j].delay_index_start;
      for(int k=start; k < len; k++) {
	int post_nid = GET_CONN_NEURON_ID((*postIds));
	//					int post_gid = GET_CONN_GRP_ID((*postIds));
	fprintf(fpg, " : %d,%d ", post_nid, j);
	postIds++;
      }
      if (snnRuntimeData.Npost[i] > maxLength)
	maxLength = snnRuntimeData.Npost[i];
      if ((start+len) >= snnRuntimeData.Npost[i])
	break;
    }
    fprintf(fpg, "\n");
  }
  fflush(fpg);
  return maxLength;
}

void SNN::printNetworkInfo(FILE* const fpg) {
	int maxLengthPost = -1;
	int maxLengthPre  = -1;
	printGroupInfo2(fpg);
	printConnectionInfo2(fpg);
	fprintf(fpg, "#Flat Network Info Format \n");
	fprintf(fpg, "#(neuron id : length (number of connections) : neuron_id0,delay0 : neuron_id1,delay1 : ... \n");
	for(int g=0; g < numGrp; g++) {
		int postM = printPostConnection2(g, fpg);
		int numPreSynapses  = printPreConnection2(g, fpg);
		if (postM > maxLengthPost)
			maxLengthPost = postM;
		if (numPreSynapses > maxLengthPre)
			maxLengthPre = numPreSynapses;
	}
	fflush(fpg);
	fclose(fpg);
}

void SNN::printFiringRate(char *fname)
{
  static int printCnt = 0;
  FILE *fpg;
  std::string strFname;
  if (fname == NULL)
    strFname = networkName_;
  else
    strFname = fname;

  strFname += ".stat";
  if(printCnt==0)
    fpg = fopen(strFname.c_str(), "w");
  else
    fpg = fopen(strFname.c_str(), "a");

  fprintf(fpg, "#Average Firing Rate\n");
  if(printCnt==0) {
    fprintf(fpg, "#network %s: size = %d\n", networkName_.c_str(), numN);
    for(int grpId=0; grpId < numGrp; grpId++) {
      fprintf(fpg, "#group %d: name %s : size = %d\n", grpId, groupInfo[grpId].Name.c_str(), groupConfig[grpId].SizeN);
    }
  }
  fprintf(fpg, "Time %d ms\n", simTime);
  fprintf(fpg, "#activeNeurons ( <= 1.0) = fraction of neuron in the given group that are firing more than 1Hz\n");
  fprintf(fpg, "#avgFiring (in Hz) = Average firing rate of activeNeurons in given group\n");
  for(int grpId=0; grpId < numGrp; grpId++) {
    fprintf(fpg, "group %d : \t", grpId);
    int   totSpike = 0;
    int   activeCnt  = 0;
    for(int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
      if (snnRuntimeData.nSpikeCnt[i] >= 1.0) {
	totSpike += snnRuntimeData.nSpikeCnt[i];
	activeCnt++;
      }
    }
    fprintf(fpg, " activeNeurons = %3.3f : avgFiring = %3.3f  \n", activeCnt*1.0/groupConfig[grpId].SizeN, (activeCnt==0)?0.0:totSpike*1.0/activeCnt);
  }
  printCnt++;
  fflush(fpg);
  fclose(fpg);
}

// print the connection info of grpId
void SNN::printPostConnection(int grpId, FILE* const fp)
{
  for(int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
    if(fp) fprintf(fp, " %3d ( %3d ) : \t", i, snnRuntimeData.Npost[i]);
    // fetch the starting position
    post_info_t* postIds = &(snnRuntimeData.postSynapticIds[snnRuntimeData.cumulativePost[i]]);
    int  offset  = snnRuntimeData.cumulativePost[i];
    for(int j=0; j < snnRuntimeData.Npost[i]; j++, postIds++) {
      int post_nid = GET_CONN_NEURON_ID((*postIds));
      int post_gid = GET_CONN_GRP_ID((*postIds));
      assert( findGrpId(post_nid) == post_gid);
      if(fp) fprintf(fp, " %3d ( maxDelay_=%3d, Grp=%3d) ", post_nid, tmp_SynapticDelay[offset+j], post_gid);
    }
    if(fp) fprintf(fp, "\n");
    if(fp) fprintf(fp, " Delay ( %3d ) : ", i);
    for(int j=0; j < maxDelay_; j++) {
      if(fp) fprintf(fp, " %d,%d ", snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+j].delay_length,
		     snnRuntimeData.postDelayInfo[i*(maxDelay_+1)+j].delay_index_start);
    }
    if(fp) fprintf(fp, "\n");
  }
}

int SNN::printPreConnection2(int grpId, FILE* const fpg)
{
	int maxLength = -1;
	for(int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
		fprintf(fpg, " id %d : group %d : prelength %d ", i, findGrpId(i), snnRuntimeData.Npre[i]);
		post_info_t* preIds = &(snnRuntimeData.preSynapticIds[snnRuntimeData.cumulativePre[i]]);
		for(int j=0; j < snnRuntimeData.Npre[i]; j++, preIds++) {
			//if (doneReorganization && (!memoryOptimized))
			//	fprintf(fpg, ": %d,%s", GET_CONN_NEURON_ID((*preIds)), (j < snnRuntimeData.Npre_plastic[i])?"P":"F");
		}
		
		if ( snnRuntimeData.Npre[i] > maxLength)
			maxLength = snnRuntimeData.Npre[i];
		fprintf(fpg, "\n");
	}
	return maxLength;
}

void SNN::printPreConnection(int grpId, FILE* const fp)
{
	for(int i=groupConfig[grpId].StartN; i<=groupConfig[grpId].EndN; i++) {
		if(fp) fprintf(fp, " %d ( preCnt=%d, prePlastic=%d ) : (id => (wt, maxWt),(preId, P/F)\n\t", i, snnRuntimeData.Npre[i], snnRuntimeData.Npre_plastic[i]);
		post_info_t* preIds = &(snnRuntimeData.preSynapticIds[snnRuntimeData.cumulativePre[i]]);
		int  pos_i  = snnRuntimeData.cumulativePre[i];
		for(int j=0; j < snnRuntimeData.Npre[i]; j++, pos_i++, preIds++) {
			if(fp) fprintf(fp,  "  %d => (%f, %f)", j, snnRuntimeData.wt[pos_i], snnRuntimeData.maxSynWt[pos_i]);
			//if(doneReorganization && (!memoryOptimized))
			//	if(fp) fprintf(fp, ",(%d, %s)", GET_CONN_NEURON_ID((*preIds)), (j < snnRuntimeData.Npre_plastic[i])?"P":"F");
		}
		
		if(fp) fprintf(fp, "\n");
	}
}


void SNN::printNeuronState(int grpId, FILE* const fp)
{
  if (simMode_==GPU_MODE) {
    copyNeuronState(&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, grpId);
  }

  fprintf(fp, "[MODE=%s] ", simMode_string[simMode_]);
  fprintf(fp, "Group %s (%d) Neuron State Information (totSpike=%d, poissSpike=%d)\n",
	  groupInfo[grpId].Name.c_str(), grpId, spikeCountAllHost, nPoissonSpikes);

  // poisson group does not have default neuron state
  if(groupConfig[grpId].Type&POISSON_NEURON) {
    fprintf(fp, "t=%d msec ", simTime);
    int totSpikes = 0;
    for (int nid=groupConfig[grpId].StartN; nid <= groupConfig[grpId].EndN; nid++) {
      totSpikes += snnRuntimeData.nSpikeCnt[nid];
      fprintf(fp, "%d ", snnRuntimeData.nSpikeCnt[nid]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "TotalSpikes [grp=%d, %s]=  %d\n", grpId, groupInfo[grpId].Name.c_str(), totSpikes);
    return;
  }

  int totSpikes = 0;
  for (int nid=groupConfig[grpId].StartN; nid <= groupConfig[grpId].EndN; nid++) {
    // copy the neuron firing information from the GPU to the CPU...
    totSpikes += snnRuntimeData.nSpikeCnt[nid];
    if(!sim_with_conductances) {
      if(snnRuntimeData.current[nid] != 0.0)
	fprintf(fp, "t=%d id=%d v=%+3.3f u=%+3.3f I=%+3.3f nSpikes=%d\n", simTime, nid,
		snnRuntimeData.voltage[nid],     snnRuntimeData.recovery[nid],      snnRuntimeData.current[nid],
		snnRuntimeData.nSpikeCnt[nid]);
    }
    else {
      if (snnRuntimeData.gAMPA[nid]+ snnRuntimeData.gNMDA[nid]+snnRuntimeData.gGABAa[nid]+snnRuntimeData.gGABAb[nid] != 0.0)
	fprintf(fp, "t=%d id=%d v=%+3.3f u=%+3.3f I=%+3.3f gAMPA=%2.5f gNMDA=%2.5f gGABAa=%2.5f gGABAb=%2.5f nSpikes=%d\n", simTime, nid,
		snnRuntimeData.voltage[nid],     snnRuntimeData.recovery[nid],      snnRuntimeData.current[nid], snnRuntimeData.gAMPA[nid],
		snnRuntimeData.gNMDA[nid], snnRuntimeData.gGABAa[nid], snnRuntimeData.gGABAb[nid], snnRuntimeData.nSpikeCnt[nid]);
    }
  }
  fprintf(fp, "TotalSpikes [grp=%d, %s] = %d\n", grpId, groupInfo[grpId].Name.c_str(), totSpikes);
  fprintf(fp, "\n");
  fflush(fp);
}

// TODO: make KERNEL_INFO(), don't write to fpInf_
void SNN::printWeights(int preGrpId, int postGrpId) {
	int preA, preZ, postA, postZ;
	if (preGrpId==ALL) {
		preA = 0;
		preZ = numGrp;
	} else {
		preA = preGrpId;
		preZ = preGrpId+1;
	}
	if (postGrpId==ALL) {
		postA = 0;
		postZ = numGrp;
	} else {
		postA = postGrpId;
		postZ = postGrpId+1;
	}

	for (int gPost=postA; gPost<postZ; gPost++) {
		// for each postsynaptic group

		fprintf(fpInf_,"Synapses from %s to %s (+/- change in last %d ms)\n",
			(preGrpId==ALL)?"ALL":groupInfo[preGrpId].Name.c_str(), groupInfo[gPost].Name.c_str(), wtANDwtChangeUpdateInterval_);

		if (simMode_ == GPU_MODE) {
			copyWeightState (&snnRuntimeData, &gpuRuntimeData, cudaMemcpyDeviceToHost, false, gPost);
		}

		int i=groupConfig[gPost].StartN;
		unsigned int offset = snnRuntimeData.cumulativePre[i];
		for (int j=0; j<snnRuntimeData.Npre[i]; j++) {
			int gPre = snnRuntimeData.grpIds[j];
			if (gPre<preA || gPre>preZ)
				continue;

			float wt  = snnRuntimeData.wt[offset+j];
			if (!groupConfig[gPost].FixedInputWts) {
				float wtC = snnRuntimeData.wtChange[offset+j];
				fprintf(fpInf_, "%s%1.3f (%s%1.3f)\t", wt<0?"":" ", wt, wtC<0?"":"+", wtC);
			} else {
				fprintf(fpInf_, "%s%1.3f \t\t", wt<0?"":" ", wt);
			}
		}
		fprintf(fpInf_,"\n");
	}
}



