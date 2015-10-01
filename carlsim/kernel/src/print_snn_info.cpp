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
#include <connection_monitor_core.h>
#include <group_monitor_core.h>

void SNN::printMemoryInfo(FILE* const fp) {
	if (snnState == CONFIG_SNN || snnState == COMPILED_SNN || snnState == PARTITIONED_SNN) {
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
	for(int g=0; g < numGroups; g++) {
		int TNpost=0;
		int TNpre=0;
		int TNpre_plastic=0;
		
		for(int i=groupConfigs[0][g].StartN; i <= groupConfigs[0][g].EndN; i++) {
			TNpost += managerRuntimeData.Npost[i];
			TNpre  += managerRuntimeData.Npre[i];
			TNpre_plastic += managerRuntimeData.Npre_plastic[i];
		}
		
	fprintf(fp, "%s Group (num_neurons=%5d): \n\t\tNpost[%2d] = %3d, Npre[%2d]=%3d Npre_plastic[%2d]=%3d \n\t\tcumPre[%5d]=%5d cumPre[%5d]=%5d cumPost[%5d]=%5d cumPost[%5d]=%5d \n",
		groupInfo[g].Name.c_str(), groupConfigs[0][g].SizeN, g, TNpost/groupConfigs[0][g].SizeN, g, TNpre/groupConfigs[0][g].SizeN, g, TNpre_plastic/groupConfigs[0][g].SizeN,
		groupConfigs[0][g].StartN, managerRuntimeData.cumulativePre[groupConfigs[0][g].StartN],  groupConfigs[0][g].EndN, managerRuntimeData.cumulativePre[groupConfigs[0][g].EndN],
		groupConfigs[0][g].StartN, managerRuntimeData.cumulativePost[groupConfigs[0][g].StartN], groupConfigs[0][g].EndN, managerRuntimeData.cumulativePost[groupConfigs[0][g].EndN]);
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

void SNN::printStatusSpikeMonitor(int gGrpId) {
	if (gGrpId==ALL) {
		for (int g = 0; g < numGroups; g++) {
			printStatusSpikeMonitor(g);
		}
	} else {
		int netId = groupConfigMap[gGrpId].netId;
		int lGrpId = groupConfigMap[gGrpId].localGrpId;
		int monitorId = groupConfigs[netId][lGrpId].SpikeMonitorId;
		
		if (monitorId == -1) return;

		// in GPU mode, need to get data from device first
		if (simMode_ == GPU_MODE)
			fetchNeuronSpikeCount(gGrpId);

		// \TODO nSpikeCnt should really be a member of the SpikeMonitor object that gets populated if
		// printRunSummary is true or mode==COUNT.....
		// so then we can use spkMonObj->print(false); // showSpikeTimes==false
		int grpSpk = 0;
		for (int gNId = groupConfigMap[gGrpId].StartN; gNId <= groupConfigMap[gGrpId].EndN; gNId++)
			grpSpk += managerRuntimeData.nSpikeCnt[gNId]; // add up all neuronal spike counts

		// infer run duration by measuring how much time has passed since the last run summary was printed
		int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %dms (%.2f +/- %.2f Hz)",
				(float)(simTime/1000.0f),
				groupInfo[gGrpId].Name.c_str(),
				gGrpId,
				0,
				0,
				0.0f,
				0.0f);
		} else {
			// if some time has passed since last print
			float meanRate = grpSpk * 1000.0f / runDurationMs / groupConfigMap[gGrpId].SizeN;
			float std = 0.0f;
			if (groupConfigMap[gGrpId].SizeN > 1) {
				for (int gNId = groupConfigMap[gGrpId].StartN; gNId <= groupConfigMap[gGrpId].EndN; gNId++) {
					float neurRate = managerRuntimeData.nSpikeCnt[gNId] * 1000.0f / runDurationMs;
					std += (neurRate - meanRate) * (neurRate - meanRate);
				}
				std = sqrt(std / (groupConfigs[netId][lGrpId].SizeN - 1.0));
			}
	
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %ums (%.2f +/- %.2f Hz)",
				simTime / 1000.0f,
				groupInfo[gGrpId].Name.c_str(),
				gGrpId,
				grpSpk,
				runDurationMs,
				meanRate,
				std);
		}
	}
}

void SNN::printStatusGroupMonitor(int grpId) {
	if (grpId == ALL) {
		for (int g = 0; g < numGroups; g++) {
			printStatusGroupMonitor(g);
		}
	} else {
		int netId = groupConfigMap[grpId].netId;
		int lGrpId = groupConfigMap[grpId].localGrpId;
		int monitorId = groupConfigs[netId][lGrpId].GroupMonitorId;

		if (monitorId == -1) return;

		std::vector<int> peakTimeVector = groupMonCoreList[monitorId]->getPeakTimeVector();
		int numPeaks = peakTimeVector.size();

		// infer run duration by measuring how much time has passed since the last run summary was printed
		int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %dms",
				simTime/1000.0f,
				groupInfo[grpId].Name.c_str(),
				grpId,
				0,
				0);
		} else {
			// if some time has passed since last print
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %ums",
				simTime/1000.0f,
				groupInfo[grpId].Name.c_str(),
				grpId,
				numPeaks,
				runDurationMs);
		}
	}
}

// new print connection info, akin to printGroupInfo
void SNN::printConnectionInfo(short int connId) {
	ConnectConfig connConfig = connectConfigMap[connId];

	KERNEL_INFO("Connection ID %d: %s(%d) => %s(%d)", connId, groupInfo[connConfig.grpSrc].Name.c_str(),
		connConfig.grpSrc, groupInfo[connConfig.grpDest].Name.c_str(), connConfig.grpDest);
	KERNEL_INFO("  - Type                       = %s", GET_FIXED_PLASTIC(connConfig.connProp)==SYN_PLASTIC?" PLASTIC":"   FIXED")
	KERNEL_INFO("  - Min weight                 = %8.5f", 0.0f); // \TODO
	KERNEL_INFO("  - Max weight                 = %8.5f", fabs(connConfig.maxWt));
	KERNEL_INFO("  - Initial weight             = %8.5f", fabs(connConfig.initWt));
	KERNEL_INFO("  - Min delay                  = %8d", connConfig.minDelay);
	KERNEL_INFO("  - Max delay                  = %8d", connConfig.maxDelay);
	KERNEL_INFO("  - Radius X                   = %8.2f", connConfig.radX);
	KERNEL_INFO("  - Radius Y                   = %8.2f", connConfig.radY);
	KERNEL_INFO("  - Radius Z                   = %8.2f", connConfig.radZ);

	float avgPostM = ((float)connConfig.numberOfConnections)/groupConfigMap[connConfig.grpSrc].SizeN;
	float avgPreM  = ((float)connConfig.numberOfConnections)/groupConfigMap[connConfig.grpDest].SizeN;
	KERNEL_INFO("  - Avg numPreSynapses         = %8.2f", avgPreM );
	KERNEL_INFO("  - Avg numPostSynapses        = %8.2f", avgPostM );
}

// print connection info, akin to printGroupInfo
void SNN::printConnectionInfo(int netId, std::list<ConnectConfig>::iterator connIt) {

	KERNEL_INFO("  |-+ %s Connection Id %d: %s(%d) => %s(%d)", netId == groupConfigMap[connIt->grpDest].netId ? "Local" : "External", connIt->connId,
		groupInfo[connIt->grpSrc].Name.c_str(), connIt->grpSrc,
		groupInfo[connIt->grpDest].Name.c_str(), connIt->grpDest);
	KERNEL_INFO("    |- Type                       = %s", GET_FIXED_PLASTIC(connIt->connProp)==SYN_PLASTIC?" PLASTIC":"   FIXED")
	KERNEL_INFO("    |- Min weight                 = %8.5f", 0.0f); // \TODO
	KERNEL_INFO("    |- Max weight                 = %8.5f", fabs(connIt->maxWt));
	KERNEL_INFO("    |- Initial weight             = %8.5f", fabs(connIt->initWt));
	KERNEL_INFO("    |- Min delay                  = %8d", connIt->minDelay);
	KERNEL_INFO("    |- Max delay                  = %8d", connIt->maxDelay);
	KERNEL_INFO("    |- Radius X                   = %8.2f", connIt->radX);
	KERNEL_INFO("    |- Radius Y                   = %8.2f", connIt->radY);
	KERNEL_INFO("    |- Radius Z                   = %8.2f", connIt->radZ);

	float avgPostM = ((float)connIt->numberOfConnections)/groupConfigMap[connIt->grpSrc].SizeN;
	float avgPreM  = ((float)connIt->numberOfConnections)/groupConfigMap[connIt->grpDest].SizeN;
	KERNEL_INFO("    |- Avg numPreSynapses         = %8.2f", avgPreM );
	KERNEL_INFO("    |- Avg numPostSynapses        = %8.2f", avgPostM );
}

void SNN::printGroupInfo(int grpId) {
	KERNEL_INFO("Group %s(%d): ", groupInfo[grpId].Name.c_str(), grpId);
	KERNEL_INFO("  - Type                       =  %s", isExcitatoryGroup(grpId) ? "  EXCIT" :
		(isInhibitoryGroup(grpId) ? "  INHIB" : (isPoissonGroup(grpId)?" POISSON" :
		(isDopaminergicGroup(grpId) ? "  DOPAM" : " UNKNOWN"))) );
	KERNEL_INFO("  - Size                       = %8d", groupConfigMap[grpId].SizeN);
	KERNEL_INFO("  - Start Id                   = %8d", groupConfigMap[grpId].StartN);
	KERNEL_INFO("  - End Id                     = %8d", groupConfigMap[grpId].EndN);
	KERNEL_INFO("  - numPostSynapses            = %8d", groupConfigMap[grpId].numPostSynapses);
	KERNEL_INFO("  - numPreSynapses             = %8d", groupConfigMap[grpId].numPreSynapses);

	if (snnState == EXECUTABLE_SNN) {
		KERNEL_INFO("  - Avg post connections       = %8.5f", ((float)groupInfo[grpId].numPostConn)/groupConfigMap[grpId].SizeN);
		KERNEL_INFO("  - Avg pre connections        = %8.5f",  ((float)groupInfo[grpId].numPreConn)/groupConfigMap[grpId].SizeN);
	}

	if(groupConfigMap[grpId].Type&POISSON_NEURON) {
		KERNEL_INFO("  - Refractory period          = %8.5f", groupConfigMap[grpId].RefractPeriod);
	}

	if (groupConfigMap[grpId].WithSTP) {
		KERNEL_INFO("  - STP:");
		KERNEL_INFO("      - STP_A                  = %8.5f", groupConfigMap[grpId].STP_A);
		KERNEL_INFO("      - STP_U                  = %8.5f", groupConfigMap[grpId].STP_U);
		KERNEL_INFO("      - STP_tau_u              = %8d", (int) (1.0f/groupConfigMap[grpId].STP_tau_u_inv));
		KERNEL_INFO("      - STP_tau_x              = %8d", (int) (1.0f/groupConfigMap[grpId].STP_tau_x_inv));
	}

	if(groupConfigMap[grpId].WithSTDP) {
		KERNEL_INFO("  - STDP:")
		KERNEL_INFO("      - E-STDP TYPE            = %s",     groupConfigMap[grpId].WithESTDPtype==STANDARD? "STANDARD" :
			(groupConfigMap[grpId].WithESTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("      - I-STDP TYPE            = %s",     groupConfigMap[grpId].WithISTDPtype==STANDARD? "STANDARD" :
			(groupConfigMap[grpId].WithISTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("      - ALPHA_PLUS_EXC         = %8.5f", groupConfigMap[grpId].ALPHA_PLUS_EXC);
		KERNEL_INFO("      - ALPHA_MINUS_EXC        = %8.5f", groupConfigMap[grpId].ALPHA_MINUS_EXC);
		KERNEL_INFO("      - TAU_PLUS_INV_EXC       = %8.5f", groupConfigMap[grpId].TAU_PLUS_INV_EXC);
		KERNEL_INFO("      - TAU_MINUS_INV_EXC      = %8.5f", groupConfigMap[grpId].TAU_MINUS_INV_EXC);
		KERNEL_INFO("      - BETA_LTP               = %8.5f", groupConfigMap[grpId].BETA_LTP);
		KERNEL_INFO("      - BETA_LTD               = %8.5f", groupConfigMap[grpId].BETA_LTD);
		KERNEL_INFO("      - LAMBDA                 = %8.5f", groupConfigMap[grpId].LAMBDA);
		KERNEL_INFO("      - DELTA                  = %8.5f", groupConfigMap[grpId].DELTA);
	}
}

void SNN::printGroupInfo(int netId, std::list<GroupConfigRT>::iterator grpIt) {
	KERNEL_INFO("  |-+ %s Group %s(G:%d,L:%d): ", netId == grpIt->netId ? "Local" : "External", groupInfo[grpIt->grpId].Name.c_str(), grpIt->grpId, grpIt->localGrpId);
	KERNEL_INFO("    |- Type                       =  %s", isExcitatoryGroup(grpIt->grpId) ? "  EXCIT" :
		(isInhibitoryGroup(grpIt->grpId) ? "  INHIB" : (isPoissonGroup(grpIt->grpId)?" POISSON" :
		(isDopaminergicGroup(grpIt->grpId) ? "  DOPAM" : " UNKNOWN"))) );
	KERNEL_INFO("    |- Size                       = %8d", grpIt->SizeN);
	KERNEL_INFO("    |- Start Id                   = (G:%d,L:%d)", grpIt->StartN, grpIt->localStartN);
	KERNEL_INFO("    |- End Id                     = (G:%d,L:%d)", grpIt->EndN, grpIt->localEndN);
	KERNEL_INFO("    |- numPostSynapses            = %8d", grpIt->numPostSynapses);
	KERNEL_INFO("    |- numPreSynapses             = %8d", grpIt->numPreSynapses);

	//if (snnState == EXECUTABLE_SNN) {
	//	KERNEL_INFO("  - Avg post connections       = %8.5f", ((float)groupInfo[grpId].numPostConn)/groupConfigMap[grpId].SizeN);
	//	KERNEL_INFO("  - Avg pre connections        = %8.5f",  ((float)groupInfo[grpId].numPreConn)/groupConfigMap[grpId].SizeN);
	//}

	if(grpIt->Type&POISSON_NEURON) {
		KERNEL_INFO("    |- Refractory period          = %8.5f", grpIt->RefractPeriod);
	}

	if (grpIt->WithSTP) {
		KERNEL_INFO("    |- STP:");
		KERNEL_INFO("        |- STP_A                  = %8.5f", grpIt->STP_A);
		KERNEL_INFO("        |- STP_U                  = %8.5f", grpIt->STP_U);
		KERNEL_INFO("        |- STP_tau_u              = %8d", (int) (1.0f/grpIt->STP_tau_u_inv));
		KERNEL_INFO("        |- STP_tau_x              = %8d", (int) (1.0f/grpIt->STP_tau_x_inv));
	}

	if(grpIt->WithSTDP) {
		KERNEL_INFO("    |- STDP:")
		KERNEL_INFO("        |- E-STDP TYPE            = %s",     grpIt->WithESTDPtype==STANDARD? "STANDARD" :
			(grpIt->WithESTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("        |- I-STDP TYPE            = %s",     grpIt->WithISTDPtype==STANDARD? "STANDARD" :
			(grpIt->WithISTDPtype==DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("        |- ALPHA_PLUS_EXC         = %8.5f", grpIt->ALPHA_PLUS_EXC);
		KERNEL_INFO("        |- ALPHA_MINUS_EXC        = %8.5f", grpIt->ALPHA_MINUS_EXC);
		KERNEL_INFO("        |- TAU_PLUS_INV_EXC       = %8.5f", grpIt->TAU_PLUS_INV_EXC);
		KERNEL_INFO("        |- TAU_MINUS_INV_EXC      = %8.5f", grpIt->TAU_MINUS_INV_EXC);
		KERNEL_INFO("        |- BETA_LTP               = %8.5f", grpIt->BETA_LTP);
		KERNEL_INFO("        |- BETA_LTD               = %8.5f", grpIt->BETA_LTD);
		KERNEL_INFO("        |- LAMBDA                 = %8.5f", grpIt->LAMBDA);
		KERNEL_INFO("        |- DELTA                  = %8.5f", grpIt->DELTA);
	}
}

void SNN::printGroupInfo2(FILE* const fpg)
{
  fprintf(fpg, "#Group Information\n");
  for(int g=0; g < numGroups; g++) {
    fprintf(fpg, "group %d: name %s : type %s %s %s %s %s: size %d : start %d : end %d \n",
      g, groupInfo[g].Name.c_str(),
      (groupConfigs[0][g].Type&POISSON_NEURON) ? "poisson " : "",
      (groupConfigs[0][g].Type&TARGET_AMPA) ? "AMPA" : "",
      (groupConfigs[0][g].Type&TARGET_NMDA) ? "NMDA" : "",
      (groupConfigs[0][g].Type&TARGET_GABAa) ? "GABAa" : "",
      (groupConfigs[0][g].Type&TARGET_GABAb) ? "GABAb" : "",
      groupConfigs[0][g].SizeN,
      groupConfigs[0][g].StartN,
      groupConfigs[0][g].EndN);
  }
  fprintf(fpg, "\n");
  fflush(fpg);
}

