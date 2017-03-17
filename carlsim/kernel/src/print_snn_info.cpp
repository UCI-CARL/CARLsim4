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
#include <connection_monitor_core.h>
#include <group_monitor_core.h>

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
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int monitorId = groupConfigMDMap[gGrpId].spikeMonitorId;
		
		if (monitorId == -1) return;

		// copy data to the manager runtime
		fetchNeuronSpikeCount(gGrpId);

		// \TODO nSpikeCnt should really be a member of the SpikeMonitor object that gets populated if
		// printRunSummary is true or mode==COUNT.....
		// so then we can use spkMonObj->print(false); // showSpikeTimes==false
		int grpSpk = 0;
		for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++)
			grpSpk += managerRuntimeData.nSpikeCnt[gNId]; // add up all neuronal spike counts

		// infer run duration by measuring how much time has passed since the last run summary was printed
		int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %dms (%.2f +/- %.2f Hz)",
				(float)(simTime / 1000.0f),
				groupConfigMap[gGrpId].grpName.c_str(),
				gGrpId,
				0,
				0,
				0.0f,
				0.0f);
		} else {
			// if some time has passed since last print
			float meanRate = grpSpk * 1000.0f / runDurationMs / groupConfigMap[gGrpId].numN;
			float std = 0.0f;
			if (groupConfigMap[gGrpId].numN > 1) {
				for (int gNId = groupConfigMDMap[gGrpId].gStartN; gNId <= groupConfigMDMap[gGrpId].gEndN; gNId++) {
					float neurRate = managerRuntimeData.nSpikeCnt[gNId] * 1000.0f / runDurationMs;
					std += (neurRate - meanRate) * (neurRate - meanRate);
				}
				std = sqrt(std / (groupConfigs[netId][lGrpId].numN - 1.0));
			}
	
			KERNEL_INFO("(t=%.3fs) SpikeMonitor for group %s(%d) has %d spikes in %ums (%.2f +/- %.2f Hz)",
				simTime / 1000.0f,
				groupConfigMap[gGrpId].grpName.c_str(),
				gGrpId,
				grpSpk,
				runDurationMs,
				meanRate,
				std);
		}
	}
}

void SNN::printStatusGroupMonitor(int gGrpId) {
	if (gGrpId == ALL) {
		for (int g = 0; g < numGroups; g++) {
			printStatusGroupMonitor(g);
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;
		int monitorId = groupConfigMDMap[gGrpId].groupMonitorId;

		if (monitorId == -1) return;

		std::vector<int> peakTimeVector = groupMonCoreList[monitorId]->getPeakTimeVector();
		int numPeaks = peakTimeVector.size();

		// infer run duration by measuring how much time has passed since the last run summary was printed
		int runDurationMs = simTime - simTimeLastRunSummary;

		if (simTime <= simTimeLastRunSummary) {
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %dms",
				simTime / 1000.0f,
				groupConfigMap[gGrpId].grpName.c_str(),
				gGrpId,
				0,
				0);
		} else {
			// if some time has passed since last print
			KERNEL_INFO("(t=%.3fs) GroupMonitor for group %s(%d) has %d peak(s) in %ums",
				simTime / 1000.0f,
				groupConfigMap[gGrpId].grpName.c_str(),
				gGrpId,
				numPeaks,
				runDurationMs);
		}
	}
}

// new print connection info, akin to printGroupInfo
void SNN::printConnectionInfo(short int connId) {
	ConnectConfig connConfig = connectConfigMap[connId];

	KERNEL_INFO("Connection ID %d: %s(%d) => %s(%d)", connId, groupConfigMap[connConfig.grpSrc].grpName.c_str(),
		connConfig.grpSrc, groupConfigMap[connConfig.grpDest].grpName.c_str(), connConfig.grpDest);
	KERNEL_INFO("  - Type                       = %s", GET_FIXED_PLASTIC(connConfig.connProp)==SYN_PLASTIC?" PLASTIC":"   FIXED")
	KERNEL_INFO("  - Min weight                 = %8.5f", 0.0f); // \TODO
	KERNEL_INFO("  - Max weight                 = %8.5f", fabs(connConfig.maxWt));
	KERNEL_INFO("  - Initial weight             = %8.5f", fabs(connConfig.initWt));
	KERNEL_INFO("  - Min delay                  = %8d", connConfig.minDelay);
	KERNEL_INFO("  - Max delay                  = %8d", connConfig.maxDelay);
	KERNEL_INFO("  - Radius X                   = %8.2f", connConfig.connRadius.radX);
	KERNEL_INFO("  - Radius Y                   = %8.2f", connConfig.connRadius.radY);
	KERNEL_INFO("  - Radius Z                   = %8.2f", connConfig.connRadius.radZ);
	KERNEL_INFO("  - Num of synapses            = %d", connConfig.numberOfConnections);
	float avgPostM = ((float)connConfig.numberOfConnections)/groupConfigMap[connConfig.grpSrc].numN;
	float avgPreM  = ((float)connConfig.numberOfConnections)/groupConfigMap[connConfig.grpDest].numN;
	KERNEL_INFO("  - Avg numPreSynapses         = %8.2f", avgPreM );
	KERNEL_INFO("  - Avg numPostSynapses        = %8.2f", avgPostM );
}

// print connection info, akin to printGroupInfo
void SNN::printConnectionInfo(int netId, std::list<ConnectConfig>::iterator connIt) {

	KERNEL_INFO("  |-+ %s Connection Id %d: %s(%d) => %s(%d)", netId == groupConfigMDMap[connIt->grpDest].netId ? "Local" : "External", connIt->connId,
		groupConfigMap[connIt->grpSrc].grpName.c_str(), connIt->grpSrc,
		groupConfigMap[connIt->grpDest].grpName.c_str(), connIt->grpDest);
	KERNEL_INFO("    |- Type                       = %s", GET_FIXED_PLASTIC(connIt->connProp)==SYN_PLASTIC?" PLASTIC":"   FIXED")
	KERNEL_INFO("    |- Min weight                 = %8.5f", 0.0f); // \TODO
	KERNEL_INFO("    |- Max weight                 = %8.5f", fabs(connIt->maxWt));
	KERNEL_INFO("    |- Initial weight             = %8.5f", fabs(connIt->initWt));
	KERNEL_INFO("    |- Min delay                  = %8d", connIt->minDelay);
	KERNEL_INFO("    |- Max delay                  = %8d", connIt->maxDelay);
	KERNEL_INFO("    |- Radius X                   = %8.2f", connIt->connRadius.radX);
	KERNEL_INFO("    |- Radius Y                   = %8.2f", connIt->connRadius.radY);
	KERNEL_INFO("    |- Radius Z                   = %8.2f", connIt->connRadius.radZ);
	KERNEL_INFO("    |- Num of synapses            = %d", connIt->numberOfConnections);
	float avgPostM = ((float)connIt->numberOfConnections)/groupConfigMap[connIt->grpSrc].numN;
	float avgPreM  = ((float)connIt->numberOfConnections)/groupConfigMap[connIt->grpDest].numN;
	KERNEL_INFO("    |- Avg numPreSynapses         = %8.2f", avgPreM );
	KERNEL_INFO("    |- Avg numPostSynapses        = %8.2f", avgPostM );
}

void SNN::printGroupInfo(int gGrpId) {
	KERNEL_INFO("Group %s(%d): ", groupConfigMap[gGrpId].grpName.c_str(), gGrpId);
	KERNEL_INFO("  - Type                       =  %s", isExcitatoryGroup(gGrpId) ? "  EXCIT" :
		(isInhibitoryGroup(gGrpId) ? "  INHIB" : (isPoissonGroup(gGrpId)?" POISSON" :
		(isDopaminergicGroup(gGrpId) ? "  DOPAM" : " UNKNOWN"))) );
	KERNEL_INFO("  - Size                       = %8d", groupConfigMap[gGrpId].numN);
	KERNEL_INFO("  - Start Id                   = %8d", groupConfigMDMap[gGrpId].gStartN);
	KERNEL_INFO("  - End Id                     = %8d", groupConfigMDMap[gGrpId].gEndN);
	KERNEL_INFO("  - numPostSynapses            = %8d", groupConfigMDMap[gGrpId].numPostSynapses);
	KERNEL_INFO("  - numPreSynapses             = %8d", groupConfigMDMap[gGrpId].numPreSynapses);

	if (snnState == EXECUTABLE_SNN) {
		KERNEL_INFO("  - Avg post connections       = %8.5f", ((float)groupConfigMDMap[gGrpId].numPostSynapses)/groupConfigMap[gGrpId].numN);
		KERNEL_INFO("  - Avg pre connections        = %8.5f",  ((float)groupConfigMDMap[gGrpId].numPreSynapses )/groupConfigMap[gGrpId].numN);
	}

	if(groupConfigMap[gGrpId].type & POISSON_NEURON) {
		KERNEL_INFO("  - Refractory period          = %8.5f", groupConfigMDMap[gGrpId].refractPeriod);
	}

	if (groupConfigMap[gGrpId].stpConfig.WithSTP) {
		KERNEL_INFO("  - STP:");
		KERNEL_INFO("      - STP_A                  = %8.5f", groupConfigMap[gGrpId].stpConfig.STP_A);
		KERNEL_INFO("      - STP_U                  = %8.5f", groupConfigMap[gGrpId].stpConfig.STP_U);
		KERNEL_INFO("      - STP_tau_u              = %8d", (int) (1.0f/groupConfigMap[gGrpId].stpConfig.STP_tau_u_inv));
		KERNEL_INFO("      - STP_tau_x              = %8d", (int) (1.0f/groupConfigMap[gGrpId].stpConfig.STP_tau_x_inv));
	}

	if(groupConfigMap[gGrpId].stdpConfig.WithSTDP) {
		KERNEL_INFO("  - STDP:")
		KERNEL_INFO("      - E-STDP TYPE            = %s",     groupConfigMap[gGrpId].stdpConfig.WithESTDPtype == STANDARD ? "STANDARD" :
			(groupConfigMap[gGrpId].stdpConfig.WithESTDPtype == DA_MOD ? "  DA_MOD" : " UNKNOWN"));
		KERNEL_INFO("      - I-STDP TYPE            = %s",     groupConfigMap[gGrpId].stdpConfig.WithISTDPtype == STANDARD ? "STANDARD" :
			(groupConfigMap[gGrpId].stdpConfig.WithISTDPtype == DA_MOD?"  DA_MOD":" UNKNOWN"));
		KERNEL_INFO("      - ALPHA_PLUS_EXC         = %8.5f", groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_EXC);
		KERNEL_INFO("      - ALPHA_MINUS_EXC        = %8.5f", groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_EXC);
		KERNEL_INFO("      - TAU_PLUS_INV_EXC       = %8.5f", groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_EXC);
		KERNEL_INFO("      - TAU_MINUS_INV_EXC      = %8.5f", groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_EXC);
		KERNEL_INFO("      - BETA_LTP               = %8.5f", groupConfigMap[gGrpId].stdpConfig.BETA_LTP);
		KERNEL_INFO("      - BETA_LTD               = %8.5f", groupConfigMap[gGrpId].stdpConfig.BETA_LTD);
		KERNEL_INFO("      - LAMBDA                 = %8.5f", groupConfigMap[gGrpId].stdpConfig.LAMBDA);
		KERNEL_INFO("      - DELTA                  = %8.5f", groupConfigMap[gGrpId].stdpConfig.DELTA);
	}
}

void SNN::printGroupInfo(int netId, std::list<GroupConfigMD>::iterator grpIt) {
	int gGrpId = grpIt->gGrpId;
	KERNEL_INFO("  |-+ %s Group %s(G:%d,L:%d): ", netId == grpIt->netId ? "Local" : "External", groupConfigMap[grpIt->gGrpId].grpName.c_str(), grpIt->gGrpId, grpIt->lGrpId);
	KERNEL_INFO("    |- Type                       =  %s", isExcitatoryGroup(grpIt->gGrpId) ? "  EXCIT" :
		(isInhibitoryGroup(grpIt->gGrpId) ? "  INHIB" : (isPoissonGroup(grpIt->gGrpId)?" POISSON" :
		(isDopaminergicGroup(grpIt->gGrpId) ? "  DOPAM" : " UNKNOWN"))) );
	KERNEL_INFO("    |- Num of Neurons             = %8d", groupConfigMap[grpIt->gGrpId].numN);
	KERNEL_INFO("    |- Start Id                   = (G:%d,L:%d)", grpIt->gStartN, grpIt->lStartN);
	KERNEL_INFO("    |- End Id                     = (G:%d,L:%d)", grpIt->gEndN, grpIt->lEndN);
	KERNEL_INFO("    |- numPostSynapses            = %8d", grpIt->numPostSynapses);
	KERNEL_INFO("    |- numPreSynapses             = %8d", grpIt->numPreSynapses);

	if (snnState == EXECUTABLE_SNN) {
		KERNEL_INFO("    |- Avg post connections       = %8.5f", ((float)grpIt->numPostSynapses) / groupConfigMap[gGrpId].numN);
		KERNEL_INFO("    |- Avg pre connections        = %8.5f", ((float)grpIt->numPreSynapses) / groupConfigMap[gGrpId].numN);
	}

	if (groupConfigMap[gGrpId].type & POISSON_NEURON) {
		KERNEL_INFO("    |- Refractory period          = %8.5f", grpIt->refractPeriod);
	}

	if (groupConfigMap[gGrpId].stpConfig.WithSTP) {
		KERNEL_INFO("    |-+ STP:");
		KERNEL_INFO("      |- STP_A                  = %8.5f", groupConfigMap[gGrpId].stpConfig.STP_A);
		KERNEL_INFO("      |- STP_U                  = %8.5f", groupConfigMap[gGrpId].stpConfig.STP_U);
		KERNEL_INFO("      |- STP_tau_u              = %8d", (int)(1.0f / groupConfigMap[gGrpId].stpConfig.STP_tau_u_inv));
		KERNEL_INFO("      |- STP_tau_x              = %8d", (int)(1.0f / groupConfigMap[gGrpId].stpConfig.STP_tau_x_inv));
	}

	if (groupConfigMap[gGrpId].stdpConfig.WithSTDP) {
		KERNEL_INFO("    |-+ STDP:")
		KERNEL_INFO("      |- E-STDP TYPE            = %s", groupConfigMap[gGrpId].stdpConfig.WithESTDPtype == STANDARD ? "STANDARD" :
			(groupConfigMap[gGrpId].stdpConfig.WithESTDPtype == DA_MOD ? "  DA_MOD" : " UNKNOWN"));
		KERNEL_INFO("      |- I-STDP TYPE            = %s", groupConfigMap[gGrpId].stdpConfig.WithISTDPtype == STANDARD ? "STANDARD" :
			(groupConfigMap[gGrpId].stdpConfig.WithISTDPtype == DA_MOD ? "  DA_MOD" : " UNKNOWN"));
		KERNEL_INFO("      |- ALPHA_PLUS_EXC         = %8.5f", groupConfigMap[gGrpId].stdpConfig.ALPHA_PLUS_EXC);
		KERNEL_INFO("      |- ALPHA_MINUS_EXC        = %8.5f", groupConfigMap[gGrpId].stdpConfig.ALPHA_MINUS_EXC);
		KERNEL_INFO("      |- TAU_PLUS_INV_EXC       = %8.5f", groupConfigMap[gGrpId].stdpConfig.TAU_PLUS_INV_EXC);
		KERNEL_INFO("      |- TAU_MINUS_INV_EXC      = %8.5f", groupConfigMap[gGrpId].stdpConfig.TAU_MINUS_INV_EXC);
		KERNEL_INFO("      |- BETA_LTP               = %8.5f", groupConfigMap[gGrpId].stdpConfig.BETA_LTP);
		KERNEL_INFO("      |- BETA_LTD               = %8.5f", groupConfigMap[gGrpId].stdpConfig.BETA_LTD);
		KERNEL_INFO("      |- LAMBDA                 = %8.5f", groupConfigMap[gGrpId].stdpConfig.LAMBDA);
		KERNEL_INFO("      |- DELTA                  = %8.5f", groupConfigMap[gGrpId].stdpConfig.DELTA);
	}
}

void SNN::printSikeRoutingInfo() {
	if (!spikeRoutingTable.empty()) {
		KERNEL_INFO("*****************          Spike Routing Table          *************************");
		for (std::list<RoutingTableEntry>::iterator rteItr = spikeRoutingTable.begin(); rteItr != spikeRoutingTable.end(); rteItr++)
			KERNEL_INFO("    |-Source net:[%d] -> Destination net[%d]", rteItr->srcNetId, rteItr->destNetId);
	}
}

