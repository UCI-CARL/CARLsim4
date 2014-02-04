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

#if ! (_WIN32 || _WIN64)
#include <string.h>
#define strcmpi(s1,s2) strcasecmp(s1,s2)
#endif


extern MTRand_closed getRandClosed;
extern MTRand	getRand;
extern RNG_rand48* gpuRand48;

void CpuSNN::printMemoryInfo(FILE *fp)
{
  checkNetworkBuilt();

  fprintf(fp, "*************Memory Info *************\n");
  int totMemSize = cpuSnnSz.networkInfoSize+cpuSnnSz.synapticInfoSize+cpuSnnSz.neuronInfoSize+cpuSnnSz.spikingInfoSize;
  fprintf(fp, "Neuron Info Size:    %3.2f (%f MB)\n", cpuSnnSz.neuronInfoSize*100.0/totMemSize,   cpuSnnSz.neuronInfoSize/(1024.0*1024));
  fprintf(fp, "Synaptic Info Size:  %3.2f (%f MB)\n", cpuSnnSz.synapticInfoSize*100.0/totMemSize, cpuSnnSz.synapticInfoSize/(1024.0*1024));
  fprintf(fp, "Network Size:	%3.2f (%f MB)\n", cpuSnnSz.networkInfoSize*100.0/totMemSize,  cpuSnnSz.networkInfoSize/(1024.0*1024));
  fprintf(fp, "Firing Info Size:    %3.2f (%f MB)\n", cpuSnnSz.spikingInfoSize*100.0/totMemSize,   cpuSnnSz.spikingInfoSize/(1024.0*1024));
  fprintf(fp, "Additional Info:     %3.2f (%f MB)\n", cpuSnnSz.addInfoSize*100.0/totMemSize,      cpuSnnSz.addInfoSize/(1024.0*1024));
  fprintf(fp, "DebugInfo Info:      %3.2f (%f MB)\n", cpuSnnSz.debugInfoSize*100.0/totMemSize,    cpuSnnSz.debugInfoSize/(1024.0*1024));
  fprintf(fp, "**************************************\n\n");

  fprintf(fp, "************* Connection Info *************\n");
  for(int g=0; g < numGrp; g++) {
    int TNpost=0;
    int TNpre=0;
    int TNpre_plastic=0;
    for(int i=grp_Info[g].StartN; i <= grp_Info[g].EndN; i++) {
      TNpost += Npost[i];
      TNpre  += Npre[i];
      TNpre_plastic += Npre_plastic[i];
    }
    fprintf(fp, "%s Group (num_neurons=%5d): \n\t\tNpost[%2d] = %3d, Npre[%2d]=%3d Npre_plastic[%2d]=%3d \n\t\tcumPre[%5d]=%5d cumPre[%5d]=%5d cumPost[%5d]=%5d cumPost[%5d]=%5d \n",
	    grp_Info2[g].Name.c_str(), grp_Info[g].SizeN, g, TNpost/grp_Info[g].SizeN, g, TNpre/grp_Info[g].SizeN, g, TNpre_plastic/grp_Info[g].SizeN,
	    grp_Info[g].StartN, cumulativePre[grp_Info[g].StartN],  grp_Info[g].EndN, cumulativePre[grp_Info[g].EndN],
	    grp_Info[g].StartN, cumulativePost[grp_Info[g].StartN], grp_Info[g].EndN, cumulativePost[grp_Info[g].EndN]);
  }
  fprintf(fp, "**************************************\n\n");

}




// This method allows us to print all information about the neuron.
// If the enablePrint is false for a specific group, we do not print its state.
void CpuSNN::printState(const char *str)
{
  fprintf(stderr, "%s", str);
  for(int g=0; g < numGrp; g++) {
    if(grp_Info2[g].enablePrint) {
      printNeuronState(g, stderr);
      //printWeight(currentMode, g, "Displaying weights\n");
    }
  }
}

void CpuSNN::printGroupInfo(FILE* fp)
{
  //FILE* fpg=fopen("group_info.txt", "w");
  //fprintf(fpg, "Group Id : Start : End : Group Name\n");
  for(int g=0; g < numGrp; g++) {
    //fprintf(fpg, "%d %d %d %s\n", g, grp_Info[g].StartN, grp_Info[g].EndN, grp_Info2[g].Name.c_str());
    fprintf(fp, "Group %s: \n", grp_Info2[g].Name.c_str());
    fprintf(fp, "------------\n");
    fprintf(fp, "\t Size = %d\n", grp_Info[g].SizeN);
    fprintf(fp, "\t Start = %d\n", grp_Info[g].StartN);
    fprintf(fp, "\t End   = %d\n", grp_Info[g].EndN);
    fprintf(fp, "\t numPostSynapses     = %d\n", grp_Info[g].numPostSynapses);
    fprintf(fp, "\t numPreSynapses  = %d\n", grp_Info[g].numPreSynapses);
    fprintf(fp, "\t Average Post Connections = %f\n", 1.0*grp_Info2[g].numPostConn/grp_Info[g].SizeN);
    fprintf(fp, "\t Average Pre Connections = %f\n",  1.0*grp_Info2[g].numPreConn/grp_Info[g].SizeN);

    if(grp_Info[g].Type&POISSON_NEURON) {
      fprintf(fp, "\t Refractory-Period = %f\n", grp_Info[g].RefractPeriod);
    }

    fprintf(fp, "\t FIXED_WTS = %s\n", grp_Info[g].FixedInputWts? "FIXED_WTS":"PLASTIC_WTS");

    if (grp_Info[g].WithSTP) {
      fprintf(fp, "\t STP_U = %f\n", grp_Info[g].STP_U);
      fprintf(fp, "\t STP_tD = %f\n", grp_Info[g].STP_tD);
      fprintf(fp, "\t STP_tF = %f\n", grp_Info[g].STP_tF);
    }

    if(grp_Info[g].WithSTDP) {
      fprintf(fp, "\t ALPHA_LTP = %f\n", grp_Info[g].ALPHA_LTP);
      fprintf(fp, "\t ALPHA_LTD = %f\n", grp_Info[g].ALPHA_LTD);
      fprintf(fp, "\t TAU_LTP_INV = %f\n", grp_Info[g].TAU_LTP_INV);
      fprintf(fp, "\t TAU_LTD_INV = %f\n", grp_Info[g].TAU_LTD_INV);
    }

  }
  //fclose(fpg);
}

void CpuSNN::printGroupInfo2(FILE* fpg)
{
  fprintf(fpg, "#Group Information\n");
  for(int g=0; g < numGrp; g++) {
    fprintf(fpg, "group %d: name %s : type %s %s %s %s %s: size %d : start %d : end %d \n",
	    g, grp_Info2[g].Name.c_str(),
	    (grp_Info[g].Type&POISSON_NEURON) ? "poisson " : "",
	    (grp_Info[g].Type&TARGET_AMPA) ? "AMPA" : "",
	    (grp_Info[g].Type&TARGET_NMDA) ? "NMDA" : "",
	    (grp_Info[g].Type&TARGET_GABAa) ? "GABAa" : "",
	    (grp_Info[g].Type&TARGET_GABAb) ? "GABAb" : "",
	    grp_Info[g].SizeN,
	    grp_Info[g].StartN,
	    grp_Info[g].EndN);
  }
  fprintf(fpg, "\n");
  fflush(fpg);
}

void CpuSNN::printConnectionInfo2(FILE *fpg)
{
  grpConnectInfo_t* newInfo = connectBegin;

  fprintf(fpg, "#Connection Information \n");
  fprintf(fpg, "#(e.g. from => to : approx. # of post (numPostSynapses) : approx. # of pre-synaptic (numPreSynapses) : weights.. : type plastic or fixed : max and min axonal delay\n");
  while(newInfo) {
    bool synWtType	= GET_FIXED_PLASTIC(newInfo->connProp);
    fprintf(fpg, " %d => %d : %s => %s : numPostSynapses %d : numPreSynapses %d : initWeight %f : maxWeight %3.3f : type %s : maxDelay %d : minDelay %d\n",
	    newInfo->grpSrc, newInfo->grpDest, grp_Info2[newInfo->grpSrc].Name.c_str(), grp_Info2[newInfo->grpDest].Name.c_str(),
	    newInfo->numPostSynapses, newInfo->numPreSynapses, newInfo->initWt,   newInfo->maxWt,
	    (synWtType == SYN_PLASTIC)?"plastic":"fixed", newInfo->maxDelay, newInfo->minDelay);
    newInfo = newInfo->next;
  }
  fprintf(fpg, "\n");
  fflush(fpg);
}

void CpuSNN::printConnectionInfo(FILE *fp)
{
  grpConnectInfo_t* newInfo = connectBegin;

  //		fprintf(fp, "\nGlobal STDP Info: \n");
  //		fprintf(fp, "------------\n");
  //		fprintf(fp, " alpha_ltp: %f\n tau_ltp: %f\n alpha_ldp: %f\n tau_ldp: %f\n", ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

  fprintf(fp, "\nConnections: \n");
  fprintf(fp, "------------\n");
  while(newInfo) {
    bool synWtType	= GET_FIXED_PLASTIC(newInfo->connProp);
    fprintf(fp, " // (%s => %s): numPostSynapses=%d, numPreSynapses=%d, iWt=%3.3f, mWt=%3.3f, ty=%x, maxD=%d, minD=%d %s\n",
	    grp_Info2[newInfo->grpSrc].Name.c_str(), grp_Info2[newInfo->grpDest].Name.c_str(),
	    newInfo->numPostSynapses, newInfo->numPreSynapses, newInfo->initWt,   newInfo->maxWt,
	    newInfo->connProp, newInfo->maxDelay, newInfo->minDelay, (synWtType == SYN_PLASTIC)?"(*)":"");

    // 			weights of input spike generating layers need not be observed...
    //          bool synWtType	= GET_FIXED_PLASTIC(newInfo->connProp);
    //			if ((synWtType == SYN_PLASTIC) && (enableSimLogs))
    //			   storeWeights(newInfo->grpDest, newInfo->grpSrc, "logs");
    newInfo = newInfo->next;
  }

  fflush(fp);
}

void CpuSNN::printParameters(FILE* fp)
{
  assert(fp!=NULL);
  printGroupInfo(fp);
  printConnectionInfo(fp);
}

void CpuSNN::printGroupInfo(string& strName)
{
  fprintf(stderr, "String Name : %s\n", strName.c_str());
  for(int g=0; g < numGrp; g++)	{
    if(grp_Info[g].Type&POISSON_NEURON)
      fprintf(stderr, "Poisson Group %d: %s\n", g, grp_Info2[g].Name.c_str());
  }
}

// print all post-connections...
void CpuSNN::printPostConnection(FILE *fp)
{
  if(fp) fprintf(fp, "PRINTING POST-SYNAPTIC CONNECTION TOPOLOGY\n");
  if(fp) fprintf(fp, "(((((((((((((((((((((())))))))))))))))))))))\n");
  for(int i=0; i < numGrp; i++)
    printPostConnection(i,fp);
}

// print all the pre-connections...
void CpuSNN::printPreConnection(FILE *fp)
{
  if(fp) fprintf(fp, "PRINTING PRE-SYNAPTIC CONNECTION TOPOLOGY\n");
  if(fp) fprintf(fp, "(((((((((((((((((((((())))))))))))))))))))))\n");
  for(int i=0; i < numGrp; i++)
    printPreConnection(i,fp);
}

// print the connection info of grpId
int CpuSNN::printPostConnection2(int grpId, FILE* fpg)
{
  int maxLength = -1;
  for(int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
    fprintf(fpg, " id %d : group %d : postlength %d ", i, findGrpId(i), Npost[i]);
    // fetch the starting position
    post_info_t* postIds = &postSynapticIds[cumulativePost[i]];
    for(int j=0; j <= D; j++) {
      int len   = postDelayInfo[i*(D+1)+j].delay_length;
      int start = postDelayInfo[i*(D+1)+j].delay_index_start;
      for(int k=start; k < len; k++) {
	int post_nid = GET_CONN_NEURON_ID((*postIds));
	//					int post_gid = GET_CONN_GRP_ID((*postIds));
	fprintf(fpg, " : %d,%d ", post_nid, j);
	postIds++;
      }
      if (Npost[i] > maxLength)
	maxLength = Npost[i];
      if ((start+len) >= Npost[i])
	break;
    }
    fprintf(fpg, "\n");
  }
  fflush(fpg);
  return maxLength;
}

void CpuSNN::printNetworkInfo()
{
  int maxLengthPost = -1;
  int maxLengthPre  = -1;
  FILE *fpg = fopen("net_info.txt", "w");
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
  fprintf(stdout, "Max post-synaptic length = %d\n", maxLengthPost);
  fprintf(stdout, "Max pre-synaptic length = %d\n", maxLengthPre);
}

void CpuSNN::printFiringRate(char *fname)
{
  static int printCnt = 0;
  FILE *fpg;
  string strFname;
  if (fname == NULL)
    strFname = networkName;
  else
    strFname = fname;

  strFname += ".stat";
  if(printCnt==0)
    fpg = fopen(strFname.c_str(), "w");
  else
    fpg = fopen(strFname.c_str(), "a");

  fprintf(fpg, "#Average Firing Rate\n");
  if(printCnt==0) {
    fprintf(fpg, "#network %s: size = %d\n", networkName.c_str(), numN);
    for(int grpId=0; grpId < numGrp; grpId++) {
      fprintf(fpg, "#group %d: name %s : size = %d\n", grpId, grp_Info2[grpId].Name.c_str(), grp_Info[grpId].SizeN);
    }
  }
  fprintf(fpg, "Time %d ms\n", simTime);
  fprintf(fpg, "#activeNeurons ( <= 1.0) = fraction of neuron in the given group that are firing more than 1Hz\n");
  fprintf(fpg, "#avgFiring (in Hz) = Average firing rate of activeNeurons in given group\n");
  for(int grpId=0; grpId < numGrp; grpId++) {
    fprintf(fpg, "group %d : \t", grpId);
    int   totSpike = 0;
    int   activeCnt  = 0;
    for(int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
      if (nSpikeCnt[i] >= 1.0) {
	totSpike += nSpikeCnt[i];
	activeCnt++;
      }
    }
    fprintf(fpg, " activeNeurons = %3.3f : avgFiring = %3.3f  \n", activeCnt*1.0/grp_Info[grpId].SizeN, (activeCnt==0)?0.0:totSpike*1.0/activeCnt);
  }
  printCnt++;
  fflush(fpg);
  fclose(fpg);
}

// print the connection info of grpId
void CpuSNN::printPostConnection(int grpId, FILE* fp)
{
  for(int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
    if(fp) fprintf(fp, " %3d ( %3d ) : \t", i, Npost[i]);
    // fetch the starting position
    post_info_t* postIds = &postSynapticIds[cumulativePost[i]];
    int  offset  = cumulativePost[i];
    for(int j=0; j < Npost[i]; j++, postIds++) {
      int post_nid = GET_CONN_NEURON_ID((*postIds));
      int post_gid = GET_CONN_GRP_ID((*postIds));
      assert( findGrpId(post_nid) == post_gid);
      if(fp) fprintf(fp, " %3d ( D=%3d, Grp=%3d) ", post_nid, tmp_SynapticDelay[offset+j], post_gid);
    }
    if(fp) fprintf(fp, "\n");
    if(fp) fprintf(fp, " Delay ( %3d ) : ", i);
    for(int j=0; j < D; j++) {
      if(fp) fprintf(fp, " %d,%d ", postDelayInfo[i*(D+1)+j].delay_length,
		     postDelayInfo[i*(D+1)+j].delay_index_start);
    }
    if(fp) fprintf(fp, "\n");
  }
}

int CpuSNN::printPreConnection2(int grpId, FILE* fpg)
{
  int maxLength = -1;
  for(int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
    fprintf(fpg, " id %d : group %d : prelength %d ", i, findGrpId(i), Npre[i]);
    post_info_t* preIds = &preSynapticIds[cumulativePre[i]];
    for(int j=0; j < Npre[i]; j++, preIds++) {
      if (doneReorganization && (!memoryOptimized))
	fprintf(fpg, ": %d,%s", GET_CONN_NEURON_ID((*preIds)), (j < Npre_plastic[i])?"P":"F");
    }
    if ( Npre[i] > maxLength)
      maxLength = Npre[i];
    fprintf(fpg, "\n");
  }
  return maxLength;
}

void CpuSNN::printPreConnection(int grpId, FILE* fp)
{
  for(int i=grp_Info[grpId].StartN; i<=grp_Info[grpId].EndN; i++) {
    if(fp) fprintf(fp, " %d ( preCnt=%d, prePlastic=%d ) : (id => (wt, maxWt),(preId, P/F)\n\t", i, Npre[i], Npre_plastic[i]);
    post_info_t* preIds = &preSynapticIds[cumulativePre[i]];
    int  pos_i  = cumulativePre[i];
    for(int j=0; j < Npre[i]; j++, pos_i++, preIds++) {
      if(fp) fprintf(fp,  "  %d => (%f, %f)", j, wt[pos_i], maxSynWt[pos_i]);
      if(doneReorganization && (!memoryOptimized))
	if(fp) fprintf(fp, ",(%d, %s)",
		       GET_CONN_NEURON_ID((*preIds)),
		       (j < Npre_plastic[i])?"P":"F");
    }
    if(fp) fprintf(fp, "\n");
  }
}

/* deprecated
   void CpuSNN::storeWeights(int destGrp, int src_grp, const string& logname, int restoreTime )
   {
   if(!enableSimLogs)
   return;

   checkNetworkBuilt();

   // if restoreTime has some value then we restore the weight to that time, else dont restore (default -1)
   bool restore = (restoreTime == -1) ? 0 : 1;     // default false;
   int retVal;

   for(int k=0, nid = grp_Info[destGrp].StartN; nid <= grp_Info[destGrp].EndN; nid++,k++)
   {
   char fname[200];
   char dirname[200];

   if(restore)
   sprintf(dirname, "%s", logname.c_str());
   else {
   sprintf(dirname, "%s/%d", logname.c_str(), randSeed);

   sprintf(fname, "mkdir -p %s", dirname );
   retVal = system(fname);
   if(retVal == -1) {
   fprintf(stderr, "system command(%s) failed !!\n", fname);
   return;
   }

   sprintf(fname, "cp -f param.txt %s", dirname );
   retVal = system(fname);
   if(retVal == -1) {
   fprintf(stderr, "system command(%s) failed !!\n", fname);
   return;
   }
   }

   if(restore)  {
   sprintf(fname, "%s/weightsSrc%dDest%d_%d.m", dirname, src_grp, nid, restoreTime);
   fprintf( stderr, "Restoring simulation status using %s (weights from grp=%d to nid=%d\n", fname, src_grp, nid);
   }
   else {
   sprintf(fname, "%s/weightsSrc%dDest%d_%lld.m", dirname, src_grp, nid, (unsigned long long)simTimeSec);
   fprintf( stderr, "Saving simulation status using %s (weights from grp=%d to nid=%d\n", fname, src_grp, nid);
   }

   FILE *fp;

   if(restore)
   fp= fopen(fname, "r");
   else
   fp= fopen(fname, "a");

   if(fp==NULL) {
   fprintf(stderr, "Unable to open/create log file => %s\n", fname);
   return;
   }

   int  dest_grp  = findGrpId(nid);
   post_info_t* preIds    = &preSynapticIds[cumulativePre[nid]];
   float* synWts  = &wt[cumulativePre[nid]];

   assert(grpConnInfo[src_grp][dest_grp] != NULL);
   assert(synWts != NULL);

   // find a suitable match for each pre-syn id that we are interested in ..
   float maxWt    = grpConnInfo[src_grp][dest_grp]->maxWt;

   int preNum = 0;
   if(restore) {
   retVal = fscanf(fp, "%d ", &preNum);
   assert(retVal > 0);
   assert(preNum == Npre[nid]);
   }
   else
   fprintf(fp, "%d ", Npre[nid]);

   for(int i=0; i < Npre[nid]; i++, preIds++, synWts++) {
   //int preId = (*preIds) & POST_SYN_NEURON_MASK;
   int preId = GET_CONN_NEURON_ID((*preIds));
   assert(preId < (numN));
   // preId matches the src_grp that we are interested..
   if( src_grp == findGrpId(preId)) {
   if (restore) {
   retVal = fscanf(fp, " %f ", synWts);
   assert(retVal > 0);
   *synWts = maxWt*(*synWts);
   //fprintf(stderr, " %f", *synWts);
   }
   else
   fprintf(fp, " %f ", *synWts/maxWt);
   }
   }

   if (restore) {
   fprintf(fp, "\n");
   }
   else
   fprintf(stderr, "\n");
   fclose(fp);
   }
   }
*/
void CpuSNN::printNeuronState(int grpId, FILE*fp)
{
  if (currentMode==GPU_MODE) {
    copyNeuronState(&cpuNetPtrs, &cpu_gpuNetPtrs, cudaMemcpyDeviceToHost, false, grpId);
  }

  fprintf(fp, "[MODE=%s] ", (currentMode==GPU_MODE)?"GPU_MODE":"CPU_MODE");
  fprintf(fp, "Group %s (%d) Neuron State Information (totSpike=%d, poissSpike=%d)\n",
	  grp_Info2[grpId].Name.c_str(), grpId, spikeCountAll, nPoissonSpikes);

  // poisson group does not have default neuron state
  if(grp_Info[grpId].Type&POISSON_NEURON) {
    fprintf(fp, "t=%d msec ", simTime);
    int totSpikes = 0;
    for (int nid=grp_Info[grpId].StartN; nid <= grp_Info[grpId].EndN; nid++) {
      totSpikes += cpuNetPtrs.nSpikeCnt[nid];
      fprintf(fp, "%d ", cpuNetPtrs.nSpikeCnt[nid]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "TotalSpikes [grp=%d, %s]=  %d\n", grpId, grp_Info2[grpId].Name.c_str(), totSpikes);
    return;
  }

  int totSpikes = 0;
  for (int nid=grp_Info[grpId].StartN; nid <= grp_Info[grpId].EndN; nid++) {
    // copy the neuron firing information from the GPU to the CPU...
    totSpikes += cpuNetPtrs.nSpikeCnt[nid];
    if(!sim_with_conductances) {
      if(cpuNetPtrs.current[nid] != 0.0)
	fprintf(fp, "t=%d id=%d v=%+3.3f u=%+3.3f I=%+3.3f nSpikes=%d\n", simTime, nid,
		cpuNetPtrs.voltage[nid],     cpuNetPtrs.recovery[nid],      cpuNetPtrs.current[nid],
		cpuNetPtrs.nSpikeCnt[nid]);
    }
    else {
      if (cpuNetPtrs.gAMPA[nid]+ cpuNetPtrs.gNMDA[nid]+cpuNetPtrs.gGABAa[nid]+cpuNetPtrs.gGABAb[nid] != 0.0)
	fprintf(fp, "t=%d id=%d v=%+3.3f u=%+3.3f I=%+3.3f gAMPA=%2.5f gNMDA=%2.5f gGABAa=%2.5f gGABAb=%2.5f nSpikes=%d\n", simTime, nid,
		cpuNetPtrs.voltage[nid],     cpuNetPtrs.recovery[nid],      cpuNetPtrs.current[nid], cpuNetPtrs.gAMPA[nid],
		cpuNetPtrs.gNMDA[nid], cpuNetPtrs.gGABAa[nid], cpuNetPtrs.gGABAb[nid], cpuNetPtrs.nSpikeCnt[nid]);
    }
  }
  fprintf(fp, "TotalSpikes [grp=%d, %s] = %d\n", grpId, grp_Info2[grpId].Name.c_str(), totSpikes);
  fprintf(fp, "\n");
  fflush(fp);
}
