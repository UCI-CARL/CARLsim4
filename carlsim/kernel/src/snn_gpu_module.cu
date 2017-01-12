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
#include <spike_buffer.h>
#include <error_code.h>
#include <cuda_runtime.h>

#define NUM_THREADS 128
#define NUM_BLOCKS 64
#define WARP_SIZE 32

///////////////////////////////////////////////////////////////////
// Some important ideas that explains the GPU execution are as follows:
//  1. Each GPU block has a local firing table (called fireTable). The block of threads
//     reads a bunch of neurons parameters and determines if it needs to fire or not
//     Whenever a neuron need to fire, it keeps track of the fired neuron in the local
//     table. When the table is full, we go and write back the fireTable to the global
//     firing table. 
//  2. Firing information is maintained in two tables globally (timingTable and the globalFiringTable)
//     for excitatory neuron population and inhibitory neurons.
//     The globalFiringTable only stores a sequence of id corresponding to fired neurons.
//     The timingTable store the total number of fired neurons till the current time step t.
//     These two tables are flushed and adjusted every second.
//     This approach requires about half of the memory compared to the traditional AER scheme which
//     stores the firing time and firing id together.
//  For more details kindly read the enclosed report (report.pdf) in the source directory
//
//
//  timeTableD2GPU[0] always is 0 -- index into firingTableD2
//  timeTableD2GPU[maxDelay_] -- should be the number of spikes "leftover" from the previous second
//	timeTableD2GPU[maxDelay_+1]-timeTableD2GPU[maxDelay_] -- should be the number of spikes in the first ms of the current second
//  timeTableD2GPU[1000+maxDelay_] -- should be the number of spikes in the current second + the leftover spikes.
//
///////////////////////////////////////////////////////////////////

__device__ unsigned int  timeTableD2GPU[TIMING_COUNT];
__device__ unsigned int  timeTableD1GPU[TIMING_COUNT];

__device__ unsigned int	spikeCountD2SecGPU;
__device__ unsigned int	spikeCountD1SecGPU;
__device__ unsigned int spikeCountD2GPU;
__device__ unsigned int spikeCountD1GPU;

__device__ unsigned int	secD2fireCntTest;
__device__ unsigned int	secD1fireCntTest;

__device__ unsigned int spikeCountLastSecLeftD2GPU;

__device__ unsigned int spikeCountExtRxD1SecGPU;
__device__ unsigned int spikeCountExtRxD2SecGPU;
__device__ unsigned int spikeCountExtRxD2GPU;
__device__ unsigned int spikeCountExtRxD1GPU;

__device__ __constant__ RuntimeData     runtimeDataGPU;
__device__ __constant__ NetworkConfigRT	networkConfigGPU;
__device__ __constant__ GroupConfigRT   groupConfigsGPU[MAX_GRP_PER_SNN];

__device__ __constant__ float               d_mulSynFast[MAX_CONN_PER_SNN];
__device__ __constant__ float               d_mulSynSlow[MAX_CONN_PER_SNN];

__device__  int	  loadBufferCount; 
__device__  int   loadBufferSize;

texture <int,    1, cudaReadModeElementType>  timeTableD2GPU_tex;
texture <int,    1, cudaReadModeElementType>  timeTableD1GPU_tex;
texture <int,    1, cudaReadModeElementType>  groupIdInfo_tex; // groupIDInfo is allocated using cudaMalloc thus doesn't require an offset when using textures
__device__  int timeTableD1GPU_tex_offset;
__device__  int timeTableD2GPU_tex_offset;

// example of the quick synaptic table
// index     cnt
// 0000000 - 0
// 0000001 - 0
// 0000010 - 1
// 0100000 - 5
// 0110000 - 4
int quickSynIdTable[256];
__device__ int  quickSynIdTableGPU[256];
void initQuickSynIdTable(int netId) {
	void* devPtr;
	   
	for(int i = 1; i < 256; i++) {
		int cnt = 0;
		while(i) {
			if(((i >> cnt) & 1) == 1) break;
      		cnt++;
      		assert(cnt <= 7);
    	}
    	quickSynIdTable[i] = cnt;		 
	}

	cudaSetDevice(netId);
	cudaGetSymbolAddress(&devPtr, quickSynIdTableGPU);
	CUDA_CHECK_ERRORS(cudaMemcpy( devPtr, quickSynIdTable, sizeof(quickSynIdTable), cudaMemcpyHostToDevice));
}

__device__ inline bool isPoissonGroup(short int lGrpId) {
	return (groupConfigsGPU[lGrpId].Type & POISSON_NEURON);
}

__device__ inline void setFiringBitSynapses(int lNId, int synId) {
	unsigned int* tmp_I_set_p = ((unsigned int*)((char*)runtimeDataGPU.I_set + ((synId >> 5) * networkConfigGPU.I_setPitch)) + lNId);
	atomicOr(tmp_I_set_p, 1 << (synId % 32));
}

__device__ inline unsigned int* getFiringBitGroupPtr(int lNId, int synId) {
	return (((unsigned int*)((char*)runtimeDataGPU.I_set + synId * networkConfigGPU.I_setPitch)) + lNId);
}

__device__ inline int getSTPBufPos(int lNId, int simTime) {
	return (((simTime + 1) % (networkConfigGPU.maxDelay + 1)) * networkConfigGPU.STP_Pitch + lNId);
}

__device__ inline int2 getStaticThreadLoad(int bufPos) {
	return (runtimeDataGPU.neuronAllocation[bufPos]);
}

__device__ inline bool getPoissonSpike(int lNId) {
	// Random number value is less than the poisson firing probability
	// if poisson firing probability is say 1.0 then the random poisson ptr
	// will always be less than 1.0 and hence it will continiously fire
	return runtimeDataGPU.randNum[lNId - networkConfigGPU.numNReg] * 1000.0f
			< runtimeDataGPU.poissonFireRate[lNId - networkConfigGPU.numNReg];
}

__device__ inline bool getSpikeGenBit(unsigned int nidPos) {
	const int nidBitPos = nidPos % 32;
	const int nidIndex  = nidPos / 32;
	return ((runtimeDataGPU.spikeGenBits[nidIndex] >> nidBitPos) & 0x1);
}

/*!
 * \brief This device function updates the average firing rate of each neuron, which is required for homeostasis
 *
 * \param[in] lNId The neuron id to be updated
 * \param[in] lGrpId The group id of the neuron
 */
__device__ inline void updateHomeoStaticState(int lNId, int lGrpId) {
	// here the homeostasis adjustment
	runtimeDataGPU.avgFiring[lNId] *= (groupConfigsGPU[lGrpId].avgTimeScale_decay);
}

/*!
 * \brief After every time step we update the time table
 *
 * Only one cuda thread is required for updating the time table
 *
 * \param[in] simTime The current time step
 */
__global__ void kernel_updateTimeTable(int simTime) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		timeTableD2GPU[simTime + networkConfigGPU.maxDelay + 1] = spikeCountD2SecGPU + spikeCountLastSecLeftD2GPU;
		timeTableD1GPU[simTime + networkConfigGPU.maxDelay + 1] = spikeCountD1SecGPU;
	}
	__syncthreads();									     
}

/////////////////////////////////////////////////////////////////////////////////
// Device Kernel Function:  Intialization of the GPU side of the simulator    ///
// KERNEL: This kernel is called after initialization of various parameters   ///
// so that we can reset all required parameters.                              ///
/////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_initGPUMemory() {
	// FIXME: use parallel access
	int timeTableIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (timeTableIdx < TIMING_COUNT) {
		timeTableD2GPU[timeTableIdx] = 0;
		timeTableD1GPU[timeTableIdx] = 0;
	}

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		spikeCountD2SecGPU = 0;
		spikeCountD1SecGPU = 0;
		spikeCountD2GPU = 0;
		spikeCountD1GPU = 0;

		secD2fireCntTest = 0;
		secD1fireCntTest = 0;

		spikeCountLastSecLeftD2GPU = 0;

		spikeCountExtRxD2GPU = 0;
		spikeCountExtRxD1GPU = 0;
		spikeCountExtRxD2SecGPU = 0;
		spikeCountExtRxD1SecGPU = 0;
	}
}

// Allocation of the group and its id..
void SNN::allocateGroupId(int netId) {
	checkAndSetGPUDevice(netId);

	assert (gpuRuntimeData[netId].groupIdInfo == NULL);
	int3* tempNeuronAllocation = (int3*)malloc(sizeof(int3) * networkConfigs[netId].numGroups);
	for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
		int3  threadLoad;
		threadLoad.x = groupConfigs[netId][lGrpId].lStartN;
		threadLoad.y = groupConfigs[netId][lGrpId].lEndN;
		threadLoad.z = lGrpId;
		tempNeuronAllocation[lGrpId] = threadLoad;
	}

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&gpuRuntimeData[netId].groupIdInfo, sizeof(int3) * networkConfigs[netId].numGroups));
	CUDA_CHECK_ERRORS(cudaMemcpy(gpuRuntimeData[netId].groupIdInfo, tempNeuronAllocation, sizeof(int3) * networkConfigs[netId].numGroups, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaBindTexture(NULL, groupIdInfo_tex, gpuRuntimeData[netId].groupIdInfo, sizeof(int3) * networkConfigs[netId].numGroups));

	free(tempNeuronAllocation);
}

/************************ VARIOUS KERNELS FOR FIRING CALCULATION AND FIRING UPDATE ****************************/
// Static Thread Load Allocation...
// This function is necessary for static allocation of load that each CUDA-SM needs for its computation.
// We store the static load allocation using the following format
// Neuron starting position (32 bit): Group identification (16) : Buffer size (16 bit)
// if we have 3 groups. grp(1) = 400 neurons, grp(2) = 100, grp(3) = 600
// The allocated static table will look as follows..
//-------------------------
// start |  grp   |   size
//-------------------------
//    0  :   0    :   256
//  256  :   0    :   144
//  400  :   1    :   100
//  500  :   2    :   256
//  756  :   2    :   256
// 1012  :   2    :    88
//-----------------------
int SNN::allocateStaticLoad(int netId, int bufSize) {
	checkAndSetGPUDevice(netId);

	// only one thread does the static load table
	int bufferCnt = 0;
	for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
		int grpBufCnt = (int) ceil(1.0f * groupConfigs[netId][lGrpId].numN / bufSize);
		assert(grpBufCnt >= 0);
		bufferCnt += grpBufCnt;
		KERNEL_DEBUG("Grp Size = %d, Total Buffer Cnt = %d, Buffer Cnt = %d", groupConfigs[netId][lGrpId].numN, bufferCnt, grpBufCnt);
	}
	assert(bufferCnt > 0);

	int2*  tempNeuronAllocation = (int2*)malloc(sizeof(int2) * bufferCnt);
	KERNEL_DEBUG("STATIC THREAD ALLOCATION");
	KERNEL_DEBUG("------------------------");
	KERNEL_DEBUG("Buffer Size = %d, Buffer Count = %d", bufSize, bufferCnt);

	bufferCnt = 0;
	for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
		for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId += bufSize) {
			int2  threadLoad;
			// starting neuron id is saved...
			threadLoad.x = lNId;
			if ((lNId + bufSize - 1) <= groupConfigs[netId][lGrpId].lEndN)
				// grpID + full size
				threadLoad.y = (lGrpId + (bufSize << 16)); // can't support group id > 2^16
			else
				// grpID + left-over size
				threadLoad.y = (lGrpId + ((groupConfigs[netId][lGrpId].lEndN - lNId + 1) << 16)); // can't support group id > 2^16

			// fill the static load distribution here...
			int testGrpId = STATIC_LOAD_GROUP(threadLoad);
			tempNeuronAllocation[bufferCnt] = threadLoad;
			KERNEL_DEBUG("%d. Start=%d, size=%d grpId=%d:%s (SpikeMonId=%d) (GroupMonId=%d)",
					bufferCnt, STATIC_LOAD_START(threadLoad),
					STATIC_LOAD_SIZE(threadLoad),
					STATIC_LOAD_GROUP(threadLoad),
					groupConfigMap[groupConfigs[netId][testGrpId].gGrpId].grpName.c_str(),
					groupConfigMDMap[groupConfigs[netId][testGrpId].gGrpId].spikeMonitorId,
					groupConfigMDMap[groupConfigs[netId][testGrpId].gGrpId].groupMonitorId);
			bufferCnt++;
		}
	}

	assert(gpuRuntimeData[netId].allocated == false);
	// Finally writeback the total bufferCnt
	// Note down the buffer size for reference
	KERNEL_DEBUG("GPU loadBufferSize = %d, GPU loadBufferCount = %d", bufSize, bufferCnt);
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(loadBufferCount, &bufferCnt, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(loadBufferSize, &bufSize, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**) &gpuRuntimeData[netId].neuronAllocation, sizeof(int2) * bufferCnt));
	CUDA_CHECK_ERRORS(cudaMemcpy(gpuRuntimeData[netId].neuronAllocation, tempNeuronAllocation, sizeof(int2) * bufferCnt, cudaMemcpyHostToDevice));
	free(tempNeuronAllocation);
	return bufferCnt;
}

//////////////////////////////////////////////////
// 1. KERNELS used when a specific neuron fires //
//////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Device local function:      	Update the STP Variables                      ///
// update the STPU and STPX variable after firing                             ///
/////////////////////////////////////////////////////////////////////////////////

// update the spike-dependent part of du/dt and dx/dt
__device__ void firingUpdateSTP (int nid, int simTime, short int grpId) {
	// we need to retrieve the STP values from the right buffer position (right before vs. right after the spike)
	int ind_plus  = getSTPBufPos(nid, simTime);
	int ind_minus = getSTPBufPos(nid, (simTime - 1));

	// at this point, stpu[ind_plus] has already been assigned, and the decay applied
	// so add the spike-dependent part to that
	// du/dt = -u/tau_F + U * (1-u^-) * \delta(t-t_{spk})
	runtimeDataGPU.stpu[ind_plus] += groupConfigsGPU[grpId].STP_U * (1.0f - runtimeDataGPU.stpu[ind_minus]);

	// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
	runtimeDataGPU.stpx[ind_plus] -= runtimeDataGPU.stpu[ind_plus] * runtimeDataGPU.stpx[ind_minus];
}

__device__ void resetFiredNeuron(int lNId, short int lGrpId, int simTime) {
	// \FIXME \TODO: convert this to use coalesced access by grouping into a
	// single 16 byte access. This might improve bandwidth performance
	// This is fully uncoalsced access...need to convert to coalsced access..
	runtimeDataGPU.voltage[lNId] = runtimeDataGPU.Izh_c[lNId];
	runtimeDataGPU.recovery[lNId] += runtimeDataGPU.Izh_d[lNId];
	if (groupConfigsGPU[lGrpId].WithSTDP)
		runtimeDataGPU.lastSpikeTime[lNId] = simTime;
	
	if (networkConfigGPU.sim_with_homeostasis) {
		// with homeostasis flag can be used here.
		runtimeDataGPU.avgFiring[lNId] += 1000/(groupConfigsGPU[lGrpId].avgTimeScale*1000);
	}
}

/*!
 * \brief 1. Copy neuron id from local table to global firing table. 2. Reset all neuron properties of neuron id in local table
 *
 *
 * \param[in] fireTablePtr the local shared memory firing table with neuron ids of fired neuron
 * \param[in] fireCntD2 the number of neurons in local table that has fired with group's max delay == 1
 * \param[in] fireCntD1 the number of neurons in local table that has fired with group's max delay > 1
 * \param[in] simTime the current time step, stored as neuron firing time  entry
 */
__device__ void updateSpikeCount(volatile unsigned int& fireCnt, volatile unsigned int& fireCntD1, volatile unsigned int& cntD2, volatile unsigned int& cntD1, volatile int&  blkErrCode) {
	int fireCntD2 = fireCnt - fireCntD1;

	cntD2 = atomicAdd(&secD2fireCntTest, fireCntD2);
	cntD1 = atomicAdd(&secD1fireCntTest, fireCntD1);

	//check for overflow in the firing table size....
	if(secD2fireCntTest>networkConfigGPU.maxSpikesD2) {
		blkErrCode = NEW_FIRE_UPDATE_OVERFLOW_ERROR2;
		return;
	}
	else if(secD1fireCntTest>networkConfigGPU.maxSpikesD1) {
		blkErrCode = NEW_FIRE_UPDATE_OVERFLOW_ERROR1;
		return;
	}
	blkErrCode = 0;

	// get a distinct counter to store firing info
	// into the firing table
	cntD2 = atomicAdd(&spikeCountD2SecGPU, fireCntD2) + spikeCountLastSecLeftD2GPU;
	cntD1 = atomicAdd(&spikeCountD1SecGPU, fireCntD1);
}

// update the firing table...
__device__ void updateFiringTable(int lNId, short int lGrpId, volatile unsigned int& cntD2, volatile unsigned int& cntD1) {
	int pos;
	if (groupConfigsGPU[lGrpId].MaxDelay == 1) {
		// this group has a delay of only 1
		pos = atomicAdd((int*)&cntD1, 1);
		//runtimeDataGPU.firingTableD1[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.firingTableD1[pos] = lNId;
	} else {
		// all other groups is dumped here 
		pos = atomicAdd((int*)&cntD2, 1);
		//runtimeDataGPU.firingTableD2[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.firingTableD2[pos] = lNId;
	}
}

// update the firing table...
__device__ void updateExtFiringTable(int lNId, short int lGrpId) {
	int pos;
	if (groupConfigsGPU[lGrpId].MaxDelay == 1) {
		// this group has a delay of only 1
		pos = atomicAdd((int*)&runtimeDataGPU.extFiringTableEndIdxD1[lGrpId] , 1);
		//runtimeDataGPU.firingTableD1[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.extFiringTableD1[lGrpId][pos] = lNId + groupConfigsGPU[lGrpId].LtoGOffset; // convert to global neuron id
	} else {
		// all other groups is dumped here 
		pos = atomicAdd((int*)&runtimeDataGPU.extFiringTableEndIdxD2[lGrpId], 1);
		//runtimeDataGPU.firingTableD2[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.extFiringTableD2[lGrpId][pos] = lNId + groupConfigsGPU[lGrpId].LtoGOffset; // convert to global neuron id
	}
}

__device__ int updateNewFirings(int* fireTablePtr, short int* fireGrpId,
                                volatile unsigned int& fireCnt, volatile unsigned int& fireCntD1, int simTime) {
	__shared__ volatile unsigned int cntD2;
	__shared__ volatile unsigned int cntD1;
	__shared__ volatile int blkErrCode;

	blkErrCode = 0;
	if (threadIdx.x == 0) {
		updateSpikeCount(fireCnt, fireCntD1, cntD2, cntD1, blkErrCode);
	}

	__syncthreads();

	// if we overflow the spike buffer space that is available,
	// then we return with an error here...
	if (blkErrCode)
		return blkErrCode;

	for (int i = threadIdx.x; i < fireCnt; i += blockDim.x) {
		// Read the firing id from the local table.....
		int lNId = fireTablePtr[i];

		updateFiringTable(lNId, fireGrpId[i], cntD2, cntD1);

		if (groupConfigsGPU[fireGrpId[i]].hasExternalConnect)
			updateExtFiringTable(lNId, fireGrpId[i]);

		if (groupConfigsGPU[fireGrpId[i]].WithSTP)
			firingUpdateSTP(lNId, simTime, fireGrpId[i]);

		// keep track of number spikes per neuron
		runtimeDataGPU.nSpikeCnt[lNId]++;

		// only neurons would do the remaining settings...
		// pure poisson generators will return without changing anything else..
		if (IS_REGULAR_NEURON(lNId, networkConfigGPU.numNReg, networkConfigGPU.numNPois))
			resetFiredNeuron(lNId, fireGrpId[i], simTime);
	}

	__syncthreads();

	 return 0;
}

// zero GPU spike counts
__global__ void kernel_resetNSpikeCnt(int lGrpId) {
	const int totBuffers = loadBufferCount;

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int nid = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId = STATIC_LOAD_SIZE(threadLoad);
		int  grpId = STATIC_LOAD_GROUP(threadLoad);

		if ((lGrpId == ALL || lGrpId == grpId) && (nid <= lastId)) {
			runtimeDataGPU.nSpikeCnt[nid] = 0;
		}
	}
}

// wrapper to call resetSpikeCnt
void SNN::resetSpikeCnt_GPU(int gGrpId) {
	assert(gGrpId >= ALL); // ALL == -1

	if (gGrpId == ALL) {
		for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
			if (!groupPartitionLists[netId].empty()) {
				checkAndSetGPUDevice(netId);
				CUDA_CHECK_ERRORS(cudaMemset((void*)gpuRuntimeData[netId].nSpikeCnt, 0, sizeof(int) * networkConfigs[netId].numN));
			}
		}
	} else {
		int netId = groupConfigMDMap[gGrpId].netId;
		int lGrpId = groupConfigMDMap[gGrpId].lGrpId;

		checkAndSetGPUDevice(netId);
		kernel_resetNSpikeCnt<<<NUM_BLOCKS, NUM_THREADS>>>(lGrpId);
	}
}

#define LTP_GROUPING_SZ 16 //!< synaptic grouping for LTP Calculation
/*!
 * \brief Computes the STDP update values for each of fired neurons stored in the local firing table.
 *
 * \param[in] fireTablePtr the local firing table with neuron ids of fired neuron
 * \param[in] fireCnt the number of fired neurons in local firing table
 * \param[in] simTime the current time step, stored as neuron firing time entry
 */
__device__ void updateLTP(int* fireTablePtr, short int* fireGrpId, volatile unsigned int& fireCnt, int simTime) {
	for(int pos=threadIdx.x/LTP_GROUPING_SZ; pos < fireCnt; pos += (blockDim.x/LTP_GROUPING_SZ))  {
		// each neuron has two variable pre and pre_exc
		// pre: number of pre-neuron
		// pre_exc: number of neuron had has plastic connections
		short int grpId = fireGrpId[pos];

		// STDP calculation: the post-synaptic neron fires after the arrival of pre-synaptic neuron's spike
		if (groupConfigsGPU[grpId].WithSTDP) { // MDR, FIXME this probably will cause more thread divergence than need be...
			int  nid   = fireTablePtr[pos];
			unsigned int  end_p = runtimeDataGPU.cumulativePre[nid] + runtimeDataGPU.Npre_plastic[nid];
			for(unsigned int p  = runtimeDataGPU.cumulativePre[nid] + threadIdx.x % LTP_GROUPING_SZ;
					p < end_p;
					p+=LTP_GROUPING_SZ) {
				int stdp_tDiff = (simTime - runtimeDataGPU.synSpikeTime[p]);
				if (stdp_tDiff > 0) {
					if (groupConfigsGPU[grpId].WithESTDP) {
						// Handle E-STDP curves
						switch (groupConfigsGPU[grpId].WithESTDPcurve) {
						case EXP_CURVE: // exponential curve
							if (stdp_tDiff * groupConfigsGPU[grpId].TAU_PLUS_INV_EXC < 25)
								runtimeDataGPU.wtChange[p] += STDP(stdp_tDiff, groupConfigsGPU[grpId].ALPHA_PLUS_EXC, groupConfigsGPU[grpId].TAU_PLUS_INV_EXC);
							break;
						case TIMING_BASED_CURVE: // sc curve
							if (stdp_tDiff * groupConfigsGPU[grpId].TAU_PLUS_INV_EXC < 25) {
								if (stdp_tDiff <= groupConfigsGPU[grpId].GAMMA)
									runtimeDataGPU.wtChange[p] += groupConfigsGPU[grpId].OMEGA + groupConfigsGPU[grpId].KAPPA * STDP(stdp_tDiff, groupConfigsGPU[grpId].ALPHA_PLUS_EXC, groupConfigsGPU[grpId].TAU_PLUS_INV_EXC);
								else // stdp_tDiff > GAMMA
									runtimeDataGPU.wtChange[p] -= STDP(stdp_tDiff, groupConfigsGPU[grpId].ALPHA_PLUS_EXC, groupConfigsGPU[grpId].TAU_PLUS_INV_EXC);
							}
							break;
						default:
							break;
						}
					}
					if (groupConfigsGPU[grpId].WithISTDP) {
						// Handle I-STDP curves
						switch (groupConfigsGPU[grpId].WithISTDPcurve) {
						case EXP_CURVE: // exponential curve
							if (stdp_tDiff * groupConfigsGPU[grpId].TAU_PLUS_INV_INB < 25) { // LTP of inhibitory synapse, which decreases synapse weight
								runtimeDataGPU.wtChange[p] -= STDP(stdp_tDiff, groupConfigsGPU[grpId].ALPHA_PLUS_INB, groupConfigsGPU[grpId].TAU_PLUS_INV_INB);
							}
							break;
						case PULSE_CURVE: // pulse curve
							if (stdp_tDiff <= groupConfigsGPU[grpId].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
								runtimeDataGPU.wtChange[p] -= groupConfigsGPU[grpId].BETA_LTP;
							} else if (stdp_tDiff <= groupConfigsGPU[grpId].DELTA) { // LTD of inhibitory syanpse, which increase sysnapse weight
								runtimeDataGPU.wtChange[p] -= groupConfigsGPU[grpId].BETA_LTD;
							}
							break;
						default:
							break;
						}
					}
				}
			}
		}
	}
	__syncthreads();
}

#define FIRE_CHUNK_CNT 512
/*!
 * \brief This kernel is responsible for finding the neurons that need to be fired.
 *
 * We use a buffered firing table that allows neuron to gradually load
 * the buffer and make it easy to carry out the calculations in a single group.
 * A single function is used for simple neurons and also for poisson neurons.
 * The function also update LTP
 *
 * device access: spikeCountD2SecGPU, spikeCountD1SecGPU
 * net access: numNReg numNPois, numN, sim_with_stdp, sim_in_testing, sim_with_homeostasis, maxSpikesD1, maxSpikesD2
 * grp access: Type, spikeGenFunc, Noffset, withSpikeCounter, spkCntBufPos, StartN, WithSTP, avgTimeScale
               WithSTDP, WithESTDP, WithISTDP, WithESTDPCurve, With ISTDPCurve, all STDP parameters
 * rtd access: randNum, poissonFireRate, spkCntBuf, nSpikeCnt, voltage, recovery, Izh_c, Izh_d
 *             cumulativePre, Npre_plastic, (R)synSpikeTime, (W)lastSpikeTime, (W)wtChange,
 *             avgFiring
 */
__global__ 	void kernel_findFiring (int simTime) {
	__shared__ volatile unsigned int fireCnt;
	__shared__ volatile unsigned int fireCntTest;
	__shared__ volatile unsigned int fireCntD1;
	__shared__ int 		fireTable[FIRE_CHUNK_CNT];
	__shared__ short int	fireGrpId[FIRE_CHUNK_CNT];
	__shared__ volatile int errCode;

	if (threadIdx.x == 0) {
		fireCnt	  = 0; // initialize total cnt to 0
		fireCntD1  = 0; // initialize d1 cnt to 0
		fireCntTest = 0; // initialize test cnt to 0
	}

	const int totBuffers=loadBufferCount;

	__syncthreads();

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad = getStaticThreadLoad(bufPos);
		int  lNId          = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastLNId       = STATIC_LOAD_SIZE(threadLoad);
		short int lGrpId  = STATIC_LOAD_GROUP(threadLoad);
		bool needToWrite  = false;	// used by all neuron to indicate firing condition
		int  fireId       = 0;

		// threadId is valid and lies within the lastId.....
		if ((threadIdx.x < lastLNId) && (lNId < networkConfigGPU.numN)) {
			// Simple poisson spiker uses the poisson firing probability
			// to detect whether it has fired or not....
			if(isPoissonGroup(lGrpId)) { // spikes generated by spikeGenFunc
				if(groupConfigsGPU[lGrpId].isSpikeGenFunc) {
					unsigned int offset = lNId - groupConfigsGPU[lGrpId].lStartN + groupConfigsGPU[lGrpId].Noffset;
					needToWrite = getSpikeGenBit(offset);
				} else { // spikes generated by poission rate
					needToWrite = getPoissonSpike(lNId);
				}
				// Note: valid lastSpikeTime of spike gen neurons is required by userDefinedSpikeGenerator()
				if (needToWrite)
					runtimeDataGPU.lastSpikeTime[lNId] = simTime;
			} else {
				if (runtimeDataGPU.voltage[lNId] >= 30.0f) {
					needToWrite = true;
				}
			}
		}

		// loop through a few times to ensure that we have added/processed all spikes that need to be written
		// if the buffer is small relative to the number of spikes needing to be written, we may have to empty the buffer a few times...
		for (int c = 0; c < 2; c++) {
			// we first increment fireCntTest to make sure we haven't filled the buffer
			if (needToWrite)
				fireId = atomicAdd((int*)&fireCntTest, 1);

			// if there is a spike and the buffer still has space...
			if (needToWrite && (fireId <(FIRE_CHUNK_CNT))) {
				// get our position in the buffer
				fireId = atomicAdd((int*)&fireCnt, 1);

				if (groupConfigsGPU[lGrpId].MaxDelay == 1)
					atomicAdd((int*)&fireCntD1, 1);

				// store ID of the fired neuron
				needToWrite 	  = false;
				fireTable[fireId] = lNId;
				fireGrpId[fireId] = lGrpId;//setFireProperties(grpId, isInhib);
			}

			__syncthreads();

			// the local firing table is full. dump the local firing table to the global firing table before proceeding
			if (fireCntTest >= (FIRE_CHUNK_CNT)) {

				// clear the table and update...
				int retCode = updateNewFirings(fireTable, fireGrpId, fireCnt, fireCntD1, simTime);
				if (retCode != 0) return;
				// update based on stdp rule
				// KILLME !!! if (simTime > 0))
				if (networkConfigGPU.sim_with_stdp && !networkConfigGPU.sim_in_testing)
					updateLTP (fireTable, fireGrpId, fireCnt, simTime);

				// reset counters
				if (threadIdx.x == 0) {
					fireCntD1  = 0;
					fireCnt   = 0;
					fireCntTest = 0;
				}
			}
		}
	}

	__syncthreads();

	// few more fired neurons are left. we update their firing state here..
	if (fireCnt) {
		int retCode = updateNewFirings(fireTable, fireGrpId, fireCnt, fireCntD1, simTime);
		if (retCode != 0) return;

		if (networkConfigGPU.sim_with_stdp && !networkConfigGPU.sim_in_testing)
			updateLTP(fireTable, fireGrpId, fireCnt, simTime);
	}
}

//******************************** UPDATE CONDUCTANCES AND TOTAL SYNAPTIC CURRENT EVERY TIME STEP *****************************

#define LOG_CURRENT_GROUP 5
/*!
 * \brief Based on the bitvector used for indicating the presence of spike, the global conductance values are updated.
 *
 * net access: numNReg, numNPois, I_setPitch, maxDelay, STP_Pitch, sim_with_conductances,
               sim_with_NMDA_rise, sim_withGABAb_Rise, sNMDA, sGABAb
 * grp access: WithSTP, STP_A
 * rtd access: Npre, cumulativePre, I_set, preSynapticIds, grpIds, wt, stpx, stpu, connIdsPreIdx,
               gAMPA, gGABAa, gNMDA_r, gNMDA_d, gNMDA, gGABAb_r, gGABAb_d, gGABAb
 * glb access: d_mulSynFast, d_mulSynSlow
 */
__global__ void kernel_conductanceUpdate (int simTimeMs, int simTimeSec, int simTime) {
	__shared__ int sh_quickSynIdTable[256];

	// Table for quick access
	for (int i = 0; i < 256; i += blockDim.x) {
		if ((i + threadIdx.x) < 256) {
			sh_quickSynIdTable[i + threadIdx.x] = quickSynIdTableGPU[i + threadIdx.x];
		}
	}

	__syncthreads();

	const int totBuffers = loadBufferCount;
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad = getStaticThreadLoad(bufPos);
		int  postNId    = STATIC_LOAD_START(threadLoad) + threadIdx.x;
		int  lastNId    = STATIC_LOAD_SIZE(threadLoad);

		if ((threadIdx.x < lastNId) && (IS_REGULAR_NEURON(postNId, networkConfigGPU.numNReg, networkConfigGPU.numNPois))) {
			// P6-1
			// load the initial current due to noise inputs for neuron 'post_nid'
			// initial values of the conductances for neuron 'post_nid'
			float AMPA_sum		 = 0.0f;
			float NMDA_sum		 = 0.0f;
			float NMDA_r_sum 	 = 0.0f;
			float NMDA_d_sum 	 = 0.0f;
			float GABAa_sum		 = 0.0f;
			float GABAb_sum		 = 0.0f;
			float GABAb_r_sum 	 = 0.0f;
			float GABAb_d_sum 	 = 0.0f;
			int   lmt      		 = runtimeDataGPU.Npre[postNId];
			unsigned int cum_pos = runtimeDataGPU.cumulativePre[postNId];

			// find the total current to this neuron...
			for (int j = 0; (lmt) && (j <= ((lmt - 1) >> LOG_CURRENT_GROUP)); j++) {
				// because of malloc2D operation we are using pitch, post_nid, j to get
				// actual position of the input current....
				// int* tmp_I_set_p = ((int*)((char*)runtimeDataGPU.I_set + j * networkConfigGPU.I_setPitch) + post_nid);
				uint32_t* tmp_I_set_p = getFiringBitGroupPtr(postNId, j);
				uint32_t  tmp_I_set = *tmp_I_set_p;

				// table lookup based find bits that are set
				int cnt = 0;
				int tmp_I_cnt = 0;
				while (tmp_I_set) {
					int k = (tmp_I_set >> (8 * cnt)) & 0xff;
					if (k == 0) {
						cnt = cnt + 1;
						continue;
					}
					int wt_i = sh_quickSynIdTable[k];
					int wtId = (j * 32 + cnt * 8 + wt_i);

					SynInfo synInfo = runtimeDataGPU.preSynapticIds[cum_pos + wtId];
					//uint8_t  pre_grpId  = GET_CONN_GRP_ID(pre_Id);
					uint32_t  preNId  = GET_CONN_NEURON_ID(synInfo);
					short int preGrpId = runtimeDataGPU.grpIds[preNId];
					char type = groupConfigsGPU[preGrpId].Type;

					// load the synaptic weight for the wtId'th input
					float change = runtimeDataGPU.wt[cum_pos + wtId];

					// Adjust the weight according to STP scaling
					if (groupConfigsGPU[preGrpId].WithSTP) {
						int tD = 0; // \FIXME find delay
						// \FIXME I think pre_nid needs to be adjusted for the delay
						int ind_minus = getSTPBufPos(preNId, (simTime - tD - 1)); // \FIXME should be adjusted for delay
						int ind_plus = getSTPBufPos(preNId, (simTime - tD));
						// dI/dt = -I/tau_S + A * u^+ * x^- * \delta(t-t_{spk})
						change *= groupConfigsGPU[preGrpId].STP_A * runtimeDataGPU.stpx[ind_minus] * runtimeDataGPU.stpu[ind_plus];
					}

					if (networkConfigGPU.sim_with_conductances) {
						short int connId = runtimeDataGPU.connIdsPreIdx[cum_pos+wtId];
						if (type & TARGET_AMPA)
							AMPA_sum += change * d_mulSynFast[connId];
						if (type & TARGET_NMDA) {
							if (networkConfigGPU.sim_with_NMDA_rise) {
								NMDA_r_sum += change * d_mulSynSlow[connId] * networkConfigGPU.sNMDA;
								NMDA_d_sum += change * d_mulSynSlow[connId] * networkConfigGPU.sNMDA;
							} else {
								NMDA_sum += change * d_mulSynSlow[connId];
							}
						}
						if (type & TARGET_GABAa)
							GABAa_sum += change * d_mulSynFast[connId];	// wt should be negative for GABAa and GABAb
						if (type & TARGET_GABAb) {						// but that is dealt with below
							if (networkConfigGPU.sim_with_GABAb_rise) {
								GABAb_r_sum += change * d_mulSynSlow[connId] * networkConfigGPU.sGABAb;
								GABAb_d_sum += change * d_mulSynSlow[connId] * networkConfigGPU.sGABAb;
							} else {
								GABAb_sum += change * d_mulSynSlow[connId];
							}
						}
					}
					else {
						// current based model with STP (CUBA)
						// updated current for neuron 'post_nid'
						AMPA_sum += change;
					}

					tmp_I_cnt++;
					tmp_I_set = tmp_I_set & (~(1 << (8 * cnt + wt_i)));
				}

				// FIXME: move reset outside kernel for debbuing I_set, resume it later
				// reset the input if there are any bit'wt set
				if(tmp_I_cnt)
					*tmp_I_set_p = 0;

				__syncthreads();
			}

			__syncthreads();

			// P6-2
			if (networkConfigGPU.sim_with_conductances) {
				// don't add mulSynFast/mulSynSlow here, because they depend on the exact pre<->post connection, not
				// just post_nid
				runtimeDataGPU.gAMPA[postNId]        += AMPA_sum;
				runtimeDataGPU.gGABAa[postNId]       -= GABAa_sum; // wt should be negative for GABAa and GABAb
				if (networkConfigGPU.sim_with_NMDA_rise) {
					runtimeDataGPU.gNMDA_r[postNId]  += NMDA_r_sum;
					runtimeDataGPU.gNMDA_d[postNId]  += NMDA_d_sum;
				} else {
					runtimeDataGPU.gNMDA[postNId]    += NMDA_sum;
				}
				if (networkConfigGPU.sim_with_GABAb_rise) {
					runtimeDataGPU.gGABAb_r[postNId] -= GABAb_r_sum;
					runtimeDataGPU.gGABAb_d[postNId] -= GABAb_d_sum;
				} else {
					runtimeDataGPU.gGABAb[postNId]   -= GABAb_sum;
				}
			}
			else {
				runtimeDataGPU.current[postNId] += AMPA_sum;
			}
		}
	}
}

//************************ UPDATE GLOBAL STATE EVERY TIME STEP *******************************************************//

/*!
 * \brief This device function implements the equations of neuron dynamics
 *
 * \param[in] nid The neuron id to be updated
 * \param[in] grpId The group id of the neuron
 */
__device__ void updateNeuronState(int nid, int grpId) {
	float v = runtimeDataGPU.voltage[nid];
	float u = runtimeDataGPU.recovery[nid];
	float I_sum, NMDAtmp;
	float gNMDA, gGABAb;

	// loop that allows smaller integration time step for v's and u's
	for (int c = 0; c < COND_INTEGRATION_SCALE; c++) {
		I_sum = 0.0f;
		if (networkConfigGPU.sim_with_conductances) {
			NMDAtmp = (v + 80.0f) * (v + 80.0f) / 60.0f / 60.0f;
			gNMDA = (networkConfigGPU.sim_with_NMDA_rise) ? (runtimeDataGPU.gNMDA_d[nid] - runtimeDataGPU.gNMDA_r[nid]) : runtimeDataGPU.gNMDA[nid];
			gGABAb = (networkConfigGPU.sim_with_GABAb_rise) ? (runtimeDataGPU.gGABAb_d[nid] - runtimeDataGPU.gGABAb_r[nid]) : runtimeDataGPU.gGABAb[nid];
			I_sum = -(runtimeDataGPU.gAMPA[nid] * (v - 0.0f)
						+ gNMDA * NMDAtmp / (1.0f + NMDAtmp) * (v - 0.0f)
						+ runtimeDataGPU.gGABAa[nid] * (v + 70.0f)
						+ gGABAb * (v + 90.0f));
		} else {
			I_sum = runtimeDataGPU.current[nid];
		}

		// update vpos and upos for the current neuron
		v += ((0.04f * v + 5.0f) * v + 140.0f - u + I_sum + runtimeDataGPU.extCurrent[nid]) / COND_INTEGRATION_SCALE;
		if (v > 30.0f) { 
			v = 30.0f; // break the loop but evaluate u[i]
			c = COND_INTEGRATION_SCALE;
		}

		if (v < -90.0f) v = -90.0f;

		u += (runtimeDataGPU.Izh_a[nid] * (runtimeDataGPU.Izh_b[nid] * v - u) / COND_INTEGRATION_SCALE);
	}
	if(networkConfigGPU.sim_with_conductances) {
		runtimeDataGPU.current[nid] = I_sum;
	} else {
		// current must be reset here for CUBA and not kernel_STPUpdateAndDecayConductances
		runtimeDataGPU.current[nid] = 0.0f;
	}
	runtimeDataGPU.voltage[nid] = v;
	runtimeDataGPU.recovery[nid] = u;
}

/*!
 *  \brief update neuron state
 *
 * This kernel update neurons' membrance potential according to neurons' dynamics model.
 * This kernel also update variables required by homeostasis
 *
 * net access: numN, numNReg, numNPois, sim_with_conductances, sim_with_NMDA_rise, sim_with_GABAb_rise
 * grp access: WithHomeostasis, avgTimeScale_decay
 * rtd access: avgFiring, voltage, recovery, gNMDA, gNMDA_r, gNMDA_d, gGABAb, gGABAb_r, gGABAb_d, gAMPA, gGABAa,
 *             current, extCurrent, Izh_a, Izh_b
 * glb access:
 */
__global__ void kernel_neuronStateUpdate() {
	const int totBuffers = loadBufferCount;

	// update neuron state
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int nid = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int lastId = STATIC_LOAD_SIZE(threadLoad);
		int grpId = STATIC_LOAD_GROUP(threadLoad);

		if ((threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {

			if (IS_REGULAR_NEURON(nid, networkConfigGPU.numNReg, networkConfigGPU.numNPois)) {
				// P7
				// update neuron state here....
				updateNeuronState(nid, grpId);

				// P8
				if (groupConfigsGPU[grpId].WithHomeostasis)
					updateHomeoStaticState(nid, grpId);
			}
		}
	}		
}

/*!
 *  \brief Update the state of groups, which includes concentration of dopamine currently
 *
 * Update the concentration of neuronmodulator
 *
 * net access: numGroups
 * grp access: WithESTDPtype, WithISTDPtype, baseDP, decayDP
 * rtd access: grpDA, grpDABuffer
 * glb access:
 */
__global__ void kernel_groupStateUpdate(int simTime) {
	// update group state
	int grpIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// P9
	if (grpIdx < networkConfigGPU.numGroups) {
		// decay dopamine concentration
		if ((groupConfigsGPU[grpIdx].WithESTDPtype == DA_MOD || groupConfigsGPU[grpIdx].WithISTDPtype == DA_MOD) && runtimeDataGPU.grpDA[grpIdx] > groupConfigsGPU[grpIdx].baseDP) {
			runtimeDataGPU.grpDA[grpIdx] *= groupConfigsGPU[grpIdx].decayDP;
		}
		runtimeDataGPU.grpDABuffer[grpIdx * 1000 + simTime] = runtimeDataGPU.grpDA[grpIdx]; // log dopamine concentration
	}
}

//******************************** UPDATE STP STATE EVERY TIME STEP **********************************************
/*!
 * \brief This function is called for updat STP and decay coductance every time step 
 *
 * net access sim_with_conductance, sim_with_NMDA_rise, sim_with_GABAb_rise, numNReg, numNPois, numN, STP_Pitch, maxDelay
 * grp access WithSTP 
 * rtd access gAMPA, gNMDA_r, gNMDA_d, gNMDA, gBABAa, gGABAb_r, gGABAb_d, gGABAb
 * rtd access stpu, stpx
 */
__global__ void kernel_STPUpdateAndDecayConductances (int t, int sec, int simTime) {
	const int totBuffers = loadBufferCount;

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad = getStaticThreadLoad(bufPos);
		int nid         = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int lastId      = STATIC_LOAD_SIZE(threadLoad);
		int grpId       = STATIC_LOAD_GROUP(threadLoad);


    // update the conductane parameter of the current neron
		if (networkConfigGPU.sim_with_conductances && IS_REGULAR_NEURON(nid, networkConfigGPU.numNReg, networkConfigGPU.numNPois)) {
			runtimeDataGPU.gAMPA[nid]   *=  networkConfigGPU.dAMPA;
			if (networkConfigGPU.sim_with_NMDA_rise) {
				runtimeDataGPU.gNMDA_r[nid]   *=  networkConfigGPU.rNMDA;
				runtimeDataGPU.gNMDA_d[nid]   *=  networkConfigGPU.dNMDA;
			} else {
				runtimeDataGPU.gNMDA[nid]   *=  networkConfigGPU.dNMDA;
			}
			runtimeDataGPU.gGABAa[nid]  *=  networkConfigGPU.dGABAa;
			if (networkConfigGPU.sim_with_GABAb_rise) {
				runtimeDataGPU.gGABAb_r[nid]  *=  networkConfigGPU.rGABAb;
				runtimeDataGPU.gGABAb_d[nid]  *=  networkConfigGPU.dGABAb;
			} else {
				runtimeDataGPU.gGABAb[nid]  *=  networkConfigGPU.dGABAb;
			}
		}

		if (groupConfigsGPU[grpId].WithSTP && (threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {
			int ind_plus  = getSTPBufPos(nid, simTime);
			int ind_minus = getSTPBufPos(nid, (simTime-1)); // \FIXME sure?
				runtimeDataGPU.stpu[ind_plus] = runtimeDataGPU.stpu[ind_minus]*(1.0f-groupConfigsGPU[grpId].STP_tau_u_inv);
				runtimeDataGPU.stpx[ind_plus] = runtimeDataGPU.stpx[ind_minus] + (1.0f-runtimeDataGPU.stpx[ind_minus])*groupConfigsGPU[grpId].STP_tau_x_inv;
		}
	}
}

//********************************UPDATE SYNAPTIC WEIGHTS EVERY SECOND  *************************************************************

/*!
 * \brief This kernel update synaptic weights
 *
 * This kernel is called every second to adjust the timingTable and globalFiringTable
 * We do the following thing:
 * 1. We discard all firing information that happened more than 1000-maxDelay_ time step.
 * 2. We move the firing information that happened in the last 1000-maxDelay_ time step to
 * the begining of the gloalFiringTable.
 * 3. We read each value of "wtChange" and update the value of "synaptic weights wt".
 * We also clip the "synaptic weight wt" to lie within the required range.
 */
__device__ void updateSynapticWeights(int nid, unsigned int synId, int grpId, float diff_firing, float homeostasisScale, float baseFiring, float avgTimeScaleInv) {
	// This function does not get called if the neuron group has all fixed weights.
	// t_twChange is adjusted by stdpScaleFactor based on frequency of weight updates (e.g., 10ms, 100ms, 1s)	
	float t_wt = runtimeDataGPU.wt[synId];
	float t_wtChange = runtimeDataGPU.wtChange[synId];
	float t_effectiveWtChange = networkConfigGPU.stdpScaleFactor * t_wtChange;
	float t_maxWt = runtimeDataGPU.maxSynWt[synId];

	switch (groupConfigsGPU[grpId].WithESTDPtype) {
	case STANDARD:
		if (groupConfigsGPU[grpId].WithHomeostasis) {
			// this factor is slow
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += t_effectiveWtChange;
		}
		break;
	case DA_MOD:
		if (groupConfigsGPU[grpId].WithHomeostasis) {
			t_effectiveWtChange = runtimeDataGPU.grpDA[grpId] * t_effectiveWtChange;
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += runtimeDataGPU.grpDA[grpId] * t_effectiveWtChange;
		}
		break;
	case UNKNOWN_STDP:
	default:
		// we shouldn't even be here if !WithSTDP
		break;
	}

	switch (groupConfigsGPU[grpId].WithISTDPtype) {
	case STANDARD:
		if (groupConfigsGPU[grpId].WithHomeostasis) {
			// this factor is slow
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += t_effectiveWtChange;
		}
		break;
	case DA_MOD:
		if (groupConfigsGPU[grpId].WithHomeostasis) {
			t_effectiveWtChange = runtimeDataGPU.grpDA[grpId] * t_effectiveWtChange;
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += runtimeDataGPU.grpDA[grpId] * t_effectiveWtChange;
		}
		break;
	case UNKNOWN_STDP:
	default:
		// we shouldn't even be here if !WithSTDP
		break;
	}

	// It's user's choice to decay weight change or not
	// see setWeightAndWeightChangeUpdate()
	t_wtChange *= networkConfigGPU.wtChangeDecay;

	// Check the synapse is excitatory or inhibitory first
	if (t_maxWt >= 0.0f) { // excitatory synapse
		if (t_wt >= t_maxWt) t_wt = t_maxWt;
		if (t_wt < 0.0f) t_wt = 0.0f;
	} else { // inhibitory synapse
		if (t_wt <= t_maxWt) t_wt = t_maxWt;
		if (t_wt > 0.0f) t_wt = 0.0f;
	}

	runtimeDataGPU.wt[synId] = t_wt;
	runtimeDataGPU.wtChange[synId] = t_wtChange;
}


#define UPWTS_CLUSTERING_SZ	32
/*!
 * \brief this kernel updates all synaptic weights
 *
 * net access: stdpScaleFactor, wtChangeDecay
 * grp access: homeostasisScale, avgTimeScaleInv, FixedInputWts, WithESTDPtype, WithISTDOtype, WithHomeostasis
 * rtd access: Npre_plastic, cumulativePre, avgFiring, baseFiringInv, baseFiring, wt, wtChange, maxSynWt
 * glb access:
 */
__global__ void kernel_updateWeights() {
	__shared__ volatile int errCode;
	__shared__ int    		startId, lastId, grpId, totBuffers, grpNCnt;
	__shared__ int2 		threadLoad;
	// added for homeostasis
	__shared__ float		homeostasisScale, avgTimeScaleInv;

	if(threadIdx.x == 0) {
		totBuffers = loadBufferCount;
		grpNCnt	= (blockDim.x / UPWTS_CLUSTERING_SZ) + ((blockDim.x % UPWTS_CLUSTERING_SZ) != 0);
	}

	__syncthreads();

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		// if ( threadIdx.x) { // TSC: this could be a performance bug, 127 threads other than the first thread try to read
		                       // threadLoad and wirte homeostatsisScale and avgTimeScaleInv at the same time
		if (threadIdx.x == 0) {
			threadLoad  = getStaticThreadLoad(bufPos);
			startId 	= STATIC_LOAD_START(threadLoad);
			lastId  	= STATIC_LOAD_SIZE(threadLoad);
			grpId   	= STATIC_LOAD_GROUP(threadLoad);

			// load homestasis parameters
			if (groupConfigsGPU[grpId].WithHomeostasis) {
				homeostasisScale = groupConfigsGPU[grpId].homeostasisScale;
				avgTimeScaleInv = groupConfigsGPU[grpId].avgTimeScaleInv;
			} else {
				homeostasisScale = 0.0f;
				avgTimeScaleInv = 1.0f;
			}
		}

		__syncthreads();

		// the weights are fixed for this group.. so dont make any changes on
		// the weight and continue to the next set of neurons...
		if (groupConfigsGPU[grpId].FixedInputWts)
			continue;

		int nid = (threadIdx.x / UPWTS_CLUSTERING_SZ) + startId;
		// update the synaptic weights from the synaptic weight derivatives
		for(; nid < startId + lastId; nid += grpNCnt) {
			int Npre_plastic = runtimeDataGPU.Npre_plastic[nid];
			unsigned int cumulativePre = runtimeDataGPU.cumulativePre[nid];
			float diff_firing  = 0.0f;
			float baseFiring = 0.0f;

			if (groupConfigsGPU[grpId].WithHomeostasis) {
				diff_firing = (1.0f - runtimeDataGPU.avgFiring[nid] * runtimeDataGPU.baseFiringInv[nid]);
				baseFiring = runtimeDataGPU.baseFiring[nid];
			}

			const int threadIdGrp = (threadIdx.x % UPWTS_CLUSTERING_SZ);
			// use 32 threads to update 32 synapses parallely
			for(unsigned int synIdOffset = cumulativePre; synIdOffset < cumulativePre + Npre_plastic; synIdOffset += UPWTS_CLUSTERING_SZ) {
				//excitatory connection change the synaptic weights
				unsigned int synId = synIdOffset + threadIdGrp;
				if(synId < cumulativePre + Npre_plastic) {
					updateSynapticWeights(nid, synId, grpId, diff_firing, homeostasisScale, baseFiring, avgTimeScaleInv);
				}
			}
		}
	}
}

//********************************UPDATE TABLES AND COUNTERS EVERY SECOND  *************************************************************

/*!
 * \brief This kernel shift the un-processed firing information in firingTableD2 to the beginning of
 * firingTableD2 for the next second of simulation.
 *
 * net access: maxDelay
 * grp access: N/A
 * rtd access: firingTableD2
 * glb access: timeTableD2GPU
 */
__global__ void kernel_shiftFiringTable() {
	int gnthreads= blockDim.x * gridDim.x;

	for(int p = timeTableD2GPU[999], k = 0; p < timeTableD2GPU[999 + networkConfigGPU.maxDelay + 1]; p += gnthreads, k += gnthreads) {
		if ((p + threadIdx.x) < timeTableD2GPU[999 + networkConfigGPU.maxDelay + 1])
			runtimeDataGPU.firingTableD2[k + threadIdx.x] = runtimeDataGPU.firingTableD2[p + threadIdx.x];
	}
}

/*!
 * \brief This kernel shift the un-processed firing information in timeTableD1(D2)GPU to the beginning of
 * timeTableD1(D2)GPU for the next second of simulation.
 *
 * After all the threads/blocks had adjusted the firingTableD1(D2)GPU, we update the timeTableD1(D2)GPU
 * so that the firing information that happended in the last maxDelay_ time step would become
 * the first maxDelay_ time step firing information for the next second of simulation.
 * We also reset/update all spike counters to appropriate values as indicated in the second part 
 * of this kernel.
 */
__global__ void kernel_shiftTimeTable() {
	int maxDelay = networkConfigGPU.maxDelay;

	if(blockIdx.x == 0) {
		for(int i = threadIdx.x; i < maxDelay; i += blockDim.x) {
			// use i+1 instead of just i because timeTableD2GPU[0] should always be 0
			timeTableD2GPU[i + 1] = timeTableD2GPU[1000 + i + 1] - timeTableD2GPU[1000];
			timeTableD1GPU[i + 1] = timeTableD1GPU[1000 + i + 1] - timeTableD1GPU[1000];
		}
	}

	__syncthreads();

	// reset various counters for the firing information
	if((blockIdx.x == 0) && (threadIdx.x == 0)) {
		timeTableD1GPU[maxDelay]  = 0;
		spikeCountD2GPU += spikeCountD2SecGPU;
		spikeCountD1GPU += spikeCountD1SecGPU;

		spikeCountD2SecGPU = 0; 
		spikeCountD1SecGPU = 0;

		spikeCountExtRxD2SecGPU = 0;
		spikeCountExtRxD1SecGPU = 0;

		spikeCountLastSecLeftD2GPU = timeTableD2GPU[maxDelay];
		secD2fireCntTest = timeTableD2GPU[maxDelay];
		secD1fireCntTest = 0;
	}
}

//****************************** GENERATE POST-SYNAPTIC CURRENT EVERY TIME-STEP  ****************************

/*
* The sequence of handling an post synaptic spike in GPU mode:
* P1. Update synSpikeTime
* P2. Update DA,5HT,ACh,NE accordingly
* P3. Update STDP wtChange
* P4. Load wt into change (temporary variable)
* P5. Modulate change by STP (if enabled)
* P6-1. Modulate change by d_mulSynSlow and d_mulSynFast
* P6-2. Accumulate g(AMPA,NMDA,GABAa,GABAb) or current
* P7. Update v(voltage), u(recovery)
* P8. Update homeostasis
* P9. Decay and log DA,5HT,ACh,NE
*/
__device__ void generatePostSynapticSpike(int simTime, int preNId, int postNId, int synId) {
	// get the actual position of the synapses and other variables...
	unsigned int pos = runtimeDataGPU.cumulativePre[postNId] + synId;

	short int preGrpId = runtimeDataGPU.grpIds[preNId]; // STP uses preGrpId
	short int postGrpId = runtimeDataGPU.grpIds[postNId]; // STDP uses postGrpId

	setFiringBitSynapses(postNId, synId);

	// P1
	runtimeDataGPU.synSpikeTime[pos] = simTime;		  //uncoalesced access

	// P2
	// Got one spike from dopaminergic neuron, increase dopamine concentration in the target area
	if (groupConfigsGPU[preGrpId].Type & TARGET_DA) {
		atomicAdd(&(runtimeDataGPU.grpDA[postGrpId]), 0.04f);
	}

	// P3
	// STDP calculation: the post-synaptic neuron fires before the arrival of pre-synaptic neuron's spike
	if (groupConfigsGPU[postGrpId].WithSTDP && !networkConfigGPU.sim_in_testing)  {
		int stdp_tDiff = simTime - runtimeDataGPU.lastSpikeTime[postNId];
		if (stdp_tDiff >= 0) {
			if (groupConfigsGPU[postGrpId].WithESTDP) {
				// Handle E-STDP curves
				switch (groupConfigsGPU[postGrpId].WithESTDPcurve) {
				case EXP_CURVE: // exponential curve
				case TIMING_BASED_CURVE: // sc curve
					if (stdp_tDiff * groupConfigsGPU[postGrpId].TAU_MINUS_INV_EXC < 25.0f)
						runtimeDataGPU.wtChange[pos] += STDP( stdp_tDiff, groupConfigsGPU[postGrpId].ALPHA_MINUS_EXC, groupConfigsGPU[postGrpId].TAU_MINUS_INV_EXC); // uncoalesced access
					break;
				default:
					break;
				}
			}
			if (groupConfigsGPU[postGrpId].WithISTDP) {
				// Handle I-STDP curves
				switch (groupConfigsGPU[postGrpId].WithISTDPcurve) {
				case EXP_CURVE: // exponential curve
					if ((stdp_tDiff * groupConfigsGPU[postGrpId].TAU_MINUS_INV_INB) < 25.0f) { // LTD of inhibitory syanpse, which increase synapse weight
						runtimeDataGPU.wtChange[pos] -= STDP(stdp_tDiff, groupConfigsGPU[postGrpId].ALPHA_MINUS_INB, groupConfigsGPU[postGrpId].TAU_MINUS_INV_INB);
					}
					break;
				case PULSE_CURVE: // pulse curve
					if (stdp_tDiff <= groupConfigsGPU[postGrpId].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
						runtimeDataGPU.wtChange[pos] -= groupConfigsGPU[postGrpId].BETA_LTP;
					} else if (stdp_tDiff <= groupConfigsGPU[postGrpId].DELTA) { // LTD of inhibitory syanpse, which increase synapse weight
						runtimeDataGPU.wtChange[pos] -= groupConfigsGPU[postGrpId].BETA_LTD;
					}
					break;
				default:
					break;
				}
			}
		}
	}
}

#define READ_CHUNK_SZ 64
/*!
 * \brief This kernel updates and generates spikes for delays greater than 1 from the fired neuron. 
 *
 * The LTD computation is also executed by this kernel.
 *
 * net access: maxDelay, I_setPitch, sim_in_testing
 * grp access: Type, WithSTDP, WithESTDP, WithESTDPcurve, WithISDP, WithISTDPcurve, all STDP parameters
 * rtd access: firingTableD2, cumulativePost, postDelayInfo, postSynapticIds, cumulativePre, grpIds,
 *             grpDA, I_set, (W)synSpikeTime, (R)lastSpikeTime, wtChange
 * glb access: spikeCountD2SecGPU, timeTableD2GPU_tex, timeTableD2GPU_tex_offset
 */
__global__ void kernel_doCurrentUpdateD2(int simTimeMs, int simTimeSec, int simTime) {
	__shared__	volatile int sh_neuronOffsetTable[READ_CHUNK_SZ + 2];
	__shared__	int sh_delayLength[READ_CHUNK_SZ + 2];
	__shared__	int sh_delayIndexStart[READ_CHUNK_SZ + 2];
	__shared__	int sh_firingId[READ_CHUNK_SZ + 2];
	__shared__ volatile int sh_NeuronCnt;

	const int threadIdWarp = (threadIdx.x % WARP_SIZE);
	const int warpId       = (threadIdx.x / WARP_SIZE);

	// this variable is used to record the
	// number of updates done by different blocks
	if(threadIdx.x<=0)   {
		sh_NeuronCnt = 0;
	}

	__syncthreads();

	// stores the number of fired neurons at time t
	int k = tex1Dfetch(timeTableD2GPU_tex, simTimeMs + networkConfigGPU.maxDelay + 1 + timeTableD2GPU_tex_offset) - 1;

	// stores the number of fired neurons at time (t - maxDelay_)
	int k_end = tex1Dfetch(timeTableD2GPU_tex, simTimeMs + 1 + timeTableD2GPU_tex_offset);

	int t_pos  = simTimeMs;

	// we need to read (k-k_end) neurons from the firing 
	// table and do necesary updates for all these post-synaptic
	// connection in these neurons..
	while ((k >= k_end) && (k >= 0)) {
		// at any point of time EXCIT_READ_CHUNK_SZ neurons
		// read different firing id from the firing table
		if (threadIdx.x < READ_CHUNK_SZ) { // use 64 threads
			int fPos = k - (READ_CHUNK_SZ * blockIdx.x) - threadIdx.x; 
			if ((fPos >= 0) && (fPos >= k_end)) {

				// get the neuron nid here....
				//int val = runtimeDataGPU.firingTableD2[fPos];
				//int nid = GET_FIRING_TABLE_NID(val);
				int nid = runtimeDataGPU.firingTableD2[fPos];

				// find the time of firing based on the firing number fPos
				while ( !((fPos >= tex1Dfetch(timeTableD2GPU_tex, t_pos + networkConfigGPU.maxDelay + timeTableD2GPU_tex_offset)) 
					&& (fPos < tex1Dfetch(timeTableD2GPU_tex, t_pos + networkConfigGPU.maxDelay + 1 + timeTableD2GPU_tex_offset)))) {
					t_pos--;
				}

				// find the time difference between firing of the neuron and the current time
				int tD  = simTimeMs - t_pos;

				// find the various delay parameters for neuron 'nid', with a delay of 'tD'
				//sh_axonDelay[threadIdx.x]	 = tD;
				int tPos = (networkConfigGPU.maxDelay + 1) * nid + tD;
				//sh_firingId[threadIdx.x]	 	 = val;
				sh_firingId[threadIdx.x] = nid;
				sh_neuronOffsetTable[threadIdx.x]= runtimeDataGPU.cumulativePost[nid];
				sh_delayLength[threadIdx.x]      = runtimeDataGPU.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]  = runtimeDataGPU.postDelayInfo[tPos].delay_index_start;

				// This is to indicate that the current thread
				// has a valid delay parameter for post-synaptic firing generation
				atomicAdd((int*)&sh_NeuronCnt, 1);
			}
		}

		__syncthreads();

		// if cnt is zero than no more neurons need to generate
		// post-synaptic firing, then we break the loop.
		if (sh_NeuronCnt == 0) {
			break;
		}

		// first WARP_SIZE threads the post synaptic
		// firing for first neuron, and so on. each of this group
		// needs to generate (numPostSynapses/maxDelay_) spikes for every fired neuron, every second
		// for numPostSynapses=500,maxDelay_=20, we need to generate 25 spikes for each fired neuron
		// for numPostSynapses=600,maxDelay_=20, we need to generate 30 spikes for each fired neuron 
		for (int pos = warpId; pos < sh_NeuronCnt; pos += (NUM_THREADS / WARP_SIZE)) {

			int delId = threadIdWarp;

			while (delId < sh_delayLength[pos]) {
				// get the post synaptic information for specific delay
				SynInfo postInfo = runtimeDataGPU.postSynapticIds[sh_neuronOffsetTable[pos] + sh_delayIndexStart[pos] + delId];
				int postNId = GET_CONN_NEURON_ID(postInfo); // get post-neuron id
				int synId = GET_CONN_SYN_ID(postInfo);      // get synaptic id

				if (postNId < networkConfigGPU.numN) // test if post-neuron is a local neuron
					generatePostSynapticSpike(simTime, sh_firingId[pos] /* preNId */, postNId, synId);

				delId += WARP_SIZE;
			}
		} //(for all excitory neurons in table)

		__syncthreads();

		if(threadIdx.x == 0) {
			sh_NeuronCnt = 0;
		}

		k = k - (gridDim.x * READ_CHUNK_SZ);

		__syncthreads();
	}

	__syncthreads();
}

/*!
 * \brief This kernel updating and generating spikes on connections with a delay of 1ms from the fired neuron.
 *
 * This function looks mostly like kernel_doCurrentUpdateD2() but has been optimized for a fixed delay of 1ms. 
 * Ultimately we may merge this kernel with the kernel_doCurrentUpdateD2().
 * The LTD computation is also executed by this kernel.
 *
 * net access: maxDelay, I_setPitch, sim_in_testing
 * grp access: Type, grpDA, WithSTDP, WithESTDP, WithISTDP, WithESTDPcurve, WithISTDPcurve, all STDP parameters
 * rtd access: postSynapticIds, cumulativePre, grpIds, I_set, wtChange, (R)lastSpikeTime, (W)synSpikeTime
 * glb access: timeTableD1GPU, spikeCountD1SecGPU, firingTableD1
 */
__global__ void kernel_doCurrentUpdateD1(int simTimeMs, int simTimeSec, int simTime) {
	__shared__ volatile	int sh_NeuronCnt;
	__shared__ volatile int sh_neuronOffsetTable[NUM_THREADS / WARP_SIZE + 2];
	__shared__ int sh_delayLength[NUM_THREADS / WARP_SIZE + 2];
	__shared__ int sh_firingId[NUM_THREADS / WARP_SIZE + 2];
	__shared__ int sh_delayIndexStart[NUM_THREADS / WARP_SIZE + 2];
	__shared__ int sh_timing;
	__shared__ int kPosEnd;

	const int warpId       = threadIdx.x / WARP_SIZE;  // warp id
	const int numWarps     = blockDim.x / WARP_SIZE;   // number of warp
	const int threadIdWarp = threadIdx.x % WARP_SIZE;  // thread id within a warp

	// load the time table for neuron firing
	if (threadIdx.x == 0) {
		sh_timing = timeTableD1GPU[simTimeMs + networkConfigGPU.maxDelay];   // number of fired neurons at simTimeMs - 1
		kPosEnd = timeTableD1GPU[simTimeMs + networkConfigGPU.maxDelay + 1]; // number of fired neurons at simTimeMs, which is equal to spikeCountD1SecGPU
	}
	__syncthreads();

	int kPos = sh_timing + (blockIdx.x * numWarps);

	__syncthreads();

	// Do current update as long as we have some valid neuron
	while ((kPos >= 0) && (kPos < kPosEnd)) {
		int fPos = -1;
		// a group of threads (4 threads) loads the delay information
		if (threadIdx.x < numWarps) {
			sh_neuronOffsetTable[threadIdx.x] = -1;
			fPos = kPos + threadIdx.x;

			// find the neuron nid and also delay information from fPos
			if ((fPos >= 0) && (fPos < kPosEnd)) {
				atomicAdd((int*)&sh_NeuronCnt, 1);
				//int val  = runtimeDataGPU.firingTableD1[fPos];
				//int nid  = GET_FIRING_TABLE_NID(val);
				int nid = runtimeDataGPU.firingTableD1[fPos];
				int tPos = (networkConfigGPU.maxDelay + 1) * nid;
				//sh_firingId[threadIdx.x] 	 	 = val;
				sh_firingId[threadIdx.x] = nid;
				sh_neuronOffsetTable[threadIdx.x] = runtimeDataGPU.cumulativePost[nid];
				sh_delayLength[threadIdx.x]       = runtimeDataGPU.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]   = runtimeDataGPU.postDelayInfo[tPos].delay_index_start;
			}
		}

		__syncthreads();

		// no more fired neuron from table... we just break from loop
		if (sh_NeuronCnt == 0) {
			break;
		}

		__syncthreads();

		int offset = sh_neuronOffsetTable[warpId];

		if (threadIdx.x == 0) {
			sh_NeuronCnt = 0;
		}

		// 32 threads for generatePostSynapticSpike()
		if (offset >= 0) {
			int delId = threadIdWarp;

			while (delId < sh_delayLength[warpId]) {
				// get the post synaptic information for specific delay
				SynInfo postInfo = runtimeDataGPU.postSynapticIds[offset + sh_delayIndexStart[warpId] + delId];
				int postNId = GET_CONN_NEURON_ID(postInfo); // get post-neuron id
				int synId = GET_CONN_SYN_ID(postInfo);      // get synaptic id

				if (postNId < networkConfigGPU.numN) // test if post-neuron is a local neuron
					generatePostSynapticSpike(simTime, sh_firingId[warpId] /* preNId */, postNId, synId);

				delId += WARP_SIZE;
			}
		}

		__syncthreads();

		kPos = kPos + (gridDim.x * numWarps);
	}
}

__global__ void kernel_convertExtSpikesD2(int startIdx, int endIdx, int GtoLOffset) {
	int firingTableIdx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;
	int spikeCountExtRx = endIdx - startIdx; // received external spike count

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		secD2fireCntTest += spikeCountExtRx;
		spikeCountD2SecGPU += spikeCountExtRx;
		spikeCountExtRxD2GPU += spikeCountExtRx;
		spikeCountExtRxD2SecGPU += spikeCountExtRx;
	}

	// FIXME: if endIdx - startIdx > 64 * 128
	if (firingTableIdx < endIdx)
		runtimeDataGPU.firingTableD2[firingTableIdx] += GtoLOffset;
}

__global__ void kernel_convertExtSpikesD1(int startIdx, int endIdx, int GtoLOffset) {
	int firingTableIdx = startIdx + blockIdx.x * blockDim.x + threadIdx.x;
	int spikeCountExtRx = endIdx - startIdx; // received external spike count

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		secD1fireCntTest += spikeCountExtRx;
		spikeCountD1SecGPU += spikeCountExtRx;
		spikeCountExtRxD1GPU += spikeCountExtRx;
		spikeCountExtRxD1SecGPU += spikeCountExtRx;
	}

	// FIXME: if endIdx - startIdx > 64 * 128
	if (firingTableIdx < endIdx)
		runtimeDataGPU.firingTableD1[firingTableIdx] += GtoLOffset;
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies information of pre-connections to it
 *
 * This function:
 * initialize Npre_plasticInv
 * (allocate and) copy Npre, Npre_plastic, Npre_plasticInv, cumulativePre, preSynapticIds
 * (allocate and) copy Npost, cumulativePost, postSynapticIds, postDelayInfo
 *
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copying
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU
 * \since v4.0
 */
void SNN::copyPreConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, 0); // check that the destination pointer is properly allocated..

	int lengthN, lengthSyn, posN, posSyn;

	if (lGrpId == ALL) {
		lengthN = networkConfigs[netId].numNAssigned;
		posN = 0;
	} else {
		lengthN = groupConfigs[netId][lGrpId].numN;
		posN = groupConfigs[netId][lGrpId].lStartN;
	}

	// connection synaptic lengths and cumulative lengths...
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre, sizeof(short) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Npre[posN], &src->Npre[posN], sizeof(short) * lengthN, kind));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	if (!sim_with_fixedwts) {
		// presyn excitatory connections
		if(allocateMem) 
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plastic, sizeof(short) * networkConfigs[netId].numNAssigned));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Npre_plastic[posN], &src->Npre_plastic[posN], sizeof(short) * lengthN, kind));

		// Npre_plasticInv is only used on GPUs, only allocate and copy it during initialization
		if(allocateMem) {
			float* Npre_plasticInv = new float[networkConfigs[netId].numNAssigned];

			for (int i = 0; i < networkConfigs[netId].numNAssigned; i++)
				Npre_plasticInv[i] = 1.0f / managerRuntimeData.Npre_plastic[i];

			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plasticInv, sizeof(float) * networkConfigs[netId].numNAssigned));
			CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre_plasticInv, Npre_plasticInv, sizeof(float) * networkConfigs[netId].numNAssigned, kind));

			delete[] Npre_plasticInv;
		}
	}
		
	// beginning position for the pre-synaptic information
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePre, sizeof(int) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->cumulativePre[posN], &src->cumulativePre[posN], sizeof(int) * lengthN, kind));

	// Npre, cumulativePre has been copied to destination
	if (lGrpId == ALL) {
		lengthSyn = networkConfigs[netId].numPreSynNet;
		posSyn = 0;
	} else {
		lengthSyn = 0;
		for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++)
			lengthSyn += dest->Npre[lNId];

		posSyn = dest->cumulativePre[groupConfigs[netId][lGrpId].lStartN];
	}

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->preSynapticIds, sizeof(SynInfo) * networkConfigs[netId].numPreSynNet));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->preSynapticIds[posSyn], &src->preSynapticIds[posSyn], sizeof(SynInfo) * lengthSyn, kind));
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies information of post-connections to it
 *
 * This function:
 * (allocate and) copy Npost, cumulativePost, postSynapticIds, postDelayInfo
 *
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copying
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU
 * \since v4.0
 */
void SNN::copyPostConnectionInfo(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, 0);// check that the destination pointer is properly allocated..

	int lengthN, lengthSyn, posN, posSyn;

	if (lGrpId == ALL) {
		lengthN = networkConfigs[netId].numNAssigned;
		posN = 0;
	} else {
		lengthN = groupConfigs[netId][lGrpId].numN;
		posN = groupConfigs[netId][lGrpId].lStartN;
	}

	// number of postsynaptic connections
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npost, sizeof(short) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Npost[posN], &src->Npost[posN], sizeof(short) * lengthN, kind));
	
	// beginning position for the post-synaptic information
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePost, sizeof(int) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->cumulativePost[posN], &src->cumulativePost[posN], sizeof(int) * lengthN, kind));

	
	// Npost, cumulativePost has been copied to destination
	if (lGrpId == ALL) {
		lengthSyn = networkConfigs[netId].numPostSynNet;
		posSyn = 0;
	} else {
		lengthSyn = 0;
		for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++)
			lengthSyn += dest->Npost[lNId];

		posSyn = dest->cumulativePost[groupConfigs[netId][lGrpId].lStartN];
	}

	// actual post synaptic connection information...
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postSynapticIds, sizeof(SynInfo) * networkConfigs[netId].numPostSynNet));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->postSynapticIds[posSyn], &src->postSynapticIds[posSyn], sizeof(SynInfo) * lengthSyn, kind));

	// static specific mapping and actual post-synaptic delay metric
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postDelayInfo, sizeof(DelayInfo) * networkConfigs[netId].numNAssigned * (glbNetworkConfig.maxDelay + 1)));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->postDelayInfo[posN * (glbNetworkConfig.maxDelay + 1)], &src->postDelayInfo[posN * (glbNetworkConfig.maxDelay + 1)], sizeof(DelayInfo) * lengthN * (glbNetworkConfig.maxDelay + 1), kind));
}

void SNN::checkDestSrcPtrs(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int lGrpId, int destOffset) {
	// source should always be allocated
	assert(src->allocated);

	if(kind == cudaMemcpyHostToDevice) {
		assert(src->memType  == CPU_MODE);
		assert(dest->memType == GPU_MODE);

		if (allocateMem) {
			assert(!dest->allocated); // if allocateMem = true, then the destination must be empty without allocation.
			assert(lGrpId == ALL); // if allocateMem = true, then we should not specify any specific group.
		} else {
			assert(dest->allocated); // if allocateMem = false, then the destination must be allocated.
		}

		assert(destOffset == 0); // H-to-D only allows local-to-local copy
	} else if (kind == cudaMemcpyDeviceToHost) {
		assert(src->memType  == GPU_MODE);
		assert(dest->memType == CPU_MODE);

		assert(dest->allocated);

		if (lGrpId == ALL)
			assert(destOffset == 0); // if copy all content, only local-to-local is allowed
	} else {
		KERNEL_ERROR("Wrong Host-Device copy direction");
		exitSimulation(1);
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies AMPA conductance to it
 *
 * This function:
 * (allocate and) copy gAMPA
 *
 * This funcion is called by copyNeuronState() and fetchConductanceAMPA(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copy
 * \param[in] allocateMem a flag indicates whether allocating memory space before copy
 * \param[in] destOffset the offset of data destination, which is used in local-to-global copy 
 *
 * \sa copyNeuronState fetchConductanceAMPA
 * \since v3.0
 */
void SNN::copyConductanceAMPA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, destOffset);// check that the destination pointer is properly allocated..
	
	assert(isSimulationWithCOBA());

	int ptrPos, length;

	if(lGrpId == ALL) {
		ptrPos = 0;
		length = networkConfigs[netId].numNReg;
	} else {
		ptrPos = groupConfigs[netId][lGrpId].lStartN;
		length = groupConfigs[netId][lGrpId].numN;
	}
	assert(length <= networkConfigs[netId].numNReg);
	assert(length > 0);

	//conductance information
	assert(src->gAMPA  != NULL);
	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gAMPA, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gAMPA[ptrPos + destOffset], &src->gAMPA[ptrPos], sizeof(float) * length, kind));
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies NMDA conductance to it
 *
 * This function:
 * (allocate and) copy gNMDA, gNMDA_r, gNMDA_d
 *
 * This funcion is called by copyNeuronState() and fetchConductanceNMDA(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copy
 * \param[in] allocateMem a flag indicates whether allocating memory space before copy
 * \param[in] destOffset the offset of data destination, which is used in local-to-global copy 
 *
 * \sa copyNeuronState fetchConductanceNMDA
 * \since v3.0
 */
void SNN::copyConductanceNMDA(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, destOffset);// check that the destination pointer is properly allocated..
	assert(isSimulationWithCOBA());

	int ptrPos, length;

	if(lGrpId == ALL) {
		ptrPos  = 0;
		length  = networkConfigs[netId].numNReg;
	} else {
		ptrPos  = groupConfigs[netId][lGrpId].lStartN;
		length  = groupConfigs[netId][lGrpId].numN;
	}
	assert(length  <= networkConfigs[netId].numNReg);
	assert(length > 0);

	if (isSimulationWithNMDARise()) {
		assert(src->gNMDA_r != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gNMDA_r, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gNMDA_r[ptrPos], &src->gNMDA_r[ptrPos], sizeof(float) * length, kind));

		assert(src->gNMDA_d != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gNMDA_d, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gNMDA_d[ptrPos], &src->gNMDA_d[ptrPos], sizeof(float) * length, kind));
	} else {
		assert(src->gNMDA != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gNMDA, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gNMDA[ptrPos + destOffset], &src->gNMDA[ptrPos], sizeof(float) * length, kind));
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies GABAa conductance to it
 *
 * This function:
 * (allocate and) copy gGABAa
 *
 * This funcion is called by copyNeuronState() and fetchConductanceGABAa(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copy
 * \param[in] allocateMem a flag indicates whether allocating memory space before copy
 * \param[in] destOffset the offset of data destination, which is used in local-to-global copy 
 *
 * \sa copyNeuronState fetchConductanceGABAa
 * \since v3.0
 */
void SNN::copyConductanceGABAa(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, destOffset); // check that the destination pointer is properly allocated..
	assert(isSimulationWithCOBA());

	int ptrPos, length;

	if(lGrpId == ALL) {
		ptrPos  = 0;
		length  = networkConfigs[netId].numNReg;
	} else {
		ptrPos  = groupConfigs[netId][lGrpId].lStartN;
		length  = groupConfigs[netId][lGrpId].numN;
	}
	assert(length  <= networkConfigs[netId].numNReg);
	assert(length > 0);

	assert(src->gGABAa != NULL);
	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gGABAa, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gGABAa[ptrPos + destOffset], &src->gGABAa[ptrPos], sizeof(float) * length, kind));
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies GABAb conductance to it
 *
 * This function:
 * (allocate and) copy gGABAb, gGABAb_r, gGABAb_d
 *
 * This funcion is called by copyNeuronState() and fetchConductanceGABAb(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copy
 * \param[in] allocateMem a flag indicates whether allocating memory space before copy
 * \param[in] destOffset the offset of data destination, which is used in local-to-global copy 
 *
 * \sa copyNeuronState fetchConductanceGABAb
 * \since v3.0
 */
void SNN::copyConductanceGABAb(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, destOffset); // check that the destination pointer is properly allocated..
	assert(isSimulationWithCOBA());

	int ptrPos, length;

	if(lGrpId == ALL) {
		ptrPos  = 0;
		length  = networkConfigs[netId].numNReg;
	} else {
		ptrPos  = groupConfigs[netId][lGrpId].lStartN;
		length  = groupConfigs[netId][lGrpId].numN;
	}
	assert(length <= networkConfigs[netId].numNReg);
	assert(length > 0);

	if (isSimulationWithGABAbRise()) {
		assert(src->gGABAb_r != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gGABAb_r, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gGABAb_r[ptrPos], &src->gGABAb_r[ptrPos], sizeof(float) * length, kind));

		assert(src->gGABAb_d != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->gGABAb_d, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gGABAb_d[ptrPos], &src->gGABAb_d[ptrPos], sizeof(float) * length, kind));
	} else {
		assert(src->gGABAb != NULL);
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->gGABAb, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->gGABAb[ptrPos + destOffset], &src->gGABAb[ptrPos], sizeof(float) * length, kind));
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies variables related to nueron state to it
 *
 * This function:
 * (allocate and) copy voltage, recovery, current, avgFiring 
 *
 * This funcion is called by allocateSNN_GPU(). Only copying from host to device is required
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU fetchNeuronState
 * \since v3.0
 */
void SNN::copyNeuronState(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, lGrpId, 0); // check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyHostToDevice);

	int ptrPos, length;

	if(lGrpId == ALL) {
		ptrPos  = 0;
		length  = networkConfigs[netId].numNReg;
	}
	else {
		ptrPos  = groupConfigs[netId][lGrpId].lStartN;
		length  = groupConfigs[netId][lGrpId].numN;
	}

	assert(length <= networkConfigs[netId].numNReg);
	
	if (length == 0)
		return;

	if(!allocateMem && groupConfigs[netId][lGrpId].Type & POISSON_NEURON)
		return;

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->recovery, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->recovery[ptrPos], &managerRuntimeData.recovery[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->voltage, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->voltage[ptrPos], &managerRuntimeData.voltage[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	//neuron input current...
	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->current, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->current[ptrPos], &managerRuntimeData.current[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if (sim_with_conductances) {
	    //conductance information
		copyConductanceAMPA(netId, lGrpId, dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, 0);
		copyConductanceNMDA(netId, lGrpId, dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, 0);
		copyConductanceGABAa(netId, lGrpId, dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, 0);
		copyConductanceGABAb(netId, lGrpId, dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, 0);
	}

	// copying external current needs to be done separately because setExternalCurrent needs to call it, too
	// do it only from host to device
	copyExternalCurrent(netId, lGrpId, dest, cudaMemcpyHostToDevice, allocateMem);
	
	copyNeuronParameters(netId, lGrpId, dest, cudaMemcpyHostToDevice, allocateMem);

	if (sim_with_homeostasis) {
		//Included to enable homeostasis in GPU_MODE.
		// Avg. Firing...
		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->avgFiring, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->avgFiring[ptrPos], &managerRuntimeData.avgFiring[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies the spike count of each neuron to it
 *
 * This function:
 * (allocate and) copy nSpikeCnt
 *
 * This funcion is called by copyAuxiliaryData() and fetchNeuronSpikeCount(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copy
 * \param[in] allocateMem a flag indicates whether allocating memory space before copy
 * \param[in] destOffset the offset of data destination, which is used in local-to-global copy 
 *
 * \sa copyAuxiliaryData fetchNeuronSpikeCount
 * \since v4.0
 */
void SNN::copyNeuronSpikeCount(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int destOffset) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, destOffset);// check that the destination pointer is properly allocated..

	int posN, lengthN;

	if(lGrpId == ALL) {
		posN = 0;
		lengthN = networkConfigs[netId].numN;
	} else {
		posN = groupConfigs[netId][lGrpId].lStartN;
		lengthN = groupConfigs[netId][lGrpId].numN;
	}
	assert(lengthN > 0 && lengthN <= networkConfigs[netId].numN);

	// spike count information
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->nSpikeCnt, sizeof(int) * lengthN));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->nSpikeCnt[posN + destOffset], &src->nSpikeCnt[posN], sizeof(int) * lengthN, kind));
}

// FIXME: move grpDA(5HT, ACh, NE)Buffer to copyAuxiliaryData
/*!
 * \brief this function allocates device (GPU) memory sapce and copies variables related to group state to it
 *
 * This function:
 * (allocate and) copy grpDA, grp5HT, grpACh, grpNE, grpDABuffer, grp5HTBuffer, grpAChBuffer, grpNEBuffer
 *
 * This funcion is called by allocateSNN_GPU() and fetchGroupState(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copying
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU fetchGroupState
 * \since v3.0
 */
void SNN::copyGroupState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, 0);// check that the destination pointer is properly allocated..

	if (allocateMem) {
		assert(dest->memType == GPU_MODE && !dest->allocated);
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDA, sizeof(float) * networkConfigs[netId].numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HT, sizeof(float) * networkConfigs[netId].numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpACh, sizeof(float) * networkConfigs[netId].numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNE, sizeof(float) * networkConfigs[netId].numGroups));
	}
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDA, src->grpDA, sizeof(float) * networkConfigs[netId].numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HT, src->grp5HT, sizeof(float) * networkConfigs[netId].numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpACh, src->grpACh, sizeof(float) * networkConfigs[netId].numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNE, src->grpNE, sizeof(float) * networkConfigs[netId].numGroups, kind));

	if (lGrpId < 0) {
		if (allocateMem) {
			assert(dest->memType == GPU_MODE && !dest->allocated);
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDABuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HTBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpAChBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNEBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups));
		}
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDABuffer, src->grpDABuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HTBuffer, src->grp5HTBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpAChBuffer, src->grpAChBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNEBuffer, src->grpNEBuffer, sizeof(float) * 1000 * networkConfigs[netId].numGroups, kind));
	} else {
		assert(!allocateMem);
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->grpDABuffer[lGrpId * 1000], &src->grpDABuffer[lGrpId * 1000], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->grp5HTBuffer[lGrpId * 1000], &src->grp5HTBuffer[lGrpId * 1000], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->grpAChBuffer[lGrpId * 1000], &src->grpAChBuffer[lGrpId * 1000], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->grpNEBuffer[lGrpId * 1000], &src->grpNEBuffer[lGrpId * 1000], sizeof(float) * 1000, kind));
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies neural parameters to it
 *
 * This function:
 * (allocate and) copy Izh_a, Izh_b, Izh_c, Izh_d
 * initialize baseFiringInv
 * (allocate and) copy baseFiring, baseFiringInv
 *
 * This funcion is only called by copyNeuronState(). Only copying direction from host to device is required.
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa copyNeuronState
 * \since v3.0
 */
void SNN::copyNeuronParameters(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	assert(kind == cudaMemcpyHostToDevice);

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, lGrpId, 0);

	// check that the destination pointer is properly allocated..
	// cannot use checkDestSrcPtrs here because src pointer would be NULL
	if (dest->allocated && allocateMem) {
		KERNEL_ERROR("GPU Memory already allocated...");
		exitSimulation(1);
	}

	// when allocating we are allocating the memory.. we need to do it completely... to avoid memory fragmentation..
	if (allocateMem) {
		assert(lGrpId == ALL);
		assert(dest->Izh_a == NULL);
		assert(dest->Izh_b == NULL);
		assert(dest->Izh_c == NULL);
		assert(dest->Izh_d == NULL);
	}

	if(lGrpId == ALL) {
		ptrPos = 0;
		length = networkConfigs[netId].numNReg;
	}
	else {
		ptrPos = groupConfigs[netId][lGrpId].lStartN;
		length = groupConfigs[netId][lGrpId].numN;
	}

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_a, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_a[ptrPos], &(managerRuntimeData.Izh_a[ptrPos]), sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_b, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_b[ptrPos], &(managerRuntimeData.Izh_b[ptrPos]), sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_c, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_c[ptrPos], &(managerRuntimeData.Izh_c[ptrPos]), sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_d, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_d[ptrPos], &(managerRuntimeData.Izh_d[ptrPos]), sizeof(float) * length, cudaMemcpyHostToDevice));

	// pre-compute baseFiringInv for fast computation on GPUs.
	if (sim_with_homeostasis) {
		float* baseFiringInv = new float[length];
		for(int nid = 0; nid < length; nid++) {
			if (managerRuntimeData.baseFiring[nid] != 0.0f)
				baseFiringInv[nid] = 1.0f / managerRuntimeData.baseFiring[ptrPos + nid];
			else
				baseFiringInv[nid] = 0.0;
		}

		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->baseFiringInv, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->baseFiringInv[ptrPos], baseFiringInv, sizeof(float) * length, cudaMemcpyHostToDevice));

		if(allocateMem) CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->baseFiring, sizeof(float) * length));
		CUDA_CHECK_ERRORS(cudaMemcpy(&dest->baseFiring[ptrPos], managerRuntimeData.baseFiring, sizeof(float) * length, cudaMemcpyHostToDevice));

		delete [] baseFiringInv;
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies short-term plasticity (STP) state to it
 *
 * This function:
 * initialize STP_Pitch
 * (allocate and) copy stpu, stpx
 *
 * This funcion is called by allocateSNN_GPU() and fetchSTPState(). It supports bi-directional copying
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] kind the direction of copying
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU fetchSTPState
 * \since v3.0
 */
void SNN::copySTPState(int netId, int lGrpId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, lGrpId, 0); // check that the destination pointer is properly allocated..
	
	
	// STP feature is optional, do addtional check for memory space
	if(allocateMem) {
		assert(dest->stpu == NULL);
		assert(dest->stpx == NULL);
	} else {
		assert(dest->stpu != NULL);
		assert(dest->stpx != NULL);
	}
	assert(src->stpu != NULL); assert(src->stpx != NULL);

	size_t STP_Pitch;
	size_t widthInBytes = sizeof(float) * networkConfigs[netId].numN;

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpu, sizeof(float)*networkConfigs[0].numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpu[0], &src->stpu[0], sizeof(float)*networkConfigs[0].numN, kind));

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpx, sizeof(float)*networkConfigs[0].numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpx[0], &src->stpx[0], sizeof(float)*networkConfigs[0].numN, kind));

	// allocate the stpu and stpx variable
	if (allocateMem)
		CUDA_CHECK_ERRORS(cudaMallocPitch((void**)&dest->stpu, &networkConfigs[netId].STP_Pitch, widthInBytes, networkConfigs[netId].maxDelay + 1));
	if (allocateMem)
		CUDA_CHECK_ERRORS(cudaMallocPitch((void**)&dest->stpx, &STP_Pitch, widthInBytes, networkConfigs[netId].maxDelay + 1));

	assert(networkConfigs[netId].STP_Pitch > 0);
	assert(STP_Pitch > 0);				// stp_pitch should be greater than zero
	assert(STP_Pitch == networkConfigs[netId].STP_Pitch);	// we want same Pitch for stpu and stpx
	assert(networkConfigs[netId].STP_Pitch >= widthInBytes);	// stp_pitch should be greater than the width
	// convert the Pitch value to multiples of float
	assert(networkConfigs[netId].STP_Pitch % (sizeof(float)) == 0);
	if (allocateMem)
		networkConfigs[netId].STP_Pitch = networkConfigs[netId].STP_Pitch/sizeof(float);

//	fprintf(stderr, "STP_Pitch = %ld, STP_witdhInBytes = %d\n", networkConfigs[0].STP_Pitch, widthInBytes);

	float* tmp_stp = new float[networkConfigs[netId].numN];
	// copy the already generated values of stpx and stpu to the GPU
	for(int t = 0; t < networkConfigs[netId].maxDelay + 1; t++) {
		if (kind == cudaMemcpyHostToDevice) {
			// stpu in the CPU might be mapped in a specific way. we want to change the format
			// to something that is okay with the GPU STP_U and STP_X variable implementation..
			for (int n = 0; n < networkConfigs[netId].numN; n++) {
				int idx = STP_BUF_POS(n, t, glbNetworkConfig.maxDelay);
				tmp_stp[n] = managerRuntimeData.stpu[idx];
				//assert(tmp_stp[n] == 0.0f); // STP is not enabled for all groups
			}
			CUDA_CHECK_ERRORS(cudaMemcpy(&dest->stpu[t * networkConfigs[netId].STP_Pitch], tmp_stp, sizeof(float) * networkConfigs[netId].numN, cudaMemcpyHostToDevice));
			for (int n = 0; n < networkConfigs[netId].numN; n++) {
				int idx = STP_BUF_POS(n, t, glbNetworkConfig.maxDelay);
				tmp_stp[n] = managerRuntimeData.stpx[idx];
				//assert(tmp_stp[n] == 1.0f); // STP is not enabled for all groups
			}
			CUDA_CHECK_ERRORS(cudaMemcpy(&dest->stpx[t * networkConfigs[netId].STP_Pitch], tmp_stp, sizeof(float) * networkConfigs[netId].numN, cudaMemcpyHostToDevice));
		} else {
			CUDA_CHECK_ERRORS(cudaMemcpy(tmp_stp, &dest->stpu[t * networkConfigs[netId].STP_Pitch], sizeof(float) * networkConfigs[netId].numN, cudaMemcpyDeviceToHost));
			for (int n = 0; n < networkConfigs[netId].numN; n++)
				managerRuntimeData.stpu[STP_BUF_POS(n, t, glbNetworkConfig.maxDelay)] = tmp_stp[n];
			CUDA_CHECK_ERRORS(cudaMemcpy(tmp_stp, &dest->stpx[t * networkConfigs[netId].STP_Pitch], sizeof(float) * networkConfigs[netId].numN, cudaMemcpyDeviceToHost));
			for (int n = 0; n < networkConfigs[netId].numN; n++)
				managerRuntimeData.stpx[STP_BUF_POS(n, t, glbNetworkConfig.maxDelay)] = tmp_stp[n];
		}
	}
	delete [] tmp_stp;
}

/*!
 * \brief This function copies networkConfig form host to device
 *
 * This function:
 * copy networkConfig
 *
 * \param[in] netId the id of a local network whose networkConfig will be copied to device (GPU) memory
 *
 * \since v4.0
 */
void SNN::copyNetworkConfig(int netId) {
	checkAndSetGPUDevice(netId);
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(networkConfigGPU, &networkConfigs[netId], sizeof(NetworkConfigRT), 0, cudaMemcpyHostToDevice));
}

/*!
 * \brief This function copies groupConfigs form host to device
 *
 * This function:
 * copy groupConfigs
 *
 * \param[in] netId the id of a local network whose groupConfigs will be copied to device (GPU) memory
 *
 * \since v4.0
 */
void SNN::copyGroupConfigs(int netId) {
	checkAndSetGPUDevice(netId);
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(groupConfigsGPU, groupConfigs[netId], (networkConfigs[netId].numGroupsAssigned) * sizeof(GroupConfigRT), 0, cudaMemcpyHostToDevice));
}

/*!
 * \brief this function copy weight state in device (GPU) memory sapce to main (CPU) memory space
 *
 * This function:
 * copy wt, wtChange synSpikeTime
 *
 * This funcion is only called by fetchWeightState(). Only copying direction from device to host is required.
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 *
 * \sa fetchWeightState
 * \since v4.0
 */
void SNN::copyWeightState(int netId, int lGrpId, cudaMemcpyKind kind) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(&managerRuntimeData, &gpuRuntimeData[netId], cudaMemcpyDeviceToHost, false, lGrpId, 0); // check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyDeviceToHost);

	int lengthSyn, posSyn;

	// first copy pre-connections info
	copyPreConnectionInfo(netId, lGrpId, &managerRuntimeData, &gpuRuntimeData[netId], cudaMemcpyDeviceToHost, false);

	if (lGrpId == ALL) {
		lengthSyn = networkConfigs[netId].numPreSynNet;
		posSyn = 0;
	} else {
		lengthSyn = 0;
		for (int lNId = groupConfigs[netId][lGrpId].lStartN; lNId <= groupConfigs[netId][lGrpId].lEndN; lNId++)
			lengthSyn += managerRuntimeData.Npre[lNId];

		posSyn = managerRuntimeData.cumulativePre[groupConfigs[netId][lGrpId].lStartN];
	}

	assert(posSyn < networkConfigs[netId].numPreSynNet || networkConfigs[netId].numPreSynNet == 0);
	assert(lengthSyn <= networkConfigs[netId].numPreSynNet);

	CUDA_CHECK_ERRORS(cudaMemcpy(&managerRuntimeData.wt[posSyn], &gpuRuntimeData[netId].wt[posSyn], sizeof(float) * lengthSyn, cudaMemcpyDeviceToHost));

	// copy firing time for individual synapses
	//CUDA_CHECK_ERRORS(cudaMemcpy(&managerRuntimeData.synSpikeTime[cumPos_syn], &gpuRuntimeData[netId].synSpikeTime[cumPos_syn], sizeof(int) * length_wt, cudaMemcpyDeviceToHost));

	if ((!sim_with_fixedwts) || sim_with_stdp) {
		// copy synaptic weight derivative
		CUDA_CHECK_ERRORS(cudaMemcpy( &managerRuntimeData.wtChange[posSyn], &gpuRuntimeData[netId].wtChange[posSyn], sizeof(float) * lengthSyn, cudaMemcpyDeviceToHost));
	}
}


/*!
 * \brief this function allocates device (GPU) memory sapce and copies variables related to syanpses to it
 *
 * This function:
 * (allocate and) copy wt, wtChange, maxSynWt
 *
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] src pointer to runtime data source
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU
 * \since v4.0
 */
void SNN::copySynapseState(int netId, RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, src, kind, allocateMem, ALL, 0); // check that the destination pointer is properly allocated..
	
	assert(networkConfigs[netId].numPreSynNet > 0);

	// synaptic information based
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->wt, sizeof(float) * networkConfigs[netId].numPreSynNet));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->wt, src->wt, sizeof(float) * networkConfigs[netId].numPreSynNet, kind));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	// they show up in gpuUpdateLTP() and updateSynapticWeights(), two functions that do not get called if
	// sim_with_fixedwts is set
	if (!sim_with_fixedwts) {
		// synaptic weight derivative
		if(allocateMem)
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->wtChange, sizeof(float) * networkConfigs[netId].numPreSynNet));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->wtChange, src->wtChange, sizeof(float) * networkConfigs[netId].numPreSynNet, kind));

		// synaptic weight maximum value
		if(allocateMem)
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->maxSynWt, sizeof(float) * networkConfigs[netId].numPreSynNet));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->maxSynWt, src->maxSynWt, sizeof(float) * networkConfigs[netId].numPreSynNet, kind));
	}
}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies auxiliary runtime data to it
 *
 * This function:
 * (allocate and) reset spikeGenBits, poissonFireRate
 * initialize I_setLength, I_setPitch; (allocate and) reset I_set
 * (allocate and) copy synSpikeTime, lastSpikeTime
 * (allocate and) copy nSpikeCnt
 * (allocate and) copy grpIds, connIdsPreIdx
 * (allocate and) copy firingTableD1, firingTableD2
 * This funcion is only called by allocateSNN_GPU. Therefore, only copying direction from host to device is required
 *
 * \param[in] netId the id of local network, which is the same as device (GPU) id
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU
 * \since v4.0
 */
void SNN::copyAuxiliaryData(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, ALL, 0); // check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyHostToDevice);

	assert(networkConfigs[netId].numN > 0);

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->spikeGenBits, sizeof(int) * (networkConfigs[netId].numNSpikeGen / 32 + 1)));
	CUDA_CHECK_ERRORS(cudaMemset(dest->spikeGenBits, 0, sizeof(int) * (networkConfigs[netId].numNSpikeGen / 32 + 1)));

	// allocate the poisson neuron poissonFireRate
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->poissonFireRate, sizeof(float) * networkConfigs[netId].numNPois));
	CUDA_CHECK_ERRORS(cudaMemset(dest->poissonFireRate, 0, sizeof(float) * networkConfigs[netId].numNPois));

	// synaptic auxiliary data
	// I_set: a bit vector indicates which synapse got a spike

	if(allocateMem) {
		networkConfigs[netId].I_setLength = ceil(((networkConfigs[netId].maxNumPreSynN) / 32.0f));
		CUDA_CHECK_ERRORS(cudaMallocPitch((void**)&dest->I_set, &networkConfigs[netId].I_setPitch, sizeof(int) * networkConfigs[netId].numNReg, networkConfigs[netId].I_setLength));
	}
	assert(networkConfigs[netId].I_setPitch > 0 || networkConfigs[netId].maxNumPreSynN == 0);
	CUDA_CHECK_ERRORS(cudaMemset(dest->I_set, 0, networkConfigs[netId].I_setPitch * networkConfigs[netId].I_setLength));

	// synSpikeTime: an array indicates the last time when a synapse got a spike
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->synSpikeTime, sizeof(int) * networkConfigs[netId].numPreSynNet));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->synSpikeTime, managerRuntimeData.synSpikeTime, sizeof(int) * networkConfigs[netId].numPreSynNet, cudaMemcpyHostToDevice));

	// neural auxiliary data
	// lastSpikeTime: an array indicates the last time of a neuron emitting a spike
	// neuron firing time
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->lastSpikeTime, sizeof(int) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->lastSpikeTime, managerRuntimeData.lastSpikeTime, sizeof(int) * networkConfigs[netId].numNAssigned, cudaMemcpyHostToDevice));

	// auxiliary data for recording spike count of each neuron
	copyNeuronSpikeCount(netId, lGrpId, dest, &managerRuntimeData, cudaMemcpyHostToDevice, true, 0);

	// quick lookup array for local group ids
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc( (void**)&dest->grpIds, sizeof(short int) * networkConfigs[netId].numNAssigned));
	CUDA_CHECK_ERRORS(cudaMemcpy( dest->grpIds, managerRuntimeData.grpIds, sizeof(short int) * networkConfigs[netId].numNAssigned, cudaMemcpyHostToDevice));

	// quick lookup array for conn ids
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->connIdsPreIdx, sizeof(short int) * networkConfigs[netId].numPreSynNet));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->connIdsPreIdx, managerRuntimeData.connIdsPreIdx, sizeof(short int) * networkConfigs[netId].numPreSynNet, cudaMemcpyHostToDevice));

	// firing table
	if(allocateMem) {
		assert(dest->firingTableD1 == NULL);
		assert(dest->firingTableD2 == NULL);
	}

	// allocate 1ms firing table
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->firingTableD1, sizeof(int) * networkConfigs[netId].maxSpikesD1));
	if (networkConfigs[netId].maxSpikesD1 > 0)
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->firingTableD1, managerRuntimeData.firingTableD1, sizeof(int) * networkConfigs[netId].maxSpikesD1, cudaMemcpyHostToDevice));

	// allocate 2+ms firing table
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->firingTableD2, sizeof(int) * networkConfigs[netId].maxSpikesD2));
	if (networkConfigs[netId].maxSpikesD2 > 0)
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->firingTableD2, managerRuntimeData.firingTableD2, sizeof(int) * networkConfigs[netId].maxSpikesD2, cudaMemcpyHostToDevice));

	// allocate external 1ms firing table
	if (allocateMem) {
		void* devPtr;
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extFiringTableD1, sizeof(int*) * networkConfigs[netId].numGroups));
		CUDA_CHECK_ERRORS(cudaMemset(dest->extFiringTableD1, 0 /* NULL */, sizeof(int*) * networkConfigs[netId].numGroups));
		for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
			if (groupConfigs[netId][lGrpId].hasExternalConnect) {
				CUDA_CHECK_ERRORS(cudaMalloc((void**)&devPtr, sizeof(int) * groupConfigs[netId][lGrpId].numN * NEURON_MAX_FIRING_RATE));
				CUDA_CHECK_ERRORS(cudaMemset(devPtr, 0 , sizeof(int) * groupConfigs[netId][lGrpId].numN * NEURON_MAX_FIRING_RATE));
				CUDA_CHECK_ERRORS(cudaMemcpy(&dest->extFiringTableD1[lGrpId], &devPtr, sizeof(int*), cudaMemcpyHostToDevice));
			}
		}
	}

	// allocate external 2+ms firing table
	if (allocateMem) {
		void* devPtr;
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extFiringTableD2, sizeof(int*) * networkConfigs[netId].numGroups));
		CUDA_CHECK_ERRORS(cudaMemset(dest->extFiringTableD2, 0 /* NULL */, sizeof(int*) * networkConfigs[netId].numGroups));
		for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
			if (groupConfigs[netId][lGrpId].hasExternalConnect) {
				CUDA_CHECK_ERRORS(cudaMalloc((void**)&devPtr, sizeof(int) * groupConfigs[netId][lGrpId].numN * NEURON_MAX_FIRING_RATE));
				CUDA_CHECK_ERRORS(cudaMemset(devPtr, 0 , sizeof(int) * groupConfigs[netId][lGrpId].numN * NEURON_MAX_FIRING_RATE));
				CUDA_CHECK_ERRORS(cudaMemcpy(&dest->extFiringTableD2[lGrpId], &devPtr, sizeof(int*), cudaMemcpyHostToDevice));
			}
		}
	}

	// allocate external 1ms firing table index
	if (allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extFiringTableEndIdxD1, sizeof(int) * networkConfigs[netId].numGroups));
	CUDA_CHECK_ERRORS(cudaMemset(dest->extFiringTableEndIdxD1, 0, sizeof(int) * networkConfigs[netId].numGroups));


	// allocate external 2+ms firing table index
	if (allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extFiringTableEndIdxD2, sizeof(int) * networkConfigs[netId].numGroups));
	CUDA_CHECK_ERRORS(cudaMemset(dest->extFiringTableEndIdxD2, 0, sizeof(int) * networkConfigs[netId].numGroups));
}

void SNN::copyGrpIdsLookupArray(int netId, cudaMemcpyKind kind) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(&managerRuntimeData, &gpuRuntimeData[netId], cudaMemcpyDeviceToHost, false, ALL, 0);// check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyDeviceToHost);

	CUDA_CHECK_ERRORS(cudaMemcpy(managerRuntimeData.grpIds, gpuRuntimeData[netId].grpIds, sizeof(short int) *  networkConfigs[netId].numNAssigned, cudaMemcpyDeviceToHost));
}

void SNN::copyConnIdsLookupArray(int netId, cudaMemcpyKind kind) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(&managerRuntimeData, &gpuRuntimeData[netId], cudaMemcpyDeviceToHost, false, ALL, 0);// check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyDeviceToHost);

	CUDA_CHECK_ERRORS(cudaMemcpy(managerRuntimeData.connIdsPreIdx, gpuRuntimeData[netId].connIdsPreIdx, sizeof(short int) *  networkConfigs[netId].numPreSynNet, cudaMemcpyDeviceToHost));
}

void SNN::copyLastSpikeTime(int netId, cudaMemcpyKind kind) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(&managerRuntimeData, &gpuRuntimeData[netId], cudaMemcpyDeviceToHost, false, ALL, 0); // check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyDeviceToHost);

	CUDA_CHECK_ERRORS(cudaMemcpy(managerRuntimeData.lastSpikeTime, gpuRuntimeData[netId].lastSpikeTime, sizeof(int) *  networkConfigs[netId].numN, cudaMemcpyDeviceToHost));
}

// spikeGeneratorUpdate on GPUs..
void SNN::spikeGeneratorUpdate_GPU() {
	// If poisson rate has been updated, assign new poisson rate
	if (spikeRateUpdated) {
		assignPoissonFiringRate_GPU();
		spikeRateUpdated = false;
	}

	// If time slice has expired, check if new spikes needs to be generated by user-defined spike generators
	generateUserDefinedSpikes();

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			// update the random number for poisson spike generator (spikes generated by rate)
			if((networkConfigs[netId].numNPois > 0) && (gpuRuntimeData[netId].gpuRandGen != NULL)) {
				curandGenerateUniform(gpuRuntimeData[netId].gpuRandGen, gpuRuntimeData[netId].randNum, networkConfigs[netId].numNPois);
			}

			// Use spike generators (user-defined callback function)
			if (networkConfigs[netId].numNSpikeGen > 0) {
				assert(managerRuntimeData.spikeGenBits != NULL);

				// reset the bit status of the spikeGenBits...
				memset(managerRuntimeData.spikeGenBits, 0, sizeof(int) * (networkConfigs[netId].numNSpikeGen / 32 + 1));

				// fill spikeGenBits from SpikeBuffer
				fillSpikeGenBits(netId);

				// copy the spikeGenBits from the manager to the GPU..
				CUDA_CHECK_ERRORS(cudaMemcpy(gpuRuntimeData[netId].spikeGenBits, managerRuntimeData.spikeGenBits, sizeof(int) * (networkConfigs[netId].numNSpikeGen / 32 + 1), cudaMemcpyHostToDevice));
			}
		}
	}

	// tell the spike buffer to advance to the next time step
	spikeBuf->step();
}

void SNN::findFiring_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			kernel_findFiring<<<NUM_BLOCKS, NUM_THREADS>>>(simTime);
			CUDA_GET_LAST_ERROR("findFiring kernel failed\n");
		}
	}
}

void SNN::updateTimingTable_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			kernel_updateTimeTable<<<NUM_BLOCKS, NUM_THREADS>>>(simTimeMs);
			CUDA_GET_LAST_ERROR("timing Table update kernel failed\n");
		}
	}
}

void SNN::doCurrentUpdate_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			if(glbNetworkConfig.maxDelay > 1) {
				kernel_doCurrentUpdateD2<<<NUM_BLOCKS, NUM_THREADS>>>(simTimeMs,simTimeSec,simTime);
				CUDA_GET_LAST_ERROR("Kernel execution failed");
			}
		}
	}

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			kernel_doCurrentUpdateD1<<<NUM_BLOCKS, NUM_THREADS>>>(simTimeMs,simTimeSec,simTime);
			CUDA_GET_LAST_ERROR("Kernel execution failed");
		}
	}
}

void SNN::doSTPUpdateAndDecayCond_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(gpuRuntimeData[netId].allocated);

			if (sim_with_stp || sim_with_conductances) {
				kernel_STPUpdateAndDecayConductances<<<NUM_BLOCKS, NUM_THREADS>>>(simTimeMs, simTimeSec, simTime);
				CUDA_GET_LAST_ERROR("STP update\n");
			
			}
		}
	}
}

void SNN::initGPU(int netId) {
	checkAndSetGPUDevice(netId);

	assert(gpuRuntimeData[netId].allocated);

	kernel_initGPUMemory<<<NUM_BLOCKS, NUM_THREADS>>>();
	CUDA_GET_LAST_ERROR("initGPUMemory kernel failed\n");
}

//void SNN::printCurrentInfo(FILE* fp) {
//	checkAndSetGPUDevice();
//
//	KERNEL_WARN("Calling printCurrentInfo with fp is deprecated");
//	// copy neuron input current...
//	KERNEL_DEBUG("Total Synaptic updates:");
//	CUDA_CHECK_ERRORS( cudaMemcpy( managerRuntimeData.current, gpuRuntimeData[0].current, sizeof(float)*numNReg, cudaMemcpyDeviceToHost));
//		for(int i=0; i < numNReg; i++) {
//			if (managerRuntimeData.current[i] != 0.0 ) {
//				KERNEL_DEBUG("I[%d] -> %f", i, managerRuntimeData.current[i]);
//		}
//	}
//	fflush(fp);
//}

// TODO FIXME there's more...
void SNN::deleteObjects_GPU() {
	// wait for kernels to complete
	CUDA_CHECK_ERRORS(cudaThreadSynchronize());

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);

			// cudaFree all device pointers
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].voltage) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].recovery) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].current) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].extCurrent) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Npre) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Npre_plastic) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Npre_plasticInv) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Npost) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].cumulativePost) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].cumulativePre) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].synSpikeTime) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].wt) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].wtChange) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].maxSynWt) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].nSpikeCnt) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].avgFiring) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].baseFiring) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].baseFiringInv) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpDA) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grp5HT) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpACh) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpNE) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpDABuffer) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grp5HTBuffer) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpAChBuffer) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpNEBuffer) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].grpIds) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Izh_a) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Izh_b) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Izh_c) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].Izh_d) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gAMPA) );
			if (sim_with_NMDA_rise) {
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gNMDA_r) );
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gNMDA_d) );
			} else {
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gNMDA) );
			}
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gGABAa) );
			if (sim_with_GABAb_rise) {
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gGABAb_r) );
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gGABAb_d) );
			} else {
				CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].gGABAb) );
			}

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].stpu) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].stpx) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].connIdsPreIdx) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].groupIdInfo) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].neuronAllocation) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].postDelayInfo) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].postSynapticIds) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].preSynapticIds) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].I_set) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].poissonFireRate) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].lastSpikeTime) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].spikeGenBits) );

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].firingTableD2) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].firingTableD1) );

			int** tempPtrs;
			tempPtrs = new int*[networkConfigs[netId].numGroups];

			// fetch device memory address stored in extFiringTableD2
			CUDA_CHECK_ERRORS( cudaMemcpy(tempPtrs, gpuRuntimeData[netId].extFiringTableD2, sizeof(int*) * networkConfigs[netId].numGroups, cudaMemcpyDeviceToHost) );
			for (int i = 0; i < networkConfigs[netId].numGroups; i++)
				CUDA_CHECK_ERRORS( cudaFree(tempPtrs[i]) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].extFiringTableD2) );

			// fetch device memory address stored in extFiringTableD1
			CUDA_CHECK_ERRORS( cudaMemcpy(tempPtrs, gpuRuntimeData[netId].extFiringTableD1, sizeof(int*) * networkConfigs[netId].numGroups, cudaMemcpyDeviceToHost) );
			for (int i = 0; i < networkConfigs[netId].numGroups; i++)
				CUDA_CHECK_ERRORS( cudaFree(tempPtrs[i]) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].extFiringTableD1) );

			delete[] tempPtrs;

			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].extFiringTableEndIdxD2) );
			CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData[netId].extFiringTableEndIdxD1) );

			// delete random numbr generator on GPU(s)
			// Note: RNG_rand48 objects allocate device memory
			if (gpuRuntimeData[netId].gpuRandGen != NULL) curandDestroyGenerator(gpuRuntimeData[netId].gpuRandGen);
			gpuRuntimeData[netId].gpuRandGen = NULL;

			if (gpuRuntimeData[netId].randNum != NULL) CUDA_CHECK_ERRORS(cudaFree(gpuRuntimeData[netId].randNum));
			gpuRuntimeData[netId].randNum = NULL;
		}
	}

	CUDA_DELETE_TIMER(timer);
}

void SNN::globalStateUpdate_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);

			kernel_conductanceUpdate<<<NUM_BLOCKS, NUM_THREADS>>>(simTimeMs, simTimeSec, simTime);
			CUDA_GET_LAST_ERROR("kernel_conductanceUpdate failed");

			// use memset to reset I_set for debugging, resume it later
			//CUDA_CHECK_ERRORS(cudaMemset(gpuRuntimeData[netId].I_set, 0, networkConfigs[netId].I_setPitch * networkConfigs[netId].I_setLength));
		}
	}

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			// update all neuron state (i.e., voltage and recovery), including homeostasis
			kernel_neuronStateUpdate<<<NUM_BLOCKS, NUM_THREADS>>>();
			CUDA_GET_LAST_ERROR("Kernel execution failed");
		}
	}

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			// update all group state (i.e., concentration of neuronmodulators)
			// currently support 4 x 128 groups
			kernel_groupStateUpdate<<<4, NUM_THREADS>>>(simTimeMs);
			CUDA_GET_LAST_ERROR("Kernel execution failed");
		}
	}
}

void SNN::assignPoissonFiringRate_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);

			for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
				// given group of neurons belong to the poisson group....
				if (groupConfigs[netId][lGrpId].isSpikeGenerator) {
					int lNId = groupConfigs[netId][lGrpId].lStartN;
					int gGrpId = groupConfigs[netId][lGrpId].gGrpId;
					PoissonRate* rate = groupConfigMDMap[gGrpId].ratePtr;

					// if spikeGenFunc group does not have a Poisson pointer, skip
					if (groupConfigMap[gGrpId].spikeGenFunc || rate == NULL)
						continue;

					assert(gpuRuntimeData[netId].poissonFireRate != NULL);
					if (rate->isOnGPU()) {
						// rates allocated on GPU
						CUDA_CHECK_ERRORS(cudaMemcpy(&gpuRuntimeData[netId].poissonFireRate[lNId - networkConfigs[netId].numNReg], rate->getRatePtrGPU(),
							sizeof(float) * rate->getNumNeurons(), cudaMemcpyDeviceToDevice) );
					} else {
						// rates allocated on CPU
						CUDA_CHECK_ERRORS(cudaMemcpy(&gpuRuntimeData[netId].poissonFireRate[lNId - networkConfigs[netId].numNReg], rate->getRatePtrCPU(),
							sizeof(float) * rate->getNumNeurons(), cudaMemcpyHostToDevice) );
					}
				}
			}
		}
	}
}

// Note: for temporarily use, might be merged into exchangeExternalSpike
void SNN::clearExtFiringTable_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);

			// fetch extFiringTableD2
			//printf("%p\n", gpuRuntimeData[netId].extFiringTableD2); 
			//CUDA_CHECK_ERRORS(cudaMemcpy(managerRuntimeData.extFiringTableD2, gpuRuntimeData[netId].extFiringTableD2, sizeof(int*) * networkConfigs[netId].numGroups, cudaMemcpyDeviceToHost));
			//for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
			//	if (groupConfigs[netId][lGrpId].hasExternalConnect) {
			//		CUDA_CHECK_ERRORS(cudaMemset(managerRuntimeData.extFiringTableD2[lGrpId], 0 , sizeof(int) * groupConfigs[netId][lGrpId].SizeN * NEURON_MAX_FIRING_RATE));
			//	}
			//}
			//// fetch extFiringTableD1
			//CUDA_CHECK_ERRORS(cudaMemcpy(managerRuntimeData.extFiringTableD1, gpuRuntimeData[netId].extFiringTableD1, sizeof(int*) * networkConfigs[netId].numGroups, cudaMemcpyDeviceToHost));
			//for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroups; lGrpId++) {
			//	if (groupConfigs[netId][lGrpId].hasExternalConnect) {
			//		CUDA_CHECK_ERRORS(cudaMemset(managerRuntimeData.extFiringTableD1[lGrpId], 0 , sizeof(int) * groupConfigs[netId][lGrpId].SizeN * NEURON_MAX_FIRING_RATE));
			//	}
			//}

			CUDA_CHECK_ERRORS(cudaMemset(gpuRuntimeData[netId].extFiringTableEndIdxD1, 0, sizeof(int) * networkConfigs[netId].numGroups));
			CUDA_CHECK_ERRORS(cudaMemset(gpuRuntimeData[netId].extFiringTableEndIdxD2, 0, sizeof(int) * networkConfigs[netId].numGroups));
		}
	}
}

void SNN::advSimStep_GPU() {
	doSTPUpdateAndDecayCond_GPU();

	spikeGeneratorUpdate_GPU();

	findFiring_GPU();

	updateTimingTable_GPU();

	routeSpikes_GPU();

	doCurrentUpdate_GPU();

	globalStateUpdate_GPU();

	clearExtFiringTable_GPU();
}

void SNN::routeSpikes_GPU() {
	int firingTableIdxD2, firingTableIdxD1;
	int GtoLOffset;
	// ToDo: route spikes using routing table. currently only exchange spikes between GPU0 and GPU1
	// GPU0 -> GPU1
	if (!groupPartitionLists[0].empty() && !groupPartitionLists[1].empty()) {
		checkAndSetGPUDevice(0);
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableEndIdxD2, gpuRuntimeData[0].extFiringTableEndIdxD2, sizeof(int) * networkConfigs[0].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableEndIdxD1, gpuRuntimeData[0].extFiringTableEndIdxD1, sizeof(int) * networkConfigs[0].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableD2, gpuRuntimeData[0].extFiringTableD2, sizeof(int*) * networkConfigs[0].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableD1, gpuRuntimeData[0].extFiringTableD1, sizeof(int*) * networkConfigs[0].numGroups, cudaMemcpyDeviceToHost));
		//KERNEL_DEBUG("GPU0 D1ex:%d/D2ex:%d", managerRuntimeData.extFiringTableEndIdxD1[0], managerRuntimeData.extFiringTableEndIdxD2[0]);

		checkAndSetGPUDevice(1);
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD2, timeTableD2GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD1, timeTableD1GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
		firingTableIdxD2 = managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1];
		firingTableIdxD1 = managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1];
		//KERNEL_DEBUG("GPU1 D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);

		for (int lGrpId = 0; lGrpId < networkConfigs[0].numGroups; lGrpId++) {
			if (groupConfigs[0][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD2[lGrpId] > 0) {
				CUDA_CHECK_ERRORS( cudaMemcpyPeer(gpuRuntimeData[1].firingTableD2 + firingTableIdxD2, 1,
												  managerRuntimeData.extFiringTableD2[lGrpId], 0,
												  sizeof(int) * managerRuntimeData.extFiringTableEndIdxD2[lGrpId]));

				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[1].begin(); grpIt != groupPartitionLists[1].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[0][lGrpId].gGrpId)
						GtoLOffset = grpIt->GtoLOffset;
				}

				kernel_convertExtSpikesD2<<<NUM_BLOCKS, NUM_THREADS>>>(firingTableIdxD2,
																	   firingTableIdxD2 + managerRuntimeData.extFiringTableEndIdxD2[lGrpId],
																	   GtoLOffset); // [StartIdx, EndIdx)
				firingTableIdxD2 += managerRuntimeData.extFiringTableEndIdxD2[lGrpId];
			}

			if (groupConfigs[0][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD1[lGrpId] > 0) {
				CUDA_CHECK_ERRORS( cudaMemcpyPeer(gpuRuntimeData[1].firingTableD1 + firingTableIdxD1, 1,
												  managerRuntimeData.extFiringTableD1[lGrpId], 0,
												  sizeof(int) * managerRuntimeData.extFiringTableEndIdxD1[lGrpId]));

				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[1].begin(); grpIt != groupPartitionLists[1].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[0][lGrpId].gGrpId)
						GtoLOffset = grpIt->GtoLOffset;
				}

				kernel_convertExtSpikesD1<<<NUM_BLOCKS, NUM_THREADS>>>(firingTableIdxD1,
																	   firingTableIdxD1 + managerRuntimeData.extFiringTableEndIdxD1[lGrpId],
																	   GtoLOffset); // [StartIdx, EndIdx)
				firingTableIdxD1 += managerRuntimeData.extFiringTableEndIdxD1[lGrpId];

			}
			//KERNEL_DEBUG("GPU1 New D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);
		}
		managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD2;
		managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD1;
		CUDA_CHECK_ERRORS( cudaMemcpyToSymbol(timeTableD2GPU, managerRuntimeData.timeTableD2, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_ERRORS( cudaMemcpyToSymbol(timeTableD1GPU, managerRuntimeData.timeTableD1, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyHostToDevice));
	}

	// GPU1 -> GPU0
	if (!groupPartitionLists[1].empty() && !groupPartitionLists[0].empty()) {
		checkAndSetGPUDevice(1);
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableEndIdxD2, gpuRuntimeData[1].extFiringTableEndIdxD2, sizeof(int) * networkConfigs[1].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableEndIdxD1, gpuRuntimeData[1].extFiringTableEndIdxD1, sizeof(int) * networkConfigs[1].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableD2, gpuRuntimeData[1].extFiringTableD2, sizeof(int*) * networkConfigs[1].numGroups, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.extFiringTableD1, gpuRuntimeData[1].extFiringTableD1, sizeof(int*) * networkConfigs[1].numGroups, cudaMemcpyDeviceToHost));
		//KERNEL_DEBUG("GPU1 D1ex:%d/D2ex:%d", managerRuntimeData.extFiringTableEndIdxD1[0], managerRuntimeData.extFiringTableEndIdxD2[0]);

		checkAndSetGPUDevice(0);
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD2, timeTableD2GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD1, timeTableD1GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
		firingTableIdxD2 = managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1];
		firingTableIdxD1 = managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1];
		//KERNEL_DEBUG("GPU0 D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);

		for (int lGrpId = 0; lGrpId < networkConfigs[1].numGroups; lGrpId++) {
			if (groupConfigs[1][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD2[lGrpId] > 0) {
				CUDA_CHECK_ERRORS( cudaMemcpyPeer(gpuRuntimeData[0].firingTableD2 + firingTableIdxD2, 0,
												  managerRuntimeData.extFiringTableD2[lGrpId], 1,
												  sizeof(int) * managerRuntimeData.extFiringTableEndIdxD2[lGrpId]));

				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[0].begin(); grpIt != groupPartitionLists[0].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[1][lGrpId].gGrpId)
						GtoLOffset = grpIt->GtoLOffset;
				}
			
				kernel_convertExtSpikesD2<<<NUM_BLOCKS, NUM_THREADS>>>(firingTableIdxD2,
																	   firingTableIdxD2 + managerRuntimeData.extFiringTableEndIdxD2[lGrpId],
																	   GtoLOffset); // [StartIdx, EndIdx)
				firingTableIdxD2 += managerRuntimeData.extFiringTableEndIdxD2[lGrpId];
			}

			if (groupConfigs[1][lGrpId].hasExternalConnect && managerRuntimeData.extFiringTableEndIdxD1[lGrpId] > 0) {
				CUDA_CHECK_ERRORS( cudaMemcpyPeer(gpuRuntimeData[0].firingTableD1 + firingTableIdxD1, 0,
												  managerRuntimeData.extFiringTableD1[lGrpId], 1,
												  sizeof(int) * managerRuntimeData.extFiringTableEndIdxD1[lGrpId]));

				for (std::list<GroupConfigMD>::iterator grpIt = groupPartitionLists[0].begin(); grpIt != groupPartitionLists[0].end(); grpIt++) {
					if (grpIt->gGrpId == groupConfigs[1][lGrpId].gGrpId)
						GtoLOffset = grpIt->GtoLOffset;
				}
			
				kernel_convertExtSpikesD1<<<NUM_BLOCKS, NUM_THREADS>>>(firingTableIdxD1,
																	   firingTableIdxD1 + managerRuntimeData.extFiringTableEndIdxD1[lGrpId],
																	   GtoLOffset); // [StartIdx, EndIdx)
				firingTableIdxD1 += managerRuntimeData.extFiringTableEndIdxD1[lGrpId];
			}
			//KERNEL_DEBUG("GPU0 New D1:%d/D2:%d", firingTableIdxD1, firingTableIdxD2);
		}
		managerRuntimeData.timeTableD2[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD2;
		managerRuntimeData.timeTableD1[simTimeMs + glbNetworkConfig.maxDelay + 1] = firingTableIdxD1;
		CUDA_CHECK_ERRORS( cudaMemcpyToSymbol(timeTableD2GPU, managerRuntimeData.timeTableD2, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyHostToDevice));
		CUDA_CHECK_ERRORS( cudaMemcpyToSymbol(timeTableD1GPU, managerRuntimeData.timeTableD1, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyHostToDevice));
	}
	
}

/*!
 * \brief This function is called every second by SNN::runNetwork(). It updates the firingTableD1(D2)GPU and
 * timeTableD1(D2)GPU by removing older firing information.
 */
void SNN::shiftSpikeTables_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			kernel_shiftFiringTable<<<NUM_BLOCKS, NUM_THREADS>>>();
		}
	}

	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			kernel_shiftTimeTable<<<NUM_BLOCKS, NUM_THREADS>>>();
		}
	}
}

/*
 * \brief Update syanptic weights every 10ms, 100ms, or 1000ms
 *
 *
 */
void SNN::updateWeights_GPU() {
	for (int netId = 0; netId < MAX_NET_PER_SNN; netId++) {
		if (!groupPartitionLists[netId].empty()) {
			checkAndSetGPUDevice(netId);
			assert(sim_in_testing==false);
			assert(sim_with_fixedwts==false);

			kernel_updateWeights<<<NUM_BLOCKS, NUM_THREADS>>>();
		}
	}
}

//__global__ void gpu_resetFiringInformation() {
//	if(threadIdx.x==0 && blockIdx.x==0) {
//		for(int i = 0; i < ROUNDED_TIMING_COUNT; i++) {
//			timeTableD2GPU[i]   = 0;
//			timeTableD1GPU[i]   = 0;
//		}
//		spikeCountD2SecGPU=0;
//		spikeCountD1SecGPU=0;
//		secD2fireCntTest=0;
//		secD1fireCntTest=0;
//		spikeCountD2GPU=0;
//		spikeCountD1GPU=0;
//
//    //spikeCountAll1Sec=0;//assigned in fetchSpikeTables()
//	}
//
//}
//
//void SNN::resetFiringInformation_GPU() {
//	checkAndSetGPUDevice();
//
//	gpu_resetFiringInformation<<<NUM_BLOCKS,NUM_THREADS>>>();
//}

/*!
 * \brief this function allocates device (GPU) memory sapce and copies external current to it
 *
 * This function:

 * (allocate and) copy extCurrent
 *
 * This funcion is called by copyNeuronState() and setExternalCurrent. Only host-to-divice copy is required
 *
 * \param[in] netId the id of a local network, which is the same as the device (GPU) id
 * \param[in] lGrpId the local group id in a local network, which specifiy the group(s) to be copied
 * \param[in] dest pointer to runtime data desitnation
 * \param[in] allocateMem a flag indicates whether allocating memory space before copying
 *
 * \sa allocateSNN_GPU fetchSTPState
 * \since v3.0
 */
void SNN::copyExternalCurrent(int netId, int lGrpId, RuntimeData* dest, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice(netId);
	checkDestSrcPtrs(dest, &managerRuntimeData, cudaMemcpyHostToDevice, allocateMem, lGrpId, 0);// check that the destination pointer is properly allocated..
	assert(kind == cudaMemcpyHostToDevice);
	
	int posN, lengthN;

	if(lGrpId == ALL) {
		posN  = 0;
		lengthN  = networkConfigs[netId].numNReg;
	} else {
		assert(lGrpId >= 0);
		posN = groupConfigs[netId][lGrpId].lStartN;
		lengthN = groupConfigs[netId][lGrpId].numN;
	}
	assert(lengthN >= 0 && lengthN <= networkConfigs[netId].numNReg); // assert NOT poisson neurons

	//KERNEL_DEBUG("copyExternalCurrent: lGrpId=%d, ptrPos=%d, length=%d, allocate=%s", lGrpId, posN, lengthN, allocateMem?"y":"n");

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extCurrent, sizeof(float) * lengthN));
	CUDA_CHECK_ERRORS(cudaMemcpy(&(dest->extCurrent[posN]), &(managerRuntimeData.extCurrent[posN]), sizeof(float) * lengthN, cudaMemcpyHostToDevice));
}

/*!
 * \brief This function fetch the spike count in all local networks and sum the up
 */
void SNN::copyNetworkSpikeCount(int netId, cudaMemcpyKind kind,
								unsigned int* spikeCountD1, unsigned int* spikeCountD2,
								unsigned int* spikeCountExtD1, unsigned int* spikeCountExtD2) {

	checkAndSetGPUDevice(netId);
	assert(kind == cudaMemcpyDeviceToHost);

	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(spikeCountExtD2, spikeCountExtRxD2GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(spikeCountExtD1, spikeCountExtRxD1GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(spikeCountD2, spikeCountD2GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(spikeCountD1, spikeCountD1GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
}

/*!
 * \brief This function fetch spikeTables in the local network specified by netId
 *
 * \param[in] netId the id of local network of which timeTableD1(D2) and firingTableD1(D2) are copied to manager runtime data
 */
void SNN::copySpikeTables(int netId, cudaMemcpyKind kind) {
	unsigned int gpuSpikeCountD1Sec, gpuSpikeCountD2Sec, gpuSpikeCountLastSecLeftD2;

	checkAndSetGPUDevice(netId);
	assert(kind == cudaMemcpyDeviceToHost);

	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(&gpuSpikeCountLastSecLeftD2, spikeCountLastSecLeftD2GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(&gpuSpikeCountD2Sec, spikeCountD2SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS(cudaMemcpyFromSymbol(&gpuSpikeCountD1Sec, spikeCountD1SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.firingTableD2, gpuRuntimeData[netId].firingTableD2, sizeof(int)*(gpuSpikeCountD2Sec + gpuSpikeCountLastSecLeftD2), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpy(managerRuntimeData.firingTableD1, gpuRuntimeData[netId].firingTableD1, sizeof(int)*gpuSpikeCountD1Sec, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD2, timeTableD2GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(managerRuntimeData.timeTableD1, timeTableD1GPU, sizeof(int)*(1000+glbNetworkConfig.maxDelay+1), 0, cudaMemcpyDeviceToHost));
}

void SNN::configGPUDevice() {
	int devCount, devMax;
	cudaDeviceProp deviceProp;

	CUDA_CHECK_ERRORS(cudaGetDeviceCount(&devCount));
	KERNEL_INFO("CUDA devices Configuration:");
	KERNEL_INFO("  - Number of CUDA devices          = %9d", devCount);

	devMax = CUDA_GET_MAXGFLOP_DEVICE_ID();
	KERNEL_INFO("  - CUDA device ID with max GFLOPs  = %9d", devMax);
	
	// ithGPU_ gives an index number on which device to run the simulation
	if (numGPUs_ <= 0) {
		KERNEL_WARN("At lease 1 CUDA device is required to run GPU simulation");
		if (devCount == 0) {
			KERNEL_ERROR("No available CUDA device");
			exitSimulation(1);
		} else {
			KERNEL_WARN("Adjust numGPUs to the number of available CUDA devices [%d]", devCount);
			numGPUs_ = devCount;
		}
	}
	
	// adjust the number of GPUs if the usable cuda devices are fewer than specified numGPUs
	if (devCount < numGPUs_) {
		KERNEL_WARN("The available CUDA devices are fewer than specified numGPUs [%d]", numGPUs_);
		KERNEL_WARN("Adjust numGPUs to the number of available CUDA devices [%d]", devCount);
		numGPUs_ = devCount;
	}

	for (int ithGPU = 0; ithGPU < devCount; ithGPU++) {
		CUDA_CHECK_ERRORS(cudaGetDeviceProperties(&deviceProp, ithGPU));
		KERNEL_INFO("  + Use CUDA device[%1d]              = %9s", ithGPU, deviceProp.name);
		KERNEL_INFO("  + CUDA Compute Capability (CC)    =      %2d.%d", deviceProp.major, deviceProp.minor);
	}
	
	if (deviceProp.major < 2) {
		// Unmark this when CC 1.3 is deprecated
		//KERNEL_ERROR("CARLsim does not support CUDA devices older than CC 2.0");
		//exitSimulation(1);
		KERNEL_WARN("CUDA device with CC 1.3 will be deprecated in a future release");
	}

	for (int ithGPU = 0; ithGPU < devCount; ithGPU++) {
		CUDA_CHECK_ERRORS(cudaSetDevice(ithGPU));
		CUDA_DEVICE_RESET();
	}

	if (devCount >= 2) { // try to setup P2P access if more than 2 GPUs are presented
		// FIXME: generalize the initialization for mulit-GPUs up to 4 or 8
		// enable P2P access
		int canAccessPeer_0_1, canAccessPeer_1_0;
		cudaDeviceCanAccessPeer(&canAccessPeer_0_1, 0, 1);
		cudaDeviceCanAccessPeer(&canAccessPeer_1_0, 1, 0);
		// enable peer access between GPU0 and GPU1
		if (canAccessPeer_0_1 & canAccessPeer_1_0) {
			cudaSetDevice(0);
			cudaDeviceEnablePeerAccess(1, 0);
			cudaSetDevice(1);
			cudaDeviceEnablePeerAccess(0, 0);
			KERNEL_INFO("* Peer Access is enabled");
		} else {
			KERNEL_INFO("* Peer Access is not enabled");
		}
	}
}

void SNN::checkAndSetGPUDevice(int netId) {
	int currentDevice;
	cudaGetDevice(&currentDevice);

	assert(netId >= 0 && netId < numGPUs_);

	if (currentDevice != netId) {
		//KERNEL_DEBUG("Change GPU context from GPU %d to GPU %d", currentDevice, netId);
		CUDA_CHECK_ERRORS(cudaSetDevice(netId));
	}
}

// deprecated
//void SNN::copyWeightsGPU(int nid, int src_grp) {
//	checkAndSetGPUDevice("copyWeightsGPU");
//
//	assert(nid < numNReg);
//	unsigned int    cumId   =  managerRuntimeData.cumulativePre[nid];
//	float* synWts  = &(managerRuntimeData.wt[cumId]);
//	//TODO: NEEDED TO COMMENT THIS FOR CARLSIM 2.1-2.2 FILEMERGE -- KDC
//	// assert(cumId >= (nid-numNPois));
//	//assert(cumId < numPreSynapses*networkConfigs[0].numN);
//
//	CUDA_CHECK_ERRORS( cudaMemcpy( synWts, &gpuRuntimeData[0].wt[cumId], sizeof(float)*managerRuntimeData.Npre[nid], cudaMemcpyDeviceToHost));
//}

// Allocates required memory and then initialize the GPU
void SNN::allocateSNN_GPU(int netId) {
	checkAndSetGPUDevice(netId);

	// setup memory type of GPU runtime data
	gpuRuntimeData[netId].memType = GPU_MODE;

	// display some memory management info
	size_t avail, total, previous;
	float toMB = std::pow(1024.0f, 2);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("GPU Memory Management: (Total %2.3f MB)",(float)(total/toMB));
	KERNEL_INFO("Data\t\t\tSize\t\tTotal Used\tTotal Available");
	KERNEL_INFO("Init:\t\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(total)/toMB,(float)((total-avail)/toMB),
		(float)(avail/toMB));
	previous=avail;

	// allocate random number generator on GPU(s)
	if(gpuRuntimeData[netId].gpuRandGen == NULL) {
		curandCreateGenerator(&gpuRuntimeData[netId].gpuRandGen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gpuRuntimeData[netId].gpuRandGen, randSeed_ + netId);
	}

	// allocate SNN::gpuRuntimeData[0].randNum for random number generators
	CUDA_CHECK_ERRORS(cudaMalloc((void **)&gpuRuntimeData[netId].randNum, networkConfigs[netId].numNPois * sizeof(float)));

	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Random Gen:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB, (float)((total-avail)/toMB),(float)(avail/toMB));
	previous=avail;

	// initialize gpuRuntimeData[0].neuronAllocation, __device__ loadBufferCount, loadBufferSize
	allocateStaticLoad(netId, NUM_THREADS);

	allocateGroupId(netId);

	// this table is useful for quick evaluation of the position of fired neuron
	// given a sequence of bits denoting the firing..
	// initialize __device__ quickSynIdTableGPU[256]
	initQuickSynIdTable(netId);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Static Load:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB, (float)((total-avail)/toMB),(float)(avail/toMB));
	previous=avail;

	// initialize (copy from SNN) gpuRuntimeData[0].Npre, gpuRuntimeData[0].Npre_plastic, gpuRuntimeData[0].Npre_plasticInv, gpuRuntimeData[0].cumulativePre
	// initialize (copy from SNN) gpuRuntimeData[0].cumulativePost, gpuRuntimeData[0].Npost, gpuRuntimeData[0].postDelayInfo
	// initialize (copy from SNN) gpuRuntimeData[0].postSynapticIds, gpuRuntimeData[0].preSynapticIds
	copyPreConnectionInfo(netId, ALL, &gpuRuntimeData[netId], &managerRuntimeData, cudaMemcpyHostToDevice, true);
	copyPostConnectionInfo(netId, ALL, &gpuRuntimeData[netId], &managerRuntimeData, cudaMemcpyHostToDevice, true);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Conn Info:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB,(float)((total-avail)/toMB), (float)(avail/toMB));
	previous=avail;
	
	// initialize (copy from SNN) gpuRuntimeData[0].wt, gpuRuntimeData[0].wtChange, gpuRuntimeData[0].maxSynWt
	copySynapseState(netId, &gpuRuntimeData[netId], &managerRuntimeData, cudaMemcpyHostToDevice, true);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Syn State:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB,(float)((total-avail)/toMB), (float)(avail/toMB));
	previous=avail;
	
	// copy the neuron state information to the GPU..
	// initialize (copy from managerRuntimeData) gpuRuntimeData[0].recovery, gpuRuntimeData[0].voltage, gpuRuntimeData[0].current
	// initialize (copy from managerRuntimeData) gpuRuntimeData[0].gGABAa, gpuRuntimeData[0].gGABAb, gpuRuntimeData[0].gAMPA, gpuRuntimeData[0].gNMDA
	// initialize (copy from SNN) gpuRuntimeData[0].Izh_a, gpuRuntimeData[0].Izh_b, gpuRuntimeData[0].Izh_c, gpuRuntimeData[0].Izh_d
	// initialize (copy form SNN) gpuRuntimeData[0].baseFiring, gpuRuntimeData[0].baseFiringInv
	copyNeuronState(netId, ALL, &gpuRuntimeData[netId], cudaMemcpyHostToDevice, true);

	// copy STP state, considered as neuron state
	if (sim_with_stp) {
		// initialize (copy from SNN) stpu, stpx
		copySTPState(netId, ALL, &gpuRuntimeData[netId], &managerRuntimeData, cudaMemcpyHostToDevice, true);
	}
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Neuron State:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB,(float)((total-avail)/toMB), (float)(avail/toMB));
	previous=avail;
		
	// initialize (copy from SNN) gpuRuntimeData[0].grpDA(5HT,ACh,NE)
	// initialize (copy from SNN) gpuRuntimeData[0].grpDA(5HT,ACh,NE)Buffer[]
	copyGroupState(netId, ALL, &gpuRuntimeData[netId], &managerRuntimeData, cudaMemcpyHostToDevice, true);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Group State:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB",(float)(previous-avail)/toMB,(float)((total-avail)/toMB), (float)(avail/toMB));
	previous=avail;

	// initialize (cudaMemset) gpuRuntimeData[0].I_set, gpuRuntimeData[0].poissonFireRate
	// initialize (copy from SNN) gpuRuntimeData[0].firingTableD1, gpuRuntimeData[0].firingTableD2
	// initialize (cudaMalloc) gpuRuntimeData[0].spikeGenBits
	// initialize (copy from managerRuntimeData) gpuRuntimeData[0].nSpikeCnt,
	// initialize (copy from SNN) gpuRuntimeData[0].synSpikeTime, gpuRuntimeData[0].lastSpikeTime
	copyAuxiliaryData(netId, ALL, &gpuRuntimeData[netId], cudaMemcpyHostToDevice, true);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Auxiliary Data:\t\t%2.3f MB\t%2.3f MB\t%2.3f MB\n\n",(float)(previous-avail)/toMB,(float)((total-avail)/toMB), (float)(avail/toMB));
	previous=avail;

	// copy relevant pointers and network information to GPU
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(runtimeDataGPU, &gpuRuntimeData[netId], sizeof(RuntimeData), 0, cudaMemcpyHostToDevice));

	// copy data to from SNN:: to NetworkConfigRT SNN::networkConfigs[0]
	copyNetworkConfig(netId); // FIXME: we can change the group properties such as STDP as the network is running.  So, we need a way to updating the GPU when changes are made.

	// TODO: move mulSynFast, mulSynSlow to ConnectConfig structure
	// copy connection configs
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(d_mulSynFast, mulSynFast, sizeof(float) * networkConfigs[netId].numConnections, 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(d_mulSynSlow, mulSynSlow, sizeof(float) * networkConfigs[netId].numConnections, 0, cudaMemcpyHostToDevice));

	copyGroupConfigs(netId);

	KERNEL_DEBUG("Transfering group settings to GPU:");
	for (int lGrpId = 0; lGrpId < networkConfigs[netId].numGroupsAssigned; lGrpId++) {
		KERNEL_DEBUG("Settings for Group %s:", groupConfigMap[groupConfigs[netId][lGrpId].gGrpId].grpName.c_str());
		
		KERNEL_DEBUG("\tType: %d",(int)groupConfigs[netId][lGrpId].Type);
		KERNEL_DEBUG("\tNumN: %d",groupConfigs[netId][lGrpId].numN);
		KERNEL_DEBUG("\tM: %d",groupConfigs[netId][lGrpId].numPostSynapses);
		KERNEL_DEBUG("\tPreM: %d",groupConfigs[netId][lGrpId].numPreSynapses);
		KERNEL_DEBUG("\tspikeGenerator: %d",(int)groupConfigs[netId][lGrpId].isSpikeGenerator);
		KERNEL_DEBUG("\tFixedInputWts: %d",(int)groupConfigs[netId][lGrpId].FixedInputWts);
		KERNEL_DEBUG("\tMaxDelay: %d",(int)groupConfigs[netId][lGrpId].MaxDelay);
		KERNEL_DEBUG("\tWithSTDP: %d",(int)groupConfigs[netId][lGrpId].WithSTDP);
		if (groupConfigs[netId][lGrpId].WithSTDP) {
			KERNEL_DEBUG("\t\tE-STDP type: %s",stdpType_string[groupConfigs[netId][lGrpId].WithESTDPtype]);
			KERNEL_DEBUG("\t\tTAU_PLUS_INV_EXC: %f",groupConfigs[netId][lGrpId].TAU_PLUS_INV_EXC);
			KERNEL_DEBUG("\t\tTAU_MINUS_INV_EXC: %f",groupConfigs[netId][lGrpId].TAU_MINUS_INV_EXC);
			KERNEL_DEBUG("\t\tALPHA_PLUS_EXC: %f",groupConfigs[netId][lGrpId].ALPHA_PLUS_EXC);
			KERNEL_DEBUG("\t\tALPHA_MINUS_EXC: %f",groupConfigs[netId][lGrpId].ALPHA_MINUS_EXC);
			KERNEL_DEBUG("\t\tI-STDP type: %s",stdpType_string[groupConfigs[netId][lGrpId].WithISTDPtype]);
			KERNEL_DEBUG("\t\tTAU_PLUS_INV_INB: %f",groupConfigs[netId][lGrpId].TAU_PLUS_INV_INB);
			KERNEL_DEBUG("\t\tTAU_MINUS_INV_INB: %f",groupConfigs[netId][lGrpId].TAU_MINUS_INV_INB);
			KERNEL_DEBUG("\t\tALPHA_PLUS_INB: %f",groupConfigs[netId][lGrpId].ALPHA_PLUS_INB);
			KERNEL_DEBUG("\t\tALPHA_MINUS_INB: %f",groupConfigs[netId][lGrpId].ALPHA_MINUS_INB);
			KERNEL_DEBUG("\t\tLAMBDA: %f",groupConfigs[netId][lGrpId].LAMBDA);
			KERNEL_DEBUG("\t\tDELTA: %f",groupConfigs[netId][lGrpId].DELTA);
			KERNEL_DEBUG("\t\tBETA_LTP: %f",groupConfigs[netId][lGrpId].BETA_LTP);
			KERNEL_DEBUG("\t\tBETA_LTD: %f",groupConfigs[netId][lGrpId].BETA_LTD);
		}
		KERNEL_DEBUG("\tWithSTP: %d",(int)groupConfigs[netId][lGrpId].WithSTP);
		if (groupConfigs[netId][lGrpId].WithSTP) {
			KERNEL_DEBUG("\t\tSTP_U: %f",groupConfigs[netId][lGrpId].STP_U);
//				KERNEL_DEBUG("\t\tSTP_tD: %f",groupConfigs[netId][lGrpId].STP_tD);
//				KERNEL_DEBUG("\t\tSTP_tF: %f",groupConfigs[netId][lGrpId].STP_tF);
		}
		KERNEL_DEBUG("\tspikeGen: %s", groupConfigs[netId][lGrpId].isSpikeGenFunc? "is Set" : "is not set ");
	}

	// allocation of gpu runtime data is done
	gpuRuntimeData[netId].allocated = true;

	// map the timing table to texture.. saves a lot of headache in using shared memory
	void* devPtr;
	size_t offset;
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD2GPU));
	CUDA_CHECK_ERRORS(cudaBindTexture(&offset, timeTableD2GPU_tex, devPtr, sizeof(int) * TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD2GPU_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));
		
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD1GPU));
	CUDA_CHECK_ERRORS(cudaBindTexture(&offset, timeTableD1GPU_tex, devPtr, sizeof(int) * TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD1GPU_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));

	initGPU(netId);
}
