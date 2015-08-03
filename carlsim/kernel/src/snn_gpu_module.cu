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
#include <error_code.h>
#include <cuda_runtime.h>

#define ROUNDED_TIMING_COUNT  (((1000+MAX_SYN_DELAY+1)+127) & ~(127))  // (1000+maxDelay_) rounded to multiple 128

#define  FIRE_CHUNK_CNT    (512)

#define LOG_WARP_SIZE		(5)
#define WARP_SIZE			(1 << LOG_WARP_SIZE)

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

__device__ unsigned int  timeTableD2GPU[ROUNDED_TIMING_COUNT];
__device__ unsigned int  timeTableD1GPU[ROUNDED_TIMING_COUNT];

__device__ unsigned int	spikeCountD2SecGPU;
__device__ unsigned int	spikeCountD1SecGPU;
__device__ unsigned int spikeCountD2GPU;
__device__ unsigned int spikeCountD1GPU;

// I believe the following are all just test variables
__device__ unsigned int	secD2fireCntTest;
__device__ unsigned int	secD1fireCntTest;

__device__ __constant__ RuntimeData     runtimeDataGPU;
__device__ __constant__ NetworkConfigRT	networkConfigGPU;
__device__ __constant__ GroupConfigRT   groupConfigGPU[MAX_GRP_PER_SNN];

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
void initQuickSynIdTable()
{
	void* devPtr;
	   
	for(int i=1; i < 256; i++) {
		int cnt=0;
		while(i) {
			if(((i>>cnt)&1)==1) break;
      		cnt++;
      		assert(cnt<=7);
    	}
    	quickSynIdTable[i]=cnt;		 
	}
	   
	cudaGetSymbolAddress(&devPtr, quickSynIdTableGPU);
	CUDA_CHECK_ERRORS( cudaMemcpy( devPtr, quickSynIdTable, sizeof(quickSynIdTable), cudaMemcpyHostToDevice));
}

__device__ inline bool isPoissonGroup(short int& grpId, int& nid)
{
	return (groupConfigGPU[grpId].Type & POISSON_NEURON);
}

__device__ inline void setFiringBitSynapses(int& nid, int& syn_id)
{
	uint32_t* tmp_I_set_p = ((uint32_t*)((char*) runtimeDataGPU.I_set + ((syn_id>>5)*networkConfigGPU.I_setPitch)) + nid);
	int atomicVal = atomicOr(tmp_I_set_p, 1 <<(syn_id%32));
}

__device__ inline uint32_t* getFiringBitGroupPtr(int& nid, int& synGrpId)
{
	uint32_t* tmp_ptr = (((uint32_t*)((char*) runtimeDataGPU.I_set + synGrpId*networkConfigGPU.I_setPitch)) + nid);
	return tmp_ptr;
}

__device__ inline uint32_t getSTPBufPos(int nid, uint32_t t)
{
//  return (((t%STP_BUF_SIZE)*networkConfigGPU.STP_Pitch) + nid);
  return ( (t%(networkConfigGPU.maxDelay+1))*networkConfigGPU.STP_Pitch + nid);
}

__device__ inline int2 getStaticThreadLoad(int& bufPos)
{
	return (runtimeDataGPU.neuronAllocation[bufPos]);
}

__device__ inline bool getPoissonSpike_GPU (int& nid)
{
	// Random number value is less than the poisson firing probability
	// if poisson firing probability is say 1.0 then the random poisson ptr
	// will always be less than 1.0 and hence it will continiously fire
	return runtimeDataGPU.poissonRandPtr[nid-networkConfigGPU.numNReg]*(1000.0f/RNG_rand48::MAX_RANGE) < runtimeDataGPU.poissonFireRate[nid-networkConfigGPU.numNReg];
}

///////////////////////////////////////////////////////////////////
// Device local function:      update_GPU_TimingTable			///
// KERNEL: After every iteration we update the timing table		///
// so that we have the new values of the fired neurons for the	///
// current time t.												///
///////////////////////////////////////////////////////////////////
__global__ void kernel_timingTableUpdate(int t)
{
   if ( threadIdx.x == 0 && blockIdx.x == 0) {
		timeTableD2GPU[t+networkConfigGPU.maxDelay+1]  = spikeCountD2SecGPU;
		timeTableD1GPU[t+networkConfigGPU.maxDelay+1]  = spikeCountD1SecGPU;
   }
   __syncthreads();									     
}

/////////////////////////////////////////////////////////////////////////////////
// Device Kernel Function:  Intialization of the GPU side of the simulator    ///
// KERNEL: This kernel is called after initialization of various parameters   ///
// so that we can reset all required parameters.                              ///
/////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_init ()
{
	if(threadIdx.x==0 && blockIdx.x==0) {
		for(int i=0; i < ROUNDED_TIMING_COUNT; i++) {
			timeTableD2GPU[i]   = 0;
			timeTableD1GPU[i]   = 0;
		}
	}

	const int totBuffers=loadBufferCount;
	__syncthreads();
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 	 threadLoad  = getStaticThreadLoad(bufPos);
		int  	 nid        = STATIC_LOAD_START(threadLoad);
		int  	 lastId      = STATIC_LOAD_SIZE(threadLoad);
//		short int grpId   	 = STATIC_LOAD_GROUP(threadLoad);

		while ((threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {
//				int totCnt = runtimeDataGPU.Npre[nid];			// total synaptic count
//				int nCum   = runtimeDataGPU.cumulativePre[nid];	// total pre-synaptic count
			nid=nid+1; // move to the next neuron in the group..
		}
	}
}

// Allocation of the group and its id..
void SNN::allocateGroupId() {
	checkAndSetGPUDevice();

	assert (gpuRuntimeData.groupIdInfo == NULL);
	int3* tempNeuronAllocation = (int3*)malloc(sizeof(int3) * networkConfig.numGroups);
	for (int g = 0; g < networkConfig.numGroups; g++) {
		int3  threadLoad;
		threadLoad.x = groupConfig[g].StartN;
		threadLoad.y = groupConfig[g].EndN;
		threadLoad.z = g;
		tempNeuronAllocation[g] = threadLoad;
	}

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&gpuRuntimeData.groupIdInfo, sizeof(int3) * networkConfig.numGroups));
	CUDA_CHECK_ERRORS(cudaMemcpy(gpuRuntimeData.groupIdInfo, tempNeuronAllocation, sizeof(int3) * networkConfig.numGroups, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaBindTexture(NULL, groupIdInfo_tex, gpuRuntimeData.groupIdInfo, sizeof(int3) * networkConfig.numGroups));

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
int SNN::allocateStaticLoad(int bufSize) {
	checkAndSetGPUDevice();

	// only one thread does the static load table
	int bufferCnt = 0;
	for (int g=0; g<networkConfig.numGroups; g++) {
		int grpBufCnt = (int) ceil(1.0f * groupConfig[g].SizeN / bufSize);
		assert(grpBufCnt>=0);
		bufferCnt += grpBufCnt;
		KERNEL_DEBUG("Grp Size = %d, Total Buffer Cnt = %d, Buffer Cnt = %f", groupConfig[g].SizeN, bufferCnt, grpBufCnt);
	}
	assert(bufferCnt>0);

	int2*  tempNeuronAllocation = (int2*)malloc(sizeof(int2) * bufferCnt);
	KERNEL_DEBUG("STATIC THREAD ALLOCATION");
	KERNEL_DEBUG("------------------------");
	KERNEL_DEBUG("Buffer Size = %d, Buffer Count = %d", bufSize, bufferCnt);

	bufferCnt = 0;
	for (int g = 0; g < networkConfig.numGroups; g++) {
		for (int n = groupConfig[g].StartN; n <= groupConfig[g].EndN; n += bufSize) {
			int2  threadLoad;
			// starting neuron id is saved...
			threadLoad.x = n;
			if ((n + bufSize - 1) <= groupConfig[g].EndN)
				// grpID + full size
				threadLoad.y = (g + (bufSize << 16)); // can't support group id > 2^16
			else
				// grpID + left-over size
				threadLoad.y = (g + ((groupConfig[g].EndN - n + 1) << 16)); // can't support group id > 2^16

			// fill the static load distribution here...
			int testg = STATIC_LOAD_GROUP(threadLoad);
			tempNeuronAllocation[bufferCnt] = threadLoad;
			KERNEL_DEBUG("%d. Start=%d, size=%d grpId=%d:%s (SpikeMonId=%d) (GroupMonId=%d)",
					bufferCnt, STATIC_LOAD_START(threadLoad),
					STATIC_LOAD_SIZE(threadLoad),
					STATIC_LOAD_GROUP(threadLoad),
					groupInfo[testg].Name.c_str(),
					groupConfig[testg].SpikeMonitorId,
					groupConfig[testg].GroupMonitorId);
			bufferCnt++;
		}
	}

	assert(gpuRuntimeData.allocated==false);
	// Finally writeback the total bufferCnt
	// Note down the buffer size for reference
	KERNEL_DEBUG("GPU loadBufferSize = %d, GPU loadBufferCount = %d", bufSize, bufferCnt);
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(loadBufferCount, &bufferCnt, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(loadBufferSize, &bufSize, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**) &gpuRuntimeData.neuronAllocation, sizeof(int2) * bufferCnt));
	CUDA_CHECK_ERRORS(cudaMemcpy(gpuRuntimeData.neuronAllocation, tempNeuronAllocation, sizeof(int2) * bufferCnt, cudaMemcpyHostToDevice));
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
__device__ void firingUpdateSTP (int& nid, int& simTime, short int&  grpId) {
	// we need to retrieve the STP values from the right buffer position (right before vs. right after the spike)
	int ind_plus  = getSTPBufPos(nid, simTime);
	int ind_minus = getSTPBufPos(nid, (simTime-1));

	// at this point, stpu[ind_plus] has already been assigned, and the decay applied
	// so add the spike-dependent part to that
	// du/dt = -u/tau_F + U * (1-u^-) * \delta(t-t_{spk})
	runtimeDataGPU.stpu[ind_plus] += groupConfigGPU[grpId].STP_U*(1.0f-runtimeDataGPU.stpu[ind_minus]);

	// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
	runtimeDataGPU.stpx[ind_plus] -= runtimeDataGPU.stpu[ind_plus]*runtimeDataGPU.stpx[ind_minus];
}

__device__ void resetFiredNeuron(int& nid, short int & grpId, int& simTime)
{
	// \FIXME \TODO: convert this to use coalesced access by grouping into a
	// single 16 byte access. This might improve bandwidth performance
	// This is fully uncoalsced access...need to convert to coalsced access..
	runtimeDataGPU.voltage[nid] = runtimeDataGPU.Izh_c[nid];
	runtimeDataGPU.recovery[nid] += runtimeDataGPU.Izh_d[nid];
	if (groupConfigGPU[grpId].WithSTDP)
		runtimeDataGPU.lastSpikeTime[nid] = simTime;
	
	if (networkConfigGPU.sim_with_homeostasis) {
		// with homeostasis flag can be used here.
		runtimeDataGPU.avgFiring[nid] += 1000/(groupConfigGPU[grpId].avgTimeScale*1000);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Device local function:      gpu_newFireUpdate                                           ///
/// Description: 1. Copy neuron id from local table to global firing table.                 ///
///		 2. Reset all neuron properties of neuron id in local table 		    			///
//                                                                                          ///
/// fireTablePtr:  local shared memory firing table with neuron ids of fired neuron         ///
/// fireCntD2:      number of excitatory neurons in local table that has fired               ///
/// fireCntD1:      number of inhibitory neurons in local table that has fired               ///
/// simTime:      current global time step..stored as neuron firing time  entry             ///
////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void updateFiringCounter(volatile unsigned int& fireCnt, volatile unsigned int& fireCntD1, volatile unsigned int& cntD2, volatile unsigned int& cntD1, volatile int&  blkErrCode)
{
	int fireCntD2 = fireCnt-fireCntD1;

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
	cntD2 = atomicAdd(&spikeCountD2SecGPU, fireCntD2);
	cntD1 = atomicAdd(&spikeCountD1SecGPU, fireCntD1);
}

// update the firing table...
__device__ void updateFiringTable(int& nId, short int& grpId, volatile unsigned int& cntD2, volatile unsigned int& cntD1)
{
	int pos;
	if (groupConfigGPU[grpId].MaxDelay == 1) {
		// this group has a delay of only 1
		pos = atomicAdd((int*)&cntD1, 1);
		//runtimeDataGPU.firingTableD1[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.firingTableD1[pos] = nId;
	} else {
		// all other groups is dumped here 
		pos = atomicAdd((int*)&cntD2, 1);
		//runtimeDataGPU.firingTableD2[pos]  = SET_FIRING_TABLE(nid, grpId);
		runtimeDataGPU.firingTableD2[pos] = nId;
	}
}

__device__ int newFireUpdate (	int* 	fireTablePtr,
								short int* fireGrpId,
								volatile unsigned int& 	fireCnt,
								volatile unsigned int& 	fireCntD1,
								int& 	simTime)
{
__shared__ volatile unsigned int cntD2;
__shared__ volatile unsigned int cntD1;
__shared__ volatile int blkErrCode;

	blkErrCode = 0;
	if (0==threadIdx.x) {
		updateFiringCounter(fireCnt, fireCntD1, cntD2, cntD1, blkErrCode);
	}

	__syncthreads();

	// if we overflow the spike buffer space that is available,
	// then we return with an error here...
	if (blkErrCode)
		return blkErrCode;

	for (int i=threadIdx.x; i < fireCnt; i+=(blockDim.x)) {

		// Read the firing id from the local table.....
		int nid  = fireTablePtr[i];

		updateFiringTable(nid, fireGrpId[i], cntD2, cntD1);

		if (groupConfigGPU[fireGrpId[i]].WithSTP)
			firingUpdateSTP(nid, simTime, fireGrpId[i]);

		// keep track of number spikes per neuron
		runtimeDataGPU.nSpikeCnt[nid]++;

		// only neurons would do the remaining settings...
		// pure poisson generators will return without changing anything else..
		if (IS_REGULAR_NEURON(nid, networkConfigGPU.numNReg, networkConfigGPU.numNPois))
			resetFiredNeuron(nid, fireGrpId[i], simTime);
	}

	__syncthreads();

	 return 0;
}

// zero GPU spike counts
__global__ void gpu_resetSpikeCnt(int _startGrp, int _endGrp)
{
	//global calls are done serially while device calls are done per thread
	//because this is a global call we want to make sure it only executes
	//once (i.e. for a particular index).  
	//TODO: I need to test and make sure this still works. TAGS:UPDATE.
	if((blockIdx.x==0)&&(threadIdx.x==0)) {
		//groupConfig seems to be accessible. -- KDC
		for( int grpId=_startGrp; grpId < _endGrp; grpId++) {
			int startN = groupConfigGPU[grpId].StartN;
			int endN   = groupConfigGPU[grpId].EndN+1;
			for (int i=startN; i < endN; i++){
				runtimeDataGPU.nSpikeCnt[i] = 0;
			}
		}
	}
}

// wrapper to call resetSpikeCnt
void SNN::resetSpikeCnt_GPU(int _startGrp, int _endGrp) {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;
	gpu_resetSpikeCnt<<<gridSize,blkSize>>> (_startGrp,_endGrp);
}

__device__ void findGrpId_GPU(int& nid, int& grpId)
{
	for (int g=0; g < networkConfigGPU.numGroups; g++) {
		//uint3 groupIdInfo = {1, 1, 1};
		int startN  = tex1Dfetch (groupIdInfo_tex, g*3);
		int endN    = tex1Dfetch (groupIdInfo_tex, g*3+1);
		// printf("%d:%s s=%d e=%d\n", g, groupInfo[g].Name.c_str(), groupConfig[g].StartN, groupConfig[g].EndN);
		//if ((nid >= groupIdInfo.x) && (nid <= groupIdInfo.y)) {
		// grpId = groupIdInfo.z;
		if ((nid >= startN) && (nid <= endN)) {
			grpId = tex1Dfetch (groupIdInfo_tex, g*3+2);
			return;
		}
	}
	grpId = -1;
	return;
}

///////////////////////////////////////////////////////////////////////////
/// Device local function:      gpu_updateLTP
/// Description:                Computes the STDP update values for each of fired
///                             neurons stored in the local firing table.
///
/// fireTablePtr:  local shared memory firing table with neuron ids of fired neuron
/// fireCnt:      number of fired neurons in local firing table
/// simTime:     current global time step..stored as neuron firing time  entry
///////////////////////////////////////////////////////////////////////////
// synaptic grouping for LTP Calculation
#define		LTP_GROUPING_SZ     16
__device__ void gpu_updateLTP(	int*     		fireTablePtr,
				short int*  		fireGrpId,
				volatile unsigned int&   fireCnt,
				int&      		simTime)
{
	for(int pos=threadIdx.x/LTP_GROUPING_SZ; pos < fireCnt; pos += (blockDim.x/LTP_GROUPING_SZ))  {
		// each neuron has two variable pre and pre_exc
		// pre: number of pre-neuron
		// pre_exc: number of neuron had has plastic connections
		short int grpId = fireGrpId[pos];
		// STDP calculation: the post-synaptic neron fires after the arrival of pre-synaptic neuron's spike
		if (groupConfigGPU[grpId].WithSTDP) { // MDR, FIXME this probably will cause more thread divergence than need be...
			int  nid   = fireTablePtr[pos];
			unsigned int  end_p = runtimeDataGPU.cumulativePre[nid] + runtimeDataGPU.Npre_plastic[nid];
			for(unsigned int p  = runtimeDataGPU.cumulativePre[nid] + threadIdx.x % LTP_GROUPING_SZ;
					p < end_p;
					p+=LTP_GROUPING_SZ) {
				int stdp_tDiff = (simTime - runtimeDataGPU.synSpikeTime[p]);
				if (stdp_tDiff > 0) {
					if (groupConfigGPU[grpId].WithESTDP) {
						// Handle E-STDP curves
						switch (groupConfigGPU[grpId].WithESTDPcurve) {
						case EXP_CURVE: // exponential curve
							if (stdp_tDiff * groupConfigGPU[grpId].TAU_PLUS_INV_EXC < 25)
								runtimeDataGPU.wtChange[p] += STDP(stdp_tDiff, groupConfigGPU[grpId].ALPHA_PLUS_EXC, groupConfigGPU[grpId].TAU_PLUS_INV_EXC);
							break;
						case TIMING_BASED_CURVE: // sc curve
							if (stdp_tDiff * groupConfigGPU[grpId].TAU_PLUS_INV_EXC < 25) {
									if (stdp_tDiff <= groupConfigGPU[grpId].GAMMA)
										runtimeDataGPU.wtChange[p] += groupConfigGPU[grpId].OMEGA + groupConfigGPU[grpId].KAPPA * STDP(stdp_tDiff, groupConfigGPU[grpId].ALPHA_PLUS_EXC, groupConfigGPU[grpId].TAU_PLUS_INV_EXC);
									else // stdp_tDiff > GAMMA
										runtimeDataGPU.wtChange[p] -= STDP(stdp_tDiff, groupConfigGPU[grpId].ALPHA_PLUS_EXC, groupConfigGPU[grpId].TAU_PLUS_INV_EXC);
							}
							break;
						default:
							break;
						}
					}
					if (groupConfigGPU[grpId].WithISTDP) {
						// Handle I-STDP curves
						switch (groupConfigGPU[grpId].WithISTDPcurve) {
						case EXP_CURVE: // exponential curve
							if (stdp_tDiff * groupConfigGPU[grpId].TAU_PLUS_INV_INB < 25) { // LTP of inhibitory synapse, which decreases synapse weight
								runtimeDataGPU.wtChange[p] -= STDP(stdp_tDiff, groupConfigGPU[grpId].ALPHA_PLUS_INB, groupConfigGPU[grpId].TAU_PLUS_INV_INB);
							}
							break;
						case PULSE_CURVE: // pulse curve
							if (stdp_tDiff <= groupConfigGPU[grpId].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
								runtimeDataGPU.wtChange[p] -= groupConfigGPU[grpId].BETA_LTP;
							} else if (stdp_tDiff <= groupConfigGPU[grpId].DELTA) { // LTD of inhibitory syanpse, which increase sysnapse weight
								runtimeDataGPU.wtChange[p] -= groupConfigGPU[grpId].BETA_LTD;
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

__device__ inline bool getSpikeGenBit_GPU (unsigned int& nidPos)
{
	const int nidBitPos = nidPos%32;
	const int nidIndex  = nidPos/32;
	return ((runtimeDataGPU.spikeGenBits[nidIndex]>>nidBitPos)&0x1);
}

// setSpikeGenBit for given neuron and group..
void SNN::setSpikeGenBit_GPU(int nid, int grp) {
	checkAndSetGPUDevice();

	unsigned int nidPos    = (nid - groupConfig[grp].StartN + groupConfig[grp].Noffset);
	unsigned int nidBitPos = nidPos%32;
	unsigned int nidIndex  = nidPos/32;

	assert(nidIndex < (NgenFunc/32+1));

	snnRuntimeData.spikeGenBits[nidIndex] |= (1 << nidBitPos);
}



///////////////////////////////////////////////////////////////////////////
/// Device KERNEL function:     kernel_findFiring
// -----------------------
// KERNEL: findFiring
// -----------------------
// This kernel is responsible for finding the neurons that need to be fired.
// We use a buffered firing table that allows neuron to gradually load
// the buffer and make it easy to carry out the calculations in a single group.
// A single firing function is used for simple neurons and also for poisson neurons
///////////////////////////////////////////////////////////////////////////
__global__ 	void kernel_findFiring (int t, int sec, int simTime) {
	__shared__ volatile unsigned int fireCnt;
	__shared__ volatile unsigned int fireCntTest;
	__shared__ volatile unsigned int fireCntD1;
	__shared__ int 		fireTable[FIRE_CHUNK_CNT];
	__shared__ short int	fireGrpId[FIRE_CHUNK_CNT];
	__shared__ volatile int errCode;

	if (0==threadIdx.x) {
		fireCnt	  = 0; // initialize total cnt to 0
		fireCntD1  = 0; // initialize inh. cnt to 0
	}

	const int totBuffers=loadBufferCount;

	__syncthreads();

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int  nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);
		short int grpId   = STATIC_LOAD_GROUP(threadLoad);
		bool needToWrite = false;	// used by all neuron to indicate firing condition
		int  fireId      = 0;

		// threadId is valid and lies within the lastId.....
		if ((threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {
			// Simple poisson spiker uses the poisson firing probability
			// to detect whether it has fired or not....
			if( isPoissonGroup(grpId, nid) ) {
				if(groupConfigGPU[grpId].spikeGen) {
					unsigned int  offset      = nid-groupConfigGPU[grpId].StartN+groupConfigGPU[grpId].Noffset;
					needToWrite = getSpikeGenBit_GPU(offset);
				}
				else {
					needToWrite = getPoissonSpike_GPU(nid);
					// meow
					if (needToWrite && groupConfigGPU[grpId].withSpikeCounter) {
						int bufPos = groupConfigGPU[grpId].spkCntBufPos;
						int bufNeur = nid-groupConfigGPU[grpId].StartN;
						runtimeDataGPU.spkCntBuf[bufPos][bufNeur]++;
					}
				}
			}
			else {
				if (runtimeDataGPU.voltage[nid] >= 30.0f) {
					needToWrite = true;
					if (groupConfigGPU[grpId].withSpikeCounter) {
						int bufPos = groupConfigGPU[grpId].spkCntBufPos;
						int bufNeur = nid-groupConfigGPU[grpId].StartN;
						runtimeDataGPU.spkCntBuf[bufPos][bufNeur]++;
					}
				}
			}
		}

		// loop through a few times to ensure that we have added/processed all spikes that need to be written
		// if the buffer is small relative to the number of spikes needing to be written, we may have to empty the buffer a few times...
		for (uint8_t c=0;c<2;c++) {
			// we first increment fireCntTest to make sure we haven't filled the buffer
			if (needToWrite)
				fireId = atomicAdd((int*)&fireCntTest, 1);

			// if there is a spike and the buffer still has space...
			if (needToWrite && (fireId <(FIRE_CHUNK_CNT))) {
				// get our position in the buffer
				fireId = atomicAdd((int*)&fireCnt, 1);

				if (groupConfigGPU[grpId].MaxDelay == 1)
					atomicAdd((int*)&fireCntD1, 1);

				// store ID of the fired neuron
				needToWrite 	  = false;
				fireTable[fireId] = nid;
				fireGrpId[fireId] = grpId;//setFireProperties(grpId, isInhib);
			}

			__syncthreads();

			// table is full.. dump the local table to the global table before proceeding
			if (fireCntTest >= (FIRE_CHUNK_CNT)) {

				// clear the table and update...
				int retCode = newFireUpdate(fireTable,  fireGrpId, fireCnt, fireCntD1, simTime);
				if (retCode != 0) return;
				// update based on stdp rule
				// KILLME !!! if (simTime > 0))
				if (networkConfigGPU.sim_with_stdp && !networkConfigGPU.sim_in_testing)
					gpu_updateLTP (fireTable, fireGrpId, fireCnt, simTime);

				// reset counters
				if (0==threadIdx.x) {
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
		int retCode = newFireUpdate(fireTable, fireGrpId, fireCnt, fireCntD1, simTime);
		if (retCode != 0) return;

		if (networkConfigGPU.sim_with_stdp && !networkConfigGPU.sim_in_testing)
			gpu_updateLTP(fireTable, fireGrpId, fireCnt, simTime);
	}
}

//******************************** UPDATE CONDUCTANCES AND TOTAL SYNAPTIC CURRENT EVERY TIME STEP *****************************

#define LOG_CURRENT_GROUP 5
#define CURRENT_GROUP	  (1 << LOG_CURRENT_GROUP)

// Based on the bitvector used for indicating the presence of spike
// the global conductance values are updated..
__global__ void kernel_globalConductanceUpdate (int t, int sec, int simTime) {
	__shared__ int sh_quickSynIdTable[256];

	// Table for quick access
	for(int i=0; i < 256; i+=blockDim.x){
		if((i+threadIdx.x) < 256){
			sh_quickSynIdTable[i+threadIdx.x]=quickSynIdTableGPU[i+threadIdx.x];
		}
	}

	__syncthreads();

	const int totBuffers=loadBufferCount;
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int  post_nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);

		if ((threadIdx.x < lastId) && (IS_REGULAR_NEURON(post_nid, networkConfigGPU.numNReg, networkConfigGPU.numNPois))) {
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
			int   lmt      		 = runtimeDataGPU.Npre[post_nid];
			unsigned int cum_pos = runtimeDataGPU.cumulativePre[post_nid];

			// find the total current to this neuron...
			for(int j=0; (lmt)&&(j <= ((lmt-1)>>LOG_CURRENT_GROUP)); j++) {
				// because of malloc2D operation we are using pitch, post_nid, j to get
				// actual position of the input current....
				// int* tmp_I_set_p = ((int*)((char*)runtimeDataGPU.I_set + j * networkConfigGPU.I_setPitch) + post_nid);
				uint32_t* tmp_I_set_p  = getFiringBitGroupPtr(post_nid, j);

				uint32_t  tmp_I_set     = *tmp_I_set_p;

				// table lookup based find bits that are set
				int cnt = 0;
				int tmp_I_cnt = 0;
				while(tmp_I_set) {
					int k=(tmp_I_set>>(8*cnt))&0xff;
					if (k==0) {
						cnt = cnt+1;
						continue;
					}
					int wt_i = sh_quickSynIdTable[k];
					int wtId = (j*32 + cnt*8 + wt_i);

					SynInfo pre_Id   = runtimeDataGPU.preSynapticIds[cum_pos + wtId];
					//uint8_t  pre_grpId  = GET_CONN_GRP_ID(pre_Id);
					uint32_t  pre_nid  = GET_CONN_NEURON_ID(pre_Id);
					short int pre_grpId = runtimeDataGPU.grpIds[pre_nid];
					char type = groupConfigGPU[pre_grpId].Type;

					// load the synaptic weight for the wtId'th input
					float change = runtimeDataGPU.wt[cum_pos + wtId];

					// Adjust the weight according to STP scaling
					if(groupConfigGPU[pre_grpId].WithSTP) {
						int tD = 0; // \FIXME find delay
						// \FIXME I think pre_nid needs to be adjusted for the delay
						int ind_minus = getSTPBufPos(pre_nid,(simTime-tD-1)); // \FIXME should be adjusted for delay
						int ind_plus = getSTPBufPos(pre_nid,(simTime-tD));
						// dI/dt = -I/tau_S + A * u^+ * x^- * \delta(t-t_{spk})
						change *= groupConfigGPU[pre_grpId].STP_A * runtimeDataGPU.stpx[ind_minus] * runtimeDataGPU.stpu[ind_plus];
					}

					if (networkConfigGPU.sim_with_conductances) {
						short int connId = runtimeDataGPU.cumConnIdPre[cum_pos+wtId];
						if (type & TARGET_AMPA)
							AMPA_sum += change*d_mulSynFast[connId];
						if (type & TARGET_NMDA) {
							if (networkConfigGPU.sim_with_NMDA_rise) {
								NMDA_r_sum += change*d_mulSynSlow[connId]*networkConfigGPU.sNMDA;
								NMDA_d_sum += change*d_mulSynSlow[connId]*networkConfigGPU.sNMDA;
							} else {
								NMDA_sum += change*d_mulSynSlow[connId];
							}
						}
						if (type & TARGET_GABAa)
							GABAa_sum += change*d_mulSynFast[connId];	// wt should be negative for GABAa and GABAb
						if (type & TARGET_GABAb) {						// but that is dealt with below
							if (networkConfigGPU.sim_with_GABAb_rise) {
								GABAb_r_sum += change*d_mulSynSlow[connId]*networkConfigGPU.sGABAb;
								GABAb_d_sum += change*d_mulSynSlow[connId]*networkConfigGPU.sGABAb;
							} else {
								GABAb_sum += change*d_mulSynSlow[connId];
							}
						}
					}
					else {
						// current based model with STP (CUBA)
						// updated current for neuron 'post_nid'
						AMPA_sum +=  change;
					}

					tmp_I_cnt++;
					tmp_I_set = tmp_I_set & (~(1<<(8*cnt+wt_i)));
				}

				// reset the input if there are any bit'wt set
				if(tmp_I_cnt)
					*tmp_I_set_p = 0;

				__syncthreads();
			}

			__syncthreads();

			if (networkConfigGPU.sim_with_conductances) {
				// don't add mulSynFast/mulSynSlow here, because they depend on the exact pre<->post connection, not
				// just post_nid
				runtimeDataGPU.gAMPA[post_nid]        += AMPA_sum;
				runtimeDataGPU.gGABAa[post_nid]       -= GABAa_sum; // wt should be negative for GABAa and GABAb
				if (networkConfigGPU.sim_with_NMDA_rise) {
					runtimeDataGPU.gNMDA_r[post_nid]  += NMDA_r_sum;
					runtimeDataGPU.gNMDA_d[post_nid]  += NMDA_d_sum;
				} else {
					runtimeDataGPU.gNMDA[post_nid]    += NMDA_sum;
				}
				if (networkConfigGPU.sim_with_GABAb_rise) {
					runtimeDataGPU.gGABAb_r[post_nid] -= GABAb_r_sum;
					runtimeDataGPU.gGABAb_d[post_nid] -= GABAb_d_sum;
				} else {
					runtimeDataGPU.gGABAb[post_nid]   -= GABAb_sum;
				}
			}
			else {
				runtimeDataGPU.current[post_nid] += AMPA_sum;
			}
		}
	}
}

//************************ UPDATE GLOBAL STATE EVERY TIME STEP *******************************************************//

__device__ void updateNeuronState(int& nid, int& grpId) {
	float v = runtimeDataGPU.voltage[nid];
	float u = runtimeDataGPU.recovery[nid];
	float I_sum, NMDAtmp;
	float gNMDA, gGABAb;

	// loop that allows smaller integration time step for v's and u's
	for (int c=0; c<COND_INTEGRATION_SCALE; c++) {
		I_sum = 0.0f;
		if (networkConfigGPU.sim_with_conductances) {
			NMDAtmp = (v+80.0f)*(v+80.0f)/60.0f/60.0f;
			gNMDA = (networkConfigGPU.sim_with_NMDA_rise) ? (runtimeDataGPU.gNMDA_d[nid]-runtimeDataGPU.gNMDA_r[nid]) : runtimeDataGPU.gNMDA[nid];
			gGABAb = (networkConfigGPU.sim_with_GABAb_rise) ? (runtimeDataGPU.gGABAb_d[nid]-runtimeDataGPU.gGABAb_r[nid]) : runtimeDataGPU.gGABAb[nid];
			I_sum = -(   runtimeDataGPU.gAMPA[nid]*(v-0.0f)
					   + gNMDA*NMDAtmp/(1.0f+NMDAtmp)*(v-0.0f)
					   + runtimeDataGPU.gGABAa[nid]*(v+70.0f)
					   + gGABAb*(v+90.0f)
					 );
		}
		else {
			I_sum = runtimeDataGPU.current[nid];
		}

		// update vpos and upos for the current neuron
		v += ((0.04f*v+5.0f)*v+140.0f-u+I_sum+runtimeDataGPU.extCurrent[nid])/COND_INTEGRATION_SCALE;
		if (v > 30.0f) { 
			v = 30.0f; // break the loop but evaluate u[i]
			c=COND_INTEGRATION_SCALE;
		}
		if (v < -90.0f)
			v = -90.0f;
		u += (runtimeDataGPU.Izh_a[nid]*(runtimeDataGPU.Izh_b[nid]*v-u)/COND_INTEGRATION_SCALE);
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

// homeostasis in GPU_MODE
__device__ inline void updateHomeoStaticState(int& pos, int& grpId)
{
	// here the homeostasis adjustment
	if (groupConfigGPU[grpId].WithHomeostasis)
		runtimeDataGPU.avgFiring[pos] *= (groupConfigGPU[grpId].avgTimeScale_decay);
}

//!
/*! Global Kernel function:      kernel_globalStateUpdate
 *  \brief update neuron state
 *  change this with selective upgrading technique used for firing neurons
 */
__global__ void kernel_globalStateUpdate (int t, int sec, int simTime)
{
	const int totBuffers = loadBufferCount;

	// update neuron state
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int nid = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId = STATIC_LOAD_SIZE(threadLoad);
		int  grpId = STATIC_LOAD_GROUP(threadLoad);

		if ((threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {

			if (IS_REGULAR_NEURON(nid, networkConfigGPU.numNReg, networkConfigGPU.numNPois)) {

				// update neuron state here....
				updateNeuronState(nid, grpId);

				// TODO: Could check with homeostasis flag here
				updateHomeoStaticState(nid, grpId);
			}
		}
	}		
}

//!
/*! Global Kernel function:      kernel_globalGroupStateUpdate
 *  \brief update group state
 *  update the concentration of neuronmodulator
 */
__global__ void kernel_globalGroupStateUpdate (int t)
{
	// update group state
	int grpIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (grpIdx < networkConfigGPU.numGroups) {
		// decay dopamine concentration
		if ((groupConfigGPU[grpIdx].WithESTDPtype == DA_MOD || groupConfigGPU[grpIdx].WithISTDPtype == DA_MOD) && runtimeDataGPU.grpDA[grpIdx] > groupConfigGPU[grpIdx].baseDP) {
			runtimeDataGPU.grpDA[grpIdx] *= groupConfigGPU[grpIdx].decayDP;
		}
		runtimeDataGPU.grpDABuffer[grpIdx][t] = runtimeDataGPU.grpDA[grpIdx]; // log dopamine concentration
	}
}

//******************************** UPDATE STP STATE  EVERY TIME STEP **********************************************
///////////////////////////////////////////////////////////
/// 	Global Kernel function: gpu_STPUpdate		///
/// 	This function is called every time step			///
///////////////////////////////////////////////////////////
__global__ void kernel_STPUpdateAndDecayConductances (int t, int sec, int simTime)
{
	// global id
	// int gid=threadIdx.x + blockDim.x*blockIdx.x;

	const int totBuffers=loadBufferCount;
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		int nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);
		uint32_t  grpId  = STATIC_LOAD_GROUP(threadLoad);


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

		if (groupConfigGPU[grpId].WithSTP && (threadIdx.x < lastId) && (nid < networkConfigGPU.numN)) {
			int ind_plus  = getSTPBufPos(nid, simTime);
			int ind_minus = getSTPBufPos(nid, (simTime-1)); // \FIXME sure?
				runtimeDataGPU.stpu[ind_plus] = runtimeDataGPU.stpu[ind_minus]*(1.0f-groupConfigGPU[grpId].STP_tau_u_inv);
				runtimeDataGPU.stpx[ind_plus] = runtimeDataGPU.stpx[ind_minus] + (1.0f-runtimeDataGPU.stpx[ind_minus])*groupConfigGPU[grpId].STP_tau_x_inv;
		}
	}
}

//********************************UPDATE SYNAPTIC WEIGHTS EVERY SECOND  *************************************************************

//////////////////////////////////////////////////////////////////
/// Global Kernel function:      kernel_updateWeights_static   ///
// KERNEL DETAILS:
//   This kernel is called every second to adjust the timingTable and globalFiringTable
//   We do the following thing:
//   1. We discard all firing information that happened more than 1000-maxDelay_ time step.
//   2. We move the firing information that happened in the last 1000-maxDelay_ time step to
//      the begining of the gloalFiringTable.
//   3. We read each value of "wtChange" and update the value of "synaptic weights wt".
//      We also clip the "synaptic weight wt" to lie within the required range.
//////////////////////////////////////////////////////////////////

// Rewritten updateSynapticWeights to include homeostasis.  We should consider using flag.
__device__ void updateSynapticWeights(int& nid, unsigned int& jpos, int& grpId, float& diff_firing, float& homeostasisScale, float& baseFiring, float& avgTimeScaleInv)
{
	// This function does not get called if the neuron group has all fixed weights.
	// t_twChange is adjusted by stdpScaleFactor based on frequency of weight updates (e.g., 10ms, 100ms, 1s)	
	float t_wt = runtimeDataGPU.wt[jpos];
	float t_wtChange = runtimeDataGPU.wtChange[jpos];
	float t_effectiveWtChange = networkConfigGPU.stdpScaleFactor * t_wtChange;
	float t_maxWt = runtimeDataGPU.maxSynWt[jpos];

	switch (groupConfigGPU[grpId].WithESTDPtype) {
	case STANDARD:
		if (groupConfigGPU[grpId].WithHomeostasis) {
			// this factor is slow
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += t_effectiveWtChange;
		}
		break;
	case DA_MOD:
		if (groupConfigGPU[grpId].WithHomeostasis) {
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

	switch (groupConfigGPU[grpId].WithISTDPtype) {
	case STANDARD:
		if (groupConfigGPU[grpId].WithHomeostasis) {
			// this factor is slow
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1.0f+fabs(diff_firing)*50.0f);
		} else {
			t_wt += t_effectiveWtChange;
		}
		break;
	case DA_MOD:
		if (groupConfigGPU[grpId].WithHomeostasis) {
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

	runtimeDataGPU.wt[jpos] = t_wt;
	runtimeDataGPU.wtChange[jpos] = t_wtChange;
}


#define UPWTS_CLUSTERING_SZ	32
__global__ void kernel_updateWeights()
{
	__shared__ volatile int errCode;
	__shared__ int    		startId, lastId, grpId, totBuffers, grpNCnt;
	__shared__ int2 		threadLoad;
	// added for homeostasis
	__shared__ float		homeostasisScale, avgTimeScaleInv;

	if(threadIdx.x==0) {
		totBuffers=loadBufferCount;
		grpNCnt	= (blockDim.x/UPWTS_CLUSTERING_SZ) + ((blockDim.x%UPWTS_CLUSTERING_SZ)!=0);
	}

	__syncthreads();

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		if ( threadIdx.x) {
			threadLoad  = getStaticThreadLoad(bufPos);
			startId 	= STATIC_LOAD_START(threadLoad);
			lastId  	= STATIC_LOAD_SIZE(threadLoad);
			grpId   	= STATIC_LOAD_GROUP(threadLoad);

			// homestasis functions
			if (groupConfigGPU[grpId].WithHomeostasis) {
				homeostasisScale = groupConfigGPU[grpId].homeostasisScale;
				avgTimeScaleInv = groupConfigGPU[grpId].avgTimeScaleInv;
			} else {
				homeostasisScale = 0.0f;
				avgTimeScaleInv = 1.0f;
			}
		}

		__syncthreads();

	// the weights are fixed for this group.. so dont make any changes on
	// the weight and continue to the next set of neurons...
	if (groupConfigGPU[grpId].FixedInputWts)
		continue;

		int nid=(threadIdx.x/UPWTS_CLUSTERING_SZ) + startId;
		// update the synaptic weights from the synaptic weight derivatives
		for(; nid < startId+lastId; nid+=grpNCnt) {
			int Npre_plastic = runtimeDataGPU.Npre_plastic[nid];
			unsigned int cumulativePre = runtimeDataGPU.cumulativePre[nid];
			float diff_firing  = 0.0f;
			float baseFiring = 0.0f;

			if (groupConfigGPU[grpId].WithHomeostasis) {
				diff_firing  = (1.0f-runtimeDataGPU.avgFiring[nid]*runtimeDataGPU.baseFiringInv[nid]);
				baseFiring = runtimeDataGPU.baseFiring[nid];
			}

			const int threadIdGrp   = (threadIdx.x%UPWTS_CLUSTERING_SZ);
			// synaptic grouping
			for(unsigned int j=cumulativePre; j < (cumulativePre+Npre_plastic); j+=UPWTS_CLUSTERING_SZ) {
				//excitatory connection change the synaptic weights
				unsigned int jpos=j+threadIdGrp;
				if(jpos < (cumulativePre+Npre_plastic)) {
					updateSynapticWeights(nid, jpos, grpId, diff_firing, homeostasisScale, baseFiring, avgTimeScaleInv);
				}
			}
		}
	}
}

__global__ void kernel_updateFiring_static() {
	int gnthreads=blockDim.x*gridDim.x;

	// Shift the firing table so that the initial information in
	// the firing table contain the firing information for the last maxDelay_ time step
	for(int p=timeTableD2GPU[999],k=0;
		p<timeTableD2GPU[999+networkConfigGPU.maxDelay+1];
		p+=gnthreads,k+=gnthreads) {
		if((p+threadIdx.x)<timeTableD2GPU[999+networkConfigGPU.maxDelay+1])
			runtimeDataGPU.firingTableD2[k+threadIdx.x]=runtimeDataGPU.firingTableD2[p+threadIdx.x];
	}
}

//********************************UPDATE TABLES AND COUNTERS EVERY SECOND  *************************************************************
// KERNEL DESCRIPTION:
// This is the second part of the previous kernel "kernel_updateWeightsFiring"
// After all the threads/blocks had adjusted the firing table and the synaptic weights,
// we update the timingTable so that the firing information that happended in the last maxDelay_
// time step would become the first maxDelay_ time step firing information for the next cycle of simulation.
// We also reset/update various counters to appropriate values as indicated in the second part 
// of this kernel.
__global__ void kernel_updateFiring()
{
	// CHECK !!!
	int maxDelay_ = networkConfigGPU.maxDelay;
	// reset the firing table so that we have the firing information
	// for the last maxDelay_ time steps to be used for the next cycle of the simulation
	if(blockIdx.x==0) {
		for(int i=threadIdx.x; i < maxDelay_; i+=blockDim.x) {
			// use i+1 instead of just i because timeTableD2GPU[0] should always be 0
			timeTableD2GPU[i+1] = timeTableD2GPU[1000+i+1]-timeTableD2GPU[1000];
			timeTableD1GPU[i+1] = timeTableD1GPU[1000+i+1]-timeTableD1GPU[1000];
		}
	}

	__syncthreads();

	// reset various counters for the firing information
	if((blockIdx.x==0)&&(threadIdx.x==0)) {
		timeTableD1GPU[networkConfigGPU.maxDelay]  = 0;
		spikeCountD2GPU += spikeCountD2SecGPU;
		spikeCountD1GPU += spikeCountD1SecGPU;
		spikeCountD2SecGPU	= timeTableD2GPU[networkConfigGPU.maxDelay];
		secD2fireCntTest	= timeTableD2GPU[networkConfigGPU.maxDelay];
		spikeCountD1SecGPU	= 0;
		secD1fireCntTest = 0;
	}
}

/// THIS KERNEL IS USED BY BLOCK_CONFIG_VERSION
__global__ void kernel_updateFiring2()
{
	// reset various counters for the firing information
	if((blockIdx.x==0)&&(threadIdx.x==0)) {
		// timeTableD1GPU[networkConfigGPU.maxDelay]  = 0;
		spikeCountD2GPU	+= spikeCountD2SecGPU;
		spikeCountD1GPU	+= spikeCountD1SecGPU;
		spikeCountD2SecGPU	= 0; //timeTableD2GPU[networkConfigGPU.maxDelay];
		spikeCountD1SecGPU	= 0;
		secD2fireCntTest = 0; //timeTableD2GPU[networkConfigGPU.maxDelay];
		secD1fireCntTest = 0;
	}
}

//****************************** GENERATE POST-SYNAPTIC CURRENT EVERY TIME-STEP  ****************************
__device__ int generatePostSynapticSpike(int& simTime, int& firingId, int& myDelayIndex, volatile int& offset, bool unitDelay)
{
	int errCode = false;

	// get the post synaptic information for specific delay
	SynInfo post_info = runtimeDataGPU.postSynapticIds[offset+myDelayIndex];

	// get neuron id
	int nid = GET_CONN_NEURON_ID(post_info);//(post_info&POST_SYN_NEURON_MASK);

	// get synaptic id
	int syn_id = GET_CONN_SYN_ID(post_info); //(post_info>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;

	// get the actual position of the synapses and other variables...
	unsigned int pos_ns = runtimeDataGPU.cumulativePre[nid] + syn_id;

	short int pre_grpId = runtimeDataGPU.grpIds[firingId];
	//short int pre_grpId = GET_FIRING_TABLE_GID(firingId);
	//int pre_nid = GET_FIRING_TABLE_NID(firingId);

	// Error MNJ... this should have been from nid.. not firingId...
	// int  nid  = GET_FIRING_TABLE_NID(firingId);
//	int    post_grpId;		// STP uses pre_grpId, STDP used post_grpId...
//	findGrpId_GPU(nid, post_grpId);
	int post_grpId = runtimeDataGPU.grpIds[nid];

	if(post_grpId == -1)
		return CURRENT_UPDATE_ERROR4;

	// Got one spike from dopaminergic neuron, increase dopamine concentration in the target area
	if (groupConfigGPU[pre_grpId].Type & TARGET_DA) {
		atomicAdd(&(runtimeDataGPU.grpDA[post_grpId]), 0.04f);
	}

	setFiringBitSynapses(nid, syn_id);

	runtimeDataGPU.synSpikeTime[pos_ns] = simTime;		  //uncoalesced access

	// STDP calculation: the post-synaptic neuron fires before the arrival of pre-synaptic neuron's spike
	if (groupConfigGPU[post_grpId].WithSTDP && !networkConfigGPU.sim_in_testing)  {
		int stdp_tDiff = simTime-runtimeDataGPU.lastSpikeTime[nid];
		if (stdp_tDiff >= 0) {
			if (groupConfigGPU[post_grpId].WithESTDP) {
				// Handle E-STDP curves
				switch (groupConfigGPU[post_grpId].WithESTDPcurve) {
				case EXP_CURVE: // exponential curve
				case TIMING_BASED_CURVE: // sc curve
					if (stdp_tDiff * groupConfigGPU[post_grpId].TAU_MINUS_INV_EXC < 25.0f)
						runtimeDataGPU.wtChange[pos_ns] += STDP( stdp_tDiff, groupConfigGPU[post_grpId].ALPHA_MINUS_EXC, groupConfigGPU[post_grpId].TAU_MINUS_INV_EXC); // uncoalesced access
					break;
				default:
					break;
				}
			}
			if (groupConfigGPU[post_grpId].WithISTDP) {
				// Handle I-STDP curves
				switch (groupConfigGPU[post_grpId].WithISTDPcurve) {
				case EXP_CURVE: // exponential curve
					if ((stdp_tDiff * groupConfigGPU[post_grpId].TAU_MINUS_INV_INB) < 25.0f) { // LTD of inhibitory syanpse, which increase synapse weight
						runtimeDataGPU.wtChange[pos_ns] -= STDP(stdp_tDiff, groupConfigGPU[post_grpId].ALPHA_MINUS_INB, groupConfigGPU[post_grpId].TAU_MINUS_INV_INB);
					}
					break;
				case PULSE_CURVE: // pulse curve
					if (stdp_tDiff <= groupConfigGPU[post_grpId].LAMBDA) { // LTP of inhibitory synapse, which decreases synapse weight
						runtimeDataGPU.wtChange[pos_ns] -= groupConfigGPU[post_grpId].BETA_LTP;
					} else if (stdp_tDiff <= groupConfigGPU[post_grpId].DELTA) { // LTD of inhibitory syanpse, which increase synapse weight
						runtimeDataGPU.wtChange[pos_ns] -= groupConfigGPU[post_grpId].BETA_LTD;
					}
					break;
				default:
					break;
				}
			}
		}
	}
	
	return errCode;
}

#define NUM_THREADS 			128
#define EXCIT_READ_CHUNK_SZ		(NUM_THREADS>>1)

//  KERNEL DESCRIPTION:-
//  This kernel is required for updating and generating spikes for delays greater than 1 from the fired neuron. 
//  The LTD computation is also executed by this approach kernel.
__global__ void kernel_doCurrentUpdateD2(int simTimeMs, int simTimeSec, int simTime)
{
	__shared__	volatile int sh_neuronOffsetTable[EXCIT_READ_CHUNK_SZ+2];
	__shared__	int sh_delayLength[EXCIT_READ_CHUNK_SZ+2];
	__shared__	int sh_delayIndexStart[EXCIT_READ_CHUNK_SZ+2];
	__shared__	int sh_firingId[EXCIT_READ_CHUNK_SZ+2];
	//__shared__	int sh_axonDelay[EXCIT_READ_CHUNK_SZ+2];
	__shared__ volatile int sh_NeuronCnt;

	const int threadIdSwarp	= (threadIdx.x%WARP_SIZE);
	const int swarpId		= (threadIdx.x/WARP_SIZE);
	int updateCnt	  	= 0;

	__shared__ volatile int sh_blkErrCode;

	// this variable is used to record the
	// number of updates done by different blocks
	if(threadIdx.x<=0)   {
		sh_NeuronCnt = 0;
	}

	__syncthreads();

	// stores the number of fired neurons at time t
	int k      = spikeCountD2SecGPU - 1;

	// stores the number of fired neurons at time (t - maxDelay_)
	int k_end  = tex1Dfetch (timeTableD2GPU_tex, simTimeMs+1+timeTableD2GPU_tex_offset);

	int t_pos  = simTimeMs;

	// we need to read (k-k_end) neurons from the firing 
	// table and do necesary updates for all these post-synaptic
	// connection in these neurons..
	while((k>=k_end) &&(k>=0)) {

		// at any point of time EXCIT_READ_CHUNK_SZ neurons
		// read different firing id from the firing table
		if (threadIdx.x<EXCIT_READ_CHUNK_SZ) {
			int fPos = k - (EXCIT_READ_CHUNK_SZ*blockIdx.x) - threadIdx.x; 
			if ((fPos >= 0) && (fPos >= k_end)) {

				// get the neuron nid here....
				//int val = runtimeDataGPU.firingTableD2[fPos];
				//int nid = GET_FIRING_TABLE_NID(val);
				int nid = runtimeDataGPU.firingTableD2[fPos];

				// find the time of firing based on the firing number fPos
				while ( !((fPos >= tex1Dfetch(timeTableD2GPU_tex, t_pos+networkConfigGPU.maxDelay+timeTableD2GPU_tex_offset)) 
					&& (fPos <  tex1Dfetch(timeTableD2GPU_tex, t_pos+networkConfigGPU.maxDelay+1+timeTableD2GPU_tex_offset)))) {
					t_pos--;
				}

				// find the time difference between firing of the neuron and the current time
				int tD  = simTimeMs - t_pos;

				// find the various delay parameters for neuron 'nid', with a delay of 'tD'
				//sh_axonDelay[threadIdx.x]	 = tD;
				int tPos = (networkConfigGPU.maxDelay+1)*nid+tD;
				//sh_firingId[threadIdx.x]	 	 = val;
				sh_firingId[threadIdx.x] = nid;
				sh_neuronOffsetTable[threadIdx.x]= runtimeDataGPU.cumulativePost[nid];
				sh_delayLength[threadIdx.x]      = runtimeDataGPU.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]  = runtimeDataGPU.postDelayInfo[tPos].delay_index_start;

				// This is to indicate that the current thread
				// has a valid delay parameter for post-synaptic firing generation
				atomicAdd((int*)&sh_NeuronCnt,1);
			}
		}

		__syncthreads();

		// if cnt is zero than no more neurons need to generate
		// post-synaptic firing, then we break the loop.
		int cnt = sh_NeuronCnt;
		updateCnt += cnt;
		if (cnt==0) {
			break;
		}

		// first WARP_SIZE threads the post synaptic
		// firing for first neuron, and so on. each of this group
		// needs to generate (numPostSynapses/maxDelay_) spikes for every fired neuron, every second
		// for numPostSynapses=500,maxDelay_=20, we need to generate 25 spikes for each fired neuron
		// for numPostSynapses=600,maxDelay_=20, we need to generate 30 spikes for each fired neuron 
		for (int pos=swarpId; pos < cnt; pos += (NUM_THREADS/WARP_SIZE)) {

			int   delId     = threadIdSwarp;

			while(delId < sh_delayLength[pos]) {

			int delIndex = sh_delayIndexStart[pos]+delId;

				sh_blkErrCode = generatePostSynapticSpike(simTime,
						sh_firingId[pos],				// presynaptic nid
						delIndex, 	// delayIndex
						sh_neuronOffsetTable[pos], 		// offset
						false);							// false for unitDelay type..

				delId += WARP_SIZE;
			}
		} //(for all excitory neurons in table)

		__syncthreads();

		if(threadIdx.x==0) {
			sh_NeuronCnt = 0;
		}

		k = k - (gridDim.x*EXCIT_READ_CHUNK_SZ);

		__syncthreads();
	}

	__syncthreads();
}

//  KERNEL DESCRIPTION:-
//  This kernel is required for updating and generating spikes on connections
//  with a delay of 1ms from the fired neuron. This function looks
//  mostly like the previous kernel but has been optimized for a fixed delay of 1ms. 
//  Ultimately we may merge this kernel with the previous kernel.
__global__ void kernel_doCurrentUpdateD1(int simTimeMs, int simTimeSec, int simTime)
{
	__shared__ volatile	int sh_NeuronCnt;
	__shared__ volatile int sh_neuronOffsetTable[NUM_THREADS/WARP_SIZE+2];
//		__shared__	int sh_firedTimeTable[NUM_THREADS/WARP_SIZE+2];
	__shared__	int sh_delayLength[NUM_THREADS/WARP_SIZE+2];
	__shared__	int sh_firingId[NUM_THREADS/WARP_SIZE+2];
	__shared__	int sh_delayIndexStart[NUM_THREADS/WARP_SIZE+2];
	__shared__	int sh_timing;

	const int swarpId		= threadIdx.x/WARP_SIZE;  // swarp id within warp
	const int numSwarps     = blockDim.x/WARP_SIZE;   // number of sub-warps (swarps)
	const int threadIdSwarp	= threadIdx.x%WARP_SIZE;  // thread id within swarp

	__shared__ volatile int sh_blkErrCode;

	// load the time table for neuron firing
	int computedNeurons = 0;
	if (0==threadIdx.x) {
		sh_timing = timeTableD1GPU[simTimeMs+networkConfigGPU.maxDelay]; // ??? check check ???
	}
	__syncthreads();

	int kPos = sh_timing + (blockIdx.x*numSwarps);

	__syncthreads();

	// Do as long as we have some valid neuron
	while((kPos >=0)&&(kPos < spikeCountD1SecGPU)) {
		int fPos = -1;
		// a group of threads loads the delay information
		if (threadIdx.x < numSwarps) {
			sh_neuronOffsetTable[threadIdx.x] = -1;
			fPos = kPos + threadIdx.x;

			// find the neuron nid and also delay information from fPos
			if((fPos>=0)&&(fPos < spikeCountD1SecGPU)) {
				atomicAdd((int*)&sh_NeuronCnt,1);
				//int val  = runtimeDataGPU.firingTableD1[fPos];
				//int nid  = GET_FIRING_TABLE_NID(val);
				int nid = runtimeDataGPU.firingTableD1[fPos];
				int tPos = (networkConfigGPU.maxDelay+1)*nid;
				//sh_firingId[threadIdx.x] 	 	 = val;
				sh_firingId[threadIdx.x] = nid;
				sh_neuronOffsetTable[threadIdx.x]= runtimeDataGPU.cumulativePost[nid];
				sh_delayLength[threadIdx.x]      = runtimeDataGPU.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]  = runtimeDataGPU.postDelayInfo[tPos].delay_index_start;
			}
		}

		__syncthreads();

		// useful to measure the load balance for each block..
		if(threadIdx.x==0)  computedNeurons += sh_NeuronCnt;

		// no more fired neuron from table... we just break from loop
		if (sh_NeuronCnt==0) {
			break;
		}

		__syncthreads();

		int offset = sh_neuronOffsetTable[swarpId];

		if (threadIdx.x == 0) {
			sh_NeuronCnt = 0;
		}

		if (offset>=0) {
			int delId=threadIdSwarp;

			while(delId < sh_delayLength[swarpId]) {

				int delIndex = (sh_delayIndexStart[swarpId]+delId);

				sh_blkErrCode = generatePostSynapticSpike(simTime,
						sh_firingId[swarpId],				// presynaptic nid
						delIndex,							// delayIndex
						sh_neuronOffsetTable[swarpId], 		// offset
						true);								// true for unit delay connection..

				delId += WARP_SIZE;
			}
		}

		__syncthreads();

		kPos = kPos + (gridDim.x*numSwarps);
	}
}

void SNN::copyPostConnectionInfo(RuntimeData* dest, bool allocateMem) {
	checkAndSetGPUDevice();

	assert(dest->memType == GPU_MODE);
	if (allocateMem) {
		assert(dest->allocated == false);
	} else {
		assert(dest->allocated == true);
	}
	assert(snnState == OPTIMIZED_PARTITIONED_SNN || snnState == EXECUTABLE_SNN);

	// beginning position for the post-synaptic information
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePost, sizeof(snnRuntimeData.cumulativePost[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->cumulativePost, snnRuntimeData.cumulativePost, sizeof(int) * numN, cudaMemcpyHostToDevice));

	// number of postsynaptic connections
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npost, sizeof(snnRuntimeData.Npost[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy( dest->Npost, snnRuntimeData.Npost, sizeof(snnRuntimeData.Npost[0]) * numN, cudaMemcpyHostToDevice));

	// static specific mapping and actual post-synaptic delay metric
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postDelayInfo, sizeof(snnRuntimeData.postDelayInfo[0]) * numN * (maxDelay_ + 1)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->postDelayInfo, snnRuntimeData.postDelayInfo, sizeof(snnRuntimeData.postDelayInfo[0]) * numN * (maxDelay_ + 1), cudaMemcpyHostToDevice));

	// actual post synaptic connection information...
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postSynapticIds, sizeof(snnRuntimeData.postSynapticIds[0]) * (numPostSynNet + 10)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->postSynapticIds, snnRuntimeData.postSynapticIds, sizeof(snnRuntimeData.postSynapticIds[0]) * (numPostSynNet + 10), cudaMemcpyHostToDevice)); //FIXME: why +10 post synapses
	networkConfig.numPostSynNet = numPostSynNet;

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->preSynapticIds, sizeof(snnRuntimeData.preSynapticIds[0]) * (numPreSynNet + 10)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->preSynapticIds, snnRuntimeData.preSynapticIds, sizeof(snnRuntimeData.preSynapticIds[0]) * (numPreSynNet + 10), cudaMemcpyHostToDevice)); //FIXME: why +10 post synapses
	networkConfig.numPreSynNet = numPreSynNet;
}

void SNN::copyConnections(RuntimeData* dest, int kind, bool allocateMem) {
	checkAndSetGPUDevice();
	// void* devPtr;
	// allocateMem memory only if destination memory is not allocated !!!
	assert(allocateMem && (dest->allocated != 1));
	if(kind == cudaMemcpyHostToDevice) {
		assert(dest->memType == GPU_MODE);
	} else {
		assert(dest->memType == CPU_MODE);
	}

	networkConfig.I_setLength = ceil(((maxNumPreSynGrp) / 32.0f));
	if(allocateMem)
		cudaMallocPitch((void**)&dest->I_set, &networkConfig.I_setPitch, sizeof(int) * numNReg, networkConfig.I_setLength);
	assert(networkConfig.I_setPitch > 0 || maxNumPreSynGrp==0);
	CUDA_CHECK_ERRORS(cudaMemset(dest->I_set, 0, networkConfig.I_setPitch * networkConfig.I_setLength));

	// connection synaptic lengths and cumulative lengths...
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre, sizeof(dest->Npre[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre, snnRuntimeData.Npre, sizeof(dest->Npre[0]) * numN, cudaMemcpyHostToDevice));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	if (!sim_with_fixedwts) {
		// presyn excitatory connections
		if(allocateMem) 
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plastic, sizeof(dest->Npre_plastic[0]) * numN));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre_plastic, snnRuntimeData.Npre_plastic, sizeof(dest->Npre_plastic[0]) * numN, cudaMemcpyHostToDevice));

		float* Npre_plasticInv = new float[numN];
		for (int i = 0; i < numN; i++)
			Npre_plasticInv[i] = 1.0f / snnRuntimeData.Npre_plastic[i];

		if(allocateMem)
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plasticInv, sizeof(dest->Npre_plasticInv[0]) * numN));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre_plasticInv, Npre_plasticInv, sizeof(dest->Npre_plasticInv[0]) * numN, cudaMemcpyHostToDevice));

		delete[] Npre_plasticInv;
	}
		
	// beginning position for the pre-synaptic information
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePre, sizeof(int) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->cumulativePre, snnRuntimeData.cumulativePre, sizeof(int) * numN, cudaMemcpyHostToDevice));

  // allocate randomPtr.... containing the firing information for the random firing neurons...
  // I'm not sure this is implemented. -- KDC
  //		if(allocateMem)  CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->randId, sizeof(int)*numN));
  //		networkConfig.numRandNeurons=numRandNeurons;

  // copy the properties of the noise generator here.....
  // I'm not sure this is implemented. -- KDC
  //		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->noiseGenProp, sizeof(noiseGenProperty_t)*numNoise));
  //		CUDA_CHECK_ERRORS( cudaMemcpy( dest->noiseGenProp, &noiseGenGroup[0], sizeof(noiseGenProperty_t)*numNoise, cudaMemcpyHostToDevice));

	// allocate the poisson neuron poissonFireRate
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->poissonFireRate, sizeof(dest->poissonFireRate[0]) * numNPois));
	CUDA_CHECK_ERRORS(cudaMemset(dest->poissonFireRate, 0, sizeof(dest->poissonFireRate[0]) * numNPois));

	copyPostConnectionInfo(dest, allocateMem);
}

void SNN::checkDestSrcPtrs(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	if(kind==cudaMemcpyHostToDevice) {
		assert(dest->memType == GPU_MODE);
		assert(src->memType  == CPU_MODE);
	}
	else {
		assert(dest->memType == CPU_MODE);
		assert(src->memType  == GPU_MODE);
	}

	if (allocateMem) {
		// if allocateMem = false, then the destination must be allocated..
		assert(dest->allocated==false);

		// if allocateMem = true, then we should not specify any specific group.
		assert(grpId == -1);
	}
	else {
		// if allocateMem = true, then the destination must be empty without allocation..
		assert(dest->allocated == true);
	}

	// source should always be allocated...
	assert(src->allocated==true);
}

void SNN::copyFiringStateFromGPU (int grpId) {
	checkAndSetGPUDevice();

	int ptrPos, length;

	if(grpId == -1) {
		ptrPos  = 0;
		length = numN;
	}
	else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}

	assert(length>0 && length <= numN);

	RuntimeData* dest = &snnRuntimeData;
	RuntimeData* src  = &gpuRuntimeData;

	// Spike Cnt. Firing...
	CUDA_CHECK_ERRORS( cudaMemcpy(&dest->nSpikeCnt[ptrPos], &src->nSpikeCnt[ptrPos], sizeof(int)*length, 
		cudaMemcpyDeviceToHost) );
}

void SNN::copyConductanceAMPA(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	//conductance information
	assert(src->gAMPA  != NULL);
	if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gAMPA, sizeof(float)*length));
	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gAMPA[ptrPos], &src->gAMPA[ptrPos], sizeof(float)*length, kind));
}

void SNN::copyConductanceNMDA(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	if (isSimulationWithNMDARise()) {
		assert(src->gNMDA_r != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gNMDA_r, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gNMDA_r[ptrPos], &src->gNMDA_r[ptrPos], sizeof(float)*length, kind));

		assert(src->gNMDA_d != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gNMDA_d, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gNMDA_d[ptrPos], &src->gNMDA_d[ptrPos], sizeof(float)*length, kind));
	} else {
		assert(src->gNMDA  != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gNMDA, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gNMDA[ptrPos], &src->gNMDA[ptrPos], sizeof(float)*length, kind));
	}
}

void SNN::copyConductanceGABAa(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	assert(src->gGABAa != NULL);
	if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gGABAa, sizeof(float)*length));
	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gGABAa[ptrPos], &src->gGABAa[ptrPos], sizeof(float)*length, kind));
}

void SNN::copyConductanceGABAb(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	if (isSimulationWithGABAbRise()) {
		assert(src->gGABAb_r != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gGABAb_r, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy(&dest->gGABAb_r[ptrPos],&src->gGABAb_r[ptrPos],sizeof(float)*length,kind) );

		assert(src->gGABAb_d != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gGABAb_d, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy(&dest->gGABAb_d[ptrPos],&src->gGABAb_d[ptrPos],sizeof(float)*length,kind) );
	} else {
		assert(src->gGABAb != NULL);
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gGABAb, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gGABAb[ptrPos], &src->gGABAb[ptrPos], sizeof(float)*length, kind));
	}
}

void SNN::copyConductanceState(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	copyConductanceAMPA( dest, src, kind, allocateMem, grpId);
	copyConductanceNMDA( dest, src, kind, allocateMem, grpId);
	copyConductanceGABAa(dest, src, kind, allocateMem, grpId);
	copyConductanceGABAb(dest, src, kind, allocateMem, grpId);
}

void SNN::copyNeuronState(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	int ptrPos, length, length2;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
		length2 = numN;
	}
	else {
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
		length2 = length;
	}

	assert(length  <= numNReg);
	assert(length2 <= numN);
	assert(length > 0);
	assert(length2 > 0);

	// when allocating we are allocating the memory.. we need to do it completely... to avoid memory fragmentation..
	if(allocateMem)
		assert(grpId == -1);

	// Spike Cnt. Firing...
	if (allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->nSpikeCnt, sizeof(int) * length2));
	CUDA_CHECK_ERRORS(cudaMemcpy( &dest->nSpikeCnt[ptrPos], &src->nSpikeCnt[ptrPos], sizeof(int) * length2, kind));

	if( !allocateMem && groupConfig[grpId].Type & POISSON_NEURON)
		return;

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->recovery, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->recovery[ptrPos], &src->recovery[ptrPos], sizeof(float) * length, kind));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->voltage, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->voltage[ptrPos], &src->voltage[ptrPos], sizeof(float) * length, kind));

	if (sim_with_conductances) {
	    //conductance information
	    copyConductanceState(dest, src, kind, allocateMem, grpId);
	}

	//neuron input current...
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->current, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->current[ptrPos], &src->current[ptrPos], sizeof(float) * length, kind));

	// copying external current needs to be done separately because setExternalCurrent needs to call it, too
	// do it only from host to device
	if (kind==cudaMemcpyHostToDevice) {
		copyExternalCurrent(dest, src, allocateMem, grpId);
	}

	if (sim_with_homeostasis) {
		//Included to enable homeostasis in GPU_MODE.
		// Avg. Firing...
		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->avgFiring, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->avgFiring[ptrPos], &src->avgFiring[ptrPos], sizeof(float)*length, kind));
	}
}

void SNN::copyGroupState(RuntimeData* dest, RuntimeData* src,  cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	if (allocateMem) {
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDA, sizeof(float) * numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HT, sizeof(float) * numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpACh, sizeof(float) * numGroups)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNE, sizeof(float) * numGroups));
	}
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDA, src->grpDA, sizeof(float) * numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HT, src->grp5HT, sizeof(float) * numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpACh, src->grpACh, sizeof(float) * numGroups, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNE, src->grpNE, sizeof(float) * numGroups, kind));

	if (grpId < 0) {
		for (int i = 0; i < numGroups; i++) {
			if (allocateMem) {
				CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDABuffer[i], sizeof(float) * 1000)); 
				CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HTBuffer[i], sizeof(float) * 1000)); 
				CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpAChBuffer[i], sizeof(float) * 1000)); 
				CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNEBuffer[i], sizeof(float) * 1000));
			}
			CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDABuffer[i], src->grpDABuffer[i], sizeof(float) * 1000, kind));
			CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HTBuffer[i], src->grp5HTBuffer[i], sizeof(float) * 1000, kind));
			CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpAChBuffer[i], src->grpAChBuffer[i], sizeof(float) * 1000, kind));
			CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNEBuffer[i], src->grpNEBuffer[i], sizeof(float) * 1000, kind));
		}
	} else {
		if (allocateMem) {
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDABuffer[grpId], sizeof(float) * 1000)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HTBuffer[grpId], sizeof(float) * 1000)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpAChBuffer[grpId], sizeof(float) * 1000)); 
			CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNEBuffer[grpId], sizeof(float) * 1000));
		}
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDABuffer[grpId], src->grpDABuffer[grpId], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HTBuffer[grpId], src->grp5HTBuffer[grpId], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpAChBuffer[grpId], src->grpAChBuffer[grpId], sizeof(float) * 1000, kind));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNEBuffer[grpId], src->grpNEBuffer[grpId], sizeof(float) * 1000, kind));
	}
}

// copy neuron parameters from host to device
void SNN::copyNeuronParametersFromHostToDevice(RuntimeData* dest, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	// cannot use checkDestSrcPtrs here because src pointer would be NULL
	if (dest->allocated && allocateMem) {
		KERNEL_ERROR("GPU Memory already allocated...");
		exitSimulation(1);
	}

	// when allocating we are allocating the memory.. we need to do it completely... to avoid memory fragmentation..
	if (allocateMem) {
		assert(grpId == -1);
		assert(dest->Izh_a == NULL);
		assert(dest->Izh_b == NULL);
		assert(dest->Izh_c == NULL);
		assert(dest->Izh_d == NULL);
	}

	// copy is always from host to device
	cudaMemcpyKind kind = cudaMemcpyHostToDevice;

	if(grpId == -1) {
		ptrPos = 0;
		length = numNReg;
	}
	else {
		ptrPos = groupConfig[grpId].StartN;
		length = groupConfig[grpId].SizeN;
	}

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_a, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_a[ptrPos], &(snnRuntimeData.Izh_a[ptrPos]), sizeof(float) * length, kind));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_b, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_b[ptrPos], &(snnRuntimeData.Izh_b[ptrPos]), sizeof(float) * length, kind));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_c, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_c[ptrPos], &(snnRuntimeData.Izh_c[ptrPos]), sizeof(float) * length, kind));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_d, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_d[ptrPos], &(snnRuntimeData.Izh_d[ptrPos]), sizeof(float) * length, kind));

	if (sim_with_homeostasis) {
		//Included to enable homeostatic plasticity in GPU_MODE. 
		// Base Firing...
		//float baseFiringInv[length];
		float* baseFiringInv = new float[length];
		for(int i=0; i < length; i++) {
			if (snnRuntimeData.baseFiring[i]!=0.0)
				baseFiringInv[i] = 1.0/snnRuntimeData.baseFiring[ptrPos+i];
			else
				baseFiringInv[i] = 0.0;
		}

		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->baseFiringInv, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->baseFiringInv[ptrPos], baseFiringInv, sizeof(float)*length, kind));

		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->baseFiring, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->baseFiring[ptrPos], snnRuntimeData.baseFiring, sizeof(float)*length, kind));

		delete [] baseFiringInv;
	}
}

void SNN::copySTPState(RuntimeData* dest, RuntimeData* src, cudaMemcpyKind kind, bool allocateMem) {
	checkAndSetGPUDevice();

	if(allocateMem) {
		assert(dest->stpu==NULL);
		assert(dest->stpx==NULL);
	} else {
		assert(dest->stpu != NULL);
		assert(dest->stpx != NULL);
	}
	assert(src->stpu != NULL); assert(src->stpx != NULL);

	size_t STP_Pitch;
	size_t widthInBytes = sizeof(float)*networkConfig.numN;

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpu, sizeof(float)*numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpu[0], &src->stpu[0], sizeof(float)*numN, kind));

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpx, sizeof(float)*numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpx[0], &src->stpx[0], sizeof(float)*numN, kind));

	// allocate the stpu and stpx variable
	if (allocateMem)
		CUDA_CHECK_ERRORS( cudaMallocPitch ((void**) &dest->stpu, &networkConfig.STP_Pitch, widthInBytes, networkConfig.maxDelay+1));
	if (allocateMem)
		CUDA_CHECK_ERRORS( cudaMallocPitch ((void**) &dest->stpx, &STP_Pitch, widthInBytes, networkConfig.maxDelay+1));

	assert(networkConfig.STP_Pitch > 0);
	assert(STP_Pitch > 0);				// stp_pitch should be greater than zero
	assert(STP_Pitch == networkConfig.STP_Pitch);	// we want same Pitch for stpu and stpx
	assert(networkConfig.STP_Pitch >= widthInBytes);	// stp_pitch should be greater than the width
	// convert the Pitch value to multiples of float
	assert(networkConfig.STP_Pitch % (sizeof(float)) == 0);
	if (allocateMem)
		networkConfig.STP_Pitch = networkConfig.STP_Pitch/sizeof(float);

//	fprintf(stderr, "STP_Pitch = %ld, STP_witdhInBytes = %d\n", networkConfig.STP_Pitch, widthInBytes);

	float* tmp_stp = new float[networkConfig.numN];
	// copy the already generated values of stpx and stpu to the GPU
	for(int t=0; t<networkConfig.maxDelay+1; t++) {
		if (kind==cudaMemcpyHostToDevice) {
			// stpu in the CPU might be mapped in a specific way. we want to change the format
			// to something that is okay with the GPU STP_U and STP_X variable implementation..
			for (int n=0; n < networkConfig.numN; n++) {
				tmp_stp[n]=snnRuntimeData.stpu[STP_BUF_POS(n,t)];
				assert(tmp_stp[n] == 0.0f);
			}
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpu[t*networkConfig.STP_Pitch], tmp_stp, sizeof(float)*networkConfig.numN, cudaMemcpyHostToDevice));
			for (int n=0; n < networkConfig.numN; n++) {
				tmp_stp[n]=snnRuntimeData.stpx[STP_BUF_POS(n,t)];
				assert(tmp_stp[n] == 1.0f);
			}
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpx[t*networkConfig.STP_Pitch], tmp_stp, sizeof(float)*networkConfig.numN, cudaMemcpyHostToDevice));
		}
		else {
			CUDA_CHECK_ERRORS( cudaMemcpy( tmp_stp, &dest->stpu[t*networkConfig.STP_Pitch], sizeof(float)*networkConfig.numN, cudaMemcpyDeviceToHost));
			for (int n=0; n < networkConfig.numN; n++)
				snnRuntimeData.stpu[STP_BUF_POS(n,t)]=tmp_stp[n];
			CUDA_CHECK_ERRORS( cudaMemcpy( tmp_stp, &dest->stpx[t*networkConfig.STP_Pitch], sizeof(float)*networkConfig.numN, cudaMemcpyDeviceToHost));
			for (int n=0; n < networkConfig.numN; n++)
				snnRuntimeData.stpx[STP_BUF_POS(n,t)]=tmp_stp[n];
		}
	}
	delete [] tmp_stp;
}

void SNN::copyNetworkConfig() {
	checkAndSetGPUDevice();
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(networkConfigGPU, &networkConfig, sizeof(NetworkConfigRT), 0, cudaMemcpyHostToDevice));
}

void SNN::copyWeightState (RuntimeData* dest, RuntimeData* src,  cudaMemcpyKind kind, bool allocateMem, int grpId) {
	checkAndSetGPUDevice();

	unsigned int length_wt, cumPos_syn;

	assert(allocateMem==0);

  // check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	int numCnt = 0;
	if (grpId == -1)
		numCnt = 1;
	else
		numCnt = groupConfig[grpId].SizeN;

	for (int i=0; i < numCnt; i++) {
		if (grpId == -1) {
			length_wt 	= numPreSynNet;
			cumPos_syn  = 0;
		} else {
			int id = groupConfig[grpId].StartN + i;
			length_wt 	= dest->Npre[id];
			cumPos_syn 	= dest->cumulativePre[id];
		}

		assert (cumPos_syn < numPreSynNet || numPreSynNet==0);
		assert (length_wt <= numPreSynNet);

	    //MDR FIXME, allocateMem option is VERY wrong
	    // synaptic information based

	    //if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wt, sizeof(float)*length_wt));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->wt[cumPos_syn], &src->wt[cumPos_syn], sizeof(float)*length_wt,  kind));

	    // firing time for individual synapses
	    //if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*length_wt));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->synSpikeTime[cumPos_syn], &src->synSpikeTime[cumPos_syn], sizeof(int)*length_wt, kind));

		if ((!sim_with_fixedwts) || sim_with_stdp) {
			// synaptic weight derivative
			//if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*length_wt));
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->wtChange[cumPos_syn], &src->wtChange[cumPos_syn], sizeof(float)*length_wt, kind));
		}
	}
}

// allocate necessary memory for the GPU...
void SNN::copyState(RuntimeData* dest, bool allocateMem) {
	checkAndSetGPUDevice();

	assert(numN != 0);

	if (dest->allocated && allocateMem) {
		KERNEL_ERROR("GPU Memory already allocated..");
		return;
	}

	// copyState is unidirectional from host to device
	cudaMemcpyKind kind = cudaMemcpyHostToDevice;

	// synaptic information based
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wt, sizeof(float)*numPreSynNet));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->wt,snnRuntimeData.wt, sizeof(float)*numPreSynNet, kind));

	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->cumConnIdPre, sizeof(short int)*numPreSynNet));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->cumConnIdPre, snnRuntimeData.cumConnIdPre, sizeof(short int)*numPreSynNet, kind));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	// they show up in gpuUpdateLTP() and updateSynapticWeights(), two functions that do not get called if
	// sim_with_fixedwts is set
	if (!sim_with_fixedwts) {
		// synaptic weight derivative
		if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*numPreSynNet));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->wtChange, snnRuntimeData.wtChange, sizeof(float)*numPreSynNet, kind));

		// synaptic weight maximum value
		if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->maxSynWt, sizeof(float)*numPreSynNet));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->maxSynWt, snnRuntimeData.maxSynWt, sizeof(float)*numPreSynNet, kind));
	}

	// firing time for individual synapses
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*numPreSynNet));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->synSpikeTime, snnRuntimeData.synSpikeTime, sizeof(int)*numPreSynNet, kind));
	networkConfig.preSynLength = numPreSynNet;

	if(allocateMem) {
		assert(dest->firingTableD1 == NULL);
		assert(dest->firingTableD2 == NULL);
	}

	// allocate 1ms firing table
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->firingTableD1, sizeof(int)*networkConfig.maxSpikesD1));
	if (networkConfig.maxSpikesD1>0) CUDA_CHECK_ERRORS( cudaMemcpy( dest->firingTableD1, snnRuntimeData.firingTableD1, sizeof(int)*networkConfig.maxSpikesD1, kind));

	// allocate 2+ms firing table
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->firingTableD2, sizeof(int)*networkConfig.maxSpikesD2));
	if (networkConfig.maxSpikesD2>0) CUDA_CHECK_ERRORS( cudaMemcpy( dest->firingTableD2, snnRuntimeData.firingTableD2, sizeof(int)*networkConfig.maxSpikesD2, kind));

	// we don't need this data structure if the network doesn't have any plastic synapses at all
	if (!sim_with_fixedwts) {
		// neuron firing time
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->lastSpikeTime, sizeof(int)*numNReg));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->lastSpikeTime, snnRuntimeData.lastSpikeTime, sizeof(int)*numNReg, kind));
	}

	// grp ids
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->grpIds, sizeof(short int)*numN));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->grpIds, snnRuntimeData.grpIds, sizeof(short int)*numN, kind));
		
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->spikeGenBits, sizeof(int) * (NgenFunc / 32 + 1)));

	// copy the group state to the GPU
	copyGroupState(dest, &snnRuntimeData, cudaMemcpyHostToDevice, allocateMem);


	// copy the neuron state information to the GPU..
	copyNeuronState(dest, &snnRuntimeData, kind, allocateMem);

	// copy neuron parameters Izh_a, Izh_b, etc., baseFiring, baseFiringInv from host vars to device pointer
	copyNeuronParametersFromHostToDevice(dest, allocateMem);

	if (sim_with_stp) {
		copySTPState(dest, &snnRuntimeData, kind, allocateMem);
	}
}

// spikeGeneratorUpdate on GPUs..
void SNN::spikeGeneratorUpdate_GPU() {
	checkAndSetGPUDevice();

	assert(gpuRuntimeData.allocated);

	// this part of the code is useful for poisson spike generator function..
	if((numNPois > 0) && (gpuPoissonRand != NULL)) {
		gpuPoissonRand->generate(numNPois, RNG_rand48::MAX_RANGE);
	}

	// this part of the code is invoked when we use spike generators
	if (NgenFunc) {
		resetCounters();

		assert(snnRuntimeData.spikeGenBits!=NULL);

		// reset the bit status of the spikeGenBits...
		memset(snnRuntimeData.spikeGenBits, 0, sizeof(int)*(NgenFunc/32+1));

		// If time slice has expired, check if new spikes needs to be generated....
		updateSpikeGenerators();

		// fill spikeGenBits accordingly...
		generateSpikes();

		// copy the spikeGenBits from the CPU to the GPU..
		CUDA_CHECK_ERRORS( cudaMemcpy( gpuRuntimeData.spikeGenBits, snnRuntimeData.spikeGenBits, sizeof(int)*(NgenFunc/32+1), cudaMemcpyHostToDevice));
	}
}

void SNN::findFiring_GPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();
		
	assert(gpuRuntimeData.allocated);

	kernel_findFiring <<<gridSize,blkSize >>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR("findFiring kernel failed\n");
	return;
}

// get spikes from GPU SpikeCounter
// grpId cannot be ALL (can only get 1 bufPos at a time)
int* SNN::getSpikeCounter_GPU(int grpId) {
	checkAndSetGPUDevice();

	assert(grpId>=0); assert(grpId<numGroups);

	int bufPos = groupConfig[grpId].spkCntBufPos;
	CUDA_CHECK_ERRORS( cudaMemcpy(spkCntBuf[bufPos],gpuRuntimeData.spkCntBufChild[bufPos],
		groupConfig[grpId].SizeN*sizeof(int),cudaMemcpyDeviceToHost) );

	return spkCntBuf[bufPos];
}

// reset SpikeCounter
// grpId and connectId cannot be ALL (this is handled by the CPU side)
void SNN::resetSpikeCounter_GPU(int grpId) {
	checkAndSetGPUDevice();

	assert(grpId>=0); assert(grpId<numGroups);

	int bufPos = groupConfig[grpId].spkCntBufPos;
	CUDA_CHECK_ERRORS( cudaMemset(gpuRuntimeData.spkCntBufChild[bufPos],0,groupConfig[grpId].SizeN*sizeof(int)) );
}


void SNN::updateTimingTable_GPU() {
	checkAndSetGPUDevice();

	assert(gpuRuntimeData.allocated);

	int blkSize  = 128;
	int gridSize = 64;
	kernel_timingTableUpdate <<<gridSize,blkSize >>> (simTimeMs);
	CUDA_GET_LAST_ERROR("timing Table update kernel failed\n");

	return;
}

void SNN::doCurrentUpdate_GPU() {
	checkAndSetGPUDevice();

	assert(gpuRuntimeData.allocated);

	int blkSize  = 128;
	int gridSize = 64;

	if(maxDelay_ > 1) {
		kernel_doCurrentUpdateD2 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
		CUDA_GET_LAST_ERROR("Kernel execution failed");
	}


	kernel_doCurrentUpdateD1 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
	CUDA_GET_LAST_ERROR("Kernel execution failed");
}

void SNN::doSTPUpdateAndDecayCond_GPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();
		
	assert(gpuRuntimeData.allocated);

	if (sim_with_stp || sim_with_conductances) {
		kernel_STPUpdateAndDecayConductances <<<gridSize, blkSize>>>(simTimeMs, simTimeSec, simTime);
		CUDA_GET_LAST_ERROR("STP update\n");
	}
}

void SNN::initGPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();

	assert(gpuRuntimeData.allocated);

	kernel_init <<< gridSize, blkSize >>> ();
	CUDA_GET_LAST_ERROR("initGPU kernel failed\n");
}

void SNN::printCurrentInfo(FILE* fp) {
	checkAndSetGPUDevice();

	KERNEL_WARN("Calling printCurrentInfo with fp is deprecated");
	// copy neuron input current...
	KERNEL_DEBUG("Total Synaptic updates:");
	CUDA_CHECK_ERRORS( cudaMemcpy( snnRuntimeData.current, gpuRuntimeData.current, sizeof(float)*numNReg, cudaMemcpyDeviceToHost));
		for(int i=0; i < numNReg; i++) {
			if (snnRuntimeData.current[i] != 0.0 ) {
				KERNEL_DEBUG("I[%d] -> %f", i, snnRuntimeData.current[i]);
		}
	}
	fflush(fp);
}

// TODO FIXME there's more...
void SNN::deleteObjects_GPU() {
	checkAndSetGPUDevice();

	// wait for kernels to complete
	CUDA_CHECK_ERRORS(cudaThreadSynchronize());

	// cudaFree all device pointers
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.voltage) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.recovery) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.current) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.extCurrent) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Npre) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Npre_plastic) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Npre_plasticInv) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Npost) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.cumulativePost) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.cumulativePre) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.synSpikeTime) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.wt) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.wtChange) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.maxSynWt) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.nSpikeCnt) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.curSpike) ); // \FIXME exists but never used...
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.firingTableD2) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.firingTableD1) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.avgFiring) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.baseFiring) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.baseFiringInv) );

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpDA) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grp5HT) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpACh) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpNE) );
	for (int i = 0; i < numGroups; i++) {
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpDABuffer[i]) );
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grp5HTBuffer[i]) );
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpAChBuffer[i]) );
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpNEBuffer[i]) );
	}

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.grpIds) );


	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Izh_a) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Izh_b) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Izh_c) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.Izh_d) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gAMPA) );
	if (sim_with_NMDA_rise) {
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gNMDA_r) );
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gNMDA_d) );
	} else {
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gNMDA) );
	}
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gGABAa) );
	if (sim_with_GABAb_rise) {
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gGABAb_r) );
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gGABAb_d) );
	} else {
		CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.gGABAb) );
	}

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.stpu) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.stpx) );

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.cumConnIdPre) );

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.groupIdInfo) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.neuronAllocation) );

	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.postDelayInfo) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.postSynapticIds) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.preSynapticIds) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.I_set) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.poissonFireRate) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.lastSpikeTime) );
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.spikeGenBits) );

	// delete all real-time spike monitors
	CUDA_CHECK_ERRORS( cudaFree(gpuRuntimeData.spkCntBuf));
	for (int i=0; i<numSpkCnt; i++)
		CUDA_CHECK_ERRORS(cudaFree(gpuRuntimeData.spkCntBufChild[i]));

	CUDA_DELETE_TIMER(timer);
}

void SNN::globalStateUpdate_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;

	kernel_globalConductanceUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR("kernel_globalConductanceUpdate failed");

	// update all neuron state (i.e., voltage and recovery)
	kernel_globalStateUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR("Kernel execution failed");

	// update all group state (i.e., concentration of neuronmodulators)
	// currently support 4 x 128 groups
	kernel_globalGroupStateUpdate <<<4, blkSize>>> (simTimeMs);
	CUDA_GET_LAST_ERROR("Kernel execution failed");
}

void SNN::assignPoissonFiringRate_GPU() {
	checkAndSetGPUDevice();

	assert(gpuRuntimeData.poissonFireRate != NULL);
	for (int grpId=0; grpId < numGroups; grpId++) {
		// given group of neurons belong to the poisson group....
		if (groupConfig[grpId].isSpikeGenerator) {
			int nid = groupConfig[grpId].StartN;
			PoissonRate* rate = groupConfig[grpId].RatePtr;

			// if SpikeGen group does not have a Poisson pointer, skip
			if (groupConfig[grpId].spikeGen || rate == NULL)
				continue;

			if (rate->isOnGPU()) {
				// rates allocated on GPU
				CUDA_CHECK_ERRORS( cudaMemcpy( &gpuRuntimeData.poissonFireRate[nid-numNReg], rate->getRatePtrGPU(),
					sizeof(float)*rate->getNumNeurons(), cudaMemcpyDeviceToDevice) );
			} else {
				// rates allocated on CPU
				CUDA_CHECK_ERRORS( cudaMemcpy( &gpuRuntimeData.poissonFireRate[nid-numNReg], rate->getRatePtrCPU(),
					sizeof(float)*rate->getNumNeurons(), cudaMemcpyHostToDevice) );
			}
		}
	}
}

void SNN::doGPUSim() {
	checkAndSetGPUDevice();
	// for all Spike Counters, reset their spike counts to zero if simTime % recordDur == 0
	if (sim_with_spikecounters) {
		checkSpikeCounterRecordDur();
	}
	
	int blkSize  = 128;
	int gridSize = 64;

	doSTPUpdateAndDecayCond_GPU(gridSize, blkSize);

	// \TODO this should probably be in spikeGeneratorUpdate_GPU
	if (spikeRateUpdated) {
		assignPoissonFiringRate_GPU();
		spikeRateUpdated = false;
	}
	spikeGeneratorUpdate_GPU();

	findFiring_GPU(gridSize, blkSize);

	updateTimingTable_GPU();

	doCurrentUpdate_GPU();

	globalStateUpdate_GPU();
}

void SNN::updateFiringTable_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;
	//void* devPtr;

	//kernel_updateWeightsFiring  <<<gridSize, blkSize>>> ();
	kernel_updateFiring_static<<<gridSize, blkSize>>>();

	kernel_updateFiring<<<gridSize, blkSize>>>();
}

void SNN::updateWeights_GPU() {
	checkAndSetGPUDevice();

	assert(sim_in_testing==false);
	assert(sim_with_fixedwts==false);

	int blkSize  = 128;
	int gridSize = 64;

	kernel_updateWeights<<<gridSize, blkSize>>>();
}

__global__ void gpu_resetFiringInformation()
{
	if(threadIdx.x==0 && blockIdx.x==0) {
		for(int i=0; i < ROUNDED_TIMING_COUNT; i++) {
			timeTableD2GPU[i]   = 0;
			timeTableD1GPU[i]   = 0;
		}
		spikeCountD2SecGPU=0;
		spikeCountD1SecGPU=0;
		secD2fireCntTest=0;
		secD1fireCntTest=0;
		spikeCountD2GPU=0;
		spikeCountD1GPU=0;


    //spikeCountAll1Sec=0;//assigned in copyFiringInfo_GPU()
	}

}


void SNN::resetFiringInformation_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;

	gpu_resetFiringInformation<<<gridSize,blkSize>>>();
}


void SNN::copyExternalCurrent(RuntimeData* dest, RuntimeData* src, bool allocateMem, int grpId) {
	// copy external current from CPU to GPU
	int ptrPos, length;

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	}
	else {
		assert(grpId>=0);
		assert(!isPoissonGroup(grpId));
		ptrPos  = groupConfig[grpId].StartN;
		length  = groupConfig[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	KERNEL_DEBUG("copyExternalCurrent: grpId=%d, ptrPos=%d, length=%d, allocate=%s", grpId, ptrPos, length, 
		allocateMem?"y":"n");

	// when allocating we are allocating the memory.. we need to do it completely... to avoid memory fragmentation..
	if(allocateMem)
		assert(grpId == -1);

	if(allocateMem) {
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->extCurrent, sizeof(float) * length));
	}

	CUDA_CHECK_ERRORS(cudaMemcpy(&(dest->extCurrent[ptrPos]), &(src->extCurrent[ptrPos]), sizeof(float) * length, 
		cudaMemcpyHostToDevice));
}


void SNN::copyFiringInfo_GPU()
{
	unsigned int gpuSpikeCountD1Sec, gpuSpikeCountD2Sec;
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &gpuSpikeCountD2Sec, spikeCountD2SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &gpuSpikeCountD1Sec, spikeCountD1SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
	spikeCountSec = gpuSpikeCountD1Sec + gpuSpikeCountD2Sec;
	spikeCountD1Sec  = gpuSpikeCountD1Sec;
	spikeCountD2Sec = gpuSpikeCountD2Sec;
	assert(gpuSpikeCountD1Sec<=maxSpikesD1);
	assert(gpuSpikeCountD2Sec<=maxSpikesD2);
	CUDA_CHECK_ERRORS( cudaMemcpy(snnRuntimeData.firingTableD2, gpuRuntimeData.firingTableD2, sizeof(int)*gpuSpikeCountD2Sec, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpy(snnRuntimeData.firingTableD1, gpuRuntimeData.firingTableD1, sizeof(int)*gpuSpikeCountD1Sec, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(timeTableD2, timeTableD2GPU, sizeof(int)*(1000+maxDelay_+1), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(timeTableD1, timeTableD1GPU, sizeof(int)*(1000+maxDelay_+1), 0, cudaMemcpyDeviceToHost));

	// \TODO: why is this here? The CPU side doesn't have it. And if you can call updateSpikeMonitor() now at any time
	// it might look weird without a time stamp.
//	KERNEL_INFO("Total spikes Multiple Delays=%d, 1Ms Delay=%d", gpuSpikeCountD2Sec,gpuSpikeCountD1Sec);
}


void SNN::allocateNetworkConfig() {
	networkConfig.numN  = numN;
	networkConfig.maxDelay  = maxDelay_;
	networkConfig.numNExcReg = numNExcReg;
	networkConfig.numNInhReg	= numNInhReg;
	networkConfig.numNReg = numNReg;
	assert(numNReg == (numNExcReg + numNInhReg));
	networkConfig.numNPois = numNPois;
	networkConfig.numNExcPois = numNExcPois;		
	networkConfig.numNInhPois = numNInhPois;
	assert(numNPois == (numNExcPois + numNInhPois));
	networkConfig.maxSpikesD2 = maxSpikesD2;
	networkConfig.maxSpikesD1 = maxSpikesD1;
	networkConfig.sim_with_fixedwts = sim_with_fixedwts;
	networkConfig.sim_with_conductances = sim_with_conductances;
	networkConfig.sim_with_homeostasis = sim_with_homeostasis;
	networkConfig.sim_with_stdp = sim_with_stdp;
	networkConfig.sim_with_stp = sim_with_stp;
	networkConfig.sim_in_testing = sim_in_testing;
	networkConfig.numGroups = numGroups;
	networkConfig.numConnections = numConnections;
	networkConfig.stdpScaleFactor = stdpScaleFactor_;
	networkConfig.wtChangeDecay = wtChangeDecay_;

	networkConfig.sim_with_NMDA_rise = sim_with_NMDA_rise;
	networkConfig.sim_with_GABAb_rise = sim_with_GABAb_rise;
	networkConfig.dAMPA = dAMPA;
	networkConfig.rNMDA = rNMDA;
	networkConfig.dNMDA = dNMDA;
	networkConfig.sNMDA = sNMDA;
	networkConfig.dGABAa = dGABAa;
	networkConfig.rGABAb = rGABAb;
	networkConfig.dGABAb = dGABAb;
	networkConfig.sGABAb = sGABAb;

	return;
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
	if (ithGPU_ < 0 || ithGPU_ >= devCount) {
		KERNEL_ERROR("CUDA device[%d] does not exist, please choose from [0,%d]", ithGPU_, devCount - 1);
		exitSimulation(1);
	}

	CUDA_CHECK_ERRORS(cudaGetDeviceProperties(&deviceProp, ithGPU_));
	KERNEL_INFO("  - Use CUDA device[%1d]              = %9s", ithGPU_, deviceProp.name);
	KERNEL_INFO("  - CUDA Compute Capability (CC)    =      %2d.%d\n", deviceProp.major, deviceProp.minor);

    if (deviceProp.major < 2) {
		// Unmark this when CC 1.3 is deprecated
		//KERNEL_ERROR("CARLsim does not support CUDA devices older than CC 2.0");
		//exitSimulation(1);
		KERNEL_WARN("CUDA device with CC 1.3 will be deprecated in a future release");
	}

	CUDA_CHECK_ERRORS(cudaSetDevice(ithGPU_));
	CUDA_DEVICE_RESET();
}

void SNN::checkAndSetGPUDevice() {
	int currentDevice;
	cudaGetDevice(&currentDevice);

	if (currentDevice != ithGPU_) {
		KERNEL_DEBUG("Inconsistent GPU context [%d %d]", currentDevice, ithGPU_);
		cudaSetDevice(ithGPU_);
	}
}

void SNN::copyWeightsGPU(int nid, int src_grp) {
	checkAndSetGPUDevice();

	assert(nid < numNReg);
	unsigned int    cumId   =  snnRuntimeData.cumulativePre[nid];
	float* synWts  = &(snnRuntimeData.wt[cumId]);
	//TODO: NEEDED TO COMMENT THIS FOR CARLSIM 2.1-2.2 FILEMERGE -- KDC
	// assert(cumId >= (nid-numNPois));
	//assert(cumId < numPreSynapses*numN);

	CUDA_CHECK_ERRORS( cudaMemcpy( synWts, &gpuRuntimeData.wt[cumId], sizeof(float)*snnRuntimeData.Npre[nid], cudaMemcpyDeviceToHost));
}

// Allocates required memory and then initialize the GPU
void SNN::allocateSNN_GPU() {
	checkAndSetGPUDevice();
	// \FIXME why is this even here? shouldn't this be checked way earlier? and then in CPU_MODE, too...
	if (maxDelay_ > MAX_SYN_DELAY) {
		KERNEL_ERROR("You are using a synaptic delay (%d) greater than MAX_SYN_DELAY defined in config.h",maxDelay_);
		exitSimulation(1);
	}

	// if we have already allocated the GPU data.. dont do it again...
	if(gpuPoissonRand != NULL) return;

	int gridSize = 64; int blkSize  = 128;

	int numN=0;
	for (int g=0;g<numGroups;g++) {
		numN += groupConfig[g].SizeN;
	}

	// generate the random number for the poisson neuron here...
	if(gpuPoissonRand == NULL) {
		gpuPoissonRand = new RNG_rand48(randSeed_);
	}

	gpuPoissonRand->generate(numNPois, RNG_rand48::MAX_RANGE);

	// initialize SNN::gpuRuntimeData.poissonRandPtr, save the random pointer as poisson generator....
	gpuRuntimeData.poissonRandPtr = (unsigned int*) gpuPoissonRand->get_random_numbers();

	//ensure that we dont do all the above optimizations again		
	assert(snnState == OPTIMIZED_PARTITIONED_SNN);

	// display some memory management info
	size_t avail, total, previous;
	float toGB = std::pow(1024.0f,3);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("GPU Memory Management: (Total %2.3f GB)",(float)(total/toGB));
	KERNEL_INFO("Data\t\t\tSize\t\tTotal Used\tTotal Available");
	KERNEL_INFO("Init:\t\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(total)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// setup memory type of gpu runtime data
	gpuRuntimeData.memType = GPU_MODE;

	allocateNetworkConfig();

	// initialize gpuRuntimeData.neuronAllocation, __device__ loadBufferCount, loadBufferSize
	allocateStaticLoad(blkSize);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Static Load:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB, (float)((total-avail)/toGB),(float)(avail/toGB));
	previous=avail;

	allocateGroupId();
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Group Id:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,(float)((total-avail)/toGB), (float)(avail/toGB));
	previous=avail;

	// this table is useful for quick evaluation of the position of fired neuron
	// given a sequence of bits denoting the firing..
	// initialize __device__ quickSynIdTableGPU[256]
	initQuickSynIdTable();

	
	// initialize (cudaMemset) gpuRuntimeData.I_set, gpuRuntimeData.poissonFireRate
	// initialize (copy from SNN) gpuRuntimeData.Npre, gpuRuntimeData.Npre_plastic, gpuRuntimeData.Npre_plasticInv, gpuRuntimeData.cumulativePre
	// initialize (copy from SNN) gpuRuntimeData.cumulativePost, gpuRuntimeData.Npost, gpuRuntimeData.postDelayInfo
	// initialize (copy from SNN) gpuRuntimeData.postSynapticIds, gpuRuntimeData.preSynapticIds
	// copy data to SNN:networkConfig.numPostSynNet, numPreSynNet
	copyConnections(&gpuRuntimeData,  cudaMemcpyHostToDevice, 1);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Conn Info:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,(float)((total-avail)/toGB), (float)(avail/toGB));
	previous=avail;

	// initialize (copy from SNN) gpuRuntimeData.wt, gpuRuntimeData.wtChange, gpuRuntimeData.maxSynWt, gpuRuntimeData.synSpikeTime
	// initialize (copy from SNN) gpuRuntimeData.firingTableD1, gpuRuntimeData.firingTableD2, gpuRuntimeData.lastSpikeTime
	// initialize (cudaMalloc) gpuRuntimeData.spikeGenBits
	// initialize (copy from snnRuntimeData) gpuRuntimeData.nSpikeCnt, gpuRuntimeData.recovery, gpuRuntimeData.voltage, gpuRuntimeData.current
	// initialize (copy from snnRuntimeData) gpuRuntimeData.gGABAa, gpuRuntimeData.gGABAb, gpuRuntimeData.gAMPA, gpuRuntimeData.gNMDA
	// initialize (copy from SNN) gpuRuntimeData.Izh_a, gpuRuntimeData.Izh_b, gpuRuntimeData.Izh_c, gpuRuntimeData.Izh_d
	// copy data to SNN:networkConfig.preSynLength
	// initialize (copy from SNN) stpu, stpx
	// initialize (copy from SNN) gpuRuntimeData.grpDA, gpuRuntimeData.grp5HT, gpuRuntimeData.grpACh, gpuRuntimeData.grpNE
	copyState(&gpuRuntimeData, 1);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("State Info:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB\n\n",(float)(previous-avail)/toGB,(float)((total-avail)/toGB), (float)(avail/toGB));
	previous=avail;

	// copy Spike Counters
	// 2D arrays are a bit tricky... We can't just copy spkCntBuf over. We still have to use spkCntBuf for
	// cudaMalloc(), but then we need to cudaMemcpy() that array of pointers to the pointer that we got from the
	// first cudaMalloc().
	CUDA_CHECK_ERRORS( cudaMalloc( (void**) &(gpuRuntimeData.spkCntBuf), sizeof(int*)*MAX_GRP_PER_SNN));
	for (int g=0; g<numGroups; g++) {
		if (!groupConfig[g].withSpikeCounter)
			continue; // skip group if it doesn't have a spkMonRT

		int bufPos = groupConfig[g].spkCntBufPos; // retrieve pos in spike buf

		// allocate child pointers
		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &(gpuRuntimeData.spkCntBufChild[bufPos]), sizeof(int)*groupConfig[g].SizeN));

		// copy child pointer to device
		CUDA_CHECK_ERRORS( cudaMemcpy(&(gpuRuntimeData.spkCntBuf[bufPos]), &(gpuRuntimeData.spkCntBufChild[bufPos]),
			sizeof(int*), cudaMemcpyHostToDevice) );

		// copy data
		CUDA_CHECK_ERRORS( cudaMemcpy(gpuRuntimeData.spkCntBufChild[bufPos], spkCntBuf[bufPos],
			sizeof(int)*groupConfig[g].SizeN, cudaMemcpyHostToDevice) );
	}

	// copy relevant pointers and network information to GPU
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(runtimeDataGPU, &gpuRuntimeData, sizeof(RuntimeData), 0, cudaMemcpyHostToDevice));

	// copy data to from SNN:: to NetworkConfigRT SNN::networkConfig
	copyNetworkConfig(); // FIXME: we can change the group properties such as STDP as the network is running.  So, we need a way to updating the GPU when changes are made.


	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(d_mulSynFast, mulSynFast, sizeof(float)*numConnections, 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(d_mulSynSlow, mulSynSlow, sizeof(float)*numConnections, 0, cudaMemcpyHostToDevice));

	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(groupConfigGPU, groupConfig, (networkConfig.numGroups) * sizeof(GroupConfigRT), 0, cudaMemcpyHostToDevice));

	KERNEL_DEBUG("Transfering group settings to GPU:");
	for (int i=0;i<numGroups;i++) {
		KERNEL_DEBUG("Settings for Group %s:", groupInfo[i].Name.c_str());
		
		KERNEL_DEBUG("\tType: %d",(int)groupConfig[i].Type);
		KERNEL_DEBUG("\tSizeN: %d",groupConfig[i].SizeN);
		KERNEL_DEBUG("\tMaxFiringRate: %d",(int)groupConfig[i].MaxFiringRate);
		KERNEL_DEBUG("\tRefractPeriod: %f",groupConfig[i].RefractPeriod);
		KERNEL_DEBUG("\tM: %d",groupConfig[i].numPostSynapses);
		KERNEL_DEBUG("\tPreM: %d",groupConfig[i].numPreSynapses);
		KERNEL_DEBUG("\tspikeGenerator: %d",(int)groupConfig[i].isSpikeGenerator);
		KERNEL_DEBUG("\tFixedInputWts: %d",(int)groupConfig[i].FixedInputWts);
		KERNEL_DEBUG("\tMaxDelay: %d",(int)groupConfig[i].MaxDelay);
		KERNEL_DEBUG("\tWithSTDP: %d",(int)groupConfig[i].WithSTDP);
		if (groupConfig[i].WithSTDP) {
			KERNEL_DEBUG("\t\tE-STDP type: %s",stdpType_string[groupConfig[i].WithESTDPtype]);
			KERNEL_DEBUG("\t\tTAU_PLUS_INV_EXC: %f",groupConfig[i].TAU_PLUS_INV_EXC);
			KERNEL_DEBUG("\t\tTAU_MINUS_INV_EXC: %f",groupConfig[i].TAU_MINUS_INV_EXC);
			KERNEL_DEBUG("\t\tALPHA_PLUS_EXC: %f",groupConfig[i].ALPHA_PLUS_EXC);
			KERNEL_DEBUG("\t\tALPHA_MINUS_EXC: %f",groupConfig[i].ALPHA_MINUS_EXC);
			KERNEL_DEBUG("\t\tI-STDP type: %s",stdpType_string[groupConfig[i].WithISTDPtype]);
			KERNEL_DEBUG("\t\tTAU_PLUS_INV_INB: %f",groupConfig[i].TAU_PLUS_INV_INB);
			KERNEL_DEBUG("\t\tTAU_MINUS_INV_INB: %f",groupConfig[i].TAU_MINUS_INV_INB);
			KERNEL_DEBUG("\t\tALPHA_PLUS_INB: %f",groupConfig[i].ALPHA_PLUS_INB);
			KERNEL_DEBUG("\t\tALPHA_MINUS_INB: %f",groupConfig[i].ALPHA_MINUS_INB);
			KERNEL_DEBUG("\t\tLAMBDA: %f",groupConfig[i].LAMBDA);
			KERNEL_DEBUG("\t\tDELTA: %f",groupConfig[i].DELTA);
			KERNEL_DEBUG("\t\tBETA_LTP: %f",groupConfig[i].BETA_LTP);
			KERNEL_DEBUG("\t\tBETA_LTD: %f",groupConfig[i].BETA_LTD);
		}
		KERNEL_DEBUG("\tWithSTP: %d",(int)groupConfig[i].WithSTP);
		if (groupConfig[i].WithSTP) {
			KERNEL_DEBUG("\t\tSTP_U: %f",groupConfig[i].STP_U);
//				KERNEL_DEBUG("\t\tSTP_tD: %f",groupConfig[i].STP_tD);
//				KERNEL_DEBUG("\t\tSTP_tF: %f",groupConfig[i].STP_tF);
		}
		KERNEL_DEBUG("\tspikeGen: %s",groupConfig[i].spikeGen==NULL?"Is Null":"Is set");
		KERNEL_DEBUG("\tspikeMonitorRT: %s",groupConfig[i].withSpikeCounter?"Is set":"Is Null");
		if (groupConfig[i].withSpikeCounter) {
			KERNEL_DEBUG("\trecordDur: %d",groupConfig[i].spkCntRecordDur);
		} 	
		KERNEL_DEBUG("\tspikeGen: %s",groupConfig[i].spikeGen==NULL?"Is Null":"Is set");
	}

	// allocation of gpu runtime data is done
	gpuRuntimeData.allocated = true;

	// map the timing table to texture.. saves a lot of headache in using shared memory
	void* devPtr;
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD2GPU));
	size_t offset;
	CUDA_CHECK_ERRORS(cudaBindTexture(&offset, timeTableD2GPU_tex, devPtr, sizeof(int) * ROUNDED_TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD2GPU_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));
		
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD1GPU));
	CUDA_CHECK_ERRORS(cudaBindTexture(&offset, timeTableD1GPU_tex, devPtr, sizeof(int) * ROUNDED_TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS(cudaGetSymbolAddress(&devPtr, timeTableD1GPU_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));

	// initialize (memset) gpuRuntimeData.current
	CUDA_CHECK_ERRORS(cudaMemset(gpuRuntimeData.current, 0, sizeof(float) * numNReg));
//	CUDA_CHECK_ERRORS(cudaMemset(gpuRuntimeData.extCurrent, 0, sizeof(float)*numNReg));
//	copyExternalCurrent(&gpuRuntimeData, &snnRuntimeData, true);
	initGPU(gridSize, blkSize);

	snnState = EXECUTABLE_SNN;
}

void SNN::printSimSummary() {
	checkAndSetGPUDevice();

	float etime;
	if(simMode_ == GPU_MODE) {
		stopGPUTiming();
		etime = gpuExecutionTime;
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &spikeCountD2Sec, spikeCountD2SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &spikeCountD1Sec, spikeCountD1SecGPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCountSec = spikeCountD1Sec + spikeCountD2Sec;
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &spikeCountD2, spikeCountD2GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol( &spikeCountD1, spikeCountD1GPU, sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCount      = spikeCountD1 + spikeCountD2;
	}
	else {
		stopCPUTiming();
		etime = cpuExecutionTime;
	}

	KERNEL_INFO("\n");
	KERNEL_INFO("********************      %s Simulation Summary      ***************************",
		simMode_==GPU_MODE?"GPU":"CPU");

	KERNEL_INFO("Network Parameters: \tnumNeurons = %d (numNExcReg:numNInhReg = %2.1f:%2.1f)", 
		numN, 100.0*numNExcReg/numN, 100.0*numNInhReg/numN);
	KERNEL_INFO("\t\t\tnumSynapses = %d", numPostSynNet);
	KERNEL_INFO("\t\t\tmaxDelay = %d", maxDelay_);
	KERNEL_INFO("Simulation Mode:\t%s",sim_with_conductances?"COBA":"CUBA");
	KERNEL_INFO("Random Seed:\t\t%d", randSeed_);
	KERNEL_INFO("Timing:\t\t\tModel Simulation Time = %lld sec", (unsigned long long)simTimeSec);
	KERNEL_INFO("\t\t\tActual Execution Time = %4.2f sec", etime/1000.0);
	KERNEL_INFO("Average Firing Rate:\t2+ms delay = %3.3f Hz", spikeCountD2/(1.0*simTimeSec*numNExcReg));
	KERNEL_INFO("\t\t\t1ms delay = %3.3f Hz", spikeCountD1/(1.0*simTimeSec*numNInhReg));
	KERNEL_INFO("\t\t\tOverall = %3.3f Hz", spikeCount/(1.0*simTimeSec*numN));
	KERNEL_INFO("Overall Firing Count:\t2+ms delay = %d", spikeCountD2);
	KERNEL_INFO("\t\t\t1ms delay = %d", spikeCountD1);
	KERNEL_INFO("\t\t\tTotal = %d", spikeCount);
	KERNEL_INFO("*********************************************************************************\n");
}