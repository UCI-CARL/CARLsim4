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
#include <error_code.h>
#include <cuda_runtime.h>

#if __CUDA3__
// includes, library
#include "cudpp/cudpp.h"
#elif __CUDA5__

#endif

#define ROUNDED_TIMING_COUNT  (((1000+MAX_SynapticDelay+1)+127) & ~(127))  // (1000+maxDelay_) rounded to multiple 128

#define  FIRE_CHUNK_CNT    (512)

#define LOG_WARP_SIZE		(5)
#define WARP_SIZE			(1 << LOG_WARP_SIZE)

#define MAX_NUM_BLOCKS 200
#define LOOP_CNT		10

#define GPU_LTP(t)   (gpuNetInfo.ALPHA_LTP*__expf(-(t)/gpuNetInfo.TAU_LTP))
#define GPU_LTD(t)   (gpuNetInfo.ALPHA_LTD*__expf(-(t)/gpuNetInfo.TAU_LTD))

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
//  timingTableD2[0] always is 0 -- index into firingTableD2
//  timingTableD2[maxDelay_] -- should be the number of spikes "leftover" from the previous second
//	timingTableD2[maxDelay_+1]-timingTableD2[maxDelay_] -- should be the number of spikes in the first ms of the current second
//  timingTableD2[1000+maxDelay_] -- should be the number of spikes in the current second + the leftover spikes.
//
///////////////////////////////////////////////////////////////////

__device__ int  timingTableD2[ROUNDED_TIMING_COUNT];
__device__ int  timingTableD1[ROUNDED_TIMING_COUNT];

__device__ unsigned int	secD2fireCnt;
__device__ unsigned int	secD1fireCnt;
__device__ unsigned int spikeCountD2;
__device__ unsigned int spikeCountD1;

// I believe the following are all just test variables
__device__ unsigned int	secD2fireCntTest;
__device__ unsigned int	secD1fireCntTest;

__device__ __constant__ network_ptr_t		gpuPtrs;
__device__ __constant__ network_info_t		gpuNetInfo;
__device__ __constant__ group_info_t		gpuGrpInfo[MAX_GRP_PER_SNN];
//	__device__ __constant__ noiseGenProperty_t*	gpu_noiseGenGroup;

__device__ __constant__ float				constData[256];

__device__  int	  loadBufferCount; 
__device__  int   loadBufferSize;

float data[256];


texture <int,    1, cudaReadModeElementType>  timingTableD2_tex;
texture <int,    1, cudaReadModeElementType>  timingTableD1_tex;
texture <int,    1, cudaReadModeElementType>  groupIdInfo_tex; // groupIDInfo is allocated using cudaMalloc thus doesn't require an offset when using textures
__device__  int timingTableD1_tex_offset;
__device__  int timingTableD2_tex_offset;

	
__device__ int generatedErrors = 0;
__device__ int	 tmp_val[MAX_NUM_BLOCKS][LOOP_CNT];
__device__ int	 retErrCode=NO_KERNEL_ERRORS;
__device__ float retErrVal[MAX_NUM_BLOCKS][20];

#define INIT_CHECK(enable, src, val)		\
  {						\
    if(enable)					\
      if(threadIdx.x==0) 			\
	src = val;				\
  }


#define ERROR_CHECK_COND(enable, src, val, cond, retVal)	\
  {								\
    if(enable)							\
      if (!(cond)) {						\
	src = val;						\
	return retVal;						\
      }								\
  }

#define ERROR_CHECK_COND_NORETURN(enable, src, val, cond)	\
  {								\
    if(enable)							\
      if (!(cond)) {						\
	src = val;						\
	return;							\
      }								\
  }

#define ERROR_CHECK(enable, src, val)		\
  {						\
    if(enable)					\
      if( blockIdx.x >= MAX_NUM_BLOCKS) {	\
	src = val;				\
	return;					\
      }						\
  }

#define UPDATE_ERROR_CHECK4(src, val1, val2, val3, val4)	\
  {								\
    if (src && (threadIdx.x==0))	 {			\
      retErrCode = src;						\
      if(ENABLE_MORE_CHECK) {					\
	retErrVal[blockIdx.x][0]=0xdead;			\
	retErrVal[blockIdx.x][1]=4;				\
	retErrVal[blockIdx.x][2]=val1;				\
	retErrVal[blockIdx.x][3]=val2;				\
	retErrVal[blockIdx.x][4]=val3;				\
	retErrVal[blockIdx.x][5]=val4;				\
      }								\
    }								\
  }
		
#define	MEASURE_LOADING			0

#define MEASURE_GPU(pos,val)			\
  {						\
    if(MEASURE_LOADING)				\
      if (threadIdx.x==0)			\
	tmp_val[blockIdx.x][pos] += val;	\
  }

// example of the quick synaptic table
// index     cnt
// 0000000 - 0
// 0000001 - 0
// 0000010 - 1
// 0100000 - 5
// 0110000 - 4
int tableQuickSynId[256];
__device__ int  gpu_tableQuickSynId[256];
void initTableQuickSynId()
{
	void* devPtr;
	   
	for(int i=1; i < 256; i++) {
		int cnt=0;
		while(i) {
			if(((i>>cnt)&1)==1) break;
      		cnt++;
      		assert(cnt<=7);
    	}
    	tableQuickSynId[i]=cnt;		 
	}
	   
	cudaGetSymbolAddress(&devPtr, gpu_tableQuickSynId);
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpy( devPtr, tableQuickSynId, sizeof(tableQuickSynId), cudaMemcpyHostToDevice));
}
	
#if defined(__CUDA3__) || defined(__NO_ATOMIC_ADD__)
	static __device__ inline float atomicAddf(float* address, float value)
	{
		float old = value;
		float ret = atomicExch(address, 0.0f);
		float new_old = ret + old;

	  	while ((old = atomicExch(address, new_old)) != 0.0f) {
			new_old = atomicExch(address, 0.0f);
			new_old += old;
		}

		return ret;
	}
#endif

__device__ inline bool isPoissonGroup(short int& grpId, unsigned int& nid)
{
	bool poiss = (gpuGrpInfo[grpId].Type & POISSON_NEURON);

	if (poiss) {
		ERROR_CHECK_COND(TESTING, retErrCode, ERROR_FIRING_2, (nid >= gpuNetInfo.numNReg), 0);
	}

	return poiss;
}

__device__ inline void setFiringBitSynapses(unsigned int& nid, int& syn_id)
{
	uint32_t* tmp_I_set_p = ((uint32_t*)((char*) gpuPtrs.I_set + ((syn_id>>5)*gpuNetInfo.I_setPitch)) + nid);
	int atomicVal = atomicOr(tmp_I_set_p, 1 <<(syn_id%32));
}

__device__ inline uint32_t* getFiringBitGroupPtr(unsigned int& nid, int& synGrpId)
{
	uint32_t* tmp_ptr = (((uint32_t*)((char*) gpuPtrs.I_set + synGrpId*gpuNetInfo.I_setPitch)) + nid);
	return tmp_ptr;
}

__device__ inline uint32_t getSTPBufPos(unsigned int nid, uint32_t t)
{
//  return (((t%STP_BUF_SIZE)*gpuNetInfo.STP_Pitch) + nid);
  return ( (t%(gpuNetInfo.maxDelay+1))*gpuNetInfo.STP_Pitch + nid);
}

__device__ inline int2 getStaticThreadLoad(int& bufPos)
{
	return (gpuPtrs.neuronAllocation[bufPos]);
}

__device__ inline bool getPoissonSpike_GPU (unsigned int& nid)
{
	// Random number value is less than the poisson firing probability
	// if poisson firing probability is say 1.0 then the random poisson ptr
	// will always be less than 1.0 and hence it will continiously fire
	ERROR_CHECK_COND(TESTING, retErrCode, POISSON_COUNT_ERROR_0, (nid >= gpuNetInfo.numNReg), 0);

	return gpuPtrs.poissonRandPtr[nid-gpuNetInfo.numNReg]*(1000.0/RNG_rand48::MAX_RANGE) < gpuPtrs.poissonFireRate[nid-gpuNetInfo.numNReg];
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
		timingTableD2[t+gpuNetInfo.maxDelay+1]  = secD2fireCnt;
		timingTableD1[t+gpuNetInfo.maxDelay+1]  = secD1fireCnt;
   }
   __syncthreads();									     
}

/////////////////////////////////////////////////////////////////////////////////
// Device Kernel Function:  Intialization of the GPU side of the simulator    ///
// KERNEL: This kernel is called after initialization of various parameters   ///
// so that we can reset all required parameters. 			      ///
/////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_init ()
{
	if(threadIdx.x==0 && blockIdx.x==0) {
		for(int i=0; i < ROUNDED_TIMING_COUNT; i++) {
			timingTableD2[i]   = 0;
			timingTableD1[i]   = 0;
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
		short int grpId   	 = STATIC_LOAD_GROUP(threadLoad);

		// errors...
		ERROR_CHECK_COND_NORETURN(TESTING, retErrCode, KERNEL_INIT_ERROR0, (grpId   < gpuNetInfo.numGrp));
		ERROR_CHECK_COND_NORETURN(TESTING, retErrCode, KERNEL_INIT_ERROR1, (lastId  < gpuNetInfo.numN));

		while ((threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {
//				int totCnt = gpuPtrs.Npre[nid];			// total synaptic count
//				int nCum   = gpuPtrs.cumulativePre[nid];	// total pre-synaptic count
			nid=nid+1; // move to the next neuron in the group..
		}
	}
}


// following are initialized in CpuSNNinitGPUparams
__device__ int 	 testFireCnt1;
__device__ int 	 testFireCnt2;
__device__ float testFireCntf1;
__device__ float testFireCntf2;


// Allocation of the group and its id..
void CpuSNN::allocateGroupId() {
	checkAndSetGPUDevice();

	assert (cpu_gpuNetPtrs.groupIdInfo == NULL);
	int3* tempNeuronAllocation = (int3*)malloc(sizeof(int3) * net_Info.numGrp);
	for (int g = 0; g < net_Info.numGrp; g++) {
		int3  threadLoad;
		threadLoad.x = grp_Info[g].StartN;
		threadLoad.y = grp_Info[g].EndN;
		threadLoad.z = g;
		tempNeuronAllocation[g] = threadLoad;
	}

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&cpu_gpuNetPtrs.groupIdInfo, sizeof(int3) * net_Info.numGrp));
	CUDA_CHECK_ERRORS_MACRO(cudaMemcpy(cpu_gpuNetPtrs.groupIdInfo, tempNeuronAllocation, sizeof(int3) * net_Info.numGrp, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS_MACRO(cudaBindTexture(NULL, groupIdInfo_tex, cpu_gpuNetPtrs.groupIdInfo, sizeof(int3) * net_Info.numGrp));

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
int CpuSNN::allocateStaticLoad(int bufSize) {
	checkAndSetGPUDevice();

	// only one thread does the static load table
	int   bufferCnt = 0;
	for (int g = 0; g < net_Info.numGrp; g++) {
		bufferCnt += (int) ceil(1.0 * grp_Info[g].SizeN / bufSize);
		KERNEL_DEBUG("Grp Size = %d, Total Buffer Cnt = %d, Buffer Cnt = %f", grp_Info[g].SizeN, bufferCnt, ceil(1.0*grp_Info[g].SizeN/bufSize));
	}
	assert(bufferCnt > 0);

	int2*  tempNeuronAllocation = (int2*)malloc(sizeof(int2) * bufferCnt);
	KERNEL_DEBUG("STATIC THREAD ALLOCATION");
	KERNEL_DEBUG("------------------------");
	KERNEL_DEBUG("Buffer Size = %d, Buffer Count = %d", bufSize, bufferCnt);

	bufferCnt = 0;
	for (int g = 0; g < net_Info.numGrp; g++) {
		for (int n = grp_Info[g].StartN; n <= grp_Info[g].EndN; n += bufSize) {
			int2  threadLoad;
			// starting neuron id is saved...
			threadLoad.x = n;
			if ((n + bufSize - 1) <= grp_Info[g].EndN)
				// grpID + full size
				threadLoad.y = (g + (bufSize << 16)); // can't support group id > 2^16
			else
				// grpID + left-over size
				threadLoad.y = (g + ((grp_Info[g].EndN - n + 1) << 16)); // can't support group id > 2^16

			// fill the static load distribution here...
			int testg = STATIC_LOAD_GROUP(threadLoad);
			tempNeuronAllocation[bufferCnt] = threadLoad;
			KERNEL_DEBUG("%d. Start=%d, size=%d grpId=%d:%s (SpikeMonId=%d) (GroupMonId=%d) (ConnMonId=%d)",
					bufferCnt, STATIC_LOAD_START(threadLoad),
					STATIC_LOAD_SIZE(threadLoad),
					STATIC_LOAD_GROUP(threadLoad),
					grp_Info2[testg].Name.c_str(),
					grp_Info[testg].SpikeMonitorId,
					grp_Info[testg].GroupMonitorId,
					grp_Info[testg].ConnectionMonitorId);
			bufferCnt++;
		}
	}

	assert(cpu_gpuNetPtrs.allocated==false);
	// Finally writeback the total bufferCnt
	// Note down the buffer size for reference
	KERNEL_DEBUG("GPU loadBufferSize = %d, GPU loadBufferCount = %d", bufSize, bufferCnt);
	CUDA_CHECK_ERRORS_MACRO(cudaMemcpyToSymbol(loadBufferCount, &bufferCnt, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS_MACRO(cudaMemcpyToSymbol(loadBufferSize, &bufSize, sizeof(int), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMalloc((void**) &cpu_gpuNetPtrs.neuronAllocation, sizeof(int2) * bufferCnt));
	CUDA_CHECK_ERRORS_MACRO(cudaMemcpy(cpu_gpuNetPtrs.neuronAllocation, tempNeuronAllocation, sizeof(int2) * bufferCnt, cudaMemcpyHostToDevice));
	free(tempNeuronAllocation);
	return bufferCnt;
}

///////////////////////////////////////////////
// 1. KERNELS used when a specific neuron fires //
///////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// Device local function:      	Update the STP Variables		  ///
// update the STPU and STPX variable after firing			      ///
/////////////////////////////////////////////////////////////////////////////////

// update the spike-dependent part of du/dt and dx/dt
__device__ void firingUpdateSTP (unsigned int& nid, int& simTime, short int&  grpId) {
	// we need to retrieve the STP values from the right buffer position (right before vs. right after the spike)
	int ind_plus  = getSTPBufPos(nid, simTime);
	int ind_minus = getSTPBufPos(nid, (simTime-1));

	// at this point, stpu[ind_plus] has already been assigned, and the decay applied
	// so add the spike-dependent part to that
	// du/dt = -u/tau_F + U * (1-u^-) * \delta(t-t_{spk})
	gpuPtrs.stpu[ind_plus] += gpuGrpInfo[grpId].STP_U*(1.0-gpuPtrs.stpu[ind_minus]);

	// dx/dt = (1-x)/tau_D - u^+ * x^- * \delta(t-t_{spk})
	gpuPtrs.stpx[ind_plus] -= gpuPtrs.stpu[ind_plus]*gpuPtrs.stpx[ind_minus];
}

__device__ void setFiringBit(unsigned int& nid)
{
	gpuPtrs.neuronFiring[nid] |= 0x1;
}

__device__ void measureFiringLoad(volatile int& fireCnt, volatile int& fireCntD1)
{
	if (0==threadIdx.x) {
		MEASURE_GPU(1, (fireCnt-fireCntD1));
		MEASURE_GPU(2, fireCntD1);
	}
	__syncthreads();
}

__device__ void resetFiredNeuron(unsigned int& nid, short int & grpId, int& simTime)
{
	// \FIXME \TODO: convert this to use coalesced access by grouping into a
	// single 16 byte access. This might improve bandwidth performance
	// This is fully uncoalsced access...need to convert to coalsced access..
	gpuPtrs.voltage[nid] = gpuPtrs.Izh_c[nid];
	gpuPtrs.recovery[nid] += gpuPtrs.Izh_d[nid];
	if (gpuGrpInfo[grpId].WithSTDP)
		gpuPtrs.lastSpikeTime[nid] = simTime;
	
	if (gpuNetInfo.sim_with_homeostasis) {
		// with homeostasis flag can be used here.
		gpuPtrs.avgFiring[nid] += 1000/(gpuGrpInfo[grpId].avgTimeScale*1000);
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
	if(secD2fireCntTest>gpuNetInfo.maxSpikesD2) {
		blkErrCode = NEW_FIRE_UPDATE_OVERFLOW_ERROR2;
		return;
	}
	else if(secD1fireCntTest>gpuNetInfo.maxSpikesD1) {
		blkErrCode = NEW_FIRE_UPDATE_OVERFLOW_ERROR1;
		return;
	}
	blkErrCode = 0;

	// get a distinct counter to store firing info
	// into the firing table
	cntD2 = atomicAdd(&secD2fireCnt, fireCntD2);
	cntD1 = atomicAdd(&secD1fireCnt, fireCntD1);

	MEASURE_GPU(1, fireCntD2);
	MEASURE_GPU(2, fireCntD1);
}

// update the firing table...
__device__ void updateFiringTable(unsigned int& nid, short int& grpId, volatile unsigned int& cntD2, volatile unsigned int& cntD1)
{
	int pos;
	if (gpuGrpInfo[grpId].MaxDelay == 1) {
		// this group has a delay of only 1
		pos = atomicAdd((int*)&cntD1, 1);
		gpuPtrs.firingTableD1[pos]  = SET_FIRING_TABLE(nid, grpId);
	} else {
		// all other groups is dumped here 
		pos = atomicAdd((int*)&cntD2, 1);
		gpuPtrs.firingTableD2[pos]  = SET_FIRING_TABLE(nid, grpId);
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

	UPDATE_ERROR_CHECK4(blkErrCode,cntD2,fireCnt,cntD1,fireCntD1);

	// if we overflow the spike buffer space that is available,
	// then we return with an error here...
	if (blkErrCode)
		return blkErrCode;

	for (int i=threadIdx.x; i < fireCnt; i+=(blockDim.x)) {

		// Read the firing id from the local table.....
		unsigned int nid  = fireTablePtr[i];

		//storeTestSpikedNeurons(gpuPtrs.testVar2, fireCnt, fireTablePtr, NULL);
		ERROR_CHECK_COND (ENABLE_MORE_CHECK, blkErrCode, ERROR_FIRING_3, (nid < gpuNetInfo.numN), blkErrCode);

		// set the LSB bit indicating the current neuron has
		if(TESTING) {
			setFiringBit(nid);
		}
		updateFiringTable(nid, fireGrpId[i], cntD2, cntD1);

		if (gpuGrpInfo[fireGrpId[i]].WithSTP)
			firingUpdateSTP(nid, simTime, fireGrpId[i]);

		// keep track of number spikes per neuron
		gpuPtrs.nSpikeCnt[nid]++;

		// only neurons would do the remaining settings...
		// pure poisson generators will return without changing anything else..
		if (IS_REGULAR_NEURON(nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois))
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
		//grp_Info seems to be accessible. -- KDC
		for( int grpId=_startGrp; grpId < _endGrp; grpId++) {
			int startN = gpuGrpInfo[grpId].StartN;
			int endN   = gpuGrpInfo[grpId].EndN+1;
			for (int i=startN; i < endN; i++){
				gpuPtrs.nSpikeCnt[i] = 0;
			}
		}
	}
}

// wrapper to call resetSpikeCnt
void CpuSNN::resetSpikeCnt_GPU(int _startGrp, int _endGrp) {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;
	gpu_resetSpikeCnt<<<gridSize,blkSize>>> (_startGrp,_endGrp);
}

__device__ void findGrpId_GPU(unsigned int& nid, int& grpId)
{
	for (int g=0; g < gpuNetInfo.numGrp; g++) {
		//uint3 groupIdInfo = {1, 1, 1};
		int startN  = tex1Dfetch (groupIdInfo_tex, g*3);
		int endN    = tex1Dfetch (groupIdInfo_tex, g*3+1);
		// printf("%d:%s s=%d e=%d\n", g, grp_Info2[g].Name.c_str(), grp_Info[g].StartN, grp_Info[g].EndN);
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
		if (gpuGrpInfo[grpId].WithSTDP) { // MDR, FIXME this probably will cause more thread divergence than need be...
			int  nid   = fireTablePtr[pos];
			unsigned int  end_p = gpuPtrs.cumulativePre[nid] + gpuPtrs.Npre_plastic[nid];
			for(unsigned int p  = gpuPtrs.cumulativePre[nid] + threadIdx.x % LTP_GROUPING_SZ;
					p < end_p;
					p+=LTP_GROUPING_SZ) {
				int stdp_tDiff = (simTime - gpuPtrs.synSpikeTime[p]);
				if ((stdp_tDiff > 0) && ((stdp_tDiff*gpuGrpInfo[grpId].TAU_LTP_INV)<25)) {
					gpuPtrs.wtChange[p] += STDP(stdp_tDiff, gpuGrpInfo[grpId].ALPHA_LTP, gpuGrpInfo[grpId].TAU_LTP_INV);
				}
			}
		}
	}
	__syncthreads();
}

__device__ inline int assertionFiringParam(short int& grpId, int& lastId)
{
	ERROR_CHECK_COND(TESTING, retErrCode, ERROR_FIRING_0, (grpId   < gpuNetInfo.numGrp), 0);
	ERROR_CHECK_COND(TESTING, retErrCode, ERROR_FIRING_1, (lastId  < gpuNetInfo.numN), 0);
	return 0;
}

__device__ inline bool getSpikeGenBit_GPU (unsigned int& nidPos)
{
	const int nidBitPos = nidPos%32;
	const int nidIndex  = nidPos/32;
	return ((gpuPtrs.spikeGenBits[nidIndex]>>nidBitPos)&0x1);
}

// setSpikeGenBit for given neuron and group..
void CpuSNN::setSpikeGenBit_GPU(unsigned int nid, int grp) {
	checkAndSetGPUDevice();

	unsigned int nidPos    = (nid - grp_Info[grp].StartN + grp_Info[grp].Noffset);
	unsigned int nidBitPos = nidPos%32;
	unsigned int nidIndex  = nidPos/32;

	assert(nidIndex < (NgenFunc/32+1));

	cpuNetPtrs.spikeGenBits[nidIndex] |= (1 << nidBitPos);
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

	// Ignore this unless you are doing real debugging gpu code...
	INIT_CHECK(ENABLE_MORE_CHECK, retErrCode, NO_KERNEL_ERRORS);

	const int totBuffers=loadBufferCount;

	__syncthreads();

	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		unsigned int  nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);
		short int grpId   = STATIC_LOAD_GROUP(threadLoad);
		bool needToWrite = false;	// used by all neuron to indicate firing condition
		int  fireId      = 0;

		assertionFiringParam(grpId, lastId);

		// threadId is valid and lies within the lastId.....
		if ((threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {
			// Simple poisson spiker uses the poisson firing probability
			// to detect whether it has fired or not....
			if( isPoissonGroup(grpId, nid) ) {
				if(gpuGrpInfo[grpId].spikeGen) {
					unsigned int  offset      = nid-gpuGrpInfo[grpId].StartN+gpuGrpInfo[grpId].Noffset;
					needToWrite = getSpikeGenBit_GPU(offset);
				}
				else
					needToWrite = getPoissonSpike_GPU(nid);
			}
			else {
				if (gpuPtrs.voltage[nid] >= 30.0) {
					needToWrite = true;
					if (gpuGrpInfo[grpId].withSpikeCounter) {
						int bufPos = gpuGrpInfo[grpId].spkCntBufPos;
						int bufNeur = nid-gpuGrpInfo[grpId].StartN;
						gpuPtrs.spkCntBuf[bufPos][bufNeur]++;
						/* NOTE: When compiling with -arch sm_13 the following warning might pop up:
						 * src/snn_gpu.cu(740): Warning:Cannot tell what pointer points to, assuming global memory space
						 * This warning can be safely ignored.
						 * It is a limitation of pre-Fermi cards, and "assuming global memory space" is correct in
						 * 99.999% of cases. If you target a Fermi device, you shouldn't compile with sm_13... So if
						 * you switch that to, e.g., sm_20, the warning will go away. For more information see:
						 * http://stackoverflow.com/questions/5212875/resolving-thrust-cuda-warnings-cannot-tell-what-pointer-points-to
						 */
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

				if (gpuGrpInfo[grpId].MaxDelay == 1)
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
				if (gpuNetInfo.sim_with_stdp)
					gpu_updateLTP (fireTable, fireGrpId, fireCnt, simTime);

				// reset counters
				if (0==threadIdx.x) {
					fireCntD1  = 0;
					fireCnt   = 0;
					fireCntTest = 0;
					MEASURE_GPU(0, 1);
				}
			}
		}
	}

	__syncthreads();

	// few more fired neurons are left. we update their firing state here..
	if (fireCnt) {
		int retCode = newFireUpdate(fireTable, fireGrpId, fireCnt, fireCntD1, simTime);
		if (retCode != 0) return;

		if (gpuNetInfo.sim_with_stdp)
			gpu_updateLTP(fireTable, fireGrpId, fireCnt, simTime);
		MEASURE_GPU(0, 1);
	}
}

//******************************** UPDATE CONDUCTANCES AND TOTAL SYNAPTIC CURRENT EVERY TIME STEP *****************************

#define LOG_CURRENT_GROUP 5
#define CURRENT_GROUP	  (1 << LOG_CURRENT_GROUP)

__device__ inline int assertConductanceStates()
{
	// error checking done here
	ERROR_CHECK_COND(ENABLE_MORE_CHECK, retErrCode, GLOBAL_CONDUCTANCE_ERROR_0, (blockIdx.x < MAX_NUM_BLOCKS), 0);
	return 0;
}

// Based on the bitvector used for indicating the presence of spike
// the global conductance values are updated..
__global__ void kernel_globalConductanceUpdate (int t, int sec, int simTime) {
	__shared__ int sh_tableQuickSynId[256];

	// Table for quick access
	for(int i=0; i < 256; i+=blockDim.x){
		if((i+threadIdx.x) < 256){
			sh_tableQuickSynId[i+threadIdx.x]=gpu_tableQuickSynId[i+threadIdx.x];
		}
	}

	__syncthreads();

	const int totBuffers=loadBufferCount;
	for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
		// KILLME !!! This can be further optimized ....
		// instead of reading each neuron group separately .....
		// read a whole buffer and use the result ......
		int2 threadLoad  = getStaticThreadLoad(bufPos);
		unsigned int  post_nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);

		// do some error checking...
		assertConductanceStates();

		if ((threadIdx.x < lastId) && (IS_REGULAR_NEURON(post_nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois))) {
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
			int   lmt      		 = gpuPtrs.Npre[post_nid];
			unsigned int cum_pos = gpuPtrs.cumulativePre[post_nid];

			// find the total current to this neuron...
			for(int j=0; (lmt)&&(j <= ((lmt-1)>>LOG_CURRENT_GROUP)); j++) {
				// because of malloc2D operation we are using pitch, post_nid, j to get
				// actual position of the input current....
				// int* tmp_I_set_p = ((int*)((char*)gpuPtrs.I_set + j * gpuNetInfo.I_setPitch) + post_nid);
				uint32_t* tmp_I_set_p  = getFiringBitGroupPtr(post_nid, j);

				ERROR_CHECK_COND_NORETURN(ENABLE_MORE_CHECK, retErrCode, GLOBAL_CONDUCTANCE_ERROR_1, (tmp_I_set_p!=0));

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
					int wt_i = sh_tableQuickSynId[k];
					int wtId = (j*32 + cnt*8 + wt_i);

					post_info_t pre_Id   = gpuPtrs.preSynapticIds[cum_pos + wtId];
					uint8_t  pre_grpId  = GET_CONN_GRP_ID(pre_Id);
					uint32_t  pre_nid  = GET_CONN_NEURON_ID(pre_Id);
					char type = gpuGrpInfo[pre_grpId].Type;

					// load the synaptic weight for the wtId'th input
					float change = gpuPtrs.wt[cum_pos + wtId];

					// Adjust the weight according to STP scaling
					if(gpuGrpInfo[pre_grpId].WithSTP) {
						int tD = 0; // \FIXME find delay
						// \FIXME I think pre_nid needs to be adjusted for the delay
						int ind_minus = getSTPBufPos(pre_nid,(simTime-tD-1)); // \FIXME should be adjusted for delay
						int ind_plus = getSTPBufPos(pre_nid,(simTime-tD));
						// dI/dt = -I/tau_S + A * u^+ * x^- * \delta(t-t_{spk})
						change *= gpuGrpInfo[pre_grpId].STP_A * gpuPtrs.stpx[ind_minus] * gpuPtrs.stpu[ind_plus];
					}

					if (gpuNetInfo.sim_with_conductances) {
						short int connId = gpuPtrs.cumConnIdPre[cum_pos+wtId];
						if (type & TARGET_AMPA)
							AMPA_sum += change*gpuPtrs.mulSynFast[connId];
						if (type & TARGET_NMDA) {
							if (gpuNetInfo.sim_with_NMDA_rise) {
								NMDA_r_sum += change*gpuPtrs.mulSynSlow[connId]*gpuNetInfo.sNMDA;
								NMDA_d_sum += change*gpuPtrs.mulSynSlow[connId]*gpuNetInfo.sNMDA;
							} else {
								NMDA_sum += change*gpuPtrs.mulSynSlow[connId];
							}
						}
						if (type & TARGET_GABAa)
							GABAa_sum += change*gpuPtrs.mulSynFast[connId];	// wt should be negative for GABAa and GABAb
						if (type & TARGET_GABAb) {						// but that is dealt with below
							if (gpuNetInfo.sim_with_GABAb_rise) {
								GABAb_r_sum += change*gpuPtrs.mulSynSlow[connId]*gpuNetInfo.sGABAb;
								GABAb_d_sum += change*gpuPtrs.mulSynSlow[connId]*gpuNetInfo.sGABAb;
							} else {
								GABAb_sum += change*gpuPtrs.mulSynSlow[connId];
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

			if (gpuNetInfo.sim_with_conductances) {
				// don't add mulSynFast/mulSynSlow here, because they depend on the exact pre<->post connection, not
				// just post_nid
				gpuPtrs.gAMPA[post_nid]        += AMPA_sum;
				gpuPtrs.gGABAa[post_nid]       -= GABAa_sum; // wt should be negative for GABAa and GABAb
				if (gpuNetInfo.sim_with_NMDA_rise) {
					gpuPtrs.gNMDA_r[post_nid]  += NMDA_r_sum;
					gpuPtrs.gNMDA_d[post_nid]  += NMDA_d_sum;
				} else {
					gpuPtrs.gNMDA[post_nid]    += NMDA_sum;
				}
				if (gpuNetInfo.sim_with_GABAb_rise) {
					gpuPtrs.gGABAb_r[post_nid] -= GABAb_r_sum;
					gpuPtrs.gGABAb_d[post_nid] -= GABAb_d_sum;
				} else {
					gpuPtrs.gGABAb[post_nid]   -= GABAb_sum;
				}
			}
			else {
				gpuPtrs.current[post_nid] += AMPA_sum;
			}
		}
	}
}

//************************ UPDATE GLOBAL STATE EVERY TIME STEP *******************************************************//

__device__ void updateNeuronState(unsigned int& nid, int& grpId) {
	double v = gpuPtrs.voltage[nid];
	double u = gpuPtrs.recovery[nid];
	double I_sum, NMDAtmp;
	double gNMDA, gGABAb;

	// loop that allows smaller integration time step for v's and u's
	for (int c=0; c<COND_INTEGRATION_SCALE; c++) {
		I_sum = 0.0;
		if (gpuNetInfo.sim_with_conductances) {
			NMDAtmp = (v+80.0)*(v+80.0)/60.0/60.0;
			gNMDA = (gpuNetInfo.sim_with_NMDA_rise) ? (gpuPtrs.gNMDA_d[nid]-gpuPtrs.gNMDA_r[nid]) : gpuPtrs.gNMDA[nid];
			gGABAb = (gpuNetInfo.sim_with_GABAb_rise) ? (gpuPtrs.gGABAb_d[nid]-gpuPtrs.gGABAb_r[nid]) : gpuPtrs.gGABAb[nid];
			I_sum = -(   gpuPtrs.gAMPA[nid]*(v-0.0)
					   + gNMDA*NMDAtmp/(1.0+NMDAtmp)*(v-0.0)
					   + gpuPtrs.gGABAa[nid]*(v+70.0)
					   + gGABAb*(v+90.0)
					 );
		}
		else {
			I_sum = gpuPtrs.current[nid];
		}

		// update vpos and upos for the current neuron
		v += ((0.04*v+5.0)*v+140.0-u+I_sum+gpuPtrs.extCurrent[nid])/COND_INTEGRATION_SCALE;
		if (v > 30.0) { 
			v = 30.0; // break the loop but evaluate u[i]
			c=COND_INTEGRATION_SCALE;
		}
		if (v < -90.0)
			v = -90.0;
		u += (gpuPtrs.Izh_a[nid]*(gpuPtrs.Izh_b[nid]*v-u)/COND_INTEGRATION_SCALE);
	}
	if(gpuNetInfo.sim_with_conductances) {
		gpuPtrs.current[nid] = I_sum;
	} else {
		// current must be reset here for CUBA and not kernel_STPUpdateAndDecayConductances
		gpuPtrs.current[nid] = 0.0;
	}
	gpuPtrs.voltage[nid] = v;
	gpuPtrs.recovery[nid] = u;
}

__device__ inline int assertGlobalStates()
{
	// error checking done here
	ERROR_CHECK_COND(ENABLE_MORE_CHECK, retErrCode, GLOBAL_STATE_ERROR_0, (blockIdx.x < MAX_NUM_BLOCKS), 0);
	return 0;
}

// homeostasis in GPU_MODE
__device__ inline void updateHomeoStaticState(unsigned int& pos, int& grpId)
{
	// here the homeostasis adjustment
	if (gpuGrpInfo[grpId].WithHomeostasis)
		gpuPtrs.avgFiring[pos] *= (gpuGrpInfo[grpId].avgTimeScale_decay);
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
		unsigned int nid = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId = STATIC_LOAD_SIZE(threadLoad);
		int  grpId = STATIC_LOAD_GROUP(threadLoad);

		// do some error checking...
		assertGlobalStates();

		if ((threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {

			if (IS_REGULAR_NEURON(nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois)) {

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
__global__ void kernel_globalGroupStateUpdate (int t, int sec, int simTime)
{
	// update group state
	int grpIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (grpIdx < gpuNetInfo.numGrp) {
		// decay dopamine concentration
		if (gpuPtrs.grpDA[grpIdx] > gpuGrpInfo[grpIdx].baseDP)
			gpuPtrs.grpDA[grpIdx] *= gpuGrpInfo[grpIdx].decayDP;
	}
}

//******************************** UPDATE STP STATE  EVERY TIME STEP **********************************************

///////////////////////////////////////////////////////////
// simple assertions for STP values..
///////////////////////////////////////////////////////////
__device__ inline int assertSTPConditions()
{
	// error checking done here
	ERROR_CHECK_COND(ENABLE_MORE_CHECK, retErrCode, STP_ERROR, (blockDim.x < MAX_NUM_BLOCKS), 0);
	return 0;
}


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
		unsigned int nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
		int  lastId      = STATIC_LOAD_SIZE(threadLoad);
		uint32_t  grpId  = STATIC_LOAD_GROUP(threadLoad);


    // update the conductane parameter of the current neron
		if (gpuNetInfo.sim_with_conductances && IS_REGULAR_NEURON(nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois)) {
			gpuPtrs.gAMPA[nid]   *=  gpuNetInfo.dAMPA;
			if (gpuNetInfo.sim_with_NMDA_rise) {
				gpuPtrs.gNMDA_r[nid]   *=  gpuNetInfo.rNMDA;
				gpuPtrs.gNMDA_d[nid]   *=  gpuNetInfo.dNMDA;
			} else {
				gpuPtrs.gNMDA[nid]   *=  gpuNetInfo.dNMDA;
			}
			gpuPtrs.gGABAa[nid]  *=  gpuNetInfo.dGABAa;
			if (gpuNetInfo.sim_with_GABAb_rise) {
				gpuPtrs.gGABAb_r[nid]  *=  gpuNetInfo.rGABAb;
				gpuPtrs.gGABAb_d[nid]  *=  gpuNetInfo.dGABAb;
			} else {
				gpuPtrs.gGABAb[nid]  *=  gpuNetInfo.dGABAb;
			}
		}

    // check various STP asserts here....
		assertSTPConditions();

		if (gpuGrpInfo[grpId].WithSTP && (threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {
			int ind_plus  = getSTPBufPos(nid, simTime);
			int ind_minus = getSTPBufPos(nid, (simTime-1)); // \FIXME sure?
				gpuPtrs.stpu[ind_plus] = gpuPtrs.stpu[ind_minus]*(1.0-gpuGrpInfo[grpId].STP_tau_u_inv);
				gpuPtrs.stpx[ind_plus] = gpuPtrs.stpx[ind_minus] + (1.0-gpuPtrs.stpx[ind_minus])*gpuGrpInfo[grpId].STP_tau_x_inv;
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
	float t_wt = gpuPtrs.wt[jpos];
	float t_wtChange = gpuPtrs.wtChange[jpos];
	float t_effectiveWtChange = gpuNetInfo.stdpScaleFactor * t_wtChange;
	float t_maxWt = gpuPtrs.maxSynWt[jpos];

	switch (gpuGrpInfo[grpId].WithSTDPtype) {
	case STANDARD:
		if (gpuGrpInfo[grpId].WithHomeostasis) {
			// this factor is slow
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1+fabs(diff_firing)*50);
		} else {
			t_wt += t_effectiveWtChange;
		}
		break;
	case DA_MOD:
		if (gpuGrpInfo[grpId].WithHomeostasis) {
			t_effectiveWtChange = gpuPtrs.grpDA[grpId] * t_effectiveWtChange;
			t_wt += (diff_firing*t_wt*homeostasisScale + t_effectiveWtChange) * baseFiring * avgTimeScaleInv / (1+fabs(diff_firing)*50);
		} else {
			t_wt += gpuPtrs.grpDA[grpId] * t_effectiveWtChange;
		}
		break;
	case UNKNOWN_STDP:
	default:
		// we shouldn't even be here if !WithSTDP
		break;
	}
	
	// It's user's choice to decay weight change or not
	// see setWeightAndWeightChangeUpdate()
	t_wtChange *= gpuNetInfo.wtChangeDecay;

	if (t_wt > t_maxWt) t_wt = t_maxWt;
	if (t_wt < 0)  	  t_wt = 0.0f;

	gpuPtrs.wt[jpos] = t_wt;
	gpuPtrs.wtChange[jpos] = t_wtChange;
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
			if (gpuGrpInfo[grpId].WithHomeostasis) {
				homeostasisScale = gpuGrpInfo[grpId].homeostasisScale;
				avgTimeScaleInv = gpuGrpInfo[grpId].avgTimeScaleInv;
			} else {
				homeostasisScale = 0;
				avgTimeScaleInv = 1.0;
			}
		}

		__syncthreads();

	// the weights are fixed for this group.. so dont make any changes on
	// the weight and continue to the next set of neurons...
	if (gpuGrpInfo[grpId].FixedInputWts)
		continue;

		int nid=(threadIdx.x/UPWTS_CLUSTERING_SZ) + startId;
		// update the synaptic weights from the synaptic weight derivatives
		for(; nid < startId+lastId; nid+=grpNCnt) {
			int Npre_plastic = gpuPtrs.Npre_plastic[nid];
			unsigned int cumulativePre = gpuPtrs.cumulativePre[nid];
			float diff_firing  = 0.0;
			float baseFiring = 0;

			if (gpuGrpInfo[grpId].WithHomeostasis) {
				diff_firing  = (1.0-gpuPtrs.avgFiring[nid]*gpuPtrs.baseFiringInv[nid]);
				baseFiring = gpuPtrs.baseFiring[nid];
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
	for(int p=timingTableD2[999],k=0;
		p<timingTableD2[999+gpuNetInfo.maxDelay+1];
		p+=gnthreads,k+=gnthreads) {
		if((p+threadIdx.x)<timingTableD2[999+gpuNetInfo.maxDelay+1])
			gpuPtrs.firingTableD2[k+threadIdx.x]=gpuPtrs.firingTableD2[p+threadIdx.x];
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
	int maxDelay_ = gpuNetInfo.maxDelay;
	// reset the firing table so that we have the firing information
	// for the last maxDelay_ time steps to be used for the next cycle of the simulation
	if(blockIdx.x==0) {
		for(int i=threadIdx.x; i < maxDelay_; i+=blockDim.x) {
			// use i+1 instead of just i because timingTableD2[0] should always be 0
			timingTableD2[i+1] = timingTableD2[1000+i+1]-timingTableD2[1000];
			timingTableD1[i+1] = timingTableD1[1000+i+1]-timingTableD1[1000];
		}
	}

	__syncthreads();

	// reset various counters for the firing information
	if((blockIdx.x==0)&&(threadIdx.x==0)) {
		timingTableD1[gpuNetInfo.maxDelay]  = 0;
		spikeCountD2	+= secD2fireCnt;
		spikeCountD1	+= secD1fireCnt;
		secD2fireCnt	= timingTableD2[gpuNetInfo.maxDelay];
		secD2fireCntTest	= timingTableD2[gpuNetInfo.maxDelay];
		secD1fireCnt	= 0;
		secD1fireCntTest = 0;
	}
}

/// THIS KERNEL IS USED BY BLOCK_CONFIG_VERSION
__global__ void kernel_updateFiring2()
{
	// reset various counters for the firing information
	if((blockIdx.x==0)&&(threadIdx.x==0)) {
		// timingTableD1[gpuNetInfo.maxDelay]  = 0;
		spikeCountD2	+= secD2fireCnt;
		spikeCountD1	+= secD1fireCnt;
		secD2fireCnt	= 0; //timingTableD2[gpuNetInfo.maxDelay];
		secD1fireCnt	= 0;
		secD2fireCntTest = 0; //timingTableD2[gpuNetInfo.maxDelay];
		secD1fireCntTest = 0;
	}
}

//****************************** GENERATE POST-SYNAPTIC CURRENT EVERY TIME-STEP  ****************************
__device__ int generatePostSynapticSpike(int& simTime, int& firingId, int& myDelayIndex, volatile int& offset, bool unitDelay)
{
	int errCode = false;

	// get the post synaptic information for specific delay
	post_info_t post_info = gpuPtrs.postSynapticIds[offset+myDelayIndex];

	// get neuron id
	unsigned int nid = GET_CONN_NEURON_ID(post_info);//(post_info&POST_SYN_NEURON_MASK);

	// get synaptic id
	int syn_id = GET_CONN_SYN_ID(post_info); //(post_info>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK;

	// get the actual position of the synapses and other variables...
	unsigned int pos_ns = gpuPtrs.cumulativePre[nid] + syn_id;

	short int pre_grpId = GET_FIRING_TABLE_GID(firingId);
	//int pre_nid = GET_FIRING_TABLE_NID(firingId);

	// Error MNJ... this should have been from nid.. not firingId...
	// int  nid  = GET_FIRING_TABLE_NID(firingId);
//	int    post_grpId;		// STP uses pre_grpId, STDP used post_grpId...
//	findGrpId_GPU(nid, post_grpId);
	int post_grpId = gpuPtrs.grpIds[nid];

	if(post_grpId == -1)
		return CURRENT_UPDATE_ERROR4;

	if(ENABLE_MORE_CHECK) {
		if (nid >= gpuNetInfo.numN)       	 errCode = (CURRENT_UPDATE_ERROR1|unitDelay);
		if (syn_id >= gpuPtrs.Npre[nid]) errCode = (CURRENT_UPDATE_ERROR2|unitDelay);
		if (errCode) return errCode;
	}

	// Got one spike from dopaminergic neuron, increase dopamine concentration in the target area
	if (gpuGrpInfo[pre_grpId].Type & TARGET_DA) {
#if defined(__CUDA3__) || defined(__NO_ATOMIC_ADD__)
		atomicAddf(&(gpuPtrs.grpDA[post_grpId]), 0.04f);
#else
		atomicAdd(&(gpuPtrs.grpDA[post_grpId]), 0.04f);
#endif
	}

	setFiringBitSynapses(nid, syn_id);

	gpuPtrs.synSpikeTime[pos_ns] = simTime;		  //uncoalesced access

	if (gpuGrpInfo[post_grpId].WithSTDP)  {
		int stdp_tDiff = simTime-gpuPtrs.lastSpikeTime[nid];
		if ((stdp_tDiff >= 0) && ((stdp_tDiff*gpuGrpInfo[post_grpId].TAU_LTD_INV)<25)) {
			gpuPtrs.wtChange[pos_ns] -= STDP( stdp_tDiff, gpuGrpInfo[post_grpId].ALPHA_LTD, gpuGrpInfo[post_grpId].TAU_LTD_INV); // uncoalesced access
		}
	}
	
	return errCode;
}

__device__ void CHECK_DELAY_ERROR (int& t_pos, volatile int& sh_blkErrCode) {
	if(ENABLE_MORE_CHECK) {
		if (!((t_pos+(int)gpuNetInfo.maxDelay) >= 0)) {
			int i=2;
			sh_blkErrCode = CURRENT_UPDATE_ERROR3;
			retErrVal[blockIdx.x][0] = 0xdead;
			/* retErrVal[blockIdx.x][i++] = simTimeMs;
			retErrVal[blockIdx.x][i++] = t_pos;
			retErrVal[blockIdx.x][i++] = fPos;
			retErrVal[blockIdx.x][i++] = tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.maxDelay-1 + timingTableD2_tex_offset);
			retErrVal[blockIdx.x][i++] = tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.maxDelay + timingTableD2_tex_offset);*/
			retErrVal[blockIdx.x][1]   = i;
		}
	}
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

	if(ENABLE_MORE_CHECK) {			
		if(threadIdx.x<=0) 
			sh_blkErrCode = 0;		
	}

	// this variable is used to record the
	// number of updates done by different blocks
	if(threadIdx.x<=0)   {
		sh_NeuronCnt = 0;
	}

	__syncthreads();

	// stores the number of fired neurons at time t
	int k      = secD2fireCnt - 1;

	// stores the number of fired neurons at time (t - maxDelay_)
	int k_end  = tex1Dfetch (timingTableD2_tex, simTimeMs+1+timingTableD2_tex_offset);

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
				int val = gpuPtrs.firingTableD2[fPos];
				int nid = GET_FIRING_TABLE_NID(val);

				// find the time of firing based on the firing number fPos
				while ( !((fPos >= tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.maxDelay+timingTableD2_tex_offset)) 
					&& (fPos <  tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.maxDelay+1+timingTableD2_tex_offset)))) {
					t_pos = t_pos - 1;
					CHECK_DELAY_ERROR(t_pos, sh_blkErrCode);
				}

				// find the time difference between firing of the neuron and the current time
				int tD  = simTimeMs - t_pos;

				// find the various delay parameters for neuron 'nid', with a delay of 'tD'
				//sh_axonDelay[threadIdx.x]	 = tD;
				int tPos = (gpuNetInfo.maxDelay+1)*nid+tD;
				sh_firingId[threadIdx.x]	 	 = val;
				sh_neuronOffsetTable[threadIdx.x]= gpuPtrs.cumulativePost[nid];
				sh_delayLength[threadIdx.x]      = gpuPtrs.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]  = gpuPtrs.postDelayInfo[tPos].delay_index_start;

				// This is to indicate that the current thread
				// has a valid delay parameter for post-synaptic firing generation
				atomicAdd((int*)&sh_NeuronCnt,1);
			}
		}

		__syncthreads();

		if(ENABLE_MORE_CHECK)
			if (sh_blkErrCode) break;

		// if cnt is zero than no more neurons need to generate
		// post-synaptic firing, then we break the loop.
		int cnt = sh_NeuronCnt;
		updateCnt += cnt;
		if(cnt==0)  break;

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

				if(ENABLE_MORE_CHECK) {
					if (sh_blkErrCode) break;
				}

				delId += WARP_SIZE;
			}
		} //(for all excitory neurons in table)

		__syncthreads();

		if(threadIdx.x==0)  
			sh_NeuronCnt = 0;

		if(ENABLE_MORE_CHECK) 
			if(sh_blkErrCode) break;

		k = k - (gridDim.x*EXCIT_READ_CHUNK_SZ);

		__syncthreads();
	}

	__syncthreads();

	MEASURE_GPU(3, updateCnt);

	if(ENABLE_MORE_CHECK)	
		if (sh_blkErrCode) {
			retErrCode = sh_blkErrCode;
			return;
		}	
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

	if(ENABLE_MORE_CHECK) {			
		if(threadIdx.x<=0) 
			sh_blkErrCode = 0;		
	}

	// load the time table for neuron firing
	int computedNeurons = 0;
	if (0==threadIdx.x) {
		sh_timing = timingTableD1[simTimeMs+gpuNetInfo.maxDelay]; // ??? check check ???
	}
	__syncthreads();

	int kPos = sh_timing + (blockIdx.x*numSwarps);

	__syncthreads();

	// Do as long as we have some valid neuron
	while((kPos >=0)&&(kPos < secD1fireCnt)) {
		int fPos = -1;
		// a group of threads loads the delay information
		if (threadIdx.x < numSwarps) {
			sh_neuronOffsetTable[threadIdx.x] = -1;
			fPos = kPos + threadIdx.x;

			// find the neuron nid and also delay information from fPos
			if((fPos>=0)&&(fPos < secD1fireCnt)) {
				atomicAdd((int*)&sh_NeuronCnt,1);
				int val  = gpuPtrs.firingTableD1[fPos];
				int nid  = GET_FIRING_TABLE_NID(val);
				int tPos = (gpuNetInfo.maxDelay+1)*nid;
				sh_firingId[threadIdx.x] 	 	 = val;
				sh_neuronOffsetTable[threadIdx.x]= gpuPtrs.cumulativePost[nid];
				sh_delayLength[threadIdx.x]      = gpuPtrs.postDelayInfo[tPos].delay_length;
				sh_delayIndexStart[threadIdx.x]  = gpuPtrs.postDelayInfo[tPos].delay_index_start;
			}
		}

		__syncthreads();

		// useful to measure the load balance for each block..
		if(threadIdx.x==0)  computedNeurons += sh_NeuronCnt;

		// no more fired neuron from table... we just break from loop
		if (sh_NeuronCnt==0)	break;

		__syncthreads();

		int offset = sh_neuronOffsetTable[swarpId];

		if (threadIdx.x == 0) 
			sh_NeuronCnt = 0;

		if (offset>=0) {
			int delId=threadIdSwarp;

			while(delId < sh_delayLength[swarpId]) {

				int delIndex = (sh_delayIndexStart[swarpId]+delId);

				sh_blkErrCode = generatePostSynapticSpike(simTime,
						sh_firingId[swarpId],				// presynaptic nid
						delIndex,							// delayIndex
						sh_neuronOffsetTable[swarpId], 		// offset
						true);								// true for unit delay connection..

				if(ENABLE_MORE_CHECK) {
					if (sh_blkErrCode) break;
				}

				delId += WARP_SIZE;
			}
		}

		__syncthreads();

		if(ENABLE_MORE_CHECK)
			if(sh_blkErrCode)  break;		

		kPos = kPos + (gridDim.x*numSwarps);
	}

	MEASURE_GPU(4, computedNeurons);

	if(ENABLE_MORE_CHECK) {
	   if (sh_blkErrCode) {
		     retErrCode = sh_blkErrCode;
		     return;
	   }
	}
}

float errVal[MAX_NUM_BLOCKS][20];
/**********************************************************************************************/
// helper functions..
// check what is the errors...that has happened during
// the previous iteration. This part is useful in debug or test mode
// the simulator kernel sets appropriate errors  and returns some important values in the 
// array "retErrVal".
/**********************************************************************************************/
int CpuSNN::checkErrors(std::string calledKernel, int numBlocks) {
	checkAndSetGPUDevice();

	int errCode = NO_KERNEL_ERRORS;
	#if(!ENABLE_MORE_CHECK)
	return errCode;
	#else
	void* devPtr;
	int errCnt  = 0;

	cudaThreadSynchronize();
	cudaGetSymbolAddress(&devPtr, retErrCode);
	CUDA_CHECK_ERRORS( cudaMemcpy(&errCode, devPtr, sizeof(int), cudaMemcpyDeviceToHost));

	cudaGetSymbolAddress(&devPtr, generatedErrors);
	CUDA_CHECK_ERRORS( cudaMemcpy(&errCnt, devPtr, sizeof(int), cudaMemcpyDeviceToHost));

	if (errCode != NO_KERNEL_ERRORS) {

		KERNEL_ERROR("(Error in Kernel <<< %s >>> , RETURN ERROR CODE = %x, total Errors = %d", calledKernel.c_str(), errCode, errCnt);

		cudaGetSymbolAddress(&devPtr, CUDA_CONVERT_SYMBOL(retErrVal));
		CUDA_CHECK_ERRORS( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));

		for(int j=0; j < errCnt; j++) {
			if (1) /*errVal[j][0]==0xdead) */ {
				KERNEL_ERROR("Block: %d, Err code = %x, Total err val is %f", j, (int)errVal[j][0]	, errVal[j][1]);
				for(int i=2; i < (errVal[j][1]); i++) {
					KERNEL_ERROR("ErrVal[%d][%d] = %f", j, i, errVal[j][i]);
					getchar();
				}
			}
		}

		errCode = NO_KERNEL_ERRORS;
		cudaGetSymbolAddress(&devPtr, retErrCode);
		CUDA_CHECK_ERRORS( cudaMemcpy(devPtr, &errCode, sizeof(int), cudaMemcpyHostToDevice));

		cudaGetSymbolAddress(&devPtr, generatedErrors);
		CUDA_CHECK_ERRORS( cudaMemcpy(devPtr, &errCnt, sizeof(int), cudaMemcpyHostToDevice));

	}

	fflush(fpErr_);
	fflush(fpOut_);
	return errCode;
#endif
}

void CpuSNN::copyPostConnectionInfo(network_ptr_t* dest, int allocateMem) {
	checkAndSetGPUDevice();

	assert(dest->memType == GPU_MODE);
	if (allocateMem) {
		assert(dest->allocated == false);
	}
	else {
		assert(dest->allocated == true);
	}
	assert(doneReorganization == true);

	// beginning position for the post-synaptic information
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePost, sizeof(cumulativePost[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->cumulativePost, cumulativePost, sizeof(int) * numN, cudaMemcpyHostToDevice));

	// number of postsynaptic connections
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npost, sizeof(Npost[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy( dest->Npost, Npost, sizeof(Npost[0]) * numN, cudaMemcpyHostToDevice));

	// static specific mapping and actual post-synaptic delay metric
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postDelayInfo, sizeof(postDelayInfo[0]) * numN * (maxDelay_ + 1)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->postDelayInfo, postDelayInfo, sizeof(postDelayInfo[0]) * numN * (maxDelay_ + 1), cudaMemcpyHostToDevice));

	// actual post synaptic connection information...
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->postSynapticIds, sizeof(postSynapticIds[0]) * (postSynCnt + 10)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->postSynapticIds, postSynapticIds, sizeof(postSynapticIds[0]) * (postSynCnt + 10), cudaMemcpyHostToDevice)); //FIXME: why +10 post synapses
	net_Info.postSynCnt = postSynCnt;

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->preSynapticIds, sizeof(preSynapticIds[0]) * (preSynCnt + 10)));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->preSynapticIds, preSynapticIds, sizeof(preSynapticIds[0]) * (preSynCnt + 10), cudaMemcpyHostToDevice)); //FIXME: why +10 post synapses
	net_Info.preSynCnt = preSynCnt;
}

void CpuSNN::copyConnections(network_ptr_t* dest, int kind, int allocateMem) {
	checkAndSetGPUDevice();
	// void* devPtr;
	// allocateMem memory only if destination memory is not allocated !!!
	assert(allocateMem && (dest->allocated != 1));
	if(kind == cudaMemcpyHostToDevice) {
		assert(dest->memType == GPU_MODE);
	}
	else {
		assert(dest->memType == CPU_MODE);
	}

	net_Info.I_setLength = ceil(((numPreSynapses) / 32.0));
	if(allocateMem)
		cudaMallocPitch((void**)&dest->I_set, &net_Info.I_setPitch, sizeof(int) * numNReg, net_Info.I_setLength);
	assert(net_Info.I_setPitch > 0);
	CUDA_CHECK_ERRORS(cudaMemset(dest->I_set, 0, net_Info.I_setPitch * net_Info.I_setLength));

	// connection synaptic lengths and cumulative lengths...
	if(allocateMem) 
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre, sizeof(dest->Npre[0]) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre, Npre, sizeof(dest->Npre[0]) * numN, cudaMemcpyHostToDevice));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	if (!sim_with_fixedwts) {
		// presyn excitatory connections
		if(allocateMem) 
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plastic, sizeof(dest->Npre_plastic[0]) * numN));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre_plastic, Npre_plastic, sizeof(dest->Npre_plastic[0]) * numN, cudaMemcpyHostToDevice));

		float* Npre_plasticInv = new float[numN];
		for (int i = 0; i < numN; i++)
			Npre_plasticInv[i] = 1.0 / Npre_plastic[i];

		if(allocateMem)
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Npre_plasticInv, sizeof(dest->Npre_plasticInv[0]) * numN));
		CUDA_CHECK_ERRORS(cudaMemcpy(dest->Npre_plasticInv, Npre_plasticInv, sizeof(dest->Npre_plasticInv[0]) * numN, cudaMemcpyHostToDevice));

		delete[] Npre_plasticInv;
	}
		
	// beginning position for the pre-synaptic information
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->cumulativePre, sizeof(int) * numN));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->cumulativePre, cumulativePre, sizeof(int) * numN, cudaMemcpyHostToDevice));

  // allocate randomPtr.... containing the firing information for the random firing neurons...
  // I'm not sure this is implemented. -- KDC
  //		if(allocateMem)  CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->randId, sizeof(int)*numN));
  //		net_Info.numRandNeurons=numRandNeurons;

  // copy the properties of the noise generator here.....
  // I'm not sure this is implemented. -- KDC
  //		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->noiseGenProp, sizeof(noiseGenProperty_t)*numNoise));
  //		CUDA_CHECK_ERRORS( cudaMemcpy( dest->noiseGenProp, &noiseGenGroup[0], sizeof(noiseGenProperty_t)*numNoise, cudaMemcpyHostToDevice));

	// allocate the poisson neuron poissonFireRate
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->poissonFireRate, sizeof(dest->poissonFireRate[0]) * numNPois));
	CUDA_CHECK_ERRORS(cudaMemset(dest->poissonFireRate, 0, sizeof(dest->poissonFireRate[0]) * numNPois));

	// neuron firing recently or not...
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->neuronFiring, sizeof(int) * numN));
	CUDA_CHECK_ERRORS(cudaMemset(dest->neuronFiring, 0, sizeof(int) * numN));

	copyPostConnectionInfo(dest, allocateMem);
}

void CpuSNN::checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
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

void CpuSNN::copyFiringStateFromGPU (int grpId) {
	checkAndSetGPUDevice();

	int ptrPos, length;

	if(grpId == -1) {
		ptrPos  = 0;
		length = numN;
	}
	else {
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
	}

	assert(length>0 && length <= numN);

	network_ptr_t* dest = &cpuNetPtrs;
	network_ptr_t* src  = &cpu_gpuNetPtrs;

	// Spike Cnt. Firing...
	CUDA_CHECK_ERRORS( cudaMemcpy(&dest->nSpikeCnt[ptrPos], &src->nSpikeCnt[ptrPos], sizeof(int)*length, 
		cudaMemcpyDeviceToHost) );
}

void CpuSNN::copyConductanceAMPA(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	//conductance information
	assert(src->gAMPA  != NULL);
	if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gAMPA, sizeof(float)*length));
	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gAMPA[ptrPos], &src->gAMPA[ptrPos], sizeof(float)*length, kind));
}

void CpuSNN::copyConductanceNMDA(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
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

void CpuSNN::copyConductanceGABAa(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
	}
	assert(length  <= numNReg);
	assert(length > 0);

	assert(src->gGABAa != NULL);
	if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->gGABAa, sizeof(float)*length));
	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->gGABAa[ptrPos], &src->gGABAa[ptrPos], sizeof(float)*length, kind));
}

void CpuSNN::copyConductanceGABAb(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	int ptrPos, length;

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	} else {
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
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

void CpuSNN::copyConductanceState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	assert(isSimulationWithCOBA());

	// check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	copyConductanceAMPA( dest, src, kind, allocateMem, grpId);
	copyConductanceNMDA( dest, src, kind, allocateMem, grpId);
	copyConductanceGABAa(dest, src, kind, allocateMem, grpId);
	copyConductanceGABAb(dest, src, kind, allocateMem, grpId);
}

void CpuSNN::copyNeuronState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId) {
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
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
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

	if( !allocateMem && grp_Info[grpId].Type & POISSON_NEURON)
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

void CpuSNN::copyGroupState(network_ptr_t* dest, network_ptr_t* src,  cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	if (allocateMem) {
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpDA, sizeof(float) * numGrp)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grp5HT, sizeof(float) * numGrp)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpACh, sizeof(float) * numGrp)); 
		CUDA_CHECK_ERRORS(cudaMalloc((void**) &dest->grpNE, sizeof(float) * numGrp));
	}
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpDA, src->grpDA, sizeof(float) * numGrp, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grp5HT, src->grp5HT, sizeof(float) * numGrp, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpACh, src->grpACh, sizeof(float) * numGrp, kind));
	CUDA_CHECK_ERRORS(cudaMemcpy(dest->grpNE, src->grpNE, sizeof(float) * numGrp, kind));
}

void CpuSNN::copyNeuronParameters(network_ptr_t* dest, int kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	int ptrPos, length;

	if (dest->allocated && allocateMem) {
		KERNEL_ERROR("GPU Memory already allocated...");
		return;
	}

	// for neuron parameter the copy is one-directional...
	assert(kind == cudaMemcpyHostToDevice);

	if(grpId == -1) {
		ptrPos = 0;
		length = numNReg;
	}
	else {
		ptrPos = grp_Info[grpId].StartN;
		length = grp_Info[grpId].SizeN;
	}

	// when allocating we are allocating the memory.. we need to do it completely... to avoid memory fragmentation..
	if(allocateMem)
		assert(grpId == -1);

	//neuron information...
	if(allocateMem)
		assert(dest->Izh_a == NULL);

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_a, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_a[ptrPos], &Izh_a[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_b, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_b[ptrPos], &Izh_b[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_c, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_c[ptrPos], &Izh_c[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->Izh_d, sizeof(float) * length));
	CUDA_CHECK_ERRORS(cudaMemcpy(&dest->Izh_d[ptrPos], &Izh_d[ptrPos], sizeof(float) * length, cudaMemcpyHostToDevice));

	if (sim_with_homeostasis) {
		//Included to enable homeostatic plasticity in GPU_MODE. 
		// Base Firing...
		//float baseFiringInv[length];
		float* baseFiringInv = new float[length];
		for(int i=0; i < length; i++) {
			if (baseFiring[i]!=0.0)
				baseFiringInv[i] = 1.0/baseFiring[ptrPos+i];
			else
				baseFiringInv[i] = 0.0;
		}

		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->baseFiringInv, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->baseFiringInv[ptrPos], baseFiringInv, sizeof(float)*length, cudaMemcpyHostToDevice));

		if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->baseFiring, sizeof(float)*length));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->baseFiring[ptrPos], baseFiring, sizeof(float)*length, cudaMemcpyHostToDevice));

		delete [] baseFiringInv;
	}
}

void CpuSNN::copySTPState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem) {
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
	size_t widthInBytes = sizeof(float)*net_Info.numN;

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpu, sizeof(float)*numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpu[0], &src->stpu[0], sizeof(float)*numN, kind));

//	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->stpx, sizeof(float)*numN));
//	CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpx[0], &src->stpx[0], sizeof(float)*numN, kind));

	// allocate the stpu and stpx variable
	if (allocateMem)
		CUDA_CHECK_ERRORS( cudaMallocPitch ((void**) &dest->stpu, &net_Info.STP_Pitch, widthInBytes, net_Info.maxDelay+1));
	if (allocateMem)
		CUDA_CHECK_ERRORS( cudaMallocPitch ((void**) &dest->stpx, &STP_Pitch, widthInBytes, net_Info.maxDelay+1));

	assert(net_Info.STP_Pitch > 0);
	assert(STP_Pitch > 0);				// stp_pitch should be greater than zero
	assert(STP_Pitch == net_Info.STP_Pitch);	// we want same Pitch for stpu and stpx
	assert(net_Info.STP_Pitch >= widthInBytes);	// stp_pitch should be greater than the width
	// convert the Pitch value to multiples of float
	assert(net_Info.STP_Pitch % (sizeof(float)) == 0);
	if (allocateMem)
		net_Info.STP_Pitch = net_Info.STP_Pitch/sizeof(float);

//	fprintf(stderr, "STP_Pitch = %ld, STP_witdhInBytes = %d\n", net_Info.STP_Pitch, widthInBytes);

	float* tmp_stp = new float[net_Info.numN];
	// copy the already generated values of stpx and stpu to the GPU
	for(int t=0; t<net_Info.maxDelay+1; t++) {
		if (kind==cudaMemcpyHostToDevice) {
			// stpu in the CPU might be mapped in a specific way. we want to change the format
			// to something that is okay with the GPU STP_U and STP_X variable implementation..
			for (int n=0; n < net_Info.numN; n++) {
				tmp_stp[n]=stpu[STP_BUF_POS(n,t)];
				assert(tmp_stp[n] == 0.0f);
			}
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpu[t*net_Info.STP_Pitch], tmp_stp, sizeof(float)*net_Info.numN, cudaMemcpyHostToDevice));
			for (int n=0; n < net_Info.numN; n++) {
				tmp_stp[n]=stpx[STP_BUF_POS(n,t)];
				assert(tmp_stp[n] == 1.0f);
			}
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->stpx[t*net_Info.STP_Pitch], tmp_stp, sizeof(float)*net_Info.numN, cudaMemcpyHostToDevice));
		}
		else {
			CUDA_CHECK_ERRORS( cudaMemcpy( tmp_stp, &dest->stpu[t*net_Info.STP_Pitch], sizeof(float)*net_Info.numN, cudaMemcpyDeviceToHost));
			for (int n=0; n < net_Info.numN; n++)
				stpu[STP_BUF_POS(n,t)]=tmp_stp[n];
			CUDA_CHECK_ERRORS( cudaMemcpy( tmp_stp, &dest->stpx[t*net_Info.STP_Pitch], sizeof(float)*net_Info.numN, cudaMemcpyDeviceToHost));
			for (int n=0; n < net_Info.numN; n++)
				stpx[STP_BUF_POS(n,t)]=tmp_stp[n];
		}
	}
	delete [] tmp_stp;
}

void CpuSNN::copyWeightState (network_ptr_t* dest, network_ptr_t* src,  cudaMemcpyKind kind, int allocateMem, int grpId) {
	checkAndSetGPUDevice();

	unsigned int length_wt, cumPos_syn;

	assert(allocateMem==0);

  // check that the destination pointer is properly allocated..
	checkDestSrcPtrs(dest, src, kind, allocateMem, grpId);

	int numCnt = 0;
	if (grpId == -1)
		numCnt = 1;
	else
		numCnt = grp_Info[grpId].SizeN;

	for (int i=0; i < numCnt; i++) {
		if (grpId == -1) {
			length_wt 	= preSynCnt;
			cumPos_syn  = 0;
			assert(0);
		} else {
			int id = grp_Info[grpId].StartN + i;
			length_wt 	= dest->Npre[id];
			cumPos_syn 	= dest->cumulativePre[id];
		}

		assert (cumPos_syn < preSynCnt);
		assert (length_wt <= preSynCnt);

    //MDR FIXME, allocateMem option is VERY wrong
    // synaptic information based

    //if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wt, sizeof(float)*length_wt));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->wt[cumPos_syn], &src->wt[cumPos_syn], sizeof(float)*length_wt,  kind));

    // firing time for individual synapses
    //if(allocateMem) CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*length_wt));
		CUDA_CHECK_ERRORS( cudaMemcpy( &dest->synSpikeTime[cumPos_syn], &src->synSpikeTime[cumPos_syn], sizeof(int)*length_wt, kind));

    // added this to CARLsim 2.1 - 2.2 file merge -- KDC
		if ((!sim_with_fixedwts) || (sim_with_stdp)) {
			// synaptic weight derivative
			//if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*length_wt));
			CUDA_CHECK_ERRORS( cudaMemcpy( &dest->wtChange[cumPos_syn], &src->wtChange[cumPos_syn], sizeof(float)*length_wt, kind));
		}
	}
}

// allocate necessary memory for the GPU...
void CpuSNN::copyState(network_ptr_t* dest, int allocateMem) {
	checkAndSetGPUDevice();

	assert(numN != 0);
	assert(preSynCnt !=0);

	if (dest->allocated && allocateMem) {
		KERNEL_ERROR("GPU Memory already allocated..");
		return;
	}

	// copyState is unidirectional from host to device
	cudaMemcpyKind kind = cudaMemcpyHostToDevice;

	// synaptic information based
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wt, sizeof(float)*preSynCnt));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->wt, wt, sizeof(float)*preSynCnt, kind));

	// conductance ratios
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->mulSynFast, sizeof(float)*numConnections));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->mulSynFast, mulSynFast, sizeof(float)*numConnections, kind));
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->mulSynSlow, sizeof(float)*numConnections));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->mulSynSlow, mulSynSlow, sizeof(float)*numConnections, kind));
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->cumConnIdPre, sizeof(short int)*preSynCnt));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->cumConnIdPre, cumConnIdPre, sizeof(short int)*preSynCnt, kind));

	// we don't need these data structures if the network doesn't have any plastic synapses at all
	// they show up in gpuUpdateLTP() and updateSynapticWeights(), two functions that do not get called if
	// sim_with_fixedwts is set
	if (!sim_with_fixedwts) {
		// synaptic weight derivative
		if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*preSynCnt));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->wtChange, wtChange, sizeof(float)*preSynCnt, kind));

		// synaptic weight maximum value
		if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->maxSynWt, sizeof(float)*preSynCnt));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->maxSynWt, maxSynWt, sizeof(float)*preSynCnt, kind));
	}

	// firing time for individual synapses
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*preSynCnt));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->synSpikeTime, synSpikeTime, sizeof(int)*preSynCnt, kind));
	net_Info.preSynLength = preSynCnt;

	if(allocateMem) {
		assert(dest->firingTableD1 == NULL);
		assert(dest->firingTableD2 == NULL);
	}

	// allocate 1ms firing table
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->firingTableD1, sizeof(int)*net_Info.maxSpikesD1));
	if (net_Info.maxSpikesD1>0) CUDA_CHECK_ERRORS( cudaMemcpy( dest->firingTableD1, firingTableD1, sizeof(int)*net_Info.maxSpikesD1, kind));

	// allocate 2+ms firing table
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->firingTableD2, sizeof(int)*net_Info.maxSpikesD2));
	if (net_Info.maxSpikesD2>0) CUDA_CHECK_ERRORS( cudaMemcpy( dest->firingTableD2, firingTableD2, sizeof(int)*net_Info.maxSpikesD2, kind));

	// we don't need this data structure if the network doesn't have any plastic synapses at all
	if (!sim_with_fixedwts) {
		// neuron firing time..
		if(allocateMem)     CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->lastSpikeTime, sizeof(int)*numNReg));
		CUDA_CHECK_ERRORS( cudaMemcpy( dest->lastSpikeTime, lastSpikeTime, sizeof(int)*numNReg, kind));
	}

	// grp ids
	if(allocateMem)		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &dest->grpIds, sizeof(short int)*numN));
	CUDA_CHECK_ERRORS( cudaMemcpy( dest->grpIds, grpIds, sizeof(short int)*numN, kind));
		
	if(allocateMem)
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&dest->spikeGenBits, sizeof(int) * (NgenFunc / 32 + 1)));

	// copy the group state to the GPU
	copyGroupState(dest, &cpuNetPtrs, cudaMemcpyHostToDevice, allocateMem);


	// copy the neuron state information to the GPU..
	copyNeuronState(dest, &cpuNetPtrs, kind, allocateMem);

	copyNeuronParameters(dest, kind, allocateMem);

	if (sim_with_stp) {
		copySTPState(dest, &cpuNetPtrs, kind, allocateMem);
	}
}

// spikeGeneratorUpdate on GPUs..
void CpuSNN::spikeGeneratorUpdate_GPU() {
	checkAndSetGPUDevice();

	assert(cpu_gpuNetPtrs.allocated);

	// this part of the code is useful for poisson spike generator function..
	if((numNPois > 0) && (gpuPoissonRand != NULL)) {
		gpuPoissonRand->generate(numNPois, RNG_rand48::MAX_RANGE);
	}

	// this part of the code is invoked when we use spike generators
	if (NgenFunc) {
		resetCounters();

		assert(cpuNetPtrs.spikeGenBits!=NULL);

		// reset the bit status of the spikeGenBits...
		memset(cpuNetPtrs.spikeGenBits, 0, sizeof(int)*(NgenFunc/32+1));

		// If time slice has expired, check if new spikes needs to be generated....
		updateSpikeGenerators();

		// fill spikeGenBits accordingly...
		generateSpikes();

		// copy the spikeGenBits from the CPU to the GPU..
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpy( cpu_gpuNetPtrs.spikeGenBits, cpuNetPtrs.spikeGenBits, sizeof(int)*(NgenFunc/32+1), cudaMemcpyHostToDevice));
	}
}

void CpuSNN::findFiring_GPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();

	int errCode;
		
	assert(cpu_gpuNetPtrs.allocated);

	kernel_findFiring <<<gridSize,blkSize >>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR("findFiring kernel failed\n");
	errCode = checkErrors("kernel_findFiring", gridSize);
	assert(errCode == NO_KERNEL_ERRORS);

	if(MEASURE_LOADING)
		printGpuLoadBalance(false,MAX_BLOCKS);

	return;
}

void CpuSNN::printGpuLoadBalance(bool init, int numBlocks) {
	checkAndSetGPUDevice();

	void* devPtr;
	static int	 cpu_tmp_val[MAX_BLOCKS][LOOP_CNT];
	cudaGetSymbolAddress(&devPtr, tmp_val);

	if(init) {
		// reset the load balance variable..
		CUDA_CHECK_ERRORS( cudaMemset(devPtr, 0, sizeof(tmp_val)));
		return;
	}

	CUDA_CHECK_ERRORS( cudaMemcpy(&cpu_tmp_val, devPtr, sizeof(cpu_tmp_val), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemset(devPtr, 0, sizeof(tmp_val)));

	KERNEL_DEBUG("GPU Load Balancing Information");

	for (int i=0; i < numBlocks; i++) {
		if (cpu_tmp_val[i][0] != 0)
			KERNEL_DEBUG("[%d] Fired Neuron = %d \n", i, cpu_tmp_val[i][1]+cpu_tmp_val[i][2]);
	}

	KERNEL_DEBUG("\n");
}

// get spikes from GPU SpikeCounter
// grpId cannot be ALL (can only get 1 bufPos at a time)
int* CpuSNN::getSpikeCounter_GPU(int grpId) {
	checkAndSetGPUDevice();

	assert(grpId>=0); assert(grpId<numGrp);

	int bufPos = grp_Info[grpId].spkCntBufPos;
	CUDA_CHECK_ERRORS( cudaMemcpy(spkCntBuf[bufPos],cpu_gpuNetPtrs.spkCntBufChild[bufPos],
		grp_Info[grpId].SizeN*sizeof(int),cudaMemcpyDeviceToHost) );

	return spkCntBuf[bufPos];
}

// reset SpikeCounter
// grpId and connectId cannot be ALL (this is handled by the CPU side)
void CpuSNN::resetSpikeCounter_GPU(int grpId) {
	checkAndSetGPUDevice();

	assert(grpId>=0); assert(grpId<numGrp);

	int bufPos = grp_Info[grpId].spkCntBufPos;
	CUDA_CHECK_ERRORS( cudaMemset(cpu_gpuNetPtrs.spkCntBufChild[bufPos],0,grp_Info[grpId].SizeN*sizeof(int)) );
}


void CpuSNN::updateTimingTable_GPU() {
	checkAndSetGPUDevice();

	assert(cpu_gpuNetPtrs.allocated);

	int blkSize  = 128;
	int gridSize = 64;
	kernel_timingTableUpdate <<<gridSize,blkSize >>> (simTimeMs);
	CUDA_GET_LAST_ERROR("timing Table update kernel failed\n");
	int errCode = checkErrors("kernel_timingTableUpdate", gridSize);
	assert(errCode == NO_KERNEL_ERRORS);

	return;
}

void CpuSNN::doCurrentUpdate_GPU() {
	checkAndSetGPUDevice();

	assert(cpu_gpuNetPtrs.allocated);

	int blkSize  = 128;
	int gridSize = 64;
	int errCode;

	if(maxDelay_ > 1) {
		kernel_doCurrentUpdateD2 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
		CUDA_GET_LAST_ERROR_MACRO("Kernel execution failed");
		errCode = checkErrors("kernel_updateCurrentE", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);
	}


	kernel_doCurrentUpdateD1 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
	CUDA_GET_LAST_ERROR_MACRO("Kernel execution failed");
	errCode = checkErrors("kernel_updateCurrentI", gridSize);
	assert(errCode == NO_KERNEL_ERRORS);
}

void CpuSNN::doSTPUpdateAndDecayCond_GPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();

	int errCode;
		
	assert(cpu_gpuNetPtrs.allocated);

	if (sim_with_stp || sim_with_conductances) {
		kernel_STPUpdateAndDecayConductances <<<gridSize, blkSize>>>(simTimeMs, simTimeSec, simTime);
		CUDA_GET_LAST_ERROR("STP update\n");
		errCode = checkErrors("gpu_STPUpdate", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);
	}
}


__global__ void kernel_check_GPU_init2 (int simTime)
{
//	  	int gid=threadIdx.x + blockDim.x*blockIdx.x;
  	int nid[]   = {0,1,2,3};

  	for(int k=0; k < 4; k++) {
  		int i=1;
		for(int t=0; t < 5; t++) {
			retErrVal[k][i++]= gpuPtrs.stpu[nid[k]]*gpuPtrs.stpx[nid[k]];
		}
	  	retErrVal[k][0] = i;
  	}
}

__global__ void kernel_check_GPU_init2_1 (int simTime)
{
//  	int gid=threadIdx.x + blockDim.x*blockIdx.x;
  	int nid=0;
	 	for(int k=0; k < gpuNetInfo.numN; k++) {
		if ( gpuPtrs.neuronFiring[k]) {
			int i=1;
			retErrVal[nid][i++]= k;
			for(int t=0; t < 5; t++) {
				float stp = gpuPtrs.stpu[k]*gpuPtrs.stpx[k];
				retErrVal[nid][i++]= stp;
			}
			//gpuPtrs.neuronFiring[k]=0;
		  	retErrVal[nid][0] = i;
		  	nid++;
		  	if(nid==(MAX_NUM_BLOCKS-1))
		  		break;
		}
  	}
}
///////////////////////////////////////////////////////////////////////////
/// Device local function:      check_GPU_initialization		///
//  In this kernel we return some important parameters, data values of
//  the initialized SNN network using the retErrVal array.
//  This is to just to ensure that the initialization has been
//  done correctly and we can concentrate on the actual kernel code
//  for bugs rather than errors due to incorrect initialization.
///////////////////////////////////////////////////////////////////////////
__global__ void kernel_check_GPU_init ()
{
       	int i=1;

//    		int gid=threadIdx.x + blockDim.x*blockIdx.x;

	if(threadIdx.x==0 && blockIdx.x==0) {
    	//float hstep = gpuNetInfo.maxWeight/(HISTOGRAM_SIZE);
    	retErrVal[0][i++]= gpuNetInfo.numN;
    	retErrVal[0][i++]= gpuNetInfo.numPostSynapses;
    	retErrVal[0][i++]= gpuNetInfo.numNReg;
    	retErrVal[0][i++]= gpuNetInfo.numNExcReg;
    	retErrVal[0][i++]= gpuNetInfo.numNInhReg;
    	retErrVal[0][i++]= gpuPtrs.wt[0];
    	retErrVal[0][i++]= gpuNetInfo.numNPois;
    	retErrVal[0][i++]= gpuPtrs.poissonRandPtr[0];
    	retErrVal[0][i++]= gpuPtrs.poissonRandPtr[1];
    	retErrVal[0][i++]= gpuNetInfo.maxSpikesD1;
    	retErrVal[0][i++]= gpuNetInfo.maxSpikesD2;
    	retErrVal[0][i++]= gpuNetInfo.sim_with_fixedwts;
    	retErrVal[0][i++]= gpuNetInfo.sim_with_conductances;
    	retErrVal[0][i++]= gpuNetInfo.sim_with_stdp;
    	retErrVal[0][i++]= gpuNetInfo.sim_with_stp;
    	retErrVal[0][i++]= gpuGrpInfo[0].SpikeMonitorId;
    	retErrVal[0][i++]= gpuGrpInfo[1].SpikeMonitorId;
    	retErrVal[0][i++]= gpuGrpInfo[0].WithSTP;
    	retErrVal[0][i++]= gpuGrpInfo[1].WithSTP;
		retErrVal[0][i++]= gpuGrpInfo[1].avgTimeScale;
    	retErrVal[0][i++]= 123456789.0;
    	retErrVal[0][0]  = i;
    	i = 1;
    	retErrVal[1][i++]= tex1Dfetch(timingTableD2_tex, 0+timingTableD2_tex_offset);
    	retErrVal[1][i++]= tex1Dfetch(timingTableD2_tex, 1+timingTableD2_tex_offset);
    	retErrVal[1][i++]= loadBufferCount;
    	retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 0);
    	retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 1);
    	retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 2);
    	retErrVal[1][i++]= gpuGrpInfo[0].WithSTDP;
    	retErrVal[1][i++]= gpuGrpInfo[1].WithSTDP;
    	retErrVal[1][i++]= gpuGrpInfo[2].WithSTDP;
    	retErrVal[1][i++]= STATIC_LOAD_START(gpuPtrs.neuronAllocation[0]);
    	retErrVal[1][i++]= STATIC_LOAD_GROUP(gpuPtrs.neuronAllocation[0]);
    	retErrVal[1][i++]= STATIC_LOAD_SIZE(gpuPtrs.neuronAllocation[0]);
    	retErrVal[1][i++]= gpuNetInfo.STP_Pitch;
    	retErrVal[1][i++]= gpuGrpInfo[2].FixedInputWts;
		retErrVal[1][i++]= gpuPtrs.baseFiringInv[gpuGrpInfo[1].StartN];
		retErrVal[1][i++]= gpuPtrs.baseFiringInv[gpuGrpInfo[0].StartN];
    	retErrVal[1][i++]= gpuGrpInfo[0].STP_U;
    	//retErrVal[1][i++]= gpuGrpInfo[0].STP_tD;
    	//retErrVal[1][i++]= gpuGrpInfo[0].STP_tF;
    	retErrVal[1][i++]= gpuPtrs.stpu[0];
    	retErrVal[1][i++]= gpuPtrs.stpx[0];
    	retErrVal[1][i++]= 123456789.0;
    	retErrVal[1][0]  = i;
    	return;
	}
}

void CpuSNN::initGPU(int gridSize, int blkSize) {
	checkAndSetGPUDevice();

	assert(cpu_gpuNetPtrs.allocated);

	kernel_init <<< gridSize, blkSize >>> ();
	CUDA_GET_LAST_ERROR("initGPU kernel failed\n");
	int errCode = checkErrors("kernel_init", gridSize);
	assert(errCode == NO_KERNEL_ERRORS);

	printGpuLoadBalance(true,false);

	checkInitialization();

	checkInitialization2();
}

void CpuSNN::checkInitialization(char* testString) {
	checkAndSetGPUDevice();

	KERNEL_DEBUG("gpu_checkInitialization()");
	void *devPtr;

	assert(cpu_gpuNetPtrs.allocated);

	memset(errVal, 0, sizeof(errVal));
	cudaGetSymbolAddress(&devPtr, retErrVal);
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &errVal, sizeof(errVal), cudaMemcpyHostToDevice));

	int testTable[10];
	// we write some number in this table..
	// write that to the GPU memory. Read it back
	// using the texture access and check if everything is correctly initialized
	testTable[0] = 11; testTable[1] = 12; testTable[2] = 13;
	testTable[3] = 14; testTable[4] = 15; testTable[5] = 16;
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(timingTableD2, testTable, sizeof(int) * (10), 0, cudaMemcpyHostToDevice));
//MDR this check fails because it assumes too much about the network...
//		kernel_check_GPU_init <<< 1, 128 >>> ();
	CUDA_GET_LAST_ERROR("check GPU failed\n");

  // read back the intialization and ensure that they are okay
  KERNEL_DEBUG("%s Checking initialization of GPU...", testString?testString:"");
  CUDA_CHECK_ERRORS( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));
  for(int i=0; i < 4; i++) {
    for(int j=0; j < (int)errVal[i][0]; j++) {
      KERNEL_DEBUG("val[%3d][%3d] = %f", i, j, errVal[i][j]);
    }
    KERNEL_DEBUG("******************");
  }
  KERNEL_DEBUG("Checking done...");
  CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(timingTableD2, timeTableD2, sizeof(int)*(10), 0, cudaMemcpyHostToDevice));
}

void CpuSNN::checkInitialization2(char* testString) {
	checkAndSetGPUDevice();

	KERNEL_DEBUG("CheckInitialization2: Time = %d", simTime);
	void *devPtr;
	memset(errVal, 0, sizeof(errVal));
	cudaGetSymbolAddress(&devPtr, retErrVal);
	CUDA_CHECK_ERRORS( cudaMemcpy(devPtr, &errVal, sizeof(errVal), cudaMemcpyHostToDevice));

	if (sim_with_stp) {
		kernel_check_GPU_init2 <<< 1, 128>>> (simTime);
		CUDA_GET_LAST_ERROR("kernel_check_GPU_init2_1 failed\n");
		int errCode = checkErrors("kernel_init2_1", 1);
		assert(errCode == NO_KERNEL_ERRORS);
	}

	// read back the intialization and ensure that they are okay
	KERNEL_DEBUG("%s Checking Initialization of STP Variable Correctly in GPU...", (testString?testString:""));
	CUDA_CHECK_ERRORS( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));
	for(int i=0; i < 4; i++) {
		for(int j=0; j < (int)errVal[i][0]; j++) {
			KERNEL_DEBUG("val[%3d][%3d] = %f", i, j, errVal[i][j]);
			}
		KERNEL_DEBUG("******************");
	}
}

void CpuSNN::printCurrentInfo(FILE* fp) {
	checkAndSetGPUDevice();

	KERNEL_WARN("Calling printCurrentInfo with fp is deprecated");
	// copy neuron input current...
	KERNEL_DEBUG("Total Synaptic updates:");
	CUDA_CHECK_ERRORS( cudaMemcpy( current, cpu_gpuNetPtrs.current, sizeof(float)*numNReg, cudaMemcpyDeviceToHost));
		for(int i=0; i < numNReg; i++) {
			if (current[i] != 0.0 ) {
				KERNEL_DEBUG("I[%d] -> %f", i, current[i]);
		}
	}
	fflush(fp);
}

// TODO FIXME there's more...
void CpuSNN::deleteObjects_GPU() {
	checkAndSetGPUDevice();

	// cudaFree all device pointers
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.voltage) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.recovery) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.current) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.extCurrent) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Npre) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Npre_plastic) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Npre_plasticInv) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Npost) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.cumulativePost) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.cumulativePre) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.synSpikeTime) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.wt) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.wtChange) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.maxSynWt) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.mulSynFast) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.mulSynSlow) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.nSpikeCnt) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.curSpike) ); // \FIXME exists but never used...
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.firingTableD2) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.firingTableD1) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.avgFiring) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.neuronFiring) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.baseFiring) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.baseFiringInv) );

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.grpDA) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.grp5HT) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.grpACh) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.grpNE) );

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.grpIds) );


	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Izh_a) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Izh_b) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Izh_c) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.Izh_d) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gAMPA) );
	if (sim_with_NMDA_rise) {
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gNMDA_r) );
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gNMDA_d) );
	} else {
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gNMDA) );
	}
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gGABAa) );
	if (sim_with_GABAb_rise) {
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gGABAb_r) );
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gGABAb_d) );
	} else {
		CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.gGABAb) );
	}

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.stpu) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.stpx) );

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.cumConnIdPre) );

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.groupIdInfo) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.neuronAllocation) );

	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.postDelayInfo) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.postSynapticIds) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.preSynapticIds) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.I_set) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.poissonFireRate) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.lastSpikeTime) );
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.spikeGenBits) );

	// delete all real-time spike monitors
	CUDA_CHECK_ERRORS( cudaFree(cpu_gpuNetPtrs.spkCntBuf));
	for (int i=0; i<numSpkCnt; i++)
		CUDA_CHECK_ERRORS(cudaFree(cpu_gpuNetPtrs.spkCntBufChild[i]));

	CUDA_DELETE_TIMER(timer);
}


void CpuSNN::testSpikeSenderReceiver(FILE* fpLog, int simTime) {
	checkAndSetGPUDevice();

	KERNEL_WARN("Calling testSpikeSenderReceiver with fpLog is deprecated");
	if(0) {
		int EnumFires = 0; int InumFires = 0;
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &EnumFires, secD2fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &InumFires, secD1fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
		KERNEL_DEBUG(" ***********( t = %d) FIRE COUNTS ************** %d %d", simTime, EnumFires, InumFires);
		int   numFires;
		float fireCnt;
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &numFires,  testFireCnt1, sizeof(int), 0, cudaMemcpyDeviceToHost));
		KERNEL_DEBUG("testFireCnt1 = %d", numFires);
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &numFires,  testFireCnt2, sizeof(int), 0, cudaMemcpyDeviceToHost));
		KERNEL_DEBUG("testFireCnt2 = %d", numFires);
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &fireCnt,  testFireCntf1, sizeof(int), 0, cudaMemcpyDeviceToHost));
		KERNEL_DEBUG("testFireCntFloat1 = %f", fireCnt);
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &fireCnt,  testFireCntf2, sizeof(int), 0, cudaMemcpyDeviceToHost));
		KERNEL_DEBUG("testFireCntFloat2 = %f", fireCnt);
		KERNEL_DEBUG(" *************************");

		if (sim_with_homeostasis) {
			// MDR originally had this.
			float* baseFiringInv = new float[numN];
			for (int i=0;i<numN;i++) baseFiringInv[i] = 1/baseFiring[i];
				CUDA_CHECK_ERRORS( cudaMemcpy( baseFiringInv, cpu_gpuNetPtrs.baseFiringInv, sizeof(float)*numN, cudaMemcpyDeviceToHost));
			delete(baseFiringInv);
			CUDA_CHECK_ERRORS( cudaMemcpy( avgFiring, cpu_gpuNetPtrs.avgFiring, sizeof(float)*numN, cudaMemcpyDeviceToHost));
			for(int g=0; g < numGrp; g++) {
				KERNEL_DEBUG("Homeostasis values");
				if (!grp_Info[g].FixedInputWts) {
					KERNEL_DEBUG("Group %s:\t", grp_Info2[g].Name.c_str());
					for(int j=grp_Info[g].StartN; j <= grp_Info[g].EndN; j++) {
						KERNEL_DEBUG(" %d: Home =%f (Base =%f)  ", j, avgFiring[j], 1/baseFiringInv[j]);
						KERNEL_DEBUG(" %d: Home =%f (Base =%f)  ", j, avgFiring[j], 1/baseFiringInv[j]);
					}
				}
			}
		}
	}
}

void CpuSNN::globalStateUpdate_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;

	kernel_globalConductanceUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR_MACRO("kernel_globalConductanceUpdate failed");

	// update all neuron state (i.e., voltage and recovery)
	kernel_globalStateUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR_MACRO("Kernel execution failed");
	int errCode = checkErrors("kernel_globalStateUpdate", gridSize);
	assert(errCode == NO_KERNEL_ERRORS);

	// update all group state (i.e., concentration of neuronmodulators)
	// currently support 4 x 128 groups
	kernel_globalGroupStateUpdate <<<4, blkSize>>> (simTimeMs, simTimeSec, simTime);
	CUDA_GET_LAST_ERROR_MACRO("Kernel execution failed");
	errCode = checkErrors("kernel_globalGroupStateUpdate", 4);
	assert(errCode == NO_KERNEL_ERRORS);
}

void CpuSNN::assignPoissonFiringRate_GPU() {
	checkAndSetGPUDevice();

	assert(cpu_gpuNetPtrs.poissonFireRate != NULL);
	for (int grpId=0; grpId < numGrp; grpId++) {
		// given group of neurons belong to the poisson group....
		if (grp_Info[grpId].isSpikeGenerator) {
			int nid    = grp_Info[grpId].StartN;
			PoissonRate* rate = grp_Info[grpId].RatePtr;

			// TODO::: what do we need to do with the refPeriod...
			// does GPU use the refPeriod ???
			//float refPeriod = grp_Info[grpId].RefractPeriod;
				
			if (grp_Info[grpId].spikeGen || rate == NULL) return;

			CUDA_CHECK_ERRORS( cudaMemcpy( &cpu_gpuNetPtrs.poissonFireRate[nid-numNReg], rate->rates, sizeof(float)*rate->len, rate->onGPU?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
		}
	}
}

void CpuSNN::doGPUSim() {
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

#if (ENABLE_MORE_CHECK)
	unsigned int gpu_secD2fireCnt, gpu_secD1fireCnt;

	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, CUDA_CONVERT_SYMBOL(secD2fireCnt), sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, CUDA_CONVERT_SYMBOL(secD1fireCnt), sizeof(int), 0, cudaMemcpyDeviceToHost));

	KERNEL_DEBUG("gpu_secD1fireCnt: %d max: %d gpu_secD2fireCnt: %d max: %d\n",gpu_secD1fireCnt,maxSpikesD1,gpu_secD2fireCnt,maxSpikesD2);

	assert(gpu_secD1fireCnt<=maxSpikesD1);
	assert(gpu_secD2fireCnt<=maxSpikesD2);
#endif

	globalStateUpdate_GPU();
}

void CpuSNN::updateFiringTable_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;
	//void* devPtr;

	//kernel_updateWeightsFiring  <<<gridSize, blkSize>>> ();
	kernel_updateFiring_static<<<gridSize, blkSize>>>();

	kernel_updateFiring<<<gridSize, blkSize>>>();
}

void CpuSNN::updateWeights_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;

	kernel_updateWeights<<<gridSize, blkSize>>>();
}

void CpuSNN::showStatus_GPU() {
	checkAndSetGPUDevice();

	int gpu_secD1fireCnt, gpu_secD2fireCnt;
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, secD2fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, secD1fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
	spikeCountAll1sec = gpu_secD1fireCnt + gpu_secD2fireCnt;
	secD1fireCnt  = gpu_secD1fireCnt;
	//printWeight(-1);
}

__global__ void gpu_resetFiringInformation()
{
	if(threadIdx.x==0 && blockIdx.x==0) {
		for(int i=0; i < ROUNDED_TIMING_COUNT; i++) {
			timingTableD2[i]   = 0;
			timingTableD1[i]   = 0;
		}
		secD2fireCnt=0;
		secD1fireCnt=0;
		secD2fireCntTest=0;
		secD1fireCntTest=0;
		spikeCountD2=0;
		spikeCountD1=0;


    //spikeCountAll1Sec=0;//assigned in copyFiringInfo_GPU()
	}

}


void CpuSNN::resetFiringInformation_GPU() {
	checkAndSetGPUDevice();

	int blkSize  = 128;
	int gridSize = 64;

	gpu_resetFiringInformation<<<gridSize,blkSize>>>();
}


void CpuSNN::copyExternalCurrent(network_ptr_t* dest, network_ptr_t* src, int allocateMem, int grpId) {
	// copy external current from CPU to GPU
	int ptrPos, length;

	if(grpId == -1) {
		ptrPos  = 0;
		length  = numNReg;
	}
	else {
		assert(grpId>=0);
		assert(!isPoissonGroup(grpId));
		ptrPos  = grp_Info[grpId].StartN;
		length  = grp_Info[grpId].SizeN;
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


void CpuSNN::copyFiringInfo_GPU()
{
	unsigned int gpu_secD1fireCnt, gpu_secD2fireCnt;
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, secD2fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, secD1fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
	spikeCountAll1sec = gpu_secD1fireCnt + gpu_secD2fireCnt;
	secD1fireCnt  = gpu_secD1fireCnt;
	assert(gpu_secD1fireCnt<=maxSpikesD1);
	assert(gpu_secD2fireCnt<=maxSpikesD2);
	CUDA_CHECK_ERRORS( cudaMemcpy(firingTableD2, cpu_gpuNetPtrs.firingTableD2, sizeof(int)*gpu_secD2fireCnt, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpy(firingTableD1, cpu_gpuNetPtrs.firingTableD1, sizeof(int)*gpu_secD1fireCnt, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(timeTableD2, timingTableD2, sizeof(int)*(1000+maxDelay_+1), 0, cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERRORS( cudaMemcpyFromSymbol(timeTableD1, timingTableD1, sizeof(int)*(1000+maxDelay_+1), 0, cudaMemcpyDeviceToHost));

	// \TODO: why is this here? The CPU side doesn't have it. And if you can call updateSpikeMonitor() now at any time
	// it might look weird without a time stamp.
//	KERNEL_INFO("Total spikes Multiple Delays=%d, 1Ms Delay=%d", gpu_secD2fireCnt,gpu_secD1fireCnt);
}


void CpuSNN::allocateNetworkParameters() {
	net_Info.numN  = numN;
	net_Info.numPostSynapses  = numPostSynapses;
	net_Info.maxDelay  = maxDelay_;
	net_Info.numNExcReg = numNExcReg;
	net_Info.numNInhReg	= numNInhReg;
	net_Info.numNReg = numNReg;
	assert(numNReg == (numNExcReg + numNInhReg));
	net_Info.numNPois = numNPois;
	net_Info.numNExcPois = numNExcPois;		
	net_Info.numNInhPois = numNInhPois;
	assert(numNPois == (numNExcPois + numNInhPois));
	//net_Info.numNoise  = numNoise;
	net_Info.maxSpikesD2 = maxSpikesD2;
	net_Info.maxSpikesD1 = maxSpikesD1;
	net_Info.sim_with_fixedwts = sim_with_fixedwts;
	net_Info.sim_with_conductances = sim_with_conductances;
	net_Info.sim_with_homeostasis = sim_with_homeostasis;
	net_Info.sim_with_stdp = sim_with_stdp;
//	net_Info.sim_with_modulated_stdp = sim_with_modulated_stdp;
	net_Info.sim_with_stp = sim_with_stp;
	net_Info.numGrp = numGrp;
	net_Info.stdpScaleFactor = stdpScaleFactor_;
	net_Info.wtChangeDecay = wtChangeDecay_;
	cpu_gpuNetPtrs.memType = GPU_MODE;

	net_Info.sim_with_NMDA_rise = sim_with_NMDA_rise;
	net_Info.sim_with_GABAb_rise = sim_with_GABAb_rise;
	net_Info.dAMPA = dAMPA;
	net_Info.rNMDA = rNMDA;
	net_Info.dNMDA = dNMDA;
	net_Info.sNMDA = sNMDA;
	net_Info.dGABAa = dGABAa;
	net_Info.rGABAb = rGABAb;
	net_Info.dGABAb = dGABAb;
	net_Info.sGABAb = sGABAb;

	return;
}

void CpuSNN::configGPUDevice() {
	int devCount, devMax;
	cudaDeviceProp deviceProp;

	CUDA_CHECK_ERRORS(cudaGetDeviceCount(&devCount));
	KERNEL_INFO("CUDA devices Configuration:");
	KERNEL_INFO("  - Number of CUDA devices          = %8d", devCount);

	devMax = CUDA_GET_MAXGFLOP_DEVICE_ID();
	KERNEL_INFO("  - CUDA device ID with max GFLOPs  = %8d", devMax);

	// ithGPU_ gives an index number on which device to run the simulation
	if (ithGPU_ < 0 || ithGPU_ >= devCount) {
		KERNEL_ERROR("CUDA device[%d] does not exist, please choose from [0,%d]", ithGPU_, devCount - 1);
		exitSimulation(1);
	}

	CUDA_CHECK_ERRORS(cudaGetDeviceProperties(&deviceProp, ithGPU_));
	KERNEL_INFO("  - Use CUDA device[%d]: \"%s\"", ithGPU_, deviceProp.name);
	KERNEL_INFO("  - CUDA Compute Capability         =     %2d.%d\n", deviceProp.major, deviceProp.minor);

	if (deviceProp.major < 2) {
		KERNEL_ERROR("CARLsim does not support CUDA devices older than version 2.0");
		exitSimulation(1);
	}

	CUDA_CHECK_ERRORS(cudaSetDevice(ithGPU_));
	CUDA_DEVICE_RESET();
}

void CpuSNN::checkAndSetGPUDevice() {
	int currentDevice;
	cudaGetDevice(&currentDevice);

	if (currentDevice != ithGPU_) {
		KERNEL_DEBUG("Inconsistent GPU context [%d %d]", currentDevice, ithGPU_);
		cudaSetDevice(ithGPU_);
	}
}

void CpuSNN::copyWeightsGPU(unsigned int nid, int src_grp) {
	checkAndSetGPUDevice();

	assert(nid < numNReg);
	unsigned int    cumId   =  cumulativePre[nid];
	float* synWts  = &wt[cumId];
	//TODO: NEEDED TO COMMENT THIS FOR CARLSIM 2.1-2.2 FILEMERGE -- KDC
	// assert(cumId >= (nid-numNPois));
	//assert(cumId < numPreSynapses*numN);

	CUDA_CHECK_ERRORS( cudaMemcpy( synWts, &cpu_gpuNetPtrs.wt[cumId], sizeof(float)*Npre[nid], cudaMemcpyDeviceToHost));
}

// Allocates required memory and then initialize the GPU
void CpuSNN::allocateSNN_GPU() {
	checkAndSetGPUDevice();
	// \FIXME why is this even here? shouldn't this be checked way earlier? and then in CPU_MODE, too...
	if (maxDelay_ > MAX_SynapticDelay) {
		KERNEL_ERROR("You are using a synaptic delay (%d) greater than MAX_SynapticDelay defined in config.h",maxDelay_);
		exitSimulation(1);
	}

	// if we have already allocated the GPU data.. dont do it again...
	if(gpuPoissonRand != NULL) return;

	int gridSize = 64; int blkSize  = 128;

	//checkGPUDevice();

	int numN=0;
	for (int g=0;g<numGrp;g++) {
		numN += grp_Info[g].SizeN;
	}

	// generate the random number for the poisson neuron here...
	if(gpuPoissonRand == NULL) {
		gpuPoissonRand = new RNG_rand48(randSeed_);
	}

	gpuPoissonRand->generate(numNPois, RNG_rand48::MAX_RANGE);

	// initialize CpuSNN::cpu_gpuNetPtrs.poissonRandPtr, save the random pointer as poisson generator....
	cpu_gpuNetPtrs.poissonRandPtr = (unsigned int*) gpuPoissonRand->get_random_numbers();

	//ensure that we dont do all the above optimizations again		
	assert(doneReorganization == true);

	// display some memory management info
	size_t avail, total, previous;
	float toGB = 1024.0 * 1024.0 * 1024.0;
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("GPU Memory Management: (Total %2.3f GB)",(float)(total/toGB));
	KERNEL_INFO("Data\t\t\tSize\t\tTotal Used\tTotal Available");
	KERNEL_INFO("Init:\t\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(total)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// copy data to from CpuSNN:: to network_info_t CpuSNN::net_Info
	allocateNetworkParameters();
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Ntw Params:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// initialize cpu_gpuNetPtrs.neuronAllocation, __device__ loadBufferCount, loadBufferSize
	allocateStaticLoad(blkSize);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Static Load:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,
		(float)((total-avail)/toGB),(float)(avail/toGB));
	previous=avail;

	allocateGroupId();
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Group Id:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// this table is useful for quick evaluation of the position of fired neuron
	// given a sequence of bits denoting the firing..
	// initialize __device__ gpu_tableQuickSynId[256]
	initTableQuickSynId();

	// initialize (cudaMemset) cpu_gpuNetPtrs.I_set, cpu_gpuNetPtrs.poissonFireRate, cpu_gpuNetPtrs.neuronFiring
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.Npre, cpu_gpuNetPtrs.Npre_plastic, cpu_gpuNetPtrs.Npre_plasticInv, cpu_gpuNetPtrs.cumulativePre
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.cumulativePost, cpu_gpuNetPtrs.Npost, cpu_gpuNetPtrs.postDelayInfo
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.postSynapticIds, cpu_gpuNetPtrs.preSynapticIds
	// copy data to CpuSNN:net_Info.postSynCnt, preSynCnt
	copyConnections(&cpu_gpuNetPtrs,  cudaMemcpyHostToDevice, 1);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("Conn Info:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB",(float)(previous-avail)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.wt, cpu_gpuNetPtrs.wtChange, cpu_gpuNetPtrs.maxSynWt, cpu_gpuNetPtrs.synSpikeTime
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.firingTableD1, cpu_gpuNetPtrs.firingTableD2, cpu_gpuNetPtrs.lastSpikeTime
	// initialize (cudaMalloc) cpu_gpuNetPtrs.spikeGenBits
	// initialize (copy from cpuNetPtrs) cpu_gpuNetPtrs.nSpikeCnt, cpu_gpuNetPtrs.recovery, cpu_gpuNetPtrs.voltage, cpu_gpuNetPtrs.current
	// initialize (copy from cpuNetPtrs) cpu_gpuNetPtrs.gGABAa, cpu_gpuNetPtrs.gGABAb, cpu_gpuNetPtrs.gAMPA, cpu_gpuNetPtrs.gNMDA
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.Izh_a, cpu_gpuNetPtrs.Izh_b, cpu_gpuNetPtrs.Izh_c, cpu_gpuNetPtrs.Izh_d
	// copy data to CpuSNN:net_Info.preSynLength
	// initialize (copy from CpuSNN) stpu, stpx
	// initialize (copy from CpuSNN) cpu_gpuNetPtrs.grpDA, cpu_gpuNetPtrs.grp5HT, cpu_gpuNetPtrs.grpACh, cpu_gpuNetPtrs.grpNE
	copyState(&cpu_gpuNetPtrs, 1);
	cudaMemGetInfo(&avail,&total);
	KERNEL_INFO("State Info:\t\t%2.3f GB\t%2.3f GB\t%2.3f GB\n\n",(float)(previous-avail)/toGB,(float)((total-avail)/toGB),
		(float)(avail/toGB));
	previous=avail;

	// copy Spike Counters
	// 2D arrays are a bit tricky... We can't just copy spkCntBuf over. We still have to use spkCntBuf for
	// cudaMalloc(), but then we need to cudaMemcpy() that array of pointers to the pointer that we got from the
	// first cudaMalloc().
	CUDA_CHECK_ERRORS( cudaMalloc( (void**) &(cpu_gpuNetPtrs.spkCntBuf), sizeof(int*)*MAX_GRP_PER_SNN));
	for (int g=0; g<numGrp; g++) {
		if (!grp_Info[g].withSpikeCounter)
			continue; // skip group if it doesn't have a spkMonRT

		int bufPos = grp_Info[g].spkCntBufPos; // retrieve pos in spike buf

		// allocate child pointers
		CUDA_CHECK_ERRORS( cudaMalloc( (void**) &(cpu_gpuNetPtrs.spkCntBufChild[bufPos]), sizeof(int)*grp_Info[g].SizeN));

		// copy child pointer to device
		CUDA_CHECK_ERRORS( cudaMemcpy(&(cpu_gpuNetPtrs.spkCntBuf[bufPos]), &(cpu_gpuNetPtrs.spkCntBufChild[bufPos]),
			sizeof(int*), cudaMemcpyHostToDevice) );

		// copy data
		CUDA_CHECK_ERRORS( cudaMemcpy(cpu_gpuNetPtrs.spkCntBufChild[bufPos], spkCntBuf[bufPos],
			sizeof(int)*grp_Info[g].SizeN, cudaMemcpyHostToDevice) );
	}

	// copy relevant pointers and network information to GPU
	void* devPtr;
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(gpuPtrs, &cpu_gpuNetPtrs, sizeof(network_ptr_t), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(gpuNetInfo, &net_Info, sizeof(network_info_t), 0, cudaMemcpyHostToDevice));
	// FIXME: we can change the group properties such as STDP as the network is running.  So, we need a way to updating the GPU when changes are made.

	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol(gpuGrpInfo, grp_Info, (net_Info.numGrp) * sizeof(group_info_t), 0, cudaMemcpyHostToDevice));

	KERNEL_DEBUG("Transfering group settings to GPU:");
	for (int i=0;i<numGrp;i++) {
		KERNEL_DEBUG("Settings for Group %s:", grp_Info2[i].Name.c_str());
		
		KERNEL_DEBUG("\tType: %d",(int)grp_Info[i].Type);
		KERNEL_DEBUG("\tSizeN: %d",grp_Info[i].SizeN);
		KERNEL_DEBUG("\tMaxFiringRate: %d",(int)grp_Info[i].MaxFiringRate);
		KERNEL_DEBUG("\tRefractPeriod: %f",grp_Info[i].RefractPeriod);
		KERNEL_DEBUG("\tM: %d",grp_Info[i].numPostSynapses);
		KERNEL_DEBUG("\tPreM: %d",grp_Info[i].numPreSynapses);
		KERNEL_DEBUG("\tspikeGenerator: %d",(int)grp_Info[i].isSpikeGenerator);
		KERNEL_DEBUG("\tFixedInputWts: %d",(int)grp_Info[i].FixedInputWts);
		KERNEL_DEBUG("\tMaxDelay: %d",(int)grp_Info[i].MaxDelay);
		KERNEL_DEBUG("\tWithSTDP: %d",(int)grp_Info[i].WithSTDP);
		if (grp_Info[i].WithSTDP) {
			KERNEL_DEBUG("\t\tSTDP type: %s",stdpType_string[grp_Info[i].WithSTDPtype]);
			KERNEL_DEBUG("\t\tTAU_LTP_INV: %f",grp_Info[i].TAU_LTP_INV);
			KERNEL_DEBUG("\t\tTAU_LTD_INV: %f",grp_Info[i].TAU_LTD_INV);
			KERNEL_DEBUG("\t\tALPHA_LTP: %f",grp_Info[i].ALPHA_LTP);
			KERNEL_DEBUG("\t\tALPHA_LTD: %f",grp_Info[i].ALPHA_LTD);
		}
		KERNEL_DEBUG("\tWithSTP: %d",(int)grp_Info[i].WithSTP);
		if (grp_Info[i].WithSTP) {
			KERNEL_DEBUG("\t\tSTP_U: %f",grp_Info[i].STP_U);
//				KERNEL_DEBUG("\t\tSTP_tD: %f",grp_Info[i].STP_tD);
//				KERNEL_DEBUG("\t\tSTP_tF: %f",grp_Info[i].STP_tF);
		}
		KERNEL_DEBUG("\tspikeGen: %s",grp_Info[i].spikeGen==NULL?"Is Null":"Is set");
		KERNEL_DEBUG("\tspikeMonitorRT: %s",grp_Info[i].withSpikeCounter?"Is set":"Is Null");
		if (grp_Info[i].withSpikeCounter) {
			int bufPos = grp_Info[i].spkCntBufPos;
			for (int j=0; j<grp_Info[i].SizeN; j++)
				KERNEL_DEBUG("\t%d",spkCntBuf[bufPos][j]);
			KERNEL_DEBUG("\trecordDur: %d",grp_Info[i].spkCntRecordDur);
		} 	
		KERNEL_DEBUG("\tspikeGen: %s",grp_Info[i].spikeGen==NULL?"Is Null":"Is set");
	}

	cpu_gpuNetPtrs.allocated = true;

	// map the timing table to texture.. saves a lot of headache in using shared memory
	CUDA_CHECK_ERRORS_MACRO(cudaGetSymbolAddress(&devPtr, timingTableD2));
	size_t offset;
	CUDA_CHECK_ERRORS_MACRO(cudaBindTexture(&offset, timingTableD2_tex, devPtr, sizeof(int) * ROUNDED_TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS_MACRO (cudaGetSymbolAddress(&devPtr, timingTableD2_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));
		
	CUDA_CHECK_ERRORS_MACRO(cudaGetSymbolAddress(&devPtr, timingTableD1));
	CUDA_CHECK_ERRORS_MACRO(cudaBindTexture(&offset, timingTableD1_tex, devPtr, sizeof(int) * ROUNDED_TIMING_COUNT));
	offset = offset / sizeof(int);
	CUDA_CHECK_ERRORS_MACRO(cudaGetSymbolAddress(&devPtr, timingTableD1_tex_offset));
	CUDA_CHECK_ERRORS(cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));

	// initialize (memset) cpu_gpuNetPtrs.current
	CUDA_CHECK_ERRORS(cudaMemset(cpu_gpuNetPtrs.current, 0, sizeof(float) * numNReg));
//	CUDA_CHECK_ERRORS(cudaMemset(cpu_gpuNetPtrs.extCurrent, 0, sizeof(float)*numNReg));
//	copyExternalCurrent(&cpu_gpuNetPtrs, &cpuNetPtrs, true);
	initGPU(gridSize, blkSize);
}

//This gets called when updateNetwork is called so setSTDP, setSTP, or setHomeostasis
//should now work in GPU_MODE.
void CpuSNN::copyGrpInfo_GPU() {
	checkAndSetGPUDevice();

	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol("gpuGrpInfo", grp_Info, (net_Info.numGrp)*sizeof(group_info_t), 0, cudaMemcpyHostToDevice));
}

//Copy all the appropriate state variables back to the GPU now that they have been
//reset. TAGS:UPDATE -- KDC
void CpuSNN::copyUpdateVariables_GPU() {
	checkAndSetGPUDevice();

	//we now copy all variables that get reset in updateNetwork.
	CUDA_CHECK_ERRORS( cudaMemcpy( cpu_gpuNetPtrs.wt, wt, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));
	if ((!sim_with_fixedwts) || (sim_with_stdp)){
		CUDA_CHECK_ERRORS( cudaMemcpy( cpu_gpuNetPtrs.wtChange, wtChange, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));
		CUDA_CHECK_ERRORS( cudaMemcpy( cpu_gpuNetPtrs.maxSynWt, maxSynWt, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));
	}
	CUDA_CHECK_ERRORS( cudaMemcpy( cpu_gpuNetPtrs.synSpikeTime, synSpikeTime, sizeof(int)*preSynCnt, cudaMemcpyHostToDevice));

  // copy the neuron state information to the GPU..
	copyNeuronState(&cpu_gpuNetPtrs, &cpuNetPtrs, cudaMemcpyHostToDevice, 0);
	copyNeuronParameters(&cpu_gpuNetPtrs, cudaMemcpyHostToDevice, 0);
	if(sim_with_stp){
		copySTPState(&cpu_gpuNetPtrs, &cpuNetPtrs, cudaMemcpyHostToDevice, 0);
	}

//finally copy this to something the GPU can access.
	CUDA_CHECK_ERRORS(cudaMemcpyToSymbol("gpuPtrs", &cpu_gpuNetPtrs, sizeof(network_ptr_t), 0, cudaMemcpyHostToDevice));

}

//This gets called in updateNetwork to copy all the reset data back to the GPU.
//TAGS:UPDATE -- KDC
void CpuSNN::updateNetwork_GPU(bool resetFiringInfo) {
	checkAndSetGPUDevice();

	//need to copy this data so STDP, STP, conductances can be reset.
	copyGrpInfo_GPU();

	if(resetFiringInfo)
		resetFiringInformation_GPU();

	copyUpdateVariables_GPU();
}

void CpuSNN::printSimSummary() {
	checkAndSetGPUDevice();

	float etime;
	if(simMode_ == GPU_MODE) {
		stopGPUTiming();
		etime = gpuExecutionTime;
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &spikeCountD2Host, secD2fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &spikeCountD1Host, secD1fireCnt, sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCountAll1sec = spikeCountD1 + spikeCountD2;
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &spikeCountD2Host, spikeCountD2, sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_CHECK_ERRORS_MACRO( cudaMemcpyFromSymbol( &spikeCountD1Host, spikeCountD1, sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCountAll      = spikeCountD1Host + spikeCountD2Host;
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
	KERNEL_INFO("\t\t\tnumSynapses = %d", postSynCnt);
	KERNEL_INFO("\t\t\tmaxDelay = %d", maxDelay_);
	KERNEL_INFO("Simulation Mode:\t%s",sim_with_conductances?"COBA":"CUBA");
	KERNEL_INFO("Random Seed:\t\t%d", randSeed_);
	KERNEL_INFO("Timing:\t\t\tModel Simulation Time = %lld sec", (unsigned long long)simTimeSec);
	KERNEL_INFO("\t\t\tActual Execution Time = %4.2f sec", etime/1000.0);
	KERNEL_INFO("Average Firing Rate:\t2+ms delay = %3.3f Hz", spikeCountD2Host/(1.0*simTimeSec*numNExcReg));
	KERNEL_INFO("\t\t\t1ms delay = %3.3f Hz", spikeCountD1Host/(1.0*simTimeSec*numNInhReg));
	KERNEL_INFO("\t\t\tOverall = %3.3f Hz", spikeCountAll/(1.0*simTimeSec*numN));
	KERNEL_INFO("Overall Firing Count:\t2+ms delay = %d", spikeCountD2Host);
	KERNEL_INFO("\t\t\t1ms delay = %d", spikeCountD1Host);
	KERNEL_INFO("\t\t\tTotal = %d", spikeCountAll);
	KERNEL_INFO("*********************************************************************************\n");
}



/* MDR -- Deprecated
/////////////////////////////////////////////////////////////////////////////////
// Device Kernel Function:      Intialization  Noise  Input currents	      ///
// KERNEL: This is useful for generating suitable thalamic inputs	      ///
// to the network. The ids of neurons to be used is stored in the randId array and 
// the kernel reads the randId array to initialize each input thalamic current
// of the selected neurons to appropriate values..
// future modification is to generate the random numbers using GPU itself instead of
// creating the random number at the CPU side and doing a memcopy operation.  ///
/////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_initThalInput(int setVoltage)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
	   
const int idBegin = bid*blockDim.x + tid;
const int idStep  = blockDim.x*gridDim.x;	   
int accPos = 0;
noiseGenProperty_t*	gpu_noiseGenGroup = (noiseGenProperty_t *) gpuPtrs.noiseGenProp;
for (int i=0; i < gpuNetInfo.numNoise; i++)
{
float  currentStrength = gpu_noiseGenGroup[i].currentStrength;
int    nrands 		 	= gpu_noiseGenGroup[i].rand_ncount;

int randCnt = accPos;
for(int j=idBegin; j < nrands; j+=idStep) {
int randId = gpuPtrs.randId[randCnt+j];
// fprintf(fpLog, " %d %d \n", simTimeSec*1000+simTimeMs, randNeuronId[randCnt]);
// assuming there is only one driver at a time.
// More than one noise input is not correct... 			 
if(setVoltage)
gpuPtrs.voltage[randId] = currentStrength;
else	
gpuPtrs.current[randId] = currentStrength;
}
accPos += nrands;
}
__syncthreads();
}
	

// Initialize the Thalamic input to network
void CpuSNN::initThalInput_GPU()
{
DBG(2, fpLog, AT, "gpu_initThalInput()");

assert(cpu_gpuNetPtrs.allocated);

int blkSize  = 128;
int gridSize = 64;
bool setVoltage = true;

// Initialize the Thalamic input current into the network
if(numNoise) {
// copy the generated random number into the randId array
CUDA_CHECK_ERRORS( cudaMemcpy(cpu_gpuNetPtrs.randId, randNeuronId, sizeof(int)*numRandNeurons, cudaMemcpyHostToDevice));

kernel_initThalInput <<<gridSize,blkSize >>> (setVoltage);
CUDA_GET_LAST_ERROR("initThalInput kernel failed\n");
int errCode = checkErrors("kernel_initThalInput", gridSize);
}
}

*/
