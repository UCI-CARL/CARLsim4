//
// Copyright (c) 2011 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	#include "config.h"
	#include "snn.h"
	#include "errorCode.h"
	#include "cuda_runtime.h"
	
	// includes, library
	#include "cudpp/cudpp.h"

	// includes, project
	#include "cutil_inline.h"
	#include "cutil_math.h"

	RNG_rand48* gpuPoissonRand = NULL;

	#define ROUNDED_TIMING_COUNT  (((1000+MAX_SynapticDelay+1)+127) & ~(127))  // (1000+D) rounded to multiple 128

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
    //  timingTableD2[D] -- should be the number of spikes "leftover" from the previous second
    //	timingTableD2[D+1]-timingTableD2[D] -- should be the number of spikes in the first ms of the current second
    //  timingTableD2[1000+D] -- should be the number of spikes in the current second + the leftover spikes.
    //
    ///////////////////////////////////////////////////////////////////

	__device__ int  timingTableD2[ROUNDED_TIMING_COUNT];
	__device__ int  timingTableD1[ROUNDED_TIMING_COUNT];
	__device__ int	testVarCnt=0;
	__device__ int	testVarCnt2=0;
	__device__ int	testVarCnt1=0;
	__device__ unsigned int	secD2fireCnt=0;
	__device__ unsigned int	secD1fireCnt=0;
	__device__ unsigned int	secD2fireCntTest=0;
	__device__ unsigned int	secD1fireCntTest=0;
	__device__ unsigned int spikeCountD2=0;
	__device__ unsigned int spikeCountD1=0;

	__device__ int  generatedSpikesE=0;
	__device__ int  generatedSpikesI=0;
	__device__ int  receivedSpikesE=0;
	__device__ int  receivedSpikesI=0;
	__device__ int  senderIdE[1000];
	__device__ int  senderIdI[1000];
	__device__ int  receiverId[1000];

	__device__ __constant__ network_ptr_t		gpuPtrs;
	__device__ __constant__ network_info_t		gpuNetInfo;
	__device__ __constant__ group_info_t		gpuGrpInfo[MAX_GRP_PER_SNN];
//	__device__ __constant__ noiseGenProperty_t*	gpu_noiseGenGroup;

	__device__ __constant__ float				constData[256];

	__device__  int	  loadBufferCount; 
	__device__  int   loadBufferSize;

	float data[256];

	float* currentVal;

	texture <int,    1, cudaReadModeElementType>  timingTableD2_tex;
	texture <int,    1, cudaReadModeElementType>  timingTableD1_tex;
	texture <int,    1, cudaReadModeElementType>  groupIdInfo_tex; // groupIDInfo is allocated using cudaMalloc thus doesn't require an offset when using textures
	__device__  int timingTableD1_tex_offset;
	__device__  int timingTableD2_tex_offset;

	
	__device__ int generatedErrors = 0;
	__device__ int	 tmp_val[MAX_NUM_BLOCKS][LOOP_CNT];
	__device__ int	 retErrCode=NO_KERNEL_ERRORS;
	__device__ float retErrVal[MAX_NUM_BLOCKS][20];

	#define INIT_CHECK(enable, src, val)\
	{									\
		if(enable)						\
		  if(threadIdx.x==0) 			\
			src = val;					\
	}


	#define ERROR_CHECK_COND(enable, src, val, cond, retVal) \
	{									\
		if(enable) 						\
			if (!(cond)) { 				\
				src = val;				\
				return retVal;				\
			}							\
	}

	#define ERROR_CHECK_COND_NORETURN(enable, src, val, cond) \
	{									\
		if(enable) 						\
			if (!(cond)) { 				\
				src = val;				\
				return;					\
			}							\
	}

	#define ERROR_CHECK(enable, src, val)		\
	{											\
		if(enable) 								\
			if( blockIdx.x >= MAX_NUM_BLOCKS) { \
				src = val;						\
				return;							\
			}									\
	}

	#define UPDATE_ERROR_CHECK4(src, val1, val2, val3, val4)\
	{											\
		if (src && (threadIdx.x==0))	 {		\
		   retErrCode = src;					\
		   if(ENABLE_MORE_CHECK) {				\
				retErrVal[blockIdx.x][0]=0xdead;\
				retErrVal[blockIdx.x][1]=4;		\
				retErrVal[blockIdx.x][2]=val1;	\
				retErrVal[blockIdx.x][3]=val2;	\
				retErrVal[blockIdx.x][4]=val3;	\
				retErrVal[blockIdx.x][5]=val4;	\
			}									\
		}										\
	}
		
	#define	MEASURE_LOADING			0

	#define MEASURE_GPU(pos,val)	\
		{							\
			if(MEASURE_LOADING)		\
				if (threadIdx.x==0)	\
					tmp_val[blockIdx.x][pos] += val;\
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
	   CUDA_SAFE_CALL( cudaMemcpy( devPtr, tableQuickSynId, sizeof(tableQuickSynId), cudaMemcpyHostToDevice));            
	}
	
	__device__ inline bool isPoissonGroup(uint16_t& grpId, unsigned int& nid)
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
//		int val=atomicAdd(&testVarCnt2, 3);
//		gpuPtrs.testVar2[val]   = nid+1;
//		gpuPtrs.testVar2[val+1] = synGrpId+1;
//		gpuPtrs.testVar2[val+2] = *tmp_ptr+1;
		return tmp_ptr;
	}

	__device__ inline uint32_t getSTPBufPos(unsigned int nid, uint32_t t)
	{
		// TODO: MNJ CHECK THIS FOR CORRECTNESS..
		return (((t%STP_BUF_SIZE)*gpuNetInfo.STP_Pitch) + nid);
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
			timingTableD2[t+gpuNetInfo.D+1]  = secD2fireCnt;
			timingTableD1[t+gpuNetInfo.D+1]  = secD1fireCnt;
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
			uint16_t grpId   	 = STATIC_LOAD_GROUP(threadLoad);

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

	__device__ int 	 testFireCnt1  = 0;
	__device__ int 	 testFireCnt2  = 0;
	__device__ float testFireCntf1 = 0.0;
	__device__ float testFireCntf2 = 0.0;

	// Allocation of the group and its id..
	void CpuSNN::allocateGroupId()
	{
		assert (cpu_gpuNetPtrs.groupIdInfo == NULL);
		int3* tmp_neuronAllocation = (int3 *) malloc(sizeof(int3)*net_Info.numGrp);
		for (int g=0; g < net_Info.numGrp; g++) {
			int3  threadLoad;
			threadLoad.x = grp_Info[g].StartN;
			threadLoad.y = grp_Info[g].EndN;
			threadLoad.z = g;
			tmp_neuronAllocation[g] = threadLoad;
		}
		cutilSafeCall  ( cudaMalloc ((void**) &cpu_gpuNetPtrs.groupIdInfo, sizeof(int3)*net_Info.numGrp));
		CUDA_SAFE_CALL ( cudaMemcpy ( cpu_gpuNetPtrs.groupIdInfo, tmp_neuronAllocation, sizeof(int3)*net_Info.numGrp, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL ( cudaBindTexture (NULL, groupIdInfo_tex, cpu_gpuNetPtrs.groupIdInfo, sizeof(int3)*net_Info.numGrp));
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
	int CpuSNN::allocateStaticLoad(int bufSize)
	{
		FILE* fpArr[]={fpLog, stderr};
		// only one thread does the static load table
		int   bufferCnt = 0;
		for (int g=0; g < net_Info.numGrp; g++) {
			bufferCnt += (int) ceil(1.0*grp_Info[g].SizeN/bufSize);
			fprintf(fpLog, "Grp Size = %d, Total Buffer Cnt = %d, Buffer Cnt = %f\n",  grp_Info[g].SizeN, bufferCnt, ceil(1.0*grp_Info[g].SizeN/bufSize));
		}
		assert(bufferCnt > 0);

		int2*  tmp_neuronAllocation = (int2 *) malloc(sizeof(int2)*bufferCnt);
		for(int i=0; i < 2; i++) {
			if (i == 0 || (showLogMode >= 2)) {
				fprintf(fpArr[i], "STATIC THREAD ALLOCATION\n");
				fprintf(fpArr[i], "------------------------\n");
				fprintf(fpArr[i], "Buffer Size = %d, Buffer Count = %d\n", bufSize, bufferCnt);
				}
		}
		bufferCnt = 0;
		for (int g=0; g < net_Info.numGrp; g++) {
			for (int n=grp_Info[g].StartN; n <= grp_Info[g].EndN; n += bufSize) {
				int2  threadLoad;
				// starting neuron id is saved...
				threadLoad.x = n;
				if ((n + bufSize-1) <= grp_Info[g].EndN)
					// grpID + full size
					threadLoad.y = (g + (bufSize << 16));
				else
					// grpID + left-over size
					threadLoad.y = (g + ((grp_Info[g].EndN-n+1) << 16));

				// fill the static load distribution here...
				int testg = STATIC_LOAD_GROUP(threadLoad);
				tmp_neuronAllocation[bufferCnt] = threadLoad;
				for(int i=0; i < 2; i++) {
					if (i == 0 || (showLogMode >= 2)) {
						fprintf(fpArr[i], "%d. Start=%d, size=%d grpId=%d:%s (MonId=%d)\n",
							bufferCnt, STATIC_LOAD_START(threadLoad),
							STATIC_LOAD_SIZE(threadLoad),
							STATIC_LOAD_GROUP(threadLoad),
							grp_Info2[testg].Name.c_str(),
							grp_Info[testg].MonitorId);
					}
				}
				bufferCnt++;
			}
		}

		assert(cpu_gpuNetPtrs.allocated==false);
		// Finally writeback the total bufferCnt
		// Note down the buffer size for reference
		fprintf(fpLog, "GPU loadBufferSize = %d, GPU loadBufferCount = %d\n", bufSize, bufferCnt);
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( "loadBufferCount",  &bufferCnt, sizeof(int), 0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL( cudaMemcpyToSymbol( "loadBufferSize",   &bufSize,   sizeof(int), 0, cudaMemcpyHostToDevice));
		cutilSafeCall(  cudaMalloc((void**) &cpu_gpuNetPtrs.neuronAllocation, sizeof(int2)*bufferCnt));
		CUDA_SAFE_CALL( cudaMemcpy( cpu_gpuNetPtrs.neuronAllocation, tmp_neuronAllocation, sizeof(int2)*bufferCnt, cudaMemcpyHostToDevice));

		return bufferCnt;
	}

    ///////////////////////////////////////////////
	// 1. KERNELS used when a specific neuron fires //
    ///////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////
	// Device local function:      	Update the STP Variables		  ///
	// update the STPU and STPX variable after firing			      ///
	/////////////////////////////////////////////////////////////////////////////////

	__device__ void firingUpdateSTP (unsigned int& nid, int& simTime, uint16_t&  grpId)
	{
		// implements Mongillo, Barak and Tsodyks model of Short term plasticity
		uint32_t ind   = getSTPBufPos(nid, simTime);
		uint32_t ind_1 = getSTPBufPos(nid, (simTime-1)); // MDR -1 is correct, we use the value before the decay has been applied for the current time step.

		gpuPtrs.stpx[ind] = gpuPtrs.stpx[ind] - gpuPtrs.stpu[ind_1]*gpuPtrs.stpx[ind_1];
		gpuPtrs.stpu[ind] = gpuPtrs.stpu[ind] + gpuGrpInfo[grpId].STP_U*(1-gpuPtrs.stpu[ind_1]);
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

	__device__ void resetFiredNeuron(unsigned int& nid, uint16_t & grpId, int& simTime)
 	{
		// TODO: convert this to use coalesced access by grouping into a
		// single 16 byte access. This might improve bandwidth performance
		// This is fully uncoalsced access...need to convert to coalsced access..
		gpuPtrs.voltage[nid] = gpuPtrs.Izh_c[nid];
		gpuPtrs.recovery[nid] += gpuPtrs.Izh_d[nid];
		gpuPtrs.lastSpikeTime[nid] = simTime;
		gpuPtrs.nSpikeCnt[nid]++;
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
	__device__ void updateFiringTable(unsigned int& nid, uint16_t& grpId, volatile unsigned int& cntD2, volatile unsigned int& cntD1)
	{
		int pos;
		if (gpuGrpInfo[grpId].MaxDelay == 1) {
			// this group has a delay of only 1
			pos = atomicAdd((int*)&cntD1, 1);
			gpuPtrs.firingTableD1[pos]  = SET_FIRING_TABLE(nid, grpId);
		}
		else {
			// all other groups is dumped here 
			pos = atomicAdd((int*)&cntD2, 1);
			gpuPtrs.firingTableD2[pos]  = SET_FIRING_TABLE(nid, grpId);
		}
	}

	__device__ int newFireUpdate (	int* 	fireTablePtr,
									uint16_t* fireGrpId,
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

			// only neurons would do the remaining settings...
			// pure poisson generators will return without changing anything else..
			if (IS_REGULAR_NEURON(nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois))
				resetFiredNeuron(nid, fireGrpId[i], simTime);
			else {
				gpuPtrs.nSpikeCnt[nid]++;
			}
		}

		__syncthreads();

		return 0;
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
									uint16_t*  		fireGrpId,
									volatile unsigned int&   fireCnt,
									int&      		simTime)
	{
		for(int pos=threadIdx.x/LTP_GROUPING_SZ; pos < fireCnt; pos += (blockDim.x/LTP_GROUPING_SZ))  {
			// each neuron has two variable pre and pre_exc
			// pre: number of pre-neuron
			// pre_exc: number of neuron had has plastic connections
			uint16_t grpId = fireGrpId[pos];
			if (gpuGrpInfo[grpId].WithSTDP) { // MDR, FIXME this probably will cause more thread divergence than need be...
				int  nid   = fireTablePtr[pos];
				int  end_p = gpuPtrs.cumulativePre[nid] + gpuPtrs.Npre_plastic[nid];
				for(int p  = gpuPtrs.cumulativePre[nid] + threadIdx.x%LTP_GROUPING_SZ;
						p < end_p;
						p+=LTP_GROUPING_SZ) {
					int stdp_tDiff = (simTime - gpuPtrs.synSpikeTime[p]);
					if ((stdp_tDiff > 0) && ((stdp_tDiff*gpuGrpInfo[grpId].TAU_LTP_INV)<25)) {
						gpuPtrs.wtChange[p] += STDP(stdp_tDiff, gpuGrpInfo[grpId].ALPHA_LTP, gpuGrpInfo[grpId].TAU_LTP_INV);
//						int val = atomicAdd(&testVarCnt, 4);
//						gpuPtrs.testVar[val]   = 1+nid;
//						gpuPtrs.testVar[val+1] = 1+p-gpuPtrs.cumulativePre[nid];
//						gpuPtrs.testVar[val+2] = 1+gpuPtrs.wtChange[p];
//						gpuPtrs.testVar[val+3] = 1+stdp_tDiff;
					}
				}
			}
		}
		__syncthreads();
	}

	__device__ inline int assertionFiringParam(uint16_t& grpId, int& lastId)
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
	void CpuSNN::setSpikeGenBit_GPU(unsigned int nid, int grp)
	{
		unsigned int nidPos    = (nid - grp_Info[grp].StartN + grp_Info[grp].Noffset);
		unsigned int nidBitPos = nidPos%32;
		unsigned int nidIndex  = nidPos/32;

		assert(nidIndex < (NgenFunc/32+1));

		//fprintf(stderr, "time = %d, nid = %d\n", simTime, nid);

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
	__global__ 	void kernel_findFiring (int t, int sec, int simTime)
	{
		__shared__ volatile unsigned int fireCnt;
		__shared__ volatile unsigned int fireCntTest;
		__shared__ volatile unsigned int fireCntD1;
		__shared__ int 		fireTable[FIRE_CHUNK_CNT];
		__shared__ uint16_t	fireGrpId[FIRE_CHUNK_CNT];
		__shared__ volatile int errCode;

		if (0==threadIdx.x) {
			fireCnt	  = 0; // initialize total cnt to 0
			fireCntD1  = 0; // initialize inh. cnt to 0
		}

		// Ignore this unless you are doing real debugging gpu code...
		INIT_CHECK(ENABLE_MORE_CHECK, retErrCode, NO_KERNEL_ERRORS);

		const int totBuffers=loadBufferCount;

		__syncthreads();

		for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x)
		{
			// KILLME !!! This can be further optimized ....
			// instead of reading each neuron group separately .....
			// read a whole buffer and use the result ......
			int2 threadLoad  = getStaticThreadLoad(bufPos);
			unsigned int  nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
			int  lastId      = STATIC_LOAD_SIZE(threadLoad);
			uint16_t grpId   = STATIC_LOAD_GROUP(threadLoad);
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
				else  {
					if (gpuPtrs.voltage[nid] >= 30.0) {
						needToWrite = true;
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
	__global__ void kernel_globalConductanceUpdate (int t, int sec, int simTime)
	{
		__shared__ int sh_tableQuickSynId[256];

		// Table for quick access
		for(int i=0; i < 256; i+=blockDim.x){
			if((i+threadIdx.x) < 256){
				sh_tableQuickSynId[i+threadIdx.x]=gpu_tableQuickSynId[i+threadIdx.x];
			}
		}

		//global id
//		int gid=threadIdx.x + blockDim.x*blockIdx.x;

		__syncthreads();

		const int totBuffers=loadBufferCount;
		for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
			// KILLME !!! This can be further optimized ....
			// instead of reading each neuron group separately .....
			// read a whole buffer and use the result ......
			int2 threadLoad  = getStaticThreadLoad(bufPos);
			unsigned int  post_nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
			int  lastId      = STATIC_LOAD_SIZE(threadLoad);
			//int  grpId       = STATIC_LOAD_GROUP(threadLoad);

			// do some error checking...
			assertConductanceStates();

			//if ((threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {
			if ((threadIdx.x < lastId) && (IS_REGULAR_NEURON(post_nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois))) {

				// load the initial current due to noise inputs for neuron 'post_nid'
				// initial values of the conductances for neuron 'post_nid'
				float AMPA_sum   =  0.0;
				float NMDA_sum   =  0.0;
				float GABAa_sum   =  0.0;
				float GABAb_sum   =  0.0;
				int   lmt       =  gpuPtrs.Npre[post_nid];
				int   cum_pos   =  gpuPtrs.cumulativePre[post_nid];
				
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
						if (k==0) { cnt = cnt+1; continue; }
						int wt_i = sh_tableQuickSynId[k];
						int wtId = (j*32 + cnt*8 + wt_i);

						// load the synaptic weight for the wtId'th input
						float wt = gpuPtrs.wt[cum_pos + wtId];

						post_info_t pre_Id   = gpuPtrs.preSynapticIds[cum_pos + wtId];
						uint8_t  pre_grpId  = GET_CONN_GRP_ID(pre_Id);
						uint32_t  pre_nid  = GET_CONN_NEURON_ID(pre_Id);
						char type = gpuGrpInfo[pre_grpId].Type;

						// Adjust the weight according to STP scaling
						if(gpuGrpInfo[pre_grpId].WithSTP) {
							wt *= gpuPtrs.stpx[pre_nid]*gpuPtrs.stpu[pre_nid];
						}
						
						if (gpuNetInfo.sim_with_conductances) {
							if (type & TARGET_AMPA) AMPA_sum += wt;
							if (type & TARGET_NMDA) NMDA_sum += wt;
							if (type & TARGET_GABAa) GABAa_sum += wt; // wt should be negative for GABAa and GABAb, but that is delt with below
							if (type & TARGET_GABAb) GABAb_sum += wt;
						}
						else {
							// current based model with STP (CUBA)
							// updated current for neuron 'post_nid'
							AMPA_sum +=  wt;
						}

						tmp_I_cnt++;
						tmp_I_set = tmp_I_set & (~(1<<(8*cnt+wt_i)));
					}

					// reset the input if there are any bit'wt set
					if(tmp_I_cnt) *tmp_I_set_p = 0;

					__syncthreads();
				}

				__syncthreads();

				if (gpuNetInfo.sim_with_conductances) {
					gpuPtrs.gAMPA[post_nid]   += AMPA_sum;
					gpuPtrs.gNMDA[post_nid]   += NMDA_sum;
					gpuPtrs.gGABAa[post_nid]  -= GABAa_sum; // wt should be negative for GABAa and GABAb
					gpuPtrs.gGABAb[post_nid]  -= GABAb_sum;
				}
				else {
					gpuPtrs.current[post_nid] += AMPA_sum;
				}
			}
		}
	}

	//******************************** UPDATE GLOBAL STATE EVERY TIME STEP *********************************************************

	__device__ void updateNeuronState(unsigned int& nid, int& grpId)
	{
		float v      =  gpuPtrs.voltage[nid];
		float u      =  gpuPtrs.recovery[nid];
//		float I_inp     =  gpuPtrs.current[pos];
		float I_sum;

		// loop that allows smaller integration time step for v's and u's
		for (int c=0; c<COND_INTEGRATION_SCALE; c++) {
			I_sum = 0.0;
			if (gpuNetInfo.sim_with_conductances) {
				float NMDAtmp = (v+80)*(v+80)/60/60;
				// should we  add the noise current only one time or for the entire millisecond period ???
				I_sum = - ( gpuPtrs.gAMPA[nid]*(v-0)
					+ gpuPtrs.gNMDA[nid]*NMDAtmp/(1+NMDAtmp)*(v-0)
					+ gpuPtrs.gGABAa[nid]*(v+70)
					+ gpuPtrs.gGABAb[nid]*(v+90));
			}
			else
				I_sum = gpuPtrs.current[nid];

			// update vpos and upos for the current neuron
			v += ((0.04f*v+5)*v+140-u+I_sum)/COND_INTEGRATION_SCALE;
			if (v > 30)  { v = 30; c=COND_INTEGRATION_SCALE; }// break the loop but evaluate u[i]
			if (v < -90) v = -90;
			u += (gpuPtrs.Izh_a[nid]*(gpuPtrs.Izh_b[nid]*v-u)/COND_INTEGRATION_SCALE);
		}

		gpuPtrs.voltage[nid] 	     = v;
		gpuPtrs.recovery[nid] 	     = u;
	}

	__device__ inline int assertGlobalStates()
	{
		// error checking done here
		ERROR_CHECK_COND(ENABLE_MORE_CHECK, retErrCode, GLOBAL_STATE_ERROR_0, (blockIdx.x < MAX_NUM_BLOCKS), 0);
		return 0;
	}

	///////////////////////////////////////////////////////////
	/// Global Kernel function:      gpu_globalStateUpdate
	/// Description:
	//  change this with selective upgrading
	//  technique used for firing neurons
	///////////////////////////////////////////////////////////
	__global__ void kernel_globalStateUpdate (int t, int sec, int simTime)
	{
		const int totBuffers=loadBufferCount;

		for (int bufPos = blockIdx.x; bufPos < totBuffers; bufPos += gridDim.x) {
			// KILLME !!! This can be further optimized ....
			// instead of reading each neuron group separately .....
			// read a whole buffer and use the result ......
			int2 threadLoad  = getStaticThreadLoad(bufPos);
			unsigned int  nid        = (STATIC_LOAD_START(threadLoad) + threadIdx.x);
			int  lastId      = STATIC_LOAD_SIZE(threadLoad);
			int  grpId       = STATIC_LOAD_GROUP(threadLoad);

			// do some error checking...
			assertGlobalStates();

			if ((threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {

				if (IS_REGULAR_NEURON(nid,gpuNetInfo.numNReg, gpuNetInfo.numNPois)) {

					// update neuron state here....
					updateNeuronState(nid, grpId);
				}
			}
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
			if (gpuGrpInfo[grpId].WithConductances && IS_REGULAR_NEURON(nid, gpuNetInfo.numNReg, gpuNetInfo.numNPois)) {
				gpuPtrs.gAMPA[nid]   *=  gpuGrpInfo[grpId].dAMPA;
				gpuPtrs.gNMDA[nid]   *=  gpuGrpInfo[grpId].dNMDA;
				gpuPtrs.gGABAa[nid]  *=  gpuGrpInfo[grpId].dGABAa;
				gpuPtrs.gGABAb[nid]  *=  gpuGrpInfo[grpId].dGABAb;
			}

			// check various STP asserts here....
			assertSTPConditions();

			if (gpuGrpInfo[grpId].WithSTP && (threadIdx.x < lastId) && (nid < gpuNetInfo.numN)) {
				uint32_t ind   = getSTPBufPos(nid, simTime);
				uint32_t ind_1 = getSTPBufPos(nid, (simTime));

//				TODO: update this to more optimized floating multiplication rather than division.....
				gpuPtrs.stpx[ind] = gpuPtrs.stpx[ind_1] + (1-gpuPtrs.stpx[ind_1])/gpuGrpInfo[grpId].STP_tD;
				gpuPtrs.stpu[ind] = gpuPtrs.stpu[ind_1] + (gpuGrpInfo[grpId].STP_U - gpuPtrs.stpu[ind_1])/gpuGrpInfo[grpId].STP_tF;
			}
		}
	}

	__device__ void updateTimingTable()
	{
		int gnthreads=blockDim.x*gridDim.x;

		// Shift the firing table so that the initial information in
		// the firing table contain the firing information for the last D time step
		for(int p=timingTableD2[999],k=0;
			p<timingTableD2[999+gpuNetInfo.D+1];
			p+=gnthreads,k+=gnthreads) {
			if((p+threadIdx.x)<timingTableD2[999+gpuNetInfo.D+1])
				gpuPtrs.firingTableD2[k+threadIdx.x]=gpuPtrs.firingTableD2[p+threadIdx.x];
		}
	}

	//********************************UPDATE SYNAPTIC WEIGHTS EVERY SECOND  *************************************************************

	//////////////////////////////////////////////////////////////////
	/// Global Kernel function:      kernel_updateWeights_static   ///
	// KERNEL DETAILS:
	//   This kernel is called every second to adjust the timingTable and globalFiringTable
	//   We do the following thing:
	//   1. We discard all firing information that happened more than 1000-D time step.
	//   2. We move the firing information that happened in the last 1000-D time step to
	//      the begining of the gloalFiringTable.
	//   3. We read each value of "wtChange" and update the value of "synaptic weights wt".
	//      We also clip the "synaptic weight wt" to lie within the required range.
	//////////////////////////////////////////////////////////////////
	__device__ void updateSynapticWeights(int& nid, int& jpos, int& grpId)
	{
		float t_wt   	  = gpuPtrs.wt[jpos];
		float t_wtChange  = gpuPtrs.wtChange[jpos];
		float t_maxWt 	  = gpuPtrs.maxSynWt[jpos];

		t_wt += t_wtChange;

		//MDR - don't decay weight cahnges, just set to 0
		//t_wtChange*=0.90f;
		t_wtChange = 0;

		if (t_wt>t_maxWt) t_wt=t_maxWt;
		if (t_wt<0)  	  t_wt=0.0f;

		gpuPtrs.wt[jpos] = t_wt;
		gpuPtrs.wtChange[jpos] = t_wtChange;
	}


#define UPWTS_CLUSTERING_SZ	32
	__global__ void kernel_updateWeightsFiring_static()
	{
		//int gid=blockIdx.x*blockDim.x+threadIdx.x;
		//int gnthreads=blockDim.x*gridDim.x;
		__shared__ volatile int errCode;
		__shared__ int    		startId, lastId, grpId, totBuffers, grpNCnt;
		__shared__ int2 		threadLoad;

		if(threadIdx.x==0) {
			totBuffers=loadBufferCount;
			grpNCnt	= (blockDim.x/UPWTS_CLUSTERING_SZ) + ((blockDim.x%UPWTS_CLUSTERING_SZ)!=0);
		}

		updateTimingTable();

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
				int cumulativePre = gpuPtrs.cumulativePre[nid];

				const int threadIdGrp   = (threadIdx.x%UPWTS_CLUSTERING_SZ);
				// synaptic grouping
				for(int j=cumulativePre; j < (cumulativePre+Npre_plastic); j+=UPWTS_CLUSTERING_SZ) {
					//excitatory connection change the synaptic weights
					int jpos=j+threadIdGrp;
					if(jpos < (cumulativePre+Npre_plastic)) {
						//fprintf(fpProgLog,"%1.2f %1.2f \t", wt[offset+j]*10, wtChange[offset+j]*10);
//						if (nid == gpuGrpInfo[grpId].StartN)  {
//							if (testVarCnt < gpuNetInfo.numN) {
//								int val = atomicAdd(&testVarCnt, 2);
//								gpuPtrs.testVar[val]   = gpuPtrs.wt[jpos]*10;
//								gpuPtrs.testVar[val+1] = gpuPtrs.wtChange[jpos]*10;
//							}
//						}
						updateSynapticWeights(nid, jpos, grpId);
					}
				}
			}
		}
	}

	//********************************UPDATE TABLES AND COUNTERS EVERY SECOND  *************************************************************
	// KERNEL DESCRIPTION:
	// This is the second part of the previous kernel "kernel_updateWeightsFiring"
	// After all the threads/blocks had adjusted the firing table and the synaptic weights,
	// we update the timingTable so that the firing information that happended in the last D
	// time step would become the first D time step firing information for the next cycle of simulation.
	// We also reset/update various counters to appropriate values as indicated in the second part 
	// of this kernel.
	__global__ void kernel_updateWeightsFiring()
	{
		// CHECK !!!
		int D = gpuNetInfo.D;
		// reset the firing table so that we have the firing information
		// for the last D time steps to be used for the next cycle of the simulation
		if(blockIdx.x==0) {
			for(int i=threadIdx.x; i < D; i+=blockDim.x) {
				// use i+1 instead of just i because timingTableD2[0] should always be 0
				timingTableD2[i+1] = timingTableD2[1000+i+1]-timingTableD2[1000];
				timingTableD1[i+1] = timingTableD1[1000+i+1]-timingTableD1[1000];
			}
		}

		__syncthreads();

		// reset various counters for the firing information
		if((blockIdx.x==0)&&(threadIdx.x==0)) {
			timingTableD1[gpuNetInfo.D]  = 0;
			spikeCountD2	+= secD2fireCnt;
			spikeCountD1	+= secD1fireCnt;
			secD2fireCnt	= timingTableD2[gpuNetInfo.D];
			secD2fireCntTest	= timingTableD2[gpuNetInfo.D];
			secD1fireCnt	= 0;
			secD1fireCntTest = 0;
		}
	}

	/// THIS KERNEL IS USED BY BLOCK_CONFIG_VERSION
	__global__ void kernel_updateWeightsFiring2()
	{
		// reset various counters for the firing information
		if((blockIdx.x==0)&&(threadIdx.x==0)) {
//			timingTableD1[gpuNetInfo.D]  = 0;
			spikeCountD2	+= secD2fireCnt;
			spikeCountD1	+= secD1fireCnt;
			secD2fireCnt	= 0; //timingTableD2[gpuNetInfo.D];
			secD1fireCnt	= 0;
			secD2fireCntTest = 0; //timingTableD2[gpuNetInfo.D];
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
		int pos_ns = gpuPtrs.cumulativePre[nid] + syn_id;

//		uint16_t pre_grpId = GET_FIRING_TABLE_GID(firingId);
//		int pre_nid = GET_FIRING_TABLE_NID(firingId);

		// Error MNJ... this should have been from nid.. not firingId...
		// int  nid  = GET_FIRING_TABLE_NID(firingId);
		int    post_grpId;		// STP uses pre_grpId, STDP used post_grpId...
		findGrpId_GPU(nid, post_grpId);

		if(post_grpId == -1)
			return CURRENT_UPDATE_ERROR4;

		if(ENABLE_MORE_CHECK) {
			if (nid >= gpuNetInfo.numN)       	 errCode = (CURRENT_UPDATE_ERROR1|unitDelay);
			if (syn_id >= gpuPtrs.Npre[nid]) errCode = (CURRENT_UPDATE_ERROR2|unitDelay);
			if (errCode) return errCode;
		}

		setFiringBitSynapses(nid, syn_id);

		gpuPtrs.synSpikeTime[pos_ns] = simTime;		  //uncoalesced access

		if (gpuGrpInfo[post_grpId].WithSTDP)  {
			int stdp_tDiff = simTime-gpuPtrs.lastSpikeTime[nid];
			if ((stdp_tDiff >= 0) && ((stdp_tDiff*gpuGrpInfo[post_grpId].TAU_LTD_INV)<25)) {
				gpuPtrs.wtChange[pos_ns] -= STDP( stdp_tDiff, gpuGrpInfo[post_grpId].ALPHA_LTD, gpuGrpInfo[post_grpId].TAU_LTD_INV); // uncoalesced access
				// gpuPtrs.wtChange[pos_ns] -= GPU_LTD(stdp_tDiff);	  //uncoalesced access
//				int val = atomicAdd(&testVarCnt, 4);
//				gpuPtrs.testVar[val]   = 1+nid;
//				gpuPtrs.testVar[val+1] = 1+syn_id;
//				gpuPtrs.testVar[val+2] = 1+gpuPtrs.wtChange[pos_ns];
//				gpuPtrs.testVar[val+3] = 1+stdp_tDiff;
			}
		}
	
		return errCode;
	}

	__device__ void CHECK_DELAY_ERROR (int& t_pos, volatile int& sh_blkErrCode)
	{
		if(ENABLE_MORE_CHECK) {
			if (!((t_pos+(int)gpuNetInfo.D) >= 0)) {
				int i=2;
				sh_blkErrCode = CURRENT_UPDATE_ERROR3;
				retErrVal[blockIdx.x][0] = 0xdead;
				/* retErrVal[blockIdx.x][i++] = simTimeMs;
				retErrVal[blockIdx.x][i++] = t_pos;
				retErrVal[blockIdx.x][i++] = fPos;
				retErrVal[blockIdx.x][i++] = tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.D-1 + timingTableD2_tex_offset);
				retErrVal[blockIdx.x][i++] = tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.D + timingTableD2_tex_offset);*/
				retErrVal[blockIdx.x][1]   = i;
			}
		}
	}

	#define NUM_THREADS 			128
	#define EXCIT_READ_CHUNK_SZ		(NUM_THREADS>>1)

   	//  KERNEL DESCRIPTION:-
	//  This kernel is required for updating and generating spikes for delays greater than 1 from the fired neuron. 
	//  The LTD computation is also executed by this approach kernel.
	__global__ void gpu_doCurrentUpdateD2(int simTimeMs, int simTimeSec, int simTime)
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

		// stores the number of fired neurons at time (t - D)
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
					while ( !((fPos >= tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.D+timingTableD2_tex_offset)) 
						&& (fPos <  tex1Dfetch(timingTableD2_tex, t_pos+gpuNetInfo.D+1+timingTableD2_tex_offset)))) {
						t_pos = t_pos - 1;
						CHECK_DELAY_ERROR(t_pos, sh_blkErrCode);
					}

					// find the time difference between firing of the neuron and the current time
					int tD  = simTimeMs - t_pos;

					// find the various delay parameters for neuron 'nid', with a delay of 'tD'
					//sh_axonDelay[threadIdx.x]	 = tD;
					int tPos = (gpuNetInfo.D+1)*nid+tD;
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
			// needs to generate (numPostSynapses/D) spikes for every fired neuron, every second
			// for numPostSynapses=500,D=20, we need to generate 25 spikes for each fired neuron
			// for numPostSynapses=600,D=20, we need to generate 30 spikes for each fired neuron 
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
	__global__ void gpu_doCurrentUpdateD1(int simTimeMs, int simTimeSec, int simTime)
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
			sh_timing = timingTableD1[simTimeMs+gpuNetInfo.D]; // ??? check check ???
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
					int tPos = (gpuNetInfo.D+1)*nid;
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
	int CpuSNN::checkCudaErrors(string calledKernel, int numBlocks)
	{
	   int errCode = NO_KERNEL_ERRORS;
	   #if(!ENABLE_MORE_CHECK)
		   return errCode;
		#else
	   void* devPtr;
	   int errCnt  = 0;
	   
	   
	   cudaThreadSynchronize();
	   cudaGetSymbolAddress(&devPtr, retErrCode);
	   cutilSafeCall( cudaMemcpy(&errCode, devPtr, sizeof(int), cudaMemcpyDeviceToHost));

	   cudaGetSymbolAddress(&devPtr, generatedErrors);
   	   cutilSafeCall( cudaMemcpy(&errCnt, devPtr, sizeof(int), cudaMemcpyDeviceToHost));

	   if (errCode != NO_KERNEL_ERRORS) {

		  fprintf(stderr, "\n (Error in Kernel <<< %s >>> , RETURN ERROR CODE = %x, total Errors = %d\n", calledKernel.c_str(), errCode, errCnt);

		  cudaGetSymbolAddress(&devPtr, "retErrVal");
		  cutilSafeCall( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));

		  for(int j=0; j < errCnt; j++) {
			  if (1) /*errVal[j][0]==0xdead) */ {
				  fprintf(stderr, "Block: %d, Err code = %x, Total err val is %f\n", j, (int)errVal[j][0]	, errVal[j][1]);
				  for(int i=2; i < (errVal[j][1]); i++) {
					  fprintf(stderr, "ErrVal[%d][%d] = %f\n", j, i, errVal[j][i]);
					  getchar();
				  }
			  }
		  }

		   errCode = NO_KERNEL_ERRORS;
		   cudaGetSymbolAddress(&devPtr, retErrCode);
		   cutilSafeCall( cudaMemcpy(devPtr, &errCode, sizeof(int), cudaMemcpyHostToDevice));

		   cudaGetSymbolAddress(&devPtr, generatedErrors);
	   	   cutilSafeCall( cudaMemcpy(devPtr, &errCnt, sizeof(int), cudaMemcpyHostToDevice));

	   }

	   fflush(stderr);
	   fflush(stdout);
	   return errCode;
	#endif
	}

	void CpuSNN::copyPostConnectionInfo(network_ptr_t* dest, int allocateMem)
	{
		assert(dest->memType == GPU_MODE);
		if (allocateMem) {
			assert(dest->allocated==false);
		}
		else {
			assert(dest->allocated == true);
		}
		assert(doneReorganization == true);

		// beginning position for the post-synaptic information
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->cumulativePost, sizeof(cumulativePost[0])*numN));
		cutilSafeCall( cudaMemcpy( dest->cumulativePost, cumulativePost, sizeof(int)*numN, cudaMemcpyHostToDevice));

		// number of postsynaptic connections
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Npost, sizeof(Npost[0])*numN));
		cutilSafeCall( cudaMemcpy( dest->Npost, Npost, sizeof(Npost[0])*numN, cudaMemcpyHostToDevice));

		// static specific mapping and actual post-synaptic delay metric
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->postDelayInfo, sizeof(postDelayInfo[0])*numN*(D+1)));
		cutilSafeCall( cudaMemcpy( dest->postDelayInfo, postDelayInfo, sizeof(postDelayInfo[0])*numN*(D+1), cudaMemcpyHostToDevice));

		// actual post synaptic connection information...
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->postSynapticIds, sizeof(postSynapticIds[0])*(postSynCnt+10)));
		cutilSafeCall( cudaMemcpy( dest->postSynapticIds, postSynapticIds, sizeof(postSynapticIds[0])*(postSynCnt+10), cudaMemcpyHostToDevice));
		net_Info.postSynCnt = postSynCnt;

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->preSynapticIds, sizeof(preSynapticIds[0])*(preSynCnt+10)));
		cutilSafeCall( cudaMemcpy( dest->preSynapticIds, preSynapticIds, sizeof(preSynapticIds[0])*(preSynCnt+10), cudaMemcpyHostToDevice));
		net_Info.preSynCnt = preSynCnt;
	}

	void CpuSNN::copyConnections(network_ptr_t* dest, int kind, int allocateMem)
	{
		// void* devPtr;
		// allocateMem memory only if destination memory is not allocated !!!
		assert(allocateMem && (dest->allocated != 1));
		if(kind==cudaMemcpyHostToDevice) {
			assert(dest->memType == GPU_MODE);
		}
		else {
			assert(dest->memType == CPU_MODE);
		}

		net_Info.I_setLength = ceil(((numPreSynapses)/32.0));
		if(allocateMem)
			cudaMallocPitch( (void**) &dest->I_set, &net_Info.I_setPitch, sizeof(int)*numNReg, net_Info.I_setLength);
		assert(net_Info.I_setPitch > 0);
		cutilSafeCall( cudaMemset( dest->I_set, 0, net_Info.I_setPitch*net_Info.I_setLength));

	#if TESTING
		fprintf(stdout, "numNReg=%d numPostSynapses = %d, I_set = %x, I_setPitch = %d,  I_setLength = %d\n", numNReg, numPostSynapses, dest->I_set, net_Info.I_setPitch, net_Info.I_setLength);
	#endif

		// connection synaptic lengths and cumulative lengths...
		if(allocateMem)   	cutilSafeCall( cudaMalloc((void**) &dest->Npre, sizeof(dest->Npre[0])*numN));
		cutilSafeCall( cudaMemcpy( dest->Npre, Npre, sizeof(dest->Npre[0])*numN, cudaMemcpyHostToDevice));

		// presyn excitatory connections
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Npre_plastic, sizeof(dest->Npre_plastic[0])*numN));
		cutilSafeCall( cudaMemcpy( dest->Npre_plastic, Npre_plastic, sizeof(dest->Npre_plastic[0])*numN, cudaMemcpyHostToDevice));

		float* Npre_plasticInv = new float[numN];
		for (int i=0;i<numN;i++) Npre_plasticInv[i] = 1.0/Npre_plastic[i];

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Npre_plasticInv, sizeof(dest->Npre_plasticInv[0])*numN));
		cutilSafeCall( cudaMemcpy( dest->Npre_plasticInv, Npre_plasticInv, sizeof(dest->Npre_plasticInv[0])*numN, cudaMemcpyHostToDevice));

		delete[] Npre_plasticInv;

		// beginning position for the pre-synaptic information
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->cumulativePre, sizeof(int)*numN));
		cutilSafeCall( cudaMemcpy( dest->cumulativePre, cumulativePre, sizeof(int)*numN, cudaMemcpyHostToDevice));

		// allocate randomPtr.... containing the firing information for the random firing neurons...
//		if(allocateMem)  cutilSafeCall( cudaMalloc( (void**) &dest->randId, sizeof(int)*numN));
//		net_Info.numRandNeurons=numRandNeurons;

		// copy the properties of the noise generator here.....
//		if(allocateMem) cutilSafeCall( cudaMalloc( (void**) &dest->noiseGenProp, sizeof(noiseGenProperty_t)*numNoise));
//		cutilSafeCall( cudaMemcpy( dest->noiseGenProp, &noiseGenGroup[0], sizeof(noiseGenProperty_t)*numNoise, cudaMemcpyHostToDevice));

		// allocate the poisson neuron poissonFireRate
		if(allocateMem) cutilSafeCall( cudaMalloc( (void**) &dest->poissonFireRate, sizeof(dest->poissonFireRate[0])*numNPois));
		cutilSafeCall( cudaMemset( dest->poissonFireRate, 0, sizeof(dest->poissonFireRate[0])*numNPois));

		// neuron firing recently or not...
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->neuronFiring, sizeof(int)*numN));
		cutilSafeCall( cudaMemset( dest->neuronFiring, 0, sizeof(int)*numN));

		copyPostConnectionInfo(dest, allocateMem);

		// neuron testing
		testVar  = new float[numN];
		testVar2 = new float[numN];
		cpuSnnSz.addInfoSize += sizeof(float)*numN*2;

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->testVar, sizeof(float)*numN));
		cutilSafeCall( cudaMemset( dest->testVar, 0, sizeof(float)*numN));

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->testVar2, sizeof(float)*numN));
		cutilSafeCall( cudaMemset( dest->testVar2, 0, sizeof(float)*numN));
	}

	void CpuSNN::checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId)
	{
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

	void CpuSNN::copyFiringStateFromGPU (int grpId)
	{
		int ptrPos, length, length2;

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

		network_ptr_t* dest = &cpuNetPtrs;
		network_ptr_t* src  = &cpu_gpuNetPtrs;
		cudaMemcpyKind kind = cudaMemcpyDeviceToHost;

		// Spike Cnt. Firing...
		cutilSafeCall( cudaMemcpy( &dest->nSpikeCnt[ptrPos], &src->nSpikeCnt[ptrPos], sizeof(int)*length2, kind));

	}

	void CpuSNN::copyNeuronState(network_ptr_t* dest, network_ptr_t* src,  cudaMemcpyKind kind, int allocateMem, int grpId)
	{
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
		if (allocateMem) cutilSafeCall( cudaMalloc( (void**) &dest->nSpikeCnt, sizeof(int)*length2));
		cutilSafeCall( cudaMemcpy( &dest->nSpikeCnt[ptrPos], &src->nSpikeCnt[ptrPos], sizeof(int)*length2, kind));

		if( !allocateMem && grp_Info[grpId].Type&POISSON_NEURON)
			return;

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->recovery, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->recovery[ptrPos], &src->recovery[ptrPos], sizeof(float)*length, kind));

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->voltage, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->voltage[ptrPos], &src->voltage[ptrPos], sizeof(float)*length, kind));

		if (sim_with_conductances) {
			//conductance information
			assert(src->gGABAa != NULL);
			assert(src->gGABAb != NULL);
			assert(src->gNMDA  != NULL);
			assert(src->gAMPA  != NULL);

			if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->gGABAa, sizeof(float)*length));
			cutilSafeCall( cudaMemcpy( &dest->gGABAa[ptrPos], &src->gGABAa[ptrPos], sizeof(float)*length, kind));

			if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->gGABAb, sizeof(float)*length));
			cutilSafeCall( cudaMemcpy( &dest->gGABAb[ptrPos], &src->gGABAb[ptrPos], sizeof(float)*length, kind));

			if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->gAMPA, sizeof(float)*length));
			cutilSafeCall( cudaMemcpy( &dest->gAMPA[ptrPos], &src->gAMPA[ptrPos], sizeof(float)*length, kind));

			if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->gNMDA, sizeof(float)*length));
			cutilSafeCall( cudaMemcpy( &dest->gNMDA[ptrPos], &src->gNMDA[ptrPos], sizeof(float)*length, kind));
		}

		//neuron input current...
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->current, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->current[ptrPos], &src->current[ptrPos], sizeof(float)*length, kind));
	}

	void CpuSNN::copyNeuronParameters(network_ptr_t* dest, int kind, int allocateMem, int grpId)
	{
		int ptrPos, length;

		if (dest->allocated && allocateMem) {
			fprintf(stderr, "GPU Memory already allocated.. \n");
			return;
		}

		// for neuron parameter the copy is one-directional...
		assert(kind==cudaMemcpyHostToDevice);

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
		if(allocateMem)     assert(dest->Izh_a == NULL);

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Izh_a, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->Izh_a[ptrPos], &Izh_a[ptrPos], sizeof(float)*length, cudaMemcpyHostToDevice));

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Izh_b, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->Izh_b[ptrPos], &Izh_b[ptrPos], sizeof(float)*length, cudaMemcpyHostToDevice));

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Izh_c, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->Izh_c[ptrPos], &Izh_c[ptrPos], sizeof(float)*length, cudaMemcpyHostToDevice));

		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->Izh_d, sizeof(float)*length));
		cutilSafeCall( cudaMemcpy( &dest->Izh_d[ptrPos], &Izh_d[ptrPos], sizeof(float)*length, cudaMemcpyHostToDevice));
	}

	void assertSTPState(network_ptr_t* dest, network_ptr_t* src, int kind, int allocateMem)
	{
		if(allocateMem) {
			assert(dest->stpu==NULL);
			assert(dest->stpx==NULL);
		}
		else {
			assert(dest->stpu != NULL);
			assert(dest->stpx != NULL);
		}

		assert(src->stpu != NULL);
		assert(src->stpx != NULL);
	}

	void CpuSNN::copySTPState(network_ptr_t* dest, network_ptr_t* src, int kind, int allocateMem)
	{
		assert(stpu != NULL);
		assert(stpx != NULL);

		size_t STP_Pitch;
		size_t widthInBytes = sizeof(float)*net_Info.numN;

		assertSTPState(dest,src, kind,allocateMem);

		// allocate the stpu and stpx variable
		if (allocateMem)
			cutilSafeCall( cudaMallocPitch ((void**) &dest->stpu, &net_Info.STP_Pitch, widthInBytes, STP_BUF_SIZE));
		if (allocateMem)
			cutilSafeCall( cudaMallocPitch ((void**) &dest->stpx, &STP_Pitch, widthInBytes, STP_BUF_SIZE));

		assert(net_Info.STP_Pitch > 0);
		assert(STP_Pitch > 0);				// stp_pitch should be greater than zero
		assert(STP_Pitch == net_Info.STP_Pitch);	// we want same Pitch for stpu and stpx
		assert(net_Info.STP_Pitch >= widthInBytes);	// stp_pitch should be greater than the width
		// convert the Pitch value to multiples of float
		assert(net_Info.STP_Pitch % (sizeof(float)) == 0);
		if (allocateMem) net_Info.STP_Pitch = net_Info.STP_Pitch/sizeof(float);

	#if TESTING
		fprintf(stdout, "STP_Pitch = %d, STP_witdhInBytes = %d\n", net_Info.STP_Pitch, widthInBytes);
	#endif

		float* tmp_stp = new float[net_Info.numN];
		// copy the already generated values of stpx and stpu to the GPU
		for(int t=0; t < STP_BUF_SIZE; t++) {
			if (kind==cudaMemcpyHostToDevice) {
				// stpu in the CPU might be mapped in a specific way. we want to change the format
				// to something that is okay with the GPU STP_U and STP_X variable implementation..
				for (int n=0; n < net_Info.numN; n++) {
					tmp_stp[n]=stpu[STP_BUF_POS(n,t)];
					assert(tmp_stp[n] != 0);
				}
				cutilSafeCall( cudaMemcpy( &dest->stpu[t*net_Info.STP_Pitch], tmp_stp, sizeof(float)*net_Info.numN, cudaMemcpyHostToDevice));
				for (int n=0; n < net_Info.numN; n++) {
					tmp_stp[n]=stpx[STP_BUF_POS(n,t)];
					assert(tmp_stp[n] != 0);
				}
				cutilSafeCall( cudaMemcpy( &dest->stpx[t*net_Info.STP_Pitch], tmp_stp, sizeof(float)*net_Info.numN, cudaMemcpyHostToDevice));
			}
			else {
				cutilSafeCall( cudaMemcpy( tmp_stp, &dest->stpu[t*net_Info.STP_Pitch], sizeof(float)*net_Info.numN, cudaMemcpyDeviceToHost));
				for (int n=0; n < net_Info.numN; n++)
					stpu[STP_BUF_POS(n,t)]=tmp_stp[n];
				cutilSafeCall( cudaMemcpy( tmp_stp, &dest->stpx[t*net_Info.STP_Pitch], sizeof(float)*net_Info.numN, cudaMemcpyDeviceToHost));
				for (int n=0; n < net_Info.numN; n++)
					stpx[STP_BUF_POS(n,t)]=tmp_stp[n];
			}
		}
		delete [] tmp_stp;
	}

	void CpuSNN::copyWeightState (network_ptr_t* dest, network_ptr_t* src,  cudaMemcpyKind kind, int allocateMem, int grpId)
	{
		int length_wt, cumPos_syn;

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
			}
			else {
				int id = grp_Info[grpId].StartN + i;
				length_wt 	= dest->Npre[id];
				cumPos_syn 	= dest->cumulativePre[id];
			}

			assert (cumPos_syn < preSynCnt);
			assert (cumPos_syn >=  0);

			assert (length_wt <= preSynCnt);
			assert (length_wt >= 0);
//MDR FIXME, allocateMem option is VERY wrong
			// synaptic information based
			
//			if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->wt, sizeof(float)*length_wt));
			cutilSafeCall( cudaMemcpy( &dest->wt[cumPos_syn], &src->wt[cumPos_syn], sizeof(float)*length_wt,  kind));

			// synaptic weight derivative
//			if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*length_wt));
			cutilSafeCall( cudaMemcpy( &dest->wtChange[cumPos_syn], &src->wtChange[cumPos_syn], sizeof(float)*length_wt, kind));

			// firing time for individual synapses
//			if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*length_wt));
			cutilSafeCall( cudaMemcpy( &dest->synSpikeTime[cumPos_syn], &src->synSpikeTime[cumPos_syn], sizeof(int)*length_wt, kind));
		}
	}

	// allocate necessary memory for the GPU...
	void CpuSNN::copyState(network_ptr_t* dest, int kind, int allocateMem)
	{
		assert(numN != 0);
		assert(preSynCnt !=0);

		if (dest->allocated && allocateMem) {
			fprintf(stderr, "GPU Memory already allocated.. \n");
			return;
		}

		// synaptic information based
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->wt, sizeof(float)*preSynCnt));
		cutilSafeCall( cudaMemcpy( dest->wt, wt, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));

		// synaptic weight derivative
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->wtChange, sizeof(float)*preSynCnt));
		cutilSafeCall( cudaMemcpy( dest->wtChange, wtChange, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));

		// firing time for individual synapses
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->synSpikeTime, sizeof(int)*preSynCnt));
		cutilSafeCall( cudaMemcpy( dest->synSpikeTime, synSpikeTime, sizeof(int)*preSynCnt, cudaMemcpyHostToDevice));
		net_Info.preSynLength = preSynCnt;

		// synaptic weight maximum value...
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->maxSynWt, sizeof(float)*preSynCnt));
		cutilSafeCall( cudaMemcpy( dest->maxSynWt, maxSynWt, sizeof(float)*preSynCnt, cudaMemcpyHostToDevice));

		assert(net_Info.maxSpikesD1 != 0);
		if(allocateMem) {
			assert(dest->firingTableD1 == NULL);
			assert(dest->firingTableD2 == NULL);
		}

		// allocate 1ms firing table
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->firingTableD1, sizeof(int)*net_Info.maxSpikesD1));
		if (net_Info.maxSpikesD1>0) cutilSafeCall( cudaMemcpy( dest->firingTableD1, firingTableD1, sizeof(int)*net_Info.maxSpikesD1, cudaMemcpyHostToDevice));

		// allocate 2+ms firing table
		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->firingTableD2, sizeof(int)*net_Info.maxSpikesD2));
		if (net_Info.maxSpikesD2>0) cutilSafeCall( cudaMemcpy( dest->firingTableD2, firingTableD2, sizeof(int)*net_Info.maxSpikesD2, cudaMemcpyHostToDevice));

		// neuron firing time..
		if(allocateMem)     cutilSafeCall( cudaMalloc( (void**) &dest->lastSpikeTime, sizeof(int)*numNReg));
		cutilSafeCall( cudaMemcpy( dest->lastSpikeTime, lastSpikeTime, sizeof(int)*numNReg, cudaMemcpyHostToDevice));

		if(allocateMem)		cutilSafeCall( cudaMalloc( (void**) &dest->spikeGenBits, sizeof(int)*(NgenFunc/32+1)));

		// copy the neuron state information to the GPU..
		copyNeuronState(dest, &cpuNetPtrs, cudaMemcpyHostToDevice, allocateMem);

		copyNeuronParameters(dest, cudaMemcpyHostToDevice, allocateMem);

		if (sim_with_stp) {
			copySTPState(dest, &cpuNetPtrs, cudaMemcpyHostToDevice, allocateMem);
		}
	}

	// spikeGeneratorUpdate on GPUs..
	void CpuSNN::spikeGeneratorUpdate_GPU()
	{
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
			CUDA_SAFE_CALL( cudaMemcpy( cpu_gpuNetPtrs.spikeGenBits, cpuNetPtrs.spikeGenBits, sizeof(int)*(NgenFunc/32+1), cudaMemcpyHostToDevice));
		}
	}
	
	void CpuSNN::findFiring_GPU()
	{
		DBG(2, fpLog, AT, "gpu_findFiring()");

		int blkSize  = 128;
		int gridSize = 64;
		int errCode;
		
		assert(cpu_gpuNetPtrs.allocated);
//		if(cpu_gpuNetPtrs.allocated == false)  
//			allocateSNN_GPU();

		//checkInitialization2("checking STP setting before firing : ");

		if (sim_with_stp || sim_with_conductances) {
			kernel_STPUpdateAndDecayConductances <<<gridSize, blkSize>>>(simTimeMs, simTimeSec, simTime);
			cutilCheckMsg("STP update\n");
			errCode = checkCudaErrors("gpu_STPUpdate", gridSize);
			assert(errCode == NO_KERNEL_ERRORS);
		}

		spikeGeneratorUpdate_GPU();

		// printTestVarInfo(fpLog, "Testing STP :", true, true, false, 0, 2, 7);

		kernel_findFiring <<<gridSize,blkSize >>> (simTimeMs, simTimeSec, simTime);
		cutilCheckMsg("findFiring kernel failed\n");
		errCode = checkCudaErrors("kernel_findFiring", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);


		//printTestVarInfo(stderr, "LTP", true, false, false, 1, 4, 0);

		if(MEASURE_LOADING) printGpuLoadBalance(false);

		return;
	}

	void CpuSNN::printGpuLoadBalance(bool init, int numBlocks, FILE*fp)
	{
		void* devPtr;

		static int	 cpu_tmp_val[MAX_BLOCKS][LOOP_CNT];

		cudaGetSymbolAddress(&devPtr, "tmp_val");

		if(init) {
			// reset the load balance variable..
			cutilSafeCall( cudaMemset(devPtr, 0, sizeof(tmp_val)));
			return;
		}

		cutilSafeCall( cudaMemcpy(&cpu_tmp_val, devPtr, sizeof(cpu_tmp_val), cudaMemcpyDeviceToHost));
		cutilSafeCall( cudaMemset(devPtr, 0, sizeof(tmp_val)));

		fprintf(fp, "GPU Load Balancing Information\n");

		for (int i=0; i < numBlocks; i++) {
			if (cpu_tmp_val[i][0] != 0)
				fprintf(fp, "[%d] Fired Neuron = %d \n", i, cpu_tmp_val[i][1]+cpu_tmp_val[i][2]);
		}

		fprintf(fp, "\n");

		fflush(fp);

	}

	// copy the spike from the GPU to the CPU..
	void CpuSNN::updateSpikeMonitor_GPU()
	{
		// copy the neuron firing information from the GPU to the CPU..
		copyFiringInfo_GPU();
	}

	void CpuSNN::updateTimingTable_GPU()
	{
		DBG(2, fpLog, AT, "gpu_updateTimingTable()");

		assert(cpu_gpuNetPtrs.allocated);
//		if(cpu_gpuNetPtrs.allocated == false)
//			allocateSNN_GPU();

		int blkSize  = 128;
		int gridSize = 64;
		kernel_timingTableUpdate <<<gridSize,blkSize >>> (simTimeMs);
		cutilCheckMsg("timing Table update kernel failed\n");
		int errCode = checkCudaErrors("kernel_timingTableUpdate", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);

		//printTestVarInfo(stderr, true, true, true);
		//printFiredId(stderr, false, 2);
		//getchar();

		return;
	}

	void CpuSNN::doCurrentUpdate_GPU()
	{
		DBG(2, fpLog, AT, "gpu_doCurrentUpdate()");

		assert(cpu_gpuNetPtrs.allocated);
//		if(cpu_gpuNetPtrs.allocated == false)
//			allocateSNN_GPU();

		int blkSize  = 128;
		int gridSize = 64;
		int errCode;

		if(D > 1) {
			gpu_doCurrentUpdateD2 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
			CUT_CHECK_ERROR("Kernel execution failed");
			errCode = checkCudaErrors("kernel_updateCurrentE", gridSize);
			assert(errCode == NO_KERNEL_ERRORS);
		}

		// printTestVarInfo(stderr, "Variable Delay", true, false, false, 0);

		gpu_doCurrentUpdateD1 <<<gridSize, blkSize>>>(simTimeMs,simTimeSec,simTime);
		CUT_CHECK_ERROR("Kernel execution failed");	
		errCode = checkCudaErrors("kernel_updateCurrentI", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);

		//printTestVarInfo(stderr, "LTD", true, false, false, 0, 3, 0);
	}

	__device__ float getSTPScaleFactor (int& nid, const int& simTime, int& del)
	{
		// Get the STP scaling factor
		// Read stpu and stpx. Generate stp_scale after re-evaluation
		uint32_t ind   = getSTPBufPos(nid, (unsigned)(simTime-del));
		//(((((unsigned)(simTime-tD))%STP_BUF_SIZE)*gpuNetInfo.STP_Pitch) + id);
		return (gpuPtrs.stpx[ind])*(gpuPtrs.stpu[ind]);
	}

	__global__ void kernel_check_GPU_init2 (int simTime)
	{
//	  	int gid=threadIdx.x + blockDim.x*blockIdx.x;
	  	int nid[]   = {0,1,2,3};

	  	for(int k=0; k < 4; k++) {
	  		int i=1;
			for(int t=0; t < 5; t++) {
				float stp = getSTPScaleFactor(nid[k], simTime, t);
				int buf = getSTPBufPos(nid[k], simTime-t);
				retErrVal[k][i++]= stp;
				retErrVal[k][i++]= buf;
			}
		  	retErrVal[k][0] = i;
	  	}
	}

	__global__ void kernel_check_GPU_init2_1 (int simTime)
	{
//	  	int gid=threadIdx.x + blockDim.x*blockIdx.x;
	  	int nid=0;
	  	for(int k=0; k < gpuNetInfo.numN; k++) {
			if ( gpuPtrs.neuronFiring[k]) {
				int i=1;
//				int pos = atomicAdd(&testVarCnt2, 7);
//				gpuPtrs.testVar2[pos]   = nid;
//				gpuPtrs.testVar2[pos+1] = ind_1;
//				gpuPtrs.testVar2[pos+2] = gpuPtrs.stpu[ind_1];
//				gpuPtrs.testVar2[pos+3] = gpuPtrs.stpx[ind_1];
//				gpuPtrs.testVar2[pos+4] = gpuPtrs.stpu[ind];
//				gpuPtrs.testVar2[pos+5] = gpuPtrs.stpx[ind];
//				gpuPtrs.testVar2[pos+6] = gpuPtrs.stpu[ind]*gpuPtrs.stpx[ind];
				retErrVal[nid][i++]= k;
				for(int t=0; t < 5; t++) {
					float stp = getSTPScaleFactor(k, simTime, t);
					int buf = getSTPBufPos(k, simTime-t);
					retErrVal[nid][i++]= stp;
					retErrVal[nid][i++]= buf;
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
        		retErrVal[0][i++]= gpuNetInfo.sim_with_conductances;
        		retErrVal[0][i++]= gpuGrpInfo[0].MonitorId;
        		retErrVal[0][i++]= gpuGrpInfo[1].MonitorId;
        		retErrVal[0][i++]= gpuGrpInfo[0].WithSTP;
        		retErrVal[0][i++]= gpuGrpInfo[1].WithSTP;
        		retErrVal[0][i++]= 123456789.0;
        		retErrVal[0][0]  = i;
        		i = 1;
        		retErrVal[1][i++]= tex1Dfetch(timingTableD2_tex, 0+timingTableD2_tex_offset);
        		retErrVal[1][i++]= tex1Dfetch(timingTableD2_tex, 1+timingTableD2_tex_offset);
        		retErrVal[1][i++]= loadBufferCount;
        		retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 0);
        		retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 1);
        		retErrVal[1][i++]= tex1Dfetch(groupIdInfo_tex, 2);
        		unsigned int id = 124;
        		int grpId; findGrpId_GPU(id, grpId);
        		retErrVal[1][i++]= gpuGrpInfo[0].WithSTDP;
        		retErrVal[1][i++]= gpuGrpInfo[1].WithSTDP;
        		retErrVal[1][i++]= gpuGrpInfo[2].WithSTDP;
        		retErrVal[1][i++]= STATIC_LOAD_START(gpuPtrs.neuronAllocation[0]);
        		retErrVal[1][i++]= STATIC_LOAD_GROUP(gpuPtrs.neuronAllocation[0]);
        		retErrVal[1][i++]= STATIC_LOAD_SIZE(gpuPtrs.neuronAllocation[0]);
        		retErrVal[1][i++]= gpuNetInfo.STP_Pitch;
        		retErrVal[1][i++]= gpuGrpInfo[2].FixedInputWts;
        		retErrVal[1][i++]= gpuGrpInfo[0].STP_U;
        		retErrVal[1][i++]= gpuGrpInfo[0].STP_tD;
        		retErrVal[1][i++]= gpuGrpInfo[0].STP_tF;
        		uint32_t ind     = getSTPBufPos(0, 1); //(((simTime%STP_BUF_SIZE)*gpuNetInfo.STP_Pitch) + nid);
        		retErrVal[1][i++]= gpuPtrs.stpu[ind];
        		retErrVal[1][i++]= gpuPtrs.stpx[ind];
        		retErrVal[1][i++]= 123456789.0;
        		retErrVal[1][0]  = i;
        		return;
        	}
    }

	void CpuSNN::initGPU(int gridSize, int blkSize)
	{
		DBG(2, fpLog, AT, "gpu_initGPU()");

		assert(cpu_gpuNetPtrs.allocated);
//		if(cpu_gpuNetPtrs.allocated == false)  
//			allocateSNN_GPU();

		kernel_init <<< gridSize, blkSize >>> ();
		cutilCheckMsg("initGPU kernel failed\n");
		int errCode = checkCudaErrors("kernel_init", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);

		printGpuLoadBalance(true);

		checkInitialization();

		checkInitialization2();
	}

	void CpuSNN::checkInitialization(char* testString)
	{
		DBG(2, fpLog, AT, "gpu_checkInitialization()");
		void *devPtr;

		assert(cpu_gpuNetPtrs.allocated);
//		if(cpu_gpuNetPtrs.allocated == false)  
//			allocateSNN_GPU();

		memset(errVal, 0, sizeof(errVal));
		cudaGetSymbolAddress(&devPtr, retErrVal);
		cutilSafeCall( cudaMemcpy(devPtr, &errVal, sizeof(errVal), cudaMemcpyHostToDevice));

		int testTable[10];
		// we write some number in this table..
		// write that to the GPU memory. Read it back
		// using the texture access and check if everything is correctly initialized
		testTable[0] = 11; testTable[1] = 12; testTable[2] = 13;
		testTable[3] = 14; testTable[4] = 15; testTable[5] = 16;
		cutilSafeCall( cudaMemcpyToSymbol("timingTableD2", testTable, sizeof(int)*(10), 0, cudaMemcpyHostToDevice));
//MDR this check fails because it assumes too much about the network...
//		kernel_check_GPU_init <<< 1, 128 >>> ();
		cutilCheckMsg("check GPU failed\n");

		// printTestVarInfo(stderr);
		// read back the intialization and ensure that they are okay
		fprintf(fpLog, "%s Checking initialization of GPU...\n", testString?testString:"");
		cutilSafeCall( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));
		for(int i=0; i < 4; i++) {
			for(int j=0; j < (int)errVal[i][0]; j++) {
				fprintf(fpLog, "val[%3d][%3d] = %f\n", i, j, errVal[i][j]);
			}
			fprintf(fpLog, "******************\n");
		}
		fprintf(fpLog, "Checking done...\n");
		cutilSafeCall( cudaMemcpyToSymbol("timingTableD2", timeTableD2, sizeof(int)*(10), 0, cudaMemcpyHostToDevice));
		fflush(fpLog);
	}

	void CpuSNN::checkInitialization2(char* testString)
	{
		fprintf(fpLog, "CheckInitialization2: Time = %d\n", simTime);
		void *devPtr;
		memset(errVal, 0, sizeof(errVal));
		cudaGetSymbolAddress(&devPtr, retErrVal);
		cutilSafeCall( cudaMemcpy(devPtr, &errVal, sizeof(errVal), cudaMemcpyHostToDevice));

		if (sim_with_stp) {
			kernel_check_GPU_init2 <<< 1, 128>>> (simTime);
			cutilCheckMsg("kernel_check_GPU_init2_1 failed\n");
			int errCode = checkCudaErrors("kernel_init2_1", 1);
			assert(errCode == NO_KERNEL_ERRORS);
		}

		// read back the intialization and ensure that they are okay
		fprintf(fpLog, "%s Checking Initialization of STP Variable Correctly in GPU...\n", (testString?testString:""));
		cutilSafeCall( cudaMemcpy(&errVal, devPtr, sizeof(errVal), cudaMemcpyDeviceToHost));
		for(int i=0; i < 4; i++) {
			for(int j=0; j < (int)errVal[i][0]; j++) {
				fprintf(fpLog, "val[%3d][%3d] = %f\n", i, j, errVal[i][j]);
			}
			fprintf(fpLog, "******************\n");
		}
	}

	void CpuSNN::printCurrentInfo(FILE* fp)
	{
		// copy neuron input current...
		fprintf(fp, "Total Synaptic updates: \n");
		cutilSafeCall( cudaMemcpy( current, cpu_gpuNetPtrs.current, sizeof(float)*numNReg, cudaMemcpyDeviceToHost));
		for(int i=0; i < numNReg; i++) {
			if (current[i] != 0.0 ) {
				fprintf(fp, "I[%d] -> %f\n", i, current[i]);
			}
		}
		fprintf(fp, "\n");
		fflush(fp);
	}

	void CpuSNN::printFiringInfo(FILE* fp, int myGrpId)
	{
		//printNeuronState(myGrpId, stderr);
	}

	void CpuSNN::printTestVarInfo(FILE* fp, char* testString, bool test1, bool test2, bool test12, int subVal, int grouping1, int grouping2)
	{
		int cnt=0;

		fflush(stdout);

		if(test1 || test12)
			cutilSafeCall( cudaMemcpy( testVar, cpu_gpuNetPtrs.testVar, sizeof(float)*numN, cudaMemcpyDeviceToHost));

		if(test2 || test12)
			cutilSafeCall( cudaMemcpy( testVar2, cpu_gpuNetPtrs.testVar2, sizeof(float)*numN, cudaMemcpyDeviceToHost));

		int gcnt=0;
		bool firstPrint = true;
		if(test12) {
			for(int i=0; i < numN; i++) {
				if ((testVar[i] != 0.0) || (testVar2[i] != 0.0)) {
					if(firstPrint) {
						fprintf(fp, "\ntime=%d: Testing Variable 1 and 2: %s\n", simTime, testString);
						firstPrint = false;
					}
					fprintf(fp, "testVar12[%d] -> %f : %f\n", i, testVar[i]-subVal, testVar2[i]-subVal);
					testVar[i]  = 0.0;
					testVar2[i] = 0.0;
					if(gcnt++==grouping1) { fprintf(fp, "\n"); gcnt=0;}
 				}
			}
//			fprintf(fp, "\n");
			fflush(fp);
			test1 = false;
			test2 = false;
			cutilSafeCall( cudaMemcpy( cpu_gpuNetPtrs.testVar,  testVar, sizeof(float)*numN, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
			cutilSafeCall( cudaMemcpy( cpu_gpuNetPtrs.testVar2, testVar2, sizeof(float)*numN, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt2", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
		}

		gcnt=0;
		firstPrint = true;
		if(test1) {
			// copy neuron input current...
			for(int i=0; i < numN; i++) {
				if (testVar[i] != 0.0 ) {
					if(firstPrint) {
						fprintf(fp, "\ntime=%d: Testing Variable 1 and 2: %s\n", simTime, testString);
						firstPrint = false;
					}
					if(gcnt==0) fprintf(fp, "testVar[%d] -> ", i);
					fprintf(fp, "%d\t", i, testVar[i]-subVal);
					testVar[i] = 0.0;
					if(++gcnt==grouping1) { fprintf(fp, "\n"); gcnt=0;}
				}
			}
//			fprintf(fp, "\n");
			fflush(fp);
			cnt=0;
			cutilSafeCall( cudaMemcpy( cpu_gpuNetPtrs.testVar,  testVar, sizeof(float)*numN, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
		}

		gcnt=0;
		firstPrint = 1;
		if(test2) {
			for(int i=0; i < numN; i++) {
				if (testVar2[i] != 0.0 ) {
					if(firstPrint) {
						fprintf(fp, "\ntime=%d: Testing Variable 1 and 2: %s\n", simTime, testString);
						firstPrint = 0;
					}
					if(gcnt==0) fprintf(fp, "testVar2[%d] -> ", i);
					fprintf(fp, "%d\t", i, testVar2[i]-subVal);
					testVar2[i] = 0.0;
					if(++gcnt==grouping2) { fprintf(fp, "\n"); gcnt=0;}
				}
			}
//			fprintf(fp, "\n");
			fflush(fp);

			cutilSafeCall( cudaMemcpy( cpu_gpuNetPtrs.testVar2, testVar2, sizeof(float)*numN, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt2", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
		}
		return;
	}

	void CpuSNN::testSpikeSenderReceiver(FILE* fpLog, int simTime)
	{
		if(0) {
			int EnumFires = 0; int InumFires = 0;
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &EnumFires, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &InumFires, "secD1fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
			fprintf(stdout, " ***********( t = %d) FIRE COUNTS ************** %d %d\n", simTime, EnumFires, InumFires);
			int   numFires;
			float fireCnt;
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &numFires,  "testFireCnt1", sizeof(int), 0, cudaMemcpyDeviceToHost));
			fprintf(stdout, "testFireCnt1 = %d\n", numFires);
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &numFires,  "testFireCnt2", sizeof(int), 0, cudaMemcpyDeviceToHost));
			fprintf(stdout, "testFireCnt2 = %d\n", numFires);
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &fireCnt,  "testFireCntf1", sizeof(int), 0, cudaMemcpyDeviceToHost));
			fprintf(stdout, "testFireCntFloat1 = %f\n", fireCnt);
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &fireCnt,  "testFireCntf2", sizeof(int), 0, cudaMemcpyDeviceToHost));
			fprintf(stdout, "testFireCntFloat2 = %f\n", fireCnt);
			fprintf(stdout, " *************************\n");

			fprintf(fpLog, "\n");
		}

#if 0
			static int* firingTableD2 = (int*) malloc(sizeof(int)*(EnumFires+1));	
			static int* firingTableD1 = (int*) malloc(sizeof(int)*(InumFires+1));	
			fprintf(stdout, "Total Fired Neuron in GPU = E=%d + I=%d\n", EnumFires, InumFires);

			if (EnumFires > 0) 
				CUDA_SAFE_CALL( cudaMemcpy( firingTableD2, cpu_gpuNetPtrs.firingTableD2, sizeof(int)*EnumFires, cudaMemcpyDeviceToHost));

			if (InumFires > 0) 
				CUDA_SAFE_CALL( cudaMemcpy( firingTableD1, cpu_gpuNetPtrs.firingTableD1, sizeof(int)*InumFires, cudaMemcpyDeviceToHost));

			static int cumCntI = 0;
			static int cumcntD2 = 0;
			fprintf(stdout, "\n");
			fprintf(stdout, "current threshold crossing neurons (GPU) = %d\n", EnumFires+InumFires-cumcntD2-cumCntI);
			for(int i=cumCntI; i < InumFires; i++) {
				fprintf(stdout, " %d " , firingTableD1[i]);
			}
			cumCntI = InumFires;

			for(int i=cumcntD2; i < EnumFires; i++) {
				fprintf(stdout, " %d " , firingTableD2[i]);
			}
			cumcntD2 = EnumFires;
			getchar();
#endif

#if 0 && (TESTING)
			int sendId[1000];
			int recId[1000];
			int genI, genE, recE, recI;
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &recE, "receivedSpikesE", sizeof(int), 0, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &recI, "receivedSpikesI", sizeof(int), 0, cudaMemcpyDeviceToHost));

			//fprintf(stderr, "generatedE = %d, receivedE = %d generatedI = %d, receivedI = %d\n", genE, recE, genI, recI);
			fprintf(fpLog, "generatedE = %d, receivedE = %d generatedI = %d, receivedI = %d\n", genE, recE, genI, recI);

			//KILLME !!!
			if ((genE != recE) || (genI != recI)) {
				fprintf( stderr, "ERROR !! generatedE = %d, receivedE = %d generatedI = %d, receivedI = %d\n", genE, recE, genI, recI);
				fprintf( fpLog, "ERROR !! generatedE = %d, receivedE = %d generatedI = %d, receivedI = %d\n", genE, recE, genI, recI);
				fflush(fpLog);
				assert(0);
			}

			//assert(genE == recE);
			//assert(genI == recI);

			static int cntTest = 0;

			if(cntTest++ == 1000) {
				genI = 0; genE = 0; recI = 0; recE = 0;
				CUDA_SAFE_CALL( cudaMemcpyToSymbol( "receivedSpikesE", &recE, sizeof(int), 0, cudaMemcpyHostToDevice));	
				CUDA_SAFE_CALL( cudaMemcpyToSymbol( "receivedSpikesI", &recI, sizeof(int), 0, cudaMemcpyHostToDevice));	
				cntTest = 0;
			}
#endif
		/*
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &sendId, "senderIdE", sizeof(int)*genE, 0, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &sendId[genE], "senderIdI", sizeof(int)*genI, 0, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &recId, "receiverId", sizeof(int)*(recI+recE), 0, cudaMemcpyDeviceToHost));

		int cnt=0;
		for(int i = 0; i < genE+genI; i++) {
			int found = false;
			for(int j=0; j < 0; j++) {
				if(recId[j] == (sendId[i])) {
					recId[j]= -1;
					cnt++;
					found = false;
				}
			}
			if(!found)
			   fprintf(fpLog, "sendE[%d] = %d, syn = %d\n", i, sendId[i]&POST_SYN_NEURON_MASK
			                                                 , ((sendId[i]>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK));
		}
		if(cnt != (recI+recE)) {
			for(int i = 0; i < recE+recI; i++) {
				if(recId[i]!=-1) {
					fprintf(fpLog, "extra[%d] = %d, syn = %d\n", i, 
						  recId[i]&POST_SYN_NEURON_MASK,
						((recId[i]>>POST_SYN_NEURON_BITS)&POST_SYN_CONN_MASK));
				}
			}
		}
		*/
	}

	void CpuSNN::globalStateUpdate_GPU()
	{
		DBG(2, fpLog, AT, "gpu_globalStateUpdate()");

		int blkSize  = 128;
		int gridSize = 64;

		kernel_globalConductanceUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);

		kernel_globalStateUpdate <<<gridSize, blkSize>>> (simTimeMs, simTimeSec, simTime);
		CUT_CHECK_ERROR("Kernel execution failed");
		int errCode = checkCudaErrors("gpu_globalStateUpdate", gridSize);
		assert(errCode == NO_KERNEL_ERRORS);


//		printTestVarInfo(stderr, "globalStateUpdate_GPU", true, false, false, 0, 1, 0);
//		int recE, recI;
//		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &recE, "receivedSpikesE", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &recI, "receivedSpikesI", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		fprintf(stderr, "t=%d, receivedE = %d, receivedI = %d\n", simTime, recE, recI);
//		recE=0;recI=0;
//		CUDA_SAFE_CALL( cudaMemcpyToSymbol( "receivedSpikesE", &recE, sizeof(int), 0, cudaMemcpyHostToDevice));
//		CUDA_SAFE_CALL( cudaMemcpyToSymbol( "receivedSpikesI", &recI, sizeof(int), 0, cudaMemcpyHostToDevice));
	}

	void CpuSNN::assignPoissonFiringRate_GPU()
	{
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
				
				cutilSafeCall( cudaMemcpy( &cpu_gpuNetPtrs.poissonFireRate[nid-numNReg], rate->rates, sizeof(float)*rate->len, rate->onGPU?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
			}
		}
	}

	void CpuSNN::doGPUSim()
	{
		if (spikeRateUpdated) {
			assignPoissonFiringRate_GPU();
			spikeRateUpdated = false;
		}

//		initThalInput_GPU();
		//printTestVarInfo(stderr, "initThalInput", true, false, false);

		findFiring_GPU();
		//printTestVarInfo(stderr, "findFiring", true, false, false);


		updateTimingTable_GPU();

		doCurrentUpdate_GPU();

		#if (ENABLE_MORE_CHECK)
			unsigned int gpu_secD2fireCnt, gpu_secD1fireCnt;

			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, "secD1fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));

			printf("gpu_secD1fireCnt: %d max: %d gpu_secD2fireCnt: %d max: %d\n",gpu_secD1fireCnt,maxSpikesD1,gpu_secD2fireCnt,maxSpikesD2);

			assert(gpu_secD1fireCnt<=maxSpikesD1);
			assert(gpu_secD2fireCnt<=maxSpikesD2);
		#endif

		globalStateUpdate_GPU();
		// printState("globalState update\n");
		//printTestVarInfo(stderr, "globalStateUpdate_GPU", true, false, false, 0, 1);

		for (int grpId=20; 0 & grpId < 25; grpId++) {
			int ptrPos  = grp_Info[grpId].StartN;
			int length  = grp_Info[grpId].SizeN;
			cutilSafeCall( cudaMemcpy( &cpuNetPtrs.current[ptrPos], &cpu_gpuNetPtrs.current[ptrPos], sizeof(float)*length, cudaMemcpyDeviceToHost));
			for(int i=0; i < length; i++) {
				fprintf(stderr, "current %d -> %f\n", ptrPos+i, cpuNetPtrs.current[ptrPos+i]);
			}
		}

		if(0) {
			int cnt=0;
			testSpikeSenderReceiver(fpLog, simTime);
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL( cudaMemcpyToSymbol( "testVarCnt2", &cnt, sizeof(int), 0, cudaMemcpyHostToDevice));
		}

//		int tmpCnt;
//		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &tmpCnt, "testVarCnt1", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		int EnumFires = 0;
//		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &EnumFires, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		fprintf(stderr, " (t=%d) fireCnt=%d, testVarCnt=%d\n", simTime, EnumFires, tmpCnt);

		return;

		//getchar();
		//if(simTime%100==99)
		//updateSpikeMonitor();
		//showStatus(GPU_MODE,1, fpLog);
		//simTime++;
		//exit(0);
	}

	void CpuSNN::updateStateAndFiringTable_GPU()
	{			
		DBG(2, fpLog, AT, "gpu_updateStateAndFiringTable()");

		int blkSize  = 128;
		int gridSize = 64;
//		void* devPtr;

		//kernel_updateWeightsFiring  <<<gridSize, blkSize>>> ();
		kernel_updateWeightsFiring_static  <<<gridSize, blkSize>>> ();

		kernel_updateWeightsFiring <<<gridSize, blkSize>>> ();

		 //printTestVarInfo(stderr, "STDP", true, false, false, 0, 2, 0);

//		// the firing id is stored in firingTableD1, and the firing bit pattern is stored in firingTableD2...
//		int gpu_secD2fireCnt;
//		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		//CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, "secD1fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
//		fprintf(stderr, "Total spikes before next time sec is %d\n", gpu_secD2fireCnt);
	}

	void CpuSNN::showStatus_GPU()
	{
		int gpu_secD1fireCnt, gpu_secD2fireCnt;
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, "secD1fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCountAll1sec = gpu_secD1fireCnt + gpu_secD2fireCnt;
		secD1fireCnt  = gpu_secD1fireCnt;
		
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

	void CpuSNN::copyFiringInfo_GPU()
	{
		unsigned int gpu_secD1fireCnt, gpu_secD2fireCnt;
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD2fireCnt, "secD2fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &gpu_secD1fireCnt, "secD1fireCnt", sizeof(int), 0, cudaMemcpyDeviceToHost));
		spikeCountAll1sec = gpu_secD1fireCnt + gpu_secD2fireCnt;
		secD1fireCnt  = gpu_secD1fireCnt;
		assert(gpu_secD1fireCnt<=maxSpikesD1);
		assert(gpu_secD2fireCnt<=maxSpikesD2);
		cutilSafeCall( cudaMemcpy(firingTableD2, cpu_gpuNetPtrs.firingTableD2, sizeof(int)*gpu_secD2fireCnt, cudaMemcpyDeviceToHost));
		cutilSafeCall( cudaMemcpy(firingTableD1, cpu_gpuNetPtrs.firingTableD1, sizeof(int)*gpu_secD1fireCnt, cudaMemcpyDeviceToHost));
		cutilSafeCall( cudaMemcpyFromSymbol(timeTableD2, "timingTableD2", sizeof(int)*(1000+D+1), 0, cudaMemcpyDeviceToHost));
		cutilSafeCall( cudaMemcpyFromSymbol(timeTableD1, "timingTableD1", sizeof(int)*(1000+D+1), 0, cudaMemcpyDeviceToHost));
		fprintf(stderr, "Total spikes Multiple Delays=%d, 1Ms Delay=%d\n", gpu_secD2fireCnt,gpu_secD1fireCnt);
		//getchar();
	}

	// initialize the probes to appropriate values
	void CpuSNN::gpuProbeInit (network_ptr_t* dest) 
	{
		if(dest->allocated || numProbe==0)
			return;

		probeParam_t* n = neuronProbe;
		uint32_t* probeId;
		probeId = (uint32_t*) calloc(numProbe, sizeof(uint32_t)*net_Info.numN);
		int cnt = 0;
		while(n) {
			int nid  = n->nid;
			probeId[nid] = cnt++;
			n = n->next;
		}
		assert(cnt == numProbe);
		// allocate the destination probes on the GPU
		cutilSafeCall( cudaMalloc( (void**) &dest->probeId, sizeof(uint32_t)*net_Info.numN*numProbe));
		cutilSafeCall( cudaMemcpy( dest->probeId, &probeId, sizeof(uint32_t)*net_Info.numN*numProbe, cudaMemcpyHostToDevice));
		// allocate and assign the probes
		cutilSafeCall( cudaMalloc( (void**) &dest->probeV, 	   1000*sizeof(float)*numProbe));
		cutilSafeCall( cudaMalloc( (void**) &dest->probeI, 	   1000*sizeof(float)*numProbe));

		free(probeId);
    }

	void CpuSNN::allocateNetworkParameters()
	{
		net_Info.numN  = numN;
		net_Info.numPostSynapses  = numPostSynapses;
		net_Info.D  = D;
		net_Info.numNExcReg = numNExcReg;
		net_Info.numNInhReg	= numNInhReg;
		net_Info.numNReg = numNReg;
		assert(numNReg == (numNExcReg+numNInhReg));
		net_Info.numNPois = numNPois;
		net_Info.numNExcPois = numNExcPois;		
		net_Info.numNInhPois = numNInhPois;
		assert(numNPois== (numNExcPois+numNInhPois));
//		net_Info.numNoise  = numNoise;
		net_Info.maxSpikesD2 = maxSpikesD2;
		net_Info.maxSpikesD1 = maxSpikesD1;
		net_Info.numProbe = numProbe;
		net_Info.sim_with_conductances = sim_with_conductances;
		net_Info.sim_with_stp = sim_with_stp;
		net_Info.numGrp = numGrp;
		cpu_gpuNetPtrs.memType = GPU_MODE;
		
		return;
	}

	void checkGPUDevice()
	{
		int dev = cutGetMaxGflopsDeviceId();
		fprintf(stdout, "Device with maximum GFLOPs is : %d\n", dev);
		cudaDeviceProp deviceProp;
		dev = 0;
		cutilSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
		fprintf(stdout, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		if (deviceProp.major == 1 && deviceProp.minor < 3) {
			printf("GPU SNN does not support NVidia cards older than version 1.3\n");
			//exit(1);
		}
		printf("CUDA Device is of type %d.%d\n", deviceProp.major, deviceProp.minor);
		assert(deviceProp.major >= 1);
		//assert(deviceProp.minor >= 3);		
		cudaThreadExit();
		cudaSetDevice( dev );
		cutilCheckMsg("cudaSetDevice failed\n");	
	}

	void CpuSNN::copyWeightsGPU(unsigned int nid, int src_grp)
	{
		assert(nid < numNReg);
		unsigned int    cumId   =  cumulativePre[nid];
		float* synWts  = &wt[cumId];
		assert(cumId >= (nid-numNPois));
		//assert(cumId < numPreSynapses*numN);

#if 0
		fprintf(fpLog, "OLD WEIGHTS\n");
		for(int i=0; i < Npre[nid]; i++) {		
			fprintf(fpLog, " %f ",  synWts[i]);
		}
		fprintf(fpLog, "\n");
#endif
		cutilSafeCall( cudaMemcpy( synWts, &cpu_gpuNetPtrs.wt[cumId], sizeof(float)*Npre[nid], cudaMemcpyDeviceToHost));

#if 0
		fprintf(fpLog, "NEW WEIGHTS\n");
		for(int i=0; i < Npre[nid]; i++) {		
			fprintf(fpLog, " %f ",  synWts[i]);
		}
		fprintf(fpLog, "\n");
#endif
	}

	// Allocates required memory and then initialize the GPU
	void CpuSNN::allocateSNN_GPU()
	{
		if (D > MAX_SynapticDelay) {
			printf("Error, you are using a synaptic delay (%d) greater than MAX_SynapticDelay defined in config.h\n",D);
			assert(0);
		}
	
		// if we have already allocated the GPU data.. dont do it again...
		if(gpuPoissonRand != NULL)
			return;

		int gridSize = 64; int blkSize  = 128;

		checkGPUDevice();

		int numN=0;
		for (int g=0;g<numGrp;g++) {
			numN += grp_Info[g].SizeN;
		}

		// generate the random number for the poisson neuron here...
		if(gpuPoissonRand == NULL) {
			gpuPoissonRand = new RNG_rand48(randSeed);
		}

		gpuPoissonRand->generate(numNPois, RNG_rand48::MAX_RANGE);
		
		// save the random pointer as poisson generator....
		cpu_gpuNetPtrs.poissonRandPtr = (unsigned int*) gpuPoissonRand->get_random_numbers();

		//ensure that we dont do all the above optimizations again		
		assert(doneReorganization == true);		

		allocateNetworkParameters();

		allocateStaticLoad(blkSize);

		allocateGroupId();

		// this table is useful for quick evaluation of the position of fired neuron
		// given a sequence of bits denoting the firing..
		initTableQuickSynId();
		gpuProbeInit(&cpu_gpuNetPtrs);
		copyConnections(&cpu_gpuNetPtrs,  cudaMemcpyHostToDevice, 1);
		copyState(&cpu_gpuNetPtrs, cudaMemcpyHostToDevice, 1);
		
		// copy relevant pointers and network information to GPU
		void* devPtr;
		cutilSafeCall( cudaMemcpyToSymbol("gpuPtrs",    &cpu_gpuNetPtrs, sizeof(network_ptr_t), 0, cudaMemcpyHostToDevice));
		cutilSafeCall( cudaMemcpyToSymbol("gpuNetInfo", &net_Info, sizeof(network_info_t), 0, cudaMemcpyHostToDevice));
		// FIXME: we can chance the group properties such as STDP as the network is running.  So, we need a way to updating the GPU when changes are made.

		cutilSafeCall( cudaMemcpyToSymbol("gpuGrpInfo", grp_Info, (net_Info.numGrp)*sizeof(group_info_t), 0, cudaMemcpyHostToDevice));

		if (showLogMode >= 3) {
			fprintf(stderr,"Transfering group settings to GPU:\n");
			for (int i=0;i<numGrp;i++) {
				fprintf(stderr,"Settings for Group %s: \n", grp_Info2[i].Name.c_str());
		
				fprintf(stderr,"\tType: %d\n",(int)grp_Info[i].Type);
				fprintf(stderr,"\tSizeN: %d\n",grp_Info[i].SizeN);
				fprintf(stderr,"\tMaxFiringRate: %d\n",(int)grp_Info[i].MaxFiringRate);
				fprintf(stderr,"\tRefractPeriod: %f\n",grp_Info[i].RefractPeriod);
				fprintf(stderr,"\tM: %d\n",grp_Info[i].numPostSynapses);
				fprintf(stderr,"\tPreM: %d\n",grp_Info[i].numPreSynapses);
				fprintf(stderr,"\tspikeGenerator: %d\n",(int)grp_Info[i].isSpikeGenerator);
				fprintf(stderr,"\tFixedInputWts: %d\n",(int)grp_Info[i].FixedInputWts);
				fprintf(stderr,"\tMaxDelay: %d\n",(int)grp_Info[i].MaxDelay);
				fprintf(stderr,"\tWithSTDP: %d\n",(int)grp_Info[i].WithSTDP);
				if (grp_Info[i].WithSTDP) {
					fprintf(stderr,"\t\tTAU_LTP_INV: %f\n",grp_Info[i].TAU_LTP_INV);
					fprintf(stderr,"\t\tTAU_LTD_INV: %f\n",grp_Info[i].TAU_LTD_INV);
					fprintf(stderr,"\t\tALPHA_LTP: %f\n",grp_Info[i].ALPHA_LTP);
					fprintf(stderr,"\t\tALPHA_LTD: %f\n",grp_Info[i].ALPHA_LTD);
				}
				fprintf(stderr,"\tWithConductances: %d\n",(int)grp_Info[i].WithConductances);
				if (grp_Info[i].WithConductances) {
					fprintf(stderr,"\t\tdAMPA: %f\n",grp_Info[i].dAMPA);
					fprintf(stderr,"\t\tdNMDA: %f\n",grp_Info[i].dNMDA);
					fprintf(stderr,"\t\tdGABAa: %f\n",grp_Info[i].dGABAa);
					fprintf(stderr,"\t\tdGABAb: %f\n",grp_Info[i].dGABAb);
				}
				fprintf(stderr,"\tWithSTP: %d\n",(int)grp_Info[i].WithSTP);
				if (grp_Info[i].WithSTP) {
					fprintf(stderr,"\t\tSTP_U: %f\n",grp_Info[i].STP_U);
					fprintf(stderr,"\t\tSTP_tD: %f\n",grp_Info[i].STP_tD);
					fprintf(stderr,"\t\tSTP_tF: %f\n",grp_Info[i].STP_tF);
				}
				fprintf(stderr,"\tspikeGen: %s\n",grp_Info[i].spikeGen==NULL?"Is Null":"Is set");
			}
		}

		cpu_gpuNetPtrs.allocated = true;

		// map the timing table to texture.. saves a lot of headache in using shared memory
		CUDA_SAFE_CALL ( cudaGetSymbolAddress(&devPtr, "timingTableD2"));
		size_t offset;
		CUDA_SAFE_CALL ( cudaBindTexture(&offset, timingTableD2_tex, devPtr, sizeof(int)*ROUNDED_TIMING_COUNT));
		offset = offset/sizeof(int);
		CUDA_SAFE_CALL ( cudaGetSymbolAddress(&devPtr, "timingTableD2_tex_offset"));
		cutilSafeCall( cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));
		
		CUDA_SAFE_CALL ( cudaGetSymbolAddress(&devPtr, "timingTableD1"));
		CUDA_SAFE_CALL ( cudaBindTexture(&offset, timingTableD1_tex, devPtr, sizeof(int)*ROUNDED_TIMING_COUNT));
		offset = offset/sizeof(int);
		CUDA_SAFE_CALL ( cudaGetSymbolAddress(&devPtr, "timingTableD1_tex_offset"));
		cutilSafeCall( cudaMemcpy(devPtr, &offset, sizeof(int), cudaMemcpyHostToDevice));

		cutilSafeCall( cudaMemset( cpu_gpuNetPtrs.current, 0, sizeof(float)*numNReg));

		initGPU(gridSize, blkSize);
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
//		if(cpu_gpuNetPtrs.allocated == false)  
//			allocateSNN_GPU();

		int blkSize  = 128;
		int gridSize = 64;
		bool setVoltage = true;

		// Initialize the Thalamic input current into the network
		if(numNoise) {
			// copy the generated random number into the randId array
			cutilSafeCall( cudaMemcpy(cpu_gpuNetPtrs.randId, randNeuronId, sizeof(int)*numRandNeurons, cudaMemcpyHostToDevice));

			kernel_initThalInput <<<gridSize,blkSize >>> (setVoltage);
			cutilCheckMsg("initThalInput kernel failed\n");
			int errCode = checkCudaErrors("kernel_initThalInput", gridSize);
		}
	}

*/
