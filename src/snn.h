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

#ifndef _SNN_GOLD_H_
#define _SNN_GOLD_H_

#include <mtrand.h>
#include <gpu_random.h>
#include <config.h>
#include <propagated_spike_buffer.h>
#include <poisson_rate.h>

#if __CUDA3__
	#include <cuda.h>
	#include <cutil_inline.h>
	#include <cutil_math.h>
#elif __CUDA5__
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <helper_cuda.h>
	#include <helper_functions.h>
	#include <helper_timer.h>
	#include <helper_math.h>
#endif

#include "CUDAVersionControl.h"


// enable easy testing of private members
#if REGRESSION_TESTING
	#define private public
 	#define protected public
#endif


extern RNG_rand48* gpuRand48; //!< Used by all network to generate global random number

#define MAX_GRPS_PER_BLOCK 		100
#define MAX_BLOCKS         		120

#define ALL -1 //!< used for the set* methods to specify all groups and/or configIds

#define SYN_FIXED      0
#define SYN_PLASTIC    1


// Bit flags to be used to specify the type of neuron.  Future types can be added in the future such as Dopamine, etc.
// Yes, they should be bit flags because some neurons release more than one transmitter at a synapse.
#define UNKNOWN_NEURON	(0)
#define POISSON_NEURON	(1<<0)
#define TARGET_AMPA	(1<<1)
#define TARGET_NMDA	(1<<2)
#define TARGET_GABAa	(1<<3)
#define TARGET_GABAb	(1<<4)

#define INHIBITORY_NEURON 		(TARGET_GABAa | TARGET_GABAb)
#define EXCITATORY_NEURON 		(TARGET_NMDA | TARGET_AMPA)
#define EXCITATORY_POISSON 		(EXCITATORY_NEURON | POISSON_NEURON)
#define INHIBITORY_POISSON		(INHIBITORY_NEURON | POISSON_NEURON)
#define IS_INHIBITORY_TYPE(type)	(((type)&TARGET_GABAa) || ((type)&TARGET_GABAb))
#define IS_EXCITATORY_TYPE(type)	(!IS_INHIBITORY_TYPE(type))


#define CONN_SYN_NEURON_BITS	20                               //!< last 20 bit denote neuron id. 1 Million neuron possible
#define CONN_SYN_BITS			(32 -  CONN_SYN_NEURON_BITS)	 //!< remaining 12 bits denote connection id
#define CONN_SYN_NEURON_MASK    ((1 << CONN_SYN_NEURON_BITS) - 1)
#define CONN_SYN_MASK      		((1 << CONN_SYN_BITS) - 1)
#define GET_CONN_NEURON_ID(a) (((unsigned int)a.postId) & CONN_SYN_NEURON_MASK)
#define GET_CONN_SYN_ID(b)    (((unsigned int)b.postId) >> CONN_SYN_NEURON_BITS)
#define GET_CONN_GRP_ID(c)    (c.grpId)
//#define SET_CONN_ID(a,b)      ((b) > CONN_SYN_MASK) ? (fprintf(stderr, "Error: Syn Id exceeds maximum limit (%d)\n", CONN_SYN_MASK)): (((b)<<CONN_SYN_NEURON_BITS)+((a)&CONN_SYN_NEURON_MASK))


#define CONNECTION_INITWTS_RANDOM    	0
#define CONNECTION_CONN_PRESENT  		1
#define CONNECTION_FIXED_PLASTIC		2
#define CONNECTION_INITWTS_RAMPUP		3
#define CONNECTION_INITWTS_RAMPDOWN		4

#define SET_INITWTS_RANDOM(a)		((a&1) << CONNECTION_INITWTS_RANDOM)
#define SET_CONN_PRESENT(a)		((a&1) << CONNECTION_CONN_PRESENT)
#define SET_FIXED_PLASTIC(a)		((a&1) << CONNECTION_FIXED_PLASTIC)
#define SET_INITWTS_RAMPUP(a)		((a&1) << CONNECTION_INITWTS_RAMPUP)
#define SET_INITWTS_RAMPDOWN(a)		((a&1) << CONNECTION_INITWTS_RAMPDOWN)

#define GET_INITWTS_RANDOM(a)		(((a) >> CONNECTION_INITWTS_RANDOM)&1)
#define GET_CONN_PRESENT(a)		(((a) >> CONNECTION_CONN_PRESENT)&1)
#define GET_FIXED_PLASTIC(a)		(((a) >> CONNECTION_FIXED_PLASTIC)&1)
#define GET_INITWTS_RAMPUP(a)		(((a) >> CONNECTION_INITWTS_RAMPUP)&1)
#define GET_INITWTS_RAMPDOWN(a)		(((a) >> CONNECTION_INITWTS_RAMPDOWN)&1)


/****************************/


// FIXME: 
/////    !!!!!!! EVEN MORE IMPORTANT : IS THIS STILL BEING USED?? !!!!!!!!!!

/////    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
////     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
///      Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
///      <--- numNExcReg --> | <-- numNInhReg --> | <-- numNInhPois -> | <---numNExcPois-->
///      <------REGULAR NEURON REGION ----------> | <----- POISSON NEURON REGION --------->
///      <----numNReg=(numNExcReg+numNInhReg)---> | <--numNPois=(numNInhPois+numNExcPois)->
////     <--------------------- ALL NEURONS ( numN=numNReg+numNPois) --------------------->
////	This organization scheme is only used/needed for the gpu_static code.
#define IS_POISSON_NEURON(nid, numNReg, numNPois) ((nid) >= (numNReg) && ((nid) < (numNReg+numNPois)))
#define IS_REGULAR_NEURON(nid, numNReg, numNPois) (((nid) < (numNReg)) && ((nid) < (numNReg+numNPois)))
#define IS_INHIBITORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) >= (numNExcReg)) && ((nid) < (numNReg + numNInhPois)))
#define IS_EXCITATORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) < (numNReg)) && (((nid) < (numNExcReg)) || ((nid) >=  (numNReg + numNInhPois))))

#if __CUDACC__
inline bool isExcitatoryNeuron (unsigned int& nid, unsigned int& numNInhPois, unsigned int& numNReg, unsigned int& numNExcReg, unsigned int& numN)
{
	return ((nid < numN) && ((nid < numNExcReg) || (nid >= numNReg + numNInhPois)));
}
inline bool isInhibitoryNeuron (unsigned int& nid, unsigned int& numNInhPois, unsigned int& numNReg, unsigned int& numNExcReg, unsigned int& numN)
{
	return ((nid >= numNExcReg) && (nid < (numNReg + numNInhPois)));
}
#endif

#define STATIC_LOAD_START(n)  (n.x)
#define STATIC_LOAD_GROUP(n)  (n.y &0xff)
#define STATIC_LOAD_SIZE(n)   ((n.y >> 16) & 0xff)

#define MAX_NUMBER_OF_NEURONS_BITS  (20)
#define MAX_NUMBER_OF_GROUPS_BITS   (32-MAX_NUMBER_OF_NEURONS_BITS)
#define MAX_NUMBER_OF_NEURONS_MASK  ((1 << MAX_NUMBER_OF_NEURONS_BITS)-1)
#define MAX_NUMBER_OF_GROUPS_MASK   ((1 << MAX_NUMBER_OF_GROUPS_BITS)-1)
#define SET_FIRING_TABLE(nid, gid)  (((gid) << MAX_NUMBER_OF_NEURONS_BITS) | (nid))
#define GET_FIRING_TABLE_NID(val)   ((val) & MAX_NUMBER_OF_NEURONS_MASK)
#define GET_FIRING_TABLE_GID(val)   (((val) >> MAX_NUMBER_OF_NEURONS_BITS) & MAX_NUMBER_OF_GROUPS_MASK)

//!< Used for in the function getConnectionId
#define CHECK_CONNECTION_ID(n,total) { assert(n >= 0); assert(n < total); }

// Macros for STP
// we keep a history of STP values to compute resource change over time
// the macro is slightly faster than an inline function, but we should consider changing it anyway because
// it's unsafe
#define STP_BUF_SIZE 32
#define STP_BUF_POS(nid,t)  (nid*STP_BUF_SIZE+((t)%STP_BUF_SIZE))


// use these macros for logging / error printing
// every message will be printed to one of fpOut_, fpErr_, fpDeb_ depending on the nature of the message
// Additionally, every message gets printed to some log file fpLog_. This is different from fpDeb_ for
// the case in which you want the two to be different (e.g., developer mode, in which you would like to
// see all debug info (stdout) but also have it saved to a file
#define CARLSIM_ERROR(formatc, ...) {	CARLSIM_ERROR_PRINT(fpErr_,formatc,##__VA_ARGS__); \
CARLSIM_DEBUG_PRINT(fpLog_,formatc,##__VA_ARGS__); }
#define CARLSIM_WARN(formatc, ...) {	CARLSIM_WARN_PRINT(fpErr_,formatc,##__VA_ARGS__); \
CARLSIM_DEBUG_PRINT(fpLog_,formatc,##__VA_ARGS__); }
#define CARLSIM_INFO(formatc, ...) {	CARLSIM_INFO_PRINT(fpOut_,formatc,##__VA_ARGS__); \
CARLSIM_DEBUG_PRINT(fpLog_,formatc,##__VA_ARGS__); }
#define CARLSIM_DEBUG(formatc, ...) {	CARLSIM_DEBUG_PRINT(fpDeb_,formatc,##__VA_ARGS__); \
CARLSIM_DEBUG_PRINT(fpLog_,formatc,##__VA_ARGS__); }

#define CARLSIM_ERROR_PRINT(fp, formatc, ...) fprintf(fp,"\033[31;1m[ERROR %s:%d] " formatc "\033[0m \n",__FILE__,__LINE__,##__VA_ARGS__)
#define CARLSIM_WARN_PRINT(fp, formatc, ...) fprintf(fp,"\033[33;1m[WARNING %s:%d] " formatc "\033[0m \n",__FILE__,__LINE__,##__VA_ARGS__)
#define CARLSIM_INFO_PRINT(fp, formatc, ...) fprintf(fp,formatc "\n",##__VA_ARGS__)
#define CARLSIM_DEBUG_PRINT(fp, formatc, ...) fprintf(fp,"[DEBUG %s:%d] " formatc "\n",__FILE__,__LINE__,##__VA_ARGS__)

/*!
 * \brief Logger modes
 * The logger mode defines where to print all status, error, and debug messages. Several predefined
 * modes exist (USER, DEVELOPER, SILENT). However, the user can also set each file pointer to a
 * location of their choice (CUSTOM mode).
 * The following logger modes exist:
 *  USER 		User mode, for experiment-oriented simulations. Errors and warnings go to stderr,
 *              status information goes to stdout. Debug information can only be found in the log file.
 *  DEVELOPER   Developer mode, for developing and debugging code. Same as user, but additionally,
 *              all debug information is printed to stdout.
 *  SILENT      Silent mode, no output is generated.
 *  CUSTOM      Custom mode, the user can set the location of all the file pointers.
 * 
 * The following file pointers exist:
 *  fpOut_	where CARLSIM_INFO messages go
 *  fpErr_ 	where CARLSIM_ERROR and CARLSIM_WARN messages go
 *  fpDeb_ 	where CARLSIM_DEBUG messages go
 *  fpLog_ 	typically a log file, where all of the above messages go
 *
 * The file pointers are automatically set to different locations, depending on the loggerMode:
 *
 *          |    USER    | DEVELOPER  |   SILENT   |  CUSTOM
 * ---------|------------|------------|------------|---------
 * fpOut_   |   stdout   |   stdout   | /dev/null  |    ?
 * fpErr_   |   stderr   |   stderr   | /dev/null  |    ?
 * fpDeb_   | /dev/null  |   stdout   | /dev/null  |    ?
 * fpLog_   | debug.log  | debug.log  | /dev/null  |    ?
 *
 * Location of the debug log file can be set in any mode using CARLsim::setLogDebugFp.
 * In mode CUSTOM, the other file pointers can be set using CARLsim::setLogsFp.
 */
 enum loggerMode_t { USER, DEVELOPER, SILENT, CUSTOM, UNKNOWN };

/*!
 * \brief simulation mode
 * CARLsim supports execution either on standard x86 central processing units (CPUs) or off-the-shelf NVIDIA GPUs.
 * 
 * When creating a new CARLsim object, you can choose from the following:
 * CPU_MODE:	run on a single CPU core
 * GPU_MODE:	run on a single GPU card
 * 
 * When running GPU mode on a multi-GPU system, you can specify on which CUDA device to establish a context (ithGPU,
 * 0-indexed) when you create a new CpuSNN object.
 * The simulation mode will be fixed throughout the lifetime of a CpuSNN object.
 */
 enum simMode_t {CPU_MODE, GPU_MODE};

//! connection types, used internally (externally it's a string)
 enum conType_t { CONN_RANDOM, CONN_ONE_TO_ONE, CONN_FULL, CONN_FULL_NO_DIRECT, CONN_USER_DEFINED, CONN_UNKNOWN};



// forward-declaration
 class CpuSNN;

//! used for fine-grained control over spike generation, using a callback mechanism
/*! Spike generation can be performed using spike generators. Spike generators are dummy-neurons that have their spikes
 * specified externally either defined by a Poisson firing rate or via a spike injection mechanism. Spike generators can
 * have post-synaptic connections with STDP and STP, but unlike Izhikevich neurons, they do not receive any pre-synaptic
 * input. For more information on spike generators see Section Neuron groups: Spike generators in the Tutorial.
 *
 * For fine-grained control over spike generation, individual spike times can be specified per neuron in each group.
 * This is accomplished using a callback mechanism, which is called at each time step, to specify whether a neuron has
 * fired or not. */
 class SpikeGenerator {
 public:
 	SpikeGenerator() {};

	//! controls spike generation using a callback
	/*! \attention The virtual method should never be called directly */
 	virtual unsigned int nextSpikeTime(CpuSNN* s, int grpId, int i, unsigned int currentTime) = 0;
 };

//! used for fine-grained control over spike generation, using a callback mechanism
/*!
 * The user can choose from a set of primitive pre-defined connection topologies, or he can implement a topology of
 * their choice by using a callback mechanism. In the callback mechanism, the simulator calls a method on a user-defined
 * class in order to determine whether a connection should be made or not. The user simply needs to define a method that
 * specifies whether a connection should be made between a pre-synaptic neuron and a post-synaptic neuron, and the
 * simulator will automatically call the method for all possible pre- and post-synaptic pairs. The user can then specify
 * the connection's delay, initial weight, maximum weight, and whether or not it is plastic.
 */
 class ConnectionGenerator {
 public:
 	ConnectionGenerator() {};

	//! specifies which synaptic connections (per group, per neuron, per synapse) should be made
	/*! \attention The virtual method should never be called directly */
 	virtual void connect(CpuSNN* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt,
 		float& delay, bool& connected) = 0;
 };


//! can be used to create a custom spike monitor
/*! To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
 * basic statistics, store spike trains, or perform more complicated output monitoring. Spike monitors are registered
 * for a group and are called automatically by the simulator every second. Similar to an address event representation
 * (AER), the spike monitor indicates which neurons spiked by using the neuron ID within a group (0-indexed) and the
 * time of the spike. Only one spike monitor is allowed per group.*/
 class SpikeMonitor {
 public:
 	SpikeMonitor() {};

  //! Controls actions that are performed when certain neurons fire (user-defined).
  /*! \attention The virtual method should never be called directly */
 	virtual void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts) = 0;
 };



typedef struct {
	short  delay_index_start;
	short  delay_length;
} delay_info_t;

typedef struct {
	int      postId;
	uint8_t  grpId;
} post_info_t;

typedef struct network_info_s  {
	size_t		STP_Pitch;		//!< numN rounded upwards to the nearest 256 boundary
	unsigned int		numN,numPostSynapses,D,numNExcReg,numNInhReg, numNReg;
	unsigned int		I_setLength;
	size_t		I_setPitch;
	unsigned int		preSynLength;
	//	unsigned int		numRandNeurons;
	//	unsigned int		numNoise;
	unsigned int		postSynCnt;
	unsigned int		preSynCnt;
	unsigned int		maxSpikesD2,maxSpikesD1;
	unsigned int   	numNExcPois;
	unsigned int	numNInhPois;
	unsigned int	numNPois;
	unsigned int	numGrp;
	bool sim_with_fixedwts;
	bool sim_with_conductances;
	bool sim_with_stdp;
	bool sim_with_stp;

	bool sim_with_NMDA_rise;	//!< a flag to inform whether to compute NMDA rise time
	bool sim_with_GABAb_rise;	//!< a flag to inform whether to compute GABAb rise time
	float dAMPA;				//!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	float rNMDA;				//!< multiplication factor for rise time of NMDA
	float dNMDA;				//!< multiplication factor for decay time of NMDA
	float dGABAa;				//!< multiplication factor for decay time of GABAa
	float rGABAb;				//!< multiplication factor for rise time of GABAb
	float dGABAb;				//!< multiplication factor for decay time of GABAb
} network_info_t;

//! nid=neuron id, sid=synapse id, grpId=group id. 
inline post_info_t SET_CONN_ID(int nid, int sid, int grpId) {
	if (sid > CONN_SYN_MASK) {
		fprintf(stderr, "Error: Syn Id (%d) exceeds maximum limit (%d) for neuron %d\n", sid, CONN_SYN_MASK, nid);
		assert(0);
	}
	post_info_t p;
	p.postId = (((sid)<<CONN_SYN_NEURON_BITS)+((nid)&CONN_SYN_NEURON_MASK));
	p.grpId  = grpId;
	return p;
}

//! connection infos...
typedef struct connectData_s {
	int 	  				grpSrc, grpDest;
	uint8_t	  				maxDelay,  minDelay;
	float	  				initWt, maxWt;
	float 					mulSynFast;				//!< factor to be applied to either gAMPA or gGABAa
	float 					mulSynSlow;				//!< factor to be applied to either gNMDA or gGABAb
	int	  	  				numPostSynapses;
	int	  	  				numPreSynapses;
	uint32_t  				connProp;
	ConnectionGenerator*	conn;
	conType_t 				type;
	float					p; 						//!< connection probability
	short int				connId;					//!< connectID of the element in the linked list
	bool					newUpdates;
	int		   				numberOfConnections;
	struct connectData_s* next;
} grpConnectInfo_t;

typedef struct network_ptr_s  {
	float	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current;

	// conductances and stp values
	float	*gNMDA, *gAMPA, *gGABAa, *gGABAb;
	float *gNMDA_r, *gNMDA_d, *gGABAb_r, *gGABAb_d;
	int*	I_set;
	simMode_t		memType; // FIXME: horrible naming
	int		allocated;			//!< true if all data has been allocated..
	float	*stpx, *stpu;

	unsigned short	*Npre;				//!< stores the number of input connections to the neuron
	unsigned short	*Npre_plastic;			//!< stores the number of plastic input connections
	float	*Npre_plasticInv;			//!< stores the 1/number of plastic input connections, for use on the GPU
	unsigned short	*Npost;				//!< stores the number of output connections from a neuron.
	unsigned int		*lastSpikeTime;			//!< storees the firing time of the neuron
	float	*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
	float	 *maxSynWt;			//!< maximum synaptic weight for given connection..
	uint32_t *synSpikeTime;
	uint32_t *neuronFiring;
	unsigned int		*cumulativePost;
	unsigned int		*cumulativePre;

	float 	*mulSynFast, *mulSynSlow;
	short int *cumConnIdPre;	//!< connectId, per synapse, presynaptic cumulative indexing

	short int *grpIds;

	/*!
	 * \brief 10 bit syn id, 22 bit neuron id, ordered based on delay
	 *
	 * allows maximum synapses of 1024 and maximum network size of 4 million neurons, with 64 bit representation. we can
	 * have larger networks for simulation
	 */
	 post_info_t	*postSynapticIds;

	 post_info_t	*preSynapticIds;
	delay_info_t    *postDelayInfo;  	//!< delay information
	unsigned int*		firingTableD1;
	unsigned int*		firingTableD2;

	float*		poissonFireRate;
	unsigned int*		poissonRandPtr;		//!< firing random number. max value is 10,000
	int2*		neuronAllocation;
	int3*		groupIdInfo;		//!< used for group Id calculations...
	short int*	synIdLimit;			//
	float*		synMaxWts;			//
	unsigned int*		nSpikeCnt;

	int** spkCntBuf; //!< for copying 2D array to GPU (see CpuSNN::allocateSNN_GPU)
	int* spkCntBufChild[MAX_GRP_PER_SNN]; //!< child pointers for above

	//!< homeostatic plasticity variables
	float*	baseFiringInv;
	float*	baseFiring;
	float*	avgFiring;

	float* 		testVar;
	float*		testVar2;
	uint32_t*	spikeGenBits;
	bool*		curSpike;
} network_ptr_t;

typedef struct group_info_s
{
  // properties of group of neurons size, location, initial weights etc.
	PoissonRate*		RatePtr;
	int			StartN;
	int			EndN;
	char		Type;
	int			SizeN;
	int			NumTraceN;
	short int  	MaxFiringRate; //!< this is for the monitoring mechanism, it needs to know what is the maximum firing rate in order to allocate a buffer big enough to store spikes...
	int			MonitorId;
	float   	RefractPeriod;
	int		CurrTimeSlice; //!< timeSlice is used by the Poisson generators in order to note generate too many or too few spikes within a window of time
	int		NewTimeSlice;
	uint32_t 	SliceUpdateTime;
	int 		FiringCount1sec;
	int 		numPostSynapses;
	int 		numPreSynapses;
	bool 		isSpikeGenerator;
	bool 		WithSTP;
	bool 		WithSTDP;
	bool 		WithHomeostasis;
//	bool 		WithConductances;
	int		homeoId;
	bool		FixedInputWts;
	int			Noffset;
	int8_t		MaxDelay;

	float		STP_U;
	float		STP_tau_u_inv;
	float		STP_tau_x_inv;
	float		TAU_LTP_INV;
	float		TAU_LTD_INV;
	float		ALPHA_LTP;
	float		ALPHA_LTD;

	bool withSpikeCounter; //!< if this flag is set, we want to keep track of how many spikes per neuron in the group
	int spkCntRecordDur; //!< record duration, after which spike buffer gets reset
	int spkCntRecordDurHelper; //!< counter to help make fast modulo
	int spkCntBufPos; //!< which position in the spike buffer the group has

	//!< homeostatic plasticity variables
	float		avgTimeScale;
	float 	avgTimeScale_decay;
	float		avgTimeScaleInv;
	float		homeostasisScale;

	SpikeGenerator* spikeGen;
	bool		newUpdates;  //!< FIXME this flag has mixed meaning and is not rechecked after the simulation is started
} group_info_t;

/*!
 * this group need not be shared with the GPU
 * separate group which has unique properties of
 * neuron in the current group.
 */
 typedef struct group_info2_s
 {
 	std::string		Name;
 	short		ConfigId;
  // properties of group of neurons size, location, initial weights etc.
  //<! homeostatic plasticity variables
 	float 		baseFiring;
 	float 		baseFiringSD;
 	float 		Izh_a;
 	float 		Izh_a_sd;
 	float 		Izh_b;
 	float 		Izh_b_sd;
 	float 		Izh_c;
 	float 		Izh_c_sd;
 	float 		Izh_d;
 	float 		Izh_d_sd;

  /*!
   * \brief when we call print state, should the group properties be printed.
   * default is false and we do not want any prints for the current group
   */
   bool		enablePrint;
   int			numPostConn;
   int			numPreConn;
   int			maxPostConn;
   int			maxPreConn;
   int			sumPostConn;
   int			sumPreConn;
} group_info2_t;



// FIXME: deprecated???
/*typedef struct grpConnInfo_s {
  int16_t	srcGrpId;			//!< group id
  int	srcStartN;		//!< starting neuron to begin computation
  int	srcEndN;			//!< ending neuron to stop computation
  int	grpDelayVector;	//!< a vector with ones in position having a given delay
  int	grpMaxM;			//!< the maximum value of the number of post-synaptic connections
  bool	hasCommonDelay;   //!< 'true' if the grpDelayVector is same as the neuron DelayVector
  bool	hasRandomConn;	//!< set to 'true' if the group has random connections
  int*	randomDelayPointer; //
  int16_t	fixedDestGrpCnt;	//!< destination group count
  int*	fixedDestGrps;		//!< connected destination groups array, (x=destGrpId, y=startN, z=endN, w=function pointer)
  int*	fixedDestParam;	//!< connected destination parameters ,  (x=Start, y=Width, z=Stride, w=height)
} grpConnInfo_t;
*/




/// **************************************************************************************************************** ///
/// CPUSNN CORE CLASS
/// **************************************************************************************************************** ///

/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
 class CpuSNN {

/// **************************************************************************************************************** ///
/// PUBLIC METHODS
/// **************************************************************************************************************** ///
 public:
 	CpuSNN(std::string& name, simMode_t simMode, loggerMode_t loggerMode, int ithGPU, int nConfig, int randSeed);
 	~CpuSNN();

	// +++++ PUBLIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	const static unsigned int MAJOR_VERSION = 2; //!< major release version, as in CARLsim X
	const static unsigned int MINOR_VERSION = 2; //!< minor release version, as in CARLsim 2.X



	// +++++ PUBLIC METHODS: SETTING UP A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// NOTE: there should be no default argument values in here, this should be handled by the user interface

	//! make connection of type "random","full","one-to-one" from each neuron in grpId1 to neurons in grpId2
	short int connect(int gIDpre, int gIDpost, const std::string& _type, float initWt, float maxWt, float _C,
		uint8_t minDelay, uint8_t maxDelay, float mulSynFast, float mulSynSlow, bool synWtType);

	//! make custom connections from grpId1 to grpId2
	// TODO: describe maxM and maxPreM
	short int connect(int gIDpre, int gIDpost, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
		bool synWtType,	int maxM, int maxPreM);


	//! creates a group of Izhikevich spiking neurons
	int createGroup(const std::string& grpName, int nNeur, int neurType, int configId);

	//! creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons). 
	int createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType, int configId);


  	//! sets custom values for conductance decay (\tau_decay) or disables conductances alltogether
  	// these will be applied to all connections in a network
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb,
		int configId);


	/*!
	 * \brief Sets the homeostasis parameters. g is the grpID, enable=true(false) enables(disables) homeostasis,
	 * configId is the configuration ID that homeostasis will be enabled/disabled, homeostasisScale is strength of
	 * homeostasis compared to the strength of normal LTP/LTD from STDP (which is 1), and avgTimeScale is the time
	 * frame over which the average firing rate is averaged (it should be larger in scale than STDP timescales).
	 */
	 void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale, int configId);

	//! Sets homeostatic target firing rate (enforced through homeostatic synaptic scaling)
	 void setHomeoBaseFiringRate(int groupId, float baseFiring, float baseFiringSD, int configId);


	//! Sets the Izhikevich parameters a, b, c, and d of a neuron group.
	/*! Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d standard
	 * deviation a_sd, b_sd, c_sd, and d_sd, respectively. */
	 void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
	 	float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId);

	//! Sets STDP params alphas and taus of a neuron group (post-synaptically)
	// TODO: make per connection, not per group
	 void setSTDP(int grpId, bool isSet, float alphaLTP, float tauLTP, float alphaLTD, float tauLTD, int configId);

	/*!
	 * \brief Sets STP params U, tau_u, and tau_x of a neuron group (pre-synaptically)
	 * CARLsim implements the short-term plasticity model of (Tsodyks & Markram, 1998).
	 * du/dt = -u/STP_tau_u + STP_U * (1-u-) * \delta(t-t_spk)
	 * dx/dt = (1-x)/STP_tau_x - u+ * x- * \delta(t-t_spk)
	 * dI/dt = -I/tau_S + A * u+ * x- * \delta(t-t_spk)
	 * where u- means value of variable u right before spike update, and x+ means value of variable x right after
	 * the spike update, and A is the synaptic weight.
	 * The STD effect is modeled by a normalized variable (0<=x<=1), denoting the fraction of resources that remain
	 * available after neurotransmitter depletion.
	 * The STF effect is modeled by a utilization parameter u, representing the fraction of available resources ready for
	 * use (release probability). Following a spike, (i) u increases due to spike-induced calcium influx to the
	 * presynaptic terminal, after which (ii) a fraction u of available resources is consumed to produce the post-synaptic
	 * current. Between spikes, u decays back to zero with time constant STP_tau_u (\tau_F), and x recovers to value one
	 * with time constant STP_tau_x (\tau_D).
	 * \param[in] grpId       pre-synaptic group id. STP will apply to all neurons of that group!
	 * \param[in] isSet       a flag whether to enable/disable STP
	 * \param[in] STP_tau_u   decay constant of u (\tau_F)
	 * \param[in] STP_tau_x   decay constant of x (\tau_D)
	 * \param[in] configId    configuration ID of group
	 */
	 void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x, int configId);



	// +++++ PUBLIC METHODS: RUNNING A SIMULATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief run the simulation for n sec
	 * \param[in] enablePrint 	enable printing of status information
	 * \param[in] copyState 	enable copying of data from device to host
	 */
	 int runNetwork(int _nsec, int _nmsec, bool enablePrint, bool copyState);



	// +++++ PUBLIC METHODS: INTERACTING WITH A SIMULATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! reads the network state from file
	 void readNetwork(FILE* fid);

	/*!
	 * \brief Reassigns fixed weights to values passed into the function in a single 1D float matrix called
	 * weightMatrix.  The user passes the connection ID (connectID), the weightMatrix, the matrixSize, and 
	 * configuration ID (configID).  This function only works for fixed synapses and for connections of type
	 * CONN_USER_DEFINED. Only the weights are changed, not the maxWts, delays, or connected values
	 */
	void reassignFixedWeights(short int connectId, float weightMatrix[], int matrixSize, int configId);

	void resetSpikeCntUtil(int grpId = -1); //!< resets spike count for particular neuron group

	/*!
	 * \brief reset Spike Counter to zero
	 * Manually resets the spike buffers of a Spike Counter to zero (for a specific group).
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time.
	 * \param grpId the group for which to reset the spike counts. Set to ALL if you want to reset all Spike Counters.
	 * \param configId the config id for which to reset the spike counts. Set to ALL if you want to reset all configIds
	 */
	void resetSpikeCounter(int grpId, int configId);

	/*!
	 * \brief A Spike Counter keeps track of the number of spikes per neuron in a group.
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur).
	 * After that, the spike buffers get reset to zero number of spikes.
	 * Works for excitatory/inhibitory neurons.
	 * The recording time can be set to any x number of ms, so that after x ms the spike counts will be reset
	 * to zero. If x==-1, then the spike counts will never be reset (should only overflow after 97 days of sim).
	 * Also, spike counts can be manually reset at any time by calling snn->resetSpikeCounter(group);
	 * At any time, you can call getSpikeCounter to get the spiking information out.
	 * You can have only one spike counter per group. However, a group can have both a SpikeMonitor and a SpikeCounter.
	 * \param grpId the group for which you want to enable a SpikeCounter
	 * \param recordDur number of ms for which to record spike numbers. Spike numbers will be reset to zero after
	 * this. Set frameDur to -1 to never reset spike counts. Default: -1.
	 */
	void setSpikeCounter(int grpId, int recordDur, int configId);
	
	//! sets up a spike generator
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId);

	//! sets up a spike monitor registered with a callback to process the spikes, there can only be one
	//! SpikeMonitor per group
	void setSpikeMonitor(int gid, SpikeMonitor* spikeMon, int configId);

	//! assign spike rate to poisson group
	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod, int configId);

	//! Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	//! weight values back to their default values by setting resetWeights = true.
	void updateNetwork(bool resetFiringInfo, bool resetWeights);

	//!< writes the network state to file
	void writeNetwork(FILE* fid);

	//! function writes population weights from gIDpre to gIDpost to file fname in binary.
	void writePopWeights(std::string fname, int gIDpre, int gIDpost, int configId);




	// +++++ PUBLIC METHODS: LOGGING / PLOTTING +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! how often network status should be printed (seconds)
	void setLogCycle(int showStatusCycle);

	/*!
	 * \brief Sets the file pointer of the debug log file
	 * \param[in] fpLog file pointer to new log file
	 */
	 void setLogDebugFp(FILE* fpLog);

	/*!
	 * \brief Sets the file pointers for all log files
	 * \param[in] fpOut file pointer for status info
	 * \param[in] fpErr file pointer for errors/warnings
	 * \param[in] fpDeb file pointer for debug info
	 * \param[in] fpLog file pointer for debug log file that contains all the above info
	 */
	 void setLogsFp(FILE* fpOut, FILE* fpErr, FILE* fpDeb, FILE* fpLog);


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	grpConnectInfo_t* getConnectInfo(short int connectId, int configId); //!< required for homeostasis
	int  getConnectionId(short int connId, int configId);

	// FIXME: fix this
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays);

	int  getGroupId(int groupId, int configId);
	group_info_t getGroupInfo(int groupId, int configId);
	std::string getGroupName(int grpId, int configId);

	loggerMode_t getLoggerMode() { return loggerMode_; }

	int getNumConfigurations()	{ return nConfig_; }	//!< gets number of network configurations
	int getNumConnections(short int connectionId);		//!< gets number of connections associated with a connection ID
	int getNumGroups() { return numGrp; }

	/*!
	 * \brief Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
	 * and the size of the 1D array in size.  gIDpre(post) is the group ID for the pre(post)synaptic group, 
	 * weights is a pointer to a single dimensional array of floats, size is the size of that array which is 
	 * returned to the user, and configID is the configuration ID of the SNN.  NOTE: user must free memory from
	 * weights to avoid a memory leak.  
	 */
	 void getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId = 0);

	 int getSimMode()		{ return simMode_; }
	 uint64_t getSimTime()		{ return simTime; }
	 uint32_t getSimTimeSec()	{ return simTimeSec; }
	 uint32_t getSimTimeMs()		{ return simTimeMs; }

	// TODO: same as spikeMonRT
	//TODO: may need to make it work for different configurations. -- KDC
	/*!
 	 * \brief Returns pointer to nSpikeCnt, which is a 1D array of the number of spikes every neuron in the group
	 *  has fired.  Takes the grpID and the simulation mode (CPU_MODE or GPU_MODE) as arguments.
	 */
	 unsigned int* getSpikeCntPtr(int grpId=ALL);

	/*!
	 * \brief return the number of spikes per neuron for a certain group
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur) at any point in time.
	 * \param grpId	the group for which you want the spikes
	 * \return pointer to array of ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (int) since the last reset.
	 */
	int* getSpikeCounter(int grpId, int configId);

	// FIXME: fix this
	// TODO: maybe consider renaming getPopWeightChanges
	 float* getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges);


	 int grpStartNeuronId(int g) { return grp_Info[g].StartN; }
	 int grpEndNeuronId(int g)   { return grp_Info[g].EndN; }
	 int grpNumNeurons(int g)    { return grp_Info[g].SizeN; }

	 bool isExcitatoryGroup(int g) { return (grp_Info[g].Type&TARGET_AMPA) || (grp_Info[g].Type&TARGET_NMDA); }
	 bool isInhibitoryGroup(int g) { return (grp_Info[g].Type&TARGET_GABAa) || (grp_Info[g].Type&TARGET_GABAb); }
	 bool isPoissonGroup(int g) { return (grp_Info[g].Type&POISSON_NEURON); }

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 */
	 void setCopyFiringStateFromGPU(bool _enableGPUSpikeCntPtr);

	 void setGroupInfo(int groupId, group_info_t info, int configId=ALL);
	 void setPrintState(int grpId, bool _status);


/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

	private:
	// +++++ CPU MODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	void CpuSNNinit();	//!< all unsafe operations of constructor

	void buildNetworkInit(unsigned int nNeur, unsigned int nPostSyn, unsigned int nPreSyn, unsigned int maxDelay);

	int  addSpikeToTable(int id, int g); //!< add the entry that the current neuron has spiked

	void buildGroup(int groupId);
	void buildNetwork();
	void buildPoissonGroup(int groupId);

	/*!
	 * \brief reset Spike Counters to zero if simTime % recordDur == 0
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur)
	 * after this period of time, the spike buffers need to be reset
	 * this function checks simTime vs. recordDur and resets the spike buffers if necessary
	 */
	void checkSpikeCounterRecordDur();
	
	void compactConnections(); //!< minimize any other wastage in that array by compacting the store
	void connectFull(grpConnectInfo_t* info);
	void connectOneToOne(grpConnectInfo_t* info);
	void connectRandom(grpConnectInfo_t* info);
	void connectUserDefined(grpConnectInfo_t* info);

	void deleteObjects();			//!< deallocates all used data structures in snn_cpu.cpp

	void doD1CurrentUpdate();
	void doD2CurrentUpdate();
	void doGPUSim();
	void doSnnSim();
	void doSTPUpdateAndDecayCond();

	void exitSimulation(int val);	//!< deallocates all dynamical structures and exits

	void findFiring();
	int  findGrpId(int nid); //!< for given neuron id, find group id

	void generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD);
	void generateSpikes();
	void generateSpikes(int grpId);
	void generateSpikesFromFuncPtr(int grpId);
	void generateSpikesFromRate(int grpId);

	int  getPoissNeuronPos(int nid);
	float getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId);

	void globalStateUpdate();

	void initSynapticWeights(); //!< init all wt to appropriate values (total size of syn conn is 'length')

	void makePtrInfo();				//!< creates CPU net ptrs

	unsigned int poissonSpike(unsigned int currTime, float frate, int refractPeriod); //!< for generateSpikesFromRate

	// NOTE: all these printer functions should be in printSNNInfo.cpp
	// FIXME: are any of these actually supposed to be public?? they are not yet in carlsim.h
	void printConnection(const std::string& fname);
	void printConnection(FILE* fp);
	void printConnection(int grpId, FILE* fp); //!< print the connection info of grpId
	void printConnectionInfo(FILE* fp);
	void printConnectionInfo2(FILE *fpg);
	void printCurrentInfo(FILE* fp); //!< for GPU debugging
	void printFiringRate(char *fname=NULL);
	void printGpuLoadBalance(bool init, int numBlocks, FILE* fp); //!< for GPU debugging
//	void printGpuLoadBalance(bool init=false, int numBlocks = MAX_BLOCKS, const FILE* fp); //!< for GPU debugging
	void printGroupInfo(FILE* fp);
	void printGroupInfo(std::string& strName);
	void printGroupInfo2(FILE* fpg);
	void printMemoryInfo(FILE* fp); //!< prints memory info to file
	void printNetworkInfo(FILE* fp);
	void printNeuronState(int grpId, FILE* fp);
	void printParameters(FILE *fp);
	void printPostConnection(FILE* fp); //!< print all post connections
	void printPostConnection(int grpId, FILE* fp);
	int  printPostConnection2(int grpId, FILE* fpg);
	void printPreConnection(FILE* fp); //!< print all pre connections
	void printPreConnection(int grpId, FILE* fp);
	int  printPreConnection2(int grpId, FILE* fpg);
	void printSimSummary(); 	//!< prints a simulation summary at the end of sim
	void printState(FILE* fp);
//	void printState(const char *str = "", const FILE* fp);
	void printTestVarInfo(FILE* fp, char* testString, bool test1=true, bool test2=true, bool test12=false,
							int subVal=0, int grouping1=0, int grouping2=0); //!< for GPU debugging
	void printTuningLog(FILE* fp);
	void printWeight(int grpId, const char *str = "");

	// FIXME: difference between the options? is one deprecated or are both still used?
	#if READNETWORK_ADD_SYNAPSES_FROM_FILE
	int readNetwork_internal(bool onlyPlastic);
	#else
	int readNetwork_internal();
	#endif

	void reorganizeDelay();
	void reorganizeNetwork(bool removeTempMemory);

	void resetConductances();
	void resetCounters();
	void resetCPUTiming();
	void resetCurrent();
	void resetFiringInformation(); //!< resets the firing information when updateNetwork is called
	void resetGroups();
	void resetNeuron(unsigned int nid, int grpId);
	void resetPointers(bool deallocate=false);
	void resetPoissonNeuron(unsigned int nid, int grpId);
	void resetPropogationBuffer();
	void resetSpikeCnt(int grpId=ALL);					//!< Resets the spike count for a particular group.
	void resetSynapticConnections(bool changeWeights=false);
	void resetTimingTable();

	inline void setConnection(int srcGrpId, int destGrpId, unsigned int src, unsigned int dest, float synWt,
		float maxWt, uint8_t dVal, int connProp, short int connId);

	void setGrpTimeSlice(int grpId, int timeSlice); //!< used for the Poisson generator. TODO: further optimize
	int setRandSeed(int seed);	//!< setter function for const member randSeed_

	void setupNetwork(bool removeTempMemory=true);

	void showStatus();

	void startCPUTiming();
	void stopCPUTiming();

	void swapConnections(int nid, int oldPos, int newPos);

	void updateAfterMaxTime();
	void updateParameters(int* numN, int* numPostSynapses, int* D, int nConfig=1);
	void updateSpikesFromGrp(int grpId);
	void updateSpikeGenerators();
	void updateSpikeGeneratorsInit();
	void updateSpikeMonitor(); //!< copy required spikes from firing buffer to spike buffer
	int  updateSpikeTables();
	void updateStateAndFiringTable();
	bool updateTime(); //!< updates simTime, returns true when a new second is started


	// +++++ GPU MODE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	// TODO: consider moving to snn_gpu.h
	void CpuSNNinit_GPU();	//!< initializes params needed in snn_gpu.cu (gets called in CpuSNN constructor)

	void allocateGroupId();
//	void allocateGroupParameters();
	void allocateNetworkParameters();
	void allocateSNN_GPU(); //!< allocates required memory and then initialize the GPU
	int  allocateStaticLoad(int bufSize);

	void assignPoissonFiringRate_GPU();

	void checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId);
	int  checkErrors(std::string kernelName, int numBlocks);
	int  checkErrors(int numBlocks);
	void checkGPUDevice();
	void checkInitialization(char* testString=NULL);
	void checkInitialization2(char* testString=NULL);

	void copyConnections(network_ptr_t* dest, int kind, int allocateMem);
	void copyFiringInfo_GPU();
	void copyFiringStateFromGPU (int grpId = -1);
	void copyGrpInfo_GPU(); //!< Used to copy grp_info to gpu for setSTDP, setSTP, and setHomeostasis
	void copyNeuronParameters(network_ptr_t* dest, int kind, int allocateMem, int grpId = -1);
	void copyNeuronState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId=-1);
	void copyParameters();
	void copyPostConnectionInfo(network_ptr_t* dest, int allocateMem);
	void copyState(network_ptr_t* dest, int allocateMem);
	void copySTPState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem);
	void copyUpdateVariables_GPU(); //!< copies wt / neuron state / STP state info from host to device
	void copyWeightsGPU(unsigned int nid, int src_grp);
	void copyWeightState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, //!< copy presynaptic info
		int allocateMem, int grpId=-1);

	void deleteObjects_GPU();		//!< deallocates all used data structures in snn_gpu.cu
	void doCurrentUpdate_GPU();
	void dumpSpikeBuffToFile_GPU(int gid);
	void findFiring_GPU();

	/*!
	 * \brief return the number of spikes per neuron for a certain group in GPU mode
	 * A Spike Counter keeps track of all spikes per neuron for a certain time period (recordDur)
	 * at any point in time.
	 * \param grpId	the group for which you want the spikes
	 * \return pointer to array of unsigned ints. Number of elements in array is the number of neurons in group.
	 * Each entry is the number of spikes for this neuron (unsigned int) since the last reset.
	 */
	int* getSpikeCounter_GPU(int grpId, int configId);

	void globalStateUpdate_GPU();
	void initGPU(int gridSize, int blkSize);

	void resetFiringInformation_GPU(); //!< resets the firing information in GPU_MODE when updateNetwork is called
	void resetGPUTiming();
	void resetSpikeCnt_GPU(int _startGrp, int _endGrp); //!< Utility function to clear spike counts in the GPU code.

	/*!
	 * \brief reset spike counter to zero in GPU mode
	 * Buffers get reset to zero automatically after recordDur. However, you can reset the buffer manually at any
	 * point in time through calling the public equivalent. This one gets called in
	 * CpuSNN::resetSpikeCounter if we're running GPU mode.
	 * \param grpId	the group for which you want to reset the spikes
	 */
	void resetSpikeCounter_GPU(int grpId, int configId);

	void setSpikeGenBit_GPU(unsigned int nid, int grp);
	void showStatus_GPU();
	void spikeGeneratorUpdate_GPU();
	void startGPUTiming();
	void stopGPUTiming();
	void testSpikeSenderReceiver(FILE* fpLog, int simTime);
	void updateNetwork_GPU(bool resetFiringInfo); //!< Allows parameters to be reset in the middle of the simulation
	void updateSpikeMonitor_GPU();
	void updateStateAndFiringTable_GPU();
	void updateTimingTable_GPU();


	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	FILE* readNetworkFID;

	const std::string networkName_;	//!< network name
	const simMode_t simMode_;		//!< current simulation mode (CPU_MODE or GPU_MODE) FIXME: give better name
	const loggerMode_t loggerMode_;	//!< current logger mode (USER, DEVELOPER, SILENT, CUSTOM)
	const int ithGPU_;				//!< on which CUDA device to establish a context (only in GPU_MODE)
	const int nConfig_;				//!< number of network configurations
	const int randSeed_;			//!< random number seed to use


	//! temporary variables created and deleted by network after initialization
	uint8_t			*tmp_SynapticDelay;

	bool simulatorDeleted;
	bool spikeRateUpdated;

	float prevCpuExecutionTime;
	float cpuExecutionTime;
	float prevGpuExecutionTime;
	float gpuExecutionTime;


	//! properties of the network (number of groups, network name, allocated neurons etc..)
	bool			doneReorganization;
	bool			memoryOptimized;

	int				numGrp;
	int				numConnections;		//!< number of connection calls (as in snn.connect(...))
	//! keeps track of total neurons/presynapses/postsynapses currently allocated
	unsigned int	allocatedN;
	unsigned int	allocatedPre;
	unsigned int	allocatedPost;

	grpConnectInfo_t* connectBegin;
	short int 	*cumConnIdPre;		//!< connId, per synapse, presynaptic cumulative indexing
	float 		*mulSynFast;	//!< scaling factor for fast synaptic currents, per connection
	float 		*mulSynSlow;	//!< scaling factor for slow synaptic currents, per connection

	short int *grpIds;

	//! Buffer to store spikes
	PropagatedSpikeBuffer* pbuf;

	bool sim_with_conductances;		//!< flag to inform whether we run in COBA mode (true) or CUBA mode (false)
	bool sim_with_NMDA_rise;	//!< a flag to inform whether to compute NMDA rise time
	bool sim_with_GABAb_rise;	//!< a flag to inform whether to compute GABAb rise time
	float dAMPA;				//!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	float rNMDA;				//!< multiplication factor for rise time of NMDA
	float dNMDA;				//!< multiplication factor for decay time of NMDA
	float dGABAa;				//!< multiplication factor for decay time of GABAa
	float rGABAb;				//!< multiplication factor for rise time of GABAb
	float dGABAb;				//!< multiplication factor for decay time of GABAb

	bool sim_with_fixedwts;
	bool sim_with_stdp;
	bool sim_with_stp;
	bool sim_with_spikecounters; //!< flag will be true if there are any spike counters around
	//! flag to enable the copyFiringStateInfo from GPU to CPU
	bool enableGPUSpikeCntPtr;

	// spiking network related info.. neurons, synapses and network parameters
	int	        	numN,numNReg,numPostSynapses,D,numNExcReg,numNInhReg, numPreSynapses;
	int   			numNExcPois, numNInhPois, numNPois;
	float       	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current;
	bool			*curSpike;
	unsigned int         	*nSpikeCnt;     //!< spike counts per neuron
	unsigned short       	*Npre;			//!< stores the number of input connections to the neuron
	unsigned short			*Npre_plastic;	//!< stores the number of excitatory input connection to the input
	unsigned short       	*Npost;			//!< stores the number of output connections from a neuron.
	uint32_t    	*lastSpikeTime;	//!< stores the most recent spike time of the neuron
	float			*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
	float	 		*maxSynWt;		//!< maximum synaptic weight for given connection..
	uint32_t    	*synSpikeTime;	//!< stores the spike time of each synapse
	unsigned int		postSynCnt; //!< stores the total number of post-synaptic connections in the network
	unsigned int		preSynCnt; //!< stores the total number of pre-synaptic connections in the network
	float			*intrinsicWeight;
	//added to include homeostasis. -- KDC
	float					*baseFiring;
	float                 *avgFiring;
	unsigned int	        *nextTaste;
	unsigned int	        *nextDeath;
	unsigned int		*cumulativePost;
	unsigned int		*cumulativePre;
	post_info_t		*preSynapticIds;
	post_info_t		*postSynapticIds;		//!< 10 bit syn id, 22 bit neuron id, ordered based on delay
	delay_info_t    *postDelayInfo;      	//!< delay information

	//! size of memory used for different parts of the network
	typedef struct snnSize_s {
		unsigned int		neuronInfoSize;
		unsigned int		synapticInfoSize;
		unsigned int		networkInfoSize;
		unsigned int		spikingInfoSize;
		unsigned int		debugInfoSize;
		unsigned int		addInfoSize;	//!< includes random number generator etc.
		unsigned int		blkInfoSize;
		unsigned int		monitorInfoSize;
	} snnSize_t;

	snnSize_t cpuSnnSz;
	snnSize_t gpuSnnSz;
	unsigned int 	postConnCnt;
	unsigned int	preConnCnt;

	//! firing info
	unsigned int		*timeTableD2;
	unsigned int		*timeTableD1;
	unsigned int     *firingTableD2;
	unsigned int     *firingTableD1;
	unsigned int		maxSpikesD1, maxSpikesD2;

	//time and timestep
	unsigned int	simTimeMs;
	uint64_t        simTimeSec;		//!< this is used to store the seconds.
	unsigned int	simTime;		//!< this value is not reset but keeps increasing to its max value.
	unsigned int		spikeCountAll1sec, secD1fireCntHost, secD2fireCntHost;	//!< firing counts for each second
	unsigned int		spikeCountAll, spikeCountD1Host, spikeCountD2Host;	//!< overall firing counts values
	unsigned int     nPoissonSpikes;

	//cuda keep track of performance...
	#if __CUDA3__
	unsigned int    timer;
	#elif __CUDA5__
	StopWatchInterface *timer;
	#endif
	float		cumExecutionTime;
	float           lastExecutionTime;

	FILE*	fpOut_;			//!< fp of where to write all simulation output (status info) if not in silent mode
	FILE*	fpErr_;			//!< fp of where to write all errors if not in silent mode
	FILE*	fpDeb_;			//!< fp of where to write all debug info if not in silent mode
	FILE*	fpLog_;
	int showStatusCycle_;	//!< how often to call showStatus (seconds)
	int showStatusCnt_; //!< internal counter to implement fast version of !(simTimeSec%showStatusCycle_)

	//spike monitor code...
	unsigned int			numSpikeMonitor;
	unsigned int			monGrpId[MAX_GRP_PER_SNN];
	unsigned int			monBufferPos[MAX_GRP_PER_SNN];
	unsigned int			monBufferSize[MAX_GRP_PER_SNN];
	unsigned int*		monBufferFiring[MAX_GRP_PER_SNN];
	unsigned int*		monBufferTimeCnt[MAX_GRP_PER_SNN];
	SpikeMonitor*	monBufferCallback[MAX_GRP_PER_SNN];

	unsigned int	numSpikeGenGrps;

	int numSpkCnt; //!< number of real-time spike monitors in the network
	int* spkCntBuf[MAX_GRP_PER_SNN]; //!< the actual buffer of spike counts (per group, per neuron)


	/* Tsodyks & Markram (1998), where the short-term dynamics of synapses is characterized by three parameters:
	   U (which roughly models the release probability of a synaptic vesicle for the first spike in a train of spikes),
	   D (time constant for recovery from depression), and F (time constant for recovery from facilitation). */
	   float *stpu;
	   float *stpx;
	   
	   float *gAMPA;
	   float *gNMDA;
	   float *gNMDA_r;
	   float *gNMDA_d;
	   float *gGABAa;
	   float *gGABAb;
	   float *gGABAb_r;
	   float *gGABAb_d;

	   network_info_t 	net_Info;

	   network_ptr_t  		cpu_gpuNetPtrs;
	   network_ptr_t   	cpuNetPtrs;

	//int   Noffset;
	int	  NgenFunc;					//!< this counts the spike generator offsets...

	bool finishedPoissonGroup;		//!< This variable is set after we have finished
	//!< creating the poisson group...

	bool showGrpFiringInfo;

	// gpu related info...
	// information about various data allocated at GPU side...
	unsigned int	gpu_tStep, gpu_simSec;		//!< this is used to store the seconds.
	unsigned int	gpu_simTime;				//!< this value is not reset but keeps increasing to its max value.

	group_info_t	  	grp_Info[MAX_GRP_PER_SNN];
	group_info2_t		grp_Info2[MAX_GRP_PER_SNN];
	float*			testVar, *testVar2;
	uint32_t*	spikeGenBits;


  /* deprecated
     void initThalInput();
     void initThalInput_GPU();

     void updateNetwork();

     // input noise related codes...
     void randomNoiseCurrent(float neuronPercentage, float currentStrength, int groupId=-1);

     void updateRandomProperty(); // for random thalamic input


     // noise generator related info...
     int			numNoise;		// Total number of noise generators
     int			numRandNeurons;	// Total number of neurons selected each time step
     int*		randNeuronId;	// Store the ids of neuron which will get random input current..
     noiseGenProperty_t	noiseGenGroup[MAX_GRP_PER_SNN];

  */
 };

/*
  typedef struct noiseGenProperty_s  {
  float	neuronPercentage;	// percentage of neuron that needs random inputs..
  float	currentStrength;	// strength of the noise current applied...
  int		groupId;			// group id.. cross references to group properties
  int		nstart;				// start of the neuron id
  int		ncount;				// total neuron in current group
  int		rand_ncount;		// (ncount*neuronPercentage/100) neurons will be selected
  } noiseGenProperty_t;

*/
#endif
