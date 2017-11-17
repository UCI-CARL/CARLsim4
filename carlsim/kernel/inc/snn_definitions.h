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

#ifndef _SNN_DEFINITIONS_H_
#define _SNN_DEFINITIONS_H_

// TODO: as Kris put it, this should really be called something like
// some_random_macros_and_hardware_limitation_dependent_param_checks.h ... for example, the MAX_... defines
// should really be private members of SNN. These ranges are limited by the data structures that implement
// the corresponding functionality. For example, you can't just set MAX_CONN_PER_SNN > 32768, because connIds
// are stored as short int.

//    NEURON ORGANIZATION/ARRANGEMENT MAP
//    <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory --> | <--------- EXTERNAL NEURONS --------->
//    Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson |
//    <--- numNExcReg --> | <-- numNInhReg --> | <-- numNInhPois -> | <---numNExcPois--> |
//    <------REGULAR NEURON REGION ----------> | <----- POISSON NEURON REGION ---------> |
//    <----numNReg=(numNExcReg+numNInhReg)---> | <--numNPois=(numNInhPois+numNExcPois)-> | <------------ numNExtern ------------>
//    <------------------ ALL LOCAL NEURONS (numN=numNReg+numNPois) -------------------> | <- ALL EXTERNAL NEURONS (numNExtern)->
//    <-------------------------------- ALL ASSIGNED NEURONS (numNAssigned=numN+numNExtern) ------------------------------------>
//    Note: this organization scheme is only used/needed for the gpu_static code.
#define IS_POISSON_NEURON(nid, numNReg, numNPois) ((nid) >= (numNReg) && ((nid) < (numNReg + numNPois)))
#define IS_REGULAR_NEURON(nid, numNReg, numNPois) (((nid) < (numNReg)) && ((nid) < (numNReg + numNPois)))
#define IS_INHIBITORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) >= (numNExcReg)) && ((nid) < (numNReg + numNInhPois)))
#define IS_EXCITATORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) < (numNReg)) && (((nid) < (numNExcReg)) || ((nid) >=  (numNReg + numNInhPois))))
#define IS_LOCAL_NEURON(nid, numN, numNAssigned) ((nid) < (numN))
#define IS_EXTERNAL_NEURON(nid, numN, numNAssigned) ((nid) >= (numN) && (nid) < (numNAssigned))

#define STATIC_LOAD_START(n)  (n.x)
#define STATIC_LOAD_GROUP(n)  (n.y & 0xff)
#define STATIC_LOAD_SIZE(n)   ((n.y >> 16) & 0xff)

//#define MAX_NUMBER_OF_NEURONS_BITS  (20)
//#define MAX_NUMBER_OF_GROUPS_BITS   (32 - MAX_NUMBER_OF_NEURONS_BITS)
//#define MAX_NUMBER_OF_NEURONS_MASK  ((1 << MAX_NUMBER_OF_NEURONS_BITS) - 1)
//#define MAX_NUMBER_OF_GROUPS_MASK   ((1 << MAX_NUMBER_OF_GROUPS_BITS) - 1)
//#define SET_FIRING_TABLE(nid, gid)  (((gid) << MAX_NUMBER_OF_NEURONS_BITS) | (nid))
//#define GET_FIRING_TABLE_NID(val)   ((val) & MAX_NUMBER_OF_NEURONS_MASK)
//#define GET_FIRING_TABLE_GID(val)   (((val) >> MAX_NUMBER_OF_NEURONS_BITS) & MAX_NUMBER_OF_GROUPS_MASK)

//!< Used for in the function getConnectionId
#define CHECK_CONNECTION_ID(n,total) { assert(n >= 0); assert(n < total); }

// Macros for STP
// we keep a history of STP values to compute resource change over time
// there are two problems to solve:
// 1) parallelism. we update postsynaptic current changes in synapse parallelism, but stpu and stpx need to be updated
//    only once for each pre-neuron (in neuron parallelism)
// 2) non-zero delays. as a post-neuron you want the spike to be weighted by what the utility and resource
//    variables were when pre spiked, not from the time at which the spike arrived at post.
// the macro is slightly faster than an inline function, but we should consider changing it anyway because
// it's unsafe
//#define STP_BUF_SIZE 32
// \FIXME D is the SNN member variable for the max delay in the network, give it a better name dammit!!
// we actually need D+1 entries. Say D=1ms. Then to update the current we need u^+ (right after the pre-spike, so
// at t) and x^- (right before the spike, so at t-1).
#define STP_BUF_POS(nid, t, maxDelay) (nid * (maxDelay + 1) + ((t + 1) % (maxDelay + 1)))

// use these macros for logging / error printing
// every message will be printed to one of fpOut_, fpErr_, fpDeb_ depending on the nature of the message
// Additionally, every message gets printed to some log file fpLog_. This is different from fpDeb_ for
// the case in which you want the two to be different (e.g., developer mode, in which you would like to
// see all debug info (stdout) but also have it saved to a file
#define KERNEL_ERROR(formatc, ...) {	KERNEL_ERROR_PRINT(fpErr_,formatc,##__VA_ARGS__); \
										KERNEL_DEBUG_PRINT(fpLog_,"ERROR",formatc,##__VA_ARGS__); }
#define KERNEL_WARN(formatc, ...) {		KERNEL_WARN_PRINT(fpErr_,formatc,##__VA_ARGS__); \
										KERNEL_DEBUG_PRINT(fpLog_,"WARN",formatc,##__VA_ARGS__); }
#define KERNEL_INFO(formatc, ...) {		KERNEL_INFO_PRINT(fpInf_,formatc,##__VA_ARGS__); \
										KERNEL_DEBUG_PRINT(fpLog_,"INFO",formatc,##__VA_ARGS__); }
#define KERNEL_DEBUG(formatc, ...) {	KERNEL_DEBUG_PRINT(fpDeb_,"DEBUG",formatc,##__VA_ARGS__); \
										KERNEL_DEBUG_PRINT(fpLog_,"DEBUG",formatc,##__VA_ARGS__); }

// cast to FILE* in case we're getting a const FILE* in
#define KERNEL_ERROR_PRINT(fp, formatc, ...) fprintf((FILE*)fp,"\033[31;1m[ERROR %s:%d] " formatc "\033[0m \n",__FILE__,__LINE__,##__VA_ARGS__)
#define KERNEL_WARN_PRINT(fp, formatc, ...) fprintf((FILE*)fp,"\033[33;1m[WARNING %s:%d] " formatc "\033[0m \n",__FILE__,__LINE__,##__VA_ARGS__)
#define KERNEL_INFO_PRINT(fp, formatc, ...) fprintf((FILE*)fp,formatc "\n",##__VA_ARGS__)
#define KERNEL_DEBUG_PRINT(fp, type, formatc, ...) fprintf((FILE*)fp,"[" type " %s:%d] " formatc "\n",__FILE__,__LINE__,##__VA_ARGS__)


#define MAX_NUM_POST_SYN 100000
#define MAX_NUM_PRE_SYN 200000
#define MAX_SYN_DELAY 20

// increasing the following numbers will increase the load on constant memory
// until a hard limit is reached, which is given by the datatype of the variable
#define MAX_CONN_PER_SNN 256	// hard limit: 2^16
#define MAX_GRP_PER_SNN 128		// hard limit: 2^16
#define MAX_NET_PER_SNN 32		// the maximum number of local networks in a simulation

#ifdef __NO_CUDA__
	#define CPU_RUNTIME_BASE 0
#else
	#define CPU_RUNTIME_BASE 8
#endif

#define NUM_CPU_CORES sysconf(_SC_NPROCESSORS_ONLN)

#define GPU_RUNTIME_BASE 0

#define COND_INTEGRATION_SCALE	2

#define NEURON_MAX_FIRING_RATE 500

#define STDP(t,a,b)       ((a)*exp(-(t)*(b))) // consider to use __expf(), which is accelerated by GPU hardware

#define MAX_TIME_SLICE 1000
#define MAX_SIMULATION_TIME     INT_MAX
#define LARGE_NEGATIVE_VALUE    (-(1 << 30))

#define TIMING_COUNT  1024 // (1000+maxDelay_) rounded to multiple 128


#define MAX_SPIKE_MON_BUFFER_SIZE 52428800 // about 50 MB. size is in bytes. Max size of reduced AER vector in spikeMonitorCore objects.
#define LONG_SPIKE_MON_DURATION 600000 // about 10 minutes
#define LARGE_SPIKE_MON_GRP_SIZE 5000 // about 10 minutes

#define MAX_NEURON_MON_BUFFER_SIZE 52428800 // about 50 MB. size is in bytes. (???)
#define LONG_NEURON_MON_DURATION 100000       // about 100 seconds
#define MAX_NEURON_MON_GRP_SZIE 128

// This flag is used when having a common poisson generator for both CPU and GPU simulation
// We basically use the CPU poisson generator. Evaluate if there is any firing due to the
// poisson neuron. Copy that curFiring status to the GPU which uses that for evaluation
// of poisson firing
#define TESTING_CPU_GPU_POISSON 			(0)

#define MAX_GRPS_PER_BLOCK 		100
#define MAX_BLOCKS         		120

//#define CONN_SYN_NEURON_BITS	20                               //!< last 20 bit denote neuron id. 1 Million neuron possible
//#define CONN_SYN_BITS			(32 -  CONN_SYN_NEURON_BITS)	 //!< remaining 12 bits denote connection id
//#define CONN_SYN_NEURON_MASK    ((1 << CONN_SYN_NEURON_BITS) - 1)
//#define CONN_SYN_MASK      		((1 << CONN_SYN_BITS) - 1)
//#define GET_CONN_NEURON_ID(a) (((unsigned int)a.postId) & CONN_SYN_NEURON_MASK)
//#define GET_CONN_SYN_ID(b)    (((unsigned int)b.postId) >> CONN_SYN_NEURON_BITS)
//#define GET_CONN_GRP_ID(c)    (c.grpId)
//#define SET_CONN_ID(a,b)      ((b) > CONN_SYN_MASK) ? (fprintf(stderr, "Error: Syn Id exceeds maximum limit (%d)\n", CONN_SYN_MASK)): (((b)<<CONN_SYN_NEURON_BITS)+((a)&CONN_SYN_NEURON_MASK))

#define GROUP_ID_MASK 0x0000ffff
#define SYNAPSE_ID_MASK 0x0000ffff
#define MAX_SYN_PER_NEURON 65535
#define NUM_SYNAPSE_BITS  (16)

#define GET_CONN_NEURON_ID(val) (val.nId)
#define GET_CONN_SYN_ID(val) (val.gsId & SYNAPSE_ID_MASK)
#define GET_CONN_GRP_ID(val) ((val.gsId >> NUM_SYNAPSE_BITS) & GROUP_ID_MASK)

#define CONNECTION_INITWTS_RANDOM    	0
#define CONNECTION_CONN_PRESENT  		1
#define CONNECTION_FIXED_PLASTIC		2
#define CONNECTION_INITWTS_RAMPUP		3
#define CONNECTION_INITWTS_RAMPDOWN		4

#define SET_INITWTS_RANDOM(a)		((a & 1) << CONNECTION_INITWTS_RANDOM)
#define SET_CONN_PRESENT(a)		((a & 1) << CONNECTION_CONN_PRESENT)
#define SET_FIXED_PLASTIC(a)		((a & 1) << CONNECTION_FIXED_PLASTIC)
#define SET_INITWTS_RAMPUP(a)		((a & 1) << CONNECTION_INITWTS_RAMPUP)
#define SET_INITWTS_RAMPDOWN(a)		((a & 1) << CONNECTION_INITWTS_RAMPDOWN)

#define GET_INITWTS_RANDOM(a)		(((a) >> CONNECTION_INITWTS_RANDOM) & 1)
#define GET_CONN_PRESENT(a)		(((a) >> CONNECTION_CONN_PRESENT) & 1)
#define GET_FIXED_PLASTIC(a)		(((a) >> CONNECTION_FIXED_PLASTIC) & 1)
#define GET_INITWTS_RAMPUP(a)		(((a) >> CONNECTION_INITWTS_RAMPUP) & 1)
#define GET_INITWTS_RAMPDOWN(a)		(((a) >> CONNECTION_INITWTS_RAMPDOWN) & 1)

#endif
