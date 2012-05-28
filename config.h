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

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <math.h>

#if _WIN32 || _WIN64
	typedef __int8 int8_t;
	typedef __int16 int16_t;
	typedef __int32 int32_t;
	typedef __int64 int64_t;
	typedef unsigned __int8 uint8_t;
	typedef unsigned __int16 uint16_t;
	typedef unsigned __int32 uint32_t;
	typedef unsigned __int64 uint64_t;
#else
	#include <stdint.h>
#endif

#define MAX_numPostSynapses (10000)
#define MAX_numPreSynapses (20000)
#define MAX_SynapticDelay (20)

//#define CONDUCTANCES 		1
#define COND_INTEGRATION_SCALE	2
//#define STP

#define UNKNOWN_NEURON_MAX_FIRING_RATE    	25
#define INHIBITORY_NEURON_MAX_FIRING_RATE 	1000
#define EXCITATORY_NEURON_MAX_FIRING_RATE 	1000
#define POISSON_MAX_FIRING_RATE 	  	1000

#define STDP(t,a,b)       ((a)*exp(-(t)*(b)))

//#define LTD(t,a,b)       (ALPHA_LTD*exp(-(t)/TAU_LTD))
//#define LTP(t,a,b)       (ALPHA_LTP*exp(-(t)/TAU_LTP))
//#define LTD(t,a,b)       (ALPHA_LTD*exp(-(t)/TAU_LTD))

#define GPU_LTP(t)   (gpuNetInfo.ALPHA_LTP*__expf(-(t)/gpuNetInfo.TAU_LTP))
#define GPU_LTD(t)   (gpuNetInfo.ALPHA_LTD*__expf(-(t)/gpuNetInfo.TAU_LTD))

#define PROPOGATED_BUFFER_SIZE  (1023)
#define MAX_SIMULATION_TIME     ((uint32_t)(0x7fffffff))
#define LARGE_NEGATIVE_VALUE    (-(1 << 30))

#define S_SCALING		 (1.0f)
#define S_MAX			 (10.0f/S_SCALING)

#define HISTOGRAM_SIZE		100

#define DEBUG_LEVEL		0

#define MAX_GRP_PER_SNN 250

// This option effects readNetwork()'s behavior.  Setting this option to 1 will cause 
// the network file to be read twice, once for plastic synapses and then again for 
// fixed synapses.  For large networks this could be a substantial speed reduction; 
// however, it uses much less memory than setting it to 0.
#define READNETWORK_ADD_SYNAPSES_FROM_FILE 1

#define INHIBITORY_STDP

//#define NEURON_NOISE

#define STP_BUF_SIZE 32

// useful during testing and development. carries out series of checks 
// to ensure simulator is working correctly
#ifndef TESTING
 #define TESTING 					(0)
#endif

// does more kind of checking in the kernel to ensure nothing is screwed up...
#ifndef ENABLE_MORE_CHECK
 #define ENABLE_MORE_CHECK				(0)
#endif


// This flag is used when having a common poisson generator for both CPU and GPU simulation
// We basically use the CPU poisson generator. Evaluate if there is any firing due to the
// poisson neuron. Copy that curFiring status to the GPU which uses that for evaluation
// of poisson firing
#define TESTING_CPU_GPU_POISSON 			(0)



	/****************************/
	//  debug related stuff.. inspiration from
	// http://www.decompile.com/cpp/faq/file_and_line_error_string.htm
	#define STRINGIFY(x) #x
	#define TOSTRING(x) STRINGIFY(x)
	#define AT __FILE__ ":" TOSTRING(__LINE__)

	inline void error(FILE *fp, const char *location, const char *msg, int sec, int step)
	{
	  fprintf(fp, "(wt=%d,ms=%d) Error at %s: %s\n", sec, step, location, msg);
	}

	inline void debug(FILE *fp, const char *location, const char *msg, int sec, int step)
	{
	  fprintf(fp, "(wt=%d,ms=%d) Executing %s: %s\n", sec, step, location, msg);
	}

	#define DBG(num,fp,loc,msg)  if( DEBUG_LEVEL >= num ) debug(fp,loc,msg,simTimeSec,simTimeMs);


#define MAX_NEURON_CHUNK_SIZE 				   (750)
/*

#define FACTOR			1
#define IMAGE_SIZE  	(16*FACTOR)

#define MAX_CV_WINDOW_PIXELS 				   (1000)

#define MIN_CV_WINDOW_PIXELS 				   (100)
*/
#endif
