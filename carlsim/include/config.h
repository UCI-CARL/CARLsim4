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

#ifndef _CONFIG_H_
#define _CONFIG_H_

// TODO: as Kris put it, this should really be called something like
// some_random_macros_and_hardware_limitation_dependent_param_checks.h ... for example, the MAX_... defines
// should really be private members of CpuSNN. These ranges are limited by the data structures that implement
// the corresponding functionality. For example, you can't just set MAX_nConnections > 32768, because connIds
// are stored as short int. 

#define MAX_nPostSynapses 10000
#define MAX_nPreSynapses 20000
#define MAX_SynapticDelay 20
#define MAX_nConnections 32768			//!< max allowed number of connect() calls by the user (used for mulSynFast)
#define MAX_nConfig 100

//#define CONDUCTANCES 		1
#define COND_INTEGRATION_SCALE	2
//#define STP

#define UNKNOWN_NEURON_MAX_FIRING_RATE    	25
#define INHIBITORY_NEURON_MAX_FIRING_RATE 	520
#define EXCITATORY_NEURON_MAX_FIRING_RATE 	520
#define POISSON_MAX_FIRING_RATE 	  		520

#define STDP(t,a,b)       ((a)*exp(-(t)*(b)))

#define GPU_LTP(t)   (gpuNetInfo.ALPHA_LTP*__expf(-(t)/gpuNetInfo.TAU_LTP))
#define GPU_LTD(t)   (gpuNetInfo.ALPHA_LTD*__expf(-(t)/gpuNetInfo.TAU_LTD))

#define PROPAGATED_BUFFER_SIZE  (1023)
#define MAX_SIMULATION_TIME     ((uint32_t)(0x7fffffff))
#define LARGE_NEGATIVE_VALUE    (-(1 << 30))

#define S_SCALING		 (1.0f)
#define S_MAX			 (10.0f/S_SCALING)

#define HISTOGRAM_SIZE		100

#define DEBUG_LEVEL		0

#define MAX_GRP_PER_SNN 250


// set to 1 if doing regression tests
// will make private members public and disable output/logging
//#define REGRESSION_TESTING 0


// This option effects readNetwork()'s behavior.  Setting this option to 1 will cause 
// the network file to be read twice, once for plastic synapses and then again for 
// fixed synapses.  For large networks this could be a substantial speed reduction; 
// however, it uses much less memory than setting it to 0.
#define READNETWORK_ADD_SYNAPSES_FROM_FILE 1

#define INHIBITORY_STDP

//#define NEURON_NOISE

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


#define MAX_NEURON_CHUNK_SIZE 				   (750)

#endif
