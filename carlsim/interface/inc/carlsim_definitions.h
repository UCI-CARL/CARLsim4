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

#ifndef _CARLSIM_DEFINITIONS_H_
#define _CARLSIM_DEFINITIONS_H_



//! CARLsim common definitions
#define ALL -1 //!< used for the set* methods to specify all groups or a specific group
#define ANY -1 //!< used for create* method to specify any GPU or a specific GPU


#define SYN_FIXED      false
#define SYN_PLASTIC    true

// Bit flags to be used to specify the type of neuron.  Future types can be added in the future such as Dopamine, etc.
// Yes, they should be bit flags because some neurons release more than one transmitter at a synapse.
#define UNKNOWN_NEURON	(0)
#define POISSON_NEURON	(1 << 0)
#define TARGET_AMPA	(1 << 1)
#define TARGET_NMDA	(1 << 2)
#define TARGET_GABAa	(1 << 3)
#define TARGET_GABAb	(1 << 4)
#define TARGET_DA		(1 << 5)
#define TARGET_5HT		(1 << 6)
#define TARGET_ACh		(1 << 7)
#define TARGET_NE		(1 << 8)

#define INHIBITORY_NEURON 		(TARGET_GABAa | TARGET_GABAb)
#define EXCITATORY_NEURON 		(TARGET_NMDA | TARGET_AMPA)
#define DOPAMINERGIC_NEURON		(TARGET_DA | EXCITATORY_NEURON)
#define EXCITATORY_POISSON 		(EXCITATORY_NEURON | POISSON_NEURON)
#define INHIBITORY_POISSON		(INHIBITORY_NEURON | POISSON_NEURON)
#define IS_INHIBITORY_TYPE(type)	(((type) & TARGET_GABAa) || ((type) & TARGET_GABAb))
#define IS_EXCITATORY_TYPE(type)	(!IS_INHIBITORY_TYPE(type))

#define CARLSIM_ERROR(where, what) fprintf(stderr,"\033[31;1m[USER ERROR %s] %s\033[0m\n",where,what)
#define CARLSIM_WARN(where, what) fprintf(stderr,"\033[33;1m[USER WARNING %s] %s\033[0m\n",where,what)
#define CARLSIM_INFO(where, what) fprintf(stdout,"\033[%s] %s\n",where,what)

#define MAX_NUM_CUDA_DEVICES 8

// maximum number of compartmental connections allowed per group
#define MAX_NUM_COMP_CONN 4

#endif