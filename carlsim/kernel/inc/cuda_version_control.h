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

#ifndef _CUDA_VERSION_CONTROL_H_
#define _CUDA_VERSION_CONTROL_H_

#ifndef __NO_CUDA__
	#include <cuda.h>
	#include <cuda_runtime.h>

	// we no longer support CUDA3 and CUDA4, but keep cuda_version_control.h for
	// handling future CUDA toolkit API differences
	#if defined(__CUDA5__) || defined(__CUDA6__) || defined(__CUDA7__) || defined(__CUDA8__) || defined(__CUDA9__)
		#include <helper_cuda.h>
		#include <helper_functions.h>
		#include <helper_timer.h>
		//#include <helper_math.h>

		#define CUDA_CREATE_TIMER(x) sdkCreateTimer(&(x))
		#define CUDA_DELETE_TIMER(x) sdkDeleteTimer(&(x))
		#define CUDA_RESET_TIMER(x) sdkResetTimer(&(x))
		#define CUDA_START_TIMER(x) sdkStartTimer(&(x))
		#define CUDA_STOP_TIMER(x) sdkStopTimer(&(x))
		#define CUDA_GET_TIMER_VALUE(x) sdkGetTimerValue(&(x))

		#define CUDA_CHECK_ERRORS(x) checkCudaErrors(x)
		#define CUDA_GET_LAST_ERROR(x) getLastCudaError(x)

		#define CUDA_GET_MAXGFLOP_DEVICE_ID gpuGetMaxGflopsDeviceId
		#define CUDA_DEVICE_RESET cudaDeviceReset
	#endif
#endif

#endif /* _CUDA_VERSION_CONTROL_H_ */
