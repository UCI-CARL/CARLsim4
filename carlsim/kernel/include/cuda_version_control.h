/*
 * cuda_version_control.h
 *
 * This file keeps track of CUDA calls from different CUDA toolkit versions.
 *
 *  Created on: Nov 3, 2013
 *      Author: tingshuc
 *  Modfied on: Jan 5, 2015
 *      Author: MB
 */

#ifndef _CUDA_VERSION_CONTROL_H_
#define _CUDA_VERSION_CONTROL_H_

#include <cuda.h>
#include <cuda_runtime.h>

// we no longer support CUDA3 and CUDA4, but keep cuda_version_control.h for
// handling future CUDA toolkit API differences
#if defined(__CUDA5__) || defined(__CUDA6__) || defined(__CUDA7__)
	#include <helper_cuda.h>
	#include <helper_functions.h>
	#include <helper_timer.h>
	#include <helper_math.h>

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

#endif /* _CUDA_VERSION_CONTROL_H_ */