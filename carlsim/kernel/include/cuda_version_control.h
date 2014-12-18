/*
 * CUDAVersionControl.h
 *
 *  Created on: Nov 3, 2013
 *      Author: tingshuc
 *  Modfied on: Dec, 18, 2014
 *      Author: MB
 */

#ifndef _CUDAVERSIONCONTROL_H_
#define _CUDAVERSIONCONTROL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#if __CUDA3__
	// includes, projec
	#include <cudpp/cudpp.h>
	#include <cutil_inline.h>
	#include <cutil_math.h>

	#define CUDA_CHECK_ERRORS(x) cutilSafeCall(x)
	#define CUDA_CHECK_ERRORS_MACRO(x) CUDA_SAFE_CALL(x)

	#define CUDA_GET_LAST_ERROR(x) cutilCheckMsg(x)
	#define CUDA_GET_LAST_ERROR_MACRO(x) CUT_CHECK_ERROR(x)

	#define CUDA_CREATE_TIMER(x) cutCreateTimer(&(x))
	#define CUDA_DELETE_TIMER(x) cutDeleteTimer(x)
	#define CUDA_RESET_TIMER(x) cutResetTimer(x)
	#define CUDA_START_TIMER(x) cutStartTimer(x)
	#define CUDA_STOP_TIMER(x) cutStopTimer(x)
	#define CUDA_GET_TIMER_VALUE(x) cutGetTimerValue(x)

	#define CUDA_GET_MAXGFLOP_DEVICE_ID cutGetMaxGflopsDeviceId
	#define CUDA_DEVICE_RESET cudaThreadExit
#elif defined(__CUDA5__) || defined(__CUDA6__)
	#include <helper_cuda.h>
	#include <helper_functions.h>
	#include <helper_timer.h>
	#include <helper_math.h>

	// those two are different in CUDA3, but same here
	#define CUDA_CHECK_ERRORS(x) checkCudaErrors(x)
	#define CUDA_CHECK_ERRORS_MACRO(x) checkCudaErrors(x)

	#define CUDA_CREATE_TIMER(x) sdkCreateTimer(&(x))
	#define CUDA_DELETE_TIMER(x) sdkDeleteTimer(&(x))
	#define CUDA_RESET_TIMER(x) sdkResetTimer(&(x))
	#define CUDA_START_TIMER(x) sdkStartTimer(&(x))
	#define CUDA_STOP_TIMER(x) sdkStopTimer(&(x))
	#define CUDA_GET_TIMER_VALUE(x) sdkGetTimerValue(&(x))

	#define CUDA_GET_LAST_ERROR(x) getLastCudaError(x)
	#define CUDA_GET_LAST_ERROR_MACRO(x) getLastCudaError(x)

	#define CUDA_GET_MAXGFLOP_DEVICE_ID gpuGetMaxGflopsDeviceId
	#define CUDA_DEVICE_RESET cudaDeviceReset

#endif

#endif /* _CUDAVERSIONCONTROL_H_ */