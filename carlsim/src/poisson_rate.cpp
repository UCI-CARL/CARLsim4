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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */

#include <poisson_rate.h>

#if __CUDA3__
    #include <cuda.h>
    #include <cutil_inline.h>
    #include <cutil_math.h>
#else
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "helper_cuda.h"
#endif

#include <cuda_version_control.h>


PoissonRate::PoissonRate(float* _rates, uint32_t _len, bool _onGPU) {
	rates = _rates;
	len = _len;
	onGPU = _onGPU;
	allocatedRatesInternally = false;
};

PoissonRate::PoissonRate(uint32_t _len, bool _onGPU) {
	if (_onGPU) {
	    CUDA_CHECK_ERRORS(cudaMalloc((void**)&rates, _len * sizeof(float)));
	} else {
		rates = new float[_len];
	}
	len = _len;
	onGPU = _onGPU;
	allocatedRatesInternally = true;
};

// destructor
PoissonRate::~PoissonRate() {
	if (allocatedRatesInternally) {
		if (onGPU) {
			CUDA_CHECK_ERRORS(cudaThreadSynchronize()); // wait for kernel to complete
			CUDA_CHECK_ERRORS(cudaFree(rates)); // free memory
		}
		else {
			delete[] rates;
		}
	}
}
