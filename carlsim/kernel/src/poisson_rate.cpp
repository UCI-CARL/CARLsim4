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

#include <assert.h>					// assert
#include <carlsim_definitions.h> 	// ALL

#include <cuda_version_control.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// constructor
PoissonRate::PoissonRate(int nNeur, bool onGPU, int refPeriod): nNeur_(nNeur), onGPU_(onGPU), refPeriod_(refPeriod) {
	assert(nNeur>0); assert(refPeriod>=1);

	h_rates_ = NULL;
	d_rates_ = NULL;

	if (onGPU) {
		// allocate rates on device and set to zero
	    CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_rates_, sizeof(float)*nNeur));
	    CUDA_CHECK_ERRORS(cudaMemset(d_rates_, 0, sizeof(float)*nNeur));
	} else {
		// allocate rates on host and set to zero
		h_rates_ = new float[nNeur];
		memset(h_rates_, 0, sizeof(float)*nNeur);
	}
}

// destructor
PoissonRate::~PoissonRate() {
	if (isOnGPU()) {
		// clean up device
		if (d_rates_!=NULL) {
			CUDA_CHECK_ERRORS(cudaThreadSynchronize()); // wait for kernel to complete
			CUDA_CHECK_ERRORS(cudaFree(d_rates_)); // free memory
			d_rates_ = NULL;
		}
	} else {
		// clean up host
		if (h_rates_!=NULL)
			delete[] h_rates_;
		h_rates_ = NULL;
	}
}

// get rate of certain neuron
float PoissonRate::getRate(int neurId) {
	assert(neurId>=0 && neurId<getNumNeurons());

	if (isOnGPU()) {
		// get data from device (might have kernel launch overhead because float is small)
		float h_d_rate = 0.0f;
//		float *h_d_rate = (float*)malloc(sizeof(float));
		CUDA_CHECK_ERRORS( cudaMemcpy(&h_d_rate, &(d_rates_[neurId]), sizeof(float), cudaMemcpyDeviceToHost) );
		return h_d_rate;
	} else {
		// data is on host
		return h_rates_[neurId];
	}
}

// get all rates as vector
std::vector<float> PoissonRate::getRates() {
	if (isOnGPU()) {
		// get data from device
		float *h_d_rates = (float*)malloc(sizeof(float)*getNumNeurons());
		CUDA_CHECK_ERRORS( cudaMemcpy(h_d_rates, d_rates_, sizeof(float)*getNumNeurons(), cudaMemcpyHostToDevice) );
		std::vector<float> rates(h_d_rates, h_d_rates + sizeof(h_d_rates)/sizeof(h_d_rates[0])); // copy to vec
		return rates;
	} else {
		// data is on host
		std::vector<float> rates(h_rates_, h_rates_ + sizeof(h_rates_)/sizeof(h_rates_[0])); // copy to vec
		return rates;
	}
}

// get pointer to rate array on CPU
float* PoissonRate::getRatePtrCPU() {
	assert(!isOnGPU());
	return h_rates_;
}

// get pointer to rate array on GPU
float* PoissonRate::getRatePtrGPU() {
	assert(isOnGPU());
	return d_rates_;
}

// set rate of a specific neuron
void PoissonRate::setRate(int neurId, float rate) {
	if (neurId==ALL) {
		setRates(rate);
	} else {
		assert(neurId>=0 && neurId<getNumNeurons());
		if (isOnGPU()) {
			// copy float to device (might have kernel launch overhead because float is small)
			CUDA_CHECK_ERRORS( cudaMemcpy(&(d_rates_[neurId]), &rate, sizeof(float), cudaMemcpyHostToDevice) );
		} else {
			// set float in host array
			h_rates_[neurId] = rate;
		}
	}
}

// set rate of all neurons to same value
void PoissonRate::setRates(float rate) {
	assert(rate>=0.0f);

	std::vector<float> rates(getNumNeurons(), rate);
	setRates(rates);
}

// set rates with vector
void PoissonRate::setRates(const std::vector<float>& rate) {
	assert(rate.size()==getNumNeurons());

	if (isOnGPU()) {
		// copy to device
		float *h_rates_arr = new float[getNumNeurons()];
		std::copy(rate.begin(), rate.end(), h_rates_arr);
		CUDA_CHECK_ERRORS( cudaMemcpy(d_rates_, h_rates_arr, sizeof(float)*getNumNeurons(), cudaMemcpyHostToDevice) );
		delete[] h_rates_arr;
	} else {
		// set host array
		std::copy(rate.begin(), rate.end(), h_rates_);
	}
}