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
 * Ver 6/13/2016
 */
#include <cassert> // assert
#include <cstdio>  // printf
#include <cstring> // string, memset
#include <cstdlib> // malloc, free, rand

#include <poisson_rate.h>
#include <carlsim_definitions.h> // ALL
#ifndef __NO_CUDA__
	#include <cuda_version_control.h>
#endif



class PoissonRate::Impl {
public:
	// +++++ PUBLIC METHODS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	Impl(int nNeur, bool onGPU): nNeur_(nNeur), onGPU_(onGPU) {
		assert(nNeur>0);

		h_rates_ = NULL;
		d_rates_ = NULL;

		if (onGPU) {
#ifndef __NO_CUDA__
			// allocate rates on device and set to zero
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_rates_, sizeof(float)*nNeur));
			CUDA_CHECK_ERRORS(cudaMemset(d_rates_, 0, sizeof(float)*nNeur));
#else
			printf("Cannot use onGPU when compiled without CUDA library.\n");
			assert(false);
#endif
		} else {
			// allocate rates on host and set to zero
			h_rates_ = new float[nNeur];
			memset(h_rates_, 0, sizeof(float)*nNeur);
		}
	}

	~Impl() {
		if (isOnGPU()) {
#ifndef __NO_CUDA__
			// clean up device
			if (d_rates_!=NULL) {
				CUDA_CHECK_ERRORS(cudaFree(d_rates_)); // free memory
				d_rates_ = NULL;
			}
#endif
		} else {
			// clean up host
			if (h_rates_!=NULL) {
				delete[] h_rates_;
			}
			h_rates_ = NULL;
		}
	}

	int getNumNeurons() {
		return nNeur_;
	}

	// get rate of certain neuron
	float getRate(int neurId) {
		assert(neurId>=0 && neurId<getNumNeurons());

		if (isOnGPU()) {
#ifndef __NO_CUDA__
			// get data from device (might have kernel launch overhead because float is small)
			float h_d_rate = 0.0f;
			CUDA_CHECK_ERRORS( cudaMemcpy(&h_d_rate, &(d_rates_[neurId]), sizeof(float), cudaMemcpyDeviceToHost) );
			return h_d_rate;
#endif
		} else {
			// data is on host
			return h_rates_[neurId];
		}
	}

	// get all rates as vector
	std::vector<float> getRates() {
		if (isOnGPU()) {
#ifndef __NO_CUDA__
			// get data from device
			float *h_d_rates = (float*)malloc(sizeof(float)*getNumNeurons());
			CUDA_CHECK_ERRORS( cudaMemcpy(h_d_rates, d_rates_, sizeof(float)*getNumNeurons(), cudaMemcpyDeviceToHost) );

			// copy data to vec
			std::vector<float> rates(h_d_rates, h_d_rates + getNumNeurons());
			free(h_d_rates);

			return rates;
#endif
		} else {
			// data is on host
			std::vector<float> rates(h_rates_, h_rates_ + getNumNeurons()); // copy to vec
			return rates;
		}
	}

	// get pointer to rate array on CPU
	float* getRatePtrCPU() {
		assert(!isOnGPU());
		return h_rates_;
	}

	// get pointer to rate array on GPU
	float* getRatePtrGPU() {
		assert(isOnGPU());
		return d_rates_;
	}

	bool isOnGPU() {
		return onGPU_;
	}

	// set rate of a specific neuron
	void setRate(int neurId, float rate) {
		if (neurId==ALL) {
			setRates(rate);
		} else {
			assert(neurId>=0 && neurId<getNumNeurons());
			if (isOnGPU()) {
#ifndef __NO_CUDA__
				// copy float to device (might have kernel launch overhead because float is small)
				CUDA_CHECK_ERRORS( cudaMemcpy(&(d_rates_[neurId]), &rate, sizeof(float), cudaMemcpyHostToDevice) );
#endif
			} else {
				// set float in host array
				h_rates_[neurId] = rate;
			}
		}
	}

	// set rate of all neurons to same value
	void setRates(float rate) {
		assert(rate>=0.0f);

		std::vector<float> rates(getNumNeurons(), rate);
		setRates(rates);
	}

	// set rates with vector
	void setRates(const std::vector<float>& rate) {
		assert(rate.size()==getNumNeurons());

		if (isOnGPU()) {
#ifndef __NO_CUDA__
			// copy to device
			float *h_rates_arr = new float[getNumNeurons()];
			std::copy(rate.begin(), rate.end(), h_rates_arr);
			CUDA_CHECK_ERRORS( cudaMemcpy(d_rates_, h_rates_arr, sizeof(float)*getNumNeurons(), cudaMemcpyHostToDevice) );
			delete[] h_rates_arr;
#endif
		} else {
			// set host array
			std::copy(rate.begin(), rate.end(), h_rates_);
		}
	}


private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// +++++ PRIVATE STATIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	float *h_rates_;	//!< pointer to host allocation of underlying firing rate array
	float *d_rates_;	//!< pointer to device allocation of underlying firing rate array
	const int nNeur_;	//!< number of neurons to manage
	const bool onGPU_;	//!< whether allocated on GPU (true) or CPU (false)
};


// ****************************************************************************************************************** //
// POISSONRATE API IMPLEMENTATION
// ****************************************************************************************************************** //

// create and destroy a pImpl instance
PoissonRate::PoissonRate(int nNeur, bool onGPU) : _impl( new Impl(nNeur, onGPU) ) {}
PoissonRate::~PoissonRate() { delete _impl; }

int PoissonRate::getNumNeurons() { return _impl->getNumNeurons(); }
float PoissonRate::getRate(int neurId) { return _impl->getRate(neurId); }
std::vector<float> PoissonRate::getRates() { return _impl->getRates(); }
float* PoissonRate::getRatePtrCPU() { return _impl->getRatePtrCPU(); }
float* PoissonRate::getRatePtrGPU() { return _impl->getRatePtrGPU(); }
bool PoissonRate::isOnGPU() { return _impl->isOnGPU(); }
void PoissonRate::setRate(int neurId, float rate) { _impl->setRate(neurId, rate); }
void PoissonRate::setRates(float rate) { _impl->setRates(rate); }
void PoissonRate::setRates(const std::vector<float>& rates) { _impl->setRates(rates); }
