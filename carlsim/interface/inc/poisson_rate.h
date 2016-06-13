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
 * Ver 11/26/2014
 */

#ifndef _POISSON_RATE_H_
#define _POISSON_RATE_H_

#include <vector>

/*!
 * \brief Class for generating Poisson spike trains
 *
 * The PoissonRate class allows a user create spike trains whose inter-spike interval follows a Poisson process. The
 * object can then be linked to a spike generator group (created via CARLsim::createSpikeGeneratorGroup) by calling
 * CARLsim::setSpikeRate.
 *
 * All firing rates will be initialized to zero. The user then has a number of options to manipulate the mean firing
 * rate of each neuron. The same rate can be applied to all neurons by calling PoissonRate::setRates(float rate).
 * Individual rates can be applied from a vector by calling PoissonRate::setRates(const std::vector<float>& rates).
 * The rate of a single neuron can be manipulated by calling PoissonRate::setRate(int neurId, float rate).
 *
 * Example usage:
 * \code
 * // create a PoissonRate object on GPU for a group of 50 neurons
 * PoissonRate myRate(50, true);
 *
 * // let all rates be zero except the one for neurId=42, set that to 20 Hz
 * myRate.setRate(42, 20.0f);
 *
 * // apply to spike generator group (say, g0) of CARLsim object in SETUP or EXECUTION state
 * // and run the network for a second
 * sim.setSpikeRate(g0, &myRate);
 * sim.runNetwork(1,0);
 *
 * // now change the rates of all neurons to 12 Hz
 * myRate.setRates(12.0f);
 * sim.setSpikeRate(g0, &myRate);
 * sim.runNetwork(1,0);
 * \endcode
 * 
 * \attention The mean firing rate will keep getting applied to any instances of CARLsim::runNetwork until the user
 * changes the values and calls CARLsim::setSpikeRate again.
 * \note A PoissonRate object can be allocated either on the CPU or the GPU. However, GPU allocation is only supported
 * if the CARLsim simulation is run in GPU_MODE.
 * \since v3.0
 */
class PoissonRate {
public:
	/*!
	 * \brief PoissonRate constructor
	 *
	 * Creates a new instance of class PoissonRate.
	 * \param[in] nNeur the number of neurons for which to generate Poisson spike trains
	 * \param[in] onGPU whether to allocate the rate vector on GPU (true) or CPU (false)
	 * \since v2.0
	 */
	PoissonRate(int nNeur, bool onGPU=false);

	/*!
	 * \brief PoissonRate destructor
	 *
	 * Cleans up all the memory upon object deletion.
	 * \since v2.0
	 */
	~PoissonRate();

	/*!
	 * \brief Returns the number of neurons for which to generate Poisson spike trains
	 *
	 * This function returns the number of neurons for which to generate Poisson spike trains. This number is defined
	 * at initialization and cannot be changed during the object lifetime.
	 * \returns number of neurons
	 * \since v3.0
	 */
	int getNumNeurons();

	/*!
	 * \brief Returns the mean firing rate of a specific neuron ID
	 *
	 * This function returns the mean firing rate assigned to a specific neuron ID. The neuron ID is 0-indexed and
	 * should thus be in the range [ 0 , getNumNeurons() ). It is completely independent from CARLsim neuron IDs.
	 * \param[in] neurId the neuron ID (0-indexed)
	 * \returns mean firing rate
	 * \since v3.0
	 */
	float getRate(int neurId);

	/*!
	 * \brief Returns a vector of firing rates, one element per neuron
	 *
	 * This function returns all the mean firing rates in a vector, one vector element per neuron.
	 * \returns vector of firing rates
	 * \since v3.0
	 */
	std::vector<float> getRates();

	/*!
	 * \brief Returns pointer to CPU-allocated firing rate array (deprecated)
	 *
	 * This function returns a pointer to the underlying firing rate array if allocated on the CPU. This pointer does
	 * not exist when the PoissonRate object is allocated on GPU.
	 *
	 * \deprecated This function is deprecated, as it should not be exposed to the high-level UI API. Use
	 * PoissonRate::getRates instead.
	 */
	float* getRatePtrCPU();

	/*!
	 * \brief Returns pointer to GPU-allocated firing rate array (deprecated)
	 *
	 * This function returns a pointer to the underlying firing rate array if allocated on the GPU. This pointer does
	 * not exist when the PoissonRate object is allocated on CPU. 
	 *
	 * \deprecated This function is deprecated, as it should not be exposed to the high-level UI API. Use
	 * PoissonRate::getRates instead.
	 */
	float* getRatePtrGPU();

	/*!
	 * \brief Checks whether the firing rates are allocated on CPU or GPU
	 *
	 * This function checks whether the firing rates are allocated either on CPU or GPU.
	 * \returns a flag whether allocated on GPU (true) or CPU (false)
	 * \since v3.0
	 */
	bool isOnGPU();

	/*!
	 * \brief Sets the mean firing rate of a particular neuron ID
	 *
	 * This function sets the firing rate of a particular neuron ID. The neuron ID is 0-indexed and should thus be in
	 * the range [ 0 , getNumNeurons() ). It is completely independent from CARLsim neuron IDs.
	 * \param[in] neurId the neuron ID (0-indexed)
	 * \param[in] rate the firing rate to set
	 * \since v3.0
	 */
	void setRate(int neurId, float rate);

	/*!
	 * \brief Assigns the same mean firing rate to all neurons
	 *
	 * This function assigns the same firing rate to all the neurons.
	 *
	 * \param[in] rate the firing rate to set
	 * \since v3.0
	 */
	void setRates(float rate);

	/*!
	 * \brief Sets the mean firing rate of each neuron from a vector
	 *
	 * This function sets the firing rate of each neuron from a vector of firing rates, one rate per neuron.
	 *
	 * \param[in] rates vector of firing rates (size should be equivalent to PoissonRate::getNumNeurons())
	 * \since v3.0
	 */
	void setRates(const std::vector<float>& rates);


private:
	// This class provides a pImpl for the CARLsim User API.
	// \see https://marcmutz.wordpress.com/translated-articles/pimp-my-pimpl/
	class Impl;
	Impl* _impl;
};

#endif