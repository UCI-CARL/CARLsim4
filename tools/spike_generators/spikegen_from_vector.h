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
#ifndef _SPIKEGEN_FROM_VECTOR_H_
#define _SPIKEGEN_FROM_VECTOR_H_

#include <callback.h>
#include <vector>

class CARLsim; // forward-declaration

/*!
 * \brief a SpikeGenerator that schedules spikes from a vector of spike times
 *
 * This class implements a SpikeGenerator that schedules spikes from a vector of spike times (ms).
 * \TODO change to vector of AER or 2D vector, so that each neuron can have their own spike times.
 */
class SpikeGeneratorFromVector : public SpikeGenerator {
public:
	/*!
	 * \brief SpikeGeneratorFromVector constructor
	 * \param[in] spkTimes a vector of all spike times
	 */
	SpikeGeneratorFromVector(std::vector<int> spkTimes);

	//! PeriodicSpikeGenerator destructor
	~SpikeGeneratorFromVector() {}

	/*!
	 * \brief schedules the next spike time
	 *
	 * This function schedules the next spike time, given the currentTime and the lastScheduledSpikeTime. It implements
	 * the virtual function of the base class.
	 * \param[in] sim pointer to a CARLsim object
	 * \param[in] grpId current group ID for which to schedule spikes
	 * \param[in] nid current neuron ID for which to schedule spikes
	 * \param[in] currentTime current time (ms) at which spike scheduler is called
	 * \param[in] lastScheduledSpikeTime the last time (ms) at which a spike was scheduled for this nid, grpId
	 * \returns the next spike time (ms)
	 */
	int nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice);

private:
	void checkSpikeVector();
	
	std::vector<int> spkTimes_;	//!< vector of spike times
	int currentIndex_;			//!< keeps track of current position in the vector
	int size_;					//!< size of vector for fast access
};

#endif