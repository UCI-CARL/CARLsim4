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
#ifndef _SPIKEGEN_FROM_FILE_H_
#define _SPIKEGEN_FROM_FILE_H_

#include <callback.h>
#include <string>

/*!
 * \brief a SpikeGenerator that schedules spikes from a spike file binary
 *
 * This class implements a SpikeGenerator that schedules spikes from a spike file binary that has been created
 * with a SpikeMonitor.
 */
class SpikeGeneratorFromFile : public SpikeGenerator {
public:
	/*!
	 * \brief SpikeGeneratorFromFile constructor
	 * \param[in] fileName file name of spike file (must be createde from SpikeMonitor)
	 */
	SpikeGeneratorFromFile(std::string fileName);

	//! PeriodicSpikeGenerator destructor
	~SpikeGeneratorFromFile();

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
	unsigned int nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
		unsigned int lastScheduledSpikeTime);

private:
	void openFile();

	bool needToInit_;
	std::string fileName_;		//!< file name
	FILE* fpBegin_;				//!< pointer to beginning of file
	long int* fpOffsetNeur_;	//!< file pointer array to store last read for each neuron
};

#endif