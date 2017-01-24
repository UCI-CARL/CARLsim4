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
#ifndef _SPIKEGEN_FROM_FILE_H_
#define _SPIKEGEN_FROM_FILE_H_

#include <callback.h>
#include <string>
#include <vector>


class CARLsim;

/*!
 * \brief a SpikeGeneratorFromFile schedules spikes from a spike file binary
 *
 * This class implements a SpikeGenerator that schedules spikes exactly as specified by a spike file binary. The
 * spike file must have been created with a SpikeMonitor.
 *
 * The easiest used-case is wanting to re-run a simulation with the exact same spike trains.
 * For example, if a spike file contains two AER events (in the format <neurId,spikeTime>): <2,123> and <10,12399>,
 * then SpikeGeneratorFromFile will deliver a spike to neuron ID 2 exactly at simulation time 123ms and a spike to
 * neuron ID 10 at simulation time 12399.
 * Running the simulation for longer than that will have no effect, since there are no spikes left to schedule
 * after that.
 *
 * If in this simulation a SpikeMonitor is set on the same group, then the newly generated spike file should be
 * identical to the one used as an input to SpikeGeneratorFromFile.
 *
 * It is possible to off-set the spike times by adding a constant offsetTimeMs to every scheduled spike time.
 * Thus the AER events <2,123> and <10,12399> will be converted to <2,123+offsetTimeMs> and <10,12399+offsetTimeMs>.
 * This offset can be passed to both the constructor and SpikeGeneratorFromFile::rewind, and it can assume both
 * positive or negative. Default value is zero.
 *
 * It is also possible to repeatedly parse the spike file, adding different offsetTimeMs offsets per loop.
 * This can be achieved by passing an optional argument to SpikeGeneratorFromFile::rewind.
 *
 * Upon initialization, the class parses and buffers all spikes from the spike file in a format that allows for
 * more efficient scheduling. Note that this might take up a lot of memory if you have a large and highly active
 * neuron group.
 *
 * Usage example:
 * \code
 * // configure a CARLsim network
 * CARLsim sim("LoadFromFile", CPU_MODE, USER);
 * int gIn = sim.createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);
 * 
 * // Initialize a SpikeGeneratorFromFile object from a previously recorded
 * // spike file (make sure that group had 10 neurons, too!)
 * SpikeGeneratorFromFile SGF("results/spk_input.dat");
 *
 * // assign spike generator to group
 * sim.setSpikeGenerator(gIn, &SGF);
 *
 * // continue configuring ...
 * sim.setupNetwork();
 * 
 * // schedule the exact same spikes as in the file
 * // let's assume the first spike time occurs at t=42ms and the last at t=967ms
 * sim.runNetwork(1,0);
 *
 * // SGF can rewind to the beginning of the spike file, but now add offset of 1000ms: this can be done
 * // either by hardcoding the number or by calling CARLsim::getSimTime:
 * SGF.rewind((int)sim.getSimTime());
 *
 * // now spikes will be scheduled again, but the first spike is at t=42+1000ms, and the last at t=967+1000ms
 * sim.runNetwork(1,0);
 * \endcode
 *
 * \note Make sure the new neuron group has the exact same number of neurons as the group that was used to record
 * the spike file.
 * \attention Upon initializiation, all spikes from the spike file will be buffered as vectors of ints, which might
 * take up a lot of memory if you have a large and highly active neuron group.
 * \since v3.0
 */
class SpikeGeneratorFromFile : public SpikeGenerator {
public:
	/*!
	 * \brief SpikeGeneratorFromFile constructor
	 *
	 * \param[in] fileName file name of spike file (must be created from SpikeMonitor)
	 * \param[in] offsetTimeMs optional offset (ms) that will be applied to all scheduled spike times. Can assume
	 *                         both positive and negative values. Default: 0.
	 */
	SpikeGeneratorFromFile(std::string fileName, int offsetTimeMs=0);

	//! SpikeGeneratorFromFile destructor
	~SpikeGeneratorFromFile();

	/*!
	 * \brief Loads a new spike file
	 *
	 * This function loads a new spike file (must be created from SpikeMonitor).
	 * This allows changing files mid-simulation, which would otherwise not be possible without re-compiling the
	 * network, because CARLsim::setSpikeGenerator can only be called in ::CONFIG_STATE.
	 *
	 * \param[in] fileName file name of spike file (must be created from SpikeMonitor)
	 * \param[in] offsetTimeMs optional offset (ms) that will be applied to all scheduled spike times. Can assume
	 *                         both positive and negative values. Default: 0.
	 * \since v3.1
	 */
	void loadFile(std::string fileName, int offsetTimeMs=0);

	/*!
	 * \brief Rewinds the spike file to beginning of file
	 *
	 * This function rewinds the spike file to the begining and applies an offset to the spike times.
	 * This means that the SpikeGenerator will continue to schedule spikes (starting over at the beginning of
	 * the spike file), but this time it will add offsetTimeMs to all spike times.
	 *
	 * Most often, this offset is the current simulation time (in ms), which can be retrieved via
	 * CARLsim::getSimTime.
	 *
	 * Specifying an offset is necessary, because SpikeGeneratorFromFile has no direct access to CARLsim, and
	 * thus cannot know how much time has already been simulated.
	 *
	 * Consider the following illustrative example:
	 * -# Assume spike file contains only two spikes <neurId,spikeTime>: <2,123> and <6,987>
	 * -# If we run CARLsim for a second, SpikeGeneratorFromFile will schedule exactly these two spikes at times
	 *   t1=123ms and t2=987ms
	 * -# If we run CARLsim for longer, no additional spikes will be scheduled.
	 * -# However, calling SpikeGeneratorFromFile::rewind with offset 1000ms (which is what CARLsim::getSimTime
	 *    will return at that point) will re-schedule all spikes, but now t1=1000+123ms and t2=100+123ms
	 *
	 * \param[in] offsetTimeMs offset (ms) that will be applied to all scheduled spike times. Can assume
	 *                         both positive and negative values.
	 */
	void rewind(int offsetTimeMs);

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
	 * \param[in] endOfTimeSlice the end of the current scheduling time slice (ms). A spike delivered at a time
	 *                           >= endOfTimeSlice will not be scheduled by CARLsim
	 * \returns the next spike time (ms)
	 */
	int nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice);

private:
	void openFile();
	void init();

	std::string fileName_;		//!< file name
	FILE* fpBegin_;				//!< pointer to beginning of file
	int szByteHeader_;          //!< number of bytes in header section
                                //!< \FIXME: there should be a standardized SpikeReader++ utility

	//! A 2D vector of spike times, first dim=neuron ID, second dim=spike times.
	//! This makes it easy to keep track of which spike needs to be scheduled next, by maintaining
	//! a vector of iterators.
	std::vector< std::vector<int> > spikes_;

	//! A vector of iterators to easily keep track of which spike to schedule next (per neuron)
	std::vector< std::vector<int>::iterator > spikesIt_;

	int nNeur_;                 //!< number of neurons in the group
	int offsetTimeMs_;			//!< offset (ms) to add to every scheduled spike time
};

#endif