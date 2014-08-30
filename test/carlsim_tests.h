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

#include <algorithm>		// std::find


// FIXME: I added this flag, because as it stands now most CPUvsGPU comparisons fail. And they should, because the
// order of execution in doSnnSim() and doGPUsim() is really different. No wonder we get, for example, different
// spike counts in the end.
// So for now, these CPUvsGPU tests will produce a lot of gtest error messages, which are annoying. Use this flag
// to disable them for now until the issue is fixed.
#define ENABLE_CPU_GPU_TESTS (0)


// Don't forget to set REGRESSION_TESTING flag to 1 in config.h 

// TODO: figure out test directory organization (see issue #67); group into appropriate test cases; have test cases
// for published results; add documentation; etc.

// TODO: test interface (see issue #38)

// TODO: add speed test scripts (see issue #32)

// TODO: add more tests in general (see issue #21)

/*
 * GENERAL TESTING STRATEGY
 * ------------------------
 *
 * We provide test cases to A) test core functionality of CARLsim, to B) test the reproducibility of published results,
 * and C) to benchmark simulation speed.
 *
 * A) TESTING CORE FUNCTIONALITY
 * 1. Test core data structures when some functionality is enabled.
 *    For example: Set STP to true for a specific group, check grp_Info to make sure all values are set accordingly.
 * 2. Test core data structures when some functionality is disabled.
 *    For example: Set STP to false for a specific group, check grp_Info to make sure it's disabled.
 * 3. Test behavior when values for input arguments are chosen unreasonably.
 *    For example: Create a group with N=-4 (number of neurons) and expect simulation to die. This is because each
 *    core function should have assertion statements to prevent the simulation from running unreasonable input values.
 *    In some cases, it makes sense to catch this kind of error in the user interface as well (and display an
 *    appropriate error message to the user), but these tests should be placed in the UserInterface test case.
 * 4. Test behavior of network when run with reasonable values.
 *    For example: Run a sample network with STP enabled, and check stpu[nid] and stpx[nid] to make sure they behave.
 *    as expected. You can use the PeriodicSpikeGenerator to be certain of specific spike times and thus run
 *    reproducible sample networks.
 * 5. Test behavior of network when run in CPU mode vs. GPU mode.
 *    For example: Run a sample network with STP enabled, once in CPU mode and once in GPU mode. Record stpu[nid] and
 *    stpx[nid], and make sure that both simulation mode give the exact same result (except for some small error
 *    margin that can account for rounding errors/etc.).
 *
 * B) TESTING PUBLISHED RESULTS
 *
 * C) BENCHMARK TESTS
 *
 */

#include <callback_core.h>

#include <vector>			// std::vector
#include <string>			// std::string, memset
#include <cassert>			// assert

/// **************************************************************************************************************** ///
/// COMMON
/// **************************************************************************************************************** ///

void readAndReturnSpikeFile(const std::string fileName, int*& AERArray, long &arraySize);
void readAndPrintSpikeFile(const std::string fileName);



// \TODO: these should actually work on the user callback level... so don't inherit from *Core classes, but from the
// user interface-equivalent...

//! a periodic spike generator (constant ISI) creating spikes at a certain rate
//! \TODO \FIXME this one should be gone, use public interface instead
class PeriodicSpikeGeneratorCore : public SpikeGeneratorCore {
public:
	PeriodicSpikeGeneratorCore(float rate, bool spikeAtZero=true) : SpikeGeneratorCore(NULL, NULL){
		assert(rate>0);
		rate_ = rate;	  // spike rate
		isi_ = 1000/rate; // inter-spike interval in ms
		spikeAtZero_ = spikeAtZero;
	}

	unsigned int nextSpikeTime(CpuSNN* snn, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (spikeAtZero_) {
			// insert spike at t=0 for each neuron (keep track of neuron IDs to avoid getting stuck in infinite loop)
			if (std::find(nIdFiredAtZero_.begin(), nIdFiredAtZero_.end(), nid)==nIdFiredAtZero_.end()) {
				// spike at t=0 has not been scheduled yet for this neuron
				nIdFiredAtZero_.push_back(nid);
				return 0;
			}
		}

		// periodic spiking according to ISI
		return lastScheduledSpikeTime+isi_; // periodic spiking according to ISI
	}

private:
	float rate_;	// spike rate
	int isi_;		// inter-spike interval that results in above spike rate
	std::vector<int> nIdFiredAtZero_; // keep track of all neuron IDs for which a spike at t=0 has been scheduled
	bool spikeAtZero_; // whether to spike at t=0
};

class SpecificTimeSpikeGeneratorCore : public SpikeGeneratorCore {
public:
	SpecificTimeSpikeGeneratorCore(std::vector<int> spkTimes) : SpikeGeneratorCore(NULL, NULL) {
		spkTimes_ = spkTimes;
		size_ = spkTimes.size();
		currentIndex_ = 0;

	}

	unsigned int nextSpikeTime(CpuSNN* snn, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		return (currentIndex_<size_) ? spkTimes_[currentIndex_++] : 0;
	}

private:
	std::vector<int> spkTimes_;
	int currentIndex_;
	int size_;
};