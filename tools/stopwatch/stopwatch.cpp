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
#include "stopwatch.h"

#include <stdio.h>
#include <algorithm>		// std::find
#include <assert.h>			// assert

#include <carlsim_log_definitions.h>	// CARLSIM_ERROR, CARLSIM_WARN, CARLSIM_INFO


// ****************************************************************************************************************** //
// STOPWATCH UTILITY PRIVATE IMPLEMENTATION
// ****************************************************************************************************************** //

/*!
 * \brief Private implementation of the Stopwatch Utility
 *
 * This class provides a timer with milliseconds resolution.
 * \see http://stackoverflow.com/questions/1861294/how-to-calculate-execution-time-of-a-code-snippet-in-c/1861337#1861337
 * \since v3.1
 */
class Stopwatch::Impl {
public:
	// +++++ PUBLIC METHODS: CONSTRUCTOR / DESTRUCTOR +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	Impl(bool startTimer) {
		_isTimerOn = false;
		reset();
		if (startTimer) {
			start("start");
		}
	}

	~Impl() {
		// nothing to see here
	}

	// resets the timer
	void reset() {
		if (_isTimerOn) {
			CARLSIM_WARN("Stopwatch::reset", "Cannot reset timer when timer is on.");
			return;
		}

		_isTimerOn = false;
		_startTimeMs.clear();
		_stopTimeMs.clear();
		_lapTimeMs.clear();
		_accumTimeMs = 0;
		_tags.clear();
	}

	// starts/continues the timer
	void start(const std::string& tag) {
		if (_isTimerOn) {
			CARLSIM_WARN("Stopwatch::start", "Cannot start timer when timer is already on.");
			return;
		}

		// start/continue timer
		_isTimerOn = true;
		_tags.push_back(tag);
		_startTimeMs.push_back( getCurrentTime() );
	}

	// stops the timer
	uint64_t stop(bool printMessage=true, FILE* fileStream=NULL) {
		if (!_isTimerOn) {
			CARLSIM_WARN("Stopwatch::stop", "Cannot stop timer when timer is already off.");
			return _accumTimeMs;
		}

		// pause/stop timer and update exe time
		_isTimerOn = false;
		_stopTimeMs.push_back( getCurrentTime() );
		uint64_t lapMs = _stopTimeMs.back() - _startTimeMs.back();
		_lapTimeMs.push_back(lapMs);

		// keep track of accumulated record time
		// check for arithmetic overflow
		assert(_accumTimeMs + lapMs >= _accumTimeMs);
		_accumTimeMs += lapMs;

		if (printMessage) {
			print(fileStream);
		}

		return _accumTimeMs;
	}

	// prints a summary to a file
	void print(FILE* fileStream=NULL) const {
		if (_isTimerOn) {
			CARLSIM_WARN("Stopwatch::print", "Cannot print when timer is on.");
			return;
		}

		if (fileStream == NULL) {
			fileStream = stdout; // default
		}

		fprintf(fileStream, "\n--------------------------------------------------------------------------------\n");
		fprintf(fileStream, "| Stopwatch                                                                    |\n");
		fprintf(fileStream, "|------------------------------------------------------------------------------|\n");
		fprintf(fileStream, "|                  Tag         Start          Stop           Lap         Total |\n");

		uint64_t totalMsSoFar = 0;
		for (unsigned int i=0; i<_lapTimeMs.size(); i++) {
			totalMsSoFar += _lapTimeMs[i];
			uint64_t startMs = _startTimeMs[i] - _startTimeMs[0];
			uint64_t stopMs = _stopTimeMs[i] - _startTimeMs[0];
			uint64_t lapMs = _lapTimeMs[i];

			fprintf(fileStream, "| %20.20s  %02lu:%02lu:%02lu.%03lu  %02lu:%02lu:%02lu.%03lu "
				" %02lu:%02lu:%02lu.%03lu  %02lu:%02lu:%02lu.%03lu |\n",
				_tags[i].c_str(),
				startMs/3600000, (startMs/1000/60)%60, (startMs/1000)%60, startMs%1000,
				stopMs/3600000, (stopMs/1000/60)%60, (stopMs/1000)%60, stopMs%1000,
				lapMs/3600000, (lapMs/1000/60)%60, (lapMs/1000)%60, lapMs%1000,
				totalMsSoFar/3600000, (totalMsSoFar/1000/60)%60, (totalMsSoFar/1000)%60, totalMsSoFar%1000);
		}
		fprintf(fileStream, "--------------------------------------------------------------------------------\n");
	}

	// lap is equivalent to stop-start
	// returns current lap time
	uint64_t lap(const std::string& tag) {
		if (!_isTimerOn) {
			CARLSIM_WARN("Stopwatch::lap", "Cannot use lap when timer is off.");
			return 0;
		}

		stop(false);
		start(tag);
		return _lapTimeMs.back();
	}

	// returns lap time, look-up by tag
	uint64_t getLapTime(const std::string& tag) const {
		unsigned int pos = std::find(_tags.begin(), _tags.end(), tag) - _tags.begin();
		if (pos >=0 && pos < _tags.size()) {
			if (pos >= _lapTimeMs.size()) {
				CARLSIM_WARN("Stopwatch::getLapTime(tag)", "Cannot look up current lap time until timer stopped.");
				return 0;
			}
			printf("pos = %u, time = %lu\n",pos,_lapTimeMs[pos]);
			return _lapTimeMs[pos];
		} else {
			CARLSIM_WARN("Stopwatch::getLapTime(tag)", "Invalid tag specified.");
			return 0;
		}
	}

	// returns lap time, look-up by index
	uint64_t getLapTime(unsigned int index) const {
		if (index < _lapTimeMs.size()) {
			return _lapTimeMs[index];
		} else {
			CARLSIM_WARN("Stopwatch::getLapTime(index)", "Invalid index specified.");
			return 0;
		}
	}

private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	uint64_t getCurrentTime() const {
	#if defined(WIN32) || defined(WIN64)
		// Windows
		FILETIME ft;
		LARGE_INTEGER li;

		// Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
		// to a LARGE_INTEGER structure.
		GetSystemTimeAsFileTime(&ft);
		li.LowPart = ft.dwLowDateTime;
		li.HighPart = ft.dwHighDateTime;

		uint64_t ret = li.QuadPart;
		ret -= 116444736000000000LL; // Convert from file time to UNIX epoch time.
		ret /= 10000; // From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals

		return ret;
	#else
		// Unix
		struct timeval tv;

		gettimeofday(&tv, NULL);

		uint64_t ret = tv.tv_usec;
		// Convert from micro seconds (10^-6) to milliseconds (10^-3)
		ret /= 1000;

		// Adds the seconds (10^0) after converting them to milliseconds (10^-3)
		ret += (tv.tv_sec * 1000);

		return ret;
	#endif
	}

	// +++++ PRIVATE STATIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
	std::vector<uint64_t> _startTimeMs;		// vector of start times (Unix/Windows timestamps)
	std::vector<uint64_t> _stopTimeMs;		// vector of end times (Unix/Windows timestamps)
	std::vector<uint64_t> _lapTimeMs;		// vector lap times (stop-start for every lap)
	uint64_t _accumTimeMs;					// acumulated record time (sum of all laps)
	std::vector<std::string> _tags;			// string tag for every lap
	bool _isTimerOn;						// flag to indicate whether stopwatch is running (true) or not (false)

};


// ****************************************************************************************************************** //
// STOPWATCH API IMPLEMENTATION
// ****************************************************************************************************************** //

// create and destroy a pImpl instance
Stopwatch::Stopwatch(bool startTimer) : _impl( new Impl(startTimer) ) {}
Stopwatch::~Stopwatch() { delete _impl; }

void Stopwatch::start(const std::string& tag) { _impl->start(tag); }
uint64_t Stopwatch::stop(bool printMessage, FILE* printFile) { return _impl->stop(printMessage, printFile); }
uint64_t Stopwatch::lap(const std::string& tag) { return _impl->lap(tag); }
void Stopwatch::reset() { _impl->reset(); }

uint64_t Stopwatch::getLapTime(const std::string& tag) const { return _impl->getLapTime(tag); }
uint64_t Stopwatch::getLapTime(int index) const { return _impl->getLapTime(index); }

void Stopwatch::print(FILE* fileStream) const { _impl->print(fileStream); }