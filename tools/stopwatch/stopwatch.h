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
#ifndef STOPWATCH_H
#define STOPWATCH_H

#if defined(WIN32) || defined(WIN64)
	#include <Windows.h>
#else
	#include <sys/time.h>
	#include <ctime>
#endif

#include <stdint.h>
#include <vector>	// std::vector
#include <string>	// std::string

/*!
 * \brief Utility to measure script execution time (wall-clock time) in milliseconds
 *
 * This class provides a means to measure script execution time (wall-clock time) with millisecond precision.
 *
 * The stopwatch can start, stop, and lap.
 * It can be used to time different segments of a simulation; for example, to record the time it took to build
 * a network vs. the time it took to run the network. Every segment (or lap) can be annotated with a user-defined
 * string for easier analysis.
 * The method Stopwatch::lap does also return the execution time for the previous lap. Stopwatch::stop also
 * returns the accumulated record time over all laps.
 *
 * Code example:
 * \code
 * int main() {
 *    // create and start timer (controlled by optional flag startTimer=true)
 *	  Stopwatch watch;
 *
 *	  // ---------------- CONFIG STATE -------------------
 *	  CARLsim sim("HelloWorld", CPU_MODE, USER);
 *
 *	  // configure network
 *
 *	  // ---------------- SETUP STATE -------------------
 *	  // build the network
 *	  watch.lap("setupNetwork"); // tag this lap with string "setupNetwork"
 *	  sim.setupNetwork();
 *
 *	  // ---------------- RUN STATE -------------------
 *	  watch.lap("runNetwork"); // tag this lap with string "runNetwork"
 *	  sim.runNetwork(10,0);
 *
 *	  // print stopwatch summary
 *	  watch.stop();
 * }
 * \endcode
 *
 * Stopping the stopwatch will produce nicely formatted console output indicating the starting and stopping
 * time of each lap, as well as the accumulated time.
 * \code
 *	 ------------------------------------------------------------------------
 *	| Stopwatch                                                            |
 *	|----------------------------------------------------------------------|
 *	|          Tag         Start          Stop           Lap         Total |
 *	|        start  00:00:00.000  00:00:00.004  00:00:00.004  00:00:00.004 |
 *	| setupNetwork  00:00:00.004  00:00:00.064  00:00:00.060  00:00:00.064 |
 *	|   runNetwork  00:00:00.064  00:00:06.651  00:00:06.587  00:00:06.651 |
 *	------------------------------------------------------------------------
 * \endcode
 * 
 * \since v3.1
 */
class Stopwatch {
public:
	/*!
	 * \brief Utility to measure script execution time (wall-clock time) in milliseconds
	 *
	 * This class provides a means to measure script execution time (wall-clock time) with millisecond precision.
	 *
	 * By default, calling the constructor will also start the timer. Optionally, <tt>startTimer</tt> can be set to
	 * false. In this case, the first call to Stopwatch::start will start the timer.
	 *
	 * \param startTimer flag to indicate whether to automatically start the timer (true) or not (false)
	 */
	Stopwatch(bool startTimer=true);

	/*!
	 * \brief Stopwatch destructor
	 *
	 * The destructor deallocates all data structures.
	 */
	~Stopwatch();

	/*!
	 * \brief Starts/restarts the timer
	 *
	 * This method starts a new lap. The lap can be given a user-defined string for a tag.
	 *
	 * A timer can (theoretically) handle an arbitrary number of start/stop cycles.
	 *
	 * \param tag user-defined string name for the new lap
	 *
	 * \see Stopwatch::lap
	 * \see Stopwatch::stop
	 * \see Stopwatch::reset
	 * \since v3.1
	 */
	void start(const std::string& tag = "");

	/*!
	 * \brief Stops the timer
	 *
	 * This method ends a lap. By default, the method will also print a summary to a file (which is stdout by
	 * default). Optionally, the print mechanism can be turned off, or the file stream redirected.
	 *
	 * A timer can (theoretically) handle an arbitrary number of start/stop cycles.
	 *
	 * The method will also return the total accumulated time (ms) over all recorded laps.
	 *
	 * \param printMessage flag to indicate whether to print a record summary (true) or not (false)
	 * \param printFile file stream where to print the summary (by default: NULL, redirects to stdout)
	 * \returns total accumulated recorded time (ms)
	 *
	 * \note Opening and closing the filestream must be handled by the user.
	 * \see Stopwatch::start
	 * \see Stopwatch::lap
	 * \see Stopwatch::reset
	 * \since v3.1
	 */
	uint64_t stop(bool printMessage=true, FILE* printFile=NULL);

	/*!
	 * \brief Ends the current lap and starts a new lap
	 *
	 * This method ends the current lap and starts a new lap. The new lap can be given a user-defined
	 * string for a tag.
	 *
	 * This method is equivalent to calling Stopwatch::stop and Stopwatch::start.
	 *
	 * The method will also return the execution time of the just-ended lap.
	 *
	 * \param tag user-defined string name for the new lap
	 * \returns execution time of the just-ended lap
	 *
	 * \note This method can only be called when the timer is on (i.e., after calling Stopwatch::start, but before
	 *       calling Stopwatch::stop).
	 * \see Stopwatch::start
	 * \see Stopwatch::stop
	 * \see Stopwatch::reset
	 * \since v3.1
	 */
	uint64_t lap(const std::string& tag = "");

	/*!
	 * \brief Resets the timer
	 *
	 * This method resets the timer, effectively deleting all previously recorded data.
	 *
	 * \note This method can only be called when the timer is off (i.e., after calling Stopwatch::stop).
	 * \see Stopwatch::start
	 * \see Stopwatch::lap
	 * \see Stopwatch::stop
	 * \since v3.1
	 */
	void reset();

	/*!
	 * \brief Looks up the execution time of a lap by its tag
	 *
	 * This method looks up a certain lap by its specified tag, and if found, returns the lap's recorded
	 * execution time. If the tag cannot be found, a warning is printed and value 0 is returned.
	 *
	 * \param tag user-defined string name of the lap to look up
	 * \returns recorded execution time of certain lap (or 0 if tag invalid)
	 *
	 * \note This method can only be called when the timer is off (i.e., after calling Stopwatch::stop).
	 * \see Stopwatch::getLapTime(int) const
	 * \since v3.1
	 */
	uint64_t getLapTime(const std::string& tag) const;

	/*!
	 * \brief Looks up the execution time of a lap by its vector index
	 *
	 * This method looks up a certain lap by its position in the list of all recorded laps (0-indexed).
	 * For example, the first lap will have index 0, the second will have index 1, and so on.
	 * If the index is valid, the method returns the lap's recorded execution time.
	 * If the index is invalid, a warning is printed and value 0 is returned.
	 *
	 * \param index index of the lap in the list of recorded laps
	 * \returns recorded execution time of certain lap (or 0 if index invalid)
	 *
	 * \note This method can only be called when the timer is off (i.e., after calling Stopwatch::stop).
	 * \see Stopwatch::getLapTime(const std::string&) const
	 * \since v3.1
	 */
	uint64_t getLapTime(int index) const;

	/*!
	 * \brief Prints a summary to file or console
	 *
	 * This method prints a summary to either file or console, depending on optional argument <tt>fileStream</tt>.
	 * By default, <tt>fileStream</tt> is <tt>NULL</tt>, making the stream redirect to stdout.
	 *
	 * Example output:
	 * \code
	 *	 ------------------------------------------------------------------------
	 *	| Stopwatch                                                            |
	 *	|----------------------------------------------------------------------|
	 *	|          Tag         Start          Stop           Lap         Total |
	 *	|        start  00:00:00.000  00:00:00.004  00:00:00.004  00:00:00.004 |
	 *	| setupNetwork  00:00:00.004  00:00:00.064  00:00:00.060  00:00:00.064 |
	 *	|   runNetwork  00:00:00.064  00:00:06.651  00:00:06.587  00:00:06.651 |
	 *	------------------------------------------------------------------------
	 * \endcode
	 * Each lap is presented on its own row, annotated by its tag.
	 * The start time of the first lap is always 0. All other times are reported relative to that start time.
	 * The time format is: number of hours:number of minutes:number of second.number of milliseconds.
	 * The column "Total" contains the recorded execution time accumulated over all laps.
	 *
	 * \param fileStream file stream where to print the summary (default: NULL, which redirects stream to stdout)
	 *
	 * \note This method can only be called when the timer is off (i.e., after calling Stopwatch::stop).
	 * \since v3.1
	 */
	void print(FILE* fileStream=NULL) const;

private:
	// This class provides a pImpl for the CARLsim User API.
	// \see https://marcmutz.wordpress.com/translated-articles/pimp-my-pimpl/
	class Impl;
	Impl* _impl;
};

#endif // STOPWATCH_H