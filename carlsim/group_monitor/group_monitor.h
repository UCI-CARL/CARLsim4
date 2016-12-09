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

// paradigm shift: run this on spikes.

#ifndef _GROUP_MON_H_
#define _GROUP_MON_H_

#include <carlsim_datastructures.h>
#include <vector>					// std::vector

class SNN; 			// forward declaration of SNN class
class GroupMonitorCore; // forward declaration of implementation

/*!
 * \brief Class GroupMonitor
 *
 * The GroupMonitor class allows a user record group data (only support dopamine concentration for now) from a particular
 * neuron group. First the method CARLsim::setGroupMonitor must be called with the group ID of the desired group as an
 * argument. The setGroupMonitor call returns a pointer to a GroupMonitor object which can be queried for group data.
 *
 * Group data will not be recorded until the GroupMonitor member function startRecording() is called.
 * Before any metrics can be computed, the user must call stopRecording(). In general, a new recording period
 * (the time period between startRecording and stopRecording calls) can be started at any point in time, and can
 * last any number of milliseconds. The GroupMonitor has a PersistentMode, which is off by default. When
 * PersistentMode is off, only the last recording period will be considered. When PersistentMode is on, all the
 * recording periods will be considered. By default, PersistentMode can be switched on/off by calling
 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
 * getRecordingTotalTime().
 *
 * GroupMonitor objects should only be used after setupNetwork has been called.
 * GroupMonitor objects will be deallocated automatically. The caller should not delete(free) GroupMonitor objects
 *
 * Example usage:
 * \code
 * // configure a network etc. ...
 *
 * sim.setupNetwork();
 *
 * // create a GroupMonitor pointer to grab the pointer from setGroupMonitor.
 * GroupMonitor* daGroupMon;
 * // call setGroupMonitor with carlsim object, sim, with the group ID, daGrpId, as an argument.
 * daGroupMon=sim.setGroupMonitor(daGrpId);
 * // begin recording group data for DA group
 * daGroupMon->startRecording();
 * // run simulation that generates spikes for 20 seconds.
 * sim.runNetwork(20);
 * // stop recording group data for DA group so we can get spike statistics.
 * daGroupMon->stopRecording();
 * // print a summary of the group data information
 * daGroupMon->print();
 * // get the average value of group data (only support dopamine concentration for now) of DA group
 * float avgDAValue = daGroupMon->getMeanValue();
 * \endcode
 *
 * \TODO finish documentation
 */
class GroupMonitor {
 public:
	/*!
	 * \brief GroupMonitor constructor
	 *
	 * Creates a new instance of the GroupMonitor class.
	 *
	 */
	GroupMonitor(GroupMonitorCore* groupMonitorCorePtr);

	/*!
	 * \brief GroupMonitor destructor.
	 *
	 * Cleans up all the memory upon object deletion.
	 *
	 */
	virtual ~GroupMonitor();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	/*!
	 * \brief Recording status (true=recording, false=not recording)
	 *
	 * Gets record status as a bool. True means it is recording, false means it is not recording.
	 * \returns bool that is true if object is recording, false otherwise.
	 */
	bool isRecording();

	/*!
	 * \brief Starts a new recording period
	 *
	 * This function starts a new recording period. From that moment onward, the data vector will be populated
	 * with all the concentrations of enabled neuromodulators of the group. Before any metrics can be computed,
	 * the user must call stopRecording().
	 * Recording periods must be ended with stopRecording(). In general, a new recording period can be started at any
	 * point in time, and can last any number of milliseconds.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 */
	void startRecording();

	/*!
	 * \brief Ends a recording period
	 *
	 * This function ends a recording period, at which point the data vector will no longer be populated
	 * with new neuromodulator data. In general, a recording period can be ended at any point in time,
	 * and last any number of milliseconds.
	 */
	void stopRecording();

	/*!
	 * \brief Returns the total recording time (ms)
	 *
	 * This function returns the total amount of recording time upon which the calculated metrics are based.
	 * If PersistentMode is off, this number is equivalent to getRecordingStopTime()-getRecordingStartTime().
	 * If PersistentMode is on, this number is equivalent to the total time accumulated over all past recording
	 * periods. Note that this is not necessarily equivalent to getRecordingStopTime()-getRecordingStartTime(), as
	 * there might have been periods in between where recording was off.
	 * \returns the total recording time (ms)
	 */
	int getRecordingTotalTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingStartTime().
	 * \returns the simulation time (ms) of the last call to startRecording()
	 */
	int getRecordingLastStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the first call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the first call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingLastStartTime().
	 * \returns the simulation time (ms) of the first call to startRecording()
	 */
	int getRecordingStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to stopRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to stopRecording().
	 * \returns the simulation time (ms) of the last call to stopRecording()
	 */
	int getRecordingStopTime();

	/*!
	 * \brief Returns a flag that indicates whether PersistentMode is on (true) or off (false)
	 *
	 * This function returns a flag that indicates whether PersistentMode is currently on (true) or off (false).
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 */
	bool getPersistentData();

	/*!
	 * \brief Sets PersistentMode either on (true) or off (false)
	 *
	 * This function sets PersistentMode either on (true) or off (false).
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time.
	 * The current state of PersistentMode can be retrieved by calling getPersistentData().
	 */
	void setPersistentData(bool persistentData);

	/*!
	 * \brief return the group data vector
	 *
	 * This function returns a vector containing all group data (only support dopamine concentration for now)
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of float values presenting dopamine concentration
	 */
	std::vector<float> getDataVector();

	/*!
	 * \brief return a vector of the timestamps for group data
	 *
	 * This function returns a vector containing all timestamps for group data.
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of int values presenting the timestamps
	 */
	std::vector<int> getTimeVector();

	/*!
	 * \brief return a vector of peak values in group data
	 *
	 * This function returns a vector containing all peak values for group data.
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of float values which are peaks (local maximum values) in group data
	 */
	std::vector<float> getPeakValueVector();

	/*!
	 * \brief return a vector of the timestamps for peak values in group data
	 *
	 * This function returns a vector containing all timestamps of peaks (local maximum value) in group data.
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of int values presenting the timestamps of peaks
	 */
	std::vector<int> getPeakTimeVector();

	/*!
	 * \brief return a vector of peak values in group data (sorted in decending order)
	 *
	 * This function returns a vector containing all sorted peak values in group data (sorted in decending order).
	 * In other word, the first element in the vector is the highest peak value in recording duration.
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of float values presenting sorted peaks
	 */
	std::vector<float> getSortedPeakValueVector();

	/*!
	 * \brief return a vector of the timestamps for peak values in group data (sorted in decending order)
	 *
	 * This function returns a vector containing all timestamps for sorted peak values (sorted in decending order)
	 * In other word, the first element in the vector is the timestamp of the highest peak in recording duration.
	 * If PersistentMode is off, only the last recording period will be considered for calculating metrics.
	 * If PersistentMode is on, all the recording periods will be considered. By default, PersistentMode is off, but
	 * can be switched on at any point in time by calling setPersistentData(bool).
	 * \returns 1D vector of int values presenting the timestamps of sorted peaks
	 */
	std::vector<int> getSortedPeakTimeVector();

 private:
	//! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
	GroupMonitorCore* groupMonitorCorePtr_;
};

#endif
