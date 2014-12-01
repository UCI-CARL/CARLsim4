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
 * *************************************************************************
 * CARLsim
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 11/26/2014
 */

// paradigm shift: run this on spikes.

#ifndef _GROUP_MON_H_
#define _GROUP_MON_H_

#include <carlsim_datastructures.h> // spikeMonMode_t
#include <vector>					// std::vector

class CpuSNN; 			// forward declaration of CpuSNN class
class GroupMonitorCore; // forward declaration of implementation

/*! To retrieve group status, a group-monitoring callback mechanism is used. This mechanism allows the user to monitor
 * basic status of a group (currently support concentrations of neuromodulator). Group monitors are registered
 * for a group and are called automatically by the simulator every second. The parameter would be the group ID, an
 * array of data, number of elements in that array.
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
	~GroupMonitor();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	/*!
	 *\brief Truncates the data vector
	 */
	void clear();

	/*!
	 * \brief Recording status (true=recording, false=not recording)
	 *
	 * Gets record status as a bool. True means it is recording, false means it is not recording.
	 * \returns bool that is true if object is recording, false otherwise.
	 */
	bool isRecording();

	/*!
	 *\brief prints the data (neuromodulator) data vector.
	 */
	void print();

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
	unsigned int getRecordingTotalTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingStartTime().
	 * \returns the simulation time (ms) of the last call to startRecording()
	 */
	unsigned int getRecordingLastStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the first call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the first call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingLastStartTime().
	 * \returns the simulation time (ms) of the first call to startRecording()
	 */
	unsigned int getRecordingStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to stopRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to stopRecording().
	 * \returns the simulation time (ms) of the last call to stopRecording()
	 */
	unsigned int getRecordingStopTime();

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


 private:
	//! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
	GroupMonitorCore* groupMonitorCorePtr_;

};

#endif
