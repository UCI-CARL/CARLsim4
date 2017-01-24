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

#ifndef _SPIKE_MON_H_
#define _SPIKE_MON_H_

#include <carlsim_datastructures.h> // SpikeMonMode
#include <vector>					// std::vector

class SNN; 			// forward declaration of SNN class
class SpikeMonitorCore; // forward declaration of implementation

/*!
 * \brief Class SpikeMonitor
 *
 * The SpikeMonitor class allows a user record spike data from a particular neuron group. First
 * the method CARLsim::setSpikeMonitor must be called with the group ID of the desired group as an
 * argument. The setSpikeMonitor call returns a pointer to a SpikeMonitor object which can be queried for spike data.
 *
 * There are two different modes that define what information is collected exactly.
 * - AER:   AER mode will collect the exact spike times of all neurons in the group and store them in a 2D spike
 *          spike vector. The first dimension of the vector is neuron id, the second dimension is spike times. Each
 *          element spkVector[i] is thus a vector of all spike times for the i-th neuron in the group.
 *          This mode is activated by default.
 *          Because of the sheer amount of information, it is unwise to run this mode for extended periods of time.
 *          Note that recording in this mode may significantly slow down your simulation.
 * - COUNT:	SpikeCount mode will only collect spike count information, such as the number of spikes per neuron. This
 *          mode cannot retrieve exact spike times. Thus it is not possible to calculate some of the more elaborate
 *          metrics, such as spike-time correlations.
 *
 * Spike data will not be recorded until the SpikeMonitor member function startRecording() is called.
 * Before any metrics can be computed, the user must call stopRecording(). In general, a new recording period
 * (the time period between startRecording and stopRecording calls) can be started at any point in time, and can
 * last any number of milliseconds. The SpikeMonitor has a PersistentMode, which is off by default. When
 * PersistentMode is off, only the last recording period will be considered. When PersistentMode is on, all the
 * recording periods will be considered. By default, PersistentMode can be switched on/off by calling
 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
 * getRecordingTotalTime().
 *
 * The above configurations do not affect the generation of the .dat spike file. That file will always contain all the
 * spikes of all the neurons in the group, for the entire simulation.
 *
 * SpikeMonitor objects should only be used after setupNetwork has been called.
 * SpikeMonitor objects will be deallocated automatically.
 *
 * Example usage:
 * \code
 * // configure a network etc. ...
 *
 * sim.setupNetwork();
 *
 * // create a SpikeMonitor pointer to grab the pointer from setSpikeMonitor.
 * SpikeMonitor* spikeMonExc;
 * // call setSpikeMonitor with carlsim object, sim, with the group ID, excGrpId, as an argument.
 * spikeMonExc=sim.setSpikeMonitor(excGrpId);
 * // begin recording spike data for group excGrpId
 * spikeMonExc->startRecording();
 * // run simulation that generates spikes for 20 seconds.
 * sim.runNetwork(20);
 * // stop recording data for group excGrpId so we can get spike statistics.
 * spikeMonExc->stopRecording();
 * // print a summary of the spike information
 * spikeMonExc->print();
 * // get the average firing rate of each of the neurons in group excGrpId
 * vector<float> excFRs = spikeMonExc->getAllFiringRates();
 * \endcode
 *
 * For additional information on how to use the SpikeMonitor object, please see Examples/spikeInfo.
 *
 * \TODO finish documentation
 */
class SpikeMonitor {
 public:
	/*!
	 * \brief SpikeMonitor constructor
	 *
	 * Creates a new instance of the SpikeMonitor class.
	 *
	 */
	SpikeMonitor(SpikeMonitorCore* spikeMonitorCorePtr);

	/*!
	 * \brief SpikeMonitor destructor.
	 *
	 * Cleans up all the memory upon object deletion.
	 *
	 */
	~SpikeMonitor();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	/*!
	 *\brief Truncates the 2D spike vector
	 *
	 * This function truncates all the data found in the 2D spike vector.
	 * If PersistentMode is off, this function will be called automatically in startRecording(), such that all
	 * spikes from previous recording periods will be discarded. By default, PersistentMode is off.
	 * If PersistentMode is on, the user can call this function after any number of recordings. However, isRecording()
	 * must always be off.
	 */
	void clear();

	/*!
	 * \brief Returns the average firing rate of all the neurons in the group as a vector of floats
	 *
	 * This function returns the average firing rate for each neuron in the group.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * Use getGroupFiringRate() to get the mean firing rate of the entire group.
	 * \returns float vector for the average firing rate of each neuron.
	 */
	std::vector<float> getAllFiringRates();

	/*!
	 * \brief Returns all the neuronal mean firing rates in ascending order
	 *
	 * This function returns a vector of neuronal mean firing rates, sorted in ascending order. The size of the vector
	 * is the same as the number of neurons in the group. Firing rates are converted to spikes/sec (Hz).
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns a vector of neuronal mean firing rates in ascending order
	 */
	std::vector<float> getAllFiringRatesSorted();

	/*!
	 * \brief returns the largest neuronal mean firing rate in the group
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns float value of the largest neuronal firing rate of the group.
	 */
	float getMaxFiringRate();

	/*!
	 * \brief returns the smallest neuronal mean firing rate in the group
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns float value of the smallest neuronal firing rate of the group.
	 */
	float getMinFiringRate();

	/*!
	 * \brief returns the mean firing rate of a specific neuron in the group
	 *
	 * This function returns the average firing rate of a specific neuron in the group in spikes/sec (Hz), averaged over
	 * the recording time window.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * Use getPopMeanFiringRate to find the population mean firing rate.
	 * \param[in] neurId the neuron ID (0-indexed, must be smaller than getNumNeurons)
	 */
	float getNeuronMeanFiringRate(int neurId);

	/*!
	 * \brief returns the total number of spikes of a specific neuron in the group
	 *
	 * This function returns the total number of spikes emitted by a specific neuron in the recording period, which is
	 * equal to the number of elements in the 2D spike vector.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * Use getGroupNumSpikes to find the number of spikes of all the neurons in the group.
	 * \param[in] neurId the neuron ID (0-indexed, must be smaller than getNumNeurons)
	 */
	int getNeuronNumSpikes(int neurId);

	/*!
	 * \brief Returns the number of neurons that fall within this particular min/max range (inclusive).
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns int value of the number of neurons that have a firing rate within the min/max range.
	 */
	int getNumNeuronsWithFiringRate(float min, float max);

	/*!
	 * \brief returns the number of neurons that are silent.
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns int of the number of neurons that are silent
	 */
	int getNumSilentNeurons();

	/*!
	 * \brief returns the percentage of total neurons in that are in the range specified by the user,
	 * min/max (inclusive).
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \param min minimum value of range (inclusive) to be searched.
	 * \param max maximum value of range (inclusive) to be searched.
	 * \returns float of the percentage of total neurons that are in this range.
	 */
	float getPercentNeuronsWithFiringRate(float min, float max);

	/*!
	 * \brief returns the percentage of total neurons in group that are silent.
	 *
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns float of the percentage of total neurons that are silent.
	 */
	float getPercentSilentNeurons();

	/*!
	 * \brief Returns the mean firing rate of the entire neuronal population
	 *
	 * This function returns the average firing rate of all the neurons in the group in spikes/sec (Hz), averaged over
	 * the recording time window.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns the average firing rate of all the neurons in the group
	 */
	float getPopMeanFiringRate();

	/*!
	 * \brief Returns the standard deviation of firing rates in the entire neuronal population
	 *
	 * This function returns the standard deviation of firing rates of all the neurons in the group in spikes/sec (Hz),
	 * averaged over the recording time window.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * \returns the standard deviation of firing rates of all the neurons in the group
	 */
	float getPopStdFiringRate();

	/*!
	 * \brief Returns the total number of spikes in the group
	 *
	 * This function returns the total number of spikes in the group, which is equal to the number of elements
	 * in the 2D spike vector.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 * Use getNeuronNumSpikes to find the number of spikes of a specific neuron in the group.
	 */
	int getPopNumSpikes();

	/*!
	 *\brief returns the 2D spike vector
	 *
	 * This function returns a 2D spike vector containing all the spikes of all the neurons in the group.
	 * The first dimension of the vector is neurons, the second dimension is spike times. Each element spkVector[i]
	 * is thus a vector of all spike times for the i-th neuron in the group.
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
	 *\returns 2D vector where first dimension is neurons, second dimension is spike times
	 */
	std::vector<std::vector<int> > getSpikeVector2D();

	/*!
	 * \brief Recording status (true=recording, false=not recording)
	 *
	 * Gets record status as a bool. True means it is recording, false means it is not recording.
	 * \returns bool that is true if object is recording, false otherwise.
	 */
	bool isRecording();

	/*!
	 *\brief prints the 2D spike vector.
	 *
	 * This function prints all the spiking information in the group in legible format.
	 * \param[in] printSpikeTimes whether to print the list of spike times for each neuron
	 */
	void print(bool printSpikeTimes=true);

	/*!
	 * \brief Starts a new recording period
	 *
	 * This function starts a new recording period. From that moment onward, the 2D spike vector will be populated
	 * with all the spikes of all the neurons in the group. Before any metrics can be computed, the user must call
	 * stopRecording().
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
	 * This function ends a recording period, at which point the 2D spike vector will no longer be populated
	 * with spikes. In general, a recording period can be ended at any point in time, and last any number of
	 * milliseconds.
	 * From this moment onward, a variety of metrics can be computed, which are based on the spikes found in the 2D
	 * spike vector. It is also possible to retrieve the raw spike vector itself by calling getSpikeVector2D().
	 * If PersistentMode is off, only the last recording period will be considered. If PersistentMode is on, all the
	 * recording periods will be considered. By default, PersistentMode is off, and can be switched on by calling
	 * setPersistentData(bool). The total time over which the metric is calculated can be retrieved by calling
	 * getRecordingTotalTime().
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
	long int getRecordingTotalTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingStartTime().
	 * \returns the simulation time (ms) of the last call to startRecording()
	 */
	long int getRecordingLastStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the first call to startRecording()
	 *
	 * This function returns the simulation time (timestamp) of the first call to startRecording().
	 * If PersistentMode is off, this number is equivalent to getRecordingLastStartTime().
	 * \returns the simulation time (ms) of the first call to startRecording()
	 */
	long int getRecordingStartTime();

	/*!
	 * \brief Returns the simulation time (ms) of the last call to stopRecording()
	 *
	 * This function returns the simulation time (timestamp) of the last call to stopRecording().
	 * \returns the simulation time (ms) of the last call to stopRecording()
	 */
	long int getRecordingStopTime();

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
	 * \brief Returns the current SpikeMonitor mode
	 *
	 * This function returns the current SpikeMonitor mode.
	 * COUNT:	Will collect only spike count information (such as number of spikes per neuron),
	 *          not the explicit spike times. COUNT mode cannot retrieve exact spike times per
	 *          neuron, and is thus not capable of computing spike train correlation etc.
	 * AER:     Will collect spike information in AER format (will collect both neuron IDs and
	 *          spike times).
	 */
	SpikeMonMode getMode();

	/*!
	 * \brief Sets the current SpikeMonitor mode
	 *
	 * This function returns the current SpikeMonitor mode.
	 * COUNT:	Will collect only spike count information (such as number of spikes per neuron),
	 *          not the explicit spike times. COUNT mode cannot retrieve exact spike times per
	 *          neuron, and is thus not capable of computing spike train correlation etc.
	 * AER:     Will collect spike information in AER format (will collect both neuron IDs and
	 *          spike times).
	 */
	void setMode(SpikeMonMode mode=AER);

	/*!
	 * \brief Sets the name of the spike file binary
	 *
	 * This function sets the name of the spike file binary. It can be called at any time during
	 * the simulation, but must be called outside of startRecording / stopRecording periods.
	 * The function will close the previous file stream (the one that was set in
	 * CARLsim::setSpikeMonitor), and from that moment on direct new incoming spikes into a
	 * new file.
	 * This allows a user to record spikes to different files during a simulation, for example
	 * to "training.dat" and "testing.dat".
	 * In order to stop recording to file, pass string "NULL".
	 *
	 * \param[in] logFileName path to binary file or "NULL" (for not recording to file at all)
	 * \attention Make sure the directory exists!
	 * \since v3.0
	 */
	void setLogFile(const std::string& logFileName);

 private:
  //! This is a pointer to the actual implementation of the class. The user should never directly instantiate it.
  SpikeMonitorCore* spikeMonitorCorePtr_;

};

#endif
