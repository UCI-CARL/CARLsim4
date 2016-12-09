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

#ifndef _SPIKE_MON_CORE_H_
#define _SPIKE_MON_CORE_H_

#include <carlsim_datastructures.h>	// SpikeMonMode
#include <stdio.h>					// FILE
#include <vector>					// std::vector

class SNN; // forward declaration of SNN class


/*
 * \brief SpikeMonitor private core implementation
 *
 * Naming convention for methods:
 * - getPop*: 		a population metric (single value) that applies to the entire group; e.g., getPopMeanFiringRate.
 * - getNeuron*: 	a neuron metric (single value), about a specific neuron (requires neurId); e.g., getNeuronNumSpikes.
 * - getAll*: 		a metric (vector) that is based on all neurons in the group; e.g. getAllFiringRates.
 * - getNum*:		a number metric, returns an int
 * - getPercent*:	a percentage metric, returns a float
 * - get*:			all the others
 */
class SpikeMonitorCore {
public:
	//! constructor (called by CARLsim::setSpikeMonitor)
	SpikeMonitorCore(SNN* snn, int monitorId, int grpId);

	//! destructor, cleans up all the memory upon object deletion
	~SpikeMonitorCore();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	//! returns a list of firing rates for all neurons in the group (sorted by neuron ID ascending)
	std::vector<float> getAllFiringRates();

	//! returns a list of firing rates for all neurons in the group (sorted by firing rate ascending)
	std::vector<float> getAllFiringRatesSorted();

	//! returns the group ID
	int getGrpId() { return grpId_; }

	//! returns number of neurons in the group
	int getGrpNumNeurons() { return nNeurons_; }

	//! returns the largest recorded firing rate
	float getMaxFiringRate();

	//! returns the smallest recorded firing rate
	float getMinFiringRate();

	//! returns recording mode
	SpikeMonMode getMode() { return mode_; }

	//! returns the SpikeMonitor ID
	int getMonitorId() { return monitorId_; }

	//! returns the recorded mean firing rate for a specific neuron
	float getNeuronMeanFiringRate(int neurId);

	//! returns the number of recorded spikes of a specific neuron
	int getNeuronNumSpikes(int neurId);

	//! returns number of neurons whose firing rate was in [min,max] during recording
	int getNumNeuronsWithFiringRate(float min, float max);

	//! returns number of neurons that didn't spike while recording
	int getNumSilentNeurons();

	//! returns percentage of neurons whose firing rate was in [min,max] during recording
	float getPercentNeuronsWithFiringRate(float min, float max);

	//! returns percentage of neurons that didn't spike during recording
	float getPercentSilentNeurons();

	//! returns status of PersistentData mode
	bool getPersistentData() { return persistentData_; }

	//! returns the recorded mean firing rate of the group
	float getPopMeanFiringRate();

	//! returns the total number of recorded spikes in the group
	int getPopNumSpikes();

	//! computes the standard deviation of firing rates in the group
	float getPopStdFiringRate();

	//! returns the total recorded time in ms
	long int getRecordingTotalTime() { return totalTime_; }

	//! retunrs the timestamp of the first startRecording in ms
	long int getRecordingStartTime() { return startTime_; }

	//! returns the timestamp of the last startRecording in ms
	long int getRecordingLastStartTime() { return startTimeLast_; }

	//! returns the timestamp of stopRecording
	long int getRecordingStopTime() { return stopTime_; }

	//! returns the 2D AER vector
	std::vector<std::vector<int> > getSpikeVector2D();

	//! returns recording status
	bool isRecording() { return recordSet_; }

	//! prints the AER vector in human-readable format
	void print(bool printSpikeTimes);

	//! inserts a (time,neurId) tupel into the 2D spike vector
	void pushAER(int time, int neurId);

	//! sets recording mode
	void setMode(SpikeMonMode mode) { mode_ = mode; }

	//! sets status of PersistentData mode
	void setPersistentData(bool persistentData) { persistentData_ = persistentData; }

	//! starts recording AER data
	void startRecording();

	//! stops recording AER data
	void stopRecording();


	// +++++ PUBLIC METHODS THAT SHOULD NOT BE EXPOSED TO INTERFACE +++++++++//

	//! deletes data from the 2D spike vector
	void clear();

	//! returns a pointer to the spike file
	FILE* getSpikeFileId() { return spikeFileId_; }

	//! sets pointer to spike file
	void setSpikeFileId(FILE* spikeFileId);

	//! returns timestamp of last SpikeMonitor update
	long int getLastUpdated() { return spkMonLastUpdated_; }

	//! sets timestamp of last SpikeMonitor update
	void setLastUpdated(long int lastUpdate) { spkMonLastUpdated_ = lastUpdate; }

    //! returns true if spike buffer is close to maxAllowedBufferSize
    bool isBufferBig();

    //! returns the approximate size of the spike vector in bytes
    long int getBufferSize();

    //! returns the total accumulated time
    long int getAccumTime();

private:
	//! initialization method
	void init();

	//! reads AER vector and updates firing rate member var
	void calculateFiringRates();

	//! reads AER vector and updates sorted firing rate member var
	void sortFiringRates();

	//! writes the header section (file signature, version number) of a spike file
	void writeSpikeFileHeader();

	//! whether we have to perform calculateFiringRates()
	bool needToCalculateFiringRates_;

	//! whether we have to perform sortFiringRates()
	bool needToSortFiringRates_;

	//! whether we have to write header section of spike file
	bool needToWriteFileHeader_;

	SNN* snn_;	//!< private CARLsim implementation
	int monitorId_;	//!< current SpikeMonitor ID
	int grpId_;		//!< current group ID
	int nNeurons_;	//!< number of neurons in the group

	FILE* spikeFileId_;	//!< file pointer to the spike file or NULL
	int spikeFileSignature_; //!< int signature of spike file
	float spikeFileVersion_; //!< version number of spike file

	//! Used to analyzed the spike information
	std::vector<std::vector<int> > spkVector_;

	std::vector<float> firingRates_;
	std::vector<float> firingRatesSorted_;

	bool recordSet_;			//!< flag that indicates whether we're currently recording
	long int startTime_;	 	//!< time (ms) of first call to startRecording
	long int startTimeLast_; 	//!< time (ms) of last call to startRecording
	long int stopTime_;		 	//!< time (ms) of stopRecording
	long int totalTime_;		//!< the total amount of recording time (over all recording periods)
	long int accumTime_;

	long int spkMonLastUpdated_;//!< time (ms) when group was last run through updateSpikeMonitor

	//! whether data should be persistent (true) or clear() should be automatically called by startRecording (false)
	bool persistentData_;

    //! Indicates if we have returned true at least once in isBufferBig(). Gets reset in stopRecording(). Used to warn the user only once.
	bool userHasBeenWarned_;
    SpikeMonMode mode_;

	// file pointers for error logging
	const FILE* fpInf_;
	const FILE* fpErr_;
	const FILE* fpDeb_;
	const FILE* fpLog_;
};


#endif
