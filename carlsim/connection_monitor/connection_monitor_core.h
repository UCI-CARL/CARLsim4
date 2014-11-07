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
 * Ver 7/29/2014
 */

#ifndef _SPIKE_MON_CORE_H_
#define _SPIKE_MON_CORE_H_

#include <carlsim_datastructures.h>	// spikeMonMode_t
#include <stdio.h>					// FILE
#include <vector>					// std::vector

class CpuSNN; // forward declaration of CpuSNN class


/*
 * \brief ConnectionMonitor private core implementation
 *
 * Naming convention for methods:
 * - getPop*: 		a population metric (single value) that applies to the entire group; e.g., getPopMeanFiringRate.
 * - getNeuron*: 	a neuron metric (single value), about a specific neuron (requires neurId); e.g., getNeuronNumSpikes. 
 * - getAll*: 		a metric (vector) that is based on all neurons in the group; e.g. getAllFiringRates.
 * - getNum*:		a number metric, returns an int
 * - getPercent*:	a percentage metric, returns a float
 * - get*:			all the others
 */
class ConnectionMonitorCore {
public: 
	/*! 
	 * \brief constructor
	 *
	 * Creates a new instance of the analysis class. 
	 * Takes a CpuSNN pointer to an object as an input and assigns it to a CpuSNN* member variable for easy reference.
	 * \param[in] snn pointer to current CpuSNN object
	 * \param[in] grpId the ID of the group we're monitoring
	 */
	ConnectionMonitorCore(CpuSNN* snn, int monitorId, int grpId); 

	//! destructor, cleans up all the memory upon object deletion
	~ConnectionMonitorCore();
	
	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	//! deletes data from the 2D spike vector
	void clear();
	
	/*!
	 * \brief returns the average firing rate for a certain group averaged
	 * over the simulation time duration. The time duration over
	 * which the neurons are averaged is calculated automatically when
	 * the user calls startRecording()/stopRecording().
	 * \param void.
	 * \return float value for the average firing rate of the whole group. 
	 */
	float getPopMeanFiringRate();

	//! computes the standard deviation of firing rates in the group
	float getPopStdFiringRate();

	/*!
	 * \brief returns the total number of spikes in the group
	 *
	 * This function returns the total number of spikes in the group, which is equal to the number of elements
	 * in the 2D spike vector.
	 * Use getNeuronNumSpikes to find the number of spikes of a specific neuron in the group
	 */
	int getPopNumSpikes();

	/*!
	 * \brief return the average firing rate for each neuron in a group of
	 * neurons over the simulation time duration. The time duration over
	 * which the neurons are averaged is calculated automatically when
	 * the user calls startRecording()/stopRecording().
	 * \param void
	 * \return float vector for the average firing rate of each neuron.
	 */
	std::vector<float> getAllFiringRates();

	/*!
	 * \brief return the number of neurons that have exactly 0 Hz firing
	 * rate.
	 * \param void
	 * \return int value of the number of silent neurons.
	 */
	float getMaxFiringRate();
	
	/*!
	 * \brief return the number of neurons that have exactly 0 Hz firing
	 * rate.
	 * \param void
	 * \return int value of the number of silent neurons.
	 */
	float getMinFiringRate();

	/*!
	 * \brief returns the mean firing rate of a specific neuron in the group
	 *
	 * This function returns the mean firing rate of a specific neuron in the group, averaged over the recording
	 * period.
	 */
	float getNeuronMeanFiringRate(int neurId);

	/*!
	 * \brief returns the number of spikes of a specific neuron in the group
	 *
	 * This function returns the number of spikes of a specific neuron in the group, which is equal to the number of
	 * elements of spikeVector2D[neurId].
	 * Use getPopNumSpikes to find the total number of spikes in the group
	 */
	int getNeuronNumSpikes(int neurId);


	/*!
	 * \brief return the ascending sorted firing rate for a particular 
	 * group of  neurons over the simulation time duration. The time 
	 * duration over which the neurons are averaged is calculated 
	 * automatically when the user calls startRecording()/stopRecording().
	 * \param void
	 * \return float value for the max firing rate of each neuron. The
	 * firing rate is taken every second and the max firing rate is taken
	 * from those values.
	 */
	std::vector<float> getAllFiringRatesSorted();
	
	/*!
	 * \brief return the number of neurons that fall within this particular
	 * min/max range (inclusive). The time duration over which the neurons 
	 * are averaged is calculated automatically when the user calls 
	 * startRecording()/stopRecording().
	 * \param void
	 * \return int value of the number of neurons that have a firing rate
	 * within the min/max range.
	 */
	int getNumNeuronsWithFiringRate(float min, float max);
	
	/*!
	 * \brief returns the number of neurons that are silent.
	 *
	 * \param min minimum value of range (inclusive) to be searched.
	 * \param max maximum value of range (inclusive) to be searched.
	 * \return int of the number of neurons that are silent.
	 */
	int getNumSilentNeurons();

	/*!
	 * \brief returns the percentage of total neurons in that are in the range
	 * specified by the user, min/max (inclusive). 
	 * \param min minimum value of range (inclusive) to be searched.
	 * \param max maximum value of range (inclusive) to be searched.
	 * \return float of the percentage of total neurons that are in this range.
	 */
	float getPercentNeuronsWithFiringRate(float min, float max);

	/*!
	 * \brief returns the percentage of total neurons in group that are silent.
	 * \param void
	 * \return float of the percentage of total neurons that are silent.
	 */
	float getPercentSilentNeurons();
	
	/*!
	 *\brief returns the AER vector.
	 *\param void.
	 *\return AER vector is returned.
	 */
	std::vector<std::vector<int> > getSpikeVector2D();

	/*!
	 *\brief prints the AER vector.
	 *\param void.
	 *\return AER vector is printed.
	 */
	void print(bool printSpikeTimes);

	//! inserts a (time,neurId) tupel into the 2D spike vector
	void pushAER(int time, int neurId);

	/*!
	 * \brief starts copying AER data to AER data structure every second.
	 * \param void
	 * \return void
	 */	
	void startRecording();
	
	/*!
	 * \brief stops copying AER data to AER data structure every second.
	 * \param void
	 * \return void
	 */
	void stopRecording();

	
	// +++++ PUBLIC SETTERS/GETTERS: +++++++++++++++++++++++++++++++++++++++//

	int getGrpId() { return grpId_; }
	int getGrpNumNeurons() { return nNeurons_; }
	int getMonitorId() { return monitorId_; }
	FILE* getSpikeFileId() { return spikeFileId_; }

	long int getRecordingTotalTime() { return totalTime_; }
	long int getRecordingStartTime() { return startTime_; }
	long int getRecordingLastStartTime() { return startTimeLast_; }
	long int getRecordingStopTime() { return stopTime_; }

	bool isRecording() { return recordSet_; }

	bool getPersistentData() { return persistentData_; }
	void setPersistentData(bool persistentData) { persistentData_ = persistentData; }

	spikeMonMode_t getMode() { return mode_; }
	void setMode(spikeMonMode_t mode) { mode_ = mode; }

	long int getLastUpdated() { return spkMonLastUpdated_; }
	void setLastUpdated(long int lastUpdate) { spkMonLastUpdated_ = lastUpdate; }

	/*!
	 * \brief returns the monBufferFid
	 * \param void
	 * \return unsigned int
	 */
	void setSpikeFileId(FILE* spikeFileId);
	
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

	CpuSNN* snn_;	//!< private CARLsim implementation
	int monitorId_;	//!< current ConnectionMonitor ID
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

	long int spkMonLastUpdated_;//!< time (ms) when group was last run through updateConnectionMonitor

	//! whether data should be persistent (true) or clear() should be automatically called by startRecording (false)
	bool persistentData_;

	spikeMonMode_t mode_;

	// file pointers for error logging
	const FILE* fpInf_;
	const FILE* fpErr_;
	const FILE* fpDeb_;
	const FILE* fpLog_;
};


#endif
