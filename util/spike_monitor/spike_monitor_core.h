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
 * Ver 2/21/2014
 */

#ifndef _SPIKE_MON_CORE_H_
#define _SPIKE_MON_CORE_H_

// we need the AER data structure
#include <carlsim_datastructures.h>
#include <algorithm>
#include <vector>

class CpuSNN; // forward declaration of CpuSNN class

class SpikeMonitorCore {
public: 
	/*! 
	 * \brief analysis constructor.
	 *
	 * Creates a new instance of the analysis class. 
	 * Takes a CpuSNN pointer to an object as an input and assigns it to a CpuSNN* member variable for easy reference.
	 * \param[in] snn pointer to current CpuSNN object
	 * \param[in] grpId the ID of the group we're monitoring
	 *	 
	 */
	SpikeMonitorCore(CpuSNN* snn, int monitorId, int grpId); 
	/*! 
	 * \brief SpikeMonitorCore destructor.
	 *
	 * Cleans up all the memory upon object deletion.
	 *
	 */
	~SpikeMonitorCore();
	
	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	/*!
	 *\brief deletes data from the AER vector.
	 *\param void.
	 *\return void.
	 */
	void clear();
	
	/*!
	 * \brief return the average firing rate for a certain group averaged
	 * over the simulation time duration. The time duration over
	 * which the neurons are averaged is calculated automatically when
	 * the user calls startRecording()/stopRecording().
	 * \param void.
	 * \return float value for the average firing rate of the whole group. 
	 */
	float getGroupFiringRate();

	/*!
	 * \brief return the average firing rate for each neuron in a group of
	 * neurons over the simulation time duration. The time duration over
	 * which the neurons are averaged is calculated automatically when
	 * the user calls startRecording()/stopRecording().
	 * \param void
	 * \return float vector for the average firing rate of each neuron.
	 */
	std::vector<float> getNeuronFiringRate();

	/*!
	 * \brief return the number of neurons that have exactly 0 Hz firing
	 * rate.
	 * \param void
	 * \return int value of the number of silent neurons.
	 */
	float getNeuronMaxFiringRate();
	
	/*!
	 * \brief return the number of neurons that have exactly 0 Hz firing
	 * rate.
	 * \param void
	 * \return int value of the number of silent neurons.
	 */
	float getNeuronMinFiringRate();

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
	std::vector<float> getNeuronSortedFiringRate();
	
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
	 * \brief Return the current size of the AER vector.
	 * \param void
	 * \return the current size of the AER vector.
	 */
	long int getSize();

	/*!
	 *\brief returns the AER vector.
	 *\param void.
	 *\return AER vector is returned.
	 */
	std::vector<AER> getVector();

	/*!
	 *\brief prints the AER vector.
	 *\param void.
	 *\return AER vector is printed.
	 */
	void print();

	/*!
	 * \brief put the nid and time values in an AER vector structure
	 * \param CARLsim* s pointer to CARLsim object so we can access simTime.
	 * \param grpId of group being counted.
	 * \param neurIds unsigned int* array of neuron ids that have fired.
	 * \param timeCnts unsigned int* array of times neurons have fired.
	 * \param timeInterval is the time interval over which these neurons
	 * were recorded.
	 * \return void
	 */	
	void pushAER(int grpId, unsigned int* neurIds, unsigned int* timeCnts, int numMsMin, int numMsMax);

	void pushAER(AER aer);

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
	int getGrpNumNeurons() { return numN_; }
	int getMonitorId() { return monitorId_; }
	FILE* getSpikeFileId() { return spikeFileId_; }

	long int getRecordingTotalTime() { return totalTime_; }
	long int getRecordingStartTime() { return startTime_; }
	long int getRecordingLastStartTime() { return startTimeLast_; }
	long int getRecordingStopTime() { return stopTime_; }

	bool isRecording() { return recordSet_; }

	bool getPersistentMode() { return persistentData_; }
	void setPersistentMode(bool persistentData_);

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

	//! whether we have to perform calculateFiringRates()
	bool needToCalculateFiringRates_;

	//! whether we have to perform sortFiringRates()
	bool needToSortFiringRates_;

	CpuSNN* snn_;
	int monitorId_;
	int grpId_;
	int numN_;

	FILE* spikeFileId_;	//!< file pointer to the spike file or NULL

	// Used to analyzed the spike information
//	std::vector<AER>::const_iterator it_begin_;
//	std::vector<AER>::const_iterator it_end_;
	std::vector<AER> spkVector_;

	//! this flag will be true whenever the firing rate needs updating (e.g., new data is coming in)
	bool needToUpdateFiringRate_;
	std::vector<float> firingRate_;
	std::vector<float> firingRateSorted_;
	std::vector<int> tmpSpikeCount_;

	bool recordSet_;
	long int startTime_;	 //!< time (ms) of first call to startRecording
	long int startTimeLast_; //!< time (ms) of last call to startRecording
	long int stopTime_;		 //!< time (ms) of stopRecording
	long int totalTime_;
	long int accumTime_;

	//! whether data should be persistent (true) or clear() should be automatically called by startRecording (false)
	bool persistentData_;

	// file pointers for error logging
	const FILE* fpInf_;
	const FILE* fpErr_;
	const FILE* fpDeb_;
	const FILE* fpLog_;
};


#endif
