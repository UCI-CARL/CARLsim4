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

#ifndef _GROUP_MON_CORE_H_
#define _GROUP_MON_CORE_H_

#include <carlsim_datastructures.h>	// Neuromodulator
#include <stdio.h>					// FILE
#include <vector>					// std::vector

class SNN; // forward declaration of SNN class

/*!
 * \brief GroupMonitor private core implementation
 *
 * \sa GroupMonitor
 */
class GroupMonitorCore {
public: 
	//! constructor (called by CARLsim::setGroupMonitor)
	GroupMonitorCore(SNN* snn, int monitorId, int grpId); 

	//! destructor, cleans up all the memory upon object deletion
	~GroupMonitorCore();


	// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

	//! returns the group ID
	int getGrpId() { return grpId_; }

	//! returns number of neurons in the group
	int getGrpNumNeurons() { return nNeurons_; }

	//! returns the GroupMonitor ID
	int getMonitorId() { return monitorId_; }

	//! returns status of PersistentData mode	
	bool getPersistentData() { return persistentData_; }

	//! returns the total recorded time in ms
	int getRecordingTotalTime() { return totalTime_; }

	//! retunrs the timestamp of the first startRecording in ms
	int getRecordingStartTime() { return startTime_; }

	//! returns the timestamp of the last startRecording in ms
	int getRecordingLastStartTime() { return startTimeLast_; }

	//! returns the timestamp of stopRecording
	int getRecordingStopTime() { return stopTime_; }

	//! returns recording status
	bool isRecording() { return recordSet_; }

	//! inserts group data (time, value) into the vectors
	void pushData(int time, float data);

	//! sets status of PersistentData mode
	void setPersistentData(bool persistentData) { persistentData_ = persistentData; }

	//! starts recording group data
	void startRecording();
	
	//! stops recording group data
	void stopRecording();

	//! get the group data
	std::vector<float> getDataVector();

	//! get the timestamps for group data
	std::vector<int> getTimeVector();

	//! get the peak values of group data	
	std::vector<float> getPeakValueVector();

	//! get the timestamps for peak values
	std::vector<int> getPeakTimeVector();

	//! get the sorted peak values of group data
	std::vector<float> getSortedPeakValueVector();

	//! get the sorted timestamps for peak values
	std::vector<int> getSortedPeakTimeVector();

	// +++++ PUBLIC METHODS THAT SHOULD NOT BE EXPOSED TO INTERFACE +++++++++//

	//! deletes the data vector
	void clear();

	//! returns a pointer to the group data file
	FILE* getGroupFileId() { return groupFileId_; }

	//! sets pointer to group data file
	void setGroupFileId(FILE* groupFileId);
	
	//! returns timestamp of last GroupMonitor update
	int getLastUpdated() { return grpMonLastUpdated_; }

	//! sets timestamp of last GroupMonitor update
	void setLastUpdated(unsigned int lastUpdate) { grpMonLastUpdated_ = lastUpdate; }

private:
	//! initialization method
	void init();

	//! writes the header section (file signature, version number) of a group data file
	void writeGroupFileHeader();

	//! whether we have to write header section of group data file
	bool needToWriteFileHeader_;

	SNN* snn_;	//!< private CARLsim implementation
	int monitorId_;	//!< current GroupMonitor ID
	int grpId_;		//!< current group ID
	int nNeurons_;	//!< number of neurons in the group

	FILE* groupFileId_;	//!< file pointer to the group data file or NULL
	int groupFileSignature_; //!< int signature of group data file
	float groupFileVersion_; //!< version number of group data file

	//! Used for analyzing the group data (only support dopamine concentration for now)
	std::vector<int> timeVector_;
	std::vector<float> dataVector_;

	bool recordSet_;			//!< flag that indicates whether we're currently recording
	int startTime_;	 			//!< time (ms) of first call to startRecording
	int startTimeLast_; 		//!< time (ms) of last call to startRecording
	int stopTime_;		 		//!< time (ms) of stopRecording
	int totalTime_;				//!< the total amount of recording time (over all recording periods)
	int accumTime_;

	int grpMonLastUpdated_;		//!< time (ms) when group was last run through updateGroupMonitor

	//! whether data should be persistent (true) or clear() should be automatically called by startRecording (false)
	bool persistentData_;

	// file pointers for error logging
	const FILE* fpInf_;
	const FILE* fpErr_;
	const FILE* fpDeb_;
	const FILE* fpLog_;
};

#endif
