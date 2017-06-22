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
* Ver 05/24/2017
*/

#include <neuron_monitor_core.h>

#include <snn.h>				// CARLsim private implementation
#include <snn_definitions.h>	// KERNEL_ERROR, KERNEL_INFO, ...

#include <algorithm>			// std::sort

NeuronMonitorCore::NeuronMonitorCore(SNN* snn, int monitorId, int grpId) {
	snn_ = snn;
	grpId_= grpId;
	monitorId_ = monitorId;
	nNeurons_ = -1;
	neuronFileId_ = NULL;
	recordSet_ = false;
	neuronMonLastUpdated_ = 0;

	persistentData_ = false;
    userHasBeenWarned_ = false;
	needToWriteFileHeader_ = true;
    neuronFileSignature_ = 206661979;
	neuronFileVersion_ = 0.2f;

	// defer all unsafe operations to init function
	init();
}

void NeuronMonitorCore::init() {
	nNeurons_ = snn_->getGroupNumNeurons(grpId_);
	assert(nNeurons_>0);

	// so the first dimension is neuron ID
	vectorV_.resize(nNeurons_);
    vectorU_.resize(nNeurons_);
    vectorI_.resize(nNeurons_);

	clear();

	// use KERNEL_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();
}

NeuronMonitorCore::~NeuronMonitorCore() {
	if (neuronFileId_!=NULL) {
		fclose(neuronFileId_);
		neuronFileId_ = NULL;
	}
}

void NeuronMonitorCore::clear() {
	assert(!isRecording());
	recordSet_ = false;
    userHasBeenWarned_ = false;
	startTime_ = -1;
	startTimeLast_ = -1;
	stopTime_ = -1;
	accumTime_ = 0;
	totalTime_ = -1;

	for (int i=0; i<nNeurons_; i++){
		vectorV_[i].clear();
        vectorU_[i].clear();
        vectorI_[i].clear();
    }
}

void NeuronMonitorCore::pushNeuronState(int neurId, float V, float U, float I) {
	assert(isRecording());

	vectorV_[neurId].push_back(V);
    vectorU_[neurId].push_back(U);
    vectorI_[neurId].push_back(I);
}

void NeuronMonitorCore::startRecording() {
	assert(!isRecording());

	if (!persistentData_) {
		// if persistent mode is off (default behavior), automatically call clear() here
		clear();
	}

	// call updateNeuronMonitor to make sure neuron state file and neuron state vector are up-to-date
	// Caution: must be called before recordSet_ is set to true!
	snn_->updateNeuronMonitor(grpId_);

	recordSet_ = true;
	long int currentTime = snn_->getSimTimeSec()*1000+snn_->getSimTimeMs();

	if (persistentData_) {
		// persistent mode on: accumulate all times
		// change start time only if this is the first time running it
		startTime_ = (startTime_<0) ? currentTime : startTime_;
		startTimeLast_ = currentTime;
		accumTime_ = (totalTime_>0) ? totalTime_ : 0;
	}
	else {
		// persistent mode off: we only care about the last probe
		startTime_ = currentTime;
		startTimeLast_ = currentTime;
		accumTime_ = 0;
	}
}

void NeuronMonitorCore::stopRecording() {
	assert(isRecording());
	assert(startTime_>-1 && startTimeLast_>-1 && accumTime_>-1);

	// call updateNeuronMonitor to make sure neuron state file and neuron state vector are up-to-date
	// Caution: must be called before recordSet_ is set to false!
	snn_->updateNeuronMonitor(grpId_);

	recordSet_ = false;
    userHasBeenWarned_ = false;
	stopTime_ = snn_->getSimTimeSec()*1000+snn_->getSimTimeMs();

	// total time is the amount of time of the last probe plus all accumulated time from previous probes
	totalTime_ = stopTime_-startTimeLast_ + accumTime_;
	assert(totalTime_>=0);
}

// returns the total accumulated time.
long int NeuronMonitorCore::getAccumTime(){
    return accumTime_;
}

void NeuronMonitorCore::setNeuronFileId(FILE* neuronFileId) {
	assert(!isRecording());

	// close previous file pointer if exists
	if (neuronFileId_!=NULL) {
		fclose(neuronFileId_);
		neuronFileId_ = NULL;
	}

	// set it to new file id
	neuronFileId_=neuronFileId;

	if (neuronFileId_==NULL)
		needToWriteFileHeader_ = false;
	else {
		// file pointer has changed, so we need to write header (again)
		needToWriteFileHeader_ = true;
		writeNeuronFileHeader();
	}
}

// write the header section of the neuron state file
void NeuronMonitorCore::writeNeuronFileHeader() {
	if (!needToWriteFileHeader_)
		return;

	// write file signature
	if (!fwrite(&neuronFileSignature_,sizeof(int),1,neuronFileId_))
		KERNEL_ERROR("NeuronMonitorCore: writeNeuronFileHeader has fwrite error");

	// write version number
	if (!fwrite(&neuronFileVersion_,sizeof(float),1,neuronFileId_))
		KERNEL_ERROR("NeuronMonitorCore: writeNeuronFileHeader has fwrite error");

	// write grid dimensions
	Grid3D grid = snn_->getGroupGrid3D(grpId_);
	int tmpInt = grid.numX;
	if (!fwrite(&tmpInt,sizeof(int),1,neuronFileId_))
		KERNEL_ERROR("NeuronMonitorCore: writeNeuronFileHeader has fwrite error");

	tmpInt = grid.numY;
	if (!fwrite(&tmpInt,sizeof(int),1,neuronFileId_))
		KERNEL_ERROR("NeuronMonitorCore: writeNeuronFileHeader has fwrite error");

	tmpInt = grid.numZ;
	if (!fwrite(&tmpInt,sizeof(int),1,neuronFileId_))
		KERNEL_ERROR("NeuronMonitorCore: writeNeuronFileHeader has fwrite error");


	needToWriteFileHeader_ = false;
}

long int NeuronMonitorCore::getBufferSize(){
    long int bufferSize=0; // in bytes
    for(int i=0; i<vectorV_.size();i++){
        bufferSize+=vectorV_[i].size()*sizeof(int);
    }
    return 3 * bufferSize;
}

// check if the state vector is getting large. If it is, return true once until
// stopRecording is called.
bool NeuronMonitorCore::isBufferBig(){
    if(userHasBeenWarned_)
        return false;
    else {
        //check if buffer is too big
        if(this->getBufferSize()>MAX_NEURON_MON_BUFFER_SIZE){
            userHasBeenWarned_=true;
            return true;
        }
        else {
            return false;
        }
    }
}

std::vector<std::vector<float> > NeuronMonitorCore::getVectorV(){
	assert(!isRecording());
	return vectorV_;
}

std::vector<std::vector<float> > NeuronMonitorCore::getVectorU(){
	assert(!isRecording());
	return vectorU_;
}

std::vector<std::vector<float> > NeuronMonitorCore::getVectorI(){
	assert(!isRecording());
	return vectorI_;
}

void NeuronMonitorCore::print() {
	assert(!isRecording());

	// how many spike times to display per row
	int dispVoltsPerRow = 7;

	// spike times only available in AER mode
	KERNEL_INFO("| Neur ID | volt");
	KERNEL_INFO("|- - - - -|- - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - -")

	for (int i=0; i<nNeurons_; i++) {
		char buffer[100];
#if defined(WIN32) || defined(WIN64)
		_snprintf(buffer, 100, "| %7d | ", i);
#else
		snprintf(buffer, 100, "| %7d | ", i);
#endif
		int nV = vectorV_[i].size();
		for (int j=0; j<nV; j++) {
			char volts[10];
#if defined(WIN32) || defined(WIN64)
			_snprintf(volts, 10, "%4.4f ", vectorV_[i][j]);
#else
			snprintf(volts, 10, "%4.4f ", vectorV_[i][j]);
#endif
			strcat(buffer, volts);
			if (j%dispVoltsPerRow == dispVoltsPerRow-1 && j<nV-1) {
				KERNEL_INFO("%s",buffer);
				strcpy(buffer,"|         |");
			}
		}
		KERNEL_INFO("%s",buffer);
	}
}