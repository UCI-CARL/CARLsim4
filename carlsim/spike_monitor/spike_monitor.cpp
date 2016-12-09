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
#include <spike_monitor.h>

#include <spike_monitor_core.h>	// SpikeMonitor private implementation
#include <user_errors.h>		// fancy user error messages

#include <sstream>				// std::stringstream
#include <algorithm>			// std::transform


// we aren't using namespace std so pay attention!
SpikeMonitor::SpikeMonitor(SpikeMonitorCore* spikeMonitorCorePtr){
	// make sure the pointer is NULL
	spikeMonitorCorePtr_ = spikeMonitorCorePtr;
}

SpikeMonitor::~SpikeMonitor() {
	delete spikeMonitorCorePtr_;
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void SpikeMonitor::clear(){
	std::string funcName = "clear()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	spikeMonitorCorePtr_->clear();
}

float SpikeMonitor::getPopMeanFiringRate() {
	std::string funcName = "getPopMeanFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getPopMeanFiringRate();
}

float SpikeMonitor::getPopStdFiringRate() {
	std::string funcName = "getPopStdFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getPopStdFiringRate();
}

int SpikeMonitor::getPopNumSpikes() {
	std::string funcName = "getPopNumSpikes()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	// \TODO
	UserErrors::assertTrue(getMode()==AER, UserErrors::UNKNOWN, funcName, "",
		"This function is not yet supported in this mode.");

	return spikeMonitorCorePtr_->getPopNumSpikes();	
}

float SpikeMonitor::getMaxFiringRate(){
	std::string funcName = "getMaxFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getMaxFiringRate();
}

float SpikeMonitor::getMinFiringRate(){
	std::string funcName = "getMinFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getMinFiringRate();
}

std::vector<float> SpikeMonitor::getAllFiringRates(){
	std::string funcName = "getAllFiringRates()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getAllFiringRates();
}

float SpikeMonitor::getNeuronMeanFiringRate(int neurId) {
	std::string funcName = "getNeuronMeanFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNeuronMeanFiringRate(neurId);

}

int SpikeMonitor::getNeuronNumSpikes(int neurId) {
	std::string funcName = "getNeuronNumSpikes()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	// \TODO
	UserErrors::assertTrue(getMode()==AER, UserErrors::UNKNOWN, funcName, "",
		"This function is not yet supported in this mode.");

	return spikeMonitorCorePtr_->getNeuronNumSpikes(neurId);
}

// need to do error check here and maybe throw CARLsim errors.
int SpikeMonitor::getNumNeuronsWithFiringRate(float min, float max){
	std::string funcName = "getNumNeuronsWithFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNumNeuronsWithFiringRate(min,max);
}

// need to do error check here and maybe throw CARLsim errors.
float SpikeMonitor::getPercentNeuronsWithFiringRate(float min, float max) {
	std::stringstream funcName; funcName << "getPercentNeuronsWithFiringRate(" << min << "," << max << ")";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName.str(), "Recording");
	UserErrors::assertTrue(min>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "min");
	UserErrors::assertTrue(max>=0.0f, UserErrors::CANNOT_BE_NEGATIVE, funcName.str(), "max");
	UserErrors::assertTrue(max>=min, UserErrors::CANNOT_BE_LARGER, funcName.str(), "min", "max");

	return spikeMonitorCorePtr_->getPercentNeuronsWithFiringRate(min,max);
}

int SpikeMonitor::getNumSilentNeurons(){
	std::string funcName = "getNumSilentNeurons()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNumSilentNeurons();
}

float SpikeMonitor::getPercentSilentNeurons(){
	std::string funcName = "getPercentSilentNeurons()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getPercentSilentNeurons();
}

std::vector<std::vector<int> > SpikeMonitor::getSpikeVector2D() {
	std::string funcName = "getSpikeVector2D()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");
	UserErrors::assertTrue(getMode()==AER, UserErrors::CAN_ONLY_BE_CALLED_IN_MODE, funcName, funcName, "AER");

	return spikeMonitorCorePtr_->getSpikeVector2D();
}

std::vector<float> SpikeMonitor::getAllFiringRatesSorted(){
	std::string funcName = "getAllFiringRatesSorted()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getAllFiringRatesSorted();
}

bool SpikeMonitor::isRecording(){
	return spikeMonitorCorePtr_->isRecording();
}

void SpikeMonitor::print(bool printSpikeTimes) {
	std::string funcName = "print()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	spikeMonitorCorePtr_->print(printSpikeTimes);
}

void SpikeMonitor::startRecording() {
	std::string funcName = "startRecording()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	spikeMonitorCorePtr_->startRecording();
}

void SpikeMonitor::stopRecording(){
	std::string funcName = "stopRecording()";
	UserErrors::assertTrue(isRecording(), UserErrors::MUST_BE_ON, funcName, "Recording");

	spikeMonitorCorePtr_->stopRecording();
}

long int SpikeMonitor::getRecordingTotalTime() {
	std::string funcName = "getRecordingTotalTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getRecordingTotalTime();
}

long int SpikeMonitor::getRecordingLastStartTime() {
	std::string funcName = "getRecordingLastStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getRecordingLastStartTime();
}

long int SpikeMonitor::getRecordingStartTime() {
	std::string funcName = "getRecordingStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getRecordingStartTime();
}

long int SpikeMonitor::getRecordingStopTime() {
	std::string funcName = "getRecordingStopTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getRecordingStopTime();
}

bool SpikeMonitor::getPersistentData() {
	return spikeMonitorCorePtr_->getPersistentData();
}

void SpikeMonitor::setPersistentData(bool persistentData) {
	spikeMonitorCorePtr_->setPersistentData(persistentData);
}

SpikeMonMode SpikeMonitor::getMode() {
	return spikeMonitorCorePtr_->getMode();
}

void SpikeMonitor::setMode(SpikeMonMode mode) {
	// \TODO
	UserErrors::assertTrue(false, UserErrors::UNKNOWN, "setMode()", "",
		"This function call is not yet supported.");

	spikeMonitorCorePtr_->setMode(mode);
}

void SpikeMonitor::setLogFile(const std::string& fileName) {
	std::string funcName = "setLogFile";

	FILE* fid;
	std::string fileNameLower = fileName;
	std::transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(), ::tolower);

	if (fileNameLower == "null") {
		// user does not want a binary created
		fid = NULL;
	} else {
		fid = fopen(fileName.c_str(),"wb");
		if (fid==NULL) {
			// default case: print error and exit
			std::string fileError = " Double-check file permissions and make sure directory exists.";
			UserErrors::assertTrue(false, UserErrors::FILE_CANNOT_OPEN, funcName, fileName, fileError);
		}
	}

	// tell new file id to core object
	spikeMonitorCorePtr_->setSpikeFileId(fid);
}