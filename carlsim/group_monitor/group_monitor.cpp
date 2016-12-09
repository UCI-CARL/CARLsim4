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
#include <group_monitor.h>

#include <group_monitor_core.h>	// GroupMonitor private implementation
#include <user_errors.h>		// fancy user error messages

#include <sstream>				// std::stringstream

// we aren't using namespace std so pay attention!
GroupMonitor::GroupMonitor(GroupMonitorCore* groupMonitorCorePtr){
	// make sure the pointer is NULL
	groupMonitorCorePtr_ = groupMonitorCorePtr;
}

GroupMonitor::~GroupMonitor() {
	delete groupMonitorCorePtr_;
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

bool GroupMonitor::isRecording(){
	return groupMonitorCorePtr_->isRecording();
}

void GroupMonitor::startRecording() {
	std::string funcName = "startRecording()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	groupMonitorCorePtr_->startRecording();
}

void GroupMonitor::stopRecording(){
	std::string funcName = "stopRecording()";
	UserErrors::assertTrue(isRecording(), UserErrors::MUST_BE_ON, funcName, "Recording");

	groupMonitorCorePtr_->stopRecording();
}

int GroupMonitor::getRecordingTotalTime() {
	std::string funcName = "getRecordingTotalTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingTotalTime();
}

int GroupMonitor::getRecordingLastStartTime() {
	std::string funcName = "getRecordingLastStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingLastStartTime();
}

int GroupMonitor::getRecordingStartTime() {
	std::string funcName = "getRecordingStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingStartTime();
}

int GroupMonitor::getRecordingStopTime() {
	std::string funcName = "getRecordingStopTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingStopTime();
}

bool GroupMonitor::getPersistentData() {
	return groupMonitorCorePtr_->getPersistentData();
}

void GroupMonitor::setPersistentData(bool persistentData) {
	groupMonitorCorePtr_->setPersistentData(persistentData);
}

std::vector<float> GroupMonitor::getDataVector(){
	std::string funcName = "getDataVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getDataVector();
}

std::vector<int> GroupMonitor::getTimeVector(){
	std::string funcName = "getTimeVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getTimeVector();
}

std::vector<int> GroupMonitor::getPeakTimeVector() {
	std::string funcName = "getPeakTimeVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getPeakTimeVector();
}

std::vector<int> GroupMonitor::getSortedPeakTimeVector() {
	std::string funcName = "getSortedPeakTimeVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getSortedPeakTimeVector();
}

std::vector<float> GroupMonitor::getPeakValueVector() {
	std::string funcName = "getPeakValueVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getPeakValueVector();
}

std::vector<float> GroupMonitor::getSortedPeakValueVector() {
	std::string funcName = "getSortedPeakValueVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getSortedPeakValueVector();
}
