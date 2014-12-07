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
