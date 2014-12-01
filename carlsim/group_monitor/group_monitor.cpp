#include <group_monitor.h>

#include <group_monitor_core.h>	// GroupMonitor private implementation
#include <user_errors.h>		// fancy user error messages

#include <sstream>				// std::stringstream

// we aren't using namespace std so pay attention!
GroupMonitor::GroupMonitor(GroupMonitorCore* spikeMonitorCorePtr){
	// make sure the pointer is NULL
	groupMonitorCorePtr_ = spikeMonitorCorePtr;
}

GroupMonitor::~GroupMonitor() {
	delete groupMonitorCorePtr_;
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void GroupMonitor::clear(){
	std::string funcName = "clear()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	groupMonitorCorePtr_->clear();
}

bool GroupMonitor::isRecording(){
	return groupMonitorCorePtr_->isRecording();
}

void GroupMonitor::print() {
	std::string funcName = "print()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	groupMonitorCorePtr_->print();
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

unsigned int GroupMonitor::getRecordingTotalTime() {
	std::string funcName = "getRecordingTotalTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingTotalTime();
}

unsigned int GroupMonitor::getRecordingLastStartTime() {
	std::string funcName = "getRecordingLastStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingLastStartTime();
}

unsigned int GroupMonitor::getRecordingStartTime() {
	std::string funcName = "getRecordingStartTime()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return groupMonitorCorePtr_->getRecordingStartTime();
}

unsigned int GroupMonitor::getRecordingStopTime() {
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
