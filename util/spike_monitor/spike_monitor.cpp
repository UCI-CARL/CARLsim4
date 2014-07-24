#include <spike_monitor.h>

#include <snn.h>
#include <spike_monitor_core.h>
#include <user_errors.h>

#include <iostream>		// std::cout, std::endl
#include <sstream>		// std::stringstream

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
	spikeMonitorCorePtr_->clear();
	return;
}

float SpikeMonitor::getGroupFiringRate() {
	std::string funcName = "getGroupFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getGroupFiringRate();
}

float SpikeMonitor::getNeuronMaxFiringRate(){
	std::string funcName = "getNeuronMaxFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNeuronMaxFiringRate();
}

float SpikeMonitor::getNeuronMinFiringRate(){
	std::string funcName = "getNeuronMinFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNeuronMinFiringRate();
}

std::vector<float> SpikeMonitor::getNeuronFiringRate(){
	std::string funcName = "getNeuronFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNeuronFiringRate();
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

long int SpikeMonitor::getSize() {
	std::string funcName = "getSize()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getSize();
}

std::vector<AER> SpikeMonitor::getVector() {
	std::string funcName = "getVector()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getVector();
}

std::vector<float> SpikeMonitor::getNeuronSortedFiringRate(){
	std::string funcName = "getNeuronSortedFiringRate()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->getNeuronSortedFiringRate();
}

bool SpikeMonitor::isRecording(){
	return spikeMonitorCorePtr_->isRecording();
}

void SpikeMonitor::print() {
	std::string funcName = "print()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->print();
}

void SpikeMonitor::startRecording() {
	std::string funcName = "startRecording()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->startRecording();
}

void SpikeMonitor::stopRecording(){
	std::string funcName = "stopRecording()";
	UserErrors::assertTrue(isRecording(), UserErrors::MUST_BE_ON, funcName, "Recording");

	return spikeMonitorCorePtr_->stopRecording();
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

bool SpikeMonitor::getPersistentMode() { return spikeMonitorCorePtr_->getPersistentMode(); }
void SpikeMonitor::setPersistentMode(bool persistentData) {
	return spikeMonitorCorePtr_->setPersistentMode(persistentData);
}