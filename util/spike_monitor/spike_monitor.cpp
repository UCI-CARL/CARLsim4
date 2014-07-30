#include <spike_monitor.h>

#include <spike_monitor_core.h>	// SpikeMonitor private implementation
#include <user_errors.h>		// fancy user error messages

#include <iostream>				// std::cout, std::endl
#include <sstream>				// std::stringstream

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

int SpikeMonitor::getPopNumSpikes() {
	std::string funcName = "getPopNumSpikes()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

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

int SpikeMonitor::getNeuronNumSpikes(int neurId) {
	std::string funcName = "getNeuronNumSpikes()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

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

void SpikeMonitor::print() {
	std::string funcName = "print()";
	UserErrors::assertTrue(!isRecording(), UserErrors::CANNOT_BE_ON, funcName, "Recording");

	spikeMonitorCorePtr_->print();
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

bool SpikeMonitor::getPersistentMode() {
	return spikeMonitorCorePtr_->getPersistentMode();
}

void SpikeMonitor::setPersistentMode(bool persistentData) {
	spikeMonitorCorePtr_->setPersistentMode(persistentData);
}