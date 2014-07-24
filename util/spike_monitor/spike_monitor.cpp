#include <snn.h>
#include <iostream>
#include <spike_monitor.h>
#include <spike_monitor_core.h>

// we aren't using namespace std so pay attention!
SpikeMonitor::SpikeMonitor(SpikeMonitorCore* spikeMonitorCorePtr){
	// make sure the pointer is NULL
	spikeMonitorCorePtr_ = spikeMonitorCorePtr;
}

SpikeMonitor::~SpikeMonitor(){
	// Should we delete the new SpikeMonitorCore object here?
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void SpikeMonitor::clear(){
	spikeMonitorCorePtr_->clear();
	return;
}

float SpikeMonitor::getGrpFiringRate() {
	return spikeMonitorCorePtr_->getGrpFiringRate();
}

float SpikeMonitor::getMaxNeuronFiringRate(){
	return spikeMonitorCorePtr_->getMaxNeuronFiringRate();
}

float SpikeMonitor::getMinNeuronFiringRate(){
	return spikeMonitorCorePtr_->getMinNeuronFiringRate();
}

std::vector<float> SpikeMonitor::getNeuronFiringRate(){
	return spikeMonitorCorePtr_->getNeuronFiringRate();
}

// need to do error check here and maybe throw CARLsim errors.
int SpikeMonitor::getNumNeuronsWithFiringRate(float min, float max){
	return spikeMonitorCorePtr_->getNumNeuronsWithFiringRate(min,max);
}

// need to do error check here and maybe throw CARLsim errors.
float SpikeMonitor::getPercentNeuronsWithFiringRate(float min, float max){
	return spikeMonitorCorePtr_->getPercentNeuronsWithFiringRate(min,max);
}

int SpikeMonitor::getNumSilentNeurons(){
	return spikeMonitorCorePtr_->getNumSilentNeurons();
}

float SpikeMonitor::getPercentSilentNeurons(){
	return spikeMonitorCorePtr_->getPercentSilentNeurons();
}

unsigned int SpikeMonitor::getSize(){
	return spikeMonitorCorePtr_->getSize();
}

std::vector<AER> SpikeMonitor::getVector(){
	return spikeMonitorCorePtr_->getVector();
}

std::vector<float> SpikeMonitor::getSortedNeuronFiringRate(){
	return spikeMonitorCorePtr_->getSortedNeuronFiringRate();
}

bool SpikeMonitor::isRecording(){
	return spikeMonitorCorePtr_->isRecording();
}

void SpikeMonitor::print(){
	return spikeMonitorCorePtr_->print();
}

void SpikeMonitor::startRecording(){
	return spikeMonitorCorePtr_->startRecording();
}

void SpikeMonitor::stopRecording(){
	return spikeMonitorCorePtr_->stopRecording();
}

int SpikeMonitor::getRecordingTotalTime() { return spikeMonitorCorePtr_->getRecordingTotalTime(); }
int SpikeMonitor::getRecordingStartTime() { return spikeMonitorCorePtr_->getRecordingStartTime(); }
int SpikeMonitor::getRecordingStopTime() { return spikeMonitorCorePtr_->getRecordingStopTime(); }