#include <spike_monitor_core.h>

#include <snn.h>
#include <snn_definitions.h>		// CARLSIM_ERROR, CARLSIM_INFO, ...

#include <iostream>		// std::cout, std::endl



// we aren't using namespace std so pay attention!
SpikeMonitorCore::SpikeMonitorCore(CpuSNN* snn, int monitorId, int grpId) {
	snn_ = snn;
	grpId_= grpId;
	monitorId_ = monitorId;
	numN_ = -1;
	spikeFileId_ = NULL;
	persistentData_ = false;

	// defer all unsafe operations to init function
	init();
}

void SpikeMonitorCore::init() {
	numN_ = snn_->getGroupNumNeurons(grpId_);
	assert(numN_>0);

	clear();

	// use CARLSIM_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();
}

SpikeMonitorCore::~SpikeMonitorCore() {
	if (spikeFileId_!=NULL) {
		fclose(spikeFileId_);
		spikeFileId_ = NULL;
	}
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void SpikeMonitorCore::clear() {
	recordSet_ = false;
	spkVector_.clear();
	startTime_ = -1;
	startTimeLast_ = -1;
	stopTime_ = -1;
	accumTime_ = 0;
	totalTime_ = -1;

	needToCalculateFiringRates_ = true;
	needToSortFiringRates_ = true;
	firingRate_.clear();
	firingRateSorted_.clear();
	tmpSpikeCount_.clear();
	firingRate_.assign(numN_,0);
	firingRateSorted_.assign(numN_,0);
	tmpSpikeCount_.assign(numN_,0);
}

float SpikeMonitorCore::getGroupFiringRate() {
	assert(!isRecording());

	if (totalTime_==0)
		return 0.0f;

	return spkVector_.size()*1000.0/(totalTime_*numN_);
}

float SpikeMonitorCore::getNeuronMaxFiringRate() {
	assert(!isRecording());

	// if necessary, get data structures up-to-date
	sortFiringRates();

	return firingRateSorted_.back();
}

float SpikeMonitorCore::getNeuronMinFiringRate(){
	assert(!isRecording());

	// if necessary, get data structures up-to-date
	sortFiringRates();

	return firingRateSorted_.front();
}

std::vector<float> SpikeMonitorCore::getNeuronFiringRate() {
	assert(!isRecording());

	// if necessary, get data structures up-to-date
	calculateFiringRates();

	return firingRate_;
}

int SpikeMonitorCore::getNumNeuronsWithFiringRate(float min, float max){
	assert(!isRecording());
	assert(min>=0.0f && max>=0.0f);
	assert(max>=min);

	// if necessary, get data structures up-to-date
	sortFiringRates();

	int counter = 0;
	std::vector<float>::const_iterator it_begin = firingRateSorted_.begin();
	std::vector<float>::const_iterator it_end = firingRateSorted_.end();
	for(std::vector<float>::const_iterator it=it_begin; it!=it_end; it++){
		if((*it) >= min && (*it) <= max)
			counter++;
	}

	return counter;
}

int SpikeMonitorCore::getNumSilentNeurons() {
	assert(!isRecording());

	return getNumNeuronsWithFiringRate(0.0f, 0.0f);
}

// \TODO need to do error check on interface
float SpikeMonitorCore::getPercentNeuronsWithFiringRate(float min, float max) {
	assert(!isRecording());

	return getNumNeuronsWithFiringRate(min,max)*100.0/numN_;
}

float SpikeMonitorCore::getPercentSilentNeurons(){
	assert(!isRecording());

	return getNumNeuronsWithFiringRate(0,0)*100.0/numN_;
}

long int SpikeMonitorCore::getSize(){
	assert(!isRecording());

	return spkVector_.size();	
}

std::vector<AER> SpikeMonitorCore::getVector(){
	assert(!isRecording());

	return spkVector_;
}

std::vector<float> SpikeMonitorCore::getNeuronSortedFiringRate() {
	assert(!isRecording());

	// if necessary, get data structures up-to-date
	sortFiringRates();

	return firingRateSorted_;
}

void SpikeMonitorCore::print() {
	assert(!isRecording());

	std::cout << "Format: Time (ms) : neuron id\n";
	std::vector<AER>::const_iterator it_begin = spkVector_.begin();
	std::vector<AER>::const_iterator it_end = spkVector_.end();
	for(std::vector<AER>::const_iterator it=it_begin; it!=it_end; it++){
		std::cout << it->time << " : "; 
		std::cout << it->nid << std::endl;
		std::cout.flush();
	}
}

void SpikeMonitorCore::pushAER(AER aer) {
	assert(isRecording());
	spkVector_.push_back(aer);
}

void SpikeMonitorCore::startRecording() {
	assert(!isRecording());

	if (!persistentData_) {
		// if persistent mode is off (default behavior), automatically call clear() here
		clear();
	}

	// call updateSpikeMonitor to make sure spike file and spike vector are up-to-date
	// Caution: must be called before recordSet_ is set to true!
	snn_->updateSpikeMonitor(grpId_);

	needToUpdateFiringRate_ = true;
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

void SpikeMonitorCore::stopRecording() {
	assert(isRecording());
	assert(startTime_>-1 && startTimeLast_>-1 && accumTime_>-1);

	// call updateSpikeMonitor to make sure spike file and spike vector are up-to-date
	// Caution: must be called before recordSet_ is set to false!
	snn_->updateSpikeMonitor(grpId_);

	recordSet_ = false;
	stopTime_ = snn_->getSimTimeSec()*1000+snn_->getSimTimeMs();

	// total time is the amount of time of the last probe plus all accumulated time from previous probes
	totalTime_ = stopTime_-startTimeLast_ + accumTime_;
	assert(totalTime_>=0);
}

void SpikeMonitorCore::setPersistentMode(bool persistentData) {
	persistentData_ = persistentData;
}

void SpikeMonitorCore::setSpikeFileId(FILE* spikeFileId) {
	assert(!isRecording());

	spikeFileId_=spikeFileId;
}


// calculate average firing rate for every neuron if we haven't done so already
void SpikeMonitorCore::calculateFiringRates() {
	// only update if we have to
	if (!needToCalculateFiringRates_)
		return;

	// clear, so we get the same answer every time.
	tmpSpikeCount_.assign(numN_,0);
	firingRate_.assign(numN_,0);

	// this really shouldn't happen at this stage, but if recording time is zero, return all zeros
	if (totalTime_==0) {
		CARLSIM_WARN("SpikeMonitorCore:: calculateFiringRates has 0 totalTime");
		return;
	}

	// read all AER events and assign them to neuron IDs to get # spikes per neuron
	std::vector<AER>::const_iterator it_begin = spkVector_.begin();
	std::vector<AER>::const_iterator it_end = spkVector_.end();
	for(std::vector<AER>::const_iterator it=it_begin; it!=it_end; it++) {
		assert(it->nid >=0 && it->nid < numN_);
		tmpSpikeCount_[it->nid]++;
	}

	// compute firing rate
	assert(totalTime_>0); // avoid division by zero
	for(int i=0;i<numN_;i++) {
		firingRate_[i]=tmpSpikeCount_[i]*1000.0/totalTime_;
	}

	needToCalculateFiringRates_ = false;
}

// sort firing rates if we haven't done so already
void SpikeMonitorCore::sortFiringRates() {
	// only sort if we have to
	if (!needToSortFiringRates_)
		return;

	// first make sure firing rate vector is up-to-date
	calculateFiringRates();

	firingRateSorted_=firingRate_;
	std::sort(firingRateSorted_.begin(),firingRateSorted_.end());

	needToSortFiringRates_ = false;
}