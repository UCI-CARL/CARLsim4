#include <snn.h>
#include <iostream>
#include <spike_monitor_core.h>
#include <snn_definitions.h>		// CARLSIM_ERROR, CARLSIM_INFO, ...



// we aren't using namespace std so pay attention!
SpikeMonitorCore::SpikeMonitorCore(CpuSNN* snn, int grpId) {
	snn_ = snn;
	grpId_= grpId;

	recordSet_ = false;
	startTime_ = -1;
	endTime_ = -1;
	totalTime_ = -1;
	accumTime_ = -1;
	numN_ = -1;
	lastStopRecTime_ = -1;

	// defer all unsafe operations to init function
	init();
}

void SpikeMonitorCore::init() {
	numN_ = snn_->getGroupNumNeurons(grpId_);
	firingRate_.assign(numN_,0);
	tmpSpikeCount_.assign(numN_,0);

	// use CARLSIM_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();
}

SpikeMonitorCore::~SpikeMonitorCore(){}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void SpikeMonitorCore::clear(){
	spkVector_.clear();
	accumTime_ = 0;
	totalTime_ = 0;
	firingRate_.clear();
	tmpSpikeCount_.clear();
	firingRate_.assign(numN_,0);
	tmpSpikeCount_.assign(numN_,0);
}

float SpikeMonitorCore::getGrpFiringRate() {
	// case where we are done recording (easy case)
	fprintf(stderr, "recordSet=%s, size=%d, totalTime=%d, numN=%d\n",recordSet_?"y":"n",spkVector_.size(), totalTime_, numN_);
	if(!recordSet_ && totalTime_>0) {
		return spkVector_.size()*1000.0/(totalTime_*numN_);
	}
	else{
		CARLSIM_ERROR("You have to stop recording before using getGrpFiringRate.");
//		snn_->exitSimulation();
	}
}

float SpikeMonitorCore::getMaxNeuronFiringRate(){
	this->getSortedNeuronFiringRate();
	return sortedFiringRate_.back();
}

float SpikeMonitorCore::getMinNeuronFiringRate(){
	this->getSortedNeuronFiringRate();
	return sortedFiringRate_.front();
}

std::vector<float> SpikeMonitorCore::getNeuronFiringRate(){
	// clear, so we get the same answer every time.
	tmpSpikeCount_.assign(numN_,0);
	firingRate_.assign(numN_,0);

	if(!recordSet_ && totalTime_>0){
		// calculate average firing rate for every neuron
		int tmpNid;
		for(std::vector<AER>::iterator it=it_begin_; it!=it_end_; it++){
			tmpSpikeCount_[(*it).nid]++;
		}
		for(int i=0;i<numN_;i++){
			firingRate_[i]=tmpSpikeCount_[i]*1000.0/totalTime_;
		}
		return firingRate_;
	}
	else{
		CARLSIM_ERROR("You have to stop recording before using getNeuronFiringRate.\n");
		snn_->exitSimulation();
	}
}
// \TODO need to do error check on interface
int SpikeMonitorCore::getNumNeuronsWithFiringRate(float min, float max){
	this->getSortedNeuronFiringRate();
	int counter = 0;
	for(std::vector<float>::iterator it=sortedFiringRate_.begin(); it!=sortedFiringRate_.end(); it++){
		if((*it) >= min && (*it) <= max)
			counter++;
	}
	return counter;
}

int SpikeMonitorCore::getNumSilentNeurons(){
	int numSilent = this->getNumNeuronsWithFiringRate(0,0);
	return numSilent;
}

// \TODO need to do error check on interface
float SpikeMonitorCore::getPercentNeuronsWithFiringRate(float min, float max) {
	return this->getNumNeuronsWithFiringRate(min,max)*100.0/numN_;
}

float SpikeMonitorCore::getPercentSilentNeurons(){
	return this->getNumNeuronsWithFiringRate(0,0)*100.0/numN_;
}

unsigned int SpikeMonitorCore::getSize(){
	return spkVector_.size();	
}

std::vector<AER> SpikeMonitorCore::getVector(){
	return spkVector_;
}

std::vector<float> SpikeMonitorCore::getSortedNeuronFiringRate(){
	// clear, so we get the same answer every time.
	tmpSpikeCount_.assign(numN_,0);
	firingRate_.assign(numN_,0);
	
	if(!recordSet_ && totalTime_>0){
		// calculate average firing rate for every neuron
		int tmpNid;
		for(std::vector<AER>::iterator it=it_begin_; it!=it_end_; it++){
			tmpSpikeCount_[(*it).nid]++;
		}
		for(int i=0;i<numN_;i++){
			firingRate_[i]=tmpSpikeCount_[i]*1000.0/totalTime_;
		}
		sortedFiringRate_=firingRate_;
		std::sort(sortedFiringRate_.begin(),sortedFiringRate_.end());
		return sortedFiringRate_;
	}else{
		CARLSIM_ERROR("You have to stop recording before using getSortedNeuronFiringRate.");
		snn_->exitSimulation();
	}
}

bool SpikeMonitorCore::isRecording(){
	return recordSet_;
}

void SpikeMonitorCore::print(){
	
	std::cout << "Format: Time (ms) : neuron id\n";
	//use an iterator
	for(std::vector<AER>::iterator it=it_begin_;it!=it_end_;it++){
		std::cout << (*it).time << " : "; 
		std::cout << (*it).nid << std::endl;
		std::cout.flush();
	}
}

// \FIXME: The following code snippet is a near 1-to-1 copy of CpuSNN::writeSpikesToFile. Consider merging for style
// and safety. Otherwise it is dangerously easy to get the two functions out of sync...
void SpikeMonitorCore::pushAER(int grpId, unsigned int* neurIds, unsigned int* timeCnts, int numMsMin, int numMsMax){
	int pos = 0; // keep track of position in flattened list of neuron IDs

	fprintf(stderr,"Use of this function is deprecated. Use SpikeMonitorCore::pushAER(AER aer) instead.");

	// current time is last completed second in milliseconds (plus t to be added below)
	// special case is after each completed second where !getSimTimeMs(): here we look 1s back
	int currentTimeSec = snn_->getSimTimeSec();
	if (!snn_->getSimTimeMs())
		currentTimeSec--;

	for (int t=numMsMin; t<numMsMax; t++) {
		for(int i=0; i<timeCnts[t];i++,pos++) {
			AER aer(currentTimeSec*1000 + t, neurIds[pos]);
			spkVector_.push_back(aer);
		}
	}
	it_begin_=spkVector_.begin();
	it_end_=spkVector_.end();
}

void SpikeMonitorCore::pushAER(AER aer) {
	spkVector_.push_back(aer);
}

void SpikeMonitorCore::startRecording(){
	if (recordSet_) {
		fprintf(stderr,"You have to stop recording first before you can start again.");
		snn_->exitSimulation();
	}

	// call updateSpikeMonitor to make sure spike file and spike vector are up-to-date
	// Caution: must be called before recordSet_ is set to true!
	snn_->updateSpikeMonitor(grpId_);

	recordSet_ = true;
	startTime_ = snn_->getSimTimeSec()*1000+snn_->getSimTimeMs();
	// in case we run this multiple times
	if(totalTime_!=-1)
		accumTime_ = totalTime_;
	else
		accumTime_ = 0;
}

void SpikeMonitorCore::stopRecording() {
	if (!recordSet_) {
		fprintf(stderr,"You have to start recording first before you can stop.");
		snn_->exitSimulation();
	}

	// call updateSpikeMonitor to make sure spike file and spike vector are up-to-date
	// Caution: must be called before recordSet_ is set to false!
	snn_->updateSpikeMonitor(grpId_);

	recordSet_ = false;
	endTime_ = snn_->getSimTimeSec()*1000+snn_->getSimTimeMs();
	totalTime_ = endTime_ - startTime_ + accumTime_;

	// check for overflow
	assert(totalTime_>=0);
	assert(totalTime_ >= endTime_-startTime_);
}

int SpikeMonitorCore::getSpikeMonGrpId() { return spikeMonGrpId_; }

void SpikeMonitorCore::setMonBufferPos(unsigned int monBufferPos){
	monBufferPos_=monBufferPos;
	return;
}
	
unsigned int SpikeMonitorCore::getMonBufferPos() { return monBufferPos_; }

void SpikeMonitorCore::incMonBufferPos() { monBufferPos_++; }

void SpikeMonitorCore::setMonBufferSize(unsigned int monBufferSize) { monBufferSize_=monBufferSize; }
	
unsigned int SpikeMonitorCore::getMonBufferSize() { return monBufferSize_; }
	
void SpikeMonitorCore::setMonBufferFiring(unsigned int* monBufferFiring) { monBufferFiring_=monBufferFiring; }
	
unsigned int* SpikeMonitorCore::getMonBufferFiring() { return monBufferFiring_; }

void SpikeMonitorCore::setMonBufferFid(FILE* monBufferFid) { monBufferFid_=monBufferFid; }

FILE* SpikeMonitorCore::getMonBufferFid() {	return monBufferFid_; }

void SpikeMonitorCore::setMonBufferTimeCnt(unsigned int* monBufferTimeCnt) { monBufferTimeCnt_=monBufferTimeCnt; }
	
unsigned int* SpikeMonitorCore::getMonBufferTimeCnt() { return monBufferTimeCnt_; }

void SpikeMonitorCore::zeroMonBufferTimeCnt(unsigned int timeSize) {
	memset(monBufferTimeCnt_,0,sizeof(int)*(timeSize));
}
