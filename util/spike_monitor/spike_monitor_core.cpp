#include <spike_monitor.h>
#include <snn.h>
#include <iostream>

// we aren't using namespace std so pay attention!
SpikeMonitorCore::SpikeMonitorCore(){
	vectorSize_ = spkVector_.size();
	recordSet_ = false;
	startTime_ = -1;
	endTime_ = -1;
	grpId_ = -1;
	totalTime_ = -1;
	accumTime_ = -1;
	numN_ = -1;
	return;
}

void SpikeMonitorCore::init(CpuSNN* snn, int grpId){
	// now we have a reference to the current CpuSNN object
	snn_ = snn;
	grpId_= grpId;
	numN_ = snn_->getGroupNumNeurons(grpId_);
	firingRate_.assign(numN_,0);
	tmpSpikeCount_.assign(numN_,0);
	return;
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
	return;
}

float SpikeMonitorCore::getGrpFiringRate(){
	
	vectorSize_ = spkVector_.size();
	// case where we are done recording (easy case)
	if(!recordSet_ && totalTime_>0)
		return (float)vectorSize_/((float)totalTime_*(float)numN_);
	else{
		printf("You have to stop recordiing before using this function.\n");
		exit(1);
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
		std::vector<AER>::const_iterator it;
		int tmpNid;
		for(it=it_begin_;it!=it_end_;it++){
			tmpSpikeCount_[(*it).nid]++;
		}
		for(int i=0;i<numN_;i++){
			firingRate_[i]=(float)tmpSpikeCount_[i]/(float)totalTime_;
		}
		return firingRate_;
	}
	else{
		printf("You have to stop recording before using this function.\n");
		exit(1);
	}
}
// need to do error check on interface
int SpikeMonitorCore::getNumNeuronsWithFiringRate(float min, float max){
	this->getSortedNeuronFiringRate();
	std::vector<float>::const_iterator it;
	int counter = 0;
	for(it=sortedFiringRate_.begin();it!=sortedFiringRate_.end();it++){
		if((*it) >= min && (*it) <= max)
			counter++;
	}
	return counter;
}

int SpikeMonitorCore::getNumSilentNeurons(){
	int numSilent = this->getNumNeuronsWithFiringRate(0,0);
	return numSilent;
}

// need to do error check on interface
float SpikeMonitorCore::getPercentNeuronsWithFiringRate(float min, float max){
	this->getSortedNeuronFiringRate();
	std::vector<float>::const_iterator it;
	int counter = 0;
	for(it=sortedFiringRate_.begin();it!=sortedFiringRate_.end();it++){
		if((*it) >= min && (*it) <= max)
			counter++;
	}
	return (float)counter/numN_;
}

float SpikeMonitorCore::getPercentSilentNeurons(){
	float numSilent = this->getNumNeuronsWithFiringRate(0,0);
	
	return numSilent/numN_;
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
		std::vector<AER>::const_iterator it;
		int tmpNid;
		for(it=it_begin_;it!=it_end_;it++){
			tmpSpikeCount_[(*it).nid]++;
		}
		for(int i=0;i<numN_;i++){
			firingRate_[i]=(float)tmpSpikeCount_[i]/(float)totalTime_;
		}
		sortedFiringRate_=firingRate_;
		std::sort(sortedFiringRate_.begin(),sortedFiringRate_.end());
		return sortedFiringRate_;
	}else{
		printf("You have to stop recording before using this function.\n");
		exit(1);
	}
}

bool SpikeMonitorCore::isRecording(){
	return recordSet_;
}

void SpikeMonitorCore::print(){
	
	std::cout << "Format: Time (ms) : neuron id\n";
	//use an iterator
	std::vector<AER>::const_iterator it;
	for(it=it_begin_;it!=it_end_;it++){
		std::cout << (*it).time << " : "; 
		std::cout << (*it).nid << std::endl;
		std::cout.flush();
	}
	return;
}

void SpikeMonitorCore::pushAER(int grpId, unsigned int* neurIds, unsigned int* timeCnts, int timeInterval){
	int pos    = 0; // keep track of position in flattened list of neuron IDs
	for (int t=0; t < timeInterval; t++) {
		for(int i=0; i<timeCnts[t];i++,pos++) {
			// timeInterval might be < 1000 at the end of a simulation
			AER aer;
			int time = t + snn_->getSimTime() - timeInterval;
			int id   = neurIds[pos];
			aer.time = time;
			aer.nid = id;
			spkVector_.push_back(aer);
		}
	}
	it_begin_=spkVector_.begin();
	it_end_=spkVector_.end();
	return;
}

void SpikeMonitorCore::startRecording(){
	recordSet_ = true;
	startTime_ = snn_->getSimTimeSec();
	// in case we run this multiple times
	if(totalTime_!=-1)
		accumTime_ = totalTime_;
	else
		accumTime_ = 0;
	return;
}

void SpikeMonitorCore::stopRecording(){
	recordSet_ = false;
	endTime_ = snn_->getSimTimeSec();
	totalTime_ = endTime_ - startTime_ + accumTime_;
	return;
}

void SpikeMonitorCore::setSpikeMonGrpId(unsigned int spikeMonGrpId){
	spikeMonGrpId_=spikeMonGrpId;
	return;
}

unsigned int SpikeMonitorCore::getSpikeMonGrpId(){
	return spikeMonGrpId_;
}

void SpikeMonitorCore::setMonBufferPos(unsigned int monBufferPos){
	monBufferPos_=monBufferPos;
	return;
}
	
unsigned int SpikeMonitorCore::getMonBufferPos(){
	return monBufferPos_;
}

void SpikeMonitorCore::incMonBufferPos(){
	monBufferPos_++;
	return;
}

void SpikeMonitorCore::setMonBufferSize(unsigned int monBufferSize){
	monBufferSize_=monBufferSize;
	return;
}
	
unsigned int* SpikeMonitorCore::getMonBufferSize(){
	return monBufferSize_;
}
	
void SpikeMonitorCore::setMonBufferFiring(unsigned int* monBufferFiring){
	monBufferFiring_=monBufferFiring;
	return;
}
	
unsigned int SpikeMonitorCore::getMonBufferFiring(){
	return monBufferFiring_;
}

void SpikeMonitorCore::setMonBufferFid(FILE* monBufferFid){
	monBufferFid_=monBufferFid;
	return;
}

FILE* SpikeMonitorCore::getMonBufferFid(){
	return monBufferFid_;
}

void SpikeMonitorCore::setMonBufferTimeCnt(unsigned int* monBufferTimeCnt){
	monBufferTimeCnt_=monBufferTimeCnt;
	return;
}
	
unsigned int* SpikeMonitorCore::getMonBufferTimeCnt(){
	return monBufferTimeCnt_;
}

void SpikeMonitorCore::zeroMonBufferTimeCnt(unsigned int timeSize){
	memset(monBufferTimeCnt_,0,sizeof(int)*(timeSize));
	return;
}
