#include <spike_info.h>
#include <snn.h>
#include <iostream>

// we aren't using namespace std so pay attention!
// we need to pass a reference for the iterators, not just a value.
SpikeInfo::SpikeInfo(){
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

void SpikeInfo::init(CpuSNN* snn, int grpId){
	// now we have a reference to the current CpuSNN object
	snn_ = snn;
	grpId_= grpId;
	numN_ = snn_->getGroupNumNeurons(grpId_);
	printf("init: before\n");
	printf("numN_: %d\n",numN_);
	printf("firingRate_.max_size(): %lu\n",firingRate_.max_size());
	firingRate_.assign(numN_,0);
	printf("firingRate_.size(): %u\n", firingRate_.size());
	tmpSpikeCount_.assign(numN_,0);
	printf("tmpSpikeCount_.size(): %u\n", tmpSpikeCount_.size());
	printf("init: after\n");
	return;
}

SpikeInfo::~SpikeInfo(){}

float SpikeInfo::getGrpFiringRate(){
	
	vectorSize_ = spkVector_.size();
	// case where we are done recording (easy case)
	if(!recordSet_ && totalTime_>0)
		return (float)vectorSize_/((float)totalTime_*(float)numN_);
	else{
		printf("You have to stop recordiing before using this function.\n");
		exit(1);
	}
}

std::vector<float> SpikeInfo::getNeuronFiringRate(){

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
void SpikeInfo::pushAER(int grpId, unsigned int* neurIds, unsigned int* timeCnts, int timeInterval){
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

void SpikeInfo::print(){
	
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

unsigned int SpikeInfo::getSize(){
	return spkVector_.size();	
}

void SpikeInfo::startRecording(){
	recordSet_ = true;
	startTime_ = snn_->getSimTimeSec();
	// in case we run this multiple times
	if(totalTime_!=-1)
		accumTime_ = totalTime_;
	else
		accumTime_ = 0;
	return;
}

void SpikeInfo::stopRecording(){
	recordSet_ = false;
	endTime_ = snn_->getSimTimeSec();
	totalTime_ = endTime_ - startTime_ + accumTime_;
	return;
}

bool SpikeInfo::isRecording(){
	return recordSet_;
}

std::vector<AER> SpikeInfo::getVector(){
	return spkVector_;
}

void SpikeInfo::clear(){
	spkVector_.clear();
	accumTime_ = 0;
	totalTime_ = 0;
	firingRate_.clear();
	tmpSpikeCount_.clear();
	firingRate_.assign(numN_,0);
	tmpSpikeCount_.assign(numN_,0);
	return;
}
