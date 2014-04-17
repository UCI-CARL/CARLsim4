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
	return;
}

SpikeInfo::init(CpuSNN* snn){
	// now we have a reference to the current CpuSNN object
	snn_=snn;
	return;
}

SpikeInfo::~SpikeInfo(){}

float SpikeInfo::getGrpFiringRate(int timeDuration, int sizeN){
	
	return (float)vectorSize_/((float)timeDuration*(float)sizeN);
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
	startTime_ = 
	return;
}

void SpikeInfo::stopRecording(){
	recordSet_ = false;
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
	return;
}
