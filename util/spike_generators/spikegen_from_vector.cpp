#include <spikegen_from_vector.h>

SpikeGeneratorFromVector::SpikeGeneratorFromVector(std::vector<int> spkTimes) {
	spkTimes_ = spkTimes;
	size_ = spkTimes.size();
	currentIndex_ = 0;
}

unsigned int SpikeGeneratorFromVector::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime) {

	return (currentIndex_<size_) ? spkTimes_[currentIndex_++] : 0;
}
