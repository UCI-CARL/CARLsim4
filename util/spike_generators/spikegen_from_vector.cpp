#include <spikegen_from_vector.h>

#include <user_errors.h>	// fancy error messages
#include <sstream>			// std::stringstream

SpikeGeneratorFromVector::SpikeGeneratorFromVector(std::vector<int> spkTimes) {
	spkTimes_ = spkTimes;
	size_ = spkTimes.size();
	currentIndex_ = 0;

	checkSpikeVector();
}

unsigned int SpikeGeneratorFromVector::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime) {

	return (currentIndex_<size_) ? spkTimes_[currentIndex_++] : 0;
}

void SpikeGeneratorFromVector::checkSpikeVector() {
	UserErrors::assertTrue(size_>0,UserErrors::CANNOT_BE_ZERO, "SpikeGeneratorFromVector", "Vector size");
	for (int i=0; i<size_; i++) {
		std::stringstream var; var << "spkTimes[" << currentIndex_ << "]";
		UserErrors::assertTrue(spkTimes_[i]>0,UserErrors::CANNOT_BE_ZERO, "SpikeGeneratorFromVector", var.str());
	}	
}