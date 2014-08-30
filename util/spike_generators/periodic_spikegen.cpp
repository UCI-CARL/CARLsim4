#include <periodic_spikegen.h>

#include <user_errors.h>	// fancy error messages
#include <algorithm>		// std::find
#include <vector>			// std::vector
#include <cassert>			// assert

PeriodicSpikeGenerator::PeriodicSpikeGenerator(float rate, bool spikeAtZero) {
	assert(rate>0);
	rate_ = rate;	  // spike rate
	isi_ = 1000/rate; // inter-spike interval in ms
	spikeAtZero_ = spikeAtZero;

	checkFiringRate();
}

unsigned int PeriodicSpikeGenerator::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime) {
//		fprintf(stderr,"currentTime: %u lastScheduled: %u\n",currentTime,lastScheduledSpikeTime);

	if (spikeAtZero_) {
		// insert spike at t=0 for each neuron (keep track of neuron IDs to avoid getting stuck in infinite loop)
		if (std::find(nIdFiredAtZero_.begin(), nIdFiredAtZero_.end(), nid)==nIdFiredAtZero_.end()) {
			// spike at t=0 has not been scheduled yet for this neuron
			nIdFiredAtZero_.push_back(nid);
			return 0;
		}
	}

	// periodic spiking according to ISI
	return lastScheduledSpikeTime+isi_;
}

void PeriodicSpikeGenerator::checkFiringRate() {
	UserErrors::assertTrue(rate_>0, UserErrors::MUST_BE_POSITIVE, "PeriodicSpikeGenerator", "Firing rate");
}