#include "simple_weight_tuner.h"

#include <carlsim.h>	// CARLsim, SpikeMonitor
#include <math.h>		// fabs
#include <stdio.h>		// printf

SimpleWeightTuner::SimpleWeightTuner(CARLsim *sim, short int connId, float errorMargin, float initWt) {
	sim_ = sim;
	errorMargin_ = errorMargin;
	connId_ = connId;
	isDone_ = false;
	wtStepSize_ = 0.1;
	wtShouldIncrease_ = true;

	grpId_ = -1;
	targetRate_ = -1;

	wtTrack_ = initWt;

}


SimpleWeightTuner::~SimpleWeightTuner() {
	
}

bool SimpleWeightTuner::done() {
	return isDone_;
}

void SimpleWeightTuner::setTargetFiringRate(int grpId, float targetRate) {
	grpId_ = grpId;
	targetRate_ = targetRate;
	currentError_ = targetRate;

	SM_ = sim_->setSpikeMonitor(grpId,"NULL");
}

void SimpleWeightTuner::iterate(int runDurationMs) {
	if (done())
		return;

	// else iterate
	SM_->startRecording();
	sim_->runNetwork(runDurationMs/1000, runDurationMs%1000, false);
	SM_->stopRecording();

	float thisRate = SM_->getPopMeanFiringRate();

	printf("rate=%f, target=%f, weight=%f, stepSize=%f\n", thisRate, targetRate_, wtTrack_, wtStepSize_);

	if (fabs(thisRate - targetRate_) <= errorMargin_) {
		isDone_ = true;
		return;
	}

	if (wtStepSize_>0 && thisRate>targetRate_ || wtStepSize_<0 && thisRate<targetRate_) {
		// we stepped too far to the right or too far to the left
		// turn around and cut step size in half
		wtStepSize_ = -wtStepSize_/2.0f;
	}

	// find new weight
	sim_->biasWeights(connId_, wtStepSize_);
	wtTrack_ += wtStepSize_;

	printf("new step size=%f, new weight=%f\n",wtStepSize_,wtTrack_);
}