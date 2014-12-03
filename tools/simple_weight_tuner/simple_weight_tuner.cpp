#include "simple_weight_tuner.h"

#include <carlsim.h>              // CARLsim, SpikeMonitor
#include <math.h>                 // fabs
#include <stdio.h>                // printf
#include <limits>                 // double::max
#include <assert.h>               // assert

// ****************************************************************************************************************** //
// CONSTRUCTOR / DESTRUCTOR
// ****************************************************************************************************************** //

SimpleWeightTuner::SimpleWeightTuner(CARLsim *sim, double errorMargin, int maxIter, double stepSizeFraction) {
	assert(sim!=NULL);
	assert(errorMargin>0);
	assert(maxIter>0);
	assert(stepSizeFraction>0.0f && stepSizeFraction<=1.0f);

	sim_ = sim;
	errorMargin_ = errorMargin;
	stepSizeFraction_ = stepSizeFraction;
	maxIter_ = maxIter;


	connId_ = -1;
	wtRange_ = NULL;
	wtInit_ = -1.0;

	grpId_ = -1;
	targetRate_ = -1.0;

	wtStepSize_ = -1.0;
	cntIter_ = 0;

	wtShouldIncrease_ = true;
	adjustRange_ = true;

	needToInitConnection_ = true;
	needToInitTargetFiring_ = true;

	needToInitAlgo_ = true;
}

SimpleWeightTuner::~SimpleWeightTuner() {
	if (wtRange_!=NULL)
		delete wtRange_;
	wtRange_=NULL;
}



// ****************************************************************************************************************** //
// PUBLIC METHODS
// ****************************************************************************************************************** //

// user function to reset algo
void SimpleWeightTuner::reset() {
	needToInitAlgo_ = true;
	initAlgo();
}

bool SimpleWeightTuner::done(bool printMessage) {
	// algo not initalized: we're not done
	if (needToInitConnection_ || needToInitTargetFiring_ || needToInitAlgo_)
		return false;

	// success: margin reached
	if (fabs(currentError_) < errorMargin_) {
		if (printMessage) {
			printf("SimpleWeightTuner successful: Error margin reached.\n");
		}
		return true;
	}

	// failure: max iter reached
	if (cntIter_ >= maxIter_) {
		if (printMessage) {
			printf("SimpleWeightTuner failed: Max number of iterations reached.\n");
		}
		return true;
	}

	// else we're not done
	return false;
}

void SimpleWeightTuner::setConnectionToTune(short int connId, double initWt, bool adjustRange) {
	assert(connId>=0 && connId<sim_->getNumConnections());

	connId_ = connId;
	wtInit_ = initWt;
	adjustRange_ = adjustRange;

	needToInitConnection_ = false;
	needToInitAlgo_ = true;
}

void SimpleWeightTuner::setTargetFiringRate(int grpId, double targetRate) {
	grpId_ = grpId;
	targetRate_ = targetRate;
	currentError_ = targetRate;

	// check whether group has SpikeMonitor
	SM_ = sim_->getSpikeMonitor(grpId);
	if (SM_==NULL) {
		// setSpikeMonitor has not been called yet
		SM_ = sim_->setSpikeMonitor(grpId,"NULL");
	}

	needToInitTargetFiring_ = false;
	needToInitAlgo_ = true;
}

void SimpleWeightTuner::iterate(int runDurationMs, bool printStatus) {
	assert(runDurationMs>0);

	// if we're done, don't iterate
	if (done(printStatus)) {
		return;
	}

	// make sure we have initialized algo
	assert(!needToInitConnection_);
	assert(!needToInitTargetFiring_);
	if (needToInitAlgo_)
		initAlgo();

	// else iterate
	SM_->startRecording();
	sim_->runNetwork(runDurationMs/1000, runDurationMs%1000, false);
	SM_->stopRecording();

	double thisRate = SM_->getPopMeanFiringRate();

	if (printStatus) {
		printf("#%d: rate=%.4fHz, target=%.4fHz, error=%.4f, errorMargin=%.4f\n", cntIter_, thisRate, targetRate_, thisRate-targetRate_, errorMargin_);
	}

	currentError_ = thisRate - targetRate_;
	cntIter_++;

	// check if we're done now
	if (done(printStatus)) {
		return;
	}

	// else update parameters
	if (wtStepSize_>0 && thisRate>targetRate_ || wtStepSize_<0 && thisRate<targetRate_) {
		// we stepped too far to the right or too far to the left
		// turn around and cut step size in half
		wtStepSize_ = -wtStepSize_/2.0;
	}

	// find new weight
	sim_->biasWeights(connId_, wtStepSize_);
}


// ****************************************************************************************************************** //
// PRIVATE METHODS
// ****************************************************************************************************************** //

// need to call this whenever connection or target firing changes
// or when user calls reset
void SimpleWeightTuner::initAlgo() {
	if (!needToInitAlgo_)
		return;

	// make sure we have all the data structures we need
	assert(!needToInitConnection_);
	assert(!needToInitTargetFiring_);

	// update weight ranges
	RangeWeight wt = sim_->getWeightRange(connId_);
	wtRange_ = new RangeWeight(wt.min, wt.init, wt.max);

	// reset algo
	wtShouldIncrease_ = true;
	wtStepSize_ = stepSizeFraction_ * (wtRange_->max - wtRange_->min);
	currentError_ = std::numeric_limits<double>::max();

	// initialize weights
	if (wtInit_>=0) {
		// start at some specified initWt
		if (wt.init != wtInit_) {
			// specified starting point is not what is specified in connect

			// make sure we're in the right CARLsim state
			if (sim_->getCARLsimState()!=EXE_STATE)
				sim_->runNetwork(0,0);

			sim_->biasWeights(connId_, wtInit_ - wt.init, adjustRange_);
		}
	}

	needToInitAlgo_ = false;
}