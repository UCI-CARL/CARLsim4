/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/
#include "simple_weight_tuner.h"

#include <carlsim.h>              // CARLsim, SpikeMonitor
#include <math.h>                 // fabs
#include <stdio.h>                // printf
#include <limits>                 // double::max
#include <assert.h>               // assert

// ****************************************************************************************************************** //
// SIMPLEWEIGHTTUNER UTILITY PRIVATE IMPLEMENTATION
// ****************************************************************************************************************** //

/*!
 * \brief Private implementation of the Stopwatch Utility
 *
 * This class provides a timer with milliseconds resolution.
 * \see http://stackoverflow.com/questions/1861294/how-to-calculate-execution-time-of-a-code-snippet-in-c/1861337#1861337
 * \since v3.1
 */
class SimpleWeightTuner::Impl {
public:
	// +++++ PUBLIC METHODS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	Impl(CARLsim *sim, double errorMargin, int maxIter, double stepSizeFraction) {
		assert(sim!=NULL);
		assert(errorMargin>0);
		assert(maxIter>0);
		assert(stepSizeFraction>0.0f && stepSizeFraction<=1.0f);

		sim_ = sim;
		assert(sim_->getCARLsimState()!=RUN_STATE);

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

	~Impl() {
		if (wtRange_!=NULL)
			delete wtRange_;
		wtRange_=NULL;
	}

// user function to reset algo
void reset() {
	needToInitAlgo_ = true;
	initAlgo();
}

bool done(bool printMessage) {
	// algo not initalized: we're not done
	if (needToInitConnection_ || needToInitTargetFiring_ || needToInitAlgo_)
		return false;

	// success: margin reached
	if (fabs(currentError_) < errorMargin_) {
		if (printMessage) {
			printf("SimpleWeightTuner successful: Error margin reached in %d iterations.\n",cntIter_);
		}
		return true;
	}

	// failure: max iter reached
	if (cntIter_ >= maxIter_) {
		if (printMessage) {
			printf("SimpleWeightTuner failed: Max number of iterations (%d) reached.\n",maxIter_);
		}
		return true;
	}

	// else we're not done
	return false;
}

void setConnectionToTune(short int connId, double initWt, bool adjustRange) {
	assert(connId>=0 && connId<sim_->getNumConnections());

	connId_ = connId;
	wtInit_ = initWt;
	adjustRange_ = adjustRange;

	needToInitConnection_ = false;
	needToInitAlgo_ = true;
}

void setTargetFiringRate(int grpId, double targetRate) {
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

void iterate(int runDurationMs, bool printStatus) {
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

	// in case the user has already been messing with the SpikeMonitor, we need to make sure that
	// PersistentMode is off
	SM_->setPersistentData(false);

	// now iterate
	SM_->startRecording();
	sim_->runNetwork(runDurationMs/1000, runDurationMs%1000, false);
	SM_->stopRecording();

	double thisRate = SM_->getPopMeanFiringRate();
	if (printStatus) {
		printf("#%d: rate=%.4fHz, target=%.4fHz, error=%.7f, errorMargin=%.7f\n", cntIter_, thisRate, targetRate_,
			thisRate-targetRate_, errorMargin_);
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
		// note that this should work for inhibitory connections, too: they have negative weights, so adding
		// to the weight will actually decrease it (make it less negative)
		wtStepSize_ = -wtStepSize_/2.0;
	}

	// find new weight
	sim_->biasWeights(connId_, wtStepSize_, adjustRange_);
}

private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

// need to call this whenever connection or target firing changes
// or when user calls reset
void initAlgo() {
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
#if defined(WIN32) || defined(WIN64)
	currentError_ = DBL_MAX;
#else
	currentError_ = std::numeric_limits<double>::max();
#endif

	// make sure we're in the right CARLsim state
	if (sim_->getCARLsimState()!=RUN_STATE)
		sim_->runNetwork(0,0,false);

	// initialize weights
	if (wtInit_>=0) {
		// start at some specified initWt
		if (wt.init != wtInit_) {
			// specified starting point is not what is specified in connect

			sim_->biasWeights(connId_, wtInit_ - wt.init, adjustRange_);
		}
	}

	needToInitAlgo_ = false;
}


	// +++++ PRIVATE STATIC PROPERTIES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// +++++ PRIVATE PROPERTIES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// flags that manage state
	bool needToInitConnection_;     //!< flag indicating whether to initialize connection params
	bool needToInitTargetFiring_;   //!< flag indicating whether to initialize target firing params
	bool needToInitAlgo_;           //!< flag indicating whether to initialize algorithm

	// CARLsim data structures
	CARLsim *sim_;                  //!< pointer to CARLsim object
	SpikeMonitor *SM_;              //!< pointer to SpikeMonitor object
	int grpId_;                     //!< CARLsim group ID
	short int connId_;              //!< CARLsim connection ID
	RangeWeight* wtRange_;          //!< pointer to CARLsim RangeWeight struct

	// termination condition params
	int maxIter_;                   //!< maximum number of iterations (termination condition)
	double errorMargin_;            //!< error margin for firing rate (termination condition)
	double targetRate_;             //!< target firing rate specified in setTargetFiringRate

	// params that are updated every iteration step
	int cntIter_;                   //!< current count of iteration number
	double wtStepSize_;             //!< current weight step size
	bool wtShouldIncrease_;         //!< flag indicating the direction of weight change (increase=true, decrease=false)
	double currentError_;           //!< current firing error

	// options
	bool adjustRange_;              //!< flag indicating whether to update [minWt,maxWt] when weight goes out of bounds
	double wtInit_;                 //!< initial weight specified in setConnectionToTune
	double stepSizeFraction_;       //!< initial weight step size
};


// ****************************************************************************************************************** //
// SIMPLEWEIGHTTUNER API IMPLEMENTATION
// ****************************************************************************************************************** //

// create and destroy a pImpl instance
SimpleWeightTuner::SimpleWeightTuner(CARLsim* sim, double errorMargin, int maxIter, double stepSizeFraction) :
	_impl( new Impl(sim, errorMargin, maxIter, stepSizeFraction) ) {}
SimpleWeightTuner::~SimpleWeightTuner() { delete _impl; }

void SimpleWeightTuner::setConnectionToTune(short int connId, double initWt, bool adjustRange) {
	_impl->setConnectionToTune(connId, initWt, adjustRange);
}
void SimpleWeightTuner::setTargetFiringRate(int grpId, double targetRate) { 
	_impl->setTargetFiringRate(grpId, targetRate);
}
void SimpleWeightTuner::iterate(int runDurationMs, bool printStatus) { _impl->iterate(runDurationMs, printStatus); }
bool SimpleWeightTuner::done(bool printMessage) { return _impl->done(printMessage); }
void SimpleWeightTuner::reset() { _impl->reset(); }
