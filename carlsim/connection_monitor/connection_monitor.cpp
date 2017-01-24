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
#include <connection_monitor.h>
#include <connection_monitor_core.h>	// ConnectionMonitor private implementation

#include <user_errors.h>				// fancy user error messages
#include <sstream>						// std::stringstream


ConnectionMonitor::ConnectionMonitor(ConnectionMonitorCore* connMonCorePtr){
	connMonCorePtr_ = connMonCorePtr;
}

ConnectionMonitor::~ConnectionMonitor() {
	delete connMonCorePtr_;
}


// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

std::vector< std::vector<float> > ConnectionMonitor::calcWeightChanges() {
	return connMonCorePtr_->calcWeightChanges();
}

short int ConnectionMonitor::getConnectId() {
	return connMonCorePtr_->getConnectId();
}

int ConnectionMonitor::getFanIn(int neurPostId) {
	std::string funcName = "getFanIn()";
	UserErrors::assertTrue(neurPostId<getNumNeuronsPost(), UserErrors::MUST_BE_SMALLER, funcName, "neurPostId", 
		"getNumNeuronsPost()");
	return connMonCorePtr_->getFanIn(neurPostId);
}

int ConnectionMonitor::getFanOut(int neurPreId) {
	std::string funcName = "getFanOut()";
	UserErrors::assertTrue(neurPreId<getNumNeuronsPre(), UserErrors::MUST_BE_SMALLER, funcName, "neurPreId", 
		"getNumNeuronsPre()");
	return connMonCorePtr_->getFanOut(neurPreId);
}

int ConnectionMonitor::getNumNeuronsPre() {
	return connMonCorePtr_->getNumNeuronsPre();
}

int ConnectionMonitor::getNumNeuronsPost() {
	return connMonCorePtr_->getNumNeuronsPost();
}

double ConnectionMonitor::getMaxWeight(bool getCurrent) {
	return (double) (connMonCorePtr_->getMaxWeight(getCurrent));
}

double ConnectionMonitor::getMinWeight(bool getCurrent) {
	return (double) (connMonCorePtr_->getMinWeight(getCurrent));
}

int ConnectionMonitor::getNumSynapses() {
	return connMonCorePtr_->getNumSynapses();
}

int ConnectionMonitor::getNumWeightsChanged(double minAbsChanged) {
	std::string funcName = "getNumWeightsChanged()";
	UserErrors::assertTrue(minAbsChanged>=0.0, UserErrors::CANNOT_BE_NEGATIVE, funcName, "minAbsChanged");
	return connMonCorePtr_->getNumWeightsChanged(minAbsChanged);
}

int ConnectionMonitor::getNumWeightsInRange(double minValue, double maxValue) {
	std::string funcName = "getNumWeightsInRange()";
	UserErrors::assertTrue(maxValue >= minValue, UserErrors::CANNOT_BE_SMALLER, funcName, "maxValue", "minValue");
	return connMonCorePtr_->getNumWeightsInRange(minValue,maxValue);
}

int ConnectionMonitor::getNumWeightsWithValue(double value) {
	std::string funcName = "getNumWeightsWithValue()";
	return connMonCorePtr_->getNumWeightsWithValue(value);
}

double ConnectionMonitor::getPercentWeightsChanged(double minAbsChanged) {
	return getNumWeightsChanged(minAbsChanged)*100.0/getNumSynapses();
}

double ConnectionMonitor::getPercentWeightsInRange(double minVal, double maxVal) {
	return getNumWeightsInRange(minVal,maxVal)*100.0/getNumSynapses();
}

double ConnectionMonitor::getPercentWeightsWithValue(double value) {
	return getNumWeightsWithValue(value)*100.0/getNumSynapses();
}

long int ConnectionMonitor::getTimeMsCurrentSnapshot() {
	return connMonCorePtr_->getTimeMsCurrentSnapshot();
}

long int ConnectionMonitor::getTimeMsLastSnapshot() {
	return connMonCorePtr_->getTimeMsLastSnapshot();
}

long int ConnectionMonitor::getTimeMsSinceLastSnapshot() {
	return connMonCorePtr_->getTimeMsSinceLastSnapshot();
}

double ConnectionMonitor::getTotalAbsWeightChange() {
	return connMonCorePtr_->getTotalAbsWeightChange();
}

void ConnectionMonitor::print() {
	connMonCorePtr_->print();
}

void ConnectionMonitor::printSparse(int neurPostId, int maxConn, int connPerLine) {
	std::string funcName = "printSparse()";
	UserErrors::assertTrue(neurPostId<getNumNeuronsPost(), UserErrors::MUST_BE_SMALLER, funcName, "neurPostId", 
		"getNumNeuronsPost()");
	UserErrors::assertTrue(maxConn>0, UserErrors::MUST_BE_POSITIVE, funcName, "maxConn");
	UserErrors::assertTrue(connPerLine>0, UserErrors::MUST_BE_POSITIVE, funcName, "connPerLine");
	connMonCorePtr_->printSparse(neurPostId,maxConn,connPerLine);
}

void ConnectionMonitor::setUpdateTimeIntervalSec(int intervalSec) {
	std::string funcName = "setUpdateTimeIntervalSec()";
	UserErrors::assertTrue(intervalSec==-1 || intervalSec>=1, UserErrors::MUST_BE_SET_TO, funcName, "intervalSec",
		"-1 or >= 1.");
	connMonCorePtr_->setUpdateTimeIntervalSec(intervalSec);
}

std::vector< std::vector<float> > ConnectionMonitor::takeSnapshot() {
	return connMonCorePtr_->takeSnapshot();
}