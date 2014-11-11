#include <connection_monitor.h>
#include <connection_monitor_core.h>	// ConnectionMonitor private implementation

#include <user_errors.h>		// fancy user error messages
#include <sstream>				// std::stringstream


// we aren't using namespace std so pay attention!
ConnectionMonitor::ConnectionMonitor(ConnectionMonitorCore* connMonCorePtr){
	// make sure the pointer is NULL
	connMonCorePtr_ = connMonCorePtr;
}

ConnectionMonitor::~ConnectionMonitor() {
	delete connMonCorePtr_;
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

std::vector< std::vector<double> > ConnectionMonitor::calcWeightChanges() {
	return connMonCorePtr_->calcWeightChanges();
}

short int ConnectionMonitor::getConnectId() {
	return connMonCorePtr_->getConnectId();
}

int ConnectionMonitor::getFanIn(int neurPostId) {
	return connMonCorePtr_->getFanIn(neurPostId);
}

int ConnectionMonitor::getFanOut(int neurPreId) {
	return connMonCorePtr_->getFanOut(neurPreId);
}

int ConnectionMonitor::getNumWeightsChanged(double minAbsChanged) {
	return connMonCorePtr_->getNumWeightsChanged(minAbsChanged);
}

double ConnectionMonitor::getPercentWeightsChanged(double minAbsChanged) {
	return connMonCorePtr_->getPercentWeightsChanged(minAbsChanged);
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
	connMonCorePtr_->printSparse(neurPostId,maxConn,connPerLine);
}


std::vector< std::vector<double> > ConnectionMonitor::takeSnapshot() {
	return connMonCorePtr_->takeSnapshot();
}