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
