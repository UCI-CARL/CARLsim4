#include <connection_monitor_core.h>

#include <snn.h>				// CARLsim private implementation
#include <snn_definitions.h>	// KERNEL_ERROR, KERNEL_INFO, ...

#include <algorithm>			// std::sort



// we aren't using namespace std so pay attention!
ConnectionMonitorCore::ConnectionMonitorCore(CpuSNN* snn, int monitorId, short int connId) {
	snn_ = snn;
	connId_= connId;
	monitorId_ = monitorId;
	connFileId_ = NULL;

	needToWriteFileHeader_ = true;
	connFileSignature_ = 202029319;
	connFileVersion_ = 0.1f;

	// defer all unsafe operations to init function
	init();
}

void ConnectionMonitorCore::init() {
//	clear();

	// use KERNEL_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();
}

ConnectionMonitorCore::~ConnectionMonitorCore() {
	if (connFileId_!=NULL) {
		fclose(connFileId_);
		connFileId_ = NULL;
	}
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

void ConnectionMonitorCore::clear() {
	
}


void ConnectionMonitorCore::setConnectFileId(FILE* connFileId) {
	// \TODO consider the case where this function is called more than once
	if (connFileId_!=NULL)
		KERNEL_ERROR("ConnectionMonitorCore: setConnectFileId has already been called.");

	connFileId_=connFileId;

	if (connFileId_==NULL)
		needToWriteFileHeader_ = false;
	else {
		// for now: file pointer has changed, so we need to write header (again)
		needToWriteFileHeader_ = true;
		writeConnectFileHeader();
	}
}


// write the header section of the spike file
// this should be done once per file, and should be the very first entries in the file
void ConnectionMonitorCore::writeConnectFileHeader() {
	if (!needToWriteFileHeader_)
		return;

	// write file signature
	if (!fwrite(&connFileSignature_,sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");

	// write version number
	if (!fwrite(&connFileVersion_,sizeof(float),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");

	needToWriteFileHeader_ = false;
}