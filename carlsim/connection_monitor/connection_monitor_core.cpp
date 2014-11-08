#include <connection_monitor_core.h>

#include <snn.h>				// CARLsim private implementation
#include <snn_definitions.h>	// KERNEL_ERROR, KERNEL_INFO, ...

#include <algorithm>			// std::sort
#include <iomanip>				// std::setfill, std::setw



// we aren't using namespace std so pay attention!
ConnectionMonitorCore::ConnectionMonitorCore(CpuSNN* snn,int monitorId,short int connId,int grpIdPre,int grpIdPost) {
	snn_ = snn;
	connId_= connId;
	grpIdPre_ = grpIdPre;
	grpIdPost_ = grpIdPost;
	monitorId_ = monitorId;
	connFileId_ = NULL;

	needToWriteFileHeader_ = true;
	connFileSignature_ = 202029319;
	connFileVersion_ = 0.1f;

	// defer all unsafe operations to init function
	init();
}

void ConnectionMonitorCore::init() {
	nNeurPre_ = snn_->getGroupNumNeurons(grpIdPre_);
	nNeurPost_ = snn_->getGroupNumNeurons(grpIdPost_);

	// init weight matrix with right dimensions
	for (int i=0; i<nNeurPre_; i++) {
		std::vector<float> wt;
		for (int j=0; j<nNeurPost_; j++) {
			wt.push_back(0.0f);
		}
		wtMat_.push_back(wt);
	}

	printf("created wtMat %dx%d\n",nNeurPre_,nNeurPost_);

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
	// empty weight matrix
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			wtMat_[i][j] = NAN;
		}
	}
}

void ConnectionMonitorCore::print() {
	takeSnapshot();

	KERNEL_INFO("ConnectionMonitor ID=%d: %d(%s) => %d(%s)",connId_, grpIdPre_, snn_->getGroupName(grpIdPre_).c_str(),
		grpIdPost_, snn_->getGroupName(grpIdPost_).c_str());

	// generate header
	std::stringstream header, header2;
	header  << " pre\\post |";
	header2 << "----------|";
	for (int j=0; j<nNeurPost_; j++) {
		header  << std::setw(9) << std::setfill(' ') << j << " |";
		header2 << "-----------";
	}
	KERNEL_INFO("%s",header.str().c_str());
	KERNEL_INFO("%s",header2.str().c_str());

	for (int i=0; i<nNeurPre_; i++) {
		std::stringstream line;
		line << std::setw(9) << std::setfill(' ') << i << " |";
		for (int j=0; j<nNeurPost_; j++) {
			line << std::fixed << std::setprecision(4) << (isnan(wtMat_[i][j])?"      ":(wtMat_[i][j]>=0?"   ":"  ")) << wtMat_[i][j]  << "  ";
		}
		KERNEL_INFO("%s",line.str().c_str());
	}
}

void ConnectionMonitorCore::printSparse() {
	takeSnapshot();
	KERNEL_INFO("ConnectionMonitor ID=%d: %d(%s) => %d(%s)",connId_, grpIdPre_, snn_->getGroupName(grpIdPre_).c_str(),
		grpIdPost_, snn_->getGroupName(grpIdPost_).c_str());

	std::stringstream line;
	int nConn = 0;
	int connPerLine = 30;
	int maxIntDigits = ceil(log10((double)max(nNeurPre_,nNeurPost_)));
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			if (!isnan(wtMat_[i][j])) {
				line << "(" << std::setw(maxIntDigits) << i << "=>" << std::setw(maxIntDigits) << j << ") " << std::fixed << std::setprecision(4) << wtMat_[i][j] << "   ";
				if (!(++nConn % connPerLine)) {
					KERNEL_INFO("%s",line.str().c_str());
					line.str(std::string());
				}
			}
		}
	}
	// flush
	if (nConn % connPerLine)
		KERNEL_INFO("%s",line.str().c_str());

}

std::vector< std::vector<float> > ConnectionMonitorCore::takeSnapshot() {
	snn_->updateConnectionMonitor(connId_);
	return wtMat_;
}

void ConnectionMonitorCore::updateWeight(int preId, int postId, float wt) {
//	printf("updating wt[%d][%d]=%f\n",preId,postId,wt);
	assert(preId < nNeurPre_);
	assert(postId < nNeurPost_);
	wtMat_[preId][postId] = wt;
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