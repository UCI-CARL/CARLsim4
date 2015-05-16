#include <connection_monitor_core.h>

#include <snn.h>				// CARLsim private implementation
#include <snn_definitions.h>	// KERNEL_ERROR, KERNEL_INFO, ...

#include <sstream>				// std::stringstream
#include <algorithm>			// std::sort
#include <iomanip>				// std::setfill, std::setw



// we aren't using namespace std so pay attention!
ConnectionMonitorCore::ConnectionMonitorCore(CpuSNN* snn,int monitorId,short int connId,int grpIdPre,int grpIdPost) {
	snn_ = snn;
	connId_= connId;
	grpIdPre_ = grpIdPre;
	grpIdPost_ = grpIdPost;
	monitorId_ = monitorId;

	wtTime_ = -1;
	wtTimeLast_ = -1;
	wtTimeWrite_ = -1;

	connFileId_ = NULL;
	needToWriteFileHeader_ = true;
	needToInit_ = true;
	connFileSignature_ = 202029319;
	connFileVersion_ = 0.4f;

	minWt_ = -1.0f;
	maxWt_ = -1.0f;

	connFileTimeIntervalSec_ = 1;
}

void ConnectionMonitorCore::init() {
	if (!needToInit_)
		return;

	nNeurPre_  = snn_->getGroupNumNeurons(grpIdPre_);
	nNeurPost_ = snn_->getGroupNumNeurons(grpIdPost_);
	isPlastic_ = snn_->isConnectionPlastic(connId_);
	nSynapses_ = snn_->getNumSynapticConnections(connId_);

	grpConnectInfo_t* connInfo = snn_->getConnectInfo(connId_);
	minWt_ = 0.0f; // for now, no non-zero min weights allowed
	maxWt_ = fabs(connInfo->maxWt);

	assert(nNeurPre_>0);
	assert(nNeurPost_>0);

	// use KERNEL_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();

	// init weight matrix with right dimensions
	for (int i=0; i<nNeurPre_; i++) {
		std::vector<float> wt;
		for (int j=0; j<nNeurPost_; j++) {
			wt.push_back(NAN);
		}
		wtMat_.push_back(wt);
		wtMatLast_.push_back(wt);
	}

	// then load current weigths from CpuSNN into weight matrix
	updateStoredWeights();

	// store the initial snapshot to file if update interval set
	if (connFileTimeIntervalSec_ > 0) {
		writeConnectFileSnapshot(0, wtMat_);
	}
}

ConnectionMonitorCore::~ConnectionMonitorCore() {
	if (connFileId_!=NULL) {
		// flush: store last snapshot to file if update interval set
		if (connFileTimeIntervalSec_ > 0) {
			// make sure CpuSNN is not already deallocated!
			assert(snn_!=NULL);
			writeConnectFileSnapshot(snn_->getSimTime(), snn_->getWeightMatrix2D(connId_));
		}

		// then close file and clean up
		fclose(connFileId_);
		connFileId_ = NULL;
		needToInit_ = true;
		needToWriteFileHeader_ = true;
	}
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

// calculate weight changes since last update (element-wise )
std::vector< std::vector<float> > ConnectionMonitorCore::calcWeightChanges() {
	updateStoredWeights();
	std::vector< std::vector<float> > wtChange(nNeurPre_, vector<float>(nNeurPost_));

	// take the naive approach for now
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			wtChange[i][j] = wtMat_[i][j] - wtMatLast_[i][j];
		}
	}

	return wtChange;
}


// reset weight matrix
void ConnectionMonitorCore::clear() {
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			wtMat_[i][j] = NAN;
			wtMatLast_[i][j] = NAN;
		}
	}
}

// find number of incoming synapses for a specific post neuron
int ConnectionMonitorCore::getFanIn(int neurPostId) {
	assert(neurPostId<nNeurPost_);
	int nSyn = 0;
	for (int i=0; i<nNeurPre_; i++) {
		if (!isnan(wtMat_[i][neurPostId])) {
			nSyn++;
		}
	}
	return nSyn;
}

// find number of outgoing synapses of a specific pre neuron
int ConnectionMonitorCore::getFanOut(int neurPreId) {
	assert(neurPreId<nNeurPre_);
	int nSyn = 0;
	for (int j=0; j<nNeurPost_; j++) {
		if (!isnan(wtMat_[neurPreId][j])) {
			nSyn++;
		}
	}
	return nSyn;
}

// calculate total absolute amount of weight change
double ConnectionMonitorCore::getTotalAbsWeightChange() {
	std::vector< std::vector<float> > wtChange = calcWeightChanges();
	double wtTotalChange = 0.0;
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			// skip entries in matrix where no synapse exists
			if (isnan(wtMat_[i][j]))
				continue;
			wtTotalChange += fabs(wtChange[i][j]);
		}
	}
	return wtTotalChange;
}

// find (number of synapses whose weigths changed)/(total number synapses)
double ConnectionMonitorCore::getPercentWeightsChanged(double minAbsChange) {
	assert(minAbsChange>=0.0);
	int nChanged = getNumWeightsChanged(minAbsChange);

	return nChanged*100.0/nSynapses_;
}

// find number of synapses whose weights changed
int ConnectionMonitorCore::getNumWeightsChanged(double minAbsChange) {
	assert(minAbsChange>=0.0);
	std::vector< std::vector<float> > wtChange = calcWeightChanges();

	int nChanged = 0;
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			// skip entries in matrix where no synapse exists
			if (isnan(wtMat_[i][j]))
				continue;

			if (fabs(wtChange[i][j]) >= minAbsChange) {
				nChanged++;
			}
		}
	}
	return nChanged;
}

void ConnectionMonitorCore::print() {
	updateStoredWeights();


	KERNEL_INFO("(t=%.3fs) ConnectionMonitor ID=%d: %d(%s) => %d(%s)", (double)wtTime_/1000.0f, connId_,
		grpIdPre_, snn_->getGroupName(grpIdPre_).c_str(),
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
			line << std::fixed << std::setprecision(4) << (isnan(wtMat_[i][j])?"      ":(wtMat_[i][j]>=0?"   ":"  "))
				<< wtMat_[i][j]  << "  ";
		}
		KERNEL_INFO("%s",line.str().c_str());
	}
}

void ConnectionMonitorCore::printSparse(int neurPostId, int maxConn, int connPerLine) {
	assert(neurPostId<nNeurPost_);
	assert(maxConn>0);
	assert(connPerLine>0);

	updateStoredWeights();
	KERNEL_INFO("(t=%.3fs) ConnectionMonitor ID=%d %d(%s) => %d(%s): [preId,postId] wt (+/-wtChange in %ldms) "
		"show first %d", (double)wtTime_/1000.0f, connId_,
		grpIdPre_, snn_->getGroupName(grpIdPre_).c_str(), grpIdPost_, snn_->getGroupName(grpIdPost_).c_str(),
		getTimeMsSinceLastSnapshot(), maxConn);

	int postA, postZ;
	if (neurPostId==ALL) {
		postA = 0;
		postZ = nNeurPost_;
	} else {
		postA = neurPostId;
		postZ = neurPostId;
	}

	std::vector< std::vector<float> > wtChange;
	if (isPlastic_) {
		wtChange = calcWeightChanges();
	}

	std::stringstream line;
	int nConn = 0;
	int maxIntDigits = ceil(log10((double)max(nNeurPre_,nNeurPost_)));
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=postA; j<postZ; j++) {
			// display only so many connections
			if (nConn>=maxConn)
				break;

			if (!isnan(wtMat_[i][j])) {
				line << "[" << std::setw(maxIntDigits) << i << "," << std::setw(maxIntDigits) << j << "] "
					<< std::fixed << std::setprecision(4) << wtMat_[i][j];
				if (isPlastic_) {
					line << " (" << ((wtChange[i][j]<0)?"":"+");
					line << std::setprecision(4) << wtChange[i][j] << ")";
				}
				line << "   ";
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

void ConnectionMonitorCore::setConnectFileId(FILE* connFileId) {
	// \TODO consider the case where this function is called more than once
	if (connFileId_!=NULL)
		KERNEL_ERROR("ConnectionMonitorCore: setConnectFileId has already been called.");

	connFileId_=connFileId;

	if (connFileId_==NULL) {
		needToWriteFileHeader_ = false;
	}
	else {
		// for now: file pointer has changed, so we need to write header (again)
		needToWriteFileHeader_ = true;
		writeConnectFileHeader();
	}
}

void ConnectionMonitorCore::setUpdateTimeIntervalSec(int intervalSec) {
	assert(intervalSec==-1 || intervalSec==1);
	connFileTimeIntervalSec_ = intervalSec;
}

// updates the internally stored last two snapshots (current one and last one)
void ConnectionMonitorCore::updateStoredWeights() {
	fprintf(stderr,"%u: updateStoredWeights\n",snn_->getSimTime());
	if (snn_->getSimTime() > wtTime_) {
		fprintf(stderr,"%u: updateStoredWeights need to update\n",snn_->getSimTime());
		// time has advanced: get new weights
		wtMatLast_ = wtMat_;
		wtTimeLast_ = wtTime_;

		wtMat_ = snn_->getWeightMatrix2D(connId_);
		wtTime_ = snn_->getSimTime();
	}
}

// returns a current snapshot
std::vector< std::vector<float> > ConnectionMonitorCore::takeSnapshot() {
	updateStoredWeights();
	return wtMat_;
}


// write the header section of the spike file
// this should be done once per file, and should be the very first entries in the file
void ConnectionMonitorCore::writeConnectFileHeader() {
	init();

	if (!needToWriteFileHeader_)
		return;

	// write file signature
	if (!fwrite(&connFileSignature_,sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");

	// write version number
	if (!fwrite(&connFileVersion_,sizeof(float),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");

	// write connection id
	if (!fwrite(&connId_,sizeof(short int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");

	// write pre group info: group id and Grid3D dimensions
	Grid3D gridPre = snn_->getGroupGrid3D(grpIdPre_);
	if (!fwrite(&grpIdPre_,sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPre.x),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPre.y),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPre.z),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");

	// write post group info: group id and # neurons
	Grid3D gridPost = snn_->getGroupGrid3D(grpIdPost_);
	if (!fwrite(&grpIdPost_,sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPost.x),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPost.y),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");
	if (!fwrite(&(gridPost.z),sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");

	// write number of synapses
	if (!fwrite(&nSynapses_,sizeof(int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");

	// write synapse type (fixed=false, plastic=true)
	if (!fwrite(&isPlastic_,sizeof(bool),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileHeader has fwrite error");

	// write minWt and maxWt
	if (!fwrite(&minWt_,sizeof(float),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");
	if (!fwrite(&maxWt_,sizeof(float),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitorCore: writeConnectFileHeader has fwrite error");


	// \TODO: write delays

	needToWriteFileHeader_ = false;
}

void ConnectionMonitorCore::writeConnectFileSnapshot(unsigned int simTimeMs, std::vector< std::vector<float> > wts) {
	// don't write if we have already written this timestamp to file (or file doesn't exist)
	if ((long int)simTimeMs <= wtTimeWrite_ || connFileId_==NULL) {
		return;
	}

	wtTimeWrite_ = (long int)simTimeMs;

	// write time stamp
	if (!fwrite(&wtTimeWrite_,sizeof(long int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileSnapshot has fwrite error");

	// write all weights
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			if (!fwrite(&wts[i][j],sizeof(float),1,connFileId_)) {
				KERNEL_ERROR("ConnectionMonitor: writeConnectFileSnapshot has fwrite error");
			}
		}
	}
}
