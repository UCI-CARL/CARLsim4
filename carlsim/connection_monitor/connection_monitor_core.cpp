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
	simTimeMs_ = -1;
	simTimeSinceLastMs_ = 0;
	simTimeMsLastWrite_ = -1;

	connFileId_ = NULL;
	needToWriteFileHeader_ = true;
	needToInit_ = true;
	connFileSignature_ = 202029319;
	connFileVersion_ = 0.2f;

	tookSnapshot_ = false;

	connFileTimeIntervalSec_ = 1;
}

void ConnectionMonitorCore::init() {
	if (!needToInit_)
		return;

	nNeurPre_  = snn_->getGroupNumNeurons(grpIdPre_);
	nNeurPost_ = snn_->getGroupNumNeurons(grpIdPost_);
	isPlastic_ = snn_->isConnectionPlastic(connId_);
	nSynapses_ = snn_->getNumSynapticConnections(connId_);

	assert(nNeurPre_>0);
	assert(nNeurPost_>0);

	// use KERNEL_{ERROR|WARNING|etc} typesetting (const FILE*)
	fpInf_ = snn_->getLogFpInf();
	fpErr_ = snn_->getLogFpErr();
	fpDeb_ = snn_->getLogFpDeb();
	fpLog_ = snn_->getLogFpLog();

	// init weight matrix with right dimensions
	for (int i=0; i<nNeurPre_; i++) {
		std::vector<float> wt, wtChange;
		for (int j=0; j<nNeurPost_; j++) {
			wt.push_back(NAN);
			wtChange.push_back(0.0f);
		}
		wtMat_.push_back(wt);
		wtLastMat_.push_back(wtChange);
	}

	// then load current weigths from CpuSNN into weight matrix
	takeSnapshot();

	needToInit_ = false;
}

ConnectionMonitorCore::~ConnectionMonitorCore() {
	if (connFileId_!=NULL) {
		// flush: advance timestep so that the last weight snapshot will be written to file
		updateTime(simTimeMs_+1);

		fclose(connFileId_);
		connFileId_ = NULL;
		needToInit_ = true;
		needToWriteFileHeader_ = true;
	}
}

// +++++ PUBLIC METHODS: +++++++++++++++++++++++++++++++++++++++++++++++//

// calculate weight changes since last update (element-wise )
std::vector< std::vector<float> > ConnectionMonitorCore::calcWeightChanges() {
	takeSnapshot();
	std::vector< std::vector<float> > wtChange(nNeurPre_, vector<float>(nNeurPost_));

	// take the naive approach for now
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			wtChange[i][j] = wtMat_[i][j] - wtLastMat_[i][j];
		}
	}

	return wtChange;
}


// reset weight matrix
void ConnectionMonitorCore::clear() {
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			wtMat_[i][j] = NAN;
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

bool ConnectionMonitorCore::needToWriteSnapshot() {
	// don't write if no file exists
	if (connFileId_==NULL)
		return false;

	// don't write to file at init
	if (simTimeMs_<0)
		return false;

	// don't write to file if we already have for this time step
	if (simTimeMs_==simTimeMsLastWrite_)
		return false;

	if (tookSnapshot_) {
		// we just took a manual snapshot, so we need to store it in binary
		return true;
	} else {
		// no manual snapshot taken, check interval
		return (connFileTimeIntervalSec_ != -1);
	}

	return true;
}

void ConnectionMonitorCore::print() {
	takeSnapshot();

	KERNEL_INFO("(t=%.3fs) ConnectionMonitor ID=%d: %d(%s) => %d(%s)", (double)simTimeMs_/1000.0f, connId_,
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

	takeSnapshot();
	KERNEL_INFO("(t=%.3fs) ConnectionMonitor ID=%d %d(%s) => %d(%s): [preId,postId] wt (+/-wtChange in %ldms) "
		"show first %d", (double)simTimeMs_/1000.0f, connId_,
		grpIdPre_, snn_->getGroupName(grpIdPre_).c_str(), grpIdPost_, snn_->getGroupName(grpIdPost_).c_str(),
		simTimeSinceLastMs_, maxConn);

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

std::vector< std::vector<float> > ConnectionMonitorCore::takeSnapshot() {
	tookSnapshot_ = true;
	snn_->updateConnectionMonitor(connId_);

	return wtMat_;
}

// returns true if ConnMon needed updating
bool ConnectionMonitorCore::updateTime(unsigned int simTimeMs) {
	long int currTime = (long int)simTimeMs; // get rid of unsigned

	bool needToUpdate = false;
	if (currTime > simTimeMs_) {
		fprintf(stderr,"in updateTime if, currTime=%ld, simTimeMs_=%ld\n",currTime,simTimeMs_);
		// time has advances since last storage
		needToUpdate = true;

		// write weights of last timestep to file
		// currently time interval can only be 1 or -1
		writeConnectFileSnapshot();

		tookSnapshot_ = false;

		// copy wtMat_ to lastWtMat_ and set wtMat_=0
		wtLastMat_.swap(wtMat_);
		clear();

		// update timers
		simTimeSinceLastMs_ = currTime-simTimeMs_; // delta t since last update
		simTimeMs_ = currTime;
		assert(simTimeSinceLastMs_>0);
	}

	return needToUpdate;
}

// CpuSNN uses float in weight matrix to save storage
// We want to do arithmetic on them, so use double instead (standard)
void ConnectionMonitorCore::updateWeight(int preId, int postId, float wt) {
	assert(preId < nNeurPre_);
	assert(postId < nNeurPost_);
	wtMat_[preId][postId] = wt;
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

	// \TODO: write delays

	needToWriteFileHeader_ = false;
}

void ConnectionMonitorCore::writeConnectFileSnapshot() {
	fprintf(stderr,"t=%ld, tookSnap=%s, needToWrite=%s\n",simTimeMs_,tookSnapshot_?"y":"n",needToWriteSnapshot()?"y":"n");
	if (!needToWriteSnapshot())
		return;

	tookSnapshot_ = false;

	simTimeMsLastWrite_ = simTimeMs_;
	fprintf(stderr,"writing to file at t=%ld\n",simTimeMs_);

	// write time stamp
	if (!fwrite(&simTimeMs_,sizeof(long int),1,connFileId_))
		KERNEL_ERROR("ConnectionMonitor: writeConnectFileSnapshot has fwrite error");

	// write all weights
	for (int i=0; i<nNeurPre_; i++) {
		for (int j=0; j<nNeurPost_; j++) {
			if (!fwrite(&wtMat_[i][j],sizeof(float),1,connFileId_)) {
				KERNEL_ERROR("ConnectionMonitor: writeConnectFileSnapshot has fwrite error");
			}
		}
	}
}
