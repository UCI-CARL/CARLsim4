#include <spikegen_from_file.h>

#include <carlsim.h>
//#include <user_errors.h>		// fancy user error messages

#include <stdio.h>				// fopen, fread, fclose
#include <string.h>				// std::string
#include <assert.h>				// assert


SpikeGeneratorFromFile::SpikeGeneratorFromFile(std::string fileName) {
	fileName_ = fileName;
	fpBegin_ = NULL;
	fpOffsetNeur_ = NULL;

	nNeur_ = -1;
	szByteHeader_ = -1;


	// needToInit_ = true;
	// needToAllocate_ = true;

	// move unsafe operations out of constructor
	openFile();
	init();
}

SpikeGeneratorFromFile::~SpikeGeneratorFromFile() {

	// if (fpOffsetNeur_!=NULL) delete[] fpOffsetNeur_;
	fclose(fpBegin_);
}

// rewind file pointers to beginning
void SpikeGeneratorFromFile::rewind() {
	// clear all spike times
	for (int i=0; i<nNeur_; i++) {
		spikes_[i].clear();
	}

	// needToInit_ = true;
	// init();
}

void SpikeGeneratorFromFile::openFile() {
	std::string funcName = "openFile("+fileName_+")";
	fpBegin_ = fopen(fileName_.c_str(),"rb");
	UserErrors::assertTrue(fpBegin_!=NULL, UserErrors::FILE_CANNOT_OPEN, funcName, fileName_);

	// \TODO there should be a common/standard way to read spike files
	// \FIXME: this is a hack...to get the size of the header section
	// needs to be updated every time header changes
	szByteHeader_ = 4*sizeof(int)+1*sizeof(float);
	fseek(fpBegin_	, sizeof(int)+sizeof(float), SEEK_SET); // skipping signature+version

	// get number of neurons from header
	nNeur_ = 1;
	int grid;
	for (int i=1; i<=3; i++) {
		size_t result = fread(&grid, sizeof(int), 1, fpBegin_);
		nNeur_ *= grid;
	}

	// make sure number of neurons is now valid
	assert(nNeur_>0);

	// reset file pointer to beginning of file
	fseek(fpBegin_, 0, SEEK_SET);
}

void SpikeGeneratorFromFile::init() {
	assert(nNeur_>0);

	// allocate
	for (int i=0; i<nNeur_; i++) {
		spikes_.push_back(std::vector<int>());
	}

	// read spike file
	FILE* fp = fpBegin_;
	fseek(fp, szByteHeader_, SEEK_SET);

	int tmpTime = -1;
	int tmpNeurId = -1;
	size_t result;
//	int maxTime = -1;

	while (!feof(fp)) {
		result = fread(&tmpTime, sizeof(int), 1, fp); // i-th time
		result = fread(&tmpNeurId, sizeof(int), 1, fp); // i-th nid
		spikes_[tmpNeurId].push_back(tmpTime);
//		maxTime = tmpTime;
	}

	for (int neurId=0; neurId<1; neurId++) {
		printf("[%d]: ",neurId);
		for (int i=0; i<spikes_[neurId].size(); i++) {
			printf("%d ",spikes_[neurId][i]);
		}
		printf("\n");
	}

	// initialize iterators
	for (int i=0; i<nNeur_; i++) {
		spikesIt_.push_back(spikes_[i].begin());
	}


//	printf("maxTime=%d\n",maxTime);
}

unsigned int SpikeGeneratorFromFile::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime, unsigned int endOfTimeSlice) {
	assert(nNeur_>0);
	assert(nid < nNeur_);

	if (spikesIt_[nid] != spikes_[nid].end()) {
		if (*(spikesIt_[nid]) < endOfTimeSlice) {
			if (nid==0) {
				printf("[0]: currTime=%u, lastTime=%u, endOfTime=%u, nextSpike=%u\n", currentTime,
					lastScheduledSpikeTime, endOfTimeSlice, *(spikesIt_[nid]));
			}
			return (unsigned int)*(spikesIt_[nid]++);
		}
	}

	return -1; // large positive number
	
	// FILE* fp = fpBegin_;
	// fseek(fpBegin_, szByteHeader_, SEEK_SET);
	// fseek(fp, fpOffsetNeur_[nid], SEEK_CUR);

	// int tmpTime = -1;
	// int tmpNeurId = -1;

	// // read the next time and neuron ID in the file
	// size_t result;
	// result = fread(&tmpTime, sizeof(int), 1, fp); // i-th time
	// result = fread(&tmpNeurId, sizeof(int), 1, fp); // i-th nid
	// fpOffsetNeur_[nid] += sizeof(int)*2;

	// // chances are this neuron ID is not the one we want, so we have to keep reading until we find the right one
	// while (tmpNeurId!=nid && !feof(fp)) {
	// 	result = fread(&tmpTime, sizeof(int), 1, fp); // j-th time
	// 	result = fread(&tmpNeurId, sizeof(int), 1, fp); // j-th nid
	// 	fpOffsetNeur_[nid] += sizeof(int)*2;
	// }

	// // if eof was reached, there are no more spikes for this neuron ID
	// if (feof(fp))
	// 	return -1; // large pos number

	// // else return the valid spike time
	// return tmpTime;
}
