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

	needToInit_ = true;
	needToAllocate_ = true;

	// move unsafe operations out of constructor
	openFile();
	init();
}

SpikeGeneratorFromFile::~SpikeGeneratorFromFile() {
	if (fpOffsetNeur_!=NULL) delete[] fpOffsetNeur_;
	fclose(fpBegin_);
}

// rewind file pointers to beginning
void SpikeGeneratorFromFile::rewind() {
	needToInit_ = true;
	init();
}

void SpikeGeneratorFromFile::openFile() {
	std::string funcName = "openFile("+fileName_+")";
	fpBegin_ = fopen(fileName_.c_str(),"rb");
	UserErrors::assertTrue(fpBegin_!=NULL, UserErrors::FILE_CANNOT_OPEN, funcName, fileName_);

	// \TODO there should be a common/standard way to read spike files
	// \FIXME: this is a hack...to get the size of the header section
	// needs to be updated every time header changes
	szByteHeader_ = 4*sizeof(int)+1*sizeof(float);

	// get number of neurons from header
	nNeur_ = 1;
	// \FIXME: same as above, this is a hack... use SpikeReader++
	fseek(fpBegin_	, sizeof(int)+sizeof(float), SEEK_SET);
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
	if (needToAllocate_) {
		// for each neuron, store a file pointer offset in #bytes from the SEEK_SET
		// this way we'll know exactly what the last spike was that we read per neuron
		fpOffsetNeur_ = new long int[nNeur_];
		needToAllocate_ = false;
	}
	if (needToInit_) {
		// init to zeros
		memset(fpOffsetNeur_, 0, sizeof(long int)*nNeur_);
		needToInit_ = false;
	}
}

unsigned int SpikeGeneratorFromFile::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime) {
	assert(nNeur_>0);

	FILE* fp = fpBegin_;
	fseek(fpBegin_, szByteHeader_, SEEK_SET);
	fseek(fp, fpOffsetNeur_[nid], SEEK_CUR);

	int tmpTime = -1;
	int tmpNeurId = -1;

	// read the next time and neuron ID in the file
	size_t result;
	result = fread(&tmpTime, sizeof(int), 1, fp); // i-th time
	result = fread(&tmpNeurId, sizeof(int), 1, fp); // i-th nid
	fpOffsetNeur_[nid] += sizeof(int)*2;

	// chances are this neuron ID is not the one we want, so we have to keep reading until we find the right one
	while (tmpNeurId!=nid && !feof(fp)) {
		result = fread(&tmpTime, sizeof(int), 1, fp); // j-th time
		result = fread(&tmpNeurId, sizeof(int), 1, fp); // j-th nid
		fpOffsetNeur_[nid] += sizeof(int)*2;
	}

	// if eof was reached, there are no more spikes for this neuron ID
	if (feof(fp))
		return -1; // large pos number

	// else return the valid spike time
	return tmpTime;
}
