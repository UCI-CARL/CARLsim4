#include <spikegen_from_file.h>

#include <carlsim.h>
//#include <user_errors.h>		// fancy user error messages

#include <stdio.h>				// fopen, fread, fclose
#include <string.h>				// std::string


SpikeGeneratorFromFile::SpikeGeneratorFromFile(std::string fileName) {
	fileName_ = fileName;
	fpBegin_ = NULL;
	fpOffsetNeur_ = NULL;
	needToInit_ = true;

	// move unsafe operations out of constructor
	openFile();
}

SpikeGeneratorFromFile::~SpikeGeneratorFromFile() {
	if (fpOffsetNeur_!=NULL) delete[] fpOffsetNeur_;
	fclose(fpBegin_);
}


void SpikeGeneratorFromFile::openFile() {
	std::string funcName = "openFile("+fileName_+")";
	fpBegin_ = fopen(fileName_.c_str(),"rb");
	UserErrors::assertTrue(fpBegin_!=NULL, UserErrors::FILE_CANNOT_OPEN, funcName, fileName_);

	// \TODO add signature to spike file so that we can make sure we have the right file
}

unsigned int SpikeGeneratorFromFile::nextSpikeTime(CARLsim* sim, int grpId, int nid, unsigned int currentTime, 
	unsigned int lastScheduledSpikeTime) {

	if (needToInit_) {
		int nNeur = sim->getGroupNumNeurons(grpId);

		// for each neuron, store a file pointer offset in #bytes from the SEEK_SET
		// this way we'll know exactly what the last spike was that we read per neuron
		fpOffsetNeur_ = new long int[nNeur];
		memset(fpOffsetNeur_, 0, sizeof(long int)*nNeur);

		needToInit_ = false;
	}

	FILE* fp = fpBegin_;
	fseek(fp, fpOffsetNeur_[nid], SEEK_SET);

	int tmpTime = -1;
	int tmpNeurId = -1;

	// read the next time and neuron ID in the file
	fread(&tmpTime, sizeof(int), 1, fp); // i-th time
	fread(&tmpNeurId, sizeof(int), 1, fp); // i-th nid
	fpOffsetNeur_[nid] += sizeof(int)*2;

	// chances are this neuron ID is not the one we want, so we have to keep reading until we find the right one
	while (tmpNeurId!=nid && !feof(fp)) {
		fread(&tmpTime, sizeof(int), 1, fp); // j-th time
		fread(&tmpNeurId, sizeof(int), 1, fp); // j-th nid
		fpOffsetNeur_[nid] += sizeof(int)*2;
	}

	// if eof was reached, there are no more spikes for this neuron ID
	if (feof(fp))
		return -1; // large pos number

	// else return the valid spike time
	return tmpTime;
}
