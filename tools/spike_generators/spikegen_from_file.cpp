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
#include <spikegen_from_file.h>

#include <carlsim.h>
//#include <user_errors.h>		// fancy user error messages

#include <stdio.h>				// fopen, fread, fclose
#include <string.h>				// std::string
#include <assert.h>				// assert

// #define VERBOSE

SpikeGeneratorFromFile::SpikeGeneratorFromFile(std::string fileName, int offsetTimeMs) {
	fileName_ = fileName;
	fpBegin_ = NULL;

	nNeur_ = -1;
	szByteHeader_ = -1;
	offsetTimeMs_ = offsetTimeMs;

	// move unsafe operations out of constructor
	openFile();
	init();
}

SpikeGeneratorFromFile::~SpikeGeneratorFromFile() {
	if (fpBegin_ != NULL) {
		fclose(fpBegin_);
	}
	fpBegin_ = NULL;
}

void SpikeGeneratorFromFile::loadFile(std::string fileName, int offsetTimeMs) {
	// close previously opened file (if any)
	if (fpBegin_ != NULL) {
		fclose(fpBegin_);
	}
	fpBegin_ = NULL;

	// update file name and open
	fileName_ = fileName;
	offsetTimeMs_ = offsetTimeMs;
	openFile();
	init();
}

// rewind file pointers to beginning
void SpikeGeneratorFromFile::rewind(int offsetTimeMs) {
	offsetTimeMs_ = offsetTimeMs;

	// reset all iterators
	spikesIt_.clear();
	for (int i=0; i<nNeur_; i++) {
		spikesIt_.push_back(spikes_[i].begin());
	}
}

void SpikeGeneratorFromFile::openFile() {
	std::string funcName = "openFile("+fileName_+")";
	fpBegin_ = fopen(fileName_.c_str(),"rb");
	UserErrors::assertTrue(fpBegin_!=NULL, UserErrors::FILE_CANNOT_OPEN, funcName, fileName_);

	// \TODO there should be a common/standard way to read spike files
	// \FIXME: this is a hack...to get the size of the header section
	// needs to be updated every time header changes
	FILE* fp = fpBegin_;
	szByteHeader_ = 4*sizeof(int)+1*sizeof(float);
	fseek(fp, sizeof(int)+sizeof(float), SEEK_SET); // skipping signature+version

	// get number of neurons from header
	nNeur_ = 1;
	int grid;
	for (int i=1; i<=3; i++) {
		size_t result = fread(&grid, sizeof(int), 1, fp);
		UserErrors::assertTrue(result == 1, UserErrors::FILE_CANNOT_READ, funcName, fileName_);
		nNeur_ *= grid;
	}

	// make sure number of neurons is now valid
	assert(nNeur_>0);
}

void SpikeGeneratorFromFile::init() {
	assert(nNeur_>0);

	// allocate spike vector
	// we organize AER format into a 2D spike vector: first dim=neuron, second dim=spike times
	// then we just need to maintain an iterator for each neuron to know which spike to schedule next
	spikes_.clear();
	for (int i=0; i<nNeur_; i++) {
		spikes_.push_back(std::vector<int>());
	}

	// read spike file
	FILE* fp = fpBegin_;
	fseek(fp, szByteHeader_, SEEK_SET); // skip header section

	std::string funcName = "readFile("+fileName_+")";
	int tmpTime = -1;
	int tmpNeurId = -1;
	size_t result1, result2;

	while (!feof(fp)) {
		result1 = fread(&tmpTime, sizeof(int), 1, fp); // i-th time
		//UserErrors::assertTrue(result == 1, UserErrors::FILE_CANNOT_READ, funcName, fileName_);
		result2 = fread(&tmpNeurId, sizeof(int), 1, fp); // i-th nid
		//UserErrors::assertTrue(result == 1, UserErrors::FILE_CANNOT_READ, funcName, fileName_);
		if (result1 + result2 == 2) // validate the size of reading
			spikes_[tmpNeurId].push_back(tmpTime); // add spike time to 2D vector
	}

#ifdef VERBOSE
	for (int neurId=0; neurId<1; neurId++) {
		printf("[%d]: ",neurId);
		for (int i=0; i<spikes_[neurId].size(); i++) {
			printf("%d ",spikes_[neurId][i]);
		}
		printf("\n");
	}
#endif

	// initialize iterators
	rewind(offsetTimeMs_);
}

int SpikeGeneratorFromFile::nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) {
	assert(nNeur_>0);
	assert(nid < nNeur_);

	if (spikesIt_[nid] != spikes_[nid].end()) {
		// if there are spikes left in the vector ...

		if (*(spikesIt_[nid])+offsetTimeMs_ < endOfTimeSlice) {
			// ... and if the next spike time is in the current scheduling time slice:
#ifdef VERBOSE
			if (nid==0) {
			printf("[%d][%d]: currTime=%u, lastTime=%u, endOfTime=%u, offsetTimeMs=%u, nextSpike=%u\n", grpId, nid,
				currentTime, lastScheduledSpikeTime, endOfTimeSlice, offsetTimeMs_,
				(unsigned int) (*(spikesIt_[nid])+offsetTimeMs_));
			}
#endif
			// return the next spike time and update iterator
			return (unsigned int)(*(spikesIt_[nid]++)+offsetTimeMs_);
		}
	}

	// if the next spike time is not a valid number, return a large positive number instead
	// this will signal CARLsim to break the nextSpikeTime loop
	return -1; // large positive number
}
