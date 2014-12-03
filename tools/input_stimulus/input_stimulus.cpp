#include <input_stimulus.h>

#include <poisson_rate.h>
#include <string>
#include <cassert> // assert
#include <stdio.h> // fopen, fread, fclose
#include <stdlib.h> // exit

// constructor
InputStimulus::InputStimulus(std::string fileName, bool wrapAroundEOF) {
	fileId_ = NULL;
	fileName_ = fileName;
	wrapAroundEOF_ = wrapAroundEOF;

	stimFrame_ = NULL;
	stimFrameNr_ = -1;

	stimWidth_ = -1;
	stimHeight_ = -1;
	stimLength_ = -1;

	stimFramePoiss_ = NULL;

	stimChannels_ = -1;
	stimType_ = STIM_UNKNOWN;

	fileSignature_ = 304698591; // v1.0 file signature

	// read the header section of the binary file
	readHeader();
}

// destructor
InputStimulus::~InputStimulus() {
	if (stimFrame_!=NULL)
		delete[] stimFrame_;
	stimFrame_=NULL;

	if (stimFramePoiss_!=NULL)
		delete[] stimFramePoiss_;
	stimFramePoiss_=NULL;

	if (fileId_!=NULL)
		fclose(fileId_);
}

// reads the next frame and returns the char array
unsigned char* InputStimulus::readFrame() {
	readFramePrivate();
	return stimFrame_;
}

// reads the next frame and returns the PoissonRate object
PoissonRate* InputStimulus::readFrame(float maxPoisson) {
	assert(maxPoisson!=0);

	// read next frame
	readFramePrivate();

	// create new Poisson object, assign 
	stimFramePoiss_ = new PoissonRate(stimWidth_*stimHeight_*stimChannels_);
	for (int i=0; i<stimWidth_*stimHeight_*stimChannels_; i++)
		stimFramePoiss_->setRate(i, stimFrame_[i]*1.0/255.0*maxPoisson); // scale firing rates

	return stimFramePoiss_;
}


// private method: reads next frame and assigns char array and Poisson Rate
void InputStimulus::readFramePrivate() {
	// TODO: So far we only support grayscale images
	assert(stimChannels_==1);
	assert(stimType_==STIM_GRAY);

	// make sure type is set
	assert(stimType_!=STIM_UNKNOWN);

	// keep at most one frame in memory
	if (stimFrame_!=NULL)
		delete[] stimFrame_;
	if (stimFramePoiss_!=NULL)
		delete[] stimFramePoiss_;
	stimFrame_ = NULL; stimFramePoiss_ = NULL;

	// have we reached EOF?
	if (feof(fileId_) || (stimFrameNr_==stimLength_-1)) {
		// reset frame index
		stimFrameNr_ = -1;

		if (!wrapAroundEOF_) {
			// we've reached end of file, print a warning
			fprintf(stderr,"WARNING: End of file reached, starting from the top\n");
		}

		// rewind position of file stream to first frame
		rewind();
	}

	// read new frame
	stimFrame_ = new unsigned char[stimWidth_*stimHeight_*stimChannels_];
	size_t result = fread(stimFrame_, sizeof(unsigned char), stimWidth_*stimHeight_*stimChannels_, fileId_);
	if (result!=(size_t) (stimWidth_*stimHeight_*stimChannels_)) {
		fprintf(stderr,"INPUTSTIM ERROR: Error while reading stimulus frame (expected %d elements, found %d\n",
			stimWidth_*stimHeight_*stimChannels_, (int)result);
		exit(1);
	}

	// initialized as -1, so after reading first frame this sits at 0
	stimFrameNr_++;
}

// reads the header section of the binary file
void InputStimulus::readHeader() {
	fileId_ = fopen(fileName_.c_str(),"rb");
	if (fileId_==NULL) {
		fprintf(stderr,"INPUTSTIM ERROR: Could not open stimulus file %s\n",fileName_.c_str());
		exit(1);
	}

	bool readErr = false; // keep track of reading errors
	size_t result;
	int tmpInt;
	float tmpFloat;
	char tmpChar;

	// read signature
	result = fread(&tmpInt, sizeof(int), 1, fileId_);
	readErr |= (result!=1);
	if (tmpInt != fileSignature_) {
		fprintf(stderr,"INPUTSTIM ERROR: Unknown file signature\n");
		exit(1);
	}

	// read version number
	result = fread(&tmpFloat, sizeof(float), 1, fileId_);
	readErr |= (result!=1);
	if (tmpFloat != 1.0) {
		fprintf(stderr,"INPUTSTIM ERROR: Unknown file version (%1.1f), must have 1.0\n",tmpFloat);
		exit(1);
	}

	// read number of channels
	result = fread(&tmpChar, sizeof(char), 1, fileId_);
	readErr |= (result!=1);
	stimChannels_ = (int)tmpChar;
	switch (stimChannels_) {
	case 1:
		stimType_ = STIM_GRAY;
		break;
	case 3:
		stimType_ = STIM_RGB;
		break;
	default:
		fprintf(stderr,"INPUTSTIM ERROR: Unknown stimulus type encountered (%d channels)\n",stimChannels_);
		exit(1);
	}

	// read stimulus dimensions
	fread(&tmpInt, sizeof(int), 1, fileId_);
	readErr |= (result!=1);
	stimWidth_ = tmpInt;
	fread(&tmpInt, sizeof(int), 1, fileId_);
	readErr |= (result!=1);
	stimHeight_ = tmpInt;
	fread(&tmpInt, sizeof(int), 1, fileId_);
	readErr |= (result!=1);
	stimLength_ = tmpInt;

	// any reading errors encountered?
	if (readErr) {
		fprintf(stderr,"INPUTSTIM ERROR: Error while reading file %s\n",fileName_.c_str());
		exit(1);
	}

	// store the size of the header section (in bytes)
	fileHeaderSize_ = ftell(fileId_);
}

// rewind position of file stream to first frame
void InputStimulus::rewind() {
	fseek(fileId_, fileHeaderSize_, SEEK_SET);
}