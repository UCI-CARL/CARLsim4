#include "visual_stimulus.h"

#include <poisson_rate.h>
#include <string>
#include <cassert> // assert
#include <stdio.h> // fopen, fread, fclose
#include <stdlib.h> // exit

class VisualStimulus::Impl {
public:
	// +++++ PUBLIC METHODS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	Impl(std::string fileName, bool wrapAroundEOF) {
		_fileId = NULL;
		_fileName = fileName;
		_wrapAroundEOF = wrapAroundEOF;

		_frame = NULL;
		_frameNum = -1;

		_width = -1;
		_height = -1;
		_length = -1;

		_framePoisson = NULL;

		_channels = -1;
		_type = UNKNOWN_STIM;

		_version = 1.0f;
		_fileSignature = 293390619; // v1.0 file signature

		// read the header section of the binary file
		readHeader();
	}

	~Impl() {
		if (_frame!=NULL)
			delete[] _frame;
		_frame=NULL;

		if (_framePoisson!=NULL)
			delete _framePoisson;
		_framePoisson=NULL;

		if (_fileId!=NULL)
			fclose(_fileId);
	}

	// reads the next frame and returns the char array
	unsigned char* readFrameChar() {
		readFramePrivate();
		return _frame;
	}

	// reads the next frame and returns the PoissonRate object
	PoissonRate* readFramePoisson(float maxPoisson, float minPoisson) {
		assert(maxPoisson>0);
		assert(maxPoisson>minPoisson);

		// read next frame
		readFramePrivate();

		// create new Poisson object, assign 
		_framePoisson = new PoissonRate(_width*_height*_channels);
		for (int i=0; i<_width*_height*_channels; i++) {
			_framePoisson->setRate(i, _frame[i]*(maxPoisson-minPoisson)/255.0f + minPoisson); // scale firing rates
		}
		
		return _framePoisson;
	}

	// rewind position of file stream to first frame
	void rewind() {
		fseek(_fileId, _fileHeaderSizeBytes, SEEK_SET);
	}

	void print() {
		fprintf(stdout, "VisualStimulus loaded (\"%s\", Type %d, Size %dx%dx%dx%d).\n", _fileName.c_str(), _type, 
			_width, _height, _channels, _length);
	}

	int getWidth()  { return _width; }
	int getHeight() { return _height; }
	int getLength() { return _length; }
	int getChannels() { return _channels; }
	stimType_t getType() { return _type; }

	unsigned char* getCurrentFrameChar() { return _frame; }
	PoissonRate* getCurrentFramePoisson() { return _framePoisson; }
	int getCurrentFrameNumber() { return _frameNum; }


private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	// reads next frame and assigns char array and Poisson Rate
	void readFramePrivate() {
		// make sure type is set
		assert(_type!=UNKNOWN_STIM);

		// keep at most one frame in memory
		if (_frame!=NULL)
			delete[] _frame;
		if (_framePoisson!=NULL)
			delete _framePoisson;
		_frame = NULL; _framePoisson = NULL;

		// have we reached EOF?
		if (feof(_fileId) || (_frameNum==_length-1)) {
			// reset frame index
			_frameNum = -1;

			if (!_wrapAroundEOF) {
				// we've reached end of file, print a warning
				fprintf(stderr,"WARNING: End of file reached, starting from the top\n");
			}

			// rewind position of file stream to first frame
			rewind();
		}

		// read new frame
		_frame = new unsigned char[_width*_height*_channels];
		size_t result = fread(_frame, sizeof(unsigned char), _width*_height*_channels, _fileId);
		if (result!=(size_t) (_width*_height*_channels)) {
			fprintf(stderr,"VisualStimulus Error: Error while reading stimulus frame (expected %d elements, found %d\n",
				_width*_height*_channels, (int)result);
			exit(1);
		}

		// initialized as -1, so after reading first frame this sits at 0
		_frameNum++;
	}

	// reads the header section of the binary file
	void readHeader() {
		_fileId = fopen(_fileName.c_str(),"rb");
		if (_fileId==NULL) {
			fprintf(stderr,"VisualStimulus Error: Could not open stimulus file %s\n",_fileName.c_str());
			exit(1);
		}

		bool readErr = false; // keep track of reading Errors
		size_t result;
		int tmpInt;
		float tmpFloat;
		char tmpChar;

		// read signature
		result = fread(&tmpInt, sizeof(int), 1, _fileId);
		readErr |= (result!=1);
		if (tmpInt != _fileSignature) {
			fprintf(stderr,"VisualStimulus Error: Unknown file signature\n");
			exit(1);
		}

		// read version number
		result = fread(&tmpFloat, sizeof(float), 1, _fileId);
		readErr |= (result!=1);
		if (tmpFloat != _version) {
			fprintf(stderr,"VisualStimulus Error: Unknown file version (%1.1f), must have 1.0\n",tmpFloat);
			exit(1);
		}

		// read stimulus type
		result = fread(&tmpInt, sizeof(int), 1, _fileId);
		readErr |= (result!=1);
		if (tmpInt < UNKNOWN_STIM || tmpInt > COMPOUND_STIM) {
			fprintf(stderr,"VisualStimulus Error: Unknown stimulus type found (%d)\n",tmpInt);
			exit(1);
		}
		_type = static_cast<stimType_t>(tmpInt);

		// read number of channels
		result = fread(&tmpChar, sizeof(char), 1, _fileId);
		readErr |= (result!=1);
		_channels = (int)tmpChar;

		// read stimulus dimensions
		result = fread(&tmpInt, sizeof(int), 1, _fileId);		readErr |= (result!=1);
		_width = tmpInt;
		result = fread(&tmpInt, sizeof(int), 1, _fileId);		readErr |= (result!=1);
		_height = tmpInt;
		result = fread(&tmpInt, sizeof(int), 1, _fileId);		readErr |= (result!=1);
		_length = tmpInt;

		// any reading Errors encountered?
		if (readErr) {
			fprintf(stderr,"VisualStimulus Error: Error while reading file %s\n",_fileName.c_str());
			exit(1);
		}

		// store the size of the header section (in bytes)
		_fileHeaderSizeBytes = ftell(_fileId);
	}


	// +++++ PRIVATE MEMBERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	FILE* _fileId;				//!< pointer to FILE stream
	std::string _fileName;		//!< file name

	int _version;				//!< VisualStimulusToolbox major/minor version
	int _fileSignature;			//!< a unique file signature used for VisualStimulus files

	long _fileHeaderSizeBytes;	//!< the number of bytes in the header section
	bool _wrapAroundEOF;		//!< if EOF is reached, whether to start reading from the top

	unsigned char* _frame;		//!< char array of current frame
	int _frameNum;				//!< current frame index (0-indexed)

	PoissonRate* _framePoisson;	//!< pointer to a PoissonRate object that contains the current frame

	int _width;					//!< stimulus width in number of pixels (neurons)
	int _height;				//!< stimulus height in number of pixels (neurons)
	int _length;				//!< stimulus length in number of frames

	int _channels;				//!< number of channels (1=grayscale, 3=RGB)
	stimType_t _type;			//!< stimulus type
};


// ****************************************************************************************************************** //
// VISUALSTIMULUS API IMPLEMENTATION
// ****************************************************************************************************************** //

// create and destroy a pImpl instance
VisualStimulus::VisualStimulus(std::string fileName, bool wrapAroundEOF) : _impl( new Impl(fileName, wrapAroundEOF) ) {}
VisualStimulus::~VisualStimulus() { delete _impl; }

unsigned char* VisualStimulus::readFrameChar() { return _impl->readFrameChar(); }
PoissonRate* VisualStimulus::readFramePoisson(float maxPoisson, float minPoisson) {
	return _impl->readFramePoisson(maxPoisson, minPoisson);
}
void VisualStimulus::rewind() { _impl->rewind(); }
void VisualStimulus::print() { _impl->print(); }

int VisualStimulus::getWidth() { return _impl->getWidth(); }
int VisualStimulus::getHeight() { return _impl->getHeight(); }
int VisualStimulus::getLength() { return _impl->getLength(); }
int VisualStimulus::getChannels() { return _impl->getChannels(); }
stimType_t VisualStimulus::getType() { return _impl->getType(); }
unsigned char* VisualStimulus::getCurrentFrameChar() { return _impl->getCurrentFrameChar(); }
PoissonRate* VisualStimulus::getCurrentFramePoisson() { return _impl->getCurrentFramePoisson(); }
int VisualStimulus::getCurrentFrameNumber() { return _impl->getCurrentFrameNumber(); }
