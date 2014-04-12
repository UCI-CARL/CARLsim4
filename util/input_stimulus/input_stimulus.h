#ifndef _INPUT_STIMULUS_H_
#define _INPUT_STIMULUS_H_

#include <string>

class PoissonRate;

class InputStimulus {
public:
	// +++++ PUBLIC METHODS: CONSTRUCTOR / DESTRUCTOR / MEMBERS +++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! default constructor
	InputStimulus(std::string fileName, float maxPoisson=50.0f, bool wrapAroundEOF=true);

	//! default destructor
	~InputStimulus();

	//! list of stimulus file types
	enum stimType_t {STIM_GRAY, STIM_RGB, STIM_UNKNOWN};
	

	// +++++ PUBLIC METHODS: READING FRAMES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	//! reads the next image frame and returns a pointer to the char array
	unsigned char* readFrame();

	//! reads the next image frame and returns a pointer to a suitable PoissonRate object
	// potential problem: freeing the memory
	PoissonRate* readFrame(float maxPoisson);


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	int getStimulusWidth()  { return stimWidth_; }
	int getStimulusHeight() { return stimHeight_; }
	int getStimulusLength() { return stimLength_; }
	stimType_t getStimulusType() { return stimType_; }

	unsigned char* getCurrentFrame() { return stimFrame_; }; //! returns a pointer to the current image frame
	int getCurrentFrameNumber() { return stimFrameNr_; } //! returns the current frame number (0-indexed)

	void setMaxPoisson(float maxPoisson);

private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	void readFramePrivate();	//!< reads the next frame
	void readHeader();	//!< reads the header section of the binary file


	// +++++ PRIVATE MEMBERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	FILE* fileId_;	//!< pointer to FILE stream
	std::string fileName_;	//!< file name
	int fileSignature_;	//!< a unique file signature used for InputStimulus files

	long fileHeaderSize_;	//!< the number of bytes in the header section
	bool wrapAroundEOF_;	//!< if EOF is reached, whether to start reading from the top

	unsigned char* stimFrame_;	//!< char array of current frame
	int stimFrameNr_;	//!< current frame index (0-indexed)

	PoissonRate* stimFramePoiss_;	//!< pointer to a PoissonRate object that contains the current frame
	float stimMaxPoisson_;	//!< to which firing rate a grayscale value of 255 should be mapped to

	int stimWidth_;		//!< stimulus width in number of pixels (neurons)
	int stimHeight_;	//!< stimulus height in number of pixels (neurons)
	int stimLength_;	//!< stimulus length in number of frames

	int stimChannels_;	//!< number of channels (1=grayscale, 3=RGB)
	stimType_t stimType_;	//!< stimulus type (grayscale, RGB, etc.)
};

#endif