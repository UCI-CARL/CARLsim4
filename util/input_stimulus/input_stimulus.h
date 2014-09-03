#ifndef _INPUT_STIMULUS_H_
#define _INPUT_STIMULUS_H_

#include <string>
class PoissonRate;

/*!
 * \brief Class to integrate CARLsim with a stimulus created using InputStimulus.m
 * Version: 4/11/14
 * Author: Michael Beyeler <mbeyeler@uci.edu>
 *
 * This class provides a means to integrate CARLsim with a stimulus created using InputStimulus.m. The class reads
 * from binary file, and returns either the raw character array or a PoissonRate object with the same size as number
 * of pixels in a stimulus frame.
 * 
 * Common workflow:
 * \arg 1. Create a stimulus in Matlab:
 * \verbatim
 * >> IS = InputStimulus(32,32);
 * >> IS.addSinGrating
 * >> IS.saveToFile
 * \endverbatim
 * Successfully saved stimulus to file "inpGrating_gray_32x32x80.dat"
 *
 * \arg 2. In your main_$(example).cpp, read from binary file:
 * \code
 * InputStimulus IS("inpGrating_gray_32x32x80.dat");
 * int videoLength = IS.getStimulusLength();
 * \endcode
 *
 * \arg 3. Assign each frame to a SpikeGenerator group and run network
 * \code
 * for (int i=0; i<videoLength; i++) {
 *     PoissonRate * rates = IS.readFrame(50.0f); // grayscale value 255 will be mapped to 50 Hz
 *     snn.setSpikeRate(g1, rates); // for this to work, there must be 32x32=1024 neurons in g1
 *     snn.runNetwork(1,0); // run the network
 * }
 * \endcode
 */
class InputStimulus {
public:
	// +++++ PUBLIC METHODS: CONSTRUCTOR / DESTRUCTOR / MEMBERS +++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Default constructor
	 *
	 * Instantiates an InputStimulus object.
	 * \param[in] fileName        path to binary file that was created using InputStimulus.m
	 * \param[in] wrapAroundEOF   after reaching the end of the file, whether to jump back to the top and start reading
	 *                            the first frame again. If this flag is false and the EOF is reached, a warning will
	 *                            be printed. Default: true.
	 */
	InputStimulus(std::string fileName, bool wrapAroundEOF=true);

	//! default destructor
	~InputStimulus();

	/*!
	 * \brief List of stimulus file types
	 * 
	 * STIM_GRAY:      grayscale image
	 * STIM_RGB:       RGB image
	 * STIM_UNKNOWN:   unknown image type
	 */
	enum stimType_t {STIM_GRAY, STIM_RGB, STIM_UNKNOWN};
	

	// +++++ PUBLIC METHODS: READING FRAMES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	/*!
	 * \brief Reads the next image frame and returns a pointer to the char array
	 * 
	 * Advances the frame index by 1 (getCurrentFrameNumber) and returns the raw grayscale values for each pixel. The
	 * char array will have a total of getStimulusWidth()*getStimulusHeight()*getStimulusChannels() entries. The 
	 * order is columns first (so the first getStimulusWidth() number of pixels will correspond to the top image row).
	 * Currently, only grayscale images are supported.
	 * Note that this will advance the frame index. If you want to access the char array or PoissonRate object of a
	 * frame that has already been read, use getCurrentFrameChar() or getCurrentFramePoisson() instead.
	 *
	 * \returns  pointer to the char array of raw grayscale values
	 */
	unsigned char* readFrame();

	/*!
	 * \brief Reads the next image frame and returns a pointer to a PoissonRate object
	 *
	 * Advances the frame index by 1 (getCurrentFrameNumber) and returns a pointer to a PoissonRate object. The object
	 * will have a total of getStimulusWidth()*getStimulusHeight()*getStimulusChannels() entries. The order is columns
	 * first (so the first getStimulusWidth() number of pixels will correspond to the top image row). The PoissonRate
	 * object can then be applied to a group:
	 * PoissonRate* rates = IS.readFrame(50.0f);
	 * snn.setSpikeRate(g1, rates);
	 * Currently, only grayscale images are supported.
	 * Note that this will advance the frame index. If you want to access the char array or PoissonRate object of a
	 * frame that has already been read, use getCurrentFrameChar() or getCurrentFramePoisson() instead.
	 *
	 * \param[in] maxPoisson      the range of grayscale values [0,255] will be linearly mapped to the range of Poisson
	 *                            rates [0,maxPoisson]. Default: 50 Hz.
	 * \returns  pointer to a PoissonRate object
	 */
	PoissonRate* readFrame(float maxPoisson);

	/*!
	 * \brief Rewinds the file pointer to the top
	 *
	 * This function rewinds the file pointer back to the beginning of the file, so that the user can re-start
	 * reading the stimulus from the top.
	 */
	void rewind();


	// +++++ PUBLIC METHODS: GETTERS / SETTERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	int getStimulusWidth()  { return stimWidth_; } //!< returns the stimulus width (number of pixels)
	int getStimulusHeight() { return stimHeight_; } //!< returns the stimulus height (number of pixels)
	int getStimulusLength() { return stimLength_; } //!< returns the stimulus length (number of frames)
	int getStimulusChannels() { return stimChannels_; } //!< returns the number of channels (1=grayscale, 3=RGB)
	stimType_t getStimulusType() { return stimType_; } //!< returns the stimulus type (STIM_GRAY, STIM_RGB, etc.)

	unsigned char* getCurrentFrameChar() { return stimFrame_; } //!< returns char array of current frame
	PoissonRate* getCurrentFramePoisson() { return stimFramePoiss_; } //!< returns PoissonRate object of current frame
	int getCurrentFrameNumber() { return stimFrameNr_; } //! returns the current frame number (0-indexed)

private:
	// +++++ PRIVATE METHODS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	void readFramePrivate();		//!< reads the next frame
	void readHeader();				//!< reads the header section of the binary file


	// +++++ PRIVATE MEMBERS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

	FILE* fileId_;					//!< pointer to FILE stream
	std::string fileName_;			//!< file name
	int fileSignature_;				//!< a unique file signature used for InputStimulus files

	long fileHeaderSize_;			//!< the number of bytes in the header section
	bool wrapAroundEOF_;			//!< if EOF is reached, whether to start reading from the top

	unsigned char* stimFrame_;		//!< char array of current frame
	int stimFrameNr_;				//!< current frame index (0-indexed)

	PoissonRate* stimFramePoiss_;	//!< pointer to a PoissonRate object that contains the current frame

	int stimWidth_;					//!< stimulus width in number of pixels (neurons)
	int stimHeight_;				//!< stimulus height in number of pixels (neurons)
	int stimLength_;				//!< stimulus length in number of frames

	int stimChannels_;				//!< number of channels (1=grayscale, 3=RGB)
	stimType_t stimType_;			//!< stimulus type (grayscale, RGB, etc.)
};

#endif