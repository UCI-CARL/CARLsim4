/*
 * Copyright (c) 2014 Regents of the University of California. All rights reserved.
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
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */

#ifndef _CARLSIM_DATASTRUCTURES_H_
#define _CARLSIM_DATASTRUCTURES_H_

#include <ostream>			// print struct info
#include <user_errors.h>	// CARLsim user errors
#include <cmath>			// sqrt

/*!
 * \brief Logger modes
 *
 * The logger mode defines where to print all status, error, and debug messages. Several predefined
 * modes exist (USER, DEVELOPER, SHOWTIME, SILENT). However, the user can also set each file pointer to a
 * location of their choice (CUSTOM mode).
 * The following logger modes exist:
 *  USER 		User mode, for experiment-oriented simulations. Errors and warnings go to stderr,
 *              status information goes to stdout. Debug information can only be found in the log file.
 *  DEVELOPER   Developer mode, for developing and debugging code. Same as user, but additionally,
 *              all debug information is printed to stdout.
 *  SHOWTIME    Showtime mode, will only output warnings and errors. 
 *  SILENT      Silent mode, no output is generated.
 *  CUSTOM      Custom mode, the user can set the location of all the file pointers.
 *
 * The following file pointers exist:
 *  fpOut_	where CARLSIM_INFO messages go
 *  fpErr_ 	where CARLSIM_ERROR and CARLSIM_WARN messages go
 *  fpDeb_ 	where CARLSIM_DEBUG messages go
 *  fpLog_ 	typically a log file, where all of the above messages go
 *
 * The file pointers are automatically set to different locations, depending on the loggerMode:
 *
 * \verbatim
 *          |    USER    | DEVELOPER  |  SHOWTIME  |   SILENT   |  CUSTOM
 * ---------|------------|------------|------------|------------|---------
 * fpOut_   |   stdout   |   stdout   | /dev/null  | /dev/null  |    ?
 * fpErr_   |   stderr   |   stderr   |   stderr   | /dev/null  |    ?
 * fpDeb_   | /dev/null  |   stdout   | /dev/null  | /dev/null  |    ?
 * fpLog_   | debug.log  | debug.log  | debug.log  | /dev/null  |    ?
 * \endverbatim
 *
 * Location of the debug log file can be set in any mode using CARLsim::setLogDebugFp.
 * In mode CUSTOM, the other file pointers can be set using CARLsim::setLogsFp.
 */
enum loggerMode_t {
	 USER,  DEVELOPER,  SHOWTIME,  SILENT,  CUSTOM,  UNKNOWN_LOGGER
};
static const char* loggerMode_string[] = {
	"USER","DEVELOPER","SHOWTIME","SILENT","CUSTOM","Unknown mode"
};

/*!
 * \brief simulation mode
 *
 * CARLsim supports execution either on standard x86 central processing units (CPUs) or off-the-shelf NVIDIA GPUs.
 *
 * When creating a new CARLsim object, you can choose from the following:
 * CPU_MODE:	run on a single CPU core
 * GPU_MODE:	run on a single GPU card
 *
 * When running GPU mode on a multi-GPU system, you can specify on which CUDA device to establish a context (ithGPU,
 * 0-indexed) when you create a new CpuSNN object.
 * The simulation mode will be fixed throughout the lifetime of a CpuSNN object.
 */
enum simMode_t {
	 CPU_MODE,  GPU_MODE,  UNKNOWN_SIM
};
static const char* simMode_string[] = {
	"CPU mode","GPU mode","Unknown mode"
};

// \TODO: extend documentation, add relevant references
/*!
 * \brief STDP flavors
 *
 * CARLsim supports two different flavors of STDP.
 * STANDARD:	The standard model of Bi & Poo (2001), nearest-neighbor.
 * DA_MOD:      Dopamine-modulated STDP, nearest-neighbor.
 */
enum stdpType_t {
	 STANDARD,       DA_MOD,                   UNKNOWN_STDP
};
static const char* stdpType_string[] = {
	"Standard STDP","Dopamine-modulated STDP","Unknown mode"
};


/*!
 * \brief SpikeMonitor mode
 *
 * SpikeMonitors can be run in different modes:
 * COUNT:	Will collect only spike count information (such as number of spikes per neuron),
 *          not the explicit spike times. COUNT mode cannot retrieve exact spike times per
 *          neuron, and is thus not capable of computing spike train correlation etc.
 * AER:     Will collect spike information in AER format (will collect both neuron IDs and
 *          spike times).
 */
enum spikeMonMode_t {
	COUNT,            AER
};
static const char* spikeMonMode_string[] = {
	"SpikeCount Mode","SpikeTime Mode"
};


/*!
 * \brief Update frequency for weights
 *
 * CARLsim supports different update frequency for weight update and weightChange update
 * INTERVAL_10MS: the update interval will be 10 ms, which is 100Hz update frequency
 * INTERVAL_100MS: the update interval will be 100 ms, which is 10Hz update frequency
 * INTERVAL_1000MS: the update interval will be 1000 ms, which is 1Hz update frequency
 */
enum updateInterval_t {
	INTERVAL_10MS, INTERVAL_100MS, INTERVAL_1000MS
};
static const char* updateInterval_string[] = {
	"10 ms interval", "100 ms interval", "1000 ms interval"
};

/*!
 * \brief CARLsim states
 *
 * A CARLsim simulation goes through the following states:
 * CONFIG 		configuration state, where the neural network is configured
 * SETUP 		setup state, where the neural network is prepared for execution
 * EXECUTION 	execution state, where the simulation is executed
 * Certain methods can only be called in certain states. Check their documentation to see which method can be called
 * in which state.
 * Certain methods perform state transitions. CARLsim::setupNetwork will change the state from CONFIG to SETUP. The
 * first call to CARLsim::runNetwork will change the state from SETUP to EXECUTION.
 */
enum carlsimState_t {
	CONFIG_STATE, SETUP_STATE, EXE_STATE
};
static const char* carlsimState_string[] = {
	"Configuration state", "Setup state", "Execution state"
};

/*!
 * \brief a range struct for synaptic delays
 *
 * Synaptic delays can range between 1 and 20 ms. The struct maintains two fields: min and max.
 * \param[in] min the lower bound for delay values
 * \param[in] max the upper bound for delay values
 * Examples:
 *   RangeDelay(2) => all delays will be 2 (delay.min=2, delay.max=2)
 *   RangeDelay(1,10) => delays will be in range [1,10]
 */
struct RangeDelay {
	RangeDelay(int _val) : min(_val), max(_val) {}
	RangeDelay(int _min, int _max) : min(_min), max(_max) {
		UserErrors::assertTrue(_min<=_max, UserErrors::CANNOT_BE_LARGER, "RangeDelay", "minDelay", "maxDelay");
	}

	friend std::ostream& operator<<(std::ostream &strm, const RangeDelay &d) {
		return strm << "delay=[" << d.min << "," << d.max << "]";
	}
	const int min,max;
};

/*!
 * \brief a range struct for synaptic weight magnitudes
 *
 * Plastic synaptic weights are initialized to initWt, and can range between some minWt and some maxWt. Fixed weights
 * will always have the same value. All weight values should be non-negative (equivalent to weight *magnitudes*), even
 * for inhibitory connections.
 * \param[in] min the lower bound for weight values
 * \param[in] init the initial value for weight values
 * \param[in] max the upper bound for weight values
 * Examples:
 *   RangeWeight(0.1)         => all weights will be 0.1 (wt.min=0.1, wt.max=0.1, wt.init=0.1)
 *   RangeWeight(0.0,0.2)     => If pre is excitatory: all weights will be in range [0.0,0.2], and wt.init=0.0. If pre
 *                               is inhibitory: all weights will be in range [-0.2,0.0], and wt.init=0.0.
 *   RangeWeight(0.0,0.1,0.2) => If pre is excitatory: all weights will be in range [0.0,0.2], and wt.init=0.1. If pre
 *                               is inhibitory: all weights will be in range [-0.2,0.0], and wt.init=0.0.
 */
struct RangeWeight {
	RangeWeight(double _val) : init(_val), max(_val), min(0) {}
	RangeWeight(double _min, double _max) : min(_min), init(_min), max(_max) {
		UserErrors::assertTrue(_min<=_max, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "minWt", "maxWt");
	}
	RangeWeight(double _min, double _init, double _max) : min(_min), init(_init), max(_max) {
		UserErrors::assertTrue(_min<=_init, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "minWt", "initWt");
		UserErrors::assertTrue(_init<=_max, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "initWt", "maxWt");
	}

	friend std::ostream& operator<<(std::ostream &strm, const RangeWeight &w) {
		return strm << "wt=[" << w.min << "," << w.init << "," << w.max << "]";
	}
	double min, init, max; 
};

typedef struct GroupSTDPInfo_s {
	bool 		WithSTDP;
	bool		WithESTDP;
	bool		WithISTDP;
	stdpType_t  WithESTDPtype;
	stdpType_t  WithISTDPtype;
	float		TAU_LTP_INV;
	float		TAU_LTD_INV;
	float		ALPHA_LTP;
	float		ALPHA_LTD;
	float		BETA_LTP;
	float		BETA_LTD;
	float		LAMDA;
	float		DELTA;
} GroupSTDPInfo_t;

typedef struct GroupNeuromodulatorInfo_s {
	float		baseDP;		//!< baseline concentration of Dopamine
	float		base5HT;	//!< baseline concentration of Serotonin
	float		baseACh;	//!< baseline concentration of Acetylcholine
	float		baseNE;		//!< baseline concentration of Noradrenaline
	float		decayDP;		//!< decay rate for Dopaamine
	float		decay5HT;		//!< decay rate for Serotonin
	float		decayACh;		//!< decay rate for Acetylcholine
	float		decayNE;		//!< decay rate for Noradrenaline
} GroupNeuromodulatorInfo_t;

/*!
 * \brief struct to assign 3D coordinates to neurons in a group
 *
 * Neurons of a group can be arranged topographically, so that they virtually lie on a 3D grid. This simplifies the
 * creation of topographic connections in the network.
 * Each neuron thus gets assigned a (x,y,z) location on a grid (integer coordinates). Neuron numbers will be
 * assigned in order to location; where the first dimension specifies the width, the second dimension is height,
 * and the third dimension is depth. Grid3D(2,2,2) would thus assign neurId 0 to location (0,0,0), neurId 1 to (1,0,0),
 * neurId 3 to (0,1,0), neurId 6 to (2,2,1), and so on.
 * The third dimension can be thought of as a depth (z-coordinate in 3D), a cortical column (each of which consists
 * of a 2D arrangement of neurons on a plane), or a channel (such as RGB channels, each of which consists of a 2D
 * arrangements of neurons coding for (x,y) coordinates of an image). For the user's convenience, the struct thus
 * provides members Grid3D::depth, Grid3D::column, and Grid3D::channels, which differ only semantically.
 * \param[in] w the width of the 3D grid (1st dim)
 * \param[in] h the height of the 3D grid (2nd dim)
 * \param[in] z the depth of the 3D grid (3rd dim; also called column or channel)
 * Examples:
 *   Grid3D(10)         => creates 10 neurons on a 1D line, neurId=2 == (2,0,0), neurId=9 == (9,0,0)
 *   Grid3D(10,2)       => creates 10x2 neurons on a 2D plane, neurId=10 == (0,1,0), neurId=13 == (3,1,0)
 *   Grid3D(10,2,3)     => creates 10x2x3 neurons on a 3D grid, neurId=19 == (9,1,0), neurId=20 == (0,0,1)
 * Members:
 *   x, width	                   the width of the 3D grid (1st dim)
 *   y, height                     the height of the 3D grid (2nd dim)
 *   z, depth, columns, channels   the depth of the 3D grid (3rd dim)
 *   N                             the total number of neurons on the grid, N=x*y*z
 */
struct Grid3D {
    Grid3D(int w) : x(w), y(1), z(1), width(w), height(1), depth(1), columns(1), channels(1), N(w) {
        UserErrors::assertTrue(w>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "width");
    }
    Grid3D(int w, int h) : x(w), y(h), z(1), width(w), height(h), depth(1), columns(1), channels(1), N(w*h) {
        UserErrors::assertTrue(w>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "width");
        UserErrors::assertTrue(h>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "height");
    }
    Grid3D(int w, int h, int d) : x(w), y(h), z(d), width(w), height(h), depth(d), columns(d), channels(d), N(w*h*d) {
         UserErrors::assertTrue(w>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "width");
         UserErrors::assertTrue(h>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "height");
         UserErrors::assertTrue(d>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "depth");
    }

    friend std::ostream& operator<<(std::ostream &strm, const Grid3D &g) {
        return strm << "Grid3D=[" << g.x << "," << g.y << "," << g.z << "]";
    }

    int width, height, depth;
    int columns, channels;
    int x, y, z;
    int N;
};

/*!
 * \brief a point in 3D space
 *
 * A point in 3D space. Coordinates (x,y,z) are of double precision.
 * \param[in] x x-coordinate
 * \param[in] y y-coordinate
 * \param[in] z z-coordinate
 */
struct Point3D {
public:
	Point3D(int _x, int _y, int _z) : x(1.0*_x), y(1.0*_y), z(1.0*_z) {}
	Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

	// print struct info
    friend std::ostream& operator<<(std::ostream &strm, const Point3D &p) {
		strm.precision(2);
        return strm << "Point3D=(" << p.x << "," << p.y << "," << p.z << ")";
    }

    // overload operators
    Point3D operator+(const double a) const { return Point3D(x+a,y+a,z+a); }
    Point3D operator+(const Point3D& p) const { return Point3D(x+p.x,y+p.y,z+p.z); }
    Point3D operator-(const double a) const { return Point3D(x-a,y-a,z-a); }
    Point3D operator-(const Point3D& p) const { return Point3D(x-p.x,y-p.y,z-p.z); }
    Point3D operator*(const double a) const { return Point3D(x*a,y*a,z*a); }
    Point3D operator*(const Point3D& p) const { return Point3D(x*p.x,y*p.y,z*p.z); }
    Point3D operator/(const double a) const { return Point3D(x/a,y/a,z/a); }
    Point3D operator/(const Point3D& p) const { return Point3D(x/p.x,y/p.y,z/p.z); }
    bool operator==(const Point3D& p) const { return Equals(p); }
    bool operator!=(const Point3D& p) const { return !Equals(p); }
    bool operator<(const Point3D& p) const { return (CompareTo(p)<0); }
    bool operator>(const Point3D& p) const { return (CompareTo(p)>0); }
    bool operator<=(const Point3D& p) const { return (CompareTo(p)<=0); }
    bool operator>=(const Point3D& p) const { return (CompareTo(p)>=0); }
	
	// coordinates
	double x, y, z;

private:
	bool Equals(const Point3D& p) const { return (x==p.x && y==p.y); }
	int CompareTo(const Point3D& p) const { return (x>p.x&&y>p.y) ? 1 : ( (x<p.x&&y<p.y) ? -1 : 0); }
};


/*
// \TODO not sure where to put the following... they're functional, but carlsim_datastructures.h is not the
// right place...

//! calculate distance between two points \FIXME maybe move to carlsim_helper.h or something...
double dist(Point3D& p1, Point3D& p2) {
	Point3D p( (p1-p2)*(p1-p2) );
	return sqrt(p.x*p.x+p.y*p.y);
//	return norm(p); // can't find norm
}

//! calculate norm \FIXME maybe move to carlsim_helper.h or something...
double norm(Point3D& p) {
	return sqrt(p.x*p.x+p.y*p.y);
}

//! check whether certain point lies on certain grid \FIXME maybe move to carlsim_helper.h or something...
bool isPointOnGrid(Point3D& p, Grid3D& g) {
	// point needs to have non-negative coordinates
	if (p.x<0 || p.y<0 || p.z<0)
		return false;
		
	// point needs to have all integer coordinates
	if (floor(p.x)!=p.x || floor(p.y)!=p.y || floor(p.z)!=p.z)
		return false;
		
	// point needs to be within ranges
	if (p.x>=g.x || p.y>=g.y || p.z>=g.z)
		return false;
		
	// passed all tests
	return true;
}
*/

#endif