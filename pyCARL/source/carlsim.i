// Include interace files if necessary

%include <std_string.i>
%include <std_vector.i>
%include <std_vectora.i>
%include <stl.i>
%include <std_shared_ptr.i>
%include <std_array.i>
%include <file.i>
%include <std_carray.i>
%include <std_alloc.i>
%include <std_container.i>
%include <std_list.i>

// SWIG must be explicitly told about the stl containers that are used

namespace std {
	%template(vectori) vector<int>;
	%template(vectord) vector<double>;
    %template(vectorf) vector<float>;
    %template(vectorvf) vector<vector<float>>;
};

// creating a module in SWIG and addign the required hearders to it

%module carlsim
%{

/* Put headers and other declarations here */
#include "../../carlsim/interface/inc/carlsim.h"
#include "../../carlsim/interface/inc/carlsim_datastructures.h"
#include "../../carlsim/interface/inc/carlsim_definitions.h"
#include "../../carlsim/interface/inc/callback.h"
#include "../../carlsim/interface/inc/poisson_rate.h"
#include "../../carlsim/monitor/spike_monitor.h"
#include "../../carlsim/monitor/connection_monitor.h"
#include "../../carlsim/monitor/group_monitor.h"
#include "../../carlsim/interface/inc/linear_algebra.h"
#include "../../carlsim/kernel/inc/snn.h"
#include "../../carlsim/kernel/inc/snn_datastructures.h"
#include "../../carlsim/kernel/inc/error_code.h"
#include "../../carlsim/kernel/inc/snn_definitions.h"
#include "../../carlsim/kernel/inc/spike_buffer.h"
//#include <../carlsim/kernel/inc/cuda_version_control.h>
#include "../../tools/spike_generators/spikegen_from_vector.h"
#include "../../tools/visual_stimulus/visual_stimulus.h"
%}

#include <stopwatch.h>
#include <time.h>
#include <stdio.h>

// explicitly menion the macros from C++ as the SWIG compiler needs to know about them 

#define SYN_FIXED      false
#define SYN_PLASTIC    true

#define TARGET_AMPA	(1 << 1)
#define TARGET_NMDA	(1 << 2)
#define UNKNOWN_NEURON  (0)
#define POISSON_NEURON  (1 << 0)
 
#define TARGET_GABAa    (1 << 3)
#define TARGET_GABAb    (1 << 4)
#define TARGET_DA       (1 << 5)
#define TARGET_5HT      (1 << 6) 
#define TARGET_ACh      (1 << 7)
#define TARGET_NE       (1 << 8)

#define INHIBITORY_NEURON       (TARGET_GABAa | TARGET_GABAb)
#define EXCITATORY_NEURON       (TARGET_NMDA | TARGET_AMPA)
#define DOPAMINERGIC_NEURON     (TARGET_DA | EXCITATORY_NEURON)
#define EXCITATORY_POISSON      (EXCITATORY_NEURON | POISSON_NEURON)
#define INHIBITORY_POISSON      (INHIBITORY_NEURON | POISSON_NEURON)
#define IS_INHIBITORY_TYPE(type)    (((type) & TARGET_GABAa) || ((type) & TARGET_GABAb))
#define IS_EXCITATORY_TYPE(type)    (!IS_INHIBITORY_TYPE(type))


#define ALL -1
#define ANY -1

// Enums, structs and other constants must be mentioned explicitly as the compiler shall be able to find them. any error stating unknown variable or enum , struct shall be included explicitly from the .h file in c++(carlsim) to here


enum STDPCurve {
	EXP_CURVE,           //!< standard exponential curve
	PULSE_CURVE,         //!< symmetric pulse curve
	TIMING_BASED_CURVE,  //!< timing-based curve
	UNKNOWN_CURVE        //!< unknown curve type
};



enum ComputingBackend {
	CPU_CORES,
	GPU_CORES
		};


enum STDPType {
	STANDARD,         //!< standard STDP of Bi & Poo (2001), nearest-neighbor
	DA_MOD,           //!< dopamine-modulated STDP, nearest-neighbor
	UNKNOWN_STDP
};
enum SpikeMonMode {
	COUNT,      //!< mode in which only spike count information is collected
	AER,        //!< mode in which spike information is collected in AER format
};
	
enum SimMode {
	CPU_MODE,     //!< model is run on CPU core(s)
	GPU_MODE,     //!< model is run on GPU card(s)
	HYBRID_MODE   //!< model is run on CPU Core(s), GPU card(s) or both
	};

enum LoggerMode {
	 USER,            //!< User mode, for experiment-oriented simulations.
	 DEVELOPER,       //!< Developer mode, for developing and debugging code.
	 SHOWTIME,        //!< Showtime mode, will only output warnings and errors.
	 SILENT,          //!< Silent mode, no output is generated.
	 CUSTOM,          //!< Custom mode, the user can set the location of all the file pointers.
	 UNKNOWN_LOGGER
};

enum Neuromodulator {
	NM_DA,		//!< dopamine
	NM_5HT,		//!< serotonin
	NM_ACh,		//!< acetylcholine
	NM_NE,		//!< noradrenaline
	NM_UNKNOWN	//!< unknown type
};

enum UpdateInterval {
	INTERVAL_10MS,		//!< the update interval will be 10 ms, which is 100Hz update frequency
	INTERVAL_100MS,		//!< the update interval will be 100 ms, which is 10Hz update frequency
	INTERVAL_1000MS		//!< the update interval will be 1000 ms, which is 1Hz update frequency
};

enum CARLsimState {
	CONFIG_STATE,		//!< configuration state, where the neural network is configured
	SETUP_STATE,		//!< setup state, where the neural network is prepared for execution and monitors are set
	RUN_STATE			//!< run state, where the model is stepped
};



//CARLsim class and function prototypes can be found here and addition to the carlsim class shall be made here
class CARLsim{
	
	public: 

///////////////////// creating carlsim object////////////////////// 
	CARLsim(const std::string& netName = "SNN", SimMode preferredSimMode = CPU_MODE, LoggerMode loggerMode = USER, int ithGPUs = 0, int randSeed = -1);
	~CARLsim();



// ////////////////creating groups and spikegenerator group////////////////////

	int createSpikeGeneratorGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);
	
	int createSpikeGeneratorGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	int createGroup(const std::string& grpName, int nNeur, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

	int createGroup(const std::string& grpName, const Grid3D& grid, int neurType, int preferredPartition = ANY, ComputingBackend preferredBackend = CPU_CORES);

    int getNumSynapticConnections(short int connectionId);
///////////////////// connection of the neuron groups /////////////////

	short int connect(int grpId1, int grpId2, const std::string& connType, const RangeWeight& wt, float connProb,
		const RangeDelay& delay=RangeDelay(1), const RadiusRF& radRF=RadiusRF(-1.0), bool synWtType=SYN_FIXED,
		float mulSynFast=1.0, float mulSynSlow=1.0);


	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, bool synWtType=SYN_FIXED);
	short int connect(int grpId1, int grpId2, ConnectionGenerator* conn, float mulSynFast, float mulSynSlow,
					bool synWtType=SYN_FIXED);

// ////////////////setting of conductances //////////////////////////////////////

	void setConductances(bool isSet);
	void setConductances(bool isSet, int tdAMPA, int tdNMDA, int tdGABAa, int tdGABAb);
	void setConductances(bool isSet, int tdAMPA, int trNMDA, int tdNMDA, int tdGABAa, int trGABAb, int tdGABAb);


////////////////// setting neuron parameters ///////////////

	void setNeuronParameters(int grpId, float izh_a, float izh_b, float izh_c, float izh_d) {
		std::string funcName = "setNeuronParameters(\""+getGroupName(grpId)+"\")";
		UserErrors::assertTrue(!isPoissonGroup(grpId), UserErrors::WRONG_NEURON_TYPE, funcName, funcName);
		UserErrors::assertTrue(carlsimState_==CONFIG_STATE, UserErrors::CAN_ONLY_BE_CALLED_IN_STATE, funcName, 
			funcName, "CONFIG.");

		// set standard deviations of Izzy params to zero
		snn_->setNeuronParameters(grpId, izh_a, 0.0, izh_b, 0.0, izh_c, 0.0, izh_d, 0.0);
	}

	void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1);



	/////////////setting of monitors /////////////////////////

	ConnectionMonitor* setConnectionMonitor(int grpIdPre, int grpIdPost, const std::string& fname);

	GroupMonitor* setGroupMonitor(int grpId, const std::string& fname);

	SpikeMonitor* setSpikeMonitor(int grpId, const std::string& fileName);

	


	

 /////////////// homeostatis and stdp parameter setting ///////////

	void setHomeostasis(int grpId, bool isSet);

	void setHomeostasis(int grpId, bool isSet, float homeoScale, float avgTimeScale);

	void setHomeoBaseFiringRate(int grpId, float baseFiring, float baseFiringSD=0.0f);

	void setNeuromodulator(int grpId, float baseDP, float tauDP, float base5HT, float tau5HT,
							float baseACh, float tauACh, float baseNE, float tauNE);

	void setESTDP(int grpId, bool isSet);		
	void setSTDP(int grpId, bool isSet);
	void setISTDP(int grpId, bool isSet);
	void setSTP(int grpId, bool isSet);
	void setESTDP(int grpId, bool isSet, STDPType type, ExpCurve curve);				
	void setSTDP(int grpId, bool isSet, STDPType type, float alphaPlus, float tauPlus, float alphaMinus, float tauMinus);
	void setESTDP(int grpId, bool isSet, STDPType type, TimingBasedCurve curve);
	void setISTDP(int grpId, bool isSet, STDPType type, ExpCurve curve);
	void setISTDP(int grpId, bool isSet, STDPType type, PulseCurve curve);
	void setSTP(int grpId, bool isSet, float STP_U, float STP_tau_u, float STP_tau_x);	

	void setWeightAndWeightChangeUpdate(UpdateInterval wtANDwtChangeUpdateInterval, bool enableWtChangeDecay, float wtChangeDecay=0.9f);
	
	void saveSimulation(const std::string& fileName, bool saveSynapseInfo=true);
	void biasWeights(short int connId, float bias, bool updateWeightRange=false);
	void loadSimulation(FILE* fid);
	void scaleWeights(short int connId, float scale, bool updateWeightRange=false);
	void setExternalCurrent(int grpId, const std::vector<float>& current);

	void setExternalCurrent(int grpId, float current);
	void setSpikeGenerator(int grpId, SpikeGenerator* spikeGenFunc);
        
        SpikeMonitor* getSpikeMonitor(int grpId);
	
        int getSimTime();
	int getSimTimeSec();
	int getSimTimeMsec();

	/////////////////// setup and run network /////////////////

    void startTesting(bool updateWeights=true);
	void stopTesting();
	void setupNetwork();
	int runNetwork(int nSec, int nMsec=0, bool printRunSummary=true);

};

/////// poisson rate class ////////////////

class PoissonRate {
public:
	PoissonRate(int nNeur, bool onGPU=false);
	~PoissonRate();
	
	int getNumNeurons();
	
	float getRate(int neurId);
	std::vector<float> getRates();
	void setRate(int neurId, float rate);
	void setRates(float rate);
	void setRates(const std::vector<float>& rates);
};

/////////////// ConnectionMonitor class ////////////
%extend ConnectionMonitor {
         std::vector<float> takeSnapshot1D(){
                std::vector< std::vector<float> > weights = self->takeSnapshot();
                int len = weights.size();
                int length = len * weights[0].size();
                std::vector<float> returnValues (length, 0);
                for (int i = 0; i < len; i++){
                        for (int j = 0; j < weights[0].size(); j++){
                                returnValues[len*i + j] = weights[i][j];
                        }
                }
		returnValues.push_back(weights[0].size());
                return returnValues;
        }
        void testPrint(std::vector<int> data){
                for (int x: data){
                        printf("%i\n", x);
                }
        }
}
class ConnectionMonitor {
 public:

	ConnectionMonitor(ConnectionMonitorCore* connMonCorePtr);

	~ConnectionMonitor();

	std::vector< std::vector<float> > calcWeightChanges();

	short int getConnectId();

	double getMaxWeight(bool getCurrent=false);

	double getMinWeight(bool getCurrent=false);

	int getNumNeuronsPre();

	int getNumNeuronsPost();

	int getNumSynapses();

	int getNumWeightsChanged(double minAbsChanged=1e-5);

	int getNumWeightsInRange(double minValue, double maxValue);

	int getNumWeightsWithValue(double value);

	double getPercentWeightsInRange(double minValue, double maxValue);

	double getPercentWeightsWithValue(double value);

	double getPercentWeightsChanged(double minAbsChanged=1e-5);

	long int getTimeMsCurrentSnapshot();

	long int getTimeMsLastSnapshot();

	long int getTimeMsSinceLastSnapshot();

	double getTotalAbsWeightChange();

	void print();

	void printSparse(int neurPostId=ALL, int maxConn=100, int connPerLine=4);


//	void setUpdateTimeIntervalSec(int intervalSec);

	std::vector< std::vector<float> > takeSnapshot();


};

///////////////////// GroupMonitor class ////////////////

class GroupMonitor {
 public:

	GroupMonitor(GroupMonitorCore* groupMonitorCorePtr);

	virtual ~GroupMonitor();

	bool isRecording();

	void startRecording();
	void stopRecording();

	int getRecordingTotalTime();

	int getRecordingLastStartTime();

	int getRecordingStartTime();

	int getRecordingStopTime();

	bool getPersistentData();

	void setPersistentData(bool persistentData);

	std::vector<float> getDataVector();

	std::vector<float> getPeakValueVector();

	std::vector<int> getPeakTimeVector();

	std::vector<float> getSortedPeakValueVector();

	std::vector<int> getSortedPeakTimeVector();

};

//////////////////////// SpikeMonitor class ////////////

class SpikeMonitor {
 public:

	SpikeMonitor(SpikeMonitorCore* spikeMonitorCorePtr);

	~SpikeMonitor();

	void clear();

	std::vector<float> getAllFiringRates();

	std::vector<float> getAllFiringRatesSorted();

	float getMaxFiringRate();

	float getMinFiringRate();

	float getNeuronMeanFiringRate(int neurId);

	int getNeuronNumSpikes(int neurId);

	int getNumNeuronsWithFiringRate(float min, float max);

	int getNumSilentNeurons();

	float getPercentNeuronsWithFiringRate(float min, float max);

	float getPercentSilentNeurons();

	float getPopMeanFiringRate();

	float getPopStdFiringRate();

	int getPopNumSpikes();

	std::vector<std::vector<int> > getSpikeVector2D();

	bool isRecording();

	void print(bool printSpikeTimes=true);

	void startRecording();

	void stopRecording();

	long int getRecordingTotalTime();

	long int getRecordingLastStartTime();

	long int getRecordingStartTime();

	long int getRecordingStopTime();

	bool getPersistentData();

	void setPersistentData(bool persistentData);

	SpikeMonMode getMode();

	void setMode(SpikeMonMode mode=AER);

	void setLogFile(const std::string& logFileName);

};


/////////////// SpikeGenerator class /////////////

class SpikeGenerator {
public:
	//SpikeGenerator() {};
    virtual ~SpikeGenerator() {}

	/*!
	 * \brief controls spike generation using a callback mechanism
	 *
	 * \attention The virtual method should never be called directly
	 * \param s pointer to the simulator object
	 * \param grpId the group id
	 * \param i the neuron index in the group
	 * \param currentTime the current simluation time
	 * \param lastScheduledSpikeTime the last spike time which was scheduled
	 * \param endOfTimeSlice the end of the current scheduling time slice. Spike times after this will not be scheduled.
	 */
	virtual int nextSpikeTime(CARLsim* s, int grpId, int i, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) = 0;
};

/*!
 * The user can choose from a set of primitive pre-defined connection topologies, or he can implement a topology of
 * their choice by using a callback mechanism. In the callback mechanism, the simulator calls a method on a user-defined
 * class in order to determine whether a connection should be made or not. The user simply needs to define a method that
 * specifies whether a connection should be made between a pre-synaptic neuron and a post-synaptic neuron, and the
 * simulator will automatically call the method for all possible pre- and post-synaptic pairs. The user can then specify
 * the connection's delay, initial weight, maximum weight, and whether or not it is plastic.
 */

//////////////// ConnectionGenerator class /////////////

class ConnectionGenerator {
public:
	// ConnectionGenerator() {}
    virtual ~ConnectionGenerator() {}
	/*!
	 * \brief specifies which synaptic connections (per group, per neuron, per synapse) should be made
	 *
	 * \attention The virtual method should never be called directly */
	virtual void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt,
							float& delay, bool& connected) = 0;

	


};
 
class SpikeGeneratorFromVector : public SpikeGenerator {
public:
    SpikeGeneratorFromVector(std::vector<int> spkTimes);
    ~SpikeGeneratorFromVector() {}

    int nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice);

private:
    void checkSpikeVector();
    std::vector<int> spkTimes_; 
    int currentIndex_;
    int size_;                  
};

SpikeGeneratorFromVector::SpikeGeneratorFromVector(std::vector<int> spkTimes) {
    spkTimes_ = spkTimes;
   size_ = spkTimes.size();
    currentIndex_ = 0;

    checkSpikeVector();
}

int SpikeGeneratorFromVector::nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) {

    // schedule spike if vector index valid and spike within scheduling time slice
    if (currentIndex_ < size_ && spkTimes_[currentIndex_] < endOfTimeSlice) {
        return spkTimes_[currentIndex_++];
    }

    return -1; // -1: large positive number
}

void SpikeGeneratorFromVector::checkSpikeVector() {
    UserErrors::assertTrue(size_>0,UserErrors::CANNOT_BE_ZERO, "SpikeGeneratorFromVector", "Vector size");
    for (int i=0; i<size_; i++) {
        std::stringstream var; var << "spkTimes[" << currentIndex_ << "]";
        UserErrors::assertTrue(spkTimes_[i]>0,UserErrors::CANNOT_BE_ZERO, "SpikeGeneratorFromVector", var.str());
    }   
}


class VisualStimulus {
public:
    VisualStimulus(std::string fileName, bool wrapAroundEOF=true);
    ~VisualStimulus();
    
    unsigned char* readFrameChar();
    PoissonRate* readFramePoisson(float maxPoisson, float minPoisson=0.0f);
    void rewind();
    void print();

	int getWidth();
	int getHeight();
	int getLength();
	int getChannels();
	stimType_t getType();

	unsigned char* getCurrentFrameChar();
	PoissonRate* getCurrentFramePoisson();
	int getCurrentFrameNumber();

private:
	class Impl;
	Impl* _impl;
};

/////////////// other data structures that are necessary for the complete library implementation that include 3d groups , time based curves , pulse based curves, stdp parameters , neuromodulators etc////////////////

struct Grid3D {
	Grid3D() : numX(-1), numY(-1), numZ(-1), N(-1),
	                 distX(-1.0f), distY(-1.0f), distZ(-1.0f),
	                 offsetX(-1.0f), offsetY(-1.0f), offsetZ(-1.0f) {
	}

    Grid3D(int _x) : numX(_x), numY(1), numZ(1), N(_x),
	                 distX(1.0f), distY(1.0f), distZ(1.0f),
	                 offsetX(1.0f), offsetY(1.0f), offsetZ(1.0f) {
        UserErrors::assertTrue(_x > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
    }

	Grid3D(int _x, float _distX, float _offsetX) : numX(_x), numY(1), numZ(1), N(_x),
	                                               distX(_distX), distY(1.0f), distZ(1.0f),
	                                               offsetX(_offsetX), offsetY(1.0f), offsetZ(1.0f) {
		UserErrors::assertTrue(_x > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
		UserErrors::assertTrue(_distX > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distX");
	}

    Grid3D(int _x, int _y) : numX(_x), numY(_y), numZ(1), N(_x * _y),
	                         distX(1.0f), distY(1.0f), distZ(1.0f),
	                         offsetX(1.0f), offsetY(1.0f), offsetZ(1.0f) {
        UserErrors::assertTrue(_x > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
        UserErrors::assertTrue(_y > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numY");
    }

	Grid3D(int _x, float _distX, float _offsetX, int _y, float _distY, float _offsetY)
		: numX(_x), numY(_y), numZ(1), N(_x * _y),
		  distX(_distX), distY(_distY), distZ(1.0f),
		  offsetX(_offsetX), offsetY(_offsetY), offsetZ(1.0f) {
		UserErrors::assertTrue(_x > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
		UserErrors::assertTrue(_distX > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distX");
		UserErrors::assertTrue(_y > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numY");
		UserErrors::assertTrue(_distY > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distY");
	}
    Grid3D(int _x, int _y, int _z) : numX(_x), numY(_y), numZ(_z), N(_x * _y * _z),
	                                 distX(1.0f), distY(1.0f), distZ(1.0f),
	                                 offsetX(1.0f), offsetY(1.0f), offsetZ(1.0f) {
         UserErrors::assertTrue(_x>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
         UserErrors::assertTrue(_y>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numY");
         UserErrors::assertTrue(_z>0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numZ");
    }
	Grid3D(int _x, float _distX, float _offsetX, int _y, float _distY, float _offsetY, int _z, float _distZ, float _offsetZ)
		: numX(_x), numY(_y), numZ(_z), N(_x * _y * _z),
		  distX(_distX), distY(_distY), distZ(_distZ),
		  offsetX(_offsetX), offsetY(_offsetY), offsetZ(_offsetZ) {
		UserErrors::assertTrue(_x > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numX");
		UserErrors::assertTrue(_distX > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distX");
		UserErrors::assertTrue(_y > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numY");
		UserErrors::assertTrue(_distY > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distY");
		UserErrors::assertTrue(_z > 0, UserErrors::MUST_BE_POSITIVE, "Grid3D", "numZ");
		UserErrors::assertTrue(_distZ > 0.0f, UserErrors::MUST_BE_POSITIVE, "Grid3D", "distZ");
	}

    

    int numX, numY, numZ;
    float distX, distY, distZ;
    float offsetX, offsetY, offsetZ;
    int N;
};

typedef struct GroupSTDPInfo_s {
	bool 		WithSTDP;			//!< enable STDP flag
	bool		WithESTDP;			//!< enable E-STDP flag
	bool		WithISTDP;			//!< enable I-STDP flag
	STDPType  WithESTDPtype;		//!< the type of E-STDP (STANDARD or DA_MOD)
	STDPType  WithISTDPtype;		//!< the type of I-STDP (STANDARD or DA_MOD)
	STDPCurve WithESTDPcurve;		//!< the E-STDP curve
	STDPCurve WithISTDPcurve;		//!< the I-STDP curve
	float		TAU_PLUS_INV_EXC;	//!< the inverse of time constant plus, if the exponential or timing-based E-STDP curve is used
	float		TAU_MINUS_INV_EXC;	//!< the inverse of time constant minus, if the exponential or timing-based E-STDP curve is used
	float		ALPHA_PLUS_EXC;		//!< the amplitude of alpha plus, if the exponential or timing-based E-STDP curve is used
	float		ALPHA_MINUS_EXC;	//!< the amplitude of alpha minus, if the exponential or timing-based E-STDP curve is used
	float		TAU_PLUS_INV_INB;	//!< the inverse of tau plus, if the exponential I-STDP curve is used
	float		TAU_MINUS_INV_INB;	//!< the inverse of tau minus, if the exponential I-STDP curve is used
	float		ALPHA_PLUS_INB;		//!< the amplitude of alpha plus, if the exponential I-STDP curve is used
	float		ALPHA_MINUS_INB;	//!< the amplitude of alpha minus, if the exponential I-STDP curve is used
	float		GAMMA;				//!< the turn over point if the timing-based E-STDP curve is used
	float		BETA_LTP;			//!< the amplitude of inhibitory LTP if the pulse I-STDP curve is used
	float		BETA_LTD;			//!< the amplitude of inhibitory LTD if the pulse I-STDP curve is used
	float		LAMBDA;				//!< the range of inhibitory LTP if the pulse I-STDP curve is used
	float		DELTA;				//!< the range of inhibitory LTD if the pulse I-STDP curve is used
} GroupSTDPInfo;


typedef struct GroupNeuromodulatorInfo_s {
	float		baseDP;		//!< baseline concentration of Dopamine
	float		base5HT;	//!< baseline concentration of Serotonin
	float		baseACh;	//!< baseline concentration of Acetylcholine
	float		baseNE;		//!< baseline concentration of Noradrenaline
	float		decayDP;		//!< decay rate for Dopaamine
	float		decay5HT;		//!< decay rate for Serotonin
	float		decayACh;		//!< decay rate for Acetylcholine
	float		decayNE;		//!< decay rate for Noradrenaline
} GroupNeuromodulatorInfo;


struct ExpCurve {
	ExpCurve(float _alphaPlus, float _tauPlus, float _alphaMinus, float _tauMinus) : alphaPlus(_alphaPlus), tauPlus(_tauPlus), alphaMinus(_alphaMinus), tauMinus(_tauMinus) {
		UserErrors::assertTrue(_tauPlus > 0.0f, UserErrors::MUST_BE_POSITIVE, "ExpCurve", "tauPlus");
		UserErrors::assertTrue(_tauMinus > 0.0f, UserErrors::MUST_BE_POSITIVE, "ExpCurve", "tauMinus");

		stdpCurve = EXP_CURVE;
	}

	STDPCurve stdpCurve; //!< the type of STDP curve
	float alphaPlus; //!< the amplitude of the exponential curve at pre-post side
	float tauPlus; //!< the time constant of the exponential curve at pre-post side
	float alphaMinus; //!< the amplitude of the exponential curve at post-pre side
	float tauMinus; //!< the time constant of the exponential curve at post-pre side
};

struct TimingBasedCurve {
	TimingBasedCurve(float _alphaPlus, float _tauPlus, float _alphaMinus, float _tauMinus, float _gamma) : alphaPlus(_alphaPlus), tauPlus(_tauPlus), alphaMinus(_alphaMinus), tauMinus(_tauMinus) , gamma(_gamma) {
		UserErrors::assertTrue(_alphaPlus > 0.0f, UserErrors::MUST_BE_POSITIVE, "TimingBasedCurve", "alphaPlus");
		UserErrors::assertTrue(_alphaMinus < 0.0f, UserErrors::MUST_BE_NEGATIVE, "TimingBasedCurve", "alphaMinus");
		UserErrors::assertTrue(_tauPlus > 0.0f, UserErrors::MUST_BE_POSITIVE, "TimingBasedCurve", "tauPlus");
		UserErrors::assertTrue(_tauMinus > 0.0f, UserErrors::MUST_BE_POSITIVE, "TimingBasedCurve", "tauMinus");
		UserErrors::assertTrue(_gamma > 0.0f, UserErrors::MUST_BE_POSITIVE, "TimingBasedCurve", "gamma");
		UserErrors::assertTrue(_tauPlus >= _gamma, UserErrors::CANNOT_BE_SMALLER, "TimingBasedCurve", "tauPlus >= gamma");

		stdpCurve = TIMING_BASED_CURVE;
	}

	STDPCurve stdpCurve; //!< the type of STDP curve
	float alphaPlus; //!< the amplitude of the exponential curve at pre-post side
	float tauPlus; //!< the time constant of the exponential curve at pre-post side
	float alphaMinus; //!< the amplitude of the exponential curve at post-pre side
	float tauMinus; //!< the time constant of the exponential curve at post-pre side
	float gamma; //!< the turn-over point
};

struct PulseCurve {
	PulseCurve(float _betaLTP, float _betaLTD, float _lambda, float _delta) : betaLTP(_betaLTP), betaLTD(_betaLTD), lambda(_lambda), delta(_delta) {
		UserErrors::assertTrue(_betaLTP > 0.0f, UserErrors::MUST_BE_POSITIVE, "PulseCurve", "betaLTP");
		UserErrors::assertTrue(_betaLTD < 0.0f, UserErrors::MUST_BE_NEGATIVE, "PulseCurve", "betaLTD");
		UserErrors::assertTrue(_lambda > 0.0f, UserErrors::MUST_BE_POSITIVE, "PulseCurve", "lambda");
		UserErrors::assertTrue(_delta > 0.0f, UserErrors::MUST_BE_POSITIVE, "PulseCurve", "delta");
		UserErrors::assertTrue(_lambda < _delta, UserErrors::MUST_BE_SMALLER, "PulseCurve", "lambda < delta");

		stdpCurve = PULSE_CURVE;
	}

	STDPCurve stdpCurve; //!< the type of STDP curve
	float betaLTP; //!< the amplitude of inhibitory LTP
	float betaLTD; //!< the amplitude of inhibitory LTD
	float lambda; //!< the range of inhibitory LTP
	float delta; //!< the range of inhibitory LTD
};




struct RangeWeight {
	RangeWeight(double _val) {
		init = _val;
		max = _val;
		min = 0;
	}
	RangeWeight(double _min, double _max) {
		UserErrors::assertTrue(_min<=_max, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "minWt", "maxWt");
		min = _min;
		init = _min;
		max = _max;
	}
	RangeWeight(double _min, double _init, double _max) {
		UserErrors::assertTrue(_min<=_init, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "minWt", "initWt");
		UserErrors::assertTrue(_init<=_max, UserErrors::CANNOT_BE_LARGER, "RangeWeight", "initWt", "maxWt");
		min = _min;
		init = _init;
		max = _max;
	}

	
	double min, init, max;
};



struct RangeDelay {
	RangeDelay(int _val) {
		min = _val;
		max = _val;
	}
	RangeDelay(int _min, int _max) {
		UserErrors::assertTrue(_min<=_max, UserErrors::CANNOT_BE_LARGER, "RangeDelay", "minDelay", "maxDelay");
		min = _min;
		max = _max;
	}

	
	int min,max;
};

struct RadiusRF {
	RadiusRF() : radX(0.0), radY(0.0), radZ(0.0) {}
	RadiusRF(double rad) : radX(rad), radY(rad), radZ(rad) {}
	RadiusRF(double rad_x, double rad_y, double rad_z) : radX(rad_x), radY(rad_y), radZ(rad_z) {}



	double radX, radY, radZ;
};
