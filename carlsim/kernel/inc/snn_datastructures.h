/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 5/22/2015
 */

#ifndef _SNN_DATASTRUCTURES_H_
#define _SNN_DATASTRUCTURES_H_

// include CUDA version-dependent macros and include files
#include <cuda_version_control.h>
#include <curand.h>

//! connection types, used internally (externally it's a string)
enum conType_t { CONN_RANDOM, CONN_ONE_TO_ONE, CONN_FULL, CONN_FULL_NO_DIRECT, CONN_GAUSSIAN, CONN_USER_DEFINED, CONN_UNKNOWN};

//! the state of spiking neural network, used with in kernel.
enum SNNState {
	CONFIG_SNN,
	COMPILED_SNN,
	PARTITIONED_SNN,
	EXECUTABLE_SNN
};

typedef struct DelayInfo_s {
	short  delay_index_start;
	short  delay_length;
} DelayInfo;

typedef struct SynInfo_s {
	int	gsId; //!< group id and synapse id
	int nId; //!< neuron id
} SynInfo;

typedef struct ConnectionInfo_s {
	int grpSrc;
	int grpDest;
	int nSrc;
	int nDest;
	int srcGLoffset;
	float initWt;
	float maxWt;
	int preSynId;
	short int connId;
	uint8_t delay;

	bool operator== (const struct ConnectionInfo_s& conn) {
		return (nSrc + srcGLoffset == conn.nSrc);
	}
} ConnectionInfo;

/*!
 * \brief The configuration of a connection
 *
 * This structure contains the configurations of connections that are created during configuration state.
 * The configurations are later processed by compileNetwork() and translated to meta data which are ready to
 * be linked.
 * \see CARLsimState
 */
typedef struct ConnectConfig_s {
	int                      grpSrc;
	int                      grpDest;
	uint8_t                  maxDelay;
	uint8_t                  minDelay;
	float                    maxWt;
	float                    initWt;
	float                    minWt;
	float                    radX;
	float                    radY;
	float                    radZ;
	float                    mulSynFast;				//!< factor to be applied to either gAMPA or gGABAa
	float                    mulSynSlow;				//!< factor to be applied to either gNMDA or gGABAb
	int                      connectionMonitorId;
	uint32_t                 connProp;
	ConnectionGeneratorCore* conn;
	conType_t                type;
	float                    p; 						//!< connection probability
	short int                connId;					//!< connectID of the element in the linked list
	int                      numberOfConnections;
} ConnectConfig;

/*!
 * \brief The configuration of a group
 *
 * This structure contains the configuration of groups that are created during configuration state.
 * The configurations are later processed by compileNetwork() and translated to meata data which are ready
 * to be linked.
 * \see CARLsimState
 */
typedef struct GroupConfig_s {
	GroupConfig_s() : grpName("N/A"), preferredNetId(-2), type(0), numN(-1), isSpikeGenerator(false)
	{}

	// properties of neural group size and location
	std::string  grpName;
	int			 preferredNetId;
	unsigned int type;
	int          numN;
	bool isSpikeGenerator;

	Grid3D grid; //<! location information of neurons
	NeuralDynamicsConfig neuralDynamicsConfig;
	STPConfig stpConfig;
	HomeostasisConfig homeoConfig;
	NeuromodulatorConfig neuromodulatorConfig;
} GroupConfig;

typedef struct GroupConfigMD_s {
	GroupConfigMD_s() : gGrpId(-1), gStartN(-1), gEndN(-1),
						lGrpId(-1), lStartN(-1), lEndN(-1),
					    netId(-1), maxIncomingDelay(1), fixedInputWts(true), hasExternalConnect(false),
						LtoGOffset(0), GtoLOffset(0), numPostSynapses(0), numPreSynapses(0)
	{}

	int gGrpId;
	int gStartN;
	int gEndN;
	int netId;
	int lGrpId;
	int lStartN;
	int lEndN;
	int LtoGOffset;
	int GtoLOffset;
	int numPostSynapses;
	int numPreSynapses;
	int maxIncomingDelay;
	bool fixedInputWts;
	bool hasExternalConnect;

	bool operator== (const struct GroupConfigMD_s& grp) {
		return (gGrpId == grp.gGrpId);
	}
} GroupConfigMD;

//!< neural dynamics configuration
typedef struct NeuralDynamicsConfig_s {
	NeuralDynamicsConfig_s() : Izh_a(-1.0f), Izh_a_sd(-1.0f), Izh_b(-1.0f), Izh_b_sd(-1.0f),
							   Izh_c(-1.0f), Izh_c_sd(-1.0f), Izh_d(-1.0f), Izh_d_sd(-1.0f)
	{}
	float 		Izh_a;
	float 		Izh_a_sd;
	float 		Izh_b;
	float 		Izh_b_sd;
	float 		Izh_c;
	float 		Izh_c_sd;
	float 		Izh_d;
	float 		Izh_d_sd;
} NeuralDynamicsConfig;

//!< short-term plasiticity configurations
typedef struct STPConfig_s {
	STPConfig_s() : STP_A(-1.0f), STP_U(-1.0f), STP_tau_u_inv(-1.0f), STP_tau_x_inv(-1.0f)
	{}

	float STP_A; // scaling factor
	float STP_U;
	float STP_tau_u_inv; // facilitatory
	float STP_tau_x_inv; // depressive
} STPConfig;

//!< homeostatic plasticity configurations
typedef struct HomeostasisConfig_s {
	HomeostasisConfig_s() : baseFiring(-1.0f), baseFiringSD(-1.0f),
							avgTimeScale(-1.0f), avgTimeScaleDecay(-1.0f),
							homeostasisScale(-1.0f)
	{}

	float baseFiring;
	float baseFiringSD;
	float avgTimeScale;
	float avgTimeScaleDecay;
	float homeostasisScale;
} HomeostasisConfig;

//!< neuromodulator configurations
typedef struct NeuromodulatorConfig_s {
	NeuromodulatorConfig_s() : baseDP(-1.0f), base5HT(-1.0f), baseACh(-1.0f), baseNE(-1.0f),
							   decayDP(-1.0f), decay5HT(-1.0f), decayACh(-1.0f)
	{}

	float baseDP;   //!< baseline concentration of Dopamine
	float base5HT;  //!< baseline concentration of Serotonin
	float baseACh;  //!< baseline concentration of Acetylcholine
	float baseNE;   //!< baseline concentration of Noradrenaline
	float decayDP;  //!< decay rate for Dopamine
	float decay5HT; //!< decay rate for Serotonin
	float decayACh; //!< decay rate for Acetylcholine
	float decayNE;  //!< decay rate for Noradrenaline
} NeuromodulatorConfig;

typedef struct RuntimeData_s {
	float* voltage;
	float* recovery;
	float* Izh_a;
	float* Izh_b;
	float* Izh_c;
	float* Izh_d;
	float* current;
	float* extCurrent;

	// conductances and stp values
	float* gNMDA;   //!< conductance of gNMDA
	float* gNMDA_r;
	float* gNMDA_d;
	float* gAMPA;   //!< conductance of gAMPA
	float* gGABAa;  //!< conductance of gGABAa
	float* gGABAb;  //!< conductance of gGABAb
	float* gGABAb_r;
	float* gGABAb_d;

	int* I_set; //!< an array of bits indicating which synapse got a spike, only used on GPU

	SimMode	memType;
	bool     allocated; //!< true if all data has been allocated..

	/* Tsodyks & Markram (1998), where the short-term dynamics of synapses is characterized by three parameters:
	   U (which roughly models the release probability of a synaptic vesicle for the first spike in a train of spikes),
	   maxDelay_ (time constant for recovery from depression), and F (time constant for recovery from facilitation). */
	float* stpx;
	float* stpu;

	unsigned short*	Npre;				//!< stores the number of input connections to a neuron
	unsigned short*	Npre_plastic;		//!< stores the number of plastic input connections to a neuron
	float*          Npre_plasticInv;	//!< stores the 1/number of plastic input connections, only used on GPU
	unsigned short* Npost;				//!< stores the number of output connections from a neuron.

	int* lastSpikeTime; //!< stores the last spike time of a neuron
	int* synSpikeTime;  //!< stores the last spike time of a synapse

	float* wtChange; //!< stores the weight change of a synaptic connection
	float* wt;       //!< stores the weight change of a synaptic connection
	float* maxSynWt; //!< maximum synaptic weight for a connection
	
	unsigned int* cumulativePost;
	unsigned int* cumulativePre;

	short int* connIdsPreIdx; //!< connectId, per synapse, presynaptic cumulative indexing
	short int* grpIds;

	/*!
	 * \brief 10 bit syn id, 22 bit neuron id, ordered based on delay
	 *
	 * allows maximum synapses of 1024 and maximum network size of 4 million neurons, with 64 bit representation. we can
	 * have larger networks for simulation
	 */
	SynInfo* postSynapticIds;
	SynInfo* preSynapticIds;

	DelayInfo* postDelayInfo;  	//!< delay information

	int* firingTableD1;
	int* firingTableD2;

	int** extFiringTableD1; //!< external firing table, only used on GPU
	int** extFiringTableD2; //!< external firing table, only used on GPU
	
	int* extFiringTableEndIdxD1;
	int* extFiringTableEndIdxD2;

	float* poissonFireRate;
	float* gpuRandNums;		//!< firing random number. max value is 10,000

	int2* neuronAllocation;		//!< .x: [31:0] index of the first neuron, .y: [31:16] number of neurons, [15:0] group id
	int3* groupIdInfo;			//!< .x , .y: the start and end index of neurons in a group, .z: gourd id, used for group Id calculations

	int*  nSpikeCnt;

	//!< homeostatic plasticity variables
	float* baseFiringInv; //!< only used on GPU
	float* baseFiring;
	float* avgFiring;

	/*!
	 * neuromodulator concentration for each group
	 */
	float* grpDA;
	float* grp5HT;
	float* grpACh;
	float* grpNE;

	// group monitor assistive buffers
	float* grpDABuffer;
	float* grp5HTBuffer;
	float* grpAChBuffer;
	float* grpNEBuffer;

	unsigned int* spikeGenBits;

	curandGenerator_t gpuRandGen;
} RuntimeData;

typedef struct GlobalNetworkConfig_s {
	GlobalNetworkConfig_s() : numN(0), numNReg(0), numNPois(0),
							  numNExcReg(0), numNInhReg(0), numNExcPois(0), numNInhPois(0),
							  maxDelay(-1)
	{}

	int numN;		  //!< number of neurons in the global network
	int numNExcReg;   //!< number of regular excitatory neurons the global network
	int numNInhReg;   //!< number of regular inhibitory neurons the global network
	int numNReg;      //!< number of regular (spking) neurons the global network
	int numNExcPois;  //!< number of excitatory poisson neurons the global network
	int numNInhPois;  //!< number of inhibitory poisson neurons the global network
	int numNPois;     //!< number of poisson neurons the global network
	int maxDelay;	  //!< maximum axonal delay in the gloabl network
} GlobalNetworkConfig;

//! runtime network configuration
/*!
 *	This structure contains the network configuration that is required for GPU simulation.
 *	The data in this structure are copied to device memory when running GPU simulation.
 *	\sa SNN
 */
typedef struct NetworkConfigRT_s  {
	// global configuration for maximum axonal delay
	int maxDelay; //!< maximum axonal delay in the gloabl network

	// configurations for boundries of neural types
	int numN;         //!< number of neurons in the spiking neural network
	int numNExcReg;   //!< number of regular excitatory neurons
	int numNInhReg;   //!< number of regular inhibitory neurons
	int numNReg;      //!< number of regular (spking) neurons
	int numNExcPois;  //!< number of excitatory poisson neurons
	int numNInhPois;  //!< number of inhibitory poisson neurons
	int numNPois;     //!< number of poisson neurons
	int numNSpikeGen; //!< number of poisson neurons generating spikes based on callback functions
	int numNRateGen;  //!< number of poisson neurons generating spikes based on firing rate
	int numNExternal; //!< number of external neurons in the view of this local network 
	int numNAssigned; //!< number of total neurons assigned to this local network

	// configurations for runtime data sizes
	unsigned int I_setLength; //!< used for GPU only
	size_t       I_setPitch;  //!< used for GPU only
	size_t       STP_Pitch;   //!< numN rounded upwards to the nearest 256 boundary, used for GPU only
	int numPostSynNet;        //!< the total number of post-connections in a network
	int numPreSynNet;         //!< the total number of pre-connections in a network
	int maxNumPostSynN;       //!< the maximum number of post-synaptic connections among neurons
	int maxNumPreSynN;        //!< the maximum number of pre-syanptic connections among neurons 
	unsigned int maxSpikesD2; //!< the estimated maximum number of spikes with delay >= 2 in a network
	unsigned int maxSpikesD1; //!< the estimated maximum number of spikes with delay ===1 in a network

	// configurations for assigned groups and connections
	int numGroups;        //!< number of local groups in this local network
	int numAssignedGroups; //!< number of groups assigned to this local network
	int numConnections;   //!< number of local connections in this local network
	//int numAssignedConnections; //!< number of connections assigned to this local network

	// configurations for execution features
	bool sim_with_fixedwts;
	bool sim_with_conductances;
	bool sim_with_stdp;
	bool sim_with_modulated_stdp;
	bool sim_with_homeostasis;
	bool sim_with_stp;
	bool sim_in_testing;

	// stdp, da-stdp configurations
	float stdpScaleFactor;
	float wtChangeDecay;   //!< the wtChange decay

	// conductance configurations
	bool sim_with_NMDA_rise;  //!< a flag to inform whether to compute NMDA rise time
	bool sim_with_GABAb_rise; //!< a flag to inform whether to compute GABAb rise time
	double dAMPA;             //!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	double rNMDA;             //!< multiplication factor for rise time of NMDA
	double dNMDA;             //!< multiplication factor for decay time of NMDA
	double sNMDA;             //!< scaling factor for NMDA amplitude
	double dGABAa;            //!< multiplication factor for decay time of GABAa
	double rGABAb;            //!< multiplication factor for rise time of GABAb
	double dGABAb;            //!< multiplication factor for decay time of GABAb
	double sGABAb;            //!< scaling factor for GABAb amplitude
} NetworkConfigRT;

/*!
 * \brief The runtime configuration of a group
 *
 * This structure contains the configurations of groups that are created by optimizeAndPartiionNetwork(),
 * which is ready to be executed by computing backend.
 * \see CARLsimState
 * \see SNNState
 */
typedef struct GroupConfigRT_s {
	int          netId;
	int          preferedNetId; //!< user preference of netId
	int          grpId;
	int          StartN;
	int          EndN;
	int          localGrpId;
	int          localStartN;
	int          localEndN;
	int          LtoGOffset;
	int          GtoLOffset;
	unsigned int Type;
	int          SizeN;
    int          SizeX;
    int          SizeY;
    int          SizeZ;
	int          NumTraceN;
	short int    MaxFiringRate; //!< this is for the monitoring mechanism, it needs to know what is the maximum firing rate in order to allocate a buffer big enough to store spikes...
	int          SpikeMonitorId;		//!< spike monitor id
	int          GroupMonitorId; //!< group monitor id
	float        RefractPeriod;
	int          CurrTimeSlice; //!< timeSlice is used by the Poisson generators in order to not generate too many or too few spikes within a window of time
	int          SliceUpdateTime;
	int          numPostSynapses; //!< the total number of post-connections of a group
	int          numPreSynapses; //!< the total number of pre-connections of a group
	bool         isSpikeGenerator;
	bool         WithSTP;
	bool         WithSTDP;
	bool         WithESTDP;
	bool         WithISTDP;
	STDPType     WithESTDPtype;
	STDPType     WithISTDPtype;
	STDPCurve    WithESTDPcurve;
	STDPCurve    WithISTDPcurve;
	bool         WithHomeostasis;
	int          homeoId;
	bool         FixedInputWts;
	bool         hasExternalConnect;         
	int          Noffset;         //!< the offset of spike generator (poisson) neurons [0, numNPois)
	int8_t       MaxDelay;

	long int     lastSTPupdate;
	float        STP_A;
	float        STP_U;
	float        STP_tau_u_inv;
	float        STP_tau_x_inv;
	float        TAU_PLUS_INV_EXC;
	float        TAU_MINUS_INV_EXC;
	float        ALPHA_PLUS_EXC;
	float        ALPHA_MINUS_EXC;
	float        GAMMA;
	float        KAPPA;
	float        OMEGA;
	float        TAU_PLUS_INV_INB;
	float        TAU_MINUS_INV_INB;
	float        ALPHA_PLUS_INB;
	float        ALPHA_MINUS_INB;
	float        BETA_LTP;
	float        BETA_LTD;
	float        LAMBDA;
	float        DELTA;

	//!< homeostatic plasticity variables
	float avgTimeScale;
	float avgTimeScale_decay;
	float avgTimeScaleInv;
	float homeostasisScale;

	// parameters of neuromodulator
	float baseDP;  //!< baseline concentration of Dopamine
	float base5HT; //!< baseline concentration of Serotonin
	float baseACh; //!< baseline concentration of Acetylcholine
	float baseNE;  //!< baseline concentration of Noradrenaline
	float decayDP; //!< decay rate for Dopaamine
	float decay5HT;//!< decay rate for Serotonin
	float decayACh;//!< decay rate for Acetylcholine
	float decayNE; //!< decay rate for Noradrenaline

	bool 		writeSpikesToFile; 	//!< whether spikes should be written to file (needs SpikeMonitorId>-1)
	bool 		writeSpikesToArray;	//!< whether spikes should be written to file (needs SpikeMonitorId>-1)
	SpikeGeneratorCore*	spikeGenFunc;
	PoissonRate* RatePtr;
} GroupConfigRT;

/*!
 * \brief The runtime configuration of a connection
 *
 * This structure contains the configurations of connections that are created by optimizeAndPartiionNetwork(),
 * which is ready to be executed by computing backend.
 * \see CARLsimState
 * \see SNNState
 */
typedef struct ConnectConfigRT_s {
	int                      grpSrc;
	int                      grpDest;
	uint8_t                  maxDelay;
	uint8_t                  minDelay;
	float                    maxWt;
	float                    initWt;
	float                    minWt;
	float                    radX;
	float                    radY;
	float                    radZ;
	float                    mulSynFast;				//!< factor to be applied to either gAMPA or gGABAa
	float                    mulSynSlow;				//!< factor to be applied to either gNMDA or gGABAb
	int                      connectionMonitorId;
	uint32_t                 connProp;
	ConnectionGeneratorCore* conn;
	conType_t                type;
	float                    p; 						//!< connection probability
	short int                connId;					//!< connectID of the element in the linked list
	int                      numberOfConnections;
} ConnectConfigRT;

#endif
