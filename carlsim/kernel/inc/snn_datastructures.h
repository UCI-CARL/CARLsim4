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

#ifndef _SNN_DATASTRUCTURES_H_
#define _SNN_DATASTRUCTURES_H_

// include CUDA version-dependent macros and include files
#ifndef __NO_CUDA__
	#include <cuda_version_control.h>
	#include <curand.h>
#endif

/*!
* \brief type of memory pointer
*
* CARLsim supports execution either on standard x86 central processing units (CPUs) or off-the-shelf NVIDIA GPUs.
* The runtime data for CPU/GPU computing backend needs to be allocated in proper memory space.
*
* CPU_MEM: runtime data is allocated on CPU (main) memory
* GPU_MEM: runtime data is allocated on GPU memory
*
*/
enum MemType {
	CPU_MEM,     //!< runtime data is allocated on CPU (main) memory
	GPU_MEM,     //!< runtime data is allocated on GPU memory
};

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
	RadiusRF                 connRadius;
	float                    mulSynFast; //!< factor to be applied to either gAMPA or gGABAa
	float                    mulSynSlow; //!< factor to be applied to either gNMDA or gGABAb
	int                      connectionMonitorId; // ToDo: move to ConnectConfigMD
	uint32_t                 connProp;
	ConnectionGeneratorCore* conn;
	conType_t                type;
	float                    connProbability; //!< connection probability
	short int                connId; //!< connectID of the element in the linked list
	int                      numberOfConnections; // ToDo: move to ConnectConfigMD
} ConnectConfig;

/*!
 * \brief the intermediate data of connect config
 *
 * \note for futre use
 */
typedef struct ConnectConfigMD_s {
	ConnectConfigMD_s() : gConnId(-1), lConnId(-1), connectionMonitorId(-1), numberOfConnections(-1)
	{}
	int gConnId;
	int lConnId;
	int connectionMonitorId;
	int numberOfConnections;
} ConnectConfigMD;

/*!
* \brief The runtime configuration of a connection
*
* This structure contains the configurations of connections that are created by optimizeAndPartiionNetwork(),
* which is ready to be executed by computing backend.
* \see CARLsimState
* \see SNNState
*/
typedef struct ConnectConfigRT_s {
	float* mulSynFast; //!< factor to be applied to either gAMPA or gGABAa
	float* mulSynSlow; //!< factor to be applied to either gNMDA or gGABAb
} ConnectConfigRT;

typedef struct compConnectionInfo_s {
	int								grpSrc, grpDest;
	short int               		connId;
}compConnectionInfo;

/*!
* \brief The configuration of a compartmental connection
*
* This structure contains the configurations of compartmental connections that are created during configuration state.
* The configurations are later processed by compileNetwork() and translated to meta data which are ready to
* be linked.
* \see CARLsimState
*/
typedef struct compConnectConfig_s {
	int							grpSrc, grpDest;
	short int               	connId;
} compConnectConfig;

//!< neural dynamics configuration
typedef struct NeuralDynamicsConfig_s {
	NeuralDynamicsConfig_s() : Izh_a(-1.0f), Izh_a_sd(-1.0f), Izh_b(-1.0f), Izh_b_sd(-1.0f),
							   Izh_c(-1.0f), Izh_c_sd(-1.0f), Izh_d(-1.0f), Izh_d_sd(-1.0f),
							   Izh_C(-1.0f), Izh_C_sd(-1.0f), Izh_k(-1.0f), Izh_k_sd(-1.0f),
							   Izh_vr(-1.0f), Izh_vr_sd(1.0f), Izh_vt(1.0f), Izh_vt_sd(-1.0f),
							   Izh_vpeak(-1.0f), Izh_vpeak_sd(-1.0f), lif_tau_m(-1), 
							   lif_tau_ref(-1), lif_vTh(1.0f), lif_vReset(0.0f), lif_minRmem(1.0f),
							   lif_maxRmem(1.0f)
	{}
	float 		Izh_C;
	float 		Izh_C_sd;
	float 		Izh_k;
	float 		Izh_k_sd;
	float 		Izh_vr;
	float 		Izh_vr_sd;
	float 		Izh_vt;
	float 		Izh_vt_sd;
	float 		Izh_vpeak;
	float 		Izh_vpeak_sd;
	float 		Izh_a;
	float 		Izh_a_sd;
	float 		Izh_b;
	float 		Izh_b_sd;
	float 		Izh_c;
	float 		Izh_c_sd;
	float 		Izh_d;
	float 		Izh_d_sd;
	int 		lif_tau_m; //!< parameters for a LIF spiking group
	int 		lif_tau_ref;
	float 		lif_vTh;
	float 		lif_vReset;
	double		lif_minRmem;
	double		lif_maxRmem;
} NeuralDynamicsConfig;

//!< long-term plasiticity configurations
typedef struct STDPConfig_s {
	STDPConfig_s() : WithSTDP(false), WithESTDP(false), WithISTDP(false),
		WithESTDPtype(UNKNOWN_STDP), WithISTDPtype(UNKNOWN_STDP), WithESTDPcurve(UNKNOWN_CURVE), WithISTDPcurve(UNKNOWN_CURVE)
	{}
	bool WithSTDP;
	bool WithESTDP;
	bool WithISTDP;
	STDPType WithESTDPtype;
	STDPType WithISTDPtype;
	STDPCurve WithESTDPcurve;
	STDPCurve WithISTDPcurve;
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
} STDPConfig;

//!< short-term plasiticity configurations
typedef struct STPConfig_s {
	STPConfig_s() : WithSTP(false), STP_A(-1.0f), STP_U(-1.0f), STP_tau_u_inv(-1.0f), STP_tau_x_inv(-1.0f)
	{}

	bool WithSTP;
	float STP_A; // scaling factor
	float STP_U;
	float STP_tau_u_inv; // facilitatory
	float STP_tau_x_inv; // depressive
} STPConfig;

//!< homeostatic plasticity configurations
typedef struct HomeostasisConfig_s {
	HomeostasisConfig_s() : WithHomeostasis(false), baseFiring(-1.0f), baseFiringSD(-1.0f),
							avgTimeScale(-1.0f), avgTimeScaleInv(-1.0f), avgTimeScaleDecay(-1.0f),
							homeostasisScale(-1.0f)
	{}

	bool WithHomeostasis;
	float baseFiring;
	float baseFiringSD;
	float avgTimeScale;
	float avgTimeScaleInv;
	float avgTimeScaleDecay;
	float homeostasisScale;
} HomeostasisConfig;

//!< neuromodulator configurations
typedef struct NeuromodulatorConfig_s {
	NeuromodulatorConfig_s() : baseDP(1.0f), base5HT(1.0f), baseACh(1.0f), baseNE(1.0f),
							   decayDP(0.99f), decay5HT(0.99f), decayACh(0.99f), decayNE(0.99f)
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

/*!
 * \brief The configuration of a group
 *
 * This structure contains the configuration of groups that are created during configuration state.
 * The configurations are later processed by compileNetwork() and translated to meata data which are ready
 * to be linked.
 * \see CARLsimState
 */
typedef struct GroupConfig_s {
	GroupConfig_s() : grpName("N/A"), preferredNetId(-2), type(0), numN(-1), isSpikeGenerator(false), spikeGenFunc(NULL)
	{}

	// properties of neural group size and location
	std::string  grpName;
	int			 preferredNetId;
	unsigned int type;
	int          numN;
	bool isSpikeGenerator;
	bool withParamModel_9; //!< False = 4 parameter model; 1 = 9 parameter model.
	bool isLIF;
	bool withCompartments;

	float compCouplingUp;
	float compCouplingDown;

	SpikeGeneratorCore* spikeGenFunc;

	Grid3D grid; //<! location information of neurons
	NeuralDynamicsConfig neuralDynamicsConfig;
	STDPConfig stdpConfig;
	STPConfig stpConfig;
	HomeostasisConfig homeoConfig;
	NeuromodulatorConfig neuromodulatorConfig;
} GroupConfig;

typedef struct GroupConfigMD_s {
	GroupConfigMD_s() : gGrpId(-1), gStartN(-1), gEndN(-1),
						lGrpId(-1), lStartN(-1), lEndN(-1),
					    netId(-1), maxOutgoingDelay(1), fixedInputWts(true), hasExternalConnect(false),
						LtoGOffset(0), GtoLOffset(0), numPostSynapses(0), numPreSynapses(0), Noffset(0),
						spikeMonitorId(-1), neuronMonitorId(-1), groupMonitorId(-1), currTimeSlice(1000), sliceUpdateTime(0), homeoId(-1), ratePtr(NULL)
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
	int maxOutgoingDelay;
	bool fixedInputWts;
	bool hasExternalConnect;
	int spikeMonitorId;
	int neuronMonitorId;
	int groupMonitorId;
	float refractPeriod;
	int currTimeSlice; //!< timeSlice is used by the Poisson generators in order to not generate too many or too few spikes within a window of time
	int sliceUpdateTime;
	int homeoId;
	int Noffset; //!< the offset of spike generator (poisson) neurons [0, numNPois)
	PoissonRate* ratePtr;

	bool operator== (const struct GroupConfigMD_s& grp) {
		return (gGrpId == grp.gGrpId);
	}
} GroupConfigMD;

/*!
* \brief The runtime configuration of a group
*
* This structure contains the configurations of groups that are created by optimizeAndPartiionNetwork(),
* which is ready to be executed by computing backend.
* \see CARLsimState
* \see SNNState
*/
typedef struct GroupConfigRT_s {
	int          netId;         //!< published by GroupConfigMD \sa GroupConfigMD
	int          gGrpId;        //!< published by GroupConfigMD \sa GroupConfigMD
	int          gStartN;       //!< published by GroupConfigMD \sa GroupConfigMD
	int          gEndN;         //!< published by GroupConfigMD \sa GroupConfigMD
	int          lGrpId;        //!< published by GroupConfigMD \sa GroupConfigMD
	int          lStartN;       //!< published by GroupConfigMD \sa GroupConfigMD
	int          lEndN;         //!< published by GroupConfigMD \sa GroupConfigMD
	int          LtoGOffset;    //!< published by GroupConfigMD \sa GroupConfigMD
	int          GtoLOffset;    //!< published by GroupConfigMD \sa GroupConfigMD
	unsigned int Type;          //!< published by GroupConfig \sa GroupConfig
	int          numN;          //!< published by GroupConfig \sa GroupConfig
	int          numPostSynapses;   //!< the total number of post-connections of a group, published by GroupConfigMD \sa GroupConfigMD
	int          numPreSynapses;    //!< the total number of pre-connections of a group, published by GroupConfigMD \sa GroupConfigMD
	bool         isSpikeGenerator;  //!< published by GroupConfig \sa GroupConfig
	bool         isSpikeGenFunc;    //!< published by GroupConfig \sa GroupConfig
	bool         WithSTP;           //!< published by GroupConfig \sa GroupConfig
	bool         WithSTDP;          //!< published by GroupConfig \sa GroupConfig
	bool         WithESTDP;         //!< published by GroupConfig \sa GroupConfig
	bool         WithISTDP;         //!< published by GroupConfig \sa GroupConfig
	STDPType     WithESTDPtype;     //!< published by GroupConfig \sa GroupConfig
	STDPType     WithISTDPtype;     //!< published by GroupConfig \sa GroupConfig
	STDPCurve    WithESTDPcurve;    //!< published by GroupConfig \sa GroupConfig
	STDPCurve    WithISTDPcurve;    //!< published by GroupConfig \sa GroupConfig
	bool         WithHomeostasis;   //!< published by GroupConfig \sa GroupConfig
	bool         FixedInputWts;     //!< published by GroupConfigMD \sa GroupConfigMD
	bool         hasExternalConnect;//!< published by GroupConfigMD \sa GroupConfigMD
	int          Noffset;           //!< the offset of spike generator (poisson) neurons [0, numNPois), published by GroupConfigMD \sa GroupConfigMD
	int8_t       MaxDelay;          //!< published by GroupConfigMD \sa GroupConfigMD

	float        STP_A;             //!< published by GroupConfig \sa GroupConfig
	float        STP_U;             //!< published by GroupConfig \sa GroupConfig
	float        STP_tau_u_inv;     //!< published by GroupConfig \sa GroupConfig
	float        STP_tau_x_inv;     //!< published by GroupConfig \sa GroupConfig
	float        TAU_PLUS_INV_EXC;  //!< published by GroupConfig \sa GroupConfig
	float        TAU_MINUS_INV_EXC; //!< published by GroupConfig \sa GroupConfig
	float        ALPHA_PLUS_EXC;    //!< published by GroupConfig \sa GroupConfig
	float        ALPHA_MINUS_EXC;   //!< published by GroupConfig \sa GroupConfig
	float        GAMMA;             //!< published by GroupConfig \sa GroupConfig
	float        KAPPA;             //!< published by GroupConfig \sa GroupConfig
	float        OMEGA;             //!< published by GroupConfig \sa GroupConfig
	float        TAU_PLUS_INV_INB;  //!< published by GroupConfig \sa GroupConfig
	float        TAU_MINUS_INV_INB; //!< published by GroupConfig \sa GroupConfig
	float        ALPHA_PLUS_INB;    //!< published by GroupConfig \sa GroupConfig
	float        ALPHA_MINUS_INB;   //!< published by GroupConfig \sa GroupConfig
	float        BETA_LTP;          //!< published by GroupConfig \sa GroupConfig
	float        BETA_LTD;          //!< published by GroupConfig \sa GroupConfig
	float        LAMBDA;            //!< published by GroupConfig \sa GroupConfig
	float        DELTA;             //!< published by GroupConfig \sa GroupConfig

									//!< homeostatic plasticity variables
	float avgTimeScale;             //!< published by GroupConfig \sa GroupConfig
	float avgTimeScale_decay;       //!< published by GroupConfig \sa GroupConfig
	float avgTimeScaleInv;          //!< published by GroupConfig \sa GroupConfig
	float homeostasisScale;         //!< published by GroupConfig \sa GroupConfig

									// parameters of neuromodulator
	float baseDP;  //!< baseline concentration of Dopamine, published by GroupConfig \sa GroupConfig
	float base5HT; //!< baseline concentration of Serotonin, published by GroupConfig \sa GroupConfig
	float baseACh; //!< baseline concentration of Acetylcholine, published by GroupConfig \sa GroupConfig
	float baseNE;  //!< baseline concentration of Noradrenaline, published by GroupConfig \sa GroupConfig
	float decayDP; //!< decay rate for Dopaamine, published by GroupConfig \sa GroupConfig
	float decay5HT;//!< decay rate for Serotonin, published by GroupConfig \sa GroupConfig
	float decayACh;//!< decay rate for Acetylcholine, published by GroupConfig \sa GroupConfig
	float decayNE; //!< decay rate for Noradrenaline, published by GroupConfig \sa GroupConfig

	bool withParamModel_9; //!< False = 4 parameter model; 1 = 9 parameter model.
	bool isLIF; //!< True = a LIF spiking group

	bool withCompartments;
	float compCouplingUp;
	float compCouplingDown;
	int   compNeighbors[4];
	float compCoupling[4];
	short numCompNeighbors;
} GroupConfigRT;

typedef struct RuntimeData_s {
	unsigned int spikeCountSec;   //!< the total number of spikes in 1 second, used in CPU_MODE currently
	unsigned int spikeCountD1Sec; //!< the total number of spikes with axonal delay == 1 in 1 second, used in CPU_MODE currently	
	unsigned int spikeCountD2Sec; //!< the total number of spikes with axonal delay > 1 in 1 second, used in CPU_MODE currently
	unsigned int spikeCountExtRxD1Sec;
	unsigned int spikeCountExtRxD2Sec;
	unsigned int spikeCount;      //!< the total number of spikes in a simulation, used in CPU_MODE currently
	unsigned int spikeCountD1;    //!< the total number of spikes with anxonal delay == 1 in a simulation, used in CPU_MODE currently
	unsigned int spikeCountD2;    //!< the total number of spikes with anxonal delay > 1 in a simulation, used in CPU_MODE currently
	unsigned int nPoissonSpikes;  //!< the total number of spikes of poisson neurons, used in CPU_MODE currently
	unsigned int spikeCountLastSecLeftD2; //!< the nubmer of spike left in the last second, used in CPU_MODE currently
	unsigned int spikeCountExtRxD2; //!< the number of external spikes with axonal delay > 1 in a simulation, used in CPU_MODE currently
	unsigned int spikeCountExtRxD1; //!< the number of external spikes with axonal delay == 1 in a simulation, used in CPU_MODE currently

	float* voltage; //!< membrane potential for each regular neuron
	float* nextVoltage; //!< membrane potential buffer (next/future time step) for each regular neuron
	float* recovery;
	float* Izh_C;
	float* Izh_k;
	float* Izh_vr;
	float* Izh_vt;
	float* Izh_vpeak;
	float* Izh_a;
	float* Izh_b;
	float* Izh_c;
	float* Izh_d;
	float* current;
	float* totalCurrent;
	float* extCurrent;
	
	int* lif_tau_m; //!< parameters for a LIF spiking group
	int* lif_tau_ref;
	int* lif_tau_ref_c; // current refractory of the neuron
	float* lif_vTh;
	float* lif_vReset;
	float* lif_gain;
	float* lif_bias;

	//! Keeps track of all neurons that spiked at current time.
	//! Because integration step can be < 1ms we might want to keep integrating but remember that the neuron fired,
	//! so that we don't produce more than 1 spike per ms.
	bool* curSpike;

	// conductances and stp values
	float* gNMDA;   //!< conductance of gNMDA
	float* gNMDA_r;
	float* gNMDA_d;
	float* gAMPA;   //!< conductance of gAMPA
	float* gGABAa;  //!< conductance of gGABAa
	float* gGABAb;  //!< conductance of gGABAb
	float* gGABAb_r;
	float* gGABAb_d;

	int* I_set; //!< an array of bits indicating which synapse got a spike

	MemType memType;
	bool allocated; //!< true if all data has been allocated

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
	unsigned int* timeTableD1; //!< firing table, only used in CPU_MODE currently
	unsigned int* timeTableD2; //!< firing table, only used in CPU_MODE currently
	
	int* firingTableD1;
	int* firingTableD2;

	int** extFiringTableD1; //!< external firing table, only used on GPU
	int** extFiringTableD2; //!< external firing table, only used on GPU
	
	int* extFiringTableEndIdxD1;
	int* extFiringTableEndIdxD2;

	float* poissonFireRate;
	float* randNum;		//!< firing random number. max value is 10,000
#ifndef __NO_CUDA__
	int2* neuronAllocation;		//!< .x: [31:0] index of the first neuron, .y: [31:16] number of neurons, [15:0] group id
	int3* groupIdInfo;			//!< .x , .y: the start and end index of neurons in a group, .z: gourd id, used for group Id calculations
#endif

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

	// neuron monitor assistive buffers
	float* nVBuffer;
	float* nUBuffer;
	float* nIBuffer;

	unsigned int* spikeGenBits;
#ifndef __NO_CUDA__
	curandGenerator_t gpuRandGen;
#endif
} RuntimeData;

typedef struct GlobalNetworkConfig_s {
	GlobalNetworkConfig_s() : numN(0), numNReg(0), numNPois(0),
							  numNExcReg(0), numNInhReg(0), numNExcPois(0), numNInhPois(0),
							  numSynNet(0), maxDelay(-1), numN1msDelay(0), numN2msDelay(0),
							  simIntegrationMethod(FORWARD_EULER),
							  simNumStepsPerMs(2), timeStep(0.5)
	{}

	int numN;		  //!< number of neurons in the global network
	int numNExcReg;   //!< number of regular excitatory neurons in the global network
	int numNInhReg;   //!< number of regular inhibitory neurons in the global network
	int numNReg;      //!< number of regular (spking) neurons in the global network
	int numComp;	  //!< number of compartmental neurons
	int numNExcPois;  //!< number of excitatory poisson neurons in the global network
	int numNInhPois;  //!< number of inhibitory poisson neurons in the global network
	int numNPois;     //!< number of poisson neurons in the global network
	int numSynNet;    //!< number of total synaptic connections in the global network
	int maxDelay;	  //!< maximum axonal delay in the gloabl network
	int numN1msDelay; //!< number of neurons with maximum out going axonal delay = 1 ms
	int numN2msDelay; //!< number of neurons with maximum out going axonal delay >= 2 ms

	integrationMethod_t simIntegrationMethod; //!< integration method (forward-Euler or Fourth-order Runge-Kutta)
	int simNumStepsPerMs;					  //!< number of steps per 1 millisecond
	float timeStep;						      //!< inverse of simNumStepsPerMs
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
	int numN;         //!< number of neurons in th local network
	int numNExcReg;   //!< number of regular excitatory neurons
	int numNInhReg;   //!< number of regular inhibitory neurons
	int numNReg;      //!< number of regular (spking) neurons
	int numComp;	  //!< number of compartmental neurons
	int numNExcPois;  //!< number of excitatory poisson neurons
	int numNInhPois;  //!< number of inhibitory poisson neurons
	int numNPois;     //!< number of poisson neurons
	int numNSpikeGen; //!< number of poisson neurons generating spikes based on callback functions
	int numNRateGen;  //!< number of poisson neurons generating spikes based on firing rate
	int numNExternal; //!< number of external neurons in the view of the local network 
	int numNAssigned; //!< number of total neurons assigned to the local network

	// configurations for runtime data sizes
	unsigned int I_setLength; //!< used for GPU only
	size_t       I_setPitch;  //!< used for GPU only
	size_t       STP_Pitch;   //!< numN rounded upwards to the nearest 256 boundary, used for GPU only
	int numPostSynNet;        //!< the total number of post-connections in a network
	int numPreSynNet;         //!< the total number of pre-connections in a network
	int maxNumPostSynN;       //!< the maximum number of post-synaptic connections among neurons
	int maxNumPreSynN;        //!< the maximum number of pre-syanptic connections among neurons 
	unsigned int maxSpikesD2; //!< the estimated maximum number of spikes with delay >= 2 in a network
	unsigned int maxSpikesD1; //!< the estimated maximum number of spikes with delay == 1 in a network

	// configurations for assigned groups and connections
	int numGroups;        //!< number of local groups in this local network
	int numGroupsAssigned; //!< number of groups assigned to this local network
	int numConnections;   //!< number of local connections in this local network
	//int numAssignedConnections; //!< number of connections assigned to this local network

	// configurations for execution features
	bool sim_with_fixedwts;
	bool sim_with_conductances;
	bool sim_with_compartments;
	bool sim_with_stdp;
	bool sim_with_modulated_stdp;
	bool sim_with_homeostasis;
	bool sim_with_stp;
	bool sim_in_testing;
	
	// please note that spike monitor and connection monitor don't need this flag because no extra buffer is required
	// for neuron monitor, the kernel allocates extra buffers to store v, u, i values of each monitored neuron
	bool sim_with_nm; // simulation with neuron monitor

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

	integrationMethod_t simIntegrationMethod; //!< integration method (forward-Euler or Fourth-order Runge-Kutta)
	int simNumStepsPerMs;					  //!< number of steps per 1 millisecond
	float timeStep;						      //!< inverse of simNumStepsPerMs
} NetworkConfigRT;


//! runtime spike routing table entry
/*!
*	This structure contains the spike routing information, including source net id, source global group id,
*	destination net id, destination global group id
*/
typedef struct RoutingTableEntry_s {
	RoutingTableEntry_s() : srcNetId(-1), destNetId(-1)	{}

	RoutingTableEntry_s(int srcNetId_, int destNetId_) : srcNetId(srcNetId_), destNetId(destNetId_)	{}

	int srcNetId;
	int destNetId;

	bool operator== (const struct RoutingTableEntry_s& rte) const {
		return (srcNetId == rte.srcNetId && destNetId == rte.destNetId);
	}
} RoutingTableEntry;


//! CPU multithreading subroutine (that takes single argument) struct argument
/*!
*	This sturcture contains the snn object (because the 
*	multithreading routing is a static method and does not recognize this object), 
*	netID runtime used by the CPU runtime methods, local group ID, startIdx, 
*	endIdx, GtoLOffset
*/
typedef struct ThreadStruct_s {
	void* snn_pointer;
	int netId;
	int lGrpId;
	int startIdx;
	int endIdx;
	int GtoLOffset;
} ThreadStruct;

#endif
