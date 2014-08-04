/*
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 4/7/2014
 */

#ifndef _SNN_DATASTRUCTURES_H_
#define _SNN_DATASTRUCTURES_H_

#if __CUDA3__
	#include <cuda.h>
	#include <cutil_inline.h>
	#include <cutil_math.h>
#else
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <helper_cuda.h>
	#include <helper_functions.h>
	#include <helper_timer.h>
	#include <helper_math.h>
#endif


//! connection types, used internally (externally it's a string)
enum conType_t { CONN_RANDOM, CONN_ONE_TO_ONE, CONN_FULL, CONN_FULL_NO_DIRECT, CONN_USER_DEFINED, CONN_UNKNOWN};

typedef struct {
	short  delay_index_start;
	short  delay_length;
} delay_info_t;

typedef struct {
	int	postId;
	uint8_t	grpId;
} post_info_t;


//! network information structure
/*!
 *	This structure contains network information that is required for GPU simulation.
 *	The data in this structure are copied to device memory when running GPU simulation.
 *	\sa CpuSNN
 */
typedef struct network_info_s  {
	size_t			STP_Pitch;		//!< numN rounded upwards to the nearest 256 boundary
	unsigned int	numN;
	unsigned int	numPostSynapses;
	unsigned int	maxDelay;
	unsigned int	numNExcReg;
	unsigned int	numNInhReg;
	unsigned int	numNReg;
	unsigned int	I_setLength;
	size_t			I_setPitch;
	unsigned int	preSynLength;
//	unsigned int	numRandNeurons;
//	unsigned int	numNoise;
	unsigned int	postSynCnt;
	unsigned int	preSynCnt;
	unsigned int	maxSpikesD2;
	unsigned int	maxSpikesD1;
	unsigned int   	numNExcPois;
	unsigned int	numNInhPois;
	unsigned int	numNPois;
	unsigned int	numGrp;
	bool 			sim_with_fixedwts;
	bool 			sim_with_conductances;
	bool 			sim_with_stdp;
	bool 			sim_with_modulated_stdp;
	bool 			sim_with_homeostasis;
	bool 			sim_with_stp;
	float 			stdpScaleFactor;
	float 			wtChangeDecay; //!< the wtChange decay

	bool 			sim_with_NMDA_rise;	//!< a flag to inform whether to compute NMDA rise time
	bool 			sim_with_GABAb_rise;	//!< a flag to inform whether to compute GABAb rise time
	double 			dAMPA;				//!< multiplication factor for decay time of AMPA conductance (gAMPA[i] *= dAMPA)
	double 			rNMDA;				//!< multiplication factor for rise time of NMDA
	double 			dNMDA;				//!< multiplication factor for decay time of NMDA
	double 			sNMDA;				//!< scaling factor for NMDA amplitude
	double 			dGABAa;				//!< multiplication factor for decay time of GABAa
	double 			rGABAb;				//!< multiplication factor for rise time of GABAb
	double 			dGABAb;				//!< multiplication factor for decay time of GABAb
	double 			sGABAb;				//!< scaling factor for GABAb amplitude
} network_info_t;


//! connection infos...
typedef struct connectData_s {
	int 	  				grpSrc, grpDest;
	uint8_t	  				maxDelay,  minDelay;
	float	  				initWt, maxWt;
	float 					mulSynFast;				//!< factor to be applied to either gAMPA or gGABAa
	float 					mulSynSlow;				//!< factor to be applied to either gNMDA or gGABAb
	int	  	  				numPostSynapses;
	int	  	  				numPreSynapses;
	uint32_t  				connProp;
	ConnectionGeneratorCore*	conn;
	conType_t 				type;
	float					p; 						//!< connection probability
	short int				connId;					//!< connectID of the element in the linked list
	bool					newUpdates;
	int		   				numberOfConnections;
	struct connectData_s* next;
} grpConnectInfo_t;

typedef struct network_ptr_s  {
	float*	voltage;
	float*	recovery;
	float*	Izh_a;
	float*	Izh_b;
	float*	Izh_c;
	float*	Izh_d;
	float*	current;

	// conductances and stp values
	float*	gNMDA;					//!< conductance of gNMDA
	float*	gNMDA_r;
	float*	gNMDA_d;
	float*	gAMPA;					//!< conductance of gAMPA
	float*	gGABAa;				//!< conductance of gGABAa
	float*	gGABAb;				//!< conductance of gGABAb
	float*	gGABAb_r;
	float*	gGABAb_d;
	int*	I_set;
	simMode_t	memType;
	int		allocated;				//!< true if all data has been allocated..
	float*	stpx;
	float*	stpu;

	unsigned short*	Npre;				//!< stores the number of input connections to the neuron
	unsigned short*	Npre_plastic;		//!< stores the number of plastic input connections
	float*		Npre_plasticInv;	//!< stores the 1/number of plastic input connections, for use on the GPU
	unsigned short*	Npost;				//!< stores the number of output connections from a neuron.
	unsigned int*	lastSpikeTime;		//!< storees the firing time of the neuron
	float*	wtChange;
	float*	wt;				//!< stores the synaptic weight and weight change of a synaptic connection
	float*	maxSynWt;			//!< maximum synaptic weight for given connection..
	unsigned int*	synSpikeTime;
	unsigned int*	neuronFiring;
	unsigned int*	cumulativePost;
	unsigned int*	cumulativePre;

	float 	*mulSynFast, *mulSynSlow;
	short int *cumConnIdPre;	//!< connectId, per synapse, presynaptic cumulative indexing

	short int *grpIds;

	/*!
	 * \brief 10 bit syn id, 22 bit neuron id, ordered based on delay
	 *
	 * allows maximum synapses of 1024 and maximum network size of 4 million neurons, with 64 bit representation. we can
	 * have larger networks for simulation
	 */
	post_info_t*	postSynapticIds;

	post_info_t*	preSynapticIds;
	delay_info_t    *postDelayInfo;  	//!< delay information
	unsigned int*	firingTableD1;
	unsigned int*	firingTableD2;



	float*	poissonFireRate;
	unsigned int*	poissonRandPtr;		//!< firing random number. max value is 10,000
	int2*	neuronAllocation;		//!< .x: [31:0] index of the first neuron, .y: [31:16] number of neurons, [15:0] group id
	int3*	groupIdInfo;			//!< .x , .y: the start and end index of neurons in a group, .z: gourd id, used for group Id calculations
	short int*	synIdLimit;			//!<
	float*	synMaxWts;				//!<
	int*	nSpikeCnt;

	int** spkCntBuf; //!< for copying 2D array to GPU (see CpuSNN::allocateSNN_GPU)
	int* spkCntBufChild[MAX_GRP_PER_SNN]; //!< child pointers for above

	//!< homeostatic plasticity variables
	float*	baseFiringInv; // only used on GPU
	float*	baseFiring;
	float*	avgFiring;

	/*!
	 * neuromodulator concentration for each group
	 */
	float		*grpDA;
	float		*grp5HT;
	float		*grpACh;
	float		*grpNE;

	float* 		testVar;
	float*		testVar2;
	unsigned int*	spikeGenBits;
	bool*		curSpike;
} network_ptr_t;

typedef struct group_info_s
{
	// properties of group of neurons size, location, initial weights etc.
	PoissonRate*	RatePtr;
	int			StartN;
	int			EndN;
	unsigned int	Type;
	int			SizeN;
	int			NumTraceN;
	short int  	MaxFiringRate; //!< this is for the monitoring mechanism, it needs to know what is the maximum firing rate in order to allocate a buffer big enough to store spikes...
	int			SpikeMonitorId;		//!< spike monitor id
	int			GroupMonitorId; //!< group monitor id
	int			ConnectionMonitorId; //!< connection monitor id
	float   	RefractPeriod;
	int			CurrTimeSlice; //!< timeSlice is used by the Poisson generators in order to note generate too many or too few spikes within a window of time
	int			NewTimeSlice;
	uint32_t 	SliceUpdateTime;
	int 		FiringCount1sec;
	int 		numPostSynapses;
	int 		numPreSynapses;
	bool 		isSpikeGenerator;
	bool 		WithSTP;
	bool 		WithSTDP;
	stdpType_t  WithSTDPtype;
//	bool		WithModulatedSTDP;
	bool 		WithHomeostasis;
//	bool 		WithConductances;
	int		homeoId;
	bool		FixedInputWts;
	int			Noffset;
	int8_t		MaxDelay;

	long int    lastSTPupdate;
	float 		STP_A;
	float		STP_U;
	float		STP_tau_u_inv;
	float		STP_tau_x_inv;
	float		TAU_LTP_INV;
	float		TAU_LTD_INV;
	float		ALPHA_LTP;
	float		ALPHA_LTD;

	bool withSpikeCounter; //!< if this flag is set, we want to keep track of how many spikes per neuron in the group
	int spkCntRecordDur; //!< record duration, after which spike buffer gets reset
	int spkCntRecordDurHelper; //!< counter to help make fast modulo
	int spkCntBufPos; //!< which position in the spike buffer the group has

	//!< homeostatic plasticity variables
	float	avgTimeScale;
	float 	avgTimeScale_decay;
	float	avgTimeScaleInv;
	float	homeostasisScale;

	// parameters of neuromodulator
	float		baseDP;		//!< baseline concentration of Dopamine
	float		base5HT;	//!< baseline concentration of Serotonin
	float		baseACh;	//!< baseline concentration of Acetylcholine
	float		baseNE;		//!< baseline concentration of Noradrenaline
	float		decayDP;		//!< decay rate for Dopaamine
	float		decay5HT;		//!< decay rate for Serotonin
	float		decayACh;		//!< decay rate for Acetylcholine
	float		decayNE;		//!< decay rate for Noradrenaline

	bool 		writeSpikesToFile; 	//!< whether spikes should be written to file (needs SpikeMonitorId>-1)
	bool 		writeSpikesToArray;	//!< whether spikes should be written to file (needs SpikeMonitorId>-1)
	SpikeGeneratorCore*	spikeGen;
	bool		newUpdates;  //!< FIXME this flag has mixed meaning and is not rechecked after the simulation is started
} group_info_t;

/*!
 * this group need not be shared with the GPU
 * separate group which has unique properties of
 * neuron in the current group.
 */
typedef struct group_info2_s
{
  std::string		Name;
  short		ConfigId;
  // properties of group of neurons size, location, initial weights etc.
  //<! homeostatic plasticity variables
  float 		baseFiring;
  float 		baseFiringSD;
  float 		Izh_a;
  float 		Izh_a_sd;
  float 		Izh_b;
  float 		Izh_b_sd;
  float 		Izh_c;
  float 		Izh_c_sd;
  float 		Izh_d;
  float 		Izh_d_sd;

	/*!
	 * \brief when we call print state, should the group properties be printed.
	 * default is false and we do not want any prints for the current group
	 */
	bool		enablePrint;
	int			numPostConn;
	int			numPreConn;
	int			maxPostConn;
	int			maxPreConn;
	int			sumPostConn;
	int			sumPreConn;
} group_info2_t;


#endif
