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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 07/13/2013
 */

/*
 * Common Abbreviations used in CARLsim
 * snn/SNN: spiking neural network
 * stp/STP: short-term plasticity
 * stdp/STDP: spike-timing dependent plasticity
 * syn/SYN: synapse
 * wt/WT: weight
 * exc/Exc: excitatory
 * inh/Inh: inhibitory
 * grp: group
 * nid/NId/NID: neuron id
 * gid/GId/GID: group id
 * 
 * conn: connection
 * min: minimum
 * max: maximum
 * num: number
 * mem: memory
 * id/Id/ID: identification
 * info: information
 * cnt: count
 * curr: current
 * mon: monitor
 * reg/Reg: regurlar
 * pois/Pois: poisson
 */
#ifndef _SNN_GOLD_H_
#define _SNN_GOLD_H_

#include <iostream>
#include <string>
#include <map>
#include "mtrand.h"
#include "gpu_random.h"
#include "config.h"
#include "PropagatedSpikeBuffer.h"
#include "PoissonRate.h"
#include "SparseWeightDelayMatrix.h"

using std::string;
using std::map;

#if __CUDA3__
    #include <cuda.h>
    #include <cutil_inline.h>
    #include <cutil_math.h>
#elif __CUDA5__
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "helper_cuda.h"
    #include "helper_functions.h"
    #include "helper_timer.h"
    #include "helper_math.h"
#endif

#include "CUDAVersionControl.h"

extern RNG_rand48* gpuRand48; //!< Used by all network to generate global random number


#define ALL -1 //!< used for the set* methods to specify all groups and/or configIds

#define SYN_FIXED      0
#define SYN_PLASTIC    1

#define CPU_MODE 0
#define GPU_MODE 1

// Bit flags to be used to specify the type of neuron.  Future types can be added in the future such as Dopamine, etc.
// Yes, they should be bit flags because some neurons release more than one transmitter at a synapse.
#define UNKNOWN_NEURON	(0)
#define POISSON_NEURON	(1 << 0)
#define TARGET_AMPA	(1 << 1)
#define TARGET_NMDA	(1 << 2)
#define TARGET_GABAa	(1 << 3)
#define TARGET_GABAb	(1 << 4)
//#define TARGET_DA		(1 << 5)

#define INHIBITORY_NEURON 		(TARGET_GABAa | TARGET_GABAb)
#define EXCITATORY_NEURON 		(TARGET_NMDA | TARGET_AMPA)
#define EXCITATORY_POISSON 		(EXCITATORY_NEURON | POISSON_NEURON)
#define INHIBITORY_POISSON		(INHIBITORY_NEURON | POISSON_NEURON)
#define IS_INHIBITORY_TYPE(type)	(((type) & TARGET_GABAa) || ((type) & TARGET_GABAb))
#define IS_EXCITATORY_TYPE(type)	(!IS_INHIBITORY_TYPE(type))


#define CONN_SYN_NEURON_BITS	20                               //!< last 20 bit denote neuron id. 1 Million neuron possible
#define CONN_SYN_BITS			(32 -  CONN_SYN_NEURON_BITS)	 //!< remaining 12 bits denote connection id
#define CONN_SYN_NEURON_MASK    ((1 << CONN_SYN_NEURON_BITS) - 1)
#define CONN_SYN_MASK      		((1 << CONN_SYN_BITS) - 1)
#define GET_CONN_NEURON_ID(a) (((unsigned int)a.postId) & CONN_SYN_NEURON_MASK)
#define GET_CONN_SYN_ID(b)    (((unsigned int)b.postId) >> CONN_SYN_NEURON_BITS)
#define GET_CONN_GRP_ID(c)    (c.grpId)
//#define SET_CONN_ID(a,b)      ((b) > CONN_SYN_MASK) ? (fprintf(stderr, "Error: Syn Id exceeds maximum limit (%d)\n", CONN_SYN_MASK)): (((b)<<CONN_SYN_NEURON_BITS)+((a)&CONN_SYN_NEURON_MASK))


#define CONNECTION_INITWTS_RANDOM    	0
#define CONNECTION_CONN_PRESENT  		1
#define CONNECTION_FIXED_PLASTIC		2
#define CONNECTION_INITWTS_RAMPUP		3
#define CONNECTION_INITWTS_RAMPDOWN		4

#define SET_INITWTS_RANDOM(a)		((a & 1) << CONNECTION_INITWTS_RANDOM)
#define SET_CONN_PRESENT(a)		((a & 1) << CONNECTION_CONN_PRESENT)
#define SET_FIXED_PLASTIC(a)		((a & 1) << CONNECTION_FIXED_PLASTIC)
#define SET_INITWTS_RAMPUP(a)		((a & 1) << CONNECTION_INITWTS_RAMPUP)
#define SET_INITWTS_RAMPDOWN(a)		((a & 1) << CONNECTION_INITWTS_RAMPDOWN)

#define GET_INITWTS_RANDOM(a)		(((a) >> CONNECTION_INITWTS_RANDOM) & 1)
#define GET_CONN_PRESENT(a)		(((a) >> CONNECTION_CONN_PRESENT) & 1)
#define GET_FIXED_PLASTIC(a)		(((a) >> CONNECTION_FIXED_PLASTIC) & 1)
#define GET_INITWTS_RAMPUP(a)		(((a) >> CONNECTION_INITWTS_RAMPUP) & 1)
#define GET_INITWTS_RAMPDOWN(a)		(((a) >> CONNECTION_INITWTS_RAMPDOWN) & 1)

#define  checkNetworkBuilt()  {                                                         \
                if(!doneReorganization)  {                                              \
			DBG(0, fpLog, AT, "checkNetworkBuilt()");			\
                        fprintf(fpLog, "Network not yet elaborated and built...\n");    \
                        fprintf(stderr, "Network not yet elaborated and built...\n");   \
			return;                                                         \
                }                                                                       \
        }

/****************************/

#define STP_BUF_POS(nid, t)  ((nid) * STP_BUF_SIZE + ((t) % STP_BUF_SIZE))

/////    !!!!!!! IMPORTANT : NEURON ORGANIZATION/ARRANGEMENT MAP !!!!!!!!!!
////     <--- Excitatory --> | <-------- Inhibitory REGION ----------> | <-- Excitatory -->
///      Excitatory-Regular  | Inhibitory-Regular | Inhibitory-Poisson | Excitatory-Poisson
///      <--- numNExcReg --> | <-- numNInhReg --> | <-- numNInhPois -> | <---numNExcPois-->
///      <------REGULAR NEURON REGION ----------> | <----- POISSON NEURON REGION --------->
///      <----numNReg=(numNExcReg+numNInhReg)---> | <--numNPois=(numNInhPois+numNExcPois)->
////     <--------------------- ALL NEURONS ( numN=numNReg+numNPois) --------------------->
////	This organization scheme is only used/needed for the gpu_static code.
#define IS_POISSON_NEURON(nid, numNReg, numNPois) ((nid) >= (numNReg) && ((nid) < (numNReg+numNPois)))
#define IS_REGULAR_NEURON(nid, numNReg, numNPois) (((nid) < (numNReg)) && ((nid) < (numNReg+numNPois)))
#define IS_INHIBITORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) >= (numNExcReg)) && ((nid) < (numNReg + numNInhPois)))
#define IS_EXCITATORY(nid, numNInhPois, numNReg, numNExcReg, numN) (((nid) < (numNReg)) && (((nid) < (numNExcReg)) || ((nid) >=  (numNReg + numNInhPois))))

#if __CUDACC__
	inline bool isExcitatoryNeuron (unsigned int& nid, unsigned int& numNInhPois, unsigned int& numNReg, unsigned int& numNExcReg, unsigned int& numN)
	{
		return ((nid < numN) && ((nid < numNExcReg) || (nid >= numNReg + numNInhPois)));
	}
	inline bool isInhibitoryNeuron (unsigned int& nid, unsigned int& numNInhPois, unsigned int& numNReg, unsigned int& numNExcReg, unsigned int& numN)
	{
		return ((nid >= numNExcReg) && (nid < (numNReg + numNInhPois)));
	}
#endif

#define STATIC_LOAD_START(n)  (n.x)
#define STATIC_LOAD_GROUP(n)  (n.y & 0xff)
#define STATIC_LOAD_SIZE(n)   ((n.y >> 16) & 0xff)

#define MAX_NUMBER_OF_NEURONS_BITS  (20)
#define MAX_NUMBER_OF_GROUPS_BITS   (32 - MAX_NUMBER_OF_NEURONS_BITS)
#define MAX_NUMBER_OF_NEURONS_MASK  ((1 << MAX_NUMBER_OF_NEURONS_BITS) - 1)
#define MAX_NUMBER_OF_GROUPS_MASK   ((1 << MAX_NUMBER_OF_GROUPS_BITS) - 1)
#define SET_FIRING_TABLE(nid, gid)  (((gid) << MAX_NUMBER_OF_NEURONS_BITS) | (nid))
#define GET_FIRING_TABLE_NID(val)   ((val) & MAX_NUMBER_OF_NEURONS_MASK)
#define GET_FIRING_TABLE_GID(val)   (((val) >> MAX_NUMBER_OF_NEURONS_BITS) & MAX_NUMBER_OF_GROUPS_MASK)


//#define CHECK_CONNECTION_ID(n,total) { assert(n >= 0); assert(n < total); }

//Various callback functions

class CpuSNN;

//! used for fine-grained control over spike generation, using a callback mechanism
/*! Spike generation can be performed using spike generators. Spike generators are dummy-neurons that have their spikes
 * specified externally either defined by a Poisson firing rate or via a spike injection mechanism. Spike generators can
 * have post-synaptic connections with STDP and STP, but unlike Izhikevich neurons, they do not receive any pre-synaptic
 * input. For more information on spike generators see Section Neuron groups: Spike generators in the Tutorial.
 *
 * For fine-grained control over spike generation, individual spike times can be specified per neuron in each group.
 * This is accomplished using a callback mechanism, which is called at each time step, to specify whether a neuron has
 * fired or not. */
class SpikeGenerator {
	public:
		SpikeGenerator() {};

		//! controls spike generation using a callback
		/*! \attention The virtual method should never be called directly */
		virtual unsigned int nextSpikeTime(CpuSNN* s, int grpId, int i, unsigned int currentTime) { assert(false); return 0; }; // the virtual method should never be called directly
};

//! used for fine-grained control over spike generation, using a callback mechanism
/*!
 * The user can choose from a set of primitive pre-defined connection topologies, or he can implement a topology of
 * their choice by using a callback mechanism. In the callback mechanism, the simulator calls a method on a user-defined
 * class in order to determine whether a connection should be made or not. The user simply needs to define a method that
 * specifies whether a connection should be made between a pre-synaptic neuron and a post-synaptic neuron, and the
 * simulator will automatically call the method for all possible pre- and post-synaptic pairs. The user can then specify
 * the connection's delay, initial weight, maximum weight, and whether or not it is plastic.
 */
class ConnectionGenerator {
	public:
		ConnectionGenerator() {};

		//! specifies which synaptic connections (per group, per neuron, per synapse) should be made
		/*! \attention The virtual method should never be called directly */
		virtual void connect(CpuSNN* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt, float& delay, bool& connected) { assert(false); }; // the virtual method should never be called directly
};

class IzhGenerator {
	public:
		IzhGenerator() {};
		virtual void set(CpuSNN* s, int grpId, int i, float& a, float& b, float& c, float& d) {};
};

//! can be used to create a custom spike monitor
/*! To retrieve outputs, a spike-monitoring callback mechanism is used. This mechanism allows the user to calculate
 * basic statistics, store spike trains, or perform more complicated output monitoring. Spike monitors are registered
 * for a group and are called automatically by the simulator every second. Similar to an address event representation
 * (AER), the spike monitor indicates which neurons spiked by using the neuron ID within a group (0-indexed) and the
 * time of the spike. Only one spike monitor is allowed per group.*/
class SpikeMonitor {
	public:
		SpikeMonitor() {};

		//! Controls actions that are performed when certain neurons fire (user-defined).
		/*! \attention The virtual method should never be called directly */
		virtual void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts) {};
};


typedef struct {
    uint16_t	delay_index_start;
    uint16_t	delay_length;
} delay_info_t;

typedef struct {
	int32_t		postId;
	uint8_t		grpId;
} post_info_t;


//! network information structure
/*!
 *	This structure contains network information that is required for GPU simulation.
 *	The data in this structure are copied to device memory when running GPU simulation.
 *	\sa CpuSNN
 */ 
typedef struct network_info_s  {
	size_t		STP_Pitch;		//!< numN rounded upwards to the nearest 256 boundary
	uint32_t	numN;
	uint32_t	numPostSynapses;
	uint32_t	D;
	uint32_t	numNExcReg;
	uint32_t	numNInhReg;
	uint32_t	numNReg;
	uint32_t	I_setLength;
	size_t		I_setPitch;
	uint32_t	preSynLength;
//	uint32_t	numRandNeurons;
//	uint32_t	numNoise;
	uint32_t	postSynCnt;
	uint32_t	preSynCnt;
	uint32_t	maxSpikesD2;
	uint32_t	maxSpikesD1;
	uint32_t	numProbe;
	uint32_t   	numNExcPois;
	uint32_t	numNInhPois;
	uint32_t	numNPois;
	uint32_t	numGrp;
	bool		sim_with_fixedwts;
	bool		sim_with_conductances;
	bool		sim_with_stdp;
	bool		sim_with_stp;
} network_info_t;

inline post_info_t SET_CONN_ID(int nid, int sid, int grpId)
{
	if (sid > CONN_SYN_MASK) {
		fprintf(stderr, "Error: Syn Id (%d) exceeds maximum limit (%d) for neuron %d\n", sid, CONN_SYN_MASK, nid);
		assert(0);
	}
	post_info_t p;
	p.postId = (((sid)<<CONN_SYN_NEURON_BITS)+((nid)&CONN_SYN_NEURON_MASK));
	p.grpId  = grpId;
	return p;
}

typedef struct network_ptr_s  {
	float	*voltage;
	float	*recovery;
	float	*Izh_a;
	float	*Izh_b;
	float	*Izh_c;
	float	*Izh_d;
	float	*current;

	// conductances and stp values
	float	*gNMDA;					//!< conductance of gNMDA
	float	*gAMPA;					//!< conductance of gAMPA
	float	*gGABAa;				//!< conductance of gGABAa
	float	*gGABAb;				//!< conductance of gGABAb
	int		*I_set;
	int		memType;
	int		allocated;				//!< true if all data has been allocated..
	float	*stpx;
	float	*stpu;

	uint16_t	*Npre;				//!< stores the number of input connections to the neuron
	uint16_t	*Npre_plastic;		//!< stores the number of plastic input connections
	float		*Npre_plasticInv;	//!< stores the 1/number of plastic input connections, for use on the GPU
	uint16_t	*Npost;				//!< stores the number of output connections from a neuron.
	uint32_t	*lastSpikeTime;		//!< storees the firing time of the neuron
	float		*wtChange;
	float		*wt;				//!< stores the synaptic weight and weight change of a synaptic connection
	float		*maxSynWt;			//!< maximum synaptic weight for given connection..
	uint32_t	*synSpikeTime;
	uint32_t	*neuronFiring;
	uint32_t	*cumulativePost;
	uint32_t	*cumulativePre;

	/*!
	 * \brief 10 bit syn id, 22 bit neuron id, ordered based on delay
	 *
	 * allows maximum synapses of 1024 and maximum network size of 4 million neurons, with 64 bit representation. we can
	 * have larger networks for simulation
	 */
	post_info_t	*postSynapticIds;

	post_info_t	*preSynapticIds;
	delay_info_t    *postDelayInfo;  	//!< delay information
	uint32_t	*firingTableD1;
	uint32_t	*firingTableD2;
//	int*		randId;
//	void*		noiseGenProp;

	float		*probeV;
	float		*probeI;
	uint32_t	*probeId;

	float		*poissonFireRate;
	uint32_t	*poissonRandPtr;		//!< firing random number. max value is 10,000
	int2		*neuronAllocation;		//!< .x: [31:0] index of the first neuron, .y: [31:16] number of neurons, [15:0] group id
	int3		*groupIdInfo;			//!< .x , .y: the start and end index of neurons in a group, .z: gourd id, used for group Id calculations
	short int	*synIdLimit;			//!<
	float		*synMaxWts;				//!<

	uint32_t	*nSpikeCnt;

	float		*testVar;
	float		*testVar2;
	uint32_t	*spikeGenBits;
	bool		*curSpike;
} network_ptr_t;

typedef struct group_info_s
{
	// properties of group of neurons size, location, initial weights etc.
	PoissonRate	*RatePtr;
	int			StartN;
	int			EndN;
	char		Type;
	int			SizeN;
	int			NumTraceN;
	short int  	MaxFiringRate; //!< this is for the monitoring mechanism, it needs to know what is the maximum firing rate in order to allocate a buffer big enough to store spikes...
	int			MonitorId;
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
	bool 		WithConductances;
	bool		FixedInputWts;
	int			Noffset;
	int8_t		MaxDelay;

	float		STP_U;
	float		STP_tD;
	float		STP_tF;
	float		TAU_LTP_INV;
	float		TAU_LTD_INV;
	float		ALPHA_LTP;
	float		ALPHA_LTD;
	float		dAMPA;
	float		dNMDA;
	float		dGABAa;
	float		dGABAb;

	SpikeGenerator	*spikeGen;
	bool		newUpdates;  //!< FIXME this flag has mixed meaning and is not rechecked after the simulation is started
} group_info_t;

/*!
 * this group need not be shared with the GPU
 * separate group which has unique properties of
 * neuron in the current group.
 */
typedef struct group_info2_s
{
	string		Name;
	short		ConfigId;
	// properties of group of neurons size, location, initial weights etc.
	float 		Izh_a;		//!<
	float 		Izh_a_sd;
	float 		Izh_b;
	float 		Izh_b_sd;
	float 		Izh_c;
	float 		Izh_c_sd;
	float 		Izh_d;
	float 		Izh_d_sd;
	IzhGenerator*	IzhGen;

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

enum ConnType { CONN_RANDOM, CONN_ONE_TO_ONE, CONN_FULL, CONN_FULL_NO_DIRECT, CONN_USER_DEFINED, CONN_UNKNOWN};

//! connection information...
typedef struct connect_data_s {
	int 	  		grpSrc;				//!< group source
	int				grpDest;			//!< group destination
	uint8_t	  		maxDelay;			//!< maximum delay
	uint8_t			minDelay;			//!< minimum delay
	float	  		initWt;				//!< initial weight
	float			maxWt;				//!< maximum weight
	int	  	  		numPostSynapses;
	int	  	  		numPreSynapses;
	uint32_t  		connProp;
	ConnectionGenerator*		conn;
	ConnType 		type;
	float			p;
	int				connId;
	bool			newUpdates;
	int		   		numberOfConnections;
	struct connect_data_s* next;
} GroupConnectData;


#define MAX_GRPS_PER_BLOCK 		100
#define MAX_BLOCKS         		120
////////////////////////////////////
// member variable
////////////////////////////////////
typedef struct group_connect_info_s {
	int16_t		srcGrpId;		//!< group id
	int			srcStartN;		//!< starting neuron to begin computation
	int			srcEndN;		//!< ending neuron to stop computation
	int			grpDelayVector;	//!< a vector with ones in position having a given delay
	int			grpMaxM;		//!< the maximum value of the number of post-synaptic connections
	bool		hasCommonDelay; //!< 'true' if the grpDelayVector is same as the neuron DelayVector
	bool		hasRandomConn;	//!< set to 'true' if the group has random connections
	int*		randomDelayPointer; //
	int16_t		fixedDestGrpCnt;//!< destination group count
	int*		fixedDestGrps;	//!< connected destination groups array, (x=destGrpId, y=startN, z=endN, w=function pointer)
	int*		fixedDestParam;	//!< connected destination parameters ,  (x=Start, y=Width, z=Stride, w=height)
} GroupConnectInfo;



/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
class CpuSNN
{
	public:

		const static unsigned int MAJOR_VERSION = 2; //!< major release version, as in CARLsim X
		const static unsigned int MINOR_VERSION = 1; //!< minor release version, as in CARLsim 2.X

		//! SNN Constructor
		/*!
		 * \brief 
		 * \param _name the symbolic name of a spiking neural network
		 * \param _mode (optional) simulation mode, CPU_MODE: running simluation on CPUs, GPU_MODE: running simulation with GPU acceleration, default = CPU_MODE
		 * \param _randomize (optional) seed of the random number generator, default = 0. 
		 * \param _numConfig (optional, deprecated) the number of configurations, default = 1.
		 */
		CpuSNN(const string& _name, int _mode = CPU_MODE, int _randomize = 0, int _numConfig = 1);

		//! SNN Destructor
		/*!
		 * \brief clean up all allocated resource 
		 */
		~CpuSNN();

		//! Creates a group of Izhikevich spiking neurons
		/*!
		 * \param _name the symbolic name of a group
		 * \param _numN  nubmer of neurons in the group
		 * \param _nType the type of neuron
		 * \param _configId (optional, deprecated) configuration id
		 */
		int createGroup(const string& _name, unsigned int _numN, int _nType, int _configId = ALL);

		//! Creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons)
		/*!
		 * \param _name the symbolic name of a group
		 * \param _size_n  nubmer of neurons in the group
		 * \param _nType the type of neuron, currently only support EXCITATORY NEURON
		 * \param _configId (optional, deprecated) configuration id
		 */
		int createSpikeGeneratorGroup(const string& _name, int unsigned _numN, int _nType, int _configId = ALL);

		//! Sets the Izhikevich parameters a, b, c, and d of a neuron group.
		/*!
		 * \brief Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d and standard deviation _a_sd, _b_sd, _c_sd, and _d_sd, respectively
		 * \param _groupId the symbolic name of a group
		 * \param _a  the mean value of izhikevich parameter a
		 * \param _a_sd the standard deviation value of izhikevich parameter a
		 * \param _b  the mean value of izhikevich parameter b
		 * \param _b_sd the standard deviation value of izhikevich parameter b
		 * \param _c  the mean value of izhikevich parameter c
		 * \param _c_sd the standard deviation value of izhikevich parameter c
		 * \param _d  the mean value of izhikevich parameter d
		 * \param _d_sd the standard deviation value of izhikevich parameter d
		 * \param _configId (optional, deprecated) configuration id
		 */
		void setNeuronParameters(int _groupId, float _a, float _a_sd, float _b, float _b_sd, float _c, float _c_sd, float _d, float _d_sd, int _configId = ALL);

		//! Sets the Izhikevich parameters a, b, c, and d of a neuron group.
		/*!
		 * \brief Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d and standard deviation _a_sd, _b_sd, _c_sd, and _d_sd, respectively
		 * \param _groupId the symbolic name of a group
		 * \param _a  the mean value of izhikevich parameter a
		 * \param _b  the mean value of izhikevich parameter b
		 * \param _c  the mean value of izhikevich parameter c
		 * \param _d  the mean value of izhikevich parameter d
		 * \param _configId (optional, deprecated) configuration id
		 */
		void setNeuronParameters(int _groupId, float _a, float _b, float _c, float _d, int _configId = ALL);
		
		//! Sets the Izhikevich parameters of a neuron group with IzhGenerator class.
		/*!
		 * \brief Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d and standard deviation _a_sd, _b_sd, _c_sd, and _d_sd, respectively
		 * \param _groupId the symbolic name of a group
		 * \param _IzhGen the IzhGenerator
		 * \param _configId (optional, deprecated) configuration id
		 * \sa IzhGenerator
		 */
		void setNeuronParameters(int _groupId, IzhGenerator* _IzhGen, int _configId = ALL);

		void setGroupInfo(int _groupId, group_info_t _info, int _configId = ALL);

		group_info_t getGroupInfo(int _groupId, int _configId = 0);
		group_info2_t getGroupInfo2(int _groupId, int _configId = 0);

		//! Creates a graphical representation of the network topology and stores it in a dotty file (where the file name corresponds to the network name _name specified in CpuSNN()
		void printDotty ();

		void CpuSNNInit(unsigned int _numN, unsigned int _numPostSynapses, unsigned int _numPreSynapses, unsigned int _D);

		//! Creates synaptic projections from a pre-synaptic group to a post-synaptic group using a pre-defined primitive type.
		/*!
		 * \brief make from each neuron in grpId1 to 'numPostSynapses' neurons in grpId2
		 *
		 * \param _grpIdPre ID of the pre-synaptic group 
		 * \param _grpIdPost ID of the post-synaptic group 
		 * \param _connType connection type. "random": random connectivity. "one-to-one": connect the i-th neuron in pre to the i-th neuron in post. "full": connect all neurons in pre to all neurons in post (no self-connections). 
		 * \param _initWt initial weight strength (arbitrary units); should be negative for inhibitory connections 
		 * \param _maxWt upper bound on weight strength (arbitrary units); should be negative for inhibitory connections 
		 * \param _connProb connection probability 
		 * \param _minDelay the minimum delay allowed (ms) 
		 * \param _maxdelay: the maximum delay allowed (ms) 
		 * \param _synWtType: (optional) connection type, either SYN_FIXED or SYN_PLASTIC, default = SYN_FIXED. 
		 * \param _wtType: (optional) DEPRECATED
		 * \return number of created synaptic projections
		 */
		int connect(int _grpIdPre, int _grpIdPost, const string& _connType, float _initWt, float _maxWt, float _connProb, uint8_t _minDelay, uint8_t _maxDelay, bool _synWtType = SYN_FIXED, const string& _wtType = " ");

		/* Creates synaptic projections using a callback mechanism.
		 *
		 * \param _grpIdPre:ID of the pre-synaptic group 
		 * \param _grpIdPost ID of the post-synaptic group 
		 * \param _conn: pointer to an instance of class ConnectionGenerator 
		 * \param _synWtType: (optional) connection type, either SYN_FIXED or SYN_PLASTIC, default = SYN_FIXED
		 * \param _maxPostM: (optional) maximum number of post-synaptic connections (per neuron), Set to 0 for no limit, default = 0
		 * \param _maxPreM: (optional) maximum number of pre-synaptic connections (per neuron), Set to 0 for no limit, default = 0. 
		 * \return number of created synaptic projections
		 */
		int connect(int _grpIdPre, int _grpIdPost, ConnectionGenerator* _conn, bool _synWtType = SYN_FIXED, int _maxPostM = 0, int _maxPreM = 0);

		//! Run a simulation. Note, network must be correctly specified and instantiated.
		/*!
		 * \brief run the simulation for _nsec second(s) and _nmsec millisecond(s)				 
		 * \param _nsec: number of seconds to run 
		 * \param _nmsec: (optional) number of milliseconds to run, default = 0
		 * \param _ithGPU: (optional) specify on which CUDA device to establish a context, default = 0 (the first GPU)
		 * \param _enablePrint: (optional) enable the printing of essential neuronal information (such as current membrane potential, recovery variable, synaptic conductances, etc.), default = false.
		 * \param _copyState: (optional) enable the copying of essential neuronal information, default = false.
		 * \return 
		 */
		int runNetwork(int _nsec, int _nmsec = 0, int _ithGPU = 0, bool _enablePrint = false, int _copyState = false);

		bool updateTime(); //!< returns true when a new second is started

		uint64_t getSimTime();

		uint32_t getSimTimeSec();

		uint32_t getSimTimeMs();

		// grpId == -1, means all groups
		//! Disable the spike-timing-dependent plasticity (STDP) for a neuron group.  
		void setSTDP(int _grpId, bool _enable, int _configId = ALL);

		//! Set the spike-timing-dependent plasticity (STDP) for a neuron group.  
		/*
		 * \brief STDP must be defined post-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1, call setSTP on group 1. Fore details on the phenomeon, see (for example) (Bi & Poo, 2001).
		 * \param _grpId ID of the neuron group 
		 * \param _enable set to true to enable STDP for this group 
		 * \param _ALPHA_LTP max magnitude for LTP change 
		 * \param _TAU_LTP decay time constant for LTP 
		 * \param _ALPHA_LTD max magnitude for LTD change (leave positive) 
		 * \param _TAU_LTD decay time constant for LTD
		 * \param _configId (optional, deprecated) configuration id
		 */
		void setSTDP(int _grpId, bool _enable, float _ALPHA_LTP, float _TAU_LTP, float _ALPHA_LTD, float _TAU_LTD, int configId = ALL);

		// grpId == -1, means all groups
		//! Disable the short-term plasticity (STP) for a neuron group.
		void setSTP(int _grpId, bool _enable, int _configId = ALL);
		
		//! Set the short-term plasticity (STP) for a neuron group. 
		/*! STP must be defined pre-synaptically; that is, if STP should be implemented on the connections from group 0 to group 1, call setSTP on group 0. For details on the phenomenon, see (for example) (Markram et al, 1998), (Mongillo et al, 2008). 
		 * \param _grpId ID of the neuron group 
		 * \param _enable set to true to enable STP for this group 
		 * \param _STP_U the increment of u due to a spike 
		 * \param _STP_tD time constant for depression term 
		 * \param _STP_tF: time constant for faciliation term 
		 * \param _configId (optional, deprecated) configuration id
		 */
		void setSTP(int _grpId, bool _enable, float _STP_U, float _STP_tD, float _STP_tF, int configId = ALL);

		// grpId == -1, means all groups

		//! Disable the use of synaptic conductanes for a group
		void setConductances(int _grpId, bool _enable, int _configId = ALL);

		//! Sets the decay time constants for synaptic conductances of a neuron group. To disable the use of synaptic conductanes for a group, you can also use a simple wrapper: void setConductances(int grpId, bool enable). To apply conductance values to all groups, replace grpId with ALL.
		/* Synaptic channels are modeled with instantaneous rise-time and exponential decay.
		 * For details on the ODE that is implemented refer to (Izhikevich et al, 2004), and for suitable values see (Dayan & Abbott, 2001). 
		 * 
		 * \param _grpId: ID of the neuron group 
		 * \param enable: enables the use of COBA mode 
		 * \param tAMPA: time _constant of AMPA decay (ms); for example, 5.0 
		 * \param tNMDA: time constant of NMDA decay (ms); for example, 150.0 
		 * \param tGABAa: time constant of GABAa decay (ms); for example, 6.0 
		 * \param tGABAb: time constant of GABAb decay (ms); for example, 150.0 
		 * \param configId: (optional, deprecated) configuration id
		 */
		void setConductances(int _grpId, bool _enable, float _tAMPA, float _tNMDA, float _tGABAa, float _tGABAb, int _configId = ALL);

		//! sets up a spike monitor registered with a callback to process the spikes, there can only be one
		/*! SpikeMonitor per group
		 * \param _grpId ID of the neuron group
		 * \param _spikeMon (optional) spikeMonitor class
		 * \param _configId (optional, deprecated) configuration id, default = ALL
		 */
		void setSpikeMonitor(int _grpId, SpikeMonitor* _spikeMon = NULL, int _configId = ALL);

		//! a simple wrapper that uses a predetermined callback to save the data to a file
		/*! Set up a spike monitor.
		 * \param _grpId ID of the neuron group 
		 * \param _fname (optional) file name where to output spikes. Leave empty to just print to stdout 
		 * \param _configId (optional, deprecated) configuration id, default = 0
		 */
		void setSpikeMonitor(int _grpId, const string& _fileName, int _configId = 0);

		//!Sets the Poisson spike rate for a group. For information on how to set up spikeRate, see Section Poisson spike generators in the Tutorial. 
		/*!Input arguments:
		 * \param _grpId ID of the neuron group 
		 * \param spikeRate pointer to a PoissonRate instance 
		 * \param refPeriod (optional) refractive period,  default = 1
		 * \param _configId (optional, deprecated) configuration id, default = ALL
		 */
		void setSpikeRate(int _grpId, PoissonRate* _spikeRate, int _refPeriod = 1, int _configId = ALL);

		void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId = ALL);

		//! stores the pre and post synaptic neuron ids with the weight and delay
		/*
		 * \param _fp: file pointer
		 */
		void writeNetwork(FILE* _fp); 

		//! Reads a CARLsim network file. Such a file can be created using CpuSNN:writeNetwork.
		/*
		 * \brief After calling CpuSNN::readNetwork, you should run CpuSNN::runNetwork before calling fclose(_fp).
		 * \param _fp: file pointer
		 * \sa CpuSNN::writeNetwork()
		 */
		void readNetwork(FILE* _fp);

#if READNETWORK_ADD_SYNAPSES_FROM_FILE
		int readNetwork_internal(bool onlyPlastic);
#else
		int readNetwork_internal();
#endif
		//! Returns the weight strengths for all synaptic connections between a pre-synaptic and a post-synaptic neuron group.
		/*!
		 * \param _grpIdPre ID of pre-synaptic group 
		 * \param _grpIdPost ID of post-synaptic group 
		 * \param _nPre return the number of pre-synaptic neurons 
		 * \param _nPost retrun the number of post-synaptic neurons 
		 * \param _weights (optional) return the weight strengths for all synapses, default = NULL
		 * \return weight strengths for all synapses
		 */
		float* getWeights(int _grpIdPre, int _grpIdPost, int& _nPre, int& _nPost, float* _weights = NULL);
		
		//! Returns the change in weight strength in the last second (due to plasticity) for all synaptic connections between a pre-synaptic and a post-synaptic neuron group.
		/*!
		 * \param _grpIdPre ID of pre-synaptic group 
		 * \param _grpIdPost ID of post-synaptic group 
		 * \param _nPre return the number of pre-synaptic neurons 
		 * \param _nPost retrun the number of post-synaptic neurons 
		 * \param _weightChanges (optional) return changes in weight strength for all synapses, default = NULL
		 * \return changes in weight strength for all synapses
		 */
		float* getWeightChanges(int _grpIdPre, int _grpIdPost, int& _nPre, int& _nPost, float* _weightChanges = NULL);

		//! Returns the delay information for all synaptic connections between a pre-synaptic and a post-synaptic neuron group
		/*!
		 * \param _grpIdPre ID of pre-synaptic group 
		 * \param _grpIdPost ID of post-synaptic group 
		 * \param _nPre return the number of pre-synaptic neurons 
		 * \param _nPost retrun the number of post-synaptic neurons 
		 * \param _delays (optional) return the delay information for all synapses, default = NULL
		 * \return delay information for all synapses
		 */
		uint8_t* getDelays(int _grpIdPre, int _grpIdPost, int& _nPre, int& _nPost, uint8_t* _delays = NULL);
		
		//! Print a simulation summary to file
		/*
		 * \param _fp (optional) file pointer, default = stdout
		 */
		void printSimSummary(FILE *_fp = stdout);

		//! Print memory information to file.
		/*
		 * \param _fp (optional) file pointer, default = stdout
		 */
		void printMemoryInfo(FILE* _fp = stdout);

		void printTuningLog();

		void setTuningLog(const string& fname);

		//! Set the update cycle for log messages
		/*!
		 * \param _cnt to disable logging, set to 0 
		 * \param _logMode (optional) the higher this number, the more logging information will be printed, default = 0
		 * \param _fp (optional) file pointer for log info, default = NULL
		 */
		void setLogCycle(unsigned int _cnt, int _logMode = 0, FILE *_fp = NULL);

		void setProbe(int g, const string& type, int startId=0, int cnt=1, uint32_t _printProbe=0);

		int  grpStartNeuronId(int g) { return grp_Info[g].StartN; }

		int  grpEndNeuronId(int g)   { return grp_Info[g].EndN;   }

		int  grpNumNeurons(int g)    { return grp_Info[g].SizeN;  }

		void plotProbes();

		int getNumConfigurations() {
			return numConfig;
		}
		int  getGroupId(int groupId, int configId);
		int  getConnectionId(int connId, int configId);


		void setPrintState(int grpId, bool _status, int neuronId=-1)
		{
			grp_Info2[grpId].enablePrint = _status;
		}
		void printState(const char *str = "");
		void printWeight(int grpId, const char *str = "");
		void printNeuronState(int grpId, FILE *_fp = stderr);

		void setSimLogs(bool enable, string logDirName = "") {
			enableSimLogs = enable;
			if (logDirName != "") {
				simLogDirName = logDirName;
			}
		}

		void printParameters(FILE *_fp);

		void printGroupInfo(FILE* _fp);

		void printGroupInfo(string& strName);

		void printConnectionInfo(FILE* _fp);

		void printConnectionInfo2(FILE *fpg);

		void printGroupInfo2(FILE* fpg);

		int printPostConnection2(int grpId, FILE* fpg);

		int printPreConnection2(int grpId, FILE* fpg);

		void printFiringRate(char *fname=NULL);

		void printNetworkInfo();

		//! print all the connections...
		void printPostConnection(FILE *_fp = stdout);

		void printPreConnection(FILE  *_fp = stdout);

		void printConnection(const string& fname)
		{
			FILE *_fp = fopen(fname.c_str(), "w");
			printConnection(_fp);
			fclose(_fp);
		}

		void printConnection(FILE *_fp = stdout)
		{
			printPostConnection(_fp);
			printPreConnection(_fp);
		}

		//! print the connection info of grpId
		void printConnection(int grpId, FILE  *_fp = stdout)
		{
			printPostConnection(grpId, _fp);
			printPreConnection(grpId, _fp);
		}

		void printPostConnection(int grpId, FILE  *_fp = stdout);

		void printPreConnection(int grpId, FILE  *_fp = stdout);


		void showGroupStatus(int _f)        { showGrpFiringInfo  = _f; }

		void showDottyViewer(int _f)        { showDotty = _f; }

	private:
		void setGrpTimeSlice(int grpId, int timeSlice); //!< used for the Poisson generator.  It can probably be further optimized...

		void doSnnSim();

		void dumpSpikeBuffToFile_GPU(int gid);

		void assignPoissonFiringRate_GPU();

		void setSpikeGenBit_GPU(unsigned int nid, int grp);

		void generateSpikes();

		void generateSpikes(int grpId);

		void generateSpikesFromFuncPtr(int grpId);

		void generateSpikesFromRate(int grpId);

		void startCPUTiming();
		void resetCPUTiming();
		void stopCPUTiming();
		void startGPUTiming();
		void resetGPUTiming();
		void stopGPUTiming();

		void resetPointers();
		void resetSpikeCnt(int grpId = -1);
		void resetConductances();
		void resetCounters();
		void resetCurrent();
		void resetSynapticConnections(bool changeWeights=false);
		void resetTimingTable();
		void resetPoissonNeuron(unsigned int nid, int grpId);
		void resetNeuron(unsigned int nid, int grpId);
		void resetPropogationBuffer();
		void resetGroups();
		void resetFiringInformation();

		void doD1CurrentUpdate();
		void doD2CurrentUpdate();

		void findFiring();

		void doSTPUpdates();

		void generatePostSpike(unsigned int pre_i, unsigned int idx_d, unsigned int offset, unsigned int tD);

		void globalStateUpdate();

		void updateStateAndFiringTable();

		void updateSpikesFromGrp(int grpId);

		void updateSpikeGeneratorsInit();

		void updateSpikeGenerators();


		//! add the entry that the current neuron has spiked..
		int addSpikeToTable(int id, int g);

		float getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId);

		//conection related methods...
		inline void setConnection(int srcGrpId, int destGrpId, unsigned int src, unsigned int dest, float synWt, float maxWt, uint8_t dVal, int connProp);
		void connectUserDefined ( GroupConnectData* info);
		void connectRandom(GroupConnectData* info);
		void connectFull(GroupConnectData* info);
		void connectOneToOne(GroupConnectData* info);
		void connectFromMatrix(SparseWeightDelayMatrix* mat, int connProp);

		void exitSimulation(int val);

		void deleteObjects(); //!< deallocates all used data structures in snn_cpu.cpp
		void deleteObjectsGPU(); //!< deallocates all used data structures in snn_gpu.cu

		void testSpikeSenderReceiver(FILE* fpLog, int simTime);

		void printFiringInfo(FILE* _fp, int grpId=-1);

		void printFiredId(FILE* _fp, bool clear=false, int myGrpId=-1);

		void printTestVarInfo(FILE *_fp, char* testString, bool test1=true, bool test2=true, bool test12=false, int subVal=0, int grouping1=0, int grouping2=0);

		void printGpuLoadBalance(bool init = false, int numBlocks = MAX_BLOCKS, FILE*_fp = stdout);

		void printCurrentInfo(FILE *_fp);

		int checkErrors(string kernelName, int numBlocks);
		int checkErrors(int numBlocks);

		void updateParameters(int* numN, int* numPostSynapses, int* D, int nConfig=1);

		int  updateSpikeTables();

		void reorganizeNetwork(bool removeTempMemory, int simType);

		void reorganizeDelay();

		void swapConnections(int nid, int oldPos, int newPos);

		//! minimize any other wastage in that array by compacting the store
		void compactConnections();

		//! Here we reorganize all the input connections to the neurons such that the first set is all excitatory
		//! connection and next set is inhibitory connections
		void regroupConnections(FILE *_fp=NULL);

		//! initialize all the synaptic weights to appropriate values. total size of the synaptic connection is 'length'
		void initSynapticWeights();

		//! For the given neuron nid, find the group id
		int findGrpId(int nid);

		//! copy required spikes from the firing buffer to the spike buffer for later preprocessing
		void updateSpikeMonitor();

		void updateSpikeMonitor_GPU();

		void updateMonitors();

		void updateAfterMaxTime();


		//! initializes params needed in snn_gpu.cu (gets called in CpuSNN constructor)
		void CpuSNNinitGPUparams();

		//! allocates required memory and then initialize the GPU
		void allocateSNN_GPU(int ithGPU);

		void allocateNetworkParameters();

		void allocateGroupParameters();

		void buildNetwork();

		void buildGroup(int groupId);

		void buildPoissonGroup(int groupId);

		void copyWeightsGPU(unsigned int nid, int src_grp);

		void makePtrInfo();

		void copyNeuronState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId=-1);

		//! copy presynaptic information
		void copyWeightState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId=-1);

		void copyNeuronParameters(network_ptr_t* dest, int kind, int allocateMem, int grpId = -1);

		void copySTPState(network_ptr_t* dest, network_ptr_t* src, int kind, int allocateMem);

		void checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId);

		void copyConnections(network_ptr_t* dest, int kind, int allocateMem);

		void copyPostConnectionInfo(network_ptr_t* dest, int allocateMem);

		void copyState(network_ptr_t* dest, int kind, int allocateMem);

		void printGpuPostConnection(int grpId, FILE* _fp, int numBlock);

		void gpuProbeInit(network_ptr_t* dest);

		void copyParameters();

		int getPoissNeuronPos(int nid);

		void findFiring_GPU();

		void spikeGeneratorUpdate_GPU();

		void updateTimingTable_GPU();

		void doCurrentUpdate_GPU();

		void globalStateUpdate_GPU();

		void doGPUSim();

		void initGPU(int gridSize, int blkSize);

		int  allocateStaticLoad(int bufSize);

		void allocateGroupId();

		void copyFiringInfo_GPU();

		void copyFiringStateFromGPU (int grpId = -1);

		void updateStateAndFiringTable_GPU();

		void showStatus(int simType=CPU_MODE);
		void showStatus_GPU();

		void checkInitialization(char* testString=NULL);

		void checkInitialization2(char* testString=NULL);

		int getNumGroups() {
			return numGrp;
		}

		bool isExcitatoryGroup(int g) {
			return (grp_Info[g].Type&TARGET_AMPA) || (grp_Info[g].Type&TARGET_NMDA);
		}

		bool isInhibitoryGroup(int g) {
			return (grp_Info[g].Type&TARGET_GABAa) || (grp_Info[g].Type&TARGET_GABAb);
		}

		bool isPoissonGroup(int g) {
			return (grp_Info[g].Type&POISSON_NEURON);
		}

		//! \deprecated deprecated, may be removed soon...
		void setDefaultParameters(float alpha_ltp=0, float tau_ltp=0, float alpha_ltd=0, float tau_ltd=0);

		void setupNetwork(int simType=CPU_MODE, int ithGPU=0, bool removeTempMemory=true);



private:
		SparseWeightDelayMatrix* tmp_SynapseMatrix_fixed;
		SparseWeightDelayMatrix* tmp_SynapseMatrix_plastic;
		FILE* readNetworkFID;

		//! temporary variables created and deleted by network after initialization
		uint8_t			*tmp_SynapticDelay;

		bool simulatorDeleted;

		bool spikeRateUpdated;

		float prevCpuExecutionTime;
		float cpuExecutionTime;
		float prevGpuExecutionTime;
		float gpuExecutionTime;

		int		randSeed;

		int		currentMode;	//!< current operating mode

		int		numConfig;

		//! properties of the network (number of groups, network name, allocated neurons etc..)
		bool			doneReorganization;
		bool			memoryOptimized;

		string			networkName;
		int				numGrp;
		int				numConnections;
		uint32_t	allocatedN;
		uint32_t	allocatedPre;
		uint32_t	allocatedPost;

		GroupConnectData* connectBegin;

		//! Buffer to store spikes
		PropagatedSpikeBuffer* pbuf;

		bool sim_with_fixedwts;
		bool sim_with_stdp;
		bool sim_with_stp;
		bool sim_with_conductances;

		// spiking neural network related information, including neurons, synapses and network parameters
		int	        	numN;				//!< number of neurons in the spiking neural network
		int				numPostSynapses;	//!< maximum number of post-synaptic connections in groups
		int				numPreSynapses;		//!< maximum number of pre-syanptic connections in groups
		int				D;					//!< maximum axonal delay in groups
		int				numNReg;			//!< number of regular (spking) neurons
		int				numNExcReg;			//!< number of regular excitatory neurons
		int				numNInhReg;			//!< number of regular inhibitory neurons
		int   			numNExcPois;		//!< number of excitatory poisson neurons
		int				numNInhPois;		//!< number of inhibitory poisson neurons
		int				numNPois;			//!< number of poisson neurons
		float       	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current;
		bool			*curSpike;
		uint32_t        *nSpikeCnt;     //!< spike counts per neuron
		uint16_t       	*Npre;			//!< stores the number of input connections to the neuron
		uint16_t		*Npre_plastic;	//!< stores the number of excitatory input connection to the input
		uint16_t       	*Npost;			//!< stores the number of output connections from a neuron.
		uint32_t    	*lastSpikeTime;	//!< stores the most recent spike time of the neuron
		float			*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
		float	 		*maxSynWt;		//!< maximum synaptic weight for given connection..
		uint32_t    	*synSpikeTime;	//!< stores the spike time of each synapse
		uint32_t		postSynCnt;		//!< stores the total number of post-synaptic connections in the network
		uint32_t		preSynCnt;		//!< stores the total number of pre-synaptic connections in the network
		float			*intrinsicWeight;
		uint32_t		*cumulativePost;
		uint32_t		*cumulativePre;
		post_info_t		*preSynapticIds;
		post_info_t		*postSynapticIds;		//!< 10 bit syn id, 22 bit neuron id, ordered based on delay
		delay_info_t    *postDelayInfo;      	//!< delay information

		FILE*		fpDotty;

		//! size of memory used for different parts of the network
		typedef struct snnSize_s {
			uint32_t		neuronInfoSize;
			uint32_t		synapticInfoSize;
			uint32_t		networkInfoSize;
			uint32_t		spikingInfoSize;
			uint32_t		debugInfoSize;
			uint32_t		addInfoSize;	//!< includes random number generator etc.
			uint32_t		blkInfoSize;
			uint32_t		monitorInfoSize;
			uint32_t		probeInfoSize;
		} snnSize_t;

		snnSize_t cpuSnnSz;
		snnSize_t gpuSnnSz;

		uint32_t 	postConnCnt;
		uint32_t	preConnCnt;

		//! firing info
		uint32_t		*timeTableD2;
		uint32_t		*timeTableD1;
		uint32_t		*firingTableD2;
		uint32_t		*firingTableD1;
		uint32_t		maxSpikesD1;
		uint32_t		maxSpikesD2;

		//time and timestep
		uint64_t	simTimeSec;		//!< this is used to store the seconds.
		uint32_t	simTimeMs;
		uint32_t	simTime;		//!< this value is not reset but keeps increasing to its max value. The unit is millisecond.
		uint32_t	spikeCountAll1sec;
		uint32_t	secD1fireCntHost;
		uint32_t	secD2fireCntHost;	//!< firing counts for each second
		uint32_t	spikeCountAll;
		uint32_t	spikeCountD1Host;
		uint32_t	spikeCountD2Host;	//!< overall firing counts values
		uint32_t	numPoissonSpikes;

		//cuda keep track of performance...
#if __CUDA3__
		unsigned int    timer;
#elif __CUDA5__
		StopWatchInterface* timer;
#endif
		float		cumExecutionTime;
		float		lastExecutionTime;

		//debug file
		FILE	*fpProgLog;
		FILE	*fpLog;
		FILE	*fpTuningLog;
		int		cntTuning;
		FILE 	*fpParam;
		int		showLog;
		int		showLogMode;			//!< each debug statement has a mode. If log set to high mode, more logs generated
		int		showLogCycle;			//!< how often do we need to update the log


		//spike monitor code...
		uint32_t	numSpikeMonitor;
		uint32_t	monGrpId[MAX_GRP_PER_SNN];
		uint32_t	monBufferPos[MAX_GRP_PER_SNN];
		uint32_t	monBufferSize[MAX_GRP_PER_SNN];
		uint32_t	*monBufferFiring[MAX_GRP_PER_SNN];
		uint32_t	*monBufferTimeCnt[MAX_GRP_PER_SNN];
		SpikeMonitor	*monBufferCallback[MAX_GRP_PER_SNN];

		unsigned int	numSpikeGenGrps;

		//current/voltage probe code...
		unsigned int	numProbe;
		typedef struct probeParam_s {
			uint32_t		printProbe;
			uint32_t 		debugCnt;
			unsigned int     	nid;
			int			type;
			float		*bufferI;
			float		*bufferV;
			float		*bufferFRate;
			bool		*spikeBins;
			int			cumCount;
			float			vmax;
			float   		vmin;
			float			imax;
			float			imin;
			float			fmax;
			float			hfmax;
			struct probeParam_s 	*next;
		} probeParam_t;

		probeParam_t	*neuronProbe;

		/* Markram et al. (1998), where the short-term dynamics of synapses is characterized by three parameters:
		U (which roughly models the release probability of a synaptic vesicle for the first spike in a train of spikes),
		D (time constant for recovery from depression), and F (time constant for recovery from facilitation). */
		float	*stpu;
		float	*stpx;
		float	*gAMPA;
		float	*gNMDA;
		float	*gGABAa;
		float	*gGABAb;

		bool 		enableSimLogs;
		string		simLogDirName;

		network_info_t 	net_Info;

		network_ptr_t  		cpu_gpuNetPtrs;
		network_ptr_t   	cpuNetPtrs;

		//int   Noffset;
		int	  NgenFunc;					//!< this counts the spike generator offsets...

		bool finishedPoissonGroup;		//!< This variable is set after we have finished
										//!< creating the poisson group...
		bool showDotty;

		bool showGrpFiringInfo;

		// gpu related info...
		// information about various data allocated at GPU side...
		unsigned int	gpu_tStep, gpu_simSec;		//!< this is used to store the seconds.
		unsigned int	gpu_simTime;				//!< this value is not reset but keeps increasing to its max value.

		group_info_t	  	grp_Info[MAX_GRP_PER_SNN];
		group_info2_t		grp_Info2[MAX_GRP_PER_SNN];
		float		*testVar, *testVar2;
		uint32_t	*spikeGenBits;


		/* these are deprecated, and replaced by writeNetwork(FILE*)
		void storePostWeight (int destGrp, int srcNid, const string& fname, const string& name);

		void dumpHistogram();
		void dumpHistogram(int* table_sHist, int* table_sdHist, FILE *fph, int sec, char* fname, int histSize=HISTOGRAM_SIZE);

		void printHistogram(const char* histName=NULL, bool dontOverWrite=0, int histSize=HISTOGRAM_SIZE);

		void putHistogram(FILE *_fp, int sec, const char* fname, int histSize=HISTOGRAM_SIZE, bool dontOverWrite=false);

		void generateHistogram(int* sHist, int* sdHist, int histSize, FILE *_fp);

		void generateHistogram_GPU(int* sHist, int* sdHist, int histSize);

		void storeWeights(int dest_grp, int src_grp, const string& dirname, int restoreTime = -1);

		void saveConnectionWeights();

		//histogram related
		int		stopHistogram;
		int		histFileCnt;
		int		rasterFileCnt;
		*/

		/* deprecated
			void initThalInput();
			void initThalInput_GPU();

			void plotFiringRate(FILE* _fp = NULL, int x=0, int y=0, int y_limit=10000);

			void plotFiringRate(const string& fname, int x, int y, int y_limit);

			void getScaledWeights(void* img, int dest_grp, int src_grp, int resx=1, int resy=1);

			void getScaledWeights1D (void* imgPtr, unsigned int nid, int src_grp, int repx, int repy);

			void showWeightPattern1D (int destGrp, int srcGrp, int locx, int locy);

			void showWeightPattern (int destGrp, int srcGrp, int locx, int locy, int size = IMAGE_SIZE);

			void showWeightRatePattern1D (int destGrp, int srcGrp);

			void getScaledWeightRates1D(unsigned int nid, int src_grp);

			void plotSpikeBuff(int gid, int row, int col=900);

			void setImgWin(int monId, int localId, int t);

			void plotBuffInit(int gid);


		void updateNetwork();

		// input noise related codes...
		void randomNoiseCurrent(float neuronPercentage, float currentStrength, int groupId=-1);

		void updateRandomProperty(); // for random thalamic input


		// noise generator related info...
		int			numNoise;		// Total number of noise generators
		int			numRandNeurons;	// Total number of neurons selected each time step
		int*		randNeuronId;	// Store the ids of neuron which will get random input current..
		noiseGenProperty_t	noiseGenGroup[MAX_GRP_PER_SNN];

		*/
};

/*
typedef struct noiseGenProperty_s  {
	float	neuronPercentage;	// percentage of neuron that needs random inputs..
	float	currentStrength;	// strength of the noise current applied...
	int		groupId;			// group id.. cross references to group properties
	int		nstart;				// start of the neuron id
	int		ncount;				// total neuron in current group
	int		rand_ncount;		// (ncount*neuronPercentage/100) neurons will be selected
} noiseGenProperty_t;

*/

#endif
