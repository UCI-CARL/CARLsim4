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
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 2/21/2014
 */ 

#ifndef _SNN_GOLD_H_
#define _SNN_GOLD_H_

#include "mtrand.h"
#include "gpu_random.h"
#include "config.h"
#include "propagated_spike_buffer.h"
#include "poisson_rate.h"
#include "sparse_weight_delay_matrix.h"

#if __CUDA3__
	#include <cuda.h>
	#include <cutil_inline.h>
	#include <cutil_math.h>
#elif __CUDA5__
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <helper_cuda.h>
	#include <helper_functions.h>
	#include <helper_timer.h>
	#include <helper_math.h>
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
#define POISSON_NEURON	(1<<0)
#define TARGET_AMPA	(1<<1)
#define TARGET_NMDA	(1<<2)
#define TARGET_GABAa	(1<<3)
#define TARGET_GABAb	(1<<4)

#define INHIBITORY_NEURON 		(TARGET_GABAa | TARGET_GABAb)
#define EXCITATORY_NEURON 		(TARGET_NMDA | TARGET_AMPA)
#define EXCITATORY_POISSON 		(EXCITATORY_NEURON | POISSON_NEURON)
#define INHIBITORY_POISSON		(INHIBITORY_NEURON | POISSON_NEURON)
#define IS_INHIBITORY_TYPE(type)	(((type)&TARGET_GABAa) || ((type)&TARGET_GABAb))
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

#define SET_INITWTS_RANDOM(a)		((a&1) << CONNECTION_INITWTS_RANDOM)
#define SET_CONN_PRESENT(a)		((a&1) << CONNECTION_CONN_PRESENT)
#define SET_FIXED_PLASTIC(a)		((a&1) << CONNECTION_FIXED_PLASTIC)
#define SET_INITWTS_RAMPUP(a)		((a&1) << CONNECTION_INITWTS_RAMPUP)
#define SET_INITWTS_RAMPDOWN(a)		((a&1) << CONNECTION_INITWTS_RAMPDOWN)

#define GET_INITWTS_RANDOM(a)		(((a) >> CONNECTION_INITWTS_RANDOM)&1)
#define GET_CONN_PRESENT(a)		(((a) >> CONNECTION_CONN_PRESENT)&1)
#define GET_FIXED_PLASTIC(a)		(((a) >> CONNECTION_FIXED_PLASTIC)&1)
#define GET_INITWTS_RAMPUP(a)		(((a) >> CONNECTION_INITWTS_RAMPUP)&1)
#define GET_INITWTS_RAMPDOWN(a)		(((a) >> CONNECTION_INITWTS_RAMPDOWN)&1)

#define  checkNetworkBuilt()  {						\
    if(!doneReorganization)  {						\
      DBG(0, fpLog, AT, "checkNetworkBuilt()");				\
      fprintf(fpLog, "Network not yet elaborated and built...\n");	\
      fprintf(stderr, "Network not yet elaborated and built...\n");	\
      return;								\
    }									\
  }

/****************************/

#define STP_BUF_POS(nid,t)  (nid*STP_BUF_SIZE+((t)%STP_BUF_SIZE))


// FIXME: 
/////    !!!!!!! EVEN MORE IMPORTANT : IS THIS STILL BEING USED?? !!!!!!!!!!

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
#define STATIC_LOAD_GROUP(n)  (n.y &0xff)
#define STATIC_LOAD_SIZE(n)   ((n.y >> 16) & 0xff)

#define MAX_NUMBER_OF_NEURONS_BITS  (20)
#define MAX_NUMBER_OF_GROUPS_BITS   (32-MAX_NUMBER_OF_NEURONS_BITS)
#define MAX_NUMBER_OF_NEURONS_MASK  ((1 << MAX_NUMBER_OF_NEURONS_BITS)-1)
#define MAX_NUMBER_OF_GROUPS_MASK   ((1 << MAX_NUMBER_OF_GROUPS_BITS)-1)
#define SET_FIRING_TABLE(nid, gid)  (((gid) << MAX_NUMBER_OF_NEURONS_BITS) | (nid))
#define GET_FIRING_TABLE_NID(val)   ((val) & MAX_NUMBER_OF_NEURONS_MASK)
#define GET_FIRING_TABLE_GID(val)   (((val) >> MAX_NUMBER_OF_NEURONS_BITS) & MAX_NUMBER_OF_GROUPS_MASK)

//!< Used for in the function getConnectionId
#define CHECK_CONNECTION_ID(n,total) { assert(n >= 0); assert(n < total); }

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
  short  delay_index_start;
  short  delay_length;
} delay_info_t;

typedef struct {
  int      postId;
  uint8_t  grpId;
} post_info_t;

typedef struct network_info_s  {
  size_t		STP_Pitch;		//!< numN rounded upwards to the nearest 256 boundary
  unsigned int		numN,numPostSynapses,D,numNExcReg,numNInhReg, numNReg;
  unsigned int		I_setLength;
  size_t		I_setPitch;
  unsigned int		preSynLength;
  //	unsigned int		numRandNeurons;
  //	unsigned int		numNoise;
  unsigned int		postSynCnt;
  unsigned int		preSynCnt;
  unsigned int		maxSpikesD2,maxSpikesD1;
  unsigned int   	numNExcPois;
  unsigned int	numNInhPois;
  unsigned int	numNPois;
  unsigned int	numGrp;
  bool		sim_with_fixedwts;
  bool		sim_with_conductances;
  bool		sim_with_stdp;
  bool		sim_with_stp;
} network_info_t;

//! nid=neuron id, sid=synapse id, grpId=group id. 
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
  float	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current;

  // conductances and stp values
  float	*gNMDA, *gAMPA, *gGABAa, *gGABAb;
  int*	I_set;
  int		memType;
  int		allocated;			//!< true if all data has been allocated..
  float	*stpx, *stpu;

  unsigned short	*Npre;				//!< stores the number of input connections to the neuron
  unsigned short	*Npre_plastic;			//!< stores the number of plastic input connections
  float	*Npre_plasticInv;			//!< stores the 1/number of plastic input connections, for use on the GPU
  unsigned short	*Npost;				//!< stores the number of output connections from a neuron.
  unsigned int		*lastSpikeTime;			//!< storees the firing time of the neuron
  float	*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
  float	 *maxSynWt;			//!< maximum synaptic weight for given connection..
  uint32_t *synSpikeTime;
  uint32_t *neuronFiring;
  unsigned int		*cumulativePost;
  unsigned int		*cumulativePre;

  /*!
   * \brief 10 bit syn id, 22 bit neuron id, ordered based on delay
   *
   * allows maximum synapses of 1024 and maximum network size of 4 million neurons, with 64 bit representation. we can
   * have larger networks for simulation
   */
  post_info_t	*postSynapticIds;

  post_info_t	*preSynapticIds;
  delay_info_t    *postDelayInfo;  	//!< delay information
  unsigned int*		firingTableD1;
  unsigned int*		firingTableD2;
  //	int*		randId;
  //	void*		noiseGenProp;


  float*		poissonFireRate;
  unsigned int*		poissonRandPtr;		//!< firing random number. max value is 10,000
  int2*		neuronAllocation;
  int3*		groupIdInfo;		//!< used for group Id calculations...
  short int*	synIdLimit;			//
  float*		synMaxWts;			//

  unsigned int*		nSpikeCnt;

  //!< homeostatic plasticity variables
  float*	baseFiringInv;
  float*	baseFiring;
  float*	avgFiring;



  float* 		testVar;
  float*		testVar2;
  uint32_t*	spikeGenBits;
  bool*		curSpike;
} network_ptr_t;

typedef struct group_info_s
{
  // properties of group of neurons size, location, initial weights etc.
  PoissonRate*		RatePtr;
  int			StartN;
  int			EndN;
  char		Type;
  int			SizeN;
  int			NumTraceN;
  short int  	MaxFiringRate; //!< this is for the monitoring mechanism, it needs to know what is the maximum firing rate in order to allocate a buffer big enough to store spikes...
  int			MonitorId;
  float   	RefractPeriod;
  int		CurrTimeSlice; //!< timeSlice is used by the Poisson generators in order to note generate too many or too few spikes within a window of time
  int		NewTimeSlice;
  uint32_t 	SliceUpdateTime;
  int 		FiringCount1sec;
  int 		numPostSynapses;
  int 		numPreSynapses;
  bool 		isSpikeGenerator;
  bool 		WithSTP;
  bool 		WithSTDP;
  bool 		WithHomeostasis;
  bool 		WithConductances;
  int		homeoId;
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

  //!< homeostatic plasticity variables
  float		avgTimeScale;
  float 	avgTimeScale_decay;
  float		avgTimeScaleInv;
  float		homeostasisScale;

  SpikeGenerator* spikeGen;
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

enum conType_t { CONN_RANDOM, CONN_ONE_TO_ONE, CONN_FULL, CONN_FULL_NO_DIRECT, CONN_USER_DEFINED, CONN_UNKNOWN};

//! connection infos...
typedef struct connectData_s {
  int 	  		grpSrc, grpDest;
  uint8_t	  		maxDelay,  minDelay;
  float	  		initWt, maxWt;
  int	  	  		numPostSynapses;
  int	  	  		numPreSynapses;
  uint32_t  		connProp;
  ConnectionGenerator*		conn;
  conType_t 		type;
  float			p;
  int				connId;
  bool			newUpdates;
  int		   		numberOfConnections;
  struct connectData_s* next;
} grpConnectInfo_t;


#define MAX_GRPS_PER_BLOCK 		100
#define MAX_BLOCKS         		120
////////////////////////////////////
// member variable
////////////////////////////////////
typedef struct grpConnInfo_s {
  int16_t	srcGrpId;			//!< group id
  int	srcStartN;		//!< starting neuron to begin computation
  int	srcEndN;			//!< ending neuron to stop computation
  int	grpDelayVector;	//!< a vector with ones in position having a given delay
  int	grpMaxM;			//!< the maximum value of the number of post-synaptic connections
  bool	hasCommonDelay;   //!< 'true' if the grpDelayVector is same as the neuron DelayVector
  bool	hasRandomConn;	//!< set to 'true' if the group has random connections
  int*	randomDelayPointer; //
  int16_t	fixedDestGrpCnt;	//!< destination group count
  int*	fixedDestGrps;		//!< connected destination groups array, (x=destGrpId, y=startN, z=endN, w=function pointer)
  int*	fixedDestParam;	//!< connected destination parameters ,  (x=Start, y=Width, z=Stride, w=height)
} grpConnInfo_t;




/// **************************************************************************************************************** ///
/// CPUSNN CORE CLASS
/// **************************************************************************************************************** ///


/*!
 * \brief Contains all of CARLsim's core functionality
 *
 * This is a more elaborate description of our main class.
 */
class CpuSNN {
public:
	CpuSNN(const std::string& _name, int _numConfig=1, int randomize=0, int mode=CPU_MODE);
	~CpuSNN();

	const static unsigned int MAJOR_VERSION = 2; //!< major release version, as in CARLsim X
	const static unsigned int MINOR_VERSION = 2; //!< minor release version, as in CARLsim 2.X


	/// ************************************************************************************************************ ///
	/// PUBLIC METHODS: SETTING UP A SIMULATION
	/// ************************************************************************************************************ ///


	//! make connection of type "random","full","one-to-one" from each neuron in grpId1 to neurons in grpId2
	int connect(int gIDpre, int gIDpost, const std::string& _type, float initWt, float maxWt, float _C,
					uint8_t minDelay, uint8_t maxDelay, bool synWtType = SYN_FIXED);

	//! make custom connections from grpId1 to grpId2
	// TODO: describe maxM and maxPreM
	int connect(int gIDpre, int gIDpost, ConnectionGenerator* conn, bool synWtType=SYN_FIXED, int maxM=0,int maxPreM=0);


	//! creates a group of Izhikevich spiking neurons
	int createGroup(const std::string& grpName, unsigned int nNeur, int neurType, int configId=ALL);

	//! creates a spike generator group (dummy-neurons, not Izhikevich spiking neurons). 
	int createSpikeGeneratorGroup(const std::string& grpName, int unsigned nNeur, int neurType, int configId=ALL);


  	//! sets custom values for conductance decay (\tau_decay) or disables conductances alltogether
	void setConductances(int grpId, bool isSet, float tdAMPA, float tdNMDA, float tdGABAa, float tdGABAb, int configId);


	//! Sets the Izhikevich parameters a, b, c, and d of a neuron group.
	/*! Parameter values for each neuron are given by a normal distribution with mean _a, _b, _c, _d standard
	 * deviation a_sd, b_sd, c_sd, and d_sd, respectively. */
	void setNeuronParameters(int grpId, float izh_a, float izh_a_sd, float izh_b, float izh_b_sd,
								float izh_c, float izh_c_sd, float izh_d, float izh_d_sd, int configId=ALL);




	/// ************************************************************************************************************ ///
	/// PUBLIC METHODS: RUNNING A SIMULATION
	/// ************************************************************************************************************ ///



	/// ************************************************************************************************************ ///
	/// PUBLIC METHODS: INTERACTING WITH A SIMULATION
	/// ************************************************************************************************************ ///

	void copyGrpInfo_GPU(); //!< Used to copy grp_info to gpu for setSTDP, setSTP, and setHomeostasis

	//! Copies synaptic weight information, neuronal state information, and STP state information from the host
	//! to the device (GPU).
	void copyUpdateVariables_GPU();

	//added this for Jay's functionality -- KDC
	//TAGS:TODO: may need to make it work for different configurations. -- KDC
	/*!
 	 * \brief Returns pointer to nSpikeCnt, which is a 1D array of the number of spikes every neuron in the group
	 *  has fired.  Takes the grpID and the simulation mode (CPU_MODE or GPU_MODE) as arguments.
	 */
	unsigned int* getSpikeCntPtr(int grpId=ALL, int simType=CPU_MODE);

	void readNetwork(FILE* fid);						//!< reads the network state from file

	// This function calls the callback connect function for a particular
	// connection (or number of identical configurations of a connection).
	// This function can only be used for fixed weights and for connections
	// of type CONN_USER_DEFINED.  Only the weights are changed, not the 
	// maxWts, delays, or the connected values. -- KDC
	// TODO: figure out scope; is this a user function?
	/*!
	 * \brief Reassigns fixed weights to values passed into the function in a single 1D float matrix called
	 * weightMatrix.  The user passes the connection ID (connectID), the weightMatrix, the matrixSize, and 
	 * configuration ID (configID).  This function only works for fixed synapses.
	 */
	void reassignFixedWeights(int connectId, float weightMatrix [], int matrixSize, int configId = ALL);

	void resetSpikeCnt(int grpId = -1);					//!< Resets the spike count for a particular group.
	void resetSpikeCnt_GPU(int _startGrp, int _endGrp); //!< Utility function to clear spike counts in the GPU code.
	void resetSpikeCntUtil(int grpId = -1); //!< resets spike count for particular neuron group

	/*!
	 * \brief Resets either the neuronal firing rate information by setting resetFiringRate = true and/or the
	 * weight values back to their default values by setting resetWeights = true.
	 */
	void updateNetwork(bool resetFiringInfo, bool resetWeights);
	void updateNetwork(); //!< Original updateNetwork() function used by JMN
	void updateNetwork_GPU(bool resetFiringInfo); //!< Allows parameters to be reset in the middle of the simulation


	void writeNetwork(FILE* fid); 						//!< writes the network state to file

	/*!
	 * \brief function writes population weights from gIDpre to gIDpost to file fname in binary.
	 */
	void writePopWeights(std::string fname, int gIDpre, int gIDpost, int configId = 0);




	/// ************************************************************************************************************ ///
	/// PUBLIC METHODS: PLOTTING / LOGGING
	/// ************************************************************************************************************ ///

	void setLogCycle(unsigned int _cnt, int mode=0, FILE *fp=NULL);

	// NOTE: all these printer functions should be in printSNNInfo.cpp
	void printConnection(const std::string& fname);
	void printConnection(FILE *fp=stdout);
	void printConnection(int grpId, FILE  *fp=stdout); //!< print the connection info of grpId
	void printConnectionInfo(FILE* fp);
	void printConnectionInfo2(FILE *fpg);
	void printCurrentInfo(FILE *fp); //!< for GPU debugging
	void printFiringRate(char *fname=NULL);
	void printGpuLoadBalance(bool init = false, int numBlocks = MAX_BLOCKS, FILE*fp = stdout); //!< for GPU debugging
	void printGroupInfo(FILE* fp);
	void printGroupInfo(std::string& strName);
	void printGroupInfo2(FILE* fpg);
	void printMemoryInfo(FILE* fp=stdout); //!< prints memory info to file
	void printNetworkInfo();
	void printNeuronState(int grpId, FILE *fp = stderr);
	void printParameters(FILE *fp);
	void printPostConnection(FILE *fp = stdout); //!< print all post connections
	void printPostConnection(int grpId, FILE  *fp = stdout);
	int  printPostConnection2(int grpId, FILE* fpg);
	void printPreConnection(FILE  *fp = stdout); //!< print all pre connections
	void printPreConnection(int grpId, FILE  *fp = stdout);
	int  printPreConnection2(int grpId, FILE* fpg);
	void printSimSummary(FILE *fp = stdout); //!< prints a simulation summary to file
	void printState(const char *str = "");
	void printTestVarInfo(FILE *fp, char* testString, bool test1=true, bool test2=true, bool test12=false,
							int subVal=0, int grouping1=0, int grouping2=0); //!< for GPU debugging
	void printTuningLog();



	// added this to output regular neuron state variables
	// to a binary file at every timestep. -- KDC
	/*!
	 * \brief Debugging function that outputs regular neuron state variables to a binary file at every timestep.
	 * Warning: drastically reduces speed of simulation.  Only use for debugging purposes.
	 */
	// void printNeuronStateBinary(std::string fname, int grpId, int configId=0, int count=-1);




	/// ************************************************************************************************************ ///
	/// PUBLIC METHODS: GETTERS / SETTERS
	/// ************************************************************************************************************ ///

	grpConnectInfo_t* getConnectInfo(int connectId, int configId=0); //!< required for homeostasis
	int  getConnectionId(int connId, int configId);

	// FIXME: fix this
	uint8_t* getDelays(int gIDpre, int gIDpost, int& Npre, int& Npost, uint8_t* delays=NULL);

	int  getGroupId(int groupId, int configId);
	group_info_t getGroupInfo(int groupId, int configId=0);
	group_info2_t getGroupInfo2(int groupId, int configId=0);
	int  getNextGroupId(int); //!< used in setHomeostasis

	int getNumConfigurations()	{ return numConfig; }	//!< gets number of network configurations
	int getNumConnections(int connectionId);			//!< gets number of connections associated with a connection ID
	int getNumGroups() { return numGrp; }

	/*!
	 * \brief Writes weights from synaptic connections from gIDpre to gIDpost.  Returns a pointer to the weights
	 * and the size of the 1D array in size.  gIDpre(post) is the group ID for the pre(post)synaptic group, 
	 * weights is a pointer to a single dimensional array of floats, size is the size of that array which is 
	 * returned to the user, and configID is the configuration ID of the SNN.  NOTE: user must free memory from
	 * weights to avoid a memory leak.  
	 */
	void getPopWeights(int gIDpre, int gIDpost, float*& weights, int& size, int configId = 0);

	uint64_t getSimTime()		{ return simTime; }
	uint32_t getSimTimeSec()	{ return simTimeSec; }
	uint32_t getSimTimeMs()		{ return simTimeMs; }

	// FIXME: fix this
	// TODO: maybe consider renaming getPopWeightChanges
	float* getWeightChanges(int gIDpre, int gIDpost, int& Npre, int& Npost, float* weightChanges=NULL);


	int grpStartNeuronId(int g) { return grp_Info[g].StartN; }
	int grpEndNeuronId(int g)   { return grp_Info[g].EndN; }
	int grpNumNeurons(int g)    { return grp_Info[g].SizeN; }

	/*!
	 * \brief Sets enableGpuSpikeCntPtr to true or false.  True allows getSpikeCntPtr_GPU to copy firing
	 * state information from GPU kernel to cpuNetPtrs.  Warning: setting this flag to true will slow down
	 * the simulation significantly.
	 */
	void setCopyFiringStateFromGPU(bool _enableGPUSpikeCntPtr);


	void setGroupInfo(int groupId, group_info_t info, int configId=ALL);
	void setPrintState(int grpId, bool _status, int neuronId=-1);
	void setSimLogs(bool enable, std::string logDirName = "");
	void setTuningLog(std::string fname);











  /*!
   * \brief run the simulation for n sec
   * simType can either be CPU_MODE or GPU_MODE
   * ithGPU: specify on which CUDA device to establish a context
   */
  int runNetwork(int _nsec, int _tstep = 0, int simType = CPU_MODE, int ithGPU = 0, bool enablePrint=false, int copyState=false);

  bool updateTime(); //!< returns true when a new second is started

  // grpId == -1, means all groups
  void setSTDP(int grpId, bool enable, int configId=ALL);
  void setSTDP(int grpId, bool enable, float _ALPHA_LTP, float _TAU_LTP, float _ALPHA_LTD, float _TAU_LTD, int configId=ALL);

  // g == -1, means all groups
  void setSTP(int g, bool enable, int configId=ALL);
  void setSTP(int g, bool enable, float STP_U, float STP_tD, float STP_tF, int configId=ALL);


 /*!
   * \brief Sets the homeostasis parameters. g is the grpID, enable=true(false) enables(disables) homeostasis,
   * and configId is the configuration ID that homeostasis will be enabled/disabled.
   */
  void setHomeostasis(int g, bool enable, int configId=0);
  /*!
   * \brief Sets the homeostasis parameters. g is the grpID, enable=true(false) enables(disables) homeostasis,
   * configId is the configuration ID that homeostasis will be enabled/disabled, homeostasisScale is strength of
   * homeostasis compared to the strength of normal LTP/LTD from STDP (which is 1), and avgTimeScale is the time
   * frame over which the average firing rate is averaged (it should be larger in scale than STDP timescales).
   */
  void setHomeostasis(int g, bool enable, float homeostasisScale, float avgTimeScale, int configId=0);
  
  
  /*!
   * \brief Sets the homeostatic target firing rate.  Neurons will try to attain this firing rate using 
   * homeostatic synaptic scaling.
   */
  void setBaseFiring(int groupId, int configId, float _baseFiring, float _baseFiringSD);

  //! sets up a spike monitor registered with a callback to process the spikes, there can only be one
  //! SpikeMonitor per group
  void setSpikeMonitor(int gid, SpikeMonitor* spikeMon=NULL, int configId=ALL);

  //! a simple wrapper that uses a predetermined callback to save the data to a file
  void setSpikeMonitor(int gid, const std::string& fname, int configId=0);

  void setSpikeRate(int grpId, PoissonRate* spikeRate, int refPeriod=1, int configId=ALL);
  void setSpikeGenerator(int grpId, SpikeGenerator* spikeGen, int configId=ALL);

  



 





/// **************************************************************************************************************** ///
/// PRIVATE METHODS
/// **************************************************************************************************************** ///

private:
	void CpuSNNInit(unsigned int nNeur, unsigned int nPostSyn, unsigned int nPreSyn, unsigned int maxDelay);

	//conection related methods
	void connectFromMatrix(SparseWeightDelayMatrix* mat, int connProp);
	void connectFull(grpConnectInfo_t* info);
	void connectOneToOne(grpConnectInfo_t* info);
	void connectRandom(grpConnectInfo_t* info);
	void connectUserDefined(grpConnectInfo_t* info);

	void deleteObjects();			//!< deallocates all used data structures in snn_cpu.cpp
	void deleteObjectsGPU();		//!< deallocates all used data structures in snn_gpu.cu

	float getWeights(int connProp, float initWt, float maxWt, unsigned int nid, int grpId);

	void exitSimulation(int val);	//!< deallocates all dynamical structures and exits

	void makePtrInfo();				//!< creates CPU net ptrs

	void printWeight(int grpId, const char *str = "");

	#if READNETWORK_ADD_SYNAPSES_FROM_FILE
		int readNetwork_internal(bool onlyPlastic);
	#else
		int readNetwork_internal();
	#endif

	void resetConductances();
	void resetCounters();
	void resetCurrent();
	void resetNeuron(unsigned int nid, int grpId);
	void resetPointers();
	void resetTimingTable();

	inline void setConnection(int srcGrpId, int destGrpId, unsigned int src, unsigned int dest, float synWt,
								float maxWt, uint8_t dVal, int connProp);

	int  updateSpikeTables();






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

	


  void resetSynapticConnections(bool changeWeights=false);
  void resetPoissonNeuron(unsigned int nid, int grpId);
  void resetPropogationBuffer();
  void resetGroups();
  void resetFiringInformation();
  //! Resets the firing information of GPU_MODE program when updateNetwork is called.
  void resetFiringInformation_GPU();


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



  void testSpikeSenderReceiver(FILE* fpLog, int simTime);


  int checkErrors(std::string kernelName, int numBlocks);
  int checkErrors(int numBlocks);

  void updateParameters(int* numN, int* numPostSynapses, int* D, int nConfig=1);


  void reorganizeNetwork(bool removeTempMemory, int simType);

  void reorganizeDelay();

  void swapConnections(int nid, int oldPos, int newPos);

  //! minimize any other wastage in that array by compacting the store
  void compactConnections();

  //! Here we reorganize all the input connections to the neurons such that the first set is all excitatory
  //! connection and next set is inhibitory connections
  void regroupConnections(FILE *fp=NULL);

  //! initialize all the synaptic weights to appropriate values. total size of the synaptic connection is 'length'
  void initSynapticWeights();

  //! For the given neuron nid, find the group id
  int findGrpId(int nid);

  //! copy required spikes from the firing buffer to the spike buffer for later preprocessing
  void updateSpikeMonitor();

  void updateSpikeMonitor_GPU();

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


  void copyNeuronState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId=-1);

  //! copy presynaptic information
  void copyWeightState(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId=-1);

  void copyNeuronParameters(network_ptr_t* dest, int kind, int allocateMem, int grpId = -1);

  void copySTPState(network_ptr_t* dest, network_ptr_t* src, int kind, int allocateMem);

  void checkDestSrcPtrs(network_ptr_t* dest, network_ptr_t* src, cudaMemcpyKind kind, int allocateMem, int grpId);

  void copyConnections(network_ptr_t* dest, int kind, int allocateMem);

  void copyPostConnectionInfo(network_ptr_t* dest, int allocateMem);

  void copyState(network_ptr_t* dest, int kind, int allocateMem);



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


  bool isExcitatoryGroup(int g) {
    return (grp_Info[g].Type&TARGET_AMPA) || (grp_Info[g].Type&TARGET_NMDA);
  }

  bool isInhibitoryGroup(int g) {
    return (grp_Info[g].Type&TARGET_GABAa) || (grp_Info[g].Type&TARGET_GABAb);
  }

  bool isPoissonGroup(int g) {
    return (grp_Info[g].Type&POISSON_NEURON);
  }

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
	int			currentMode;	//!< current operating mode
	int			numConfig;

	//! properties of the network (number of groups, network name, allocated neurons etc..)
	bool			doneReorganization;
	bool			memoryOptimized;

	std::string			networkName;
	int				numGrp;
	int				numConnections;
	//! keeps track of total neurons/presynapses/postsynapses currently allocated
	unsigned int	allocatedN;
	unsigned int	allocatedPre;
	unsigned int	allocatedPost;

	grpConnectInfo_t* connectBegin;

	//! Buffer to store spikes
	PropagatedSpikeBuffer* pbuf;

	bool sim_with_fixedwts;
	bool sim_with_stdp;
	bool sim_with_stp;
	bool sim_with_conductances;
	//! flag to enable the copyFiringStateInfo from GPU to CPU
	bool enableGPUSpikeCntPtr;

	// spiking network related info.. neurons, synapses and network parameters
	int	        	numN,numNReg,numPostSynapses,D,numNExcReg,numNInhReg, numPreSynapses;
	int   			numNExcPois, numNInhPois, numNPois;
	float       	*voltage, *recovery, *Izh_a, *Izh_b, *Izh_c, *Izh_d, *current;
	bool			*curSpike;
	unsigned int         	*nSpikeCnt;     //!< spike counts per neuron
	unsigned short       	*Npre;			//!< stores the number of input connections to the neuron
	unsigned short			*Npre_plastic;	//!< stores the number of excitatory input connection to the input
	unsigned short       	*Npost;			//!< stores the number of output connections from a neuron.
	uint32_t    	*lastSpikeTime;	//!< stores the most recent spike time of the neuron
	float			*wtChange, *wt;	//!< stores the synaptic weight and weight change of a synaptic connection
	float	 		*maxSynWt;		//!< maximum synaptic weight for given connection..
	uint32_t    	*synSpikeTime;	//!< stores the spike time of each synapse
	unsigned int		postSynCnt; //!< stores the total number of post-synaptic connections in the network
	unsigned int		preSynCnt; //!< stores the total number of pre-synaptic connections in the network
	float			*intrinsicWeight;
	//added to include homeostasis. -- KDC
	float					*baseFiring;
	float                 *avgFiring;
	unsigned int	        *nextTaste;
	unsigned int	        *nextDeath;
	unsigned int		*cumulativePost;
	unsigned int		*cumulativePre;
	post_info_t		*preSynapticIds;
	post_info_t		*postSynapticIds;		//!< 10 bit syn id, 22 bit neuron id, ordered based on delay
	delay_info_t    *postDelayInfo;      	//!< delay information

	//! size of memory used for different parts of the network
	typedef struct snnSize_s {
		unsigned int		neuronInfoSize;
		unsigned int		synapticInfoSize;
		unsigned int		networkInfoSize;
		unsigned int		spikingInfoSize;
		unsigned int		debugInfoSize;
		unsigned int		addInfoSize;	//!< includes random number generator etc.
		unsigned int		blkInfoSize;
		unsigned int		monitorInfoSize;
	} snnSize_t;

	snnSize_t cpuSnnSz;
	snnSize_t gpuSnnSz;
	unsigned int 	postConnCnt;
	unsigned int	preConnCnt;

	//! firing info
	unsigned int		*timeTableD2;
	unsigned int		*timeTableD1;
	unsigned int     *firingTableD2;
	unsigned int     *firingTableD1;
	unsigned int		maxSpikesD1, maxSpikesD2;

	//time and timestep
	unsigned int	simTimeMs;
	uint64_t        simTimeSec;		//!< this is used to store the seconds.
	unsigned int	simTime;		//!< this value is not reset but keeps increasing to its max value.
	unsigned int		spikeCountAll1sec, secD1fireCntHost, secD2fireCntHost;	//!< firing counts for each second
	unsigned int		spikeCountAll, spikeCountD1Host, spikeCountD2Host;	//!< overall firing counts values
	unsigned int     nPoissonSpikes;

	//cuda keep track of performance...
	#if __CUDA3__
		unsigned int    timer;
	#elif __CUDA5__
		StopWatchInterface* timer;
	#endif
	float		cumExecutionTime;
	float           lastExecutionTime;

	//debug file
	FILE*	fpProgLog;
	FILE*	fpLog;
	FILE* 	fpTuningLog;
	int		cntTuning;
	FILE 	*fpParam;
	int		showLog;
	int		showLogMode;			//!< each debug statement has a mode. If log set to high mode, more logs generated
	int		showLogCycle;			//!< how often do we need to update the log


	//spike monitor code...
	unsigned int			numSpikeMonitor;
	unsigned int			monGrpId[MAX_GRP_PER_SNN];
	unsigned int			monBufferPos[MAX_GRP_PER_SNN];
	unsigned int			monBufferSize[MAX_GRP_PER_SNN];
	unsigned int*		monBufferFiring[MAX_GRP_PER_SNN];
	unsigned int*		monBufferTimeCnt[MAX_GRP_PER_SNN];
	SpikeMonitor*	monBufferCallback[MAX_GRP_PER_SNN];

	unsigned int	numSpikeGenGrps;


	/* Markram et al. (1998), where the short-term dynamics of synapses is characterized by three parameters:
	   U (which roughly models the release probability of a synaptic vesicle for the first spike in a train of spikes),
	   D (time constant for recovery from depression), and F (time constant for recovery from facilitation). */
	float*		stpu;
	float*		stpx;
	float*		gAMPA;
	float*		gNMDA;
	float*		gGABAa;
	float*		gGABAb;

	bool 		enableSimLogs;
	std::string		simLogDirName;

	network_info_t 	net_Info;

	network_ptr_t  		cpu_gpuNetPtrs;
	network_ptr_t   	cpuNetPtrs;

	//int   Noffset;
	int	  NgenFunc;					//!< this counts the spike generator offsets...

	bool finishedPoissonGroup;		//!< This variable is set after we have finished
	//!< creating the poisson group...

	bool showGrpFiringInfo;

	// gpu related info...
	// information about various data allocated at GPU side...
	unsigned int	gpu_tStep, gpu_simSec;		//!< this is used to store the seconds.
	unsigned int	gpu_simTime;				//!< this value is not reset but keeps increasing to its max value.

	group_info_t	  	grp_Info[MAX_GRP_PER_SNN];
	group_info2_t		grp_Info2[MAX_GRP_PER_SNN];
	float*			testVar, *testVar2;
	uint32_t*	spikeGenBits;


  /* deprecated
     void initThalInput();
     void initThalInput_GPU();

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
