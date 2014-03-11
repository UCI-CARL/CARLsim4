#include <carlsim.h>		// include CARLsim
#include "gtest/gtest.h"	// include Google testing scripts

#include <limits.h>


// Don't forget to set REGRESSION_TESTING flag to 1 in config.h 

// TODO: figure out test directory organization (see issue #67); group into appropriate test cases; have test cases
// for published results; add documentation; etc.

// TODO: test interface (see issue #38)

// TODO: add speed test scripts (see issue #32)

// TODO: add more tests in general (see issue #21)

/*
 * GENERAL TESTING STRATEGY
 * ------------------------
 *
 * We provide test cases to A) test core functionality of CARLsim, to B) test the reproducibility of published results,
 * and C) to benchmark simulation speed.
 *
 * A) TESTING CORE FUNCTIONALITY
 * 1. Test core data structures when some functionality is enabled.
 *    For example: Set STP to true for a specific group, check grp_Info to make sure all values are set accordingly.
 * 2. Test core data structures when some functionality is disabled.
 *    For example: Set STP to false for a specific group, check grp_Info to make sure it's disabled.
 * 3. Test behavior when values for input arguments are chosen unreasonably.
 *    For example: Create a group with N=-4 (number of neurons) and expect simulation to die. This is because each
 *    core function should have assertion statements to prevent the simulation from running unreasonable input values.
 *    In some cases, it makes sense to catch this kind of error in the user interface as well (and display an
 *    appropriate error message to the user), but these tests should be placed in the UserInterface test case.
 * 4. Test behavior of network when run with reasonable values.
 *    For example: Run a sample network with STP enabled, and check stpu[nid] and stpx[nid] to make sure they behave.
 *    as expected. You can use the PeriodicSpikeGenerator to be certain of specific spike times and thus run
 *    reproducible sample networks.
 * 5. Test behavior of network when run in CPU mode vs. GPU mode.
 *    For example: Run a sample network with STP enabled, once in CPU mode and once in GPU mode. Record stpu[nid] and
 *    stpx[nid], and make sure that both simulation mode give the exact same result (except for some small error
 *    margin that can account for rounding errors/etc.).
 *
 * B) TESTING PUBLISHED RESULTS
 *
 * C) BENCHMARK TESTS
 *
 */


/// **************************************************************************************************************** ///
/// COMMON
/// **************************************************************************************************************** ///

//! a periodic spike generator (constant ISI) creating spikes at a certain rate
class PeriodicSpikeGenerator : public SpikeGenerator {
public:
	PeriodicSpikeGenerator(float rate) {
		assert(rate>0);
		rate_ = rate;	  // spike rate
		isi_ = 1000/rate; // inter-spike interval in ms
	}

	unsigned int nextSpikeTime(CpuSNN* snn, int grpId, int nid, unsigned int currentTime) {
		return currentTime+isi_; // periodic spiking according to ISI
	}

private:
	float rate_;	// spike rate
	int isi_;		// inter-spike interval that results in above spike rate
};


//! a spike monitor that counts the number of spikes per neuron, and also the total number of spikes
//! used to test the behavior of SpikeCounter
class SpikeMonitorPerNeuron: public SpikeMonitor {
private:
	const int nNeur_; // number of neurons in the group
	int* spkPerNeur_; // number of spikes per neuron
	long long spkTotal_; // number of spikes in group (across all neurons)

public:
	SpikeMonitorPerNeuron(int numNeur) : nNeur_(numNeur) {
		// we're gonna count the spikes each neuron emits
		spkPerNeur_ = new int[nNeur_];
		memset(spkPerNeur_,0,sizeof(int)*nNeur_);
		spkTotal_ = 0;
	}
		
	// destructor, delete all dynamically allocated data structures
	~SpikeMonitorPerNeuron() { delete spkPerNeur_; }
		
	int* getSpikes() { return spkPerNeur_; }
	long long getSpikesTotal() { return spkTotal_; }

	// the update function counts the spikes per neuron in the current second
	void update(CpuSNN* s, int grpId, unsigned int* NeuronIds, unsigned int *timeCounts) {
		int pos = 0;
		for (int t=0; t<1000; t++) {
			for (int i=0; i<timeCounts[t]; i++,pos++) {
				// turns out id will be enumerated between 0..numNeur_; it is NOT preSynIds[]
				// or postSynIds[] or whatever...
				int id = NeuronIds[pos];
				assert(id>=0); assert(id<nNeur_);
				spkPerNeur_[id]++;
				spkTotal_++;
			}
		}
	}
};



/// **************************************************************************************************************** ///
/// CARLSIM INTERFACE
/// **************************************************************************************************************** ///

//! trigger all UserErrors
TEST(Interface, setSpikeCounterUserError) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->setSpikeCounter(ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, getSpikeCounterUserError) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setSpikeCounter(g1);
	EXPECT_DEATH({sim->getSpikeCounter(ALL);},"");
	EXPECT_DEATH({sim->getSpikeCounter(g1,ALL);},"");
	delete sim;
}

/// **************************************************************************************************************** ///
/// SPIKE COUNTER
/// **************************************************************************************************************** ///

TEST(SpikeCounter, setSpikeCounterTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int recordDur = rand()%1000 + 1;

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setSpikeCounter(g1,recordDur,ALL);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.withSpikeCounter);
				EXPECT_FLOAT_EQ(grpInfo.spkCntRecordDur,recordDur);
			}
			delete sim;
		}
	}
}

//! expect CARLsim to die if SpikeCounter is called with silly params
TEST(SpikeCounter, setSpikeCounterDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	std::string name = "SNN";
	CpuSNN* sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);

	// grpId
	EXPECT_DEATH({sim->setSpikeCounter(g1+1,1,ALL);},"");
	EXPECT_DEATH({sim->setSpikeCounter(-1,1,ALL);},"");
	EXPECT_DEATH({sim->setSpikeCounter(sim->numGrp,1,ALL);},"");

	// configId
	EXPECT_DEATH({sim->setSpikeCounter(g1,1,-2);},"");
	EXPECT_DEATH({sim->setSpikeCounter(g1,1,2);},"");
	EXPECT_DEATH({sim->setSpikeCounter(g1,1,100);},"");

	delete sim;
}

//! expects the number of spikes recorded by spike counter to be equal to spike monitor values
TEST(SpikeCounter, SpikeCntvsSpikeMon) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	int* spikesCnt;
	int* spikesMon;

	// FIXME: this will fail in GPU mode... Because number of spikes will be zero. However, even the "official" Spike
	// Monitor is returning zero spikes. Is this due to the ordering in doGPUsim()?
	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,USER,0,nConfig,42);
			int recordDur = -1;

			int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON,ALL);
			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->connect(g0,g1,"full",0.001f,0.001f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
			sim->setConductances(ALL,true,5.0f,150.0f,6.0f,150.0f,ALL);

			sim->setSpikeCounter(g1,recordDur,ALL);
			SpikeMonitorPerNeuron* spikePN = new SpikeMonitorPerNeuron(10);
			sim->setSpikeMonitor(g1,spikePN,ALL);

			PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
			sim->setSpikeGenerator(g0, spk50, ALL);

			// after some time expect some number of spikes
			sim->runNetwork(1,0,false,false);
			spikesMon = spikePN->getSpikes();
			for (int c=0; c<nConfig; c++) { // for each configId
				spikesCnt = sim->getSpikeCounter(g1,c);
				for (int i=0; i<10; i++) { // for each neuron
					EXPECT_EQ(spikesCnt[i],14);
					EXPECT_EQ(spikesCnt[i]*nConfig,spikesMon[i]); // SpkMon should have the same, but all configId's together
				}
			}

			delete sim;
			delete spikePN;
			delete spk50;
		}
	}
}

//! expects certain number of spikes, CPU same as GPU
TEST(SpikeCounter, CPUvsGPU) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	int* spikes;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int recordDur = -1;

			int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON,ALL);
			int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->connect(g0,g1,"full",0.001f,0.001f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
			sim->setConductances(ALL,true,5.0f,150.0f,6.0f,150.0f,ALL);

			sim->setSpikeCounter(g1,recordDur,ALL);
			sim->setSpikeMonitor(g1,NULL,ALL);

			PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
			sim->setSpikeGenerator(g0, spk50, ALL);

			// after some time expect some number of spikes
			sim->runNetwork(0,750,false,false);
			spikes = sim->getSpikeCounter(g1,0);
			EXPECT_EQ(spikes[0],10);

			// reset different group and expect same number
			sim->resetSpikeCounter(g0,ALL);
			spikes = sim->getSpikeCounter(g1,0);
			EXPECT_EQ(spikes[0],10);

			// reset group and expect zero
			sim->resetSpikeCounter(g1,ALL);
			spikes = sim->getSpikeCounter(g1,0);
			EXPECT_EQ(spikes[0],0);

			// run some more and expect number
			sim->runNetwork(2,134,false,false);
			spikes = sim->getSpikeCounter(g1,0);
			EXPECT_EQ(spikes[0],28);

			delete spk50;
			delete sim;
		}
	}
}

/// **************************************************************************************************************** ///
/// SHORT-TERM PLASTICITY STP
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSTP to true
 * This function tests the information stored in the group info struct after enabling STP via setSTP
 */
TEST(STP, setSTPTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setSTP(g1,true,STP_U,STP_tF,STP_tD,ALL);					// exact values matter

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTP); 							// STP must be enabled
				EXPECT_FLOAT_EQ(grpInfo.STP_U,STP_U); 					// check exact values
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_u_inv,1.0f/STP_tF);
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_x_inv,1.0f/STP_tD);
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setSTP to false
 * This function tests the information stored in the group info struct after disabling STP via setSTP
 */
TEST(STP, setSTPFalse) {
	int maxConfig = rand()%10 + 10;	// create network by varying nConfig from 1..maxConfig, with some
	int nConfigStep = rand()%3 + 2; // step size nConfigStep
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTP(g1,false,0.1f,100,200); 					// exact values don't matter

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_FALSE(grpInfo.WithSTP);						// STP must be disabled
			}
			delete sim;
		}
	}
}


//! expect CARLsim to die if setSTP is called with silly params
TEST(STP, setSTPdeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);

	// grpId
	EXPECT_DEATH({sim->setSTP(-2,true,0.1f,10,10,ALL);},"");

	// STP_U
	EXPECT_DEATH({sim->setSTP(g1,true,0.0f,10,10,ALL);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,1.1f,10,10,ALL);},"");

	// STP_tF / STP_tD
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,-10,10,ALL);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,-10,ALL);},"");

	// configId
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,-2);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,2);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,101);},"");
}

/*
 * \brief check whether CPU mode reproduces some pre-recorded stpu and stpx (internal variables)
 * This test ensures that CARLsim quantitatively reproduces STP behavior across machines. This exact network has been
 * run before, and STP variables (stpu and stpx) have been recorded for a time window of 50 ms. The network should
 * reproduce these values at all times and on all platforms.
 * If this test fails, then the low-level behavior of STP has been changed.
 * There is a separate test that observes the impact of stpu and stpx on the post-synaptic current.
 * There is a separate test that makes sure CPU mode yields the same result as GPU mode.
 */
TEST(STP, internalCPUvsData) {
	// run network, compare to these pre-recorded values
	float stpu[50] = {	0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
						0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1900,0.1878,0.1856,0.1834,0.1813,0.1792,
						0.1771,0.1751,0.1730,0.1710,0.1690,0.1671,0.1651,0.1632,0.1613,0.1594,0.1576,0.1557,0.1539,
						0.1521,0.3118,0.3082,0.3046,0.3010,0.2975,0.2941,0.2907,0.2873,0.2839,0.2806 };
	float stpx[50] = {	1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
						1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,0.8100,0.8102,0.8104,0.8106,0.8108,0.8110,
						0.8111,0.8113,0.8115,0.8117,0.8119,0.8121,0.8123,0.8125,0.8127,0.8129,0.8130,0.8132,0.8134,
						0.8136,0.5601,0.5605,0.5609,0.5614,0.5618,0.5623,0.5627,0.5631,0.5636,0.5640 };

	std::string name = "SNN";
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP
	float abs_error = 1e-4f;		// allowed error margin

	CpuSNN* sim = new CpuSNN(name,CPU_MODE,USER,0,1,randSeed);
	int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
	int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
	sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
	sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
	sim->setConductances(ALL,true,5.0f,10.0f,15.0f,20.0f,ALL);
	sim->setSTP(g0,true,0.19f,86,992,ALL); // the exact values are not important

	PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
	sim->setSpikeGenerator(g0, spk50, ALL);

	for (int i=0; i<50; i++) {
		sim->runNetwork(0,1,false,true); // enable copyState
		EXPECT_NEAR(sim->stpu[1], stpu[i], abs_error);
		EXPECT_NEAR(sim->stpx[1], stpx[i], abs_error);
	}

	delete spk50;
	delete sim;
}

/*
 * \brief check whether CPU mode reproduces some pre-recorded post-synaptic current (external, behavior)
 * This test ensures that CARLsim quantitatively reproduces STP behavior across machines. This exact network has been
 * run before, and post-synaptic currents (affected by pre-synaptic STP) have been recorded for a time window of 50 ms.
 * The network should reproduce these values at all times and on all platforms.
 * If this test fails, then the low-level behavior or protocol of STP (the order in which STP variables are updated and
 * current changes are computed) has been changed.
 * There is a separate test that observes the internal STP variables stpu and stpx.
 * There is a separate test that makes sure CPU mode yields the same result as GPU mode.
 */
TEST(STP, externalCPUvsData) {
	// run network, compare to these pre-recorded values of post-synaptic current
	float current[50] = {	0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
							0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.2756,0.2209,0.1772,0.1422,0.1142,0.0918,
							0.0738,0.0594,0.0478,0.0385,0.0310,0.0251,0.0202,0.0164,0.0133,0.0108,0.0087,0.0071,0.0058,
							0.0047,0.3706,0.2971,0.2385,0.1916,0.1540,0.1238,0.0996,0.0802,0.0646,0.0521};

	std::string name = "SNN";
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP
	float abs_error = 1e-4f;		// allowed error margin

	CpuSNN* sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,randSeed);
	int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
	int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
	sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
	sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
	sim->setConductances(ALL,true,5.0f,10.0f,15.0f,20.0f,ALL);
	sim->setSTP(g0,true,0.19f,86,992,ALL); // the exact values are not important

	PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
	sim->setSpikeGenerator(g0, spk50, ALL);

	for (int i=0; i<50; i++) {
		sim->runNetwork(0,1,false,true); // enable copyState
		EXPECT_NEAR(sim->current[0], current[i], abs_error); // check post-synaptic current to see effect of pre-STP
	}

	delete spk50;
	delete sim;
}

/*!
 * \brief check whether CPU and GPU mode return the same stpu and stpx
 * This test creates a STP connection with random parameter values, runs a simulation
 * for 300 ms, and checks whether CPU_MODE and GPU_MODE return the same internal
 * variables stpu and stpx. Input is periodic 20 Hz spiking.
 */
TEST(STP, internalCPUvsGPU) {
	CpuSNN* sim = NULL;
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};

	float stpu[600] = {0.0f};
	float stpx[600] = {0.0f};

	int nConfig = 1;
	int randSeed = rand() % 1000;
	float STP_U = (float) rand()/RAND_MAX;
	int STP_tD = rand() % 100;
	int STP_tF = rand() % 500 + 500;
	float abs_error = 1e-4f; // error allowed for CPU<->GPU mode

	for (int j=0; j<2; j++) {
		sim = new CpuSNN(name,simModes[j],USER,0,nConfig,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(ALL,true,5.0f,10.0f,15.0f,20.0f,ALL);
		sim->setSTP(g0,true,STP_U,STP_tD,STP_tF,ALL);

		PeriodicSpikeGenerator* spk20 = new PeriodicSpikeGenerator(20.0f);
		sim->setSpikeGenerator(g0, spk20, ALL);

		for (int i=0; i<300; i++) {
			sim->runNetwork(0,1,false,true); // enable copyState
			stpu[j*300+i] = sim->stpu[1];
			stpx[j*300+i] = sim->stpx[1];
		}

		delete spk20;
		delete sim;
	}

	// compare stpu and stpx for both sim modes
	for (int i=0; i<300; i++) {
		EXPECT_NEAR(stpu[i],stpu[i+300],abs_error); // EXPECT_FLOAT_EQ sometimes works, too
		EXPECT_NEAR(stpx[i],stpx[i+300],abs_error);	// but _NEAR is better
	}

	// check init default values
	EXPECT_FLOAT_EQ(stpu[0],0.0f);
	EXPECT_FLOAT_EQ(stpx[0],1.0f);
	EXPECT_FLOAT_EQ(stpu[300],0.0f);
	EXPECT_FLOAT_EQ(stpx[300],1.0f);
}

// FIXME: The following test fails, but this is expected because of issue #61). There is the possibility of rounding
//        errors when going from CPU to GPU. But, the order of computation in the innermost loop (doSnnSim, doGPUsim)
//        is quite different, which makes it not so hard to believe that the resulting output would be different, too.
/*!
 * \brief check whether CPU and GPU mode return the post-synaptic current, affected by pre-synaptic STP (external)
 * This test creates a STP connection with random parameter values, runs a simulation
 * for 300 ms, and checks whether CPU_MODE and GPU_MODE return the same external behavior (post-synaptic current
 * affected by pre-synaptic STP). Input is periodic 20 Hz spiking.
 */
TEST(STP, externalCPUvsGPU) {
	CpuSNN* sim = NULL;
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};

	float current[600] = {0.0f};

	int nConfig = 1;
	int randSeed = rand() % 1000;
	float STP_U = (float) rand()/RAND_MAX;
	int STP_tD = rand() % 100;
	int STP_tF = rand() % 500 + 500;
	float abs_error = 1e-4f; // error allowed for CPU<->GPU mode

	for (int j=0; j<2; j++) {
		sim = new CpuSNN(name,simModes[j],USER,0,nConfig,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(ALL,true,5.0f,10.0f,15.0f,20.0f,ALL);
		sim->setSTP(g0,true,STP_U,STP_tD,STP_tF,ALL);

		PeriodicSpikeGenerator* spk20 = new PeriodicSpikeGenerator(20.0f);
		sim->setSpikeGenerator(g0, spk20, ALL);

		for (int i=0; i<300; i++) {
			sim->runNetwork(0,1,false,true); // enable copyState
			current[j*300+i] = sim->current[0];
		}

		delete spk20;
		delete sim;
	}

	// compare stpu and stpx for both sim modes
	for (int i=0; i<300; i++) {
		EXPECT_NEAR(current[i],current[i+300],abs_error); // EXPECT_FLOAT_EQ sometimes works, too
	}
}


/// **************************************************************************************************************** ///
/// CORE FUNCTIONALITY
/// **************************************************************************************************************** ///

//! check all possible (valid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinit) {
	CpuSNN* sim = NULL;

	// Problem: The first two modes will print to stdout, and close it in the end; so all subsequent calls to sdout
	// via GTEST fail
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};
	loggerMode_t loggerModes[4] = {USER, DEVELOPER, SILENT, CUSTOM};
	for (int i=0; i<4; i++) {
		for (int j=0; j<2; j++) {
			int nConfig = rand() % 100 + 1;
			int randSeed = rand() % 1000;
			sim = new CpuSNN(name,simModes[j],loggerModes[i],0,nConfig,randSeed);

			EXPECT_EQ(sim->networkName_,name);
			EXPECT_EQ(sim->getNumConfigurations(),nConfig);
			EXPECT_EQ(sim->randSeed_,randSeed);
			EXPECT_EQ(sim->getSimMode(),simModes[j]);
			EXPECT_EQ(sim->getLoggerMode(),loggerModes[i]);

			delete sim;
		}
	}

	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,0);
	EXPECT_EQ(sim->randSeed_,123);
	delete sim;

	// time(NULL)
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,-1);
	EXPECT_NE(sim->randSeed_,-1);
	EXPECT_NE(sim->randSeed_,0);
	delete sim;
}

// FIXME: enabling the following generates a segfault
//! check all possible (invalid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinitDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
	std::string name="SNN";

	// ithGPU
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,USER,-1,1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// nConfig
	EXPECT_DEATH({sim = new CpuSNN(name,GPU_MODE,USER,0,0,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,USER,0,101,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
}

//! Death tests for createGroup (test all possible silly values)
TEST(CORE, createGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({int N=-10; sim->createGroup("excit", N, EXCITATORY_NEURON, ALL);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, -3, ALL);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, EXCITATORY_NEURON, 2);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, EXCITATORY_NEURON, -2);},"");

	if (sim!=NULL)
		delete sim;
}

//! Death tests for createSpikeGenerator (test all possible silly values)
TEST(CORE, createSpikeGeneratorGroupSilly) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({int N=-10; sim->createSpikeGeneratorGroup("excit", N, EXCITATORY_NEURON, ALL);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, -3, ALL);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON, 2);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON, -2);},"");

	if (sim!=NULL)
		delete sim;
}


//! Death tests for setNeuronParameters (test all possible silly values)
TEST(CORE, setNeuronParametersSilly) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);
	int g0=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, -0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, -10.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, -0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, -10.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, -2.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, -8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, -10.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, 2);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, -2);},"");

	if (sim!=NULL)
		delete sim;
}


//! connect with certain mulSynFast, mulSynSlow and observe connectInfo
TEST(CORE, connect) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;

	CpuSNN* sim = NULL;
	grpConnectInfo_t* connInfo;
	std::string typeStr;
	std::string name="SNN";

	int conn[4] 		= {-1};
	conType_t type[4] 	= {CONN_RANDOM,CONN_ONE_TO_ONE,CONN_FULL,CONN_FULL_NO_DIRECT};
	float initWt[4] 	= {0.05f, 0.1f, 0.21f, 0.42f};
	float maxWt[4] 		= {0.05f, 0.1f, 0.21f, 0.42f};
	float prob[4] 		= {0.1, 0.2, 0.3, 0.4};
	int minDelay[4] 	= {1,2,3,4};
	int maxDelay[4] 	= {1,2,3,4};
	float mulSynFast[4] = {0.2f, 0.8f, 1.2f, 0.0};
	float mulSynSlow[4] = {0.0f, 2.4f, 11.1f, 10.0f};
	int synType[4] 		= {SYN_FIXED,SYN_PLASTIC,SYN_FIXED,SYN_PLASTIC};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			for (int i=0; i<4; i++) {
				sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

				int g0=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
				int g1=sim->createGroup("excit0", 10, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
				int g2=sim->createGroup("excit1", 10, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

				if (type[i]==CONN_RANDOM) typeStr = "random";
				else if (type[i]==CONN_ONE_TO_ONE) typeStr = "one-to-one";
				else if (type[i]=CONN_FULL) typeStr = "full";
				else if (type[i]=CONN_FULL_NO_DIRECT) typeStr = "full-no-direct";

				conn[i] = sim->connect(g0, g1, typeStr, initWt[i], maxWt[i], prob[i], minDelay[i], maxDelay[i], 
											mulSynFast[i], mulSynSlow[i], synType[i]);

				for (int c=0; c<nConfig; c++) {
					connInfo = sim->getConnectInfo(conn[i],c);
					EXPECT_FLOAT_EQ(connInfo->initWt,initWt[i]);
					EXPECT_FLOAT_EQ(connInfo->maxWt,maxWt[i]);
					EXPECT_FLOAT_EQ(connInfo->p,prob[i]);
					EXPECT_FLOAT_EQ(connInfo->mulSynFast,mulSynFast[i]);
					EXPECT_FLOAT_EQ(connInfo->mulSynSlow,mulSynSlow[i]);
					EXPECT_EQ(connInfo->minDelay,minDelay[i]);
					EXPECT_EQ(connInfo->maxDelay,maxDelay[i]);
					EXPECT_EQ(connInfo->type,type[i]);
					EXPECT_EQ(GET_FIXED_PLASTIC(connInfo->connProp),synType[i]);
				}
				delete sim;
			}
		}
	}
}

// TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output




/// **************************************************************************************************************** ///
/// SPIKE-TIMING-DEPENDENT PLASTICITY STDP
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(STDP, setSTDPTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,true,alphaLTP,tauLTP,alphaLTD,tauLTD);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTDP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTP,alphaLTP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTD,alphaLTD);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTP_INV,1.0/tauLTP);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTD_INV,1.0/tauLTD);
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setSTDP to false
 * This function tests the information stored in the group info struct after disabling STDP via setSTDP
 */
TEST(STDP, setSTDPFalse) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,false,alphaLTP,tauLTP,alphaLTD,tauLTD);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_FALSE(grpInfo.WithSTDP);
			}
			delete sim;
		}
	}
}





/// **************************************************************************************************************** ///
/// CONDUCTANCE-BASED MODEL (COBA)
/// **************************************************************************************************************** ///

//! Death tests for setConductances (test all possible silly values)
TEST(COBA, setCondSilly) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);
	int g0=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({sim->setConductances(g0+1, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(-2, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, -5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, -150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, -6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, -150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, 150.0f, 2);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, 150.0f, -2);},"");

	if (sim!=NULL)
		delete sim;
}

/*!
 * \brief testing setConductances to true
 * This function tests the information stored in the group info struct after calling setConductances and enabling COBA.
 */
TEST(COBA, setCondTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim = NULL;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->setConductances(grps[0],true,tAMPA,tNMDA,tGABAa,tGABAb,ALL);
			sim->setConductances(grps[1],true,tAMPA,tNMDA,tGABAa,tGABAb,ALL);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_TRUE(grpInfo.WithConductances);
					EXPECT_FLOAT_EQ(grpInfo.dAMPA,1.0-1.0/tAMPA);
					EXPECT_FLOAT_EQ(grpInfo.dNMDA,1.0-1.0/tNMDA);
					EXPECT_FLOAT_EQ(grpInfo.dGABAa,1.0-1.0/tGABAa);
					EXPECT_FLOAT_EQ(grpInfo.dGABAb,1.0-1.0/tGABAb);
				}
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setConductances to true using default values
 * This function tests the information stored in the group info struct after calling setConductances and enabling COBA.
 * Actual conductance values are set via the interface function setDefaultConductanceDecay
 */
TEST(COBA, setCondTrueDefault) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CARLsim* sim;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			sim->setDefaultConductanceDecay(tAMPA,tNMDA,tGABAa,tGABAb);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(grps[1], 0.02f, 0.2f, -65.0f, 8.0f);

			sim->setConductances(grps[0],true);
			sim->setConductances(grps[1],true);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_TRUE(grpInfo.WithConductances);
					EXPECT_FLOAT_EQ(grpInfo.dAMPA,1.0-1.0/tAMPA);
					EXPECT_FLOAT_EQ(grpInfo.dNMDA,1.0-1.0/tNMDA);
					EXPECT_FLOAT_EQ(grpInfo.dGABAa,1.0-1.0/tGABAa);
					EXPECT_FLOAT_EQ(grpInfo.dGABAb,1.0-1.0/tGABAb);
				}
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setConductances to false
 * This function tests the information stored in the group info struct after calling setConductances and disabling COBA.
 */
TEST(COBA, setCondFalse) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim = NULL;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->setConductances(grps[0],false,tAMPA,tNMDA,tGABAa,tGABAb, ALL);
			sim->setConductances(grps[1],false,tAMPA,tNMDA,tGABAa,tGABAb, ALL);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_FALSE(grpInfo.WithConductances);
				}
			}
			delete sim;
		}
	}
}

// TODO: test to trigger error that not all groups have conductances enabled


TEST(COBA, disableSynReceptors) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = 1; //rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim = NULL;
	group_info_t grpInfo;
	int grps[4] = {-1};

	int expectSpkCnt[4] = {200, 160, 0, 0};
	int expectSpkCntStd = 10;

	std::string expectCond[4] = {"AMPA","NMDA","GABAa","GABAb"};
	float expectCondVal[4] = {0.14, 2.2, 0.17, 2.2};
	float expectCondStd[4] = {0.025,0.2,0.025,0.2,};

	int nInput = 1000;
	int nOutput = 10;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g0=sim->createSpikeGeneratorGroup("spike", nInput, EXCITATORY_NEURON, ALL);
			int g1=sim->createSpikeGeneratorGroup("spike", nInput, INHIBITORY_NEURON, ALL);
			grps[0]=sim->createGroup("excitAMPA", nOutput, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excitNMDA", nOutput, EXCITATORY_NEURON, ALL);
			grps[2]=sim->createGroup("inhibGABAa", nOutput, INHIBITORY_NEURON, ALL);
			grps[3]=sim->createGroup("inhibGABAb", nOutput, INHIBITORY_NEURON, ALL);

			sim->setNeuronParameters(grps[0], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[2], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[3], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);

			sim->setConductances(ALL, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);

			sim->connect(g0, grps[0], "full", 0.001f, 0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g0, grps[1], "full", 0.0005f, 0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);
			sim->connect(g1, grps[2], "full", -0.001f, -0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g1, grps[3], "full", -0.0005f, -0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);

			PoissonRate poissIn1(nInput);
			PoissonRate poissIn2(nInput);
			for (int i=0; i<nInput; i++) {
				poissIn1.rates[i] = 30.0f;
				poissIn2.rates[i] = 30.0f;
			}
			sim->setSpikeRate(g0,&poissIn1,1,ALL);
			sim->setSpikeRate(g1,&poissIn2,1,ALL);

			sim->runNetwork(1,0,false,false);

			if (mode) {
				// GPU_MODE: copy from device to host
				for (int g=0; g<4; g++)
					sim->copyNeuronState(&(sim->cpuNetPtrs), &(sim->cpu_gpuNetPtrs), cudaMemcpyDeviceToHost, false, grps[g]);
			}

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<4; g++) { // all groups
					grpInfo = sim->getGroupInfo(grps[g],c);

					EXPECT_TRUE(grpInfo.WithConductances);
					for (int n=grpInfo.StartN; n<=grpInfo.EndN; n++) {
//						printf("%d[%d]: AMPA=%f, NMDA=%f, GABAa=%f, GABAb=%f\n",g,n,sim->gAMPA[n],sim->gNMDA[n],sim->gGABAa[n],sim->gGABAb[n]);
						if (expectCond[g]=="AMPA") {
							EXPECT_GT(sim->gAMPA[n],0.0f);
							EXPECT_NEAR(sim->gAMPA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gAMPA[n],0.0f);

						if (expectCond[g]=="NMDA") {
							EXPECT_GT(sim->gNMDA[n],0.0f);
							EXPECT_NEAR(sim->gNMDA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gNMDA[n],0.0f);

						if (expectCond[g]=="GABAa") {
							EXPECT_GT(sim->gGABAa[n],0.0f);
							EXPECT_NEAR(sim->gGABAa[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAa[n],0.0f);

						if (expectCond[g]=="GABAb") {
							EXPECT_GT(sim->gGABAb[n],0.0f);
							EXPECT_NEAR(sim->gGABAb[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAb[n],0.0f);
					}
				}
			}
			delete sim;
		}
	}	
}


/// **************************************************************************************************************** ///
/// CURRENT-BASED MODEL (CUBA)
/// **************************************************************************************************************** ///