#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// SPIKE COUNTER INTERFACE TESTS: Tests input from user-interface
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSpikeCounter for user errors
 * This function tests the setSpikeCounter for user errors.
 */
TEST(SpikeCounter, setSpikeCounterUserError) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->setSpikeCounter(ALL);},"");
	delete sim;
}

/*!
 * \brief testing getSpikeCounter for user errors
 * This function tests the getSpikeCounter for user errors.
 */
TEST(SpikeCounter, getSpikeCounterUserError) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setSpikeCounter(g1);
	EXPECT_DEATH({sim->getSpikeCounter(ALL);},"");
	EXPECT_DEATH({sim->getSpikeCounter(g1,ALL);},"");
	delete sim;
}


/// **************************************************************************************************************** ///
/// SPIKE COUNTER IMPLEMENTATION TESTS: Tests implementation functionality and related data structures
/// **************************************************************************************************************** ///

/*!
 * \brief test spike counter core behavior.
 * CARLsim shoudl die if SpikeCounter is called with silly params.
 */
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

/*!
 * \brief verify SpikeCounter function output.
 * Compare SpikeCounter recorded spikes to spike monitor values.
 */
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

/*!
 * \brief verfying SpikeCounter output on CPU/GPU
 * Compares SpikeCounter spike count between CPU and GPU
 */

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
