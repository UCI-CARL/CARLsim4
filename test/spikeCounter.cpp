#include <snn.h>

#include "carlsim_tests.h"

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

//! expects certain number of spikes, CPU vs. some pre-recorded data
TEST(SpikeCounter, SpikeCntVsData) {
	std::string name = "SNN";
	int maxConfig = rand()%10+10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	int* spikes;

	for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
		sim = new CpuSNN(name,CPU_MODE,SILENT,0,nConfig,42);

		int recordDur = -1;

		int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON,ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON,ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		sim->connect(g0,g1,"full",0.001f,0.001f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(true,5,0,150,6,0,150,ALL);

		sim->setSpikeCounter(g1,recordDur,ALL);
		sim->setSpikeMonitor(g1,NULL,ALL);

		PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
		sim->setSpikeGenerator(g0, spk50, ALL);

		// after some time expect some number of spikes
		sim->setupNetwork(true);
		sim->runNetwork(0,750,false);
		for (int c=0; c<nConfig; c++) {
			spikes = sim->getSpikeCounter(g1,c);
			EXPECT_EQ(spikes[0],16);
		}

		// reset different group and expect same number
		sim->resetSpikeCounter(g0,ALL);
		for (int c=0; c<nConfig; c++) {
			spikes = sim->getSpikeCounter(g1,c);
			EXPECT_EQ(spikes[0],16);
		}

		// reset group (one configId) and expect zero for it, same number for others
		sim->resetSpikeCounter(g1,nConfig-1);
		for (int c=0; c<nConfig; c++) {
			spikes = sim->getSpikeCounter(g1,c);
			if (c==nConfig-1)
				EXPECT_EQ(spikes[0],0);
			else
				EXPECT_EQ(spikes[0],16);
		}

		// reset group and expect zero
		sim->resetSpikeCounter(g1,ALL);
		for (int c=0; c<nConfig; c++) {
			spikes = sim->getSpikeCounter(g1,c);
			EXPECT_EQ(spikes[0],0);
		}

		// run some more and expect number
		sim->setupNetwork(true);
		sim->runNetwork(2,134,false);
		for (int c=0; c<nConfig; c++) {
			spikes = sim->getSpikeCounter(g1,0);
			EXPECT_EQ(spikes[0],42);
		}

		delete spk50;
		delete sim;
	}
}

//! expects the number of spikes recorded by spike counter to be equal to spike monitor values
TEST(SpikeCounter, SpikeCntvsSpikeMon) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f; // the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	int* spikesCnt;
	int* spikesMon;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,USER,0,nConfig,42);

			int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON,ALL);
			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->connect(g0,g1,"full",0.001f,0.001f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
			sim->setConductances(true,5,0,150,6,0,150,ALL);

			sim->setSpikeCounter(g1,-1,ALL);
			SpikeMonitorPerNeuron* spikePN = new SpikeMonitorPerNeuron(10);
			sim->setSpikeMonitor(g1,spikePN,ALL);

			PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
			sim->setSpikeGenerator(g0, spk50, ALL);

			// after some time expect some number of spikes
			sim->setupNetwork(true);
			sim->runNetwork(1,0,false);
			spikesMon = spikePN->getSpikes();

			// spike counter has to loop over all configIds
			int nSpikes[10] = {0};
			for (int c=0; c<nConfig; c++) { // for each configId
				spikesCnt = sim->getSpikeCounter(g1,c);
				for (int i=0; i<10; i++) { // for each neuron
					nSpikes[i] += spikesCnt[i];
				}
			}

			// SpkMon should have the same number of spikes as all configs together for SpikeCounter
			for (int i=0; i<10; i++) {
				EXPECT_EQ(nSpikes[i],spikesMon[i]);
			}

			delete sim;
			delete spikePN;
			delete spk50;
		}
	}
}

#if ENABLE_CPU_GPU_TESTS
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

	int *spkCnt;
	int timeMs = 500;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			// record spikes in CPU mode, compare GPU values
			int* spikesCPU = new int[timeMs*nConfig]; // time x number config

			int recordDur = -1;

			int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON,ALL);
			int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->connect(g0,g1,"full",0.001f,0.001f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
			sim->setConductances(true,5,0,150,6,0,150,ALL);

			sim->setSpikeCounter(g1,recordDur,ALL);
			sim->setSpikeMonitor(g1,NULL,ALL);

			PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
			sim->setSpikeGenerator(g0, spk50, ALL);

			// after some time expect some number of spikes
			for (int tt=0; tt<timeMs; tt++) {
				sim->runNetwork(0,1,false,false); // run for 1ms
				if (!mode) {
					// CPU mode: record spikes
					for (int c=0; c<nConfig; c++) {
						spikesCPU[tt*nConfig+c] = sim->getSpikeCounter(g1,c)[0];
					}
				} else {
					// GPU mode: compare to recorded spikes
					for (int c=0; c<nConfig; c++) {
						EXPECT_EQ(sim->getSpikeCounter(g1,c)[0],spikesCPU[tt*nConfig+c]);
					}
				}
			}

			delete spikesCPU;
			delete spk50;
			delete sim;
		}
	}
}
#endif
