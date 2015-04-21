#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>

#if defined(WIN32) || defined(WIN64)
#include <periodic_spikegen.h>
#endif

/// **************************************************************************************************************** ///
/// SPIKE COUNTER
/// **************************************************************************************************************** ///

//! expect CARLsim to die if SpikeCounter is called with silly params
TEST(SpikeCounter, setSpikeCounterDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SpikeCounter.setSpikeCounterDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);

	// grpId
	EXPECT_DEATH({sim->setSpikeCounter(g1+1,1);},"");
	EXPECT_DEATH({sim->setSpikeCounter(-1,1);},"");
	EXPECT_DEATH({sim->setSpikeCounter(2,1);},"");

	delete sim;
}

//! expects the number of spikes recorded by spike counter to be equal to spike monitor values
TEST(SpikeCounter, SpikeCntvsSpikeMon) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	float STP_U = 0.25f; // the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CARLsim* sim = NULL;

	const int nNeur = 10;
	const int nGrp = 3;
	int grpIds[nGrp];
	SpikeMonitor* SM[nGrp];

	for (int mode=0; mode<=1; mode++) {
		sim = new CARLsim("SpikeCounter.SpikeCntvsSpikeMon",mode?GPU_MODE:CPU_MODE,SHOWTIME,0,42);

		grpIds[0]=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(grpIds[0], 0.02f, 0.2f, -65.0f, 8.0f);

		grpIds[1]=sim->createSpikeGeneratorGroup("inputCallback", nNeur, EXCITATORY_NEURON);
		grpIds[2]=sim->createSpikeGeneratorGroup("inputPoisson", nNeur, EXCITATORY_NEURON);

		sim->connect(grpIds[1],grpIds[0],"full",RangeWeight(0.01f),1.0f);
		sim->setConductances(true,5,150,6,150);

		for (int g=0; g<nGrp; g++) {
			sim->setSpikeCounter(grpIds[g],-1);
			SM[g] = sim->setSpikeMonitor(grpIds[g],"NULL");
		}

		PeriodicSpikeGenerator spk50(50.0f,true); // periodic spiking @ 50 Hz
		sim->setSpikeGenerator(grpIds[1], &spk50);

		// after some time expect some number of spikes
		sim->setupNetwork(true);

		PoissonRate inPoiss(nNeur);
		inPoiss.setRates(20.0f);
		sim->setSpikeRate(grpIds[2], &inPoiss);

		for (int g=0; g<nGrp; g++) {
			SM[g]->startRecording();
		}
		sim->runNetwork(1,0);
		for (int g=0; g<nGrp; g++) {
			SM[g]->stopRecording();
		}

		// SpkMon should have the same number of spikes as SpikeCounter
		for (int g=0; g<nGrp; g++) {
			EXPECT_GT(SM[g]->getPopNumSpikes(), 0);

			if (SM[g]->getPopNumSpikes() > 0) {
				int* spkCnt = sim->getSpikeCounter(grpIds[g]);
				for (int i=0; i<nNeur; i++) {
					EXPECT_EQ(spkCnt[i], SM[g]->getNeuronNumSpikes(i));
				}
			}
		}

		delete sim;
	}
}

#if ENABLE_CPU_GPU_TESTS
//! expects certain number of spikes, CPU same as GPU
TEST(SpikeCounter, CPUvsGPU) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CARLsim* sim = NULL;

	int *spkCnt;
	int timeMs = 500;

	for (int mode=0; mode<=1; mode++) {
		sim = new CARLsim("SpikeCounter.CPUvsGPU",mode?GPU_MODE:CPU_MODE,SILENT,0,42);

		// record spikes in CPU mode, compare GPU values
		int* spikesCPU = new int[timeMs]; // time

		int recordDur = -1;

		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

		int g0=sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON);

		sim->connect(g0,g1,"full",RangeWeight(0.001f),1.0);
		sim->setConductances(true,5,150,6,150);

		sim->setSpikeCounter(g1,recordDur);
		sim->setSpikeMonitor(g1,"NULL");

		PeriodicSpikeGenerator* spk50 = new PeriodicSpikeGenerator(50.0f); // periodic spiking @ 50 Hz
		sim->setSpikeGenerator(g0, spk50);

		// after some time expect some number of spikes
		for (int tt=0; tt<timeMs; tt++) {
			sim->runNetwork(0,1,false); // run for 1ms
			if (!mode) {
				// CPU mode: record spikes
				spikesCPU[tt] = sim->getSpikeCounter(g1)[0];
			} else {
				// GPU mode: compare to recorded spikes
				EXPECT_EQ(sim->getSpikeCounter(g1)[0],spikesCPU[tt]);
			}
		}

		delete spikesCPU;
		delete spk50;
		delete sim;
	}
}
#endif
