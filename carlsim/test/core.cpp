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
#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <vector>

#include <periodic_spikegen.h>


/// **************************************************************************************************************** ///
/// Core FUNCTIONALITY
/// **************************************************************************************************************** ///

TEST(Core, getGroupGrid3D) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Core.getGroupGrid3D",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for this function to work

	for (int g=g1; g<g2; g++) {
		Grid3D getGrid = sim->getGroupGrid3D(g);
		EXPECT_EQ(getGrid.numX, grid.numX);
		EXPECT_EQ(getGrid.numY, grid.numY);
		EXPECT_EQ(getGrid.numZ, grid.numZ);
		EXPECT_EQ(getGrid.N, grid.N);
	}

	delete sim;
}

TEST(Core, getGroupIdFromString) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Core.getGroupIdFromString",CPU_MODE,SILENT,1,42);
	int g2=sim->createGroup("bananahama", Grid3D(1,2,3), INHIBITORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for this function to work

	EXPECT_EQ(sim->getGroupId("excit"), g1);
	EXPECT_EQ(sim->getGroupId("bananahama"), g2);
	EXPECT_EQ(sim->getGroupId("invalid group name"), -1); // group not found

	delete sim;
}


// This test creates a group on a grid and makes sure that the returned 3D location of each neuron is correct
TEST(Core, getNeuronLocation3D) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Core.getNeuronLocation3D",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for getNeuronLocation3D to work

	// make sure the 3D location that is returned is correct
	for (int grp=0; grp<=1; grp++) {
		// do for both spike gen and RS group

		int x = 0,y = 0, z = 0;
		for (int neurId = grp * grid.N; neurId < (grp + 1) * grid.N; neurId++) {
			Point3D loc = sim->getNeuronLocation3D(neurId);
			EXPECT_FLOAT_EQ(loc.x, x * grid.distX + grid.offsetX);
			EXPECT_FLOAT_EQ(loc.y, y * grid.distY + grid.offsetY);
			EXPECT_FLOAT_EQ(loc.z, z * grid.distZ + grid.offsetZ);

			x++;
			if (x==grid.numX) {
				x=0;
				y++;
			}
			if (y==grid.numY) {
				x=0;
				y=0;
				z++;
			}
		}
	}

	delete sim;
}

// \TODO: using external current, make sure the Izhikevich model is correctly implemented
// Run izhikevich.org MATLAB script to find number of spikes as a function of neuron type,
// input current, and time period. Build test case to reproduce the exact numbers.

TEST(Core, setExternalCurrent) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim * sim;
	int nNeur = 10;

	for (int hasCOBA=0; hasCOBA<=1; hasCOBA++) {
		for (int mode = 0; mode < TESTED_MODES; mode++) {
			sim = new CARLsim("Core.setExternalCurrent", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);
			int g1=sim->createGroup("excit1", nNeur, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			int g0=sim->createSpikeGeneratorGroup("input0", nNeur, EXCITATORY_NEURON);
			sim->connect(g0,g1,"full",RangeWeight(0.1),1.0f,RangeDelay(1));
			sim->setConductances(hasCOBA>0);
			sim->setupNetwork();
//			fprintf(stderr, "setExternalCurrent %s %s\n",hasCOBA?"COBA":"CUBA",isGPUmode?"GPU":"CPU");

			SpikeMonitor* SM = sim->setSpikeMonitor(g1,"NULL");

			// run for a bunch, observe zero spikes since ext current should be zero by default
			SM->startRecording();
			sim->runNetwork(1,0);
			SM->stopRecording();
			EXPECT_EQ(SM->getPopNumSpikes(), 0);

			// set current, observe spikes
			std::vector<float> current(nNeur,7.0f);
			sim->setExternalCurrent(g1, current);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

			// (intentionally) forget to reset current, observe spikes
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

			// reset current to zero
			sim->setExternalCurrent(g1, 0.0f);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_EQ(SM->getPopNumSpikes(), 0);

			// use convenience function to achieve same result as above
			sim->setExternalCurrent(g1, 7.0f);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

			delete sim;
		}
	}
}

TEST(Core, biasWeights) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	int nNeur = 10;
	int *nSpkHighWt = new int[nNeur];
	memset(nSpkHighWt, 0, nNeur*sizeof(int));

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("Core.biasWeights",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
		int g1=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
		int c1=sim->connect(g1, g1, "one-to-one", RangeWeight(0.5f), 1.0f, RangeDelay(1));
		sim->setConductances(true);
		sim->setupNetwork();

		// ---- run network for a while with input current and high weight
		//      observe much spiking

		SpikeMonitor* SM = sim->setSpikeMonitor(g1,"NULL");
		sim->setExternalCurrent(g1, 7.0f);

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {	
			nSpkHighWt[neurId] = SM->getNeuronNumSpikes(neurId);
		}


		// ---- run network for a while with zero weight (but still current injection)
		//      observe less spiking
		sim->biasWeights(c1, -0.25f, false);

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {
			EXPECT_LT(SM->getNeuronNumSpikes(neurId), nSpkHighWt[neurId]);
		}

		delete sim;
	}

	delete[] nSpkHighWt;
}

TEST(Core, scaleWeights) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	int nNeur = 10;
	int *nSpkHighWt = new int[nNeur];
	memset(nSpkHighWt, 0, nNeur*sizeof(int));

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("Core.scaleWeights",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
		int g1=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
		int c1=sim->connect(g1, g1, "one-to-one", RangeWeight(0.5f), 1.0f, RangeDelay(1));
		sim->setConductances(true);
		sim->setupNetwork();

		// ---- run network for a while with input current and high weight
		//      observe much spiking

		SpikeMonitor* SM = sim->setSpikeMonitor(g1,"NULL");
		sim->setExternalCurrent(g1, 7.0f);

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {	
			nSpkHighWt[neurId] = SM->getNeuronNumSpikes(neurId);
		}


		// ---- run network for a while with zero weight (but still current injection)
		//      observe less spiking
		sim->scaleWeights(c1, 0.5f, false);

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {
			EXPECT_LT(SM->getNeuronNumSpikes(neurId), nSpkHighWt[neurId]);
		}

		delete sim;
	}

	delete[] nSpkHighWt;
}

TEST(Core, setWeight) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	int nNeur = 10;
	int *nSpkHighWt = new int[nNeur];
	memset(nSpkHighWt, 0, nNeur*sizeof(int));

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("Core.setWeight",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
		int g1=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
		int c1=sim->connect(g1, g1, "one-to-one", RangeWeight(0.5f), 1.0f, RangeDelay(1));
		sim->setConductances(true);
		sim->setupNetwork();

		// ---- run network for a while with input current and high weight
		//      observe much spiking

		SpikeMonitor* SM = sim->setSpikeMonitor(g1,"NULL");
		sim->setExternalCurrent(g1, 7.0f);

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {	
			nSpkHighWt[neurId] = SM->getNeuronNumSpikes(neurId);
			sim->setWeight(c1, neurId, neurId, 0.0f, false);
		}


		// ---- run network for a while with zero weight (but still current injection)
		//      observe less spiking

		SM->startRecording();
		sim->runNetwork(2,0);
		SM->stopRecording();

		for (int neurId=0; neurId<nNeur; neurId++) {
			EXPECT_LT(SM->getNeuronNumSpikes(neurId), nSpkHighWt[neurId]);
		}

		delete sim;
	}

	delete[] nSpkHighWt;
}

TEST(Core, getDelayRange) {
	CARLsim* sim;
	int nNeur = 10;
	int minDelay = 1;
	int maxDelay = 10;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("Core.getDelayRange",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
		int g1=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
		int c1=sim->connect(g1, g1, "one-to-one", RangeWeight(0.5f), 1.0f, RangeDelay(minDelay,maxDelay));

		// config state right after connect
		RangeDelay delay = sim->getDelayRange(c1);
		EXPECT_EQ(delay.min, minDelay);
		EXPECT_EQ(delay.max, maxDelay);

		sim->setConductances(true);
		sim->setupNetwork();

		// setup state: still valid
		delay = sim->getDelayRange(c1);
		EXPECT_EQ(delay.min, minDelay);
		EXPECT_EQ(delay.max, maxDelay);

		sim->runNetwork(1,0);

		// exe state: still valid
		delay = sim->getDelayRange(c1);
		EXPECT_EQ(delay.min, minDelay);
		EXPECT_EQ(delay.max, maxDelay);

		delete sim;
	}
}

TEST(Core, getWeightRange) {
	CARLsim* sim;
	int nNeur = 10;
	float minWt = 0.0f;
	float initWt = 1.25f;
	float maxWt = 10.0f;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("Core.getWeightRange",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
		int g1=sim->createGroup("excit", nNeur, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
		int c1=sim->connect(g1, g1, "one-to-one", RangeWeight(minWt,initWt,maxWt), 1.0f, RangeDelay(1), RadiusRF(-1),
			SYN_PLASTIC);

		// config state right after connect
		RangeWeight wt = sim->getWeightRange(c1);
		EXPECT_EQ(wt.min, minWt);
		EXPECT_EQ(wt.init, initWt);
		EXPECT_EQ(wt.max, maxWt);

		sim->setConductances(true);
		sim->setupNetwork();

		// setup state: still valid
		wt = sim->getWeightRange(c1);
		EXPECT_EQ(wt.min, minWt);
		EXPECT_EQ(wt.init, initWt);
		EXPECT_EQ(wt.max, maxWt);

		sim->runNetwork(1,0);

		// exe state: still valid
		wt = sim->getWeightRange(c1);
		EXPECT_EQ(wt.min, minWt);
		EXPECT_EQ(wt.init, initWt);
		EXPECT_EQ(wt.max, maxWt);

		delete sim;
	}
}


// make sure bookkeeping for number of groups is correct during CONFIG
TEST(Core, numGroups) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim sim("Core.numGroups", CPU_MODE, SILENT, 0, 42);
	EXPECT_EQ(sim.getNumGroups(), 0);

	int nLoops = 4;
	int nNeur = 10;
	for (int i=0; i<nLoops; i++) {
		sim.createGroup("regexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+1);
		sim.createGroup("reginh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+2);
		sim.createSpikeGeneratorGroup("genexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+3);
		sim.createSpikeGeneratorGroup("geninh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+4);
	}
}

// make sure bookkeeping for number of neurons is correct during CONFIG
TEST(Core, numNeurons) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim sim("Core.numNeurons", CPU_MODE, SILENT, 0, 42);
	EXPECT_EQ(sim.getNumNeurons(), 0);
	EXPECT_EQ(sim.getNumNeuronsRegExc(), 0);
	EXPECT_EQ(sim.getNumNeuronsRegInh(), 0);
	EXPECT_EQ(sim.getNumNeuronsGenExc(), 0);
	EXPECT_EQ(sim.getNumNeuronsGenInh(), 0);

	int nNeur = 10;

	int g1 = sim.createGroup("regexc", nNeur, EXCITATORY_NEURON);
	int g2 = sim.createGroup("reginh", nNeur, INHIBITORY_NEURON);
	int g3 = sim.createSpikeGeneratorGroup("genexc", nNeur, EXCITATORY_NEURON);
	int g4 = sim.createSpikeGeneratorGroup("geninh", nNeur, INHIBITORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
	sim.setNeuronParameters(g2, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	sim.connect(g1, g2, "full", RangeWeight(0.5f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(g3, g1, "full", RangeWeight(0.5f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(g4, g1, "full", RangeWeight(0.5f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setupNetwork();

	EXPECT_EQ(sim.getNumNeurons(), 4*nNeur);
	EXPECT_EQ(sim.getNumNeuronsRegExc(), nNeur);
	EXPECT_EQ(sim.getNumNeuronsRegInh(), nNeur);
	EXPECT_EQ(sim.getNumNeuronsGenExc(), nNeur);
	EXPECT_EQ(sim.getNumNeuronsGenInh(), nNeur);
	EXPECT_EQ(sim.getNumNeurons(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh()
		+ sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
	EXPECT_EQ(sim.getNumNeuronsReg(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh());
	EXPECT_EQ(sim.getNumNeuronsGen(), sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
}

TEST(Core, startStopTestingPhase) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;

	// run twice, once with expected start/stop order, once with a bunch of additional (but
	// irrelevant start/stop calls)
	for (int run=0; run<=1; run++) {
		for (int mode = 0; mode < TESTED_MODES; mode++) {
			sim = new CARLsim("Core.startStopTestingPhase",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

			int gExc = sim->createGroup("output", 1, EXCITATORY_NEURON);
			sim->setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS
			int gIn = sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);

			int cInExc  = sim->connect(gIn, gExc, "full", RangeWeight(0.0f, 0.5f, 0.5f), 1.0f, RangeDelay(1), 
				RadiusRF(-1), SYN_PLASTIC);

			// set E-STDP to be STANDARD (without neuromodulatory influence) with an EXP_CURVE type.
			sim->setESTDP(gExc, true, STANDARD, ExpCurve(2e-4f,20.0f, -6.6e-5f,60.0f));
			sim->setHomeostasis(gExc, true, 1.0f, 10.0f);  // homeo scaling factor, avg time scale
			sim->setHomeoBaseFiringRate(gExc, 35.0f, 0.0f); // target firing, target firing st.d.

			sim->setConductances(true);
			sim->setupNetwork();
			ConnectionMonitor* CM = sim->setConnectionMonitor(gIn, gExc, "NULL");

			PoissonRate PR(10);
			PR.setRates(50.0f);
			sim->setSpikeRate(gIn, &PR);

			// training: expect weight changes due to STDP
			if (run==1) {
				sim->startTesting(); // testing function calls in SETUP_STATE
				sim->stopTesting();
			}
			sim->runNetwork(1,0);
			double wtChange = CM->getTotalAbsWeightChange();
			EXPECT_GT(CM->getTotalAbsWeightChange(), 0);
			EXPECT_EQ(CM->getTimeMsCurrentSnapshot(), 1000);
			EXPECT_EQ(CM->getTimeMsLastSnapshot(), 0);
			EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(), 1000);

			// testing: expect no weight changes
			sim->startTesting();
			if (run==1) {
				sim->runNetwork(5,0);
				sim->startTesting(); // start after start: redundant
				sim->runNetwork(5,0);
			} else {
				sim->runNetwork(10,0);
			}
			EXPECT_FLOAT_EQ(CM->getTotalAbsWeightChange(), 0.0f);
			EXPECT_EQ(CM->getTimeMsCurrentSnapshot(), 11000);
			EXPECT_EQ(CM->getTimeMsLastSnapshot(), 1000);
			EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(), 10000);

			// some more training: expect weight changes
			sim->stopTesting();
			CM->takeSnapshot();
			sim->runNetwork(5,0);
			EXPECT_GT(CM->getTotalAbsWeightChange(), 0);
			EXPECT_EQ(CM->getTimeMsCurrentSnapshot(), 16000);
			EXPECT_EQ(CM->getTimeMsLastSnapshot(), 11000);
			EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(), 5000);

			delete sim;
		}
	}
}

TEST(Core, saveLoadSimulation) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	float tauPlus = 20.0f;
	float tauMinus = 20.0f;
	float alphaPlus = 0.1f;
	float alphaMinus = 0.15f;
	int gPre, gPost;
	ConnectionMonitor* cmSave;
	ConnectionMonitor* cmLoad;
    // all neurons get input of 6 Hz.
	PeriodicSpikeGenerator spkGenG0(6.0f);
	std::vector<std::vector<float> > weightsSave;
	std::vector<std::vector<float> > weightsLoad;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba=0; coba<=1; coba++) {
			for (int isPlastic=0; isPlastic<=1; isPlastic++) {
				for (int loadSim=0; loadSim<=1; loadSim++) {
					// Run and save simulation ------------------------------ //
					CARLsim* sim = new CARLsim("Core.saveSimulation", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);
					FILE* simFid = NULL;

					gPost = sim->createGroup("pre-ex", 10, EXCITATORY_NEURON);
					sim->setNeuronParameters(gPost, 0.02f, 0.2f, -65.0f, 8.0f);
					gPre = sim->createSpikeGeneratorGroup("post-ex", 10, EXCITATORY_NEURON);
					sim->setSpikeGenerator(gPre, &spkGenG0);

					sim->connect(gPre, gPost, "full", RangeWeight(0.0, 20.0f/100, 20.0f/100), 1.0f, RangeDelay(1, 5),
						RadiusRF(-1), isPlastic?SYN_PLASTIC:SYN_FIXED);
					sim->setSTDP(gPost, isPlastic, STANDARD, alphaPlus/100, tauPlus, alphaMinus/100, tauMinus);
					sim->setConductances(coba>0);

					if (loadSim) {
					// load previous simulation
						simFid = fopen("results/sim.dat", "rb");
						sim->loadSimulation(simFid);
					}

					sim->setupNetwork();

					if (!loadSim) {
						// first run: save network at the end
						cmSave = sim->setConnectionMonitor(gPre, gPost, "NULL");
						sim->runNetwork(20, 0, false);

						weightsSave = cmSave->takeSnapshot();
						sim->saveSimulation("results/sim.dat", true);
					} else {
						// second run: load simulation
						cmLoad = sim->setConnectionMonitor(gPre, gPost, "NULL");
						sim->runNetwork(0, 2, false);
						weightsLoad = cmLoad->takeSnapshot();

						// test weights we saved are the same as weights we loaded
						for (int i = 0; i < sim->getGroupNumNeurons(gPre); i++) {
							for (int j = 0; j < sim->getGroupNumNeurons(gPost); j++) {
								if (coba) {
									EXPECT_FLOAT_EQ(weightsSave[i][j], weightsLoad[i][j]);
								} else {
									EXPECT_FLOAT_EQ(weightsSave[i][j], weightsLoad[i][j]);
								}
							}
						}
					}

					// close sim.dat
					if (simFid != NULL) fclose(simFid);
					delete sim;
				}
			}
		}
	}
}

TEST(Core, synapseIdOverflow) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim sim("Core.synapseIdOverflow", HYBRID_MODE, SILENT, 0, 42);

	int gExc = sim.createGroup("exc", 65536, EXCITATORY_NEURON, 0, CPU_CORES);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS
	int gInput = sim.createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, 0, CPU_CORES);

	// make connections more than 65535
	sim.connect(gInput, gExc, "full", RangeWeight(1.0), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	
	EXPECT_DEATH({ sim.setupNetwork(); }, ""); //sim.setupNetwork();
}
