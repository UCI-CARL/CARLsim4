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

// include CARLsim user interface and gtest
#include "gtest/gtest.h"
#include "carlsim_tests.h"
#include <carlsim.h>

/*
	class FixedRandomConnGen - Subclass of the connectionGenerator class to define custom connections between two groups
 	using a callback connect() method. The constructor takes #source neurons, #destination neurons,
 	an instance of rangeweight struct, and an instance of rangedelay struct. The callback connect() method
 	can be used to set a connection from neuron i of source group to neuron j of dest group and to set
 	the synapse parameters
*/
class FixedRandomConnGen : public ConnectionGenerator {
public:
        FixedRandomConnGen(int srcNumN, int destNumN, RangeWeight rt, RangeDelay rd) {
                maxConnections = srcNumN * destNumN;
                row = destNumN;
                fixedWt = rt.init;
                maxDelay = rd.max;
                minDelay = rd.min;

                delays = new int[maxConnections];
                connecteds = new bool[maxConnections];

                srand((int)time(NULL));

                for (int i = 0; i < maxConnections; i++) {
                        delays[i] = (rand() % (maxDelay - minDelay)) + minDelay;
                        connecteds[i] = rand() % 2 ? true : false; // 50%
                        //printf("[%d %d]", delays[i], connecteds[i]);
                }
        }

        ~FixedRandomConnGen() {
                delete [] delays;
                delete [] connecteds;
        }

        void connect(CARLsim* s, int srcGrpId, int i, int destGrpId, int j, float& weight, float& maxWt, float& delay, bool& connected) {
                weight = fixedWt;
                maxWt = fixedWt;
                delay = delays[i * row + j];
                connected = connecteds[i * row + j];
        }

private:
        int maxConnections;
        int row;
        int maxDelay;
        int minDelay;
        float fixedWt;
        int* delays;
        bool* connecteds;
};

TEST(cpuMultiRuntimes, spikesSingleVsMulti) {
	// create a network on GPU
	int gExc, gExc2, gInput;
	std::vector<std::vector<int> > spikesSingleRuntime, spikesMultiRuntimes;
	CARLsim* sim;
	FixedRandomConnGen* frConnGen = new FixedRandomConnGen(10, 10, RangeWeight(10.0f), RangeDelay(1, 20));

	int randSeed = rand();
	for (int partition = 0; partition < 10; partition++) {
		sim = new CARLsim("test kernel", HYBRID_MODE, SILENT, 0, randSeed);

		// configure the network
		gExc = sim->createGroup("exc", 10, EXCITATORY_NEURON, 0, CPU_CORES);
		sim->setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

		//int gInh = sim.createGroup("inh", 20, INHIBITORY_NEURON);
		//sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS
		gExc2 = sim->createGroup("exc2", 10, EXCITATORY_NEURON, partition, CPU_CORES);
		sim->setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS

		gInput = sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON, 0, CPU_CORES);

		sim->connect(gInput, gExc, "one-to-one", RangeWeight(50.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		sim->connect(gExc, gExc2, frConnGen, SYN_FIXED);

		sim->setConductances(false);

			//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

			// build the network
		sim->setupNetwork();

			// set some monitors
			//SpikeMonitor* smInput = sim->setSpikeMonitor(gInput, "NULL");
			//SpikeMonitor* smExc = sim->setSpikeMonitor(gExc, "NULL");
		SpikeMonitor* smExc2 = sim->setSpikeMonitor(gExc2, "NULL");
			//ConnectionMonitor* cmEE = sim->setConnectionMonitor(gExc, gExc2, "NULL");

			//setup some baseline input
		PoissonRate in(10);
		in.setRates(5.0f);
		sim->setSpikeRate(gInput, &in);

			// run for a total of 10 seconds
			// at the end of each runNetwork call, SpikeMonitor stats will be printed
			//smInput->startRecording();
			//smExc->startRecording();
		smExc2->startRecording();

		sim->runNetwork(1, 0);

			//smInput->stopRecording();
			//smExc->stopRecording();
		smExc2->stopRecording();

		if (partition == 0) { // single cpu
			spikesSingleRuntime = smExc2->getSpikeVector2D();
		}
		else {
			spikesMultiRuntimes = smExc2->getSpikeVector2D();
		}

			//smExc->print(true);
			//smExc2->print(true);
			//smInput->print(true);

		//printf("spikesSingleVsMulti - [0 , %d]", partition);

		delete sim;
	}

	for (int nId = 0; nId < spikesSingleRuntime.size(); nId++) {
		EXPECT_EQ(spikesSingleRuntime[nId].size(), spikesMultiRuntimes[nId].size()); // the same number of spikes
		for (int s = 0; s < spikesSingleRuntime[nId].size(); s++)
			EXPECT_EQ(spikesSingleRuntime[nId][s], spikesMultiRuntimes[nId][s]); // the same spike timing
	}
}

void run_shuffleGroups(int partitionA, int partitionB, int partitionC, float* smExcRate, float* smInhRate,
	float* smInputRate)
{
	int randSeed = 42;
	float pConn = 100.0f / 1000; // connection probability

	CARLsim sim("MultiRuntimes.shffleGroups", HYBRID_MODE, SILENT, 0, randSeed);

	// configure the network
	int gExc = sim.createGroup("exc", 800, EXCITATORY_NEURON, partitionA, CPU_CORES);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gInh = sim.createGroup("inh", 200, INHIBITORY_NEURON, partitionB, CPU_CORES);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	int gInput = sim.createSpikeGeneratorGroup("input", 800, EXCITATORY_NEURON, partitionC, CPU_CORES);

	sim.connect(gInput, gExc, "one-to-one", RangeWeight(30.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gExc, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gInh, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInh, gExc, "random", RangeWeight(5.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setConductances(false);

	// build the network
	sim.setupNetwork();

	// set some monitors
	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
	SpikeMonitor* smInh = sim.setSpikeMonitor(gInh, "NULL");
	SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

	//setup some baseline input
	PoissonRate in(800);
	in.setRates(1.0f);
	sim.setSpikeRate(gInput, &in);

	smInput->startRecording();
	smExc->startRecording();
	smInh->startRecording();

	sim.runNetwork(4, 0, false);

	smInput->stopRecording();
	smExc->stopRecording();
	smInh->stopRecording();

	// store output
	*smExcRate = smExc->getPopMeanFiringRate();
	*smInhRate = smInh->getPopMeanFiringRate();
	*smInputRate = smInput->getPopMeanFiringRate();
}


TEST(cpuMultiRuntimes, shuffleGroups0) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int partitionA = 0;
	for (int partitionB = 0; partitionB < 5; partitionB++) {
		for (int partitionC = 0; partitionC < 5; partitionC++) {
			float smExcRate;
			float smInhRate;
			float smInputRate;

			run_shuffleGroups(partitionA, partitionB, partitionC, &smExcRate, &smInhRate, &smInputRate);

			EXPECT_NEAR(smExcRate, 6.1, 0.4);
			EXPECT_NEAR(smInhRate, 29.0, 2.0);
			EXPECT_NEAR(smInputRate, 1.0, 0.1);
		}
	}
}

TEST(cpuMultiRuntimes, shuffleGroups1) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int partitionA = 1;
	for (int partitionB = 0; partitionB < 5; partitionB++) {
		for (int partitionC = 0; partitionC < 5; partitionC++) {
			float smExcRate;
			float smInhRate;
			float smInputRate;

			run_shuffleGroups(partitionA, partitionB, partitionC, &smExcRate, &smInhRate, &smInputRate);

			EXPECT_NEAR(smExcRate, 6.1, 0.4);
			EXPECT_NEAR(smInhRate, 29.0, 2.0);
			EXPECT_NEAR(smInputRate, 1.0, 0.1);
		}
	}
}

TEST(cpuMultiRuntimes, shuffleGroups2) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int partitionA = 2;
	for (int partitionB = 0; partitionB < 5; partitionB++) {
		for (int partitionC = 0; partitionC < 5; partitionC++) {
			float smExcRate;
			float smInhRate;
			float smInputRate;

			run_shuffleGroups(partitionA, partitionB, partitionC, &smExcRate, &smInhRate, &smInputRate);

			EXPECT_NEAR(smExcRate, 6.1, 0.4);
			EXPECT_NEAR(smInhRate, 29.0, 2.0);
			EXPECT_NEAR(smInputRate, 1.0, 0.1);
		}
	}
}

TEST(cpuMultiRuntimes, shuffleGroups3) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int partitionA = 3;
	for (int partitionB = 0; partitionB < 5; partitionB++) {
		for (int partitionC = 0; partitionC < 5; partitionC++) {
			float smExcRate;
			float smInhRate;
			float smInputRate;

			run_shuffleGroups(partitionA, partitionB, partitionC, &smExcRate, &smInhRate, &smInputRate);

			EXPECT_NEAR(smExcRate, 6.1, 0.4);
			EXPECT_NEAR(smInhRate, 29.0, 2.0);
			EXPECT_NEAR(smInputRate, 1.0, 0.1);
		}
	}
}

TEST(cpuMultiRuntimes, shuffleGroups4) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int partitionA = 4;
	for (int partitionB = 0; partitionB < 5; partitionB++) {
		for (int partitionC = 0; partitionC < 5; partitionC++) {
			float smExcRate;
			float smInhRate;
			float smInputRate;

			run_shuffleGroups(partitionA, partitionB, partitionC, &smExcRate, &smInhRate, &smInputRate);

			EXPECT_NEAR(smExcRate, 6.1, 0.4);
			EXPECT_NEAR(smInhRate, 29.0, 2.0);
			EXPECT_NEAR(smInputRate, 1.0, 0.1);
		}
	}
}
