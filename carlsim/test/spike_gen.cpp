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
#include <spikegen_from_file.h>
#include <spikegen_from_vector.h>



// tests whether the binary file created by setSpikeMonitor matches the specifications of PeriodicSpikeGenerator
TEST(spikeGenFunc, PeriodicSpikeGenerator) {
	int isi = 100; // ms
	double rate = 1000.0/isi;
	int nNeur = 5;
	CARLsim sim("PeriodicSpikeGenerator",CPU_MODE,SILENT,0,42);

	int g2 = sim.createGroup("g2", 1, EXCITATORY_NEURON);		
	sim.setNeuronParameters(g2, 0.02, 0.2, -65.0, 8.0);

	int g0 = sim.createSpikeGeneratorGroup("Input0",nNeur,EXCITATORY_NEURON);
	int g1 = sim.createSpikeGeneratorGroup("Input1",nNeur,EXCITATORY_NEURON);
	PeriodicSpikeGenerator spkGen0(rate,true);
	PeriodicSpikeGenerator spkGen1(rate,false);
	sim.setSpikeGenerator(g0, &spkGen0);
	sim.setSpikeGenerator(g1, &spkGen1);

	sim.setConductances(true);

	// add some dummy connections so we can actually run the network
	sim.connect(g0,g2,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setupNetwork();
	sim.setSpikeMonitor(g0,"spkInputGrp0.dat"); // save spikes to file
	sim.setSpikeMonitor(g1,"spkInputGrp1.dat"); // save spikes to file
	sim.runNetwork(1,0);


	// explicitly read the spike file to make sure
	int *inputArray0 = NULL, *inputArray1 = NULL;
	long inputSize0, inputSize1;
	readAndReturnSpikeFile("spkInputGrp0.dat",inputArray0,inputSize0);
	readAndReturnSpikeFile("spkInputGrp1.dat",inputArray1,inputSize1);

	bool isSize0Correct = inputSize0/2 == nNeur * (int)rate;
	bool isSize1Correct = inputSize1/2 == nNeur * ((int)rate-1);
	EXPECT_TRUE(isSize0Correct);
	EXPECT_TRUE(isSize1Correct);

	if (isSize0Correct) {
		for (int i=0; i<inputSize0; i+=2) {
			EXPECT_EQ(inputArray0[i]%isi, 0);
		}
	}

	if (isSize1Correct) {
		for (int i=0; i<inputSize1; i+=2) {
			EXPECT_EQ(inputArray1[i]%isi, 0);
		}
	}

	if (inputArray0!=NULL) delete[] inputArray0;
	if (inputArray1!=NULL) delete[] inputArray1;
}

TEST(spikeGenFunc, PeriodicSpikeGeneratorDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(0.0);},"");
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(-10.0);},"");
}

TEST(spikeGenFunc, SpikeGeneratorFromFile) {
	PoissonRate* poiss = NULL;
	SpikeGeneratorFromFile* sgf = NULL;
	CARLsim* sim = NULL;
	std::string fileName0 = "results/spk_run0.dat", fileName1 = "results/spk_run1.dat";
	std::vector< std::vector<int> > spkVec0, spkVec1;
	SpikeMonitor *SM0, *SM1;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int isCOBA=0; isCOBA<=1; isCOBA++) {
			for (int run=0; run<=1; run++) {
				sim = new CARLsim("SpikeGeneratorFromFile",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
				int g1 = sim->createGroup("g1", 1, EXCITATORY_NEURON);		
				sim->setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);

				int g0 = sim->createSpikeGeneratorGroup("g0",1,EXCITATORY_NEURON);
				if (run==1) {
					// second run: load from file and compare spike times
					sgf = new SpikeGeneratorFromFile(fileName0);
					sim->setSpikeGenerator(g0, sgf);
				}
				sim->connect(g0,g1,"full",RangeWeight(0.1f), 0.5f);
				sim->setConductances(isCOBA);
				sim->setupNetwork();

				if (run==0) {
					// first run: use Poisson spike generator as ground truth
					// generate the spike file and run in one piece
					poiss = new PoissonRate(1);
					poiss->setRates(50.0f);
					sim->setSpikeRate(g0, poiss);
					SM0 = sim->setSpikeMonitor(g0, fileName0);
					SM0->startRecording();
					sim->runNetwork(1,0,false);
					SM0->stopRecording();
					spkVec0 = SM0->getSpikeVector2D();
				} else {
					// second run: generate new spike file, schedule in slices
					SM1 = sim->setSpikeMonitor(g0, fileName1);
					SM1->startRecording();
					for (int i=0; i<200; i++) {
						sim->runNetwork(0,5,false);
					}
					SM1->stopRecording();
					spkVec1 = SM1->getSpikeVector2D();
				}

				if (run==1) {
					// make sure we have the same spikes in both spike vectors
					EXPECT_EQ(spkVec0.size(), spkVec1.size());
					if (spkVec0.size() == spkVec1.size()) {
						for (int neurId=0; neurId<spkVec0.size(); neurId++) {
							EXPECT_EQ(spkVec0[neurId].size(), spkVec1[neurId].size());
							if (spkVec0[neurId].size() == spkVec1[neurId].size()) {
								for (int spk=0; spk<spkVec0[neurId].size(); spk++) {
									EXPECT_EQ(spkVec0[neurId][spk], spkVec1[neurId][spk]);
								}
							}
						}
					}
					spkVec0.clear();
					spkVec1.clear();
				}

				// deallocate
				if (poiss != NULL) {
					delete poiss;
				}
				if (sgf != NULL) {
					delete sgf;
				}
				if (sim != NULL) {
					delete sim;
				}
				poiss = NULL;
				sgf = NULL;
				sim = NULL;
			}
		}
	}
}

TEST(spikeGenFunc, SpikeGeneratorFromFileLoadFile) {
	PoissonRate* poiss = NULL;
	SpikeGeneratorFromFile* sgf = NULL;
	CARLsim* sim = NULL;
	std::string fileName0 = "results/spk_run0.dat", fileName1 = "results/spk_run1.dat";
	std::vector< std::vector<int> > spkVec0, spkVec1;
	SpikeMonitor *SM0, *SM1;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int isCOBA=0; isCOBA<=1; isCOBA++) {
			for (int run=0; run<=1; run++) {
				sim = new CARLsim("SpikeGeneratorFromFileLoadFile",mode?GPU_MODE:CPU_MODE,SILENT,1,42);
				int g1 = sim->createGroup("g1", 1, EXCITATORY_NEURON);		
				sim->setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);

				int g0 = sim->createSpikeGeneratorGroup("g0",1,EXCITATORY_NEURON);
				if (run==1) {
					// second run: load from file and compare spike times
					sgf = new SpikeGeneratorFromFile(fileName0);
					sim->setSpikeGenerator(g0, sgf);
				}
				sim->connect(g0,g1,"full",RangeWeight(0.1f), 0.5f);
				sim->setConductances(isCOBA);
				sim->setupNetwork();

				if (run==0) {
					// first run: use Poisson spike generator as ground truth
					// generate the spike file and run in one piece
					poiss = new PoissonRate(1);
					poiss->setRates(50.0f);
					sim->setSpikeRate(g0, poiss);
					SM0 = sim->setSpikeMonitor(g0, fileName0);
					SM0->startRecording();
					sim->runNetwork(1,0,false);
					SM0->stopRecording();
					spkVec0 = SM0->getSpikeVector2D();
				} else {
					// second run: generate new spike file, schedule in slices
					SM1 = sim->setSpikeMonitor(g0, fileName1);
					SM1->startRecording();
					for (int i=0; i<200; i++) {
						sim->runNetwork(0,5,false);
					}
					SM1->stopRecording();
					spkVec1 = SM1->getSpikeVector2D();

					// make sure we have the same spikes in both spike vectors
					EXPECT_EQ(spkVec0.size(), spkVec1.size());
					if (spkVec0.size() == spkVec1.size()) {
						for (int neurId=0; neurId<spkVec0.size(); neurId++) {
							EXPECT_EQ(spkVec0[neurId].size(), spkVec1[neurId].size());
							if (spkVec0[neurId].size() == spkVec1[neurId].size()) {
								for (int spk=0; spk<spkVec0[neurId].size(); spk++) {
									EXPECT_EQ(spkVec0[neurId][spk], spkVec1[neurId][spk]);
								}
							}
						}
					}

					// load same file again, choose right offset
					int currentTime = (int)sim->getSimTime();
					sgf->loadFile(fileName0, currentTime);
					SM1->startRecording();
					for (int i=0; i<200; i++) {
						sim->runNetwork(0,5,false);
					}
					SM1->stopRecording();
					spkVec1.clear();
					spkVec1 = SM1->getSpikeVector2D();

					// make sure we have the same spikes again
					EXPECT_EQ(spkVec0.size(), spkVec1.size());
					if (spkVec0.size() == spkVec1.size()) {
						for (int neurId=0; neurId<spkVec0.size(); neurId++) {
							EXPECT_EQ(spkVec0[neurId].size(), spkVec1[neurId].size());
							if (spkVec0[neurId].size() == spkVec1[neurId].size()) {
								for (int spk=0; spk<spkVec0[neurId].size(); spk++) {
									EXPECT_EQ(spkVec0[neurId][spk]+currentTime, spkVec1[neurId][spk]);
								}
							}
						}
					}
					spkVec0.clear();
					spkVec1.clear();
				}

				// deallocate
				if (poiss != NULL) {
					delete poiss;
				}
				if (sgf != NULL) {
					delete sgf;
				}
				if (sim != NULL) {
					delete sim;
				}
				poiss = NULL;
				sgf = NULL;
				sim = NULL;
			}
		}
	}
}

TEST(spikeGenFunc, SpikeGeneratorFromFileDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("");},"");
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("thisFile/doesNot/exist.dat");},"");
}

// tests whether the binary spike file created by setSpikeMonitor contains the same spike times as specified
// by a spike vector
TEST(spikeGenFunc, SpikeGeneratorFromVector) {
	int spkTimesArr[11] = {13, 42, 99, 102, 200, 523, 738, 820, 821, 912, 989};
	std::vector<int> spkTimes(&spkTimesArr[0], &spkTimesArr[0]+11);

	int nNeur = 5;
	CARLsim sim("SpikeGeneratorFromVector",CPU_MODE,SILENT,0,42);

	int g1 = sim.createGroup("g1", 1, EXCITATORY_NEURON);		
	sim.setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);

	int g0 = sim.createSpikeGeneratorGroup("Input",nNeur,EXCITATORY_NEURON);
	SpikeGeneratorFromVector spkGen(spkTimes);
	sim.setSpikeGenerator(g0, &spkGen);

	sim.setConductances(true);

	// add some dummy connections so we can actually run the network
	sim.connect(g0,g1,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setupNetwork();
	sim.setSpikeMonitor(g0,"spkInputGrp0.dat"); // save spikes to file
	sim.runNetwork(1,0);

	// explicitly read the spike file to make sure
	int *inputArray0 = NULL;
	long inputSize0;
	readAndReturnSpikeFile("spkInputGrp0.dat",inputArray0,inputSize0);
	bool isSize0Correct = inputSize0/2 == spkTimes.size();
	EXPECT_TRUE(isSize0Correct);

	if (isSize0Correct) {
		for (int i=0; i<spkTimes.size(); i++) {
			EXPECT_EQ(inputArray0[i*2], spkTimes[i]);
		}
	}

	if (inputArray0!=NULL) delete[] inputArray0;
}

TEST(spikeGenFunc, SpikeGeneratorFromVectorDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	std::vector<int> emptyVec, negativeVec;
	negativeVec.push_back(0);
	negativeVec.push_back(-1);

	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(emptyVec);},"");
	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(negativeVec);},"");
}
