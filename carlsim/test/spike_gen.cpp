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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 11/26/2014
 */ 

#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <vector>


// tests whether the binary file created by setSpikeMonitor matches the specifications of PeriodicSpikeGenerator
TEST(SpikeGen, PeriodicSpikeGenerator) {
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
	ASSERT_EQ(inputSize0/2, nNeur * (int)rate);
	ASSERT_EQ(inputSize1/2, nNeur * ((int)rate-1));

	for (int i=0; i<inputSize0; i+=2) {
		EXPECT_EQ(inputArray0[i]%isi, 0);
	}
	for (int i=0; i<inputSize1; i+=2) {
		EXPECT_EQ(inputArray1[i]%isi, 0);
	}

	if (inputArray0!=NULL) delete[] inputArray0;
	if (inputArray1!=NULL) delete[] inputArray1;
}

TEST(SpikeGen, PeriodicSpikeGeneratorDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(0.0);},"");
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(-10.0);},"");
}

TEST(SpikeGen, SpikeGeneratorFromFile) {
	int isi = 100; // ms
	double rate = 1000.0/isi;
	int nNeur = 5;
	CARLsim sim("SpikeGeneratorFromFile",CPU_MODE,SILENT,0,42);

	int g1 = sim.createGroup("g1", 1, EXCITATORY_NEURON);		
	sim.setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);

	int g0 = sim.createSpikeGeneratorGroup("Input0",nNeur,EXCITATORY_NEURON);
	PeriodicSpikeGenerator spkGen0(rate,true);
	sim.setSpikeGenerator(g0, &spkGen0);

	sim.setConductances(true);

	// add some dummy connections so we can actually run the network
	sim.connect(g0,g1,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setupNetwork();
	sim.setSpikeMonitor(g0,"spkInputGrp0.dat"); // save spikes to file
	sim.runNetwork(1,0);


	// now that we have created the spike file, run a different network using the spike file from above
	CARLsim sim2("SpikeGeneratorFromVector2",CPU_MODE,SILENT,0,42);

	g0 = sim2.createSpikeGeneratorGroup("Input",nNeur,EXCITATORY_NEURON);
	SpikeGeneratorFromFile spkGen2("spkInputGrp0.dat");
	sim2.setSpikeGenerator(g0, &spkGen2);

	g1 = sim2.createGroup("g1", 1, EXCITATORY_NEURON);		
	sim2.setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);
	sim2.setConductances(true);

	// add some dummy connections so we can actually run the network
	sim2.connect(g0,g1,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim2.setupNetwork();
	sim2.setSpikeMonitor(g0,"spkInputFinal.dat"); // save spikes to file
	sim2.runNetwork(1,0);

	// explicitly read the spike file to make sure
	int *inputArray0 = NULL;
	long inputSize0;
	readAndReturnSpikeFile("spkInputFinal.dat",inputArray0,inputSize0);
	ASSERT_EQ(inputSize0/2, nNeur * (int)rate);

	int j=0;
	for (int i=0; i<inputSize0; i+=2)
		EXPECT_EQ(inputArray0[i]%isi, 0);

	if (inputArray0!=NULL) delete[] inputArray0;
}

TEST(SpikeGen, SpikeGeneratorFromFileDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("");},"");
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("thisFile/doesNot/exist.dat");},"");
}

// tests whether the binary spike file created by setSpikeMonitor contains the same spike times as specified
// by a spike vector
TEST(SpikeGen, SpikeGeneratorFromVector) {
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
	ASSERT_EQ(inputSize0/2, spkTimes.size());

	for (int i=0; i<spkTimes.size(); i++) {
		EXPECT_EQ(inputArray0[i*2], spkTimes[i]);
	}

	if (inputArray0!=NULL) delete[] inputArray0;
}

TEST(SpikeGen, SpikeGeneratorFromVectorDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	std::vector<int> emptyVec, negativeVec;
	negativeVec.push_back(0);
	negativeVec.push_back(-1);

	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(emptyVec);},"");
	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(negativeVec);},"");
}