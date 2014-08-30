#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <periodic_spikegen.h>
#include <spikegen_from_file.h>
#include <spikegen_from_vector.h>

#include <vector>


TEST(SpikeGen, PeriodicSpikeGenerator) {
	int isi = 100; // ms
	double rate = 1000.0/isi;
	int nNeur = 5;
	CARLsim sim("PeriodicSpikeGenerator",CPU_MODE,SILENT,0,1,42);

	int g0 = sim.createSpikeGeneratorGroup("Input0",nNeur,EXCITATORY_NEURON);
	int g1 = sim.createSpikeGeneratorGroup("Input1",nNeur,EXCITATORY_NEURON);
	PeriodicSpikeGenerator spkGen0(rate,true);
	PeriodicSpikeGenerator spkGen1(rate,false);
	sim.setSpikeGenerator(g0, &spkGen0);
	sim.setSpikeGenerator(g1, &spkGen1);

	int g2 = sim.createGroup("g2", 1, EXCITATORY_NEURON);		
	sim.setNeuronParameters(g2, 0.02, 0.2, -65.0, 8.0);
	sim.setConductances(true);

	// add some dummy connections so we can actually run the network
	sim.connect(g0,g2,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), SYN_FIXED);

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
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(0.0);},"");
	EXPECT_DEATH({PeriodicSpikeGenerator spkGen(-10.0);},"");
}

TEST(SpikeGen, SpikeGeneratorFromFile) {

}

TEST(SpikeGen, SpikeGeneratorFromFileDeath) {
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("");},"");
	EXPECT_DEATH({SpikeGeneratorFromFile spkGen("thisFile/doesNot/exist.dat");},"");
}

TEST(SpikeGen, SpikeGeneratorFromVector) {

}

TEST(SpikeGen, SpikeGeneratorFromVectorDeath) {
	std::vector<int> emptyVec, negativeVec;
	negativeVec.push_back(0);
	negativeVec.push_back(-1);

	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(emptyVec);},"");
	EXPECT_DEATH({SpikeGeneratorFromVector spkGen(negativeVec);},"");
}