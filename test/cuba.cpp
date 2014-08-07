#include <snn.h>
#include "carlsim_tests.h"

/// ****************************************************************************
/// current-based (cuba) model
/// ****************************************************************************


/*!
 * \brief testing of CUBA mode for CPU and GPU modes
 * For now this tests to make sure that the firing rates closely resemble a
 * Matlab implementation of an izhikevich CUBA neuron. The matlab script
 * is found in tests/scripts and is called runCUBA.m. There are two test cases:
 * a test case that produces lower firing (LF) rates in the output RS neuron
 * (1 Hz) and a test case that produces higher firing rates (HF) in the output
 * RS neuron (13 Hz). The parameters used as input to the Matlab script and
 * this test script are as follows:
 *
 * LF case: input firing rate: 50 Hz
 *          weight value from input to output neuron: 15
 *          run time: 1 second
 *          resulting output neuron firing rate: 1 Hz
 *
 * HF case: input firing rate: 25 Hz
 *          weight value from input to output neuron: 25
 *          run time: 1 second
 *          resulting output neuron firing rate: 13 Hz
 *
 * Note that the CARLsim cuba simulations should give the same number of spikes
 * as the Matlab file but with slightly different spike times (offset by 2-3
 * ms). This is probably due to differences in the execution order of simulation
 * functions.
 *
 * Using the Matlab script: runCUBA(runMs, rate, wt)
 * runMs is the number of ms to run the script for (1000 ms in this case)
 * rate is the firing rate of the input neuron (15 or 25 Hz for our cases)
 * wt is the strength of the weight between the input and output neuron (15 or
 * 25 for our cases)
 * The script returns the firing rate and spike times of the output RS neuron.
 */

// Testing GPU version of CUBA mode with input that should result in a
// low-firing rate (1 Hz)
TEST(CUBA, GPU_MODE_LF) {
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU

	// SpikeMonitor pointers to grab setSpikeMonitor output
	SpikeMonitor* spikeMonG1;
	SpikeMonitor* spikeMonGin;

	// create a network
	CARLsim sim("random",GPU_MODE,SILENT,ithGPU,1,42);

	// create spike generator that produces periodic (deterministic) spike trains
	PeriodicSpikeGenerator* spkGenG0 = NULL;

	int g1=sim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int gin=sim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	// Regular spiking neuron parameters
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// create our periodic spike generator
	bool spikeAtZero = true;
	spkGenG0 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim.setSpikeGenerator(gin, spkGenG0);

	sim.connect(gin,g1,"random", RangeWeight(15), 1.0f, RangeDelay(1), SYN_FIXED);

	sim.setupNetwork();

	spikeMonG1=sim.setSpikeMonitor(g1,"examples/random/results/spikes.dat"); // put spike times into spikes.dat
	spikeMonGin=sim.setSpikeMonitor(gin);

	spikeMonG1->startRecording();
	spikeMonGin->startRecording();

	sim.runNetwork(1,0,false);

	spikeMonG1->stopRecording();
	spikeMonGin->stopRecording();

	spikeMonG1->print(true);
	spikeMonGin->print(true);

	int spikeNumG1=spikeMonG1->getPopNumSpikes();
	int spikeNumGin=spikeMonGin->getPopNumSpikes();

	EXPECT_EQ(spikeNumG1,1);
	EXPECT_EQ(spikeNumGin,50);

	delete spkGenG0;
}

// Testing GPU version of CUBA mode with input that should result in a
// higher firing rate (13 Hz)
TEST(CUBA, GPU_MODE_HF) {
	// simulation details
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU

	// SpikeMonitor pointers to grab setSpikeMonitor output
	SpikeMonitor* spikeMonG1;
	SpikeMonitor* spikeMonGin;

	// create a network
	CARLsim sim("random",GPU_MODE,SILENT,ithGPU,1,42);

	// create spike generator that produces periodic (deterministic) spike trains
	PeriodicSpikeGenerator* spkGenG0 = NULL;

	int g1=sim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int gin=sim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	// Regular spiking neuron parameters
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// create our periodic spike generator
	bool spikeAtZero = true;
	spkGenG0 = new PeriodicSpikeGenerator(25.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim.setSpikeGenerator(gin, spkGenG0);

	sim.connect(gin,g1,"random", RangeWeight(25), 1.0f, RangeDelay(1), SYN_FIXED);

	sim.setupNetwork();

	spikeMonG1=sim.setSpikeMonitor(g1,"examples/random/results/spikes.dat"); // put spike times into spikes.dat
	spikeMonGin=sim.setSpikeMonitor(gin);

	spikeMonG1->startRecording();
	spikeMonGin->startRecording();

	sim.runNetwork(1,0,false);

	spikeMonG1->stopRecording();
	spikeMonGin->stopRecording();

	spikeMonG1->print(true);
	spikeMonGin->print(true);

	int spikeNumG1=spikeMonG1->getPopNumSpikes();
	int spikeNumGin=spikeMonGin->getPopNumSpikes();

	EXPECT_EQ(spikeNumG1,13);
	EXPECT_EQ(spikeNumGin,25);

	delete spkGenG0;
}

// Testing CPU version of CUBA mode with input that should result in a
// low-firing rate (1 Hz)
TEST(CUBA, CPU_MODE_LF) {
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU

	// SpikeMonitor pointers to grab setSpikeMonitor output
	SpikeMonitor* spikeMonG1;
	SpikeMonitor* spikeMonGin;

	// create a network
	CARLsim sim("random",CPU_MODE,SILENT,ithGPU,1,42);

	// create spike generator that produces periodic (deterministic) spike trains
	PeriodicSpikeGenerator* spkGenG0 = NULL;

	int g1=sim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int gin=sim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	// Regular spiking neuron parameters
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// create our periodic spike generator
	bool spikeAtZero = true;
	spkGenG0 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim.setSpikeGenerator(gin, spkGenG0);

	sim.connect(gin,g1,"random", RangeWeight(15), 1.0f, RangeDelay(1), SYN_FIXED);

	sim.setupNetwork();

	spikeMonG1=sim.setSpikeMonitor(g1,"examples/random/results/spikes.dat"); // put spike times into spikes.dat
	spikeMonGin=sim.setSpikeMonitor(gin);

	spikeMonG1->startRecording();
	spikeMonGin->startRecording();

	sim.runNetwork(1,0,false);

	spikeMonG1->stopRecording();
	spikeMonGin->stopRecording();

	spikeMonG1->print(true);
	spikeMonGin->print(true);

	int spikeNumG1=spikeMonG1->getPopNumSpikes();
	int spikeNumGin=spikeMonGin->getPopNumSpikes();

	EXPECT_EQ(spikeNumG1,1);
	EXPECT_EQ(spikeNumGin,50);

	delete spkGenG0;
}

// Testing CPU version of CUBA mode with input that should result in a
// higher firing rate (13 Hz)
TEST(CUBA, CPU_MODE_HF) {
	// simulation details
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU

	// SpikeMonitor pointers to grab setSpikeMonitor output
	SpikeMonitor* spikeMonG1;
	SpikeMonitor* spikeMonGin;

	// create a network
	CARLsim sim("random",GPU_MODE,SILENT,ithGPU,1,42);

	// create spike generator that produces periodic (deterministic) spike trains
	PeriodicSpikeGenerator* spkGenG0 = NULL;

	int g1=sim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int gin=sim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	// Regular spiking neuron parameters
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// create our periodic spike generator
	bool spikeAtZero = true;
	spkGenG0 = new PeriodicSpikeGenerator(25.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim.setSpikeGenerator(gin, spkGenG0);

	sim.connect(gin,g1,"random", RangeWeight(25), 1.0f, RangeDelay(1), SYN_FIXED);

	sim.setupNetwork();

	spikeMonG1=sim.setSpikeMonitor(g1,"examples/random/results/spikes.dat"); // put spike times into spikes.dat
	spikeMonGin=sim.setSpikeMonitor(gin);

	spikeMonG1->startRecording();
	spikeMonGin->startRecording();

	sim.runNetwork(1,0,false);

	spikeMonG1->stopRecording();
	spikeMonGin->stopRecording();

	spikeMonG1->print(true);
	spikeMonGin->print(true);

	int spikeNumG1=spikeMonG1->getPopNumSpikes();
	int spikeNumGin=spikeMonGin->getPopNumSpikes();

	EXPECT_EQ(spikeNumG1,13);
	EXPECT_EQ(spikeNumGin,25);

	delete spkGenG0;
}

// Testing fidelity of CPU and GPU versions of CUBA mode. They should have identical
// spike patterns for identical inputs. This test checks their response to a 25 Hz
// input with a weight of 25. Both output RS neurons should have a firing rate of
// 13 Hz and identical spike times.
TEST(CUBA, CPUvsGPU) {
	// simulation details
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU
	//================BEGIN CPU COMPONENT==================================
	SpikeMonitor* cpuSpikeMonG1;
	SpikeMonitor* cpuSpikeMonGin;

	CARLsim cpuSim("random",CPU_MODE,SILENT,ithGPU,1,42);

	PeriodicSpikeGenerator* cpuSpkGenG0 = NULL;

	int cpuG1=cpuSim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int cpuGin=cpuSim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	cpuSim.setNeuronParameters(cpuG1, 0.02f, 0.2f, -65.0f, 8.0f);

	bool spikeAtZero = true;

	cpuSpkGenG0 = new PeriodicSpikeGenerator(25.0f,spikeAtZero); // periodic spiking @ 50 Hz

	cpuSim.setSpikeGenerator(cpuGin, cpuSpkGenG0);

	cpuSim.connect(cpuGin,cpuG1,"random", RangeWeight(25), 1.0f, RangeDelay(1), SYN_FIXED);

	cpuSim.setupNetwork();

	cpuSpikeMonG1=cpuSim.setSpikeMonitor(cpuG1,"examples/random/results/cpuSpikes.dat"); // put spike times into spikes.dat
	cpuSpikeMonGin=cpuSim.setSpikeMonitor(cpuGin);

	cpuSpikeMonG1->startRecording();
	cpuSpikeMonGin->startRecording();

	cpuSim.runNetwork(1,0,false);

	cpuSpikeMonG1->stopRecording();
	cpuSpikeMonGin->stopRecording();

	cpuSpikeMonG1->print(true);
	cpuSpikeMonGin->print(true);

	//================END CPU COMPONENT==================================

	//================BEGIN GPU COMPONENT================================
	SpikeMonitor* gpuSpikeMonG1;
	SpikeMonitor* gpuSpikeMonGin;

	CARLsim gpuSim("random",GPU_MODE,SILENT,ithGPU,1,42);

	PeriodicSpikeGenerator* gpuSpkGenG0 = NULL;

	int gpuG1=gpuSim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	int gpuGin=gpuSim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);

	gpuSim.setNeuronParameters(gpuG1, 0.02f, 0.2f, -65.0f, 8.0f);

	gpuSpkGenG0 = new PeriodicSpikeGenerator(25.0f,spikeAtZero); // periodic spiking @ 50 Hz

	gpuSim.setSpikeGenerator(gpuGin, gpuSpkGenG0);

	gpuSim.connect(gpuGin,gpuG1,"random", RangeWeight(25), 1.0f, RangeDelay(1), SYN_FIXED);

	gpuSim.setupNetwork();

	gpuSpikeMonG1=gpuSim.setSpikeMonitor(gpuG1,"examples/random/results/gpuSpikes.dat"); // put spike times into spikes.dat
	gpuSpikeMonGin=gpuSim.setSpikeMonitor(gpuGin);

	gpuSpikeMonG1->startRecording();
	gpuSpikeMonGin->startRecording();

	gpuSim.runNetwork(1,0,false);

	gpuSpikeMonG1->stopRecording();
	gpuSpikeMonGin->stopRecording();
	//================FINAL CALC COMPONENT===============================
	// calculate the number of spikes for CPU/GPU groups and make sure
	// they are the same. Also compare the exact spike times and make
	// sure they are the same.

	// get number of spikes for both CPU/GPU groups
	int cpuSpikeNumG1=cpuSpikeMonG1->getPopNumSpikes();
	int cpuSpikeNumGin=cpuSpikeMonGin->getPopNumSpikes();
	int gpuSpikeNumG1=gpuSpikeMonG1->getPopNumSpikes();
	int gpuSpikeNumGin=gpuSpikeMonGin->getPopNumSpikes();

	// get the spike times for both CPU/GPU groups
	std::vector<std::vector<int> > cpuSpikeVectorG1 = cpuSpikeMonG1->getSpikeVector2D();
	std::vector<std::vector<int> > cpuSpikeVectorGin = cpuSpikeMonGin->getSpikeVector2D();
	std::vector<std::vector<int> > gpuSpikeVectorG1 = gpuSpikeMonG1->getSpikeVector2D();
	std::vector<std::vector<int> > gpuSpikeVectorGin = gpuSpikeMonGin->getSpikeVector2D();

	// check to make sure they have the same number of spikes
	ASSERT_EQ(cpuSpikeNumG1,gpuSpikeNumG1);
	ASSERT_EQ(cpuSpikeNumGin,gpuSpikeNumGin);

	// check to make sure the spike times are identical
	ASSERT_EQ(cpuSpikeVectorG1[0].size(),gpuSpikeVectorG1[0].size());
	ASSERT_EQ(cpuSpikeVectorGin[0].size(),gpuSpikeVectorGin[0].size());

	for(int i=0;i<cpuSpikeVectorG1[0].size();i++){
		EXPECT_EQ(cpuSpikeVectorG1[0][i],gpuSpikeVectorG1[0][i]);
	}

	for(int i=0;i<cpuSpikeVectorGin[0].size();i++){
		EXPECT_EQ(cpuSpikeVectorGin[0][i],gpuSpikeVectorGin[0][i]);
	}

	delete cpuSpkGenG0;
	delete gpuSpkGenG0;
}

