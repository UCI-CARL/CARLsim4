#include <snn.h>
#include "carlsim_tests.h"

/// ****************************************************************************
/// current-based (cuba) model
/// ****************************************************************************


/*!
 * \brief testing of CUBA mode for CPU and GPU modes
 * For now this tests to make sure that the firing rates closely resemble a
 * Matlab implementation of an izhikevich CUBA neuron.
 *
 */

// FIXME: this could be at the interface-level, if we implemented the correct
// neuron/connection monitors or if we implemented the required setters and
// getters in the interface at least temporarily.
TEST(CUBA, CPU_MODE) {

	//EXPECT_TRUE(sim->sim_with_conductances);
	//EXPECT_FLOAT_EQ(sim->dAMPA,1.0f-1.0f/tdAMPA);
	//EXPECT_TRUE(sim->sim_with_NMDA_rise);
	//EXPECT_FLOAT_EQ(sim->rNMDA,1.0f-1.0f/trNMDA);
	//EXPECT_FLOAT_EQ(sim->sNMDA, 1.0/(exp(-tmax/tdNMDA)-exp(-tmax/trNMDA))); // scaling factor, 1 over max amplitude

	//EXPECT_FLOAT_EQ(sim->dNMDA,1.0f-1.0f/tdNMDA);
	//EXPECT_FLOAT_EQ(sim->dGABAa,1.0f-1.0f/tdGABAa);
	//EXPECT_TRUE(sim->sim_with_GABAb_rise);
	//EXPECT_FLOAT_EQ(sim->rGABAb,1.0f-1.0f/trGABAb);
	//EXPECT_FLOAT_EQ(sim->sGABAb, 1.0/(exp(-tmax/tdGABAb)-exp(-tmax/trGABAb))); // scaling factor, 1 over max amplitude
	//EXPECT_FALSE(sim->sim_with_GABAb_rise);
	//EXPECT_FLOAT_EQ(sim->dGABAb,1.0f-1.0f/tdGABAb);

}

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
}

// Testing fidelity of CPU and GPU versions of CUBA mode. They should have identical
// spike patterns for identical inputs.
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
}

