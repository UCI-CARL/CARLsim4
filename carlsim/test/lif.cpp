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

TEST(LIF, cpuFI) {
	// MATLAB ground truth
	float matlab_rates[] = {0.0f, 0.0f, 0.0f, 0.0f, 33.3f, 58.8f, 76.9f, 90.9f, 100.0f, 111.1f, 125.0f};

	// create a network on CPU of single LIF neuron and a dummy Izhi neuron for making a connection
	
	// ---------------- CONFIG STATE -------------------
	
	// create a network on CPU
	int numGPUs = 1;
	int randSeed = 123;
	CARLsim* sim = new CARLsim("Simple LIF neuron tuning", CPU_MODE, SILENT, numGPUs, randSeed);

	// configure the network
	// set up a single LIF neuron network to record its fi curve
	Grid3D gridSingle(1,1,1); // pre is on a 1x1 grid
	Grid3D gridDummy(1,1,1); // dummy is on a 1x1 grid
	int gSingleLIF=sim->createGroupLIF("input", gridSingle, EXCITATORY_NEURON, 0, CPU_CORES);
	int gDummyIzh=sim->createGroup("output", gridDummy, EXCITATORY_NEURON, 1, CPU_CORES);

	// set neuron parameters
	sim->setNeuronParametersLIF(gSingleLIF, 10, 2, -50.0f, -65.0f, RangeRmem(5.0f));
	sim->setNeuronParameters(gDummyIzh, 0.02f, 0.2f, -65.0f, 8.0f);
	
	// connect
	sim->connect(gSingleLIF, gDummyIzh, "full", RangeWeight(0.05), 1.0f, RangeDelay(1));
	sim->setConductances(false);
	sim->setIntegrationMethod(FORWARD_EULER, 1);


	// ---------------- SETUP STATE -------------------
	// build the network
	sim->setupNetwork();

	// set some monitors
	SpikeMonitor* smLIF = sim->setSpikeMonitor(gSingleLIF, "NULL");

	// ---------------- RUN STATE -------------------

	// run for a total of 10 seconds for different amount of external current
	for (int i=0; i<=10; i++) {
		std::vector<float> current(1, (float)i*0.8f);
		sim->setExternalCurrent(gSingleLIF, current);
		
		smLIF->startRecording();
		sim->runNetwork(10,0);
		smLIF->stopRecording();

		EXPECT_NEAR(smLIF->getPopMeanFiringRate(), matlab_rates[i], 0.5f);
	}
	delete sim;
}

TEST(LIF, gpuFI) {
	// MATLAB ground truth
	float matlab_rates[] = {0.0f, 0.0f, 0.0f, 0.0f, 33.3f, 58.8f, 76.9f, 90.9f, 100.0f, 111.1f, 125.0f};

	// create a network on GPU of single LIF neuron and a dummy Izhi neuron for making a connection
	
	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 123;
	CARLsim* sim = new CARLsim("Simple LIF neuron tuning", GPU_MODE, SILENT, numGPUs, randSeed);

	// configure the network
	// set up a single LIF neuron network to record its fi curve
	Grid3D gridSingle(1,1,1); // pre is on a 1x1 grid
	Grid3D gridDummy(1,1,1); // dummy is on a 1x1 grid
	int gSingleLIF=sim->createGroupLIF("input", gridSingle, EXCITATORY_NEURON, 0, GPU_CORES);
	int gDummyIzh=sim->createGroup("output", gridDummy, EXCITATORY_NEURON, 0, GPU_CORES);

	// set neuron parameters
	sim->setNeuronParametersLIF(gSingleLIF, 10, 2, -50.0f, -65.0f, RangeRmem(5.0f));
	sim->setNeuronParameters(gDummyIzh, 0.02f, 0.2f, -65.0f, 8.0f);
	
	// connect
	sim->connect(gSingleLIF, gDummyIzh, "full", RangeWeight(0.05), 1.0f, RangeDelay(1));
	sim->setConductances(false);
	sim->setIntegrationMethod(FORWARD_EULER, 1);


	// ---------------- SETUP STATE -------------------
	// build the network
	sim->setupNetwork();

	// set some monitors
	SpikeMonitor* smLIF = sim->setSpikeMonitor(gSingleLIF, "NULL");

	// ---------------- RUN STATE -------------------

	// run for a total of 10 seconds for different amount of external current
	for (int i=0; i<=10; i++) {
		std::vector<float> current(1, (float)i*0.8f);
		sim->setExternalCurrent(gSingleLIF, current);
		
		smLIF->startRecording();
		sim->runNetwork(10,0);
		smLIF->stopRecording();

		EXPECT_NEAR(smLIF->getPopMeanFiringRate(), matlab_rates[i], 0.5f);
	}
	delete sim;
}

TEST(LIF, gpuVscpu10steps) {

	// create a network on GPU of single LIF neuron and a dummy Izhi neuron for making a connection
	
	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 123;
	CARLsim* sim = new CARLsim("Simple LIF neuron tuning", GPU_MODE, SILENT, numGPUs, randSeed);

	// configure the network
	// set up a single LIF neuron network to record its fi curve
	Grid3D gridSingle(1,1,1); // pre is on a 1x1 grid
	Grid3D gridDummy(1,1,1); // dummy is on a 1x1 grid
	int gSingleLIF_gpu=sim->createGroupLIF("input_gpu", gridSingle, EXCITATORY_NEURON, 0, GPU_CORES);
	int gDummyIzh_gpu=sim->createGroup("output_gpu", gridDummy, EXCITATORY_NEURON, 0, GPU_CORES);

	int gSingleLIF_cpu=sim->createGroupLIF("input_cpu", gridSingle, EXCITATORY_NEURON, 0, CPU_CORES);
	int gDummyIzh_cpu=sim->createGroup("output_cpu", gridDummy, EXCITATORY_NEURON, 1, CPU_CORES);

	// set neuron parameters
	sim->setNeuronParametersLIF(gSingleLIF_gpu, 10, 2, -50.0f, -65.0f, RangeRmem(5.0f));
	sim->setNeuronParameters(gDummyIzh_gpu, 0.02f, 0.2f, -65.0f, 8.0f);

	sim->setNeuronParametersLIF(gSingleLIF_cpu, 10, 2, -50.0f, -65.0f, RangeRmem(5.0f));
	sim->setNeuronParameters(gDummyIzh_cpu, 0.02f, 0.2f, -65.0f, 8.0f);
	
	// connect
	sim->connect(gSingleLIF_gpu, gDummyIzh_gpu, "full", RangeWeight(0.05), 1.0f, RangeDelay(1));
	sim->connect(gSingleLIF_cpu, gDummyIzh_cpu, "full", RangeWeight(0.05), 1.0f, RangeDelay(1));

	sim->setConductances(false);
	sim->setIntegrationMethod(FORWARD_EULER, 10);

	// ---------------- SETUP STATE -------------------
	// build the network
	sim->setupNetwork();

	// set some monitors
	SpikeMonitor* smLIF_gpu = sim->setSpikeMonitor(gSingleLIF_gpu, "NULL");
	SpikeMonitor* smLIF_cpu = sim->setSpikeMonitor(gSingleLIF_cpu, "NULL");

	// ---------------- RUN STATE -------------------

	// run for a total of 10 seconds for different amount of external current
	for (int i=0; i<=10; i++) {
		std::vector<float> current(1, (float)i*0.8f);
		sim->setExternalCurrent(gSingleLIF_gpu, current);
		sim->setExternalCurrent(gSingleLIF_cpu, current);
		
		smLIF_gpu->startRecording();
		smLIF_cpu->startRecording();
		sim->runNetwork(10,0);
		smLIF_gpu->stopRecording();
		smLIF_cpu->stopRecording();

		EXPECT_EQ(smLIF_gpu->getPopMeanFiringRate(), smLIF_cpu->getPopMeanFiringRate());
	}
	delete sim;
}
