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

TEST(MULTIGPUS, spikesSingleVsMulti) {
	// create a network on GPU
	int gExc, gExc2, gInput;
	std::vector<std::vector<int> > spikesSingleGPU, spikesMultiGPU;
	CARLsim* sim;
	
	for (int gpuId = 0; gpuId < 2; gpuId++) {
		sim = new CARLsim("test kernel", GPU_MODE, USER, 2, 42);

		// configure the network
		gExc = sim->createGroup("exc", 10, EXCITATORY_NEURON, 0);
		sim->setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

		//int gInh = sim.createGroup("inh", 20, INHIBITORY_NEURON);
		//sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS
		gExc2 = sim->createGroup("exc2", 10, EXCITATORY_NEURON, gpuId);
		sim->setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS

		gInput = sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON, 0);

		sim->connect(gInput, gExc, "one-to-one", RangeWeight(50.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		sim->connect(gExc, gExc2, "random", RangeWeight(10.0f), 0.4f, RangeDelay(1, 10), RadiusRF(-1), SYN_FIXED);

		sim->setConductances(false);

		//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

		// build the network
		sim->setupNetwork();

		// set some monitors
		SpikeMonitor* smInput = sim->setSpikeMonitor(gInput, "NULL");
		SpikeMonitor* smExc = sim->setSpikeMonitor(gExc, "NULL");
		SpikeMonitor* smExc2 = sim->setSpikeMonitor(gExc2, "NULL");
		//ConnectionMonitor* cmEE = sim->setConnectionMonitor(gExc, gExc2, "NULL");

		//setup some baseline input
		PoissonRate in(10);
		in.setRates(5.0f);
		sim->setSpikeRate(gInput, &in);

		// run for a total of 10 seconds
		// at the end of each runNetwork call, SpikeMonitor stats will be printed
		smInput->startRecording();
		smExc->startRecording();
		smExc2->startRecording();
	
		sim->runNetwork(1, 0);
	
		smInput->stopRecording();
		smExc->stopRecording();
		smExc2->stopRecording();

		if (gpuId == 0) { // single gpu
			spikesSingleGPU = smExc2->getSpikeVector2D();
		} else {
			spikesMultiGPU = smExc2->getSpikeVector2D();
		}

		smExc->print(true);
		smExc2->print(true);
		smInput->print(true);

		delete sim;
	}
}
