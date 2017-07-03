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

// include CARLsim user interface
#include <carlsim.h>
#include <stdio.h>
// include stopwatch for timing
#include <stopwatch.h>

int main() {
	// keep track of execution time
	Stopwatch watch;
	

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 123;
	CARLsim sim("Simple LIF neuron tuning", GPU_MODE, USER, numGPUs, randSeed);

	// configure the network
	// set up a single LIF neuron network to record its fi curve
	Grid3D gridSingle(1,1,1); // pre is on a 1x1 grid
	Grid3D gridDummy(1,1,1); // dummy is on a 1x1 grid
	int gSingleLIF=sim.createGroupLIF("input", gridSingle, EXCITATORY_NEURON, 0, GPU_CORES);
	int gDummyLIF=sim.createGroup("output", gridDummy, EXCITATORY_NEURON, 1, CPU_CORES);
	sim.setNeuronParametersLIF(gSingleLIF, 20, 2, 1.0f, 0.0f, RangeRmem(1.5f));
	//sim.setNeuronParametersLIF(gSingleLIF, 10, 2, -50.0f, -65.0f, RangeRmem(5.0f));
	//sim.setNeuronParametersLIF(gDummyLIF, 20);
	//sim.setNeuronParameters(gSingleLIF, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.setNeuronParameters(gDummyLIF, 0.02f, 0.2f, -65.0f, 8.0f);
	printf("LIF group ID: %d", gSingleLIF);
	printf("Dummy group ID: %d", gDummyLIF);
	fflush(stdout);
	sim.connect(gSingleLIF, gDummyLIF, "full", RangeWeight(0.05), 1.0f, RangeDelay(1));
	sim.setConductances(false);
	sim.setIntegrationMethod(FORWARD_EULER, 1);

//	NeuronMonitor* nmIn = sim.setNeuronMonitor(gSingleLIF,"DEFAULT");

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// set some monitors
	sim.setSpikeMonitor(gSingleLIF,"DEFAULT");
	//sim.setSpikeMonitor(gDummyLIF,"DEFAULT");

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
//	nmIn->startRecording();
	for (int i=0; i<=10; i++) {
		std::vector<float> current(1, (float)i*0.8f);
		sim.setExternalCurrent(gSingleLIF, current);
		sim.runNetwork(10,0);
	}
//	nmIn->stopRecording();
	// print stopwatch summary
	watch.stop();
//	nmIn->print();
	
	return 0;
}
