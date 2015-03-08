//
// Copyright (c) 2015 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <carlsim.h>
#include <stdio.h>
#include <vector>

int main() {
	// ---------------- CONFIG STATE -------------------
	// create a network with nPois Poisson neurons and nExc excitatory output
	// neurons
	CARLsim sim("plasticity simulation", GPU_MODE, USER);
	int nPois = 100;
	int nExc  = 1;

	PoissonRate poissRate(nPois, true); // allocate on GPU for minimal memory copies
	int gPois = sim.createSpikeGeneratorGroup("input", nPois, EXCITATORY_POISSON);
	int gExc  = sim.createGroup("output", nExc, EXCITATORY_NEURON);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.connect(gPois, gExc,  "full", 0.01f, 0.03f, 1.0, 1, 1, SYN_PLASTIC);

	// set conductances with default values
	sim.setConductances(true);

	sim.connect(gPois,gExc,"full",RangeWeight(0.0,1.0f/100, 20.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
	sim.setSTDP(gExc, true);
	sim.setESTDP(gExc, true);

	// homeostasis constants
	float homeoScale= 1.0; // homeostatic scaling factor
	float avgTimeScale = 5.0; // homeostatic time constant
	float targetFiringRate = 35.0;

	sim.setHomeostasis(gExc,true,homeoScale,avgTimeScale);
  sim.setHomeoBaseFiringRate(gExc,targetFiringRate,0);

	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();
	SpikeMonitor* SpikeMonInput  = sim.setSpikeMonitor(gPois,"DEFAULT");
	SpikeMonitor* SpikeMonOutput = sim.setSpikeMonitor(gExc,"DEFAULT");

	// ---------------- RUN STATE -------------------
	// set rate of each neuron
	std::vector <float> rates;
	for (int i=0; i<nPois; i++)
		rates.push_back((i+1)*(20.0/100));

	poissRate.setRates(rates);
	sim.setSpikeRate(gPois, &poissRate);

	// run the established network for 1 sec
	int runTimeSec = 1000; // seconds
	int runTimeMs  = 0; // milliseconds
	SpikeMonOutput->startRecording();
	sim.runNetwork(runTimeSec, runTimeMs);
	SpikeMonOutput->stopRecording();


  return 0;
}
