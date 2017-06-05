/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
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
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 5/22/2015
*/

// include CARLsim user interface
#include <carlsim.h>

#define N_EXC 1

int main() {
	// create a network on GPU
	int randSeed = 42;
	CARLsim sim("test kernel", CPU_MODE, USER, 0, randSeed);

	// configure the network
	int gExc = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON, 0, CPU_CORES);

	// 4 parameter version
	//sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	// test

	// 9 parameter version
	sim.setNeuronParameters(gExc, 100.0f, 0.7f, -60.0f, -40.0f, 0.03f, -2.0f, 35.0f, -50.0f, 100.0f); //RS
																									  // set up a dummy (connection probability of 0) connection
	int gInput = sim.createSpikeGeneratorGroup("input", N_EXC, EXCITATORY_NEURON, 0, CPU_CORES);
	sim.connect(gInput, gExc, "one-to-one", RangeWeight(30.0f), 0.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setConductances(false);

	//FORWARD_EULER
	//RUNGE_KUTTA4
	sim.setIntegrationMethod(FORWARD_EULER, 10.0f);

	// build the network
	sim.setupNetwork();

	// set some monitors
	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
	//SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

	NeuronMonitor* nmExc = sim.setNeuronMonitor(gExc, "NULL");

	//setup some baseline input

	//PoissonRate in(N_EXC);
	//in.setRates(1.0f);
	//sim.setSpikeRate(gInput, &in);

	//smInput->startRecording();
	smExc->startRecording();
	nmExc->startRecording();

	for (int t = 0; t < 1; t++) {
		sim.runNetwork(0, 100, true);
		sim.setExternalCurrent(gExc, 70);
		sim.runNetwork(0, 900, true);
	}

	//smInput->stopRecording();
	smExc->stopRecording();
	nmExc->stopRecording();

	smExc->print(true);
	nmExc->print();
	//Expected Spike Times (4 param; extCurrent = 5; Euler; 2 steps / ms): 108 196 293 390 487 584 681 778 875 972
	//Expected Spike Times (9 param; extCurrent = 70): 
	//smInput->print(false);

	return 0;
}
