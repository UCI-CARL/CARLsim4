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
	CARLsim sim("test kernel", GPU_MODE, USER, 0, randSeed);

	// configure the network
	int gExc = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// 4 parameter version
	//sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	// 9 parameter version
	sim.setNeuronParameters(gExc, 100.0f, 0.7f, -60.0f, -40.0f, 0.03f, -2.0f, 35.0f, -50.0f, 100.0f); //RS
																									  // set up a dummy (connection probability of 0) connection
	int gInput = sim.createSpikeGeneratorGroup("input", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.connect(gInput, gExc, "one-to-one", RangeWeight(30.0f), 0.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// COMPARTMENT TESTING
	// Groups for manual compartment testing
	int s = sim.createGroup("soma", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d1 = sim.createGroup("d1", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d2 = sim.createGroup("d2", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d3 = sim.createGroup("d3", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d4 = sim.createGroup("d4", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d5 = sim.createGroup("d5", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d6 = sim.createGroup("d6", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// some regular neuron groups
	int reg0 = sim.createGroup("reg0", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int reg1 = sim.createGroup("reg1", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// make them 9-param Izzy neurons
	sim.setNeuronParameters(s, 550.0f, 2.0, -59.0, -50.0, 0.0, -0.0, 24.0, -53.0, 109.0f);
	sim.setNeuronParameters(d1, 367.0f, 1.0, -59.0, -44.0, 0.0, 3.0, 20.0, -46.0, 24.0f);
	sim.setNeuronParameters(d2, 425.0f, 2.0, -59.0, -25.0, 0.0, 0.0, 13.0, -38.0, 69.0f);
	sim.setNeuronParameters(d3, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d4, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d5, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d6, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(reg0, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(reg1, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);

	// enable compartments
	sim.setCompartmentParameters(s, 1.0f, 1.0f);
	sim.setCompartmentParameters(d1, 1.0f, 1.0f);
	sim.setCompartmentParameters(d2, 1.0f, 1.0f);
	sim.setCompartmentParameters(d3, 1.0f, 1.0f);
	sim.setCompartmentParameters(d4, 1.0f, 1.0f);
	sim.setCompartmentParameters(d5, 1.0f, 1.0f);
	sim.setCompartmentParameters(d6, 1.0f, 1.0f);

	int gen = sim.createSpikeGeneratorGroup("SpikeGen", N_EXC, EXCITATORY_NEURON);

	// Compartment interface testing

	// grpIDs must be valid, cannot be identical
	//sim.connectCompartments(d3, d3);
	//sim.connectCompartments(s, -1);

	// no spike generators in connect call
	//sim.connectCompartments(gen, s);

	// groups must be of same size
	//sim.connectCompartments(s, d6);

	// connectCompartments is bidirectional: connecting same groups twice is illegal
	sim.connectCompartments(s, d1);

	//sim.connectCompartments(s, d1);
	//sim.connectCompartments(d1, s);

	// can't have both synaptic and compartmental connections on the same groups
	//sim.connect(s, d1, "full", RangeWeight(1.0f), 1.0f);
	//sim.connect(d1, s, "full", RangeWeight(1.0f), 1.0f);

	// can't be involved in more than 4 connections (d1-d4), d5 must break
	sim.connectCompartments(d2, s);
	sim.connectCompartments(d3, s);
	sim.connectCompartments(s, d4);
	//sim.connectCompartments(d5, s);

	// COMPARTMENT TESTING OVER

	sim.setConductances(true);

	//FORWARD_EULER
	//RUNGE_KUTTA4
	sim.setIntegrationMethod(RUNGE_KUTTA4, 10.0f);

	// build the network
	sim.setupNetwork();

	// set some monitors
	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
	//SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

	//setup some baseline input

	//PoissonRate in(N_EXC);
	//in.setRates(1.0f);
	//sim.setSpikeRate(gInput, &in);

	//smInput->startRecording();
	smExc->startRecording();

	for (int t = 0; t < 1; t++) {
		sim.runNetwork(0, 100, true);
		sim.setExternalCurrent(gExc, 70);
		sim.runNetwork(0, 900, true);
	}

	//smInput->stopRecording();
	smExc->stopRecording();

	smExc->print(true);
	//Expected Spike Times (4 param; extCurrent = 5; Euler; 2 steps / ms): 108 196 293 390 487 584 681 778 875 972
	//Expected Spike Times (9 param; extCurrent = 70): 
	//smInput->print(false);

	return 0;
}
