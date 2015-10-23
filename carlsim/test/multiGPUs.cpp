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

//int main() {
//	// create a network on GPU
//	int numGPUs = 2;
//	int randSeed = 42;
//	CARLsim sim("test kernel", GPU_MODE, USER, numGPUs, randSeed);
//
//	// configure the network
//	int gExc = sim.createGroup("exc", 10, EXCITATORY_NEURON);
//	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS
//
//	//int gInh = sim.createGroup("inh", 20, INHIBITORY_NEURON);
//	//sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS
//	int gExc2 = sim.createGroup("exc", 10, EXCITATORY_NEURON);
//	sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS
//
//	int gInput = sim.createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);
//
//	sim.connect(gInput, gExc, "one-to-one", RangeWeight(50.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
//	sim.connect(gExc, gExc2, "random", RangeWeight(10.0f), 0.4f, RangeDelay(1, 10), RadiusRF(-1), SYN_FIXED);
//	sim.connect(gExc2, gExc, "random", RangeWeight(0.0001f), 0.4f, RangeDelay(1, 10), RadiusRF(-1), SYN_FIXED);
//
//	sim.setConductances(false);
//
//	//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));
//
//	// build the network
//	sim.setupNetwork();
//
//	// set some monitors
//	SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "DEFAULT");
//	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "DEFAULT");
//	SpikeMonitor* smExc2 = sim.setSpikeMonitor(gExc2, "DEFAULT");
//	ConnectionMonitor* cmEE = sim.setConnectionMonitor(gExc, gExc2, "DEFAULT");
//
//	//setup some baseline input
//	PoissonRate in(10);
//	in.setRates(5.0f);
//	sim.setSpikeRate(gInput, &in);
//
//	// run for a total of 10 seconds
//	// at the end of each runNetwork call, SpikeMonitor stats will be printed
//	smInput->startRecording();
//	smExc->startRecording();
//	smExc2->startRecording();
//	
//	sim.runNetwork(0, 100);
//	
//	smInput->stopRecording();
//	smExc->stopRecording();
//	smExc2->stopRecording();
//
//	smExc->print(true);
//	smExc2->print(true);
//	smInput->print(true);
//
//	return 0;
//}
