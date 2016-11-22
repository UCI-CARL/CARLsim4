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

#define N_EXC 800
#define N_INH 200

int main() {
	// create a network on GPU
	int numGPUs = 2;
	int randSeed = 42;
	float pConn = 100.0f / (N_EXC + N_INH); // connection probability
	CARLsim sim("test kernel 2", CPU_MODE, USER, numGPUs, randSeed);

	// configure the network
	int gExc = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON, 0);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	//int gExc2 = sim.createGroup("exc2", N_EXC, EXCITATORY_NEURON, 1);
	//sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f);

	int gInh = sim.createGroup("inh", N_INH, INHIBITORY_NEURON, 1);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	//int gInh2 = sim.createGroup("inh2", N_INH, INHIBITORY_NEURON, 0);
	//sim.setNeuronParameters(gInh2, 0.1f, 0.2f, -65.0f, 2.0f);
	//int gExc2 = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON);
	//sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gInput = sim.createSpikeGeneratorGroup("input", N_EXC, EXCITATORY_NEURON, 0);

	//int gInput2 = sim.createSpikeGeneratorGroup("input2", N_EXC, EXCITATORY_NEURON, 1);

	sim.connect(gInput, gExc, "one-to-one", RangeWeight(30.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc, gExc, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gExc, "random", RangeWeight(0.0f, 6.0f, 10.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc, gInh, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gInh, gExc, "random", RangeWeight(5.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInh, gExc, "random", RangeWeight(0.0f, 5.0f, 10.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

	//sim.connect(gInput2, gExc2, "one-to-one", RangeWeight(30.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc2, gExc2, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc2, gInh2, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gInh2, gExc2, "random", RangeWeight(5.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// enable STDP on all incoming synapses to gExc
	float alphaPlus = 0.1f, tauPlus = 20.0f, alphaMinus = 0.1f, tauMinus = 20.0f;
	sim.setESTDP(gExc, true, STANDARD, ExpCurve(alphaPlus, tauPlus, -alphaMinus, tauMinus));
	sim.setISTDP(gExc, true, STANDARD, ExpCurve(-alphaPlus, tauPlus, alphaMinus, tauMinus));

	sim.setConductances(false);

	//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

	// build the network
	sim.setupNetwork();

	// set some monitors
	//SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "DEFAULT");
	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
	SpikeMonitor* smInh = sim.setSpikeMonitor(gInh, "NULL");
	SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

	//SpikeMonitor* smExc2 = sim.setSpikeMonitor(gExc2, "NULL");
	//SpikeMonitor* smInh2 = sim.setSpikeMonitor(gInh2, "NULL");
	//SpikeMonitor* smInput2 = sim.setSpikeMonitor(gInput2, "NULL");
	ConnectionMonitor* cmEE = sim.setConnectionMonitor(gExc, gExc, "DEFAULT");
	cmEE->setUpdateTimeIntervalSec(-1);

	//setup some baseline input
	PoissonRate in(N_EXC);
	in.setRates(1.0f);
	sim.setSpikeRate(gInput, &in);

	//PoissonRate in2(N_EXC);
	//in2.setRates(1.0f);
	//sim.setSpikeRate(gInput2, &in2);

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int t = 0; t < 10; t++) {
		smInput->startRecording();
		smExc->startRecording();
		smInh->startRecording();

		for (int ts = 0; ts < 1000; ts++)
			sim.runNetwork(0, 1, false);

		smInput->stopRecording();
		smExc->stopRecording();
		smInh->stopRecording();
		cmEE->takeSnapshot();

		smExc->print(false);
		smInh->print(false);
		smInput->print(false);
		cmEE->printSparse(false);
	}

	return 0;
}
