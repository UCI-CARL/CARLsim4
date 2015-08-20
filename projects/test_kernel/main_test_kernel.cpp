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

int main() {
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	CARLsim sim("test kernel", GPU_MODE, USER, numGPUs, randSeed);

	// configure the network
	// set up a COBA two-layer network with gaussian connectivity
	Grid3D gridIn(10, 10, 1); // pre is on a 10x10 grid
	Grid3D gridOut(10, 10, 1); // post is on a 10x10 grid
	int gout = sim.createGroup("output", gridOut, EXCITATORY_NEURON);
	int gin = sim.createSpikeGeneratorGroup("input", gridIn, EXCITATORY_NEURON);

	sim.setNeuronParameters(gout, 0.02f, 0.2f, -65.0f, 8.0f);
	sim.connect(gin, gout, "random", RangeWeight(0.0, 2.0f/100, 20.0f/100), 0.2f, RangeDelay(1, 5), RadiusRF(-1), SYN_PLASTIC);
	sim.setConductances(true);

	sim.setESTDP(gout, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

	// build the network
	sim.setupNetwork();

	// set some monitors
	sim.setSpikeMonitor(gin, "DEFAULT");
	sim.setSpikeMonitor(gout, "DEFAULT");
	sim.setConnectionMonitor(gin, gout, "DEFAULT");

	//setup some baseline input
	PoissonRate in(gridIn.N);
	in.setRates(10.0f);
	sim.setSpikeRate(gin, &in);

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	//for (int i=0; i<10; i++)
		sim.runNetwork(10, 0);

	return 0;
}
