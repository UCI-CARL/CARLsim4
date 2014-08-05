/*
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
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
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 3/22/14
 */

#include <carlsim.h>

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

int main()
{
	// simulation details
	int N = 1;
	int NUM_INPUT  = N;
	int NUM_OUTPUT = N;
	int ithGPU = 0; // run on first GPU

	// create a network
	CARLsim sim("random",GPU_MODE,USER,ithGPU,1,42);

	// create spike generator that produces periodic (deterministic) spike trains
	PeriodicSpikeGenerator* spkGenG0 = NULL;

	int g1=sim.createGroup("excit", NUM_INPUT, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int gin=sim.createSpikeGeneratorGroup("input", NUM_OUTPUT ,EXCITATORY_NEURON);
	// create our periodic spike generator
	bool spikeAtZero = true;
	spkGenG0 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 10 Hz
	sim->setSpikeGenerator(gin, spkGenG0);

	// 5% probability of connection
	sim.connect(gin,g1,"random", RangeWeight(0.5), 0.05f, RangeDelay(1), SYN_FIXED);

	// build the network
	sim.setupNetwork();

	sim.setSpikeMonitor(g1,"examples/random/results/spikes.dat"); // put spike times into spikes.dat
	sim.setSpikeMonitor(gin);

	//run for 10 seconds
	sim.runNetwork(10,0,false);

	return 0;
}

