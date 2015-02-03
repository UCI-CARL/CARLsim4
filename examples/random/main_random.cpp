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

#include <vector>
#include <stdio.h>

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

int main() {
	// simulation details
	int N = 100; // number of neurons
	int ithGPU = 0; // run on first GPU

	// create a network
	CARLsim sim("random", GPU_MODE, USER, ithGPU, 42);

	int g1=sim.createGroup("excit", Grid3D(3,3,3), EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2=sim.createGroup("inhib", N*0.2, INHIBITORY_NEURON);
	sim.setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	int gin=sim.createSpikeGeneratorGroup("input",Grid3D(5,5,5),EXCITATORY_NEURON);

	sim.setConductances(true,5,150,6,150);


	// 5% probability of connection
	sim.connect(gin, g1, "full", RangeWeight(0.25), 1.0f, RangeDelay(1,20), RadiusRF(3,3,3));

	// build the network
	sim.setupNetwork();

	// record spike times, save to binary
	sim.setSpikeMonitor(g1, "Default");
	sim.setSpikeMonitor(g2, "Default");
	sim.setSpikeMonitor(gin, "Default");

	// record weights of g1->g1 connection, save to binary
	sim.setConnectionMonitor(gin,g1,"Default");

	//setup some baseline input
	PoissonRate in(N*0.1);
	in.setRates(1.0f);
//	sim.setSpikeRate(gin,&in);

	// run for a total of 10 seconds
	// at the end of each runNetwork call, Spike and Connection Monitor stats will be printed
	bool printRunStats = true;
	for (int i=0; i<10; i++) {
		sim.runNetwork(1,0,printRunStats);
	}

	return 0;
}

