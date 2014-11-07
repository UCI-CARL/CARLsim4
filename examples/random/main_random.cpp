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

int main() {
	// simulation details
	int N = 1000; // number of neurons
	int ithGPU = 0; // run on first GPU

	// create a network
	CARLsim sim("random",GPU_MODE,USER,ithGPU,42);

	int g1=sim.createGroup("excit", N*0.8, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2=sim.createGroup("inhib", N*0.2, INHIBITORY_NEURON);
	sim.setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	int gin=sim.createSpikeGeneratorGroup("input",N*0.1,EXCITATORY_NEURON);

	sim.setConductances(true,5,150,6,150);

	// make random connections with 10% probability
	sim.connect(g2,g1,"random", RangeWeight(0.01), 0.1f);
	// make random connections with 10% probability, and random delays between 1 and 20
	sim.connect(g1,g2,"random", RangeWeight(0.0,0.0025,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(g1,g1,"random", RangeWeight(0.0,0.06,0.1), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	// 5% probability of connection
	sim.connect(gin, g1, "random", RangeWeight(1.0), 0.05f, RangeDelay(1,20), RadiusRF(-1));

	// here we define and set the properties of the STDP.
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.12f/100, TAU_LTD = 20.0f;
	sim.setSTDP(g1, true, STANDARD, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);


	// build the network
	sim.setupNetwork();
	sim.setSpikeMonitor(g1); // put spike times into file
	sim.setSpikeMonitor(g2); // Show basic statistics about g2
	sim.setSpikeMonitor(gin);

	sim.setConnectionMonitor(g1, g2);

	//setup some baseline input
	PoissonRate in(N*0.1);
	for (int i=0;i<N*0.1;i++) in.rates[i] = 1;
		sim.setSpikeRate(gin,&in);

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int i=0; i<10; i++)
		sim.runNetwork(1,0);

	return 0;
}

