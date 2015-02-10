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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 3/22/14
 */

#include <carlsim.h>

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

#define NUM_EXC_NEURON 800

int main()
{
	// simulation details
	std::string saveFolder = "results/";
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights = NULL;
	int size;
	float sum;
	SpikeMonitor* spikeMon1;
	SpikeMonitor* spikeMon2;
	SpikeMonitor* spikeMon3;
	int gExc, gInb, gInput;
	float BETA_LTP = 0.16f/100;
	float BETA_LTD = 0.10f/100;
	float LAMBDA = 6.0f;
	float DELTA = 20.0f;
	float ALPHA_LTP_EXC = 0.10f/100;
	float ALPHA_LTD_EXC = 0.12f/100;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;

	//FILE* fid = fopen("results/weight.csv", "w");

	// create a network
	CARLsim sim("istdp",GPU_MODE, USER,0,1,42);

	gExc = sim.createGroup("excit", NUM_EXC_NEURON, EXCITATORY_NEURON);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f);

	gInb = sim.createGroup("inhib", 200, INHIBITORY_NEURON);
	sim.setNeuronParameters(gInb, 0.1f,  0.2f, -65.0f, 2.0f);

	gInput = sim.createSpikeGeneratorGroup("input", NUM_EXC_NEURON, EXCITATORY_NEURON);

	sim.connect(gInput, gExc, "one-to-one", RangeWeight(10.0f/100), 1.0f, RangeDelay(1, 20), SYN_FIXED);
	sim.connect(gExc, gExc, "random", RangeWeight(0.0, 1.0f/100, 4.0f/100), 0.12f, RangeDelay(1, 20), SYN_PLASTIC);
	sim.connect(gExc, gInb, "random", RangeWeight(0.0, 1.0f/100, 4.0f/100), 0.12f, RangeDelay(1, 20), SYN_PLASTIC);
	//sim.connect(gInb, gExc, "random", RangeWeight(0.0, 1.0f/100, 4.0f/100), 0.1f, RangeDelay(1), SYN_PLASTIC);
	//sim.connect(gInb, gInb, "random", RangeWeight(0.0, 1.0f/100, 4.0f/100), 0.1f, RangeDelay(1), SYN_PLASTIC);
	sim.connect(gInb, gExc, "random", RangeWeight(1.0f/100), 0.1f, RangeDelay(1), SYN_FIXED);
	sim.connect(gInb, gInb, "random", RangeWeight(1.0f/100), 0.1f, RangeDelay(1), SYN_FIXED);

	// enable COBA, set up STDP, enable dopamine-modulated STDP
	sim.setConductances(true, 5, 150, 6, 150);
	sim.setESTDP(gExc, true, STANDARD, ALPHA_LTP_EXC, TAU_LTP, ALPHA_LTD_EXC, TAU_LTD);
	//sim.setISTDP(gExc, true, STANDARD, BETA_LTP, BETA_LTD, LAMBDA, DELTA);

	sim.setESTDP(gInb, true, STANDARD, ALPHA_LTP_EXC, TAU_LTP, ALPHA_LTD_EXC, TAU_LTD);
	//sim.setISTDP(gInb, true, STANDARD, BETA_LTP, BETA_LTD, LAMBDA, DELTA);

	// build the network
	sim.setupNetwork();

	spikeMon1 = sim.setSpikeMonitor(gExc);
	spikeMon2 = sim.setSpikeMonitor(gInb);
	spikeMon3 = sim.setSpikeMonitor(gInput);

	//setup some baseline input
	PoissonRate in(NUM_EXC_NEURON);
	for (int i = 0; i < NUM_EXC_NEURON; i++) in.rates[i] = 2;
		sim.setSpikeRate(gInput, &in);


	// run for 1000 seconds
	for (int t = 0; t < 1000; t++) {
		spikeMon1->startRecording();
		spikeMon2->startRecording();
		spikeMon3->startRecording();
		sim.runNetwork(10,0,true, true);
		spikeMon1->stopRecording();
		spikeMon2->stopRecording();
		spikeMon3->stopRecording();

		sim.getPopWeights(gInput, gExc, weights, size);
		sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += weights[i];
		printf("input-exc:%f\n", sum / size);

		sim.getPopWeights(gExc, gExc, weights, size);
		sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += weights[i];
		printf("exc-exc:%f\n", sum / size);

		sim.getPopWeights(gExc, gInb, weights, size);
		sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += weights[i];
		printf("exc-inb:%f\n", sum / size);

		sim.getPopWeights(gInb, gExc, weights, size);
		sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += weights[i];
		printf("inb-exc:%f\n", sum / size);

		sim.getPopWeights(gInb, gInb, weights, size);
		sum = 0.0f;
		for (int i = 0; i < size; i++)
			sum += weights[i];
		printf("inb-inb:%f\n", sum / size);
	}

	return 0;
}

