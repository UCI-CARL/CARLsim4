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

#define NUM_NEURON 50

int main()
{
	// simulation details
	std::string saveFolder = "results/";
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights = NULL;
	int size;
	SpikeMonitor* spikeMonIn1;
	SpikeMonitor* spikeMonIn2;
	SpikeMonitor* spikeMonEx;
	int gEx, gIn1, gIn2, gInput1, gInput2, gInput3, gInput4;
	float BETA_LTP = 0.10f/100;
	float BETA_LTD = 0.12f/100;
	float LAMDA = 6.0f;
	float DELTA = 20.0f;
	float ALPHA_LTP = 0.10f/100;
	float ALPHA_LTD = 0.12f/100;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	//FILE* fid = fopen("results/weight.csv", "w");

	// create a network
	CARLsim sim("istdp",GPU_MODE, USER,0,1,42);

	gEx = sim.createGroup("excit", 50, EXCITATORY_NEURON);
	sim.setNeuronParameters(gEx, 0.02f, 0.2f, -65.0f, 8.0f);

	gIn1 = sim.createGroup("inhib", 50, INHIBITORY_NEURON);
	sim.setNeuronParameters(gIn1, 0.1f,  0.2f, -65.0f, 2.0f);

	gIn2 = sim.createGroup("inhib", 50, INHIBITORY_NEURON);
	sim.setNeuronParameters(gIn2, 0.1f,  0.2f, -65.0f, 2.0f);

	gInput1=sim.createSpikeGeneratorGroup("input_1", 50, EXCITATORY_NEURON);
	gInput2=sim.createSpikeGeneratorGroup("input_2", 50, EXCITATORY_NEURON);
	gInput3=sim.createSpikeGeneratorGroup("input_3", 50, EXCITATORY_NEURON);
	gInput4=sim.createSpikeGeneratorGroup("input_4", 50, EXCITATORY_NEURON);

	sim.connect(gInput1, gEx, "full", RangeWeight(0.0, 1.0f/100, 2.0f/100), 1.0f, RangeDelay(1, 5), SYN_PLASTIC);
	sim.connect(gInput1, gIn1, "full", RangeWeight(0.0, 0.9f/100, 1.2f/100), 1.0f, RangeDelay(15, 20), SYN_PLASTIC);
	sim.connect(gInput1, gIn2, "full", RangeWeight(0.0, 0.9f/100, 1.2f/100), 1.0f, RangeDelay(15, 20), SYN_PLASTIC);

	sim.connect(gInput2, gEx, "full", RangeWeight(0.0, 1.0f/100, 2.0f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);
	sim.connect(gInput3, gIn1, "full", RangeWeight(0.0, 0.9f/100, 1.2f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);
	sim.connect(gInput4, gIn2, "full", RangeWeight(0.0, 0.9f/100, 1.2f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);

	sim.connect(gIn1, gEx, "full", RangeWeight(0.0, 0.5f/100, 1.0f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);
	sim.connect(gIn2, gEx, "full", RangeWeight(0.0, 0.5f/100, 1.0f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);

	// enable COBA, set up STDP, enable dopamine-modulated STDP
	sim.setConductances(true,5,150,6,150);
	sim.setESTDP(gEx, true, STANDARD, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	sim.setISTDP(gEx, true, STANDARD, BETA_LTP, BETA_LTD, LAMDA, DELTA);

	sim.setESTDP(gIn1, true, STANDARD, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	sim.setESTDP(gIn2, true, STANDARD, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// build the network
	sim.setupNetwork();

	spikeMonEx = sim.setSpikeMonitor(gEx);
	spikeMonIn1 = sim.setSpikeMonitor(gIn1);
	spikeMonIn2 = sim.setSpikeMonitor(gIn2);

	//setup some baseline input
	PoissonRate in1(NUM_NEURON);
	for (int i = 0; i < NUM_NEURON; i++) in1.rates[i] = 2;
		sim.setSpikeRate(gInput1, &in1);

	PoissonRate in2(NUM_NEURON);
	for (int i = 0; i < NUM_NEURON; i++) in2.rates[i] = 2;
		sim.setSpikeRate(gInput2, &in2);

	PoissonRate in3(NUM_NEURON);
	for (int i = 0; i < NUM_NEURON; i++) in3.rates[i] = 2;
		sim.setSpikeRate(gInput3, &in3);

	PoissonRate in4(NUM_NEURON);
	for (int i = 0; i < NUM_NEURON; i++) in4.rates[i] = 2;
		sim.setSpikeRate(gInput4, &in4);


	// run for 1000 seconds
	for (int t = 0; t < 1000; t++) {
		spikeMonIn1->startRecording();
		spikeMonIn2->startRecording();
		spikeMonEx->startRecording();
		sim.runNetwork(1,0,true, true);
		spikeMonIn1->stopRecording();
		spikeMonIn2->stopRecording();
		spikeMonEx->stopRecording();
		
		//spikeMonIn->print();
		//spikeMonEx->print();
		//spikeMon1->print();

		//sim.getPopWeights(gin, g1, weights, size);
		//printf("%f\n",weights[0]);
	}

	//fclose(fid);

	return 0;
}

