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
#include <stdio.h>

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

#define numGPU 2

void *RunNetwork(void* arg) {
	CARLsim* sim = (CARLsim*)arg;

	for (int t = 0; t < 100; t++)
		sim->runNetwork(1, 0);
}

int main() {

	CARLsim** network;
	pthread_t* threads;

	network = new CARLsim*[2];
	threads = new pthread_t[2];

	/** construct a CARLsim network on the heap. */
	network[0] = new CARLsim("tunePolyGroupECJ", GPU_MODE, USER, 0);
	network[1] = new CARLsim("tunePolyGroupECJ", GPU_MODE, USER, 1);

	const float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;

	// Neurons
	const int NUM_NEURONS_EXC = 800;
	const int NUM_NEURONS_INB = 200;
	const float alphaLTP = 0.1f/100;
	const float alphaLTD = 0.12f/100;
	const float tauLTP = 20.0f;
	const float tauLTD = 20.0f;
	const float betaLTP = 0.10f/100;
	const float betaLTD = 0.06f/100;
	const float lamda = 6.0f;
	const float delta = 20.0f;

	// Simulation time (each must be at least 1s due to bug in SpikeMonitor)
	const int runTime = 2;

	// Target rates for the objective function
	const float INPUT_HZ = 1.0f;
	const float EXC_TARGET_HZ   = 1.0f;
	//const float INH_TARGET_HZ   = 2.0f;

	int indiNum = 4;

	int poissonGroup[indiNum];
	int excGroup[indiNum];
	int inhGroup[indiNum];
	SpikeMonitor* excMonitor[indiNum];
	//SpikeMonitor* inhMonitor[indiNum];
	float excHz[indiNum];
	//float inhHz[indiNum];
	float excError[indiNum];
	//float inhError[indiNum];
	float fitness[indiNum];
	int startIdx[2];
	int endIdx[2];

	FILE* fd = fopen("debug2.log", "a");
	fprintf(fd, "indiNum:%d\n", indiNum);

	startIdx[0] = 0;
	endIdx[1] = indiNum;
	startIdx[1] = indiNum / numGPU;
	endIdx[0] = indiNum / numGPU;
	// NumInstances -> number of individuals
	for (int gpuId = 0; gpuId < numGPU; gpuId++) {
		for(unsigned int i = startIdx[gpuId]; i < endIdx[gpuId]; i++) {
			fprintf(fd, "gpuId %d, i %d\n", gpuId, i);
			/** Decode a genome*/
			poissonGroup[i] = network[gpuId]->createSpikeGeneratorGroup("poisson", NUM_NEURONS_EXC, EXCITATORY_NEURON);
			excGroup[i] = network[gpuId]->createGroup("exc", NUM_NEURONS_EXC, EXCITATORY_NEURON);
			inhGroup[i] = network[gpuId]->createGroup("inh", NUM_NEURONS_INB, INHIBITORY_NEURON);
			network[gpuId]->setNeuronParameters(excGroup[i], 0.02f, 0.2f, -65.0f, 8.0f);
			network[gpuId]->setNeuronParameters(inhGroup[i], 0.1f, 0.2f, -65.0f, 2.0f);
			network[gpuId]->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);

			network[gpuId]->connect(poissonGroup[i], excGroup[i], "one-to-one", RangeWeight(12.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
			network[gpuId]->connect(excGroup[i], excGroup[i], "random", RangeWeight(4.0f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC);
			network[gpuId]->connect(excGroup[i], inhGroup[i], "random", RangeWeight(4.0f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC);
			network[gpuId]->connect(inhGroup[i], excGroup[i], "random", RangeWeight(4.0f/100), 0.1f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

			network[gpuId]->setSTDP(excGroup[i], true, STANDARD, alphaLTP, tauLTP, alphaLTD, tauLTD);
		}
	}

	// can't call setupNetwork() multiple times in the loop
	network[0]->setupNetwork();
	network[1]->setupNetwork();

	// it's unnecessary to do this in the loop
	PoissonRate* const in = new PoissonRate(NUM_NEURONS_EXC);
	for (int k=0;k<NUM_NEURONS_EXC;k++)
		in->rates[k] = INPUT_HZ;

	for (int gpuId = 0; gpuId < numGPU; gpuId++) {
		for(unsigned int i = startIdx[gpuId]; i < endIdx[gpuId]; i++) {
			fprintf(fd, "%d %d\n", gpuId, i);
			network[gpuId]->setSpikeRate(poissonGroup[i],in);

			excMonitor[i] = network[gpuId]->setSpikeMonitor(excGroup[i], "/dev/null");
			//inhMonitor[i] = network->setSpikeMonitor(inhGroup[i], "/dev/null");

			// initialize all the error and fitness variables
			excHz[i]=0; //inhHz[i]=0;
			excError[i]=0; //inhError[i]=0;
			fitness[i]=0;
		}
	}

	//fprintf(fd, "0\n");
	//network[0]->runNetwork(runTime,0);
	//fprintf(fd, "1\n");
	//network[1]->runNetwork(runTime,0);
	//for (int t = 0; t < 100; t++) {
	//	network[0]->runNetwork(1,0);
	//	network[1]->runNetwork(1, 0);
	//}

	pthread_create(&threads[0], NULL, RunNetwork, (void*)network[0]);
	pthread_create(&threads[1], NULL, RunNetwork, (void*)network[1]);
	pthread_join(threads[0], NULL);
	pthread_join(threads[1], NULL);

	fprintf(fd, "a\n");

	// Evaluate last 5 second
	for(unsigned int i = 0; i < indiNum; i++) {
		excMonitor[i]->startRecording();
		//inhMonitor[i]->startRecording();
	}

	fprintf(fd, "b\n");

	network[0]->runNetwork(5,0);
	network[1]->runNetwork(5,0);
	//pthread_create(&threads[0], NULL, TunePolyGroupECJExperiment::EvalNetwork, (void*)network[0]);
	//pthread_create(&threads[1], NULL, TunePolyGroupECJExperiment::RunNetwork, NULL);
	//pthread_join(threads[0], NULL);

	fprintf(fd, "c\n");


	delete in;
	fclose(fd);
	delete network[0];
	delete network[1];
	delete [] threads;
	delete [] network;
}

