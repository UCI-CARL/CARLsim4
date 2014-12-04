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

#define GPU0 0
#define GPU1 1
#define NUM_GPU 2

#define NUM_NEURONS_EXC 800
#define NUM_NEURONS_INB 200

#if (WIN32 || WIN64)
#include <windows.h>
#define _CRT_SECURE_NO_WARNINGS

DWORD WINAPI runCARLsim(LPVOID arg)
#else
void *runCARLsim(void* arg)
#endif
{
	CARLsim* sim = (CARLsim*)arg;
	// Conductances
	float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
	// STDP
	float alphaLTP = 0.1f/100, alphaLTD = 0.12f/100, tauLTP = 20.0f, tauLTD = 20.0f;
	int gNoise, gExc, gInb;
	SpikeMonitor *excMon, *inbMon;

	gExc = sim->createGroup("exc", NUM_NEURONS_EXC, EXCITATORY_NEURON);
	sim->setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f);
	
	gInb = sim->createGroup("inh", NUM_NEURONS_INB, INHIBITORY_NEURON);
	sim->setNeuronParameters(gInb, 0.1f, 0.2f, -65.0f, 2.0f);
	
	gNoise = sim->createSpikeGeneratorGroup("poisson", NUM_NEURONS_EXC, EXCITATORY_NEURON);
			
	sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);

	sim->connect(gNoise, gExc, "one-to-one", RangeWeight(12.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim->connect(gExc, gExc, "random", RangeWeight(0.0f, 0.5f/100, 4.0f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC);
	sim->connect(gExc, gInb, "random", RangeWeight(0.0f, 2.0f/100, 4.0f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC);
	sim->connect(gInb, gExc, "random", RangeWeight(2.0f/100), 0.1f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim->setSTDP(gExc, true, STANDARD, alphaLTP, tauLTP, alphaLTD, tauLTD);

	sim->setupNetwork();

	// it's unnecessary to do this in the loop
	PoissonRate in = PoissonRate(NUM_NEURONS_EXC);
	in.setRates(2.0f); // 2 Hz

	sim->setSpikeRate(gNoise, &in);
	
	excMon = sim->setSpikeMonitor(gExc);
	inbMon = sim->setSpikeMonitor(gInb);

	for (int t = 0; t < 100; t++) {
		sim->runNetwork(1,0);
	}

	return 0;
}

int main() {

	CARLsim* sims[NUM_GPU]; /* construct a CARLsim network on the heap. */
	
	sims[GPU0] = new CARLsim("multiGPU_0", GPU_MODE, USER, 0);
	sims[GPU1] = new CARLsim("multiGPU_1", GPU_MODE, USER, 1);

#if (WIN32 || WIN64)
	HANDLE threads[NUM_GPU] = {0};

	threads[GPU0] = CreateThread(NULL, 0, runCARLsim, (LPVOID)sims[GPU0], 0, NULL);
	threads[GPU1] = CreateThread(NULL, 0, runCARLsim, (LPVOID)sims[GPU1], 0, NULL);

	WaitForSingleObject(threads[0], INFINITE);
	WaitForSingleObject(threads[1], INFINITE);

	CloseHandle(threads[GPU0]);
	CloseHandle(threads[GPU1]);
#else
	pthread_t threads[NUM_GPU] = {0};

	pthread_create(&threads[GPU0], NULL, runCARLsim, (void*)sims[GPU0]);
	pthread_create(&threads[GPU1], NULL, runCARLsim, (void*)sims[GPU1]);
	pthread_join(threads[GPU0], NULL);
	pthread_join(threads[GPU1], NULL);
#endif

	delete sims[GPU0];
	delete sims[GPU1];
}
