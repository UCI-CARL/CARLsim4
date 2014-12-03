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
 * Ver 12/1/2014
 */

#include <carlsim.h>
#include <simple_weight_tuner.h>

#include <stdio.h>	// printf

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

int main() {
	// ------------ CONFIG STATE --------------------------

	int nNeur = 1000; // number of neurons per group
	float initWt = 0.1f; // initial weight for connection

	CARLsim *sim = new CARLsim("SimpleWeightTuner", CPU_MODE, USER, 0, 42);

	int gOut=sim->createGroup("out", nNeur, EXCITATORY_NEURON);
	sim->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f);
	int gIn=sim->createSpikeGeneratorGroup("in", nNeur, EXCITATORY_NEURON);

	// random connection with 10% probability
	int c0=sim->connect(gIn, gOut, "random", RangeWeight(initWt), 0.1f, RangeDelay(1,10));

	sim->setConductances(true);


	// ------------ SETUP STATE --------------------------

	sim->setupNetwork();

	// make input 50Hz
	// apply to input group
	PoissonRate PR(nNeur);
	PR.setRates(50.0f);
	sim->setSpikeRate(gIn, &PR);


	// ------------ EXE STATE --------------------------

	// use weight tuner to find the weights that give 27.4 Hz spiking
	SimpleWeightTuner SWT(sim, 0.01, 100);
	SWT.setConnectionToTune(c0, 0.0);
	SWT.setTargetFiringRate(gOut, 27.4);

	while (!SWT.done()) {
		SWT.iterate();
	}

	// \TODO: Use ConnectionMonitor to retrieve the current set of weights

	// verify result
	for (int i=0; i<5; i++)
		sim->runNetwork(1,0);

	return 0;
}

