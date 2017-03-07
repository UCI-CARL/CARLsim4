/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
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
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#include <carlsim.h>
#include <stdio.h>

#include <periodic_spikegen.h>
#include <simple_weight_tuner.h>

#if defined(WIN32) || defined(WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	CARLsim *sim = new CARLsim("SimpleWeightTuner", CPU_MODE, USER, 0, 42);

	// output layer should have some target firing rate
	int gOut=sim->createGroup("out", 1000, EXCITATORY_NEURON);
	sim->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f);

	// hidden layer to tune first
	int gHid=sim->createGroup("hidden", 1000, EXCITATORY_NEURON);
	sim->setNeuronParameters(gHid, 0.02f, 0.2f, -65.0f, 8.0f);

	// input is a SpikeGenerator group that fires every 20 ms (50 Hz)
	PeriodicSpikeGenerator PSG(50.0f);
	int gIn=sim->createSpikeGeneratorGroup("in", 1000, EXCITATORY_NEURON);
	sim->setSpikeGenerator(gIn, &PSG);

	// random connection with 10% probability
	int c0=sim->connect(gIn, gHid, "random", RangeWeight(0.005f), 0.1f, RangeDelay(1,10));
	int c1=sim->connect(gHid, gOut, "random", RangeWeight(0.005f), 0.1f, RangeDelay(1,10));

	sim->setConductances(true);


	// ---------------- SETUP STATE -------------------

	sim->setupNetwork();

	SpikeMonitor* SpikeMonOut = sim->setSpikeMonitor(gOut, "NULL");
	SpikeMonitor* SpikeMonHidden = sim->setSpikeMonitor(gHid, "NULL");

	// accept firing rates within this range of target firing
	double targetFiringHid = 27.4;	// target firing rate for gHid
	double targetFiringOut = 42.8;	// target firing rate for gOut

	// algorithm will terminate when at least one of the termination conditions is reached
	double errorMarginHz = 0.015;	// error margin
	int maxIter = 100;				// max number of iterations

	// set up weight tuning from input -> hidden
	SimpleWeightTuner SWTin2hid(sim, errorMarginHz, maxIter);
	SWTin2hid.setConnectionToTune(c0, 0.0); // start at 0
	SWTin2hid.setTargetFiringRate(gHid, targetFiringHid);

	// set up weight tuning from hidden -> output
	SimpleWeightTuner SWThid2out(sim, errorMarginHz, maxIter);
	SWThid2out.setConnectionToTune(c1, 0.0); // start at 0
	SWThid2out.setTargetFiringRate(gOut, targetFiringOut);


	// ---------------- RUN STATE -------------------

	printf("\nSimpleWeightTuner Demo\n");
	printf("- Step 1: Tune weights from input layer to hidden layer\n");
	while (!SWTin2hid.done()) {
		SWTin2hid.iterate();
	}

	printf("\n- Step 2: Tune weights from hidden layer to output layer\n");
	while (!SWThid2out.done()) {
		SWThid2out.iterate();
	}

	printf("\n- Step 3: Verify result (gHid=%.4fHz, gOut=%.4fHz, +/- %.4fHz)\n", targetFiringHid, targetFiringOut, 
		errorMarginHz);

	SpikeMonOut->startRecording();
	SpikeMonHidden->startRecording();

	sim->runNetwork(10, 0, false);

	SpikeMonOut->stopRecording();
	SpikeMonHidden->stopRecording();

	SpikeMonOut->print(false);
	SpikeMonHidden->print(false);

	delete sim;
	return 0;
}
