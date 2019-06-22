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

// include CARLsim user interface
#include <carlsim.h>
#include <stdio.h>

// include stopwatch for timing
#include <stopwatch.h>
#include <periodic_spikegen.h>

#ifndef __NO_CUDA__
#define TESTED_MODES 2
#else
#define TESTED_MODES 1
#endif

int main() {
	// keep track of execution time
	// Stopwatch watch;
	

	// // ---------------- CONFIG STATE -------------------
	
	// // create a network on GPU
	// int numGPUs = 1;
	// int randSeed = 42;
	// CARLsim sim("hello world", CPU_MODE, USER, numGPUs, randSeed);

	// // configure the network
	// // set up a COBA two-layer network with gaussian connectivity
	// Grid3D gridIn(13,9,1); // pre is on a 13x9 grid
	// Grid3D gridOut(3,3,1); // post is on a 3x3 grid
	// int gin=sim.createSpikeGeneratorGroup("input", gridIn, EXCITATORY_NEURON);
	// int gout=sim.createGroup("output", gridOut, EXCITATORY_NEURON);
	// sim.setNeuronParameters(gout, 0.02f, 0.2f, -65.0f, 8.0f);
	// sim.connect(gin, gout, "gaussian", RangeWeight(0.05), 1.0f, RangeDelay(1), RadiusRF(3,3,1));
	// sim.setConductances(true);
	// // sim.setIntegrationMethod(FORWARD_EULER, 2);

	// // ---------------- SETUP STATE -------------------
	// // build the network
	// watch.lap("setupNetwork");
	// sim.setupNetwork();

	// // set some monitors
	// sim.setSpikeMonitor(gin,"DEFAULT");
	// sim.setSpikeMonitor(gout,"DEFAULT");
	// sim.setConnectionMonitor(gin,gout,"DEFAULT");

	// //setup some baseline input
	// PoissonRate in(gridIn.N);
	// in.setRates(30.0f);
	// sim.setSpikeRate(gin,&in);


	// // ---------------- RUN STATE -------------------
	// watch.lap("runNetwork");

	// // run for a total of 10 seconds
	// // at the end of each runNetwork call, SpikeMonitor stats will be printed
	// for (int i=0; i<10; i++) {
	// 	sim.runNetwork(1,0);
	// }

	// // print stopwatch summary
	// watch.stop();
	
	// return 0;
	int randSeed = 1000;	// randSeed must not interfere with STP

	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int isRunLong=0; isRunLong<=1; isRunLong++) {
	//int isRunLong = 1;
		for (int hasCOBA=0; hasCOBA<=1; hasCOBA++) {
		//int hasCOBA = 1;
			for (int mode = 0; mode < TESTED_MODES; mode++) {
			//int isGPUmode = 1;
				// compare
				float rateG2noSTP = -1.0f;
				float rateG3noSTP = -1.0f;

				for (int hasSTP=0; hasSTP<=1; hasSTP++) {
				//int hasSTP = 1;
					sim = new CARLsim("STP.firingRateSTDvsSTF",mode?GPU_MODE:CPU_MODE,USER,1,randSeed);
					int g2=sim->createGroup("STD", 1, EXCITATORY_NEURON);
					int g3=sim->createGroup("STF", 1, EXCITATORY_NEURON);
					sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
					sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
					int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
					int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);

					float wt = hasCOBA ? 0.2f : 18.0f;
					sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
					sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

					if (hasCOBA)
						sim->setConductances(true, 5, 0, 150, 6, 0, 150);
					else
						sim->setConductances(false);

					if (hasSTP) {
						sim->setSTP(g0, g2, true, STPu(0.45f), STPtauU(50.0f), STPtauX(750.0f)); // depressive
						sim->setSTP(g1, g3, true, STPu(0.15f), STPtauU(750.0f), STPtauX(50.0f)); // facilitative
					}

					bool spikeAtZero = true;
					spkGenG0 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g0, spkGenG0);
					spkGenG1 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g1, spkGenG1);

					sim->setupNetwork();

					sim->setSpikeMonitor(g0,"NULL");
					sim->setSpikeMonitor(g1,"NULL");
					spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
					spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

					spkMonG2->startRecording();
					spkMonG3->startRecording();
					int runTimeMs = isRunLong ? 2000 : 100;
					sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
					spkMonG2->stopRecording();
					spkMonG3->stopRecording();

					if (!hasSTP) {
						// if STP is off: record spike rate, so that afterwards we can compare it to the one with STP
						// enabled
						//spkMonG2->print(true);
						//spkMonG3->print(true);
						rateG2noSTP = spkMonG2->getPopMeanFiringRate();
						rateG3noSTP = spkMonG3->getPopMeanFiringRate();
					} else {
						//spkMonG2->print(true);
						//spkMonG3->print(true);
						//fprintf(stderr,"%s %s %s, G2 w/o=%f, G2 w/=%f\n", isRunLong?"long":"short",
						//	isGPUmode?"GPU":"CPU",
						//	hasCOBA?"COBA":"CUBA",
						//	rateG2noSTP, spkMonG2->getPopMeanFiringRate());
						//fprintf(stderr,"%s %s %s, G3 w/o=%f, G3 w/=%f\n", isRunLong?"long":"short",
						//	isGPUmode?"GPU":"CPU",
						//	hasCOBA?"COBA":"CUBA",
						//	rateG3noSTP,
						//	spkMonG3->getPopMeanFiringRate());


						// if STP is on: compare spike rate to the one recorded without STP
						if (isRunLong) {
							// the run time was relatively long, so STP should have its expected effect
							printf("Long run -- spkMonG2->getPopMeanFiringRate(), rateG2noSTP: %f -- %f\n", spkMonG2->getPopMeanFiringRate(), rateG2noSTP);
							printf("Long run -- spkMonG3->getPopMeanFiringRate(), rateG3noSTP: %f -- %f\n", spkMonG3->getPopMeanFiringRate(), rateG3noSTP);
						 // facilitative
						} else {
							// the run time was really short, so STP should have no effect (because we scale STP_A so
							// that STP has no weakening/strengthening effect on the first spike)
							printf("Short run -- spkMonG2->getPopMeanFiringRate(), rateG2noSTP: %f -- %f\n", spkMonG2->getPopMeanFiringRate(), rateG2noSTP);
							printf("Short run -- spkMonG3->getPopMeanFiringRate(), rateG3noSTP: %f -- %f\n", spkMonG3->getPopMeanFiringRate(), rateG3noSTP);
						}
					}

					delete spkGenG0, spkGenG1;
					delete sim;
				}
			}
		}
	}
}

