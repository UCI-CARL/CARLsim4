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
* (JX) Jinwei Xing <jinweix1@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK, JX
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 07/23/2019
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
	int randSeed = 1000;	// randSeed must not interfere with STP

	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int isRunLong=1; isRunLong<=1; isRunLong++) {
	//int isRunLong = 1;
		for (int hasCOBA=1; hasCOBA<=1; hasCOBA++) {
		//int hasCOBA = 1;
			for (int mode = 1; mode <= 1; mode++) {
			//int isGPUmode = 1;
				// compare
				float rateG2noSTP = -1.0f;
				float rateG3noSTP = -1.0f;

				for (int hasSTP=0; hasSTP<=1; hasSTP++) {
				//int hasSTP = 1;
					Stopwatch watch(false);
					watch.start();
					int N = 1000;
					sim = new CARLsim("STP.firingRateSTDvsSTF",mode?GPU_MODE:CPU_MODE,USER,1,randSeed);
					int g2=sim->createGroup("STD", N, EXCITATORY_NEURON, 0, GPU_CORES);//mode?GPU_CORES:CPU_CORES);
					int g3=sim->createGroup("STF", N, EXCITATORY_NEURON, 0, GPU_CORES);//mode?GPU_CORES:CPU_CORES);
					sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
					sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
					int g0=sim->createSpikeGeneratorGroup("input0", N, EXCITATORY_NEURON, 0, GPU_CORES);// mode?GPU_CORES:CPU_CORES);
					int g1=sim->createSpikeGeneratorGroup("input1", N, EXCITATORY_NEURON, 0, GPU_CORES);// mode?GPU_CORES:CPU_CORES)C

					float wt = hasCOBA ? 0.01f : 18.0f;
					sim->connect(g0,g2,"random",RangeWeight(wt),0.1f,RangeDelay(1));
					sim->connect(g1,g3,"random",RangeWeight(wt),0.1f,RangeDelay(1));

// 					if (hasCOBA)
// 						sim->setConductances(true, 5, 0, 150, 6, 0, 150);
// 					else
// 						sim->setConductances(false);

					if (hasSTP) {
                        sim->setSTP(g0, g2, true, STPu(0.45f, 0.01f),
                                     STPtauU(50.0f, 0.03f),
                                     STPtauX(750.0f, 0.05f),
                                     STPtdAMPA(5.0f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f)); // depressive
                        sim->setSTP(g1, g3, true, STPu(0.15f, 0.02f),
                                     STPtauU(750.0f, 0.04f),
                                     STPtauX(50.0f, 0.06f),
                                     STPtdAMPA(5.0f, 0.0f),
                                     STPtdNMDA(150.0f, 0.0f),
                                     STPtdGABAa(6.0f, 0.0f),
                                     STPtdGABAb(150.0f, 0.0f),
                                     STPtrNMDA(0.0f, 0.0f),
                                     STPtrGABAb(0.0f, 0.0f)); // depressive
					}

					bool spikeAtZero = true;

					spkGenG0 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 10 Hz
					sim->setSpikeGenerator(g0, spkGenG0);
					spkGenG1 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 10 Hz
					sim->setSpikeGenerator(g1, spkGenG1);
					watch.lap();

					sim->setupNetwork();

					//setup some baseline input
					// PoissonRate spkGenG0(N);
					// spkGenG0.setRates(100.0f);
					// sim -> setSpikeRate(g0, &spkGenG0);

					sim->setSpikeMonitor(g0,"NULL");
					sim->setSpikeMonitor(g1,"NULL");
					spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
					spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

					spkMonG2->startRecording();
					spkMonG3->startRecording();
					int runTimeMs = isRunLong ? 2000 : 10;

					watch.lap();
					sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
					watch.stop(false);
					spkMonG2->stopRecording();
					spkMonG3->stopRecording();

					if (!hasSTP) {
						rateG2noSTP = spkMonG2->getPopMeanFiringRate();
						rateG3noSTP = spkMonG3->getPopMeanFiringRate();
						printf("no stp config %ld, setup %ld, run %ld\n", watch.getLapTime(0),watch.getLapTime(1), watch.getLapTime(2));
					} else {
						// if STP is on: compare spike rate to the one recorded without STP
						if (isRunLong) {
							// the run time was relatively long, so STP should have its expected effect
							printf("with stp long config %ld, setup %ld, run %ld\n", watch.getLapTime(0),watch.getLapTime(1), watch.getLapTime(2));

							printf("Long run -- spkMonG2->getPopMeanFiringRate(), rateG2noSTP: %f -- %f\n", spkMonG2->getPopMeanFiringRate(), rateG2noSTP);
							printf("Long run -- spkMonG3->getPopMeanFiringRate(), rateG3noSTP: %f -- %f\n", spkMonG3->getPopMeanFiringRate(), rateG3noSTP);
						 // facilitative
						} else {
							// the run time was really short, so STP should have no effect (because we scale STP_A so
							// that STP has no weakening/strengthening effect on the first spike)
							printf("with stp short config %ld, setup %ld, run %ld\n", watch.getLapTime(0),watch.getLapTime(1), watch.getLapTime(2));
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
