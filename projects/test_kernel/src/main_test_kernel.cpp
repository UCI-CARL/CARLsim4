/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
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
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 5/22/2015
*/

// include CARLsim user interface
#include <carlsim.h>

#define N_EXC 1

int main() {
	/*

	// create a network on GPU
	int randSeed = 42;
	CARLsim sim("test kernel", GPU_MODE, USER, 0, randSeed);

	// configure the network
	int gExc = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// 4 parameter version
	//sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	// 9 parameter version
	sim.setNeuronParameters(gExc, 100.0f, 0.7f, -60.0f, -40.0f, 0.03f, -2.0f, 35.0f, -50.0f, 100.0f); //RS
																									  // set up a dummy (connection probability of 0) connection
	int gInput = sim.createSpikeGeneratorGroup("input", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.connect(gInput, gExc, "one-to-one", RangeWeight(30.0f), 0.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// COMPARTMENT TESTING
	// Groups for manual compartment testing
	int s = sim.createGroup("soma", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d1 = sim.createGroup("d1", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d2 = sim.createGroup("d2", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d3 = sim.createGroup("d3", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d4 = sim.createGroup("d4", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d5 = sim.createGroup("d5", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int d6 = sim.createGroup("d6", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// some regular neuron groups
	int reg0 = sim.createGroup("reg0", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	int reg1 = sim.createGroup("reg1", 2 * N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);

	// make them 9-param Izzy neurons
	sim.setNeuronParameters(s, 550.0f, 2.0, -59.0, -50.0, 0.0, -0.0, 24.0, -53.0, 109.0f);
	sim.setNeuronParameters(d1, 367.0f, 1.0, -59.0, -44.0, 0.0, 3.0, 20.0, -46.0, 24.0f);
	sim.setNeuronParameters(d2, 425.0f, 2.0, -59.0, -25.0, 0.0, 0.0, 13.0, -38.0, 69.0f);
	sim.setNeuronParameters(d3, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d4, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d5, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(d6, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(reg0, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim.setNeuronParameters(reg1, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);

	// enable compartments
	sim.setCompartmentParameters(s, 1.0f, 1.0f);
	sim.setCompartmentParameters(d1, 1.0f, 1.0f);
	sim.setCompartmentParameters(d2, 1.0f, 1.0f);
	sim.setCompartmentParameters(d3, 1.0f, 1.0f);
	sim.setCompartmentParameters(d4, 1.0f, 1.0f);
	sim.setCompartmentParameters(d5, 1.0f, 1.0f);
	sim.setCompartmentParameters(d6, 1.0f, 1.0f);

	int gen = sim.createSpikeGeneratorGroup("SpikeGen", N_EXC, EXCITATORY_NEURON);

	// Compartment interface testing

	// grpIDs must be valid, cannot be identical
	//sim.connectCompartments(d3, d3);
	//sim.connectCompartments(s, -1);

	// no spike generators in connect call
	//sim.connectCompartments(gen, s);

	// groups must be of same size
	//sim.connectCompartments(s, d6);

	// connectCompartments is bidirectional: connecting same groups twice is illegal
	sim.connectCompartments(s, d1);

	//sim.connectCompartments(s, d1);
	//sim.connectCompartments(d1, s);

	// can't have both synaptic and compartmental connections on the same groups
	//sim.connect(s, d1, "full", RangeWeight(1.0f), 1.0f);
	//sim.connect(d1, s, "full", RangeWeight(1.0f), 1.0f);

	// can't be involved in more than 4 connections (d1-d4), d5 must break
	sim.connectCompartments(d2, s);
	sim.connectCompartments(d3, s);
	sim.connectCompartments(s, d4);
	//sim.connectCompartments(d5, s);

	// COMPARTMENT TESTING OVER

	sim.setConductances(true);

	//FORWARD_EULER
	//RUNGE_KUTTA4
	sim.setIntegrationMethod(RUNGE_KUTTA4, 10.0f);

	// build the network
	sim.setupNetwork();

	// set some monitors
	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
	//SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

	//setup some baseline input

	//PoissonRate in(N_EXC);
	//in.setRates(1.0f);
	//sim.setSpikeRate(gInput, &in);

	//smInput->startRecording();
	smExc->startRecording();

	for (int t = 0; t < 1; t++) {
		sim.runNetwork(0, 100, true);
		sim.setExternalCurrent(gExc, 70);
		sim.runNetwork(0, 900, true);
	}

	//smInput->stopRecording();
	smExc->stopRecording();

	smExc->print(true);
	//Expected Spike Times (4 param; extCurrent = 5; Euler; 2 steps / ms): 108 196 293 390 487 584 681 778 875 972
	//Expected Spike Times (9 param; extCurrent = 70): 
	//smInput->print(false);
	*/

	// Test Compartments

for (int hasCOBA = 0; hasCOBA <= 1; hasCOBA++) {
	int cpu_numSpikesSP, cpu_numSpikesSR, cpu_numSpikesSLM, cpu_numSpikesSO;
	std::vector<std::vector<int> > cpu_spkTimesSP, cpu_spkTimesSR, cpu_spkTimesSLM, cpu_spkTimesSO;
	for (int numIntSteps = 10; numIntSteps <= 100; numIntSteps += 10) {
		for (int isGPUmode = 0; isGPUmode <= 1; isGPUmode++) {
			CARLsim* sim = new CARLsim("COMPARTMENTS.spikeTimesCPUvsGPU",
				isGPUmode ? GPU_MODE : CPU_MODE, SILENT, 0, 42);
			sim->setIntegrationMethod(RUNGE_KUTTA4, numIntSteps);
			printf("This simulation has %i steps.\n", numIntSteps);

			int N = 5;

			int grpSP = sim->createGroup("excit", N, EXCITATORY_NEURON);
			int grpSR = sim->createGroup("excit", N, EXCITATORY_NEURON);
			int grpSLM = sim->createGroup("excit", N, EXCITATORY_NEURON);
			int grpSO = sim->createGroup("excit", N, EXCITATORY_NEURON);

			sim->setNeuronParameters(grpSP, 550.0f, 2.3330991f, -59.101414f, -50.428886f, 0.0021014998f,
				-0.41361538f, 24.98698f, -53.223213f, 109.0f);//9 parameter setNeuronParametersCall (RS NEURON) (soma)
			sim->setNeuronParameters(grpSR, 367.0f, 1.1705916f, -59.101414f, -44.298294f, 0.2477681f,
				3.3198094f, 20.274296f, -46.076824f, 24.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
			sim->setNeuronParameters(grpSLM, 425.0f, 2.2577047f, -59.101414f, -25.137894f, 0.32122386f,
				0.14995363f, 13.203414f, -38.54892f, 69.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
			sim->setNeuronParameters(grpSO, 225.0f, 1.109572f, -59.101414f, -36.55802f, 0.29814243f,
				-4.385603f, 21.473854f, -40.343994f, 21.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)

			sim->setCompartmentParameters(grpSR, 28.396f, 5.526f);//SR 28 and 5
			sim->setCompartmentParameters(grpSLM, 50.474f, 0.0f);//SLM 50 and 0
			sim->setCompartmentParameters(grpSO, 0.0f, 49.14f);//SO 0 and 49
			sim->setCompartmentParameters(grpSP, 116.861f, 4.60f);// SP (somatic) 116 and 4

			int gin = sim->createSpikeGeneratorGroup("input", N, EXCITATORY_NEURON);
			sim->connect(gin, grpSP, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1));

			sim->setConductances(hasCOBA);

			sim->connectCompartments(grpSLM, grpSR);
			sim->connectCompartments(grpSR, grpSP);
			sim->connectCompartments(grpSP, grpSO);

			sim->setSTDP(grpSR, false);
			sim->setSTDP(grpSLM, false);
			sim->setSTDP(grpSO, false);
			sim->setSTDP(grpSP, false);

			sim->setupNetwork();

			SpikeMonitor* spkMonSP = sim->setSpikeMonitor(grpSP, "DEFAULT"); // put spike times into file
			SpikeMonitor* spkMonSR = sim->setSpikeMonitor(grpSR, "DEFAULT"); // put spike times into file
			SpikeMonitor* spkMonSLM = sim->setSpikeMonitor(grpSLM, "DEFAULT"); // put spike times into file
			SpikeMonitor* spkMonSO = sim->setSpikeMonitor(grpSO, "DEFAULT"); // put spike times into file

			PoissonRate in(N);
			in.setRates(0.0f);
			sim->setSpikeRate(gin, &in);//Inactive input group

			spkMonSP->startRecording();
			spkMonSR->startRecording();
			spkMonSLM->startRecording();
			spkMonSO->startRecording();
			//sim->setExternalCurrent(grpSP, 0);
			//sim->runNetwork(0, 100);
			//sim->setExternalCurrent(grpSP, 592);
			//sim->runNetwork(0, 400);
			//sim->setExternalCurrent(grpSP, 592);
			//sim->runNetwork(0, 400);
			//sim->setExternalCurrent(grpSP, 0);
			//sim->runNetwork(0, 100);

			sim->setExternalCurrent(grpSP, 600);
			sim->runNetwork(1, 0);

			spkMonSP->stopRecording();
			spkMonSR->stopRecording();
			spkMonSLM->stopRecording();
			spkMonSO->stopRecording();

			spkMonSP->print(true);
			//spikeSR->print(true);
			//spikeSLM->print(true);
			spkMonSO->print(true);

			if (isGPUmode == 0) {
				cpu_numSpikesSP = spkMonSP->getPopNumSpikes();
				cpu_numSpikesSR = spkMonSR->getPopNumSpikes();
				cpu_numSpikesSLM = spkMonSLM->getPopNumSpikes();
				cpu_numSpikesSO = spkMonSO->getPopNumSpikes();
				cpu_spkTimesSP = spkMonSP->getSpikeVector2D();
				cpu_spkTimesSR = spkMonSR->getSpikeVector2D();
				cpu_spkTimesSLM = spkMonSLM->getSpikeVector2D();
				cpu_spkTimesSO = spkMonSO->getSpikeVector2D();
			}
			else {
				//EXPECT_EQ(spkMonSP->getPopNumSpikes(), cpu_numSpikesSP);
				if (spkMonSP->getPopNumSpikes() == cpu_numSpikesSP) {
					std::vector<std::vector<int> > gpu_spkTimesSP = spkMonSP->getSpikeVector2D();
					for (int i = 0; i<cpu_spkTimesSP.size(); i++) {
						for (int j = 0; j<cpu_spkTimesSP[0].size(); j++) {
							if (numIntSteps <= 40) {
								//EXPECT_EQ(gpu_spkTimesSP[i][j], cpu_spkTimesSP[i][j]);
								//printf("GPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, gpu_spkTimesSP[i][j]);
								//printf("CPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, cpu_spkTimesSP[i][j]);
								printf("The difference between GPU & CPU spiketimes is: %i.\n", gpu_spkTimesSP[i][j] - cpu_spkTimesSP[i][j]);
							}
							else {
								// at 50 steps and up, we are allowed to get no more than 1 ms deviation
								//printf("GPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, gpu_spkTimesSP[i][j]);
								//printf("CPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, cpu_spkTimesSP[i][j]);
								printf("The difference between GPU & CPU spiketimes is: %i.\n", gpu_spkTimesSP[i][j] - cpu_spkTimesSP[i][j]);
							}
						}
					}
				}

				//EXPECT_EQ(spkMonSR->getPopNumSpikes(), cpu_numSpikesSR);
				/*if (spkMonSR->getPopNumSpikes() == cpu_numSpikesSR) {
					std::vector<std::vector<int> > gpu_spkTimesSR = spkMonSR->getSpikeVector2D();
					for (int i = 0; i<cpu_spkTimesSR.size(); i++) {
						for (int j = 0; j<cpu_spkTimesSR[0].size(); j++) {
							EXPECT_EQ(gpu_spkTimesSR[i][j], cpu_spkTimesSR[i][j]);
						}
					}
				}

				//EXPECT_EQ(spkMonSLM->getPopNumSpikes(), cpu_numSpikesSLM);
				if (spkMonSLM->getPopNumSpikes() == cpu_numSpikesSLM) {
					std::vector<std::vector<int> > gpu_spkTimesSLM = spkMonSLM->getSpikeVector2D();
					for (int i = 0; i<cpu_spkTimesSLM.size(); i++) {
						for (int j = 0; j<cpu_spkTimesSLM[0].size(); j++) {
							EXPECT_EQ(gpu_spkTimesSLM[i][j], cpu_spkTimesSLM[i][j]);
						}
					}
				}*/

				//EXPECT_EQ(spkMonSO->getPopNumSpikes(), cpu_numSpikesSO);
				if (spkMonSO->getPopNumSpikes() == cpu_numSpikesSO) {
					std::vector<std::vector<int> > gpu_spkTimesSO = spkMonSO->getSpikeVector2D();
					for (int i = 0; i<cpu_spkTimesSO.size(); i++) {
						for (int j = 0; j<cpu_spkTimesSO[0].size(); j++) {
							if (numIntSteps <= 40) {
								//printf("GPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, gpu_spkTimesSO[i][j]);
								//printf("CPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, cpu_spkTimesSO[i][j]);
								printf("The difference between GPU & CPU spiketimes is: %i.\n", gpu_spkTimesSO[i][j] - cpu_spkTimesSO[i][j]);
							}
							else {
								// at 50 steps and up, we are allowed to get no more than 1 ms deviation
								//printf("GPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, gpu_spkTimesSO[i][j]);
								//printf("CPU spike time for neuron's # %i, %i th spike is: %i.\n", i, j, cpu_spkTimesSO[i][j]);
								printf("The difference between GPU & CPU spiketimes is: %i.\n", gpu_spkTimesSO[i][j] - cpu_spkTimesSO[i][j]);
							}
						}
					}
				}
			}
	
			delete sim;
			}
		}
	}

	return 0;
}
