#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>

#if defined(WIN32) || defined(WIN64)
#include <periodic_spikegen.h>
#endif

/// ****************************************************************************
/// compartmental model
/// ****************************************************************************


TEST(COMPARTMENTS, setCompartmentParameters) {
	CARLsim sim("COMPARTMENTS.setCompartmentParameters", CPU_MODE, SILENT,
		0, 42);
	sim.setIntegrationMethod(RUNGE_KUTTA4, 10);

	int N = 5;

	int grpSP = sim.createGroup("SP soma", N, EXCITATORY_NEURON); // s
	int grpSR = sim.createGroup("SR d1", N, EXCITATORY_NEURON); // d1
	int grpSLM = sim.createGroup("SLM d2", N, EXCITATORY_NEURON); // d2
	int grpSO = sim.createGroup("SO d3", N, EXCITATORY_NEURON); // d3

	sim.setNeuronParameters(grpSP, 550.0f, 2.3330991f, -59.101414f, -50.428886f, 0.0021014998f, -0.41361538f,
		24.98698f, -53.223213f, 109.0f);//9 parameter setNeuronParametersCall (RS NEURON) (soma)
	sim.setNeuronParameters(grpSR, 367.0f, 1.1705916f, -59.101414f, -44.298294f, 0.2477681f, 3.3198094f,
		20.274296f, -46.076824f, 24.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
	sim.setNeuronParameters(grpSLM, 425.0f, 2.2577047f, -59.101414f, -25.137894f, 0.32122386f, 0.14995363f,
		13.203414f, -38.54892f, 69.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
	sim.setNeuronParameters(grpSO, 225.0f, 1.109572f, -59.101414f, -36.55802f, 0.29814243f, -4.385603f,
		21.473854f, -40.343994f, 21.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)

										// smoke test
	sim.setCompartmentParameters(ALL, 28.396f, 5.526f);
}


TEST(COMPARTMENTS, spikeTimesCPUvsData) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	float numModes;

	// expected spike times for soma
	int expectSpikeTimeSO[5][7] = { { 148, 188, 239, 322, 504, 673, 848 },
	{ 148, 187, 239, 325, 503, 674, 849 },
	{ 148, 187, 239, 324, 502, 674, 846 },
	{ 148, 187, 238, 323, 502, 672, 848 },
	{ 148, 187, 238, 323, 503, 674, 848 } };

	int expectSpikeTimeSP[5][7] = { { 148, 188, 240, 322, 504, 674, 848 },
	{ 149, 187, 239, 325, 504, 675, 849 },
	{ 149, 187, 239, 324, 502, 675, 846 },
	{ 149, 187, 239, 323, 502, 672, 848 },
	{ 149, 187, 239, 323, 504, 674, 849 } };

	for (int numIntSteps = 10; numIntSteps <= 50; numIntSteps += 10) {
		CARLsim* sim = new CARLsim("COMPARTMENTS.spikeTimesCPUvsData",
			CPU_MODE, SILENT, 0, 42);
		sim->setIntegrationMethod(RUNGE_KUTTA4, numIntSteps);

		int N = 5;

		int grpSP = sim->createGroup("SP soma", N, EXCITATORY_NEURON); // s
		int grpSR = sim->createGroup("SR d1", N, EXCITATORY_NEURON); // d1
		int grpSLM = sim->createGroup("SLM d2", N, EXCITATORY_NEURON); // d2
		int grpSO = sim->createGroup("SO d3", N, EXCITATORY_NEURON); // d3

		sim->setNeuronParameters(grpSP, 550.0f, 2.3330991f, -59.101414f, -50.428886f, 0.0021014998f, -0.41361538f,
			24.98698f, -53.223213f, 109.0f);//9 parameter setNeuronParametersCall (RS NEURON) (soma)
		sim->setNeuronParameters(grpSR, 367.0f, 1.1705916f, -59.101414f, -44.298294f, 0.2477681f, 3.3198094f,
			20.274296f, -46.076824f, 24.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
		sim->setNeuronParameters(grpSLM, 425.0f, 2.2577047f, -59.101414f, -25.137894f, 0.32122386f, 0.14995363f,
			13.203414f, -38.54892f, 69.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)
		sim->setNeuronParameters(grpSO, 225.0f, 1.109572f, -59.101414f, -36.55802f, 0.29814243f, -4.385603f,
			21.473854f, -40.343994f, 21.0f);//9 parameter setNeuronParametersCall (RS NEURON) (dendr)

		sim->setCompartmentParameters(grpSR, 28.396f, 5.526f);//SR 28 and 5
		sim->setCompartmentParameters(grpSLM, 50.474f, 0.0f);//SLM 50 and 0
		sim->setCompartmentParameters(grpSO, 0.0f, 49.14f);//SO 0 and 49
		sim->setCompartmentParameters(grpSP, 116.861f, 4.60f);// SP (somatic) 116 and 4

		int gin = sim->createSpikeGeneratorGroup("input", N, EXCITATORY_NEURON);
		sim->connect(gin, grpSP, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1));

		sim->setConductances(false);//This forces use of CUBA model.

									// Establish compartmental connections in order to form the following configuration:
									//	d3    SO
									//	|     |
									//	s     SP
									//	|     |
									//	d1    SR
									//	|     |
									//	d2    SLM
		sim->connectCompartments(grpSLM, grpSR);
		sim->connectCompartments(grpSR, grpSP);
		sim->connectCompartments(grpSP, grpSO);

		sim->setESTDP(ALL, false);
		sim->setISTDP(ALL, false);

		sim->setupNetwork();

		SpikeMonitor* spikeSP = sim->setSpikeMonitor(grpSP, "DEFAULT"); // put spike times into file
		SpikeMonitor* spikeSR = sim->setSpikeMonitor(grpSR, "DEFAULT"); // put spike times into file
		SpikeMonitor* spikeSLM = sim->setSpikeMonitor(grpSLM, "DEFAULT"); // put spike times into file
		SpikeMonitor* spikeSO = sim->setSpikeMonitor(grpSO, "DEFAULT"); // put spike times into file

		PoissonRate in(N);

		in.setRates(0.0f);
		sim->setSpikeRate(gin, &in);//Inactive input group

		spikeSP->startRecording();
		spikeSR->startRecording();
		spikeSLM->startRecording();
		spikeSO->startRecording();
		sim->setExternalCurrent(grpSP, 0);
		sim->runNetwork(0, 100);
		sim->setExternalCurrent(grpSP, 592);
		sim->runNetwork(0, 400);
		sim->setExternalCurrent(grpSP, 592);
		sim->runNetwork(0, 400);
		sim->setExternalCurrent(grpSP, 0);
		sim->runNetwork(0, 100);

		spikeSP->stopRecording();
		spikeSR->stopRecording();
		spikeSLM->stopRecording();
		spikeSO->stopRecording();

		// SP (somatic): expect 8 spikes at specific times
		EXPECT_EQ(spikeSP->getPopNumSpikes(), 7 * N);
		if (spikeSP->getPopNumSpikes() == 7 * N) {
			// only execute if #spikes matches, otherwise we'll segfault
			std::vector<std::vector<int> > spikeTimeSP = spikeSP->getSpikeVector2D();
			for (int neurId = 0; neurId<spikeTimeSP.size(); neurId++) {
				for (int spkT = 0; spkT<spikeTimeSP[neurId].size(); spkT++) {
					EXPECT_EQ(spikeTimeSP[neurId][spkT], expectSpikeTimeSP[numIntSteps / 10 - 1][spkT]);
				}
			}
		}

		// SR: expect silent
		EXPECT_EQ(spikeSR->getPopNumSpikes(), 0);

		// SLM: expect silent
		EXPECT_EQ(spikeSLM->getPopNumSpikes(), 0);

		// SO (grpSO dendritic): expect 8 spikes at specific times
		EXPECT_EQ(spikeSO->getPopNumSpikes(), 7 * N);
		if (spikeSO->getPopNumSpikes() == 7 * N) {
			// only execute if #spikes matches, otherwise we'll segfault
			std::vector<std::vector<int> > spikeTimeSO = spikeSO->getSpikeVector2D();
			for (int neurId = 0; neurId<spikeTimeSO.size(); neurId++) {
				for (int spkT = 0; spkT<spikeTimeSO[neurId].size(); spkT++) {
					// spike times such precede SP by 1 ms
					EXPECT_EQ(spikeTimeSO[neurId][spkT], expectSpikeTimeSO[numIntSteps / 10 - 1][spkT]);
				}
			}
		}

		delete sim;
	}
}



/*!
* \brief Testing CPU vs GPU consistency for compartment model
*
* This test makes sure that CPU mode and GPU mode of the compartment model produce the exact same spike times.
* FIX ME: TEST CURRENTLY FAILS AT some timesteps.
*/
#ifndef __NO_CUDA__
TEST(COMPARTMENTS, spikeTimesCPUvsGPU) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	for (int hasCOBA = 0; hasCOBA <= 1; hasCOBA++) {
		int cpu_numSpikesSP, cpu_numSpikesSR, cpu_numSpikesSLM, cpu_numSpikesSO;
		std::vector<std::vector<int> > cpu_spkTimesSP, cpu_spkTimesSR, cpu_spkTimesSLM, cpu_spkTimesSO;

		for (int numIntSteps = 10; numIntSteps <= 100; numIntSteps += 10) {
			//printf("This simulation has coba: %i. And has %i steps.\n", hasCOBA, numIntSteps);
			for (int isGPUmode = 0; isGPUmode <= 1; isGPUmode++) {
				//printf("This is GPU mode: %i.\n", isGPUmode);
				CARLsim* sim = new CARLsim("COMPARTMENTS.spikeTimesCPUvsGPU",
					isGPUmode ? GPU_MODE : CPU_MODE, SILENT, 0, 42);
				sim->setIntegrationMethod(RUNGE_KUTTA4, numIntSteps);

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
				sim->setExternalCurrent(grpSP, 600);
				sim->runNetwork(1, 0);
				spkMonSP->stopRecording();
				spkMonSR->stopRecording();
				spkMonSLM->stopRecording();
				spkMonSO->stopRecording();

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
					EXPECT_EQ(spkMonSP->getPopNumSpikes(), cpu_numSpikesSP);
					if (spkMonSP->getPopNumSpikes() == cpu_numSpikesSP) {
						std::vector<std::vector<int> > gpu_spkTimesSP = spkMonSP->getSpikeVector2D();
						for (int i = 0; i<cpu_spkTimesSP.size(); i++) {
							for (int j = 0; j<cpu_spkTimesSP[0].size(); j++) {
								if (numIntSteps <= 40) {
									EXPECT_EQ(gpu_spkTimesSP[i][j], cpu_spkTimesSP[i][j]);
								}
								else {
									// at 50 steps and up, we are allowed to get no more than 1 ms deviation
									EXPECT_NEAR(gpu_spkTimesSP[i][j], cpu_spkTimesSP[i][j], 1);
								}
							}
						}
					}

					EXPECT_EQ(spkMonSR->getPopNumSpikes(), cpu_numSpikesSR);
					if (spkMonSR->getPopNumSpikes() == cpu_numSpikesSR) {
						std::vector<std::vector<int> > gpu_spkTimesSR = spkMonSR->getSpikeVector2D();
						for (int i = 0; i<cpu_spkTimesSR.size(); i++) {
							for (int j = 0; j<cpu_spkTimesSR[0].size(); j++) {
								EXPECT_EQ(gpu_spkTimesSR[i][j], cpu_spkTimesSR[i][j]);
							}
						}
					}

					EXPECT_EQ(spkMonSLM->getPopNumSpikes(), cpu_numSpikesSLM);
					if (spkMonSLM->getPopNumSpikes() == cpu_numSpikesSLM) {
						std::vector<std::vector<int> > gpu_spkTimesSLM = spkMonSLM->getSpikeVector2D();
						for (int i = 0; i<cpu_spkTimesSLM.size(); i++) {
							for (int j = 0; j<cpu_spkTimesSLM[0].size(); j++) {
								EXPECT_EQ(gpu_spkTimesSLM[i][j], cpu_spkTimesSLM[i][j]);
							}
						}
					}

					EXPECT_EQ(spkMonSO->getPopNumSpikes(), cpu_numSpikesSO);
					if (spkMonSO->getPopNumSpikes() == cpu_numSpikesSO) {
						std::vector<std::vector<int> > gpu_spkTimesSO = spkMonSO->getSpikeVector2D();
						for (int i = 0; i<cpu_spkTimesSO.size(); i++) {
							for (int j = 0; j<cpu_spkTimesSO[0].size(); j++) {
								if (numIntSteps <= 40) {
									EXPECT_EQ(gpu_spkTimesSO[i][j], cpu_spkTimesSO[i][j]);
								}
								else {
									// at 50 steps and up, we are allowed to get no more than 1 ms deviation
									EXPECT_NEAR(gpu_spkTimesSO[i][j], cpu_spkTimesSO[i][j], 1);
								}
							}
						}
					}
				}

				delete sim;
			}
		}
	}
}
#endif