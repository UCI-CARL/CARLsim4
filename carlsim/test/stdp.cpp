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
#include "gtest/gtest.h"
#include "carlsim_tests.h"
#include <carlsim.h>

#include <periodic_spikegen.h>
#include <interactive_spikegen.h>
#include <pre_post_group_spikegen.h>



/// **************************************************************************************************************** ///
/// SPIKE-TIMING-DEPENDENT PLASTICITY STDP
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(STDP, setSTDPTrue) {
	float alphaPlus = 5.0f;		// the exact values don't matter
	float alphaMinus = -10.0f;
	float tauPlus = 15.0f;
	float tauMinus = 20.0f;
	float gamma = 10.0f;
	float betaLTP = 1.0f;
	float betaLTD = -1.2f;
	float lambda = 12.0f;
	float delta = 40.0f;
	CARLsim* sim;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int stdpType = 0; stdpType < 2; stdpType++) { // we have two stdp types {STANDARD, DA_MOD}
			for(int stdpCurve = 0; stdpCurve < 2; stdpCurve++) { // we have four stdp curves, two for ESTDP, two for ISTDP
				sim = new CARLsim("STDP.setSTDPTrue",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

				int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
				if (stdpType == 0) {
					if (stdpCurve == 0) {
						sim->setESTDP(g1, true, STANDARD, ExpCurve(alphaPlus,tauPlus,alphaMinus,tauMinus));
						sim->setISTDP(g1, true, STANDARD, ExpCurve(alphaPlus,tauPlus,alphaMinus,tauMinus));
					} else { //stdpCurve == 1
						sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(alphaPlus,tauPlus,alphaMinus,tauMinus, gamma));
						sim->setISTDP(g1, true, STANDARD, PulseCurve(betaLTP,betaLTD,lambda,delta));
					}
				} else { // stdpType == 1
					if (stdpCurve == 0) {
						sim->setESTDP(g1, true, DA_MOD, ExpCurve(alphaPlus,tauPlus,alphaMinus,tauMinus));
						sim->setISTDP(g1, true, DA_MOD, ExpCurve(alphaPlus,tauPlus,alphaMinus,tauMinus));
					} else { //stdpCurve == 1
						sim->setESTDP(g1, true, DA_MOD, TimingBasedCurve(alphaPlus,tauPlus,alphaMinus,tauMinus, gamma));
						sim->setISTDP(g1, true, DA_MOD, PulseCurve(betaLTP,betaLTD,lambda,delta));
					}
				}

				GroupSTDPInfo gInfo = sim->getGroupSTDPInfo(g1);
				EXPECT_TRUE(gInfo.WithSTDP);
				EXPECT_TRUE(gInfo.WithESTDP);
				EXPECT_TRUE(gInfo.WithISTDP);
				if (stdpType == 0) {
					EXPECT_TRUE(gInfo.WithESTDPtype == STANDARD);
					EXPECT_TRUE(gInfo.WithESTDPtype == STANDARD);
				} else { // stdpType == 1
					EXPECT_TRUE(gInfo.WithESTDPtype == DA_MOD);
					EXPECT_TRUE(gInfo.WithISTDPtype == DA_MOD);
				}

				if (stdpCurve == 0) {
					EXPECT_TRUE(gInfo.WithESTDPcurve == EXP_CURVE);
					EXPECT_TRUE(gInfo.WithISTDPcurve == EXP_CURVE);
				} else {
					EXPECT_TRUE(gInfo.WithESTDPcurve == TIMING_BASED_CURVE);
					EXPECT_TRUE(gInfo.WithISTDPcurve == PULSE_CURVE);
				}

				EXPECT_FLOAT_EQ(gInfo.ALPHA_PLUS_EXC,alphaPlus);
				EXPECT_FLOAT_EQ(gInfo.ALPHA_MINUS_EXC,alphaMinus);
				EXPECT_FLOAT_EQ(gInfo.TAU_PLUS_INV_EXC,1.0/tauPlus);
				EXPECT_FLOAT_EQ(gInfo.TAU_MINUS_INV_EXC,1.0/tauMinus);
				if (stdpCurve == 0) {
					EXPECT_FLOAT_EQ(gInfo.ALPHA_PLUS_INB,alphaPlus);
					EXPECT_FLOAT_EQ(gInfo.ALPHA_MINUS_INB,alphaMinus);
					EXPECT_FLOAT_EQ(gInfo.TAU_PLUS_INV_INB,1.0/tauPlus);
					EXPECT_FLOAT_EQ(gInfo.TAU_MINUS_INV_INB,1.0/tauMinus);
					EXPECT_FLOAT_EQ(gInfo.GAMMA, 0.0f);
				} else {
					EXPECT_FLOAT_EQ(gInfo.BETA_LTP,betaLTP);
					EXPECT_FLOAT_EQ(gInfo.BETA_LTD,betaLTD);
					EXPECT_FLOAT_EQ(gInfo.LAMBDA,lambda);
					EXPECT_FLOAT_EQ(gInfo.DELTA,delta);
					EXPECT_FLOAT_EQ(gInfo.GAMMA, gamma);
				}

				delete sim;
			}
		}
	}
}

/*!
 * \brief testing setSTDP to false
 * This function tests the information stored in the group info struct after disabling STDP via setSTDP
 */
TEST(STDP, setSTDPFalse) {
	float alphaPlus = 5.0f;		// the exact values don't matter
	float alphaMinus = -10.0f;
	float tauPlus = 15.0f;
	float tauMinus = 20.0f;
	float betaLTP = 1.0f;
	float betaLTD = -2.0f;
	float lambda = 3.0f;
	float delta = 4.0f;
	CARLsim* sim;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("STDP.setSTDPFalse",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setESTDP(g1,false,STANDARD, ExpCurve(alphaPlus,tauPlus,alphaMinus,tauMinus));
		sim->setISTDP(g1,false,STANDARD, PulseCurve(betaLTP,betaLTD,lambda,delta));

		GroupSTDPInfo gInfo = sim->getGroupSTDPInfo(g1);
		EXPECT_FALSE(gInfo.WithSTDP);
		EXPECT_FALSE(gInfo.WithESTDP);
		EXPECT_FALSE(gInfo.WithISTDP);

		EXPECT_FLOAT_EQ(gInfo.ALPHA_PLUS_EXC, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.ALPHA_MINUS_EXC, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.TAU_PLUS_INV_EXC, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.TAU_MINUS_INV_EXC, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.ALPHA_PLUS_EXC, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.ALPHA_MINUS_EXC, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.TAU_PLUS_INV_EXC, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.TAU_MINUS_INV_EXC, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.BETA_LTP, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.BETA_LTD, 0.0f);
		EXPECT_FLOAT_EQ(gInfo.LAMBDA, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.DELTA, 1.0f);
		EXPECT_FLOAT_EQ(gInfo.GAMMA, 0.0f);

		delete sim;
	}
}

/*!
 * \brief testing setSTDPNeuromodulatorParameters
 * This function tests the information stored in the group info struct after setting neuromodulator parameters
 */
TEST(STDP, setNeuromodulatorParameters) {
	float alphaPlus = 1.0f;		// the exact values don't matter
	float alphaMinus = 1.2f;
	float tauPlus = 20.0f;
	float tauMinus = 20.0f;
	float baseDP = 1.0f;
	float base5HT = 2.0f;
	float baseACh = 3.0f;
	float baseNE = 4.0f;
	float tauDP = 100.0f;
	float tau5HT = 200.0f;
	float tauACh = 300.0f;
	float tauNE = 400.0f;
	CARLsim* sim;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("STDP.setNeuromodulatorParameters",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setSTDP(g1,true,DA_MOD,alphaPlus,tauPlus,alphaMinus,tauMinus);
		sim->setNeuromodulator(g1, baseDP, tauDP, base5HT, tau5HT, baseACh, tauACh, baseNE, tauNE);

		// Temporarily mark out the testing code
		// Discuss whether carlsim user interface needs to spport GroupConfigRT
		GroupSTDPInfo gInfo = sim->getGroupSTDPInfo(g1);
		EXPECT_TRUE(gInfo.WithSTDP);
		EXPECT_TRUE(gInfo.WithESTDPtype == DA_MOD);

		GroupNeuromodulatorInfo gInfo2 = sim->getGroupNeuromodulatorInfo(g1);
		EXPECT_FLOAT_EQ(gInfo2.baseDP, baseDP);
		EXPECT_FLOAT_EQ(gInfo2.base5HT, base5HT);
		EXPECT_FLOAT_EQ(gInfo2.baseACh, baseACh);
		EXPECT_FLOAT_EQ(gInfo2.baseNE, baseNE);
		EXPECT_FLOAT_EQ(gInfo2.decayDP, 1.0 - 1.0 / tauDP);
		EXPECT_FLOAT_EQ(gInfo2.decay5HT, 1.0 - 1.0 / tau5HT);
		EXPECT_FLOAT_EQ(gInfo2.decayACh, 1.0 - 1.0 / tauACh);
		EXPECT_FLOAT_EQ(gInfo2.decayNE, 1.0 - 1.0 / tauNE);

		delete sim;
	}
}

/*!
 * \brief testing the effect of dopamine modulation
 * This function tests the effect of dopamine modulation on a single synapse (reinforcement learning).
 * The the synaptic weight modulated by dopamine is expected to be higher than that without dopamine modulation
 */
TEST(STDP, DASTDPWeightBoost_Adv) {
	float tauPlus = 20.0f;
	float tauMinus = 20.0f;
	float alphaPlus = 0.1f;
	float alphaMinus = -0.1f;
	int g1, gin, g1noise, gda;
	InteractiveSpikeGenerator* iSpikeGen = new InteractiveSpikeGenerator(500, 500);
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights;
	int size;
	SpikeMonitor* spikeMonPost;
	SpikeMonitor* spikeMonPre;
	float weightDAMod, weightNonDAMod;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int damod = 0; damod < 2; damod++) {
				CARLsim* sim = new CARLsim("STDP.DASTDPWeightBoost", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("post-ex", 1, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gin = sim->createSpikeGeneratorGroup("pre-ex", 1, EXCITATORY_NEURON);
				g1noise = sim->createSpikeGeneratorGroup("post-ex-noise", 1, EXCITATORY_NEURON);
				gda = sim->createSpikeGeneratorGroup("DA neurons", 500, DOPAMINERGIC_NEURON);

				if (coba) {
					sim->connect(gin,g1,"one-to-one", RangeWeight(0.0, 1.0f/100, 20.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
					sim->connect(g1noise, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					// enable COBA, set up STDP, enable dopamine-modulated STDP
					sim->setConductances(true,5,150,6,150);
					sim->setSTDP(g1, true, DA_MOD, alphaPlus/100, tauPlus, alphaMinus/100, tauMinus);
				} else { // cuba mode
					sim->connect(gin,g1,"one-to-one", RangeWeight(0.0, 1.0f, 20.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
					sim->connect(g1noise, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					// set up STDP, enable dopamine-modulated STDP
					sim->setSTDP(g1, true, DA_MOD, alphaPlus, tauPlus, alphaMinus, tauMinus);
					sim->setConductances(false);
				}

				sim->setNeuromodulator(ALL);

				sim->setWeightAndWeightChangeUpdate(INTERVAL_10MS, true, 0.99f);

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gda, iSpikeGen);

				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gin, g1, "NULL");

				spikeMonPost = sim->setSpikeMonitor(g1, "NULL");
				spikeMonPre = sim->setSpikeMonitor(gin, "NULL");
				sim->setSpikeMonitor(gda, "NULL");

				//setup baseline firing rate
				PoissonRate in(1);
				in.setRates(6.0f); // 6Hz
				sim->setSpikeRate(gin, &in);
				sim->setSpikeRate(g1noise, &in);

				for (int t = 0; t < 200; t++) {
					spikeMonPost->startRecording();
					spikeMonPre->startRecording();
					sim->runNetwork(1, 0, false);
					spikeMonPost->stopRecording();
					spikeMonPre->stopRecording();

					// get spike time of pre-synaptic neuron post-synaptic neuron
					spikesPre = spikeMonPre->getSpikeVector2D()[0]; // pre-neuron spikes
					spikesPost = spikeMonPost->getSpikeVector2D()[0]; // post-neuron in spikes

					// detect LTP or LTD
					for (int j = 0; j < spikesPre.size(); j++) { // j: index of the (j+1)-th spike
						for (int k = 0; k < spikesPost.size(); k++) { // k: index of the (k+1)-th spike
							int diff = spikesPost[k] - spikesPre[j]; // (post-spike time) - (pre-spike time)
							// if LTP is detected, set up reward (activate DA neurons ) to reinforcement this synapse
							if (diff > 0 && diff <= 20) {
								//printf("LTP\n");
								if (damod) {
									iSpikeGen->setQuotaAll(1);
									//printf("release DA\n");
								}
							}

							//if (diff < 0 && diff >= -20)
							//	printf("LTD\n");
						}
					}
				}

				std::vector< std::vector<float> > weights = CM->takeSnapshot();
				if (damod) {
					weightDAMod = weights[0][0];
				} else {
					weightNonDAMod = weights[0][0];
				}

				delete sim;
			}

			EXPECT_TRUE(weightDAMod >= weightNonDAMod);
			//printf("mode:%d coba:%d Non-DA w:%f DA w:%f\n", mode, coba, weightNonDAMod, weightDAMod);
		}
	}

	delete iSpikeGen;
}

#ifndef __NO_CUDA__
/*!
* \brief testing the exponential E-STDP curve
* This function tests whether E-STDP change synaptic weight as expected
* Wtih control of pre- and post-neurons' spikes, the synaptic weights of CPU and GPU mode are expected
* to be the same
*/
TEST(STDP, ESTDPExpCurveCPUvsGPU_Adv) {
	// simulation details
	int size;
	int gex1, gex2, g1;
	float ALPHA_LTP = 0.10f;
	float ALPHA_LTD = -0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;
	float cpuWeight, gpuWeight;

	for (int coba = 0; coba < 2; coba++) {
		for (int offset = -30; offset <= 30; offset += 5) {
			
			for (int mode = 0; mode < TESTED_MODES; mode++) {
				if (offset == 0) continue; // skip offset == 0;
										   // create a network
				CARLsim* sim = new CARLsim("STDP.ESTDPExpCurve", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex1 = sim->createSpikeGeneratorGroup("input-ex1", 1, EXCITATORY_NEURON, 0);
				gex2 = sim->createSpikeGeneratorGroup("input-ex2", 1, EXCITATORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gex2, gex1);

				if (coba) { // conductance-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight / 100, maxInhWeight / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ESTDP
					sim->setConductances(true, 5, 150, 6, 150);
					sim->setESTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP / 100, TAU_LTP, ALPHA_LTD / 100, TAU_LTP));
				}
				else { // current-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ESTDP
					sim->setConductances(false);
					sim->setESTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTP));
				}

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gex1, prePostSpikeGen);
				sim->setSpikeGenerator(gex2, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gex2, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);
				SpikeMonitor* SMgex1 = sim->setSpikeMonitor(gex1, "Default");
				SpikeMonitor* SMgex2 = sim->setSpikeMonitor(gex2, "Default");
				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");

				//SMgex1->startRecording();
				//SMgex2->startRecording();
				//SMg1->startRecording();

				sim->runNetwork(1, 0, false);

				//SMgex1->stopRecording();
				//SMgex2->stopRecording();
				//SMg1->stopRecording();

				//SMgex1->print(true);
				//SMgex2->print(true);
				//SMg1->print(true);

				std::vector<std::vector<float> > weights = CM->takeSnapshot();
				if (mode == 0)
					cpuWeight = weights[0][0];
				else
					gpuWeight = weights[0][0];

				delete prePostSpikeGen;
				delete sim;
			}

			EXPECT_NEAR(cpuWeight / gpuWeight, 1.0f, 0.000001f);
			//printf("coba:%d offset:%d cpu/gpu ratio:%f\n", coba, offset, cpuWeight / gpuWeight);
		}
	}
}
#endif // !__NO_CUDA__

/*!
 * \brief testing the exponential E-STDP curve
 * This function tests whether E-STDP change synaptic weight as expected
 * Wtih control of pre- and post-neurons' spikes, the synaptic weight is expected to increase or decrease to
 * maximum or minimum synaptic weight respectively.
 */
TEST(STDP, ESTDPExpCurve_Adv) {
	// simulation details
	int size;
	int gex1, gex2, g1;
	float ALPHA_LTP = 0.10f;
	float ALPHA_LTD = -0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int offset = -30; offset <= 30; offset += 5) {
				if (offset == 0) continue; // skip offset == 0;
				// create a network
				CARLsim* sim = new CARLsim("STDP.ESTDPExpCurve", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex1 = sim->createSpikeGeneratorGroup("input-ex1", 1, EXCITATORY_NEURON, 0);
				gex2 = sim->createSpikeGeneratorGroup("input-ex2", 1, EXCITATORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gex2, gex1);

				if (coba) { // conductance-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight/100, maxInhWeight/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ESTDP
					sim->setConductances(true,5,150,6,150);
					sim->setESTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP/100, TAU_LTP, ALPHA_LTD/100, TAU_LTP));
				} else { // current-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ESTDP
					sim->setConductances(false);
					sim->setESTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTP));
				}

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gex1, prePostSpikeGen);
				sim->setSpikeGenerator(gex2, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gex2, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);
				SpikeMonitor* SMgex1 = sim->setSpikeMonitor(gex1, "Default");
				SpikeMonitor* SMgex2 = sim->setSpikeMonitor(gex2, "Default");
				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");

				//SMgex1->startRecording();
				//SMgex2->startRecording();
				//SMg1->startRecording();

				sim->runNetwork(55, 0, false);

				//SMgex1->stopRecording();
				//SMgex2->stopRecording();
				//SMg1->stopRecording();

				//SMgex1->print(true);
				//SMgex2->print(true);
				//SMg1->print(true);

				std::vector<std::vector<float> > weights = CM->takeSnapshot();
				if (offset > 0) { // pre-post
					if (coba) {
						EXPECT_NEAR(maxInhWeight/100, weights[0][0], 0.005f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					} else {
						EXPECT_NEAR(maxInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				} else { // post-pre
					if (coba) {
						EXPECT_NEAR(minInhWeight/100, weights[0][0], 0.005f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					} else {
						EXPECT_NEAR(minInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				}

				delete prePostSpikeGen;
				delete sim;
			}
		}
	}
}

#ifndef __NO_CUDA__
/*!
* \brief testing the timing-based E-STDP curve
* This function tests whether E-STDP change synaptic weight as expected
* Wtih control of pre- and post-neurons' spikes, the synaptic weights of CPU and GPU mode are expected
* to be the same
*/
TEST(STDP, ESTDPTimingBasedCurveCPUvsGPU_Adv) {
	// simulation details
	int size;
	int gex1, gex2, g1;
	float ALPHA_LTP = 0.10f;
	float ALPHA_LTD = -0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float GAMMA = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;
	float cpuWeight, gpuWeight;
	
	for (int coba = 0; coba < 2; coba++) {
		for (int offset = -24; offset <= 24; offset += 3) {
			for (int mode = 0; mode < TESTED_MODES; mode++) {
				if (offset == 0) continue; // skip offset == 0;
				// create a network
				CARLsim* sim = new CARLsim("STDP.ESTDTimingBasedCurve", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex1 = sim->createSpikeGeneratorGroup("input-ex1", 1, EXCITATORY_NEURON, 0);
				gex2 = sim->createSpikeGeneratorGroup("input-ex2", 1, EXCITATORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gex2, gex1);

				if (coba) { // conductance-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight / 100, maxInhWeight / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ESTDP
					sim->setConductances(true, 5, 150, 6, 150);
					sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(ALPHA_LTP / 100, TAU_LTP, ALPHA_LTD / 100, TAU_LTP, GAMMA));
				}
				else { // current-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ESTDP
					sim->setConductances(false, 0, 0, 0, 0);
					sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTP, GAMMA));
				}

				sim->setSpikeGenerator(gex1, prePostSpikeGen);
				sim->setSpikeGenerator(gex2, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gex2, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				sim->runNetwork(1, 0, false);

				std::vector< std::vector<float> > weights = CM->takeSnapshot();

				if (mode == 0)
					cpuWeight = weights[0][0];
				else
					gpuWeight = weights[0][0];

				delete prePostSpikeGen;
				delete sim;
			}

			EXPECT_NEAR(cpuWeight / gpuWeight, 1.0f, 0.000001f);
			//printf("coba:%d offset:%d cpu/gpu ratio:%f\n", coba, offset, cpuWeight / gpuWeight);
		}
	}
}
#endif // !__NO_CUDA__

/*!
 * \brief testing the timing-based E-STDP curve
 * This function tests whether E-STDP change synaptic weight as expected
 * Wtih control of pre- and post-neurons' spikes, the synaptic weight is expected to increase or decrease to
 * maximum or minimum synaptic weith respectively.
 */
TEST(STDP, ESTDPTimingBasedCurve_Adv) {
	// simulation details
	int size;
	int gex1, gex2, g1;
	float ALPHA_LTP = 0.10f;
	float ALPHA_LTD = -0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float GAMMA = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int offset = -24; offset <= 24; offset += 3) {
				if (offset == 0) continue; // skip offset == 0;
				// create a network
				CARLsim* sim = new CARLsim("STDP.ESTDTimingBasedCurve", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex1 = sim->createSpikeGeneratorGroup("input-ex1", 1, EXCITATORY_NEURON, 0);	
				gex2 = sim->createSpikeGeneratorGroup("input-ex2", 1, EXCITATORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gex2, gex1);

				if (coba) { // conductance-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight/100, maxInhWeight/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ESTDP
					sim->setConductances(true,5,150,6,150);
					sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(ALPHA_LTP/100, TAU_LTP, ALPHA_LTD/100, TAU_LTP, GAMMA));
				} else { // current-based
					sim->connect(gex1, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gex2, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ESTDP
					sim->setConductances(false,0,0,0,0);
					sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTP, GAMMA));
				}

				sim->setSpikeGenerator(gex1, prePostSpikeGen);
				sim->setSpikeGenerator(gex2, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gex2, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				sim->runNetwork(75, 0, false);

				std::vector< std::vector<float> > weights = CM->takeSnapshot();
				if (offset > 0) { // pre-post
					if (coba) {
						if (offset == 3 || offset == 6) {
							EXPECT_NEAR(maxInhWeight / 100, weights[0][0], 0.005f);
							//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
						} else {
							EXPECT_NEAR(minInhWeight / 100, weights[0][0], 0.005f);
							//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
						}
					} else {
						if (offset == 3 || offset == 6 || offset == 9) {
							EXPECT_NEAR(maxInhWeight, weights[0][0], 0.5f);
							//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
						} else {
							EXPECT_NEAR(minInhWeight, weights[0][0], 0.5f);
							//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
						}
					}
				} else { // post-pre
					if (coba) {
						EXPECT_NEAR(minInhWeight/100, weights[0][0], 0.005f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					} else {
						EXPECT_NEAR(minInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				}

				delete prePostSpikeGen;
				delete sim;
			}
		}
	}
}

#ifndef __NO_CUDA__
/*!
* \brief testing the pulse I-STDP curve
* This function tests whether I-STDP change synaptic weight as expected
* Wtih control of pre- and post-neurons' spikes, the synaptic weights of CPU and GPU mode are expected
* to be the same
*/
TEST(STDP, ISTDPPulseCurveCPUvsGPU_Adv) {
	// simulation details
	int size;
	int gin, gex, g1;
	float BETA_LTP = 0.10f;
	float BETA_LTD = -0.14f;
	float LAMBDA = 9.0f;
	float DELTA = 40.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;
	float cpuWeight, gpuWeight;

	for (int coba = 0; coba < 2; coba++) {
		for (int offset = -15; offset <= 15; offset += 10) {
			for (int mode = 0; mode < TESTED_MODES; mode++) {
				// create a network
				CARLsim* sim = new CARLsim("STDP.ISTDPPulseCurve", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex = sim->createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON, 0);
				gin = sim->createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gin, gex);

				if (coba) { // conductance-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight / 100, maxInhWeight / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ISTDP
					sim->setConductances(true, 5, 150, 6, 150);
					sim->setISTDP(g1, true, STANDARD, PulseCurve(BETA_LTP / 100, BETA_LTD / 100, LAMBDA, DELTA));
				}
				else { // current-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ISTDP
					sim->setConductances(false, 0, 0, 0, 0);
					sim->setISTDP(g1, true, STANDARD, PulseCurve(BETA_LTP, BETA_LTD, LAMBDA, DELTA));
				}

				sim->setSpikeGenerator(gex, prePostSpikeGen);
				sim->setSpikeGenerator(gin, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gin, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");
				SpikeMonitor* SMgin = sim->setSpikeMonitor(gin, "Default");
				SpikeMonitor* SMgex = sim->setSpikeMonitor(gex, "Default");

				//SMg1->startRecording();
				//SMgin->startRecording();
				//SMgex->startRecording();

				sim->runNetwork(1, 0, false);

				//SMg1->stopRecording();
				//SMgin->stopRecording();
				//SMgex->stopRecording();

				//SMgin->print(true);
				//SMgex->print(true);
				//SMg1->print(true);

				std::vector< std::vector<float> > weights = CM->takeSnapshot();

				if (mode == 0)
					cpuWeight = weights[0][0];
				else
					gpuWeight = weights[0][0];

				delete prePostSpikeGen;
				delete sim;
			}

			EXPECT_NEAR(cpuWeight / gpuWeight, 1.0f, 0.000001f);
			//printf("coba:%d offset:%d cpu/gpu ratio:%f\n", coba, offset, cpuWeight / gpuWeight);
		}
	}
}
#endif // !__NO_CUDA__

/*!
 * \brief testing the pulse I-STDP curve
 * This function tests whether I-STDP change synaptic weight as expected
 * Wtih control of pre- and post-neurons' spikes, the synaptic weight is expected to increase or decrease to
 * maximum or minimum synaptic weith respectively.
 */
TEST(STDP, ISTDPulseCurve_Adv) {
	// simulation details
	int size;
	int gin, gex, g1;
	float BETA_LTP = 0.10f;
	float BETA_LTD = -0.14f;
	float LAMBDA = 9.0f;
	float DELTA = 40.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int offset = -15; offset <= 15; offset += 10) {
				// create a network
				CARLsim* sim = new CARLsim("STDP.ISTDPPulseCurve", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);

				sim->setIntegrationMethod(FORWARD_EULER, 1);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex = sim->createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON, 0);
				gin = sim->createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gin, gex);

				if (coba) { // conductance-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight/100, maxInhWeight/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ISTDP
					sim->setConductances(true,5,150,6,150);
					sim->setISTDP(g1, true, STANDARD, PulseCurve(BETA_LTP / 100, BETA_LTD / 100, LAMBDA, DELTA));
				} else { // current-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ISTDP
					sim->setConductances(false,0,0,0,0);
					sim->setISTDP(g1, true, STANDARD, PulseCurve(BETA_LTP, BETA_LTD, LAMBDA, DELTA));
				}

				sim->setSpikeGenerator(gex, prePostSpikeGen);
				sim->setSpikeGenerator(gin, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gin, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");
				SpikeMonitor* SMgin = sim->setSpikeMonitor(gin, "Default");
				SpikeMonitor* SMgex = sim->setSpikeMonitor(gex, "Default");

				//SMg1->startRecording();
				//SMgin->startRecording();
				//SMgex->startRecording();

				// run for 20 seconds
				sim->runNetwork(20,0, false);

				//SMg1->stopRecording();
				//SMgin->stopRecording();
				//SMgex->stopRecording();

				//SMgin->print(true);
				//SMgex->print(true);
				//SMg1->print(true);

				std::vector< std::vector<float> > weights = CM->takeSnapshot();
				if (offset == -5 || offset == 5) { // I-STDP LTP
					if (coba) {
						EXPECT_NEAR(maxInhWeight/100, weights[0][0], 0.0075f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					} else {
						EXPECT_NEAR(maxInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				} else { // I-STDP LTD
					if (coba) {
						EXPECT_NEAR(minInhWeight/100, weights[0][0], 0.0075f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					} else {
						EXPECT_NEAR(minInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				}

				delete prePostSpikeGen;
				delete sim;
			}
		}
	}
}

#ifndef __NO_CUDA__
/*!
* \brief testing the exponential I-STDP curve
* This function tests whether I-STDP change synaptic weight as expected
* Wtih control of pre- and post-neurons' spikes, the synaptic weights of CPU and GPU mode are expected
* to be the same
*/
TEST(STDP, ISTDPExpCurveCPUvsGPU_Adv) {
	// simulation details
	int size;
	int gin, gex, g1;
	float ALPHA_LTP = -0.10f;
	float ALPHA_LTD = 0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;
	float cpuWeight, gpuWeight;

	for (int coba = 0; coba < 2; coba++) {
		for (int offset = -24; offset <= 24; offset += 3) {
			if (offset == 0) continue; // skip offset == 0;
			for (int mode = 0; mode < TESTED_MODES; mode++) {
				// create a network
				CARLsim* sim = new CARLsim("STDP.ISTDPPulseCurve", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex = sim->createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON, 0);
				gin = sim->createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gin, gex);

				if (coba) { // conductance-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight / 100, maxInhWeight / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ISTDP
					sim->setConductances(true, 5, 150, 6, 150);
					sim->setISTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP / 100, TAU_LTP, ALPHA_LTD / 100, TAU_LTD));
				}
				else { // current-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ISTDP
					sim->setConductances(false, 0, 0, 0, 0);
					sim->setISTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD));
				}

				sim->setSpikeGenerator(gex, prePostSpikeGen);
				sim->setSpikeGenerator(gin, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gin, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");
				SpikeMonitor* SMgin = sim->setSpikeMonitor(gin, "Default");
				SpikeMonitor* SMgex = sim->setSpikeMonitor(gex, "Default");

				//SMg1->startRecording();
				//SMgin->startRecording();
				//SMgex->startRecording();
				//CM->takeSnapshot();

				sim->runNetwork(1, 0, false);

				//SMg1->stopRecording();
				//SMgin->stopRecording();
				//SMgex->stopRecording();
				//CM->takeSnapshot();

				//SMgin->print(true);
				//SMgex->print(true);
				//SMg1->print(true);

				//CM->printSparse();

				std::vector< std::vector<float> > weights = CM->takeSnapshot();

				if (mode == 0)
					cpuWeight = weights[0][0];
				else
					gpuWeight = weights[0][0];

				delete prePostSpikeGen;
				delete sim;
			}

			EXPECT_NEAR(cpuWeight / gpuWeight, 1.0f, 0.000001f);
			//printf("coba:%d offset:%d cpu:%f cpu/gpu ratio:%f\n", coba, offset, cpuWeight, cpuWeight / gpuWeight);
		}
	}
}
#endif // !__NO_CUDA__

/*!
* \brief testing the Exponential I-STDP curve
* This function tests whether I-STDP change synaptic weight as expected
* Wtih control of pre- and post-neurons' spikes, the synaptic weight is expected to increase or decrease to
* maximum or minimum synaptic weith respectively.
*/
TEST(STDP, ISTDPExpCurve_Adv) {
	// simulation details
	int size;
	int gin, gex, g1;
	float ALPHA_LTP = -0.10f;
	float ALPHA_LTD = 0.14f;
	float TAU_LTP = 20.0f;
	float TAU_LTD = 20.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;

	for (int mode = 0; mode < TESTED_MODES; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int offset = -24; offset <= 24; offset += 3) {
				if (offset == 0) continue; // skip offset == 0;
				// create a network
				CARLsim* sim = new CARLsim("STDP.ISTDPPulseCurve", mode ? GPU_MODE : CPU_MODE, SILENT, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON, 0);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex = sim->createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON, 0);
				gin = sim->createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON, 0);

				PrePostGroupSpikeGenerator* prePostSpikeGen = new PrePostGroupSpikeGenerator(100, offset, gin, gex);

				if (coba) { // conductance-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight / 100, maxInhWeight / 100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// enable COBA, set up ISTDP
					sim->setConductances(true, 5, 150, 6, 150);
					sim->setISTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP / 100, TAU_LTP, ALPHA_LTD / 100, TAU_LTD));
				}
				else { // current-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

					// set up ISTDP
					sim->setConductances(false, 0, 0, 0, 0);
					sim->setISTDP(g1, true, STANDARD, ExpCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD));
				}

				sim->setSpikeGenerator(gex, prePostSpikeGen);
				sim->setSpikeGenerator(gin, prePostSpikeGen);

				// build the network
				sim->setupNetwork();

				ConnectionMonitor* CM = sim->setConnectionMonitor(gin, g1, "NULL");
				CM->setUpdateTimeIntervalSec(-1);

				SpikeMonitor* SMg1 = sim->setSpikeMonitor(g1, "Default");
				SpikeMonitor* SMgin = sim->setSpikeMonitor(gin, "Default");
				SpikeMonitor* SMgex = sim->setSpikeMonitor(gex, "Default");

				//SMg1->startRecording();
				//SMgin->startRecording();
				//SMgex->startRecording();

				// run for 25 seconds
				sim->runNetwork(25, 0, false);

				//SMg1->stopRecording();
				//SMgin->stopRecording();
				//SMgex->stopRecording();

				//SMgin->print(true);
				//SMgex->print(true);
				//SMg1->print(true);

				std::vector< std::vector<float> > weights = CM->takeSnapshot();
				if (offset > 0) { // pre-post
					if (coba) {
						EXPECT_NEAR(minInhWeight / 100, weights[0][0], 0.005f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
					else {
						EXPECT_NEAR(minInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				}
				else { // post-pre
					if (coba) {
						EXPECT_NEAR(maxInhWeight / 100, weights[0][0], 0.005f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
					else {
						EXPECT_NEAR(maxInhWeight, weights[0][0], 0.5f);
						//printf("mode:%d coba:%d offset:%d w:%f\n", mode, coba, offset, weights[0][0]);
					}
				}

				delete prePostSpikeGen;
				delete sim;
			}
		}
	}
}
