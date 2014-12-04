#include "gtest/gtest.h"
#include "carlsim_tests.h"
#include <carlsim.h>


/// **************************************************************************************************************** ///
/// SPIKE-TIMING-DEPENDENT PLASTICITY STDP
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(STDP, setSTDPTrue) {
	std::string name="SNN";
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int stdpType = 0; stdpType < 2; stdpType++) { // we have two stdp types {STANDARD, DA_MOD}
			sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			if (stdpType == 0)
				sim->setSTDP(g1,true,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);
			else
				sim->setSTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);

			GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1);
			EXPECT_TRUE(gInfo.WithSTDP);
			if (stdpType == 0)
				EXPECT_TRUE(gInfo.WithSTDPtype == STANDARD);
			else
				EXPECT_TRUE(gInfo.WithSTDPtype == DA_MOD);
			EXPECT_FLOAT_EQ(gInfo.ALPHA_LTP,alphaLTP);
			EXPECT_FLOAT_EQ(gInfo.ALPHA_LTD,alphaLTD);
			EXPECT_FLOAT_EQ(gInfo.TAU_LTP_INV,1.0/tauLTP);
			EXPECT_FLOAT_EQ(gInfo.TAU_LTD_INV,1.0/tauLTD);

			delete sim;
		}
	}
}

/*!
 * \brief testing setSTDP to false
 * This function tests the information stored in the group info struct after disabling STDP via setSTDP
 */
TEST(STDP, setSTDPFalse) {
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,42);

		int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setSTDP(g1,false,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);

		GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1);
		EXPECT_FALSE(gInfo.WithSTDP);

		delete sim;
	}
}

TEST(STDP, setNeuromodulatorParameters) {
	std::string name="SNN";
	float alphaLTP = 1.0f;		// the exact values don't matter
	float alphaLTD = 1.2f;
	float tauLTP = 20.0f;
	float tauLTD = 20.0f;
	float baseDP = 1.0f;
	float base5HT = 2.0f;
	float baseACh = 3.0f;
	float baseNE = 4.0f;
	float tauDP = 100.0f;
	float tau5HT = 200.0f;
	float tauACh = 300.0f;
	float tauNE = 400.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,42);

		int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setSTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);
		sim->setNeuromodulator(g1, baseDP, tauDP, base5HT, tau5HT,
								baseACh, tauACh, baseNE, tauNE);

		// Temporarily mark out the testing code
		// Discuss whether carlsim user interface needs to spport group_info_t
		GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1);
		EXPECT_TRUE(gInfo.WithSTDP);
		EXPECT_TRUE(gInfo.WithSTDPtype == DA_MOD);

		GroupNeuromodulatorInfo_t gInfo2 = sim->getGroupNeuromodulatorInfo(g1);
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

TEST(STDP, DASTDPweightBoost) {
	float tauLTP = 20.0f;
	float tauLTD = 20.0f;
	float alphaLTP = 0.1f;
	float alphaLTD = 0.122f;
	CARLsim* sim;
	int g1, gin, g1noise, gda;
	InteractiveSpikeGenerator* iSpikeGen = new InteractiveSpikeGenerator(500, 500);
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights;
	int size;
	SpikeMonitor* spikeMonPost;
	SpikeMonitor* spikeMonPre;
	float weightDAMod, weightNonDAMod;

	for (int mode = 0; mode < 2; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int damod = 0; damod < 2; damod++) {
				sim = new CARLsim("SNN", mode?GPU_MODE:CPU_MODE, SILENT, 0, 42);

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
					sim->setSTDP(g1, true, DA_MOD, alphaLTP/100, tauLTP, alphaLTD/100, tauLTD);
				} else {
					sim->connect(gin,g1,"one-to-one", RangeWeight(0.0, 1.0f, 20.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
					sim->connect(g1noise, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					sim->connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
					// set up STDP, enable dopamine-modulated STDP
					sim->setSTDP(g1, true, DA_MOD, alphaLTP, tauLTP, alphaLTD, tauLTD);
					sim->setConductances(false);
				}

				sim->setWeightAndWeightChangeUpdate(INTERVAL_10MS, true, 0.99f);

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gda, iSpikeGen);

				sim->setupNetwork();

				spikeMonPost = sim->setSpikeMonitor(g1,"NULL");
				spikeMonPre = sim->setSpikeMonitor(gin,"NULL");
				sim->setSpikeMonitor(gda);

				//setup baseline firing rate
				PoissonRate in(1);
				in.setRates(6.0f); // 6Hz
				sim->setSpikeRate(gin, &in);
				sim->setSpikeRate(g1noise, &in);

				for (int t = 0; t < 200; t++) {
					spikeMonPost->startRecording();
					spikeMonPre->startRecording();
					sim->runNetwork(1, 0, false, false);
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
								if (damod) iSpikeGen->setQuotaAll(1);
							}

							//if (diff < 0 && diff >= -20)
							//	printf("LTD\n");
						}
					}
				}

				sim->getPopWeights(gin, g1, weights, size);
				//printf("%d %d %d %f\n", mode, coba, damod, weights[0]);
				if (damod)
					weightDAMod = weights[0];
				else
					weightNonDAMod = weights[0];

				delete sim;
			}

			EXPECT_TRUE(weightDAMod >= weightNonDAMod);
		}
	}

	delete iSpikeGen;
}