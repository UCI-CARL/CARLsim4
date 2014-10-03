#include "gtest/gtest.h"
#include <snn.h>
#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// SPIKE-TIMING-DEPENDENT PLASTICITY STDP
/// **************************************************************************************************************** ///

/*!
 * \brief controller for spike timing
 */
class QuotaSpikeController: public SpikeGenerator {
private:
	int quota;
public:
	QuotaSpikeController() {
		quota = 0;
	}

	unsigned int nextSpikeTime(CARLsim* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (quota > 0 && lastScheduledSpikeTime < currentTime + 500) {
			quota--;
			return currentTime + 500;
		}

		return 0xFFFFFFFF;
	}

	void setQuota(int nQuota) {
		quota = nQuota;
	}
};

class TwoGroupsSpikeController: public SpikeGenerator {
private:
	int isi;
	int offset;
	int g1;
	int g2;
	bool setOffset;
public:
	TwoGroupsSpikeController(int interSpikeInterval, int offsetToFirstGroup, int firstGroup, int secondGroup) {
		isi = interSpikeInterval;
		offset = offsetToFirstGroup;
		g1 = firstGroup;
		g2 = secondGroup;
		setOffset = false;
	}

	unsigned int nextSpikeTime(CARLsim* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (grpId == g1) // gin
			return lastScheduledSpikeTime + isi;
		else if (grpId == g2) { // gex
			if (!setOffset) {
				setOffset = true;
				return lastScheduledSpikeTime + isi + offset;
			} else
				return lastScheduledSpikeTime + isi;
		}

		return 0xFFFFFFFF;
	}

	void updateOffset(int newOffset) {
		offset = newOffset;
		setOffset = false;
	}
};

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(STDP, setSTDPTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	float betaLTP = 1.0f;
	float betaLTD = 1.2f;
	float lamda = 12.0f;
	float delta = 40.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			for (int stdpType = 0; stdpType < 2; stdpType++) { // we have two stdp types {STANDARD, DA_MOD}
				sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

				int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
				if (stdpType == 0) {
					sim->setESTDP(g1,true,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);
					sim->setISTDP(g1,true,DA_MOD,betaLTP,betaLTD,lamda,delta);
				} else {
					sim->setESTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);
					sim->setISTDP(g1,true,STANDARD,betaLTP,betaLTD,lamda,delta);
				}

				for (int c=0; c<nConfig; c++) {
					GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
					EXPECT_TRUE(gInfo.WithSTDP);
					EXPECT_TRUE(gInfo.WithESTDP);
					EXPECT_TRUE(gInfo.WithISTDP);
					if (stdpType == 0)
						EXPECT_TRUE(gInfo.WithESTDPtype == STANDARD);
					else
						EXPECT_TRUE(gInfo.WithESTDPtype == DA_MOD);
					if (stdpType == 0)
						EXPECT_TRUE(gInfo.WithISTDPtype == DA_MOD);
					else
						EXPECT_TRUE(gInfo.WithISTDPtype == STANDARD);
					EXPECT_FLOAT_EQ(gInfo.ALPHA_LTP_EXC,alphaLTP);
					EXPECT_FLOAT_EQ(gInfo.ALPHA_LTD_EXC,alphaLTD);
					EXPECT_FLOAT_EQ(gInfo.TAU_LTP_INV_EXC,1.0/tauLTP);
					EXPECT_FLOAT_EQ(gInfo.TAU_LTD_INV_EXC,1.0/tauLTD);
					EXPECT_FLOAT_EQ(gInfo.BETA_LTP,betaLTP);
					EXPECT_FLOAT_EQ(gInfo.BETA_LTD,betaLTD);
					EXPECT_FLOAT_EQ(gInfo.LAMDA,lamda);
					EXPECT_FLOAT_EQ(gInfo.DELTA,delta);
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
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float alphaLTP = 5.0f;		// the exact values don't matter
	float alphaLTD = 10.0f;
	float tauLTP = 15.0f;
	float tauLTD = 20.0f;
	float betaLTP = 1.0f;
	float betaLTD = 2.0f;
	float lamda = 3.0f;
	float delta = 4.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setESTDP(g1,false,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);
			sim->setISTDP(g1,false,STANDARD,betaLTP,betaLTD,lamda,delta);

			for (int c=0; c<nConfig; c++) {
				GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
				EXPECT_FALSE(gInfo.WithSTDP);
				EXPECT_FALSE(gInfo.WithESTDP);
				EXPECT_FALSE(gInfo.WithISTDP);
			}

			delete sim;
		}
	}
}

/*!
 * \brief testing setSTDPNeuromodulatorParameters
 * This function tests the information stored in the group info struct after setting neuromodulator parameters
 */
TEST(STDP, setNeuromodulatorParameters) {
	// create network by varying nConfig from 1...maxConfig, with step size nConfigStep
	std::string name="SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
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
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);
			sim->setNeuromodulator(g1, baseDP, tauDP, base5HT, tau5HT,
									baseACh, tauACh, baseNE, tauNE);

			for (int c=0; c<nConfig; c++) {
				GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
				EXPECT_TRUE(gInfo.WithSTDP);
				EXPECT_TRUE(gInfo.WithESTDPtype == DA_MOD);

				GroupNeuromodulatorInfo_t gInfo2 = sim->getGroupNeuromodulatorInfo(g1, c);
				EXPECT_FLOAT_EQ(gInfo2.baseDP, baseDP);
				EXPECT_FLOAT_EQ(gInfo2.base5HT, base5HT);
				EXPECT_FLOAT_EQ(gInfo2.baseACh, baseACh);
				EXPECT_FLOAT_EQ(gInfo2.baseNE, baseNE);
				EXPECT_FLOAT_EQ(gInfo2.decayDP, 1.0 - 1.0 / tauDP);
				EXPECT_FLOAT_EQ(gInfo2.decay5HT, 1.0 - 1.0 / tau5HT);
				EXPECT_FLOAT_EQ(gInfo2.decayACh, 1.0 - 1.0 / tauACh);
				EXPECT_FLOAT_EQ(gInfo2.decayNE, 1.0 - 1.0 / tauNE);
			}

			delete sim;
		}
	}
}

/*!
 * \brief testing the effect of dopamine modulation
 * This function tests the effect of dopamine modulation on a single synapse (reinforcement learning).
 * The the synaptic weight modulated by dopamine is expected to be higher than that without dopamine modulation
 */
TEST(STDP, DASTDPWeightBoost) {
	float tauLTP = 20.0f;
	float tauLTD = 20.0f;
	float alphaLTP = 0.1f;
	float alphaLTD = 0.122f;
	CARLsim* sim;
	int g1, gin, g1noise, gda;
	QuotaSpikeController* spikeCtrl = new QuotaSpikeController();
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
				sim = new CARLsim("SNN", mode?GPU_MODE:CPU_MODE, SILENT, 0, 1, 42);

				g1 = sim->createGroup("post-ex", 1, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gin = sim->createSpikeGeneratorGroup("pre-ex", 1, EXCITATORY_NEURON);
				g1noise = sim->createSpikeGeneratorGroup("post-ex-noise", 1, EXCITATORY_NEURON);
				gda = sim->createSpikeGeneratorGroup("DA neurons", 500, DOPAMINERGIC_NEURON);

				if (coba) {
					sim->connect(gin,g1,"one-to-one", RangeWeight(0.0, 1.0f/100, 20.0f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);
					sim->connect(g1noise, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), SYN_FIXED);
					sim->connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), SYN_FIXED);
					// enable COBA, set up STDP, enable dopamine-modulated STDP
					sim->setConductances(true,5,150,6,150);
					sim->setSTDP(g1, true, DA_MOD, alphaLTP/100, tauLTP, alphaLTD/100, tauLTD);
				} else {
					sim->connect(gin,g1,"one-to-one", RangeWeight(0.0, 1.0f, 20.0f), 1.0f, RangeDelay(1), SYN_PLASTIC);
					sim->connect(g1noise, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), SYN_FIXED);
					sim->connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), SYN_FIXED);
					// set up STDP, enable dopamine-modulated STDP
					sim->setSTDP(g1, true, DA_MOD, alphaLTP, tauLTP, alphaLTD, tauLTD);
				}

				sim->setWeightAndWeightChangeUpdate(INTERVAL_10MS, INTERVAL_10MS, 100);

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gda, spikeCtrl);

				sim->setupNetwork();

				spikeMonPost = sim->setSpikeMonitor(g1);
				spikeMonPre = sim->setSpikeMonitor(gin);
				sim->setSpikeMonitor(gda);

				//setup baseline firing rate
				PoissonRate in(1);
				in.rates[0] = 6; // 6Hz
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
								if (damod) spikeCtrl->setQuota(500);
							}

							//if (diff < 0 && diff >= -20)
							//printf("LTD\n");
						}
					}
				}

				sim->getPopWeights(gin, g1, weights, size);
				//printf("%f\n",weights[0]);
				if (damod)
					weightDAMod = weights[0];
				else
					weightNonDAMod = weights[0];

				delete sim;
			}

			EXPECT_TRUE(weightDAMod >= weightNonDAMod);
		}
	}

	delete spikeCtrl;
}

/*!
 * \brief testing the I-STDP fuction
 * This function tests whether I-STDP change synaptic weight as expected
 * Wtih control of pre- and post-neurons' spikes, the synaptic weight is expected to increase or decrease to
 * maximum or minimum synaptic weith respectively.
 */
TEST(STDP, ISTDPWeightChange) {
	// simulation details
	float* weights = NULL;
	int size;
	SpikeMonitor* spikeMon1;
	SpikeMonitor* spikeMonIn;
	SpikeMonitor* spikeMonEx;
	TwoGroupsSpikeController* spikeCtrl;
	int gin, gex, g1;
	float BETA_LTP = 0.10f;
	float BETA_LTD = 0.14f;
	float LAMDA = 9.0f;
	float DELTA = 40.0f;
	float maxInhWeight = 10.0f;
	float initWeight = 5.0f;
	float minInhWeight = 0.0f;
	CARLsim* sim;

	for (int mode = 0; mode < 2; mode++) {
		for (int coba = 0; coba < 2; coba++) {
			for (int offset = -15; offset <= 15; offset += 10) {
				//printf("mode:%d coba:%d offset:%d\n", mode, coba, offset);
				// create a network
				sim = new CARLsim("istdp", mode?GPU_MODE:CPU_MODE, SILENT, 0, 1, 42);

				g1 = sim->createGroup("excit", 1, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

				gex = sim->createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON);
				gin = sim->createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON);

				spikeCtrl = new TwoGroupsSpikeController(100, offset, gin, gex);

				if (coba) { // conductance-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight/100, maxInhWeight/100), 1.0f, RangeDelay(1), SYN_PLASTIC);

					// enable COBA, set up ISTDP
					sim->setConductances(true,5,150,6,150);
					sim->setISTDP(g1, true, STANDARD, BETA_LTP/100, BETA_LTD/100, LAMDA, DELTA);
				} else { // current-based
					sim->connect(gex, g1, "one-to-one", RangeWeight(40.0f), 1.0f, RangeDelay(1), SYN_FIXED);
					sim->connect(gin, g1, "one-to-one", RangeWeight(minInhWeight, initWeight, maxInhWeight), 1.0f, RangeDelay(1), SYN_PLASTIC);

					// set up ISTDP
					sim->setISTDP(g1, true, STANDARD, BETA_LTP, BETA_LTD, LAMDA, DELTA);
				}

				// set up spike controller on DA neurons
				sim->setSpikeGenerator(gex, spikeCtrl);
				sim->setSpikeGenerator(gin, spikeCtrl);

				// build the network
				sim->setupNetwork();

				spikeMon1 = sim->setSpikeMonitor(g1);
				spikeMonIn = sim->setSpikeMonitor(gin);
				spikeMonEx = sim->setSpikeMonitor(gex);

				// run for 10 seconds
				for (int t = 0; t < 20; t++) {
					spikeMon1->startRecording();
					spikeMonIn->startRecording();
					spikeMonEx->startRecording();
					sim->runNetwork(1,0, true, true);
					spikeMon1->stopRecording();
					spikeMonIn->stopRecording();
					spikeMonEx->stopRecording();
					//sim->getPopWeights(gin, g1, weights, size);
					//printf("%f\n",weights[0]);
				}

				sim->getPopWeights(gin, g1, weights, size);

				if (offset == -5 || offset == 5) { // I-STDP LTP
					if (coba) {
						EXPECT_NEAR(-maxInhWeight/100, weights[0], 0.005f);
					} else {
						EXPECT_NEAR(-maxInhWeight, weights[0], 0.5f);
					}
				} else { // I-STDP LTD
					if (coba) {
						EXPECT_NEAR(-minInhWeight/100, weights[0], 0.005f);
					} else {
						EXPECT_NEAR(-minInhWeight, weights[0], 0.5f);
					}
				}

				delete spikeCtrl;
				delete sim;
			}
		}
	}
}