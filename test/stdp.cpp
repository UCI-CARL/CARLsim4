#include <snn.h>

#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// SPIKE-TIMING-DEPENDENT PLASTICITY STDP
/// **************************************************************************************************************** ///

// FIXME: this is missing dopamine-modulated STDP

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
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			for (int stdpType = 0; stdpType < 2; stdpType++) { // we have two stdp types {STANDARD, DA_MOD}
				sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

				int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
				if (stdpType == 0)
					sim->setSTDP(g1,true,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);
				else
					sim->setSTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);

				for (int c=0; c<nConfig; c++) {
					GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
					EXPECT_TRUE(gInfo.WithSTDP);
					if (stdpType == 0)
						EXPECT_TRUE(gInfo.WithSTDPtype == STANDARD);
					else
						EXPECT_TRUE(gInfo.WithSTDPtype == DA_MOD);
					EXPECT_FLOAT_EQ(gInfo.ALPHA_LTP,alphaLTP);
					EXPECT_FLOAT_EQ(gInfo.ALPHA_LTD,alphaLTD);
					EXPECT_FLOAT_EQ(gInfo.TAU_LTP_INV,1.0/tauLTP);
					EXPECT_FLOAT_EQ(gInfo.TAU_LTD_INV,1.0/tauLTD);
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
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,false,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);

			for (int c=0; c<nConfig; c++) {
				GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
				EXPECT_FALSE(gInfo.WithSTDP);
			}

			delete sim;
		}
	}
}

TEST(STDP, setNeuromodulator) {
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

			// Temporarily mark out the testing code
			// Discuss whether carlsim user interface needs to spport group_info_t
			for (int c=0; c<nConfig; c++) {
				GroupSTDPInfo_t gInfo = sim->getGroupSTDPInfo(g1,c);
				EXPECT_TRUE(gInfo.WithSTDP);
				EXPECT_TRUE(gInfo.WithSTDPtype == DA_MOD);

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

TEST(STDP, dastdpSelectivity) {
	std::string name="SNN";
	float alphaLTP = 0.1f/100;		// the exact values don't matter
	float alphaLTD = 0.12f/100;
	float tauLTP = 20.0f;
	float tauLTD = 20.0f;
	CARLsim* sim;

	for (int mode = 0; mode < 2; mode++) {
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);

		int g1=sim->createGroup("pre-excit", 10, EXCITATORY_NEURON);
		int g2=sim->createGroup("post-excit", 10, EXCITATORY_NEURON);

		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setSTDP(g1,true,DA_MOD,alphaLTP,tauLTP,alphaLTD,tauLTD);

		sim->connect(g1, g2, "full", RangeWeight(0.0, 0.25f/100, 0.5f/100), 1.0f, RangeDelay(1, 20), SYN_PLASTIC);

		sim->setupNetwork();

		for (int t = 0; t < 10; t++)
			sim->runNetwork(1,0);

		delete sim;
	}
}
