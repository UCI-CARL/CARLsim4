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
			sim = new CARLsim(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,true,STANDARD,alphaLTP,tauLTP,alphaLTD,tauLTD);

			// Temporarily mark out the testing code
			// Discuss whether carlsim user interface needs to spport group_info_t
			/*
			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTDP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTP,alphaLTP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTD,alphaLTD);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTP_INV,1.0/tauLTP);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTD_INV,1.0/tauLTD);
			}
			*/
			delete sim;
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

			// Temporarily mark out the testing code
			// Discuss whether carlsim interface needs to support group_int_t
			/*
			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_FALSE(grpInfo.WithSTDP);
			}
			*/
			delete sim;
		}
	}
}

TEST(STDP, setDASTDP) {
}

TEST(STDP, setNeuromodulator) {

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
