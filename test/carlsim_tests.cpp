// Testing COBA

//#define REGRESSION_TESTING

#include <carlsim.h>
#include <limits.h>
#include "gtest/gtest.h"


//! check all possible (valid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinit) {
	CpuSNN* sim;

	// Problem: The first two modes will print to stdout, and close it in the end; so all subsequent calls to sdout
	// via GTEST fail
	loggerMode_t modes[4] = {USER, DEVELOPER, SILENT, CUSTOM};
	for (int i=0; i<4; i++) {
		for (int j=0; j<=1; j++) {
			std::string name = "SNN";
			int nConfig = rand() % 100 + 1;
			int randSeed = rand() % 1000;
			sim = new CpuSNN(name,nConfig,randSeed,j,modes[i]);

			EXPECT_EQ(sim->networkName,name);
			EXPECT_EQ(sim->getNumConfigurations(),nConfig);
			EXPECT_EQ(sim->randSeed,randSeed);
			EXPECT_EQ(sim->getSimMode(),j);
			EXPECT_EQ(sim->getLoggerMode(),modes[i]);

			delete sim;
		}
	}

	sim = new CpuSNN("SNN",1,0);
	EXPECT_EQ(sim->randSeed,123);
	delete sim;

	// time(NULL)
	sim = new CpuSNN("SNN",1,-1);
	EXPECT_NE(sim->randSeed,-1);
	EXPECT_NE(sim->randSeed,0);
	delete sim;
}

// FIXME: enabling the following generates a segfault
//! check all possible (invalid) ways of instantiating CpuSNN
/*TEST(CORE, CpuSNNinitDeath) {
	CpuSNN* sim;

	EXPECT_DEATH({sim = new CpuSNN("SNN",-1);},"");
	if (sim!=NULL) delete sim;

	EXPECT_DEATH({sim = new CpuSNN("SNN",101);},"");
	if (sim!=NULL) delete sim;

	EXPECT_DEATH({sim = new CpuSNN("SNN",1,42,-1);},"");
	if (sim!=NULL) delete sim;

	EXPECT_DEATH({sim = new CpuSNN("SNN",1,42,2);},"");
	if (sim!=NULL) delete sim;

	EXPECT_DEATH({sim = new CpuSNN("SNN",1,42,CPU_MODE,UNKNOWN);},"");
	if (sim!=NULL) delete sim;
}*/

//! Death tests for createGroup (test all possible silly values)
TEST(CORE, createGroupSilly) {
	CpuSNN* sim;
	sim = new CpuSNN("SNN",1,42,CPU_MODE,SILENT);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({int N=-10; sim->createGroup("excit", N, EXCITATORY_NEURON, ALL);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, -3, ALL);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, EXCITATORY_NEURON, 2);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, EXCITATORY_NEURON, -2);},"");

	if (sim!=NULL)
		delete sim;
}

//! Death tests for createSpikeGenerator (test all possible silly values)
TEST(CORE, createSpikeGeneratorGroupSilly) {
	CpuSNN* sim;
	sim = new CpuSNN("SNN",1,42,CPU_MODE,SILENT);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({int N=-10; sim->createSpikeGeneratorGroup("excit", N, EXCITATORY_NEURON, ALL);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, -3, ALL);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON, 2);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON, -2);},"");

	if (sim!=NULL)
		delete sim;
}


//! Death tests for setNeuronParameters (test all possible silly values)
TEST(CORE, setNeuronParametersSilly) {
	CpuSNN* sim;
	sim = new CpuSNN("SNN",1,42,CPU_MODE,SILENT);
	int g0=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, -0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, -10.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, -0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, -10.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, -2.0f, 8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, -8.0f, 0.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, -10.0f, ALL);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, 2);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, -2);},"");

	if (sim!=NULL)
		delete sim;
}









//! Death tests for setConductances (test all possible silly values)
TEST(COBA, setCondSilly) {
	CpuSNN* sim;
	sim = new CpuSNN("SNN",1,42,CPU_MODE,SILENT);
	int g0=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps or values>numConfig, etc.
	EXPECT_DEATH({sim->setConductances(g0+1, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(-2, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, -5.0f, 150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, -150.0f, 6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, -6.0f, 150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, -150.0f, ALL);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, 150.0f, 2);},"");
	EXPECT_DEATH({sim->setConductances(g0, true, 5.0f, 150.0f, 6.0f, 150.0f, -2);},"");

	if (sim!=NULL)
		delete sim;
}

/*!
 * \brief testing setConductances to true
 * This function tests the information stored in the group info struct after calling setConductances and enabling COBA.
 */
TEST(COBA, setCondTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,SILENT);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->setConductances(grps[0],true,tAMPA,tNMDA,tGABAa,tGABAb,ALL);
			sim->setConductances(grps[1],true,tAMPA,tNMDA,tGABAa,tGABAb,ALL);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_TRUE(grpInfo.WithConductances);
					EXPECT_FLOAT_EQ(grpInfo.dAMPA,1.0-1.0/tAMPA);
					EXPECT_FLOAT_EQ(grpInfo.dNMDA,1.0-1.0/tNMDA);
					EXPECT_FLOAT_EQ(grpInfo.dGABAa,1.0-1.0/tGABAa);
					EXPECT_FLOAT_EQ(grpInfo.dGABAb,1.0-1.0/tGABAb);
				}
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setConductances to true using default values
 * This function tests the information stored in the group info struct after calling setConductances and enabling COBA.
 * Actual conductance values are set via the interface function setDefaultConductanceDecay
 */
TEST(COBA, setCondTrueDefault) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CARLsim* sim;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,0,SILENT);

			sim->setDefaultConductanceDecay(tAMPA,tNMDA,tGABAa,tGABAb);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(grps[1], 0.02f, 0.2f, -65.0f, 8.0f);

			sim->setConductances(grps[0],true);
			sim->setConductances(grps[1],true);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_TRUE(grpInfo.WithConductances);
					EXPECT_FLOAT_EQ(grpInfo.dAMPA,1.0-1.0/tAMPA);
					EXPECT_FLOAT_EQ(grpInfo.dNMDA,1.0-1.0/tNMDA);
					EXPECT_FLOAT_EQ(grpInfo.dGABAa,1.0-1.0/tGABAa);
					EXPECT_FLOAT_EQ(grpInfo.dGABAb,1.0-1.0/tGABAb);
				}
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setConductances to false
 * This function tests the information stored in the group info struct after calling setConductances and disabling COBA.
 */
TEST(COBA, setCondFalse) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim;
	group_info_t grpInfo;
	int grps[2] = {-1};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,SILENT);

			grps[0]=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excit", 10, EXCITATORY_NEURON, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

			sim->setConductances(grps[0],false,tAMPA,tNMDA,tGABAa,tGABAb, ALL);
			sim->setConductances(grps[1],false,tAMPA,tNMDA,tGABAa,tGABAb, ALL);

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<=1; g++) {
					grpInfo = sim->getGroupInfo(grps[g],c);
					EXPECT_FALSE(grpInfo.WithConductances);
				}
			}
			delete sim;
		}
	}
}

// TODO: test to trigger error that not all groups have conductances enabled












//! connect with certain mulSynFast, mulSynSlow and observe connectInfo
TEST(Connect, connect) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;

	CpuSNN* sim;
	grpConnectInfo_t* connInfo;
	std::string typeStr;

	int conn[4] 		= {-1};
	conType_t type[4] 	= {CONN_RANDOM,CONN_ONE_TO_ONE,CONN_FULL,CONN_FULL_NO_DIRECT};
	float initWt[4] 	= {0.05f, 0.1f, 0.21f, 0.42f};
	float maxWt[4] 		= {0.05f, 0.1f, 0.21f, 0.42f};
	float prob[4] 		= {0.1, 0.2, 0.3, 0.4};
	int minDelay[4] 	= {1,2,3,4};
	int maxDelay[4] 	= {1,2,3,4};
	float mulSynFast[4] = {0.2f, 0.8f, 1.2f, 0.0};
	float mulSynSlow[4] = {0.0f, 2.4f, 11.1f, 10.0f};
	int synType[4] 		= {SYN_FIXED,SYN_PLASTIC,SYN_FIXED,SYN_PLASTIC};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			for (int i=0; i<4; i++) {
				sim = new CpuSNN("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,SILENT);

				int g0=sim->createSpikeGeneratorGroup("spike", 10, EXCITATORY_NEURON, ALL);
				int g1=sim->createGroup("excit0", 10, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
				int g2=sim->createGroup("excit1", 10, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

				if (type[i]==CONN_RANDOM) typeStr = "random";
				else if (type[i]==CONN_ONE_TO_ONE) typeStr = "one-to-one";
				else if (type[i]=CONN_FULL) typeStr = "full";
				else if (type[i]=CONN_FULL_NO_DIRECT) typeStr = "full-no-direct";

				conn[i] = sim->connect(g0, g1, typeStr, initWt[i], maxWt[i], prob[i], minDelay[i], maxDelay[i], 
											mulSynFast[i], mulSynSlow[i], synType[i]);

				for (int c=0; c<nConfig; c++) {
					connInfo = sim->getConnectInfo(conn[i],c);
					EXPECT_FLOAT_EQ(connInfo->initWt,initWt[i]);
					EXPECT_FLOAT_EQ(connInfo->maxWt,maxWt[i]);
					EXPECT_FLOAT_EQ(connInfo->p,prob[i]);
					EXPECT_FLOAT_EQ(connInfo->mulSynFast,mulSynFast[i]);
					EXPECT_FLOAT_EQ(connInfo->mulSynSlow,mulSynSlow[i]);
					EXPECT_EQ(connInfo->minDelay,minDelay[i]);
					EXPECT_EQ(connInfo->maxDelay,maxDelay[i]);
					EXPECT_EQ(connInfo->type,type[i]);
					EXPECT_EQ(GET_FIXED_PLASTIC(connInfo->connProp),synType[i]);
				}
				delete sim;
			}
		}
	}
}

// TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output




// Testing STDP

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(STDP, setSTDPTrue) {
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
			sim = new CARLsim("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,0,SILENT);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,true,alphaLTP,tauLTP,alphaLTD,tauLTD);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTDP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTP,alphaLTP);
				EXPECT_FLOAT_EQ(grpInfo.ALPHA_LTD,alphaLTD);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTP_INV,1.0/tauLTP);
				EXPECT_FLOAT_EQ(grpInfo.TAU_LTD_INV,1.0/tauLTD);
			}
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
			sim = new CARLsim("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,0,SILENT);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTDP(g1,false,alphaLTP,tauLTP,alphaLTD,tauLTD);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_FALSE(grpInfo.WithSTDP);
			}
			delete sim;
		}
	}
}




// Testing STP

/*!
 * \brief testing setSTP to true
 * This function tests the information stored in the group info struct after enabling STP via setSTP
 */
TEST(STDP, setSTPTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 5.0f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",nConfig,42,mode?GPU_MODE:CPU_MODE,0,SILENT);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTP(g1,true,STP_U,STP_tD,STP_tF);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTP);
				EXPECT_FLOAT_EQ(grpInfo.STP_U,STP_U);
				EXPECT_FLOAT_EQ(grpInfo.STP_tD,STP_tD);
				EXPECT_FLOAT_EQ(grpInfo.STP_tF,STP_tF);
			}
			delete sim;
		}
	}
}