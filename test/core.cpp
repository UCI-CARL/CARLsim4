#include <snn.h>

#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// CORE FUNCTIONALITY
/// **************************************************************************************************************** ///

//! check all possible (valid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinit) {
	CpuSNN* sim = NULL;

	// Problem: The first two modes will print to stdout, and close it in the end; so all subsequent calls to sdout
	// via GTEST fail
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};
	loggerMode_t loggerModes[5] = {USER, DEVELOPER, SHOWTIME, SILENT, CUSTOM};
	for (int i=0; i<4; i++) {
		for (int j=0; j<2; j++) {
			int nConfig = rand() % 100 + 1;
			int randSeed = rand() % 1000;
			sim = new CpuSNN(name,simModes[j],loggerModes[i],0,nConfig,randSeed);

			EXPECT_EQ(sim->networkName_,name);
			EXPECT_EQ(sim->getNumConfigurations(),nConfig);
			EXPECT_EQ(sim->randSeed_,randSeed);
			EXPECT_EQ(sim->getSimMode(),simModes[j]);
			EXPECT_EQ(sim->getLoggerMode(),loggerModes[i]);

			delete sim;
		}
	}

	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,0);
	EXPECT_EQ(sim->randSeed_,123);
	delete sim;

	// time(NULL)
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,-1);
	EXPECT_NE(sim->randSeed_,-1);
	EXPECT_NE(sim->randSeed_,0);
	delete sim;
}

// FIXME: enabling the following generates a segfault
//! check all possible (invalid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinitDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CpuSNN* sim = NULL;
	std::string name="SNN";

	// sim mode
	EXPECT_DEATH({sim = new CpuSNN(name,UNKNOWN_SIM,USER,0,1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// logger mode
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,UNKNOWN_LOGGER,0,1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// ithGPU
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,USER,-1,1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// nConfig
	EXPECT_DEATH({sim = new CpuSNN(name,GPU_MODE,USER,0,0,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,USER,0,101,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
}

//! Death tests for createGroup (test all possible silly values)
TEST(CORE, createGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);

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
TEST(CORE, createSpikeGeneratorGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);

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
TEST(CORE, setNeuronParametersDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CpuSNN* sim = NULL;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);
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


//! connect with certain mulSynFast, mulSynSlow and observe connectInfo
TEST(CORE, connect) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;

	CpuSNN* sim = NULL;
	grpConnectInfo_t* connInfo;
	std::string typeStr;
	std::string name="SNN";

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
				sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

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

TEST(CORE, setConductancesTrue) {
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			int tdAMPA  = rand()%100 + 1;
			int trNMDA  = (nConfig==1) ? 0 : rand()%100 + 1;
			int tdNMDA  = rand()%100 + trNMDA + 1; // make sure it's larger than trNMDA
			int tdGABAa = rand()%100 + 1;
			int trGABAb = (nConfig==nConfigStep+1) ? 0 : rand()%100 + 1;
			int tdGABAb = rand()%100 + trGABAb + 1; // make sure it's larger than trGABAb

			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);
			sim->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb,ALL);
			EXPECT_TRUE(sim->sim_with_conductances);
			EXPECT_FLOAT_EQ(sim->dAMPA,1.0f-1.0f/tdAMPA);
			if (trNMDA) {
				EXPECT_TRUE(sim->sim_with_NMDA_rise);
				EXPECT_FLOAT_EQ(sim->rNMDA,1.0f-1.0f/trNMDA);
			} else {
				EXPECT_FALSE(sim->sim_with_NMDA_rise);
			}
			EXPECT_FLOAT_EQ(sim->dNMDA,1.0f-1.0f/tdNMDA);
			EXPECT_FLOAT_EQ(sim->dGABAa,1.0f-1.0f/tdGABAa);
			if (trGABAb) {
				EXPECT_TRUE(sim->sim_with_GABAb_rise);
				EXPECT_FLOAT_EQ(sim->rGABAb,1.0f-1.0f/trGABAb);
			} else {
				EXPECT_FALSE(sim->sim_with_GABAb_rise);
			}
			EXPECT_FLOAT_EQ(sim->dGABAb,1.0f-1.0f/tdGABAb);

			delete sim;
		}
	}
}

// TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output

//! test all possible valid ways of setting conductances to true
// FIXME: this could be interface level, but then there would be no way to test net_Info struct
