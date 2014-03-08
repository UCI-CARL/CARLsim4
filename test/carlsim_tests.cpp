// Testing COBA

// Don't forget to set REGRESSION_TESTING flag to 1 in config.h 

#include <carlsim.h>
#include <limits.h>
#include "gtest/gtest.h"

//! a periodic spike generator (constant ISI) creating spikes at a certain rate
class PeriodicSpikeGenerator : public SpikeGenerator {
public:
	PeriodicSpikeGenerator(float rate) {
		assert(rate>0);
		rate_ = rate;	  // spike rate
		isi_ = 1000/rate; // inter-spike interval in ms
	}

	unsigned int nextSpikeTime(CpuSNN* snn, int grpId, int nid, unsigned int currentTime) {
		return currentTime+isi_; // periodic spiking according to ISI
	}

private:
	float rate_;	// spike rate
	int isi_;		// inter-spike interval that results in above spike rate
};

/*!
 * \brief check whether CPU and GPU mode return the same stpu and stpx
 * This test creates a STP connection with random parameter values, runs a simulation
 * for 300 ms, and checks whether CPU_MODE and GPU_MODE return the same internal
 * variables stpu and stpx. Input is periodic 20 Hz spiking.
 */
TEST(STP, testCPUvsGPU) {
	CpuSNN* sim;
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};

	float stpu[600] = {0.0f};
	float stpx[600] = {0.0f};

	int nConfig = 1;
	int randSeed = rand() % 1000;
	float STP_U = (float) rand()/RAND_MAX;
	int STP_tD = rand() % 100;
	int STP_tF = rand() % 500 + 500;

	for (int j=0; j<2; j++) {
		sim = new CpuSNN(name,simModes[j],USER,0,nConfig,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(ALL,true,5.0f,10.0f,15.0f,20.0f,ALL);
		sim->setSTP(g0,true,STP_U,STP_tD,STP_tF,ALL);

		PeriodicSpikeGenerator* spk20 = new PeriodicSpikeGenerator(20.0f);
		sim->setSpikeGenerator(g0, spk20, ALL);

		for (int i=0; i<300; i++) {
			sim->runNetwork(0,1,false,true); // enable copyState
			stpu[j*300+i] = sim->stpu[1];
			stpx[j*300+i] = sim->stpx[1];
		}

		delete spk20;
		delete sim;
	}

	// compare stpu and stpx for both sim modes
	for (int i=0; i<300; i++) {
		EXPECT_FLOAT_EQ(stpu[i],stpu[i+300]);
		EXPECT_FLOAT_EQ(stpx[i],stpx[i+300]);
	}

	// check init default values
	EXPECT_FLOAT_EQ(stpu[0],0.0f);
	EXPECT_FLOAT_EQ(stpx[0],1.0f);
	EXPECT_FLOAT_EQ(stpu[300],0.0f);
	EXPECT_FLOAT_EQ(stpx[300],1.0f);
}


//! check all possible (valid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinit) {
	CpuSNN* sim;

	// Problem: The first two modes will print to stdout, and close it in the end; so all subsequent calls to sdout
	// via GTEST fail
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};
	loggerMode_t loggerModes[4] = {USER, DEVELOPER, SILENT, CUSTOM};
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
TEST(CORE, createSpikeGeneratorGroupSilly) {
	CpuSNN* sim;
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
TEST(CORE, setNeuronParametersSilly) {
	CpuSNN* sim;
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

	CpuSNN* sim;
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

// TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output




// Testing STDP

/*!
 * \brief testing setSTDP to true
 * This function tests the information stored in the group info struct after enabling STDP via setSTDP
 */
TEST(CORE, setSTDPTrue) {
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
TEST(CORE, setSTDPFalse) {
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
TEST(CORE, setSTPTrue) {
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
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTP(g1,true,STP_U,STP_tD,STP_tF);

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTP);
				EXPECT_FLOAT_EQ(grpInfo.STP_U,STP_U);
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_u_inv,1.0f/STP_tF);
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_x_inv,1.0f/STP_tD);
			}
			delete sim;
		}
	}
}






//! Death tests for setConductances (test all possible silly values)
TEST(COBA, setCondSilly) {
	CpuSNN* sim;
	std::string name="SNN";
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,42);
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
	std::string name="SNN";
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
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

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
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

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
	std::string name="SNN";
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
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

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


TEST(COBA, disableSynReceptors) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = 1; //rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim;
	group_info_t grpInfo;
	int grps[4] = {-1};

	int expectSpkCnt[4] = {200, 160, 0, 0};
	int expectSpkCntStd = 10;

	std::string expectCond[4] = {"AMPA","NMDA","GABAa","GABAb"};
	float expectCondVal[4] = {0.14, 2.2, 0.17, 2.2};
	float expectCondStd[4] = {0.025,0.2,0.025,0.2,};

	int nInput = 1000;
	int nOutput = 10;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g0=sim->createSpikeGeneratorGroup("spike", nInput, EXCITATORY_NEURON, ALL);
			int g1=sim->createSpikeGeneratorGroup("spike", nInput, INHIBITORY_NEURON, ALL);
			grps[0]=sim->createGroup("excitAMPA", nOutput, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excitNMDA", nOutput, EXCITATORY_NEURON, ALL);
			grps[2]=sim->createGroup("inhibGABAa", nOutput, INHIBITORY_NEURON, ALL);
			grps[3]=sim->createGroup("inhibGABAb", nOutput, INHIBITORY_NEURON, ALL);

			sim->setNeuronParameters(grps[0], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[2], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[3], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);

			sim->setConductances(ALL, true, 5.0f, 150.0f, 6.0f, 150.0f, ALL);

			sim->connect(g0, grps[0], "full", 0.001f, 0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g0, grps[1], "full", 0.0005f, 0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);
			sim->connect(g1, grps[2], "full", -0.001f, -0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g1, grps[3], "full", -0.0005f, -0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);

			PoissonRate poissIn1(nInput);
			PoissonRate poissIn2(nInput);
			for (int i=0; i<nInput; i++) {
				poissIn1.rates[i] = 30.0f;
				poissIn2.rates[i] = 30.0f;
			}
			sim->setSpikeRate(g0,&poissIn1,1,ALL);
			sim->setSpikeRate(g1,&poissIn2,1,ALL);

			sim->runNetwork(1,0,false,false);

			if (mode) {
				// GPU_MODE: copy from device to host
				for (int g=0; g<4; g++)
					sim->copyNeuronState(&(sim->cpuNetPtrs), &(sim->cpu_gpuNetPtrs), cudaMemcpyDeviceToHost, false, grps[g]);
			}

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<4; g++) { // all groups
					grpInfo = sim->getGroupInfo(grps[g],c);

					EXPECT_TRUE(grpInfo.WithConductances);
					for (int n=grpInfo.StartN; n<=grpInfo.EndN; n++) {
//						printf("%d[%d]: AMPA=%f, NMDA=%f, GABAa=%f, GABAb=%f\n",g,n,sim->gAMPA[n],sim->gNMDA[n],sim->gGABAa[n],sim->gGABAb[n]);
						if (expectCond[g]=="AMPA") {
							EXPECT_GT(sim->gAMPA[n],0.0f);
							EXPECT_NEAR(sim->gAMPA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gAMPA[n],0.0f);

						if (expectCond[g]=="NMDA") {
							EXPECT_GT(sim->gNMDA[n],0.0f);
							EXPECT_NEAR(sim->gNMDA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gNMDA[n],0.0f);

						if (expectCond[g]=="GABAa") {
							EXPECT_GT(sim->gGABAa[n],0.0f);
							EXPECT_NEAR(sim->gGABAa[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAa[n],0.0f);

						if (expectCond[g]=="GABAb") {
							EXPECT_GT(sim->gGABAb[n],0.0f);
							EXPECT_NEAR(sim->gGABAb[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAb[n],0.0f);
					}
				}
			}
			delete sim;
		}
	}	
}