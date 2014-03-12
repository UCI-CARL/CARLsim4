#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// CONDUCTANCE-BASED MODEL (COBA) INTERFACE TESTS: Tests input from user-interface
/// **************************************************************************************************************** ///

//! Death tests for setConductances (test all possible silly values)
TEST(COBA, setCondSilly) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CpuSNN* sim = NULL;
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
	CpuSNN* sim = NULL;
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
	CpuSNN* sim = NULL;
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
	CpuSNN* sim = NULL;
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
