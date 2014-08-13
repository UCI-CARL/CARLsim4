#include <snn.h>

#include "carlsim_tests.h"
#include <vector>

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
	for (int i=0; i<5; i++) {
		for (int j=0; j<2; j++) {
			int nConfig = rand() % 100 + 1;
			int randSeed = rand() % 1000;
			sim = new CpuSNN(name,simModes[j],loggerModes[i],0,nConfig,randSeed);

			EXPECT_EQ(sim->getNetworkName(),name);
			EXPECT_EQ(sim->getNumConfigurations(),nConfig);
			EXPECT_EQ(sim->getRandSeed(),randSeed);
			EXPECT_EQ(sim->getSimMode(),simModes[j]);
			EXPECT_EQ(sim->getLoggerMode(),loggerModes[i]);

			delete sim;
		}
	}

	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,0);
	EXPECT_EQ(sim->getRandSeed(),123);
	delete sim;

	// time(NULL)
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,-1);
	EXPECT_NE(sim->getRandSeed(),-1);
	EXPECT_NE(sim->getRandSeed(),0);
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

                Grid3D neur(10,1,1);
				int g0=sim->createSpikeGeneratorGroup("spike", neur, EXCITATORY_NEURON, ALL);
				int g1=sim->createGroup("excit0", neur, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
				int g2=sim->createGroup("excit1", neur, EXCITATORY_NEURON, ALL);
				sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

				if (type[i]==CONN_RANDOM) typeStr = "random";
				else if (type[i]==CONN_ONE_TO_ONE) typeStr = "one-to-one";
				else if (type[i]=CONN_FULL) typeStr = "full";
				else if (type[i]=CONN_FULL_NO_DIRECT) typeStr = "full-no-direct";

				conn[i] = sim->connect(g0, g1, typeStr, initWt[i], maxWt[i], prob[i],
					minDelay[i],maxDelay[i], mulSynFast[i], mulSynSlow[i], synType[i]);

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

// This test creates a group on a grid and makes sure that the returned 3D location of each neuron is correct
TEST(CORE, getNeuronLocation3D) {
	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,USER,0,1,42);
	Grid3D grid(2,3,4);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for getNeuronLocation3D to work

	// make sure the 3D location that is returned is correct
	for (int grp=0; grp<=1; grp++) {
		// do for both spike gen and RS group

		int x=0,y=0,z=0;
		for (int neurId=grp*grid.N; neurId<(grp+1)*grid.N; neurId++) {
			Point3D loc = sim->getNeuronLocation3D(neurId);
			EXPECT_EQ(loc.x, x);
			EXPECT_EQ(loc.y, y);
			EXPECT_EQ(loc.z, z);

			x++;
			if (x==grid.x) {
				x=0;
				y++;
			}
			if (y==grid.y) {
				x=0;
				y=0;
				z++;
			}
		}
	}

	delete sim;
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
			EXPECT_TRUE(sim->isSimulationWithCOBA());
			EXPECT_FALSE(sim->isSimulationWithCUBA());
//			EXPECT_FLOAT_EQ(sim->dAMPA,1.0f-1.0f/tdAMPA);
			if (trNMDA) {
				EXPECT_TRUE(sim->isSimulationWithNMDARise());
//				EXPECT_FLOAT_EQ(sim->rNMDA,1.0f-1.0f/trNMDA);
			} else {
				EXPECT_FALSE(sim->isSimulationWithNMDARise());
			}
//			EXPECT_FLOAT_EQ(sim->dNMDA,1.0f-1.0f/tdNMDA);
//			EXPECT_FLOAT_EQ(sim->dGABAa,1.0f-1.0f/tdGABAa);
			if (trGABAb) {
				EXPECT_TRUE(sim->isSimulationWithGABAbRise());
//				EXPECT_FLOAT_EQ(sim->rGABAb,1.0f-1.0f/trGABAb);
			} else {
				EXPECT_FALSE(sim->isSimulationWithGABAbRise());
			}
//			EXPECT_FLOAT_EQ(sim->dGABAb,1.0f-1.0f/tdGABAb);

			delete sim;
		}
	}
}

// \TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// \TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// \TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// \TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output

//! test all possible valid ways of setting conductances to true
// \FIXME: this could be interface level, but then there would be no way to test net_Info struct


TEST(CORE, firingRateCPUvsGPU) {
	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG0 = NULL, *spkMonG1 = NULL, *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int hasCOBA=0; hasCOBA<=0; hasCOBA++) {
		float rateG0CPU = -1.0f;
		float rateG1CPU = -1.0f;
		float rateG2CPU = -1.0f;
		float rateG3CPU = -1.0f;
		std::vector<std::vector<int> > spkTimesG0, spkTimesG1, spkTimesG2, spkTimesG3;

		int runTimeMs = 1000;//rand() % 9500 + 500;
		float wt = hasCOBA ? 0.15f : 15.0f;

PoissonRate in(1);

		for (int isGPUmode=0; isGPUmode<=0; isGPUmode++) {
			CARLsim* sim = new CARLsim("SNN",isGPUmode?GPU_MODE:CPU_MODE,SILENT,0,1,42);
			int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
			int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);
			int g2=sim->createGroup("excit2", 1, EXCITATORY_NEURON);
			int g3=sim->createGroup("excit3", 1, EXCITATORY_NEURON);
			sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);

			sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
			sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

			if (hasCOBA)
				sim->setConductances(true,5, 0, 150, 6, 0, 150);

			bool spikeAtZero = true;
			spkGenG0 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking
			sim->setSpikeGenerator(g0, spkGenG0);
			spkGenG1 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking
			sim->setSpikeGenerator(g1, spkGenG1);

			sim->setupNetwork();

//	for (int i=0;i<1;i++) in.rates[i] = 15;
//		sim->setSpikeRate(g0,&in);
//		sim->setSpikeRate(g1,&in);

			spkMonG0 = sim->setSpikeMonitor(g0,"NULL");
			spkMonG1 = sim->setSpikeMonitor(g1,"NULL");
			spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
			spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

			spkMonG0->startRecording();
			spkMonG1->startRecording();
			spkMonG2->startRecording();
			spkMonG3->startRecording();
			sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
			spkMonG0->stopRecording();
			spkMonG1->stopRecording();
			spkMonG2->stopRecording();
			spkMonG3->stopRecording();

			if (!isGPUmode) {
				// CPU mode: record rates, so that we can compare them with GPU mode
				rateG0CPU = spkMonG0->getPopMeanFiringRate();
				rateG1CPU = spkMonG1->getPopMeanFiringRate();
				rateG2CPU = spkMonG2->getPopMeanFiringRate();
				rateG3CPU = spkMonG3->getPopMeanFiringRate();
				spkTimesG0 = spkMonG0->getSpikeVector2D();
				spkTimesG1 = spkMonG1->getSpikeVector2D();
				spkTimesG2 = spkMonG2->getSpikeVector2D();
				spkTimesG3 = spkMonG3->getSpikeVector2D();
//				for (int i=0; i<spkTimesG2[0].size(); i++)
//					fprintf(stderr, "%d\n",spkTimesG2[0][i]);
			} else {
				// GPU mode: compare rates to CPU mode
				ASSERT_FLOAT_EQ( spkMonG0->getPopMeanFiringRate(), rateG0CPU);
				ASSERT_FLOAT_EQ( spkMonG1->getPopMeanFiringRate(), rateG1CPU);
				ASSERT_FLOAT_EQ( spkMonG2->getPopMeanFiringRate(), rateG2CPU);
				ASSERT_FLOAT_EQ( spkMonG3->getPopMeanFiringRate(), rateG3CPU);

				std::vector<std::vector<int> > spkT = spkMonG2->getSpikeVector2D();
				ASSERT_EQ(spkTimesG2[0].size(), spkT[0].size());
				for (int i=0; i<spkTimesG2[0].size(); i++)
					EXPECT_EQ(spkTimesG2[0][i], spkT[0][i]);
//					fprintf(stderr, "%d\t%d\n",(i<spkTimesG2[0].size())?spkTimesG2[0][i]:-1, (i<spkT[0].size())?spkT[0][i]:-1);
			}

//			delete spkGenG0;
//			delete spkGenG1;
			delete sim;
		}
	}
}
