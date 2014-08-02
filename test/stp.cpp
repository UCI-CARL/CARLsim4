#include <snn.h>

#include "carlsim_tests.h"

/// **************************************************************************************************************** ///
/// SHORT-TERM PLASTICITY STP
/// **************************************************************************************************************** ///

/*!
 * \brief testing setSTP to true
 *
 * This function tests the information stored in the group info struct after enabling STP via setSTP
 * \TODO use public user interface
 */
TEST(STP, setSTPTrue) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON,ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setSTP(g1,true,STP_U,STP_tF,STP_tD,ALL);					// exact values matter

			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_TRUE(grpInfo.WithSTP); 							// STP must be enabled
				EXPECT_FLOAT_EQ(grpInfo.STP_A,1.0f/STP_U);
				EXPECT_FLOAT_EQ(grpInfo.STP_U,STP_U); 					// check exact values
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_u_inv,1.0f/STP_tF);
				EXPECT_FLOAT_EQ(grpInfo.STP_tau_x_inv,1.0f/STP_tD);
			}
			delete sim;
		}
	}
}

/*!
 * \brief testing setSTP to false
 * This function tests the information stored in the group info struct after disabling STP via setSTP
 */
TEST(STP, setSTPFalse) {
	int maxConfig = rand()%10 + 10;	// create network by varying nConfig from 1..maxConfig, with some
	int nConfigStep = rand()%3 + 2; // step size nConfigStep
	CARLsim* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setSTP(g1,false,0.1f,100,200); 					// exact values don't matter

			// Temporarily mark out the testing code
			// Discuss whether carlsim interface needs to support group_int_t
			/*
			for (int c=0; c<nConfig; c++) {
				group_info_t grpInfo = sim->getGroupInfo(g1,c);
				EXPECT_FALSE(grpInfo.WithSTP);						// STP must be disabled
			}
			*/
			delete sim;
		}
	}
}


//! expect CARLsim to die if setSTP is called with silly params
TEST(STP, setSTPdeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);

	// grpId
	EXPECT_DEATH({sim->setSTP(-2,true,0.1f,10,10,ALL);},"");

	// STP_U
	EXPECT_DEATH({sim->setSTP(g1,true,0.0f,10,10,ALL);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,1.1f,10,10,ALL);},"");

	// STP_tF / STP_tD
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,-10,10,ALL);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,-10,ALL);},"");

	// configId
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,-2);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,2);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,10,101);},"");
}

//! test the effect of short-term depression
TEST(STP, spikeRateSTDSTF) {
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP

	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int isRunLong=1; isRunLong<=1; isRunLong++) {
		for (int hasCOBA=1; hasCOBA<=1; hasCOBA++) {
			for (int isGPUmode=1; isGPUmode<=1; isGPUmode++) {
			// compare 
				float rateG2noSTP = -1.0f;
				float rateG3noSTP = -1.0f;

				for (int hasSTP=0; hasSTP<=1; hasSTP++) {
					CARLsim* sim = new CARLsim("SNN",isGPUmode?GPU_MODE:CPU_MODE,USER,0,1,randSeed);
					int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
					int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);
					int g2=sim->createGroup("STD", 1, EXCITATORY_NEURON);
					int g3=sim->createGroup("STF", 1, EXCITATORY_NEURON);
					sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
					sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);

					float wt = hasCOBA ? 0.1f : 1.0f;
					sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
					sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

					if (hasCOBA)
						sim->setConductances(true,5, 0, 150, 6, 0, 150);

					if (hasSTP) {
						sim->setSTP(g0, true, 0.45f, 50.0f,   750.0f); // depressive
						sim->setSTP(g1, true, 0.15f, 750.0f, 50.0f); // facilitative
					}

					bool spikeAtZero = true;
					spkGenG0 = new PeriodicSpikeGenerator(5.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g0, spkGenG0);
					spkGenG1 = new PeriodicSpikeGenerator(5.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g1, spkGenG1);

					sim->setupNetwork();

					sim->setSpikeMonitor(g0,"NULL");
					sim->setSpikeMonitor(g1,"NULL");
					spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
					spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

					spkMonG2->startRecording();
					spkMonG3->startRecording();
					int runTimeMs = isRunLong ? 2000 : 10;
					sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
					spkMonG2->stopRecording();
					spkMonG3->stopRecording();

					if (!hasSTP) {
						// if STP is off: record spike rate, so that afterwards we can compare it to the one with STP
						// enabled
						rateG2noSTP = spkMonG2->getPopMeanFiringRate();
						rateG3noSTP = spkMonG3->getPopMeanFiringRate();
					} else {
						fprintf(stderr,"%s %s, G2 w/o=%f, G2 w/=%f\n",isGPUmode?"GPU":"CPU",hasCOBA?"COBA":"CUBA", rateG2noSTP, spkMonG2->getPopMeanFiringRate());
						fprintf(stderr,"%s %s, G3 w/o=%f, G3 w/=%f\n",isGPUmode?"GPU":"CPU",hasCOBA?"COBA":"CUBA", rateG3noSTP, spkMonG3->getPopMeanFiringRate());
						// if STP is on: compare spike rate to the one recorded without STP

						if (isRunLong) {
							// the run time was relatively long, so STP should have its expected effect
							EXPECT_TRUE( spkMonG2->getPopMeanFiringRate() < rateG2noSTP); // depressive
							EXPECT_TRUE( spkMonG3->getPopMeanFiringRate() > rateG3noSTP); // facilitative
						} else {
							// the run time was really short, so STP should have no effect (because we scale STP_A so
							// that STP has no weakening/strengthening effect on the first spike)
							EXPECT_FLOAT_EQ( spkMonG2->getPopMeanFiringRate(), rateG2noSTP); // equivalent
							EXPECT_FLOAT_EQ( spkMonG3->getPopMeanFiringRate(), rateG3noSTP); // equivalent							
						}
					}

					delete spkGenG0;
					delete spkGenG1;
					delete sim;
				}
			}
		}
	}
}

/*
 * \brief check whether CPU mode reproduces some pre-recorded stpu and stpx (internal variables)
 *
 * This test ensures that CARLsim quantitatively reproduces STP behavior across machines. This exact network has been
 * run before, and STP variables (stpu and stpx) have been recorded for a time window of 50 ms. The network should
 * reproduce these values at all times and on all platforms.
 * If this test fails, then the low-level behavior of STP has been changed.
 * There is a separate test that observes the impact of stpu and stpx on the post-synaptic current.
 * There is a separate test that makes sure CPU mode yields the same result as GPU mode.
 */
/*TEST(STP, internalCPUvsData) {
	// run network, compare to these pre-recorded values
	float stpu[50] = {	0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
						0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.1900,0.1878,0.1856,0.1834,0.1813,0.1792,
						0.1771,0.1751,0.1730,0.1710,0.1690,0.1671,0.1651,0.1632,0.1613,0.1594,0.1576,0.1557,0.1539,
						0.1521,0.3118,0.3082,0.3046,0.3010,0.2975,0.2941,0.2907,0.2873,0.2839,0.2806 };
	float stpx[50] = {	1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
						1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,0.8100,0.8102,0.8104,0.8106,0.8108,0.8110,
						0.8111,0.8113,0.8115,0.8117,0.8119,0.8121,0.8123,0.8125,0.8127,0.8129,0.8130,0.8132,0.8134,
						0.8136,0.5601,0.5605,0.5609,0.5614,0.5618,0.5623,0.5627,0.5631,0.5636,0.5640 };

	std::string name = "SNN";
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP
	float abs_error = 1e-2f;		// allowed error margin

	CpuSNN* sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,randSeed);
	int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
	int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
	sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
	sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
	sim->setConductances(true,5,10,15,20,25,30,ALL);
	sim->setSTP(g0,true,0.19f,86,992,ALL); // the exact values are not important

	bool spikeAtZero = false;
	PeriodicSpikeGeneratorCore* spk50 = new PeriodicSpikeGeneratorCore(50.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim->setSpikeGenerator(g0, spk50, ALL);

	sim->setupNetwork(true);
	for (int i=0; i<50; i++) {
		sim->runNetwork(0,1,false,true); // enable copyState
		EXPECT_NEAR(sim->stpu[1], stpu[i], abs_error);
		EXPECT_NEAR(sim->stpx[1], stpx[i], abs_error);
	}

	delete spk50;
	delete sim;
}*/

/*
 * \brief check whether CPU mode reproduces some pre-recorded post-synaptic current (external, behavior)
 * This test ensures that CARLsim quantitatively reproduces STP behavior across machines. This exact network has been
 * run before, and post-synaptic currents (affected by pre-synaptic STP) have been recorded for a time window of 50 ms.
 * The network should reproduce these values at all times and on all platforms.
 * If this test fails, then the low-level behavior or protocol of STP (the order in which STP variables are updated and
 * current changes are computed) has been changed.
 * There is a separate test that observes the internal STP variables stpu and stpx.
 * There is a separate test that makes sure CPU mode yields the same result as GPU mode.
 */
/*TEST(STP, externalCPUvsData) {
	// run network, compare to these pre-recorded values of post-synaptic current
	float current[50] = {	0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,
							0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.6182,0.4954,0.3994,0.3231,0.2622,0.2134,
							0.1742,0.1429,0.1177,0.0975,0.0812,0.0681,0.0575,0.0490,0.0421,0.0364,0.0318,0.0280,0.0249,
							0.0223,0.8409,0.6767,0.5483,0.4461,0.3643,0.2986,0.2458,0.2032,0.1690,0.1413};

	std::string name = "SNN";
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP
	float abs_error = 5e-2f;		// allowed error margin

	CpuSNN* sim = new CpuSNN(name,CPU_MODE,SILENT,0,1,randSeed);
	int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
	int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
	sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
	sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
	sim->setConductances(true,5,10,15,20,25,30,ALL);
	sim->setSTP(g0,true,0.19f,86,992,ALL); // the exact values are not important, as long as current matches

	bool spikeAtZero = false;
	PeriodicSpikeGeneratorCore* spk50 = new PeriodicSpikeGeneratorCore(50.0f,spikeAtZero); // periodic spiking @ 50 Hz
	sim->setSpikeGenerator(g0, spk50, ALL);

	sim->setupNetwork(true);
	for (int i=0; i<50; i++) {
		sim->runNetwork(0,1,false,true); // enable copyState
//		fprintf(stderr,"%.4f,",sim->current[0]);
		EXPECT_NEAR(sim->current[0], current[i], abs_error); // check post-synaptic current to see effect of pre-STP
	}

	delete spk50;
	delete sim;
}*/

/*!
 * \brief check whether CPU and GPU mode return the same stpu and stpx
 * This test creates a STP connection with random parameter values, runs a simulation
 * for 300 ms, and checks whether CPU_MODE and GPU_MODE return the same internal
 * variables stpu and stpx. Input is periodic 20 Hz spiking.
 */
/*TEST(STP, internalCPUvsGPU) {
	CpuSNN* sim = NULL;
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};

	float stpu[600] = {0.0f};
	float stpx[600] = {0.0f};

	int nConfig = 1;
	int randSeed = rand() % 1000;
	float STP_U = (float) rand()/RAND_MAX;
	int STP_tD = rand() % 100;
	int STP_tF = rand() % 500 + 500;
	float abs_error = 1e-2f; // error allowed for CPU<->GPU mode

	for (int j=0; j<2; j++) {
		sim = new CpuSNN(name,simModes[j],SILENT,0,nConfig,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(true,5,10,15,20,25,30,ALL);
		sim->setSTP(g0,true,STP_U,STP_tD,STP_tF,ALL);

		bool spikeAtZero = false;
		PeriodicSpikeGeneratorCore* spk20 = new PeriodicSpikeGeneratorCore(20.0f,spikeAtZero);
		sim->setSpikeGenerator(g0, spk20, ALL);

		sim->setupNetwork(true);
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
		EXPECT_NEAR(stpu[i],stpu[i+300],abs_error); // EXPECT_FLOAT_EQ sometimes works, too
		EXPECT_NEAR(stpx[i],stpx[i+300],abs_error);	// but _NEAR is better
	}

	// check init default values
	EXPECT_FLOAT_EQ(stpu[0],0.0);
	EXPECT_FLOAT_EQ(stpx[0],1.0);
	EXPECT_FLOAT_EQ(stpu[300],0.0);
	EXPECT_FLOAT_EQ(stpx[300],1.0);
}*/

#if ENABLE_CPU_GPU_TESTS
// FIXME: The following test fails, but this is expected because of issue #61). There is the possibility of rounding
//        errors when going from CPU to GPU. But, the order of computation in the innermost loop (doSnnSim, doGPUsim)
//        is quite different, which makes it not so hard to believe that the resulting output would be different, too.
/*!
 * \brief check whether CPU and GPU mode return the post-synaptic current, affected by pre-synaptic STP (external)
 * This test creates a STP connection with random parameter values, runs a simulation
 * for 300 ms, and checks whether CPU_MODE and GPU_MODE return the same external behavior (post-synaptic current
 * affected by pre-synaptic STP). Input is periodic 20 Hz spiking.
 */
TEST(STP, externalCPUvsGPU) {
	CpuSNN* sim = NULL;
	std::string name = "SNN";
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};

	float current[600] = {0.0f};

	int nConfig = 1;
	int randSeed = rand() % 1000;
	float STP_U = (float) rand()/RAND_MAX;
	int STP_tD = rand() % 100;
	int STP_tF = rand() % 500 + 500;
	float abs_error = 1e-2f; // error allowed for CPU<->GPU mode

	for (int j=0; j<2; j++) {
		sim = new CpuSNN(name,simModes[j],USER,0,nConfig,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON, ALL);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"full",0.01f,0.01f,1.0f,1,1,1.0f,1.0f,SYN_FIXED);
		sim->setConductances(true,5,10,15,20,25,30,ALL);
		sim->setSTP(g0,true,STP_U,STP_tD,STP_tF,ALL);

		bool spikeAtZero = false;
		PeriodicSpikeGeneratorCore* spk20 = new PeriodicSpikeGeneratorCore(20.0f,spikeAtZero);
		sim->setSpikeGenerator(g0, spk20, ALL);

		for (int i=0; i<300; i++) {
			sim->runNetwork(0,1,true); // enable copyState
			current[j*300+i] = sim->current[0];
		}

		delete spk20;
		delete sim;
	}

	// compare stpu and stpx for both sim modes
	for (int i=0; i<300; i++) {
		EXPECT_NEAR(current[i],current[i+300],abs_error); // EXPECT_FLOAT_EQ sometimes works, too
	}
}
#endif
