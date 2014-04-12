#include "gtest\gtest.h"
#include <carlsim.h>
#include <snn.h>
#include "carlsim_tests.h"

//! trigger all UserErrors
// TODO: add more error checking
TEST(Interface, connect) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->connect(g1,g1,"random",0.01f,0.1f,1);},""); // g2 cannot be PoissonGroup
	EXPECT_DEATH({sim->connect(g1,g1,"random",-0.01f,0.1f,1);},""); // weight cannot be negative
}

//! trigger all UserErrors
TEST(Interface, getSpikeCounter) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setSpikeCounter(g1);
	EXPECT_DEATH({sim->getSpikeCounter(ALL);},"");
	EXPECT_DEATH({sim->getSpikeCounter(g1,ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setConductances) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,USER,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1,g1,"random",0.01f,0.1f,1);

	// set custom values, no rise times
	EXPECT_DEATH({sim->setConductances(true,-1,2,3,4);},"");
	EXPECT_DEATH({sim->setConductances(true,1,-2,3,4);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,-3,4);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,3,-4);},"");

	// set custom values, all
	EXPECT_DEATH({sim->setConductances(true,-1,2,3,4,5,6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,-2,3,4,5,6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,-3,4,5,6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,3,-4,5,6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,3,4,-5,6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,3,4,5,-6);},"");
	EXPECT_DEATH({sim->setConductances(true,1,2,2,4,5,6);},""); // tdNMDA==trNMDA
	EXPECT_DEATH({sim->setConductances(true,1,2,3,4,5,5);},""); // tdGABAb==trGABAb

	// calling setConductances after runNetwork
	sim->setupNetwork();
	sim->runNetwork(0,0);
	EXPECT_DEATH({sim->setConductances(true);},"");
	EXPECT_DEATH({sim->setConductances(false,1,2,3,4);},"");
	EXPECT_DEATH({sim->setConductances(false,1,2,3,4,5,6);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setSpikeCounter) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->setSpikeCounter(ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setDefaultConductanceTimeConstants) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT);
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(-1,2,3,4,5,6);},""); // negative values
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,-2,3,4,5,6);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,-3,4,5,6);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,3,-4,5,6);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,3,4,-5,6);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,3,4,5,-6);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,2,4,5,6);},"");  // trNMDA==tdNMDA
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1,2,3,4,5,5);},"");  // trGABAb==tdGABAb
	delete sim;
}
