#include "gtest/gtest.h"
#include <carlsim.h>
#include "carlsim_tests.h"

class DummyCG: public ConnectionGenerator {
public:
	DummyCG() {}
	~DummyCG() {}

	void connect(CARLsim* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay,
		bool& connected) {
		weight = 1.0f;
		maxWt = 1.0f;
		delay = 1;
		connected = true;
	}
};

//! trigger all UserErrors
// TODO: add more error checking
TEST(Interface, connectDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.connectDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f,-65.0f,8.0f);

	// regular connect call
	EXPECT_DEATH(sim->connect(g1,g1,"random",RangeWeight(0.1f),0.1f),""); // g-post cannot be PoissonGroup
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(-0.01f),0.1f),""); // weight cannot be negative
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.01f,0.1f,0.1f),0.1f),""); // wt.min>0
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.0f,0.01f,0.1f),0.1f),""); // SYN_FIXED wt.init!=wt.max
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.0f,0.01f,0.1f),-0.1f),""); // prob<0
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.0f,0.01f,0.1f),2.3f),""); // prob>1
	EXPECT_DEATH(sim->connect(g1,g2,"one-to-one",RangeWeight(0.1f),0.1f,RangeDelay(1),RadiusRF(3,0,0)),""); // rad>0
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.1f),0.1f,RangeDelay(1),RadiusRF(-1),SYN_FIXED,-1.0f,0.0f),""); // mulSynFast<0
	EXPECT_DEATH(sim->connect(g1,g2,"random",RangeWeight(0.1f),0.1f,RangeDelay(1),RadiusRF(-1),SYN_FIXED,0.0f,-1.0f),""); // mulSynSlow<0

	// custom ConnectionGenerator
	ConnectionGenerator* CGNULL = NULL;
	DummyCG* CG = new DummyCG;
	EXPECT_DEATH({sim->connect(g1,g2,CGNULL);},""); // CG=NULL
	EXPECT_DEATH({sim->connect(g1,g1,CG);},""); // g-post cannot be PoissonGroup
	EXPECT_DEATH({sim->connect(g1,g2,CG,SYN_FIXED,-1,100);},""); // maxM<0
	EXPECT_DEATH({sim->connect(g1,g2,CG,SYN_FIXED,100,-1);},""); // maxPreM<0

	// custom ConnectionGenerator with mulSyns
	EXPECT_DEATH({sim->connect(g1,g2,CGNULL,1.0f,1.0f);},""); // CG=NULL
	EXPECT_DEATH({sim->connect(g1,g1,CG,1.0f,1.0f);},""); // g-post cannot be PoissonGroup
	EXPECT_DEATH({sim->connect(g1,g2,CG,-1.0f,1.0f,SYN_FIXED,0,0);},""); // mulSynFast<0
	EXPECT_DEATH({sim->connect(g1,g2,CG,1.0f,-1.0f,SYN_FIXED,0,0);},""); // mulSynSlow<0
	EXPECT_DEATH({sim->connect(g1,g2,CG,1.0f,1.0f,SYN_FIXED,-2,0);},""); // maxM<0
	EXPECT_DEATH({sim->connect(g1,g2,CG,1.0f,1.0f,SYN_FIXED,0,-4);},""); // maxPreM<0

	delete CG;
	delete sim;
}

//! Death tests for createGroup (test all possible silly values)
// \FIXME this should be interface-level
TEST(Interface, createGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,0,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrp, etc.
	EXPECT_DEATH({sim->createGroup("excit", -10, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createGroup("excit", 10, -3);},"");
	EXPECT_DEATH({sim->createGroup("excit", Grid3D(-10,1,1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createGroup("excit", Grid3D(1,-1,1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createGroup("excit", Grid3D(10,1,-1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createGroup("excit", Grid3D(1,1,1), -3);},"");

	if (sim!=NULL)
		delete sim;
}

//! Death tests for createSpikeGenerator (test all possible silly values)
// \FIXME make interface-level
TEST(Interface, createSpikeGeneratorGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.createSpikeGeneratorGroupDeath",CPU_MODE,SILENT,0,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrps, etc.
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", -10, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", 10, -3);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", Grid3D(-10,1,1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", Grid3D(1,-1,1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", Grid3D(10,1,-1), EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->createSpikeGeneratorGroup("excit", Grid3D(1,1,1), -3);},"");

	if (sim!=NULL)
		delete sim;
}

TEST(Interface, getGroupGrid3DDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.getGroupGrid3D",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setConductances(true);
	sim->setupNetwork();

	EXPECT_DEATH({sim->getGroupGrid3D(-1);},"");
	EXPECT_DEATH({sim->getGroupGrid3D(1);},"");

	delete sim;
}

TEST(Interface, getNeuronLocation3DDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,0,42);
	Grid3D grid(2,3,4);
	int g1=sim->createGroup("excit", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setConductances(true);
	sim->setupNetwork();

	EXPECT_DEATH({sim->getNeuronLocation3D(-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(grid.x*grid.y*grid.z);},"");

	EXPECT_DEATH({sim->getNeuronLocation3D(-1,-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1, grid.x*grid.y*grid.z);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1+1,0);},"");

	delete sim;
}

TEST(Interface, loggerDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = NULL;

	sim = new CARLsim("Interface.loggerDeath",CPU_MODE,CUSTOM,0,42);
	EXPECT_DEATH({sim->setLogFile("meow.log");},"");
	if (sim!=NULL) delete sim; sim = NULL;

	sim = new CARLsim("Interface.loggerDeath",CPU_MODE,SILENT,0,42);
	EXPECT_DEATH({sim->setLogsFpCustom();},"");
	if (sim!=NULL) delete sim; sim = NULL;

	EXPECT_DEATH({CARLsim* sim = new CARLsim("Interface.loggerDeath",CPU_MODE,UNKNOWN_LOGGER,0,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
}

TEST(Interface, biasWeightsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.biasWeightsDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setConductances(true);

	EXPECT_DEATH({sim->biasWeights(c1, 0.1, false);},""); // CONFIG state

	sim->setupNetwork();
	sim->runNetwork(0,20);

	EXPECT_DEATH({sim->biasWeights(c1+1, 0.1, false);},""); // invalid connId
	EXPECT_DEATH({sim->biasWeights(-1,   0.1, false);},""); // invalid connId
}

TEST(Interface, scaleWeightsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.scaleWeightsDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));

	EXPECT_DEATH({sim->scaleWeights(c1, 0.1, false);},""); // CONFIG state

	sim->setConductances(true);
	sim->setupNetwork();
	sim->runNetwork(0,20);

	EXPECT_DEATH({sim->scaleWeights(c1+1, 0.1, false);},""); // invalid connId
	EXPECT_DEATH({sim->scaleWeights(-1,   0.1, false);},""); // invalid connId
	EXPECT_DEATH({sim->scaleWeights(0,   -1.0, false);},""); // scale<0
}

TEST(Interface, setWeightDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setWeightDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));

	EXPECT_DEATH({sim->setWeight(c1, 0, 0, 0.1, false);},""); // CONFIG state

	sim->setConductances(true);
	sim->setupNetwork();
	sim->runNetwork(0,20);

	EXPECT_DEATH({sim->setWeight(c1+1, 0,  0,  0.1, false);},""); // invalid connId
	EXPECT_DEATH({sim->setWeight(-1,   0,  0,  0.1, false);},""); // connId<0
	EXPECT_DEATH({sim->setWeight(0,   -1,  0,  0.1, false);},""); // neurIdPre<0
	EXPECT_DEATH({sim->setWeight(0,  101,  0,  0.1, false);},""); // invalid neurIdPre
	EXPECT_DEATH({sim->setWeight(0,    0, -1,  0.1, false);},""); // neurIdPost<0
	EXPECT_DEATH({sim->setWeight(0,    0,101,  0.1, false);},""); // invalid neurIdPost
	EXPECT_DEATH({sim->setWeight(0,    0,  0, -1.0, false);},""); // weight<0
}

TEST(Interface, getDelayRangeDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.getDelayRangeDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1,10));
	EXPECT_DEATH({sim->getDelayRange(c1+1);},"");
	EXPECT_DEATH({sim->getDelayRange(-1);},"");

	sim->setConductances(true);

	sim->setupNetwork();
	EXPECT_DEATH({sim->getDelayRange(c1+1);},"");
	EXPECT_DEATH({sim->getDelayRange(-1);},"");

	sim->runNetwork(0,20);
	EXPECT_DEATH({sim->getDelayRange(c1+1);},"");
	EXPECT_DEATH({sim->getDelayRange(-1);},"");
}

TEST(Interface, getWeightRangeDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.getWeightRangeDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.0, 0.1, 0.1), 1.0f, RangeDelay(1,10));
	EXPECT_DEATH({sim->getWeightRange(c1+1);},"");
	EXPECT_DEATH({sim->getWeightRange(-1);},"");

	sim->setConductances(true);

	sim->setupNetwork();
	EXPECT_DEATH({sim->getWeightRange(c1+1);},"");
	EXPECT_DEATH({sim->getWeightRange(-1);},"");

	sim->runNetwork(0,20);
	EXPECT_DEATH({sim->getWeightRange(c1+1);},"");
	EXPECT_DEATH({sim->getWeightRange(-1);},"");
}

//! trigger all UserErrors
TEST(Interface, getSpikeCounterDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.getSpikeCounterDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setSpikeCounter(g1);
	EXPECT_DEATH({sim->getSpikeCounter(ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setConductancesDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setConductancesDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1,g1,"random",RangeWeight(0.01),0.1f,RangeDelay(1));

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
	sim->setConductances(false);
	sim->setupNetwork();
	sim->runNetwork(0,0);
	EXPECT_DEATH({sim->setConductances(true);},"");
	EXPECT_DEATH({sim->setConductances(false,1,2,3,4);},"");
	EXPECT_DEATH({sim->setConductances(false,1,2,3,4,5,6);},"");
	delete sim;
}

TEST(Interface, setExternalCurrentDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setExternalCurrentDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int g0=sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);
	sim->connect(g0,g1,"random",RangeWeight(0.01),0.1f,RangeDelay(1));

	// calling setExternalCurrent in CONFIG
	float current = 0.0f;
	std::vector<float> vecCurrent(10, current);
	EXPECT_DEATH({sim->setExternalCurrent(g1,vecCurrent);},"");
	EXPECT_DEATH({sim->setExternalCurrent(g1,current);},"");

	sim->setConductances(true);
	sim->setupNetwork();

	// calling setExternalCurrent in correct state but with invalid input arguments
	EXPECT_DEATH({sim->setExternalCurrent(100,vecCurrent);},""); // grpId out of bounds
	EXPECT_DEATH({sim->setExternalCurrent(100,current);},""); // grpId out of bounds
	EXPECT_DEATH({sim->setExternalCurrent(-1,vecCurrent);},""); // ALL not allowed
	EXPECT_DEATH({sim->setExternalCurrent(-1,current);},""); // ALL not allowed
	EXPECT_DEATH({sim->setExternalCurrent(g0,vecCurrent);},""); // calling on SpikeGen grp
	EXPECT_DEATH({sim->setExternalCurrent(g0,current);},""); // calling on SpikeGen grp
	std::vector<float> vecCurrent2(20, 0.1f);
	EXPECT_DEATH({sim->setExternalCurrent(g1,vecCurrent2);},""); // current wrong size

	delete sim;
}


//! Death tests for setNeuronParameters (test all possible silly values)
TEST(Interface, setNeuronParametersDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = NULL;
	sim = new CARLsim("Interface.setNeuronParametersDeath",CPU_MODE,SILENT,0,42);
	int g0=sim->createGroup("excit", Grid3D(10,1,1), EXCITATORY_NEURON);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGrpsetc.
	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.2f, -65.0f, 8.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.2f, -65.0f, 8.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, -0.02f, 0.2f, -65.0f, 8.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, -0.2f, -65.0f, 8.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, -8.0f);},"");

	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, -0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, -10.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, -0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, -10.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, -2.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, -8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, -10.0f);},"");

	if (sim!=NULL)
		delete sim;
}

TEST(Interface, CARLsimConstructorDeathGPU) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim *sim1=NULL, *sim2=NULL, *sim3=NULL;
	EXPECT_DEATH({sim1 = new CARLsim("Interface.CARLsimConstructorDeathGPU", GPU_MODE, SILENT, -1);},"");
	EXPECT_DEATH({sim2 = new CARLsim("Interface.CARLsimConstructorDeathGPU", GPU_MODE, SILENT, 42);},"");
	// This test will fail if the machine has 8 GPUs
	EXPECT_DEATH({sim3 = new CARLsim("Interface.CARLsimConstructorDeathGPU", GPU_MODE, SILENT, 7);},"");
	if (sim1 != NULL) delete sim1;
	if (sim2 != NULL) delete sim2;
	if (sim3 != NULL) delete sim3;
}

TEST(Interface, AllocateGPUConflict) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim *sim1=NULL, *sim2=NULL;

	sim1 = new CARLsim("Interface.AllocateGPUConflict_A", GPU_MODE, SILENT, 0);
	EXPECT_DEATH({sim2 = new CARLsim("Interface.AllocateGPUConflict_B", GPU_MODE, SILENT, 0);},"");
	if (sim1 != NULL) delete sim1;

	sim1 = new CARLsim("Interface.AllocateGPUConflict", GPU_MODE, SILENT, 0);

	if (sim1 != NULL) delete sim1;
	if (sim2 != NULL) delete sim2;
}

//! trigger all UserErrors
TEST(Interface, setSpikeCounter) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setSpikeCounter",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->setSpikeCounter(ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setDefaultConductanceTimeConstants) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setDefaultConductanceTimeConstants",CPU_MODE,SILENT);
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

//! test APIs those are called at wrong state
TEST(Interface, CARLsimState) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	int g1, g2, i, j;
	float wM[4];
	float* w;
	CARLsim* sim = new CARLsim("Interface.CARLsimState",CPU_MODE,SILENT,0,42);
	//----- CONFIG_STATE zone -----

	g1 = sim->createGroup("excit", 80, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g1,"random", RangeWeight(0.0,0.001,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);


	// test APIs that can't be called at CONFIG_STATE
	EXPECT_DEATH({sim->runNetwork(1, 0);},"");
	EXPECT_DEATH({sim->saveSimulation("test.dat", true);},"");
	EXPECT_DEATH({sim->setSpikeRate(g1, NULL);},"");
	EXPECT_DEATH({sim->writePopWeights("test.dat", 0, 1);},"");
	EXPECT_DEATH({sim->getDelays(0, 1, i, j);},"");
	EXPECT_DEATH({sim->getGroupId("hello");},"");
	EXPECT_DEATH({sim->getGroupStartNeuronId(0);},"");
	EXPECT_DEATH({sim->getGroupEndNeuronId(0);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(0);},"");
	EXPECT_DEATH({sim->getNumPreSynapses();},"");
	EXPECT_DEATH({sim->getNumPostSynapses();},"");
	EXPECT_DEATH({sim->getSpikeCounter(0);},"");
	EXPECT_DEATH({sim->resetSpikeCounter(0);},"");

	sim->setConductances(true);

	// test buildNetwork(), change carlsimState_ from CONFIG_STATE to SETUP_STATE
	EXPECT_TRUE(sim->getCARLsimState() == CONFIG_STATE);
	sim->setupNetwork();
	EXPECT_TRUE(sim->getCARLsimState() == SETUP_STATE);
	//----- SETUP_STATE zone -----

	// test APIs that can't be called at SETUP_STATE
	EXPECT_DEATH({g2 = sim->createGroup("excit", 80, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({g2 = sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->connect(g1,g1,"random", RangeWeight(0.0,0.001,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);},"");
	EXPECT_DEATH({sim->setConductances(true);},"");
	EXPECT_DEATH({sim->setConductances(true,1, 2, 3, 4);},"");
	EXPECT_DEATH({sim->setConductances(true, 1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setHomeostasis(g1, true);},"");
	EXPECT_DEATH({sim->setHomeostasis(g1, true, 1.0, 2.0);},"");
	EXPECT_DEATH({sim->setHomeoBaseFiringRate(g1, 1.0, 2.0);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g1, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setNeuromodulator(g1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);},"");
	EXPECT_DEATH({sim->setNeuromodulator(g1, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setSTDP(g1, true);},"");
	EXPECT_DEATH({sim->setSTDP(g1, true, STANDARD, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setESTDP(g1, true);},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, ExpCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true);},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setSTP(g1, true, 1.0, 2.0, 3.0);},"");
	EXPECT_DEATH({sim->setSTP(g1, true);},"");
	EXPECT_DEATH({sim->getSpikeCounter(0);},"");
	EXPECT_DEATH({sim->setSpikeCounter(g1);},"");
	EXPECT_DEATH({sim->getSpikeCounter(g1);},"");
	EXPECT_DEATH({sim->setWeightAndWeightChangeUpdate(INTERVAL_1000MS, true, 0.9f);},"");
	EXPECT_DEATH({sim->setupNetwork();},"");
	EXPECT_DEATH({sim->loadSimulation(NULL);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setDefaultHomeostasisParams(1.0, 2.0);},"");
	EXPECT_DEATH({sim->setDefaultSaveOptions("test.dat", true);},"");
	EXPECT_DEATH({sim->setDefaultSTDPparams(1.0, 2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultSTPparams(1, 1.0, 2.0, 3.0);},"");

	// test runNetwork(), change carlsimState_ from SETUP_STATE to RUN_STATE
	EXPECT_TRUE(sim->getCARLsimState() == SETUP_STATE);
	sim->runNetwork(1, 0);
	EXPECT_TRUE(sim->getCARLsimState() == RUN_STATE);
	//----- RUN_STATE zone -----

	// test APIs that can't be called at RUN_STATE
	EXPECT_DEATH({sim->setupNetwork();},"");
	EXPECT_DEATH({sim->loadSimulation(NULL);},"");
	EXPECT_DEATH({g2 = sim->createGroup("excit", 80, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({g2 = sim->createSpikeGeneratorGroup("input", 10, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({sim->connect(g1,g1,"random", RangeWeight(0.0,0.001,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);},"");
	//sim->connect
	//sim->connect
	EXPECT_DEATH({sim->setConductances(true);},"");
	EXPECT_DEATH({sim->setConductances(true,1, 2, 3, 4);},"");
	EXPECT_DEATH({sim->setConductances(true, 1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setHomeostasis(g1, true);},"");
	EXPECT_DEATH({sim->setHomeostasis(g1, true, 1.0, 2.0);},"");
	EXPECT_DEATH({sim->setHomeoBaseFiringRate(g1, 1.0, 2.0);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g1, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setNeuromodulator(g1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);},"");
	EXPECT_DEATH({sim->setNeuromodulator(g1, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setSTDP(g1, true);},"");
	EXPECT_DEATH({sim->setSTDP(g1, true, STANDARD, 1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setESTDP(g1, true);},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, ExpCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true);},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setSTP(g1, true, 1.0, 2.0, 3.0);},"");
	EXPECT_DEATH({sim->setSTP(g1, true);},"");
	EXPECT_DEATH({sim->setWeightAndWeightChangeUpdate(INTERVAL_1000MS, true, 0.9f);},"");
	EXPECT_DEATH({sim->setConnectionMonitor(0, 1, "Default");},"");
	EXPECT_DEATH({sim->setGroupMonitor(0, "Default");},"");
	EXPECT_DEATH({sim->setSpikeCounter(g1);},"");
	//EXPECT_DEATH({sim->setSpikeGenerator(0, SpikeGenerator* spikeGen);},"");
	EXPECT_DEATH({sim->setSpikeMonitor(0, "Default");},"");
	//EXPECT_DEATH({sim->setSpikeMonitor(0, const std::string& fname, int configId=0);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setDefaultHomeostasisParams(1.0, 2.0);},"");
	EXPECT_DEATH({sim->setDefaultSaveOptions("test.dat", true);},"");
	EXPECT_DEATH({sim->setDefaultSTDPparams(1.0, 2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultSTPparams(1, 1.0, 2.0, 3.0);},"");

	delete sim;
}

TEST(Interface, setDefaultSTDPparamsDeath) {
	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,0,42);

	int	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

//	EXPECT_DEATH({sim->setDefaultESTDPparams(-1.0, 2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, -2.0, 3.0, 4.0,STANDARD);},"");
//	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, 2.0, -3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, 2.0, 3.0, -4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, 2.0, 3.0, 4.0,UNKNOWN_STDP);},"");
    EXPECT_DEATH({sim->setDefaultISTDPparams(-1.0, 2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(1.0, -2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(.0, 2.0, -3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(1.0, 2.0, 3.0, -4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(1.0, 2.0, 3.0, 4.0,UNKNOWN_STDP);},"");

	delete sim;
}

TEST(Interface, setSTDPDeath) {
	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,0,42);

	int	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, ExpCurve(1.0, -2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, ExpCurve(1.0, 2.0, 3.0, -4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(-1.0, -2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(1.0, -2.0, -3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, PulseCurve(1.0, -2.0, 3.0, -4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(-1.0, 2.0, -3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(1.0, -2.0, -3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(1.0, 2.0, 3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(1.0, 2.0, -3.0, -4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, TimingBasedCurve(1.0, 2.0, -3.0, -4.0, -5.0));},"");

	delete sim;
}
