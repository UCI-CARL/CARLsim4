#include "gtest/gtest.h"
#include <carlsim.h>
#include <snn.h>
#include "carlsim_tests.h"

//! trigger all UserErrors
// TODO: add more error checking
TEST(Interface, connectDeath) {
	CARLsim* sim = new CARLsim("Interface.connectDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->connect(g1,g1,"random",RangeWeight(0.01f,0.1f),0.1);},""); // g2 cannot be PoissonGroup
	EXPECT_DEATH({sim->connect(g1,g1,"random",RangeWeight(-0.01f,0.1f),0.1);},""); // weight cannot be negative
	EXPECT_DEATH({sim->connect(g1,g1,"random",RangeWeight(0.1f),0.1,RangeDelay(1),RadiusRF(0,0,0));},""); // radius=0
	EXPECT_DEATH({sim->connect(g1,g1,"one-to-one",RangeWeight(0.1f),0.1,RangeDelay(1),RadiusRF(3,0,0));},""); // rad>0
	delete sim;
}

//! Death tests for createGroup (test all possible silly values)
// \FIXME this should be interface-level
TEST(Interface, createGroupDeath) {
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
	CARLsim* sim = new CARLsim("Interface.getGroupGrid3D",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setupNetwork();

	EXPECT_DEATH({sim->getGroupGrid3D(-1);},"");
	EXPECT_DEATH({sim->getGroupGrid3D(1);},"");

	delete sim;
}

TEST(Interface, getNeuronLocation3DDeath) {
	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,0,42);
	Grid3D grid(2,3,4);
	int g1=sim->createGroup("excit", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setupNetwork();

	EXPECT_DEATH({sim->getNeuronLocation3D(-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(grid.x*grid.y*grid.z);},"");

	EXPECT_DEATH({sim->getNeuronLocation3D(-1,-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1, grid.x*grid.y*grid.z);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1+1,0);},"");

	delete sim;
}

//! trigger all UserErrors
TEST(Interface, getSpikeCounterDeath) {
	CARLsim* sim = new CARLsim("Interface.getSpikeCounterDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createGroup("excit", 10, EXCITATORY_NEURON);
	sim->setSpikeCounter(g1);
	EXPECT_DEATH({sim->getSpikeCounter(ALL);},"");
	delete sim;
}

//! trigger all UserErrors
TEST(Interface, setConductancesDeath) {
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

	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g1,"random", RangeWeight(0.0,0.001,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);


	// test APIs that can't be called at CONFIG_STATE
	EXPECT_DEATH({sim->runNetwork(1, 0);},"");
	EXPECT_DEATH({sim->saveSimulation("test.dat", true);},"");
	EXPECT_DEATH({sim->reassignFixedWeights(0, wM, 4);},"");
	EXPECT_DEATH({sim->setSpikeRate(g1, NULL);},"");
	EXPECT_DEATH({sim->writePopWeights("test.dat", 0, 1);},"");
	EXPECT_DEATH({sim->getDelays(0, 1, i, j);},"");
	EXPECT_DEATH({sim->getGroupId("hello");},"");
	EXPECT_DEATH({sim->getGroupStartNeuronId(0);},"");
	EXPECT_DEATH({sim->getGroupEndNeuronId(0);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(0);},"");
	EXPECT_DEATH({sim->getNumPreSynapses();},"");
	EXPECT_DEATH({sim->getNumPostSynapses();},"");
	EXPECT_DEATH({sim->getPopWeights(0, 1, w, i);},"");
	EXPECT_DEATH({sim->getSpikeCounter(0);},"");
	EXPECT_DEATH({sim->resetSpikeCounter(0);},"");

	sim->setConductances(true);

	// test buildNetwork(), change carlsimState_ from CONFIG_STATE to SETUP_STATE
	EXPECT_TRUE(sim->getCarlsimState() == CONFIG_STATE);
	sim->setupNetwork();
	EXPECT_TRUE(sim->getCarlsimState() == SETUP_STATE);
	//----- SETUP_STATE zone -----

	// test APIs that can't be called at SETUP_STATE
	EXPECT_DEATH({g2 = sim->createGroup("excit", 800, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({g2 = sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON);},"");
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
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true);},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setSTP(g1, true, 1.0, 2.0, 3.0);},"");
	EXPECT_DEATH({sim->setSTP(g1, true);},"");
	EXPECT_DEATH({sim->setWeightAndWeightChangeUpdate();},"");
	EXPECT_DEATH({sim->setupNetwork();},"");
	EXPECT_DEATH({sim->loadSimulation(NULL);},"");
	EXPECT_DEATH({sim->getSpikeCounter(0);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setDefaultHomeostasisParams(1.0, 2.0);},"");
	EXPECT_DEATH({sim->setDefaultSaveOptions("test.dat", true);},"");
	EXPECT_DEATH({sim->setDefaultSTDPparams(1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultSTPparams(1, 1.0, 2.0, 3.0);},"");

	// test runNetwork(), change carlsimState_ from SETUP_STATE to EXE_STATE
	EXPECT_TRUE(sim->getCarlsimState() == SETUP_STATE);
	sim->runNetwork(1, 0);
	EXPECT_TRUE(sim->getCarlsimState() == EXE_STATE);
	//----- EXE_STATE zone -----

	// test APIs that can't be called at EXE_STATE
	EXPECT_DEATH({sim->setupNetwork();},"");
	EXPECT_DEATH({sim->loadSimulation(NULL);},"");
	EXPECT_DEATH({sim->reassignFixedWeights(0, wM, 4);},"");
	EXPECT_DEATH({g2 = sim->createGroup("excit", 800, EXCITATORY_NEURON);},"");
	EXPECT_DEATH({g2 = sim->createSpikeGeneratorGroup("input", 100, EXCITATORY_NEURON);},"");
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
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true);},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setSTP(g1, true, 1.0, 2.0, 3.0);},"");
	EXPECT_DEATH({sim->setSTP(g1, true);},"");
	EXPECT_DEATH({sim->setWeightAndWeightChangeUpdate();},"");
	EXPECT_DEATH({sim->setConnectionMonitor(0, 1);},"");
	EXPECT_DEATH({sim->setGroupMonitor(0);},"");
	EXPECT_DEATH({sim->setSpikeCounter(0);},"");
	//EXPECT_DEATH({sim->setSpikeGenerator(0, SpikeGenerator* spikeGen);},"");
	EXPECT_DEATH({sim->setSpikeMonitor(0);},"");
	//EXPECT_DEATH({sim->setSpikeMonitor(0, const std::string& fname, int configId=0);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setDefaultHomeostasisParams(1.0, 2.0);},"");
	EXPECT_DEATH({sim->setDefaultSaveOptions("test.dat", true);},"");
	EXPECT_DEATH({sim->setDefaultSTDPparams(1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultSTPparams(1, 1.0, 2.0, 3.0);},"");

	delete sim;
}

TEST(Interface, setDefaultSTDPparamsDeath) {
	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,0,42);

	int	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	EXPECT_DEATH({sim->setDefaultESTDPparams(-1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, -2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, 2.0, -3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultESTDPparams(1.0, 2.0, 3.0, -4.0);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(-1.0, 2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(1.0, -2.0, 3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(.0, 2.0, -3.0, 4.0);},"");
	EXPECT_DEATH({sim->setDefaultISTDPparams(1.0, 2.0, 3.0, -4.0);},"");

	delete sim;
}

TEST(Interface, setSTDPDeath) {
	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,0,42);

	int	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(-1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(1.0, -2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(1.0, 2.0, -3.0, 4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HebbianCurve(1.0, 2.0, 3.0, -4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(-1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(1.0, -2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(1.0, 2.0, -3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, ConstantSymmetricCurve(1.0, 2.0, 3.0, -4.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HalfHebbianCurve(-1.0, 2.0, 3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HalfHebbianCurve(1.0, -2.0, 3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HalfHebbianCurve(1.0, 2.0, -3.0, 4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HalfHebbianCurve(1.0, 2.0, 3.0, -4.0, 5.0));},"");
	EXPECT_DEATH({sim->setESTDP(g1, true, STANDARD, HalfHebbianCurve(1.0, 2.0, 3.0, -4.0, -5.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, LinearSymmetricCurve(-1.0, 2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, LinearSymmetricCurve(1.0, -2.0, 3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, LinearSymmetricCurve(1.0, 2.0, -3.0, 4.0));},"");
	EXPECT_DEATH({sim->setISTDP(g1, true, STANDARD, LinearSymmetricCurve(1.0, 2.0, 3.0, -4.0));},"");

	delete sim;
}
