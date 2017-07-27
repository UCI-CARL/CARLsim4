/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/
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

	CARLsim* sim = new CARLsim("Interface.connectDeath",CPU_MODE,SILENT,1,42);
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

	// custom ConnectionGenerator with mulSyns
	EXPECT_DEATH({sim->connect(g1,g2,CGNULL,1.0f,1.0f);},""); // CG=NULL
	EXPECT_DEATH({sim->connect(g1,g1,CG,1.0f,1.0f);},""); // g-post cannot be PoissonGroup
	EXPECT_DEATH({sim->connect(g1,g2,CG,-1.0f,1.0f,SYN_FIXED);},""); // mulSynFast<0
	EXPECT_DEATH({sim->connect(g1,g2,CG,1.0f,-1.0f,SYN_FIXED);},""); // mulSynSlow<0

	delete CG;
	delete sim;
}

TEST(Interface, connectCompartmentsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.connectCompartmentsDeath", CPU_MODE, SILENT, 0, 42);

	// set up compartmental groups
	int N = 5; //number of neurons
	int s = sim->createGroup("soma", N, EXCITATORY_NEURON);
	int d1 = sim->createGroup("d1", N, EXCITATORY_NEURON);
	int d2 = sim->createGroup("d2", N, EXCITATORY_NEURON);
	int d3 = sim->createGroup("d3", N, EXCITATORY_NEURON);
	int d4 = sim->createGroup("d4", N, EXCITATORY_NEURON);
	int d5 = sim->createGroup("d5", N, EXCITATORY_NEURON);
	int d6 = sim->createGroup("d6", 2 * N, EXCITATORY_NEURON);

	// some regular neuron groups
	int reg0 = sim->createGroup("reg0", 2 * N, EXCITATORY_NEURON);
	int reg1 = sim->createGroup("reg1", 2 * N, EXCITATORY_NEURON);

	// make them 9-param Izzy neurons
	sim->setNeuronParameters(s, 550.0f, 2.0, -59.0, -50.0, 0.0, -0.0, 24.0, -53.0, 109.0f);
	sim->setNeuronParameters(d1, 367.0f, 1.0, -59.0, -44.0, 0.0, 3.0, 20.0, -46.0, 24.0f);
	sim->setNeuronParameters(d2, 425.0f, 2.0, -59.0, -25.0, 0.0, 0.0, 13.0, -38.0, 69.0f);
	sim->setNeuronParameters(d3, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim->setNeuronParameters(d4, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim->setNeuronParameters(d5, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim->setNeuronParameters(d6, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim->setNeuronParameters(reg0, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);
	sim->setNeuronParameters(reg1, 225.0f, 1.0, -59.0, -36.0, 0.0, -4.0, 21.0, -40.0, 21.0f);

	// enable compartments
	sim->setCompartmentParameters(s, 1.0f, 1.0f);
	sim->setCompartmentParameters(d1, 1.0f, 1.0f);
	sim->setCompartmentParameters(d2, 1.0f, 1.0f);
	sim->setCompartmentParameters(d3, 1.0f, 1.0f);
	sim->setCompartmentParameters(d4, 1.0f, 1.0f);
	sim->setCompartmentParameters(d5, 1.0f, 1.0f);
	sim->setCompartmentParameters(d6, 1.0f, 1.0f);

	int gen = sim->createSpikeGeneratorGroup("SpikeGen", N, EXCITATORY_NEURON);

	// grpIDs must be valid, cannot be identical
	EXPECT_DEATH({ sim->connectCompartments(sim->getNumGroups(), d1); }, "");
	EXPECT_DEATH({ sim->connectCompartments(s, 10); }, "");
	EXPECT_DEATH({ sim->connectCompartments(s, -1); }, "");
	EXPECT_DEATH({ sim->connectCompartments(-1, s); }, "");
	EXPECT_DEATH({ sim->connectCompartments(d3, d3); }, "");

	// no spike generators in connect call
	EXPECT_DEATH({ sim->connectCompartments(gen, s); }, "");
	EXPECT_DEATH({ sim->connectCompartments(d2, gen); }, "");
	EXPECT_DEATH({ sim->connectCompartments(gen, gen); }, "");

	// groups must be of same size
	EXPECT_DEATH({ sim->connectCompartments(s, d6); }, "");


	// connectCompartments is bidirectional: connecting same groups twice is illegal
	sim->connectCompartments(s, d1);
	EXPECT_DEATH({ sim->connectCompartments(s, d1); }, "");
	EXPECT_DEATH({ sim->connectCompartments(d1, s); }, "");

	// can't have both synaptic and compartmental connections on the same groups
	EXPECT_DEATH({ sim->connect(s, d1, "full", RangeWeight(1.0f), 1.0f); }, "");
	EXPECT_DEATH({ sim->connect(d1, s, "full", RangeWeight(1.0f), 1.0f); }, "");
	sim->connect(d3, d2, "full", RangeWeight(1.0f), 1.0f);
	EXPECT_DEATH({ sim->connectCompartments(d3, d2); }, "");
	EXPECT_DEATH({ sim->connectCompartments(d2, d3); }, "");

	// can't be involved in more than 4 connections (d1-d4), d5 must break
	sim->connectCompartments(d2, s);
	sim->connectCompartments(d3, s);
	sim->connectCompartments(s, d4);
	EXPECT_DEATH({ sim->connectCompartments(d5, s); }, "");
	EXPECT_DEATH({ sim->connectCompartments(s, d5); }, "");

	// use compartment connections on regular neurons
	// must break during setupNetwork (in verifyCompartments)
	sim->connectCompartments(reg0, reg1);
	EXPECT_DEATH({ sim->setupNetwork(); }, "");
}

//! Death tests for createGroup (test all possible silly values)
// \FIXME this should be interface-level
TEST(Interface, createGroupDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,1,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGroups, etc.
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

	CARLsim* sim = new CARLsim("Interface.createSpikeGeneratorGroupDeath",CPU_MODE,SILENT,1,42);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGroups, etc.
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

	CARLsim* sim = new CARLsim("Interface.getGroupGrid3D",CPU_MODE,SILENT,1,42);
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

	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,1,42);
	Grid3D grid(2,3,4);
	int g1=sim->createGroup("excit", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));
	sim->setConductances(true);
	sim->setupNetwork();

	EXPECT_DEATH({sim->getNeuronLocation3D(-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(grid.numX * grid.numY * grid.numZ);},"");

	EXPECT_DEATH({sim->getNeuronLocation3D(-1,-1);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1, grid.numX * grid.numY * grid.numZ);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(g1+1,0);},"");

	delete sim;
}

TEST(Interface, loggerDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	CARLsim* sim = NULL;

	sim = new CARLsim("Interface.loggerDeath",CPU_MODE,CUSTOM,1,42);
	EXPECT_DEATH({sim->setLogFile("meow.log");},"");
	if (sim!=NULL) delete sim; sim = NULL;

	sim = new CARLsim("Interface.loggerDeath",CPU_MODE,SILENT,1,42);
	EXPECT_DEATH({sim->setLogsFpCustom();},"");
	if (sim!=NULL) delete sim; sim = NULL;

	EXPECT_DEATH({CARLsim* sim = new CARLsim("Interface.loggerDeath",CPU_MODE,UNKNOWN_LOGGER,1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
}

TEST(Interface, biasWeightsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.biasWeightsDeath",CPU_MODE,SILENT,1,42);
	int g1=sim->createGroup("excit", Grid3D(10,10,1), EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);
	int c1=sim->connect(g1, g1, "full", RangeWeight(0.01), 1.0f, RangeDelay(1));

	EXPECT_DEATH({sim->biasWeights(c1, 0.1, false);},""); // CONFIG state

	sim->setConductances(true);

	EXPECT_DEATH({sim->biasWeights(c1, 0.1, false);},""); // CONFIG state

	sim->setupNetwork();
	sim->runNetwork(0,20);

	EXPECT_DEATH({sim->biasWeights(c1+1, 0.1, false);},""); // invalid connId
	EXPECT_DEATH({sim->biasWeights(-1,   0.1, false);},""); // invalid connId
}

TEST(Interface, scaleWeightsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.scaleWeightsDeath",CPU_MODE,SILENT,1,42);
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

	CARLsim* sim = new CARLsim("Interface.setWeightDeath",CPU_MODE,SILENT,1,42);
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

	CARLsim* sim = new CARLsim("Interface.getDelayRangeDeath",CPU_MODE,SILENT,1,42);
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

	CARLsim* sim = new CARLsim("Interface.getWeightRangeDeath",CPU_MODE,SILENT,1,42);
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
TEST(Interface, setConductancesDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setConductancesDeath",CPU_MODE,SILENT,1,42);
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

	CARLsim* sim = new CARLsim("Interface.setExternalCurrentDeath",CPU_MODE,SILENT,1,42);
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
	EXPECT_DEATH({sim->setExternalCurrent(g0,vecCurrent);},""); // calling on spikeGenFunc grp
	EXPECT_DEATH({sim->setExternalCurrent(g0,current);},""); // calling on spikeGenFunc grp
	std::vector<float> vecCurrent2(20, 0.1f);
	EXPECT_DEATH({sim->setExternalCurrent(g1,vecCurrent2);},""); // current wrong size

	delete sim;
}

TEST(Interface, setHomeostasisDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim sim("Interface.setHomeostasisDeath",CPU_MODE,SILENT,0,42);
	int g1=sim.createGroup("excit", 10, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f,-65.0f,8.0f);

	int g0=sim.createSpikeGeneratorGroup("input0", 10, EXCITATORY_NEURON);

	sim.connect(g0,g1,"random",RangeWeight(0.01),0.1f,RangeDelay(1));

	sim.setConductances(true);

	// calling homeostasis on on spikeGenFunc
	EXPECT_DEATH({sim.setHomeostasis(g0, true);},"");
	EXPECT_DEATH({sim.setHomeoBaseFiringRate(g0, 20.0f, 0.0f);},"");

	// calling homeo base firing first
	EXPECT_DEATH({sim.setHomeoBaseFiringRate(g1, 20.0f, 0.0f);},"");

	// will not fail just yet (no plastic connections), but will fail in setupNetwork
	sim.setHomeostasis(g1, true);
	sim.setHomeoBaseFiringRate(g1, 20.0f, 0.0f);

	EXPECT_DEATH({sim.setupNetwork();},"");
}


//! Death tests for setNeuronParameters (test all possible silly values)
TEST(Interface, setNeuronParametersDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = NULL;
	sim = new CARLsim("Interface.setNeuronParametersDeath",CPU_MODE,SILENT,1,42);
	int g0=sim->createGroup("excit", Grid3D(10,1,1), EXCITATORY_NEURON);

	// set silly values to all possible input arguments
	// e.g., negative values for things>=0, values>numGroups, etc.
	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.2f, -65.0f, 8.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.2f, -65.0f, 8.0f);},"");

	EXPECT_DEATH({sim->setNeuronParameters(-2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0+1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, -10.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, -10.0f, -65.0f, 0.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, -10.0f, 8.0f, 0.0f);},"");
	EXPECT_DEATH({sim->setNeuronParameters(g0, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, -10.0f);},"");

	if (sim!=NULL)
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
	CARLsim* sim = new CARLsim("Interface.CARLsimState",CPU_MODE,SILENT,1,42);
	//----- CONFIG_STATE zone -----

	g1 = sim->createGroup("excit", 80, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g1,"random", RangeWeight(0.0,0.001,0.005), 0.1f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);


	// test APIs that can't be called at CONFIG_STATE
	EXPECT_DEATH({sim->runNetwork(1, 0);},"");
	EXPECT_DEATH({sim->saveSimulation("test.dat", true);},"");
	EXPECT_DEATH({sim->setSpikeRate(g1, NULL);},"");
	EXPECT_DEATH({sim->getDelays(0, 1, i, j);},"");
	EXPECT_DEATH({sim->getGroupStartNeuronId(0);},"");
	EXPECT_DEATH({sim->getGroupEndNeuronId(0);},"");
	EXPECT_DEATH({sim->getNeuronLocation3D(0);},"");
	EXPECT_DEATH({sim->getNumSynapses();},"");
	EXPECT_DEATH({sim->startTesting();},"");
	EXPECT_DEATH({sim->stopTesting();},"");

	sim->setConductances(true);

	// test generateRuntimeData(), change carlsimState_ from CONFIG_STATE to SETUP_STATE
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
	SpikeGenerator* spikeGenFunc = NULL;
	std::string fname = "test";
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
	EXPECT_DEATH({sim->setSpikeGenerator(0, spikeGenFunc);},"");
	EXPECT_DEATH({sim->setSpikeMonitor(0, "Default");},"");
	EXPECT_DEATH({sim->setSpikeMonitor(0, fname);},"");
	EXPECT_DEATH({sim->setDefaultConductanceTimeConstants(1, 2, 3, 4, 5, 6);},"");
	EXPECT_DEATH({sim->setDefaultHomeostasisParams(1.0, 2.0);},"");
	EXPECT_DEATH({sim->setDefaultSaveOptions("test.dat", true);},"");
	EXPECT_DEATH({sim->setDefaultSTDPparams(1.0, 2.0, 3.0, 4.0,STANDARD);},"");
	EXPECT_DEATH({sim->setDefaultSTPparams(1, 1.0, 2.0, 3.0);},"");

	delete sim;
}

TEST(Interface, setDefaultSTDPparamsDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";
	
	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,1,42);

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
	 ::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("Interface.setSTDPDeath",CPU_MODE,SILENT,1,42);

	int	g1 = sim->createGroup("excit", 800, EXCITATORY_NEURON);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// invalid values
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

	// setting on spikeGenFunc
	int g0 = sim->createSpikeGeneratorGroup("gen", 20, EXCITATORY_NEURON);
	EXPECT_DEATH({sim->setSTDP(g0,true);},"");
	EXPECT_DEATH({sim->setSTDP(g0,true,STANDARD,1.0f,2.0f,3.0f,4.0f);},"");
	EXPECT_DEATH({sim->setESTDP(g0, true);},"");
	EXPECT_DEATH({sim->setESTDP(g0, true, STANDARD, ExpCurve(1.0f,2.0f,3.0f,4.0f));},"");
	EXPECT_DEATH({sim->setESTDP(g0, true, STANDARD, TimingBasedCurve(1.0f,2.0f,-3.0f,4.0f,5.0f));},"");
	EXPECT_DEATH({sim->setISTDP(g0, true);},"");
	EXPECT_DEATH({sim->setISTDP(g0, true, STANDARD, ExpCurve(1.0f,2.0f,3.0f,4.0f));},"");
	EXPECT_DEATH({sim->setISTDP(g0, true, STANDARD, PulseCurve(1.0f,-2.0f,3.0f,4.0f));},"");

	// setting on fixed synapses
	sim->setSTDP(g1,true,STANDARD,1.0f,2.0f,3.0f,4.0f);
	sim->connect(g0, g1, "random", RangeWeight(0.1f), 0.1f);
	sim->setConductances(true);

	EXPECT_DEATH({sim->setupNetwork();},"");

	delete sim;
}
