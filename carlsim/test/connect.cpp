#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <snn.h>
#include <vector>
#include <periodic_spikegen.h>

/// **************************************************************************************************************** ///
/// CONNECT FUNCTIONALITY
/// **************************************************************************************************************** ///

// make sure the function CpuSNN::isPoint3DinRF returns the right values for all cases
// need to use CpuSNN, because function is not in user interface
TEST(CORE, isPoint3DinRF) {
	CpuSNN snn("CONNECT.isPoint3DinRF", CPU_MODE, USER, 0, 1, 42);
	EXPECT_TRUE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0,0,0))); // same point
	EXPECT_TRUE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(10,0,0))); // on border
	EXPECT_TRUE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0,10,0)));
	EXPECT_TRUE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0,0,10)));
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(10.0001, 0.0, 0.0))); // a little too far
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0.0, 10.001, 0.0)));
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0.0, 0.0, 10.001)));
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(10,10,0))); // way too far
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(0,10,10)));
	EXPECT_FALSE(snn.isPoint3DinRF(RadiusRF(10.0), Point3D(0,0,0), Point3D(10,0,10)));
}

// \TODO make CARLsim-level
//! connect with certain mulSynFast, mulSynSlow and observe connectInfo
TEST(CONNECT, connectInfo) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;

	CARLsim* sim = NULL;
//	grpConnectInfo_t* connInfo;

	int conn[4]            = {-1};
	std::string typeStr[4] = {"random", "one-to-one", "full", "full-no-direct"};
	float initWt[4]        = {0.05f, 0.1f, 0.21f, 0.42f};
	float maxWt[4]         = {0.05f, 0.1f, 0.21f, 0.42f};
	float prob[4]          = {0.1, 0.2, 0.3, 0.4};
	int minDelay[4]        = {1,2,3,4};
	int maxDelay[4]        = {1,2,3,4};
	float mulSynFast[4]    = {0.2f, 0.8f, 1.2f, 0.0};
	float mulSynSlow[4]    = {0.0f, 2.4f, 11.1f, 10.0f};
	int synType[4]         = {SYN_FIXED,SYN_PLASTIC,SYN_FIXED,SYN_PLASTIC};

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			for (int i=0; i<4; i++) {
				sim = new CARLsim("CONNECT.connectInfo",mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

                Grid3D neur(10,1,1);
				int g0=sim->createSpikeGeneratorGroup("spike", neur, EXCITATORY_NEURON);
				int g1=sim->createGroup("excit0", neur, EXCITATORY_NEURON);
				sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
				int g2=sim->createGroup("excit1", neur, EXCITATORY_NEURON);
				sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

				conn[i] = sim->connect(g0, g1, typeStr[i], RangeWeight(0.0f,initWt[i],maxWt[i]), prob[i],
					RangeDelay(minDelay[i],maxDelay[i]), RadiusRF(-1), synType[i], mulSynFast[i], mulSynSlow[i]);

// \TODO: this should be replaced by ConnectionMonitor
/*				for (int c=0; c<nConfig; c++) {
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
				}*/
				delete sim;
			}
		}
	}
}


TEST(CONNECT, connectFull) {
	CARLsim* sim = new CARLsim("CORE.connectRadiusRF",CPU_MODE,SILENT,0,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"full",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,0)); // all in x
	int c2=sim->connect(g2,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,-1)); // all in x and z
	int c3=sim->connect(g3,g3,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2,3,0)); // 2D ellipse x,y
	int c4=sim->connect(g4,g4,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(0,2,3)); // 2D ellipse y,z
	int c5=sim->connect(g5,g5,"full",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2,2,2)); // 3D ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N * grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N * grid.x);
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N * grid.x * grid.z);
	EXPECT_EQ(sim->getNumSynapticConnections(c3), 144); // these numbers are less than what the grid would
	EXPECT_EQ(sim->getNumSynapticConnections(c4), 224); // say because of edge effects
	EXPECT_EQ(sim->getNumSynapticConnections(c5), 320);

	delete sim;
}

TEST(CONNECT, connectFullNoDirect) {
	CARLsim* sim = new CARLsim("CORE.connectRadiusRF",CPU_MODE,SILENT,0,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,0)); // all in x
	int c2=sim->connect(g2,g2,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,-1)); // all in x,z
	int c3=sim->connect(g3,g3,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2,3,0)); // ellipse x,y
	int c4=sim->connect(g4,g4,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(0,2,3)); // ellipse y,z
	int c5=sim->connect(g5,g5,"full-no-direct",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(2,2,2)); // ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	// same as connect full, but no self-connections
	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N * (grid.N-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N * (grid.x-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N * (grid.x*grid.z-1) );
	EXPECT_EQ(sim->getNumSynapticConnections(c3), 120); // these numbers are less than what the grid would
	EXPECT_EQ(sim->getNumSynapticConnections(c4), 200); // say because of edge effects
	EXPECT_EQ(sim->getNumSynapticConnections(c5), 296);

	delete sim;
}

TEST(CONNECT, connectOneToOne) {
	CARLsim* sim = new CARLsim("CORE.connectOneToOne",CPU_MODE,SILENT,0,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

	int c0=sim->connect(g0,g0,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,0)); // all in x
	int c2=sim->connect(g2,g2,"one-to-one",RangeWeight(0.1), 1.0, RangeDelay(1), RadiusRF(-1,0,-1)); // all in x,z

	sim->setupNetwork(); // need SETUP state for this function to work

	// RadiusRF should not matter
	EXPECT_EQ(sim->getNumSynapticConnections(c0), grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c1), grid.N);
	EXPECT_EQ(sim->getNumSynapticConnections(c2), grid.N);

	delete sim;
}

TEST(CONNECT, connectRandom) {
	CARLsim* sim = new CARLsim("CORE.connectRandom",CPU_MODE,SILENT,0,1,42);
	Grid3D grid(2,3,4);
	int g0=sim->createGroup("excit0", grid, EXCITATORY_NEURON);
	int g1=sim->createGroup("excit1", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	int g3=sim->createGroup("excit3", grid, EXCITATORY_NEURON);
	int g4=sim->createGroup("excit4", grid, EXCITATORY_NEURON);
	int g5=sim->createGroup("excit5", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g4, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->setNeuronParameters(g5, 0.02f, 0.2f, -65.0f, 8.0f);

	double prob = 0.2;
	int c0=sim->connect(g0,g0,"random",RangeWeight(0.1), prob, RangeDelay(1)); // full
	int c1=sim->connect(g1,g1,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(-1,0,0)); // all in x
	int c2=sim->connect(g2,g2,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(-1,0,-1)); // all in x and z
	int c3=sim->connect(g3,g3,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(2,3,0)); // 2D ellipse x,y
	int c4=sim->connect(g4,g4,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(0,2,3)); // 2D ellipse y,z
	int c5=sim->connect(g5,g5,"random",RangeWeight(0.1), prob, RangeDelay(1), RadiusRF(2,2,2)); // 3D ellipsoid

	sim->setupNetwork(); // need SETUP state for this function to work

	// from CpuSNN::connect: estimate max number of connections needed using binomial distribution
	// at 5 standard deviations
	int errorMargin = 5*sqrt(prob*(1-prob)*grid.N)+0.5;
	EXPECT_NEAR(sim->getNumSynapticConnections(c0), prob * grid.N * grid.N, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c1), prob * grid.N * grid.x, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c2), prob * grid.N * grid.x * grid.z, errorMargin);
	EXPECT_NEAR(sim->getNumSynapticConnections(c3), prob*144, errorMargin); // these numbers are less than what the 
	EXPECT_NEAR(sim->getNumSynapticConnections(c4), prob*224, errorMargin); // grid would say because of edge effects
	EXPECT_NEAR(sim->getNumSynapticConnections(c5), prob*320, errorMargin);

	delete sim;
}