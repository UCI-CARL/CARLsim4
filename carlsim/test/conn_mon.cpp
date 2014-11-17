#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <snn.h>
#include <periodic_spikegen.h>

// TODO: I should probably use a google tests figure for this to reduce the
// amount of redundant code, but I don't need to do that right now. -- KDC


class ConnectPropToPreNeurId : public ConnectionGenerator {
public:
	ConnectPropToPreNeurId(float wtScale) {
		wtScale_ = wtScale;
	}

	//! connection function, connect neuron i in scrGrp to neuron j in destGrp
	void connect(CARLsim* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay,
		bool& connected) {

		connected = true;
		delay = 1;
		weight = i*wtScale_;
	}

private:
	float wtScale_;
};


/// ****************************************************************************
/// TESTS FOR CONNECTION MONITOR 
/// ****************************************************************************

TEST(setConnMon, interfaceDeath) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	const int GRP_SIZE = 10;

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,42);
		
		int g0 = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON);
		sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

		// ----- CONFIG ------- //
		// calling setConnMon in CONFIG
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1);},"");

		// connect and advance to SETUP state
		sim->connect(g0,g1,"random",RangeWeight(0.1),0.1);
		sim->setConductances(false);
		sim->setupNetwork();

		// ----- SETUP ------- //
		// calling setConnMon on non-existent connection
		EXPECT_DEATH({sim->setConnectionMonitor(g1,g0);},"");

		// calling setConnMon twice on same group
		sim->setConnectionMonitor(g0,g1);
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1);},"");

		// advance to EXE state
		sim->runNetwork(1,0);

		// ----- EXE ------- //
		// calling setConnMon in EXE
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1);},"");
		
		delete sim;
	}
}

TEST(setConnMon, fname) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	const int GRP_SIZE = 10;
	
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("setConnMon.fname",mode?GPU_MODE:CPU_MODE,SILENT,0,42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);

		sim->connect(g1,g2,"random",RangeWeight(0.1),0.1);
		sim->setupNetwork();

		// this directory doesn't exist.
		EXPECT_DEATH({sim->setConnectionMonitor(g1,g2,"absentDirectory/testSpikes.dat");},"");
		
		delete sim;
	}
}

TEST(ConnMon, getters) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	ConnectPropToPreNeurId* connPre;

	const int GRP_SIZE_PRE = 10, GRP_SIZE_POST = 20;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,42);
		
		int g0 = sim->createGroup("g0", GRP_SIZE_PRE, EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE_POST, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE_POST, EXCITATORY_NEURON);
		sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

		connPre = new ConnectPropToPreNeurId(wtScale);
		int c0 = sim->connect(g0,g1,connPre,SYN_FIXED,1000,1000);
		sim->setupNetwork();

		ConnectionMonitor* CM = sim->setConnectionMonitor(g0,g1,"NULL");

		EXPECT_EQ(CM->getConnectId(),c0);
		EXPECT_EQ(CM->getFanIn(0),GRP_SIZE_PRE);
		EXPECT_EQ(CM->getFanOut(0),GRP_SIZE_POST);
		EXPECT_EQ(CM->getNumNeuronsPre(),GRP_SIZE_PRE);
		EXPECT_EQ(CM->getNumNeuronsPost(),GRP_SIZE_POST);
		EXPECT_EQ(CM->getNumSynapses(),GRP_SIZE_PRE*GRP_SIZE_POST);
		EXPECT_EQ(CM->getNumWeightsChanged(),0);
		EXPECT_FLOAT_EQ(CM->getPercentWeightsChanged(),0.0);
		EXPECT_EQ(CM->getTimeMsCurrentSnapshot(),0);
//		EXPECT_EQ(CM->getTimeMsLastSnapshot(),-1);
//		EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(),1);
//		EXPECT_FLOAT_EQ(CM->getTotalAbsWeightChange(),NAN);

		delete sim;
	}
}

TEST(ConnMon, takeSnapshot) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	ConnectPropToPreNeurId* connPre;

	const int GRP_SIZE = 10;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,42);
		
		int g0 = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON);
		sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

		connPre = new ConnectPropToPreNeurId(wtScale);
		sim->connect(g0,g1,connPre,SYN_FIXED,1000,1000);
		sim->setupNetwork();

		ConnectionMonitor* CM = sim->setConnectionMonitor(g0,g1,"NULL");

		std::vector< std::vector<float> > wt = CM->takeSnapshot();
		for (int i=0; i<GRP_SIZE; i++) {
			for (int j=0; j<GRP_SIZE; j++) {
				EXPECT_FALSE(isnan(wt[i][j]));
				EXPECT_FLOAT_EQ(wt[i][j], wtScale*i);
			}
		}
	}
}