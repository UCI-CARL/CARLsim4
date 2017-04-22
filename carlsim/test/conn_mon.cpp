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
#include "carlsim_tests.h"
#include <periodic_spikegen.h>
#include "carlsim.h"
#include <fstream>                // std::ifstream
#include <cmath>


// connect such that the weight of the synapse is proportional to the pre-neuron ID
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
		maxWt = (net->getGroupNumNeurons(srcGrp)-1)*wtScale_;
		//printf("[%d,%d,%f,%f]\n", i, j, weight, maxWt);
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
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		int g0 = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON, 0);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, 0);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, 0);
		sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

		// ----- CONFIG ------- //
		// calling setConnMon in CONFIG
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1,"Default");},"");

		// connect and advance to SETUP state
		sim->connect(g0,g1,"full",RangeWeight(0.1),1.0);
		sim->setConductances(false);
		sim->setupNetwork();

		// ----- SETUP ------- //
		// calling setConnMon on non-existent connection
		EXPECT_DEATH({sim->setConnectionMonitor(g1,g0,"Default");},"");

		// calling setConnMon twice on same group
		sim->setConnectionMonitor(g0,g1,"Default");
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1,"Default");},"");

		// advance to EXE state
		sim->runNetwork(1,0);

		// ----- EXE ------- //
		// calling setConnMon in EXE
		EXPECT_DEATH({sim->setConnectionMonitor(g0,g1,"Default");},"");

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
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("setConnMon.fname",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, 0);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, 0);
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
	std::vector<ConnectPropToPreNeurId*> connPre(4, NULL);
	std::vector<short int> connId(4, -1);
	std::vector<int> grpId(2, -1);

	std::vector<int> grpSize(2, -1);
	grpSize[0] = 10;
	grpSize[1] = 20;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		grpId[0] = sim->createGroup("g0", grpSize[0], EXCITATORY_NEURON, 0);
		grpId[1] = sim->createGroup("g1", grpSize[1], INHIBITORY_NEURON, 0);
		sim->setNeuronParameters(grpId[0], 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(grpId[1], 0.1f, 0.2f, -65.0f, 2.0f);

		for (int i=0; i<4; i++) {
			// the sign of the weight is not important: it will be corrected
			connPre[i] = new ConnectPropToPreNeurId(-wtScale);
		}

		int cid = 0;
		for (int gPre=0; gPre<=1; gPre++) {
			for (int gPost=0; gPost<=1; gPost++, cid++) {
				// exc to exc, exc to inh, inh to exc, inh to inh
				connId[cid] = sim->connect(grpId[gPre], grpId[gPost], connPre[cid], SYN_FIXED, 1000, 1000);
			}
		}
		sim->setConductances(true);
		sim->setupNetwork();

		cid = 0;
		for (int gPre=0; gPre<=1; gPre++) {
			for (int gPost=0; gPost<=1; gPost++, cid++) {
				ConnectionMonitor* CM = sim->setConnectionMonitor(grpId[gPre], grpId[gPost], "NULL");

				EXPECT_EQ(CM->getConnectId(),connId[cid]);
				EXPECT_EQ(CM->getFanIn(0),grpSize[gPre]);
				EXPECT_EQ(CM->getFanOut(0),grpSize[gPost]);
				EXPECT_EQ(CM->getNumNeuronsPre(),grpSize[gPre]);
				EXPECT_EQ(CM->getNumNeuronsPost(),grpSize[gPost]);
				EXPECT_EQ(CM->getNumSynapses(),grpSize[gPre]*grpSize[gPost]);
				EXPECT_EQ(CM->getNumWeightsChanged(),0);
				EXPECT_FLOAT_EQ(CM->getPercentWeightsChanged(),0.0f);
				EXPECT_EQ(CM->getTimeMsCurrentSnapshot(),0);

				EXPECT_FLOAT_EQ(CM->getMinWeight(false),0.0f);
				EXPECT_FLOAT_EQ(CM->getMinWeight(true),0.0f);
				EXPECT_FLOAT_EQ(CM->getMaxWeight(false),(grpSize[gPre]-1)*wtScale);
				EXPECT_FLOAT_EQ(CM->getMaxWeight(true),(grpSize[gPre]-1)*wtScale);

				EXPECT_EQ(CM->getNumWeightsInRange(CM->getMinWeight(false),CM->getMaxWeight(false)), grpSize[gPre]*grpSize[gPost]);
				EXPECT_EQ(CM->getNumWeightsInRange(0.0, 0.0), grpSize[gPost]);
				EXPECT_EQ(CM->getNumWeightsInRange(wtScale, 2*wtScale), 2*grpSize[gPost]);
				EXPECT_EQ(CM->getNumWeightsInRange(CM->getMaxWeight(false)*1.01, CM->getMaxWeight(false)*2), 0);
				EXPECT_EQ(CM->getNumWeightsWithValue(0.0), grpSize[gPost]);
				EXPECT_EQ(CM->getNumWeightsWithValue(wtScale), grpSize[gPost]);

				EXPECT_FLOAT_EQ(CM->getPercentWeightsInRange(CM->getMinWeight(false),CM->getMaxWeight(false)), 100.0);
				EXPECT_FLOAT_EQ(CM->getPercentWeightsInRange(0.0, 0.0), grpSize[gPost]*100.0/CM->getNumSynapses());
				EXPECT_FLOAT_EQ(CM->getPercentWeightsInRange(wtScale, 2*wtScale), 2*grpSize[gPost]*100.0/CM->getNumSynapses());
				EXPECT_FLOAT_EQ(CM->getPercentWeightsWithValue(0.0), grpSize[gPost]*100.0/CM->getNumSynapses());
				EXPECT_FLOAT_EQ(CM->getPercentWeightsWithValue(wtScale), grpSize[gPost]*100.0/CM->getNumSynapses());
			}
		}
		delete sim;
	}
}

TEST(ConnMon, takeSnapshot) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	std::vector<ConnectPropToPreNeurId*> connPre(4, NULL);
	std::vector<short int> connId(4, -1);
	std::vector<int> grpId(2, -1);

	const int GRP_SIZE = 10;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		grpId[0] = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON, 0);
		grpId[1] = sim->createGroup("g1", GRP_SIZE, INHIBITORY_NEURON, 0);
		sim->setNeuronParameters(grpId[0], 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(grpId[1], 0.1f, 0.2f, -65.0f, 2.0f);

		for (int i=0; i<4; i++) {
			// the sign of the weight is not important: it will be corrected
			connPre[i] = new ConnectPropToPreNeurId(-wtScale);
		}

		int cid = 0;
		for (int gPre=0; gPre<=1; gPre++) {
			for (int gPost=0; gPost<=1; gPost++, cid++) {
				// exc to exc, exc to inh, inh to exc, inh to inh
				connId[cid] = sim->connect(grpId[gPre], grpId[gPost], connPre[cid], SYN_FIXED, 1000, 1000);
			}
		}
		sim->setConductances(true);
		sim->setupNetwork();

		cid = 0;
		for (int gPre=0; gPre<=1; gPre++) {
			for (int gPost=0; gPost<=1; gPost++, cid++) {
				ConnectionMonitor* CM = sim->setConnectionMonitor(grpId[gPre], grpId[gPost], "NULL");
				std::vector< std::vector<float> > wt = CM->takeSnapshot();
				for (int i=0; i<GRP_SIZE; i++) {
					for (int j=0; j<GRP_SIZE; j++) {
						EXPECT_FALSE(isnan(wt[i][j]));
						EXPECT_FLOAT_EQ(wt[i][j], wtScale*i);
					}
				}
			}
		}

		delete sim;
	}
}

TEST(ConnMon, weightFile) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;

	const int GRP_SIZE = 10;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// loop over time interval options
		long fileLength[3] = {0,0,0};
		for (int interval=-1; interval<=3; interval+=2) {
			sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

			int g0 = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON, 0);
			sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);

			sim->connect(g0,g0,"full",RangeWeight(0.1f),0.1f);
			sim->setConductances(true);
			sim->setupNetwork();

			ConnectionMonitor* CM = sim->setConnectionMonitor(g0,g0,"results/weights.dat");
			CM->setUpdateTimeIntervalSec(interval);
			if (interval==-1) {
				sim->runNetwork(10,0);
			} else {
				// taking a snapshot in the beginning should not matter, because that snapshot is already
				// being recorded automatically
				CM->takeSnapshot();
				sim->runNetwork(6,0);

				// taking additional snapshots should not matter either
				CM->takeSnapshot();
				CM->takeSnapshot();
				sim->runNetwork(4,200);
			}

			delete sim;

			// make sure file size for CM binary is correct
			std::ifstream wtFile("results/weights.dat", std::ios::binary | std::ios::ate);
			EXPECT_TRUE(wtFile.is_open());
			if (wtFile) {
				wtFile.seekg( 0, std::ios::end );
				if (interval==-1) {
					fileLength[0] = wtFile.tellg(); // should contain 0 snapshots
				} else if (interval==1) {
					fileLength[1] = wtFile.tellg(); // should contain 11 snapshots
				} else {
					fileLength[2] = wtFile.tellg(); // should contain 5 snapshots
				}
			}
		}

		// we want to check the file size, but that might vary depending on the header section
		// (which might change over the course of time)
		// so choose a portable approach: estimate header size for both interval modes, and make
		// sure they're the same
		// file should have header+(number of snapshots)*((number of weights)+(timestamp as long int))*(bytes per word)

		// if interval==-1: no snapshots in the file
		int headerSize = fileLength[0] - 0*(GRP_SIZE*GRP_SIZE+2)*4;

		// if interval==1: 11 snapshots from t = 0, 1, 2, ..., 10 sec plus one from 10.200 sec
		EXPECT_EQ(headerSize, fileLength[1] - 12*(GRP_SIZE*GRP_SIZE+2)*4);

		// if interval==3: 4 snapshots from t = 0, 3, 6, 9, plus one from 10.200 sec
		EXPECT_EQ(headerSize, fileLength[2] - 5*(GRP_SIZE*GRP_SIZE+2)*4);
	}
}

TEST(ConnMon, weightChange) {
	// set this flag to make all death tests thread-safe
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	ConnectPropToPreNeurId* connPre;

	const int GRP_SIZE = 10;
	float wtScale = 0.01f;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		sim = new CARLsim("ConnMon.setConnectionMonitorDeath",mode?GPU_MODE:CPU_MODE,SILENT,1,42);

		int g0 = sim->createGroup("g0", GRP_SIZE, EXCITATORY_NEURON, 0);
		sim->setNeuronParameters(g0, 0.02f, 0.2f, -65.0f, 8.0f);

		short int c0 = sim->connect(g0,g0,"full",RangeWeight(wtScale),0.1f,RangeDelay(1),RadiusRF(-1),SYN_PLASTIC);
		sim->setConductances(true);
		sim->setupNetwork();

		// take snapshot at beginning
		ConnectionMonitor* CM = sim->setConnectionMonitor(g0, g0, "NULL");
		CM->takeSnapshot();

		// run for some time, make sure no weights changed (because there is no plasticity)
		sim->runNetwork(0,500);
		sim->runNetwork(1,0);
		EXPECT_FLOAT_EQ(CM->getTotalAbsWeightChange(), 0.0f);
		EXPECT_EQ(CM->getTimeMsCurrentSnapshot(), 1500);
		EXPECT_EQ(CM->getTimeMsLastSnapshot(), 0);
		EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(), 1500);

		// set all weights to zero
		sim->scaleWeights(c0, 0.0f);

		// Run for some time, now SNN::updateConnectionMonitor will be called, but MUST NOT
		// interfere with the takeSnapshot method.
		// So we expect the weight change to be from wtScale (at t=0.5s) to 0 (at t=1.5s), not from
		// 0 (at t=1.0s) to 0 (at t=1.5s).
		sim->runNetwork(1,0);
		EXPECT_FLOAT_EQ(CM->getTotalAbsWeightChange(), wtScale*GRP_SIZE*GRP_SIZE);
		EXPECT_EQ(CM->getTimeMsCurrentSnapshot(), 2500);
		EXPECT_EQ(CM->getTimeMsLastSnapshot(), 1500);
		EXPECT_EQ(CM->getTimeMsSinceLastSnapshot(), 1000);

		// If we call another weight method, then ConnectionMonitorCore::updateStoredWeights should not
		// update the weight matrices. Instead it should operate on the same time interval as above,
		// effectively giving the same result.
		std::vector< std::vector<float> > wtChange = CM->calcWeightChanges();
		for (int i=0; i<GRP_SIZE; i++) {
			for (int j=0; j<GRP_SIZE; j++) {
				EXPECT_FLOAT_EQ(wtChange[i][j], -wtScale);
			}
		}

		EXPECT_EQ(CM->getNumWeightsChanged(), GRP_SIZE*GRP_SIZE);
		EXPECT_FLOAT_EQ(CM->getPercentWeightsChanged(), 100.0f);

		delete sim;
	}
}
