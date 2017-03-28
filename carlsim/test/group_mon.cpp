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

#include <carlsim.h>
#include <periodic_spikegen.h>
#include <snn_definitions.h> // MAX_GRP_PER_SNN

// TODO: I should probably use a google tests figure for this to reduce the
// amount of redundant code, but I don't need to do that right now. -- KDC

/// ****************************************************************************
/// TESTS FOR SET GROUP MON 
/// ****************************************************************************

/*!
 * \brief testing to make sure grpId error is caught in setGroupMonitor.
 *
 */
TEST(setGroupMon, grpId){
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	const int GRP_SIZE = 10;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("setGroupMon.grpId", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		
		EXPECT_DEATH(sim->setGroupMonitor(ALL, "Default"),"");  // grpId = ALL (-1) and less than 0 
		EXPECT_DEATH(sim->setGroupMonitor(-4, "Default"),"");  // less than 0
		EXPECT_DEATH(sim->setGroupMonitor(2, "Default"),""); // greater than number of groups
		EXPECT_DEATH(sim->setGroupMonitor(MAX_GRP_PER_SNN, "Default"),""); // greater than number of group & and greater than max groups
		
		delete sim;
	}
}

/*!
 * \brief testing to make sure file name error is caught in setGroupMonitor.
 *
 */
TEST(setGroupMon, fname){
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim;
	const int GRP_SIZE = 10;

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("setGroupMon.fname", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);

		// this directory doesn't exist.
		EXPECT_DEATH(sim->setGroupMonitor(1, "absentDirectory/testSpikes.dat"),"");  
		
		delete sim;
	}
}

TEST(GroupMon, interfaceDeath) {
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("GroupMon.interfaceDeath", CPU_MODE, SILENT, 1, 42);

	int g1 = sim->createGroup("g1", 5, EXCITATORY_NEURON);		
	sim->setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);

	int g0 = sim->createSpikeGeneratorGroup("Input", 5, EXCITATORY_NEURON);
	GroupMonitor* grpMon = sim->setGroupMonitor(g0, "NULL");

	sim->setConductances(true);

	sim->connect(g0, g1 , "random", RangeWeight(0.01), 0.5f);

	// call setSpikeMonitor again on group, should fail
	EXPECT_DEATH({sim->setGroupMonitor(g0, "NULL");},"");

	// set up network and test all API calls that are not valid in certain modes
	sim->setupNetwork();

	// test all APIs that cannot be called when recording is on
	grpMon->startRecording();
	EXPECT_DEATH(grpMon->getDataVector(),"");
	EXPECT_DEATH(grpMon->getTimeVector(),"");
	EXPECT_DEATH(grpMon->getPeakValueVector(),"");
	EXPECT_DEATH(grpMon->getPeakTimeVector(),"");
	EXPECT_DEATH(grpMon->getSortedPeakTimeVector(),"");
	EXPECT_DEATH(grpMon->getSortedPeakValueVector(),"");
	EXPECT_DEATH(grpMon->startRecording(),"");

	delete sim;
}

TEST(GroupMon, persistentMode) {
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("GroupMon.persistentMode", CPU_MODE, SILENT, 1, 42);

	int g1 = sim->createGroup("g1", 5, EXCITATORY_NEURON);		
	sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g0 = sim->createSpikeGeneratorGroup("Input", 5, EXCITATORY_NEURON);
	GroupMonitor* grpMon = sim->setGroupMonitor(g0, "NULL");

	sim->setConductances(true);

	sim->connect(g0, g1, "random", RangeWeight(0.01), 0.5f);

	sim->setupNetwork();

	// run for half a second, then check recording getters
	grpMon->startRecording();
	sim->runNetwork(0,500);
	grpMon->stopRecording();
	EXPECT_EQ(grpMon->getRecordingTotalTime(), 500);
	EXPECT_EQ(grpMon->getRecordingStartTime(), 0);
	EXPECT_EQ(grpMon->getRecordingStopTime(), 500);

	// run for half a second, then check recording getters
	// persistent mode should be off, so only the last probe should matter
	grpMon->startRecording();
	sim->runNetwork(0,500);
	grpMon->stopRecording();
	EXPECT_EQ(grpMon->getRecordingTotalTime(), 500);
	EXPECT_EQ(grpMon->getRecordingStartTime(), 500);
	EXPECT_EQ(grpMon->getRecordingStopTime(), 1000);

	// now switch persistent mode on
	grpMon->setPersistentData(true);

	// run for half a second, and expect persistent mode on
	// start should now be what it was (500), and total time should have increased by 500
	grpMon->startRecording();
	sim->runNetwork(0,500);
	grpMon->stopRecording();
	EXPECT_EQ(grpMon->getRecordingTotalTime(), 1000);
	EXPECT_EQ(grpMon->getRecordingStartTime(), 500);
	EXPECT_EQ(grpMon->getRecordingStopTime(), 1500);

	delete sim;
}

/*
 * This test verifies that the group data (only support dopamine concentration for now) match the analytic solution.
 * A PeriodicSpikeGenerator is used to periodically generate spikes, which allows us to know the exact spike times
 * and therefore dopamine concentratoin of post-synaptic group. We run the simulation for 1 second and get the group
 * data vector thtrough GroupMonitor object. At every time step, we expect the DA value to be exactly what it should be.
 */
TEST(GroupMon, peakTimeAndValue) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for (int mode = 0; mode < TESTED_MODES; mode++) {
		// first iteration, test CPU mode, second test GPU mode
		CARLsim* sim = new CARLsim("GroupMon.peakTimeAndValue", mode?GPU_MODE:CPU_MODE, SILENT, 1, 42);
		//CARLsim* sim = new CARLsim("GroupMon.peakTimeAndValue", CPU_MODE, SILENT, 1, 42);
		float tAMPA = 5.0, tNMDA = 150.0, tGABAa = 6.0, tGABAb = 150.0;
		int g1 = sim->createGroup("g1", 10, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);

		int g0 = sim->createSpikeGeneratorGroup("Input", 10, DOPAMINERGIC_NEURON);

		sim->setConductances(true, tAMPA, tNMDA, tGABAa, tGABAb);
		// we are testing dopamine values of the post-synaptic group, not spikes. Set weight close to zero
		sim->connect(g0, g1, "one-to-one", RangeWeight(1.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

		sim->setESTDP(g1, true, DA_MOD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

		sim->setNeuromodulator(ALL);

		// use periodic spike generator to know the exact dopamine delivery
		PeriodicSpikeGenerator* spkGen = new PeriodicSpikeGenerator(10, false);
		sim->setSpikeGenerator(g0, spkGen);

		sim->setupNetwork();

		// write all spikes to file
		GroupMonitor* groupMon = sim->setGroupMonitor(g1, "NULL");

		groupMon->startRecording();
		sim->runNetwork(1, 0);
		groupMon->stopRecording();

		// get the timestamps of peaks in group data
		std::vector<int> timeVector = groupMon->getPeakTimeVector();
		for (int i = 0; i < timeVector.size(); i++) {
			EXPECT_EQ(timeVector[i], (i+1) * 100); // the peaks shoul be at 100, 200, 300 ... ,900 (ms)
			//printf("peak time:%d %d\n", timeVector[i], (i + 1) * 100);
		}

		// compare all group data and to analytic solution
		std::vector<float> dataVector = groupMon->getDataVector();
		float da = 1.0f; // default dopamine inital value
		float decay = 1.0f-1.0f/100.0f; // default doapmine decay
		for (int t = 0; t < 1000; t++) {
			if (da > 1.0f)
				da *= decay;
			if (t > 0 && t % 100 == 0) {
				// There are 10 synapses, we expect 10 spikes very 100ms (except 0ms)
				da += 10 /*number of syanpses*/ * 0.04f /*dopamine dose per spike*/;
			}

			//printf("(%d,%f,%f)", t, dataVector[t], da);
			EXPECT_NEAR(dataVector[t], da, 0.02f);
			//printf("DA value: %f %f\n", dataVector[t], da);
		}

		delete spkGen;
		delete sim;
	}
}
