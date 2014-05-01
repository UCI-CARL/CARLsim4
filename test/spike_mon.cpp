#include <snn.h>

#include "carlsim_tests.h"

/// ****************************************************************************
/// TESTS FOR SPIKE MONITOR 
/// ****************************************************************************

/*!
 * \brief testing to make sure grpId error is caught in setSpikeMonitor.
 *
 */
TEST(SPIKEMON, grpId){
	CARLsim* sim;
	const int GRP_SIZE = 10;
	
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		
		EXPECT_DEATH(sim->setSpikeMonitor(ALL),"");  // grpId = ALL (-1) and less than 0 
		EXPECT_DEATH(sim->setSpikeMonitor(-4),"");  // less than 0
		EXPECT_DEATH(sim->setSpikeMonitor(2),""); // greater than number of groups
		EXPECT_DEATH(sim->setSpikeMonitor(MAX_GRP_PER_SNN),""); // greater than number of group & and greater than max groups
		
		delete sim;
	}
}

/*!
 * \brief testing to make sure configId error is caught in setSpikeMonitor.
 *
 */
TEST(SPIKEMON, configId){
	CARLsim* sim;
	const int GRP_SIZE = 10;
	
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		
		EXPECT_DEATH(sim->setSpikeMonitor(1,"testSpikes.dat",ALL),"");  // configId = ALL (-1) and less than 0 
		EXPECT_DEATH(sim->setSpikeMonitor(1,"testSpikes.dat",-2),"");  // less than 0
		EXPECT_DEATH(sim->setSpikeMonitor(1,"testSpikes.dat",-100),"");  // less than 0

		delete sim;
	}
}


/*!
 * \brief testing to make sure file name error is caught in setSpikeMonitor.
 *
 */
TEST(SPIKEMON, fname){
	CARLsim* sim;
	const int GRP_SIZE = 10;
	
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);
		
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		// this directory doesn't exist.
		EXPECT_DEATH(sim->setSpikeMonitor(1,"absentDirectory/testSpikes.dat",0),"");  
		
		delete sim;
	}
}

/// ****************************************************************************
/// TESTS FOR SPIKE-INFO CLASS
/// ****************************************************************************

/*!
 * \brief testing to make sure clear() function works.
 *
 */
TEST(SPIKEINFO, clear){
	CARLsim* sim;
	PoissonRate* input;
	const int GRP_SIZE = 10;
	const int inputTargetFR = 5.0f;
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);

		float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
		int inputGroup = sim->createSpikeGeneratorGroup("Input",GRP_SIZE,EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		int g2 = sim->createGroup("g2", GRP_SIZE, EXCITATORY_NEURON, ALL);
		
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		double initWeight = 0.05f;
		double maxWeight = 4*initWeight;

		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		// input
		input = new PoissonRate(GRP_SIZE);
		for(int i=0;i<GRP_SIZE;i++){
			input->rates[i]=inputTargetFR;
		}
		sim->connect(inputGroup,g1,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
		sim->connect(inputGroup,g2,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
		sim->connect(g1,g2,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);

		SpikeInfo* spikeInfoG1 = sim->setSpikeMonitor(g1);
		
		sim->setSpikeRate(inputGroup,input);
		
		spikeInfoG1->startRecording();
		
		int runTime = 1;
		// run the network
		sim->runNetwork(runTime,0);
	
		spikeInfoG1->stopRecording();
		
		// we should have spikes!
		EXPECT_TRUE(spikeInfoG1->getSize() != 0);
		
		// now clear the spikes
		spikeInfoG1->clear();

		// we shouldn't have spikes!
		EXPECT_TRUE(spikeInfoG1->getSize() == 0);

		delete sim;
		delete input;
	}
}
/*!
 * \brief testing to make sure getGrpFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getMaxFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getMinFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getNeuronFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getNumNeuronsWithFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getNumSilentNeurons() function works.
 *
 */

/*!
 * \brief testing to make sure getPercentNeuronsWithFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure getPercentSilentNeurons() function works.
 *
 */

/*!
 * \brief testing to make sure getGrpSize() function works.
 *
 */

/*!
 * \brief testing to make sure getSortedNeuronFiringRate() function works.
 *
 */

/*!
 * \brief testing to make sure isRecording() function works.
 *
 */


/*!
 * \brief testing to make sure print() function works.
 *
 */


/*!
 * \brief testing to make sure startRecording() function works.
 *
 */

/*!
 * \brief testing to make sure stopRecording() function works.
 *
 */


