#include "gtest/gtest.h"
#include "carlsim_tests.h"
#include <snn.h>
#include <periodic_spikegen.h>

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
	std::string name = "SNN";
	float STP_U = 0.25f;		// the exact values don't matter
	float STP_tF = 10.0f;
	float STP_tD = 15.0f;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,42);
        Grid3D neurGrid(10,1,1);
		int g1=sim->createGroup("excit", neurGrid, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setSTP(g1,true,STP_U,STP_tF,STP_tD);					// exact values matter

		group_info_t grpInfo = sim->getGroupInfo(g1);
		EXPECT_TRUE(grpInfo.WithSTP); 							// STP must be enabled
		EXPECT_FLOAT_EQ(grpInfo.STP_A,1.0f/STP_U);
		EXPECT_FLOAT_EQ(grpInfo.STP_U,STP_U); 					// check exact values
		EXPECT_FLOAT_EQ(grpInfo.STP_tau_u_inv,1.0f/STP_tF);
		EXPECT_FLOAT_EQ(grpInfo.STP_tau_x_inv,1.0f/STP_tD);
		delete sim;
	}
}

/*!
 * \brief testing setSTP to false
 * This function tests the information stored in the group info struct after disabling STP via setSTP
 */
TEST(STP, setSTPFalse) {
	std::string name="STP.setSTPFalse";
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,42);
        Grid3D neurGrid(10,1,1);
		int g1=sim->createGroup("excit", neurGrid, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setSTP(g1,false,0.1f,100,200); 					// exact values don't matter

		group_info_t grpInfo = sim->getGroupInfo(g1);
		EXPECT_FALSE(grpInfo.WithSTP);						// STP must be disabled
		delete sim;
	}
}


//! expect CARLsim to die if setSTP is called with silly params
TEST(STP, setSTPdeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("STP.setSTPdeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createSpikeGeneratorGroup("excit", 10, EXCITATORY_NEURON);

	// grpId
	EXPECT_DEATH({sim->setSTP(-2,true,0.1f,10,10);},"");

	// STP_U
	EXPECT_DEATH({sim->setSTP(g1,true,0.0f,10,10);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,1.1f,10,10);},"");

	// STP_tF / STP_tD
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,-10,10);},"");
	EXPECT_DEATH({sim->setSTP(g1,true,0.1f,10,-10);},"");
	
	delete sim;
}

//! test the effect of short-term depression
/*!
 * \brief test the effect short-term depression (STD) and short-term facilitation (STF) on post-rate
 *
 * This test ensures that STD and STF have the expected effect on post-synaptic firing rate.
 * A SpikeGenerator @ 10 Hz is connected to an excitatory post-neuron. First, STP is disabled, and the post-rate
 * is recorded. Then we turn on STD, and expect the firing rate to decrease. Then we turn on STF (instead of STD),
 * and expect the firing rate to increase.
 * However, if the stimulation period is short (isRunLong==0, runTimeMs=10 ms), then the firing rate should not
 * change at all, because the first spike under STP should not make a difference (due to the scaling of STP_A).
 * We perform this procedure in CUBA and COBA mode.
 * \TODO \FIXME: fix STP buffer and make sure test works for delays > 1 ms
 */
TEST(STP, firingRateSTDvsSTF) {
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP

	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int isRunLong=0; isRunLong<=1; isRunLong++) {
		for (int hasCOBA=0; hasCOBA<=1; hasCOBA++) {
			for (int isGPUmode=0; isGPUmode<=1; isGPUmode++) {
				// compare
				float rateG2noSTP = -1.0f;
				float rateG3noSTP = -1.0f;

				for (int hasSTP=0; hasSTP<=1; hasSTP++) {
					CARLsim* sim = new CARLsim("STP.firingRateSTDvsSTF",isGPUmode?GPU_MODE:CPU_MODE,SILENT,0,randSeed);
					int g2=sim->createGroup("STD", 1, EXCITATORY_NEURON);
					int g3=sim->createGroup("STF", 1, EXCITATORY_NEURON);
					sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
					sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
					int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
					int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);

					float wt = hasCOBA ? 0.2f : 18.0f;
					sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
					sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

					if (hasCOBA)
						sim->setConductances(true, 5, 0, 150, 6, 0, 150);
					else
						sim->setConductances(false);

					if (hasSTP) {
						sim->setSTP(g0, true, 0.45f, 50.0f,   750.0f); // depressive
						sim->setSTP(g1, true, 0.15f, 750.0f, 50.0f); // facilitative
					}

					bool spikeAtZero = true;
					spkGenG0 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g0, spkGenG0);
					spkGenG1 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
					sim->setSpikeGenerator(g1, spkGenG1);

					sim->setupNetwork();

					sim->setSpikeMonitor(g0,"NULL");
					sim->setSpikeMonitor(g1,"NULL");
					spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
					spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

					spkMonG2->startRecording();
					spkMonG3->startRecording();
					int runTimeMs = isRunLong ? 2000 : 100;
					sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
					spkMonG2->stopRecording();
					spkMonG3->stopRecording();

					if (!hasSTP) {
						// if STP is off: record spike rate, so that afterwards we can compare it to the one with STP
						// enabled
						rateG2noSTP = spkMonG2->getPopMeanFiringRate();
						rateG3noSTP = spkMonG3->getPopMeanFiringRate();
					} else {

/*
						fprintf(stderr,"%s %s %s, G2 w/o=%f, G2 w/=%f\n", isRunLong?"long":"short",
							isGPUmode?"GPU":"CPU",
							hasCOBA?"COBA":"CUBA",
							rateG2noSTP, spkMonG2->getPopMeanFiringRate());
						fprintf(stderr,"%s %s %s, G3 w/o=%f, G3 w/=%f\n", isRunLong?"long":"short",
							isGPUmode?"GPU":"CPU",
							hasCOBA?"COBA":"CUBA",
							rateG3noSTP,
							spkMonG3->getPopMeanFiringRate());
*/

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

					delete spkGenG0, spkGenG1;
					delete sim;
				}
			}
		}
	}
}

TEST(STP, spikeTimesCPUvsGPU) {
	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	int runTimeMs = 2000;
	std::vector<std::vector<int> > spkTG2CPU, spkTG3CPU, spkTG2GPU, spkTG3GPU;

	for (int hasCOBA=0; hasCOBA<=1; hasCOBA++) {
		// compare spike times cpu vs gpu

		for (int isGPUmode=0; isGPUmode<=1; isGPUmode++) {
			CARLsim* sim = new CARLsim("SNN",isGPUmode?GPU_MODE:CPU_MODE,SILENT,0,42);
			int g2=sim->createGroup("STD", 1, EXCITATORY_NEURON);
			int g3=sim->createGroup("STF", 1, EXCITATORY_NEURON);
			sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);
			int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
			int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);

			float wt = hasCOBA ? 0.2f : 18.0f;
			sim->connect(g0,g2,"one-to-one",RangeWeight(wt),1.0f,RangeDelay(1));
			sim->connect(g1,g3,"one-to-one",RangeWeight(wt),1.0f,RangeDelay(1));

			if (hasCOBA)
				sim->setConductances(true, 5, 0, 150, 6, 0, 150);
			else
				sim->setConductances(false);

			sim->setSTP(g0, true, 0.45f, 50.0f, 750.0f); // depressive
			sim->setSTP(g1, true, 0.15f, 750.0f, 50.0f); // facilitative

			bool spikeAtZero = true;
			spkGenG0 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
			sim->setSpikeGenerator(g0, spkGenG0);
			spkGenG1 = new PeriodicSpikeGenerator(10.0f,spikeAtZero); // periodic spiking @ 15 Hz
			sim->setSpikeGenerator(g1, spkGenG1);

			sim->setupNetwork();

			spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
			spkMonG3 = sim->setSpikeMonitor(g3,"NULL");
			spkMonG2->startRecording();
			spkMonG3->startRecording();
			sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
			spkMonG2->stopRecording();
			spkMonG3->stopRecording();

			if (!isGPUmode) {
				// in CPU mode, record spike times, so that we can compare them with GPU mode later on
				spkTG2CPU = spkMonG2->getSpikeVector2D();
				spkTG3CPU = spkMonG3->getSpikeVector2D();
			} else {
				// in GPU mode, compare spike times to recorded ones from CPU mode
				spkTG2GPU = spkMonG2->getSpikeVector2D();
				spkTG3GPU = spkMonG3->getSpikeVector2D();

				// assert so we skip the following for loop if sizes don't match
				ASSERT_EQ(spkTG2CPU[0].size(), spkTG2GPU[0].size());
				for (int i=0; i<spkTG2CPU[0].size(); i++)
					EXPECT_EQ(spkTG2CPU[0][i], spkTG2GPU[0][i]);

				ASSERT_EQ(spkTG3CPU[0].size(), spkTG3GPU[0].size());
				for (int i=0; i<spkTG3CPU[0].size(); i++)
					EXPECT_EQ(spkTG3CPU[0][i], spkTG3GPU[0][i]);
			}

			delete spkGenG0, spkGenG1;
			delete sim;
		}
	}
}
