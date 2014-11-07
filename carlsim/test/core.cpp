#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <snn.h>
#include <vector>
#include <periodic_spikegen.h>

/// **************************************************************************************************************** ///
/// CORE FUNCTIONALITY
/// **************************************************************************************************************** ///

//! check all possible (valid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinit) {
	CpuSNN* sim = NULL;
	std::string name = "CORE.CpuSNNinit";

	// Problem: The first two modes will print to stdout, and close it in the end; so all subsequent calls to sdout
	// via GTEST fail
	simMode_t simModes[2] = {CPU_MODE, GPU_MODE};
	loggerMode_t loggerModes[5] = {USER, DEVELOPER, SHOWTIME, SILENT, CUSTOM};
	for (int i=0; i<5; i++) {
		for (int j=0; j<2; j++) {
			int randSeed = rand() % 1000;
			sim = new CpuSNN(name,simModes[j],loggerModes[i],0,randSeed);

			EXPECT_EQ(sim->getNetworkName(),name);
			EXPECT_EQ(sim->getRandSeed(),randSeed);
			EXPECT_EQ(sim->getSimMode(),simModes[j]);
			EXPECT_EQ(sim->getLoggerMode(),loggerModes[i]);

			delete sim;
		}
	}

	sim = new CpuSNN(name,CPU_MODE,SILENT,0,0);
	EXPECT_EQ(sim->getRandSeed(),123);
	delete sim;

	// time(NULL)
	sim = new CpuSNN(name,CPU_MODE,SILENT,0,-1);
	EXPECT_NE(sim->getRandSeed(),-1);
	EXPECT_NE(sim->getRandSeed(),0);
	delete sim;
}

// FIXME: enabling the following generates a segfault
//! check all possible (invalid) ways of instantiating CpuSNN
TEST(CORE, CpuSNNinitDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CpuSNN* sim = NULL;
	std::string name="CORE.CpuSNNinitDeath";

	// sim mode
	EXPECT_DEATH({sim = new CpuSNN(name,UNKNOWN_SIM,USER,0,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// logger mode
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,UNKNOWN_LOGGER,0,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;

	// ithGPU
	EXPECT_DEATH({sim = new CpuSNN(name,CPU_MODE,USER,-1,42);},"");
	if (sim!=NULL) delete sim; sim = NULL;
}


TEST(CORE, getGroupGrid3D) {
	CARLsim* sim = new CARLsim("CORE.getGroupGrid3D",CPU_MODE,SILENT,0,42);
	Grid3D grid(2,3,4);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for this function to work

	for (int g=g1; g<g2; g++) {
		Grid3D getGrid = sim->getGroupGrid3D(g);
		EXPECT_EQ(getGrid.x, grid.x);
		EXPECT_EQ(getGrid.y, grid.y);
		EXPECT_EQ(getGrid.z, grid.z);
		EXPECT_EQ(getGrid.N, grid.N);
	}

	delete sim;
}

TEST(CORE, getGroupIdFromString) {
	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,0,42);
	int g1=sim->createSpikeGeneratorGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
	int g2=sim->createGroup("bananahama", Grid3D(1,2,3), INHIBITORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for this function to work

	EXPECT_EQ(sim->getGroupId("excit"), g1);
	EXPECT_EQ(sim->getGroupId("bananahama"), g2);
	EXPECT_EQ(sim->getGroupId("invalid group name"), -1); // group not found

	delete sim;
}


// This test creates a group on a grid and makes sure that the returned 3D location of each neuron is correct
TEST(CORE, getNeuronLocation3D) {
	CARLsim* sim = new CARLsim("Interface.createGroupDeath",CPU_MODE,SILENT,0,42);
	Grid3D grid(2,3,4);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	sim->connect(g1,g2,"full",RangeWeight(0.1), 1.0, RangeDelay(1));
	sim->setupNetwork(); // need SETUP state for getNeuronLocation3D to work

	// make sure the 3D location that is returned is correct
	for (int grp=0; grp<=1; grp++) {
		// do for both spike gen and RS group

		int x=0,y=0,z=0;
		for (int neurId=grp*grid.N; neurId<(grp+1)*grid.N; neurId++) {
			Point3D loc = sim->getNeuronLocation3D(neurId);
			EXPECT_EQ(loc.x, x);
			EXPECT_EQ(loc.y, y);
			EXPECT_EQ(loc.z, z);

			x++;
			if (x==grid.x) {
				x=0;
				y++;
			}
			if (y==grid.y) {
				x=0;
				y=0;
				z++;
			}
		}
	}

	delete sim;
}

// tests whether a point lies on a grid
TEST(CORE, isPoint3DonGrid) {
	CpuSNN snn("CORE.isPoint3DonGrid", CPU_MODE, SILENT, 0, 42);
	EXPECT_FALSE(snn.isPoint3DonGrid(Point3D(-1,-1,-1), Grid3D(10,5,2)));
	EXPECT_FALSE(snn.isPoint3DonGrid(Point3D(0.5,0.5,0.5), Grid3D(10,5,2)));
	EXPECT_FALSE(snn.isPoint3DonGrid(Point3D(10,5,2), Grid3D(10,5,2)));

	EXPECT_TRUE(snn.isPoint3DonGrid(Point3D(0,0,0), Grid3D(10,5,2)));
	EXPECT_TRUE(snn.isPoint3DonGrid(Point3D(0.0,0.0,0.0), Grid3D(10,5,2)));
	EXPECT_TRUE(snn.isPoint3DonGrid(Point3D(1,1,1), Grid3D(10,5,2)));
	EXPECT_TRUE(snn.isPoint3DonGrid(Point3D(9,4,1), Grid3D(10,5,2)));
	EXPECT_TRUE(snn.isPoint3DonGrid(Point3D(9.0,4.0,1.0), Grid3D(10,5,2)));
}

TEST(CORE, setConductancesTrue) {
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		int tdAMPA  = rand()%100 + 1;
		int trNMDA  = rand()%100 + 1;
		int tdNMDA  = rand()%100 + trNMDA + 1; // make sure it's larger than trNMDA
		int tdGABAa = rand()%100 + 1;
		int trGABAb = rand()%100 + 1;
		int tdGABAb = rand()%100 + trGABAb + 1; // make sure it's larger than trGABAb

		sim = new CpuSNN("CORE.setConductancesTrue",mode?GPU_MODE:CPU_MODE,SILENT,0,42);
		sim->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb);
		EXPECT_TRUE(sim->isSimulationWithCOBA());
		EXPECT_FALSE(sim->isSimulationWithCUBA());
//			EXPECT_FLOAT_EQ(sim->dAMPA,1.0f-1.0f/tdAMPA);
		if (trNMDA) {
			EXPECT_TRUE(sim->isSimulationWithNMDARise());
//				EXPECT_FLOAT_EQ(sim->rNMDA,1.0f-1.0f/trNMDA);
		} else {
			EXPECT_FALSE(sim->isSimulationWithNMDARise());
		}
//			EXPECT_FLOAT_EQ(sim->dNMDA,1.0f-1.0f/tdNMDA);
//			EXPECT_FLOAT_EQ(sim->dGABAa,1.0f-1.0f/tdGABAa);
		if (trGABAb) {
			EXPECT_TRUE(sim->isSimulationWithGABAbRise());
//				EXPECT_FLOAT_EQ(sim->rGABAb,1.0f-1.0f/trGABAb);
		} else {
			EXPECT_FALSE(sim->isSimulationWithGABAbRise());
		}
//			EXPECT_FLOAT_EQ(sim->dGABAb,1.0f-1.0f/tdGABAb);

		delete sim;
	}
}

// \TODO: set both mulSynFast and mulSynSlow to 0.0, observe no spiking

// \TODO: set mulSynSlow=0, have some pre-defined mulSynFast and check output rate via spkMonRT
// \TODO: set mulSynFast=0, have some pre-defined mulSynSlow and check output rate via spkMonRT

// \TODO: connect g0->g2 and g1->g2 with some pre-defined values, observe spike output

//! test all possible valid ways of setting conductances to true
// \FIXME: this could be interface level, but then there would be no way to test net_Info struct


TEST(CORE, firingRateCPUvsGPU) {
	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG0 = NULL, *spkMonG1 = NULL, *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	for (int hasCOBA=0; hasCOBA<=0; hasCOBA++) {
		float rateG0CPU = -1.0f;
		float rateG1CPU = -1.0f;
		float rateG2CPU = -1.0f;
		float rateG3CPU = -1.0f;
		std::vector<std::vector<int> > spkTimesG0, spkTimesG1, spkTimesG2, spkTimesG3;

		int runTimeMs = 1000;//rand() % 9500 + 500;
		float wt = hasCOBA ? 0.15f : 15.0f;

		PoissonRate in(1);

		for (int isGPUmode=0; isGPUmode<=0; isGPUmode++) {
			CARLsim* sim = new CARLsim("CORE.firingRateCPUvsGPU",isGPUmode?GPU_MODE:CPU_MODE,SILENT,0,42);
			int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
			int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);
			int g2=sim->createGroup("excit2", 1, EXCITATORY_NEURON);
			int g3=sim->createGroup("excit3", 1, EXCITATORY_NEURON);
			sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
			sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);

			sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
			sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

			if (hasCOBA)
				sim->setConductances(true,5, 0, 150, 6, 0, 150);
			else
				sim->setConductances(false);

			bool spikeAtZero = true;
			spkGenG0 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking
			sim->setSpikeGenerator(g0, spkGenG0);
			spkGenG1 = new PeriodicSpikeGenerator(50.0f,spikeAtZero); // periodic spiking
			sim->setSpikeGenerator(g1, spkGenG1);

			sim->setupNetwork();

//	for (int i=0;i<1;i++) in.rates[i] = 15;
//		sim->setSpikeRate(g0,&in);
//		sim->setSpikeRate(g1,&in);

			spkMonG0 = sim->setSpikeMonitor(g0,"NULL");
			spkMonG1 = sim->setSpikeMonitor(g1,"NULL");
			spkMonG2 = sim->setSpikeMonitor(g2,"NULL");
			spkMonG3 = sim->setSpikeMonitor(g3,"NULL");

			spkMonG0->startRecording();
			spkMonG1->startRecording();
			spkMonG2->startRecording();
			spkMonG3->startRecording();
			sim->runNetwork(runTimeMs/1000, runTimeMs%1000);
			spkMonG0->stopRecording();
			spkMonG1->stopRecording();
			spkMonG2->stopRecording();
			spkMonG3->stopRecording();

			if (!isGPUmode) {
				// CPU mode: record rates, so that we can compare them with GPU mode
				rateG0CPU = spkMonG0->getPopMeanFiringRate();
				rateG1CPU = spkMonG1->getPopMeanFiringRate();
				rateG2CPU = spkMonG2->getPopMeanFiringRate();
				rateG3CPU = spkMonG3->getPopMeanFiringRate();
				spkTimesG0 = spkMonG0->getSpikeVector2D();
				spkTimesG1 = spkMonG1->getSpikeVector2D();
				spkTimesG2 = spkMonG2->getSpikeVector2D();
				spkTimesG3 = spkMonG3->getSpikeVector2D();
//				for (int i=0; i<spkTimesG2[0].size(); i++)
//					fprintf(stderr, "%d\n",spkTimesG2[0][i]);
			} else {
				// GPU mode: compare rates to CPU mode
				ASSERT_FLOAT_EQ( spkMonG0->getPopMeanFiringRate(), rateG0CPU);
				ASSERT_FLOAT_EQ( spkMonG1->getPopMeanFiringRate(), rateG1CPU);
				ASSERT_FLOAT_EQ( spkMonG2->getPopMeanFiringRate(), rateG2CPU);
				ASSERT_FLOAT_EQ( spkMonG3->getPopMeanFiringRate(), rateG3CPU);

				std::vector<std::vector<int> > spkT = spkMonG2->getSpikeVector2D();
				ASSERT_EQ(spkTimesG2[0].size(), spkT[0].size());
				for (int i=0; i<spkTimesG2[0].size(); i++)
					EXPECT_EQ(spkTimesG2[0][i], spkT[0][i]);
//					fprintf(stderr, "%d\t%d\n",(i<spkTimesG2[0].size())?spkTimesG2[0][i]:-1, (i<spkT[0].size())?spkT[0][i]:-1);
			}

//			delete spkGenG0;
//			delete spkGenG1;
			delete sim;
		}
	}
}

// make sure bookkeeping for number of groups is correct during CONFIG
TEST(CORE, numGroups) {
	CARLsim sim("CORE.numGroups", CPU_MODE, SILENT, 0, 42);
	EXPECT_EQ(sim.getNumGroups(), 0);

	int nLoops = 4;
	int nNeur = 10;
	for (int i=0; i<nLoops; i++) {
		sim.createGroup("regexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+1);
		sim.createGroup("reginh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+2);
		sim.createSpikeGeneratorGroup("genexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+3);
		sim.createSpikeGeneratorGroup("geninh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumGroups(), i*4+4);
	}
}

// make sure bookkeeping for number of neurons is correct during CONFIG
TEST(CORE, numNeurons) {
	CARLsim sim("CORE.numNeurons", CPU_MODE, SILENT, 0, 42);
	EXPECT_EQ(sim.getNumNeurons(), 0);
	EXPECT_EQ(sim.getNumNeuronsRegExc(), 0);
	EXPECT_EQ(sim.getNumNeuronsRegInh(), 0);
	EXPECT_EQ(sim.getNumNeuronsGenExc(), 0);
	EXPECT_EQ(sim.getNumNeuronsGenInh(), 0);

	int nLoops = 4;
	int nNeur = 10;

	for (int i=0; i<nLoops; i++) {
		sim.createGroup("regexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumNeurons(), i*4*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegInh(), i*nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenExc(), i*nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenInh(), i*nNeur);
		EXPECT_EQ(sim.getNumNeurons(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh()
			+ sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
		EXPECT_EQ(sim.getNumNeuronsReg(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh());
		EXPECT_EQ(sim.getNumNeuronsGen(), sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());

		sim.createGroup("reginh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumNeurons(), i*4*nNeur + 2*nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegInh(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenExc(), i*nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenInh(), i*nNeur);
		EXPECT_EQ(sim.getNumNeurons(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh()
			+ sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
		EXPECT_EQ(sim.getNumNeuronsReg(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh());
		EXPECT_EQ(sim.getNumNeuronsGen(), sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());

		sim.createSpikeGeneratorGroup("genexc", nNeur, EXCITATORY_NEURON);
		EXPECT_EQ(sim.getNumNeurons(), i*4*nNeur + 3*nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegInh(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenInh(), i*nNeur);
		EXPECT_EQ(sim.getNumNeurons(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh()
			+ sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
		EXPECT_EQ(sim.getNumNeuronsReg(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh());
		EXPECT_EQ(sim.getNumNeuronsGen(), sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());

		sim.createSpikeGeneratorGroup("geninh", nNeur, INHIBITORY_NEURON);
		EXPECT_EQ(sim.getNumNeurons(), i*4*nNeur + 4*nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsRegInh(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenExc(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeuronsGenInh(), i*nNeur + nNeur);
		EXPECT_EQ(sim.getNumNeurons(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh()
			+ sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
		EXPECT_EQ(sim.getNumNeuronsReg(), sim.getNumNeuronsRegExc() + sim.getNumNeuronsRegInh());
		EXPECT_EQ(sim.getNumNeuronsGen(), sim.getNumNeuronsGenExc() + sim.getNumNeuronsGenInh());
	}
}