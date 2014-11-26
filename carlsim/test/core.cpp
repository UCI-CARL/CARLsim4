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
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
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
	int g2=sim->createGroup("bananahama", Grid3D(1,2,3), INHIBITORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", Grid3D(2,3,4), EXCITATORY_NEURON);
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
	int g2=sim->createGroup("excit2", grid, EXCITATORY_NEURON);
	sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
	int g1=sim->createSpikeGeneratorGroup("excit", grid, EXCITATORY_NEURON);
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


// \TODO: using external current, make sure the Izhikevich model is correctly implemented
// Run izhikevich.org MATLAB script to find number of spikes as a function of neuron type,
// input current, and time period. Build test case to reproduce the exact numbers.

TEST(CORE, setExternalCurrent) {
	CARLsim * sim;
	int nNeur = 10;

	for (int hasCOBA=0; hasCOBA<=1; hasCOBA++) {
		for (int isGPUmode=0; isGPUmode<=1; isGPUmode++) {
			sim = new CARLsim("CORE.setExternalCurrent", isGPUmode?GPU_MODE:CPU_MODE, SILENT, 0, 42);
			int g1=sim->createGroup("excit1", nNeur, EXCITATORY_NEURON);
			sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);
			int g0=sim->createSpikeGeneratorGroup("input0", nNeur, EXCITATORY_NEURON);
			sim->connect(g0,g1,"full",RangeWeight(0.1),1.0f,RangeDelay(1));
			sim->setConductances(hasCOBA>0);
			sim->setupNetwork();
//			fprintf(stderr, "setExternalCurrent %s %s\n",hasCOBA?"COBA":"CUBA",isGPUmode?"GPU":"CPU");

			SpikeMonitor* SM = sim->setSpikeMonitor(g1,"NULL");

			// run for a bunch, observe zero spikes since ext current should be zero by default
			SM->startRecording();
			sim->runNetwork(1,0);
			SM->stopRecording();
			EXPECT_EQ(SM->getPopNumSpikes(), 0);

			// set current, observe spikes
			std::vector<float> current(nNeur,7.0f);
			sim->setExternalCurrent(g1, current);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

			// (intentionally) forget to reset current, observe spikes
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

			// reset current to zero
			sim->setExternalCurrent(g1, 0.0f);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_EQ(SM->getPopNumSpikes(), 0);

			// use convenience function to achieve same result as above
			sim->setExternalCurrent(g1, 7.0f);
			SM->startRecording();
			sim->runNetwork(0,500);
			SM->stopRecording();
			EXPECT_GT(SM->getPopNumSpikes(), 0); // should be >0 in all cases
			for (int i=0; i<nNeur; i++) {
				EXPECT_EQ(SM->getNeuronNumSpikes(i), 8); // but actually should be ==8
			}

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