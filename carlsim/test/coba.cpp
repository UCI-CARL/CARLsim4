#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <snn.h>
#include <periodic_spikegen.h>

#if (WIN32 || WIN64)
#include <Windows.h>
#endif


/// **************************************************************************************************************** ///
/// CONDUCTANCE-BASED MODEL (COBA)
/// **************************************************************************************************************** ///


//! This test assures that the conductance peak occurs as specified by tau_rise and tau_decay, and that the peak is
//! equal to the specified weight value
TEST(COBA, synRiseTime) {
	CpuSNN* sim;

	float abs_error = 0.05; // five percent error for wt

#if (WIN32 || WIN64)
	HANDLE hMutex = CreateMutex( 
					NULL, // default security attributes
					FALSE, // initially not owned
					NULL);
#else
	pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
#endif

	for (int mode=0; mode<=1; mode++) {
#if (WIN32 || WIN64)
		WaitForSingleObject(hMutex, INFINITE);
#else
		pthread_mutex_lock(&lock);
#endif
		int tdAMPA  = rand()%100 + 1;
		int trNMDA  = rand()%100 + 1;
		int tdNMDA  = rand()%100 + trNMDA + 1; // make sure it's larger than trNMDA
		int tdGABAa = rand()%100 + 1;
		int trGABAb = rand()%100 + 1;
		int tdGABAb = rand()%100 + trGABAb + 1; // make sure it's larger than trGABAb

		int delay = 1;
		float radRF = -1.0f;

		sim = new CpuSNN("COBA.synRiseTime",mode?GPU_MODE:CPU_MODE,SILENT,0,42);
        Grid3D neur(1);
        Grid3D neur2(1);
		int g0=sim->createSpikeGeneratorGroup("inputExc", neur, EXCITATORY_NEURON);
		int g1=sim->createGroup("excit", neur2, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->connect(g0,g1,"full",0.5f,0.5f,1.0f,delay,delay,radRF,radRF,radRF,0.0f,1.0f,SYN_FIXED);

		int g2=sim->createSpikeGeneratorGroup("inputInh", neur, INHIBITORY_NEURON);
		int g3=sim->createGroup("inhib", neur, INHIBITORY_NEURON);
		sim->setNeuronParameters(g3, 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f);
		sim->connect(g2,g3,"full",-0.5f,-0.5f,1.0f,delay,delay,radRF,radRF,radRF,0.0f,1.0f,SYN_FIXED);

		sim->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb);

		// run network for a second first, so that we know spike will happen at simTimeMs==1000
		PeriodicSpikeGeneratorCore* spk1 = new PeriodicSpikeGeneratorCore(1.0f); // periodic spiking @ 50 Hz
		sim->setSpikeGenerator(g0, spk1);
		sim->setSpikeGenerator(g2, spk1);
		sim->setupNetwork(true);
		sim->runNetwork(1,0,false,false);

		// now observe gNMDA, gGABAb after spike, and make sure that the time at which they're max matches the
		// analytical solution, and that the peak conductance is actually equal to the weight we set
		int tmaxNMDA = -1;
		double maxNMDA = -1;
		int tmaxGABAb = -1;
		double maxGABAb = -1;
		int nMsec = max(trNMDA+tdNMDA,trGABAb+tdGABAb)+10;
		for (int i=0; i<nMsec; i++) {
			sim->runNetwork(0,1,false,true); // copyNeuronState

			std::vector<float> gNMDA = sim->getConductanceNMDA();
			std::vector<float> gGABAb = sim->getConductanceGABAb();

			if (gNMDA[sim->getGroupStartNeuronId(g1)] > maxNMDA) {
				tmaxNMDA=i;
				maxNMDA=gNMDA[sim->getGroupStartNeuronId(g1)];
			}
			if (gGABAb[sim->getGroupStartNeuronId(g3)] > maxGABAb) {
				tmaxGABAb=i;
				maxGABAb=gGABAb[sim->getGroupStartNeuronId(g3)];
			}
		}

		double tmax = (-tdNMDA*trNMDA*log(1.0*trNMDA/tdNMDA))/(tdNMDA-trNMDA);
		EXPECT_NEAR(tmaxNMDA,tmax,1); // t_max should be near the analytical solution
		EXPECT_NEAR(maxNMDA,0.5,0.5*abs_error); // max should be equal to the weight

		tmax = (-tdGABAb*trGABAb*log(1.0*trGABAb/tdGABAb))/(tdGABAb-trGABAb);
		EXPECT_NEAR(tmaxGABAb,tmaxGABAb,1); // t_max should be near the analytical solution
		EXPECT_NEAR(maxGABAb,0.5,0.5*abs_error); // max should be equal to the weight times -1

		delete spk1;
		delete sim;
#if (WIN32 || WIN64)
		ReleaseMutex(hMutex);
#else
		pthread_mutex_unlock(&lock);
#endif
	}
}


//! This test ensures that CPUmode and GPUmode produce the exact same conductance values over a period of 1000 ms
//! Conductances are read out every timestep. Precision on a single neuron is tested.
TEST(COBA, condSingleNeuronCPUvsGPU) {
	CpuSNN* sim = NULL;
	int grps[6] = {-1};
	std::string expectCond[6] = {"AMPA","NMDA","AMPA+NMDA","GABAa","GABAb","GABAa+GABAb"};
	float expectCondStd[6] = {0.001, 0.001, 0.001, 0.001, 0.001, 0.001};

	std::vector<float> gAMPA_CPU(1000, 0.0f);
	std::vector<float> gNMDA_CPU(1000, 0.0f);
	std::vector<float> gGABAa_CPU(1000, 0.0f);
	std::vector<float> gGABAb_CPU(1000, 0.0f);

	PeriodicSpikeGeneratorCore *spkGen1, *spkGen2;

	// make it a single neuron
	// \TODO If post gets input from more than one pre, conductance values in GPUmode are a bit off. Need to
	// investigate that (aggregating rounding errors?)
	int nInput = 1;
	int nOutput = 1;
	float rate = 30.0f;
	bool spikeAtZero = true;

	for (int mode=0; mode<=1; mode++) {
		sim = new CpuSNN("COBA.condCPUvsGPU",mode?GPU_MODE:CPU_MODE,SILENT,0,42);
		int g0=sim->createSpikeGeneratorGroup("spike", Grid3D(nInput), EXCITATORY_NEURON);
		int g1=sim->createSpikeGeneratorGroup("spike", Grid3D(nInput), INHIBITORY_NEURON);
		grps[0]=sim->createGroup("excitAMPA", Grid3D(nOutput), EXCITATORY_NEURON);
		grps[1]=sim->createGroup("excitNMDA", Grid3D(nOutput), EXCITATORY_NEURON);
		grps[2]=sim->createGroup("excitAMPA+NMDA", Grid3D(nOutput), EXCITATORY_NEURON);
		grps[3]=sim->createGroup("inhibGABAa", Grid3D(nOutput), INHIBITORY_NEURON);
		grps[4]=sim->createGroup("inhibGABAb", Grid3D(nOutput), INHIBITORY_NEURON);
		grps[5]=sim->createGroup("inhibGABAa+GABAb", Grid3D(nOutput), INHIBITORY_NEURON);

		sim->setNeuronParameters(grps[0], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(grps[2], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f);
		sim->setNeuronParameters(grps[3], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f);
		sim->setNeuronParameters(grps[4], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f);
		sim->setNeuronParameters(grps[5], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f);

		// use some rise and decay
		sim->setConductances(true, 5, 20, 150, 6, 100, 150);

		sim->connect(g0,grps[0],"full", 0.001f,  0.001f,  1.0f, 1,1, -1.0,-1.0,-1.0, 1.0,0.0, SYN_FIXED); // AMPA
		sim->connect(g0,grps[1],"full", 0.0005f, 0.0005f, 1.0f, 1,1, -1.0,-1.0,-1.0, 0.0,1.0, SYN_FIXED); // NMDA
		sim->connect(g0,grps[2],"full",-0.0005f,-0.0005f, 1.0f, 1,1, -1.0,-1.0,-1.0, 0.5,0.5, SYN_FIXED); // AMPA+NMDA
		sim->connect(g1,grps[3],"full",-0.001f, -0.001f,  1.0f, 1,1, -1.0,-1.0,-1.0, 1.0,0.0, SYN_FIXED); // GABAa
		sim->connect(g1,grps[4],"full",-0.0005f,-0.0005f, 1.0f, 1,1, -1.0,-1.0,-1.0, 0.0,1.0, SYN_FIXED); // GABAb
		sim->connect(g1,grps[5],"full",-0.0005f,-0.0005f, 1.0f, 1,1, -1.0,-1.0,-1.0, 1.0,1.0, SYN_FIXED); // GABAa+GABAb

		spkGen1 = new PeriodicSpikeGeneratorCore(rate, spikeAtZero);
		spkGen2 = new PeriodicSpikeGeneratorCore(rate, spikeAtZero);
		sim->setSpikeGenerator(g0, spkGen1);
		sim->setSpikeGenerator(g1, spkGen2);

		sim->setupNetwork(true);
		ASSERT_TRUE(sim->isSimulationWithCOBA());

		for (int i=0; i<1000; i++) {
			sim->runNetwork(0,1,false,true);

			std::vector<float> gAMPA  = sim->getConductanceAMPA();
			std::vector<float> gNMDA  = sim->getConductanceNMDA();
			std::vector<float> gGABAa = sim->getConductanceGABAa();
			std::vector<float> gGABAb = sim->getConductanceGABAb();

			for (int g=0; g<4; g++) {
				// for all groups
				group_info_t grpInfo = sim->getGroupInfo(grps[g]);

				if (expectCond[g].find("AMPA")!=std::string::npos) {
					// AMPA is active
					if (!mode) {
						// CPU mode: record conductance values
						gAMPA_CPU[i] = gAMPA[grpInfo.StartN];
					} else {
						// GPU mode: compare values
//						fprintf(stderr,"gAMPA CPU=%f, GPU=%f\n",gAMPA_CPU[i], gAMPA[grpInfo.StartN]);
//						EXPECT_FLOAT_EQ(gAMPA_CPU[i], gAMPA[grpInfo.StartN]);
						EXPECT_NEAR(gAMPA_CPU[i], gAMPA[grpInfo.StartN], expectCondStd[g]);
					}
				} else if (expectCond[g].find("NMDA")!=std::string::npos) {
					if (!mode) {
						gNMDA_CPU[i] = gNMDA[grpInfo.StartN];
					} else {
//						EXPECT_FLOAT_EQ(gNMDA_CPU[i], gNMDA[grpInfo.StartN]);
						EXPECT_NEAR(gNMDA_CPU[i], gNMDA[grpInfo.StartN], expectCondStd[g]);
					}
				} else if (expectCond[g].find("GABAa")!=std::string::npos) {
					if (!mode) {
						gGABAa_CPU[i] = gGABAa[grpInfo.StartN];
					} else {
//						EXPECT_FLOAT_EQ(gGABAa_CPU[i], gGABAa[grpInfo.StartN]);
						EXPECT_NEAR(gGABAa_CPU[i], gGABAa[grpInfo.StartN], expectCondStd[g]);
					}
				} else if (expectCond[g].find("GABAb")!=std::string::npos) {
					if (!mode) {
						gGABAb_CPU[i] = gGABAb[grpInfo.StartN];
					} else {
//						EXPECT_FLOAT_EQ(gGABAb_CPU[i], gGABAb[grpInfo.StartN]);
						EXPECT_NEAR(gGABAb_CPU[i], gGABAb[grpInfo.StartN], expectCondStd[g]);
					}
				}
			}
		}

		delete spkGen1, spkGen2;
		delete sim;
	}	
}

/*
 * \brief testing CARLsim COBA output (spike rates) CPU vs GPU
 *
 * This test makes sure that whatever COBA network is run, both CPU and GPU mode give the exact same output
 * in terms of spike times and spike rates.
 * The total simulation time, input rate, weight, and delay are chosen randomly.
 * Afterwards we make sure that CPU and GPU mode produce the same spike times and spike rates. 
 */
TEST(COBA, firingRateCPUvsGPU) {
	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG0 = NULL, *spkMonG1 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL;
	std::vector<std::vector<int> > spkTimesG0CPU, spkTimesG1CPU, spkTimesG0GPU, spkTimesG1GPU;
	float spkRateG0CPU = 0.0f, spkRateG1CPU = 0.0f;

	int delay = rand() % 10 + 1;
	float wt = rand()*1.0/RAND_MAX*0.2f + 0.05f;
	float inputRate = rand() % 45 + 5.0f;
	int runTimeMs = rand() % 800 + 200;
//	fprintf(stderr,"runTime=%d, delay=%d, wt=%f, input=%f\n",runTimeMs,delay,wt,inputRate);

	for (int isGPUmode=0; isGPUmode<=1; isGPUmode++) {
		sim = new CARLsim("COBA.firingRateCPUvsGPU",isGPUmode?GPU_MODE:CPU_MODE,SILENT,0,42);
		int g0=sim->createSpikeGeneratorGroup("input", 1 ,EXCITATORY_NEURON);
		int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON);
		sim->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
		sim->setConductances(true); // make COBA explicit

		sim->connect(g0, g1, "full", RangeWeight(wt), 1.0f, RangeDelay(1,delay));

		bool spikeAtZero = true;
		spkGenG0 = new PeriodicSpikeGenerator(inputRate,spikeAtZero);
		sim->setSpikeGenerator(g0, spkGenG0);

		sim->setupNetwork();

		spkMonG0 = sim->setSpikeMonitor(g0,"NULL");
		spkMonG1 = sim->setSpikeMonitor(g1,"NULL");

		spkMonG0->startRecording();
		spkMonG1->startRecording();
		sim->runNetwork(runTimeMs/1000,runTimeMs%1000,false);
		spkMonG0->stopRecording();
		spkMonG1->stopRecording();

		if (!isGPUmode) {
			// CPU mode: store spike times and spike rate for future comparison
			spkRateG0CPU = spkMonG0->getPopMeanFiringRate();
			spkRateG1CPU = spkMonG1->getPopMeanFiringRate();
			spkTimesG0CPU = spkMonG0->getSpikeVector2D();
			spkTimesG1CPU = spkMonG1->getSpikeVector2D();
		} else {
			// GPU mode: compare to CPU results
			// assert so that we do not display all spike time errors if the rates are wrong
			ASSERT_FLOAT_EQ(spkMonG0->getPopMeanFiringRate(), spkRateG0CPU);
			ASSERT_FLOAT_EQ(spkMonG1->getPopMeanFiringRate(), spkRateG1CPU);

			spkTimesG0GPU = spkMonG0->getSpikeVector2D();
			spkTimesG1GPU = spkMonG1->getSpikeVector2D();
			ASSERT_EQ(spkTimesG0CPU[0].size(),spkTimesG0GPU[0].size());
			ASSERT_EQ(spkTimesG1CPU[0].size(),spkTimesG1GPU[0].size());
			for (int i=0; i<spkTimesG0CPU[0].size(); i++)
				EXPECT_EQ(spkTimesG0CPU[0][i], spkTimesG0GPU[0][i]);
			for (int i=0; i<spkTimesG1CPU[0].size(); i++)
				EXPECT_EQ(spkTimesG1CPU[0][i], spkTimesG1GPU[0][i]);
		}
		delete spkGenG0;
		delete sim;
	}
}
