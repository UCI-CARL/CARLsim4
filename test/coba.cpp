#include <snn.h>

#include "carlsim_tests.h"


// FIXME: this COULD be interface-level, but then there's no way to check sim->dNMDA, etc.
// We should ensure everything gets set up correctly
TEST(COBA, synRiseTimeSettings) {
	std::string name = "SNN";
	int maxConfig = rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	CpuSNN* sim;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			int tdAMPA  = rand()%100 + 1;
			int trNMDA  = (nConfig==1) ? 0 : rand()%100 + 1;
			int tdNMDA  = rand()%100 + trNMDA + 1; // make sure it's larger than trNMDA
			int tdGABAa = rand()%100 + 1;
			int trGABAb = (nConfig==nConfigStep+1) ? 0 : rand()%100 + 1;
			int tdGABAb = rand()%100 + trGABAb + 1; // make sure it's larger than trGABAb

			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);
			sim->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb,ALL);
			EXPECT_TRUE(sim->sim_with_conductances);
			EXPECT_FLOAT_EQ(sim->dAMPA,1.0f-1.0f/tdAMPA);
			if (trNMDA) {
				EXPECT_TRUE(sim->sim_with_NMDA_rise);
				EXPECT_FLOAT_EQ(sim->rNMDA,1.0f-1.0f/trNMDA);
				double tmax = (-tdNMDA*trNMDA*log(1.0*trNMDA/tdNMDA))/(tdNMDA-trNMDA); // t at which cond will be max
				EXPECT_FLOAT_EQ(sim->sNMDA, 1.0/(exp(-tmax/tdNMDA)-exp(-tmax/trNMDA))); // scaling factor, 1 over max amplitude

			} else {
				EXPECT_FALSE(sim->sim_with_NMDA_rise);
			}
			EXPECT_FLOAT_EQ(sim->dNMDA,1.0f-1.0f/tdNMDA);
			EXPECT_FLOAT_EQ(sim->dGABAa,1.0f-1.0f/tdGABAa);
			if (trGABAb) {
				EXPECT_TRUE(sim->sim_with_GABAb_rise);
				EXPECT_FLOAT_EQ(sim->rGABAb,1.0f-1.0f/trGABAb);
				double tmax = (-tdGABAb*trGABAb*log(1.0*trGABAb/tdGABAb))/(tdGABAb-trGABAb); // t at which cond will be max
				EXPECT_FLOAT_EQ(sim->sGABAb, 1.0/(exp(-tmax/tdGABAb)-exp(-tmax/trGABAb))); // scaling factor, 1 over max amplitude
			} else {
				EXPECT_FALSE(sim->sim_with_GABAb_rise);
			}
			EXPECT_FLOAT_EQ(sim->dGABAb,1.0f-1.0f/tdGABAb);

			delete sim;
		}
	}
}

//! This test assures that the conductance peak occurs as specified by tau_rise and tau_decay, and that the peak is
//! equal to the specified weight value
TEST(COBA, synRiseTime) {
	std::string name = "SNN";
	int maxConfig = 1;//rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	CpuSNN* sim;

	float abs_error = 0.05; // five percent error for wt

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			int tdAMPA  = rand()%100 + 1;
			int trNMDA  = rand()%100 + 1;
			int tdNMDA  = rand()%100 + trNMDA + 1; // make sure it's larger than trNMDA
			int tdGABAa = rand()%100 + 1;
			int trGABAb = rand()%100 + 1;
			int tdGABAb = rand()%100 + trGABAb + 1; // make sure it's larger than trGABAb

			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);
			int g0=sim->createSpikeGeneratorGroup("inputExc", 1, EXCITATORY_NEURON, ALL);
			int g1=sim->createGroup("excit", 1, EXCITATORY_NEURON, ALL);
			sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->connect(g0,g1,"full",0.5f,0.5f,1.0f,1,1,0.0f,1.0f,SYN_FIXED);

			int g2=sim->createSpikeGeneratorGroup("inputInh", 1, INHIBITORY_NEURON, ALL);
			int g3=sim->createGroup("inhib", 1, INHIBITORY_NEURON, ALL);
			sim->setNeuronParameters(g3, 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);
			sim->connect(g2,g3,"full",-0.5f,-0.5f,1.0f,1,1,0.0f,1.0f,SYN_FIXED);

			sim->setConductances(true,tdAMPA,trNMDA,tdNMDA,tdGABAa,trGABAb,tdGABAb,ALL);

			// run network for a second first, so that we know spike will happen at simTimeMs==1000
			PeriodicSpikeGeneratorCore* spk1 = new PeriodicSpikeGeneratorCore(1.0f); // periodic spiking @ 50 Hz
			sim->setSpikeGenerator(g0, spk1, ALL);
			sim->setSpikeGenerator(g2, spk1, ALL);
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

				if ((sim->gNMDA_d[sim->getGroupStartNeuronId(g1)]-sim->gNMDA_r[sim->getGroupStartNeuronId(g1)]) > maxNMDA) {
					tmaxNMDA=i;
					maxNMDA=sim->gNMDA_d[sim->getGroupStartNeuronId(g1)]-sim->gNMDA_r[sim->getGroupStartNeuronId(g1)];
				}
				if ((sim->gGABAb_d[sim->getGroupStartNeuronId(g3)]-sim->gGABAb_r[sim->getGroupStartNeuronId(g3)]) > maxGABAb) {
					tmaxGABAb=i;
					maxGABAb=sim->gGABAb_d[sim->getGroupStartNeuronId(g3)]-sim->gGABAb_r[sim->getGroupStartNeuronId(g3)];
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
		}
	}
}

/// **************************************************************************************************************** ///
/// CONDUCTANCE-BASED MODEL (COBA)
/// **************************************************************************************************************** ///


TEST(COBA, disableSynReceptors) {
	// create network by varying nConfig from 1...maxConfig, with
	// step size nConfigStep
	std::string name="SNN";
	int maxConfig = 1; //rand()%10 + 10;
	int nConfigStep = rand()%3 + 2;
	float tAMPA = 5.0f;		// the exact values don't matter
	float tNMDA = 10.0f;
	float tGABAa = 15.0f;
	float tGABAb = 20.0f;
	CpuSNN* sim = NULL;
	group_info_t grpInfo;
	int grps[4] = {-1};

	int expectSpkCnt[4] = {200, 160, 0, 0};
	int expectSpkCntStd = 10;

	std::string expectCond[4] = {"AMPA","NMDA","GABAa","GABAb"};
	float expectCondVal[4] = {0.14, 2.2, 0.17, 2.2};
	float expectCondStd[4] = {0.025,0.2,0.025,0.2,};

	int nInput = 1000;
	int nOutput = 10;

	for (int mode=0; mode<=1; mode++) {
		for (int nConfig=1; nConfig<=maxConfig; nConfig+=nConfigStep) {
			sim = new CpuSNN(name,mode?GPU_MODE:CPU_MODE,SILENT,0,nConfig,42);

			int g0=sim->createSpikeGeneratorGroup("spike", nInput, EXCITATORY_NEURON, ALL);
			int g1=sim->createSpikeGeneratorGroup("spike", nInput, INHIBITORY_NEURON, ALL);
			grps[0]=sim->createGroup("excitAMPA", nOutput, EXCITATORY_NEURON, ALL);
			grps[1]=sim->createGroup("excitNMDA", nOutput, EXCITATORY_NEURON, ALL);
			grps[2]=sim->createGroup("inhibGABAa", nOutput, INHIBITORY_NEURON, ALL);
			grps[3]=sim->createGroup("inhibGABAb", nOutput, INHIBITORY_NEURON, ALL);

			sim->setNeuronParameters(grps[0], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[1], 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[2], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);
			sim->setNeuronParameters(grps[3], 0.1f,  0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 2.0f, 0.0f, ALL);

			sim->setConductances(true, 5, 0, 150, 6, 0, 150, ALL);

			sim->connect(g0, grps[0], "full", 0.001f, 0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g0, grps[1], "full", 0.0005f, 0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);
			sim->connect(g1, grps[2], "full", -0.001f, -0.001f, 1.0f, 1, 1, 1.0, 0.0, SYN_FIXED);
			sim->connect(g1, grps[3], "full", -0.0005f, -0.0005f, 1.0f, 1, 1, 0.0, 1.0, SYN_FIXED);

			PoissonRate poissIn1(nInput);
			PoissonRate poissIn2(nInput);
			for (int i=0; i<nInput; i++) {
				poissIn1.rates[i] = 30.0f;
				poissIn2.rates[i] = 30.0f;
			}
			sim->setSpikeRate(g0,&poissIn1,1,ALL);
			sim->setSpikeRate(g1,&poissIn2,1,ALL);

			sim->setupNetwork(true);
			sim->runNetwork(1,0,false,false);

			if (mode) {
				// GPU_MODE: copy from device to host
				for (int g=0; g<4; g++)
					sim->copyNeuronState(&(sim->cpuNetPtrs), &(sim->cpu_gpuNetPtrs), cudaMemcpyDeviceToHost, false, grps[g]);
			}

			for (int c=0; c<nConfig; c++) {
				for (int g=0; g<4; g++) { // all groups
					grpInfo = sim->getGroupInfo(grps[g],c);

					EXPECT_TRUE(sim->sim_with_conductances);
					for (int n=grpInfo.StartN; n<=grpInfo.EndN; n++) {
//						printf("%d[%d]: AMPA=%f, NMDA=%f, GABAa=%f, GABAb=%f\n",g,n,sim->gAMPA[n],sim->gNMDA[n],sim->gGABAa[n],sim->gGABAb[n]);
						if (expectCond[g]=="AMPA") {
							EXPECT_GT(sim->gAMPA[n],0.0f);
							EXPECT_NEAR(sim->gAMPA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gAMPA[n],0.0f);

						if (expectCond[g]=="NMDA") {
							EXPECT_GT(sim->gNMDA[n],0.0f);
							EXPECT_NEAR(sim->gNMDA[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gNMDA[n],0.0f);

						if (expectCond[g]=="GABAa") {
							EXPECT_GT(sim->gGABAa[n],0.0f);
							EXPECT_NEAR(sim->gGABAa[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAa[n],0.0f);

						if (expectCond[g]=="GABAb") {
							EXPECT_GT(sim->gGABAb[n],0.0f);
							EXPECT_NEAR(sim->gGABAb[n],expectCondVal[g],expectCondStd[g]);
						}
						else
							EXPECT_FLOAT_EQ(sim->gGABAb[n],0.0f);
					}
				}
			}
			delete sim;
		}
	}	
}

TEST(COBA, spikeRateCPUvsGPU) {
	int randSeed = rand() % 1000;	// randSeed must not interfere with STP

	CARLsim *sim = NULL;
	SpikeMonitor *spkMonG0 = NULL, *spkMonG1 = NULL, *spkMonG2 = NULL, *spkMonG3 = NULL;
	PeriodicSpikeGenerator *spkGenG0 = NULL, *spkGenG1 = NULL;

	float rateG0CPU = -1.0f;
	float rateG1CPU = -1.0f;
	float rateG2CPU = -1.0f;
	float rateG3CPU = -1.0f;
	std::vector<std::vector<int> > spkTimesG0, spkTimesG1, spkTimesG2, spkTimesG3;

	int runTimeMs = 1000;//rand() % 9500 + 500;
	float wt = 0.15f;

//PoissonRate in(1);

	for (int isGPUmode=0; isGPUmode<=1; isGPUmode++) {
		CARLsim* sim = new CARLsim("SNN",isGPUmode?GPU_MODE:CPU_MODE,USER,0,1,randSeed);
		int g0=sim->createSpikeGeneratorGroup("input0", 1, EXCITATORY_NEURON);
		int g1=sim->createSpikeGeneratorGroup("input1", 1, EXCITATORY_NEURON);
		int g2=sim->createGroup("excit2", 1, EXCITATORY_NEURON);
		int g3=sim->createGroup("excit3", 1, EXCITATORY_NEURON);
		sim->setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);
		sim->setNeuronParameters(g3, 0.02f, 0.2f, -65.0f, 8.0f);

		sim->connect(g0,g2,"full",RangeWeight(wt),1.0f,RangeDelay(1));
		sim->connect(g1,g3,"full",RangeWeight(wt),1.0f,RangeDelay(1));

		sim->setConductances(true,5, 0, 150, 6, 0, 150);

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
			// assert so if the rate is not the same, don't evaluate spike times
			ASSERT_FLOAT_EQ( spkMonG0->getPopMeanFiringRate(), rateG0CPU);
			ASSERT_FLOAT_EQ( spkMonG1->getPopMeanFiringRate(), rateG1CPU);
			ASSERT_FLOAT_EQ( spkMonG2->getPopMeanFiringRate(), rateG2CPU);
			ASSERT_FLOAT_EQ( spkMonG3->getPopMeanFiringRate(), rateG3CPU);

//				fprintf(stderr,"Group 2:\n");				
			std::vector<std::vector<int> > spkT = spkMonG2->getSpikeVector2D();
			for (int i=0; i<max(spkTimesG2[0].size(),spkT[0].size()); i++) {
//					fprintf(stderr, "%d\t%d\n",(i<spkTimesG2[0].size())?spkTimesG2[0][i]:-1, (i<spkT[0].size())?spkT[0][i]:-1);
				if (i<spkTimesG2[0].size() && i<spkT[0].size())
					EXPECT_EQ( spkTimesG2[0][i], spkT[0][i]);
			}
		}
		delete spkGenG0;
		delete spkGenG1;
		delete sim;
	}
}