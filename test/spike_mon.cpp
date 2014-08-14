#include <snn.h>

#include "carlsim_tests.h"

// TODO: I should probably use a google tests figure for this to reduce the
// amount of redundant code, but I don't need to do that right now. -- KDC

/// ****************************************************************************
/// Function to read and return a 1D array with time and nid (in that order.
/// ****************************************************************************
void readAndReturnSpikeFile(const std::string fileName, int*& AERArray, long &arraySize){
	FILE* pFile;
	long lSize;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
		
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell(pFile);
	arraySize = lSize/sizeof(int);
	rewind (pFile);
	AERArray = new int[lSize];
	memset(AERArray,0,sizeof(int)*lSize);
	// allocate memory to contain the whole file:
	int* buffer = (int*) malloc (sizeof(int)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
		
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
		
	// the whole file is now loaded in the memory buffer.
	for (int i=0; i<lSize; i++) {
		int tmp = buffer[i];
		AERArray[i]=tmp;
	}

	// terminate
	fclose (pFile);
	free(buffer);
}

/// ****************************************************************************
/// Function for reading and printing spike data written to a file
/// ****************************************************************************
void readAndPrintSpikeFile(const std::string fileName){
	FILE * pFile;
	long lSize;
	int* buffer;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
			
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
		
	// allocate memory to contain the whole file:
	buffer = (int*) malloc (sizeof(int)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
		
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
		
	// the whole file is now loaded in the memory buffer.
	for(int i=0;i<lSize/sizeof(int);i=i+2){
		printf("time = %d, nid = %d\n",buffer[i],buffer[i+1]);
	}

	// terminate
	fclose (pFile);
	free (buffer);
}


/// ****************************************************************************
/// TESTS FOR SET SPIKE MON 
/// ****************************************************************************

/*!
 * \brief testing to make sure grpId error is caught in setSpikeMonitor.
 *
 */
TEST(SETSPIKEMON, grpId){
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
TEST(SETSPIKEMON, configId){
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
TEST(SETSPIKEMON, fname){
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


TEST(SPIKEMON, interfaceDeath) {
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g0 = sim->createSpikeGeneratorGroup("Input",5,EXCITATORY_NEURON);
	SpikeMonitor* spkMon = sim->setSpikeMonitor(g0);

	int g1 = sim->createGroup("g1", 5, EXCITATORY_NEURON);		
	sim->setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);
	sim->setConductances(true);

	sim->connect(g0,g1,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), SYN_FIXED);

	sim->setupNetwork();

	// test all APIS that cannot be called in certain modes
	// \TODO SpikeMonitor mode COUNT not yet implemented
	EXPECT_DEATH(spkMon->setMode(COUNT),"");
	// after mode is implemented, make sure you cannot call getSpikeVector2D in COUNT mode, etc.

	// test all APIs that cannot be called when recording is on
	spkMon->startRecording();
	EXPECT_DEATH(spkMon->getPopMeanFiringRate(),"");
	EXPECT_DEATH(spkMon->getPopStdFiringRate(),"");
	EXPECT_DEATH(spkMon->getPopNumSpikes(),"");
	EXPECT_DEATH(spkMon->getAllFiringRates(),"");
	EXPECT_DEATH(spkMon->getAllFiringRatesSorted(),"");
	EXPECT_DEATH(spkMon->getMaxFiringRate(),"");
	EXPECT_DEATH(spkMon->getMinFiringRate(),"");
	EXPECT_DEATH(spkMon->getNeuronMeanFiringRate(0),"");
	EXPECT_DEATH(spkMon->getNeuronNumSpikes(0),"");
	EXPECT_DEATH(spkMon->getNumNeuronsWithFiringRate(0,0),"");
	EXPECT_DEATH(spkMon->getNumSilentNeurons(),"");
	EXPECT_DEATH(spkMon->getPercentNeuronsWithFiringRate(0,0),"");
	EXPECT_DEATH(spkMon->getPercentSilentNeurons(),"");
	EXPECT_DEATH(spkMon->getSpikeVector2D(),"");
	EXPECT_DEATH(spkMon->print(),"");
	EXPECT_DEATH(spkMon->startRecording(),"");

	delete sim;
}


TEST(SPIKEMON, persistentMode) {
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	CARLsim* sim = new CARLsim("SNN",CPU_MODE,SILENT,0,1,42);
	int g0 = sim->createSpikeGeneratorGroup("Input",5,EXCITATORY_NEURON);
	SpikeMonitor* spkMon = sim->setSpikeMonitor(g0);

	int g1 = sim->createGroup("g1", 5, EXCITATORY_NEURON);		
	sim->setNeuronParameters(g1, 0.02, 0.2, -65.0, 8.0);
	sim->setConductances(true);

	sim->connect(g0,g1,"random", RangeWeight(0.01), 0.5f, RangeDelay(1), SYN_FIXED);

	sim->setupNetwork();

	// run for half a second, then check recording getters
	spkMon->startRecording();
	sim->runNetwork(0,500);
	spkMon->stopRecording();
	EXPECT_EQ(spkMon->getRecordingTotalTime(), 500);
	EXPECT_EQ(spkMon->getRecordingStartTime(), 0);
	EXPECT_EQ(spkMon->getRecordingStopTime(), 500);

	// run for half a second, then check recording getters
	// persistent mode should be off, so only the last probe should matter
	spkMon->startRecording();
	sim->runNetwork(0,500);
	spkMon->stopRecording();
	EXPECT_EQ(spkMon->getRecordingTotalTime(), 500);
	EXPECT_EQ(spkMon->getRecordingStartTime(), 500);
	EXPECT_EQ(spkMon->getRecordingStopTime(), 1000);

	// now switch persistent mode on
	spkMon->setPersistentData(true);

	// run for half a second, and expect persistent mode on
	// start should now be what it was (500), and total time should have increased by 500
	spkMon->startRecording();
	sim->runNetwork(0,500);
	spkMon->stopRecording();
	EXPECT_EQ(spkMon->getRecordingTotalTime(), 1000);
	EXPECT_EQ(spkMon->getRecordingStartTime(), 500);
	EXPECT_EQ(spkMon->getRecordingStopTime(), 1500);

	delete sim;
}


/*!
 * \brief testing to make sure clear() function works.
 *
 */
TEST(SPIKEMON, clear){
	CARLsim* sim;
	PoissonRate* input;
	const int GRP_SIZE = 5;
	const int inputTargetFR = 5.0f;
	int runTimeMs = 2000;

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

		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setNeuronParameters(g2, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		// input
		input = new PoissonRate(GRP_SIZE);
		for(int i=0;i<GRP_SIZE;i++){
			input->rates[i]=inputTargetFR;
		}

		// use full because random might give us a network that does not spike (depending on the random seed),
		// leading to the EXPECTs below to fail
		sim->connect(inputGroup,g1,"full", RangeWeight(initWeight), 1.0f, RangeDelay(1), SYN_FIXED);
		sim->connect(inputGroup,g2,"full", RangeWeight(initWeight), 1.0f, RangeDelay(1), SYN_FIXED);
		sim->connect(g1,g2,"full", RangeWeight(initWeight), 1.0f, RangeDelay(1), SYN_FIXED);

		SpikeMonitor* spikeMonG1 = sim->setSpikeMonitor(g1);
		sim->setupNetwork();

		sim->setSpikeRate(inputGroup,input);
		spikeMonG1->startRecording();
		
		sim->runNetwork(runTimeMs/1000,runTimeMs%1000);
	
		spikeMonG1->stopRecording();
		
		// we should have spikes!
		EXPECT_TRUE(spikeMonG1->getPopNumSpikes() > 0);
		
		// now clear the spikes and run again
		spikeMonG1->clear();
		EXPECT_TRUE(spikeMonG1->getPopNumSpikes() == 0); // shouldn't have any spikes!
		spikeMonG1->startRecording();
		sim->runNetwork(runTimeMs/1000,runTimeMs%1000);
		spikeMonG1->stopRecording();
		
		// we should have spikes again
		EXPECT_TRUE(spikeMonG1->getPopNumSpikes() > 0);

		
		delete sim;
		delete input;
	}
}



/*
 * This test verifies that the spike times written to file and AER struct match the ones from the simulation.
 * A PeriodicSpikeGenerator is used to periodically generate spikes, which allows us to know the exact spike times.
 * We run the simulation for a random number of milliseconds (most probably no full seconds), and read the spike file
 * afterwards. We expect all spike times to be multiples of the inter-spike interval, and the total number of spikes
 * to be exactly what it should be. The same must apply to the AER struct from the SpikeMonitor object.
 */
TEST(SPIKEMON, spikeTimes) {
	double rate = rand()%20 + 2.0;  // some random mean firing rate
	int isi = 1000/rate; // inter-spike interval

	const int GRP_SIZE = rand()%5 + 1; // some random group size

	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		CARLsim* sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);
		float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
		int g0 = sim->createSpikeGeneratorGroup("Input",GRP_SIZE,EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		sim->connect(g0,g1,"random", RangeWeight(0.27f), 1.0f, RangeDelay(1), SYN_FIXED);

		// use periodic spike generator to know the exact spike times
		PeriodicSpikeGenerator* spkGen = new PeriodicSpikeGenerator(rate);
		sim->setSpikeGenerator(g0, spkGen);

		sim->setupNetwork();

		// write all spikes to file
		SpikeMonitor* spikeMonG0 = sim->setSpikeMonitor(g0,"spkG0.dat",0);
		spikeMonG0->startRecording();

		// pick some random simulation time
		int runMs = (5+rand()%20) * isi;
		sim->runNetwork(runMs/1000,runMs%1000);

		spikeMonG0->stopRecording();

		// get spike vector
		std::vector<std::vector<int> > spkVector = spikeMonG0->getSpikeVector2D();

		// read spike file
		int* inputArray = NULL;
		long inputSize;
		readAndReturnSpikeFile("spkG0.dat",inputArray,inputSize);

		// sanity-check the size of the arrays
		EXPECT_EQ(inputSize/2, runMs/isi * GRP_SIZE);
		for (int i=0; i<GRP_SIZE; i++)
			EXPECT_EQ(spkVector[i].size(), runMs/isi);

		// check the spike times of spike file and AER struct
		// we expect all spike times to be a multiple of the ISI
		for (int i=0; i<inputSize; i+=2) {
			EXPECT_EQ(inputArray[i]%isi, 0);
//			EXPECT_EQ(spkVector[i/2].time % isi, 0);
		}
		for (int i=0; i<GRP_SIZE; i++)
			for (int j=0; j<spkVector[i].size(); j++)
				EXPECT_EQ(spkVector[i][j] % isi, 0);


		system("rm -rf spkG0.dat");
		if (inputArray!=NULL) delete[] inputArray;
		delete spkGen;
		delete sim;
	}
}

/*
 * This test checks for the correctness of the getGroupFiringRate method.
 * A PeriodicSpikeGenerator is used to periodically generate input spikes, so that the input spike times are known.
 * A network will then be run for a random amount of milliseconds. The activity of the input group will only be
 * recorded for a brief amount of time, whereas the activity of another group will be recorded for the full run.
 * The firing rate of the input group, which is calculated by the SpikeMonitor object, must then be based on only a
 * brief time window, whereas the spike file should contain all spikes. For the other group, both spike file and AER
 * struct should have the same number of spikes.
 */
TEST(SPIKEMON, getGroupFiringRate){
	CARLsim* sim;

	double rate = rand()%12 + 2.0f;  // some random mean firing rate
	int isi = 1000/rate; // inter-spike interval

	const int GRP_SIZE = 1;//rand()%5 + 1;
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<=1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);

		float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
		int g0 = sim->createSpikeGeneratorGroup("Input",GRP_SIZE,EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(g0,g1,"random", RangeWeight(0.27f), 1.0f, RangeDelay(1), SYN_FIXED);

		PeriodicSpikeGenerator* spkGen = new PeriodicSpikeGenerator(rate);
		sim->setSpikeGenerator(g0, spkGen);

		sim->setupNetwork();

		SpikeMonitor* spikeMonInput = sim->setSpikeMonitor(g0,"spkInputGrp.dat",0);
		SpikeMonitor* spikeMonG1 = sim->setSpikeMonitor(g1,"spkG1Grp.dat",0);

		// pick some random simulation time
		int runTimeMsOff = (5+rand()%10) * isi;
		int runTimeMsOn  = (5+rand()%20) * isi;

		// run network with recording off for g0, but recording on for G1
		spikeMonG1->startRecording();
		sim->runNetwork(runTimeMsOff/1000, runTimeMsOff%1000);

		// then start recording for some period
		spikeMonInput->startRecording();
		sim->runNetwork(runTimeMsOn/1000, runTimeMsOn%1000);
		spikeMonInput->stopRecording();

		// and run some more with recording off for input
		sim->runNetwork(runTimeMsOff/1000, runTimeMsOff%1000);

		// stopping the recording will update both AER structs and spike files
		spikeMonG1->stopRecording();

		// Note: Starting to record for 0 milliseconds at the end of a simulation is a little silly...but it should
		// be allowed, and we just want to achieve that all spikes of the input group get written to the spike file
		spikeMonInput->setPersistentData(true);
		spikeMonInput->startRecording();
		spikeMonInput->stopRecording();

		// read spike files (which are now complete because of stopRecording above)
		int* inputArray = NULL;
		long inputSize;
		readAndReturnSpikeFile("spkInputGrp.dat",inputArray,inputSize);
		int* g1Array = NULL;
		long g1Size;
		readAndReturnSpikeFile("spkG1Grp.dat",g1Array,g1Size);

		// activity in the input group was recorded only for a short period
		// the SpikeMon object must thus compute the firing rate based on only a brief time window
		EXPECT_EQ(spikeMonInput->getRecordingTotalTime(), runTimeMsOn);
		EXPECT_NEAR(spikeMonInput->getPopMeanFiringRate(), rate, 0.1); // rate must match
		EXPECT_EQ(spikeMonInput->getPopNumSpikes(), runTimeMsOn*GRP_SIZE/isi); // spikes only from brief window
		EXPECT_EQ(inputSize/2, (runTimeMsOn+2*runTimeMsOff)*GRP_SIZE/isi); // but spike file must have all spikes

		// g1 had recording on the whole time
		// its firing rate is not known explicitly, but AER should match spike file
		EXPECT_EQ(spikeMonG1->getRecordingTotalTime(), runTimeMsOn+2*runTimeMsOff);
		EXPECT_EQ(spikeMonG1->getPopNumSpikes(), g1Size/2);
		EXPECT_FLOAT_EQ(spikeMonG1->getPopMeanFiringRate(), g1Size/(2.0*GRP_SIZE) * 1000.0/(runTimeMsOn+2*runTimeMsOff));

		system("rm -rf spkInputGrp.dat");
		system("rm -rf spkG1Grp.dat");

		if (inputArray!=NULL) delete[] inputArray;
		if (g1Array!=NULL) delete[] g1Array;
		delete spkGen;
		delete sim;
	}
}

TEST(SPIKEMON, getMaxMinNeuronFiringRate){
	const int GRP_SIZE = 5;
	const int inputTargetFR = 5.0f;
	// use threadsafe version because we have deathtests
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	// loop over both CPU and GPU mode.
	for(int mode=0; mode<1; mode++){
		// first iteration, test CPU mode, second test GPU mode
		CARLsim* sim = new CARLsim("SNN",mode?GPU_MODE:CPU_MODE,SILENT,0,1,42);

		float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
		int inputGroup = sim->createSpikeGeneratorGroup("Input",GRP_SIZE,EXCITATORY_NEURON);
		int g1 = sim->createGroup("g1", GRP_SIZE, EXCITATORY_NEURON, ALL);
		
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);
		sim->connect(inputGroup,g1,"random", RangeWeight(0.27f), 0.2f, RangeDelay(1), SYN_FIXED);
		sim->setupNetwork();

		// input
		PoissonRate* input = new PoissonRate(GRP_SIZE);
		for(int i=0;i<GRP_SIZE;i++){
			input->rates[i]=inputTargetFR;
		}
		sim->setSpikeRate(inputGroup,input);

		system("rm -rf spkInputGrp.dat");
		system("rm -rf spkG1Grp.dat");
		SpikeMonitor* spikeMonInput = sim->setSpikeMonitor(inputGroup,"spkInputGrp.dat",0);
		SpikeMonitor* spikeMonG1 = sim->setSpikeMonitor(g1,"spkG1Grp.dat",0);
		
		spikeMonInput->startRecording();
		spikeMonG1->startRecording();
	 		
		int runTimeMs = 2000;
		// run the network
		sim->runNetwork(runTimeMs/1000,runTimeMs%1000);
	
		spikeMonInput->stopRecording();
		spikeMonG1->stopRecording();

		int* inputArray = NULL;
		long inputSize;
		readAndReturnSpikeFile("spkInputGrp.dat",inputArray,inputSize);
		int* g1Array = NULL;
		long g1Size;
		readAndReturnSpikeFile("spkG1Grp.dat",g1Array,g1Size);

		// divide both by two, because we are only counting spike events, for 
		// which there are two data elements (time, nid)
		int inputSpkCount[GRP_SIZE];
		int g1SpkCount[GRP_SIZE];
		memset(inputSpkCount,0,sizeof(int)*GRP_SIZE);
		memset(g1SpkCount,0,sizeof(int)*GRP_SIZE);
		for(int i=1; i<inputSize; i+=2) {
			int nid = inputArray[i];
			assert(nid>=0 && nid<GRP_SIZE);
			inputSpkCount[nid]++;
		}
		for(int i=1; i<g1Size; i+=2) {
			int nid = g1Array[i];
			assert(nid>=0 && nid<GRP_SIZE);
			g1SpkCount[nid]++;
		}

		std::vector<float> inputVector;
		std::vector<float> g1Vector;
		for(int i=0;i<GRP_SIZE;i++){
			inputVector.push_back(inputSpkCount[i]*1000.0/(float)runTimeMs);
			g1Vector.push_back(g1SpkCount[i]*1000.0/(float)runTimeMs);
		}
		// confirm the spike info information is correct here.
		std::sort(inputVector.begin(),inputVector.end());
		std::sort(g1Vector.begin(),g1Vector.end());

		spikeMonInput->getMaxFiringRate();

		// check max neuron firing
		float tmp = inputVector.back();
		EXPECT_FLOAT_EQ(spikeMonInput->getMaxFiringRate(),tmp);
		EXPECT_FLOAT_EQ(spikeMonG1->getMaxFiringRate(),g1Vector.back());

		// check min neuron firing
		EXPECT_FLOAT_EQ(spikeMonInput->getMinFiringRate(),inputVector.front());
		EXPECT_FLOAT_EQ(spikeMonG1->getMinFiringRate(),g1Vector.front());

		if (inputArray!=NULL) delete[] inputArray;
		if (g1Array!=NULL) delete[] g1Array;
		delete input;
		delete sim;
	}
}