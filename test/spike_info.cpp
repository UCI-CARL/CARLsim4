#include <snn.h>

#include "carlsim_tests.h"

// TODO: I should probably use a google tests figure for this to reduce the
// amount of redundant code, but I don't need to do that right now. -- KDC

/// ****************************************************************************
/// Function to read and return a 1D array with time and nid (in that order.
/// ****************************************************************************
void readAndReturnSpikeFile(const std::string fileName, int*& buffer, long &arraySize){
	FILE* pFile;
	long lSize;
	size_t result;
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
		
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell(pFile);
	arraySize = lSize/sizeof(uint);
	rewind (pFile);
	int* AERArray;
	AERArray = new int[lSize];
	memset(AERArray,0,sizeof(int)*lSize);
	// allocate memory to contain the whole file:
	buffer = (int*) malloc (sizeof(int)*lSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
		
	// copy the file into the buffer:
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {fputs ("Reading error",stderr); exit (3);}
		
	/* the whole file is now loaded in the memory buffer. */
	for(int i=0;i<lSize/sizeof(int);i=i+2){
		AERArray[i]=buffer[i];
	}

	// terminate
	fclose (pFile);

	return;
};

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
		
	/* the whole file is now loaded in the memory buffer. */
	for(int i=0;i<lSize/sizeof(int);i=i+2){
		printf("time = %u \n",buffer[i]);
		printf("nid = %u \n",buffer[i+1]);
	}

	// terminate
	fclose (pFile);
	free (buffer);
};

/// ****************************************************************************
/// TESTS FOR SET SPIKE INFO 
/// ****************************************************************************

/*!
 * \brief testing to make sure grpId error is caught in setSpikeMonitor.
 *
 */
TEST(SETSPIKEINFO, grpId){
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
TEST(SETSPIKEINFO, configId){
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
TEST(SETSPIKEINFO, fname){
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

/*!
 * \brief testing to make sure clear() function works.
 *
 */
TEST(SPIKEINFO, clear){
	CARLsim* sim;
	PoissonRate* input;
	const int GRP_SIZE = 5;
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
		
		int runTime = 2;
		// run the network
		sim->runNetwork(runTime,0);
	
		spikeInfoG1->stopRecording();
		
		// we should have spikes!
		EXPECT_TRUE(spikeInfoG1->getSize() != 0);
		
		// now clear the spikes
		spikeInfoG1->clear();

		// we shouldn't have spikes!
		EXPECT_TRUE(spikeInfoG1->getSize() == 0);
		
		// start recording again
		spikeInfoG1->startRecording();
		
		// run the network again
		sim->runNetwork(runTime,0);
		
		// stop recording
		spikeInfoG1->stopRecording();
		
		// we should have spikes again
		EXPECT_TRUE(spikeInfoG1->getSize() != 0);

		
		delete sim;
		delete input;
	}
}

TEST(SPIKEINFO, getGrpFiringRate){
	CARLsim* sim;
	PoissonRate* input;
	const int GRP_SIZE = 5;
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
		
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		double initWeight = 0.27f;
		double maxWeight = 4*initWeight;

		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		sim->connect(inputGroup,g1,"random", initWeight, maxWeight, 1.0f, 1, 1, SYN_FIXED);

		// input
		input = new PoissonRate(GRP_SIZE);
		for(int i=0;i<GRP_SIZE;i++){
			input->rates[i]=inputTargetFR;
		}

		sim->setSpikeRate(inputGroup,input);

		system("rm -rf spkInputGrp.dat");
		system("rm -rf spkG1Grp.dat");
		SpikeInfo* spikeInfoInput = sim->setSpikeMonitor(inputGroup,"spkInputGrp.dat",0);
		SpikeInfo* spikeInfoG1 = sim->setSpikeMonitor(g1,"spkG1Grp.dat",0);
		
		spikeInfoInput->startRecording();
		spikeInfoG1->startRecording();
	 		
		int runTime = 2;
		// run the network
		sim->runNetwork(runTime,0);
	
		spikeInfoInput->stopRecording();
		spikeInfoG1->stopRecording();

		int* inputArray;
		long inputSize;
		readAndReturnSpikeFile("spkInputGrp.dat",inputArray,inputSize);
		int* g1Array;
		long g1Size;
		readAndReturnSpikeFile("spkG1Grp.dat",g1Array,g1Size);
		sim->setSpikeRate(inputGroup,input);
		// divide both by two, because we are only counting spike events, for 
		// which there are two data elements (time, nid)
		inputSize = inputSize/2;
		g1Size = g1Size/2;
		float inputFR = (float)inputSize/(runTime*GRP_SIZE);
		float g1FR = (float)g1Size/(runTime*GRP_SIZE);

		// confirm the spike info information is correct here.
		EXPECT_FLOAT_EQ(spikeInfoInput->getGrpFiringRate(),inputFR);
		EXPECT_FLOAT_EQ(spikeInfoG1->getGrpFiringRate(),g1FR);
		
		delete inputArray;
		delete g1Array;
		delete sim;
		delete input;
	}
}

TEST(SPIKEINFO, getMaxMinNeruonFiringRate){
	CARLsim* sim;
	PoissonRate* input;
	const int GRP_SIZE = 5;
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
		
		sim->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
		double initWeight = 0.27f;
		double maxWeight = 4*initWeight;

		sim->setNeuronParameters(g1, 0.02f, 0.0f, 0.2f, 0.0f, -65.0f, 0.0f, 8.0f, 0.0f, ALL);

		sim->connect(inputGroup,g1,"random", initWeight, maxWeight, 1.0f, 1, 1, SYN_FIXED);

		// input
		input = new PoissonRate(GRP_SIZE);
		for(int i=0;i<GRP_SIZE;i++){
			input->rates[i]=inputTargetFR;
		}

		sim->setSpikeRate(inputGroup,input);

		system("rm -rf spkInputGrp.dat");
		system("rm -rf spkG1Grp.dat");
		SpikeInfo* spikeInfoInput = sim->setSpikeMonitor(inputGroup,"spkInputGrp.dat",0);
		SpikeInfo* spikeInfoG1 = sim->setSpikeMonitor(g1,"spkG1Grp.dat",0);
		
		spikeInfoInput->startRecording();
		spikeInfoG1->startRecording();
	 		
		int runTime = 2;
		// run the network
		sim->runNetwork(runTime,0);
	
		spikeInfoInput->stopRecording();
		spikeInfoG1->stopRecording();

		int* inputArray;
		long inputSize;
		readAndReturnSpikeFile("spkInputGrp.dat",inputArray,inputSize);
		int* g1Array;
		long g1Size;
		readAndReturnSpikeFile("spkG1Grp.dat",g1Array,g1Size);
		sim->setSpikeRate(inputGroup,input);
		// divide both by two, because we are only counting spike events, for 
		// which there are two data elements (time, nid)
		int inputSpkCount[GRP_SIZE];
		int g1SpkCount[GRP_SIZE];
		memset(inputSpkCount,0,sizeof(int)*GRP_SIZE);
		memset(g1SpkCount,0,sizeof(int)*GRP_SIZE);
		for(int i=0;i<inputSize;i=i+2){
			inputSpkCount[inputArray[i+1]]++;
		}
		for(int i=0;i<g1Size;i=i+2){
			g1SpkCount[g1Array[i+1]]++;
		}

		std::vector<float> inputVector;
		std::vector<float> g1Vector;
		for(int i=0;i<GRP_SIZE;i++){
			//float inputFR = ;
			//float g1FR = ;
			inputVector.push_back((float)inputSpkCount[i]/(float)runTime);
			g1Vector.push_back((float)g1SpkCount[i]/(float)runTime);
		}
		// confirm the spike info information is correct here.
		std::sort(inputVector.begin(),inputVector.end());
		std::sort(g1Vector.begin(),g1Vector.end());

		// check max neuron firing
		EXPECT_FLOAT_EQ(spikeInfoInput->getMaxNeuronFiringRate(),inputVector.back());
		EXPECT_FLOAT_EQ(spikeInfoG1->getMaxNeuronFiringRate(),g1Vector.back());

		// check min neuron firing
		EXPECT_FLOAT_EQ(spikeInfoInput->getMinNeuronFiringRate(),inputVector.front());
		EXPECT_FLOAT_EQ(spikeInfoG1->getMinNeuronFiringRate(),g1Vector.front());
		
		delete inputArray;
		delete g1Array;
		delete sim;
		delete input;
	}
}
