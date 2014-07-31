//
// Copyright (c) 2013 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//		notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//		notice, this list of conditions and the following disclaimer in the
//		documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//		products derived from this software without specific prior written
//		permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Written Kris Carlson (KDC) @ UCI
// This file makes sure update network works for fixed weights.
// Rewritten: 3/30/2014 to use part of the new parameter tuning interface.
// and to test the new analysis class.

// includes core CARLsim functionality
#include <carlsim.h>
#include <mtrand.h>
// include the PTI framework classes and functions
#include <pti.h>
// TODO: Do away with globals.
// TODO: put fitness in a separate file.

extern MTRand getRand;

using namespace std;
// -----------------------------------------------------------------------------
// BEGIN global simulation constants and variables
// -----------------------------------------------------------------------------
#define SPON_RATE				    1.0
#define REFRACTORY_PERIOD		1.0
#define PI									3.1415926535897
#define RAND_SEED						42
// the number of networks we run on the GPU simultaneously.
#define NUM_CONFIGS					10
// the total number of parameters the user is going to tune.
#define NUM_PARAMS					4

// -----------------------------------------------------------------------------
// END global simulation constants
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// BEGIN internal model global constants
// -----------------------------------------------------------------------------
#define INPUT_SIZE			5
#define EXC_SIZE				5
#define INH_SIZE				5
// -----------------------------------------------------------------------------
// END internal model global constants
// -----------------------------------------------------------------------------

int main()
{
	MTRand getRand(210499257);
	const int numConfig = 1;// for the new version we run one at a time
	const bool onGPU = false; //run on GPU
	const int ithGPU =2; // which GPU to run (0-3)
	// probably should put the rand seed here
	const int randSeed = 42;
	// fitness
	double fitness;
	// CARLsim data structures
	CARLsim* snn;
	int inputGroup;
	int excGroup;
	int inhGroup;
	// input firing rate
	float inputTargetFR=5;
//	float excTargetFR=10;
//	float inhTargetFR=20;
	
	// poissonRate spiking input pointer
	PoissonRate* input;
	// create a SpikeMonitor pointers
	SpikeMonitor* spikeMonInput;
	SpikeMonitor* spikeMonExc;
	SpikeMonitor* spikeMonInh;
	
	snn = new CARLsim("TuningFixedWeightsSNN",onGPU?GPU_MODE:CPU_MODE,USER,ithGPU,numConfig,randSeed);
			
	float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
	inputGroup=snn->createSpikeGeneratorGroup("Input",INPUT_SIZE,EXCITATORY_NEURON);
	excGroup=snn->createGroup("Exc",EXC_SIZE,EXCITATORY_NEURON);
	inhGroup=snn->createGroup("Inh",INH_SIZE,INHIBITORY_NEURON);
	// set conductance values
	snn->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
	// set Izhikevich neuron parameter values
	snn->setNeuronParameters(excGroup, 0.02f, 0.2f, -65.0f, 8.0f);
	snn->setNeuronParameters(inhGroup, 0.1f, 0.2f, -65.0f, 2.0f); 
	double initWeight = 0.05f;

	// create the connections (with a dummy weight) and grab their connection id
	snn->connect(inputGroup,excGroup,"random", RangeWeight(initWeight), 0.5f, RangeDelay(1), SYN_FIXED);
	snn->connect(excGroup,excGroup,"random", RangeWeight(initWeight), 0.5f, RangeDelay(1), SYN_FIXED);
	snn->connect(excGroup,inhGroup,"random", RangeWeight(initWeight), 0.5f, RangeDelay(1), SYN_FIXED);
	snn->connect(inhGroup,excGroup,"random", RangeWeight(initWeight), 0.5f, RangeDelay(1), SYN_FIXED);
	
	snn->setupNetwork();

	// initialize input
	input = new PoissonRate(INPUT_SIZE);
	for(int i=0;i<INPUT_SIZE;i++){
		input->rates[i]=inputTargetFR;
	}
	
	// set out spike monitors here
	spikeMonInput=snn->setSpikeMonitor(inputGroup);
	spikeMonExc=snn->setSpikeMonitor(excGroup);
	spikeMonInh=snn->setSpikeMonitor(inhGroup);

	// still have to set the firing rates (need to double check)
	snn->setSpikeRate(inputGroup,input);
	
	// -----------------------------------------------------------------------------
	// END CARLsim initialization
	// -----------------------------------------------------------------------------
	
	// now run the simulations in parallel with these parameters and evaluate them
	// we should start timing here too.
	spikeMonInput->startRecording();
	spikeMonExc->startRecording();
	spikeMonInh->startRecording();
	// run network for 1 s
	int runTimeMs = 800;
	bool printSummary = false;
	snn->runNetwork(runTimeMs/1000,runTimeMs%1000,printSummary);
	
	// stop recording
	spikeMonInput->stopRecording();
	spikeMonExc->stopRecording();
	spikeMonInh->stopRecording();

	// print all the spiking info
	spikeMonInput->print();
	spikeMonExc->print();
	spikeMonInh->print();

	// \TODO clean up the following...

	// get the output of our spike monitor
/*	float inputFR = spikeMonInput->getPopMeanFiringRate();
	cout << "inputFR = " << inputFR << " Hz" << endl;
	float excFR = spikeMonExc->getPopMeanFiringRate();
	cout << "excFR = " << excFR << " Hz" << endl;
	float inhFR = spikeMonInh->getPopMeanFiringRate();
	cout << "inhFR = " << inhFR << " Hz" << endl;

	cout << "Input: Printing individual neuron firing rates:\n";
	vector<float> inputNFR = spikeMonInput->getAllFiringRates();
	for(int i=0;i<inputNFR.size();i++){
		cout << inputNFR.at(i) << " Hz" << endl;
	}
	cout << endl;

	cout << "Input: Printing sorted individual neuron firing rates:\n";
	vector<float> inputSNFR = spikeMonInput->getAllFiringRatesSorted();
	for(int i=0;i<inputSNFR.size();i++){
		cout << inputSNFR.at(i) << " Hz" << endl;
	}
	cout << endl;
	*/

	int numNeuronsInRange = 0;
	numNeuronsInRange = spikeMonInput->getNumNeuronsWithFiringRate(0.0f,7.0f);
	cout << "Number of neurons with firing range between 0 and 7 Hz: " \
			 << numNeuronsInRange << endl;

	float percentNeuronsInRange = 0;
	percentNeuronsInRange = spikeMonInput->getPercentNeuronsWithFiringRate(0.0f,7.0f);
	cout << "Percentage of neurons with firing range between 0 and 7 Hz: " \
			 << percentNeuronsInRange << endl;

	int numSilent = 0;
	numSilent = spikeMonInput->getNumSilentNeurons();
	cout << "Number of silent neurons: " << numSilent << endl;

	int percentSilent = 0;
	percentSilent = spikeMonInput->getPercentSilentNeurons();
	cout << "Percentage of silent neurons: " << percentSilent << "%" \
			 << endl;

	float inputMaxFR = 0;
	inputMaxFR = spikeMonInput->getMaxFiringRate();
	cout << "Neuron with max. firing rate firing at: " << inputMaxFR \
			 << " Hz." << endl;

	float inputMinFR = 0;
	inputMinFR = spikeMonInput->getMinFiringRate();
	cout << "Neuron with min. firing rate firing at: " << inputMinFR	\
			 << " Hz." << endl << endl;

	spikeMonInput->clear();
	spikeMonExc->clear();
	spikeMonInh->clear();

	// run for a bunch without recording spike times
	// show only brief spike summary by enabling printSummary
	snn->runNetwork(2*runTimeMs/1000, (2*runTimeMs)%1000, true);

	spikeMonInput->startRecording();
	spikeMonExc->startRecording();
	spikeMonInh->startRecording();
	
	snn->runNetwork(runTimeMs/1000,runTimeMs%1000,printSummary);
	spikeMonInput->stopRecording();
	spikeMonExc->stopRecording();
	spikeMonInh->stopRecording();

	spikeMonInput->print();
	spikeMonExc->print();
	spikeMonInh->print();

/*	vector<float> excNFR = spikeMonExc->getAllFiringRates();
	for(int i=0;i< excNFR.size();i++){
		cout << excNFR.at(i) << " Hz" << endl;
	}

	// get the output of our spike monitor
	inputFR = spikeMonInput->getPopMeanFiringRate();
	cout << "inputFR = " << inputFR << " Hz" << endl;
	excFR = spikeMonExc->getPopMeanFiringRate();
	cout << "excFR = " << excFR << " Hz" << endl;
	inhFR = spikeMonInh->getPopMeanFiringRate();
	cout << "inhFR = " << inhFR << " Hz" << endl;

	// \FIXME what is this doing here? there's no EO in this
	fitness=fabs(excFR-excTargetFR)+fabs(inhFR-inhTargetFR);
	printf("fitness = %f\n",fitness);
	// associate the fitness values (CARLsim) with individual Id/associated parameter values (EO)

*/
	if (snn!=NULL) 
		delete snn;
	snn=NULL;
	if(input!=NULL)
		delete input;
	input=NULL;

	return 0;
}
