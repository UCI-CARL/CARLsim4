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
#define SIM_MODE						GPU_MODE 
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
#define INPUT_SIZE			10
#define EXC_SIZE				10
#define INH_SIZE				10
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
	float excTargetFR=10;
	float inhTargetFR=20;
	
	// poissonRate spiking input pointer
	PoissonRate* input;
	// create a SpikeInfo pointers
	SpikeInfo* spikeInfoInput;
	SpikeInfo* spikeInfoExc;
	SpikeInfo* spikeInfoInh;
	
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
	double maxWeight = 4*initWeight;
	// create the connections (with a dummy weight) and grab their connection id
	snn->connect(inputGroup,excGroup,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
	snn->connect(excGroup,excGroup,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
	snn->connect(excGroup,inhGroup,"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
	snn->connect(inhGroup,excGroup,"random", -1.0f*initWeight,-1.0f*maxWeight, 0.5f, 1, 1, SYN_FIXED);
	
	// initialize input
	input = new PoissonRate(INPUT_SIZE);
	for(int i=0;i<INPUT_SIZE;i++){
		input->rates[i]=inputTargetFR;
	}
	
	// set out spike monitors here
	spikeInfoInput=snn->setSpikeMonitor(inputGroup);
	spikeInfoExc=snn->setSpikeMonitor(excGroup);
	spikeInfoInh=snn->setSpikeMonitor(inhGroup);

	// still have to set the firing rates (need to double check)
	snn->setSpikeRate(inputGroup,input);
	
	// set log stats 
	snn->setLogCycle(1);
	// -----------------------------------------------------------------------------
	// END CARLsim initialization
	// -----------------------------------------------------------------------------
	
	// now run the simulations in parallel with these parameters and evaluate them
	// we should start timing here too.
	spikeInfoInput->startRecording();
	spikeInfoExc->startRecording();
	spikeInfoInh->startRecording();
	// run network for 1 s
	int runTime = 1;
	snn->runNetwork(runTime,0);
	
	// stop recording
	spikeInfoInput->stopRecording();
	spikeInfoExc->stopRecording();
	spikeInfoInh->stopRecording();

	// get the output of our spike monitor
	float inputFR = spikeInfoInput->getGrpFiringRate();
	cout << "inputFR = " << inputFR << " Hz" << endl;
	float excFR = spikeInfoExc->getGrpFiringRate();
	cout << "excFR = " << excFR << " Hz" << endl;
	float inhFR = spikeInfoInh->getGrpFiringRate();
	cout << "inhFR = " << inhFR << " Hz" << endl;
	vector<float>* inputNFR;
	inputNFR = spikeInfoInput->getNeuronFiringRate();
	for(int i=0;i<inputNFR->size();i++){
		cout << inputNFR->at(i) << " Hz" << endl;
	}

	spikeInfoInput->clear();
	spikeInfoExc->clear();
	spikeInfoInh->clear();

	snn->runNetwork(runTime,0);
	snn->runNetwork(runTime,0);

	spikeInfoInput->startRecording();
	spikeInfoExc->startRecording();
	spikeInfoInh->startRecording();
	
	snn->runNetwork(runTime,0);
	
	spikeInfoInput->stopRecording();
	spikeInfoExc->stopRecording();
	spikeInfoInh->stopRecording();

	// get the output of our spike monitor
	inputFR = spikeInfoInput->getGrpFiringRate();
	cout << "inputFR = " << inputFR << " Hz" << endl;
	excFR = spikeInfoExc->getGrpFiringRate();
	cout << "excFR = " << excFR << " Hz" << endl;
	inhFR = spikeInfoInh->getGrpFiringRate();
	cout << "inhFR = " << inhFR << " Hz" << endl;

	
	fitness=fabs(excFR-excTargetFR)+fabs(inhFR-inhTargetFR);
	printf("fitness = %f\n",fitness);
	// associate the fitness values (CARLsim) with individual Id/associated parameter values (EO)
	if (snn!=NULL) 
		delete snn;
	snn=NULL;
	if(input!=NULL)
		delete input;
	input=NULL;
	// if(spikeInfoInput!=NULL)
	// 	delete spikeInfoInput;
	// spikeInfoInput=NULL;
	// if(spikeInfoExc!=NULL)
	// 	delete spikeInfoExc;
	// spikeInfoExc=NULL;
	// if(spikeInfoInh!=NULL)
	// 	delete spikeInfoInh;
	// spikeInfoInh=NULL;
	
	return 0;
}
