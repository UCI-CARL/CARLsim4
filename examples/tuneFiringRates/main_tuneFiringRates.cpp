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
// includes the callback function to output spike data to arrays
#include "../common/writeSpikeToArray.h"
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
	
	// in case I want to time stuff
	time_t timer_start,timer_end;
	// start the timer
	time(&timer_start);
	const int numConfig = 1;// for the new version we run one at a time
	const bool onGPU = false; //run on GPU
	const int ithGPU =2; // which GPU to run (0-3)
	// probably should put the rand seed here
	const int randSeed = 42;
	// pti data structures
	ParamTuning *ptiObj;
	// keeps track of individual Id for a particular generation
	uint32_t indiId;
	// keeps track of generations
	int genCounter;
	// array that keeps track of the fitness values of the networks being evaluated in parallel
	double fitness[NUM_CONFIGS];
	// the target average firing rate is 10 Hz for every group
	double inputTargetFR=30.0f;
	double excTargetFR=10.0f; 
	double inhTargetFR=20.0f;
	// CARLsim data structures
	CARLsim** snn = new CARLsim*[NUM_CONFIGS];
	int inputGroup[NUM_CONFIGS];
	int excGroup[NUM_CONFIGS];
	int inhGroup[NUM_CONFIGS];
	// poissonRate spiking input pointer
	PoissonRate* input[NUM_CONFIGS];
	
	// -----------------------------------------------------------------------------
	// BEGIN PTI initialization
	// -----------------------------------------------------------------------------
	// must always be called first
	InitializeParamTuning("examples/tuneFiringRates/ESEA-plus.param");
	// Use contstructor to say where will some EO statistics and data
	ptiObj = new ParamTuning("examples/tuneFiringRates/results/eoOutput.txt");

	// Create parameter tuning parameters and register them here
	float min = 0.0005;
	float max = 0.5;
	// param 1: InputGroup-ExcGroup weights (fixed)
	ptiObj->addParam("InputGroup-ExcGroup", min, max);
	// param 2: ExcGroup-ExcGroup weights (fixed)
	ptiObj->addParam("ExcGroup-ExcGroup", min, max);
	// param 3: ExcGroup-InhGroup weights (fixed)
	ptiObj->addParam("ExcGroup-InhGroup", min, max);
	// param 4: InhGroup-ExcGroup weights (fixed)
	ptiObj->addParam("InhGroup-ExcGroup", min, max);
	
	// PTI adds all parameters
	ptiObj->updateParameters();
	int genomeSize = ptiObj->getVectorSize();
	printf("Parameters added\n");
	assert(genomeSize==NUM_PARAMS);

	// grabe the population size
	int popSize = ptiObj->getPopulationSize();

	// -----------------------------------------------------------------------------
	// END PTI initialization
	// -----------------------------------------------------------------------------
	
	// -----------------------------------------------------------------------------
	// BEGIN EO main EA loop
	// -----------------------------------------------------------------------------
	printf("Beginning Evolutionary Algorithm\n");
	printf("The maximum generations allowed is: %d \n",ptiObj->getMaxGen());
	// initialize all relevant counters
	genCounter=0;
	do{
		// we need to do initialization here now
  	// we can call a function that takes the snn* object and the pti object and 
		// sets the parameters
		genCounter++;
		printf("Parameter initialization for generation %d\n",genCounter);
		// We can run, at most, 1 generation of individuals in parallel.	Obviously, we can not run
		// individuals from the next generation because we have not evaluated this generation yet.
		assert(popSize == ptiObj->getPopulationSize());
		indiId = 0;
		int currentIndiId = indiId;
		// initialize fitness array to zero
		memset(fitness, 0, sizeof(double)*popSize);
		// Loop over the individuals in the population
		while(indiId < popSize){
			// Associate unique individual Id with every new parameter configuration: every iteration of 
			// this loop represents the assignment of one set of parameters so the indiId must be incremented
			currentIndiId = indiId;
			for(int configId=0; configId < NUM_CONFIGS; configId++, indiId++){
				// -----------------------------------------------------------------------------
				// BEGIN CARLsim initialization
				// -----------------------------------------------------------------------------
				//create a network
				snn[configId] = new CARLsim("TuningFixedWeightsSNN",onGPU?GPU_MODE:CPU_MODE,USER,ithGPU,numConfig,randSeed);
			
				
				float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
				inputGroup[configId]=snn[configId]->createSpikeGeneratorGroup("Input",INPUT_SIZE,EXCITATORY_NEURON);
				excGroup[configId]=snn[configId]->createGroup("Exc",EXC_SIZE,EXCITATORY_NEURON);
				inhGroup[configId]=snn[configId]->createGroup("Inh",INH_SIZE,INHIBITORY_NEURON);
				// set conductance values
				snn[configId]->setConductances(true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
				// set Izhikevich neuron parameter values
				snn[configId]->setNeuronParameters(excGroup[configId], 0.02f, 0.2f, -65.0f, 8.0f);
				snn[configId]->setNeuronParameters(inhGroup[configId], 0.1f, 0.2f, -65.0f, 2.0f); 
				double initWeight = ptiObj->getParam(indiId,"InputGroup-ExcGroup");
				double maxWeight = initWeight;
				// create the connections (with a dummy weight) and grab their connection id
				snn[configId]->connect(inputGroup[configId],excGroup[configId],"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
				initWeight = ptiObj->getParam(indiId,"InputGroup-ExcGroup");
				maxWeight = initWeight;
				snn[configId]->connect(excGroup[configId],excGroup[configId],"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
				initWeight = ptiObj->getParam(indiId,"InputGroup-ExcGroup");
				maxWeight = initWeight;
				snn[configId]->connect(excGroup[configId],inhGroup[configId],"random", initWeight, maxWeight, 0.5f, 1, 1, SYN_FIXED);
				initWeight = ptiObj->getParam(indiId,"InputGroup-ExcGroup");
				maxWeight = initWeight;
				snn[configId]->connect(inhGroup[configId],excGroup[configId],"random", -1.0f*initWeight,-1.0f*maxWeight, 0.5f, 1, 1, SYN_FIXED);

				// initialize input
				input[configId] = new PoissonRate(INPUT_SIZE);
				for(int i=0;i<INPUT_SIZE;i++){
					input[configId]->rates[i]=inputTargetFR;
				}
	
				// still have to set the firing rates (need to double check)
				snn[configId]->setSpikeRate(inputGroup[configId],input[configId]);

				// set log stats 
				snn[configId]->setLogCycle(1);
				snn[configId]->setSpikeCounter(excGroup[configId],-1);
				snn[configId]->setSpikeCounter(inhGroup[configId],-1);
				
				// just build the network				
				snn[configId]->runNetwork(0,0);
				// -----------------------------------------------------------------------------
				// END CARLsim initialization
				// -----------------------------------------------------------------------------
			}

			indiId=currentIndiId;			
			// now run the network to evaluate it
			for(int configId=0; configId < NUM_CONFIGS; configId++, indiId++){
				// now run the simulations in parallel with these parameters and evaluate them
				//evaluateFitnessV1(CARLsim **snn[],);
				// run network for 1 s
				int runTime = 2;
				snn[configId]->runNetwork(runTime,0);
				// evaluate the fitness right here
				int* excCount = snn[configId]->getSpikeCounter(excGroup[configId]);
				int* inhCount = snn[configId]->getSpikeCounter(inhGroup[configId]);
				// count all spikes as a sanity check
				int excTotalCount = 0;
				for(int neurId=0;neurId<EXC_SIZE; neurId++)
					excTotalCount = excTotalCount + excCount[neurId];
				int inhTotalCount = 0;
				for(int neurId=0;neurId<INH_SIZE; neurId++)
					inhTotalCount = inhTotalCount + inhCount[neurId];
				printf("excTotalCount = %d\n",excTotalCount);
				printf("inhTotalCount = %d\n",inhTotalCount);
				double excFR = (double) (*excCount)/((double)EXC_SIZE*runTime);
				double inhFR = (double) (*inhCount)/((double)INH_SIZE*runTime);
				printf("excFR = %f Hz\n",excFR);
				printf("inhFR = %f Hz\n",inhFR);
				fitness[configId]=fabs(excFR-excTargetFR)+fabs(inhFR-inhTargetFR);
				printf("fitness = %f\n",fitness[configId]);
				// reset all spike counters
				snn[configId]->resetSpikeCounter(-1);
				// associate the fitness values (CARLsim) with individual Id/associated parameter values (EO)
				ptiObj->setFitness(fitness[configId], indiId);
				if (snn[configId]!=NULL) 
					delete snn[configId];
				snn[configId]=NULL;
				if(input[configId]!=NULL)
					delete input[configId];
				input[configId]=NULL;
			}
			
		} // end loop over individuals
	}while(ptiObj->runEA());// this takes care of all termination conditions specified in the EO param files
	// -----------------------------------------------------------------------------
	// END EO main EA loop
	// -----------------------------------------------------------------------------
	// print out final stats
	ptiObj->printSortedPopulation();
	printf("genCounter=%d\n", genCounter);

	delete ptiObj;

	time(&timer_end);

	printf("Time to run: %.f seconds\n",difftime(timer_end,timer_start));

	return 0;
}
