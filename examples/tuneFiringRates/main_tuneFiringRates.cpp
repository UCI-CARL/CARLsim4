//
// Copyright (c) 2013 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
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

// includes core CARLsim functionality
#include <snn.h>
// includes the callback function to output spike data to arrays
#include "../common/writeSpikeToArray.h"
// include the PTI framework classes and functions
#include <pti.h>

extern MTRand getRand;

using namespace std;
// -----------------------------------------------------------------------------
// BEGIN global simulation constants and variables
// -----------------------------------------------------------------------------
#define SIM_MODE            GPU_MODE 
#define SPON_RATE 	    1.0
#define REFRACTORY_PERIOD   1.0
#define PI                  3.1415926535897
#define RAND_SEED           42
// the number of networks we run on the GPU simultaneously.
#define NUM_CONFIGS         10
// the total number of parameters the user is going to tune.
#define NUM_PARAMS          4
#define DUMMY_WEIGHT_VALUE  1.0
// maximum input/feedback spike rates
#define MAX_FIRING_INPUT    30
#define MAX_FEEDBACK        30
// -----------------------------------------------------------------------------
// END global simulation constants
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// BEGIN internal model global constants
// -----------------------------------------------------------------------------
#define INPUT_SIZE      10
#define EXC_SIZE        10
#define INH_SIZE        10
// -----------------------------------------------------------------------------
// END internal model global constants
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// BEGIN global variables (don't give me that look)
// -----------------------------------------------------------------------------
// CARLSIM core:
CpuSNN* snn;
int inputGroup; int excGroup; int inhGroup;
int input_exc_cid; int exc_exc_cid; int exc_inh_cid; int inh_exc_cid;
float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;

// poissonRate spiking input pointer
PoissonRate* input;
// array to hold the firing rates for each input neuron
float inputSpikeRate[INPUT_SIZE];

// create a WriteSpikeToArray object pointer
WriteSpikeToArray* callbackInput[NUM_CONFIGS];
WriteSpikeToArray* callbackExc[NUM_CONFIGS]; 
WriteSpikeToArray* callbackInh[NUM_CONFIGS];
// arrays to hold spike firing data
unsigned int** spikeDataInput[NUM_CONFIGS]; 
unsigned int** spikeDataExc[NUM_CONFIGS];
unsigned int** spikeDataInh[NUM_CONFIGS];
// arrays to hold size of spike firing data
int sizeInput[NUM_CONFIGS];
int sizeExc[NUM_CONFIGS];
int sizeInh[NUM_CONFIGS];

//EO/PTI core
//PTI object
ParamTuning *p;
// keeps track of global individual ID for labelling file output.
uint32_t globalIndiID;
// keeps track of individual ID for a particular generation
uint32_t IndiID;
// keeps track of generations
int genCounter;

// array that keeps track of the fitness values of the networks being evaluated in parallel
double fitness[NUM_CONFIGS];
// create an array to hold all the assigned variables
// 1st DIM: the number of SNNs to be run in parallel.
// 2nd DIM: the number of parameters.
float paramValue[NUM_CONFIGS][NUM_PARAMS];

// -----------------------------------------------------------------------------
// END global variables
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// BEGIN function prototypes
// -----------------------------------------------------------------------------
// built for clarity.  The user adds all the EO parameters here.  
void addParamsToPTI();
// built for clarity.  The parameter values from EO are assigned to CARLsim parameter values here.
void assignParamsToCARLsim(int numConfig);
// the objective/fitness function, version 1.  Tunes networks to achieve a target average firing rate.
int evaluateFitnessV1();
// -----------------------------------------------------------------------------
// END function prototypes
// -----------------------------------------------------------------------------

int main()
{
  MTRand getRand(210499257);
  
  // -----------------------------------------------------------------------------
  // BEGIN CARLsim initialization
  // -----------------------------------------------------------------------------
  //create a network
  snn = new CpuSNN("TuningFixedWeightsSNN",NUM_CONFIGS,RAND_SEED); // 'Tuned Internal Model'     
  inputGroup=snn->createSpikeGeneratorGroup("Input",INPUT_SIZE,EXCITATORY_NEURON);
  excGroup=snn->createGroup("Exc",EXC_SIZE,EXCITATORY_NEURON);
  inhGroup=snn->createGroup("Inh",INH_SIZE,INHIBITORY_NEURON);
  // set conductance values
  snn->setConductances(ALL,true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
  // set Izhikevich neuron parameter values
  snn->setNeuronParameters(excGroup, 0.02f, 0.2f, -65.0f, 8.0f);
  snn->setNeuronParameters(inhGroup, 0.1f, 0.2f, -65.0f, 2.0f); 
  
  // create the connections (with a dummy weight) and grab their connection id
  input_exc_cid=snn->connect(inputGroup,excGroup,"random", DUMMY_WEIGHT_VALUE, DUMMY_WEIGHT_VALUE, 0.5f, 1, 5, SYN_FIXED);
  exc_exc_cid=snn->connect(excGroup,excGroup,"random", DUMMY_WEIGHT_VALUE, DUMMY_WEIGHT_VALUE, 0.5f, 1, 1, SYN_FIXED);
  exc_inh_cid=snn->connect(excGroup,inhGroup,"random", DUMMY_WEIGHT_VALUE, DUMMY_WEIGHT_VALUE, 0.5f, 1, 1, SYN_FIXED);
  inh_exc_cid=snn->connect(inhGroup,excGroup,"random", DUMMY_WEIGHT_VALUE, DUMMY_WEIGHT_VALUE, 0.5f, 1, 1, SYN_FIXED);
  // note: DUMMY_WEIGHT_VALUE = 1.0

  // initialize input
  input = new PoissonRate(INPUT_SIZE);
  for(int i=0;i<INPUT_SIZE;i++){
    input->rates[i]=MAX_FIRING_INPUT;
  }
  
  // set log stats 
  snn->setLogCycle(1, 1, stdout);

  // initialize every callback object in each array
  for(int i=0; i<NUM_CONFIGS; i++){
    callbackInput[i]= new WriteSpikeToArray;
    callbackExc[i]= new WriteSpikeToArray;
    callbackInh[i]= new WriteSpikeToArray;
  }   
 
  // assign a callback function to a group and config ID
  for(int i=0; i<NUM_CONFIGS; i++){
    snn->setSpikeMonitor(inputGroup,callbackInput[i],i);  
    snn->setSpikeMonitor(excGroup,callbackExc[i],i);
    snn->setSpikeMonitor(inhGroup,callbackInh[i],i);
  }
  
  //initialize these variables
  for(int i=0; i<NUM_CONFIGS; i++){
    spikeDataInput[i]=NULL;spikeDataExc[i]=NULL;spikeDataInh[i]=NULL;
    sizeInput[i]=0;sizeExc[i]=0;sizeInh[i]=0;
  }
  // -----------------------------------------------------------------------------
  // END CARLsim initialization
  // -----------------------------------------------------------------------------
  
  // -----------------------------------------------------------------------------
  // BEGIN PTI initialization
  // -----------------------------------------------------------------------------
  // create our results directory
  system("mkdir -p results/tuneFiringRates");
  // must always be called first
  InitializeParamTuning("examples/tuneFiringRates/ESEA-plus.param");
  // Use contstructor to say where will some EO statistics and data
  p = new ParamTuning("results/tuneFiringRates/eoOutput.txt");

  // Create parameter tuning parameters and register them here
  addParamsToPTI();
  
  // PTI adds all parameters
  p->updateParameters();
  int genomeSize = p->getVectorSize();
  printf("Parameters added\n");
  assert(genomeSize==NUM_PARAMS);


  // Initialize paramValue which holds the parameter values generated in EO
  // first dim size: number of configs, second dim size: number of total parameter
  for(int j=0;j<genomeSize;j++)
    for(int i=0;i<NUM_CONFIGS;i++)
      paramValue[i][j]=0;
  // -----------------------------------------------------------------------------
  // END PTI initialization
  // -----------------------------------------------------------------------------
  
  // -----------------------------------------------------------------------------
  // BEGIN EO main EA loop
  // -----------------------------------------------------------------------------
  printf("Beginning Evolutionary Algorithm\n");
  printf("The maximum generations allowed is: %d \n",p->getMaxGen());
  // initialize all relevant counters
  genCounter=0;globalIndiID=0;
  do{
    genCounter++;
    printf("Parameter initialization for generation %d\n",genCounter);
    // We can run, at most, 1 generation of individuals in parallel.  Obviously, we can not run
    // individuals from the next generation because we have not evaluated this generation yet.
    assert(NUM_CONFIGS <= p->getPopulationSize());
    IndiID = 0;
    // Loop over the individuals in the population
    while(IndiID < p->getPopulationSize() ){
      // Associate unique individual ID with every new parameter configuration: every iteration of 
      // this loop represents the assignment of one set of parameters so the IndiID must be incremented
      for(int configID=0; configID < NUM_CONFIGS; configID++, IndiID++, globalIndiID++){
	// Assign these newly created parameters to values in CARLsim using PTI
	assignParamsToCARLsim(configID);
      }
      // call updateNetwork() to set the weight changes
      // 1st argument resets timing tables and 2nd argument resets the weights (valid for plastic weights)
      snn->updateNetwork(false,true);
      // now run the simulations in parallel with these parameters and evaluate them
      evaluateFitnessV1();
      
      // associate the fitness values (CARLsim) with individual ID/associated parameter values (EO)
      p->setFitness(fitness, IndiID, NUM_CONFIGS);
    } // end loop over individuals
  }while(p->runEA());// this takes care of all termination conditions specified in the EO param files
  // -----------------------------------------------------------------------------
  // END EO main EA loop
  // -----------------------------------------------------------------------------
  // print out final stats
  p->printSortedPopulation();
  printf("genCounter=%d\n", genCounter);

  return 0;
}

// The user adds all the parameters to be tuned by the PTI here
  
// Range for all weight values: 0.005 - 0.5.
void addParamsToPTI()
{
  float min = 0.0005;
  float max = 0.5;
  // param 1: InputGroup-ExcGroup weights (fixed)
  p->addParam("InputGroup-ExcGroup", min, max);
  // param 2: ExcGroup-ExcGroup weights (fixed)
  p->addParam("ExcGroup-ExcGroup", min, max);
  // param 3: ExcGroup-InhGroup weights (fixed)
  p->addParam("ExcGroup-InhGroup", min, max);
  // param 4: InhGroup-ExcGroup weights (fixed)
  p->addParam("InhGroup-ExcGroup", min, max);
  
  return;
}

void assignParamsToCARLsim(int configID)
{
  // dummy string to make input easier
  std::string s1;

  int paramIndex=0;
  grpConnectInfo_t* gc;

  // param 1:InputGroup-ExcGroup
  paramValue[configID][paramIndex]=p->getParam(IndiID,s1="InputGroup-ExcGroup");
  gc = snn->getConnectInfo(input_exc_cid, configID);
  gc->initWt = paramValue[configID][paramIndex];
  gc->maxWt  = paramValue[configID][paramIndex];
  paramIndex++;
  // param 2: ExcGroup-ExcGroup
  paramValue[configID][paramIndex]=p->getParam(IndiID,s1="ExcGroup-ExcGroup");
  gc = snn->getConnectInfo(exc_exc_cid, configID);
  gc->initWt = paramValue[configID][paramIndex];
  gc->maxWt  = paramValue[configID][paramIndex];
  paramIndex++;
  // param 3: ExcGroup-InhGroup
  paramValue[configID][paramIndex]=p->getParam(IndiID,s1="ExcGroup-InhGroup");
  gc = snn->getConnectInfo(exc_inh_cid, configID);
  gc->initWt = 1.0f*paramValue[configID][paramIndex];
  gc->maxWt  = 1.0f*paramValue[configID][paramIndex];
  paramIndex++;
  // param 4: InhGroup-ExcGroup
  paramValue[configID][paramIndex]=p->getParam(IndiID,s1="InhGroup-ExcGroup");
  gc = snn->getConnectInfo(inh_exc_cid, configID);
  gc->initWt = -1.0f*paramValue[configID][paramIndex];
  gc->maxWt  = -1.0f*paramValue[configID][paramIndex];
  paramIndex++;

  return;
}

// Fitness function version 1.0
// function definition for the objective/fitness function 
int evaluateFitnessV1()
{
  // run the network for 10 seconds with the input on both PM and FM
#define TRIAL_NUM      10
#define TRIAL_RUN_SEC   1
#define NUM_ITERATIONS  1

  
  float firingRateInput[NUM_CONFIGS][TRIAL_NUM];
  float firingRateExc[NUM_CONFIGS][TRIAL_NUM];
  float firingRateInh[NUM_CONFIGS][TRIAL_NUM];

  float errorInput[NUM_CONFIGS]; 
  float errorExc[NUM_CONFIGS];
  float errorInh[NUM_CONFIGS]; 
    
  float averageFiringInput[NUM_CONFIGS];
  float averageFiringExc[NUM_CONFIGS];
  float averageFiringInh[NUM_CONFIGS]; 
  
  // initialize:
  for(int i=0;i<NUM_CONFIGS;i++){
    for(int j=0;j<TRIAL_NUM;j++){
      firingRateInput[i][j]=0;
      firingRateExc[i][j]=0;
      firingRateInh[i][j]=0;
    }
    errorInput[i]=0;
    errorExc[i]=0;
    errorInh[i]=0;
    
    averageFiringInput[i]=0;
    averageFiringExc[i]=0;
    averageFiringInh[i]=0;
  } 
  // the target average firing rate is 10 Hz for every group
  float targetFiringRateInput=MAX_FIRING_INPUT;
  float targetFiringRateExc=10.0f; 
  float targetFiringRateInh=2*targetFiringRateExc;

  // reset the timing tables but keep the weights as you left them
  snn->updateNetwork(true, false);
  
  //reset the spike counter monitors
  for(int i=0; i<NUM_CONFIGS; i++){
    callbackInput[i]->resetSpikeCounter();
    callbackExc[i]->resetSpikeCounter();
    callbackInh[i]->resetSpikeCounter();
  }   

  // -----------------------------------------------------------------------------
  // FITNESS FUNCTION STEP 1: run the simulations and collect the firing rate data
  // -----------------------------------------------------------------------------
  // iterate for the amount of presentation iterations
  for(int j=0; j< NUM_ITERATIONS; j++){
    // iterate over all different stimulus inputs
    for(int i=0; i<TRIAL_NUM; i++){
      //input->rates[i] = MAX_FIRING_INPUT;
      // set the correct spike rate for each configuration.
      snn->setSpikeRate(inputGroup,input);
      // run the network for TRIAL_RUN_SEC
      snn->runNetwork(TRIAL_RUN_SEC,0,GPU_MODE);
    }
  }
  // calculate firing rate here so we can just the fitness error. 
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    callbackInput[configID]->getArrayInfo(spikeDataInput[configID],sizeInput[configID]);
    callbackExc[configID]->getArrayInfo(spikeDataExc[configID],sizeExc[configID]);
    callbackInh[configID]->getArrayInfo(spikeDataInh[configID],sizeInh[configID]);
  }
  
  float spikeCountInput[NUM_CONFIGS][TRIAL_NUM]; 
  float spikeCountExc[NUM_CONFIGS][TRIAL_NUM]; 
  float spikeCountInh[NUM_CONFIGS][TRIAL_NUM]; 

  for(int configID=0;configID<NUM_CONFIGS;configID++){
    for(int i=0;i<TRIAL_NUM;i++){
      spikeCountInput[configID][i]=0;  
      spikeCountExc[configID][i]=0;  
      spikeCountInh[configID][i]=0;
    }
  }

  // calculate the average firing rate for each neuron group by dividing each total spike count
  // by the total time the simulation ran
  int min; int max;
  // Input group
  min=0; max=999;
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    for(int j=0; j<TRIAL_RUN_SEC; j++){
      for(int i=0; i<sizeInput[configID]; i++){
	if(spikeDataInput[configID][i][0]>=min && spikeDataInput[configID][i][0]<=max){
	  spikeCountInput[configID][j]=spikeCountInput[configID][j]+1;
	}
      }
      min=min+1000;
      max=max+1000;
    }
    min=0; max=999;
  }
  
  // Exc group
  min=0; max=999;
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    for(int j=0; j<TRIAL_RUN_SEC; j++){
      for(int i=0; i<sizeExc[configID]; i++){
	if(spikeDataExc[configID][i][0]>=min && spikeDataExc[configID][i][0]<=max){
	  spikeCountExc[configID][j]=spikeCountExc[configID][j]+1;
	}
      }
      min=min+1000;
      max=max+1000;
    }
    min=0; max=999;
  }

  // Inh group
  min=0; max=999;
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    for(int j=0; j<TRIAL_RUN_SEC; j++){
      for(int i=0; i<sizeInh[configID]; i++){
	if(spikeDataInh[configID][i][0]>=min && spikeDataInh[configID][i][0]<=max){
	  spikeCountInh[configID][j]=spikeCountInh[configID][j]+1;
	}
      }
      min=min+1000;
      max=max+1000;
    }
    min=0; max=999;
  }

  // -----------------------------------------------------------------------------
  // FITNESS STEP 2: Compare firing data for each group to the target firing rate 
  // and calculate error. Sum of all errors is the fitness value, which we are 
  // trying to minimize.
  // -----------------------------------------------------------------------------
  // calculate the average firing rate at every second
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    for(int j=0; j<TRIAL_RUN_SEC; j++){
      firingRateInput[configID][j]=spikeCountInput[configID][j]/INPUT_SIZE;
      firingRateExc[configID][j]=spikeCountExc[configID][j]/EXC_SIZE;
      firingRateInh[configID][j]=spikeCountInh[configID][j]/INH_SIZE;
    }
  }
  
  // calculate the average firing rate over the whole run
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    for(int j=0; j<TRIAL_RUN_SEC; j++){
      // sum the average firing rates over all of the 1 second time trials
      averageFiringInput[configID]=averageFiringInput[configID]+firingRateInput[configID][j];
      averageFiringExc[configID]=averageFiringExc[configID]+firingRateExc[configID][j];
      averageFiringInh[configID]=averageFiringInh[configID]+firingRateInh[configID][j];
    }
    // calculate the average firing rate over all the 1 second time trials
    averageFiringInput[configID]=averageFiringInput[configID]/TRIAL_RUN_SEC;
    averageFiringExc[configID]=averageFiringExc[configID]/TRIAL_RUN_SEC;
    averageFiringInh[configID]=averageFiringInh[configID]/TRIAL_RUN_SEC;
    
    // calculate the error for each configuration
    errorInput[configID]=abs(averageFiringInput[configID]-targetFiringRateInput);
    errorExc[configID]=abs(averageFiringExc[configID]-targetFiringRateExc);
    errorInh[configID]=abs(averageFiringInh[configID]-targetFiringRateInh);
      
    // fitness is the sum of the error from all of the groups in a configuration
    fitness[configID]=+errorExc[configID]+errorInh[configID];
  }
    
  // -----------------------------------------------------------------------------
  // FITNESS STEP 3: Output data to file
  // -----------------------------------------------------------------------------
  FILE* fpResp;
  fpResp = fopen("results/tuneFiringRates/resp.txt","a");
  // output every SNN
  for(int configID=0; configID<NUM_CONFIGS; configID++){
    fprintf(fpResp,"Generation Number: %d\n",genCounter);
    fprintf(fpResp,"Global Individual ID: %d\n",globalIndiID-configID); 
    fprintf(fpResp,"Parameter Values: \n");
    fprintf(fpResp,"-----------------\n");
    for(int i=0;i<NUM_PARAMS;i++){
      fprintf(fpResp,"param %d: %f\n",i,paramValue[configID][i]);
    }
    fprintf(fpResp,"Error for each group\n:");
    fprintf(fpResp,"Input Group: Error (not counted): %f \n",errorInput[configID]);
    fprintf(fpResp,"Exc Group: Error: %f \n",errorExc[configID]);
    fprintf(fpResp,"Inh Group: Error: %f \n",errorInh[configID]);
    
    fprintf(fpResp,"Fitness: %f\n",fitness[configID]);
  }
  fprintf(fpResp,"\n\n");
  
  fclose(fpResp);

  // free all the memory
  for(int i=0;i<NUM_CONFIGS;i++){
    for(int j=0;j<sizeInput[i];j++){
      delete[] spikeDataInput[i][j];
      //printf("lots of deleting spikeDataInput[%d][%d]\n",i,j);
    }
    delete[] spikeDataInput[i];
    printf("deleting spikeDataInput[%d]\n",i); 
    for(int j=0;j<sizeExc[i];j++){
      delete[] spikeDataExc[i][j];
      //printf("lots of deleting spikeDataExc[%d][%d]\n",i,j);
    }
    delete[] spikeDataExc[i];
    printf("deleting spikeDataExc[%d]\n",i); 
    for(int j=0;j<sizeInh[i];j++){
      delete[] spikeDataInh[i][j];
      //printf("lots of deleting spikeDataInh[%d][%d]\n",i,j);
    }
    delete[] spikeDataInh[i];
    printf("deleting spikeDataInh[%d]\n",i); 
    sizeInput[i]=0;
    sizeExc[i]=0;
    sizeInh[i]=0;
  }



  return 0;
}
