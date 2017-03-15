// UCI Documentation:
// -----------------------------------------------------------------------------
/* Kris Carlson 10/31/2011
   Toy Program to test several EO-ES features.  This program must be called
   with a parameter file.  The parameter file specifies the EO engine options
   to be passed to the parser.  The real_value.h file contains the fitness
   function to be optimized.  Many different EA algorithms can be implemented
   by changing the paramter file.
*/

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif
#include <snn.h>
//TODO: THESE SHOULD BE GONE!!
#include "PropagatedSpikeBuffer.h"
//TODO: Still need this one.
#include "../common/stimGenerator.h"
// include the PTI framework classes and functions
#include <pti.h>   

using namespace std;

//#define SIM_MODE	    CPU_MODE
#define SIM_MODE            GPU_MODE 
#define SPON_RATE 	    1.0
#define REFRACTORY_PERIOD   1.0
// #define MAX_POISS_RATE    20 // also try 75 hz // now a variable
#define NUM_PATTERNS        40
#define PI                  3.1415926535897
#define RAND_SEED           42
// the number of networks we run on the GPU simultaneously.
#define NUM_CONFIG 10
#define DUMMY_WEIGHT (0.0160f) // was 0.2f
#define MAX_TARGET_FIRING_RATE  60 //used in fitness function to use as target firing rate.
// Jay's STP parameters.
#define GABOR_STP_U_Inh  	0.5
#define	GABOR_STP_tD_Inh 	800
#define	GABOR_STP_tF_Inh 	1000
#define GABOR_STP_U_Exc  	0.2
#define	GABOR_STP_tD_Exc 	700
#define	GABOR_STP_tF_Exc 	20
#define NUM_OUTPUT              4 //number of output exc and inh neurons.
#define TUNING_SIGMA            15*(PI/180)

// I hate global variables -- KDC
int inputOnGrp, inputOffGrp, bufferOnGrp, bufferOffGrp, excGrp, inhGrp;
int inputOnToBufferOn_cid, inputOffToBufferOff_cid, bufferOnToExc_cid, bufferOffToExc_cid, excToInh_cid, inhToExc_cid;
// This defines the size of the input patterns and 4 associated neural groups.
int imageX = 32;
int imageY = 32;
int nPois = imageX*imageY;
PoissonRate* spikeResponseOn[NUM_PATTERNS];
PoissonRate* spikeResponseOff[NUM_PATTERNS];
double* spikeRateOn[NUM_PATTERNS];
double* spikeRateOff[NUM_PATTERNS];
// this will serve as noise/no input to both on and off groups
// across all neurons 
PoissonRate* noiseGen;
double noiseRate=1;// 1 Hz noise rate

double* gaborPattern[NUM_PATTERNS];
double* CPSGratingPattern[NUM_PATTERNS];
//as measured from positive y axis (counter-clockwise).
//double angle[]={1*PI/8,2*PI/8,3*PI/8,4*PI/8,5*PI/8,6*PI/8,7*PI/8,8*PI/8}; 
double angle[NUM_PATTERNS];

float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;

// global vars to handle printing out information in evaluateFitness that
// has individual and generation information attached to it.  
uint32_t globalIndiId = 0;
int      globalGenCounter = 0;
char tuningCurvesDir[100] = "results/SORFs/tuningCurvesFiles";

// these variables eventually get set by the evolutionary algorithm
// perhaps they should not be global.  TAGS:TODO.
float ALPHA_LTP_INPUT[NUM_CONFIG];
float TAU_LTP_INPUT[NUM_CONFIG];
float ALPHA_LTD_INPUT[NUM_CONFIG];
float TAU_LTD_INPUT[NUM_CONFIG];
float TAU_DELTA_INPUT[NUM_CONFIG];
float ALPHA_LTP_INH[NUM_CONFIG];
float TAU_LTP_INH[NUM_CONFIG];
float ALPHA_LTD_INH[NUM_CONFIG];
float TAU_LTD_INH[NUM_CONFIG];
float TAU_DELTA_INH[NUM_CONFIG];
float MAX_POISS_RATE[NUM_CONFIG];//probably should change this to be not all caps. TAGS:TODO
float inh_weight_param[NUM_CONFIG];
float baseFiring_excGrp_param[NUM_CONFIG];
float buff_weight_param[NUM_CONFIG];
float exc_weight_param[NUM_CONFIG];
float baseFiring_inhGrp_param[NUM_CONFIG];

// Currently not using this.  Just using the built in EO steady-state
// calculator.  Both of these are used to calculate steady-state.
//double* prevFitness=NULL;
//double prevIndiId=-1;
FILE* fpResp;

// function to test against the real data.  It's a Gaussian tuning curve that
// extends across 90 degrees.  All arguments are in radians.  Except firing rate
// which is in hertz.
float tuningCurve(float _rMax, float _maxAngle, float _sigma, float _noiseRate, float _angle)
{
  return _noiseRate+(_rMax-_noiseRate)*exp(-0.5*pow((_angle-_maxAngle)/(_sigma),2));
}

// 3-component fitness function to test the fitness of each individual (SNN).
int evaluateFitness(CpuSNN& snn, double* _fitness)
{
  // Phase 0: Initialize values and set target firing rates
  // ---------------------------------------------------------------------
  int continueSimulation=0;
  int nNeurons = NUM_OUTPUT;
  int nStim    = NUM_PATTERNS;
  float response[NUM_CONFIG][nStim][nNeurons];
  float targetResponse[NUM_CONFIG][nStim][nNeurons];
  float scaledTargetResponse[NUM_CONFIG][nStim][nNeurons];
  float scaledResponse[NUM_CONFIG][nStim][nNeurons];
  float maxResponse[NUM_CONFIG][nNeurons];
  int maxStimId[NUM_CONFIG][nNeurons];

  // Initialize response array.
  for(int i=0;i<NUM_CONFIG;i++){
    for(int j=0;j<nStim;j++){
      for(int k=0;k<nNeurons;k++){
        response[i][j][k]=0;
        targetResponse[i][j][k]=0;
      }
    }
  }
  // Initialize response variables.
  for(int i=0;i<NUM_CONFIG;i++){
    for(int j=0;j<nNeurons;j++){
      maxResponse[i][j]=0;
      maxStimId[i][j]=-1;
    }
  }

  // disable STDP for all groups (and configurations)
  snn.setSTDP(ALL, false);

  // reset the timing tables but keep the weights as you left them
  snn.updateNetwork(true, false);

  // make sure we are actually copying data from the GPU to the CPU
  // TAGS:TODO this should be turned off after we are finished.
  if(SIM_MODE==GPU_MODE)
    snn.setCopyFiringStateFromGPU(true);

#define NUM_ITERATIONS      20
  //#define NUM_ITERATIONS      2 // short for testing purposes
#define TEST_RUN_SEC        1.0
#define TEST_RUN_MSEC       0.0
#define TEST_OFF_RUN_SEC    0.0
#define TEST_OFF_RUN_MSEC   500

  // Phase 1: find the average response of neurons to stimulus (20 runs of 40 patterns)
  // I ran the network for 500 ms first, to clear it, and then reset the spiking rates 
  // so there is not a lot of overlap for the calculation of the spike rates in response 
  // to the patterns.
  // ---------------------------------------------------------------------------------

  // Iterate over the NUM_ITERATIONS and network configurations. Count spikes for 
  // every neuron for every pattern and every configuration for NUM_ITERATIONS.
  for (int iter=0; iter < NUM_ITERATIONS; iter++) {
    // for each stim, sum neuronal responses
    for (int stim=0; stim < nStim; stim++) {

      // first run the network for 500 ms with noise input so we have
      // an accurate count.  (This is probably unnecessary.)
      /*
	for(int configId=0;configId<NUM_CONFIG;configId++){
	// set the correct spike rate for each configuration.
	snn.setSpikeRate(inputOnGrp, noiseGen, 1, configId);
	snn.setSpikeRate(inputOffGrp, noiseGen, 1, configId);
	}
	// run the network with noise/background input
	snn.runNetwork(TEST_RUN_SEC, TEST_RUN_MSEC, SIM_MODE);
      */  //seems to be accurate without this.
      // reset spike counters and set spike rates for each configuration.
      for(int configId=0;configId<NUM_CONFIG;configId++){
        //we need to reset the correct spike counter group each config.
        int currentGrpId = snn.getGroupId(excGrp,configId);
        snn.resetSpikeCntUtil(currentGrpId);	
        // set the correct spike rate for each configuration.
        snn.setSpikeRate(inputOnGrp, (spikeResponseOn[stim]), 1, configId);
        snn.setSpikeRate(inputOffGrp, (spikeResponseOff[stim]), 1, configId);
      }

      // run the network
      snn.runNetwork(TEST_RUN_SEC, TEST_RUN_MSEC, SIM_MODE);
      // count the spikes for each configuraton.
      for(int configId=0; configId<NUM_CONFIG; configId++){
        int currentGrpId = snn.getGroupId(excGrp,configId);
        unsigned int* spikeCnt = snn.getSpikeCntPtr(currentGrpId,SIM_MODE);
        // sum all the responses from each neuron to each stimulus for 16 runs.
        for (int neu=0; neu < nNeurons; neu++){
          assert(spikeCnt[neu]>=0);
          response[configId][stim][neu] += spikeCnt[neu];
        }
      }
    }
  }

  // Divide spike count by NUM_ITERATIONS for every neuron response
  // to every stimulus pattern for every network configuration.
  // Find max response for every neuron for every stimID.  Record
  // max response and respective stimulus ID.
  for(int configId=0; configId<NUM_CONFIG; configId++){
    for(int neu=0; neu < nNeurons; neu++) {
      // float maxResponse = 0.0;
      // int   maxStimId = -1;
      for(int stim=0; stim < nStim; stim++)  {
        // response = response/NUM_ITERATIONS, aka the average
        response[configId][stim][neu] /= NUM_ITERATIONS;
        if (response[configId][stim][neu] >= maxResponse[configId][neu]) {
          // record stim id that makes neuron spike most
          maxResponse[configId][neu] = response[configId][stim][neu];
          maxStimId[configId][neu] = stim;
        }
      }
    }
  }
  // Phase 2: Fitness Calculation
  // ---------------------------------------------------------------------------------
  // Compare the error between the desired response of the neurons to the
  // actual response of the neurons. We want to minimize this fitness.  I changed it 
  // in the pti.h file in the function real_value from maximize to minimize.

  // initialize fitness
  memset(_fitness, 0, sizeof(double)*NUM_CONFIG);

  // Fitness Compontent 1: Orthogonality/Spanning the stimulus space
  int targetDist = 0;  // we want to do integer math to get the right spacing
  int currentId = 0; // the current stimId we are examining
  int spacing = 0;
  int curMinDist = 0; int minDist = 0;
  int cumError = 0;

  float fitnessStep1[NUM_CONFIG];
  float neuronError[NUM_OUTPUT];

  // initialize fitnessStep1 to zero
  for(int configId=0;configId<NUM_CONFIG;configId++)
    fitnessStep1[configId]=0;

  // initialize neuron errors to zero
  for(int nid=0;nid<NUM_OUTPUT;nid++)
    neuronError[nid]=0;

  // calculate what 'optimal' spacing would be to span the stimulus.
  targetDist = NUM_PATTERNS/NUM_OUTPUT;
  assert(NUM_PATTERNS>=NUM_OUTPUT);  // we must have more patterns than neurons.
  assert(targetDist>=0);  // 
  // cycle through all configurations
  for(int configId=0;configId<NUM_CONFIG;configId++){
    // cycle through the maxStimId of each neuron and calculate the distance.
    cumError = 0;
    for(int nid=0;nid<NUM_OUTPUT;nid++){
      // initialize the neuronError Matrix
      neuronError[nid]=0;
      currentId = maxStimId[configId][nid];
      // intialized to the max distance between any two max stim ids 
      minDist = NUM_PATTERNS/2; 
      for(int i=0;i<NUM_OUTPUT;i++){
        if(i!=nid){
          spacing=abs(currentId-maxStimId[configId][i]);
          // we consider these angles to loop, we have a circular
          // data structure, not a linear one.  The below code takes
          // care of that.
          if(spacing>NUM_PATTERNS/2)
            spacing=(NUM_PATTERNS)-spacing;

          minDist=min(minDist,spacing);

          if(spacing==0)
            break; // break from this for loop
        }
      }
      // assign the error based on the distance between the maxId's of you
      // and your nearest neighbor.  If you are further from your nearest
      // neighbor than the target, assign the error to be zero, else take
      // the different between actual and target and take the absolute value.
      if(minDist<targetDist){
        neuronError[nid]=abs(minDist-targetDist);
      }
      //sum all neuron errors for this particular configuration
      cumError+=neuronError[nid];
    }
    fitnessStep1[configId]=cumError;
  }
  // Fitness Component 2: Tuning curves
  // 11/01/2012: why not just calculate the error real time instead of looking up
  // something from a correct response table?  The algorithm: find the max responding
  // stimulus angle.  Calculate the error to the right 20, and to the left 19.  You
  // could just go from 0 to 39 and shift by the max stim angle also. Use
  // mod to make sure you are still in the correct range.  Iterate through a for loop
  // and do the calculation there. We included a background firing rate for the target
  // tuning curve that is just equal to the noiseRate.

  // new fitness step 2 calculation
  float maxRespScaleFactor[NUM_CONFIG];
  int maxNeuronIndex[NUM_CONFIG];
  float fitnessStep2[NUM_CONFIG];
  int withinRange[NUM_CONFIG][NUM_PATTERNS][nNeurons];
  float errorSize[NUM_CONFIG][NUM_PATTERNS][nNeurons];
  float rightMatrix[NUM_PATTERNS];
  float leftMatrix[NUM_PATTERNS];

  //initialize error matrices
  for(int configId=0; configId<NUM_CONFIG; configId++){
    fitnessStep2[configId]=0;
    for(int stimId=0; stimId < NUM_PATTERNS; stimId++){ 
      for(int nid=0; nid < nNeurons; nid++){
        withinRange[configId][stimId][nid]=0;
        errorSize[configId][stimId][nid]=0;
      }
      rightMatrix[stimId]=0;// think carefully about this
      leftMatrix[stimId]=0; // think carefully about this
    }
  }

  for(int configId=0; configId<NUM_CONFIG; configId++){
    for(int neu=0; neu<nNeurons; neu++){
      if(maxResponse[configId][neu]<1)
        maxRespScaleFactor[configId]=1;
      else
        maxRespScaleFactor[configId]=maxResponse[configId][neu];

      float maxAngle = 0;
      float maxFiringRate = 0;
      float correctRate = 0;
      float cumPatternError = 0;

      for(int stim=0;stim<NUM_PATTERNS;stim++){
        //start at the max neuron and keep calculating the error
        maxAngle = angle[maxStimId[configId][neu]];
        maxFiringRate = maxRespScaleFactor[configId];
        // This is required so that we have looping around indices
        // of the angle matrix. 
        // create matrix of angles to the right and left of max.
        for(int i=0;i<NUM_PATTERNS;i++){
          rightMatrix[(maxStimId[configId][neu]+i)%NUM_PATTERNS]=angle[maxStimId[configId][neu]]+i*(PI/NUM_PATTERNS);
        }
        for(int j=0;j<NUM_PATTERNS;j++){
          leftMatrix[(NUM_PATTERNS+maxStimId[configId][neu]-j)%NUM_PATTERNS]=angle[maxStimId[configId][neu]]-j*(PI/NUM_PATTERNS);
        }
        // create correct response matrix.  Just use the matrix with the biggest response.
        targetResponse[configId][stim][neu]=max(tuningCurve(maxFiringRate, maxAngle, TUNING_SIGMA, noiseRate, rightMatrix[stim]),tuningCurve(maxFiringRate, maxAngle, TUNING_SIGMA, noiseRate, leftMatrix[stim]));
        // scale all target response and stimulus responses with respect to the MAX_TARGET_FIRING_RATE.
        assert(MAX_TARGET_FIRING_RATE>0);// this should never be 0.  It makes no sense.
        // create absolute target response matrix.  (everything is scaled for a max of MAX_TARGET_FIRING_RATE)
        scaledTargetResponse[configId][stim][neu]=max(tuningCurve(MAX_TARGET_FIRING_RATE, maxAngle, TUNING_SIGMA, noiseRate, rightMatrix[stim]),tuningCurve(MAX_TARGET_FIRING_RATE, maxAngle, TUNING_SIGMA, noiseRate, leftMatrix[stim]));

        correctRate=scaledTargetResponse[configId][stim][neu];

        // create absolute stimuluse response matrix.  (everything is scaled for a max of MAX_TARGET_FIRING_RATE)
        // if the current response is the max response for this neuron, then the max response is set to MAX_TARGET_FIRING_RATE.
        scaledResponse[configId][stim][neu]=response[configId][stim][neu]*(MAX_TARGET_FIRING_RATE/maxFiringRate);

        errorSize[configId][stim][neu]=abs(scaledResponse[configId][stim][neu]-correctRate);

        // we know the cum error will never be larger than (NUM_PATTERNS-1)*MAX_TARGET_FIRING_RATE (39*60 in my case). (1 pattern will have 0 
        // error (it is the max!) and everything other response has a max value of MAX_TARGET_FIRING_RATE. Over all neurons, the neuron cumulative 
        // error will be NUM_OUTPUT*cumulative error.

        cumPatternError = cumPatternError + errorSize[configId][stim][neu];
      }
      fitnessStep2[configId] = fitnessStep2[configId] + cumPatternError;
    }
  }

  // maybe I shouldn't have a cap?
#define FIT3_CAP 60 //the value we cap the fit3 error at.
  // Fitness Component 3
  // Calculate the difference the maximum firing rate and the target firing rate.
  float fitnessStep3[NUM_CONFIG];
  for(int i=0; i< NUM_CONFIG; i++){
    fitnessStep3[i] = 0;
  }
  // Currently: 60 Hz
  float maxTargetFiringRate=MAX_TARGET_FIRING_RATE;
  for(int configId=0; configId<NUM_CONFIG; configId++){
    for(int neu=0; neu<nNeurons; neu++){
      float tmp = sqrt(pow(maxResponse[configId][neu]-maxTargetFiringRate,2));
      // cap error at 60.
      if(tmp>FIT3_CAP)
        tmp = FIT3_CAP;
      fitnessStep3[configId]=fitnessStep3[configId]+tmp;
    }
  }
  // create normalizing terms
  float fitWorst1 = NUM_PATTERNS; 
  float fitWorst2 = NUM_PATTERNS; // not used, included for completeness
  float fitWorst3 = FIT3_CAP*NUM_OUTPUT;

  // Final fitness calculation
  float fitGood1 = 15; // this may be slightly severe, I could increase it.
  float fitGood2 = 325*NUM_OUTPUT;//124;
  float fitGood3 = 10*NUM_OUTPUT; // not used, included for completeness
  int   printCode[NUM_CONFIG];// just use this to print out correct message
  float fit3WeightFactor = 4.4;// gives max firing rate a bigger factor                                              
  
  float oldFitnessValue[NUM_CONFIG]; //for reference 
  for(int i=0; i<NUM_CONFIG;i++){
    oldFitnessValue[i]=0;
  }


  for(int configId=0; configId<NUM_CONFIG; configId++){
    // scale steps in terms of importance, the larger the multiplier, the more important.
    printCode[configId]=0;
    if(fitnessStep2[configId]>fitGood2){
      // if fit2 is poor, set the other values to max values to just concentrate on fit2.
      _fitness[configId] = fitnessStep2[configId]+fitWorst1+fit3WeightFactor*fitWorst3;
      oldFitnessValue[configId] = fitnessStep2[configId]+fitWorst1+fitWorst3;
      printCode[configId]=1;
    }
    else if(fitnessStep2[configId]<=fitGood2 && fitnessStep1[configId]>fitGood1){
      // concentrate on just fit 2 and fit 1
      _fitness[configId] = fitnessStep1[configId]+fitnessStep2[configId]+fit3WeightFactor*fitWorst3;
      oldFitnessValue[configId] = fitnessStep1[configId]+fitnessStep2[configId]+fitWorst3;
      printCode[configId]=2;
    }
    else{
      // concentrate on all three fitnesses, because fit1 and fit2 are good enough
      _fitness[configId] = fitnessStep1[configId]+fitnessStep2[configId]+fit3WeightFactor*fitnessStep3[configId];
      oldFitnessValue[configId] = fitnessStep1[configId]+fitnessStep2[configId]+fitnessStep3[configId];
      printCode[configId]=3;
    }
    assert(printCode[configId]!=0);
  }

  // 
  // Phase 3: Output Phase
  // ---------------------------------------------------------------------------------
  // print out the angle and the neurons responses
  // eventually output this to files
  // debugging:

  for(int configId=0; configId<NUM_CONFIG; configId++){
    if(printCode[configId]==1)
      fprintf(fpResp,"Optimizing tuning curve shape as overall fitness.\n");
    else if( printCode[configId]==2)
      fprintf(fpResp,"Optimizing tuning curve shape and orthogonal \n curves as overall fitness.\n");
    else
      fprintf(fpResp,"Optimizing tuning curve shape, orthogonal \n curves, and maximum firing rate as overall fitness.\n");
    fprintf(fpResp,"Orthogonality fitness for this individual = %f (<%f)\n",fitnessStep1[configId],fitGood1);
    fprintf(fpResp,"Relative firing rate fitness for this individual = %f (<%f)\n",fitnessStep2[configId],fitGood2);
    fprintf(fpResp,"Maximum firing rate fitness for this individual = %f (<%f)\n",fitnessStep3[configId],fitGood3);
    fprintf(fpResp,"Fitness for this individual (older fitness) = %f (%f)\n",_fitness[configId],oldFitnessValue[configId]);
    fprintf(fpResp,"Number of Stimulus Angles = %d\n", nStim);
    for(int stim=0; stim < nStim; stim++) {
      fprintf(fpResp, " % 8.3f ", angle[stim]*180/PI);
      if((stim+1)%10 == 0)
        fprintf(fpResp,"\n");
    }
    fprintf(fpResp, "\n");

    // calculate the average response
    fprintf(fpResp, "Number of  Neuron responses = %d\n",nNeurons);  
    fprintf(fpResp, "Format: actual firing rate (target firing rate)\n");
    fprintf(fpResp, "-----------------------------------------------\n");
    // output the response/target matrix.  On the left, the response,
    // to the right in parentheses is the target/desired firing rate
    fprintf(fpResp,"\n");
    for(int stim=0; stim<nStim; stim++){
      for(int neu=0; neu<nNeurons; neu++){
        maxRespScaleFactor[configId]=maxResponse[configId][neu];
        maxNeuronIndex[configId]=maxStimId[configId][neu];
        fprintf(fpResp,"% 8.3f (% 8.3f)", response[configId][stim][neu], targetResponse[configId][stim][neu]);
      }
      fprintf(fpResp,"\n");
    }
    // DEBUGGING
    // print out the maxStimId
    fprintf(fpResp, "Debugging: maxStimId(%d)\n",nNeurons);      
    for(int neu=0;neu<nNeurons;neu++){
      fprintf(fpResp,"%d ",maxStimId[configId][neu]);
    }
    fprintf(fpResp,"\n");
    fflush(fpResp);
  }

  // I need to fix this so I can see all the tuning curves. TAGS:TODO

  // output spiking data for all angles and neurons
  // loop over all configurations

  FILE* tuning_fid; 
  string tuningCurvesFileName[NUM_CONFIG];

  for(int configId=0; configId<NUM_CONFIG; configId++){
    // order: actual response of neurons, target response of neurons

    char tmpFileName[200];
    // I think this line is causing the reversal in the filenames and the individual numbers.
    //sprintf(tmpFileName,"%s/tuningCurves_gen_%d_indi_%d.dat",tuningCurvesDir,(globalGenCounter-1),globalIndiId-(configId));
    sprintf(tmpFileName,"%s/tuningCurves_gen_%d_indi_%d.dat",tuningCurvesDir,(globalGenCounter-1),globalIndiId-NUM_CONFIG+configId);
    tuningCurvesFileName[configId]=tmpFileName;

    tuning_fid = fopen((tuningCurvesFileName[configId]).c_str(), "wb");
    assert(tuning_fid != NULL);
    // order we write: For each neuron, write its response to every angle
    // from 0-PI (non-inclusive).
    for(int nid=0; nid<nNeurons; nid++){
      for(int stim=0; stim<nStim; stim++){
        fwrite(&(response[configId][stim][nid]),sizeof(float),1,tuning_fid);
      }
    }
    // do the same, but do it for what the target firing rates should be.
    for(int nid=0; nid<nNeurons; nid++){
      for(int stim=0; stim<nStim; stim++){
        fwrite(&(targetResponse[configId][stim][nid]),sizeof(float),1,tuning_fid);
      }
    }
    fclose(tuning_fid);
  }
  return 0;
}

int main_fitness_function(int argc, char *argv[])
{
  fpResp = fopen("results/SORFs/resp.txt","a");

  // BEGIN SNN initialization
  // --------------------------------------------------------------------------------------------------


  CpuSNN snn("SpikingNeuralNetworkEvolvedUsingEO",NUM_CONFIG,RAND_SEED);

  //  create a population or group of neurons with the label "inhib", having
  inputOnGrp     = snn.createSpikeGeneratorGroup("spike_input_on", nPois, EXCITATORY_POISSON);
  inputOffGrp    = snn.createSpikeGeneratorGroup("spike_input_off", nPois, EXCITATORY_POISSON);
  bufferOnGrp    = snn.createGroup("buffer_on",  nPois, EXCITATORY_NEURON);
  bufferOffGrp   = snn.createGroup("buffer_off", nPois, EXCITATORY_NEURON);
  excGrp         = snn.createGroup("excit", NUM_OUTPUT, EXCITATORY_NEURON);
  inhGrp         = snn.createGroup("inhib", NUM_OUTPUT, INHIBITORY_NEURON);

  // parameter value assignment
  snn.setNeuronParameters(bufferOnGrp, 0.02f, 0.2f, -65.0f, 8.0f, ALL);
  snn.setNeuronParameters(bufferOffGrp, 0.02f, 0.2f, -65.0f, 8.0f, ALL);
  snn.setNeuronParameters(excGrp, 0.02f, 0.2f,-65.0f, 8.0f, ALL);
  snn.setNeuronParameters(inhGrp, 0.1f, 0.2f, -65.0f, 2.0f, ALL);

  // connections
  inputOnToBufferOn_cid   = snn.connect(inputOnGrp,  bufferOnGrp,  "one-to-one", 0.2f, 0.6f, 1.0, 1, 1, SYN_FIXED);
  inputOffToBufferOff_cid = snn.connect(inputOffGrp, bufferOffGrp, "one-to-one", 0.2f, 0.6f, 1.0, 1, 1, SYN_FIXED);
  // The initial connections for these weights are labelled DUMMY_WT because they are
  // replaced by the evolutionary algorithm.
  bufferOnToExc_cid       = snn.connect(bufferOnGrp,  excGrp, "full", DUMMY_WEIGHT/4, DUMMY_WEIGHT, 1.0, 1, 1, SYN_PLASTIC, "random");
  bufferOffToExc_cid      = snn.connect(bufferOffGrp, excGrp, "full", DUMMY_WEIGHT/4, DUMMY_WEIGHT, 1.0, 1, 1, SYN_PLASTIC, "random");

  excToInh_cid            = snn.connect(excGrp, inhGrp, "full", DUMMY_WEIGHT/4, DUMMY_WEIGHT, 1.0,  1, 1, SYN_PLASTIC, "random");

  inhToExc_cid            = snn.connect(inhGrp, excGrp, "full", -0.21, -0.21, 1.0, 1, 1, SYN_FIXED); // original
  //inhToExc_cid            = snn.connect(inhGrp, excGrp, "full", -1*DUMMY_WEIGHT, -1*DUMMY_WEIGHT, 1.0, 1, 1, SYN_FIXED);

  //DUMMY_WEIGHT = 0.20 //DUMMY_WEIGHT/4 = 0.05



  // delete previous spike files
  system("rm excGrp_spikes.dat");
  system("rm inhGrp_spikes.dat");
  system("rm bufferOnGrp_spikes.dat");
  system("rm bufferOffGrp_spikes.dat");
  system("rm inputOnGrp_spikes.dat");
  system("rm inputOffGrp_spikes.dat");

  /*
  // I don't want to output these now.
  snn.setSpikeMonitor(excGrp, "excGrp_spikes.dat");
  snn.setSpikeMonitor(inhGrp, "inhGrp_spikes.dat");
  snn.setSpikeMonitor(bufferOnGrp, "bufferOnGrp_spikes.dat");
  snn.setSpikeMonitor(bufferOffGrp, "bufferOffGrp_spikes.dat");
  snn.setSpikeMonitor(inputOnGrp, "inputOnGrp_spikes.dat");
  snn.setSpikeMonitor(inputOffGrp,"inputOffGrp_spikes.dat");
  */

  // show logout every 100 secs, enabled with level 1 and output to stdout.
  // make it show output every 100 seconds so I don't have to look at it.
  snn.setLogCycle(100, 1, stdout);

  snn.setConductances(ALL,true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb, ALL);
  //snn.setSTDP(excGrp, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
  //#define HOMEO_FACTOR                    (5.0)
  //#define HOMEO_AVERAGE_TIME_SCALE        (10.0)
  //I may need to change the homeofactor to 0.01
#define HOMEO_FACTOR 			(0.1) //original
#define HOMEO_AVERAGE_TIME_SCALE 	(10.0) // original: begin with this value from Jay's
  snn.setHomeostasis(excGrp, true, HOMEO_FACTOR, HOMEO_AVERAGE_TIME_SCALE, ALL);
  snn.setHomeostasis(inhGrp, true, HOMEO_FACTOR, HOMEO_AVERAGE_TIME_SCALE, ALL);

  //set STP (NEW!) according to Jay's code, we set them all except the inhibitory
  //connection.  Haven't thought this through or tested it yet.
  /*
    snn.setSTP(ALL,  true, GABOR_STP_U_Exc, GABOR_STP_tD_Exc, GABOR_STP_tF_Exc, ALL);
    snn.setSTP(inhGrp, true, GABOR_STP_U_Inh, GABOR_STP_tD_Inh, GABOR_STP_tF_Inh, ALL);
    snn.setSTP(inhGrp, false, ALL); 
  */

  // END SNN initialization
  // --------------------------------------------------------------------------------------------------

  // BEGIN stimulus setup
  // --------------------------------------------------------------------------------------------------

  for(int i=0;i<NUM_PATTERNS;i++){
    CPSGratingPattern[i]=new double[imageX*imageY];
    spikeRateOn[i]=new double[imageX*imageY];
    spikeRateOff[i]=new double[imageX*imageY];
    spikeResponseOn[i]=new PoissonRate(imageX*imageY);
    spikeResponseOff[i]=new PoissonRate(imageX*imageY);
  }

  // initialize all matrices to zero entries
  for(int i=0;i<NUM_PATTERNS;i++){
    for(int j=0;j<imageX*imageY;j++){
      CPSGratingPattern[i][j]=0;
      spikeRateOn[i][j]=0;
      spikeRateOff[i][j]=0;
      (spikeResponseOn[i])->rates[j]=0;
      (spikeResponseOff[i])->rates[j]=0;
    }
    // initialize the angle array to the correct values.
    angle[i]=i*(PI/NUM_PATTERNS); // from 8 angles (0 to pi) to 40 angles (0 to pi)
  }


  noiseGen=new PoissonRate(imageX*imageY);
  for(int i=0;i<imageX*imageY;i++){
    noiseGen->rates[i]=noiseRate;
  }

  for(int i=0;i<NUM_PATTERNS;i++){
    // Create the counter-phase sinusoidal grating patterns
    CPSGrating(CPSGratingPattern[i],imageX*imageY,angle[i],0.5,PI/2,1);
  }

  for(int i=0;i<NUM_PATTERNS;i++){
    // Match the firing rate responses to gabor/CPS grating patterns
    // gabor/CPS grating pattern to match spike rate to, spike response,
    // single dimension of square matrix, max response, min
    // response, 1=on, 0=off
    getRate(CPSGratingPattern[i],spikeRateOn[i],imageY,1,0,1);
    getRate(CPSGratingPattern[i],spikeRateOff[i],imageY,1,0,0);
  }
  // set the rates in the PoissionRate objects for all eight patterns
  // these are just initilization values, because the actual value must
  // be multiplied by MAX_POISS_RATE which is specific to each network
  // configuration.
  for(int i=0;i<NUM_PATTERNS;i++){
    // for all pixels in each pattern
    for(int j=0;j<imageX*imageY;j++){
      (spikeResponseOn[i])->rates[j]=spikeRateOn[i][j];
      (spikeResponseOff[i])->rates[j]=spikeRateOff[i][j];
    }
  }

  // END stimulus setup
  // --------------------------------------------------------------------------------------------------

  // BEGIN EO initialization
  // ---------------------------------------------------------------------------------------------------

  //double fitness;
  //we need to make this into an array
  double fitness[NUM_CONFIG];

  std::string s1;

  // InitializeParamTuning(argc, argv);
  system("mkdir -p results/SORFs");
  InitializeParamTuning("examples/SORFs/ESEA-plus.param");
  // constructor
  ParamTuning *p = new ParamTuning("results/SORFs/eoOutput.txt");

  // all alphas are eventually multiplied by 0.3 originally,
  // I don't do this anymore so I should do this here!
#define STDP_SCALE                 (DUMMY_WEIGHT/500.0)//=0.2/500 = 4e-4

  //#define STDP_SCALE                 (DUMMY_WEIGHT/250.0)//(DUMMY_WEIGHT/500.0)
#define ALPHA_SCALE                 0.30
  // param 1: inhibitory to excitatory weights
  p->addParam("inhGrp_exc_wt", 0.1, 0.5); 
  // param 2: tau LTP input
  p->addParam("tau_ltp_input", 10.0, 60.0);
  // param 3: tau DELTA input
  p->addParam("tau_delta_input", -5.0, 40.0);
  // param 4: alpha LTP input
  p->addParam("alpha_ltp_input", ALPHA_SCALE*STDP_SCALE, ALPHA_SCALE*STDP_SCALE*5); 
  // param 5: alpha LTD input
  p->addParam("alpha_ltd_input", ALPHA_SCALE*STDP_SCALE, ALPHA_SCALE*STDP_SCALE*5); 
  // param 6: tau LTP inh
  p->addParam("tau_ltp_inh", 10.0, 60.0); 
  // param 7: tau DELTA inh
  p->addParam("tau_delta_inh", -5.0, 40.0);
  // param 8: alpha LTP inh
  p->addParam("alpha_ltp_inh", ALPHA_SCALE*STDP_SCALE, ALPHA_SCALE*STDP_SCALE*5); 
  // param 9: alpha LTD inh
  p->addParam("alpha_ltd_inh", ALPHA_SCALE*STDP_SCALE, ALPHA_SCALE*STDP_SCALE*5);
  // param 10: poisson rate - this should be centered around 25 Hz
  p->addParam("poiss_rate", 10.0, 40.0);
  // param 11: base firing rate of excGrp - this should be around 20 Hz
  p->addParam("baseFiring_excGrp", 10.0, 30.0);
  // param 12: buffer group to excitatory weights
  p->addParam("buffGrp_exc_wt", DUMMY_WEIGHT/4, DUMMY_WEIGHT);
  // param 13: excitatory group to inhibitory weights
  p->addParam("excGrp_inh_wt",0.1,1.0);
  // param 14: base firing rate of inhGrp
  p->addParam("baseFiring_inhGrp", 40.0, 100.0);

  // tell EO to add all parameters
  p->updateParameters();

  printf("Parameters added\n");

  int genCounter=0;

  // get number of individuals in population
  int populationSize;
  populationSize=p->getPopulationSize();

  // create string names for all individuals in population
  // DEBUG: need to fix
  // I could add genCounter into this and then do it every generation.
  // It would look something like:
  // sprintf(tmp_fileName,"exc_weights_%d_%d.dat",i,genCounter);
  /*
    string fileName[populationSize];
    for(int i=0;i<populationSize;i++){
    char tmp_fileName[100];
    sprintf(tmp_fileName,"exc_weights_%d.dat",i);
    // copy c-string to string
    fileName[i]=tmp_fileName;
    }
  */
  for(int configId=0;configId<NUM_CONFIG;configId++){
    ALPHA_LTP_INPUT[configId]=0;
    TAU_LTP_INPUT[configId]=0;
    ALPHA_LTD_INPUT[configId]=0;
    TAU_LTD_INPUT[configId]=0;
    TAU_DELTA_INPUT[configId]=0;
    ALPHA_LTP_INH[configId]=0;
    TAU_LTP_INH[configId]=0;
    ALPHA_LTD_INH[configId]=0;
    TAU_LTD_INH[configId]=0;
    TAU_DELTA_INH[configId]=0;
    MAX_POISS_RATE[configId]=0;
    inh_weight_param[configId]=0;
    baseFiring_excGrp_param[configId]=0;
    buff_weight_param[configId]=0;
    exc_weight_param[configId]=0;
    baseFiring_inhGrp_param[configId]=0;
  }

  //separate spike files and weight files into different
  //folders
  char spikeDir[100] = "results/SORFs/spikeFiles";
  char weightsDir[100] = "results/SORFs/weightFiles";
  char cmdName[100];
  sprintf(cmdName, "rm -r %s", spikeDir);
  system(cmdName);
  sprintf(cmdName, "mkdir -p %s", spikeDir);
  int response = system(cmdName);
  assert(response == 0);
  sprintf(cmdName, "rm -r %s", weightsDir);
  system(cmdName);
  sprintf(cmdName, "mkdir -p %s", weightsDir);
  response = system(cmdName);
  assert(response == 0);
  sprintf(cmdName, "rm -r %s", tuningCurvesDir);
  system(cmdName);
  sprintf(cmdName, "mkdir -p %s", tuningCurvesDir);
  response = system(cmdName);
  assert(response == 0);

  //stuff related to outputting the weights
  string initOnFileName[NUM_CONFIG];
  string initOffFileName[NUM_CONFIG];
  string midOnFileName[NUM_CONFIG];
  string midOffFileName[NUM_CONFIG];
  string finOnFileName[NUM_CONFIG];
  string finOffFileName[NUM_CONFIG];
  string initExcInhFileName[NUM_CONFIG];
  string midExcInhFileName[NUM_CONFIG];
  string finExcInhFileName[NUM_CONFIG];

  // END EO initialization
  // ---------------------------------------------------------------------------------------------------

  // BEGIN EO Parameter Tuning
  // ---------------------------------------------------------------------------------------------------

  // TAGS:TODO
  // do a check of number of configurations and the population size and throw an error if things don't
  // work out right.

  printf("genCounter=%d\n",genCounter);

  cout << "p->maxGen=" << p->maxGen << endl;

  globalGenCounter = 0;
  while (genCounter < p->maxGen){// need to make accessors for this
    genCounter++; globalGenCounter++;
    printf("Parameter initialization for generation %d",genCounter);
    // Loop over the individuals in the population
    //for (uint32_t IndiId=0; IndiId < p->getPopulationSize(); IndiId++) {
    uint32_t IndiId=0; globalIndiId=0;
    while(IndiId < p->getPopulationSize() ){
      // makes sure that we don't spend extra time
      // evaluating an individual that already has a fitness
      /*
	if (p->fitnessAlreadyExists(IndiId)){// seems to work
	printf("fitness exists for this individual, moving to next individual\n");
	continue;
	}
      */
      for(int configId=0; configId < NUM_CONFIG; configId++, IndiId++, globalIndiId++){
        bool finished = false;
        bool newNetworks = false;

        //string calculation:
        char tmpFileName[200];
        // initial weights
        sprintf(tmpFileName,"%s/init_on_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        initOnFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/init_off_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        initOffFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/init_exc_inh_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        initExcInhFileName[configId]=tmpFileName;

        // midway through the simulation weights
        sprintf(tmpFileName,"%s/mid_on_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        midOnFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/mid_off_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        midOffFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/mid_exc_inh_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        midExcInhFileName[configId]=tmpFileName;

        // final weight state
        sprintf(tmpFileName,"%s/fin_on_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        finOnFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/fin_off_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        finOffFileName[configId]=tmpFileName;

        sprintf(tmpFileName,"%s/fin_exc_inh_wts_gen_%d_indi_%d.dat",weightsDir,(genCounter-1),IndiId);
        finExcInhFileName[configId]=tmpFileName;
        //int  configIndiId = -1;//replaced by newNetworks.

        //selecting the correct configId/IndiId match.
        /*
	  while (!finished) {
	  if (IndiId >= p->getPopulationSize()) {
	  finished = true;
	  }
	  else if (p->fitnessAlreadyExists(IndiId)) {
	  IndiId++;
	  }
	  else {
	  finished = true;
	  //configIndiId = IndiId;
	  newNetworks=true;
	  }
	  }
	*/
        //condition where we have no configurations to run
        /*
	  if(!newNetworks){
	  //break out of IndiId loop to avoid simulations.
	  //set special flag to not run sim.
	  }
	*/
        // Get value of EO parameters
        // --------------------------------------------------------
        // param 1: inhibitory to excitatory weights (static)
        inh_weight_param[configId] = -1.0*(p->getParam(IndiId,s1="inhGrp_exc_wt"));
        // param 2: tau LTP input
        TAU_LTP_INPUT[configId] = p->getParam(IndiId,s1="tau_ltp_input");
        // param 3: tau DELTA input
        TAU_DELTA_INPUT[configId] = p->getParam(IndiId,s1="tau_delta_input");
        TAU_LTD_INPUT[configId]   = TAU_LTP_INPUT[configId]+TAU_DELTA_INPUT[configId];
        // param 4: alpha LTP input
        ALPHA_LTP_INPUT[configId] = p->getParam(IndiId,s1="alpha_ltp_input");
        // param 5: alpha LTD input
        ALPHA_LTD_INPUT[configId] = p->getParam(IndiId,s1="alpha_ltd_input");		
        // param 6: tau LTP inh
        TAU_LTP_INH[configId] = p->getParam(IndiId,s1="tau_ltp_inh");
        // param 7: tau DELTA input
        TAU_DELTA_INH[configId] = p->getParam(IndiId,s1="tau_delta_inh");
        TAU_LTD_INH[configId]   = TAU_LTP_INH[configId]+TAU_DELTA_INH[configId];
        //--------------------------------------------------------------------
        //  IMPORTANT: This is where you change the polarity of STDP for inh
        //  negative values for both will flip the STDP curves.  The -1 term
        //  is the flip!
        //--------------------------------------------------------------------
        // param 8: alpha LTP input
        ALPHA_LTP_INH[configId] = -1.0*p->getParam(IndiId,s1="alpha_ltp_inh");
        // param 9: alpha LTD input
        ALPHA_LTD_INH[configId] = -1.0*p->getParam(IndiId,s1="alpha_ltd_inh");
        // param 10: poisson rate
        MAX_POISS_RATE[configId]=p->getParam(IndiId,s1="poiss_rate");
        // param 11: base firing rate of excGrp
        baseFiring_excGrp_param[configId]=p->getParam(IndiId,s1="baseFiring_excGrp");
        // param 12: buffer group to excitatory group weights (plastic)
        buff_weight_param[configId] = p->getParam(IndiId,s1="buffGrp_exc_wt");
        // param 13: excitatory group to inhibitory group weights (plastic)
        exc_weight_param[configId] = p->getParam(IndiId,s1="excGrp_inh_wt");
        // param 14: base firing rate of excGrp
        baseFiring_inhGrp_param[configId]=p->getParam(IndiId,s1="baseFiring_inhGrp");

        // Set the values of SNN variables to the EO parameters
        // --------------------------------------------------------
        // param 1: inhibitory to excitatory weights (fixed)
        grpConnectInfo_t* gc = snn.getConnectInfo(inhToExc_cid, configId);
        gc->initWt = inh_weight_param[configId]/4.0f;
        gc->maxWt  = inh_weight_param[configId];
        // params 2, 3, 4, and 5 
        snn.setSTDP(excGrp, true, ALPHA_LTP_INPUT[configId], TAU_LTP_INPUT[configId], ALPHA_LTD_INPUT[configId], TAU_LTD_INPUT[configId], configId);
        // params 6, 7, 8, and 9 
        snn.setSTDP(inhGrp, true, ALPHA_LTP_INH[configId], TAU_LTP_INH[configId], ALPHA_LTD_INH[configId], TAU_LTD_INH[configId], configId);
        // param 10: MAX_POISS_RATE 
        // the spike responses are set proportional to MAX_POISS_RATE
        for(int i=0;i<NUM_PATTERNS;i++){
          // for all pixels in each pattern
          for(int j=0;j<imageX*imageY;j++){
            (spikeResponseOn[i])->rates[j]= MAX_POISS_RATE[configId]*spikeRateOn[i][j];
            (spikeResponseOff[i])->rates[j]= MAX_POISS_RATE[configId]*spikeRateOff[i][j];
          }
        }
        // param 11: base firing rate excGrp (homeostasis)
        snn.setBaseFiring(excGrp, configId, baseFiring_excGrp_param[configId], 0.0);
        // param 12: buffer to excitatory weights (plastic)
        // for on buffer to exc weights
        gc = snn.getConnectInfo(bufferOnToExc_cid, configId);
        gc->initWt = buff_weight_param[configId]/4.0f;
        gc->maxWt  = buff_weight_param[configId];
        // for off buffer to exc weights 
        gc = snn.getConnectInfo(bufferOffToExc_cid, configId);
        gc->initWt = buff_weight_param[configId]/4.0f;
        gc->maxWt  = buff_weight_param[configId];
        // param 13: exc to inh weights
        gc = snn.getConnectInfo(excToInh_cid, configId);
        gc->initWt = exc_weight_param[configId]/4.0f;
        gc->maxWt  = exc_weight_param[configId];
        // param 14: base firing rate for inhGrp (homeostasis)
        snn.setBaseFiring(inhGrp, configId, baseFiring_inhGrp_param[configId], 0.0);
      } // end loop over configurations


      // update the snn network (individual within this loop) 
      // with the new parameter values

      // When this is called before the first run, nothing will happen.
      // When this is called after the first run, we reset the timing tables
      // (first argument) and reset the weights to default values (second
      // argument).
      snn.updateNetwork(true,true);

      // initialize random number generator seed
      srand(time(0));

#ifndef REG_TESTING
      // LOOP_COUNT: loop for presentation of stimulus
      // stimulus presentation.
      // #define LOOP_COUNT 20000 // jay's original
      // original:
#define LOOP_COUNT 2500//100 minutes //250 // let's try ~8 minutes
      //#define LOOP_COUNT    3 // super short for testing
      //original:
#define PRES_RUN_SEC  2 
      //#define LOOP_COUNT    100 //about 100 minutes
      //#define PRES_RUN_SEC  60
#define PRES_RUN_MSEC 0
#define OFF_RUN_SEC   0
#define OFF_RUN_MSEC  500
#endif

      // make sure that we are not copying data needlessly from GPU
      snn.setCopyFiringStateFromGPU(false);

      // for 250 iterations of 2 s = 500 sec ~ 8 mins
      // MAIN TRAINING LOOP
      for(int iter=0;iter<LOOP_COUNT;iter++){
        int randNum;
        randNum=rand()%NUM_PATTERNS;

        //output the weights at a few different times
        if(iter==2){
          //for loop that cycles through the configs and prints them all out.
          for(int k=0;k<NUM_CONFIG;k++){
            snn.writePopWeights(initOnFileName[k],bufferOnGrp,excGrp,k);
            snn.writePopWeights(initOffFileName[k],bufferOffGrp,excGrp,k);
            snn.writePopWeights(initExcInhFileName[k],excGrp,inhGrp,k);
          }
        }

        if(iter==floor(LOOP_COUNT/2)){
          //for loop that cycles through the configs and prints them all out.
          for(int k=0;k<NUM_CONFIG;k++){
            snn.writePopWeights(midOnFileName[k],bufferOnGrp,excGrp,k);
            snn.writePopWeights(midOffFileName[k],bufferOffGrp,excGrp,k);
            snn.writePopWeights(midExcInhFileName[k],excGrp,inhGrp,k);
          }
        }

        if(iter==floor(LOOP_COUNT-1)){
          //for loop that cycles through the configs and prints them all out.
          for(int k=0;k<NUM_CONFIG;k++){
            snn.writePopWeights(finOnFileName[k],bufferOnGrp,excGrp,k);
            snn.writePopWeights(finOffFileName[k],bufferOffGrp,excGrp,k);
            snn.writePopWeights(finExcInhFileName[k],excGrp,inhGrp,k);
          }
        }
        // choose random pattern
        snn.setSpikeRate(inputOnGrp,(spikeResponseOn[randNum]));
        snn.setSpikeRate(inputOffGrp,(spikeResponseOff[randNum]));

        // run network with chosen pattern
        snn.runNetwork(PRES_RUN_SEC, PRES_RUN_MSEC, SIM_MODE);

        // now do the off interval presentation
        snn.setSpikeRate(inputOnGrp,noiseGen);
        snn.setSpikeRate(inputOffGrp,noiseGen);

        // run network with 'rest' interval
        snn.runNetwork(OFF_RUN_SEC, OFF_RUN_MSEC, SIM_MODE);
      }
      // we can print out the weights here once, before they get evaluated
      // so we can make sure it is working and the weights have not changed
      // (STDP is really off).
      // print:

      // test the param1Value with our fitness function and assign the fitness value:
      evaluateFitness(snn, fitness);

      p->setFitness(fitness, IndiId, NUM_CONFIG);

    } // end loop over individuals

    // run the evolutionary algorithm as each individual in the population
    // has been asigned a fitness.
    bool continueIter = p->runEA();
    // Stopping condition reached...
    if(!continueIter){
      fprintf(stderr," Stop condition reached...\n");
      break;
    }
  } // end while loop over generations


  p->printSortedPopulation();

  printf("genCounter=%d\n", genCounter);


  // release the memory from heap
  for(int k=0;k<NUM_PATTERNS;k++){
    // delete[] gaborPattern[k];
    delete[] CPSGratingPattern[k];
    delete[] spikeRateOn[k];
    delete[] spikeRateOff[k];
  }

  delete[] noiseGen;

  delete p;

  fclose(fpResp);

  return 0;
}

// A main that catches the exceptions

int main(int argc, char **argv)
{

  try
    {
      main_fitness_function(argc, argv);
    }
  catch (exception& e)
    {
      cout << "Exception: " << e.what() << '\n';
    }

  return 1;
}
