//
// Copyright (c) 2009 Regents of the University of California. All rights reserved.
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

#include "snn.h"
#include "PropagatedSpikeBuffer.h"
// switch homeostasis on or off.
#define WITH_HOMEO 0
//#define SIM_MODE   CPU_MODE
#define SIM_MODE   CPU_MODE

int main()
{
  // create a network with N neurons, maximum number of synapes is 100, maximum delay is 20ms, name is global
  CpuSNN snn("global");
  int nPois = 100;
  //PoissonRate rampFiringRates(nPois);
  PoissonRate* rampFiringRates;
  rampFiringRates = new PoissonRate(nPois);
  int gpois = snn.createSpikeGeneratorGroup("ramp_input", nPois, EXCITATORY_POISSON);
  int gexc  = snn.createGroup("excit", 1, EXCITATORY_NEURON);
  snn.setNeuronParameters(gexc, 0.02f, 0.2f, -65.0f, 8.0f);
  snn.connect(gpois, gexc,  "full", 0.01f, 0.03f, 1.0, 1, 1, SYN_PLASTIC);

#define COND_tAMPA 5.0
#define COND_tNMDA 150.0
#define COND_tGABAa 6.0
#define COND_tGABAb 150.0
  snn.setConductances(ALL,true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);
  
  // define and set the properties of STDP
  float ALPHA_LTP = (0.0010f/5.0);
  float TAU_LTP   = 20.0;//17.0;//20.0;
  float ALPHA_LTD = 0.00033f/5.0;//ALPHA_LTP*0.40;//0.00033f/5.0;
  float TAU_LTD   = 60.0;//34.0;//60.0;
  snn.setSTDP(ALL,true,ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

  // snn.sim_with_stp = false;
  // show log every 1 sec (0 to disable logging). You can pass a file pointer or 
  // pass stdout to specify where the log output should go.
  snn.setLogCycle(1, 1, stderr);
	
#define HOMEO_FACTOR 		  ( 1.0)
#define HOMEO_AVERAGE_TIME_SCALE  ( 5.0)
#define BASE_FIRING_RATE          (35.0)      

#if WITH_HOMEO
  // need to figure out what this extra parameter was
  //snn.setHomeostasis(gexc,true,15.0,1.0,5);
  // i think 15.0 is base firing, homeofactor is 1.0 and avgTimeScale is 5
  snn.setHomeostasis(gexc,true,HOMEO_FACTOR,HOMEO_AVERAGE_TIME_SCALE,ALL);
  snn.setBaseFiring(gexc,0,BASE_FIRING_RATE,0);
#else
  // need to figure out what this extra parameter was
  //snn.setHomeostasis(gexc,false,15.0,1.0,5);
  snn.setHomeostasis(gexc,false,ALL);
#endif

  for (int i=0;i<nPois;i++){
    //rampFiringRates->rates[i] = (i)/10.0+0.5;
    //rampFiringRates->rates[i] = (i+1);
    rampFiringRates->rates[i] = (i+1)*(20.0/100.0);
    // debugging sanity check
    cout << "ramp firing rate " << i << " = " << rampFiringRates->rates[i] << endl; 
  }

  snn.setSpikeRate(gpois, rampFiringRates);

#if WITH_HOMEO
  snn.setSpikeMonitor(gexc, "ramp_spikes_homeo.dat");
  snn.setSpikeMonitor(gpois, "poisson_spikes.dat");
  //FILE* nid = fopen("ramp_snn_homeo.dat","wb");
#else
  snn.setSpikeMonitor(gexc, "ramp_spikes_no_homeo.dat");
  snn.setSpikeMonitor(gpois, "poisson_spikes.dat");
  //FILE* nid = fopen("ramp_snn.dat","wb");
#endif

  // create the correct weights folder and delete previous ones
  char homeoWeightsDir[100] = "homeoRampWeights";
  char noHomeoWeightsDir[100] = "noHomeoRampWeights";
  char cmdName[100];
  int response = system(cmdName);
  string fileName;
  // clear out files
#if WITH_HOMEO
  sprintf(cmdName, "rm -r %s", homeoWeightsDir);
  system(cmdName);
  sprintf(cmdName, "mkdir -p %s", homeoWeightsDir);
  response = system(cmdName);
  assert(response == 0);
#else
  sprintf(cmdName, "rm -r %s", noHomeoWeightsDir);
  system(cmdName);
  sprintf(cmdName, "mkdir -p %s", noHomeoWeightsDir);
  response = system(cmdName);
  assert(response == 0);
#endif
  


	
#define LOOP_ITERATIONS 1000
  for(uint32_t i=0; i < LOOP_ITERATIONS; i++) {
    // run the established network for 1 sec
    snn.runNetwork(1, 0, SIM_MODE);
    //snn.writeNetwork(nid);
#if WITH_HOMEO
    
    //snn.writePopWeights("ramp_weights_homeo.dat",gpois,gexc);
    // generate a bunch of weight frames
    // we print out an updated weight table every 40 steps
    int frameNumber;
    //if(i%10==1){
    //frameNumber = i/10;
    if(i<1000){
      frameNumber=i;
      char tmpFileName1[100];

      //output bufferOnGrp to excGrp weights
      sprintf(tmpFileName1,"%s/homeo_wt_%d.dat",homeoWeightsDir,frameNumber);
      fileName=tmpFileName1;
      snn.writePopWeights(fileName,gpois,gexc);
    }
#else
    //snn.writePopWeights("ramp_weights_no_homeo.dat",gpois,gexc);

    int frameNumber;
    //if(i%10==1){
    //frameNumber = i/10;
    if(i<1000){
	frameNumber=i;
	char tmpFileName1[100];
	
       //output bufferOnGrp to excGrp weights
       sprintf(tmpFileName1,"%s/no_homeo_wt_%d.dat",noHomeoWeightsDir,frameNumber);
       fileName=tmpFileName1;
       snn.writePopWeights(fileName,gpois,gexc);
     }
#endif
  }		
  //fclose(nid);

  // display the details of the current simulation run
  //snn.printSimSummary();

  return 0;
}
