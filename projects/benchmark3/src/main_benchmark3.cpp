/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 i*
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. The names of its contributors may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * *********************************************************************************************** *
 * CARLsim
 * created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:
 * (MA) Mike Avery <averym@uci.edu>
 * (MB) Michael Beyeler <mbeyeler@uci.edu>,
 * (KDC) Kristofor Carlson <kdcarlso@uci.edu>
 * (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 5/22/2015
 */

// include CARLsim user interface
#include <carlsim.h>
#include <stopwatch.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h> 
#include <iostream>
#include <simple_weight_tuner.h>
using namespace std;

//class FixedSpikeGenerator : public SpikeGenerator {
//public:
//	FixedSpikeGenerator() {}
//
//	int nextSpikeTime(CARLsim* sim, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) {
//		if (lastScheduledSpikeTime <= currentTime)
//			return currentTime + nid + 100;
//		else
//			return endOfTimeSlice + 1;
//	}
//};

int main(int argc, char* argv[] ) {

        int N_NEURONS, N_INPUT;
	float pConn;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, conWeight[3];
	FILE* recordFile;	
	int N_CORES;
	int weightTuning;

	if (argc!=14) return 1; // 13 parameters are required


        // read parameters and create a network on GPU
	N_NEURONS = atoi(argv[1]);
	N_INPUT = atoi(argv[2]);
	simulateTime = atoi(argv[6]);
	inputFireRate = atof(argv[7]);
	inputWeight = atof(argv[8]);
	conWeight[0] = atof(argv[9]);
	conWeight[1] = atof(argv[10]);
	conWeight[2] = atof(argv[11]);
	N_CORES = atoi(argv[12]);
	weightTuning = atoi(argv[13]);
	recordFile = fopen(argv[5],"a");
	// create a network on GPU
        Stopwatch watch;
	
	randSeed = atoi(argv[3]);
        pConn = atof(argv[4])/N_NEURONS; // connection probability
	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	
	// configure the network
	string core;
	int groupId[4];
	int gInput;
	
	groupId[0] = sim.createGroup("exc1", N_NEURONS, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.setNeuronParameters(groupId[0], 0.02f, 0.2f, -65.0f, 8.0f); // RS

	groupId[1] = sim.createGroup("exc2", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/4.0), GPU_CORES);
	sim.setNeuronParameters(groupId[1], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
	groupId[2] = sim.createGroup("exc3", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/4.0*2), GPU_CORES);
	sim.setNeuronParameters(groupId[2], 0.02f, 0.2f, -65.0f, 8.0f); // RS

	groupId[3] = sim.createGroup("exc4", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/4.0*3), GPU_CORES);
	sim.setNeuronParameters(groupId[3], 0.02f, 0.2f, -65.0f, 8.0f); // RS

	gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);	

	//FixedSpikeGenerator* f1 = new FixedSpikeGenerator();
	//sim.setSpikeGenerator(gInput, f1);
	int c[4];
	cout<<"start to connect"<<endl;
	c[0] = sim.connect(gInput, groupId[0], "full", RangeWeight(inputWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	for(int i=0; i<3; i++){
		c[i+1] = sim.connect(groupId[i], groupId[i+1], "random", RangeWeight(conWeight[i]), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	}		


	sim.setConductances(false);

	cout<<"start to set up network"<<endl;

	//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();

//	SpikeMonitor* spkMon1 = sim.setSpikeMonitor(groupId[0], "NULL");
//	SpikeMonitor* spkMon2 = sim.setSpikeMonitor(groupId[1], "NULL");
//	SpikeMonitor* spkMon3 = sim.setSpikeMonitor(groupId[2], "NULL");
//	SpikeMonitor* spkMon4 = sim.setSpikeMonitor(groupId[3], "NULL");
//	SpikeMonitor* spkMon0 = sim.setSpikeMonitor(gInput, "NULL");

//	ConnectionMonitor * CM0 = sim.setConnectionMonitor(gInput, groupId[0], "DEFAULT");
//	ConnectionMonitor * CM1 = sim.setConnectionMonitor(groupId[0], groupId[1], "DEFAULT");
//	ConnectionMonitor * CM2 = sim.setConnectionMonitor(groupId[1], groupId[2], "DEFAULT");
//	ConnectionMonitor * CM3 = sim.setConnectionMonitor(groupId[2], groupId[3], "DEFAULT");
	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
        sim.setSpikeRate(gInput, &in);   

	//weight tuning
	FILE* fireTargetFile = fopen("fireTarget","r");
	fseek(fireTargetFile,76,SEEK_SET);
	watch.lap("Weight Tuning");
	if(weightTuning==1){

		double targetFiringRate;
		double errorMarginHz;
		int maxIter;
		double stepSize;

		fscanf(fireTargetFile, "%lf", &targetFiringRate);
		fscanf(fireTargetFile, "%lf", &errorMarginHz);
		fscanf(fireTargetFile, "%d", &maxIter);
		fscanf(fireTargetFile, "%lf", &stepSize);
		
		SimpleWeightTuner SWTin2exc0(simulator, errorMarginHz, maxIter, stepSize);            
                SWTin2exc0.setConnectionToTune(c[0], 0.0);                                              
                SWTin2exc0.setTargetFiringRate(groupId[0], targetFiringRate);

		SimpleWeightTuner SWTin2exc1(simulator, errorMarginHz, maxIter, stepSize);            
                SWTin2exc1.setConnectionToTune(c[1], 0.0);                                              
                SWTin2exc1.setTargetFiringRate(groupId[1], targetFiringRate);

		SimpleWeightTuner SWTin2exc2(simulator, errorMarginHz, maxIter, stepSize);            
                SWTin2exc2.setConnectionToTune(c[2], 0.0);                                              
                SWTin2exc2.setTargetFiringRate(groupId[2], targetFiringRate);

		SimpleWeightTuner SWTin2exc3(simulator, errorMarginHz, maxIter, stepSize);            
                SWTin2exc3.setConnectionToTune(c[3], 0.0);                                              
                SWTin2exc3.setTargetFiringRate(groupId[3], targetFiringRate);

		while(!SWTin2exc0.done()){
                        SWTin2exc0.iterate();
                }		
		int a;
		cout<<"finish first tuning"<<endl;;
	//	cin>>a;				
		while(!SWTin2exc1.done()){
                        SWTin2exc1.iterate();
                }	
		cout<<"finish second tuning"<<endl;;
	//	cin>>a;
		while(!SWTin2exc2.done()){
                        SWTin2exc2.iterate();
                }
		cout<<"3"<<endl;;
	//	cin>>a;
		while(!SWTin2exc3.done()){
                        SWTin2exc3.iterate();
                }
		cout<<"4"<<endl;
	//	cin>>a;
	}


	// set some monitors
	//ConnectionMonitor* cmEE = sim.setConnectionMonitor(gExc, gInh, "DEFAULT");

	
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed

//	spkMon0->startRecording();
//	spkMon1->startRecording();
//	spkMon2->startRecording();
//	spkMon3->startRecording();
//	spkMon4->startRecording();


	watch.lap("runNetwork");
//	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(simulateTime, 0, true);
//	}

 	watch.stop();

//	spkMon0->stopRecording();
//	spkMon1->stopRecording();
//	spkMon2->stopRecording();
//	spkMon3->stopRecording();
//	spkMon4->stopRecording();


/*	spkMon0->print();
	spkMon1->print();
	spkMon2->print();
	spkMon3->print();
	spkMon4->print();
	spkMon5->print();
	spkMon6->print();
	spkMon7->print();
	spkMon8->print();
*/
	fprintf(recordFile, "%d,%lf,%ld,%ld,%ld\n",N_NEURONS, N_NEURONS*pConn,  watch.getLapTime(0), watch.getLapTime(1),
 watch.getLapTime(3));     	
	fclose(recordFile);
}
