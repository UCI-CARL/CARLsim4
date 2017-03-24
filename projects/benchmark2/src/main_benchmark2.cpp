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

        int N_EXC1, N_EXC2, N_EXC3, N_EXC4, N_INPUT;
	float pConn1, pConn2, pConn3, pConn4;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight;
	FILE* recordFile;
	int weightTuning;	
	

	if (argc!=14) return 1; // 13 parameters are required


        // read parameters and create a network on GPU
	N_EXC1 = atoi(argv[1]);
	N_EXC2 = atoi(argv[2]);
	N_EXC3 = atoi(argv[3]);
	N_EXC4 = atoi(argv[4]);
	N_INPUT = atoi(argv[6]);
	simulateTime = atoi(argv[9]);
	inputFireRate = atof(argv[10]);
	inputWeight = atof(argv[11]);
	excWeight = atof(argv[12]);
	
	recordFile = fopen(argv[8],"a");
	// create a network on GPU
        Stopwatch watch;
	int numGPUs = 2;
	randSeed = atoi(argv[7]);

        pConn1 = atof(argv[5])/N_EXC1; // connection probability
        pConn2 = atof(argv[5])/N_EXC2; // connection probability
        pConn3 = atof(argv[5])/N_EXC3; // connection probability
        pConn4 = atof(argv[5])/N_EXC4; // connection probability

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network
	int gExc1 = sim.createGroup("exc", N_EXC1, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gExc2 = sim.createGroup("exc", N_EXC2, EXCITATORY_NEURON, 1, GPU_CORES);		
 	sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS             		

	int gExc3 = sim.createGroup("exc", N_EXC3, EXCITATORY_NEURON, 1, GPU_CORES);
	sim.setNeuronParameters(gExc3, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gExc4 = sim.createGroup("exc", N_EXC4, EXCITATORY_NEURON ,0, GPU_CORES);
	sim.setNeuronParameters(gExc4, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);




	int c0 = sim.connect(gInput, gExc1, "random", RangeWeight(inputWeight), pConn1, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	int c1 = sim.connect(gExc1, gExc2, "random", RangeWeight(excWeight), pConn2, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	int c2 = sim.connect(gExc2, gExc3, "random", RangeWeight(excWeight), pConn3, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	int c3 = sim.connect(gExc3, gExc4, "random", RangeWeight(excWeight), pConn4, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);

	sim.setConductances(false);


	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();

	//setup monitors
//	SpikeMonitor* spkMon1 = sim.setSpikeMonitor(gExc1, "NULL");
//	SpikeMonitor* spkMon2 = sim.setSpikeMonitor(gExc2, "NULL");
//	SpikeMonitor* spkMon3 = sim.setSpikeMonitor(gExc3, "NULL");
//	SpikeMonitor* spkMon4 = sim.setSpikeMonitor(gExc4, "NULL");

	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
        sim.setSpikeRate(gInput, &in);   


	// weight tuning
	watch.lap("Weight Tuning");
	if(atoi(argv[13])==1){
		
		//fetch tuning parameters
		double targetFiringExc1, targetFiringExc2, targetFiringExc3, targetFiringExc4;
		double errorMarginHz;
		int maxIter; 
		double stepSize;
		cout<<endl<<"please input traget firing rate for group1, group2, group3, group4, errorMarginHz, maximum iteration, stepSize"<<endl;
		cin>>targetFiringExc1>>targetFiringExc2>>targetFiringExc3>>targetFiringExc4>>errorMarginHz>>maxIter>>stepSize;
		cout<<endl;
		
		//tuning weights
		
		SimpleWeightTuner SWTin2exc1(simulator, errorMarginHz, maxIter, stepSize);
		SWTin2exc1.setConnectionToTune(c0, 0.0);
		SWTin2exc1.setTargetFiringRate(gExc1, targetFiringExc1);	
		
		SimpleWeightTuner SWTexc12exc2(simulator, errorMarginHz, maxIter, stepSize);
		SWTexc12exc2.setConnectionToTune(c1, 0.0);
		SWTexc12exc2.setTargetFiringRate(gExc2, targetFiringExc2);	
	
		SimpleWeightTuner SWTexc22exc3(simulator, errorMarginHz, maxIter, stepSize);
		SWTexc22exc3.setConnectionToTune(c2, 0.0);
		SWTexc22exc3.setTargetFiringRate(gExc3, targetFiringExc3);	
	
		SimpleWeightTuner SWTexc32exc4(simulator, errorMarginHz, maxIter, stepSize);
		SWTexc32exc4.setConnectionToTune(c3, 0.0);
		SWTexc32exc4.setTargetFiringRate(gExc4, targetFiringExc4);	

		while(!SWTin2exc1.done()){
			SWTin2exc1.iterate();
		}

		while(!SWTexc12exc2.done()){
			SWTexc12exc2.iterate();
		}

		while(!SWTexc22exc3.done()){
			SWTexc22exc3.iterate();
		}

		while(!SWTexc32exc4.done()){
			SWTexc32exc4.iterate();
		}
		
		
		printf("Verify result (gExc1=%.4fHz, gExc2=%.4fHz, gExc3=%.4fHz, gExc4=%.4fHz, +/- %.4fHz)\n", targetFiringExc1, targetFiringExc2, targetFiringExc3, targetFiringExc4, errorMarginHz); 
		
		//spkMon1->startRecording();
		//spkMon2->startRecording();
		//spkMon3->startRecording();
		//spkMon4->startRecording();
	
		//sim.runNetwork(1,0);
		
		//spkMon1->stopRecording();
		//spkMon2->stopRecording();
		//spkMon3->stopRecording();
		//spkMon4->stopRecording();
		
		//spkMon1->print(false);
		//spkMon2->print(false);
		//spkMon3->print(false);
		//spkMon4->print(false);
		
		
	}
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed

	watch.lap("runNetwork");
	
	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(1, 0, true);
	}

 	watch.stop();

//	fprintf(recordFile, "%ld,%ld,%ld\n", watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(3));     	
//	fclose(recordFile);
	

	return 0;	
}
