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
#include <string>
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

        int N_EXC1, N_EXC2, N_INPUT;
	float pConn1;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight;
	FILE* recordFile;
	int weightTuning;	
	int id4core1, id4core2;	
	string connectionType;

	if (argc!=16) return 1; // 15 parameters are required


        // read parameters and create a network on GPU
	N_EXC1 = atoi(argv[1]);
	N_EXC2 = atoi(argv[2]);
	N_INPUT = atoi(argv[3]);
	simulateTime = atoi(argv[4]);
	inputFireRate = atof(argv[5]);
	inputWeight = atof(argv[6]);
	excWeight = atof(argv[7]);
	
	recordFile = fopen(argv[8],"a");
	// create a network on GPU
        Stopwatch watch;
	randSeed = atoi(argv[9]);

        pConn1 = atof(argv[10])/N_EXC1; // connection probability

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network

	string core1, core2;
	core1 = argv[11];
	core2 = argv[12];

	int gExc1;
	int gExc2;
	int gInput;

	if(core1 == "GPU"){
		gExc1 = sim.createGroup("exc", N_EXC1, EXCITATORY_NEURON, 0, GPU_CORES);
		sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);
	}
	else if(core1 == "CPU"){
		gExc1 = sim.createGroup("exc", N_EXC1, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, CPU_CORES);
	}
	else{
		cout<<"wrong parameter"<<endl;
		return 1;
	}


	if(core2 == "GPU"){
		gExc2 = sim.createGroup("exc", N_EXC2, EXCITATORY_NEURON, atoi(argv[13]), GPU_CORES);
		sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS	
	}
	else if(core2 == "CPU"){
		gExc2 = sim.createGroup("exc", N_EXC2, EXCITATORY_NEURON, atoi(argv[13]), CPU_CORES);
		sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS	
	}
	else{
		cout<<"wrong parameter"<<endl;
		return 1;
	}


	connectionType = argv[15];
	int c0, c1;

	if(connectionType == "Fixed"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		c1 = sim.connect(gExc1, gExc2, "full", RangeWeight(excWeight), 1.0, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);

	}
	else if(connectionType == "STP"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "full", RangeWeight(excWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		sim.setSTP(gExc1, true);
	}
	else if(connectionType == "STDP"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "full", RangeWeight(excWeight), 1.0, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
		sim.setESTDP(gExc1, true, STANDARD, ExpCurve(2e-4f, 20.0f, -6.6e-5f, 60.0f));
	}
	else if(connectionType == "Homeostasis"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "full", RangeWeight(excWeight), 1.0, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
		sim.setESTDP(gExc1, true, STANDARD, ExpCurve(2e-4f, 20.0f, -6.6e-5f, 60.0f));
		sim.setHomeostasis(gExc1, true, 1.0f, 10.0f);
		sim.setHomeoBaseFiringRate(gExc1, 35.0f, 0.0f);	
	}
	else{
		return 1;
	}

	sim.setConductances(false);


	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();

	//setup monitors
//	SpikeMonitor* spkMon1 = sim.setSpikeMonitor(gExc1, "NULL");
//	SpikeMonitor* spkMon2 = sim.setSpikeMonitor(gExc2, "NULL");

	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
        sim.setSpikeRate(gInput, &in);   


	// weight tuning
	FILE* fireTargetFile = fopen("fireTarget","r");
	fseek(fireTargetFile, 76, SEEK_SET);
	watch.lap("Weight Tuning");
	if(atoi(argv[14])==1){
		
		//fetch tuning parameters
		double targetFiringExc1, targetFiringExc2;
		double errorMarginHz;
		int maxIter; 
		double stepSize;

		fscanf(fireTargetFile, "%lf",&targetFiringExc1);
		fscanf(fireTargetFile, "%lf",&targetFiringExc2);
 		fscanf(fireTargetFile, "%lf",&errorMarginHz);
 		fscanf(fireTargetFile, "%d",&maxIter);
 		fscanf(fireTargetFile, "%lf",&stepSize);
	
		//tuning weights
		
		SimpleWeightTuner SWTin2exc1(simulator, errorMarginHz, maxIter, stepSize);
		SWTin2exc1.setConnectionToTune(c0, 0.0);
		SWTin2exc1.setTargetFiringRate(gExc1, targetFiringExc1);	
		
		SimpleWeightTuner SWTexc12exc2(simulator, errorMarginHz, maxIter, stepSize);
		SWTexc12exc2.setConnectionToTune(c1, 0.0);
		SWTexc12exc2.setTargetFiringRate(gExc2, targetFiringExc2);	
	

		while(!SWTin2exc1.done()){
			SWTin2exc1.iterate();
		}

		while(!SWTexc12exc2.done()){
			SWTexc12exc2.iterate();
		}

		
		
		printf("Verify result (gExc1=%.4fHz, gExc2=%.4fHz,  +/- %.4fHz)\n", targetFiringExc1, targetFiringExc2, errorMarginHz); 
		
		//spkMon1->startRecording();
		//spkMon2->startRecording();
	
		//sim.runNetwork(1,0);
		
		//spkMon1->stopRecording();
		//spkMon2->stopRecording();
		
		//spkMon1->print(false);
		//spkMon2->print(false);
		
		
	}
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed

	watch.lap("runNetwork");
	
	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(1, 0, true);
	}

 	watch.stop();

	fprintf(recordFile, "%ld,%ld,%ld\n", watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(3));     	
	fclose(recordFile);
	

	return 0;	
}
