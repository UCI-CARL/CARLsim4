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
	float inputWeight, excWeight;
	FILE* recordFile;	
	int N_CORES;


	if (argc!=12) return 1; // 11 parameters are required


        // read parameters and create a network on GPU
	N_NEURONS = atoi(argv[1]);
	N_INPUT = int(N_NEURONS*atof(argv[2]));
	simulateTime = atoi(argv[6]);
	inputFireRate = atof(argv[7]);
	inputWeight = atof(argv[8]);
	excWeight = atof(argv[9]);
	N_CORES = atoi(argv[11]);
	recordFile = fopen(argv[5],"a");
	// create a network on GPU
        Stopwatch watch;
	
	randSeed = atoi(argv[3]);
        pConn = atof(argv[4])/N_NEURONS/8; // connection probability
	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);

	
	// configure the network
	string core;
	core = argv[10];
	int groupId[8];
	int gInput;
	
	cout<<"start to create groups"<<endl;
//	if(core == "GPU"){
		groupId[0] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0, GPU_CORES);
		sim.setNeuronParameters(groupId[0], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[1] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0), GPU_CORES);
		sim.setNeuronParameters(groupId[1], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
		groupId[2] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*2), GPU_CORES);
		sim.setNeuronParameters(groupId[2], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[3] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*3), GPU_CORES);
		sim.setNeuronParameters(groupId[3], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[4] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*4), GPU_CORES);
		sim.setNeuronParameters(groupId[4], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[5] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*5), GPU_CORES);
		sim.setNeuronParameters(groupId[5], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[6] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*6), GPU_CORES);
		sim.setNeuronParameters(groupId[6], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[7] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*7), GPU_CORES);
		sim.setNeuronParameters(groupId[7], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);	

/*	}
	else if(core == "CPU"){
		groupId[0] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParameters(groupId[0], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[1] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0), CPU_CORES);
		sim.setNeuronParameters(groupId[1], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
		groupId[2] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*2), CPU_CORES);
		sim.setNeuronParameters(groupId[2], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[3] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*3), CPU_CORES);
		sim.setNeuronParameters(groupId[3], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[4] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*4), CPU_CORES);
		sim.setNeuronParameters(groupId[4], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[5] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*5), CPU_CORES);
		sim.setNeuronParameters(groupId[5], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[6] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*6), CPU_CORES);
		sim.setNeuronParameters(groupId[6], 0.02f, 0.2f, -65.0f, 8.0f); // RS

		groupId[7] = sim.createGroup("exc", N_NEURONS, EXCITATORY_NEURON, 0+int(N_CORES/8.0*7), CPU_CORES);
		sim.setNeuronParameters(groupId[7], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, CPU_CORES);	
	
	}	
	else{
		return 1;
	}
*/	//FixedSpikeGenerator* f1 = new FixedSpikeGenerator();
	//sim.setSpikeGenerator(gInput, f1);
	cout<<"start to connect"<<endl;
	sim.connect(gInput, groupId[0], "full", RangeWeight(inputWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	for(int i=0; i<8; i++){
		for(int j=0; j<8; j++){
			cout<<i<<"+"<<j<<endl;
			sim.connect(groupId[i], groupId[j], "random", RangeWeight(excWeight), 1.0f, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
		}
	}		


	sim.setConductances(false);

	cout<<"start to set up network"<<endl;

	//sim.setESTDP(gExc, true, STANDARD, ExpCurve(0.1f/100, 20, -0.12f/100, 20));

	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();


	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
        sim.setSpikeRate(gInput, &in);   


	// set some monitors
//	SpikeMonitor* smExc = sim.setSpikeMonitor(gExc, "NULL");
//	SpikeMonitor* smInh = sim.setSpikeMonitor(gInh, "NULL");
//	SpikeMonitor* smInput = sim.setSpikeMonitor(gInput, "NULL");

//	SpikeMonitor* smExc2 = sim.setSpikeMonitor(gExc2, "NULL");
//	SpikeMonitor* smInh2 = sim.setSpikeMonitor(gInh2, "NULL");
//	SpikeMonitor* smInput2 = sim.setSpikeMonitor(gInput2, "NULL");
	//ConnectionMonitor* cmEE = sim.setConnectionMonitor(gExc, gInh, "DEFAULT");

	
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed

	//smInput->startRecording();
	//smExc->startRecording();
	//smInh->startRecording();
	watch.lap("runNetwork");
	cout<<"start to run network"<<endl;
	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(1, 0, true);
	}

 	watch.stop();


	fprintf(recordFile, "%ld,%ld,%ld\n", watch.getLapTime(0), watch.getLapTime(1),
 watch.getLapTime(2));     	
	fclose(recordFile);
}
