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
#define N_Network 32
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

        int N_INPUT, N_NEURON;
	float pConn1;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight;
	FILE* recordFile;
	int N_CORE;
	string connectionType;

	if (argc!=11) return 1; // 10 parameters are required


        // read parameters and create a network on GPU
	N_NEURON = atoi(argv[1]);
	N_INPUT = atoi(argv[2]);
	simulateTime = atoi(argv[5]);
	inputFireRate = atof(argv[6]);
	inputWeight = atof(argv[7]);
	excWeight = atof(argv[8]);
	
	recordFile = fopen(argv[4],"a");
	// create a network on GPU
        Stopwatch watch;
	randSeed = atoi(argv[9]);
	N_CORE = atoi(argv[10]);

        pConn1 = atof(argv[3])/N_NEURON; // connection probability

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network

	int gExc1[N_Network];
	int gExc2[N_Network];
	int gInput[N_Network];
	int c0[N_Network];
	int c1[N_Network];
	for(int i=0; i<N_Network; i++){
		gExc1[i] = sim.createGroup("exc1", N_NEURON, EXCITATORY_NEURON, 0+int(i*N_CORE/N_Network), CPU_CORES);
		sim.setNeuronParameters(gExc1[i], 0.02f, 0.2f, -65.0f, 8.0f); // RS
	
		gInput[i] = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0+int(i*N_CORE/N_Network), CPU_CORES);
		gExc2[i] = sim.createGroup("exc2", N_NEURON, EXCITATORY_NEURON, 0+int(i*N_CORE/N_Network), CPU_CORES);
		sim.setNeuronParameters(gExc2[i], 0.02f, 0.2f, -65.0f, 8.0f); // RS	
	
	}


	for(int i=0; i<N_Network; i++){
		c0[i] = sim.connect(gInput[i], gExc1[i], "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		c1[i] = sim.connect(gExc1[i], gExc2[i], "random", RangeWeight(excWeight), pConn1, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	
	}

	sim.setConductances(false);


	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();


	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
       
	for(int i=0; i<N_Network; i++){
		 sim.setSpikeRate(gInput[i], &in);   

	}
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	
	watch.lap("runNetwork");

	sim.runNetwork(simulateTime, 0, true);

 	watch.stop();

	fprintf(recordFile, "%d,%f,%ld,%ld,%ld\n",N_NEURON, pConn1*N_NEURON, watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(2));     	
	fclose(recordFile);
	

	return 0;	
}
