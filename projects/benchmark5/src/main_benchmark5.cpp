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

        int N_EXC, N_INH, N_INPUT;
	float pConn1, pConn2;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight, inhWeight;
	FILE* recordFile;
	int weightTuning;	
	int id4core1, id4core2;	
	string connectionType;

	if (argc!=12) return 1; // 11 parameters are required


        // read parameters and create a network on GPU
	N_EXC = atoi(argv[1]);
	N_INH = atoi(argv[2]);
	N_INPUT = atoi(argv[3]);
	simulateTime = atoi(argv[4]);
	inputFireRate = atof(argv[5]);
	inputWeight = atof(argv[6]);
	excWeight = atof(argv[7]);
	inhWeight = atof(argv[8]);
	recordFile = fopen(argv[9],"a");
	// create a network on GPU
        Stopwatch watch;
	randSeed = atoi(argv[10]);

        pConn1 = atof(argv[11])/N_EXC; // connection probability
	pConn2 = atof(argv[11])/N_INH;

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network


	int gExc[4];
	int gInh[4];
	int gInput[4];
	string Exc[4] ={"exc0", "exc1", "exc2", "exc3"};
	string Inh[4] ={"inh0", "inh1", "inh2", "inh3"};
	string Input[4]={"input0", "input1", "input2", "input3"};

	for (int i=0; i<4; i++)
	{
		gExc[i] = sim.createGroup(Exc[i], N_EXC, EXCITATORY_NEURON, i, CPU_CORES);
		sim.setNeuronParameters(gExc[i], 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInh[i] = sim.createGroup(Inh[i], N_INH, INHIBITORY_NEURON, i, CPU_CORES);
		sim.setNeuronParameters(gInh[i], 0.1f, 0.2f, -65.0f, 2.0f); // RS	
		gInput[i] = sim.createSpikeGeneratorGroup(Input[i], N_EXC, EXCITATORY_NEURON, i, CPU_CORES);
	}
	
	int cinput2e[4], ce2inh[4], cinh2e[4];
	for (int i=0; i<4; i++)
	{	
		cinput2e[i] = sim.connect(gInput[i], gExc[i], "one-to-one", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		ce2inh[i] = sim.connect(gExc[i], gInh[(i+1)%4], "random", RangeWeight(excWeight), pConn1, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
		cinh2e[i] = sim.connect(gInh[i], gExc[(i+3)%4], "random", RangeWeight(inhWeight), pConn2, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED); 
	}


	sim.setConductances(false);
	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();

	//setup monitors
	SpikeMonitor* spkMon1 = sim.setSpikeMonitor(gExc[0], "NULL");
	SpikeMonitor* spkMon2 = sim.setSpikeMonitor(gInh[0], "NULL");

	//setup some baseline input                           
        PoissonRate in(N_EXC);
        in.setRates(inputFireRate);                        
	for (int i=0; i<4; i++){            
       		sim.setSpikeRate(gInput[i], &in);   
	}

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
//	spkMon1->startRecording();
//	spkMon2->startRecording();

	watch.lap("runNetwork");


//	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(simulateTime, 0, true);
//	}

 	watch.stop();

//	spkMon1->stopRecording();
//	spkMon2->stopRecording();


	fprintf(recordFile, "%d,%f,%ld,%ld,%ld\n",N_INH, pConn1*N_INH, watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(2));     	
	fclose(recordFile);
	
	return 0;	
}
