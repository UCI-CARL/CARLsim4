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

  int N_EXC,N_INH, N_INPUT;
	float pConn;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, e2iWeight, i2eWeight, e2eWeight;
	FILE* recordFile;
	int N_Partition;

	if (argc!=13) return 1; // 12 parameters are required


        // read parameters and create a network on GPU
	N_EXC =	0.8*atoi(argv[1]);
	N_INH = 0.2*atoi(argv[1]);
	N_INPUT = N_EXC;
	simulateTime = atoi(argv[2]);
	inputFireRate = atof(argv[3]);
	inputWeight = atof(argv[4]);
	e2iWeight = atof(argv[5]);
	i2eWeight = atof(argv[6]);
	e2eWeight = atof(argv[7]);
	recordFile = fopen(argv[8],"a");

	Stopwatch watch;
	randSeed = atoi(argv[9]);

  pConn = atof(argv[10])/atoi(argv[1]); // connection probability
	N_Partition = atoi(argv[11]);

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network
	ComputingBackend backend;

	if(argv[12] == "CPU")
		backend = CPU_CORES;
	else
		backend = GPU_CORES;



	int gExc[8];
	int gInh[8];
	int gInput[8];
	string Exc[8] ={"exc0", "exc1", "exc2", "exc3", "exc4", "exc5", "exc6", "exc7"};
	string Inh[8] ={"inh0", "inh1", "inh2", "inh3", "inh4", "inh5", "inh6", "inh7"};
	string Input[8]={"input0", "input1", "input2", "input3", "input4", "input5", "input6", "input7"};

	for (int i=0; i<8; i++)
	{
		gExc[i] = sim.createGroup(Exc[i], N_EXC, EXCITATORY_NEURON, i*N_Partition/8, backend);
		sim.setNeuronParameters(gExc[i], 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInh[i] = sim.createGroup(Inh[i], N_INH, INHIBITORY_NEURON, (i*N_Partition/8+1)%N_Partition, backend);
		sim.setNeuronParameters(gInh[i], 0.1f, 0.2f, -65.0f, 2.0f); // RS
		gInput[i] = sim.createSpikeGeneratorGroup(Input[i], N_EXC, EXCITATORY_NEURON, i*N_Partition/8, backend);
	}

	int cinput[8], ce2i[8], ci2e[8], ce2e[8];
	for (int i=0; i<8; i++)
	{
		cinput[i] = sim.connect(gInput[i], gExc[i], "one-to-one", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		ce2i[i] = sim.connect(gExc[i], gInh[i], "random", RangeWeight(e2iWeight), pConn*1.25, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
		ci2e[i] = sim.connect(gInh[i], gExc[i], "random", RangeWeight(i2eWeight), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
		ce2e[i] = sim.connect(gExc[i], gExc[i], "random", RangeWeight(e2eWeight), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	}


	sim.setConductances(false);
	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();

	//setup monitors
	//SpikeMonitor* spkMon1 = sim.setSpikeMonitor(gExc[0], "NULL");
	//SpikeMonitor* spkMon2 = sim.setSpikeMonitor(gInh[0], "NULL");

	//setup some baseline input
        PoissonRate in(N_EXC);
        in.setRates(inputFireRate);
	for (int i=0; i<8; i++){
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

//	s:pkMon1->stopRecording();
//	spkMon2->stopRecording();


	fprintf(recordFile, "%d,%f,%ld,%ld,%ld\n",N_EXC+N_INH, pConn,  watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(2));
	fclose(recordFile);

	return 0;
}
