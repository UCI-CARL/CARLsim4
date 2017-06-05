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

        int N_NEURON, N_INPUT;
	float pConn1;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight;
	FILE* recordFile;
	string connectionType;
	int spkmon,conmon;
	if (argc!=13) return 1; // 12 parameters are required

	N_INPUT =100;
        // read parameters and create a network on GPU
	N_NEURON = atoi(argv[1]);
	simulateTime = atoi(argv[2]);
	inputFireRate = atof(argv[3]);
	inputWeight = atof(argv[4]);
	excWeight = atof(argv[5]);
	
	recordFile = fopen(argv[6],"a");
	// create a network on GPU
        Stopwatch watch;
	randSeed = atoi(argv[7]);

        pConn1 = atof(argv[8])/N_NEURON; // connection probability

	CARLsim sim("benchmark", GPU_MODE, USER, 0, randSeed);
	CARLsim *simulator = &sim;
	// configure the network

	string node;
	node  = argv[9];
	connectionType = argv[10];
	conmon = atoi(argv[12]);
	spkmon = atoi(argv[11]);

	int gExc1;
	int gExc2;
	int gInput;

	if(node == "GPU"){
		gExc1 = sim.createGroup("exc1", N_NEURON, EXCITATORY_NEURON, 0, GPU_CORES);
		sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);
		gExc2 = sim.createGroup("exc2", N_NEURON/4, EXCITATORY_NEURON, 0, GPU_CORES);
		sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS	
	
	}
	else if(node == "CPU"){
		gExc1 = sim.createGroup("exc1", N_NEURON, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS
		gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, CPU_CORES);
		gExc2 = sim.createGroup("exc2", N_NEURON/4, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS	
	}
	else{
		cout<<"wrong parameter"<<endl;
		return 1;
	}


	int c0, c1;

	if(connectionType == "Fixed"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
		c1 = sim.connect(gExc1, gExc2, "random", RangeWeight(excWeight), pConn1, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	}
	else if(connectionType == "STP"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(0, inputWeight, 2*inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "random", RangeWeight(0, excWeight, 2*excWeight), pConn1, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		sim.setSTP(gExc1, true, 0.01f, 50.0f, 750.0f );
	}
	else if(connectionType == "STDP"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(0, inputWeight, 2*inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "random", RangeWeight(0, excWeight, 2*excWeight), pConn1, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
		sim.setESTDP(gExc1, true, STANDARD, ExpCurve(0.0f, 20.0f, 0.0f, 60.0f));
		sim.setESTDP(gExc2, true, STANDARD, ExpCurve(0.0f, 20.0f, 0.0f, 60.0f));
	}
	else if(connectionType == "Homeostasis"){
		c0 = sim.connect(gInput, gExc1, "full", RangeWeight(0, inputWeight, 2*inputWeight), 1.0, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		c1 = sim.connect(gExc1, gExc2, "random", RangeWeight(0, excWeight, 2*excWeight), pConn1, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
		sim.setESTDP(gExc1, true, STANDARD, ExpCurve(0.0f, 20.0f, 0.0f, 60.0f));
		sim.setESTDP(gExc2, true, STANDARD, ExpCurve(0.0f, 20.0f, 0.0f, 60.0f));
		sim.setHomeostasis(gExc1, true, 0.0f, 2.0f);
		sim.setHomeoBaseFiringRate(gExc1, 20.0f, 0.0f);	
		sim.setHomeostasis(gExc2, true, 0.0f, 2.0f);
		sim.setHomeoBaseFiringRate(gExc2, 20.0f, 0.0f);	
	}
	else{
		return 1;
	}

	sim.setConductances(false);


	// build the network
        watch.lap("Setup Network");
	sim.setupNetwork();


	//setup some baseline input                           
        PoissonRate in(N_INPUT);
        in.setRates(inputFireRate);                                    
        sim.setSpikeRate(gInput, &in);   

	//setup monitors
	SpikeMonitor* spkMon1 = sim.setSpikeMonitor(gExc1, "NULL");
	SpikeMonitor* spkMon2 = sim.setSpikeMonitor(gExc2, "NULL");
	ConnectionMonitor* conMon1;
	ConnectionMonitor* conMon2;
	
	if(spkmon == 1){
		spkMon1->startRecording();
		spkMon2->startRecording();
	}
	if(conmon == 1){
		conMon1	= sim.setConnectionMonitor(gInput, gExc1, "NULL");
		conMon2 = sim.setConnectionMonitor(gExc1, gExc2, "NULL");
	}
	// weight tuning
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed

	watch.lap("runNetwork");
	for(float i=0; i<simulateTime; i=i+0.1){
		sim.runNetwork(0, 100, true);
		if(conmon == 1){
			std::vector< std::vector<float> > weights = conMon1->takeSnapshot();
			std::vector< std::vector<float> > weights2 = conMon2->takeSnapshot();
		}
	}
 	watch.stop();
	if(spkmon == 1){
		spkMon1->stopRecording();
		spkMon2->stopRecording();
	}

	fprintf(recordFile, "%d,%ld,%ld,%ld\n",N_NEURON, watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(2));     	
	fclose(recordFile);
	
	return 0;	
}
