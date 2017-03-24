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

        int N_EXC, N_INH, N_INPUT, N_NEURONS;
	float pConn;
	int randSeed;
	int simulateTime;
	float inputFireRate;
	float inputWeight, excWeight, inhWeight;
	FILE* recordFile;	


	if (argc!=12) return 1; // 11 parameters are required


        // read parameters and create a network on GPU
	N_NEURONS = atoi(argv[1]);
	N_EXC = int(atoi(argv[1])*atof(argv[2]));
	N_INH = N_NEURONS-N_EXC;
	N_INPUT = int(N_NEURONS*atof(argv[3]));
	simulateTime = atoi(argv[7]);
	inputFireRate = atof(argv[8]);
	inputWeight = atof(argv[9]);
	excWeight = atof(argv[10]);
	inhWeight = atof(argv[11]);
	
	recordFile = fopen(argv[6],"a");
	// create a network on GPU
        Stopwatch watch;
	int numGPUs = 2;
	randSeed = atoi(argv[4]);
        pConn = atof(argv[5])/N_NEURONS; // connection probability
	CARLsim sim("benchmark", GPU_MODE, SILENT, 0, randSeed);

	
	// configure the network
	int gExc = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS


	int gInh = sim.createGroup("inh", N_INH, INHIBITORY_NEURON, 1, GPU_CORES);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	//int gExc2 = sim.createGroup("exc", N_EXC, EXCITATORY_NEURON);
	//sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gInput = sim.createSpikeGeneratorGroup("input", N_INPUT, EXCITATORY_NEURON, 0, GPU_CORES);


	//FixedSpikeGenerator* f1 = new FixedSpikeGenerator();
	//sim.setSpikeGenerator(gInput, f1);


	sim.connect(gInput, gExc, "full", RangeWeight(inputWeight), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gExc, "random", RangeWeight(excWeight), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gInh, "random", RangeWeight(excWeight), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInh, gInh, "random", RangeWeight(inhWeight), pConn, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
        sim.connect(gInh, gExc, "random", RangeWeight(inhWeight), pConn, RangeDelay(1), RadiusRF(-1), SYN_FIXED);  

	sim.setConductances(false);



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

	for (int t = 0; t < simulateTime; t++) {
		sim.runNetwork(1, 0, true);
	}

 	watch.stop();


	fprintf(recordFile, "%ld,%ld,%ld\n", watch.getLapTime(0), watch.getLapTime(1),
 watch.getLapTime(2));     	
	fclose(recordFile);

}
