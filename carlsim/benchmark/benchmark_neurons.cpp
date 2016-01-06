/* * Copyright (c) 2015 Regents of the University of California. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
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

#define NUM_GPUS 2

int main(int argc, char* argv[]) {
	int gExc, gExc2, gInh, gInh2, gInput, gInput2;
	int numN, numExc, numInh;
	int randSeed, gpuId;
	float pConn;
	FILE* retFile;

	if (argc != 5) return 1; // 4 input parameters are required

	// setup benchmark parameters
	numN = atoi(argv[1]);
	numExc = numN * 8 / 10;
	numInh = numN * 2 / 10;
	pConn = 100.0f / (numExc + numInh); // connection probability

	randSeed = atoi(argv[2]);

	gpuId = atoi(argv[3]);

	retFile = fopen(argv[4], "a");
	
	// create CARLsim object
	Stopwatch watch(false);
	CARLsim sim("benchmark_neurons", GPU_MODE, SILENT, NUM_GPUS, randSeed);
		
	// configure the network
	watch.start();
	gExc = sim.createGroup("exc", numExc, EXCITATORY_NEURON, 0);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	gExc2 = sim.createGroup("exc2", numExc, EXCITATORY_NEURON, gpuId);
	sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f);

	gInh = sim.createGroup("inh", numInh, INHIBITORY_NEURON, gpuId);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	gInh2 = sim.createGroup("inh2", numInh, INHIBITORY_NEURON, 0);
	sim.setNeuronParameters(gInh2, 0.1f, 0.2f, -65.0f, 2.0f);

	gInput = sim.createSpikeGeneratorGroup("input", numExc / 100, EXCITATORY_NEURON, 0);

	gInput2 = sim.createSpikeGeneratorGroup("input2", numExc / 100, EXCITATORY_NEURON, gpuId);

	sim.connect(gInput, gExc, "random", RangeWeight(30.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gExc, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc, gInh, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInh, gExc, "random", RangeWeight(5.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.connect(gInput2, gExc2, "random", RangeWeight(30.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc2, gExc2, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc2, gInh2, "random", RangeWeight(6.0f), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInh2, gExc2, "random", RangeWeight(5.0f), pConn * 1.25f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.setConductances(false);

	// build the network
	watch.lap();
	sim.setupNetwork();

	// set some monitors
	//SpikeMonitor* smInput = sim->setSpikeMonitor(gInput, "DEFAULT");
	//SpikeMonitor* smExc = sim->setSpikeMonitor(gExc, "NULL");
	//SpikeMonitor* smInh = sim->setSpikeMonitor(gInh, "NULL");
	//SpikeMonitor* smInput = sim->setSpikeMonitor(gInput, "NULL");

	//SpikeMonitor* smExc2 = sim->setSpikeMonitor(gExc2, "NULL");
	//SpikeMonitor* smInh2 = sim->setSpikeMonitor(gInh2, "NULL");
	//SpikeMonitor* smInput2 = sim->setSpikeMonitor(gInput2, "NULL");

	//setup some baseline input
	PoissonRate in(numExc / 100);
	in.setRates(1.0f);
	sim.setSpikeRate(gInput, &in);

	PoissonRate in2(numExc / 100);
	in2.setRates(1.0f);
	sim.setSpikeRate(gInput2, &in2);

	// run the network for 10 seconds
	watch.lap();
	//smInput->startRecording();
	//smExc->startRecording();
	//smInh->startRecording();
	//smInput->startRecording();
	
	sim.runNetwork(10, 0);
	
	//smInput->stopRecording();
	//smExc->stopRecording();
	//smInh->stopRecording();
	//smInput->stopRecording();

	//smExc->print(true);
	//smExc2->print(true);
	//smInput->print(true);
	watch.stop(false);

	fprintf(retFile, "%ld,%ld,%ld\n", watch.getLapTime(0), watch.getLapTime(1), watch.getLapTime(2));
	printf("config %ld, setup %ld, run %ld\n", watch.getLapTime(0),watch.getLapTime(1), watch.getLapTime(2));
	fclose(retFile);

	return 0;
}
