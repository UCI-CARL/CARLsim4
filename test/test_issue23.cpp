/*
 * ISSUE 23: REAL-TIME SPIKE MONITOR
 * see list of issues on GitHub
 *
 * Feature description:
 *   If we are to integrate CARLsim with real-time systems, we need to be able to get spikes out per frame, not only per
 *   second of simulation time.
 *
 * How bug was fixed:
 *   
 *
 * How the following script tests the bug:
 *
 * Ver 01/24/14 mb
 */

#include "snn.h"
#include <vector>

extern MTRand getRand;


int main() {
	CpuSNN* snn;

	int frameDur = 50; 		// frame duration
	int frameNum = 2000;		// number of frames per simulation

	int numNeurIn = 1000;		// number of neurons in input group
	int numNeurOut = 1000;	// number of neurons in output group

	float rateIn = 10.0f; 	// mean spike rate of input group
	PoissonRate rIn(numNeurIn);

	std::vector<unsigned int>* spkOut;


	for (int sim_mode=0; sim_mode<1; sim_mode++) {
		// run in CPU and GPU mode

		for (int useSpkMonRT=0; useSpkMonRT<=0; useSpkMonRT++) {
			// run with SpikeMonRT off and on

			snn = new CpuSNN("issue23");
			int gIn = snn->createSpikeGeneratorGroup("input", numNeurIn, EXCITATORY_NEURON);
			int gOut = snn->createGroup("output", numNeurOut, EXCITATORY_NEURON);

			snn->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f); // RS
			snn->setConductances(ALL, true, 5.0, 150.0, 6.0, 150.0); // COBA
			snn->connect(gIn,gOut,"random", 0.025, 0.025, 0.1, 1, 1, SYN_FIXED);

			snn->setLogCycle(0, 0, stdout);
		
			// initialize
			snn->runNetwork(1,0, sim_mode);

			// regular spike monitors, updated every 1000ms
			snn->setSpikeMonitor(gIn);
			snn->setSpikeMonitor(gOut);

			// "real-time" spike monitor, keeps track of the number of spikes per neuron in a group
			// works for excitatory/inhibitory neurons as well as spike generators
			// the recording time can be set to any x number of ms, so that after x ms the spike counts will be reset
			// to zero. if x==-1, then the spike counts will never be reset (should only overflow after 97 days of sim)
			// also, spike counts can be manually reset at any time by calling snn->resetSpikeMonitorRealTime(group);
			// you can have only one real-time spike monitor per group. however, a group can have both a regular and a
			// real-time spike monitor
			if (useSpkMonRT) {
				snn->setSpikeMonitorRealTime(gOut,frameDur);
			}


			for (int j=0; j<numNeurIn; j++)
				rIn.rates[j] = rateIn;
			snn->setSpikeRate(gIn, &rIn);

			// main loop
			for (int i=0; i<frameNum; i++) {
				snn->runNetwork(0,frameDur);

//				if (i==frameNum/2)
//					snn->resetSpikeMonitorRealTime(ALL);
			}

			if (useSpkMonRT) {
				spkOut = snn->getSpikesRealTime(gOut);	// gets out all the spikes3
				printf("spikes of group %d:\n",gOut);
				unsigned int spkOutTotal=0;
				for (int n=0; n<snn->grpNumNeurons(gOut); n++) {
					spkOutTotal += (*spkOut)[n];
					printf("%u\t",(*spkOut)[n]);
				}
				printf("\nTotal: %u spikes\n",spkOutTotal);
			}

			// deallocate
			delete snn;
		}
	}

	printf("PASSED\n");
	return 0;	
}
