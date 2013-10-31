/*
 * BUG 14: MEMORY LEAKS
 * see https://carlserver.ss.uci.edu/bugz
 *
 * Bug description:
 *   Some heap blocks are not freed, leading to memory leak.
 *
 * How bug was fixed:
 *   I updated CpuSNN::deleteObjects and CpuSNN::deleteObjectsGPU to free all data structures allocated in the various
 *   CpuSNN init methods.
 *
 * How the following script tests the bug:
 *   Run the bash script. It will compile the below network and check for memory leaks using valgrind
 *
 * Ver 10/18/13 MB
 */

#include "snn.h"
#include <vector>

extern MTRand getRand;


int main() {
	CpuSNN* snn;
	PoissonRate rIn(1);

	snn = new CpuSNN("bug14");
	int gIn = snn->createSpikeGeneratorGroup("input", 1, EXCITATORY_NEURON);
	int gOut = snn->createGroup("output", 1, EXCITATORY_NEURON);

	snn->setNeuronParameters(gOut, 0.02f, 0.2f, -65.0f, 8.0f); // RS
	snn->setConductances(ALL, true, 5.0, 150.0, 6.0, 150.0); // COBA
	snn->connect(gIn,gOut,"full", 0.5, 0.5, 1.0, 1, 1, SYN_FIXED);

	snn->setLogCycle(0, 0, stdout);
		
	// initialize
	snn->runNetwork(1,0, CPU_MODE);

	snn->setSpikeMonitor(gIn);
	snn->setSpikeMonitor(gOut);

	rIn.rates[0] = 10.0;
	snn->setSpikeRate(gIn, &rIn);

	snn->runNetwork(1,0);

	// deallocate
	delete snn;

	// passing will be determined by valgrind
	return 0;	
}
