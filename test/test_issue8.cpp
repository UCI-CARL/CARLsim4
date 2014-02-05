/*
 * ISSUE 8: RESET / INIT
 * see list of issues on GitHub
 *
 * Bug description:
 *   Certain global variables in snn_gpu.cu are set outside class scope. If you want to run multiple networks in a single
 *   file by creating multiple CpuSNN instances, the second network run will fail because these global variables are not
 *   initialized the second time around.
 *   Also, fpParam is never fclosed, which will eventually lead to a memory leak.
 *
 * How bug was fixed:
 *   I created a new method CpuSNN::CpuSNNinitGPUparams, which gets called in the CpuSNN constructor (and hence only once
 *   per class instance).
 *   The file pointer fpParam gets fclosed in CpuSNN::deleteObjects.
 *   Also, I created a new method CpuSNN::deleteObjectsGPU, which eventually should contain all deallocation for snn_gpu.cu,
 *   similar to CpuSNN::deleteObjects in snn_cpu.cpp. However, currently this guy is empty.
 *
 * How the following script tests the bug:
 *   It creates a larger number of CpuSNN instances within a for loop.
 *   Script should fail if bug still persists; i.e. segmentation fault if fpParam is not fclosed.
 *
 * Ver 11/01/13 MB
 */

#include "snn.h"
#include <vector>

extern MTRand getRand;


int main() {
	CpuSNN* snn;
	PoissonRate rIn(1);

	for (int i=0; i<10000; i++) {
		snn = new CpuSNN("bug10");
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
	}

	printf("PASSED\n");
	return 0;	
}
