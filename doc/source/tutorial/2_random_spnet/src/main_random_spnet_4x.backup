/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
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
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

#include <carlsim.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <stopwatch.h>

int main(int argc, const char* argv[]) {
	if (argc != 4) {printf("\n\nFewer arguments provided. Aborting.\n");return 1;}

	// ---------------- CONFIG STATE -------------------
	CARLsim sim("spnet", CPU_MODE, USER, 2, 42);

	int scale = atoi(argv[1]);
	int N_CORES = atoi(argv[2]);
	FILE* recordFile;
	recordFile = fopen(argv[3],"a");
	Stopwatch watch;

	int nNeur = 1000*scale;			// number of neurons
	int nNeurExc1 = 0.267*nNeur;	// number of excitatory-1 neurons
	int nNeurExc2 = 0.267*nNeur;	// number of excitatory-2 neurons
	int nNeurExc3 = 0.267*nNeur;	// number of excitatory-3 neurons
	int nNeurInh = 0.2*nNeur;	// number of inhibitory neurons
	int nSynPerNeur = 100;  	// number of synpases per neuron
	int maxDelay = 20;      	// maximal conduction delay

	// create 80-20 network with 80% RS and 20% FS neurons
	int gExc1 = sim.createGroup("exc1", nNeurExc1, EXCITATORY_NEURON, 0, CPU_CORES);
	sim.setNeuronParameters(gExc1, 0.02f, 0.2f, -65.0f, 8.0f); // RS1
	int gExc2 = sim.createGroup("exc2", nNeurExc2, EXCITATORY_NEURON, int(N_CORES/4.0), CPU_CORES);
	sim.setNeuronParameters(gExc2, 0.02f, 0.2f, -65.0f, 8.0f); // RS2
	int gExc3 = sim.createGroup("exc3", nNeurExc3, EXCITATORY_NEURON, int((2*N_CORES)/4.0), CPU_CORES);
	sim.setNeuronParameters(gExc3, 0.02f, 0.2f, -65.0f, 8.0f); // RS3
	int gInh = sim.createGroup("inh", nNeurInh, INHIBITORY_NEURON, int((3*N_CORES)/4.0), CPU_CORES);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	// specify connectivity
	float wtExc = 6.0f;                   // synaptic weight magnitude if pre is exc
	float wtInh = 5.0f;                   // synaptic weight magnitude if pre is inh (no negative sign)
	float wtMax = 10.0f;                  // maximum synaptic weight magnitude
	float pConn = nSynPerNeur*1.0f/nNeur; // connection probability

	// gExc receives input from nSynPerNeur neurons from both gExc and gInh
	// every neuron in gExc should receive ~nSynPerNeur synapses
	sim.connect(gExc1, gExc1, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc1, gExc2, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc1, gExc3, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	sim.connect(gExc2, gExc1, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc2, gExc2, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc2, gExc3, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	sim.connect(gExc3, gExc1, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc3, gExc2, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gExc3, gExc3, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	sim.connect(gInh, gExc1, "random", RangeWeight(0.0f, wtInh, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gInh, gExc2, "random", RangeWeight(0.0f, wtInh, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gInh, gExc3, "random", RangeWeight(0.0f, wtInh, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	//sim.connect(gExc1, gExc1, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc1, gExc2, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc1, gExc3, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);

	//sim.connect(gExc2, gExc1, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc2, gExc2, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc2, gExc3, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);

	//sim.connect(gExc3, gExc1, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc3, gExc2, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExc3, gExc3, "random", RangeWeight(wtExc), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);

	//sim.connect(gInh, gExc1, "random", RangeWeight(wtInh), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gInh, gExc2, "random", RangeWeight(wtInh), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gInh, gExc3, "random", RangeWeight(wtInh), pConn, RangeDelay(1, 20), RadiusRF(-1), SYN_FIXED);

	// gInh receives input from nSynPerNeur neurons from gExc, all delays are 1ms, no plasticity
	// every neuron in gInh should receive ~nSynPerNeur synapses
	sim.connect(gExc1, gInh, "random", RangeWeight(wtExc), pConn*nNeur/(3*nNeurExc1), RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc2, gInh, "random", RangeWeight(wtExc), pConn*nNeur/(3*nNeurExc2), RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gExc3, gInh, "random", RangeWeight(wtExc), pConn*nNeur/(3*nNeurExc3), RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// enable STDP on all incoming synapses to gExc
	float alphaPlus = 0.1f, tauPlus = 20.0f, alphaMinus = 0.1f, tauMinus = 20.0f;
	sim.setESTDP(gExc1, true, STANDARD, ExpCurve(alphaPlus, tauPlus, -alphaMinus, tauMinus));
	sim.setISTDP(gExc1, true, STANDARD, ExpCurve(-alphaPlus, tauPlus, alphaMinus, tauMinus));
	sim.setESTDP(gExc2, true, STANDARD, ExpCurve(alphaPlus, tauPlus, -alphaMinus, tauMinus));
	sim.setISTDP(gExc2, true, STANDARD, ExpCurve(-alphaPlus, tauPlus, alphaMinus, tauMinus));
	sim.setESTDP(gExc3, true, STANDARD, ExpCurve(alphaPlus, tauPlus, -alphaMinus, tauMinus));
	sim.setISTDP(gExc3, true, STANDARD, ExpCurve(-alphaPlus, tauPlus, alphaMinus, tauMinus));

	// run CUBA mode
	sim.setConductances(false);


	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	SpikeMonitor* SMexc1 = sim.setSpikeMonitor(gExc1, "DEFAULT");
	SpikeMonitor* SMexc2 = sim.setSpikeMonitor(gExc2, "DEFAULT");
	SpikeMonitor* SMexc3 = sim.setSpikeMonitor(gExc3, "DEFAULT");
	SpikeMonitor* SMinh = sim.setSpikeMonitor(gInh, "DEFAULT");
	//ConnectionMonitor* CMee = sim.setConnectionMonitor(gExc, gExc, "DEFAULT");
	//ConnectionMonitor* CMei = sim.setConnectionMonitor(gInh, gExc, "DEFAULT");

	// ---------------- RUN STATE -------------------
	SMexc1->startRecording();
	SMexc2->startRecording();
	SMexc3->startRecording();
	SMinh->startRecording();

	watch.lap("runNetwork");

	for (int t=0; t<10000; t++) {
		// random thalamic input to a single neuron from either gExc or gInh
		std::vector<float> thalamCurrExc1(nNeurExc1, 0.0f);
		std::vector<float> thalamCurrExc2(nNeurExc2, 0.0f);
		std::vector<float> thalamCurrExc3(nNeurExc3, 0.0f);
		std::vector<float> thalamCurrInh(nNeurInh, 0.0f);
		float thCurr = 20.0f;

		for (int inj=0; inj<scale; inj++){

			int randNeurId = floor(drand48()*(nNeur-1) + 0.5);
			if (randNeurId < nNeurExc1) {
				// neurId belongs to gExc
				thalamCurrExc1[randNeurId] = thCurr;
			} 
			
			else if (randNeurId < (nNeurExc1 + nNeurExc2)) {
				// neurId belongs to gExc
				thalamCurrExc2[randNeurId - nNeurExc1] = thCurr;
			} 

			else if (randNeurId < (nNeurExc1 + nNeurExc2 + nNeurExc3)) {
				// neurId belongs to gExc
				thalamCurrExc3[randNeurId - (nNeurExc1 + nNeurExc2)] = thCurr;
			} 

			else {
				// neurId belongs to gInh
				thalamCurrInh[randNeurId - (nNeurExc1 + nNeurExc2 + nNeurExc3)] = thCurr;
			}
		}

		sim.setExternalCurrent(gExc1, thalamCurrExc1);
		sim.setExternalCurrent(gExc2, thalamCurrExc2);
		sim.setExternalCurrent(gExc3, thalamCurrExc3);
		sim.setExternalCurrent(gInh, thalamCurrInh);
	
		// run for 1 ms, don't generate run stats
		sim.runNetwork(0,1,false);
	}
	watch.stop();

	SMexc1->stopRecording();
	SMexc2->stopRecording();
	SMexc3->stopRecording();
	SMinh->stopRecording();

	// print firing stats (but not the exact spike times)
	SMexc1->print(false);
	SMexc2->print(false);
	SMexc3->print(false);
	SMinh->print(false);

	//CMee->printSparse();
	//CMei->printSparse();

	fprintf(recordFile, "%d,%d,%ld\n", nNeur, N_CORES, watch.getLapTime(0));
	fclose(recordFile);
	return 0;
}
