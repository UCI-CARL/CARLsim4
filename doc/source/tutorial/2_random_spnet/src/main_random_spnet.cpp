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


int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	CARLsim sim("spnet", GPU_MODE, USER, 2, 42);

	int nNeur = 1000;			// number of neurons
	int nNeurExc = 0.8*nNeur;	// number of excitatory neurons
	int nNeurInh = 0.2*nNeur;	// number of inhibitory neurons
	int nSynPerNeur = 100;  	// number of synpases per neuron
	int maxDelay = 20;      	// maximal conduction delay

	// create 80-20 network with 80% RS and 20% FS neurons
	int gExc = sim.createGroup("exc", nNeurExc, EXCITATORY_NEURON, 0, GPU_CORES);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS
	int gInh = sim.createGroup("inh", nNeurInh, INHIBITORY_NEURON, 1, GPU_CORES);
	sim.setNeuronParameters(gInh, 0.1f, 0.2f, -65.0f, 2.0f); // FS

	// specify connectivity
	float wtExc = 6.0f;                   // synaptic weight magnitude if pre is exc
	float wtInh = 5.0f;                   // synaptic weight magnitude if pre is inh (no negative sign)
	float wtMax = 10.0f;                  // maximum synaptic weight magnitude
	float pConn = nSynPerNeur*1.0f/nNeur; // connection probability

	// gExc receives input from nSynPerNeur neurons from both gExc and gInh
	// every neuron in gExc should receive ~nSynPerNeur synapses
	sim.connect(gExc, gExc, "random", RangeWeight(0.0f, wtExc, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(gInh, gExc, "random", RangeWeight(0.0f, wtInh, wtMax), pConn, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);

	// gInh receives input from nSynPerNeur neurons from gExc, all delays are 1ms, no plasticity
	// every neuron in gInh should receive ~nSynPerNeur synapses
	sim.connect(gExc, gInh, "random", RangeWeight(wtExc), pConn*nNeur/nNeurExc, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// enable STDP on all incoming synapses to gExc
	float alphaPlus = 0.1f, tauPlus = 20.0f, alphaMinus = 0.1f, tauMinus = 20.0f;
	sim.setESTDP(gExc, true, STANDARD, ExpCurve(alphaPlus, tauPlus, -alphaMinus, tauMinus));
	sim.setISTDP(gExc, true, STANDARD, ExpCurve(-alphaPlus, tauPlus, alphaMinus, tauMinus));

	// run CUBA mode
	sim.setConductances(false);


	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	SpikeMonitor* SMexc = sim.setSpikeMonitor(gExc, "DEFAULT");
	SpikeMonitor* SMinh = sim.setSpikeMonitor(gInh, "DEFAULT");
	ConnectionMonitor* CMee = sim.setConnectionMonitor(gExc, gExc, "DEFAULT");
	ConnectionMonitor* CMei = sim.setConnectionMonitor(gInh, gExc, "DEFAULT");

	// ---------------- RUN STATE -------------------
	SMexc->startRecording();
	SMinh->startRecording();
	for (int t=0; t<10000; t++) {
		// random thalamic input to a single neuron from either gExc or gInh
		std::vector<float> thalamCurrExc(nNeurExc, 0.0f);
		std::vector<float> thalamCurrInh(nNeurInh, 0.0f);
		int randNeurId = floor(drand48()*(nNeur-1) + 0.5);
		float thCurr = 20.0f;
		if (randNeurId < nNeurExc) {
			// neurId belongs to gExc
			thalamCurrExc[randNeurId] = thCurr;
		} else {
			// neurId belongs to gInh
			thalamCurrInh[randNeurId - nNeurExc] = thCurr;
		}
	
		sim.setExternalCurrent(gExc, thalamCurrExc);
		sim.setExternalCurrent(gInh, thalamCurrInh);
	
		// run for 1 ms, don't generate run stats
		sim.runNetwork(0,1,false);
	}
	SMexc->stopRecording();
	SMinh->stopRecording();

	// print firing stats (but not the exact spike times)
	SMexc->print(false);
	SMinh->print(false);

	CMee->printSparse();
	CMei->printSparse();

	return 0;
}
