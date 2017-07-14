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

#define ONE_NEURON 1

int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	CARLsim sim("spnet", HYBRID_MODE, USER, 0, 42);

	float wtExc = 40.0f;

	// create 
	int gExc = sim.createGroup("exc", ONE_NEURON, EXCITATORY_NEURON, 0, CPU_CORES);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f); // RS

	int gInput = sim.createSpikeGeneratorGroup("input", ONE_NEURON, EXCITATORY_NEURON, 0, CPU_CORES);

	// gExc receives input from nSynPerNeur neurons from both gExc and gInh
	// every neuron in gExc should receive ~nSynPerNeur synapses
	sim.connect(gInput, gExc, "full", RangeWeight(wtExc), 1.0, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// run CUBA mode
	sim.setConductances(false);

	SpikeMonitor* SMexc = sim.setSpikeMonitor(gExc, "DEFAULT");
	NeuronMonitor* NMexc = sim.setNeuronMonitor(gExc, "DEFAULT");

	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();

	// ---------------- RUN STATE -------------------
	SMexc->startRecording();
	NMexc->startRecording();

	// random thalamic input to a single neuron from either gExc or gInh
	std::vector<float> thalamCurrExc(ONE_NEURON, 10.0f);
	sim.setExternalCurrent(gExc, thalamCurrExc);

	//for (int t = 0; t < 500; t++) {

	
	
		// run for 1 ms, don't generate run stats
		sim.runNetwork(0,200,false);
	//}
	SMexc->stopRecording();
	NMexc->stopRecording();

	// print firing stats (but not the exact spike times)
	SMexc->print(false);
	NMexc->print();

	return 0;
}
