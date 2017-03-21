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
#include <stdio.h>
#include <vector>

int main() {
	// ---------------- CONFIG STATE -------------------
	// create a network with nPois Poisson neurons and nExc excitatory output
	// neurons
	CARLsim sim("plasticity simulation", GPU_MODE, USER, 2);
	int nPois = 100; // 100 input neurons
	int nExc  = 1;   // 1 output neuron

	// set up our neuron groups
	int gPois = sim.createSpikeGeneratorGroup("input", nPois, EXCITATORY_POISSON);
	int gExc  = sim.createGroup("output", nExc, EXCITATORY_NEURON);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f);

	// connect our groups with SYN_PLASTIC as the final argument.
	sim.connect(gPois,gExc,"full",RangeWeight(0.0,0.01, 0.03), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

	// set conductances with default values
	sim.setConductances(true);

	// create PoissonRate object of size nPoiss.
	PoissonRate poissRate(nPois);

	// set E-STDP parameters.
	float alpha_LTP=0.001f/5; float tau_LTP=20.0f;
	float alpha_LTD=0.00033f/5; float tau_LTD=60.0f;

	// set E-STDP to be STANDARD (without neuromodulatory influence) with an EXP_CURVE type.
	sim.setESTDP(gExc, true, STANDARD, ExpCurve(alpha_LTP, tau_LTP, -alpha_LTD, tau_LTD));

	// homeostasis constants
	float alpha = 1.0; // homeostatic scaling factor
	float T = 5.0; // homeostatic time constant
	float R_target = 35.0; // target firing rate neuron tries to achieve

	sim.setHomeostasis(gExc,true,alpha,T);
	sim.setHomeoBaseFiringRate(gExc,R_target);

	// ---------------- SETUP STATE -------------------
	sim.setupNetwork();
	SpikeMonitor* SpikeMonInput  = sim.setSpikeMonitor(gPois,"DEFAULT");
	SpikeMonitor* SpikeMonOutput = sim.setSpikeMonitor(gExc,"DEFAULT");
	ConnectionMonitor* CM = sim.setConnectionMonitor(gPois,gExc,"DEFAULT");
	// disable automatic output of synaptic weights from connection monitor
	CM->setUpdateTimeIntervalSec(-1);

	// ---------------- RUN STATE -------------------
	// set rate of each neuron
	std::vector <float> rates;
	for (int i=0; i<nPois; i++)
		rates.push_back((i+1)*(20.0/100));

	poissRate.setRates(rates);
	sim.setSpikeRate(gPois, &poissRate);

	// run the established network for 1 sec
	int runTimeSec = 1000; // seconds
	int runTimeMs  = 0; // milliseconds

	// take a snapshot of the weights before we run the simulation
	CM->takeSnapshot();

	sim.runNetwork(runTimeSec, runTimeMs);

	// take a snapshot of the weights after the simulation runs to completion
	CM->takeSnapshot();

  return 0;
}
