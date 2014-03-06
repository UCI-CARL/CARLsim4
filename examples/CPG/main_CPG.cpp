/*
 * Copyright (c) 2013 Regents of the University of California. All rights reserved.
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
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 3/4/14
 */ 

#include <carlsim.h>

class SpikeAt20Hz : public SpikeGenerator {
	unsigned int nextSpikeTime(CpuSNN* snn, int grpId, int nid, unsigned int currentTime) {
		return currentTime+50;
	}
};

int main()
{

	int nIn = 1;
	int nExc = 1;
	int nInh = 1;
	int ithGPU = 0;

	simMode_t simMode = CPU_MODE;
	int nSec = 1; // run for 10 s

	// create a network
	CARLsim sim("CPG",simMode,USER,ithGPU,1,2);

	int g0 = sim.createSpikeGeneratorGroup("input",nIn,EXCITATORY_NEURON);

	int g1 = sim.createGroup("excLeft",nExc,EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

//	int g2 = sim.createGroup("excRight",nExc,EXCITATORY_NEURON);
//	sim.setNeuronParameters(g2, 0.02f, 0.2f, -65.0f, 8.0f);

//	int g3 = sim.createGroup("inhLeft",nInh,INHIBITORY_NEURON);
//	sim.setNeuronParameters(g3, 0.1f, 0.2f, -65.0f, 2.0f);

//	int g4 = sim.createGroup("inhRight",nInh,INHIBITORY_NEURON);
//	sim.setNeuronParameters(g4, 0.1f, 0.2f, -65.0f, 2.0f);

	// use default conductance values
	sim.setConductances(ALL,true);

	// from input to exc: fixed full
	sim.connect(g0,g1,"full",0.01,1.0,1);
//	sim.connect(g0,g2,"full",0.01,1.0,1);

	// from exc to self: recurrent excitation with STD
//	sim.connect(g1,g1,"one-to-one",0.1,1.0,1);
//	sim.connect(g2,g2,"one-to-one",0.1,1.0,1);

	// from exc to other exc: inhibition
//	sim.connect(g1,g3,"full",0.01,1.0,1); sim.connect(g3,g2,"full",-0.01,1.0,1);
//	sim.connect(g2,g4,"full",0.01,1.0,1); sim.connect(g4,g1,"full",-0.01,1.0,1);

	// set STP using default values
	// FIXME: this should be something along the lines of:
	// sim.setSTP(g1,g2,true); (connection-based)

//	The neuronal firing rate R=15Hz. The parameters A=1, U=0.45, τs=20ms, τd=750ms, and τf=50ms
	sim.setSTP(g0,true,0.45,50,750);

//	0.15, τf=750ms, and τd=50
//	sim.setSTP(g0,true,0.15,750,50);

	sim.setSpikeMonitor(g0);
	sim.setSpikeMonitor(g1);
//	sim.setSpikeMonitor(g2);
//	sim.setSpikeMonitor(g3);
//	sim.setSpikeMonitor(g4);

	sim.setSpikeGenerator(g0, new SpikeAt20Hz());

	// run network
	sim.runNetwork(0,300);

	return 0;
}

