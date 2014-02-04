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
 * Ver 10/09/2013
 */ 

#include <snn.h>

#define N    1000

int main()
{
	// create a network
	CpuSNN s("global");

	int g1=s.createGroup("excit", N*0.8, EXCITATORY_NEURON);
	s.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2=s.createGroup("inhib", N*0.2, INHIBITORY_NEURON);
	s.setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	int gin=s.createSpikeGeneratorGroup("input",N*0.1,EXCITATORY_NEURON);

	// make random connections with 10% probability
	s.connect(g2,g1,"random", -1.0f/100, -1.0f/100, 0.1f, 1, 1, SYN_FIXED);
	// make random connections with 10% probability, and random delays between 1 and 20
	s.connect(g1,g2,"random", +0.25f/100, 0.5f/100, 0.1f,  1, 20, SYN_PLASTIC);
	s.connect(g1,g1,"random", +6.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);

	// 5% probability of connection
	s.connect(gin,g1,"random", +100.0f/100, 100.0f/100, 0.05f,  1, 20, SYN_FIXED);

	float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
	s.setConductances(ALL,true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);

	// here we define and set the properties of the STDP. 
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.12f/100, TAU_LTD = 20.0f;	
	s.setSTDP(g1, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// show logout every 10 secs, enabled with level 1 and output to stdout.
	s.setLogCycle(10, 1, stdout);

	// put spike times into spikes.dat
	s.setSpikeMonitor(g1,"results/random/spikes.dat");

	// Show basic statistics about g2
	s.setSpikeMonitor(g2);

	s.setSpikeMonitor(gin);

	//setup some baseline input
	PoissonRate in(N*0.1);
	for (int i=0;i<N*0.1;i++) in.rates[i] = 1;
	s.setSpikeRate(gin,&in);

	//run for 10 seconds
	for(int i=0; i < 10; i++) {
		// run the established network for a duration of 1 (sec)  and 0 (millisecond), in CPU_MODE
		s.runNetwork(1, 0, GPU_MODE);
	}

	FILE* nid = fopen("results/random/network.dat","wb");
	s.writeNetwork(nid);
	fclose(nid);

	return 0;
}

