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
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARL/CARLsim/
 * Ver 3/22/14
 */

#include <carlsim.h>

#include <stdio.h>		// printf, fopen
#include <math.h>		// expf

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

#define NUM_DA_NEURON 30
#define NUM_NEURON 10

class GroupController: public GroupMonitor {
private:
	FILE* fid;
public:
	GroupController(std::string saveFolder) {
		std::string fileName = saveFolder + "DA.csv";

		fid = fopen(fileName.c_str(), "w");
	}

	~GroupController() {
		fclose(fid);
	}

	void update(CARLsim* s, int grpId, float* daBuffer, int n) {
		for (int i = 0; i < 100 /* n is 100 currently */; i++)
			fprintf(fid, "%f ", daBuffer[i]);
	}
};

class SpikeController: public SpikeGenerator {
private:
	int rewardQuota;
public:
	SpikeController() {
		rewardQuota = 0;
	}

	unsigned int nextSpikeTime(CARLsim* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (rewardQuota > 0 && lastScheduledSpikeTime < currentTime + 500) {
			rewardQuota--;
			return currentTime + 500;
		}

		return 0xFFFFFFFF;
	}

	void setReward(int quota) {
		rewardQuota = quota;
	}
};

int main()
{
	// simulation details
	std::string saveFolder = "results/";
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights;
	int size;
	SpikeMonitor* spikeMon1;
	SpikeMonitor* spikeMonIn;
	GroupController* grpCtrl = new GroupController(saveFolder);
	SpikeController* spikeCtrl = new SpikeController();
	int gin, g1, g1noise, gda;
	float ALPHA_LTP_EXC = 0.10f/100;
	float TAU_LTP = 20.0f;
	float ALPHA_LTD_EXC = 0.125f/100;
	float TAU_LTD = 20.0f;

	// create a network
	CARLsim sim("dastdp",GPU_MODE,USER,0,42);

	g1=sim.createGroup("excit", NUM_NEURON, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	gin=sim.createSpikeGeneratorGroup("input", NUM_NEURON, EXCITATORY_NEURON);

	g1noise = sim.createSpikeGeneratorGroup("noise", NUM_NEURON, EXCITATORY_NEURON);

	gda = sim.createSpikeGeneratorGroup("DA neurons", NUM_DA_NEURON, DOPAMINERGIC_NEURON);

	sim.connect(gin, g1, "full", RangeWeight(0.0, 1.0f/100, 10.0f/100), 1.0f, RangeDelay(1,20), RadiusRF(-1), SYN_PLASTIC);
	sim.connect(g1noise, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gda, g1, "full", RangeWeight(0.0), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// enable COBA, set up STDP, enable dopamine-modulated STDP
	sim.setConductances(true,5,150,6,150);
	sim.setSTDP(g1, true, DA_MOD, ALPHA_LTP_EXC, TAU_LTP, ALPHA_LTD_EXC, TAU_LTD);
	sim.setWeightAndWeightChangeUpdate(INTERVAL_10MS, true, 0.99f);

	// set up spike controller on DA neurons
	sim.setSpikeGenerator(gda, spikeCtrl);

	// build the network
	sim.setupNetwork();

	spikeMon1 = sim.setSpikeMonitor(g1);
	spikeMonIn = sim.setSpikeMonitor(gin);
	sim.setSpikeMonitor(gda);

	sim.setGroupMonitor(g1, grpCtrl);

	// save weights to file periodically
	sim.setConnectionMonitor(gin, g1);


	//setup some baseline input
	PoissonRate in(NUM_NEURON);
	in.setRates(4.0f);
	sim.setSpikeRate(gin,&in);

	PoissonRate noise(NUM_NEURON);
	noise.setRates(4.0f);
	sim.setSpikeRate(g1noise,&noise);


	// run for 1000 seconds
	for (int t = 0; t < 1000; t++) {
		spikeMon1->startRecording();
		spikeMonIn->startRecording();
		sim.runNetwork(1,0,true, true);
		spikeMon1->stopRecording();
		spikeMonIn->stopRecording();

		// get spike time of pre-synaptic neuron post-synaptic neuron
		spikesPre = spikeMonIn->getSpikeVector2D()[0]; // first neuron in pre-synaptic group
		spikesPost = spikeMon1->getSpikeVector2D()[1]; // second neuron in post-synaptic group

		// detect LTP or LTD
		for (int j = 0; j < spikesPre.size(); j++) { // j: index of the (j+1)-th spike
			for (int k = 0; k < spikesPost.size(); k++) { // k: index of the (k+1)-th spike
				int diff = spikesPost[k] - spikesPre[j]; // (post-spike time) - (pre-spike time)
				// if LTP is detected, set up reward (activate DA neurons ) to reinforcement this synapse
				if (diff > 0 && diff <= 20) {
				//	printf("LTP\n");
					spikeCtrl->setReward(NUM_DA_NEURON);
				}

				//if (diff < 0 && diff >= -20)
				//	printf("LTD\n");
			}
		}
	}

	delete grpCtrl;
	delete spikeCtrl;

	return 0;
}

