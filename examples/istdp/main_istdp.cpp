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

#if (WIN32 || WIN64)
	#define _CRT_SECURE_NO_WARNINGS
#endif

#define NUM_DA_NEURON 30
#define NUM_NEURON 10

class SpikeController: public SpikeGenerator {
private:
	int isi;
	int offset;
	bool setOffset;
public:
	SpikeController(int interSpikeInterval, int offsetOfSecondGroup) {
		isi = interSpikeInterval;
		offset = offsetOfSecondGroup;
		setOffset = false;
	}

	unsigned int nextSpikeTime(CARLsim* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (grpId == 2) // gin
			return lastScheduledSpikeTime + isi;
		else if (grpId == 1) { // gex
			if (!setOffset) {
				setOffset = true;
				return lastScheduledSpikeTime + isi + offset;
			} else
				return lastScheduledSpikeTime + isi;
		}

		return 0xFFFFFFFF;
	}

	void updateOffset(int newOffset) {
		offset = newOffset;
		setOffset = false;
	}
};

int main()
{
	// simulation details
	std::string saveFolder = "results/";
	std::vector<int> spikesPost;
	std::vector<int> spikesPre;
	float* weights = NULL;
	int size;
	SpikeMonitor* spikeMon1;
	SpikeMonitor* spikeMonIn;
	SpikeMonitor* spikeMonEx;
	SpikeController* spikeCtrl = new SpikeController(100, 5);
	int gin, gex, g1;
	float BETA_LTP = 0.10f/100;
	float BETA_LTD = 0.12f/100;
	float LAMDA = 12.0f;
	float DELTA = 40.0f;
	FILE* fid = fopen("results/weight.csv", "w");

	// create a network
	CARLsim sim("istdp",CPU_MODE, USER,0,1,42);

	g1=sim.createGroup("excit", 1, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	gex=sim.createSpikeGeneratorGroup("input-ex", 1, EXCITATORY_NEURON);
	gin=sim.createSpikeGeneratorGroup("input-in", 1, INHIBITORY_NEURON);

	sim.connect(gex, g1, "one-to-one", RangeWeight(40.0f/100), 1.0f, RangeDelay(1), SYN_FIXED);
	sim.connect(gin, g1, "one-to-one", RangeWeight(0.0, 5.0f/100, 10.0f/100), 1.0f, RangeDelay(1), SYN_PLASTIC);

	// enable COBA, set up STDP, enable dopamine-modulated STDP
	sim.setConductances(true,5,150,6,150);
	sim.setISTDP(g1, true, STANDARD, BETA_LTP, BETA_LTD, LAMDA, DELTA);

	// set up spike controller on DA neurons
	sim.setSpikeGenerator(gex, spikeCtrl);
	sim.setSpikeGenerator(gin, spikeCtrl);

	// build the network
	sim.setupNetwork();

	spikeMon1 = sim.setSpikeMonitor(g1);
	spikeMonIn = sim.setSpikeMonitor(gin);
	spikeMonEx = sim.setSpikeMonitor(gex);


	// run for 1000 seconds
	for (int t = 0; t < 5; t++) {
		spikeMon1->startRecording();
		spikeMonIn->startRecording();
		spikeMonEx->startRecording();
		sim.runNetwork(1,0,true, true);
		spikeMon1->stopRecording();
		spikeMonIn->stopRecording();
		spikeMonEx->stopRecording();
		
		//spikeMonIn->print();
		//spikeMonEx->print();
		//spikeMon1->print();

		sim.getPopWeights(gin, g1, weights, size);
		printf("%f\n",weights[0]);
	}

	fclose(fid);

	delete spikeCtrl;

	return 0;
}

