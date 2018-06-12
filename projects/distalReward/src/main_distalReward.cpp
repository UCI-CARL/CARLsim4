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

class SpikeController: public SpikeGenerator {
private:
	int rewardQuota;
	unsigned int delay;
public:

	SpikeController(unsigned int _delay) {
		rewardQuota = 0;
		delay = _delay;
	}

	int nextSpikeTime(CARLsim* s, int grpId, int nid, int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) {
		if (rewardQuota > 0 && lastScheduledSpikeTime < currentTime + delay) {
			rewardQuota--;
			return currentTime + delay; // nid / 5
		}

		return 0xFFFFFFFF;
	}

	void setReward(int quota) {
		rewardQuota = quota;
	}

	void setDelay(unsigned int _delay) {
		delay = _delay;
	}
};

int main()
{
	// simulation details
	std::string saveFolder = "results/";
	std::vector<int> spikesExc;
	SpikeController *spikeCtrlS = new SpikeController(900);
	SpikeController *spikeCtrlDA = new SpikeController(900);
	SpikeMonitor *spikeMonExc, *spikeMonInb, *spikeMonDA, *spikeMonS;
	GroupMonitor *groupMonExc;
	ConnectionMonitor *connMonExcExc;
	int gS, gExc, gInb, gExcNoise, gInbNoise, gDA;
	int resA = 0, resB = 0, resNo = 0;
	int spikesA = 0, spikesB = 0;
	int numPre = 0, numPost = 0;
	float ALPHA_LTP_EXC = 0.10f/100;
	float TAU_LTP = 20.0f;
	float ALPHA_LTD_EXC = 0.125f/100;
	float TAU_LTD = 20.0f;

	// create a network
	CARLsim sim("dastdp", CPU_MODE, SILENT, 0,43);

	gExc = sim.createGroup("exc", 800, EXCITATORY_NEURON);
	sim.setNeuronParameters(gExc, 0.02f, 0.2f, -65.0f, 8.0f);

	gInb = sim.createGroup("inb", 200, INHIBITORY_NEURON);
	sim.setNeuronParameters(gInb, 0.1f,  0.2f, -65.0f, 2.0f);

	gExcNoise = sim.createSpikeGeneratorGroup("exc_noise", 800, EXCITATORY_NEURON);
	gInbNoise = sim.createSpikeGeneratorGroup("inb_noise", 200, EXCITATORY_NEURON);

	gS = sim.createSpikeGeneratorGroup("stimulus", 800, EXCITATORY_NEURON);
	gDA = sim.createSpikeGeneratorGroup("da", 50, DOPAMINERGIC_NEURON);

	sim.connect(gExcNoise, gExc, "one-to-one", RangeWeight(8.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	sim.connect(gInbNoise, gInb, "one-to-one", RangeWeight(8.0f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gExcNoise, gExc, "one-to-one", RangeWeight(0.1f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);
	//sim.connect(gInbNoise, gInb, "one-to-one", RangeWeight(0.1f/100), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	sim.connect(gS, gExc, "one-to-one", RangeWeight(12.0f/100), 1.0f, RangeDelay(1, 5), RadiusRF(-1), SYN_FIXED);
	sim.connect(gDA, gExc, "random", RangeWeight(0.000000001f/100), 0.01f, RangeDelay(1), RadiusRF(-1), SYN_FIXED);

	// make random connections with 10% probability
	sim.connect(gInb, gExc, "random", RangeWeight(1.0f/100), 0.1f, RangeDelay(1), RadiusRF(-1), SYN_FIXED); // 1.0f
	// make random connections with 10% probability, and random delays between 1 and 20
	sim.connect(gExc, gInb, "random", RangeWeight(0.0f, 1.0f/100, 2.4f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC); // 4.0f
	sim.connect(gExc, gExc,"random", RangeWeight(0.0f, 1.2f/100, 2.4f/100), 0.1f, RangeDelay(1, 20), RadiusRF(-1), SYN_PLASTIC); // 1.8 ~ 1.6, 6.0f

	// enable COBA, set up STDP, enable dopamine-modulated STDP
	sim.setConductances(true, 5, 150, 6, 150);
	sim.setESTDP(gExc, true, DA_MOD, ExpCurve(ALPHA_LTP_EXC, TAU_LTP, ALPHA_LTD_EXC, TAU_LTD));
	sim.setESTDP(gInb, true, DA_MOD, ExpCurve(ALPHA_LTP_EXC, TAU_LTP, ALPHA_LTD_EXC, TAU_LTD));
	
	sim.setWeightAndWeightChangeUpdate(INTERVAL_10MS, true, 0.99f);

	sim.setNeuromodulator(gExc, 1.0, 50, 1.0, 100, 1.0, 100, 1.0, 100);

	// set up spike controller on DA neurons
	sim.setSpikeGenerator(gDA, spikeCtrlDA);
	sim.setSpikeGenerator(gS, spikeCtrlS);

	// load previous simulation
	//FILE* simFid = NULL;
	//simFid = fopen("results/sim.dat", "rb");
	//if (simFid != NULL) sim.loadSimulation(simFid);

	// build the network
	sim.setupNetwork();

	// close previous simulation
	//if (simFid != NULL) fclose(simFid);

	spikeMonExc = sim.setSpikeMonitor(gExc, "Default");
	spikeMonInb = sim.setSpikeMonitor(gInb, "Default");
	spikeMonDA = sim.setSpikeMonitor(gDA, "Default");
	spikeMonS = sim.setSpikeMonitor(gS, "Default");

	groupMonExc = sim.setGroupMonitor(gExc, "Default");

	// save weights to file periodically
	connMonExcExc = sim.setConnectionMonitor(gExc, gExc, "NULL");

	numPre = connMonExcExc->getNumNeuronsPre();
	numPost = connMonExcExc->getNumNeuronsPost();

	//setup some baseline input
	PoissonRate exc_noise(800);
	exc_noise.setRates(1.0f);
	sim.setSpikeRate(gExcNoise, &exc_noise);

	PoissonRate inb_noise(200);
	inb_noise.setRates(1.0f);
	sim.setSpikeRate(gInbNoise, &inb_noise);

	FILE* analysisFid = NULL;
	FILE* analysisFid2 = NULL;
	FILE* analysisFid3 = NULL;
	FILE* analysisFid4 = NULL;
	analysisFid = fopen("results/ABSpikes.csv", "w");
	analysisFid2 = fopen("results/ABRatio.csv", "w");
	analysisFid3 = fopen("results/ABSpikesAvg.csv", "w");
	analysisFid4 = fopen("results/ABWeightAvg.csv", "w");

	// run for 1000 seconds
	for (int t = 0; t < 2000; t++) {
		if (t % 4 == 0) // set stimulus
			spikeCtrlS->setReward(50);

		spikeMonExc->startRecording();
		groupMonExc->startRecording();
		sim.runNetwork(1, 0, true);
		spikeMonExc->stopRecording();
		groupMonExc->stopRecording();

		// get spike time of groupA and groupB
		if (t % 4 == 0) {
			std::vector<std::vector<int> > spikes = spikeMonExc->getSpikeVector2D();

			// sum up spikes of group A
			int numSpikeA = 0;
			for (int nid = 100; nid < 150; nid++) {
				for (int tid = 0; tid < spikes[nid].size(); tid++) {
					int spikeTime = spikes[nid][tid];
					if (spikeTime % 1000 < 950 && spikeTime % 1000 > 900) numSpikeA++;
				}
			}
			printf("%d \t group A:%d\t",t,numSpikeA);
			fprintf(analysisFid, "%d,", numSpikeA);
			spikesA += numSpikeA;

			// sum up spikes of group B
			int numSpikeB = 0;
			for (int nid = 200; nid < 250; nid++) {
				for (int tid = 0; tid < spikes[nid].size(); tid++) {
					int spikeTime = spikes[nid][tid];
					if (spikeTime % 1000 < 950 && spikeTime % 1000 > 900) numSpikeB++;
				}
			}
			printf("group B:%d", numSpikeB);
			fprintf(analysisFid, "%d\n", numSpikeB);
			spikesB += numSpikeB;

			if (t < 1000 && numSpikeA > numSpikeB) {
				int delay = 900;
				delay -= (numSpikeA - numSpikeB) * 100;

				if (delay <= 0) delay = 10;

				spikeCtrlDA->setDelay(delay);
				spikeCtrlDA->setReward(50);
				printf("\t Reward");
			}

			if (t >= 1000 && numSpikeB > numSpikeA) {
				int delay = 900;
				delay -= (numSpikeB - numSpikeA) * 100;

				if (delay <= 0) delay = 10;

				spikeCtrlDA->setDelay(delay);
				spikeCtrlDA->setReward(50);
				printf("\t Reward");
			}
			
			printf("\n");

			if (numSpikeA > numSpikeB)
				resA++;
			else if (numSpikeA == numSpikeB)
				resNo++;
			else
				resB++;

			if (t % 80 == 0) {
				fprintf(analysisFid2, "%f,%f,%f\n", resA/20.0, resB/20.0, resNo/20.0);
				resA = 0; resB = 0; resNo = 0;
			}

			if (t % 40 == 0) {
				std::vector< std::vector<float> > weights = connMonExcExc->takeSnapshot();
				float weight;

				fprintf(analysisFid3, "%f,%f\n", spikesA/10.0, spikesB/10.0);
				spikesA = 0; spikesB = 0;

				// input weight for group A
				weight = 0.0f;
				for (int postId = 100; postId < 150; postId++) {
					for (int preId = 0; preId < numPre; preId++) {
						// accumulate weight
						if (!isnan(weights[preId][postId]))
							weight += weights[preId][postId];
					}
				}
				fprintf(analysisFid4, "%f,", weight / (50.0f * numPre));

				// input weight for group B
				weight = 0.0f;
				for (int postId = 200; postId < 250; postId++) {
					for (int preId = 0; preId < numPre; preId++) {
						// accumulate weight
						if (!isnan(weights[preId][postId]))
							weight += weights[preId][postId];
					}
				}
				fprintf(analysisFid4, "%f\n", weight / (50.0f * numPre));
			}
		}

		if (t % 4 == 1) {
			std::vector<int> timeVector = groupMonExc->getPeakTimeVector();
			std::vector<float> dataVector = groupMonExc->getPeakValueVector();
			for (int i = 0; i < timeVector.size(); i++) {
//				printf("(%d,%f)", timeVector[i], dataVector[i]);
			}
			printf("\n");
		}
	}

	fclose(analysisFid);
	fclose(analysisFid2);
	fclose(analysisFid3);
	fclose(analysisFid4);

	sim.saveSimulation(saveFolder + "sim.dat");

	delete spikeCtrlDA;
	delete spikeCtrlS;

	return 0;
}
