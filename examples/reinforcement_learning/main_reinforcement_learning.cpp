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
 * Ver 10/09/2013
 */ 

#include "snn.h"
#include "gaussian_connect.h"

#define N 4500

class DopamineController: public SpikeMonitor, SpikeGenerator {
private:
	FILE* fid;
	int nidPre;
	int nidPost;
	int daQuota[50];
	unsigned int controllerCurrentTime;
	unsigned int daReleaseTime;

public:
	DopamineController(FILE* _fid) {
		fid = _fid;
		nidPre = -1;
		nidPost = -1;
		controllerCurrentTime = 0;
		daReleaseTime = 0xFFFFFFFF;

		for (int i = 0; i < 50; i++)
			daQuota[i] = 0;
	}

	// nextSpikeTime is called every one second (1000 ms)
	unsigned int nextSpikeTime(CpuSNN* s, int grpId, int nid, unsigned int currentTime, unsigned int lastScheduledSpikeTime) {
		if (currentTime > daReleaseTime) {
			for (int i = 0; i < 50; i++)
				daQuota[i] = 5;

			daReleaseTime = 0xFFFFFFFF;
		}

		controllerCurrentTime = currentTime;

		if (daQuota[nid] >= 5) {
			daQuota[nid]--;
			printf("a %d %d %d\n", nid, currentTime, lastScheduledSpikeTime);
			return currentTime + 1;
		} else if (daQuota[nid] > 0) {
			daQuota[nid]--;
			printf("b %d %d %d\n", nid, currentTime, lastScheduledSpikeTime);
			return lastScheduledSpikeTime + 1;
		} else {
			return 0xFFFFFFFF;
		}

		return 0xFFFFFFFF;
	}

	void update(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts)
	{
		int pos = 0;
		int lastPreSpikeTime = -21;
		int lastPostSpikeTime = -21;

		for (int t = 0; t < 1000; t++) {
			for(int i = 0; i < timeCnts[t]; i++, pos++) {
				int time = t + s->getSimTime() - 1000;
				int id = Nids[pos];
				//int cnt = fwrite(&time, sizeof(int), 1, fid);
				//assert(cnt != 0);
				//cnt = fwrite(&id, sizeof(int), 1, fid);
				//assert(cnt != 0);

				if (nidPre > 0 && nidPre == id) {
					lastPreSpikeTime = t;
					fprintf(stdout, "Neuron %d spikes at %d\n", nidPre, lastPreSpikeTime);
					
					if (lastPreSpikeTime - lastPostSpikeTime > 0 && lastPreSpikeTime - lastPostSpikeTime <= 20) {
						fprintf(stdout, "LTD:(%d,%d)/(%d,%d)\n", nidPre, lastPreSpikeTime, nidPost, lastPostSpikeTime);
						fprintf(fid, "LTD:(%d,%d)/(%d,%d)@%d\n", nidPre, lastPreSpikeTime, nidPost, lastPostSpikeTime, s->getSimTime());
					}
				}

				if (nidPost > 0 && nidPost == id) {
					lastPostSpikeTime = t;
					fprintf(stdout, "Neuron %d spikes at %d\n", nidPost, lastPostSpikeTime);

					if (lastPostSpikeTime - lastPreSpikeTime > 0 && lastPostSpikeTime - lastPreSpikeTime <= 20) {
						fprintf(stdout, "LTP:(%d,%d)/(%d,%d)\n", nidPre, lastPreSpikeTime, nidPost, lastPostSpikeTime);
						fprintf(fid, "LTP:(%d,%d)/(%d,%d)@%d\n", nidPre, lastPreSpikeTime, nidPost, lastPostSpikeTime, s->getSimTime());
						//daReleaseTime = controllerCurrentTime + 500; // release dopamine 2 second later 
					}
				}
			}

			
		}

		fflush(fid);
	}

	void setPair(int _nidPre, int _nidPost) {
		nidPre = _nidPre;
		nidPost = _nidPost;
	}
};

int main()
{
	// create a network
	CpuSNN* s;
	DopamineController* daController;
	FILE* stdpLog;
	GaussianConnect* gc;
	
	stdpLog = fopen("STDP.txt", "wb");

	s = new CpuSNN("global", GPU_MODE);

	daController = new DopamineController(stdpLog);

	int g1 = s->createGroup("excit", N * 0.8, EXCITATORY_NEURON);
	s->setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	int g2 = s->createGroup("inhib", N * 0.2, INHIBITORY_NEURON);
	s->setNeuronParameters(g2, 0.1f,  0.2f, -65.0f, 2.0f);

	// 50 dopaminergeic neurons
	int gda = s->createGroup("dopaminergic", 50, DOPAMINERGIC_NEURON);
	s->setNeuronParameters(gda, 0.02f, 0.2f, -65.0f, 8.0f);

	// simulate random thalamic noise
	int gin = s->createSpikeGeneratorGroup("input", N * 0.8, EXCITATORY_NEURON);

	// the spike control on dopaminergic neurons
	int gdaControl = s->createSpikeGeneratorGroup("da control", 50, EXCITATORY_NEURON);

	s->setWeightUpdateParameter(_10MS, 100);

	// make random connections with 10% probability
	s->connect(g2, g1, "random", -4.0f/100, -4.0f/100, 0.1f, 1, 1, SYN_FIXED);
	// make random connections with 10% probability, and random delays between 1 and 20
	s->connect(g1, g2, "random", 5.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);
	
	// set up gaussian connectivity
	gc = new GaussianConnect(60, 60, 60, 60, 10.0, 10.0f/100, 1.0, 1, 20, 25);
	//s->connect(g1, g1, "random", 1.0f/100, 10.0f/100, 0.1f,  1, 20, SYN_PLASTIC);
	s->connect(g1, g1, gc, SYN_PLASTIC);

	// 5% probability of connection
	//s->connect(g1, gda, "random", +3.0f/100, 10.0f/100, 0.05f, 10, 20, SYN_FIXED);
	// Dummy synaptic weights. Dopaminergic neurons only release dopamine to the target area in the current model.
	s->connect(gda, g1, "random", 0.0, 0.0, 0.05f, 10, 20, SYN_FIXED);


	// 5% probability of connection
	s->connect(gin, g1, "one-to-one", 20.0f/100, 20.0f/100, 1.0f,  1, 20, SYN_FIXED);

	s->connect(gdaControl, gda, "one-to-one", +20.0f/100, 20.0f/100, 1.0f, 1, 1, SYN_FIXED);

	float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
	s->setConductances(ALL,true,COND_tAMPA,COND_tNMDA,COND_tGABAa,COND_tGABAb);

	// here we define and set the properties of the STDP. 
	float ALPHA_LTP = 0.10f/100, TAU_LTP = 20.0f, ALPHA_LTD = 0.08f/100, TAU_LTD = 40.0f;	
	s->setSTDP(g1, true, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);
	s->setSTDP(g2, true, true, ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD);

	// show logout every 10 secs, enabled with level 1 and output to stdout.
	s->setLogCycle(60, 0, stdout);

	// put spike times into spikes.dat
	//s->setSpikeMonitor(g1,"spikes.dat");
	s->setSpikeMonitor(g1, daController);

	// Show basic statistics about g2
	s->setSpikeMonitor(g2);

	s->setSpikeMonitor(gda);

	s->setSpikeMonitor(gin);

	s->setSpikeMonitor(gdaControl);

	//setup random thalamic noise
	PoissonRate in(N * 0.8);
	for (int i = 0; i < N * 0.8; i++)
		in.rates[i] = 1.0;
	s->setSpikeRate(gin,&in);

	//PoissonRate inDA(N * 0.05);
	//for (int i = 0; i < N * 0.05; i++)
	//	inDA.rates[i] = 1.0;
	//s->setSpikeRate(gdaControl, &inDA);

	s->setSpikeGenerator(gdaControl, (SpikeGenerator*)daController);

	// run network for 1 second
	s->runNetwork(1, 0);

	int nidPre, nidPost;
	s->getConnectionPair(0, nidPre, nidPost);
	daController->setPair(nidPre, nidPost);
	
	//run for 60 seconds
	for(int i = 0; i < 60 * 10 - 1; i++) {
		// run the established network for a duration of 1 (sec)  and 0 (millisecond), in CPU_MODE
		s->runNetwork(1, 0);
	}

	FILE* nid = fopen("network.dat","wb");
	s->writeNetwork(nid);
	fclose(nid);

	fclose(stdpLog);

	delete s;
	delete daController;

	return 0;
}

