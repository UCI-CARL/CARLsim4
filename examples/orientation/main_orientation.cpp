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
 * Ver 3/22/14
 */ 

#include <carlsim.h>
#include <mtrand.h>

#if (WIN32 || WIN64)
#include <algorithm>
#define fmin std::min
#define fmax std::max
#endif

void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow,
						float* yellow_blue, float* ME, bool GPUpointers);
extern MTRand	      getRand;


#define nrX (32)
#define nrY (nrX)

#define V1_LAYER_DIM	(nrX)
#define V4_LAYER_DIM	(nrX)

#define MTsize (3)

#define FAST_EXP_BUF_SIZE 100
#define FAST_EXP_MAX_STD 5.0
float fast_exp_buf[FAST_EXP_BUF_SIZE] = {-1};
float fast_exp(float x)
{
	if (fast_exp_buf[0] == -1) {
		for (int i=0;i<FAST_EXP_BUF_SIZE;i++) {
			fast_exp_buf[i] = expf(-i*FAST_EXP_MAX_STD/FAST_EXP_BUF_SIZE);
		}
	}
	
	x = -x;
	
	return (x<FAST_EXP_MAX_STD)?fast_exp_buf[(int)(x/FAST_EXP_MAX_STD*FAST_EXP_BUF_SIZE)]:0;
}



float orientation_proj[28][4] = {{0.311800, 0.000000, 0.449200, 0.952322},
				{0.000000, 0.518441, 0.943600, 0.230224},
				{0.248000, 0.942564, 0.499200, 0.000000},
				{0.559400, 0.928139, 0.167400, 0.000000},
				{0.938400, 0.256104, 0.000000, 0.485207},
				{0.000000, 0.007910, 0.887800, 0.661842},
				{0.830200, 0.000000, 0.000000, 0.747119},
				{0.000000, 0.000000, 0.742400, 0.827023},
				{0.906600, 0.573454, 0.000000, 0.122886},
				{0.635000, 0.000000, 0.028400, 0.883308},
				{0.000000, 0.654913, 0.860800, 0.000000},
				{0.762000, 0.774414, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000},
				{0.000000, 0.000000, 0.000000, 0.000000}};


class connectV1toV4o: public ConnectionGenerator {
public:
	connectV1toV4o(int scale, float weightScale, float (*proj)[4], float bias[4]) {
		spatialScale = scale;
		this->weightScale = weightScale;
		this->proj = proj;
		this->bias = bias;
	}

	int spatialScale;
	float weightScale;
	float (*proj)[4];	
	float* bias;
	
	void connect(CARLsim* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int v1X = i%nrX;
		int v1Y = (i/nrX)%nrY;
		int spaceTimeInd = (i/(nrX*nrY))%28;
		int scale = i/(nrX*nrY)/28;

		int edgedist = fmin(fmin(v1X,nrX-1-v1X),fmin(v1Y,nrY-1-v1Y)); // deal with the edges, which tend to have very high responses...

		int v4X = (j%(nrX/spatialScale))*spatialScale;
		int v4Y = ((j/(nrX/spatialScale))%(nrY/spatialScale))*spatialScale;
		int o = j/(nrX*nrY/spatialScale/spatialScale);

		float gaus = fast_exp(-((v4X-v1X)*(v4X-v1X)+(v4Y-v1Y)*(v4Y-v1Y))/MTsize/2);//sqrt(2*3.1459*MTsize);

		connected = getRand()<gaus*proj[spaceTimeInd][o];
		weight = proj[spaceTimeInd][o] * bias[o]*fmin(9,edgedist)/9.0*weightScale;
		delay = 1;
	}
};

class connectV4oitoV4o: public ConnectionGenerator {
public:
	connectV4oitoV4o(int scale, float weightScale) {
		spatialScale = scale;
		this->weightScale = weightScale;
	}

	int spatialScale;
	float weightScale;
	
	void connect(CARLsim* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int X = j%nrX;
		int Y = (j/nrX)%nrY;
		int o = j/(nrX*nrY);

		int iX = (i%(nrX/spatialScale))*spatialScale;
		int iY = ((i/(nrX/spatialScale))%(nrY/spatialScale))*spatialScale;
		int iOr = i/(nrX*nrY/spatialScale/spatialScale);

		float gaus = fast_exp(-((X-iX)*(X-iX)+(Y-iY)*(Y-iY))/MTsize); //for Inhibition use twice the radius...

		connected = getRand()<gaus*(o!=iOr);//cos((o-iOr+2)/4.0*2*3.1459);
		weight = weightScale;
		delay = 1;
	}
};


int main()
{
	MTRand	      getRand(210499257);

	float synscale = 1;
	float stdpscale = 1;

	synscale = synscale*0.001;
	stdpscale = stdpscale*0.04;
	synscale = synscale*4;

	int frameDur = 100;

	FILE* fid;
	bool onGPU = true;
	int ithGPU = 0;

	CARLsim s("orientation",onGPU?GPU_MODE:CPU_MODE,USER,ithGPU);


	int gV1ME = s.createSpikeGeneratorGroup("V1ME", nrX*nrY*28*3, EXCITATORY_NEURON);

	int inhibScale = 2;

	int gV4o = s.createGroup("V4o", nrX*nrY*4, EXCITATORY_NEURON);
	s.setNeuronParameters(gV4o, 0.02f, 0.2f, -65.0f, 8.0f);
	int gV4oi = s.createGroup("V4oi", nrX*nrY*4/inhibScale/inhibScale, INHIBITORY_NEURON);
	s.setNeuronParameters(gV4oi, 0.1f,  0.2f, -65.0f, 2.0f);

	float biasE[4] = {1, 1.2, 1.3, 1.2};
	float biasI[4] = {1, 0.9, 1.3, 0.95};

	s.connect(gV1ME, gV4o, new connectV1toV4o(1, synscale*4.5*2, orientation_proj, biasE), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gV4oi, new connectV1toV4o(inhibScale, synscale*1*2*2, orientation_proj, biasI), SYN_FIXED,1000,3000);

	s.connect(gV4oi, gV4o, new connectV4oitoV4o(inhibScale,-0.01*2), SYN_FIXED,1000,3000);


	s.setConductances(true);
	
	s.setSTDP(ALL, false);

	s.setSTP(ALL,false);

	s.setSpikeMonitor(gV1ME);
#if (WIN32 || WIN64)
	s.setSpikeMonitor(gV4o,"results/spkV4o.dat");
	s.setSpikeMonitor(gV4oi,"results/spkV4oi.dat");
#else
	s.setSpikeMonitor(gV4o,"examples/orientation/results/spkV4o.dat");
	s.setSpikeMonitor(gV4oi,"examples/orientation/results/spkV4oi.dat");
#endif

	// setup the network
	s.setupNetwork();

	unsigned char* vid = new unsigned char[nrX*nrY*3];

	PoissonRate me(nrX*nrY*28*3,onGPU);
	PoissonRate red_green(nrX*nrY,onGPU);
	PoissonRate green_red(nrX*nrY,onGPU);
	PoissonRate yellow_blue(nrX*nrY,onGPU);
	PoissonRate blue_yellow(nrX*nrY,onGPU);

	int VIDLEN = 4*33;

	for(long long i=0; i < VIDLEN*1; i++) {
		if (i%VIDLEN==0) {
#if (WIN32 || WIN64)
			fid = fopen("videos/orienR.dat","rb");
#else
			fid = fopen("examples/orientation/videos/orienR.dat","rb");
#endif
			if (fid==NULL) {
				printf("ERROR: could not open video file\n");
				exit(1);
			}
		}
		
		size_t result = fread(vid,1,nrX*nrY*3,fid);
		if (result!=nrX*nrY*3) {
			printf("ERROR: could not read from video file\n");
			exit(2);
		}

		calcColorME(nrX, nrY, vid, red_green.rates, green_red.rates, blue_yellow.rates, yellow_blue.rates, me.rates, onGPU);

		s.setSpikeRate(gV1ME, &me, 1);

		// run the established network for 1 (sec)  and 0 (millisecond), in GPU_MODE
		s.runNetwork(0,frameDur);

		if (i==1) {
#if (WIN32 || WIN64)
			s.saveSimulation("results/net.dat", true);
#else
			s.saveSimulation("examples/orientation/results/net.dat", true);
#endif
		}
	}
	fclose(fid);
	delete[] vid;
}
