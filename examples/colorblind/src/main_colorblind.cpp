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

//Change this!
#include <snn.h>
void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers);
extern MTRand	      getRand;


#define nrX (64)
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



enum color_name_t { BLUE=0, GREEN, RED, YELLOW};
string imageName[] = { "blue", "green", "red", "yellow"};

enum v1color_name_t    { RED_GREEN=0, BLUE_YELLOW, GREEN_RED, YELLOW_BLUE };
string v1ImageName[] = { "red-green-cells", "blue-yellow-cells",
			 "green-red-cells", "yellow-blue-cells"};

string  v4CellNameExc[] = { "Ev4magenta", "Ev4blue", "Ev4cyan", "Ev4green", "Ev4yellow", "Ev4red"};
string  v4CellNameInh[] = { "Iv4magenta", "Iv4blue", "Iv4cyan", "Iv4green", "Iv4yellow", "Iv4red"};

enum  v4CellType_t  {MAGENTA_V4=0, BLUE_V4, CYAN_V4, GREEN_V4, YELLOW_V4, RED_V4};


class v1v4Proj: public ConnectionGenerator {
public:
	v1v4Proj(int src_x, int src_y, int dest_x, int dest_y, int radius, float weightScale) {
		this->src_x=src_x;
		this->src_y=src_y;
		this->dest_x=dest_x;
		this->dest_y=dest_y;
		this->radius=radius;
		this->weightScale = weightScale;
	}

	int src_x; int src_y; int dest_x; int dest_y; int radius;
	float weightScale;
	
	void connect(CpuSNN* net, int srcGrp, int src_i, int destGrp, int dest_i, float& weight, float& maxWt, float& delay, bool& connected)
	{
		// extract x and y positions...
		int dest_i_x  = dest_i%dest_x;
		int dest_i_y  = dest_i/dest_x;
		int src_i_y = src_i/src_x;
		int src_i_x = src_i%(src_x);

		float distance2 = ((dest_i_y-src_i_y)*(dest_i_y-src_i_y))+((dest_i_x-src_i_x)*(dest_i_x-src_i_x));
		float gaus = fast_exp(-distance2/radius/radius*3);
		
		connected = gaus>0.1;
		delay     = 1;
		weight    = gaus*weightScale;
	}
};

class simpleProjection: public ConnectionGenerator {
public:
	simpleProjection(float radius, float weightScale) {
		this->weightScale = weightScale;
		localRadius2 = radius*radius;
	}

	float localRadius2;
	float weightScale;
	
	void connect(CpuSNN* net, int srcGrp, int src_i, int destGrp, int dest_i, float& weight, float& maxWt, float& delay, bool& connected)
	{
		// extract x and y position from the destination
		int dest_i_x = dest_i%V4_LAYER_DIM;
		int dest_i_y = dest_i/V4_LAYER_DIM;

		// extract x and y position from the source
		int src_i_x = src_i%V4_LAYER_DIM;
		int src_i_y = src_i/V4_LAYER_DIM;

		float distance2 = ((dest_i_y-src_i_y)*(dest_i_y-src_i_y))+((dest_i_x-src_i_x)*(dest_i_x-src_i_x));
		float gaus = fast_exp(-distance2/localRadius2*3);

		connected = gaus>0.1;
		delay       = 1.0;
		weight  = gaus*weightScale;
	}
};

int main()
{
	MTRand	      getRand(210499257);

	#define FRAMEDURATION 100

	FILE* fid;

	CpuSNN s("global");

	int v1Cells[5];
	int num_V1_groups=6;
	for (int i=RED_GREEN; i <= YELLOW_BLUE; i++) {
		v1Cells[i] = s.createSpikeGeneratorGroup(v1ImageName[i].c_str(), V1_LAYER_DIM*V1_LAYER_DIM, TARGET_AMPA);

//		s.setSTP(v1Cells[i],true, 0.2, 800, 20);
	}
	
	int   num_V4_groups = RED_V4+1;

	int v4CellsExc[num_V4_groups];
	int v4CellsInh[num_V4_groups];
	for (int i=0; i < num_V4_groups; i++) {
		v4CellsExc[i] = s.createGroup(v4CellNameExc[i].c_str(), V4_LAYER_DIM*V4_LAYER_DIM, TARGET_AMPA);
		s.setNeuronParameters(v4CellsExc[i], 0.02f, 0.2f, -65.0f, 8.0f);
		v4CellsInh[i] = s.createGroup(v4CellNameInh[i].c_str(), V4_LAYER_DIM*V4_LAYER_DIM, TARGET_GABAa);
		s.setNeuronParameters(v4CellsInh[i], 0.1f,  0.2f, -65.0f, 2.0f);
	}

	//{int src_x; int src_y; int dest_x; int dest_y; int overlap; int radius; float  prob;} projParam_t;
	float v1toV4w = 0.5;
	float v1toV4iw = 0.5;
	float radius = sqrt(3);
	
	v1v4Proj* projSecondary = new v1v4Proj(V1_LAYER_DIM, V1_LAYER_DIM, V4_LAYER_DIM, V4_LAYER_DIM, radius, v1toV4w*1.5);
	v1v4Proj* projPrimary = new v1v4Proj(V1_LAYER_DIM, V1_LAYER_DIM, V4_LAYER_DIM, V4_LAYER_DIM, radius, v1toV4w);
	v1v4Proj* projYellow = new v1v4Proj(V1_LAYER_DIM, V1_LAYER_DIM, V4_LAYER_DIM, V4_LAYER_DIM, radius, v1toV4w*2);
	v1v4Proj* projInhib = new v1v4Proj(V1_LAYER_DIM, V1_LAYER_DIM, V4_LAYER_DIM, V4_LAYER_DIM, radius, v1toV4iw);

	// feedforward connections
	s.connect(v1Cells[RED_GREEN],   v4CellsExc[MAGENTA_V4], projSecondary, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsExc[MAGENTA_V4], projSecondary, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v1Cells[GREEN_RED],   v4CellsExc[CYAN_V4], projSecondary, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsExc[CYAN_V4], projSecondary, SYN_FIXED, radius*radius*4, radius*radius*4);

	// same weight connectivity pattern, but the probability has been made 1.00 instead of 0.60 in the previous case...
	s.connect(v1Cells[RED_GREEN],   v4CellsExc[RED_V4], projPrimary, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[GREEN_RED], 	v4CellsExc[GREEN_V4], projPrimary, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsExc[BLUE_V4], projPrimary, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[YELLOW_BLUE], v4CellsExc[YELLOW_V4], projYellow, SYN_FIXED, radius*radius*4, radius*radius*4);


	// feedforward inhibition
	s.connect(v1Cells[RED_GREEN],   v4CellsInh[MAGENTA_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsInh[MAGENTA_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v1Cells[GREEN_RED],   v4CellsInh[CYAN_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsInh[CYAN_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);

	// same weight connectivity pattern, but the probability has been made 1.00 instead of 0.60 in the previous case...
	s.connect(v1Cells[RED_GREEN], 	v4CellsInh[RED_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[GREEN_RED], 	v4CellsInh[GREEN_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[BLUE_YELLOW], v4CellsInh[BLUE_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v1Cells[YELLOW_BLUE], v4CellsInh[YELLOW_V4], projInhib, SYN_FIXED, radius*radius*4, radius*radius*4);

	// laternal connections.....
	float wtScale = -0.3;

	radius *= 2; // inhibition has a larger radius
	simpleProjection* projInhToExc = new simpleProjection(radius, wtScale);
	
	s.connect(v4CellsInh[MAGENTA_V4], v4CellsExc[CYAN_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[MAGENTA_V4], v4CellsExc[YELLOW_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v4CellsInh[CYAN_V4], v4CellsExc[MAGENTA_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[CYAN_V4], v4CellsExc[YELLOW_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v4CellsInh[YELLOW_V4], v4CellsExc[CYAN_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[YELLOW_V4], v4CellsExc[MAGENTA_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
/*
	s.connect(v4CellsInh[RED_V4], v4CellsExc[GREEN_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[RED_V4], v4CellsExc[BLUE_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v4CellsInh[BLUE_V4], v4CellsExc[RED_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[BLUE_V4], v4CellsExc[GREEN_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);

	s.connect(v4CellsInh[GREEN_V4], v4CellsExc[RED_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
	s.connect(v4CellsInh[GREEN_V4], v4CellsExc[BLUE_V4], projInhToExc, SYN_FIXED, radius*radius*4, radius*radius*4);
*/
	// show log every 1 sec (0 to disable logging). You can pass a file pointer or pass stdout to specify where the log output should go.
	s.setLogCycle(1, 1, stdout);


	s.setConductances(ALL, true,5,150,6,150);
	
	s.setSTDP(ALL, false);

	s.setSTP(ALL,false);

	s.setSpikeMonitor(v1Cells[RED_GREEN],"results/colorblind/spkV1RG.dat");
	s.setSpikeMonitor(v1Cells[GREEN_RED],"results/colorblind/spkV1GR.dat");
	s.setSpikeMonitor(v1Cells[BLUE_YELLOW],"results/colorblind/spkV1BY.dat");
	s.setSpikeMonitor(v1Cells[YELLOW_BLUE],"results/colorblind/spkV1YB.dat");

	s.setSpikeMonitor(v4CellsExc[RED_V4],"results/colorblind/spkV4R.dat");
	s.setSpikeMonitor(v4CellsExc[GREEN_V4],"results/colorblind/spkV4G.dat");
	s.setSpikeMonitor(v4CellsExc[BLUE_V4],"results/colorblind/spkV4B.dat");
	s.setSpikeMonitor(v4CellsExc[YELLOW_V4],"results/colorblind/spkV4Y.dat");
	s.setSpikeMonitor(v4CellsExc[CYAN_V4],"results/colorblind/spkV4C.dat");
	s.setSpikeMonitor(v4CellsExc[MAGENTA_V4],"results/colorblind/spkV4M.dat");

	s.setSpikeMonitor(v4CellsInh[RED_V4],"results/colorblind/spkV4Ri.dat");
	s.setSpikeMonitor(v4CellsInh[GREEN_V4],"results/colorblind/spkV4Gi.dat");
	s.setSpikeMonitor(v4CellsInh[BLUE_V4],"results/colorblind/spkV4Bi.dat");
	s.setSpikeMonitor(v4CellsInh[YELLOW_V4],"results/colorblind/spkV4Yi.dat");
	s.setSpikeMonitor(v4CellsInh[CYAN_V4],"results/colorblind/spkV4Ci.dat");
	s.setSpikeMonitor(v4CellsInh[MAGENTA_V4],"results/colorblind/spkV4Mi.dat");


	unsigned char* vid = new unsigned char[nrX*nrY*3];

	bool onGPU = true;

	if (!onGPU) {
		CUDA_CHECK_ERRORS(cudaSetDevice(CUDA_GET_MAXGFLOP_DEVICE_ID()));
	}

	//initialize the GPU/network
	s.runNetwork(0,0, onGPU?GPU_MODE:CPU_MODE);

	PoissonRate me(nrX*nrY*28*3,onGPU);
	PoissonRate red_green(nrX*nrY,onGPU);
	PoissonRate green_red(nrX*nrY,onGPU);
	PoissonRate yellow_blue(nrX*nrY,onGPU);
	PoissonRate blue_yellow(nrX*nrY,onGPU);
	#define VIDLEN 12

	for(long long i=0; i < VIDLEN*1; i++) {
		if (i%VIDLEN==0) fid = fopen("videos/colorblind.dat","rb");
		fread(vid,1,nrX*nrY*3,fid);

		calcColorME(nrX, nrY, vid, red_green.rates, green_red.rates, blue_yellow.rates, yellow_blue.rates, me.rates, onGPU);

		s.setSpikeRate(v1Cells[RED_GREEN], &red_green, 1);
		s.setSpikeRate(v1Cells[GREEN_RED], &green_red, 1);
		s.setSpikeRate(v1Cells[BLUE_YELLOW], &blue_yellow, 1);
		s.setSpikeRate(v1Cells[YELLOW_BLUE], &yellow_blue, 1);

		// run the established network for 1 (sec)  and 0 (millisecond), in GPU_MODE
		s.runNetwork(0,FRAMEDURATION, onGPU?GPU_MODE:CPU_MODE);

		if (i==1) {
			FILE* nid = fopen("results/colorblind/net.dat","wb");
			s.writeNetwork(nid);
			fclose(nid);
		}
	}
	fclose(fid);
}

