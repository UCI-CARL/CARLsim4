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
void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers);
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


// these are the projections of all 28 space-time filters on to 8 different directions.  In order words, the connection strength and/or probabilty from a V1 cell to an MT cell should be proportional to these values.
float motion_proj1[28][8] = {{0.000000, 0.259878, 0.935900, 1.187461, 0.867200, 0.162722, 0.000000, 0.000000},
			{0.110700, 0.000000, 0.000000, 0.000000, 0.314500, 0.971821, 1.184400, 0.827712},
			{0.845000, 1.192282, 0.970600, 0.309813, 0.000000, 0.000000, 0.000000, 0.132187},
			{0.000000, 0.000000, 0.000000, 0.365193, 1.006300, 1.190669, 0.810300, 0.088007},
			{1.201900, 0.860752, 0.151700, 0.000000, 0.000000, 0.000000, 0.313700, 0.975304},
			{0.004700, 0.739855, 1.179800, 1.066821, 0.467100, 0.000000, 0.000000, 0.000000},
			{0.000000, 0.000000, 0.565400, 1.118660, 1.160200, 0.665687, 0.000000, 0.000000},
			{0.673600, 0.000000, 0.000000, 0.000000, 0.000000, 0.571452, 1.124100, 1.166411},
			{0.000000, 0.000000, 0.097400, 0.818143, 1.210000, 1.043427, 0.416000, 0.000000},
			{1.076800, 0.473765, 0.000000, 0.000000, 0.000000, 0.044835, 0.773500, 1.200954},
			{0.517000, 1.104656, 1.207600, 0.765528, 0.037400, 0.000000, 0.000000, 0.000000},
			{1.171200, 1.177407, 0.663900, 0.000000, 0.000000, 0.000000, 0.000000, 0.648915},
			{0.000000, 0.000000, 0.000000, 0.143771, 0.841400, 1.279526, 1.201500, 0.653029},
			{0.638700, 0.046883, 0.000000, 0.000000, 0.521100, 1.112917, 1.392500, 1.196073},
			{0.052900, 0.635141, 1.191700, 1.396553, 1.129700, 0.547459, 0.000000, 0.000000},
			{1.157900, 0.586853, 0.019700, 0.000000, 0.029100, 0.600147, 1.167300, 1.398329},
			{1.402200, 1.171603, 0.611500, 0.049991, 0.000000, 0.046597, 0.606700, 1.168209},
			{1.130200, 1.401666, 1.212900, 0.674478, 0.101800, 0.000000, 0.019100, 0.557522},
			{0.000000, 0.182583, 0.767700, 1.268699, 1.392100, 1.065617, 0.480500, 0.000000},
			{0.560600, 1.128697, 1.403200, 1.223308, 0.694400, 0.126303, 0.000000, 0.031692},
			{0.000000, 0.000000, 0.227900, 0.816815, 1.295300, 1.383066, 1.028700, 0.439785},
			{0.539900, 0.259168, 0.303800, 0.647652, 1.089300, 1.370032, 1.325400, 0.981548},
			{1.334200, 1.354695, 1.068300, 0.642780, 0.327400, 0.306905, 0.593300, 1.018820},
			{1.338700, 1.033975, 0.612400, 0.320928, 0.330300, 0.635025, 1.056600, 1.348072},
			{1.001100, 0.591370, 0.329100, 0.367924, 0.685100, 1.094830, 1.357100, 1.318276},
			{0.977300, 1.305558, 1.363500, 1.117185, 0.710900, 0.382642, 0.324700, 0.571015},
			{0.336300, 0.425885, 0.765200, 1.155478, 1.368100, 1.278515, 0.939200, 0.548922},
			{0.497600, 0.872935, 1.236300, 1.374840, 1.207400, 0.832065, 0.468700, 0.330160}};

float motion_proj2[28][8] = {{0.000000, 0.471178, 1.147200, 1.398761, 1.078500, 0.374022, 0.000000, 0.000000},
			{0.323300, 0.000000, 0.000000, 0.000000, 0.527100, 1.184421, 1.397000, 1.040312},
			{1.066000, 1.413282, 1.191600, 0.530813, 0.000000, 0.000000, 0.000000, 0.353187},
			{0.000000, 0.000000, 0.000000, 0.591793, 1.232900, 1.417269, 1.036900, 0.314607},
			{1.434600, 1.093452, 0.384400, 0.000000, 0.000000, 0.000000, 0.546400, 1.208004},
			{0.240600, 0.975755, 1.415700, 1.302721, 0.703000, 0.000000, 0.000000, 0.000000},
			{0.000000, 0.069613, 0.810500, 1.363760, 1.405300, 0.910787, 0.169900, 0.000000},
			{0.926500, 0.187248, 0.000000, 0.000000, 0.085100, 0.824352, 1.377000, 1.419311},
			{0.000000, 0.000000, 0.354100, 1.074843, 1.466700, 1.300127, 0.672700, 0.000000},
			{1.336100, 0.733065, 0.004400, 0.000000, 0.000000, 0.304135, 1.032800, 1.460254},
			{0.794200, 1.381856, 1.484800, 1.042728, 0.314600, 0.000000, 0.000000, 0.066072},
			{1.461400, 1.467607, 0.954100, 0.221685, 0.000000, 0.000000, 0.206700, 0.939115},
			{0.353800, 0.000000, 0.000000, 0.542171, 1.239800, 1.677926, 1.599900, 1.051429},
			{1.218600, 0.626783, 0.347200, 0.543627, 1.101000, 1.692817, 1.972400, 1.775973},
			{0.644200, 1.226441, 1.783000, 1.987853, 1.721000, 1.138759, 0.582200, 0.377347},
			{1.751400, 1.180353, 0.613200, 0.382171, 0.622600, 1.193647, 1.760800, 1.991829},
			{2.011300, 1.780703, 1.220600, 0.659091, 0.425100, 0.655697, 1.215800, 1.777309},
			{1.746200, 2.017666, 1.828900, 1.290478, 0.717800, 0.446334, 0.635100, 1.173522},
			{0.480200, 0.806683, 1.391800, 1.892799, 2.016200, 1.689717, 1.104600, 0.603601},
			{1.188100, 1.756197, 2.030700, 1.850808, 1.321900, 0.753803, 0.479300, 0.659192},
			{0.589600, 0.501834, 0.856200, 1.445115, 1.923600, 2.011366, 1.657000, 1.068085},
			{1.354500, 1.073768, 1.118400, 1.462252, 1.903900, 2.184632, 2.140000, 1.796148},
			{2.165000, 2.185495, 1.899100, 1.473580, 1.158200, 1.137705, 1.424100, 1.849620},
			{2.173200, 1.868475, 1.446900, 1.155428, 1.164800, 1.469525, 1.891100, 2.182572},
			{1.844200, 1.434470, 1.172200, 1.211024, 1.528200, 1.937930, 2.200200, 2.161376},
			{1.821400, 2.149658, 2.207600, 1.961285, 1.555000, 1.226742, 1.168800, 1.415115},
			{1.188500, 1.278085, 1.617400, 2.007678, 2.220300, 2.130715, 1.791400, 1.401122},
			{1.350100, 1.725435, 2.088800, 2.227340, 2.059900, 1.684565, 1.321200, 1.182660}};

float motion_proj3[28][8] = {{0.189300, 0.893778, 1.569800, 1.821361, 1.501100, 0.796622, 0.120600, 0.000000},
			{0.748500, 0.091179, 0.000000, 0.235288, 0.952300, 1.609621, 1.822200, 1.465512},
			{1.508000, 1.855282, 1.633600, 0.972813, 0.260000, 0.000000, 0.134400, 0.795187},
			{0.126700, 0.000000, 0.322700, 1.044993, 1.686100, 1.870469, 1.490100, 0.767807},
			{1.900000, 1.558852, 0.849800, 0.188196, 0.000000, 0.302748, 1.011800, 1.673404},
			{0.712400, 1.447555, 1.887500, 1.774521, 1.174800, 0.439645, 0.000000, 0.112679},
			{0.065300, 0.559813, 1.300700, 1.853960, 1.895500, 1.400987, 0.660100, 0.106840},
			{1.432300, 0.693048, 0.140400, 0.098089, 0.590900, 1.330152, 1.882800, 1.925111},
			{0.073500, 0.240073, 0.867500, 1.588243, 1.980100, 1.813527, 1.186100, 0.465357},
			{1.854700, 1.251665, 0.523000, 0.095546, 0.219700, 0.822735, 1.551400, 1.978854},
			{1.348600, 1.936256, 2.039200, 1.597128, 0.869000, 0.281344, 0.178400, 0.620472},
			{2.041800, 2.048007, 1.534500, 0.802085, 0.279800, 0.273593, 0.787100, 1.519515},
			{1.150600, 0.712474, 0.790500, 1.338971, 2.036600, 2.474726, 2.396700, 1.848229},
			{2.378400, 1.786583, 1.507000, 1.703427, 2.260800, 2.852617, 3.132200, 2.935773},
			{1.826800, 2.409041, 2.965600, 3.170453, 2.903600, 2.321359, 1.764800, 1.559947},
			{2.938400, 2.367353, 1.800200, 1.569171, 1.809600, 2.380647, 2.947800, 3.178829},
			{3.229500, 2.998903, 2.438800, 1.877291, 1.643300, 1.873897, 2.434000, 2.995509},
			{2.978200, 3.249666, 3.060900, 2.522478, 1.949800, 1.678334, 1.867100, 2.405522},
			{1.728400, 2.054883, 2.640000, 3.140999, 3.264400, 2.937917, 2.352800, 1.851801},
			{2.443100, 3.011197, 3.285700, 3.105808, 2.576900, 2.008803, 1.734300, 1.914192},
			{1.846200, 1.758434, 2.112800, 2.701715, 3.180200, 3.267966, 2.913600, 2.324685},
			{2.983700, 2.702968, 2.747600, 3.091452, 3.533100, 3.813832, 3.769200, 3.425348},
			{3.826600, 3.847095, 3.560700, 3.135180, 2.819800, 2.799305, 3.085700, 3.511220},
			{3.842200, 3.537475, 3.115900, 2.824428, 2.833800, 3.138525, 3.560100, 3.851572},
			{3.530400, 3.120670, 2.858400, 2.897224, 3.214400, 3.624130, 3.886400, 3.847576},
			{3.509600, 3.837858, 3.895800, 3.649485, 3.243200, 2.914942, 2.857000, 3.103315},
			{2.892900, 2.982485, 3.321800, 3.712078, 3.924700, 3.835115, 3.495800, 3.105522},
			{3.055100, 3.430435, 3.793800, 3.932340, 3.764900, 3.389565, 3.026200, 2.887660}};

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


enum color_name_t { BLUE=0, GREEN, RED, YELLOW};
string imageName[] = { "blue", "green", "red", "yellow"};

enum v1color_name_t    { RED_GREEN=0, BLUE_YELLOW, GREEN_RED, YELLOW_BLUE };
string v1ImageName[] = { "red-green-cells", "blue-yellow-cells",
			 "green-red-cells", "yellow-blue-cells"};

string  v4CellNameExc[] = { "Ev4magenta", "Ev4blue", "Ev4cyan", "Ev4green", "Ev4yellow", "Ev4red"};
string  v4CellNameInh[] = { "Iv4magenta", "Iv4blue", "Iv4cyan", "Iv4green", "Iv4yellow", "Iv4red"};

enum  v4CellType_t  {MAGENTA_V4=0, BLUE_V4, CYAN_V4, GREEN_V4, YELLOW_V4, RED_V4};

class connectV1toMT: public ConnectionGenerator {
public:
	connectV1toMT(int scale, float weightScale, float (*proj)[8]) {
		spatialScale = scale;
		this->weightScale = weightScale;
		this->proj = proj;
	}

	int spatialScale;
	float weightScale;
	float (*proj)[8];	
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int v1X = i%nrX;
		int v1Y = (i/nrX)%nrY;
		int spaceTimeInd = (i/(nrX*nrY))%28;
		int scale = i/(nrX*nrY)/28;

		int mtX = (j%(nrX/spatialScale))*spatialScale;
		int mtY = ((j/(nrX/spatialScale))%(nrY/spatialScale))*spatialScale;
		int dir = j/(nrX*nrY/spatialScale/spatialScale);

		float gaus = fast_exp(-((mtX-v1X)*(mtX-v1X)+(mtY-v1Y)*(mtY-v1Y))/MTsize/2);

		connected = getRand()<gaus*proj[spaceTimeInd][dir]/1.4032;
		weight = proj[spaceTimeInd][dir]/1.4032*gaus * weightScale;
		delay = 1;
	}
};

class connectMTitoMT: public ConnectionGenerator {
public:
	connectMTitoMT(int scale, float weightScale) {
		spatialScale = scale;
		this->weightScale = weightScale;
	}

	int spatialScale;
	float weightScale;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int X = j%nrX;
		int Y = (j/nrX)%nrY;
		int dir = j/(nrX*nrY);

		int iX = (i%(nrX/spatialScale))*spatialScale;
		int iY = ((i/(nrX/spatialScale))%(nrY/spatialScale))*spatialScale;
		int iDir = i/(nrX*nrY/spatialScale/spatialScale);

		float gaus = fast_exp(-((X-iX)*(X-iX)+(Y-iY)*(Y-iY))/MTsize)/sqrt(2*3.1459*MTsize*2); //for Inhibition use twice the radius...

		connected = getRand()<gaus*cos((dir-iDir+4)/8.0*2*3.1459);
		weight = cos((dir-iDir+4)/8.0*2*3.1459)*weightScale;
		delay = 1;
	}
};

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
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
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
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
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

class connectMTtoPFC: public ConnectionGenerator {
public:
	connectMTtoPFC(int num, float weightScale) {
		this->num = num;
		this->weightScale = weightScale;
	}

	int num;
	float weightScale;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int MTdir = i/(nrX*nrY);
		int PFCdir = j/num;

		connected = getRand()<cos((MTdir-PFCdir)/8.0*2*3.1459)*0.2;
		weight = cos((MTdir-PFCdir)/8.0*2*3.1459)*weightScale;
		delay = 1;
	}
};

class connectPFCitoPFC: public ConnectionGenerator {
public:
	connectPFCitoPFC(int num, int numi, float weightScale) {
		this->num = num;
		this->numi = numi;
		this->weightScale = weightScale;
	}

	int num, numi;
	float weightScale;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int PFCidir = i/numi;
		int PFCdir = j/num;

		connected = getRand()<cos((PFCdir-PFCidir+4)/8.0*2*3.1459);
		weight = weightScale;
		delay = 1;
	}
};

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

		connected   = gaus>0.1;
		delay       = 1.0;
		weight  = gaus*weightScale;
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

	#define FRAMEDURATION 100

	FILE* fid;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
//	CUDA_CHECK_ERRORS(cudaSetDevice(cutGetMaxGflopsDeviceId()));

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



	int gV1ME = s.createSpikeGeneratorGroup("V1ME", nrX*nrY*28*3, EXCITATORY_NEURON);
	int gMT1 = s.createGroup("MT1", nrX*nrY*8, EXCITATORY_NEURON);
	int gMT2 = s.createGroup("MT2", nrX*nrY*8, EXCITATORY_NEURON);
	int gMT3 = s.createGroup("MT3", nrX*nrY*8, EXCITATORY_NEURON);
	s.setNeuronParameters(gMT1, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT2, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT3, 0.02f, 0.2f, -65.0f, 8.0f);
	int gMT1i = s.createGroup("MT1i", nrX*nrY*8/4, INHIBITORY_NEURON);
	int gMT2i = s.createGroup("MT2i", nrX*nrY*8/4, INHIBITORY_NEURON);
	int gMT3i = s.createGroup("MT3i", nrX*nrY*8/4, INHIBITORY_NEURON);
	s.setNeuronParameters(gMT1i, 0.1f,  0.2f, -65.0f, 2.0f);
	s.setNeuronParameters(gMT2i, 0.1f,  0.2f, -65.0f, 2.0f);
	s.setNeuronParameters(gMT3i, 0.1f,  0.2f, -65.0f, 2.0f);

	int gPFC = s.createGroup("PFC", 50*8, EXCITATORY_NEURON);
	s.setNeuronParameters(gPFC, 0.02f, 0.2f, -65.0f, 8.0f);
	int gPFCi = s.createGroup("PFCi", 10*8, INHIBITORY_NEURON);
	s.setNeuronParameters(gPFCi, 0.1f,  0.2f, -65.0f, 2.0f);



	s.connect(gV1ME, gMT1, new connectV1toMT(1,synscale*4.5/2,motion_proj1), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT2, new connectV1toMT(1,synscale*4.5/2,motion_proj2), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT3, new connectV1toMT(1,synscale*4.5/2,motion_proj3), SYN_FIXED,1000,3000);

	s.connect(gV1ME, gMT1i, new connectV1toMT(2,synscale*3/2,motion_proj1), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT2i, new connectV1toMT(2,synscale*3/2,motion_proj2), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT3i, new connectV1toMT(2,synscale*3/2,motion_proj3), SYN_FIXED,1000,3000);

	s.connect(gMT1i, gMT1, new connectMTitoMT(2,-synscale*20), SYN_FIXED,1000,3000);
	s.connect(gMT2i, gMT2, new connectMTitoMT(2,-synscale*20), SYN_FIXED,1000,3000);
	s.connect(gMT3i, gMT3, new connectMTitoMT(2,-synscale*20), SYN_FIXED,1000,3000);
	

	s.connect(gMT1, gPFC, new connectMTtoPFC(50,synscale*0.8), SYN_FIXED);
	s.connect(gMT1, gPFCi, new connectMTtoPFC(10,synscale*0.5), SYN_FIXED);

	s.connect(gPFCi, gPFC, new connectPFCitoPFC(50,10,-synscale*1), SYN_FIXED);



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


	// show log every 1 sec (0 to disable logging). You can pass a file pointer or pass stdout to specify where the log output should go.
	s.setLogCycle(1, 1, stdout);


	s.setConductances(ALL, true,5,150,6,150);
	
	s.setSTDP(ALL, false);

	s.setSTP(ALL,false);

	s.setSpikeMonitor(v1Cells[RED_GREEN],"results/v1v4PFC/V1RG.dat");
	s.setSpikeMonitor(v1Cells[GREEN_RED],"results/v1v4PFC/V1GR.dat");
	s.setSpikeMonitor(v1Cells[BLUE_YELLOW],"results/v1v4PFC/V1BY.dat");
	s.setSpikeMonitor(v1Cells[YELLOW_BLUE],"results/v1v4PFC/V1YB.dat");

	s.setSpikeMonitor(v4CellsExc[RED_V4],"results/v1v4PFC/V4R.dat");
	s.setSpikeMonitor(v4CellsExc[GREEN_V4],"results/v1v4PFC/V4G.dat");
	s.setSpikeMonitor(v4CellsExc[BLUE_V4],"results/v1v4PFC/V4B.dat");
	s.setSpikeMonitor(v4CellsExc[YELLOW_V4],"results/v1v4PFC/V4Y.dat");
	s.setSpikeMonitor(v4CellsExc[CYAN_V4],"results/v1v4PFC/V4C.dat");
	s.setSpikeMonitor(v4CellsExc[MAGENTA_V4],"results/v1v4PFC/V4M.dat");

	s.setSpikeMonitor(v4CellsInh[RED_V4],"results/v1v4PFC/V4Ri.dat");
	s.setSpikeMonitor(v4CellsInh[GREEN_V4],"results/v1v4PFC/V4Gi.dat");
	s.setSpikeMonitor(v4CellsInh[BLUE_V4],"results/v1v4PFC/V4Bi.dat");
	s.setSpikeMonitor(v4CellsInh[YELLOW_V4],"results/v1v4PFC/V4Yi.dat");
	s.setSpikeMonitor(v4CellsInh[CYAN_V4],"results/v1v4PFC/V4Ci.dat");
	s.setSpikeMonitor(v4CellsInh[MAGENTA_V4],"results/v1v4PFC/V4Mi.dat");

	s.setSpikeMonitor(gV1ME);
	s.setSpikeMonitor(gMT1,"results/v1v4PFC/MT1.dat");
	s.setSpikeMonitor(gMT2,"results/v1v4PFC/MT2.dat");
	s.setSpikeMonitor(gMT3,"results/v1v4PFC/MT3.dat");
	s.setSpikeMonitor(gPFC,"results/v1v4PFC/PFC.dat");
	s.setSpikeMonitor(gMT1i,"results/v1v4PFC/MT1i.dat");
	s.setSpikeMonitor(gPFCi,"results/v1v4PFC/PFCi.dat");

	s.setSpikeMonitor(gV4o,"results/v1v4PFC/V4o.dat");
	s.setSpikeMonitor(gV4oi,"results/v1v4PFC/V4oi.dat");


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

	#define VIDLEN (256*3)
//	#define VIDLEN (8*33*10)

	for(long long i=0; i < VIDLEN*1; i++) {
		if (i%VIDLEN==0) fid = fopen("videos/colorcycle.dat","rb");
//		if (i%VIDLEN==0) fid = fopen("videos/rdk3.dat","rb");
		fread(vid,1,nrX*nrY*3,fid);

		calcColorME(nrX, nrY, vid, red_green.rates, green_red.rates, blue_yellow.rates, yellow_blue.rates, me.rates, onGPU);

		s.setSpikeRate(gV1ME, &me, 1);
		s.setSpikeRate(v1Cells[RED_GREEN], &red_green, 1);
		s.setSpikeRate(v1Cells[GREEN_RED], &green_red, 1);
		s.setSpikeRate(v1Cells[BLUE_YELLOW], &blue_yellow, 1);
		s.setSpikeRate(v1Cells[YELLOW_BLUE], &yellow_blue, 1);

		// run the established network for 1 (sec)  and 0 (millisecond), in GPU_MODE
		s.runNetwork(0,FRAMEDURATION, onGPU?GPU_MODE:CPU_MODE);

		if (i==1) {
			FILE* nid = fopen("results/v1v4PFC/net.dat","wb");
			s.writeNetwork(nid);
			fclose(nid);
		}
	}
	fclose(fid);
}

