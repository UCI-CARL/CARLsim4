//
// Copyright (c) 2011 Regents of the University of California. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "snn.h"
#include <sys/stat.h>
#include <errno.h>

// stim must be a file of unsigned char in RGB, arranged as R1 G1 B1 R2 G2 B2 ...
void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow,
					float* yellow_blue, float* ME, bool GPUpointers);
extern MTRand getRand;


/// **************************************************************************************************************** ///
/// specify hyper-parameters
/// **************************************************************************************************************** ///

// the dimension of the input stimulus
#define nrX (32)
#define nrY (32)

// set for debug modus (will print actual connections made, etc.)
#define DEBUG (0)

#define poolCDSsz 3
float poolCDSfilt[poolCDSsz] = {0.7866,0.1065,0.0003};

#define poolPDSsz 10
float poolPDSfilt[poolPDSsz] = {0.1330,0.1258,0.1065,0.0807,0.0547,0.0332,0.0180,0.0087,0.0038,0.0015};



/// **************************************************************************************************************** ///
/// helper functions
/// **************************************************************************************************************** ///

// required for M_PI etc.
#define _USE_MATH_DEFINES

// sum of an array
template <typename T> T sum (const T *array, const unsigned n) {
    T accum = 0;
    for (unsigned i=0; i<n; i++)
        accum += array[i];
    return accum;
}

// product of an array
template <typename T> T prod (const T *array, const unsigned n) {
    T accum = 1;
    for (unsigned i=0; i<n; i++)
        accum *= array[i];
    return accum;
}


/* fast_exp(x): fast approximation of exponential distribution
 *				FAST_EXP_BUF_SIZE determines resolution of exp values
 *				FAST_EXP_MAX_STD value where fast_exp decays to 0
 *  x			fast_exp(x)
 * -0.000000 	1.000000
 * -0.100000 	0.818731
 * -0.200000 	0.670320
 * -0.300000 	0.548812
 * -0.400000 	0.449329
 * -0.500000 	0.367879
 * -0.600000 	0.301194
 * -0.700000 	0.246597
 * -0.800000 	0.201897
 * -0.900000 	0.165299
 * -1.000000 	0.000000
 */
#define FAST_EXP_BUF_SIZE 100
#define FAST_EXP_MAX_STD 1.0
float fast_exp_buf[FAST_EXP_BUF_SIZE] = {-1};
float fast_exp(float x) {
	if (fast_exp_buf[0] == -1) {
		for (int i=0;i<FAST_EXP_BUF_SIZE;i++)
			fast_exp_buf[i] = expf(-2.0*i*FAST_EXP_MAX_STD/FAST_EXP_BUF_SIZE);
	}	
	x = -x;
	return (x<FAST_EXP_MAX_STD)?fast_exp_buf[(int)(x/FAST_EXP_MAX_STD*FAST_EXP_BUF_SIZE)]:0;
}



/// **************************************************************************************************************** ///
/// ME projections
/// **************************************************************************************************************** ///


// 8 directions in 45deg increments (right, up-right, up, up-left, left, down-left, down, down-right)
// speed: 1.5 px / frame
float motionProj1[28][8] = {{ 0.022767, 0.133918,-0.019089, 0.023261, 0.117327, 0.191374, 0.278932,-0.145564},
							{-0.015832, 0.121715,-0.068499, 0.030544, 0.154770, 0.155971, 0.131154, 0.012329},
							{ 0.022053,-0.015930, 0.032040,-0.062458, 0.142408,-0.005933,-0.015348, 0.021949},
							{ 0.028526, 0.016924,-0.007294, 0.076223,-0.054996, 0.122826, 0.138724,-0.054547},
							{-0.038627, 0.142911,-0.065407, 0.113648, 0.018516, 0.204728, 0.246668,-0.068293},
							{ 0.015789,-0.040322,-0.045610, 0.044447,-0.098194,-0.113810,-0.088908,-0.008092},
							{-0.067669, 0.000465, 0.012076,-0.060441, 0.009823,-0.101542,-0.168729, 0.134523},
							{-0.045801,-0.064207, 0.070138,-0.058892,-0.020514,-0.055737,-0.108304, 0.100377},
							{ 0.054840,-0.092988, 0.032218,-0.034154,-0.005643,-0.116633,-0.141287, 0.004476},
							{ 0.079610,-0.173034, 0.036507,-0.026012,-0.080840,-0.141932,-0.149201,-0.007094},
							{-0.002760,-0.063476, 0.055657,-0.009239,-0.155815,-0.058488,-0.055848, 0.010228},
							{-0.042512, 0.013043,-0.015809,-0.048879,-0.027626,-0.070454,-0.071868, 0.032086},
							{-0.036402,-0.038521, 0.001206,-0.027450,-0.084032,-0.148966,-0.134906,-0.010160},
							{ 0.135353,-0.139635, 0.083199,-0.008804,-0.096384,-0.057582, 0.007292,-0.140935},
							{-0.027877,-0.154818, 0.010138, 0.033938,-0.198714,-0.148831,-0.350906, 0.208434},
							{-0.103417, 0.315892,-0.173888, 0.097633, 0.083156, 0.152226, 0.165264,-0.001054},
							{ 0.082706,-0.195796, 0.177637,-0.163242, 0.048142,-0.141593,-0.178753, 0.049655},
							{-0.012464, 0.072712,-0.143530, 0.219182,-0.129355, 0.062050, 0.068393,-0.044208},
							{ 0.075080, 0.058068,-0.023023, 0.066687, 0.010372, 0.232446, 0.444257,-0.257191},
							{-0.005135, 0.099539, 0.053135,-0.138693, 0.392758, 0.193018, 0.195633,-0.050703},
							{-0.124533, 0.093023,-0.022271,-0.051476, 0.114320,-0.003362,-0.036171, 0.168819},
							{ 0.449609,-0.088170, 0.030453, 0.036227, 0.054088, 0.158473, 0.070089, 0.713758},
							{ 0.010038,-0.020296, 0.558592, 0.703615,-0.013115, 0.143155, 0.163689,-0.047228},
							{-0.127947, 0.604710, 0.642658,-0.097680,-0.076346,-0.020829,-0.010490, 0.005232},
							{ 0.726212, 0.500602,-0.139541,-0.017232, 0.032586,-0.084820,-0.104418,-0.031422},
							{ 0.004648,-0.106625,-0.092245, 0.450655, 0.687086,-0.258392,-0.271798, 0.108523},
							{-0.072646,-0.082013, 0.014441, 0.010235,-0.176118,-0.184076, 0.655413, 0.487448},
							{ 0.016393, 0.102307, 0.006109,-0.101644, 0.352339, 1.096713, 0.321430,-0.191344},
};

// 8 directions in 45deg increments (right, up-right, up, up-left, left, down-left, down, down-right)
// speed: 0.125 px / frame
float motionProj2[28][8] = {{ 0.003640, 0.933010, 0.388890, 0.197562, 0.084291, 1.245238, 0.321257, 0.301306},
							{ 1.065132, 0.179712, 0.210607, 0.209520, 0.656550, 0.178338, 0.224549, 0.282544},
							{-0.072333,-0.003296,-0.019682, 0.965563,-0.161379,-0.019895, 0.114500, 0.543728},
							{ 0.186242, 0.045697, 0.098575, 0.403170, 0.233578, 0.105243,-0.013517, 0.894730},
							{ 0.116420, 0.125421, 1.264815, 0.162434, 0.152664, 0.133810, 0.774508, 0.306905},
							{ 0.204125,-0.374294,-0.211568,-0.050406, 0.497778,-0.236053,-0.146894,-0.146810},
							{ 0.050835,-0.227469,-0.063493,-0.052891, 0.041496,-0.018163, 0.209719,-0.060457},
							{ 0.101298, 0.314486,-0.050000,-0.104841,-0.110479, 0.006605,-0.055793,-0.073382},
							{-0.165903,-0.066405, 0.106123,-0.157636,-0.232452,-0.126424, 0.590964,-0.241091},
							{-0.115457, 0.396518,-0.430683,-0.142583,-0.151111, 0.026130,-0.440278,-0.290727},
							{ 0.016502,-0.028555,-0.060128,-0.241319, 0.410043,-0.027084,-0.174658,-0.229034},
							{-0.026115, 0.011962, 0.046626, 0.051008, 0.023170,-0.016961,-0.144792,-0.204227},
							{-0.360202,-0.103151,-0.159345,-0.188471,-0.378063,-0.155427,-0.121011, 0.019412},
							{-0.111288,-0.249642,-0.120743,-0.067637,-0.207498,-0.092784,-0.160119,-0.245147},
							{-0.154268,-0.702222,-0.420057,-0.264413,-0.346866,-0.487925,-0.489101,-0.362018},
							{ 0.063441, 0.173577, 0.323178, 0.201441, 0.236601, 0.096039, 0.374005, 0.320932},
							{-0.066108,-0.262287,-0.443572,-0.149622,-0.139227,-0.154347,-0.415389,-0.108173},
							{ 0.072335,-0.034118, 0.010746,-0.155265,-0.017252, 0.014960, 0.030234,-0.195081},
							{ 0.102592, 0.328904, 0.172354, 0.199116, 0.208057, 0.186771, 0.137320, 0.245871},
							{-0.025451, 0.465996, 0.341556, 0.235990, 0.049754, 0.270328, 0.399241, 0.357376},
							{ 0.082266, 0.012787,-0.073076,-0.108275, 0.090986, 0.037951,-0.109994,-0.162016},
							{ 0.155732, 0.240473, 0.276125, 0.188415, 0.236508, 0.149633, 0.307171, 0.151194},
							{ 0.013812, 0.368624, 0.322702, 0.197739, 0.190912, 0.186632, 0.364775, 0.335970},
							{ 0.029251,-0.129720,-0.015200,-0.056951,-0.092842,-0.039413,-0.049185,-0.233354},
							{-0.061683,-0.016128,-0.173241,-0.095164,-0.047177,-0.039825,-0.176810, 0.032133},
							{-0.069734,-0.455673,-0.374674,-0.154333,-0.201482,-0.272269,-0.441561,-0.250118},
							{-0.183000,-0.344820,-0.226003,-0.125039,-0.303116,-0.219826,-0.242264,-0.126068},
							{ 0.147916, 0.400611, 0.279166, 0.102892, 0.276558, 0.268718, 0.333123, 0.135605},
};

// 8 directions in 45deg increments (right, up-right, up, up-left, left, down-left, down, down-right)
// speed: 9 px / frame
float motionProj3[28][8] = {{-4.829119,-4.831345,-4.787727,-4.705154,-4.610980,-4.567922,-4.627049,-4.751024},
							{-3.392298,-3.452437,-3.515044,-3.524595,-3.481753,-3.427692,-3.384343,-3.368358},
							{-0.699835,-0.677115,-0.635369,-0.596646,-0.592321,-0.637687,-0.686657,-0.703392},
							{-1.921730,-1.981686,-2.019515,-2.021574,-1.985321,-1.911846,-1.857618,-1.869012},
							{-3.943419,-3.890022,-3.841720,-3.825041,-3.835737,-3.852578,-3.890386,-3.942991},
							{ 1.984432, 1.999067, 2.046136, 2.099241, 2.112984, 2.080808, 2.031675, 1.996412},
							{ 1.665604, 1.616381, 1.593096, 1.606199, 1.646718, 1.684816, 1.709473, 1.707957},
							{ 1.820706, 1.820244, 1.757613, 1.661462, 1.591134, 1.586342, 1.653112, 1.754558},
							{ 2.047726, 2.011076, 1.981621, 1.971064, 1.976766, 1.991144, 2.019252, 2.051737},
							{ 3.455032, 3.487651, 3.465047, 3.393645, 3.304997, 3.250801, 3.272536, 3.364911},
							{ 1.777413, 1.812416, 1.844123, 1.837802, 1.801891, 1.774594, 1.761144, 1.759893},
							{ 1.624518, 1.678171, 1.701922, 1.683976, 1.638995, 1.588573, 1.562264, 1.577172},
							{ 3.174255, 3.200677, 3.197059, 3.165751, 3.128862, 3.106939, 3.109058, 3.135893},
							{ 0.921904, 1.002909, 1.135064, 1.216250, 1.217478, 1.162704, 1.061461, 0.952972},
							{ 5.575307, 5.579709, 5.521145, 5.398089, 5.243886, 5.175735, 5.276667, 5.464643},
							{-4.020423,-4.081317,-4.020573,-3.869434,-3.727133,-3.668267,-3.721078,-3.868836},
							{ 2.306505, 2.164379, 2.075782, 2.102967, 2.203843, 2.293563, 2.355760, 2.379644},
							{-0.656539,-0.742457,-0.850178,-0.901092,-0.851379,-0.732007,-0.638395,-0.619717},
							{-3.745841,-3.662498,-3.613674,-3.611078,-3.651567,-3.712135,-3.777342,-3.803653},
							{-4.273420,-4.307900,-4.379971,-4.441871,-4.424243,-4.348406,-4.282880,-4.263578},
							{-0.659069,-0.550851,-0.472650,-0.443654,-0.472785,-0.575314,-0.688332,-0.722381},
							{-2.974909,-3.074511,-3.116085,-3.101876,-3.061307,-3.004600,-2.941517,-2.912327},
							{-2.918789,-2.803728,-2.660725,-2.627063,-2.719522,-2.822560,-2.897712,-2.943275},
							{ 1.245166, 1.413557, 1.440300, 1.299646, 1.145624, 1.065563, 1.052906, 1.105395},
							{ 2.125134, 2.068515, 1.877911, 1.731502, 1.677265, 1.694269, 1.798597, 1.984265},
							{ 4.473787, 4.528705, 4.640234, 4.757053, 4.730868, 4.561108, 4.440899, 4.438622},
							{ 4.241033, 4.083689, 3.978842, 3.932503, 3.967739, 4.124463, 4.320540, 4.367061},
							{-3.403131,-3.411277,-3.342663,-3.188073,-2.975002,-2.880409,-3.032034,-3.272588},
};


/* cutoff 0.1
float motionProj1[28][8] = {{-0.000000, 0.098204, -0.054803, -0.000000, 0.081613, 0.155659, 0.243218, -0.181278},
{-0.051547, 0.086001, -0.104213, -0.000000, 0.119056, 0.120256, 0.095439, -0.000000},
{-0.000000, -0.051644, -0.000000, -0.098173, 0.106693, -0.041648, -0.051062, -0.000000},
{-0.000000, -0.000000, -0.043009, 0.000000, -0.090711, 0.000000, 0.103010, -0.090262},
{-0.074341, 0.107197, -0.101121, 0.077934, -0.000000, 0.169014, 0.210953, -0.104007},
{-0.019925, -0.076036, -0.081324, 0.000000, -0.133908, -0.149524, -0.124623, -0.043806},
{-0.103383, -0.035249, -0.023638, -0.096156, -0.025891, -0.137256, -0.204444, 0.098809},
{-0.081515, -0.099921, 0.000000, -0.094606, -0.056228, -0.091452, -0.144019, 0.000000},
{0.000000, -0.128702, -0.000000, -0.069869, -0.041357, -0.152347, -0.177001, -0.031239},
{0.000000, -0.208748, 0.000000, -0.061726, -0.116555, -0.177646, -0.184915, -0.042809},
{-0.038474, -0.099190, 0.000000, -0.044953, -0.191529, -0.094202, -0.091563, -0.000000},
{-0.078227, -0.000000, -0.051523, -0.084593, -0.063341, -0.106169, -0.107583, -0.000000},
{-0.072116, -0.074235, -0.034509, -0.063164, -0.119746, -0.184680, -0.170620, -0.045874},
{0.099639, -0.175349, 0.000000, -0.044518, -0.132098, -0.093296, -0.000000, -0.176650},
{-0.063591, -0.190533, -0.025576, -0.000000, -0.234428, -0.184545, -0.386621, 0.172719},
{-0.139132, 0.280178, -0.209602, 0.000000, 0.000000, 0.116512, 0.129550, -0.036768},
{0.000000, -0.231510, 0.141923, -0.198956, 0.000000, -0.177308, -0.214468, 0.000000},
{-0.048179, 0.000000, -0.179244, 0.183468, -0.165069, 0.000000, 0.000000, -0.079922},
{0.000000, 0.000000, -0.058737, 0.000000, -0.025343, 0.196732, 0.408543, -0.292905},
{-0.040849, 0.063824, 0.000000, -0.174407, 0.357043, 0.157304, 0.159919, -0.086417},
{-0.160247, 0.057308, -0.057985, -0.087190, 0.078606, -0.039077, -0.071885, 0.133105},
{0.413895, -0.123885, -0.000000, 0.000000, 0.000000, 0.122759, 0.000000, 0.678043},
{-0.025676, -0.056010, 0.522878, 0.667901, -0.048829, 0.107441, 0.127975, -0.082942},
{-0.163661, 0.568995, 0.606944, -0.133394, -0.112060, -0.056543, -0.046204, -0.030482},
{0.690498, 0.464887, -0.175255, -0.052946, -0.000000, -0.120535, -0.140133, -0.067136},
{-0.031067, -0.142339, -0.127959, 0.414941, 0.651372, -0.294106, -0.307513, 0.072809},
{-0.108360, -0.117727, -0.021273, -0.025480, -0.211832, -0.219791, 0.619699, 0.451733},
{-0.019321, 0.066593, -0.029605, -0.137358, 0.316625, 1.060998, 0.285716, -0.227058},
};
float motionProj2[28][8] = {{-0.000000, 0.897296, 0.353176, 0.161847, 0.000000, 1.209524, 0.285543, 0.265591},
{1.029417, 0.143998, 0.174893, 0.173806, 0.620836, 0.142623, 0.188835, 0.246830},
{-0.108047, -0.000000, -0.055396, 0.929848, -0.197093, -0.055609, 0.078786, 0.508013},
{0.150528, 0.000000, 0.000000, 0.367456, 0.197863, 0.000000, -0.000000, 0.859015},
{0.000000, 0.000000, 1.229100, 0.126720, 0.116950, 0.000000, 0.738794, 0.271190},
{0.168411, -0.410008, -0.247282, -0.086121, 0.462063, -0.271767, -0.182609, -0.182525},
{0.000000, -0.263183, -0.099207, -0.088605, 0.000000, -0.053877, 0.174004, -0.096171},
{0.000000, 0.278772, -0.085714, -0.140555, -0.146193, -0.000000, -0.091507, -0.109096},
{-0.201618, -0.102119, 0.000000, -0.193351, -0.268166, -0.162138, 0.555250, -0.276805},
{-0.151171, 0.360803, -0.466397, -0.178297, -0.186825, -0.000000, -0.475992, -0.326441},
{-0.000000, -0.000000, -0.095843, -0.277033, 0.374329, -0.062798, -0.210372, -0.264749},
{-0.061829, -0.000000, 0.000000, 0.000000, -0.000000, -0.052675, -0.180506, -0.239941},
{-0.395916, -0.138865, -0.195059, -0.224185, -0.413778, -0.191141, -0.156726, -0.000000},
{-0.147002, -0.285356, -0.156458, -0.103351, -0.243213, -0.128499, -0.195833, -0.280861},
{-0.189982, -0.737936, -0.455772, -0.300128, -0.382581, -0.523640, -0.524815, -0.397732},
{0.000000, 0.137863, 0.287464, 0.165727, 0.200886, 0.000000, 0.338290, 0.285218},
{-0.101822, -0.298001, -0.479286, -0.185336, -0.174942, -0.190061, -0.451103, -0.143887},
{0.000000, -0.000000, -0.000000, -0.190980, -0.052967, -0.000000, -0.000000, -0.230796},
{0.000000, 0.293190, 0.136640, 0.163402, 0.172343, 0.151056, 0.101605, 0.210156},
{-0.061165, 0.430281, 0.305841, 0.200276, 0.000000, 0.234614, 0.363526, 0.321661},
{0.000000, -0.000000, -0.108791, -0.143990, 0.000000, 0.000000, -0.145709, -0.197730},
{0.120017, 0.204758, 0.240411, 0.152700, 0.200794, 0.000000, 0.271457, 0.115479},
{-0.000000, 0.332910, 0.286988, 0.162025, 0.155198, 0.150918, 0.329061, 0.300256},
{-0.000000, -0.165435, -0.050914, -0.092666, -0.128557, -0.075127, -0.084900, -0.269069},
{-0.097398, -0.000000, -0.208955, -0.130879, -0.082892, -0.075539, -0.212524, -0.000000},
{-0.105448, -0.491387, -0.410388, -0.190047, -0.237196, -0.307983, -0.477275, -0.285832},
{-0.218714, -0.380534, -0.261717, -0.160753, -0.338830, -0.255540, -0.277978, -0.161782},
{0.112201, 0.364896, 0.243452, 0.000000, 0.240844, 0.233003, 0.297409, 0.099891},
};
float motionProj3[28][8] = {{-4.864834, -4.867060, -4.823441, -4.740868, -4.646694, -4.603636, -4.662763, -4.786739},
{-3.428012, -3.488151, -3.550758, -3.560310, -3.517467, -3.463406, -3.420058, -3.404072},
{-0.735549, -0.712829, -0.671083, -0.632360, -0.628035, -0.673401, -0.722371, -0.739107},
{-1.957444, -2.017401, -2.055229, -2.057289, -2.021035, -1.947560, -1.893333, -1.904727},
{-3.979133, -3.925736, -3.877434, -3.860755, -3.871451, -3.888292, -3.926100, -3.978706},
{1.948717, 1.963352, 2.010421, 2.063527, 2.077270, 2.045093, 1.995961, 1.960698},
{1.629890, 1.580667, 1.557382, 1.570485, 1.611004, 1.649102, 1.673758, 1.672242},
{1.784991, 1.784529, 1.721898, 1.625747, 1.555419, 1.550628, 1.617398, 1.718844},
{2.012012, 1.975361, 1.945907, 1.935350, 1.941052, 1.955430, 1.983538, 2.016023},
{3.419318, 3.451937, 3.429333, 3.357931, 3.269283, 3.215087, 3.236822, 3.329197},
{1.741699, 1.776702, 1.808409, 1.802087, 1.766176, 1.738879, 1.725429, 1.724178},
{1.588804, 1.642456, 1.666208, 1.648262, 1.603281, 1.552858, 1.526549, 1.541457},
{3.138541, 3.164963, 3.161345, 3.130037, 3.093148, 3.071225, 3.073344, 3.100179},
{0.886189, 0.967194, 1.099349, 1.180536, 1.181763, 1.126990, 1.025747, 0.917258},
{5.539593, 5.543994, 5.485430, 5.362374, 5.208172, 5.140021, 5.240953, 5.428928},
{-4.056137, -4.117032, -4.056287, -3.905148, -3.762848, -3.703982, -3.756792, -3.904550},
{2.270790, 2.128664, 2.040068, 2.067253, 2.168129, 2.257849, 2.320046, 2.343929},
{-0.692253, -0.778171, -0.885892, -0.936806, -0.887094, -0.767721, -0.674110, -0.655431},
{-3.781555, -3.698213, -3.649388, -3.646792, -3.687281, -3.747849, -3.813056, -3.839367},
{-4.309134, -4.343614, -4.415685, -4.477585, -4.459957, -4.384120, -4.318594, -4.299292},
{-0.694784, -0.586566, -0.508364, -0.479368, -0.508499, -0.611029, -0.724046, -0.758095},
{-3.010623, -3.110226, -3.151800, -3.137590, -3.097021, -3.040314, -2.977231, -2.948042},
{-2.954503, -2.839443, -2.696440, -2.662777, -2.755236, -2.858274, -2.933426, -2.978990},
{1.209452, 1.377843, 1.404586, 1.263931, 1.109909, 1.029849, 1.017192, 1.069681},
{2.089420, 2.032800, 1.842197, 1.695787, 1.641550, 1.658555, 1.762882, 1.948551},
{4.438072, 4.492991, 4.604519, 4.721339, 4.695153, 4.525393, 4.405185, 4.402908},
{4.205318, 4.047975, 3.943128, 3.896789, 3.932025, 4.088749, 4.284825, 4.331347},
{-3.438845, -3.446991, -3.378377, -3.223788, -3.010716, -2.916123, -3.067748, -3.308302},
};
*/



/// **************************************************************************************************************** ///
/// connection functions
/// **************************************************************************************************************** ///

// Connections from V1 to MT are governed by the 28x8 motionProj{1-3}, which can be either positive or negative.
// Positive connections are fed directly to MT{1-3}. Negative connections are fed to MT{1-3}inh, which then feed to
// MT{1-3}. The firing strength of MT is then given by the weighted sum of these inputs, which is the spiking equivalent
// to the Matlab function shModelMTLinear() of the S&H model.
class connectV1toMT: public ConnectionGenerator {
public:
	connectV1toMT(float weightScale, int standDev, float (*proj)[8], bool usePosWts) {
		this->weightScale = weightScale;
		this->standDev = standDev;
		this->proj = proj;
		this->usePosWts = usePosWts;
	}

	float weightScale;
	int standDev;
	float (*proj)[8];
	bool usePosWts;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		// pre (V1): get x,y coordinates, index of the space-time filter, and scale (0-2)
		int v1X = i%nrX;
		int v1Y = (i/nrX)%nrY;
		int spaceTimeInd = (i/(nrX*nrY))%28;
		int scale = i/(nrX*nrY)/28;

		// post (MT): get x,y coordinates, direction (8 in 45deg increments)
		int mtX = (j%nrX);
		int mtY = ((j/nrX)%nrY);
		int dir = j/(nrX*nrY);

		// The 1-D Gaussian filter is given by mtPoolFilt: It is pars.mtSpatialPoolingFilter in the S&H model, and
		// normpdf(-9:9,0,3) in Matlab. mtPoolFilt stores only the last 9 elements (starting at the peak). So if mtX
		// and v1X are i pixel away from each other, the Gaussian contribution factor is mtPoolFilt[i]. Apply in X and Y
		// to generate a 2-D Gaussian.
		int gaussX = abs(round(mtX-v1X));
		int gaussY = abs(round(mtY-v1Y));
//		float gauss = ((gaussX<mtPoolSz)?mtPoolFilt[gaussX]:0.0f) * ((gaussY<mtPoolSz)?mtPoolFilt[gaussY]:0.0f);
//		bool connectGauss = mtX==v1X && mtY==v1Y; // (gauss>0.001f*mtPoolFilt[0]);

		float gauss = ((gaussX<poolCDSsz)?poolCDSfilt[gaussX]:0.0f) * ((gaussY<poolCDSsz)?poolCDSfilt[gaussY]:0.0f);
		bool connectGauss = v1X==mtX && v1Y==mtY;
		
		if (usePosWts) {
			// use only the positive weights in motionProj[]
			connected = connectGauss && (proj[spaceTimeInd][dir]>0);
			weight = proj[spaceTimeInd][dir]*weightScale;
		}
		else {
			// use only the negative weights in motionProj[], note the -
			connected = connectGauss && (proj[spaceTimeInd][dir]<0);
			weight = -proj[spaceTimeInd][dir]*weightScale;
		}
	
		delay = 1;
	}
};


class connectMTtoMTpatt: public ConnectionGenerator {
public:
	connectMTtoMTpatt(float weightScale, int standDev, bool usePosWts) {
		this->weightScale = weightScale;
		this->standDev = standDev;
		this->usePosWts = usePosWts;
	}

	float weightScale;
	int standDev;
	bool usePosWts;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int preX = i%(nrX);
		int preY = (i/nrX)%(nrY);
		int prePool = i/(nrX*nrY); // direction given by groups 0-7

		int postX = j%(nrX);
		int postY = (j/nrX)%(nrY);
		int postPool = j/(nrX*nrY);

		// fast_exp(x) is 1 at x=0, and decays such that at x=-1 it is 0
//		float gausPos = fast_exp(-((postX-preX)*(postX-preX)+(postY-preY)*(postY-preY))/(2*standDev*standDev));
		int gaussX = abs(round(postX-preX));
		int gaussY = abs(round(postY-preY));

		float gausPos = ((gaussX<poolPDSsz)?poolPDSfilt[gaussX]:0.0f) * ((gaussY<poolPDSsz)?poolPDSfilt[gaussY]:0.0f);
		gausPos /= poolPDSfilt[0];
//		float gausPos = (gauss>0.01*poolPDSfilt[0])?gauss:0.0f;


		int diffPool = abs(prePool-postPool);
		if (diffPool>4)
			diffPool = 8-diffPool;
		float gausDir = cos(diffPool/8.0*2.0*M_PI);

		if (usePosWts) {
			connected = (gausPos>0.1) && (gausDir>0.1);
			weight = weightScale*gausPos*gausDir;
		}
		else {
			connected = (gausPos>0.1) && ((-gausDir)>0.1);
			weight = weightScale*gausPos*(-gausDir);

		}

		delay = 1;
	}
};


// FIXME: this is just one-to-one
class connectMTtoMTpattTunedNorm: public ConnectionGenerator {
public:
	connectMTtoMTpattTunedNorm(float weightScale, int standDev) {
		this->weightScale = weightScale;
		this->standDev = standDev;
	}

	float weightScale;
	int standDev;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int preX = i%(nrX);
		int preY = (i/nrX)%(nrY);
		int prePool = i/(nrX*nrY); // direction given by groups 0-7

		int postX = j%(nrX);
		int postY = (j/nrX)%(nrY);
		int postPool = j/(nrX*nrY);

		// fast_exp(x) is 1 at x=0, and decays such that at x=-1 it is 0
		float gausPos = fast_exp(-((postX-preX)*(postX-preX)+(postY-preY)*(postY-preY))/(2*standDev*standDev));

		int diffPool = abs(prePool-postPool);
		if (diffPool>4)
			diffPool = 8-diffPool;
		float gausDir = (diffPool<4) ? cos(diffPool/4.0*2.0*M_PI) : 0.0f;

		connected = (gausPos>0.1) && (gausDir>0.1);
		weight = weightScale*gausPos*gausDir;
		delay = 1;
	}
};



class connectGauss: public ConnectionGenerator {
public:
	connectGauss(int standDev, float weightScale, bool stayWithinPool) {
		this->standDev = standDev;
		this->weightScale = weightScale;
		this->stayWithinPool = stayWithinPool;
	}

	int standDev;
	float weightScale;
	bool stayWithinPool;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		int preX = i%(nrX);
		int preY = (i/nrX)%(nrY);
		int prePool = i/(nrX*nrY); // direction given by groups 0-7

		int postX = j%(nrX);
		int postY = (j/nrX)%(nrY);
		int postPool = j/(nrX*nrY);

		// fast_exp(x) is 1 at x=0, and decays such that at x=-1 it is 0
		float gaus = fast_exp(-((postX-preX)*(postX-preX)+(postY-preY)*(postY-preY))/(2*standDev*standDev));

		// either stayWithinPool: connect only if directions of pre and post are the same AND if gaussian
		// or !stayWithinPool: connect if gaussian
		connected = (stayWithinPool && prePool==postPool || !stayWithinPool) && 2.0*getRand()<gaus;
		weight = weightScale*gaus;
		delay = 1;
	}
};



/* connectOneToOne()
 * Connects the i-th cell in pre to the i-th cell in post.
 *
 * Input:
 * - preDim:	     {nrXpre,nrYpre}
 * - postDim:	     {nrXpost,nrYpost}
 * - weight:	     the weight value
 * - stayWithinPool: if set to true, do not connect different pools
 * Output:		     none
 * Usage:		     s.connect(gPre, gPost, new connectOneToOne(preDim,postDim,weight,stayWithinPool),SYN_FIXED);
 *
 * Pre and post populations do not need to have the same size. The function will compute the row- and column-indices
 * of each cell with respect to population size, and convert post-coordinates into pre-coordinates.
 * If the number of neurons in pre exceeds preSize, the first preSize neurons will belong to pool 0, the next to
 * pool 1, etc. If stayWithinPool==true, then neurons of pool 0 will NOT be connected to pool>0. Same goes for post.
 */
class connectOneToOne: public ConnectionGenerator {
public:
	connectOneToOne(int preDim[3], int postDim[3], float weight, bool stayWithinPool) {
		nrXpre  = preDim[0];
		nrYpre  = preDim[1];
		nrXpost = postDim[0];
		nrYpost = postDim[1];
		this->weight = weight;
		this->stayWithinPool = stayWithinPool;
	}

	int nrXpre;
	int nrYpre;
	int nrXpost;
	int nrYpost;
	float weight;
	bool stayWithinPool;
	
	void connect(CpuSNN* net, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt, float& delay, bool& connected)
	{
		// wrap around
		int preX     = i%nrXpre;
		int preY     = (i/nrXpre)%nrYpre;
		int prePool  = i/(nrXpre*nrYpre);

		int postX   = ((j%nrXpost)*nrXpre)/nrXpost;
		int postY   = (((j/nrYpost)%nrYpost)*nrYpre)/nrYpost;
		int postPool = j/(nrXpost*nrYpost);

		// if stayWithinPool is set: only connect if pre and post are in the same pool
		connected = preX==postX && preY==postY && (stayWithinPool && prePool==postPool || !stayWithinPool);
		weight = this->weight;
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

		connected = getRand()<cos((MTdir-PFCdir)/8.0*2*3.1459)*0.05;
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

		connected = getRand()<cos((PFCdir-PFCidir+4)/8.0*2*3.1459)*0.05;
		weight = weightScale;
		delay = 1;
	}
};


/// **************************************************************************************************************** ///
/// main simulation
/// **************************************************************************************************************** ///

int main()
{
	// -------------------------------------------------------------------------------------------------------------- //
	// set simulation params
	// -------------------------------------------------------------------------------------------------------------- //
	MTRand	      getRand(210499257);
	time_t		  timer_start,timer_build,timer_end;
	
	time(&timer_start);
	
	// expected format of video: R1 G1 B1 R2 G2 B2 ... e[0,255]
//	char loadVideo[]	 = "videos/mkBarSpeed_ctrst0.2_32x32x7520.dat";
//	char loadVideo[] 	 = "videos/mkGratingPlaid_ctrst0.3_32x32x2400.dat";
//	char loadVideo[]	 = "videos/mkGratingContrast_32x32x1000.dat";
	char loadVideo[]	 = "videos/mkRDK_32x32x1600.dat";
	int startAtFrame	 = 0; 						// at which frame of movie to start
	char saveFolder[]	 = "Results/MTpatternTest/"; // where to store all files (folder will be created if not exists)
	bool storeNetwork	 = false;					// store network? at beginning and end
	bool onGPU = true;					// run on GPU?
	int ithGPU 			 = 1;						// on which GPU to run (in case of carlculator: 0-3)
	int frameDur 		 = 50;						// present each frame for .. ms
	int presentEachFrame = 1;					 	// present each frame .. times (> 1 to slow down motion patterns)
	int vidLen 			 = 1600;				 	// number of frames
	float synScale		 = 0.01;				 	// some scaling factor for syn weights

	unsigned char* vid = new unsigned char[nrX*nrY*3]; // pointer to read out video
	char thisTmpSave[128]; // temp var to store save folder
	FILE* nid; // fp for network file

	// try to open video first: in case of an error, we don't have to wait until the network is allocated
	FILE* fid = fopen(loadVideo,"rb");
	if (fid==NULL) {
		printf("ERROR %d: could not open video file: %s\n",errno,strerror(errno));
		return 1;
	}

	// same goes for results folder, try to create
	// read/write/search permissions for owner and group, and with read/search permissions for others
	int status = mkdir(saveFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status==-1 && errno!=EEXIST) {
		printf("ERROR %d: could not create directory: %s\n",errno, strerror(errno));
		return 1;
	}


	// -------------------------------------------------------------------------------------------------------------- //
	// create network
	// -------------------------------------------------------------------------------------------------------------- //

	CpuSNN s("Motion Energy");

	// population sizes: {number of rows, number of columns, number of pools}
	int V1MEdim[3]		= {nrX,nrY,28*3};	// 28 space-time filters at 3 scales
	int MTdim[3]		= {nrX,nrY,8}; 		// goes for MT1, MT2, and MT3
	int MTiDim[3]		= {nrX,nrY,8};		// 8 directions
	int MTnormDim[3]	= {nrX,nrY,1};		// only 1 pool
	int PFCdim[3]		= {40,1,8};
	int PFCiDim[3]		= {40,1,8};			// for lateral inhibition


	int gV1ME      = s.createSpikeGeneratorGroup("V1ME", prod(V1MEdim,3), EXCITATORY_NEURON);
	int gMT1CDS    = s.createGroup("MT1CDS", prod(MTdim,3), EXCITATORY_NEURON);
	int gMT2CDS    = s.createGroup("MT2CDS", prod(MTdim,3), EXCITATORY_NEURON);
	int gMT3CDS    = s.createGroup("MT3CDS", prod(MTdim,3), EXCITATORY_NEURON);
	int gMT1CDSinh = s.createGroup("MT1CDSi", prod(MTiDim,3), INHIBITORY_NEURON);
	int gMT2CDSinh = s.createGroup("MT2CDSi", prod(MTiDim,3), INHIBITORY_NEURON);
	int gMT3CDSinh = s.createGroup("MT3CDSi", prod(MTiDim,3), INHIBITORY_NEURON);
	int gMTCDSnorm = s.createGroup("MTCDSnorm", prod(MTnormDim,3), INHIBITORY_NEURON);
	int gMT1PDS    = s.createGroup("MT1PDS", prod(MTdim,3), EXCITATORY_NEURON);
	int gMT1PDSinh = s.createGroup("MT1PDSinh", prod(MTiDim,3), INHIBITORY_NEURON);

	int gPFC = s.createGroup("PFC", prod(PFCdim,3), EXCITATORY_NEURON);
	int gPFCi = s.createGroup("PFCi", prod(PFCiDim,3), INHIBITORY_NEURON);


	s.setNeuronParameters(gMT1CDS, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT2CDS, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT3CDS, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT1CDSinh, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT2CDSinh, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT3CDSinh, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMTCDSnorm, 0.1f, 0.2f, -65.0f, 2.0f);
	s.setNeuronParameters(gMT1PDS, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gMT1PDSinh, 0.1f, 0.2f, -65.0f, 2.0f);

	s.setNeuronParameters(gPFC, 0.02f, 0.2f, -65.0f, 8.0f);
	s.setNeuronParameters(gPFCi, 0.1f,  0.2f, -65.0f, 2.0f);


	// show log every 1 sec (0 to disable logging). You can pass a file pointer or pass stdout to specify where the
	// log output should go.
	s.setLogCycle(1, 1, stdout);
	s.setConductances(ALL,true,5,150,6,150);	
	s.setSTDP(ALL,false);
	s.setSTP(ALL,false);


	// -------------------------------------------------------------------------------------------------------------- //
	// set up connections
	// -------------------------------------------------------------------------------------------------------------- //

	// the motion projections give the strength and probability of connection from the 28 filters onto a MT neuron
	// connect (i) positive values to MT{1-3}, and (ii) negative ones to MT{1-3}inh (although with a positive sign)

	// In the S&H model, preThresholdBlur is set, and 'mtlin' responses are blurred with pars.mtSpatialPoolingFilter,
	// which is equivalent to g=normpdf(-9:9,3); in Matlab. We model this by v1toMTstd
	// (i) positive weights
	float wt_V1_MT1 = synScale*7.0;
	float wt_V1_MT2 = synScale*8.0;
	float wt_V1_MT3 = synScale*4.0;
	int v1toMTstd = 3; // g=normpdf(-3:3,0,0.75);g/trapz(-3:3,g)
	s.connect(gV1ME, gMT1CDS, new connectV1toMT(wt_V1_MT1,v1toMTstd,motionProj1,true), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT2CDS, new connectV1toMT(wt_V1_MT2,v1toMTstd,motionProj2,true), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT3CDS, new connectV1toMT(wt_V1_MT3,v1toMTstd,motionProj3,true), SYN_FIXED,1000,3000);

	// (ii) negative weights
	float wt_V1_MT1inh = synScale*7.0;
	float wt_V1_MT2inh = synScale*8.0;
	float wt_V1_MT3inh = synScale*4.0;
	s.connect(gV1ME, gMT1CDSinh, new connectV1toMT(wt_V1_MT1inh,v1toMTstd,motionProj1,false), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT2CDSinh, new connectV1toMT(wt_V1_MT2inh,v1toMTstd,motionProj2,false), SYN_FIXED,1000,3000);
	s.connect(gV1ME, gMT3CDSinh, new connectV1toMT(wt_V1_MT3inh,v1toMTstd,motionProj3,false), SYN_FIXED,1000,3000);
	float wt_MTi_MT1 = -synScale*15;
	float wt_MTi_MT2 = -synScale*15;
	float wt_MTi_MT3 = -synScale*15;
	s.connect(gMT1CDSinh, gMT1CDS, "one-to-one", wt_MTi_MT1, wt_MTi_MT1, 1.0, 1, 1, SYN_FIXED);
	s.connect(gMT2CDSinh, gMT2CDS, "one-to-one", wt_MTi_MT2, wt_MTi_MT2, 1.0, 1, 1, SYN_FIXED);
	s.connect(gMT3CDSinh, gMT3CDS, "one-to-one", wt_MTi_MT3, wt_MTi_MT3, 1.0, 1, 1, SYN_FIXED);

	// MT normalization
	// In the S&H model, neuron activity is normalized by the activity of ALL MT neurons. We normalize in a large
	// Gaussian spatial neighborhood (wMTtoMTnormRadius) instead, but still take into account the activity of all
	// 3 MT pools within that neighborhood.
	float wt_MT_MTnorm = synScale*0.12;
	int wMTtoMTnormRadius = 3;
	bool stayWithinPool = false; // collapse 8 directions onto 1 pool
	s.connect(gMT1CDS, gMTCDSnorm, new connectGauss(wMTtoMTnormRadius,wt_MT_MTnorm,stayWithinPool),SYN_FIXED,1000,3000);
	s.connect(gMT2CDS, gMTCDSnorm, new connectGauss(wMTtoMTnormRadius,wt_MT_MTnorm,stayWithinPool),SYN_FIXED,1000,3000);
	s.connect(gMT3CDS, gMTCDSnorm, new connectGauss(wMTtoMTnormRadius,wt_MT_MTnorm,stayWithinPool),SYN_FIXED,1000,3000);
	float wt_MTnorm_MT = -synScale*1;
	stayWithinPool = false;
	s.connect(gMTCDSnorm, gMT1CDS, new connectOneToOne(MTnormDim,MTdim,wt_MTnorm_MT,stayWithinPool),SYN_FIXED,1000,3000);
	s.connect(gMTCDSnorm, gMT2CDS, new connectOneToOne(MTnormDim,MTdim,wt_MTnorm_MT,stayWithinPool),SYN_FIXED,1000,3000);
	s.connect(gMTCDSnorm, gMT3CDS, new connectOneToOne(MTnormDim,MTdim,wt_MTnorm_MT,stayWithinPool),SYN_FIXED,1000,3000);

	// MT PDS cells are given by pooling of MT CDS cell activity and tuned normalization
	// weight strength plotted against direction is Gaussian, with the strongest positive weights coming from cells that
	// have a similar direction preference, and the strongest negative weights coming from cells with opposite direction
	// preference
	float wt_MT_MTpatt = synScale*15.0;
	int MTtoMTpattStd = 4;
	s.connect(gMT1CDS, gMT1PDS, new connectMTtoMTpatt(wt_MT_MTpatt,MTtoMTpattStd,true), SYN_FIXED,1000,3000);

	// negative weights of direction pooling
	float wt_MT_MTpattInh = synScale*5.0;
	s.connect(gMT1CDS, gMT1PDSinh, new connectMTtoMTpatt(wt_MT_MTpattInh,MTtoMTpattStd,false), SYN_FIXED,1000,3000);

	// tuned normalization
	float wt_MT_MTpattInh_tunedNorm = synScale*5.0;
	s.connect(gMT1CDS, gMT1PDSinh, "one-to-one", wt_MT_MTpattInh_tunedNorm, wt_MT_MTpattInh_tunedNorm, 1.0, 1, 1, SYN_FIXED);
	float wt_MTpattInh_MTpatt = -synScale*15.0;
	s.connect(gMT1PDSinh, gMT1PDS, "one-to-one", wt_MTpattInh_MTpatt, wt_MTpattInh_MTpatt, 1.0, 1, 1, SYN_FIXED);

	s.connect(gMT1PDS, gPFC, new connectMTtoPFC(40,synScale*1.0), SYN_FIXED, 1000, 3000);
	s.connect(gMT1PDS, gPFCi, new connectMTtoPFC(10,synScale*0.5), SYN_FIXED, 1000, 3000);
	s.connect(gPFCi, gPFC, new connectPFCitoPFC(40,10,-synScale*1.0), SYN_FIXED, 1000, 3000);







	// -------------------------------------------------------------------------------------------------------------- //
	// write to file params.txt
	// -------------------------------------------------------------------------------------------------------------- //

	// write to file at beginning rather than end of experiment: this way we immediately now what's going on
	strcpy(thisTmpSave,saveFolder);
	FILE* fParamId = fopen(strcat(thisTmpSave,"params.txt"),"w");
	fprintf(fParamId, "Version d1 2\n");

	fprintf(fParamId, "remoteFolder s1 spnetME/%s\n",saveFolder);
	fprintf(fParamId, "populations s12 V1ME MT1CDS MT2CDS MT3CDS MT1CDSi MT2CDSi MT3CDSi MTCDSnorm MT1PDS MT1PDSinh PFC PFCi\n");
	fprintf(fParamId, "popSize d12 %d %d %d %d %d %d %d %d %d %d %d %d\n",prod(V1MEdim,3),prod(MTdim,3),prod(MTdim,3),prod(MTdim,3),
								prod(MTiDim,3),prod(MTiDim,3),prod(MTiDim,3),prod(MTnormDim,3),prod(MTdim,3),prod(MTiDim,3),prod(PFCdim,3),prod(PFCiDim,3));
	fprintf(fParamId, "popNrX d12 %d %d %d %d %d %d %d %d %d %d %d %d\n",V1MEdim[0],MTdim[0],MTdim[0],MTdim[0],
								MTiDim[0],MTiDim[0],MTiDim[0],MTnormDim[0],MTdim[0],MTiDim[0],PFCdim[0],PFCiDim[0]);
	fprintf(fParamId, "popNrY d12 %d %d %d %d %d %d %d %d %d %d %d %d\n",V1MEdim[1],MTdim[1],MTdim[1],MTdim[1],
								MTiDim[1],MTiDim[1],MTiDim[1],MTnormDim[1],MTdim[1],MTiDim[1],PFCdim[1],PFCiDim[1]);
	fprintf(fParamId, "loadVideo s1 %s\n",loadVideo);
	fprintf(fParamId, "synScale d1 %f\n",synScale);
	fprintf(fParamId, "simFrameDur d1 %d\n",frameDur);
	fprintf(fParamId, "presentEachFrame d1 %d\n",presentEachFrame);
	fprintf(fParamId, "startAtFrame d1 %d\n",startAtFrame);
	fprintf(fParamId, "vidLen d1 %d\n",vidLen);
	fprintf(fParamId, "onGPU s1 %s\n",(onGPU?"on":"off"));
	fprintf(fParamId, "nrX d1 %d\n",nrX);
	fprintf(fParamId, "nrY d1 %d\n",nrY);
	fclose(fParamId);


	// -------------------------------------------------------------------------------------------------------------- //
	// set all spike monitors
	// -------------------------------------------------------------------------------------------------------------- //

	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gV1ME,strcat(thisTmpSave,"spkV1ME.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT1CDS, strcat(thisTmpSave,"spkMT1CDS.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT2CDS, strcat(thisTmpSave,"spkMT2CDS.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT3CDS, strcat(thisTmpSave,"spkMT3CDS.dat"));
//	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT1CDSinh, strcat(thisTmpSave,"spkMT1CDSi.dat"));
//	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT2CDSinh, strcat(thisTmpSave,"spkMT2CDSi.dat"));
//	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT3CDSinh, strcat(thisTmpSave,"spkMT3CDSi.dat"));
//	s.setSpikeMonitor(gMT1CDSinh);
//	s.setSpikeMonitor(gMT2CDSinh);
//	s.setSpikeMonitor(gMT3CDSinh);
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMTCDSnorm, strcat(thisTmpSave,"spkMTCDSnorm.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT1PDS, strcat(thisTmpSave,"spkMT1PDS.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gMT1PDSinh, strcat(thisTmpSave,"spkMT1PDSinh.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gPFC, strcat(thisTmpSave,"spkPFC.dat"));
	strcpy(thisTmpSave,saveFolder); s.setSpikeMonitor(gPFCi, strcat(thisTmpSave,"spkPFCi.dat"));


	// -------------------------------------------------------------------------------------------------------------- //
	// run network
	// -------------------------------------------------------------------------------------------------------------- //


	// initialize the GPU/network, run on device with index ithGPU
	s.runNetwork(0,0, onGPU?GPU_MODE:CPU_MODE, ithGPU);
	
	time(&timer_build);
	

	PoissonRate me(nrX*nrY*28*3,onGPU);
	PoissonRate red_green(nrX*nrY,onGPU);
	PoissonRate green_red(nrX*nrY,onGPU);
	PoissonRate yellow_blue(nrX*nrY,onGPU);
	PoissonRate blue_yellow(nrX*nrY,onGPU);

	// store network for loading
	if (storeNetwork) {
		strcpy(thisTmpSave,saveFolder);
		nid = fopen(strcat(thisTmpSave,"netA.dat"),"wb");
		s.writeNetwork(nid);
		fclose(nid);
	}

	// movie can be offset by so many frames
	if (startAtFrame>0)
		fseek(fid,startAtFrame*nrX*nrY*3,SEEK_SET);

	for(long long i=0; i < vidLen*1; i++) {
		fread(vid,1,nrX*nrY*3,fid);

		for (int j=1;j<=presentEachFrame;j++) {
			// run motion energy model and assign spike rates
			calcColorME(nrX, nrY, vid, red_green.rates, green_red.rates, blue_yellow.rates, yellow_blue.rates, me.rates, onGPU);
			s.setSpikeRate(gV1ME, &me, 1);

			// run the established network for 1 frame
			s.runNetwork(0,frameDur, onGPU?GPU_MODE:CPU_MODE);
		}
	}

	// store network if bool is set
	if (storeNetwork) {
		strcpy(thisTmpSave,saveFolder);
		nid = fopen(strcat(thisTmpSave,"netZ.dat"),"wb");
		s.writeNetwork(nid);
		fclose(nid);
	}

	fclose(fid); // close input video file
	printf("DONE %s\n",saveFolder);
	
	time(&timer_end);
	
	printf("Time to build: %.f seconds\n",difftime(timer_build,timer_start));
	printf("Time to run: %.f seconds\n",difftime(timer_end,timer_build));
}
