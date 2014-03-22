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
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 07/13/2013
 */ 

/*
 * Large-scale simulation of cortical visual processing.
 * This code implements both (i) the color opponency model (De Valoisetal.,1958; LivingstoneandHubel,1984) as well as
 * (ii) the motion energy model (Simoncelli & Heeger, 1998). The original version of this code has been released as
 * part of the publication
 * Richert, M., Nageswaran, J.M., Dutt, N., and Krichmar, J.L. (2011). "An efficient simulation environment for modeling
 * large-scale cortical processing". Frontiers in Neuroinformatics 5, 1-15.
 *
 * Creator: Micah Richert
 * Curator: Michael Beyeler
 *
 * Ver 02/15/2013
 */



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cufft.h>

#if __CUDA3__
    #include <cutil_inline.h>
#elif __CUDA5__
    #include <helper_cuda.h>
#endif

#include <cuda_version_control.h>

#define IMUL(a, b) __mul24(a, b)

// In order to run the color opponency / motion energy model within your simulation, you need to declare the function
// in your "main_[].cpp" à la:
// void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers);
//	input arguments:
//		nrX:	input dimension, x-coordinate
//		nrY:	input dimension, y-coordinate
//		stim:	RGB input stimulus, must be a nrX*nrY*3 array of unsigned chars, where the R, G, and B values of each
//				pixel is ordered as R1 G1 B1 R2 G2 B2 ...
//	output arguments (pre-allocated):
//		red_green:		red-green channel, pre-allocated array of floats with dimensionality nrX*nrY
//		green_red:		green-red channel: pre-allocated array of floats with dimensionality nrX*nrY
//		blue_yellow:	blue-yellow channel: pre-allocated array of floats with dimensionality nrX*nrY
//		yellow_blue:	yellow-blue channel: pre-allocated array of floats with dimensionality nrX*nrY
//		ME:				motion energy, V1 complex cells: pre-allocated array of floats with dimensionality nrX*nrY*28*3


// 28 space-time orientations of V1 simple cells
#define nrDirs 28

// each of these is a unit vector (e.g. d_v1Dirs[0][1..3] => (-0.6559)^2+(0.7246)^2+(0.2113)^2 == 1
// all vectors lie on the surface of a dome (hemisphere) in 3D Fourier space
__constant__  float d_v1Dirs[3][nrDirs] = {{-0.6559, -0.1019, 0.6240, -0.7797, 0.9692, -0.2312, -0.9151, 0.4207, -0.9533, 0.8175, 0.2398, 0.8810, -0.4430, 0.0588, -0.5384, 0.5644, 0.7931, 0.5142, -0.7680, -0.0669, -0.6670, -0.2747, 0.5034, 0.5042, 0.1580, 0.1332, -0.5159, -0.3549},
                                           { 0.7246, -0.9718, 0.7496, -0.5837, -0.0810, 0.9439, 0.3203, -0.8712, -0.1593, -0.5142, 0.9304, 0.3737, -0.8031, -0.8126, 0.6004, -0.5738, 0.0024, 0.5969, 0.1436, 0.7757, -0.4004, -0.5108, 0.2375, -0.2221, -0.5140, 0.5194, -0.0870, 0.3838},
                                           { 0.2113, 0.2126, 0.2210, 0.2266, 0.2327, 0.2359, 0.2451, 0.2529, 0.2567, 0.2593, 0.2772, 0.2902, 0.3984, 0.5799, 0.5913, 0.5935, 0.6091, 0.6160, 0.6241, 0.6275, 0.6283, 0.8146, 0.8308, 0.8345, 0.8431, 0.8441, 0.8522, 0.8525}};


// this filter is used for the 3 different scales in space/time:
// first scale  == original image resolution
// second scale == first scale blurred with Gaussian (width 1), so resolution should scale down sqrt(2)
// third scale  == second scale blurred with Gaussian (width 1)
// FIXME: not sure why this guy is not normalized
#define scalingFiltSize 5
__constant__ float d_scalingFilt[scalingFiltSize] = {0.0884, 0.3536, 0.5303, 0.3536, 0.0884};
float* scalingFilt;

// d_v1Gaus defines the 1D receptive field of a V1 unit, which is then used for all three dimensions (X,Y and T)
// this guy can be reproduced in matlab with g=normpdf(-4:4,0,1.25);
// it is the same as provided by the S&H matlab code
#define v1GausSize 9
__constant__ float d_v1Gaus[v1GausSize] = {0.0007, 0.0155, 0.0903, 0.2345, 0.3179, 0.2345, 0.0903, 0.0155, 0.0007};
float* v1Gaus;

// d_complexV1Filt is the spacial filter for complex cells; it averages over "simple" V1 cells
// all simple cells must have the same space-time orientation and phase
// this guy can be reproduced in matlab with g=normpdf(-5:5,0,1.6);
// it is the same as provided by the S&H matlab code
#define complexV1FiltSize 11
__constant__ float d_complexV1Filt[complexV1FiltSize] = {0.0019, 0.0110, 0.0430, 0.1142, 0.2052, 0.2495, 0.2052, 0.1142, 0.0430, 0.0110, 0.0019};
float* complexV1Filt;

// d_normV1filt is the spatial filter used complex cell normalization
// this guy can be reproduced in matlab with: g=normpdf(-10:10,0,3.35);
#define normV1filtSize 21
__constant__ float d_normV1filt[normV1filtSize] = {0.0013, 0.0031, 0.0067, 0.0132, 0.0237, 0.0389, 0.0584, 0.0800, 0.1001, 0.1146, 0.1199, 0.1146, 0.1001, 0.0800, 0.0584, 0.0389, 0.0237, 0.0132, 0.0067, 0.0031, 0.0013};
float* normV1filt;

// difference operator for taking the first-order derivative
#define diff1filtSize 3
__constant__ float d_diff1filt[diff1filtSize] = {-1/2.0, 0, 1/2.0};
float* diff1filt;

// difference operator for taking the second-order derivative
#define diff2filtSize 3
__constant__ float d_diff2filt[diff2filtSize] = {1, -2, 1};
float* diff2filt;

// difference operator for taking the third-order derivative
// the grayscale values of our input stimuli will be convolved with d_scalingFilt in 3D
#define diff3filtSize 5
__constant__ float d_diff3filt[diff3filtSize] = {-1/2.0, 1, 0, -1, 1/2.0};
float* diff3filt;

// 3 different spatio-temporal scales (these must correspond to the 3 blobs along the x-, y-, or t-axis in Fig. 3 in
// the Simoncelli & Heeger paper... because they are scales in the x,y,t-space they cannot be temporal frequency or
// "speed" alone)
#define nrScales 3

// number of time steps to be considered in computation
// nrT = 5*(3-1)-1 = 9
#define nrT (scalingFiltSize*(nrScales-1)-1)//(scalingFiltSize*nrScales-2)

int stimBufX, stimBufY;
float* d_resp; // will be the returned ME responses
float* d_stimBuf;
float* d_scalingStimBuf; // the temporary matrix that will be filtered once for each scale...
float* d_v1GausBuf;
float* d_v1GausBuf2;
float* d_diffV1GausBuf;
float* diffV1GausBufT;

unsigned char* d_stim; // the video input
float* d_pop;

// following 3 lines are needed for dev_split(), altho we only care about d_stim
float* d_red;
float* d_green;
float* d_blue;

float* d_center;
float* d_surround;
float* d_color_tmp;
float* d_color_tmp_green;
float* d_color_tmp_yellow;


#define iDivUp(a,b) ((a)+(b)-1)/(b)

// convolve idata with filt and store output in odata
# define CONV1_THREAD_SIZE 256
__global__ void dev_conv1(float* idata, float* odata, int len, const float* filt, int filtlen) {
	__shared__ float block[CONV1_THREAD_SIZE];

	const int nrValidConv = CONV1_THREAD_SIZE - (filtlen-1);
	const int offset = (filtlen-1)/2;
	
	int xInd = blockIdx.x*nrValidConv + threadIdx.x - offset;
	int idx = blockIdx.y * len + xInd;

	block[threadIdx.x] = (xInd>=0 && xInd<len)?idata[idx]:0;

	__syncthreads();
	
	xInd += offset;
	idx += offset;

	if (xInd<len && threadIdx.x < nrValidConv) {
		float sum = 0;
		for (int i = 0; i< filtlen; i++) sum += block[threadIdx.x+i]*filt[i];
		odata[idx] = sum;
	}
}


#define CONVN_THREAD_SIZE1 16
#define CONVN_THREAD_SIZE2 31 //31 is faster than 32 because shared memory is too full
__global__ void dev_convn(float* idata, float* odata, int nrX, int nrN, int stride, int blockStride, int nrBlocks, const float* filt, int filtlen) {
	__shared__ float block[CONVN_THREAD_SIZE1*CONVN_THREAD_SIZE2];

	const int nrValidConv = (CONVN_THREAD_SIZE2-(filtlen-1));
	const int offset = (filtlen-1)/2;
	
	const int blockY = blockIdx.y/nrBlocks;
	const int b = blockIdx.y - blockY*nrBlocks;

	const int ind1 = blockIdx.x*CONVN_THREAD_SIZE1 + threadIdx.x;
	int ind2 = blockY*nrValidConv + threadIdx.y - offset;
	int idx = ind2*stride + ind1 + b*blockStride;

	const int threadxy = threadIdx.x*CONVN_THREAD_SIZE2+threadIdx.y;

	block[threadxy] = (ind2>=0 && ind2<nrN && ind1 < nrX)?idata[idx]:0;

	__syncthreads();

	ind2 += offset;
	idx += offset*stride;

	if (ind2<nrN && ind1 < nrX && threadIdx.y < nrValidConv) {
		float sum = 0;
		for (int i = 0; i< filtlen; i++) sum += block[threadxy+i]*filt[i];
		odata[idx] = sum;
	}
}

// conv2D is only used in the color model
// odata must be pre-allocated
// the result will end up in idata...
// FIXME: in conv1D the logic was: perform operation on idata and output to odata
// filtlen can not be greater than CONVN_THREAD_SIZE2
void conv2D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension	
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension	
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");

	// FIXME: shouldn't there be a tmp=idata; idata=odata; odata=tmp; here???
}

// conv3D is only used in the motion model in freq space (\omega_x,\omega_y,\omega_t)
// odata must be pre-allocated
// the result will end up in idata
// filtlen can not be greater than CONVN_THREAD_SIZE2

void conv3D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension	
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");
	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension	
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the third dimension	
	dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
	dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
        CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;
}

//will free idata
// this computes the difference / approximates the derivative of idata
float* diff(float* idata, uint3 _sizes, int order, int dim)
{
	unsigned int* sizes = (unsigned int*)&_sizes;
	int filtlen;
	float* filt;
	float* odata;

	CUDA_CHECK_ERRORS(cudaMalloc((void**)&odata, sizeof(float)*sizes[0]*sizes[1]*sizes[2]));
	
	switch (order) {
		case 1:
			filtlen = diff1filtSize;
		  	filt = diff1filt;
			break;
		case 2:
			filtlen = diff2filtSize;
		  	filt = diff2filt;
			break;
		case 3:
			filtlen = diff3filtSize;
		  	filt = diff3filt;
			break;
	}

	switch (dim) {
		case 0: {
			// convolve the first dimension	
			dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
			dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
			dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_conv1() execution failed\n");
			break;
		}
		case 1: {
			// convolve the second dimension	
			dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
			dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");
			break;
		}
		case 2: {
			// convolve the third dimension	
			dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
			dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
			CUDA_GET_LAST_ERROR("dev_convn() execution failed\n");
			break;
		}
	}

	CUDA_CHECK_ERRORS(cudaFree (idata));

	return odata;
}

__global__ void dev_accumDiffStims(float *d_resp, float *diffV1GausBuf, int nrXnrY, int c, int orderX, int orderY, int orderT) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	__shared__ float dirorders[nrDirs];

	if (threadIdx.x < nrDirs) {
		const float dir1 = d_v1Dirs[0][threadIdx.x]; // x-component
		const float dir2 = d_v1Dirs[1][threadIdx.x]; // y-component
		const float dir3 = d_v1Dirs[2][threadIdx.x]; // t-component

		float dirX = (orderX==0)?1:(orderX==1)?dir1:(orderX==2)?dir1*dir1:dir1*dir1*dir1;
		float dirY = (orderY==0)?1:(orderY==1)?dir2:(orderY==2)?dir2*dir2:dir2*dir2*dir2;
		float dirT = (orderT==0)?1:(orderT==1)?dir3:(orderT==2)?dir3*dir3:dir3*dir3*dir3;
		dirorders[threadIdx.x] = dirX*dirY*dirT;
	}

	__syncthreads();

	for(int i = tid; i < nrXnrY; i += threadN)
	{
		float d = diffV1GausBuf[i];
		for (int j=0; j<nrDirs; j++)
			d_resp[i+j*nrXnrY] += c*d*dirorders[j];
	}
}

void accumDiffStims(float *d_resp, float* diffV1GausBuf, uint3 _sizes, int orderX, int orderY, int orderT) {
	// a useful list of factorials for computing the scaling factors for the derivatives
	int factorials[4] = {1, 1, 2, 6};

	// the scaling factor for this directial derivative; similar to the binomial coefficients
	int c = 6/factorials[orderX]/factorials[orderY]/factorials[orderT];

        dev_accumDiffStims<<<iDivUp(_sizes.x*_sizes.y, 128), 128>>>(d_resp, diffV1GausBuf, _sizes.x*_sizes.y, c, orderX, orderY, orderT);
        CUDA_GET_LAST_ERROR("dev_accumDiffStims() execution failed\n");
}

// parallel half-wave rectification and squaring
__global__ void dev_halfRect2(float *data, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float d = data[i];
		d = (d>0)?d:0;
		data[i] = d*d;
	}
}

// compute the mean on the array's third dimension
// this is used to compute the mean of all 28 filter responses at a given location/scale (used in the complex cell
// normalization step)
__global__ void dev_mean3(float *idata, float *odata, int nrXnrY, int nrZ) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int blockSize = nrXnrY*nrZ;

	for(int i = tid; i < nrXnrY; i += threadN) {
		float sum = 0;
		int ind = i + blockIdx.y*blockSize;
		for (int j=0; j < nrZ; j++) sum += idata[ind+j*nrXnrY];
		odata[i+blockIdx.y*nrXnrY] = sum/nrZ;
	}
}

// population normalization of complex cell responses
// note the 0.1, probably to avoid division by zero
__global__ void dev_normalize(float *resp, float *pop, int nrXnrY, int nrZ) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	const int blockSize = nrXnrY*nrZ;

	for(int i = tid; i < nrXnrY; i += threadN) {
		float norm = pop[i+blockIdx.y*nrXnrY];
		int ind = i + blockIdx.y*blockSize;
		for (int j=0; j < nrZ; j++) resp[ind+j*nrXnrY] /= (norm + 0.1);
	}
}

// reads in stimuli in RGB format and extracts R, G, B, and grayscale values (normalized to [0,1])
__global__ void dev_split(unsigned char *idata, float *red, float *green, float *blue, float *gray, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	typedef struct rgb_s {
		unsigned char r,g,b;
	} rgb_t;

	rgb_t* rgbs = (rgb_t*)idata;

	for(int i = tid; i < len; i += threadN) {
		rgb_t rgb=rgbs[i];
		float r = rgb.r/255.0;
		float g = rgb.g/255.0;
		float b = rgb.b/255.0;

		gray[i] = (r+g+b)/3;

		red[i] = r;
		green[i] = g;
		blue[i] = b;
	}
}

// parallel subtraction
__global__ void dev_sub(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] - i2data[i];
	}
}

// parallel averaging
__global__ void dev_ave(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = (i1data[i] + i2data[i])/2;
	}
}

// parallel summing
__global__ void dev_sum(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] + i2data[i];
	}
}

// parallel half-rectification at a given scale
__global__ void dev_scaleHalfRect(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float tmp = data[i];
		data[i] = (tmp>0)?sqrt(sqrt(tmp))*scale:0;
	}
}

// parallel mulitplying with a scale factor
__global__ void dev_scale(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		data[i] *= scale;
	}
}

// parallel taking the square root and multiplying with a scale factor
__global__ void dev_scaleSqrt(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float tmp = data[i];
		data[i] = (tmp>0)?sqrt(tmp)*scale:0;
	}
}

// stim should be an array of lenght nrX*nrY*3 with the data being organized such as: R1 G1 B1 R2 G2 B2 ...
// void calcColorME(int spatFreq, int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers)
void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers)
{
	// allocate memory on the GPU
	if (nrX != stimBufX || nrY != stimBufY) {
		stimBufX = nrX;
		stimBufY = nrY;

		// allocate the response matrix
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&d_resp, sizeof(float)*nrX*nrY*nrDirs*nrScales));

		// probably should free previous buffers if they were previously allocated...

		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_stimBuf, nrX*nrY*nrT*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMemset (d_stimBuf, 0, nrX*nrY*nrT*sizeof(float)));

		CUDA_CHECK_ERRORS(cudaMalloc((void**)&diffV1GausBufT, sizeof(float)*nrX*nrY*v1GausSize));
	
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_stim, nrX*nrY*3));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_scalingStimBuf, nrX*nrY*nrT*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_v1GausBuf, nrX*nrY*nrT*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_diffV1GausBuf, nrX*nrY*nrT*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_pop, nrX*sizeof(float)*nrY*nrScales)); // mean of 28 filter responses for all x,y and spatial scales, at a given step in time

		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_red, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_green, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_blue, nrX*nrY*sizeof(float)));

		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_center, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_surround, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_color_tmp, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_color_tmp_green, nrX*nrY*sizeof(float)));
		CUDA_CHECK_ERRORS(cudaMalloc ((void**)&d_color_tmp_yellow, nrX*nrY*sizeof(float)));

		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&scalingFilt, d_scalingFilt));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&v1Gaus, d_v1Gaus));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&complexV1Filt, d_complexV1Filt));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&normV1filt, d_normV1filt));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&diff1filt, d_diff1filt));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&diff2filt, d_diff2filt));
		CUDA_CHECK_ERRORS(cudaGetSymbolAddress((void**)&diff3filt, d_diff3filt));
		
	}
	// use the preexisting filters because they are about the right size and give good results
	float* center_filt = v1Gaus;
	float* surround_filt = complexV1Filt;
	const int center_filtSize = v1GausSize;
	const int surround_filtSize = complexV1FiltSize;

	CUDA_CHECK_ERRORS(cudaMemcpy(d_stim,stim,3*nrX*nrY,cudaMemcpyHostToDevice));
	dev_split<<<iDivUp(nrX*nrY,128), 128>>>(d_stim, d_red, d_green, d_blue, &d_stimBuf[nrX*nrY*(nrT-1)], nrX*nrY);
 	CUDA_GET_LAST_ERROR("dev_split() execution failed\n");


	/* ***** COLOR MODEL ***** */

	uint3 color_sizes = make_uint3(nrX,nrY,1);

	//d_center will contain center_red
	CUDA_CHECK_ERRORS(cudaMemcpy(d_center,d_red,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_center, d_color_tmp, color_sizes, center_filt, center_filtSize);

	//d_color_tmp_green will contain center_green
	CUDA_CHECK_ERRORS(cudaMemcpy(d_color_tmp_green,d_green,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_color_tmp_green, d_color_tmp, color_sizes, center_filt, center_filtSize);

	//d_color_tmp_yellow will contain center_yellow
	dev_ave<<<iDivUp(nrX*nrY,128), 128>>>(d_center, d_color_tmp_green, d_color_tmp_yellow, nrX*nrY);

	//d_green will contain surround_green
	conv2D(d_green, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_color_tmp will contain the result
	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_center, d_green, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	CUDA_CHECK_ERRORS(cudaMemcpy(red_green,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
	
	//d_red will contain surround_red
	conv2D(d_red, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_surround will contain surround_blue
	CUDA_CHECK_ERRORS(cudaMemcpy(d_surround,d_blue,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_surround, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_color_tmp_yellow will contain the result
	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp_yellow, d_surround, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	CUDA_CHECK_ERRORS(cudaMemcpy(yellow_blue,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp_green, d_red, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	CUDA_CHECK_ERRORS(cudaMemcpy(green_red,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	//d_surround will contain surround_yellow
	dev_ave<<<iDivUp(nrX*nrY,128), 128>>>(d_red, d_green, d_surround, nrX*nrY);

	//d_blue will contain center_blue
	conv2D(d_blue, d_color_tmp, color_sizes, center_filt, center_filtSize);

	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_blue, d_surround, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	CUDA_CHECK_ERRORS(cudaMemcpy(blue_yellow,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));




	/* ***** MOTION ENERGY MODEL ***** */

	// ME responses do not depend on color responses, but on grayscale values found in d_stimBuf[]

	// shift d_stimBuf in time by 1 frame, from frame i to frame i-1
	for(int i=1;i<nrT;i++)
		CUDA_CHECK_ERRORS(cudaMemcpy(&d_stimBuf[nrX*nrY*(i-1)],&d_stimBuf[nrX*nrY*i],sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));

	// allocate d_resp, which will contain the response to all 28 (nrDirs) space-time orientation at 3 (nrScales) scales
	// for every pixel location (x,y)
	CUDA_CHECK_ERRORS(cudaMemset (d_resp, 0, sizeof(float)*nrX*nrY*nrDirs*nrScales));

	// working copy of grayscale values: copy d_stimBuf to d_scalingStimBuf
	CUDA_CHECK_ERRORS(cudaMemcpy(d_scalingStimBuf,d_stimBuf,sizeof(float)*nrX*nrY*nrT,cudaMemcpyDeviceToDevice));

	// compute the V1 simple cell responses at 3 different spatial scales
	for (int scale=1; scale<=nrScales; scale++) {
		// blur/scale the image... each time this is called stim is blurred more
		// scale 1 == original image resolution (space/time)
		if (scale > 1) {
			float* tmp;
			CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrT));

			// convolve d_scalingStimBuf by scalingFilt in 3D
			uint3 sizes = make_uint3(nrX,nrY,nrT);
			conv3D(d_scalingStimBuf, tmp, sizes, scalingFilt, scalingFiltSize);

			CUDA_CHECK_ERRORS(cudaFree(d_scalingStimBuf));
			d_scalingStimBuf = tmp;
		}

		// nrT is 9, v1GaussSize is 9, so we're taking d_scalingStimBuf[0-0+nrX*nrY*9]
		// since nrT could be greater than v1GaussSize, we take "only the part we want", quote Micah comment
		CUDA_CHECK_ERRORS(cudaMemcpy(d_v1GausBuf, &d_scalingStimBuf[nrX*nrY*((nrT-v1GausSize)/2)], sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));

		float* tmp;
		CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*v1GausSize));

		// convolve d_v1GausBuf by v1Gaus in 3D
		uint3 sizes = make_uint3(nrX,nrY,v1GausSize);
		conv3D(d_v1GausBuf, tmp, sizes, v1Gaus, v1GausSize);
		CUDA_CHECK_ERRORS(cudaFree(d_v1GausBuf));
		d_v1GausBuf = tmp;

		// go through and calculate all directional derivatives and then combine them to calculate the diferent
		// space-time oriented filters
		for (int orderT=0; orderT<=3; orderT++) {
			// reset diffV1GausBufT back to the 3D gaussian filtered version
			CUDA_CHECK_ERRORS(cudaMemcpy(diffV1GausBufT, d_v1GausBuf, sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));

			if (orderT > 0) {
				// take the derivative
				// sizes: tripel (nrX,nrY,v1GaussSize)
				diffV1GausBufT = diff(diffV1GausBufT, sizes, orderT,2);
			}

			for (int orderY=0; orderY<=3-orderT; orderY++) {
				int orderX = 3-orderY-orderT;
			
				CUDA_CHECK_ERRORS(cudaMemcpy(d_diffV1GausBuf, diffV1GausBufT, sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));

				if (orderX > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderX,0);
				if (orderY > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderY,1);

				// combine the directional derivative by the direction of the space-time filter
				accumDiffStims(&d_resp[(scale-1)*nrX*nrY*nrDirs], &d_diffV1GausBuf[nrX*nrY*(v1GausSize/2)], sizes, orderX, orderY, orderT);
			}
		}
	}
	
	// perform half-rectification on V1 simple cell responses
	// eq.4 in S&H: halfrec[L(t)]^2 = max[0,L(t)]^2
	dev_halfRect2<<<iDivUp(nrX*nrY*nrDirs*nrScales,128), 128>>>(d_resp, nrX*nrY*nrDirs*nrScales);
	CUDA_GET_LAST_ERROR("dev_halfRect2() execution failed\n");

	float* tmp;

	// The following two steps have reversed order... S&H first compute the normalized simple cell response, S_n(t),
	// and then compute the local average. We do it the other way around. We might be doing that because we need a
	// normalized responses at the complex cell level (this is our output), but S&H does only normalize at the simple
	// cell and at the MT level. Also, the order of normalizing / local averaging should not matter (interchangeable)

	// complex: convolve by d_complexV1Filt in 2D
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrDirs*nrScales));
	uint3 sizes = make_uint3(nrX,nrY,nrDirs*nrScales);
	conv2D(d_resp, tmp, sizes, complexV1Filt, complexV1FiltSize);
	CUDA_CHECK_ERRORS(cudaFree(tmp));

	// we need to associate each filter at pixel position (x,y) with a power/intensity, but there are 28 filter
	// responses at each location... so we need to (i) average over the 28 filters (3rd dimension in d_resp) and put it
	// into d_pop ...
	dim3 gridm(iDivUp(nrX*nrY,128), nrScales);
	dev_mean3<<<gridm, 128>>>(d_resp, d_pop, nrX*nrY, nrDirs);
	CUDA_GET_LAST_ERROR("dev_mean3() execution failed\n");

	// ... and (ii) sum over some spatial neighborhood
	// population normalization: convolve by d_normV1filtSize in 2D
	uint3 nsizes = make_uint3(nrX,nrY,nrScales);
	CUDA_CHECK_ERRORS(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrScales));
	conv2D(d_pop, tmp, nsizes, normV1filt, normV1filtSize);
	CUDA_CHECK_ERRORS(cudaFree(tmp));

	dev_normalize<<<gridm, 128>>>(d_resp, d_pop, nrX*nrY, nrDirs);
	CUDA_GET_LAST_ERROR("dev_normalize() execution failed\n");

	for (int scale=0; scale<nrScales; scale++)
		dev_scale<<<iDivUp(nrX*nrY*nrDirs,128), 128>>>(&d_resp[scale*nrX*nrY*nrDirs], (scale==0?1000000:(scale==1?500000:100000))/255.0/255*50, nrX*nrY*nrDirs);

	// copy response to the Host side
	CUDA_CHECK_ERRORS(cudaMemcpy(ME,d_resp,sizeof(float)*nrX*nrY*nrDirs*nrScales,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
/*
	unsigned int free, total;
	CUresult res = cuMemGetInfo(&free, &total);
	if(res != CUDA_SUCCESS) printf("!!!! cuMemGetInfo failed! (status = %x)", res);
	printf("used GPU memory %ld\n", total - free);
*/
}
