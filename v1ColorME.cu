#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cufft.h>
#include <cutil_inline.h>
#define IMUL(a, b) __mul24(a, b)

#define nrDirs 28
__constant__  float d_v1Dirs[3][nrDirs] = {{-0.6559, -0.1019, 0.6240, -0.7797, 0.9692, -0.2312, -0.9151, 0.4207, -0.9533, 0.8175, 0.2398, 0.8810, -0.4430, 0.0588, -0.5384, 0.5644, 0.7931, 0.5142, -0.7680, -0.0669, -0.6670, -0.2747, 0.5034, 0.5042, 0.1580, 0.1332, -0.5159, -0.3549},
                                           { 0.7246, -0.9718, 0.7496, -0.5837, -0.0810, 0.9439, 0.3203, -0.8712, -0.1593, -0.5142, 0.9304, 0.3737, -0.8031, -0.8126, 0.6004, -0.5738, 0.0024, 0.5969, 0.1436, 0.7757, -0.4004, -0.5108, 0.2375, -0.2221, -0.5140, 0.5194, -0.0870, 0.3838},
                                           { 0.2113, 0.2126, 0.2210, 0.2266, 0.2327, 0.2359, 0.2451, 0.2529, 0.2567, 0.2593, 0.2772, 0.2902, 0.3984, 0.5799, 0.5913, 0.5935, 0.6091, 0.6160, 0.6241, 0.6275, 0.6283, 0.8146, 0.8308, 0.8345, 0.8431, 0.8441, 0.8522, 0.8525}};


#define scalingFiltSize 5
__constant__ float d_scalingFilt[scalingFiltSize] = {0.0884, 0.3536, 0.5303, 0.3536, 0.0884};
float* scalingFilt;
#define v1GausSize 9
__constant__ float d_v1Gaus[v1GausSize] = {0.0007, 0.0155, 0.0903, 0.2345, 0.3179, 0.2345, 0.0903, 0.0155, 0.0007};
float* v1Gaus;
#define complexV1FiltSize 11
__constant__ float d_complexV1Filt[complexV1FiltSize] = {0.0019, 0.0110, 0.0430, 0.1142, 0.2052, 0.2495, 0.2052, 0.1142, 0.0430, 0.0110, 0.0019};
float* complexV1Filt;
#define normV1filtSize 21
__constant__ float d_normV1filt[normV1filtSize] = {0.0013, 0.0031, 0.0067, 0.0132, 0.0237, 0.0389, 0.0584, 0.0800, 0.1001, 0.1146, 0.1199, 0.1146, 0.1001, 0.0800, 0.0584, 0.0389, 0.0237, 0.0132, 0.0067, 0.0031, 0.0013};
float* normV1filt;

#define diff1filtSize 3
__constant__ float d_diff1filt[diff1filtSize] = {-1/2.0, 0, 1/2.0};
float* diff1filt;

#define diff2filtSize 3
__constant__ float d_diff2filt[diff2filtSize] = {1, -2, 1};
float* diff2filt;

#define diff3filtSize 5
__constant__ float d_diff3filt[diff3filtSize] = {-1/2.0, 1, 0, -1, 1/2.0};
float* diff3filt;

#define nrScales 3
#define nrT (scalingFiltSize*(nrScales-1)-1)//(scalingFiltSize*nrScales-2)

int stimBufX, stimBufY;
float* d_resp;
float* d_stimBuf;
float* d_scalingStimBuf; // the temporary matrix that will be filtered once for each scale...
float* d_v1GausBuf;
float* d_v1GausBuf2;
float* d_diffV1GausBuf;
float* diffV1GausBufT;

unsigned char* d_stim;
float* d_pop;

float* d_red;
float* d_green;
float* d_blue;

float* d_center;
float* d_surround;
float* d_color_tmp;
float* d_color_tmp_green;
float* d_color_tmp_yellow;


#define iDivUp(a,b) ((a)+(b)-1)/(b)

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

// odata must be pre-allocated
// the result will end up in idata...
// filtlen can not be greater than CONVN_THREAD_SIZE2
void conv2D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension	
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
        cutilCheckMsg("dev_conv1() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension	
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
        cutilCheckMsg("dev_convn() execution failed\n");
}

// odata must be pre-allocated
// filtlen can not be greater than CONVN_THREAD_SIZE2
void conv3D(float* idata, float* odata, dim3 _sizes, const float* filt, int filtlen) {
	unsigned int* sizes = (unsigned int*)&_sizes;
	float* tmp;

	// convolve the first dimension	
	dim3 grid1(iDivUp(sizes[0], CONV1_THREAD_SIZE-(filtlen-1)), sizes[1]*sizes[2]);
	dim3 threads1(CONV1_THREAD_SIZE, 1, 1);
	dev_conv1<<<grid1, threads1>>>(idata, odata, sizes[0], filt, filtlen);
        cutilCheckMsg("dev_conv1() execution failed\n");
	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the second dimension	
	dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
	dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
        cutilCheckMsg("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;

	// convolve the third dimension	
	dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
	dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
	dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
        cutilCheckMsg("dev_convn() execution failed\n");

	tmp = idata;
	idata = odata;
	odata = tmp;
}

//will free idata
float* diff(float* idata, uint3 _sizes, int order, int dim)
{
	unsigned int* sizes = (unsigned int*)&_sizes;
	int filtlen;
	float* filt;
	float* odata;

	cutilSafeCall(cudaMalloc((void**)&odata, sizeof(float)*sizes[0]*sizes[1]*sizes[2]));
	
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
			cutilCheckMsg("dev_conv1() execution failed\n");
			break;
		}
		case 1: {
			// convolve the second dimension	
			dim3 grid2(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[1], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[2]);
			dim3 threads2(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid2, threads2>>>(idata, odata, sizes[0], sizes[1], sizes[0], sizes[0]*sizes[1], sizes[2], filt, filtlen);
			cutilCheckMsg("dev_convn() execution failed\n");
			break;
		}
		case 2: {
			// convolve the third dimension	
			dim3 grid3(iDivUp(sizes[0], CONVN_THREAD_SIZE1), iDivUp(sizes[2], CONVN_THREAD_SIZE2-(filtlen-1))*sizes[1]);
			dim3 threads3(CONVN_THREAD_SIZE1, CONVN_THREAD_SIZE2, 1);
			dev_convn<<<grid3, threads3>>>(idata, odata, sizes[0], sizes[2], sizes[0]*sizes[1], sizes[0], sizes[1], filt, filtlen);
			cutilCheckMsg("dev_convn() execution failed\n");
			break;
		}
	}

	cutilSafeCall(cudaFree (idata));

	return odata;
}

__global__ void dev_accumDiffStims(float *d_resp, float *diffV1GausBuf, int nrXnrY, int c, int orderX, int orderY, int orderT) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	__shared__ float dirorders[nrDirs];

	if (threadIdx.x < nrDirs) {
		const float dir1 = d_v1Dirs[0][threadIdx.x];
		const float dir2 = d_v1Dirs[1][threadIdx.x];
		const float dir3 = d_v1Dirs[2][threadIdx.x];

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
	int factorials[4] = {1, 1, 2, 6};
	int c = 6/factorials[orderX]/factorials[orderY]/factorials[orderT]; // the scaling factor for this directial derivative; similar to the binomial coefficients

        dev_accumDiffStims<<<iDivUp(_sizes.x*_sizes.y, 128), 128>>>(d_resp, diffV1GausBuf, _sizes.x*_sizes.y, c, orderX, orderY, orderT);
        cutilCheckMsg("dev_accumDiffStims() execution failed\n");
}


__global__ void dev_halfRect2(float *data, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float d = data[i];
		d = (d>0)?d:0;
		data[i] = d*d;
	}
}

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

__global__ void dev_sub(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] - i2data[i];
	}
}

__global__ void dev_ave(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = (i1data[i] + i2data[i])/2;
	}
}

__global__ void dev_sum(float *i1data, float *i2data, float* odata, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		odata[i] = i1data[i] + i2data[i];
	}
}

__global__ void dev_scaleHalfRect(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float tmp = data[i];
		data[i] = (tmp>0)?sqrt(sqrt(tmp))*scale:0;
	}
}

__global__ void dev_scale(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		data[i] *= scale;
	}
}

__global__ void dev_scaleSqrt(float *data, float scale, int len) {
	const int     tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);

	for(int i = tid; i < len; i += threadN) {
		float tmp = data[i];
		data[i] = (tmp>0)?sqrt(tmp)*scale:0;
	}
}

// stim should be an array of lenght nrX*nrY*3 with the data being organized such as: R1 G1 B1 R2 G2 B2 ...
void calcColorME(int nrX, int nrY, unsigned char* stim, float* red_green, float* green_red, float* blue_yellow, float* yellow_blue, float* ME, bool GPUpointers)
{
	// allocate memory on the GPU
	if (nrX != stimBufX || nrY != stimBufY) {
		stimBufX = nrX;
		stimBufY = nrY;

		// allocate the response matrix
		cutilSafeCall(cudaMalloc((void**)&d_resp, sizeof(float)*nrX*nrY*nrDirs*nrScales));

		// probably should free previous buffers if they were previously allocated...

		cutilSafeCall(cudaMalloc ((void**)&d_stimBuf, nrX*nrY*nrT*sizeof(float)));
		cutilSafeCall(cudaMemset (d_stimBuf, 0, nrX*nrY*nrT*sizeof(float)));

		cutilSafeCall(cudaMalloc((void**)&diffV1GausBufT, sizeof(float)*nrX*nrY*v1GausSize));
	
		cutilSafeCall(cudaMalloc ((void**)&d_stim, nrX*nrY*3));
		cutilSafeCall(cudaMalloc ((void**)&d_scalingStimBuf, nrX*nrY*nrT*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_v1GausBuf, nrX*nrY*nrT*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_diffV1GausBuf, nrX*nrY*nrT*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_pop, nrX*sizeof(float)*nrY*nrScales));

		cutilSafeCall(cudaMalloc ((void**)&d_red, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_green, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_blue, nrX*nrY*sizeof(float)));

		cutilSafeCall(cudaMalloc ((void**)&d_center, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_surround, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_color_tmp, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_color_tmp_green, nrX*nrY*sizeof(float)));
		cutilSafeCall(cudaMalloc ((void**)&d_color_tmp_yellow, nrX*nrY*sizeof(float)));

		cutilSafeCall(cudaGetSymbolAddress((void**)&scalingFilt, "d_scalingFilt"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&v1Gaus, "d_v1Gaus"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&complexV1Filt, "d_complexV1Filt"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&normV1filt, "d_normV1filt"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&diff1filt, "d_diff1filt"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&diff2filt, "d_diff2filt"));
		cutilSafeCall(cudaGetSymbolAddress((void**)&diff3filt, "d_diff3filt"));
		
	}
	// use the preexisting filters because they are about the right size and give good results
	float* center_filt = v1Gaus;
	float* surround_filt = complexV1Filt;
	const int center_filtSize = v1GausSize;
	const int surround_filtSize = complexV1FiltSize;

	cutilSafeCall(cudaMemcpy(d_stim,stim,3*nrX*nrY,cudaMemcpyHostToDevice));
	dev_split<<<iDivUp(nrX*nrY,128), 128>>>(d_stim, d_red, d_green, d_blue, &d_stimBuf[nrX*nrY*(nrT-1)], nrX*nrY);
 	cutilCheckMsg("dev_split() execution failed\n");


	uint3 color_sizes = make_uint3(nrX,nrY,1);

	//d_center will contain center_red
	cutilSafeCall(cudaMemcpy(d_center,d_red,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_center, d_color_tmp, color_sizes, center_filt, center_filtSize);

	//d_color_tmp_green will contain center_green
	cutilSafeCall(cudaMemcpy(d_color_tmp_green,d_green,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_color_tmp_green, d_color_tmp, color_sizes, center_filt, center_filtSize);

	//d_color_tmp_yellow will contain center_yellow
	dev_ave<<<iDivUp(nrX*nrY,128), 128>>>(d_center, d_color_tmp_green, d_color_tmp_yellow, nrX*nrY);

	//d_green will contain surround_green
	conv2D(d_green, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_color_tmp will contain the result
	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_center, d_green, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	cutilSafeCall(cudaMemcpy(red_green,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
	
	//d_red will contain surround_red
	conv2D(d_red, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_surround will contain surround_blue
	cutilSafeCall(cudaMemcpy(d_surround,d_blue,sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));
	conv2D(d_surround, d_color_tmp, color_sizes, surround_filt, surround_filtSize);

	//d_color_tmp_yellow will contain the result
	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp_yellow, d_surround, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	cutilSafeCall(cudaMemcpy(yellow_blue,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp_green, d_red, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	cutilSafeCall(cudaMemcpy(green_red,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	//d_surround will contain surround_yellow
	dev_ave<<<iDivUp(nrX*nrY,128), 128>>>(d_red, d_green, d_surround, nrX*nrY);

	//d_blue will contain center_blue
	conv2D(d_blue, d_color_tmp, color_sizes, center_filt, center_filtSize);

	dev_sub<<<iDivUp(nrX*nrY,128), 128>>>(d_blue, d_surround, d_color_tmp, nrX*nrY);
	dev_scaleHalfRect<<<iDivUp(nrX*nrY,128), 128>>>(d_color_tmp, 50.0, nrX*nrY);
	cutilSafeCall(cudaMemcpy(blue_yellow,d_color_tmp,sizeof(float)*nrX*nrY,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));

	// shift d_stimBuf in time by 1 frame
	for(int i=1;i<nrT;i++)
		cutilSafeCall(cudaMemcpy(&d_stimBuf[nrX*nrY*(i-1)],&d_stimBuf[nrX*nrY*i],sizeof(float)*nrX*nrY,cudaMemcpyDeviceToDevice));

	cutilSafeCall(cudaMemset (d_resp, 0, sizeof(float)*nrX*nrY*nrDirs*nrScales));

	//copy d_stimBuf to d_scalingStimBuf
	cutilSafeCall(cudaMemcpy(d_scalingStimBuf,d_stimBuf,sizeof(float)*nrX*nrY*nrT,cudaMemcpyDeviceToDevice));
	for (int scale=1; scale<=nrScales; scale++) {
		if (scale > 1) {
			float* tmp;
			cutilSafeCall(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrT));

			// convolve d_scalingStimBuf by scalingFilt in 3D
			uint3 sizes = make_uint3(nrX,nrY,nrT);
			conv3D(d_scalingStimBuf, tmp, sizes, scalingFilt, scalingFiltSize);

			cutilSafeCall(cudaFree(d_scalingStimBuf));
			d_scalingStimBuf = tmp;
		}

		// extract just the part we want...
		cutilSafeCall(cudaMemcpy(d_v1GausBuf, &d_scalingStimBuf[nrX*nrY*((nrT-v1GausSize)/2)], sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));

		float* tmp;
		cutilSafeCall(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*v1GausSize));

		// convolve d_v1GausBuf by v1Gaus in 3D
		uint3 sizes = make_uint3(nrX,nrY,v1GausSize);
		conv3D(d_v1GausBuf, tmp, sizes, v1Gaus, v1GausSize);
		cutilSafeCall(cudaFree(d_v1GausBuf));
		d_v1GausBuf = tmp;

		for (int orderT=0; orderT<=3; orderT++) {
			cutilSafeCall(cudaMemcpy(diffV1GausBufT, d_v1GausBuf, sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));
			if (orderT > 0) diffV1GausBufT = diff(diffV1GausBufT, sizes, orderT,2);

			for (int orderY=0; orderY<=3-orderT; orderY++) {
				int orderX = 3-orderY-orderT;
			
				cutilSafeCall(cudaMemcpy(d_diffV1GausBuf, diffV1GausBufT, sizeof(float)*nrX*nrY*v1GausSize, cudaMemcpyDeviceToDevice));

				if (orderX > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderX,0);
				if (orderY > 0) d_diffV1GausBuf = diff(d_diffV1GausBuf, sizes, orderY,1);

				accumDiffStims(&d_resp[(scale-1)*nrX*nrY*nrDirs], &d_diffV1GausBuf[nrX*nrY*(v1GausSize/2)], sizes, orderX, orderY, orderT);
			}
		}
	}
	
	// half rect squared...
	dev_halfRect2<<<iDivUp(nrX*nrY*nrDirs*nrScales,128), 128>>>(d_resp, nrX*nrY*nrDirs*nrScales);
	cutilCheckMsg("dev_halfRect2() execution failed\n");

	float* tmp;

	// complex: convolve by d_complexV1Filt in 2D
	cutilSafeCall(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrDirs*nrScales));
	uint3 sizes = make_uint3(nrX,nrY,nrDirs*nrScales);
	conv2D(d_resp, tmp, sizes, complexV1Filt, complexV1FiltSize);
	cutilSafeCall(cudaFree(tmp));

	// normalize
	dim3 gridm(iDivUp(nrX*nrY,128), nrScales);
	dev_mean3<<<gridm, 128>>>(d_resp, d_pop, nrX*nrY, nrDirs);
	cutilCheckMsg("dev_mean3() execution failed\n");

	// population normalization: convolve by d_normV1filtSize in 2D
	uint3 nsizes = make_uint3(nrX,nrY,nrScales);
	cutilSafeCall(cudaMalloc((void**)&tmp, sizeof(float)*nrX*nrY*nrScales));
	conv2D(d_pop, tmp, nsizes, normV1filt, normV1filtSize);
	cutilSafeCall(cudaFree(tmp));

	dev_normalize<<<gridm, 128>>>(d_resp, d_pop, nrX*nrY, nrDirs);
	cutilCheckMsg("dev_normalize() execution failed\n");

	for (int scale=0; scale<nrScales; scale++)
		dev_scale<<<iDivUp(nrX*nrY*nrDirs,128), 128>>>(&d_resp[scale*nrX*nrY*nrDirs], (scale==0?1000000:(scale==1?500000:100000))/255.0/255*50, nrX*nrY*nrDirs);

	// copy response to the Host side
	cutilSafeCall(cudaMemcpy(ME,d_resp,sizeof(float)*nrX*nrY*nrDirs*nrScales,GPUpointers?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
/*
	unsigned int free, total;
	CUresult res = cuMemGetInfo(&free, &total);
	if(res != CUDA_SUCCESS) printf("!!!! cuMemGetInfo failed! (status = %x)", res);
	printf("used GPU memory %ld\n", total - free);
*/
}

