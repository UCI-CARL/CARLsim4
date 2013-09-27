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

#include "snn.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;


int cvKeyCheck()
{
	// wait for a key
	keyVal = cvWaitKey(10);
//			if(keyVal != -1) {
//				fprintf(stderr, "keyVal = %c %d %d %d \n", (char) keyVal, keyVal, (int) 'p', (int) 's', (int) 'q');
//				cvWaitKey(0);
//			}
	if ((char) keyVal == 'p') {
		fprintf(stderr, "Pause key pressed ! Press any to continue\n");
		rasterPlotAll(10, false);
		cvWaitKey(0);
	}
	else if( (char) keyVal == 's') {
		//saveConnectionWeights();
		rasterPlotAll(10, false);
		cvWaitKey(0);
	}
	else if( (char) keyVal == 'q') {
		cvWaitKey(0);
		exitSimulation(0);
	}
	else if( (char) keyVal ==  'b') {
		return keyVal;
	}
	else if( (char) keyVal ==  'n') {
		return keyVal;
	}
}


class spikeRaster_t {
	public:
		spikeRaster_t(
			static int winCounter;
			winID = winCounter++;
			
	CvMat* imgWin;
	bool firstDisplay;
	int offset;
	int maxSize;
	int yScale;
	int xPos;
	int yPos;
	int winID;
	char winName[100];
};


void spikeRaster(CpuSNN* s, int grpId, unsigned int* Nids, unsigned int* timeCnts, spikeRaster_t* sR)
{
	group_info2_t gInfo2 = s->getGroupInfo2(gid);

	cvNamedWindow(sR->winName);
	if(sR->firstDisplay) {
		cvMoveWindow(sR->winName, sR->xPos, sR->yPos);

		sR->imgWin = cvCreateMat(160, 160, CV_8UC1);
	}
	sR->firstDisplay = false;

	CvMat* imgWin = sR->imgWin;
	
	cvSetZero(imgWin);
	int pos    = 0;
	for (int t=0; t < 1000; t++) {
		for(int i=0; i<timeCnts[t];i++,pos++) {
			int id = Nids[pos];
			assert(id <= grp_Info[gid].EndN);
			assert(id >= grp_Info[gid].StartN);
			int  val = id-grp_Info[gid].StartN-sR->offset;
			if( (val < sR->maxSize) && (val >= 0)) {

				if ( sR->yScale < 2) {
					if( (sR->maxSize+10-val) < MAX_CV_WINDOW_PIXELS) {
						*((uchar*)CV_MAT_ELEM_PTR( *imgWin, sR->maxSize+10-val, t/2)) = 255.0;
					}
				}
				else {
					if ( sR->maxSize+10-val*sR->yScale < MAX_CV_WINDOW_PIXELS) {
						*((uchar*)CV_MAT_ELEM_PTR( *imgWin, sR->maxSize+10-val*sR->yScale, t/2)) = 255.0;
						*((uchar*)CV_MAT_ELEM_PTR( *imgWin, sR->maxSize+10-1-val*sR->yScale, t/2)) = 255.0;
						*((uchar*)CV_MAT_ELEM_PTR( *imgWin, sR->maxSize+10-val*sR->yScale, 1+t/2)) = 255.0;
						*((uchar*)CV_MAT_ELEM_PTR( *imgWin, sR->maxSize+10-1-val*sR->yScale, 1+t/2)) = 255.0;
					}
				}
			}
		}
	}

	cvShowImage(winName, imgWin);
}


/*


	int showInputPattern(GaborFilter* gb, int radius, float angle, bool showImage, bool firstDisplay)
	{
		static CvMat* imgWin = cvCreateMat(160, 160, CV_8UC1);
		char winName[]="mainWin";

		int factor=1;
		//float angle = angles[(int)((sizeof(angles)/4)*drand48())]; //45.0;//   = atof(argv[2]);
		float freq  = 1/10.0; //0.1105/factor;   // atof(argv[3]);
		float bw    = 1.0; //   atof(argv[4]);
		float phase = 0.0; // atof(argv[5]);
		float rho   = 1.5; //atof(argv[6]);
		float sig   = 2*factor;

		gb->GenerateFilter(radius, angle, freq, sig, phase, bw, rho);

		CvMat* img = gb->GetPixel();

		if((!showImage) && (!firstDisplay))
		  return 0;

		// create a window
		cvNamedWindow(winName);//, CV_WINDOW_AUTOSIZE);
		if(firstDisplay)
			cvMoveWindow("mainWin", 300, 500);

		cvResize(img, imgWin);

		cvShowImage(winName, imgWin);

		return 0;
	}

	void CpuSNN::getScaledWeights(void* imgPtr, int nid, int src_grp, int repx, int repy)
	{
		if(!plotUsingOpenCV)
			return;

		CvMat* img = (CvMat*) imgPtr;
		cvSetZero(img);
		int dest_grp=findGrpId(nid);
		post_info_t* preIds  = &preSynapticIds[cumulativePre[nid]];
		int  cols = img->cols;
		int  rows = img->rows;
		float* synWts   = &wt[cumulativePre[nid]];
		// find a suitable match for each pre-syn id that we are interested in ..
		float maxWt = grpConnInfo[src_grp][dest_grp]->maxWt;
		for(int i=0; i < Npre[nid]; i++, preIds++, synWts++) {
			//int preId = (*preIds) & POST_SYN_NEURON_MASK;
			int preId = GET_CONN_NEURON_ID((*preIds));
			assert(preId < (numN));
			// preId matches the src_grp that we are interested..
			if(src_grp == findGrpId(preId)) {
				int imgId = preId - grp_Info[src_grp].StartN;
				assert((imgId*repx*repy) < (cols*rows));
				int xpt = imgId%(cols/repx);
				int ypt = imgId/(cols/repx);
				for(int p=0; p < repy; p++)
				  for(int q=0; q < repx; q++) {
					*((uchar*)CV_MAT_ELEM_PTR( *img, repy*ypt+p, repx*xpt+q)) = (uchar) ((255.0*(*synWts))/maxWt);
				  }
			}
		}
	}

	void CpuSNN::getScaledWeights1D(void* imgPtr, unsigned int nid, int src_grp, int repx, int repy)
	{
		char fname[100];
		sprintf(fname, "weightsSrc%dDest%d.txt", src_grp, nid);
		//FILE *fp = fopen(fname, "w");
		//if(fp == NULL)

		if(!plotUsingOpenCV)
			return;

		CvMat* img = (CvMat*) imgPtr;
		cvSetZero(img);

		int dest_grp = findGrpId(nid);
		post_info_t* preIds  = &preSynapticIds[cumulativePre[nid]];

		// find a suitable match for each pre-syn id that we are interested in ..
		float maxWt 	 = grpConnInfo[src_grp][dest_grp]->maxWt;
		int   cols  	 = img->cols;
		float rowFactor  = (img->rows-8)/(repy*maxWt);
		float* synWts    = &wt[cumulativePre[nid]];
		float* synWtChange    = &wtChange[cumulativePre[nid]];

		if(currentMode == GPU_MODE) {
			copyWeightsGPU(nid, src_grp);
		}

		//fprintf(stderr, "%d => %d : ", src_grp, nid);
		for(int i=0; i < Npre[nid]; i++, preIds++, synWts++, synWtChange++) {
			int preId = GET_CONN_NEURON_ID((*preIds));
			//int preId = (*preIds) & POST_SYN_NEURON_MASK;
			assert(preId < (numN));
			// preId matches the src_grp that we are interested..
			if( src_grp == findGrpId(preId)) {
				int imgId = preId - grp_Info[src_grp].StartN;
				assert((imgId*repx) < cols);
				int xpt = imgId;
				if (*synWts > maxWt) {
					fprintf(stderr, "Maximum synaptic weight (%f) exceeded for this Weight (%f)...\n", maxWt, *synWts);
					assert (*synWts <= maxWt);
				}
				int ypt = (uchar)(rowFactor*(*synWts));
				//fprintf(stderr, "[%d] %f (%f) ", preId, *synWts, *synWtChange);
				for(int p=0; p < repy; p++)
				  for(int q=0; q < repx; q++) {
					*((uchar*)CV_MAT_ELEM_PTR( *img, img->rows - 1 -(4+repy*ypt+p), repx*xpt+q )) = 255.0;
				  }
			}
		}
		//fprintf(stderr, "\n");
	}

	void CpuSNN::getScaledWeightRates1D(unsigned int nid, int src_grp)
	{
		// find a suitable match for each pre-syn id that we are interested in ..
		int dest_grp = findGrpId(nid);
		float maxWt = grpConnInfo[src_grp][dest_grp]->maxWt;
		char fname[100];
		sprintf(fname, "weightRatesSrc%dDest%d.txt(max=%3.3f)", src_grp, nid, maxWt);

		post_info_t* preIds  = &preSynapticIds[cumulativePre[nid]];

		float* synWts   = &wtChange[cumulativePre[nid]];
		fprintf(stderr, "wtChange: %d => %d : ", src_grp, nid);
		for(int i=0; i < Npre[nid]; i++, preIds++, synWts++) {
			//int preId = (*preIds) & POST_SYN_NEURON_MASK;
			int preId = GET_CONN_NEURON_ID((*preIds));
			assert(preId < (numN));
			// preId matches the src_grp that we are interested..
			if( src_grp == findGrpId(preId)) {
				fprintf(stderr, "%d:%f ", preId, *synWts);
			}
		}

		fprintf(stderr, "\n");
		//fclose(fp);
	}

	void CpuSNN::showWeightRatePattern1D (int destGrp, int srcGrp)
	{

		checkNetworkBuilt();

		if(grpConnInfo[srcGrp][destGrp]==NULL) {
			fprintf(stderr, "showWeightPattern1D failed\n");
			fprintf(stderr, "No connections exists between %s(%d) to %s(%d)\n", grp_Info2[srcGrp].Name.c_str(), srcGrp, grp_Info2[destGrp].Name.c_str(), destGrp);
			return;
		}

		//char winName[100];
		//int wtImgSizeX = 2*grp_Info[srcGrp].SizeN;
		//int wtImgSizeY = 128;
		//int	   repSize = 2;
		//static CvMat* wtImg = cvCreateMat(wtImgSizeY, wtImgSizeX, CV_8UC1);

		for(int k=0, n = grp_Info[destGrp].StartN; n <= grp_Info[destGrp].EndN; n++,k++)
		{
			//sprintf(winName, "weightRateNeuron%d%d", srcGrp, n);
			//cvNamedWindow(winName, CV_WINDOW_AUTOSIZE);
			//if(firstDisplay)
			//cvMoveWindow(winName, col+0*(wtImg->cols+40), row+k*(wtImg->rows+40));
			//getScaledWeightRates1D((void*)wtImg, n, srcGrp, repSize, repSize);
			getScaledWeightRates1D(n, srcGrp);
			//cvShowImage(winName, wtImg );
		}
	}

	void CpuSNN::showWeightPattern1D(int destGrp, int srcGrp, int row, int col)
	{
		checkNetworkBuilt();

		if(!plotUsingOpenCV)
			return;

		if(grpConnInfo[srcGrp][destGrp]==NULL) {
			fprintf(stderr, "showWeightPattern1D cannot proceed\n");
			fprintf(stderr, "No connections exists between %s(%d) to %s(%d)\n", grp_Info2[srcGrp].Name.c_str(), srcGrp, grp_Info2[destGrp].Name.c_str(), destGrp);
			return;
		}

		char winName[100];
		int wtImgSizeX = 2*grp_Info[srcGrp].SizeN;
		int wtImgSizeY = 128;
		int repSize = 2;
		static CvMat* wtImg = cvCreateMat(wtImgSizeY, wtImgSizeX, CV_8UC1);

		for(int k=0, n = grp_Info[destGrp].StartN; n <= grp_Info[destGrp].EndN; n++,k++)
		{
			sprintf(winName, "g=%d:%dweightNeuron1D", srcGrp, n);
			cvNamedWindow(winName, CV_WINDOW_AUTOSIZE);
			if(firstDisplay)
			  cvMoveWindow(winName, col, row+k*(wtImg->rows+30));
			getScaledWeights1D((void*)wtImg, n, srcGrp, repSize, repSize);
			cvShowImage(winName, wtImg );
		}
	}

	void CpuSNN::showWeightPattern (int destGrp, int srcGrp, int row, int col, int size)
	{
		checkNetworkBuilt();

		if(!plotUsingOpenCV)
			return;

		if(grpConnInfo[srcGrp][destGrp]==NULL) {
			fprintf(stderr, "showWeightPattern1D failed\n");
			fprintf(stderr, "No connections exists between %s(%d) to %s(%d)\n", grp_Info2[srcGrp].Name.c_str(), srcGrp, grp_Info2[destGrp].Name.c_str(), destGrp);
			return;
		}

		int wtImgSize = 160;
		static CvMat* wtImg = cvCreateMat(wtImgSize, wtImgSize, CV_8UC1);
		int	   repSize = (wtImgSize/size);

		char winName[100];

		for(int k=0, n = grp_Info[destGrp].StartN; n <= grp_Info[destGrp].EndN; n++,k++)
		{
			sprintf(winName, "g=%d:id=%dweightNeuron", srcGrp, n);
			cvNamedWindow(winName, CV_WINDOW_AUTOSIZE);
			if(firstDisplay)
			cvMoveWindow(winName, col+0*(wtImg->cols+40), row+k*(wtImg->rows+40));
			getScaledWeights((void*)wtImg, n, srcGrp, repSize, repSize);
			cvShowImage(winName, wtImg );
		}
	}

	void CpuSNN::rasterPlot(int gid, const string& fname, int row, int col, bool dontOverWrite)
	{
		DBG(2, fpLog, AT, "rasterPlot() called");

		FILE* fpDumpFile = getRasterFilePointer(fname, dontOverWrite);

		// no we got the file pointer.. go and dump
		// if fp=null, the GPU will atleast update
		// the CVMat array with spike info to be displayed later.
		// if fp=null, then the CPU will mostly just return without
		// dumping the spike information to the file...
		dumpSpikeBuffToFile(fpDumpFile, gid);

		// enable plotting..
		if(plotUsingOpenCV) {
			if (gid==-1) {
				for (int i=0; i < numSpikeMonitor; i++) {
					int gid = monGrpId[i];
					plotSpikeBuff(gid, row, col);
				}
			}
			else
				plotSpikeBuff(gid, row, col);
		}
	}

	void CpuSNN::plotBuffInit(int gid)
	{
		DBG(2, fpLog, AT, "plotBufInit()");
		int  monId = grp_Info[gid].MonitorId;
		if(monId == -1) {
			fprintf(stderr, "RasterPlot(gid=%d) failed... Group %s not monitored..\n", gid, grp_Info2[gid].Name.c_str());
			fprintf(stderr, "call spikeMonitor(gid=%d) function in your code to ensure monitoring of group %s\n", gid, grp_Info2[gid].Name.c_str());
			return;
		}

		int maxSize = (grp_Info[gid].SizeN);
		maxSize     = (maxSize >= MAX_CV_WINDOW_PIXELS) ? MAX_CV_WINDOW_PIXELS : maxSize;
		maxSize     = (maxSize <  MIN_CV_WINDOW_PIXELS) ? MIN_CV_WINDOW_PIXELS : maxSize;
		int yscale   = maxSize/grp_Info[gid].SizeN;

		CvMat* imgWin = cvCreateMat(maxSize+20, 501, CV_8UC1);
		assert((gid >= 0)&& (gid < numGrp));

		// save the CV pointer so that we can refer to it in the future when
		// plotting useful information ....
//		monCvMatPtr[monId] = (void*) imgWin;
		monMaxSizeN[monId] = maxSize;
		monYscale[monId] = yscale;

		return;
	}

	void CpuSNN::plotSpikeBuff(int gid, int row, int col)
	{
		static int cumSize = 0;
		static int cntNext = 0;

		if(!plotUsingOpenCV)
			return;

		DBG(2, fpLog, AT, "plotSpikeBuff()");

		int  monId = grp_Info[gid].MonitorId;
		if(monId == -1) {
			fprintf(stderr, "RasterPlot(gid=%d) failed... Group %s not monitored..\n", gid, grp_Info2[gid].Name.c_str());
			fprintf(stderr, "call spikeMonitor(gid=%d) function in your code to ensure monitoring of group %s\n", gid, grp_Info2[gid].Name.c_str());
			return;
		}

		int maxSize  = monMaxSizeN[monId];
		//int yscale   = monYscale[monId];

	}

	void CpuSNN::setImgWin(int monId, int localId, int t)
	{
		assert(monId != -1);
		assert(localId < numN);

		if(!plotUsingOpenCV)
			return;
		return;
	}

	void CpuSNN::plotProbes()
	{
		if(!plotUsingOpenCV)
			return;

		char winNameI[100];
		char winNameV[100];
		char winNameH[100];
		char winNameF[100];

		static CvMat* imgWinF;
		static CvMat* imgWinH;
		static CvMat* imgWinV;
		static CvMat* imgWinI;

		static int firstSet = 0;

		int imgSizeY   = 100;
		int imgSizeX   = 500;
		float scaleX   = 1000.0/imgSizeX;

		int prevGid    = -1;
		int cnt=0;
		probeParam_t* n = neuronProbe;

		float fmin = 0.0; float hfmin = 0.0;

		while(n) {
			int retVal = 0;
			int nid  = n->nid;
			int gid	 = findGrpId(nid);
			if(prevGid == -1)  prevGid = gid;
			assert((gid >= 0)&& (gid < numGrp));
			int len = 0;

			if (n->type & PROBE_CURRENT) {
			   retVal = sprintf(winNameI, "probeCurrent:g=%d,n=%d:%s(max=%3.1f,min=%3.1f)", gid, nid-grp_Info[gid].StartN, grp_Info2[gid].Name.c_str(), n->imax, n->imin);
			   if(retVal < 0)	perror("string length error\n");
			   if(firstSet==0) {
					imgWinI = cvCreateMat(imgSizeY, imgSizeX, CV_8UC1);
				    cvNamedWindow(winNameI);
				    cvMoveWindow(winNameI, 10+cnt*imgSizeX, 100+len*(imgSizeY+30)+10);
			   }
			   cvSetZero(imgWinI);
			   len++;
			}

			if (n->type & PROBE_VOLTAGE) {
				sprintf(winNameV, "probeVoltage:g=%d,n=%d:%s(max=%3.1f,min=%3.1f)", gid, nid-grp_Info[gid].StartN, grp_Info2[gid].Name.c_str(), n->vmax, n->vmin);
				if(retVal < 0)	perror("string length error\n");
				if(firstSet==0) {
					imgWinV = cvCreateMat(imgSizeY, imgSizeX, CV_8UC1);
					cvNamedWindow(winNameV);
					cvMoveWindow(winNameV, 10+cnt*imgSizeX, 100+len*(imgSizeY+30)+10);
				}
				cvSetZero(imgWinV);
				len++;
			}

			if (n->type & PROBE_HOMEO_RATE) {
				sprintf(winNameH, "probeHomeoRate:g=%d,n=%d:%s(max=%3.1f,base=%3.1f)", gid, nid-grp_Info[gid].StartN, grp_Info2[gid].Name.c_str(), n->hfmax, baseFiring[nid]);
				if(retVal < 0)	perror("string length error\n");
				if(firstSet==0) {
					imgWinH = cvCreateMat(imgSizeY, imgSizeX, CV_8UC1);
					cvNamedWindow(winNameH);
					cvMoveWindow(winNameH, 10+cnt*imgSizeX, 100+len*(imgSizeY+30)+10);
				}
				cvSetZero(imgWinH);
				len++;
			}
			if (n->type & PROBE_FIRING_RATE) {
				sprintf(winNameF, "probeFiringRate:g=%d,n=%d:%s(max=%3.1f,base=%3.1f)", gid, nid-grp_Info[gid].StartN, grp_Info2[gid].Name.c_str(), n->fmax, baseFiring[nid]);
				if(retVal < 0)	perror("string length error\n");
				if(firstSet==0) {
					imgWinF = cvCreateMat(imgSizeY, imgSizeX, CV_8UC1);
					cvNamedWindow(winNameF);
					cvMoveWindow(winNameF, 10+cnt*imgSizeX, 100+len*(imgSizeY+30)+10);
				}
				cvSetZero(imgWinF);
				len++;
			}

			if(prevGid != gid) {
				cnt = cnt+1;
			}

			float vScale = (imgSizeY-1)/(n->vmax-n->vmin);
			float iScale = (imgSizeY-1)/(n->imax-n->imin);
			float hScale = (imgSizeY-1)/(n->hfmax-hfmin);
			float fScale = (imgSizeY-1)/(n->fmax-fmin);

			for(int i=0; i < 1000; i++) {
				int xpt = (int) (i/scaleX);
				if (n->type & PROBE_VOLTAGE) {
					int yptV = (int)(vScale*(n->bufferV[i]-n->vmin));
					if (yptV >= imgSizeY) yptV = imgSizeY-1;
					if (yptV < 0) yptV = 0;
					*((uchar*) CV_MAT_ELEM_PTR( *imgWinV, imgSizeY - 1 - yptV, xpt)) = 255.0;
				}

				if (n->type & PROBE_CURRENT) {
					int yptI = (int)(iScale*(n->bufferI[i])-n->imin);
					if (yptI >= imgSizeY) yptI = imgSizeY-1;
					if (yptI < 0) yptI = 0;
					*((uchar*) CV_MAT_ELEM_PTR( *imgWinI, imgSizeY - 1 - yptI, xpt)) = 255.0;
				}

				if (n->type & PROBE_HOMEO_RATE) {
					int yptH = (int)(hScale*(n->bufferHomeo[i])-hfmin);
					if (yptH >= imgSizeY) yptH = imgSizeY-1;
					if (yptH < 0) yptH = 0;
					*((uchar*) CV_MAT_ELEM_PTR( *imgWinH, imgSizeY - 1 - yptH, xpt)) = 255.0;
				}

				if (n->type & PROBE_FIRING_RATE) {
					int yptF = (int)(fScale*(n->bufferFRate[i])-fmin);
					if (yptF >= imgSizeY) yptF = imgSizeY-1;
					if (yptF < 0) yptF = 0;
					*((uchar*) CV_MAT_ELEM_PTR( *imgWinF, imgSizeY - 1 - yptF, xpt)) = 255.0;
				}
			}
			if (n->type & PROBE_VOLTAGE) 	 cvShowImage(winNameV, imgWinV);
			if (n->type & PROBE_CURRENT) 	 cvShowImage(winNameI, imgWinI);
			if (n->type & PROBE_HOMEO_RATE)  cvShowImage(winNameH, imgWinH);
			if (n->type & PROBE_FIRING_RATE) cvShowImage(winNameF, imgWinF);
			n = n->next;
			prevGid = gid;
		}

		firstSet = 1;

	}


typedef struct fileInfo_s {
	FILE*    fp;
	uint32_t simTime;
} fileInfo_t;


	// close files and also clear all data used
	//   by raster generation function 
	void CpuSNN::clearRasterData()
	{
		DBG(2, fpLog, AT, "clearRasterData()");

		map<string, fileInfo_t*>::const_iterator itr;

		for(itr = rasterFileInfo.begin(); itr != rasterFileInfo.end(); ++itr){
			fileInfo_t* rInfo = (*itr).second;
			fclose(rInfo->fp);
			delete rInfo;
		}
		rasterFileInfo.erase(rasterFileInfo.begin(), rasterFileInfo.end());
	}


	void CpuSNN::rasterPlotAll(int row, int col, bool newFileEveryTime)
	{
		DBG(2, fpLog, AT, "rasterPlotAll() called");

		// its always best to store the results in a single file...
		assert(newFileEveryTime == false);

		if ((currentMode == GPU_MODE) && (BLK_CONFIG_VERSION))  {
			//rasterPlot(-1, fname, row, dontOverWrite);
			for (int monId=0; monId < numSpikeMonitor; monId++) {
				string& fname = monBufferFileName[monId];
				FILE* fp = getRasterFilePointer(fname, newFileEveryTime);
				monBufferFp[monId] = fp;
			}

			// now we got the file pointer.. go and dump
			// if fp=null, the GPU will atleast update
			// the CVMat array with spike info to be displayed later.
			// if fp=null, then the CPU will mostly just return without
			// dumpign the spike information to the file..
			dumpSpikeBuffToFile(NULL, -1);

			for (int monId=0; monId < numSpikeMonitor; monId++) {
				if(monBufferFp[monId])
					fflush(monBufferFp[monId]);
			}

			// enable plotting..
			if(plotUsingOpenCV) {
				for (int i=0; i < numSpikeMonitor; i++) {
					int gid = monGrpId[i];
					plotSpikeBuff(gid, row, col);
				}
			}
		}
		else {
			// when running on CPU mode, we dump info. separately.
			for (int monId=0; monId < numSpikeMonitor; monId++) {
				string& fname = monBufferFileName[monId];
				int   gid   = monGrpId[monId];
				rasterPlot(gid, fname, row, col, newFileEveryTime);
			}
		}
	}

	FILE* CpuSNN::getRasterFilePointer(const string& fname, bool newFileEveryTime)
	{
		DBG(2, fpLog, AT, "getRasterPointer() called");
		FILE* fpDumpFile = NULL;

		assert(newFileEveryTime == false);

#define MAXIMUM_OPEN_RASTER_FILE 100

		assert(rasterFileInfo.size() < MAXIMUM_OPEN_RASTER_FILE);

		if (fname != "")
		{
				fileInfo_t* rInfo=NULL;
				string new_fname(fname);

				if (rasterFileInfo.count(fname)) {
					rInfo = rasterFileInfo[fname];
				}

				// file does not exists
				if (rInfo==NULL) {
					rInfo = new fileInfo_t;
					rInfo->simTime = -1;
					rInfo->fp = NULL;
					rasterFileInfo[fname] = rInfo;
				}

				if (newFileEveryTime) {
					if (rInfo->simTime != simTime) {
						if (rInfo->fp != NULL)
							fclose(rInfo->fp);
						appendFileName(new_fname, simTime);
						rInfo->fp = fopen(new_fname.c_str(), "w");
					}
				}
				else {
					if (rInfo->fp == NULL) {
						rInfo->fp = fopen(fname.c_str(), "w");
					}
				}

			fpDumpFile = rInfo->fp;
		}

		return fpDumpFile;

	}


	void CpuSNN::plotFiringRate(const string& fname, int x, int y, int y_limit)
	{
		FILE* fp = fopen(fname.c_str(), "a");

		plotFiringRate(fp, x, y, y_limit);

		fclose(fp);
	}

	void CpuSNN::plotFiringRate(FILE* fpAvg, int x, int y, int y_limit)
	{

#if _DEBUG
		return;
#endif
		//FILE* fpAvg = fopen("avg_firing.txt", "a");
		if(fpAvg) fprintf(fpAvg, "#-------------------\n");
		int xImgPos = x;
		int yImgPos = y;
		// display the post connectivity information as an image
		for(int g=0; g < numGrp; g++) {
			int sizeX = 128;
			int sizeY = 128;
			int scaleX = sizeX/grp_Info2[g].numX;
			int scaleY = sizeY/grp_Info2[g].numY;
			sizeX = (scaleX <= 0) ? grp_Info2[g].numX: sizeX;
			sizeY = (scaleY <= 0) ? grp_Info2[g].numY: sizeY;
			if(scaleX <=0) scaleX = 1;
			if(scaleY <=0) scaleY = 1;
			Mat m(Size(sizeX, sizeY), CV_8UC1);
			float maxFiring = 50.0f;
			float scaleFactor = 255.0/maxFiring;
			//fprintf(stderr, "sizex=%d, sizey=%d, scalex=%d, scaley=%d\n", sizeX, sizeY, scaleX, scaleY);
			bool poissonType = grp_Info[g].Type&POISSON_NEURON;
			bool excitatoryType = (grp_Info[g].Type&TARGET_AMPA) || (grp_Info[g].Type&TARGET_NMDA);
//			if (poissonType) {
//				fprintf(stderr, "poisson group = %s\n", grp_Info2[g].Name.c_str());
//			}
			int nid = grp_Info[g].StartN;
			float totAvg=0.0;
			float printAvg=0.0;
			for (int j=0; (j < grp_Info2[g].numY); j++) {
				for (int i=0; i < grp_Info2[g].numX; i++, nid++) {
					float avg = nSpikeCnt[nid];
					totAvg += avg;
					for (int x=0; x < scaleX; x++) {
						for (int y=0; y < scaleY; y++) {
							int row = scaleY*j+y;  assert(row < sizeX);
							int col = scaleX*i+x;  assert(col < sizeY);
							m.at<char>(row,col)= (char) ((int) avg*scaleFactor)&0xff;
						}
					}
					if ((!poissonType) && (excitatoryType))
						if((i== grp_Info2[g].numX/2) && (j == grp_Info2[g].numY/2))
							if(fpAvg) fprintf(fpAvg, "%3d %3d : %3d => %03.3f (%s)\n", nid, i, j, avg, grp_Info2[g].Name.c_str());
				}
			}

//			if ((!poissonType) && (IS_EXCITATORY(nid, numNInhPois, numNReg, numNExcReg, numN)))
//					fprintf(fpAvg, "%3d %3d : %3d => %03.3f \n", nid, p_Info2[g].numX/2, p_Info2[g].numY/2, totAvg);

			string name(grp_Info2[g].Name);
			imshow(name+"-conn", m);
			cvMoveWindow((name+"-conn").c_str(), xImgPos, yImgPos);

			yImgPos = yImgPos+sizeX + 30;
			if ( yImgPos > y_limit ) {
				yImgPos = y;
				xImgPos = xImgPos + sizeY + 10;
			}
		}
	}




		if(plotUsingOpenCV) {
			// wait for a key
			keyVal = cvWaitKey(10);
//			if(keyVal != -1) {
//				fprintf(stderr, "keyVal = %c %d %d %d \n", (char) keyVal, keyVal, (int) 'p', (int) 's', (int) 'q');
//				cvWaitKey(0);
//			}
			if ((char) keyVal == 'p') {
				fprintf(stderr, "Pause key pressed ! Press any to continue\n");
				rasterPlotAll(10, false);
				cvWaitKey(0);
			}
			else if( (char) keyVal == 's') {
				//saveConnectionWeights();
				rasterPlotAll(10, false);
				cvWaitKey(0);
			}
			else if( (char) keyVal == 'q') {
				cvWaitKey(0);
				exitSimulation(0);
			}
			else if( (char) keyVal ==  'b') {
				return keyVal;
			}
			else if( (char) keyVal ==  'n') {
				return keyVal;
			}
		}




	void dumpToFile(CvMat* mat)
	{
		static int cnt = 0;

		//std::stringstream fname;
		//fname << "dumpCvMat" << cnt++ << ".txt";
		char fname[100];
		sprintf(fname, "dumpCvMat%d.txt", cnt++);
		FILE* fp = fopen(fname, "w");

		if (((mat->type&0xff)==CV_32FC1)) {
			fprintf(stdout, "Values are %d\n", CV_32FC1);
			for(int row=0; row<mat->rows; row++ ) {
		    	const float* ptr = (const float*)(mat->data.ptr + row * mat->step);
		    		for( int col=0; col<mat->cols; col++ ) {
		    			fprintf(fp, "%2.1f ", *ptr);
		    			ptr++;
		    		}
		    		fprintf(fp, "\n");
			}
		}
		else if ((mat->type&0xff)==CV_8UC1) {
			int k=2;
			while(k > 0) {
				fprintf(stdout, "Values are %d\n", CV_8UC1);
				for(int row=0; row<mat->rows; row++ ) {
			    	const char* ptr = (const char*)(mat->data.ptr + row * mat->step);
			    		for( int col=0; col<mat->cols; col++ ) {
			    			if(((k==2) && (*ptr > 0)) || ((k==1) && (*ptr < 0)))
			    			  	fprintf(fp, "   ");
			    			else
			    				fprintf(fp, "%x ", ((int) *ptr)&0xff);
			    			ptr++;
			    		}
			    		fprintf(fp, "\n");
				}
				k=k-1;
			}
		}

		fclose(fp);
	}

	CvMat* getRampPattern ( int size, float min, float max, string rampType, float freq, CvMat* mat)
	{
		if (mat==NULL)
			mat = cvCreateMat(1, size, CV_32FC1);

#if _DEBUG_
		fprintf(stdout, " Printing Ramp Patter Rates : \n");
#endif

		if(rampType.find("ramp") != string::npos) {
			float step = (max - min)/size;
			float rate = min;
			for(int row=0; row<mat->rows; row++ ) {
				float* ptr = (float*)(mat->data.ptr + row * mat->step);
				for( int col=0; col<mat->cols; col++) {
					*ptr = rate; ptr++; rate += step;
				}
			}
		}
		else if ( rampType.find("sine") != string::npos) {
			float range = (max - min)/2;
			float sinFactor = (2*3.14159265)/(size/freq);
			for(int row=0; row<mat->rows; row++ ) {
				float* ptr = (float*)(mat->data.ptr + row * mat->step);
				for( int col=0; col<mat->cols; col++) { 
					*ptr = range + range*sin(sinFactor*col); 
#if _DEBUG_
				fprintf(stdout, " %f ", *ptr);
#endif
					ptr++; 
				}
			}
		}
#if _DEBUG_
		fprintf( stdout, " Printing Ramp Patter Rates : \n");
#endif
		fflush(stdout);
		return mat;
	}

	int setGaborParameters(GaborFilter* gb, int radius, float angle)
	{
		int factor=1;
		//float angle = angles[(int)((sizeof(angles)/4)*drand48())]; //45.0;//   = atof(argv[2]);
		float freq  = 1/10.0; //0.1105/factor;   // atof(argv[3]);
		float bw    = 1.0; //   atof(argv[4]);
		float phase = 0.0; // atof(argv[5]);
		float rho   = 1.5; //atof(argv[6]);
		float sig   = 2*factor;
		gb->GenerateFilter(radius, angle, freq, sig, phase, bw, rho);
		return 0;
	}

	int displayPattern (GaborFilter *gb, CvMat* spikeRate)
	{
		static CvMat* imgWin 	 = cvCreateMat(160, 160, CV_8UC1);
		static bool firstDisplay = true;
		char winName[]="mainWin";

		CvMat* spikePixel = cvCreateMat(spikeRate->rows, spikeRate->cols, CV_8UC1);

		double minVal, maxVal;
		cvMinMaxLoc(spikeRate, &minVal, &maxVal);

		for(int i=0; i < spikeRate->rows; i++) {
			for (int j=0; j < spikeRate->cols; j++) {
				CvScalar rate         = cvGet2D(spikeRate, i, j);
				double val = (int) 255*rate.val[0]/maxVal;
				CvScalar pixel;
				pixel.val[0] = val;
				cvSet2D(spikePixel, i, j, pixel);
			}
		}

		// create a window
		cvNamedWindow(winName);

		if (firstDisplay) {
			cvMoveWindow("mainWin", 300, 500);
			firstDisplay = false;
		}

		cvResize(spikePixel, imgWin);

		cvShowImage(winName, imgWin);

		return 0;
	}
	*/
