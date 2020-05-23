  #include <assert.h>
#include <cmath>
#include <vector>
#include <stdio.h>

#include "helper.h"
#include "util.h"

using namespace std;

void reconstructMT(vector<vector<float> > weights, vector<float> FRs, float* recMT) {
    int wNumRow = weights.size();
    int wNumCol = weights[0].size();
    int numNeur = FRs.size();

    assert (wNumCol == numNeur);

	float productSum;

    for (unsigned int i = 0; i < wNumRow; i ++) {
    	productSum = 0.0;
    	for (unsigned int j = 0; j < wNumCol; j ++) {
    		if (isnan(weights[i][j]) || weights[i][j] < 0) {
    			weights[i][j] = 0;
    		}
    		productSum += weights[i][j] * FRs[j];
	    }
	    recMT[i] = productSum;
    }
}  

void sortMTResponses(float** MTData, int trial, int dimFlow, int numNeurPerPixel, float* out_array) {

	float MTArray[dimFlow][dimFlow][numNeurPerPixel];

	int ind = 0;
	for (unsigned n = 0; n < numNeurPerPixel; n++) {
		for (unsigned y = 0; y < dimFlow; y ++) {
			for (unsigned x = 0; x < dimFlow; x++) {
				MTArray[x][y][n] = MTData[ind][trial];
				ind++;
			}
		}
	}
	ind = 0;
	for (unsigned n = 0; n < numNeurPerPixel; n++) {
		for (unsigned x = 0; x < dimFlow; x++) {
			for (unsigned y = 0; y < dimFlow; y++) {
				out_array[ind] = MTArray[dimFlow-1-x][y][n];
				ind++;
			}
		}
	}
}



