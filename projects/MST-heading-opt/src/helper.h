#include <assert.h>
#include <cmath>
#include <vector>
#include <stdio.h>

using namespace std;

void reconstructMT(vector<vector<float> > weights, vector<float> FRs, float* recMT);

void sortMTResponses(float** MTData, int trial, int dimFlow, int numNeurPerPixel, float* out_array);

float calcReconstructError(vector<float> origMT, vector<float> recMT);

