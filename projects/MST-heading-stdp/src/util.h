#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

float calcMean(float *in, int size);

float** loadData(string file, int numRow, int numCol);

int randGenerator (int i);

void shuffleTrials(int numTrials, int numTrain, int numTest, int *trainTrials, int *testTrials);

void calcCorrCoef(float** X, float** Y, int numRow, int numCol, float* corrCoef);

float calcPopCorrCoef(float* x,float* y, int length);
