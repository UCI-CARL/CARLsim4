#include <carlsim.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "util.h"
#include "helper.h"

using namespace std;

int main(int argc, char *argv[]) {

	for (int load_network = 0; load_network < 2; load_network++) {
		// int load_network = 1;

		int paramInd = 0;
		int numNetwork = 1;
		int numParams = 18;
		float** parameters;

		const float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
		const float REG_IZH[] = { 0.02f, 0.2f, -65.0f, 8.0f };
		const float FAST_IZH[] = { 0.1f, 0.2f, -65.0f, 2.0f };
		const float alpha = 1.0; // homeostatic scaling factor
		const float T = 10.0; // homeostatic time constant

		// MT group dimensions 
		int gridDim = 15; // dimension of flow fields
		int nNeuronPerPixel = 1;
		Grid3D MTDim(gridDim, gridDim, nNeuronPerPixel); 
		int nMT = gridDim * gridDim * nNeuronPerPixel; // 1800 MT neurons

		// MST group dimensions 
		int nMSTDim = 1;
		Grid3D MSTDim(nMSTDim, nMSTDim, 1);
		int nMST = nMSTDim * nMSTDim; // 64 MST neurons
		
		// Inh group dimensions 
		Grid3D MSTInhDim(nMSTDim, nMSTDim, nNeuronPerPixel);
		int nMSTInh = nMSTDim * nMSTDim * nNeuronPerPixel; // 8 x 8 x 8 = 512 inhibitory neurons

		// neuron groups
		int gMT;
		int gMST;
		int gMSTInh;

		// inter-group connections
		short int mtToMst;
		short int mtToInh;
		short int inhToMst;

		// network run time
		int runTimeSec = 0; //seconds
		int runTimeMs = 500; //milliseconds

		float poissBaseRate = 20.0f;
		float targetMaxFR = 250.0f;

		string trainMTDataFile = "./data/V-one-speed-sorted.csv";
		// string testMTDataFile = "./data/pursuit-V-large.csv";
		string paramsFile = "./data/params.csv";

		// training and testing parameters
		int totalNumTrial = 6000;
		int numTrial = 5;
		int numTrain = int(numTrial * 0.8);
		numTrain = 1;
		int numTest = numTrial - numTrain;
		// int numTest = 100;
		// int numNew = 10000;

		int trial;
		int trainTrials[numTrain];
		// int trainDataTrainTrials[numTrain];
		int testTrials[numTest];
		// int testDataTestTrials[numTest];

		float** trainMTData; // array to store input MT FR
		float** testMTData; // array to store input MT FR
		float sortedMTArray[nMT]; // vector to store sorted MT
		vector<float> poissRateVector; // MT FR * 20.0f

		//array to store sorted MT for all test trials
		float** testMTMatrix;
		testMTMatrix = new float*[numTest];
		for (int i = 0; i < numTest; i ++) {
			testMTMatrix[i] = new float[nMT];
		}
		// array to store reconstructed MT
		float** recMTAll;
		recMTAll = new float*[numTest];
		for (int i = 0; i < numTest; i ++) {
			recMTAll[i] = new float[nMT];
		}

		vector<float> sumNeurFR(nMST, 0.0f);
		float maxFR;

		vector<vector<float> > weights; // 2D vector to store connection weights between MT and MST

		vector<float> MSTSpikeRates; // vector to store MST FR
		float corrCoef[nMT];
		float fitness;

		SpikeMonitor* SpikeMonMST;
		ConnectionMonitor* CMMtToMst;
		
		trainMTData = loadData(trainMTDataFile, nMT, totalNumTrial); // Load MT response 
		// testMTData = loadData(testMTDataFile, nMT, 10000);
		parameters = loadData(paramsFile, numNetwork, numParams);

		// ---------------- CONFIG STATE ------------------- 
		int seed = 1774820947;
		CARLsim* const network = new CARLsim("MST-heading-opt", GPU_MODE, USER, 1, seed);

		// creat neuron groups
		gMT = network->createSpikeGeneratorGroup("MT", MTDim, EXCITATORY_POISSON);
		gMST = network->createGroup("MST", MSTDim, EXCITATORY_NEURON);
		gMSTInh = network->createGroup("inh", MSTInhDim, INHIBITORY_NEURON);

		network->setNeuronParameters(gMST, REG_IZH[0], REG_IZH[1], REG_IZH[2], REG_IZH[3]);
		network->setNeuronParameters(gMSTInh, FAST_IZH[0], FAST_IZH[1], FAST_IZH[2], FAST_IZH[3]);
		network->setConductances(true, COND_tAMPA, COND_tNMDA, COND_tGABAa, COND_tGABAb);

		float gaussRadius = parameters[paramInd][17];

		mtToMst = network->connect(gMT, gMST, "gaussian", RangeWeight(0.0f,parameters[paramInd][14], parameters[paramInd][14]), 1.0f, RangeDelay(1), RadiusRF(gaussRadius,gaussRadius,-1), SYN_PLASTIC);
		mtToInh = network->connect(gMT, gMSTInh, "random", RangeWeight(0.0f,parameters[paramInd][15], parameters[paramInd][15]), 0.1f, RangeDelay(1), -1, SYN_PLASTIC);
		inhToMst = network->connect(gMSTInh, gMST, "random", RangeWeight(0.0f,parameters[paramInd][16], parameters[paramInd][16]), 0.1f, RangeDelay(1), -1, SYN_PLASTIC);

		// set E-STDP to be STANDARD (without neuromodulatory influence) with an EXP_CURVE type.
		network->setESTDP(gMST, true, STANDARD, ExpCurve(parameters[paramInd][0], parameters[paramInd][6], parameters[paramInd][2], parameters[paramInd][7]));
		network->setISTDP(gMST, true, STANDARD, ExpCurve(parameters[paramInd][1], parameters[paramInd][8], parameters[paramInd][3], parameters[paramInd][9]));
		network->setESTDP(gMSTInh, true, STANDARD, ExpCurve(parameters[paramInd][4], parameters[paramInd][10], parameters[paramInd][5], parameters[paramInd][11]));

		// set homeostasis
		network->setHomeostasis(gMST, true, alpha, T); 
		network->setHomeostasis(gMSTInh, true, alpha, T);
		network->setHomeoBaseFiringRate(gMST, parameters[paramInd][12], 0);
		network->setHomeoBaseFiringRate(gMSTInh, parameters[paramInd][13], 0);

		// ---------------- SETUP STATE -------------------]

		FILE* fId = NULL;
		if (load_network) {
			fId = fopen("sim_MST-heading-opt.dat", "rb");
			network->loadSimulation(fId);
			network->setupNetwork();
			fclose(fId);
		}
		else {
			network->setupNetwork();
		}
		
		PoissonRate* const poissRate = new PoissonRate(nMT, true);

		//set up monitors
		string spk_monitor_name = "results/spk_MST_analysis.dat";
		string conn_monitor_name = "results/conn_MT_MST_analysis.dat";
		SpikeMonMST = network->setSpikeMonitor(gMST, spk_monitor_name);

		CMMtToMst = network->setConnectionMonitor(gMT, gMST, conn_monitor_name);
		CMMtToMst->setUpdateTimeIntervalSec(-1);

		// // ---------------- RUN STATE -------------------
		shuffleTrials(totalNumTrial, numTrain, numTest, trainTrials, testTrials); 

		// /*** TRAINING - run network with MT activities on 80% trials ***/

		if (!load_network) {
			cout << "================ Training starts ===========================" << endl;
			
			for (unsigned int tr = 0; tr < numTrain; tr++) {
				trial = trainTrials[tr];
				// sort MT responses to match with the dimensions of the neuron group
				sortMTResponses(trainMTData, trial, gridDim, nNeuronPerPixel, sortedMTArray);

				for (unsigned int neur = 0; neur < nMT; neur ++) {
					poissRateVector.push_back(sortedMTArray[neur]*poissBaseRate);
				}
				// run network with stimulus
				poissRate->setRates(poissRateVector);
				poissRateVector.clear();
				network->setSpikeRate(gMT, poissRate);
				network->runNetwork(runTimeSec, runTimeMs);

				float wtChange = CMMtToMst->getPercentWeightsChanged();
				if (wtChange < 50) {
					cout << "pct wt changed: " << wtChange << endl;
				}

				// run network for same amount of time with no stimulus
				poissRate->setRates(0.0f);
				network->setSpikeRate(gMT, poissRate);
				network->runNetwork(runTimeSec, runTimeMs);
				if (tr % 10 == 0) {
					cout << "train#" << tr << endl;	
				}
			}
			cout << "=============== Training completed ============\n";
			network->saveSimulation("sim_MST-heading-opt.dat", true);
	        }	

		
		weights = CMMtToMst->takeSnapshot();

		std::ofstream fileWts; 
		string wtsFileName;
		if (load_network) {
			wtsFileName = ("results/weights_load.csv");
		} else {
			wtsFileName = ("results/weights_save.csv");
		}
		fileWts.open(wtsFileName.c_str(), std::ofstream::out | std::ofstream::app);

		std::vector<std::vector<float> >::iterator vec_it;
		std::vector<float>::iterator it;

		// take a snapshot of the weights before we run the simulation
		// for (vec_it = weights.begin(); vec_it != weights.end(); vec_it++) {
		// 	for (it = (*vec_it).begin(); it != (*vec_it).end(); it++) {
		// 		fileWts << (*it) << ","; 
		// 	}
		// 	fileWts << endl;
		// }
                for (vec_it = weights.begin(); vec_it != weights.end(); vec_it++) {
		    for (it = (*vec_it).begin(); it != (*vec_it).end(); it++) {
			std::cout << (*it) << " "; 
	            }
	        }

		for (unsigned int m = 0; m < nMST; m ++) {
			for (unsigned int n = 0; n < nMT; n ++) {
				fileWts << m << ", " << n << ", " << weights[m][n] << "\n";
			}
			fileWts << endl;
		}
	}

	// /*** TESTING ***/
	// network->startTesting(); // turn off STDP-H

	// std::ofstream fileFitness; // tmp
	// fileFitness.open("./results/fitness.txt", std::ofstream::out | std::ofstream::app); // tmp
	// std::ofstream fileTrial; //tmp
	// fileTrial.open("./results/trials.txt", std::ofstream::out | std::ofstream::app);
	// // std::ofstream fileRecMT; //tmp
	// // fileRecMT.open("./results/recMT.txt", std::ofstream::out | std::ofstream::app);
	// std::ofstream fileMST;
	// fileMST.open("./results/MST-rot-only.csv", std::ofstream::out | std::ofstream::app);

	// // construct MT response matrix for testing
	// for (unsigned int tr = 0; tr < numTest; tr++) {
	// 	// trial = tr;
	// 	trial = testTrials[tr];
	// 	fileTrial << trial << " "; // record test trial indices
	// 	// sort MT responses to match with the dimensions of the neuron group
	// 	sortMTResponses(trainMTData, trial, gridDim, nNeuronPerPixel, sortedMTArray);

	// 	for (unsigned int neur = 0; neur < nMT; neur ++) {
	// 		testMTMatrix[tr][neur] = sortedMTArray[neur];
	// 	}
	// }	
	// fileTrial << endl;	

	// for (unsigned int tr = 0; tr < numTest; tr++) {

	// 	for (unsigned int neur = 0; neur < nMT; neur ++) {
	// 		poissRateVector.push_back(testMTMatrix[tr][neur]*poissBaseRate);
	// 	}
	// 	poissRate->setRates(poissRateVector);
	// 	poissRateVector.clear();					

	// 	network->setSpikeRate(gMT, poissRate);

	// 	SpikeMonMST->clear();
	// 	SpikeMonMST->startRecording();

	// 	network->runNetwork(runTimeSec, runTimeMs);
		
	// 	SpikeMonMST->stopRecording();
	// 	MSTSpikeRates = SpikeMonMST->getAllFiringRates();	
	// 	// summing the FR of each MST neuron across trials
	// 	for (unsigned int neur = 0; neur < nMST; neur ++) {
	// 		sumNeurFR[neur] += MSTSpikeRates[neur];
	// 	}

	// 	for (unsigned int neur = 0; neur < nMST; neur ++) {
	// 		fileMST << MSTSpikeRates[neur] << " ";
	// 	}

	// 	reconstructMT(weights, MSTSpikeRates, recMTAll[tr]);

	// 	fileMST << "\n";

	// 	MSTSpikeRates.clear();

	// 	// run network for same amount of time with no stimulus
	// 	poissRate->setRates(0.0f);
	// 	network->setSpikeRate(gMT, poissRate);
	// 	network->runNetwork(runTimeSec, runTimeMs);
	// 	if (tr % 10 == 0) {
	// 		cout << "test#" << tr << endl;	
	// 	}

	// }

	// calcCorrCoef(recMTAll, testMTMatrix, numTest, nMT, corrCoef);
	// fitness = calcMean(corrCoef, nMT);

	// for (unsigned int neur = 0; neur < nMST; neur ++) {
	// 	// calculate the mean FR for each neuron across trials
	// 	sumNeurFR[neur] /= numTest;
	// 	// max mean FR in the MT group
	// 	if (sumNeurFR[neur] > maxFR) {
	// 		maxFR = sumNeurFR[neur];
	// 	}
	// }
	// // if max mean FR greater than target max FR, subtract max mean from target
	// // and subtract the difference averaged across neurons from fitness
	// if (maxFR > targetMaxFR) {
	// 	fitness -= ((maxFR - targetMaxFR) / nMST);
	// }
	// // for (unsigned int neur = 0; neur < nMT; neur ++) {
	// // 	fileFitness << corrCoef[neur] << " ";
	// // }
	// fileFitness << "fitness-"  << ": "<< fitness << endl;
	// cout << fitness << endl;

	// sumNeurFR.clear();
	// weights.clear();

	// network->stopTesting();
	// fileFitness.close();
	// fileTrial.close();
	// fileMST.close();

}
