#include <PTI.h>
#include <carlsim.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <ctime>

#include "util.h"
#include "helper.h"

using namespace std;
using namespace CARLsim_PTI;

namespace CARLsim_PTI {
    class MSTHeadingExperiment : public Experiment {
    public:
    	MSTHeadingExperiment() {}

		void run(const ParameterInstances &parameters, std::ostream &outputStream) const {

			const float COND_tAMPA=5.0, COND_tNMDA=150.0, COND_tGABAa=6.0, COND_tGABAb=150.0;
			const float REG_IZH[] = { 0.02f, 0.2f, -65.0f, 8.0f };
			const float FAST_IZH[] = { 0.1f, 0.2f, -65.0f, 2.0f };
			const float alpha = 1.0; // homeostatic scaling factor
			const float T = 10.0; // homeostatic time constant

			int randSeed = 42;	

			bool loadNetwork = true;
			bool writeRes = true;
			string data_dir_root = "./data/";
			string result_dir_root = "./results/";

			// MT group dimensions 
			int gridDim = 15; // dimension of flow fields
			int nNeuronPerPixel = 8;
			Grid3D MTDim(gridDim, gridDim, nNeuronPerPixel); 
			int nMT = gridDim * gridDim * nNeuronPerPixel; // 1800 MT neurons

			// MST group dimensions 
			int nMSTDim = 8;
			Grid3D MSTDim(nMSTDim, nMSTDim, 1);
			int nMST = nMSTDim * nMSTDim; // 64 MST neurons
			
			// Inh group dimensions 
			Grid3D inhDim(nMSTDim, nMSTDim, nNeuronPerPixel);
			int nInh = nMSTDim * nMSTDim * nNeuronPerPixel; 

			// neuron groups
			int numIndi = parameters.getNumInstances(); // number of network individuals
			int gMT;
			int gMST[numIndi];
			int gInh[numIndi];

			// inter-group connections
			short int mtToMst[numIndi];
			short int mtToInh[numIndi];
			short int inhToMst[numIndi];

			// network run time
			int runTimeSec = 0; //seconds
			int runTimeMs = 500; //milliseconds

			float poissBaseRate = 20.0f;
			float targetMaxFR = 250.0f;

			string dataFile = "V-one-speed-sorted.csv";
			string MTDataFile = (data_dir_root + dataFile);

			// training and testing parameters
			int totalSimTrial = 6000;
			int numTrial = 500;
			int numTrain = int(numTrial * 0.8);
			int numTest = numTrial - numTrain;
			int trial;
			// arrays to store randomized test and train indices
			int trainTrials[numTrain];
			int testTrials[numTest];

			float** MTData; // array to store input MT FR
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

			vector<vector<float> > sumNeurFR(numIndi, vector<float>(nMST, 0.0f)); 
			float maxFR; // FR threshold

			vector<vector<vector<float> > > weights(numIndi); // 3D vector to store connection weights for each individual
			vector<vector<float> > MSTSpikeRates(numIndi); // vector to store MST FR for each individual

		    // 2D array to store correlation between input and reconstructed stimulis
			float** popCorrCoef;
    		popCorrCoef = new float*[numIndi];
		    for (int i = 0; i < numIndi; i ++) {
		        popCorrCoef[i] = new float[numTest];
		    }

			float popFitness[numIndi];

			SpikeMonitor* SpikeMonMST[numIndi];
			ConnectionMonitor* CMMtToMst[numIndi];
			
			MTData = loadData(MTDataFile, nMT, totalSimTrial); // Load MT response 

			// ---------------- CONFIG STATE ------------------- 
			// CARLsim* const network = new CARLsim("MST-heading", GPU_MODE, SILENT);
			CARLsim* const network = new CARLsim("MST-heading", GPU_MODE, SILENT, 1, randSeed);

			gMT = network->createSpikeGeneratorGroup("MT", MTDim, EXCITATORY_POISSON); //input
			for (unsigned int i = 0; i < numIndi; i++) {
				// creat neuron groups
				gMST[i] = network->createGroup("MST", MSTDim, EXCITATORY_NEURON);
				gInh[i] = network->createGroup("inh", inhDim, INHIBITORY_NEURON);

				network->setNeuronParameters(gMST[i], REG_IZH[0], REG_IZH[1], REG_IZH[2], REG_IZH[3]);
				network->setNeuronParameters(gInh[i], FAST_IZH[0], FAST_IZH[1], FAST_IZH[2], FAST_IZH[3]);
				network->setConductances(true, COND_tAMPA, COND_tNMDA, COND_tGABAa, COND_tGABAb);

				float gaussRadius = parameters.getParameter(i,17);

				mtToMst[i] = network->connect(gMT, gMST[i], "gaussian", RangeWeight(0.0f,parameters.getParameter(i,14), parameters.getParameter(i,14)), 1.0f, RangeDelay(1), RadiusRF(gaussRadius,gaussRadius,-1), SYN_PLASTIC);
				mtToInh[i] = network->connect(gMT, gInh[i], "random", RangeWeight(0.0f,parameters.getParameter(i,15), parameters.getParameter(i,15)), 0.1f, RangeDelay(1), -1, SYN_PLASTIC);
				inhToMst[i] = network->connect(gInh[i], gMST[i], "random", RangeWeight(0.0f,parameters.getParameter(i,16), parameters.getParameter(i,16)), 0.1f, RangeDelay(1), -1, SYN_PLASTIC);

				// set E-STDP to be STANDARD (without neuromodulatory influence) with an EXP_CURVE type.
				// network->setESTDP(gMST[i], true, STANDARD, ExpCurve(parameters.getParameter(i,0), parameters.getParameter(i,6), parameters.getParameter(i,3), parameters.getParameter(i,7)));
				// network->setESTDP(gInh[i], true, STANDARD, ExpCurve(parameters.getParameter(i,1), parameters.getParameter(i,8), parameters.getParameter(i,4), parameters.getParameter(i,9)));
				// network->setISTDP(gMST[i], true, STANDARD, ExpCurve(parameters.getParameter(i,5), parameters.getParameter(i,10), parameters.getParameter(i,2), parameters.getParameter(i,11)));
				
				network->setESTDP(gMST[i], true, STANDARD, ExpCurve(parameters.getParameter(i,0), parameters.getParameter(i,6), parameters.getParameter(i,2), parameters.getParameter(i,7)));
				network->setESTDP(gInh[i], true, STANDARD, ExpCurve(parameters.getParameter(i,1), parameters.getParameter(i,8), parameters.getParameter(i,3), parameters.getParameter(i,9)));
				network->setISTDP(gMST[i], true, STANDARD, ExpCurve(parameters.getParameter(i,4), parameters.getParameter(i,10), parameters.getParameter(i,5), parameters.getParameter(i,11)));

				// set homeostasis
				network->setHomeostasis(gMST[i], true, alpha, T); 
				network->setHomeostasis(gInh[i], true, alpha, T);
				network->setHomeoBaseFiringRate(gMST[i], parameters.getParameter(i,12), 0);
				network->setHomeoBaseFiringRate(gInh[i], parameters.getParameter(i,13), 0);
			}


			// ---------------- SETUP STATE -------------------
			FILE* fId = NULL;
			if (loadNetwork) {
				fId = fopen("sim_MST-heading-stdp.dat", "rb");
				network->loadSimulation(fId);
				network->setupNetwork();
				fclose(fId);
			}
			else {
				network->setupNetwork();
			}

			PoissonRate* const poissRate = new PoissonRate(nMT, true);

			// naming for monitors
			string spk_name_prefix = "spk_MST_";
			string conn_name_prefix = "conn_MT_MST_";
			string name_suffix = ".dat";
			string name_id;
			string spk_monitor_name;
			string conn_monitor_name;

			for (int i = 0; i < numIndi; i++) {
				stringstream name_id_ss;
				name_id_ss << i;
				name_id = name_id_ss.str();

				spk_monitor_name = (result_dir_root + spk_name_prefix + name_id + name_suffix);
				SpikeMonMST[i] = network->setSpikeMonitor(gMST[i], spk_monitor_name);
				SpikeMonMST[i]->setPersistentData(false);

				conn_monitor_name = (result_dir_root + conn_name_prefix + name_id + name_suffix);
				CMMtToMst[i] = network->setConnectionMonitor(gMT, gMST[i], conn_monitor_name);
				CMMtToMst[i]->setUpdateTimeIntervalSec(-1);
			}

			shuffleTrials(totalSimTrial, numTrain, numTest, trainTrials, testTrials); 

			// ---------------- RUN STATE -------------------
			if (!loadNetwork) {

				/*** TRAINING - run network with MT activities on 80% trials ***/
				for (unsigned int tr = 0; tr < numTrain; tr++) {
					trial = trainTrials[tr];

					// set spike rates for the input group
					for (unsigned int neur = 0; neur < nMT; neur ++) {
						poissRateVector.push_back(MTData[neur][trial]*poissBaseRate);
					}
					poissRate->setRates(poissRateVector);
					poissRateVector.clear();
					network->setSpikeRate(gMT, poissRate);

					// run network with stimulus
					network->runNetwork(runTimeSec, runTimeMs);

					// run network for same amount of time with no stimulus
					poissRate->setRates(0.0f);
					network->setSpikeRate(gMT, poissRate);
					network->runNetwork(runTimeSec, runTimeMs);
				}
				network->saveSimulation("sim_MST-heading-stdp.dat", true);
			}

			/******************* TESTING ***********************/ 
			network->startTesting(); // turn off STDP-H

			std::ofstream fileFitness; 
			string fitFileName = (result_dir_root + "fitness.txt");
	    	fileFitness.open(fitFileName.c_str(), std::ofstream::out | std::ofstream::app); 
			
			// file to store trial numbers
	    	std::ofstream fileTrial; 
			string trialFileName = (result_dir_root + "trials.csv");
	    	fileTrial.open(trialFileName.c_str(), std::ofstream::out | std::ofstream::app);

			std::ofstream fileMST; 
	    	std::ofstream fileWts; 

			if (writeRes) {
				string mstFileName = (result_dir_root + "MST-fr.csv");
		    	fileMST.open(mstFileName.c_str(), std::ofstream::out | std::ofstream::app); 
			
				string wtsFileName = (result_dir_root + "weights.csv");
		    	fileWts.open(wtsFileName.c_str(), std::ofstream::out | std::ofstream::app);
		    }

	    	// MT test data
	    	for (unsigned int tr = 0; tr < numTest; tr++) {
				trial = testTrials[tr];
				fileTrial << trial << ",";
				for (unsigned int neur = 0; neur < nMT; neur ++) {
					testMTMatrix[tr][neur] = MTData[neur][trial];
				}
			}	
			fileTrial << endl;		

			for (unsigned int i = 0; i < numIndi; i++) {
				weights[i] = CMMtToMst[i]->takeSnapshot();	
				if (writeRes) {
					for (unsigned int m = 0; m < nMST; m ++) {
						for (unsigned int n = 0; n < nMT; n ++) {
							fileWts << weights[i][m][n] << ",";
						}
					}
					fileWts << endl;
				}
			}
				
	    	for (unsigned int tr = 0; tr < numTest; tr++) {

	    		// set spike rates for the input group
				for (unsigned int neur = 0; neur < nMT; neur ++) {
					poissRateVector.push_back(testMTMatrix[tr][neur]*poissBaseRate);
				}
				poissRate->setRates(poissRateVector);
				poissRateVector.clear();					
				network->setSpikeRate(gMT, poissRate);

				// run network with stimulus and record MST activity
				for (unsigned int i = 0; i < numIndi; i++) {
					SpikeMonMST[i]->startRecording();
				}

				network->runNetwork(runTimeSec, runTimeMs);

				for (unsigned int i = 0; i < numIndi; i++) {
					SpikeMonMST[i]->stopRecording();
					MSTSpikeRates[i] = SpikeMonMST[i]->getAllFiringRates();	

					// summing FR of each MST neuron across trials (calculate mean firing rates at the end)
					for (unsigned int neur = 0; neur < nMST; neur ++) {
						sumNeurFR[i][neur] += MSTSpikeRates[i][neur];
						if (writeRes) {
							// fileMST << MSTSpikeRates[i][neur] << ",";
						}
					}
					// reconstruct input by taking the dot product of W and MST
					reconstructMT(weights[i], MSTSpikeRates[i], recMTAll[tr]);	
					// calculate correlation between input and reconstructed MT for each stimulus
					popCorrCoef[i][tr] = calcPopCorrCoef(recMTAll[tr], testMTMatrix[tr], nMT);

					MSTSpikeRates[i].clear();
				}
				if (writeRes) {
					fileMST << endl;
				}

				// run network for same amount of time with no stimulus
				poissRate->setRates(0.0f);
				network->setSpikeRate(gMT, poissRate);
				network->runNetwork(runTimeSec, runTimeMs);
			}

			for (unsigned int i = 0; i < numIndi; i++) {
				popFitness[i] = calcMean(popCorrCoef[i], numTest);

				maxFR = 0.0f;
				for (unsigned int neur = 0; neur < nMST; neur ++) {
					// calculate the mean FR for each neuron across trials
					sumNeurFR[i][neur] /= numTest;
					// find the max mean FR in MST group
					if (sumNeurFR[i][neur] > maxFR) {
						maxFR = sumNeurFR[i][neur];
					}
				}
				// if max mean FR greater than target max FR, subtract max mean from target
				// add punishment to fitness
				if (maxFR > targetMaxFR) {
					// fitness[i] -= ((maxFR - targetMaxFR) / nMST);
					popFitness[i] -= ((maxFR - targetMaxFR) / nMST);
				}

				// fileFitness << "fitness-" << i << ": " << fitness[i] << endl;
				fileFitness << "popFitness-" << i << ": " << popFitness[i] << endl;

				// outputStream << fitness[i] << endl;
				outputStream << popFitness[i] << endl; // send correlation between input and reconstructed stimulus to ECJ
			}

			network->stopTesting();
			fileFitness.close();
			fileTrial.close();

			if (writeRes) {
				fileMST.close();
				fileWts.close();
			}
		}
	};
}

int main(int argc, char* argv[]) {

	const MSTHeadingExperiment experiment;
	const PTI pti(argc, argv, std::cout, std::cin);

	pti.runExperiment(experiment);

	return 0;
}

