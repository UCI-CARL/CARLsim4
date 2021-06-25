/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
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
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

// include CARLsim user interface
#include <carlsim.h>

// Include libraries that will allow for us to perform vector operations, and
// print their results
#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>
#include <ctime>
#include <cstdlib>

// include stopwatch for timing
#include <stopwatch.h>


int main() {
	// keep track of execution time
	Stopwatch watch;

	// ---------------- CONFIG STATE -------------------

	// create a network on GPU
	int numGPUs = 0;
	int randSeed = 10;
	CARLsim sim("ca3_snn_GPU", GPU_MODE, USER, numGPUs, randSeed);

	// include header file that contains generation of groups and their
	// properties
	#include "../generateCONFIGStateSTP.h"
    
	// Set the time constants for the excitatory and inhibitory receptors, and
	// set the method of integration to numerically solve the systems of ODEs
	// involved in the SNN
	// sim.setConductances(true);
	sim.setIntegrationMethod(RUNGE_KUTTA4, 5);

//     // before calling setupNetwork, call loadSimulation
//     FILE* fId = NULL;
//     fId = fopen("ca3SNN2.dat", "rb");
//     sim.loadSimulation(fId);
    
	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

    // ... wait until after setupNetwork is called
//     fclose(fId);
    
	// include header file that contains generation of groups and their
	// properties
	#include "../generateSETUPStateSTP.h"
    
    // Declare variables that will store the start and end ID for the neurons
    // in the granule group
    int DG_start = sim.getGroupStartNeuronId(DG_Granule);
    int DG_end = sim.getGroupEndNeuronId(DG_Granule);
    int DG_range = (DG_end - DG_start) + 1;

    // Create a vector that is the length of the number of neurons in the granule population
    std::vector<int> DG_vec( boost::counting_iterator<int>( 0 ),
                             boost::counting_iterator<int>( DG_range ));

    // Define the number of granule cells to fire
    int numGranuleFire = 10;

    std::vector<int> DG_vec_A;

    // Define the location of those granule cells so that we choose the same granule cells each time we call setRates
    for (int i = 0; i < numGranuleFire; i++)
    {
        DG_vec_A.push_back(5*(i+1));
    }
    
	// Set the seed of the pseudo-random number generator based on the current system time
	std::srand(std::time(nullptr));
    
    sim.saveSimulation("ca3SNN1.dat", true); // define where to save the network structure to and save synapse info
    
	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

    // run for a total of 10 seconds
    // at the end of each runNetwork call, SpikeMonitor stats will be printed
    for (int i=0; i<20; i++)
    {
        if (i >= 0 && i < 10) 
        {
            sim.runNetwork(0,500); // run network for 500 ms
        }

        if ( i == 10)
        {
            for (int j = 0; j < numGranuleFire; j++)
            {
                int randGranCell = DG_vec.front() + DG_vec_A[j]; // choose the jth random granule cell
                DG_Granule_rate.setRate(DG_vec.at(randGranCell), DG_Granule_frate); // set the firing rate for the jth random granule cell
            }
            sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation           
            sim.runNetwork(0,25); // run network for 25 ms
        }

        if (i == 11)
        {
            DG_Granule_rate.setRates(0.4f); // set the firing rates for all granule cells back to baseline firing rate
            sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation          
            sim.runNetwork(0,75); // run network for 75 ms
        }

        if ( i == 12)
        {
            for (int j = 0; j < numGranuleFire; j++)
            {
                int randGranCell = DG_vec.front() + DG_vec_A[j]; // choose the jth random granule cell
                DG_Granule_rate.setRate(DG_vec.at(randGranCell), DG_Granule_frate); // set the firing rate for the jth random granule cell
            }
            sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation           
            sim.runNetwork(0,25); // run network for 25 ms
            }

        if (i == 13)
        {
            DG_Granule_rate.setRates(0.4f); // set the firing rates for all granule cells back to baseline firing rate
            sim.setSpikeRate(DG_Granule, &DG_Granule_rate, 1); // update the firing rates of all granule cells before the next run of the simulation
            sim.runNetwork(0,75); // run network for 75 ms
        }

        if (i >=14 && i < 20)
        {
            sim.runNetwork(0,500); // run network for 500 ms
        }
    }

    sim.saveSimulation("ca3SNN2.dat", true); // fileName, saveSynapseInfo

	// print stopwatch summary
	watch.stop();

	return 0;
}
