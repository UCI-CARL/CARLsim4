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

// Create a function that will print out all of the elements of a vector
void print(std::vector <int> const &a) {
 std::cout << "The vector elements are : ";

 for(int i=0; i < a.size(); i++)
		std::cout << a.at(i) << ' ';
}

// Create a function that will create a subset of a vector, which will can be
// used in defining Poisson rates for a fraction of neurons in a group
template<typename T>
std::vector<T> slice(std::vector<T> &v, int m, int n)
{
    std::vector<T> vec;
    std::copy(v.begin() + m, v.begin() + n + 1, std::back_inserter(vec));
    return vec;
}

int main() 
{
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

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// Declare variables that will store the start and end ID for the neurons
	// in the pyramidal group
	int pyr_start = sim.getGroupStartNeuronId(CA3_Pyramidal);
	std::cout << "Beginning neuron ID for Pyramidal Cells is : " << pyr_start;
	int pyr_end = sim.getGroupEndNeuronId(CA3_Pyramidal);
	std::cout << "Ending neuron ID for Pyramidal Cells is : " << pyr_end;
	int pyr_range = (pyr_end - pyr_start) + 1;
	std::cout << "The range for Pyramidal Cells is : " << pyr_range;

	// Create vectors that are the length of the number of neurons in the pyramidal
	// group, and another that will store the current at the position for the
	// random pyramidal cells that will be selected
	std::vector<int> pyr_vec( boost::counting_iterator<int>( 0 ),
							  boost::counting_iterator<int>( pyr_range ));
    std::vector<float> current(pyr_range, 0.0f);
    std::vector<int> pyr_vec_random( boost::counting_iterator<int>( 0 ),
									 boost::counting_iterator<int>( pyr_range ));

	// include header file that contains generation of groups and their
	// properties
	#include "../generateSETUPStateSTP.h"

    // Define the number of neurons to receive input from the external current
    int numPyramidalFire = 1;
    int count = 0;

	// Set the seed of the pseudo-random number generator based on the current system time
	std::srand(std::time(nullptr));

    // Define the vectors to be used for storing the random cells of each neuron type chosen
    std::random_shuffle ( pyr_vec_random.begin(), pyr_vec_random.end() );
    std::vector<float> randPyr;

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");
    

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int i=0; i<1017; i++) 
	{
        if (i >= 0 && i < 1000)
        {
          // Set external current for a fraction of pyramidal cells based on the random
          // seed
          for (int j = 0; j < numPyramidalFire; j++)
          {
            int randPyrCell = pyr_vec_random[count];
            randPyr.push_back(randPyrCell);
            current.at(randPyrCell) = 45000.0f;
            count += 1;
          }
          
          // Set external current for a fraction of pyramidal cells based on the random
          // seed and run the network for one ms
          sim.setExternalCurrent(CA3_Pyramidal, current);
          sim.runNetwork(0,1);
          
          // Set external current for the pyramidal that were randomly chosen
          // back to zero 
          for (int k = 0; k < randPyr.size(); k++)
          {
            current.at(randPyr[k]) = 0.0f;
          }
        }
        
		if (i == 1000)
		{
          // Set external current for the pyramidal cells to zero and run the
          // network for one ms
    		sim.setExternalCurrent(CA3_Pyramidal, 0.0f);
    		sim.runNetwork(0,1);
    	}
        if (i >= 1001 && i < 1016)
		{
    		// Run the network for 500 ms
    		sim.runNetwork(0,500);
		}
		
		if (i == 1016)
		{
    		// Run the network for 499 ms
    		sim.runNetwork(0,499);
    	}
    }

	// print stopwatch summary
	watch.stop();

	return 0;
}
