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

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	// include header file that contains generation of groups and their
	// properties
	#include "../generateSETUPStateSTP.h"

	// Set the seed of the pseudo-random number generator based on the current system time
	std::srand(std::time(nullptr));

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
	for (int i=0; i<20; i++) {
      sim.runNetwork(0,500);
		}

	// print stopwatch summary
	watch.stop();

	return 0;
}
