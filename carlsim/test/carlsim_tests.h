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
#ifndef _CARLSIM_TEST_H_
#define _CARLSIM_TEST_H_

#include <algorithm>		// std::find
#include <vector>			// std::vector
#include <string>			// std::string, memset
#include <cassert>			// assert

#ifndef __NO_CUDA__
#define TESTED_MODES 2
#else
#define TESTED_MODES 1
#endif

/*
 * GENERAL TESTING STRATEGY
 * ------------------------
 *
 * We provide test cases to A) test core functionality of CARLsim, to B) test the reproducibility of published results,
 * and C) to benchmark simulation speed.
 *
 * A) TESTING CORE FUNCTIONALITY
 * 1. Test core data structures when some functionality is enabled.
 *    For example: Set STP to true for a specific group, check groupConfig to make sure all values are set accordingly.
 * 2. Test core data structures when some functionality is disabled.
 *    For example: Set STP to false for a specific group, check groupConfig to make sure it's disabled.
 * 3. Test behavior when values for input arguments are chosen unreasonably.
 *    For example: Create a group with N=-4 (number of neurons) and expect simulation to die. This is because each
 *    core function should have assertion statements to prevent the simulation from running unreasonable input values.
 *    In some cases, it makes sense to catch this kind of error in the user interface as well (and display an
 *    appropriate error message to the user), but these tests should be placed in the UserInterface test case.
 * 4. Test behavior of network when run with reasonable values.
 *    For example: Run a sample network with STP enabled, and check stpu[nid] and stpx[nid] to make sure they behave.
 *    as expected. You can use the PeriodicSpikeGenerator to be certain of specific spike times and thus run
 *    reproducible sample networks.
 * 5. Test behavior of network when run in CPU mode vs. GPU mode.
 *    For example: Run a sample network with STP enabled, once in CPU mode and once in GPU mode. Record stpu[nid] and
 *    stpx[nid], and make sure that both simulation mode give the exact same result (except for some small error
 *    margin that can account for rounding errors/etc.).
 *
 * B) TESTING PUBLISHED RESULTS
 *
 * C) BENCHMARK TESTS
 *
 */

/// **************************************************************************************************************** ///
/// Utility Functions
/// **************************************************************************************************************** ///

void readAndReturnSpikeFile(const std::string fileName, int*& AERArray, long &arraySize);
void readAndPrintSpikeFile(const std::string fileName);

#endif // _CARLSIM_TEST_H_
