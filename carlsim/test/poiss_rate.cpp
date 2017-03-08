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
#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>

// trigger all UserErrors
TEST(PoissRate, constructDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	EXPECT_DEATH({PoissonRate poiss(0);},""); // nNeur==0
	EXPECT_DEATH({PoissonRate poiss(-1);},""); // nNeur<0
}

// testing getRate(neurId)
TEST(PoissRate, getRateNeurId) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

#ifdef __NO_CUDA__
        int numGPU = 0;
#else
        int numGPU = 1;
#endif

	int nNeur = 100;
	for (int onGPU=0; onGPU<=numGPU; onGPU++) {
		PoissonRate rate(nNeur,onGPU==true);

		for (int i=0; i<nNeur; i++) {
			EXPECT_FLOAT_EQ(rate.getRate(i), 0.0f);
		}
	}	
}

// testing getRates vector
TEST(PoissRate, getRates) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

#ifdef __NO_CUDA__
        int numGPU = 0;
#else
        int numGPU = 1;
#endif

	int nNeur = 100;
	for (int onGPU=0; onGPU<=numGPU; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		std::vector<float> ratesVec = rate.getRates();

		bool isSizeCorrect = ratesVec.size() == nNeur;
		EXPECT_TRUE(isSizeCorrect);

		if (isSizeCorrect) {
			for (int i=0; i<nNeur; i++) {
				EXPECT_FLOAT_EQ(ratesVec[i], 0.0f);
			}
		}
	}
}

// setting rates with vector
TEST(PoissRate, setRatesVector) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

#ifdef __NO_CUDA__
        int numGPU = 0;
#else
        int numGPU = 1;
#endif

	int nNeur = 100;
	for (int onGPU=0; onGPU<=numGPU; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		std::vector<float> ratesVecIn;

		for (int i=0; i<nNeur; i++)
			ratesVecIn.push_back(i);

		rate.setRates(ratesVecIn);

		std::vector<float> ratesVecOut = rate.getRates();
		bool isSizeCorrect = ratesVecOut.size() == nNeur;
		EXPECT_TRUE(isSizeCorrect);
		if (isSizeCorrect) {
			for (int i=0; i<nNeur; i++) {
				EXPECT_FLOAT_EQ(ratesVecOut[i], i);
			}
		}
	}
}

// setting all rates to same float
TEST(PoissRate, setRatesFloat) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

#ifdef __NO_CUDA__
        int numGPU = 0;
#else
        int numGPU = 1;
#endif

	int nNeur = 100;
	for (int onGPU=0; onGPU<=numGPU; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		rate.setRates(42.0f);

		std::vector<float> ratesVec = rate.getRates();
		bool isSizeCorrect = ratesVec.size() == nNeur;
		EXPECT_TRUE(isSizeCorrect);
		if (isSizeCorrect) {
			for (int i=0; i<nNeur; i++) {
				EXPECT_FLOAT_EQ(ratesVec[i], 42.0f);
			}
		}
	}
}

// setting single neuron ID to float
TEST(PoissRate, setRateNeurId) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

#ifdef __NO_CUDA__
        int numGPU = 0;
#else
        int numGPU = 1;
#endif

	int nNeur = 100;
	int neurId = 42;
	float neurIdRate = 10.25f;
	for (int onGPU=0; onGPU<=numGPU; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		rate.setRate(neurId,neurIdRate);

		std::vector<float> ratesVec = rate.getRates();
		bool isSizeCorrect = ratesVec.size() == nNeur;
		EXPECT_TRUE(isSizeCorrect);

		if (isSizeCorrect) {
			for (int i=0; i<nNeur; i++) {
				if (i!=neurId) {
					EXPECT_FLOAT_EQ(ratesVec[i], 0.0f);
				} else {
					EXPECT_FLOAT_EQ(ratesVec[neurId],neurIdRate);
				}
			}
		}
	}
}

//! \NOTE: There is no good way to further test PoissonRate and in a CARLsim environment. Running the network twice
//! will not reproduce the same spike train (because of the random seed). Comparing CPU mode to GPU mode will not work
//! because CPU and GPU use different random seeds. Comparing lambda in PoissonRate.setRate(lambda) to the one from
//! SpikeMonitor.getNeuronMeanFiringRate() will not work because the Poisson process has standard deviation == lambda.
TEST(PoissRate, runSim) {
	// \TODO test CARLsim integration
	// \TODO use cuRAND
}
