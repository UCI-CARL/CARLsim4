#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <carlsim.h>
#include <poisson_rate.h>

// trigger all UserErrors
TEST(PoissRate, constructDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	EXPECT_DEATH({PoissonRate poiss(0);},""); // nNeur==0
	EXPECT_DEATH({PoissonRate poiss(-1);},""); // nNeur<0
}

// testing getRate(neurId)
TEST(PoissRate, getRateNeurId) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	int nNeur = 100;
	for (int onGPU=0; onGPU<=1; onGPU++) {
		PoissonRate rate(nNeur,onGPU==true);

		for (int i=0; i<nNeur; i++) {
			EXPECT_FLOAT_EQ(rate.getRate(i), 0.0f);
		}
	}	
}

// testing getRates vector
TEST(PoissRate, getRates) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	int nNeur = 100;
	for (int onGPU=0; onGPU<=1; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		std::vector<float> ratesVec = rate.getRates();

		ASSERT_EQ(ratesVec.size(),nNeur);
		for (int i=0; i<nNeur; i++) {
			EXPECT_FLOAT_EQ(ratesVec[i], 0.0f);
		}
	}
}

// setting rates with vector
TEST(PoissRate, setRatesVector) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	int nNeur = 100;
	for (int onGPU=0; onGPU<=1; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		std::vector<float> ratesVecIn;

		for (int i=0; i<nNeur; i++)
			ratesVecIn.push_back(i);

		rate.setRates(ratesVecIn);

		std::vector<float> ratesVecOut = rate.getRates();
		ASSERT_EQ(ratesVecOut.size(),nNeur);
		for (int i=0; i<nNeur; i++) {
			EXPECT_FLOAT_EQ(ratesVecOut[i], i);
		}
	}
}

// setting all rates to same float
TEST(PoissRate, setRatesFloat) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	int nNeur = 100;
	for (int onGPU=0; onGPU<=1; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		rate.setRates(42.0f);

		std::vector<float> ratesVec = rate.getRates();
		ASSERT_EQ(ratesVec.size(),nNeur);
		for (int i=0; i<nNeur; i++) {
			EXPECT_FLOAT_EQ(ratesVec[i], 42.0f);
		}
	}
}

// setting single neuron ID to float
TEST(PoissRate, setRateNeurId) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	int nNeur = 100;
	int neurId = 42;
	float neurIdRate = 10.25f;
	for (int onGPU=0; onGPU<=1; onGPU++) {
		PoissonRate rate(nNeur,true==onGPU);
		rate.setRate(neurId,neurIdRate);

		std::vector<float> ratesVec = rate.getRates();
		ASSERT_EQ(ratesVec.size(),nNeur);
		for (int i=0; i<nNeur; i++) {
			if (i!=neurId) {
				EXPECT_FLOAT_EQ(ratesVec[i], 0.0f);
			} else {
				EXPECT_FLOAT_EQ(ratesVec[neurId],neurIdRate);
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