#include "gtest/gtest.h"
#include "carlsim_tests.h"

#include <poisson_rate.h>

//! trigger all UserErrors
// TODO: add more error checking
TEST(PoissRate, constructDeath) {
	::testing::FLAGS_gtest_death_test_style = "threadsafe";

	EXPECT_DEATH({PoissonRate poiss(-1);},""); // nNeur<0
	EXPECT_DEATH({PoissonRate poiss(1,true,0);},""); // refPeriod=0
	EXPECT_DEATH({PoissonRate poiss(1,true,-3);},""); // refPeriod<0
}

TEST(PoissRate, getSetRateNeurId) {

}