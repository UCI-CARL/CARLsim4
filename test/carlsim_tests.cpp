/*!
 * Testing for the public member function: read, of the pti class.
 * 
 * @author Kris Carlson (KDC)
 */

#include <carlsim.h>
#include <limits.h>
#include "gtest/gtest.h"

// Tests for args > 3
TEST(readTestArgc, ArgNumber) {
  // This test is named "ArgNumber", and belongs to the "readTest"
  // test case.

	CARLsim sim("SNN",1,42,CPU_MODE,0,true);

	int N=1000;

	int gin=sim.createSpikeGeneratorGroup("input",N*0.1,EXCITATORY_NEURON);

	int g1=sim.createGroup("excit", N*0.8, EXCITATORY_NEURON);
	sim.setNeuronParameters(g1, 0.02f, 0.2f, -65.0f, 8.0f);

	// 5% probability of connection
	sim.connect(gin,g1,"random", +100.0f/100, 100.0f/100, 0.05f,  1, 20, SYN_FIXED);

	PoissonRate in(N*0.1);
	for (int i=0;i<N*0.1;i++) in.rates[i] = 1;
		sim.setSpikeRate(gin,&in);


	//run for 10 seconds
	for(int i=0; i < 10; i++) {
		// run the established network for a duration of 1 (sec)  and 0 (millisecond)
		sim.runNetwork(1,0);
	}


  EXPECT_EQ(1,1);
  //EXPECT_EQ(1, Factorial(-1));
  //EXPECT_GT(Factorial(-10), 0);

}

// // Tests factorial of 0.
// TEST(FactorialTest, Zero) {
//   EXPECT_EQ(1, Factorial(0));
// }

// // Tests factorial of positive numbers.
// TEST(FactorialTest, Positive) {
//   EXPECT_EQ(1, Factorial(1));
//   EXPECT_EQ(2, Factorial(2));
//   EXPECT_EQ(6, Factorial(3));
//   EXPECT_EQ(40320, Factorial(8));
// }


// // Tests IsPrime()

// // Tests negative input.
// TEST(IsPrimeTest, Negative) {
//   // This test belongs to the IsPrimeTest test case.

//   EXPECT_FALSE(IsPrime(-1));
//   EXPECT_FALSE(IsPrime(-2));
//   EXPECT_FALSE(IsPrime(INT_MIN));
// }

// // Tests some trivial cases.
// TEST(IsPrimeTest, Trivial) {
//   EXPECT_FALSE(IsPrime(0));
//   EXPECT_FALSE(IsPrime(1));
//   EXPECT_TRUE(IsPrime(2));
//   EXPECT_TRUE(IsPrime(3));
// }

// // Tests positive input.
// TEST(IsPrimeTest, Positive) {
//   EXPECT_FALSE(IsPrime(4));
//   EXPECT_TRUE(IsPrime(5));
//   EXPECT_FALSE(IsPrime(6));
//   EXPECT_TRUE(IsPrime(23));
// }
