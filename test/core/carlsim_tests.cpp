/*!
 * Testing for the public member function: read, of the pti class.
 * 
 * @author Kris Carlson (KDC)
 */

#include <limits.h>
#include "gtest/gtest.h"

// Tests for args > 3
TEST(readTestArgc, ArgNumber) {
  // This test is named "ArgNumber", and belongs to the "readTest"
  // test case.

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
