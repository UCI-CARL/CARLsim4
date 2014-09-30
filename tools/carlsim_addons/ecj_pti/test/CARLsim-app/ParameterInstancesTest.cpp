#include "ParameterInstances.h"

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace CARLsim_PTI;
using std::string;

class ParameterInstancesTest : public ::testing::Test {
 protected:
    std::stringstream emptyInput;
    std::stringstream goodInput;
    std::stringstream outOfOrderInput;
    std::stringstream oddColumnInput;
    std::stringstream unevenInput;
    std::stringstream nonFloatInput;

    ParameterInstances sut;
    ParameterInstancesTest():
        emptyInput(""),
        goodInput(string("2.82, 6.51, 7.37, 7.67, 2.32, 3.95, 2.22, 5.21\n") +
                string("2.81, 8.24, 5.11, 7.82, 3.62, 7.00, 1.98, 5.70\n") +
                string("1.92, 9.74, 2.77, 5.04, 3.74, 6.72, 6.96, 9.26\n") +
                string("2.95, 6.70, 7.12, 8.85, 4.15, 5.40, 3.19, 6.19")),
        oddColumnInput(string("2.82, 6.51, 7.37, 7.67, 2.32, 3.95, 2.22\n") +
                string("2.81, 8.24, 5.11, 7.82, 3.62, 7.00, 5.70\n") +
                string("1.92, 9.74, 2.77, 5.04, 3.74, 6.72, 6.96\n") +
                string("2.95, 6.70, 7.12, 8.85, 4.15, 5.40, 6.19")),
        unevenInput(string("2.82, 6.51, 7.37, 7.67, 2.32, 3.95, 2.22, 5.21\n") +
                string("2.81, 8.24, 5.11, 7.82, 3.62, 7.00, 1.98, 5.70\n") +
                string("1.92, 9.74, 2.77, 5.04, 3.74, 6.72\n") +
                string("2.95, 6.70, 7.12, 8.85, 4.15, 5.40, 3.19, 6.19")),
        nonFloatInput(string("2.82, 6.51, 7.37, 7.67, 2.32, 3.95, 2.22, 5.21\n") +
                string("2.81, 8.24, 5.11, 7.82, NA, 7.00, 1.98, 5.70\n") +
                string("1.92, 9.74, 2.77, 5.04, 3.74, 6.72, 6.96, 9.26\n") +
                string("2.95, 6.70, 7.12, 8.85, 4.15, 5.40, 3.19, 6.19")),
        sut(goodInput) {};
};

TEST_F(ParameterInstancesTest, Constructor) {
    EXPECT_EQ(sut.getNumInstances(), 4) << "Constructor deduced incorrect number of rows.";
    EXPECT_EQ(sut.getNumParameters(), 8) << "Construct deduced incorrect number of parameters.";
    EXPECT_TRUE(sut.repOK());
}

TEST_F(ParameterInstancesTest, EmptyInput) {
    ParameterInstances sut(emptyInput);
    EXPECT_EQ(sut.getNumInstances(), 0) << "Found data here, but the input was empty.";
    EXPECT_EQ(sut.getNumParameters(), 0) << "No individuals, but this is saying I have > 0 parameters.";
    EXPECT_TRUE(sut.repOK());
    EXPECT_DEATH(sut.getParameter(0, 0), "");
}

TEST_F(ParameterInstancesTest, GoodInput) {
    EXPECT_EQ(sut.getNumInstances(), 4) << "Constructor deduced incorrect number of rows.";
    EXPECT_EQ(sut.getNumParameters(), 8) << "Construct deduced incorrect number of parameters.";
    float expected00 = 2.82;
    float expected02 = 7.37;
    float expected06 = 2.22;
    float expected30 = 2.95;
    float expected36 = 3.19;
    float nExpected12 = 8.82;

    // Check some values
    EXPECT_FLOAT_EQ(sut.getParameter(0, 0), expected00);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 2), expected02);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 6), expected06);
    EXPECT_FLOAT_EQ(sut.getParameter(3, 0), expected30);
    EXPECT_FLOAT_EQ(sut.getParameter(3, 6), expected36);
    EXPECT_NE(sut.getParameter(1, 2), nExpected12);

    EXPECT_TRUE(sut.repOK());
    EXPECT_DEATH(sut.getParameter(4, 0), "");
    EXPECT_DEATH(sut.getParameter(0, 8), "");
}

TEST_F(ParameterInstancesTest, OddColumnInput) {
    EXPECT_EQ(sut.getNumInstances(), 4) << "Constructor deduced incorrect number of rows.";
    EXPECT_EQ(sut.getNumParameters(), 8) << "Construct deduced incorrect number of parameters.";
    float expected00 = 2.82;
    float expected02 = 7.37;
    float expected06 = 2.22;
    float expected30 = 2.95;
    float expected36 = 3.19;
    float nExpected12 = 8.82;

    // Check some values
    EXPECT_FLOAT_EQ(sut.getParameter(0, 0), expected00);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 2), expected02);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 6), expected06);
    EXPECT_FLOAT_EQ(sut.getParameter(3, 0), expected30);
    EXPECT_FLOAT_EQ(sut.getParameter(3, 6), expected36);
    EXPECT_NE(sut.getParameter(1, 2), nExpected12);

    EXPECT_TRUE(sut.repOK());
    EXPECT_DEATH(sut.getParameter(4, 0), "");
    EXPECT_DEATH(sut.getParameter(0, 8), "");
}

TEST_F(ParameterInstancesTest, UnevenInput) {
    EXPECT_THROW(new ParameterInstances(unevenInput), std::invalid_argument) << "Constructor accepted input with unequal row lengths.";
}

TEST_F(ParameterInstancesTest, NonFloatInput) {
    EXPECT_THROW(new ParameterInstances(nonFloatInput), std::invalid_argument) << "Constructor accepted input with non-float value.";
}

TEST_F(ParameterInstancesTest, FileInput) {
    std::ifstream input("test_data/pti_tests_data.csv", std::ifstream::in);
    ASSERT_TRUE(input.is_open());
    ParameterInstances sut(input);
    float expected00 = -1;
    float expected02 = -1;
    float expected06 = 3;
    float expected20 = 0.00001;
    float expected36 = 100;
    float nExpected12 = 0.01;

    // Check some values
    EXPECT_FLOAT_EQ(sut.getParameter(0, 0), expected00);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 2), expected02);
    EXPECT_FLOAT_EQ(sut.getParameter(0, 6), expected06);
    EXPECT_FLOAT_EQ(sut.getParameter(2, 0), expected20);
    EXPECT_FLOAT_EQ(sut.getParameter(3, 6), expected36);
    EXPECT_NE(sut.getParameter(1, 2), nExpected12);

    EXPECT_TRUE(sut.repOK());
    EXPECT_DEATH(sut.getParameter(4, 0), "");
    EXPECT_DEATH(sut.getParameter(0, 8), "");
}
