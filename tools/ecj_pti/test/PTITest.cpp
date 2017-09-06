#include <PTI.h>
#include <Experiment.h>
#include <ParameterInstances.h>
#include <Util.h>

#include <gtest/gtest.h>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace CARLsim_PTI;

/*! A test fixture setting up a Logger that writes to a std::stringstream,
 * where it is easy for us to verify the output. */
class PTITest : public ::testing::Test {
 protected:
    std::stringstream defaultInputStream;
    std::stringstream outputStream;
    PTI sut;

    PTITest():
        defaultInputStream(string("2.82, 6.51, 7.37, 7.67, 2.32, 3.95, 2.22, 5.21\n") +
                string("2.81, 8.24, 5.11, 7.82, 3.62, 7.00, 1.98, 5.70\n") +
                string("1.92, 9.74, 2.77, 5.04, 3.74, 6.72, 6.96, 9.26\n") +
                string("2.95, 6.70, 7.12, 8.85, 4.15, 5.40, 3.19, 6.19")),
        sut(0, NULL, outputStream, defaultInputStream) {};
};

class TestExperiment : public Experiment {
public:
    TestExperiment() {}

    void run(const ParameterInstances &parameters, std::ostream &outputStream) const {
        for(unsigned int i = 0; i < parameters.getNumInstances(); i++) {
            float sum = 0.0;
            for (unsigned int j = 0; j < parameters.getNumParameters(); j++) {
                const float p = parameters.getParameter(i, j);
                sum += p;
            }
            outputStream << sum << endl;
        }
    }
};

TEST_F(PTITest, RunWithDefaultInputStream) {
    const TestExperiment experiment;
    sut.runExperiment(experiment);
    const float expected[4] = { 38.07, 42.28, 46.15, 44.54999999999999 };
    string strLine;
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(getline(outputStream, strLine));
        EXPECT_FLOAT_EQ(stringToFloat(strLine), expected[i]) << "Experiment returned incorrect sum of parameters.";
    }
    EXPECT_FALSE(getline(outputStream, strLine)) << "Test input had more lines than expected.";
}

TEST_F(PTITest, ConstructWithFileDNE) {
    const char* const argv[2] = { "-f", "/awepohbpoihaef" };
    EXPECT_THROW(const PTI sut(2, argv, outputStream, defaultInputStream), std::invalid_argument);
}

TEST_F(PTITest, RunWithFile) {
    const TestExperiment experiment;
    const char* const argv[2] = { "-f", "test_data/pti_tests_data.csv" };
    const PTI sut(2, argv, outputStream, defaultInputStream);
    sut.runExperiment(experiment);
    const float expected[4] = { 15, 4.26, 0.00036, 401.48999000000 };
    string strLine;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(getline(outputStream, strLine));
        EXPECT_FLOAT_EQ(stringToFloat(strLine), expected[i]) << "Experiment returned incorrect sum of parameters.";
    }
    EXPECT_FALSE(getline(outputStream, strLine)) << "Test input had more lines than expected.";
}
