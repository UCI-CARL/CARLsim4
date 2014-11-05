#include "Logger.h"

#include <gtest/gtest.h>
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace CARLsim_PTI;

/*! A test fixture setting up a Logger that writes to a std::stringstream,
 * where it is easy for us to verify the output. */
class LoggerTest : public ::testing::Test {
 protected:
    const char* const output1;
    const char* const output2;
    const char* const output3;
    std::stringstream stream;
    const Logger::Level defaultLevel;
    
    Logger sut;
    LoggerTest():
        output1("Test output 1."),
        output2("Test output 2."),
        output3("Test output 3."),
        defaultLevel(Logger::USER),
        sut(defaultLevel, stream) {};
};

TEST_F(LoggerTest, Constructor) {
    EXPECT_EQ(sut.getLevel(), defaultLevel) << "Constructor failed to assign local variable.";
    EXPECT_EQ(stream.str(), std::string("")) << "Brand new Logger is non-empty.";
}

TEST_F(LoggerTest, LogOneLine) {
    sut.log(Logger::USER, output1);
    EXPECT_EQ(stream.str(), std::string(output1) + "\n") << "Logger mangled output.";
}

TEST_F(LoggerTest, LogMultipleLines) {
    sut.log(Logger::USER, output1);
    sut.log(Logger::USER, output2);
    sut.log(Logger::USER, output3);
    
    std::string expected = std::string(output1) + "\n" +
                            std::string(output2) + "\n" +
                            std::string(output3) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger mangled output.";
}

TEST_F(LoggerTest, LogLevelDEBUG) {
    Logger sut(Logger::DEBUG, stream);
    
    sut.log(Logger::DEBUG, output1);
    sut.log(Logger::DEVELOPER, output2);
    sut.log(Logger::USER, output3);
    sut.log(Logger::WARNING, output1);
    sut.log(Logger::ERROR, output2);
    
    std::string expected = std::string(output1) + "\n" +
                            std::string(output2) + "\n" +
                            std::string(output3) + "\n" +
                            std::string(output1) + "\n" +
                            std::string(output2) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger interpreted Level hierarchy incorrectly.";
}

TEST_F(LoggerTest, LogLevelDEVELOPER) {
    Logger sut(Logger::DEVELOPER, stream);
    
    sut.log(Logger::DEBUG, output1);
    sut.log(Logger::DEVELOPER, output2);
    sut.log(Logger::USER, output3);
    sut.log(Logger::WARNING, output1);
    sut.log(Logger::ERROR, output2);
    
    std::string expected = std::string(output2) + "\n" +
                            std::string(output3) + "\n" +
                            std::string(output1) + "\n" +
                            std::string(output2) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger interpreted Level hierarchy incorrectly.";
}

TEST_F(LoggerTest, LogLevelUSER) {
    sut.log(Logger::DEBUG, output1);
    sut.log(Logger::DEVELOPER, output2);
    sut.log(Logger::USER, output3);
    sut.log(Logger::WARNING, output1);
    sut.log(Logger::ERROR, output2);
    
    std::string expected = std::string(output3) + "\n" +
                            std::string(output1) + "\n" +
                            std::string(output2) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger interpreted Level hierarchy incorrectly.";
}

TEST_F(LoggerTest, LogLevelWARNING) {
    Logger sut(Logger::WARNING, stream);
    
    sut.log(Logger::DEBUG, output1);
    sut.log(Logger::DEVELOPER, output2);
    sut.log(Logger::USER, output3);
    sut.log(Logger::WARNING, output1);
    sut.log(Logger::ERROR, output2);
    
    std::string expected = std::string(output1) + "\n" +
                            std::string(output2) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger interpreted Level hierarchy incorrectly.";
}

TEST_F(LoggerTest, LogLevelERROR) {
    Logger sut(Logger::ERROR, stream);
    
    sut.log(Logger::DEBUG, output1);
    sut.log(Logger::DEVELOPER, output2);
    sut.log(Logger::USER, output3);
    sut.log(Logger::WARNING, output1);
    sut.log(Logger::ERROR, output2);
    
    std::string expected = std::string(output2) + "\n";
    EXPECT_EQ(stream.str(), expected) << "Logger interpreted Level hierarchy incorrectly.";
}