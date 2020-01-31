#include "Logger.h"
#include <stdexcept>
#include <string>
#include <cassert>
#include <typeinfo>
#include <ostream>

Logger::Logger(const Level level, std::ostream &logStream)
: loggerLevel(level), logStream(logStream) {}

void Logger::log(const Level messageLevel, const char * const message) {
    assert(message != NULL);
    if (messageLevel >= loggerLevel)
        logStream << message << std::endl;
}

Logger::Level Logger::getLevel() { return loggerLevel; }
