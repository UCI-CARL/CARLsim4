#ifndef LOGGER
#define LOGGER

#include <iostream>

/*! A simple logging class.  This will write messages to the given std::ostream
 * as long as their level is higher than or equal to the configured 
 * loggerLevel.  ERROR is the highest (always printed), DEBUG is the lowest.
 */
namespace CARLsim_PTI {
    class Logger {
    public:
        enum Level { DEBUG=0, DEVELOPER, USER, WARNING, ERROR };
        
        /*! Construct a Logger that outputs only messages higher than or
         * equal to level, and writes the messages to the given ostream.
         * 
         * \param level Messages below this level are ignored.
         * \param logStream The ostream to write messages to when they are
         * received.  The Logger holds a reference to the stream, but does not
         * own it -- the caller is responsible for closing file streams, etc,
         * after they are done with the Logger.  If no ostream is given,
         * messages are written to std::cout by default.
         */
        Logger(const Level level, std::ostream &logStream = std::cout);
        
        /*! Write a message to the log at the given level.
         *
         * \param level Messages below this level will be ignored.
         * \param message c-string with the text to write to the log.
         */
        virtual void log(const Level level, const char * const message);
        
        /*! View the logging level this Logger is configured at. */
        Level getLevel();
    private:
        const Level loggerLevel;
        std::ostream &logStream;
    };
}
#endif
