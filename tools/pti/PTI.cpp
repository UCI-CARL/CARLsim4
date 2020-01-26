#include "PTI.h"
#include "ParameterInstances.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

using namespace std;

struct PTI::PTIImpl {
    ostream &outputStream;
    const unique_ptr<ParameterInstances> instances;
    
    PTIImpl(const char* const fileName, istream &defaultInputStream, ostream &outputStream):
            outputStream(outputStream),
            instances(loadParameterInstances(fileName, defaultInputStream)) {
    }
    
    static const char* getStringArgument(const char* const option, const int argc, const char* const argv[]) {
        assert(option != NULL);
        for (int i = 0; i < argc - 1; i++) {
            if (0 == strcmp(option, argv[i]))
                return argv[i+1];
        }
        return NULL;
    }
    
    static int getIntegerArgument(const char* const option, const int argc, const char* const argv[]) {
        assert(option != NULL);
        if (argc < 0)
            throw invalid_argument(string("PTI::PTIImpl: argc is negative."));
        if (argv == NULL)
            throw invalid_argument(string("PTI::PTIImpl: argv is NULL."));
        for (int i = 0; i < argc - 1; i++) {
            if (0 == strcmp(option, argv[i]))
                return atoi(argv[i+1]);
        }
        return -1;
    }
    
    /*! Parse command line arguments and read in the parameter values to be
        * used in an Experiment, allocating a ParameterInstances on the heap. */
    static ParameterInstances* loadParameterInstances(const char* const fileName, istream &defaultInputStream) {
        // I asked the following SO question while deciding how to write this method: http://stackoverflow.com/questions/23049166/initialize-polymorphic-variable-on-stack
        istream * const input(fileName ? new ifstream(fileName, ifstream::in) : &defaultInputStream); 
        
        if (fileName) {
            if (!dynamic_cast<ifstream*>(input)->is_open())
                throw invalid_argument(string("PTI::PTIImpl: Failed to open file") + string(fileName) + string("."));
        }
        
        ParameterInstances* const result = new ParameterInstances(*input);
        
        if (fileName) {
            dynamic_cast<ifstream*>(input)->close();
            delete(input);
        }
        return result;
    }
    
    bool repOK() const {
        return instances.get() != NULL;
    }
};

PTI::PTI(const int argc, const char* const argv[], ostream &outputStream):
        impl(new PTIImpl(PTIImpl::getStringArgument("-f", argc, argv), cin, outputStream)) {
    
    assert(repOK());
}

PTI::PTI(const int argc, const char* const argv[], ostream &outputStream, istream &defaultInputStream):
        impl(new PTIImpl(PTIImpl::getStringArgument("-f", argc, argv), defaultInputStream, outputStream)) {
    
    assert(repOK());
}

// Empty destructor allows us to get away with defining the pimpl in an auto_ptr.  See http://stackoverflow.com/questions/311166/stdauto-ptr-or-boostshared-ptr-for-pimpl-idiom
PTI::~PTI() { };

string PTI::usage() const {
  return string("\nThis program should be called with 3 arguments:\n\n") +
    string("./a.out [filename]\n\n") +
    string("<filename> is the name of the csv parameter file to be read.\n\n") +
    string("Format of csv file: Each row represents a single \nindividual, while \
  each csv represents a min or max value for a parameter. \nEach csv is a float.\
  If there are 4 individuals with 4 parameters, \nthen there will be four rows, \
  each with 8 csv (2 for each parameter).\n\n");
}

void PTI::runExperiment(const Experiment& experiment) const {
    experiment.run(*(impl.get()->instances.get()), impl.get()->outputStream);
    assert(repOK());
}

bool PTI::repOK() const {
    return impl.get()->repOK();
}
