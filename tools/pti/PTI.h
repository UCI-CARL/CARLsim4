#ifndef PTI_H
#define PTI_H

#include "Experiment.h"

#include <string>
#include <memory>
#include <iostream>

class PTI {
public:
    PTI(const int argc, const char * const argv[], std::ostream &outputStream);
    PTI(const int argc, const char * const argv[], std::ostream &outputStream, std::istream &defaultInputStream);
    ~PTI();
    void runExperiment(const Experiment &experiment) const;
    std::string usage() const;
    bool repOK() const;
private:
    struct PTIImpl;
    const std::unique_ptr<PTIImpl> impl;
};

#endif // PTI_H
