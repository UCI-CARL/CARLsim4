#ifndef PTI_H
#define PTI_H

#include "Experiment.h"

#include <string>
#include <memory>
#include <iostream>

using namespace std;

/*!
 * 
 */
class PTI {
public:
    PTI(const int argc, const char * const argv[], ostream &outputStream);
    PTI(const int argc, const char * const argv[], ostream &outputStream, istream &defaultInputStream);
    ~PTI();
    void runExperiment(const Experiment &experiment) const;
    string usage() const;
    bool repOK() const;
private:
    struct PTIImpl;
    const unique_ptr<PTIImpl> impl;
};

#endif // PTI_H
