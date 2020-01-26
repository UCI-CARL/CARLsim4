#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "ParameterInstances.h"

#include <ostream>

class Experiment {
public:
    virtual void run(const ParameterInstances &parameters, std::ostream &outputStream) const = 0;
};


#endif // EXPERIMENT_H
