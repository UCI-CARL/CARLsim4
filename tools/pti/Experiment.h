#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "ParameterInstances.h"

#include <ostream>


/*! This class defines a high-level interface that can be used with an external tool to tune the parameters of an
 * SNN.  An Experiment defines a parameterized SNN model in such a way that many independent instances of the 
 * model can be executed (potentially in parallel) using different parameter configurations.
 * 
 * The intent is that this interface can be used in conjunction with an external optimization tool (such as the <a 
 * href="https://cs.gmu.edu/~eclab/projects/ecj/">ECJ metaheuristics toolkit</a>) to tune parameters of a simulation.
 */
class Experiment {
public:
    /*! Execute the model once for each parameter set in `parameters`, and write data about the results to 
     * `outputStream`, one result per line.
     * 
     * Use this function, for example, to implement a fitness function for turning the parameters of a model.
     */
    virtual void run(const ParameterInstances &parameters, std::ostream &outputStream) const = 0;
};


#endif // EXPERIMENT_H
