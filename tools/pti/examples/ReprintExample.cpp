/*! \brief A program that demonstrates using PTI to load parameters and run an
 * Experiment using them.
 * 
 * The "Experiment" is very simple: It just prints out all the parameters in
 * tab-delimited format. 
 */
#include "PTI.h"
#include "Experiment.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;

class ReprintExperiment : public Experiment {
public:
    ReprintExperiment() {}

    void run(const ParameterInstances &parameters, std::ostream &outputStream) const {
        for(unsigned int i = 0; i < parameters.getNumInstances(); i++) {
            for (unsigned int j = 0; j < parameters.getNumParameters(); j++) {
                const float p = parameters.getParameter(i, j);
                outputStream << p << "\t";
            }
            outputStream << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    /* First we Initialize an Experiment and a PTI object.  The PTI parses CLI
     * arguments, and then loads the Parameters from a file (if one has been
     * specified by the user) or else from a default istream (std::cin here). */
    const ReprintExperiment experiment;
    const PTI pti(argc, argv, std::cout, std::cin);
    
    /* The PTI will now cheerfully iterate through all the Parameter sets and
     * run your Experiment on each one, printing the results to the specified
     * ostream (std::cout here). */
    pti.runExperiment(experiment);

    return 0;
}
