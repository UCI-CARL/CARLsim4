/*! \brief Using PTI to search for strings that maximize or minimize the sum
 * of all parameter values.
 *
 */
#include "PTI.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>

using namespace std;
using namespace CARLsim_PTI;

namespace CARLsim_PTI {
  class SumExperiment : public Experiment {
  public:
    SumExperiment() {}

    void run(const ParameterInstances &parameters, std::ostream &outputStream) const {
      srand(time(NULL));
      for(unsigned int i = 0; i < parameters.getNumInstances(); i++) {
	float sum = 0.0;
	for (unsigned int j = 0; j < parameters.getNumParameters(); j++) {
	  const float p = parameters.getParameter(i, j);
	  sum += p;
	}
	// Add some random noise
	sum += ((float) (rand()%parameters.getNumParameters()))/10.0f;
	outputStream << sum << endl;
      }
    }
  };
}

int main(int argc, char* argv[]) {
  /* First we Initialize an Experiment and a PTI object.  The PTI parses CLI
   * arguments, and then loads the Parameters from a file (if one has been
   * specified by the user) or else from a default istream (std::cin here). */
  const SumExperiment experiment;
  const PTI pti(argc, argv, std::cout, std::cin);

  /* The PTI will now cheerfully iterate through all the Parameter sets and
   * run your Experiment on it, printing the results to the specified
   * ostream (std::cout here). */
  pti.runExperiment(experiment);

  return 0;
}

