#ifndef PTI_H
#define PTI_H

#include "Experiment.h"

#include <string>
#include <memory>
#include <iostream>

using namespace std;

/*!
 * The CARLsim parameter-tuning interface (PTI) provides a standardized interface for tuning a model with an 
 * external optimization tool.
 * 
 * The PTI class reads a list of real-valued parameter vectors from a file or istream and passes them to an 
 * Experiment object to have their performance evaluated.  You should implement a subclass of Experiment 
 * that executes your model and calculates some kind of information about its behavior, which PTI will write to
 * an ostream.
 * 
 * The intent is that PTI and an Experiment can be used in the main method of a simple program:
 * 
 * \snippet examples/AbortExample.cpp PTI
 * 
 * A program like this can serve as the interface between CARLsim an an optimization tool (such as the <a 
 * href="https://cs.gmu.edu/~eclab/projects/ecj/">ECJ metaheuristics toolkit</a>).
 */
class PTI {
public:
    /*!
     * Parse command-line arguments and set up a PTI instance that writes fitnesses or phenotypes to the specified 
     * ostream.
     * 
     * If a file is specified in the arguments via the option '-f filename', then parameter vectors are 
     * read from the file.  Otherwise, the are read from std::cin.
     * 
     * @param argc The number of command-line arguments
     * @param argv Array of command-line argument
     * @param outputStream An ostream to write fitness or phenotype values to
     */
    PTI(const int argc, const char * const argv[], ostream &outputStream);

    /*!
     * Parse command-line arguments and set up a PTI instance that writes fitnesses or phenotypes to the specified 
     * ostream.
     * 
     * If a file is specified in the arguments via the option '-f filename', then parameter vectors are 
     * read from the file.  Otherwise, they are read from the specified default istream.
     * 
     * @param argc The number of command-line arguments
     * @param argv Array of command-line argument
     * @param outputStream An ostream to write fitness or phenotype values to
     * @param defaultInputStream An istream to receive parameter vectors from if no file is specified.
     */
    PTI(const int argc, const char * const argv[], ostream &outputStream, istream &defaultInputStream);
    ~PTI();

    /*!
     * Execute an Experiment on the incoming parameter vectors, telling it to write results to the 
     * outgoing ostream.
     * 
     * @param experiment The Experiment whose Experiment::run() method should be executed.
     */
    void runExperiment(const Experiment &experiment) const;

    /*!
     * A string explaining how to use the command-line options.
     */
    string usage() const;

    /*!
     * Representation invariant: used for debugging; always returns true unless the class's state is inconsistent.
     */
    bool repOK() const;
    
private:
    struct PTIImpl;
    const unique_ptr<PTIImpl> impl;
};

#endif // PTI_H
