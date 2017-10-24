#ifndef PARAMETER_INSTANCES_H
#define PARAMETER_INSTANCES_H

#include <istream>
#include <vector>

namespace CARLsim_PTI {

    /*! Receives a CSV of doubleing point values from an istream and interprets
     * them as minimum and maximum bounds on a number of parameters.
     */
    class ParameterInstances {
    public:
        /*! Read the file in istream, interpretting each comma-delimited row as
         * an "individual" and each pair of columns as Parameter.
         * 
         * Throws std::invalid_argument if a non-double value is found, if the
         * file has an odd number of columns, if two rows have an unequal number
         * of columns, or if the first column in each pair has a value greater
         * than the second column (since we must have min <= max).
         */
        ParameterInstances(std::istream &inputStream, const bool firstColumnIsSubPopulation = false);
        
        ~ParameterInstances();

        double getParameter(const unsigned int instance, const unsigned int parameter) const;

        std::vector<double> getInstance(const unsigned int instance) const;
        
        unsigned int getSubPopulation(const unsigned int instance) const;
        
        /*! The number of rows that were found in the input. */
        unsigned int getNumInstances() const;

        /*! Half the number of columns that were found in the input. */
        unsigned int getNumParameters() const;
        
        /*! This returns false only if this is in an inconsistent state. Under
         * correct usage, this should always return true. */
        bool repOK() const;

    private:
        struct ParameterInstancesImpl;
        const ParameterInstancesImpl &impl;
    };
}

#endif // PARAMETER_INSTANCES_H
