#include "ParameterInstances.h"
#include "Util.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <istream>
#include <sstream>
#include <typeinfo>
#include <vector>

using namespace std;
using namespace CARLsim_PTI;

namespace CARLsim_PTI {
    struct ParameterInstances::ParameterInstancesImpl {
        vector< vector< double > > instanceVectors;
        vector< unsigned int > subPopulations;
        
        ParameterInstancesImpl(istream &inputStream, const bool firstColumnIsSubPopulation) {
            readCSV(inputStream, instanceVectors, subPopulations, firstColumnIsSubPopulation);
            assert(repOK());
        }

        void readCSV(istream &inputStream, vector< vector<double> > &instanceVectors, vector< unsigned int > &subPopulations, const bool firstColumnIsSubPopulation) {
            string strLine;
            // Iterate through each individual (row))
            while (getline(inputStream, strLine)) {
                // Determine which subPopulation this individual belongs to
                if (firstColumnIsSubPopulation) {
                    std::string subPop = strLine.substr(0, strLine.find(','));
                    subPopulations.push_back(atoi(subPop.c_str()));
                }
                else
                    subPopulations.push_back(0);
                // The rest of the row is its genome
                instanceVectors.push_back(readCSVLine(strLine, firstColumnIsSubPopulation));
            }
            if (!allRowsEqualLength(instanceVectors))
                throw invalid_argument(string(typeid(*this).name()) + string(": rows are not of equal length."));
        }

        vector<double> readCSVLine(const string &strLine, const bool ignoreFirstColumn) {
            string doubleString;
            stringstream strLineStream(strLine);

            vector<double> lineVector;
            
            if (ignoreFirstColumn)
                getline(strLineStream, doubleString, ',');
            while (getline(strLineStream, doubleString, ','))
                lineVector.push_back(stringToDouble(doubleString));

            return lineVector;
        }

        static bool allRowsEqualLength(const vector< vector<double> > &instanceVectors) {
            if (instanceVectors.size() == 0)
                return true;
            const unsigned int numParametersPerInstance = instanceVectors[0].size();
            for (unsigned int i = 1; i < instanceVectors.size(); i++)
                if (instanceVectors[i].size() != numParametersPerInstance)
                    return false;
            return true;
        }
        
        vector<double> getInstance(const unsigned int instance) const {
            assert(instance < getNumInstances());
            return instanceVectors[instance];
        }

        double getParameter(const unsigned int instance, const unsigned int parameter) const {
            assert(instance < getNumInstances());
            assert(parameter < getNumParameters());
            return instanceVectors[instance][parameter];
        }

        unsigned int getNumInstances() const {
            return instanceVectors.size();
        }

        unsigned int getNumParameters() const {
            if (0 == getNumInstances())
                return 0;
            return instanceVectors[0].size();
        }

        bool repOK() const {
            return allRowsEqualLength(instanceVectors);
        }

        unsigned int getSubPopulation(const unsigned int instance) const {
            assert(instance < getNumInstances());
            return subPopulations[instance];
        }
    };
}

ParameterInstances::ParameterInstances(istream &inputStream, const bool firstColumnIsSubPopulation):
 impl(*new ParameterInstancesImpl(inputStream, firstColumnIsSubPopulation)) {
    assert(repOK());
}

ParameterInstances::~ParameterInstances() {
    delete(&impl);
}

vector<double> ParameterInstances::getInstance(const unsigned int instance) const {
    assert(instance < getNumInstances());
    return impl.getInstance(instance);
}

double ParameterInstances::getParameter(const unsigned int instance, const unsigned int parameter) const {
    assert(instance < getNumInstances());
    assert(parameter < getNumParameters());
    return impl.getParameter(instance, parameter);
}

unsigned int ParameterInstances::getNumInstances() const {
    return impl.getNumInstances();
}

unsigned int ParameterInstances::getNumParameters() const {
    return impl.getNumParameters();
}

bool ParameterInstances::repOK() const {
    return impl.repOK();
}


unsigned int ParameterInstances::getSubPopulation(const unsigned int instance) const {
    return impl.getSubPopulation(instance);
}
