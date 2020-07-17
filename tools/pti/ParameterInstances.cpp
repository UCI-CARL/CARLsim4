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


/********************************************************
 * Private methods for pointer-to-implementation pattern.
 ********************************************************/
struct ParameterInstances::ParameterInstancesImpl {
    vector< vector< double > > instanceVectors;
    
    ParameterInstancesImpl(istream &inputStream) {
        readCSV(inputStream, instanceVectors);
        assert(repOK());
    }

    void readCSV(istream &inputStream, vector< vector<double> > &instanceVectors) {
        string strLine;
        while (getline(inputStream, strLine))
            instanceVectors.push_back(readCSVLine(strLine));
        if (!allRowsEqualLength(instanceVectors))
            throw invalid_argument(string(typeid(*this).name()) + string(": rows are not of equal length."));
    }

    vector<double> readCSVLine(const string &strLine) {
        string doubleString;
        stringstream strLineStream(strLine);

        vector<double> lineVector;

        while (getline(strLineStream, doubleString, ',')) {
            lineVector.push_back(stringToDouble(doubleString));
        }

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
};


/********************************************************
 * Public Methods
 ********************************************************/


ParameterInstances::ParameterInstances(istream &inputStream):
 impl(*new ParameterInstancesImpl(inputStream)) {
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
