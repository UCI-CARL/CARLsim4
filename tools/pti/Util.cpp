#include "Util.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! Take a string representing a decimal point number and convert it to a
    * float.
    * 
    * Throws std::invalid_argument if the string does not hold a decimal point
    * number. */
float stringToFloat(const string &str) {
    stringstream stream(str);
    float value;
    stream >> value;

    if (stream.fail())
        throw std::invalid_argument(string("stringToFloat: Could not convert ") + str + string(" to float."));

    return value;
}

/*! Take a string representing a decimal point number and convert it to a
    * double.
    * 
    * Throws std::invalid_argument if the string does not hold a decimal point
    * number. */
double stringToDouble(const string &str) {
    stringstream stream(str);
    double value;
    stream >> value;

    if (stream.fail())
        throw std::invalid_argument(string("stringToDouble: Could not convert ") + str + string(" to double."));

    return value;
}

/*! Consder to floats equal if they are within epsilon of each other. */
bool equals(const float x, const float y, float epsilon) {
    return abs(x - y) < epsilon;
}

/*! Consder to floats equal if they are within epsilon of each other. */
bool equals(const double x, const double y, double epsilon) {
    return abs(x - y) < epsilon;
}
