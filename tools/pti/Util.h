#ifndef UTIL_H
#define UTIL_H

#include <string>

using namespace std;

float stringToFloat(const string &str);
double stringToDouble(const string &str);
bool equals(float x, float y, float epsilon = 0.000001);
bool equals(double x, double y, double epsilon = 0.000001);

#endif // UTIL_H
