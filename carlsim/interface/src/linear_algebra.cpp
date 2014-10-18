#include <linear_algebra.h>

#include <cmath>			// sqrt


double dist(Point3D& p1, Point3D& p2) {
	return norm((p1-p2)*(p1-p2));
}

//! calculate norm \FIXME maybe move to carlsim_helper.h or something...
double norm(Point3D p) {
	return sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
}