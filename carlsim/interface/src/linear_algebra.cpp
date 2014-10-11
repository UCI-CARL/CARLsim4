#include <linear_algebra.h>

#include <user_errors.h>	// CARLsim user errors
#include <cmath>			// sqrt


double dist(Point3D& p1, Point3D& p2) {
	Point3D p( (p1-p2)*(p1-p2) );
	return sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
//	return norm(p); // can't find norm
}

//! calculate norm \FIXME maybe move to carlsim_helper.h or something...
double norm(Point3D p) {
	return sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
}