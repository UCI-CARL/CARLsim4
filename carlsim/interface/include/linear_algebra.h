/*
 * Copyright (c) 2014 Regents of the University of California. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. The names of its contributors may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * *********************************************************************************************** *
 * CARLsim
 * created by: 		(MDR) Micah Richert, (JN) Jayram M. Nageswaran
 * maintained by:	(MA) Mike Avery <averym@uci.edu>, (MB) Michael Beyeler <mbeyeler@uci.edu>,
 *					(KDC) Kristofor Carlson <kdcarlso@uci.edu>
 *					(TSC) Ting-Shuo Chou <tingshuc@uci.edu>
 *
 * CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
 * Ver 10/11/2014
 */

#ifndef _LINEAR_ALGEBRA_H_
#define _LINEAR_ALGEBRA_H_

#include <ostream>			// print struct info

unsigned long int get_time_ms64();

/*!
 * \brief a point in 3D space
 *
 * A point in 3D space. Coordinates (x,y,z) are of double precision.
 * \param[in] x x-coordinate
 * \param[in] y y-coordinate
 * \param[in] z z-coordinate
 */
struct Point3D {
public:
	Point3D(int _x, int _y, int _z) : x(1.0*_x), y(1.0*_y), z(1.0*_z) {}
	Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

	// print struct info
    friend std::ostream& operator<<(std::ostream &strm, const Point3D &p) {
		strm.precision(2);
        return strm << "Point3D=(" << p.x << "," << p.y << "," << p.z << ")";
    }

    // overload operators
    Point3D operator+(const double a) const { return Point3D(x+a,y+a,z+a); }
    Point3D operator+(const Point3D& p) const { return Point3D(x+p.x,y+p.y,z+p.z); }
    Point3D operator-(const double a) const { return Point3D(x-a,y-a,z-a); }
    Point3D operator-(const Point3D& p) const { return Point3D(x-p.x,y-p.y,z-p.z); }
    Point3D operator*(const double a) const { return Point3D(x*a,y*a,z*a); }
    Point3D operator*(const Point3D& p) const { return Point3D(x*p.x,y*p.y,z*p.z); }
    Point3D operator/(const double a) const { return Point3D(x/a,y/a,z/a); }
    Point3D operator/(const Point3D& p) const { return Point3D(x/p.x,y/p.y,z/p.z); }
    bool operator==(const Point3D& p) const { return Equals(p); }
    bool operator!=(const Point3D& p) const { return !Equals(p); }
    bool operator<(const Point3D& p) const { return (CompareTo(p)<0); }
    bool operator>(const Point3D& p) const { return (CompareTo(p)>0); }
    bool operator<=(const Point3D& p) const { return (CompareTo(p)<=0); }
    bool operator>=(const Point3D& p) const { return (CompareTo(p)>=0); }
	
	// coordinates
	double x, y, z;

private:
	bool Equals(const Point3D& p) const { return (x==p.x && y==p.y && z==p.z); }
	int CompareTo(const Point3D& p) const { return (x>p.x&&y>p.y&&z>p.z) ? 1 : ( (x<p.x&&y<p.y&&z<p.z) ? -1 : 0); }
};

double dist(Point3D& p1, Point3D& p2);

//! calculate norm^2
double norm2(const Point3D& p);

//! calculate norm \FIXME maybe move to carlsim_helper.h or something...
double norm(const Point3D& p);

//! check whether certain point lies on certain grid \FIXME maybe move to carlsim_helper.h or something...
//bool isPointOnGrid(Point3D& p, Grid3D& g);

#endif