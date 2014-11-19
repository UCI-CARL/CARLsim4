#include <linear_algebra.h>

#include <sys/time.h>		// gettimeofday
#include <cmath>			// sqrt

/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
* windows and linux.
*/
unsigned long int get_time_ms64() {
#ifdef WIN32
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;
	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	* to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	unsigned long int ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */
	return ret;
#else
	/* Linux */
	struct timeval tv;
	gettimeofday(&tv, NULL);
	unsigned long int ret = tv.tv_usec;
	/* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
	ret /= 1000;
	/* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
	ret += (tv.tv_sec * 1000);
	return ret;
#endif
}

double dist(Point3D& p1, Point3D& p2) {
	return norm((p1-p2)*(p1-p2));
}

//! calculate norm \FIXME maybe move to carlsim_helper.h or something...
double norm(Point3D p) {
	return sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
}