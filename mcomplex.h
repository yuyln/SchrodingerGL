#ifndef __MCOMPLEX_H
#define __MCOMPLEX_H

#ifndef OPENCL_COMP
#include <math.h>
#endif


typedef struct {
	double r, i;
} complex;

complex ccon(complex c) {
	c.i *= -1.0;
	return c;
}

complex cadd(complex c1, complex c2) {
	c1.r += c2.r;
	c1.i += c2.i;
	return c1;
}

complex csub(complex c1, complex c2) {
	c1.r -= c2.r;
	c1.i -= c2.i;
	return c1;
}

complex cmul(complex c1, complex c2) {
	return (complex) { c1.r * c2.r - c1.i * c2.i,
	                   c1.i * c2.r + c1.r * c2.i };
}

complex cdiv(complex c1, complex c2) {
	double div = c2.r * c2.r + c2.i * c2.i;
	return (complex) { (c1.r * c2.r + c1.i * c2.i) / div,
					   (c1.i * c2.r - c1.r * c2.i) / div };
}

complex cexp_(double phase) {
	return (complex) { cos(phase), sin(phase) };
}

double carg_(complex c) {
	return atan2(c.i, c.r);
}

double cmod2(complex c) {
	return c.r * c.r + c.i * c.i;
}



#endif //__MCOMPLEX_H
