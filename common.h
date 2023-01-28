#include "mcomplex.h"
#define HBAR 1.0//(1.054571817e-34)
#define QE 1.0//(1.60217663e-19)
#define ME 1.0//(9.1093837e-31)
#define e0 1.0//(8.8541878128e-12)
#define E0 1.0//(2.1798721788e-18)
#define a0 1.0//(5.2917723276e-11)

double V(double x, double y) {
	double xU = x / a0;
	double yU = y / a0;
	int XX = xU > 0.1 && xU < 0.11;
	int upper = XX && yU > 0.15 && yU < 1.0;
	int inner = XX && yU > -0.05 && yU < 0.05;
	int lower = XX && yU > -1.0 && yU < -0.15;
	return (upper || inner || lower) * 30000.0;
	return -(xU > 0.3 && xU < 0.6 && yU > 0.3 && yU < 0.6) * 3000.0;
}

int boundary(int i, int limit) {
	if (i >= limit) return i % limit;
	if (i < 0) return (i % limit) + limit;
	return i;
}

complex laplacian(const complex *psi, int row, int col, int nrows, int ncols,
				  double dx, double dy, complex dpsi) {
	int colR = boundary(col + 1, ncols);
	int colL = boundary(col - 1, ncols);
	int rowU = boundary(row + 1, nrows);
	int rowD = boundary(row - 1, nrows);
	
	complex right = psi[ncols * row + colR];
	complex left = psi[ncols * row + colL];
	complex up = psi[ncols * rowU + col];
	complex down = psi[ncols * rowD + col];
	complex current = cadd(psi[ncols * row + col], dpsi);

	if (col + 1 >= ncols) right = (complex){0, 0};
	if (col - 1 < 0) left = (complex){0, 0};
	if (row + 1 >= nrows) up = (complex){0, 0};
	if (row - 1 < 0) down = (complex){0, 0};
							 
	

	complex laplacian_x = cmul(csub(cadd(right, left),
									cmul(current, (complex){2, 0})),
							   (complex){1.0 / (dx * dx), 0});

	complex laplacian_y = cmul(csub(cadd(up, down),
									cmul(current, (complex){2, 0})),
							   (complex){1.0 / (dy * dy), 0});

	return cadd(laplacian_x, laplacian_y);	
}

complex dpsi_dt(double x0, double y0, int row, int col, int nrows, int ncols, double mass, double dx, double dy,
		        double dt, double t, const complex *psi0) {
	(void)mass;
	(void)t;
	complex r1 = {0}, r2 = {0}, r3 = {0}, r4 = {0};
	int current = row * ncols + col;
	double x = x0 + col * dx;
	double y = y0 + row * dy;
	
	{//RK1
    	double v = V(x, y);
		complex l = laplacian(psi0, row, col, nrows, ncols, dx, dy, (complex){0.0, 0.0});
		l = cmul((complex){0, 1.0 / 2.0}, l);
		complex v_psi = cmul(psi0[current], (complex){0, -v});
		r1 = cadd(l, v_psi);
	}
	{//RK2
		double v = V(x, y);
		complex l = laplacian(psi0, row, col, nrows, ncols, dx, dy, cmul(r1, (complex){0.5 * dt, 0.0}));
		l = cmul((complex){0, 1.0 / 2.0}, l);
		complex v_psi = cmul(cadd(psi0[current], cmul(r1, (complex){0.5 * dt, 0})), (complex){0, -v});
		r2 = cadd(l, v_psi);		
	}
	{//RK3
		double v = V(x, y);
		complex l = laplacian(psi0, row, col, nrows, ncols, dx, dy, cmul(r2, (complex){0.5 * dt, 0.0}));
		l = cmul((complex){0, 1.0 / 2.0}, l);
		complex v_psi = cmul(cadd(psi0[current], cmul(r2, (complex){0.5 * dt, 0})), (complex){0, -v});
		r3 = cadd(l, v_psi);	
	}
	{//RK4
		double v = V(x, y);
		complex l = laplacian(psi0, row, col, nrows, ncols, dx, dy, cmul(r3, (complex){dt, 0.0}));
		l = cmul((complex){0, 1.0 / 2.0}, l);
		complex v_psi = cmul(cadd(psi0[current], cmul(r3, (complex){dt, 0})), (complex){0, -v});
		r4 = cadd(l, v_psi);	
	}
	complex RK = cadd(cadd(r1, cmul(r2, (complex){2.0, 0.0})), cadd(cmul(r3, (complex){2.0, 0.0}), r4));
	return cmul(RK, (complex){dt / 6.0, 0.0});
}
