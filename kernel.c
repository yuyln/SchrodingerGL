#include "common.h"

kernel void Step(int nrows, int ncols, double mass, double dx, double dy, 
				 double dt, double t, global complex *psi, global const complex *psi0,
				 double x0, double y0, global double* potential) {
	int i = get_global_id(0);
	int col = i % ncols;
	int row = (i - col) / ncols;
	complex dpsidt = dpsi_dt(x0, y0, row, col, nrows, ncols, mass, dx, dy, dt, t, psi0);
	psi[i] = cadd(psi0[i], dpsidt);
	double x = x0 + col * dx;
	double y = y0 + row * dy;
	potential[i] = V(x, y, t);
}

kernel void Normalize(global complex *psi, const double norm) {
	int i = get_global_id(0);
	psi[i] = cmul(psi[i], (complex){1.0 / norm, 0.0});
}

kernel void psi2(const global complex *psi, global double *psi_2) {
	int i = get_global_id(0);
	psi_2[i] = cmul(psi[i], ccon(psi[i])).r;
}

kernel void to_rgba(global complex *psi0, global float4 *psi_tex, double max_norm2,
					global double *potential, double max_pot, double min_pot,
					global float4 *pot_tex) {
	int i = get_global_id(0);
	double psi_2 = cmul(psi0[i], ccon(psi0[i])).r / max_norm2;
	double phase = carg_(psi0[i]);
	psi_tex[i].x = psi_2;
	psi_tex[i].y = psi_2;
	psi_tex[i].z = psi_2;
	psi_tex[i].x *= 0.5 * sin(phase) + 0.5;
	psi_tex[i].y *= 0.5 * sin(phase + 3.14 / 2) + 0.5;
	psi_tex[i].z *= 0.5 * sin(phase + 3.14) + 0.5;
	psi_tex[i].w = psi_2;

    double a = 1.0 / (min_pot - max_pot);
	double b = -max_pot / (min_pot - max_pot);

	pot_tex[i].x = a * potential[i] + b;
	pot_tex[i].y = a * potential[i] + b;
	pot_tex[i].z = a * potential[i] + b;
	pot_tex[i].w = 1 - (a * potential[i] + b);
}

kernel void Att(const global complex *psi, global complex *psi0) {
	int i = get_global_id(0);
	psi0[i].r = psi[i].r;
	psi0[i].i = psi[i].i;
}
