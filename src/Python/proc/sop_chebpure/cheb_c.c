#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double cheb_pol(double x_val, int degree) {
	// Evaluates the Chebyshev polynomial of "degree" in point "x_val"
	if (degree == 0) {
			return 1;
		}	
	else if (degree == 1) {
			return x_val;
		}
	else {
		return 2 * x_val * cheb_pol(x_val, degree-1) - cheb_pol(x_val, degree-2);
	}
}

void cheb_vect(double *u_vects, double *geo, int ngeos, int dofs, int *chebdim) {
	// Modifies the vector of Chebyshev polynomials
	int jkappa = 0;
	int gcount = 0;
	for (int k = 0; k < ngeos; ++k) {
		for (int i = 0; i < dofs; ++i) {
			for (int j = 0; j < chebdim[i]; ++j) {
				u_vects[jkappa] = cheb_pol(geo[gcount], j);
				jkappa += 1;
			}
			gcount +=1;
		}
	}
}
