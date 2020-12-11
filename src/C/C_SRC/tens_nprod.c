#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifndef CHEBDIM
#define CHEBDIM 10
#endif
#ifndef DOFS
#define DOFS 6
#endif

void kronvectmul(double *u_n, double *v_n, double *kron) {
	// Evaluates the kronecker product of two vectors
    for(int i = 0; i < DOFS; i++) {
        for (int j = 0; j < DOFS; j++) {
            kron[i * DOFS + j] = u_n[i] * v_n[j];
        }
    }
    return;
}

double chebpolyn(double x_val, int degree) {
	// Evaluates the Chebyshev polynomial of "degree" in point "x_val"
	if (degree == 0) {
			return 1;
		}	
	else if (degree == 1) {
			return x_val;
		}
	else {
		return 2 * x_val * chebpolyn(x_val, degree-1) - chebpolyn(x_val, degree-2);
	}
}

double chebserie(double coeffs[CHEBDIM], double x_val) {
	// Evaluates the Chebyshev series with "coeffs" in point "x_val"
	double serieval = 0; 
	for (int i = 0; i < CHEBDIM; i++) {
		serieval += coeffs[i] * chebpolyn(x_val, CHEBDIM);
	}
	return serieval;
}

double sop_fbr(double *omega, double *cora, int nconf) {
	double pot = 0;
	for (int i = 0; i < nconf; i++) {
		pot += omega[i] * cora[i];
	}
	return pot;
}

int main() {

	// Point evaluated

	double XVAL[DOFS] = {1, 1, 1, 1, 1, 1};

	// Define relevant variables

	int GDIM[DOFS] = {5, 5, 5, 5, 5, 5}; // m_k array
	int NJOTAS = 5; // Obvious gotcha (need to make it array of dim DOFS)

	int NCHEB = 0; // total number of parameters
	int NCONF = 1; // total number of configurations
	int NPARS = 0; // total number of parameters

	for (int i = 0; i < DOFS; i++) {
		NCHEB += GDIM[i] * CHEBDIM;
		NCONF *= GDIM[i];
	}

	NPARS = NCONF + NCHEB;

	double CHEB[NPARS]; // array with Chebyshev coeff
	double CORA[NCONF]; // array with core tensor coeff
	double OMEGA[NCONF]; // The omega matrix

	// Read the parameters of the SOP expansion

	FILE * parfil;

	parfil = fopen("./params_sop", "r");
	if (!parfil)
		exit(EXIT_FAILURE);
	for (int i = 0; i < NPARS; i++) {
		if (i <= NCHEB) {
			fscanf(parfil, "%lf", &CHEB[i]);		
		}
		else {
			fscanf(parfil, "%lf", &CORA[i]);		
		}
	}

	fclose(parfil);


	double sercoeff[CHEBDIM];
	double uvect[NJOTAS]; // Placeholder for factor vectors
	double uvect0[5] = {1, 0, 0, 0, 0};
	for (int i = 0; i < DOFS; i++) {
		for (int j = 0; j < NJOTAS; j++) {
			for (int k = 0; k < CHEBDIM; k++) {
				sercoeff[k] = CHEB[i + j + k];
			}
			uvect[j] = chebserie(sercoeff, XVAL[i]);
		}
		kronvectmul(uvect, uvect0, OMEGA);
		uvect0 = uvect;
	}

	return 0;
}
