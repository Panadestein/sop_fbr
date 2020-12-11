"""Generates the Tucker decomposition of the Henon Heiles
potential with the HOOI algorithm"""
#  import sys
import numpy as np
from numpy.polynomial import chebyshev as cheby
import matplotlib.pyplot as plt
import tensorly as tl
import tensorly.decomposition as tldec

# Constants

PLOTS = False
CHEBDIM = 5
GDIM = np.array([6, 6, 6])
NDIM = len(GDIM)
LMB = 0.111803

# Define Henon Heiles potential for 3D


def henhai(x_1, x_2, x_3):
    """Returns the value of the HH potential at the point (x_1, x_2, x_3)"""
    v_hh = ((x_1**2 + x_2**2 + x_3**2) / 2 +
            LMB * (x_1**2 * x_2 + x_2**2 * x_3 - (x_2**3 + x_3**3) / 3))
    return v_hh


# Define grids

X = np.linspace(-9, 7, num=30)
Y = np.linspace(-9, 7, num=30)
Z = np.linspace(-9, 7, num=30)
G_REF = [X, Y, Z]
X_M, Y_M, Z_M = np.meshgrid(X, Y, Z, indexing='ij')

# Compute reference energies tensor

E_REF = np.vectorize(henhai)(X_M, Y_M, Z_M)

# Tensor decomposition and error estimation

CORA, MATS = tldec.tucker(E_REF, ranks=GDIM)
E_RECONS = tl.tucker_tensor.tucker_to_tensor((CORA, MATS))
RMSE = np.sqrt(((E_RECONS - E_REF) ** 2).mean())
print(RMSE)

# Approximate factor matrices with Chebyshev polynomial series

CHEB_COEFF = []
for idx, elem in enumerate(MATS):
    for j_kp in elem.T:
        coeffs, errors = cheby.chebfit(G_REF[idx], j_kp, CHEBDIM, full=True)
        print(errors[0])
        CHEB_COEFF.append(coeffs)
CHEB_COEFF = np.array(CHEB_COEFF).reshape((NDIM, -1, CHEBDIM + 1))

# Plot SPPs and compare with originals

if PLOTS:
    for KAP in range(NDIM):
        for JAP in range(8):
            NEW_G = np.linspace(G_REF[KAP][0], G_REF[KAP][-1], num=100)
            plt.plot(G_REF[KAP], MATS[KAP][:, JAP], '-o')
            plt.plot(NEW_G, cheby.chebval(NEW_G, CHEB_COEFF[KAP, JAP]))
            plt.legend([r'REFE', r'CHEB'])
            plt.title(r'DOF = %d  jk = %d' % (KAP, JAP))
            plt.show()

# Save SOP parameters

PARAMS = np.concatenate((CHEB_COEFF.flatten(), CORA.flatten()))
np.savetxt('params_sopfbr', PARAMS)
