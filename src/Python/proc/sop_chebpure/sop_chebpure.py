"""Evaluates and test the transformed SOP (pure Chebyshev Omega)"""
#  import sys
import numpy as np
from numpy.polynomial import chebyshev as cheby
import scipy.constants as sc

# Constants

AUCM = sc.physical_constants['hartree-inverse meter relationship'][0] * 1e-2

# Define Chebyshev labels and functions

PAFNUTY = {'cheb0': lambda x: 1.0,
           'cheb1': lambda x: x,
           'cheb2': lambda x: 2 * x ** 2 - 1.0,
           'cheb3': lambda x: 4 * x ** 3 - 3 * x,
           'cheb4': lambda x: 8 * x ** 4 - 8 * x ** 2 + 1.0,
           'cheb5': lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
           'cheb6': lambda x: 32 * x ** 6 - 48 * x ** 4 + 18 * x ** 2 - 1.0,
           'cheb7': lambda x: 64 * x ** 7 - 112 * x ** 5 + 56 * x ** 3 - 7 * x,
           'cheb8': lambda x: (128 * x ** 8 - 256 * x ** 6 +
                               160 * x ** 4 - 32 * x ** 2 + 1.0),
           'cheb9': lambda x: (256 * x ** 9 - 576 * x ** 7 +
                               432 * x ** 5 - 120 * x ** 3 + 9 * x),
           'cheb10': lambda x: (512 * x ** 10 - 1280 * x ** 8 +
                                1120 * x ** 6 - 400 * x ** 4 +
                                50 * x ** 2 - 1.0)}

# Check that the chebyshev implementation is correct

CHKBE = False
if CHKBE:
    print(cheby.chebval(1.5, [1e3, 2, 3e-3, 4, 5]))
    print(1e3 * PAFNUTY['cheb0'](1.5) + 2 * PAFNUTY['cheb1'](1.5) +
          3e-3 * PAFNUTY['cheb2'](1.5) + 4 * PAFNUTY['cheb3'](1.5) +
          5 * PAFNUTY['cheb4'](1.5))


def chebvect(point):
    """Returns the vector of Chebyshev series"""
    vect = []
    for term in range(11):
        vect.append(PAFNUTY['cheb%d' % term](point))
    return vect


# Load new SOP

SOP = np.loadtxt('pes_section', delimiter='|', dtype=str,
                 converters={1: lambda s: s.strip(),
                             2: lambda s: s.strip(),
                             3: lambda s: s.strip()})

# Define SOP function


def sop_cheb_omg(r_1, r_2, theta):
    """Computes the energy of the Chebyshev-Omega PES"""
    suma = 0
    for elem in SOP:
        suma += (float(elem[0]) * PAFNUTY[elem[1]](r_1) *
                 PAFNUTY[elem[2]](r_2) * PAFNUTY[elem[3]](theta))
    return suma


# Compute RMSE with respect to referece energies

DATA = np.loadtxt('ref_ab')
E_AB = DATA[:, -1]

E_SOP = []
for r1, r2, th in DATA[:, :-1]:
    E_SOP.append(sop_cheb_omg(r1, r2, th))
E_SOP = np.array(E_SOP)

RMSE = np.sqrt(((E_AB - E_SOP) ** 2).mean())
print(RMSE)
