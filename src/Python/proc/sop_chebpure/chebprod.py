"""Computes Energy of Chebyshev expanded expression.
Uses tensor n-mode products instead the Omega matrix"""
from ctypes import CDLL, c_int, c_void_p
import numpy as np
import numpy.ctypeslib as npct
import tensorly as tl

# Chebyshev vector function (C wrapper)

ARRAY_1D_INT = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
ARRAY_1D_DOUBLE = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

CFUNC = CDLL("./cheb_c.so")
cheb_ten = CFUNC.cheb_vect
cheb_ten.restype = c_void_p
cheb_ten.argtypes = [ARRAY_1D_DOUBLE, ARRAY_1D_DOUBLE,
                     c_int, c_int, ARRAY_1D_INT]

# Load reference data

DATA = np.loadtxt('ref_ab')
NGEOS = DATA.shape[0]
G_AB = DATA[:, :-1].flatten()
E_AB = DATA[:, -1]

# System definition

CHEBDIM = np.array([6, 6, 6], dtype=np.int32)
CTEN_DIM = np.array([6, 6, 6])
NDIM = len(CTEN_DIM)
NCHEB = np.sum(CTEN_DIM * CHEBDIM)

# Generate factorized tensor

PARAMS = np.loadtxt('params_sopfbr')
CHEB = PARAMS[:NCHEB].reshape(NDIM, -1, CHEBDIM[0]).transpose(0, 2, 1)
CORA = PARAMS[NCHEB:].reshape(CTEN_DIM)
ATEN_A = tl.tenalg.multi_mode_dot(CORA, CHEB)
ATEN_B = tl.tenalg.mode_dot(CORA, CHEB[0], mode=0)
ATEN_C = tl.tenalg.multi_mode_dot(CORA, [CHEB[0], CHEB[1]], modes=[0, 1])

# Compute Chebyshev tensor

CHEBFLAT = np.zeros(CHEBDIM[0] * NGEOS * NDIM)
cheb_ten(CHEBFLAT, G_AB, NGEOS, NDIM, CHEBDIM)
MATRICES = np.reshape(CHEBFLAT, (NGEOS, NDIM, -1))

# Compute energies

E_SOP_A = np.zeros(NGEOS)
E_SOP_B = np.zeros(NGEOS)
E_SOP_C = np.zeros(NGEOS)
for idx, cvects in enumerate(MATRICES):
    E_SOP_A[idx] = tl.tenalg.multi_mode_dot(ATEN_A, cvects)
    E_SOP_B[idx] = tl.tenalg.multi_mode_dot(
        ATEN_B, [CHEB[1], CHEB[2], *cvects],
        modes=[1, 2, 0, 1, 2])
    E_SOP_C[idx] = tl.tenalg.multi_mode_dot(
        ATEN_C, [CHEB[2], *cvects],
        modes=[2, 0, 1, 2])

# Compute RMSE

RMSE_A = np.sqrt(((E_AB - E_SOP_A) ** 2).mean())
RMSE_B = np.sqrt(((E_AB - E_SOP_B) ** 2).mean())
RMSE_C = np.sqrt(((E_AB - E_SOP_C) ** 2).mean())
print("FULL EXP", RMSE_A)
print("MODE 0", RMSE_B)
print("MODE 0 and MODE 1", RMSE_C)
