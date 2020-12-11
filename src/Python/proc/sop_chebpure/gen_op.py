"""Generates the MCTDH operator file
for a given the set of SOP-FBR parameters"""
import itertools
import scipy.constants as sc
import numpy as np
import tensorly as tl

# Constants

AUCM = sc.physical_constants['hartree-inverse meter relationship'][0] * 1e-2

# System definition

GDIM = np.array([6, 6, 6])
NDIM = len(GDIM)
CHEBDIM = 6
NCHEB = np.sum(GDIM) * CHEBDIM
COMBCORA = [np.arange(i) for i in GDIM]
COMBCHEB = [np.arange(i) for i in [CHEBDIM, CHEBDIM, CHEBDIM]]

# Optimized parameters

PARAMS = np.loadtxt('params_sopfbr')
CHEB = PARAMS[:NCHEB].reshape(NDIM, -1, CHEBDIM).transpose(0, 2, 1)
CORA = PARAMS[NCHEB::]
CORATEN = CORA.reshape(GDIM)
NEWCORA = np.zeros((CHEBDIM ** NDIM))
NEWCORATEN = tl.tenalg.multi_mode_dot(CORATEN, [CHEB[0], CHEB[1], CHEB[2]],
                                      modes=[0, 1, 2])
np.savetxt('newten', NEWCORATEN.flatten())

# Generate PES section of the Hamiltonian

PRODS = "{:.14E} | cheb{:d} | cheb{:d} | cheb{:d} \n"

with open('pes_section', 'w') as soppes:
    for cchebs, set_chebs in enumerate(itertools.product(*COMBCHEB)):
        for cjota, set_j in enumerate(itertools.product(*COMBCORA)):
            TERNA = CORA[cjota]
            for kdof in np.arange(NDIM):
                TERNA *= CHEB[kdof, set_chebs[kdof], set_j[kdof]]
            NEWCORA[cchebs] += TERNA
        soppes.write(PRODS.format(NEWCORA[cchebs], *set_chebs))
