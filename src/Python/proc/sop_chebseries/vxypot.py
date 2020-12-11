"""
The present module performs the optimization of the coefficients of the SOP-FBR
representation of the 6D PES of the HONO cis-trans isomerization. It has
dependencies with on the Numpy, Tensorly and Scipy packages
"""
import numpy as np
from numpy.polynomial import chebyshev as cheby
from scipy.optimize import minimize as mini
import scipy.constants as sc
import tensorly as tl
import pot_ros as pot

# Constants

AUCM = sc.physical_constants['hartree-inverse meter relationship'][0] * 1e-2

# System paramters

CHEBDIM = np.array([8, 8, 8, 8, 8, 12])
CTEN_DIM = np.array([5, 5, 5, 5, 5, 12])
CCHEB_SLICE = np.cumsum(CTEN_DIM)
RESER = np.inf
THRESH = 10000000

# Define ROSMUS wrapper


def fixros(*q_array):
    """Wraps evil rosmus surface"""
    r_1, r_2, t_2, r_3, t_1, p_h = q_array
    return pot.potr(r_3, r_1, r_2, t_1, t_2, p_h)


# SOP-FBR


def pot_grid(chebs, core):
    """Computes the SOP potential for the reference geometries using the
    tensor n-mode approach"""

    v_matrices = []
    idx_cheb = 0
    for kdof, m_kp in enumerate(CTEN_DIM):
        v_kp = np.zeros((len(G_AB[kdof]), m_kp))
        for j_kp in np.arange(m_kp):
            for i_kp, val in enumerate(G_AB[kdof]):
                v_kp[i_kp, j_kp] = cheby.chebval(
                    val, chebs[idx_cheb:idx_cheb + CHEBDIM[kdof]])
            idx_cheb += CHEBDIM[kdof]
        v_matrices.append(v_kp)

    core = core.reshape(CTEN_DIM)
    prod = tl.tucker_tensor.tucker_to_tensor((core, v_matrices))

    return prod


# RMS


def rho(carray):
    """Objective function for the optimization
    of the full parameter space
    """
    e_sop = pot_grid(carray[:NCHEB], carray[NCHEB:])

    rms = np.sqrt(np.mean(np.square(e_sop - E_AB)))
    np.savetxt('e_sop', e_sop.flatten())
    print(rms)

    with open("rms", "a") as file_target:
        file_target.write(str(rms) + "\n")

    return rms


# Reference data and parameter guess input

R1_0 = 1.82169598494  # OH, r1
R2_0 = 2.21097956679  # N=O, r2
T2_0 = 1.93207948196  # ONO, t2
R3_0 = 2.6947494549  # O-N, r3
T1_0 = 1.77849050778  # HON, t1
PH_0 = 3.14159265359  # torsion, p

#  Original grid
R1MIN, R1MAX = 1.30, 2.45
R2MIN, R2MAX = 1.90, 2.60
T2MIN, T2MAX = -0.65, -0.10
R3MIN, R3MAX = 2.10, 3.25
T1MIN, T1MAX = -0.65, 0.25
PHMIN, PHMAX = 0, np.pi

NCHEB = np.sum(CTEN_DIM * CHEBDIM)
CARRAY = np.loadtxt('params_init')
CARRAY[NCHEB:][np.abs(CARRAY[NCHEB:]) < THRESH] = 0.

# This is a low energy grid (near equilibrium values)
#  R1MIN, R1MAX = 1.75, 1.9
#  R2MIN, R2MAX = 2.1, 2.4
#  R3MIN, R3MAX = 2.5, 2.9
#  T2MIN, T2MAX = np.cos([1.8, 2.0])
#  T1MIN, T1MAX = np.cos([1.3, 1.8])
#  PHMIN, PHMAX = 0, np.pi

R1 = np.linspace(R1MIN, R1MAX, num=8)
R2 = np.linspace(R2MIN, R2MAX, num=8)
T2 = np.linspace(T2MIN, T2MAX, num=8)
R3 = np.linspace(R3MIN, R3MAX, num=8)
T1 = np.linspace(T1MIN, T1MAX, num=8)
PH = np.linspace(PHMIN, PHMAX, num=10)

E_0 = fixros(R1_0, R2_0, T2_0, R3_0, T1_0, PH_0)

R1_M, R2_M, T2_M, R3_M, T1_M, PH_M = np.meshgrid(
    R1, R2, T2, R3, T1, PH, indexing='ij')

G_AB = [R1, R2, T2, R3, T1, PH]

E_AB = (np.vectorize(fixros)(R1_M, R2_M, np.arccos(T2_M), R3_M,
                             np.arccos(T1_M), PH_M) - E_0) * AUCM
np.savetxt('e_ref', E_AB.flatten())

# Total number of Chebyshev polinomial's coefficients

NCHEB = np.sum(CTEN_DIM * CHEBDIM)

# Total parameter array and core tensor

CARRAY = np.loadtxt('params_init')
CARRAY[NCHEB:][np.abs(CARRAY[NCHEB:]) < THRESH] = 0.

# Fitting process

TOL = 1e-6
ITERS = 0
while RESER >= TOL:
    if ITERS == 10:
        break

    # Optimize Chebyshev series coefficients

    PARAMS_OPT_CHEB = mini(rho, CARRAY,
                           method='Powell', options={'maxfev': 9000})
    CARRAY = PARAMS_OPT_CHEB.x
    RESER = PARAMS_OPT_CHEB.fun

    # Track RMSE evolution

    np.savetxt('opt_params_%d' % ITERS, CARRAY)
    with open('out_reser', 'a') as outr:
        outr.write(str(ITERS) + " RMS =  " + str(RESER) + "\n")

    ITERS += 1

np.savetxt('opt_params_final', CARRAY)
