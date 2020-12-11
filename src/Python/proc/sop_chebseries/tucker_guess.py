"""
The present module decomposes a reference tensor in Tucker form
using the Tensorly library
"""
import numpy as np
import scipy.constants as sc
import tensorly as tl
import tensorly.decomposition as tldec
import pot_ros as pot

# System paramters

D0DIM = np.array([5, 5, 5, 5, 5, 12])
AUCM = sc.physical_constants['hartree-inverse meter relationship'][0] * 1e-2

# Define ROSMUS wrapper


def fixros(r_1, r_2, t_2, r_3, t_1, p_h):
    """Wraps evil rosmus surface"""
    return pot.potr(r_3, r_1, r_2, t_1, t_2, p_h)


# Reference data and parameter guess input (X -> contracted DOF)
# KEO HONO  r1 = OH, r2 = N=O, r3 = ON, th1 = HON, th2 = ONO, p1 = torsion

R2_0 = 2.21097956679  # N=O, r2
R3_0 = 2.6947494549  # O-N, r3
R1_0 = 1.82169598494  # OH, r1
T2_0 = 1.93207948196  # ONO, t2
T1_0 = 1.77849050778  # HON, t1
PH_0 = 3.14159265359  # torsion, p

# This is the full grid

R1MIN, R1MAX = 1.30, 2.45
R2MIN, R2MAX = 1.90, 2.60
T2MIN, T2MAX = -0.65, -0.10
R3MIN, R3MAX = 2.10, 3.25
T1MIN, T1MAX = -0.65, 0.25
PHMIN, PHMAX = 0, np.pi

# Read MCTDH grids

with open('ref_grid', 'r') as fgrid:
    DVR = fgrid.read().split("#")

ARR_S = [np.array(elem.strip().split(), dtype=np.float64) for elem in DVR]

R3, R1, R2, T1, T2, PH = ARR_S

# Print relevant information

for ix, crd in enumerate([R1, R2, T2, R3, T1, PH]):
    np.savetxt('dof_%d' % ix, crd)

# Compute energies in grid
# KEO modes    |  r_2   | r_3    |  r_1    |   t_2       |    t_1       |  p_1

#  E_0 = pot.potr(R3_0, R1_0, R2_0, T1_0, T2_0, PH_0)
print(R1_0, R2_0, T2_0, R3_0, T1_0, PH_0)
E_0 = fixros(R1_0, R2_0, T2_0, R3_0, T1_0, PH_0)
print(fixros(1.3, 1.9, -0.25, 2.1, 0.25, 1.42) * AUCM - E_0 * AUCM)
raise SystemExit

R1_M, R2_M, T2_M, R3_M, T1_M, PH_M = np.meshgrid(
    R1, R2, T2, R3, T1, PH, indexing='ij')

#E_AB = (np.vectorize(fixros)(R1_M, R2_M, np.arccos(T2_M), R3_M,
#                             np.arccos(T1_M), PH_M) - E_0) * AUCM


GRID = np.meshgrid(R1, R2, T2, R3, T1, PH, indexing='ij')
INTCOORD = np.vstack(list(map(np.ravel, GRID))).T
np.savetxt('cord_lar_ene', INTCOORD)


CORA, MATS = tldec.tucker(E_AB, ranks=D0DIM)
E_PRIMA = tl.tucker_tensor.tucker_to_tensor((CORA, MATS))
rms = np.sqrt(((E_PRIMA - E_AB) ** 2).mean())

for idx, mat in enumerate(MATS):
    np.savetxt('evec_%d_5' % idx, mat)

print(f'RMS = {rms} cm-1')

with open('error_tucker', 'w') as filerr:
    filerr.write(str(rms))

np.savetxt('e_ab', E_AB.flatten())
np.savetxt('core', CORA.flatten())
