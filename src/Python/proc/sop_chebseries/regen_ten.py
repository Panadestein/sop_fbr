#########################
# Once fitted the SPPs with Cheb
# rebuild the resulting energy tensor
# and compare to reference energy tensor
#########################


"""Regenerates core tensor using SPPs from Chebyshev fits"""
import numpy as np
import tensorly as tl
#  from tensorly.tenalg import kronecker as kronti

D0DIM_5 = np.array([5, 5, 5, 5, 5, 12])
CORA_5 = np.loadtxt('core').reshape(D0DIM_5)
EVEC_01 = np.loadtxt('./new_evec_0')
EVEC_02 = np.loadtxt('./new_evec_1')
EVEC_03 = np.loadtxt('./new_evec_2')
EVEC_04 = np.loadtxt('./new_evec_3')
EVEC_05 = np.loadtxt('./new_evec_4')
EVEC_06 = np.loadtxt('./new_evec_5')
E_AB = np.loadtxt('e_ab')
MATS_5 = [EVEC_01, EVEC_02, EVEC_03, EVEC_04, EVEC_05, EVEC_06]

# Omega approach (MEM and CPU extensive)

#  OMEGA = kronti(MATS)
#  E_APP = np.matmul(OMEGA, CORA)

# Tucker approach

E_APP_5 = tl.tucker_tensor.tucker_to_tensor((CORA_5, MATS_5)).flatten()
np.savetxt('e_sop', E_APP_5.flatten())

RMS = np.sqrt(((E_APP_5 - E_AB) ** 2).mean())
print(RMS)
