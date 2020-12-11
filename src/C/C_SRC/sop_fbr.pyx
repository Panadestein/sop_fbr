"""
The present module contains the Cython SOP_FBR function
"""
cimport cheb_c
cimport numpy as np
import numpy as np
import tensorly as tl

# SOP-FBR function


cpdef double vchpot(np.ndarray[np.float64_t, ndim=1] chebs,
             np.ndarray[np.float64_t, ndim=1] core,
             np.ndarray[np.float64_t, ndim=1] q_array,
             np.ndarray[np.int64_t, ndim=1] gdim,
             int dof, int chebdim):
    """ Computes the value of the SOP-FBR potential by first
        conforming the vij(k) matrices, then reshaping
        the core tensor, and penforming the tensor dot product.
    """
    cdef double prod
    cdef np.ndarray[np.float64_t, ndim=1] u_vect
    cdef np.ndarray[np.float64_t, ndim=2] v_vectors

    u_vect = np.zeros(dof * chebdim, dtype=np.float64)
    cheb_c.cheb_vect(chebs, u_vect, dof, chebdim)
    v_matrices = np.reshape(u_vect, (dof, -1)) 

    prod = tl.tucker_tensor.tucker_to_tensor((core, v_matrices))

    return prod

