#############################
# Get the Cheb fir to the guess SPPs
#############################


"""Fits SPPs to Chebyshev series"""
import numpy as np
#  from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
#  from scipy.optimize import curve_fit
from numpy.polynomial import chebyshev as cheby
#  from numpy.polynomial.chebyshev import Chebyshev as cheo
#  from numpy.polynomial import legendre as legend
#  from numpy.polynomial import laguerre as lager
#  from numpy.polynomial import hermite as hermes
#  from numpy.polynomial import polynomial as pol

# Define a generic fitting function


def messer(x_dvr, a0, a1, a2, a3, a4):
    """An amplified cosine function"""
    val = a0 + a1 * np.sin(a2 * x_dvr) * np.cos(a3 * x_dvr + a4)
    return val


WEIGHT = np.concatenate(([0.3] * 4, np.ones(22), [0.3] * 4))

chebs = []
for DOF, JOTAS in enumerate([5, 5, 5, 5, 5, 12]):
    EVEC = np.loadtxt('evec_%d_5' % DOF)
    COORD = np.loadtxt('dof_%d' % DOF)
    NEWCOORD = np.linspace(COORD[0], COORD[-1], num=20)
    new_evec = []
    new_evec_aug = []
    for JVAL in range(JOTAS):
        if DOF == 5:
            CHEBDIM = 11
        else:
            CHEBDIM = 7
        elran = np.linspace(COORD[0], COORD[-1], num=100)
        coeff, stuff = cheby.chebfit(COORD, EVEC[:, JVAL],
                                     CHEBDIM, full=True)
        #  coeff, stuff1 = legend.legfit(COORD, EVEC[:, JVAL],
        #                                CHEBDIM, full=True)
        #  coeff, stuff2 = lager.lagfit(COORD, EVEC[:, JVAL],
        #                               CHEBDIM, full=True)
        #  coeff, stuff3 = hermes.hermfit(COORD, EVEC[:, JVAL],
        #                                 CHEBDIM, full=True)
        #  coeff, stuff4 = pol.polyfit(COORD, EVEC[:, JVAL],
        #                              CHEBDIM, full=True)
        #  elchy, stufy = cheo.fit(COORD, EVEC[:, JVAL], CHEBDIM,
        #                          domain=None, full=True)
        #  popt, pcov = curve_fit(messer, COORD,
        #                         EVEC[:, JVAL], maxfev=10000000)
        #  espli = splrep(COORD, EVEC[:, JVAL], k=5)
        chebs.append(coeff)
        #  print(coeff)
        #  print("#")
        #  print(elchy)
        #  print(elchy.domain)
        #  print(elchy.window)
        #  print(stuff[0][0], stuff1[0][0], stuff2[0][0], stuff3[0][0],
        #        stuff4[0][0])
        if False:
            plt.title("dof = %d, j = %d, err = %f" % (DOF, JVAL, stuff[0][0]))
            plt.plot(COORD, EVEC[:, JVAL], 'o')
            plt.plot(elran, cheby.chebval(elran, coeff))
            #  plt.plot(elran, splev(elran, espli))
            #  plt.plot(elran, elchy(elran))
            #  plt.plot(elran, messer(elran, *popt))
            plt.legend(['SPP', 'CHEB', 'SPLINE', 'MESSER'])
            plt.show()
        new_evec.append(cheby.chebval(COORD, coeff))
        new_evec_aug.append(cheby.chebval(NEWCOORD, coeff))
        #  if stuff[0][0] >= 0.1:
        #  print(f"Mal asunto DOF = {DOF}, j = {JVAL}") #  ERR = {stuff[0][0]}"
    new_evec = np.array(new_evec)
    print(new_evec.shape)
    #  new_evec_aug = np.array(new_evec_aug)
    np.savetxt('new_evec_%d' % DOF, new_evec.reshape(-1, COORD.shape[0]).T)
    #  np.savetxt('new_evec_aug_%d' % DOF,
    #             new_evec_aug.reshape(-1, NEWCOORD.shape[0]).T)
chebs = np.concatenate(chebs)
np.savetxt('new_chebs', chebs.flatten())
