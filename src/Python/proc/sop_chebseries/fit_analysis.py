##########################
# Correlation plot
# Cumulative RMSE
##########################

import numpy as np
import matplotlib.pyplot as plt

# Defining variables and parsing output files

e_ab = np.loadtxt('e_ref')
e_srp = np.loadtxt('e_sop')

index = np.argsort(e_ab)
e_ab_sorted = np.sort(e_ab)
e_srp_sorted = e_srp[index]

rms = []
for i in np.arange(len(e_ab)):
    val = np.sqrt(((e_ab_sorted[0:i] - e_srp_sorted[0:i]) ** 2).mean())
    rms.append(val)
rms = np.array(rms)

# Plot the energies

fig, ax = plt.subplots()
plt.ylabel(r'$E_{ref} \, (cm^{-1})$', fontsize=16)
plt.xlabel(r'$E_{sop} \, (cm^{-1})$', fontsize=16)
ax.plot(e_ab, e_srp, 'r.', markersize=2, rasterized=True)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig("hono_sop12.pdf", bbox_inches="tight", dpi=300)

plt.plot(e_ab_sorted, rms)
plt.xlabel(r'$E_{ref} \, (cm^{-1})$')
plt.ylabel(r'RMSE $(cm^{-1})$')
plt.savefig("h2o_fit_reg.pdf", bbox_inches="tight", dpi=300)
