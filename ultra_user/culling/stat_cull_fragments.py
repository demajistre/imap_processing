import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import ultra_user.culling.cull_util as cull_util
import spiceypy
import ultra_user.planets.ENA_planets as ENA_planets
import imap_processing.spice.time as spiceTime
from importlib import reload

# repointings to focus (Nov 20 - Dec 6)
repointings90 = cull_util.get_pointings(64,70)
repointings45 = cull_util.get_pointings(64,70,sensor='45')

energy_ranges = cull_util.l1c_energy_ranges()
c90 = cull_util.runculls(repointings90,energy_ranges)
c45 = cull_util.runculls(repointings45,energy_ranges)

for rp in repointings90:
    print(f"{rp}: {c90['scull'][rp]['converge']} : {c90['scull'][rp]['iterations']}")
#yields (45 gives same answer!):
#64: [ True  True  True  True False False] : [1 3 2 2 5 5]
#65: [ True  True  True  True  True False] : [1 1 1 2 1 5]
#66: [ True  True  True  True  True False] : [1 1 1 1 1 5]
#67: [ True  True  True  True  True False] : [1 1 1 1 1 5]
#68: [False False  True  True  True False] : [5 5 2 1 1 5]
#69: [ True  True False False  True False] : [1 1 5 5 1 5]

# looking at 64 and multiple iterations
repoint=64
nen=len(energy_ranges[:,0])
cullData = c90['cullData'][repoint]
csum = cullData.get_count_summary()
spinStart = cullData.spinbins[:,0]

premask = c90['ecull'][repoint]['mask']
for ie in range(nen):
    premask[ie,:] = np.logical_and(premask[ie,:],c90['vcull'][repoint]['binMask'])
scullMask = c90['scull'][repoint]['mask']
fullMask = np.logical_and(premask,scullMask)



ylims = [30,20,15,10]

nch=4
fig, axs = plt.subplots(nch)
for ic in range(nch):
    ii = np.nonzero(premask[ic,:])[0]
    jj = np.nonzero(fullMask[ic,:])[0]
    axs[ic].plot(spinStart,csum[:,ic],'r')
    axs[ic].plot(spinStart[ii], csum[ii, ic], 'b')
    axs[ic].plot(spinStart[jj], csum[jj, ic], 'g')
    axs[ic].set_ylim(0,ylims[ic])
plt.show()
