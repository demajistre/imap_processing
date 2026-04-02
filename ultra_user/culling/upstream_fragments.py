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


spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0116.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
spiceypy.furnsh('data/imap/spice/spk/imap_pred_od024_20260217_20260331_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260217_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_353_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_354_2025_356_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_356_2025_358_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_358_2025_360_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_051_002.ah.bc')

repoint = 55
de = MyUltraFile.L1Bde(repoint,sensor='45').data
xspin = MyUltraFile.L1Bxspin(repoint,sensor='45').data
l1c = MyUltraFile.L1C(repoint,sensor='45').data
status = MyUltraFile.L1Bstatus(repoint,sensor='45').data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

energy_ranges = cull_util.l1c_energy_ranges()

repointings90 = cull_util.get_pointings(27,180)
repointings45 = cull_util.get_pointings(27,180,sensor='45')
#

c90 = cull_util.runculls(repointings90,energy_ranges)
c90u = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_stat_v1")

pointing = 87

###
cnt_sum = c90['cnt_sum'][pointing]
ecull = c90['ecull'][pointing]
cnt_sumu = c90['cnt_sum'][pointing]
ecullu = c90['ecull'][pointing]
# these are the same for both culls
sb = c90['cullData'][pointing].spinbins
umask1 = c90u['upcull1'][pointing]['mask']
scaledc1 = c90u['upcull1'][pointing]['scaled_counts']
totalsc1 = c90u['upcull1'][pointing]['totalScaled']
thresh1 = c90u['upcull1'][pointing]['thresh']
scaledc2 = c90u['upcull2'][pointing]['scaled_counts']
totalsc2 = c90u['upcull2'][pointing]['totalScaled']
thresh2 = c90u['upcull2'][pointing]['thresh']
umask2 = c90u['upcull2'][pointing]['mask']

# checking results

lims =[30,25,20,10]
fig, axs = plt.subplots(4)
for ich in range(4):
    mask = c90['cullData'][pointing].currentMask['bin_mask']
    masku = c90u['cullData'][pointing].currentMask['bin_mask']
    csum = c90['cnt_sum'][pointing]
    sb = c90['cullData'][pointing].spinbins
    ii = np.nonzero(mask[ich, :])[0]
    iiu = np.nonzero(masku[ich, :])[0]

    axs[ich].plot(sb, csum[:, ich], "r")
    axs[ich].plot(sb[ii], csum[ii, ich], 'b')
    axs[ich].plot(sb[iiu], csum[iiu, ich], 'g')
    axs[ich].set_ylim(0, lims[ich])
plt.show()
