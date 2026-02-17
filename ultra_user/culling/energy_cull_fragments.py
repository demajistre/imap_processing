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
from importlib import reload


spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')

spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260203_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_037_002.ah.bc')

repoint = 47
de = MyUltraFile.L1Bde(repoint,sensor='45').data
xspin = MyUltraFile.L1Bxspin(repoint,sensor='45').data
l1c = MyUltraFile.L1C(repoint,sensor='45').data
status = MyUltraFile.L1Bstatus(repoint,sensor='45').data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

energy_ranges = cull_util.l1c_energy_ranges()

repointings90 = cull_util.get_pointings(27,153)
repointings45 = cull_util.get_pointings(27,153,sensor='45')
#increase bins for more stats
spin_range=100
c90 = cull_util.runculls(repointings90,energy_ranges,spin_range=spin_range)
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',spin_range=spin_range)

full_sum, start_spin = cull_util.full_cntsum(c90)

max_hi_energy_cnts=250
bin_size = 1
ihi = 4
mean_chan0,bedge,bn0 = binned_statistic(full_sum[:,ihi],full_sum[:,ihi],bins=range(0,max_hi_energy_cnts,bin_size))
count_chan0,bedge,bn0 = binned_statistic(full_sum[:,ihi],full_sum[:,ihi],'count',bins=range(0,max_hi_energy_cnts,bin_size))
nbin=len(bedge)-1
nx = len(bn0)

mean_vals = np.ndarray((ihi,nbin))
median_vals = np.ndarray((ihi,nbin))
bn = np.ndarray((ihi,nx))
for i in range(ihi):
    mean_chan0, bedge, bn0 = binned_statistic(full_sum[:, ihi], full_sum[:, i],
                                              bins=range(0, max_hi_energy_cnts, bin_size))
    mdeian_chan0, bedge, bn0 = binned_statistic(full_sum[:, ihi], full_sum[:, i],'median',
                                              bins=range(0, max_hi_energy_cnts, bin_size))
    mean_vals[i,:] = mean_chan0
    median_vals[i,:] = mdeian_chan0
    bn[i,:]=bn0

for ic in range(ihi):
    plt.plot(bedge[:-1],mean_vals[ic,:])
plt.show()

for ic in range(ihi):
    plt.plot(bedge[:-1],median_vals[ic,:])
plt.show()

frac = np.ndarray(nbin)
for i in range(0,nbin):
    frac[i] = np.sum(count_chan0[:i])/np.sum(count_chan0)
plt.plot(range(0,nbin),frac)
plt.show()

#Narrative - select a threshold high energy rate for each energy
#before the slope of the curves start to increase a second time
# 0- 80
# 1- 75
# 2- 60
# 3- 35

thresh0 = np.array([200,150,60,35])
thresh_per_spin = thresh0/spin_range
# for standard 20 spin intervals
thresh20 = thresh_per_spin*20.
print(f'recommended threshold in counts/spin: {thresh_per_spin}')

# current state
c90 = cull_util.runculls(repointings90,energy_ranges)
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45')

cull_util.cullplot(c90)
cull_util.cullplot(c45)

