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

repointings90 = cull_util.get_pointings(27,153)
repointings45 = cull_util.get_pointings(27,153,sensor='45')
#increase bins for more stats
c90 = cull_util.runculls(repointings90,energy_ranges)
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45')

cull_util.cullplot(c90)
cull_util.cullplot(c45)

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
    plt.plot(bedge[:-1],mean_vals[ic,:],label=ic)
    plt.xlim(0,120)
    plt.ylim(0,30)
    plt.legend()
plt.show()

for ic in range(ihi):
    plt.plot(bedge[:-1]/c90['cullData'][repoint].spin_range,mean_vals[ic,:],label=ic)
    plt.xlim(0,10)
    plt.ylim(0,30)
    plt.legend()
plt.show()


for ic in range(ihi):
    plt.plot(bedge[:-1],median_vals[ic,:],label=ic)
    plt.xlim(0,120)
    plt.ylim(0,30)
    plt.legend()
plt.show()

# by eye
energy_ranges = cull_util.l1c_energy_ranges(base_ebin=3)
hist_breakpoints_full = [60,50,45,40,40]
#hist_breakpoints_spin = np.array(hist_breakpoints_full)/c90['cullData'][repoint].spin_range
hist_breakpoints_spin = np.array([2., 1.5, 0.6, 0.2,.2])

c90v1 = cull_util.runculls(repointings90,energy_ranges,sep_threshold_per_spin=hist_breakpoints_spin,nAddChans=3)
c45v1 = cull_util.runculls(repointings45,energy_ranges,sep_threshold_per_spin=hist_breakpoints_spin,sensor='45',nAddChans=3)
