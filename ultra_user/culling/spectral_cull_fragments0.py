import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import ultra_user.culling.cull_util as cull_util
import spiceypy
import pickle
import ultra_user.planets.ENA_planets as ENA_planets
import imap_processing.spice.time as spiceTime
import ultra_user.utils.ultra_utils as utils
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
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_084_001.ah.bc')

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

repointings90 = cull_util.get_pointings(27,183)
repointings45 = cull_util.get_pointings(27,183,sensor='45')
#increase bins for more stats
c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_stat_v1")
c90s = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45s = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")


culls = [c45,c90,c45s,c90s]
rp = [repointings45,repointings90,repointings45,repointings90]

nch = 6
for icull in range(len(rp)):
    print(icull)
    cull = culls[icull]
    pointings = np.array(rp[icull])
    npoint = len(pointings)
    nchan = len(cull['scull'][80]['converge'])

    conv = np.ndarray((npoint, nchan), dtype=bool)
    cfrac = np.ndarray((npoint, nchan), dtype=float)
    culledRate = np.ndarray((npoint, nchan), dtype=float)
    culledRateErr = np.ndarray((npoint, nchan), dtype=float)
    for ic in range(npoint):
        #print(pointings[ic])
        conv[ic, :] = cull['scull'][pointings[ic]]['converge']
        cfrac[ic, :] = cull['cullData'][pointings[ic]].currentCullFraction()
        cmask = cull['cullData'][pointings[ic]].currentMask['bin_mask']
        for chan in range(nchan):
            ii = np.nonzero(cmask[chan, :])[0]
            nii = len(ii)
            if nii > 1:
                tcounts = np.sum(cull['cnt_sum'][pointings[ic]][ii, chan])
                culledRate[ic, chan] = tcounts / nii
                culledRateErr[ic, chan] = np.sqrt(tcounts) / nii
            else:
                culledRate[ic, chan] = 0
                culledRateErr[ic, chan] = 0
    cull['conv'] = conv
    cull['cfrac']= cfrac
    cull['culledRate'] =culledRate
    cull['culledRateErr'] = culledRateErr
    cull['pointings'] = pointings


# a good reference pointing is 90 - fully converged with over 90% accepted in all chans
pGood = 90
#our problem child
pProb = 87

pointing = pProb

c0 = c90
c1 = c90s

csum = c0['cnt_sum'][pointing]
met = (c0['cullData'][pointing].binTimes[:,0]+c0['cullData'][pointing].binTimes[:,1])*.5
hichan = c0['cullData'][pointing].combine_spin_bins(5, 5)
ut = np.array(spiceTime.met_to_utc(met),dtype='datetime64')
mask0 = np.transpose(c0['cullData'][pointing].currentMask['bin_mask'])
mask1 = np.transpose(c1['cullData'][pointing].currentMask['bin_mask'])
chanlims=[40,30,15,10,5]
elabel = list()
nen = len(energy_ranges[:,0])
ctofs = utils.kev2ctof(energy_ranges)
for ic in range(nen):
    elabel.append(f"{ctofs[ic,0]:.1f}-{ctofs[ic,1]:.1f}")


fig, axs = plt.subplots(nch+1,layout='constrained',sharex=True,figsize=(8,10))
axs[nch].plot(ut,hichan,'b')
axs[nch].set_yscale('log')
axs[nch].set_ylabel('< 8.9')
for ich in range(nch):
    ii = np.nonzero(mask0[:,ich])[0]
    jj = np.nonzero(mask1[:,ich])[0]
    axs[ich].plot(ut,csum[:,ich],"r",lw=.5)
    axs[ich].plot(ut[ii],csum[ii,ich],"g",lw=2)
    axs[ich].plot(ut[jj], csum[jj, ich], "b", lw=2)
    axs[ich].set_ylim(0,chanlims[ich])
    axs[ich].set_ylabel(elabel[ich])
    axs[ich].xaxis.set_major_formatter(mdates.DateFormatter('%d:%H:%m'))
plt.show()


