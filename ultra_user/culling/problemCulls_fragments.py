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

# sep_threshold_per_spin = np.array([4., 2., 1.25, 0.9, 0.2,.2]) # original
sep_threshold_per_spin = np.array([4., 2., 1.25, 0.45, 0.1,.1])
c90t = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_stat_v1",sep_threshold_per_spin=sep_threshold_per_spin)
c45t = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_stat_v1",sep_threshold_per_spin=sep_threshold_per_spin)

culls = [c45,c90t,c45t,c45t]
rp = [repointings90,repointings90,repointings45,repointings45]

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


cull=c90

nc=4
fig, axs = plt.subplots(nc,layout='constrained',sharex=True,figsize=(8,10))
for ic in range(nc):
    axs[ic].errorbar(cull['pointings'], cull['culledRate'][:, ic], cull['culledRateErr'][:, ic], fmt='r.')
    ii = np.nonzero(cull['conv'][:,ic])[0]
    axs[ic].errorbar(cull['pointings'][ii], cull['culledRate'][ii, ic], cull['culledRateErr'][ii, ic],fmt='g.')
    axs[ic].set_title(f"{energy_ranges[ic,0]} - {energy_ranges[ic,1]} kev")
    axs[ic].set_ylabel(f"Count Rate")
plt.xlabel("Pointing")
plt.show()


fig, axs = plt.subplots(nc,layout='constrained',sharex=True,figsize=(8,10))
for ic in range(nc):
    axs[ic].errorbar(c90['pointings'], c90['culledRate'][:, ic], c90['culledRateErr'][:, ic], fmt='g.')
    axs[ic].errorbar(c90t['pointings'], c90t['culledRate'][:, ic], c90t['culledRateErr'][:, ic], fmt='r.')
    axs[ic].set_title(f"{energy_ranges[ic,0]} - {energy_ranges[ic,1]} kev")
    axs[ic].set_ylabel(f"Count Rate")
    axs[ic].set_xlim(120,185)
plt.xlabel("Pointing")
plt.show()


fig, axs = plt.subplots(nc,layout='constrained',sharex=True,figsize=(8,10))
for ic in range(nc):
    axs[ic].errorbar(c45['pointings'], c45['culledRate'][:, ic], c45['culledRateErr'][:, ic], fmt='g.')
    axs[ic].errorbar(c45t['pointings'], c45t['culledRate'][:, ic], c45t['culledRateErr'][:, ic], fmt='r.')
    axs[ic].set_title(f"{energy_ranges[ic,0]} - {energy_ranges[ic,1]} kev")
    axs[ic].set_ylabel(f"Count Rate")
    axs[ic].set_xlim(120,185)
plt.xlabel("Pointing")
plt.show()


###################

#sensor='90'
#cull = c90
pointings = np.array(repointings90, dtype=int)
sensor='45'
cull = c45
#pointings = np.array(repointings45, dtype=int)
#sensor='90'
#cull = c90t
#pointings = np.array(repointings90, dtype=int)
#sensor='45'
#cull = c45t
#pointings = np.array(repointings45, dtype=int)


# quick fraction check
nch = 6
frac= np.ndarray((len(pointings), nch))
ic = 0
for p in repointings90:
    frac[ic, :] = cull['cullFrac'][p]
    ic += 1

# a good reference pointing is 90 - fully converged with over 90% accepted in all chans
pGood = 90
#our problem child
pProb = 87

plt.plot(cull['cnt_sum'][pGood][:,0:4])
plt.yscale('log')
plt.show()

plt.plot(cull['cnt_sum'][pProb][:,0:4])
plt.yscale('log')
plt.show()

# spectral cull?
csum = c90['cnt_sum'][pProb]
ic = 2
dif = csum[:,ic+1]-csum[:,ic]
sdif = np.sqrt(csum[:,ic-1]+csum[:,ic])
plt.plot(dif-sdif)
plt.show()
print(len(np.nonzero(dif-sdif >0)[0]))


# accumulate some data

npoint = len(pointings)
nchan = len(cull['scull'][pGood]['converge'])

conv = np.ndarray((npoint,nchan),dtype=bool)
cfrac = np.ndarray((npoint,nchan),dtype=float)
culledRate = np.ndarray((npoint,nchan),dtype=float)
culledRateErr = np.ndarray((npoint,nchan),dtype=float)

for ic in range(npoint):
    print(pointings[ic])
    conv[ic,:] = cull['scull'][pointings[ic]]['converge']
    cfrac[ic,:] = cull['cullData'][pointings[ic]].currentCullFraction()
    cmask =cull['cullData'][pointings[ic]].currentMask['bin_mask']
    for chan in range(nchan):
        ii = np.nonzero(cmask[chan,:])[0]
        nii = len(ii)
        if nii > 1:
            tcounts = np.sum(cull['cnt_sum'][pointings[ic]][ii,chan])
            culledRate[ic, chan] = tcounts/nii
            culledRateErr[ic, chan] = np.sqrt(tcounts)/nii
        else:
            culledRate[ic, chan] = 0
            culledRateErr[ic, chan] = 0

myData={'sensor':sensor, 'pointings':pointings, 'culledRate':culledRate, 'culledRateErr':culledRateErr,
        'energy_ranges':energy_ranges, 'cullFraction':cfrac, 'convergence':conv}
pickname = f"/Users/demajr1/tmp/pointing_rates_{sensor}_v0.pkl"
pickle.dump(myData,open(pickname,'wb'))
