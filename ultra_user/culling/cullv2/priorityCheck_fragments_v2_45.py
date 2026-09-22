import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.culling.UltraCull0 as Ultracull0
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.CullQFvalidation as CullVal
import spiceypy
import imap_processing.spice.time as spiceTime
import pickle
import cdflib
import pickle
from scipy.stats import binned_statistic
from scipy.stats import norm
from datetime import datetime
import ultra_user.utils.spiceLoader as spiceLoader

from importlib import reload


spiceLoad = spiceLoader.spiceLoader()
spicefiles = spiceLoad.getFileList()
spiceLoad.loadSpice()

#old kernel enumeration
#spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0116.tsc')
#spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
#spiceypy.furnsh('data/imap/spice/spk/imap_pred_od040_20260816_20260927_v01.bsp')
#spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260816_v01.bsp')
#spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
#spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
#spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_353_2025_354_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_354_2025_356_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_356_2025_358_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_358_2025_360_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_131_002.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_085_2026_176_001.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_176_2026_220_002.ah.bc')
#spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_190_2026_253_001.ah.bc')

pickleDir = '/Users/demajr1/files/imap_ultra/culling/cullPickles/'

sensor="45"
energy_ranges = cull_util.l1c_energy_ranges()
cullList = ['c0','c1','c2','c3','c4']
pointings = dict()
culls=dict()
for cull in cullList:
    p = cull_util.get_pointings_from_calPeriods(cull,sensor=sensor)
    culls[cull] = cull_util.runculls(p, energy_ranges, sensor=sensor, cullPackage="hiEnergy_upstream_spectral_stat_v1")
    pvset = set(culls[cull]['vcull'].keys())
    p_missing = set(p).difference(pvset)
    for mp in p_missing:
        p.remove(mp)
    pointings[cull] = p


cntSum = dict()
cntTot = dict()
sepStat=dict()
binThresholds=dict()
for cull in cullList:
    ii = np.nonzero(culls[cull]['vcull'][pointings[cull][0]]['binMask'])[0]
    csum = culls[cull]['cnt_sum'][pointings[cull][0]][ii, :]
    cdata0 = culls[cull]['cullData'][pointings[cull][0]]
    for ip, point in enumerate(pointings[cull]):
        if ip != 0:
            print(f"Cull {cull}, pointing {point} ({ip} of {len(pointings[cull])})")
            ii = np.nonzero(culls[cull]['vcull'][pointings[cull][ip]]['binMask'])[0]
            if len(ii) > 0:
                csum = np.concatenate((csum, culls[cull]['cnt_sum'][pointings[cull][ip]][ii, :]))
    cntSum[cull] = csum
    cntTot[cull] = np.sum(csum, axis=1)
    sepStat[cull] = cull_util.sepStats(culls[cull], repointings=pointings[cull], loud=True, binsize=[1, 1, 1, 1, 1],
                                       maxval=[200, 200, 200, 200, 200])
    binThresholds[cull] = cdata0.sep_threshold_per_spin*cdata0.spin_range



plotCulls=['c0','c1','c2','c3','c4']
maxy=3
fig, axs = plt.subplots(5,sharex=True,figsize=(8,11))
for ic in range(5):
    for cull in plotCulls:
        axs[ic].plot(sepStat[cull]['sepStat'][ic]['bcen'],
                     sepStat[cull]['sepStat'][ic]['avg']/sepStat[cull]['sepStat'][ic]['avg'][0],
                     marker='.',label=cull)
        axs[ic].plot(binThresholds[cull][ic]*np.array([1,1]),[0,maxy],color='red')
    axs[ic].set_ylabel(f"{np.sqrt(energy_ranges[ic, 0] * energy_ranges[ic, 1]):.1f} keV")
    axs[ic].set_xscale('log')
    axs[ic].set_ylim(0,maxy)
plt.suptitle(f"Energy bin histgrams (normalized)")
plt.xlabel("high energy bin center")
plt.legend()
plt.show()

# setting threholds
cdata0 = culls[cull]['cullData'][pointings[cull][0]]
oldThresholds = cdata0.sep_threshold_per_spin*cdata0.spin_range
binThresholds['c0'] = [120,50,8,2,1.5]
binThresholds['c1'] = [90,65,24,5,2]
binThresholds['c2'] = [200,80,35,11,4]
binThresholds['c3'] = [200,120,20,18,3]
binThresholds['c4'] = [100,100,30,25,8]

strictThresholds = binThresholds.copy()
strictThresholds['c0'] = [60,8,3,1,1.5]
strictThresholds['c1'] = [60,25,8,3,1.5]
strictThresholds['c2'] = [200,30,8,4,3.5]
strictThresholds['c3'] = [120,25,9,9,3]
strictThresholds['c4'] = [30,30,12,11.5,5]

# now do separate plots so we can tune more carefully
cull='c4'
maxy=2
miny=0.5
fig, axs = plt.subplots(5,sharex=True,figsize=(8,11))
for ic in range(5):
    axs[ic].errorbar(sepStat[cull]['sepStat'][ic]['bcen'],
                 sepStat[cull]['sepStat'][ic]['avg'] / sepStat[cull]['sepStat'][ic]['avg'][0],
                 yerr=sepStat[cull]['sepStat'][ic]['errMean'] / sepStat[cull]['sepStat'][ic]['avg'][0],
                 marker='.', label=cull)
    axs[ic].plot(binThresholds[cull][ic] * np.array([1, 1]), [0, maxy], color='orange')
    axs[ic].plot(strictThresholds[cull][ic] * np.array([1, 1]), [0, maxy], color='red')
    axs[ic].plot(oldThresholds[ic] * np.array([1, 1]), [0, maxy],'--', color='purple')
    axs[ic].plot([np.min(sepStat[cull]['sepStat'][ic]['bcen']),np.max(sepStat[cull]['sepStat'][ic]['bcen'])],[1,1],'--')
    axs[ic].set_ylabel(f"{np.sqrt(energy_ranges[ic, 0] * energy_ranges[ic, 1]):.1f} keV")
    axs[ic].set_xscale('log')
    axs[ic].set_ylim(miny,maxy)
    axs2=axs[ic].twinx()
    axs2.set_ylabel("Cumulative Fraction")
    axs2.plot(sepStat[cull]['sepStat'][ic]['bcen'],
              np.interp(sepStat[cull]['sepStat'][ic]['bcen'],
                        sepStat[cull]['sepCumbin'],sepStat[cull]['sepCumfrac']), color='green')
    axs2.set_ylim(0,1)
plt.suptitle(f"Energy bin histgrams (normalized) for {cull} ({sensor})")
plt.xlabel("high energy bin center")
fig.tight_layout()
plt.show()

# work out new method - cumulative error
ichan=2
cull='c2'

cumdif = dict()
echans = np.array(range(len(energy_ranges[:,0])))
for cull in cullList:
    weights = sepStat[cull]['sepStat'][0]['count']
    cdif = np.ndarray((len(weights),len(echans)-1))
    for ichan in range(len(echans)-1):
        norm =np.mean(sepStat[cull]['sepStat'][ichan]['avg'][0])
        normAvg = sepStat[cull]['sepStat'][ichan]['avg'] / norm
        cumAvg = np.zeros_like(weights)
        cumAvg[0] = 1
        for ic in range(1, len(weights)):
            cdif[ic,ichan] = np.sum(normAvg[0:ic]*weights[0:ic])/np.sum(weights[0:ic]) -1
    cumdif[cull] = cdif

bincen = sepStat[cull]['sepStat'][0]['bcen']
altThresholds = dict()
for cull in cullList:
    thresholds = np.full(len(energy_ranges[:,0]),200.)
    thresholds[len(echans)-1]=0
    for ichan in range(len(echans)-1):
        tbins = np.where(cumdif[cull][:,ichan] > 0.05)[0]
        if len(tbins) != 0:
            thresholds[ichan] = bincen[tbins[0]]
    altThresholds[cull]=thresholds

cull='c4'
emid = np.sqrt(energy_ranges[:,0]*energy_ranges[:,1])
for ichan in range(len(echans)-1):
    plt.plot(bincen,cumdif[cull][:,ichan],label=f"{emid[ichan]:.1f} keV")
plt.plot([0,200],[0.05,0.05])
plt.xscale('log')
plt.ylim(-.05,.1)
plt.xlabel('high energy counts')
plt.ylabel('cumulative background')
plt.title(f"Cal {cull} U{sensor}")
plt.legend()
plt.show()


