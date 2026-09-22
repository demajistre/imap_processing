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
from importlib import reload

spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0116.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
spiceypy.furnsh('data/imap/spice/spk/imap_pred_od040_20260816_20260927_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260816_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_353_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_354_2025_356_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_356_2025_358_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_358_2025_360_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_131_002.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_085_2026_176_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_176_2026_220_002.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_190_2026_231_002.ah.bc')
pickleDir = '/Users/demajr1/files/imap_ultra/culling/cullPickles/'

sensor="90"
energy_ranges = cull_util.l1c_energy_ranges()
p1 = cull_util.get_pointings_from_calPeriods('c1',sensor=sensor)
p2 = cull_util.get_pointings_from_calPeriods('c2',sensor=sensor)

c1 = cull_util.runculls(p1,energy_ranges,sensor=sensor,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c2 = cull_util.runculls(p2,energy_ranges,sensor=sensor,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c1r = cull_util.runculls(p1,energy_ranges,sensor=sensor,cullPackage="hiEnergy_upstream_spectral_stat_v1",useRawOnly=True)
c2r = cull_util.runculls(p2,energy_ranges,sensor=sensor,cullPackage="hiEnergy_upstream_spectral_stat_v1",useRawOnly=True)

culls = {'c1':c1,'c2':c2,'c1r':c1r,'c2r':c2r}
pointings = {'c1':p1,'c2':p2,'c1r':p1,'c2r':p2}

cntSum = dict()
cntTot = dict()
sepStat=dict()
for ic,cull in enumerate(culls.keys()):
    ii = np.nonzero(culls[cull]['vcull'][pointings[cull][0]]['binMask'])[0]
    csum = culls[cull]['cnt_sum'][pointings[cull][0]][ii,:]
    for ip,point in enumerate(pointings[cull]):
        if ip !=0 :
            print(f"Cull {cull}, pointing {point} ({ip} of {len(pointings[cull])})")
            ii = np.nonzero(culls[cull]['vcull'][pointings[cull][ip]]['binMask'])[0]
            if len(ii) > 0:
                csum = np.concatenate((csum,culls[cull]['cnt_sum'][pointings[cull][ip]][ii,:]))
    cntSum[cull] = csum
    cntTot[cull] = np.sum(csum,axis=1)
    sepStat[cull] = cull_util.sepStats(culls[cull], loud=True, binsize=[1, 1, 1, 1, 1],
                                       maxval=[100, 100, 100, 100, 100])

#sepStat=dict()
#for ic,cull in enumerate(culls.keys()):
#    sepStat[cull]= cull_util.sepStats(culls[cull],loud=True,binsize = [1,1,1,1,1],maxval=[100,100,100,100,100])

plt.plot(cntSum['c2r'][:,5],label='raw')
plt.plot(cntSum['c2'][:,5],label='priority 1')
plt.xlabel('spin bin')
plt.ylabel('counts')
plt.yscale('log')
plt.legend()
plt.show()


plt.plot(cntSum['c2'][:,5]/cntSum['c2r'][:,5])
plt.xlabel('spin bin')
plt.ylabel('ratio of p1 to raw (highest energy)')
plt.show()

fig, axs = plt.subplots(6,sharex=True,figsize=(8,11))
for ic in range(6):
    axs[ic].plot(cntSum['c2'][:,ic]/cntSum['c2r'][:,ic])
    axs[ic].set_ylabel(f"{np.sqrt(energy_ranges[ic,0]*energy_ranges[ic,1]):.1f} keV")
plt.xlabel('spin bin')
plt.suptitle("ratio of priority 1 to raw")
plt.show()

fig, axs = plt.subplots(3,figsize=(8,11))
ic=0
for p in range(238,241):
    print(p)
    axs[ic].plot(culls['c2']['cnt_sum'][p][:,5],label='priority')
    axs[ic].plot(culls['c2r']['cnt_sum'][p][:,5],label='raw')
    axs[ic].set_title(f"pointing {p} high energy channel")
    ic=ic+1
    plt.legend()
plt.show()


plt.plot(sepStat['c1']['sepStat'][1]['bcen'],sepStat['c1']['sepStat'][1]['avg'])
plt.show()

cullLabels={'c1': "cal period 1, priority 1",'c1r': "cal period 1, raw",
        'c2': "cal period 2, priority 1",'c2r': "cal period 2, raw"}

plotCulls=['c1','c2']
fig, axs = plt.subplots(5,sharex=True,figsize=(8,11))
for ic in range(5):
    for cull in plotCulls:
        axs[ic].plot(sepStat[cull]['sepStat'][ic]['bcen'],
                     sepStat[cull]['sepStat'][ic]['avg']/sepStat[cull]['sepStat'][ic]['avg'][0],
                     marker='.',label=cullLabels[cull])
    axs[ic].set_ylabel(f"{np.sqrt(energy_ranges[ic, 0] * energy_ranges[ic, 1]):.1f} keV")
    axs[ic].set_xscale('log')
    #axs[ic].set_yscale('log')
    axs[ic].set_ylim(0,5)
plt.suptitle(f"Energy bin histgrams (normalized)")
plt.xlabel("high energy bin center")
plt.legend()
plt.show()




# single plot
cull = 'c1r'
fig, axs = plt.subplots(5,sharex=True,figsize=(8,11))
for ic in range(5):
    axs[ic].plot(sepStat[cull]['sepStat'][ic]['bcen'],sepStat[cull]['sepStat'][ic]['avg'],marker='.')
    axs[ic].set_ylabel(f"{np.sqrt(energy_ranges[ic, 0] * energy_ranges[ic, 1]):.1f} keV")
    axs[ic].set_xscale('log')
    axs[ic].set_yscale('log')
plt.suptitle(f"{cullLabels[cull]}")
plt.xlabel("high energy bin center")
plt.show()
