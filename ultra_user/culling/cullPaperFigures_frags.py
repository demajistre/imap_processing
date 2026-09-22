import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.CullQFvalidation as CullVal
from ultra_user.utils.ultra_utils import PointingDef
import imap_processing.spice.time as spiceTime
from scipy.stats import binned_statistic
from scipy.stats import norm
from scipy.stats import poisson
import spiceypy
import pickle
import cdflib
from importlib import reload

spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0116.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
spiceypy.furnsh('data/imap/spice/spk/imap_pred_od031_20260511_20260622_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260511_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_353_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_354_2025_356_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_356_2025_358_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_358_2025_360_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_131_002.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_085_2026_133_002.ah.bc')

energy_ranges = cull_util.l1c_energy_ranges()
#repointings90 = cull_util.get_pointings(96,183)
#repointings45 = cull_util.get_pointings(96,183,sensor='45')
#repointings90 = cull_util.get_pointings(130,222) #3mo pointings
#repointings45 = cull_util.get_pointings(130,222,sensor='45')
repointings90 = cull_util.get_pointings(96,222) #3mo pointings
repointings45 = cull_util.get_pointings(96,222,sensor='45')

c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

figRoot = '/Users/demajr1/files/imap_ultra/culling/cullpaper/'
pointings = PointingDef()
# example data set
repoint = 215
cull_util.cullplot(c90,repointings=[repoint],rawOnly=True,saveFile=figRoot+'sample_raw_pointing.png')

# voltage cull
vrepoint = 143 #133 #143
vpointTime = pointings.get_pointing_timerange(vrepoint)
vSum = c90['cullData'][vrepoint].get_dvolt_summary()
vTime = c90['cullData'][vrepoint].center_spin()['center_time']
vcnts = c90['cullData'][vrepoint].get_count_summary()
tHrs = (vTime-vpointTime[0])/3600

fig,ax1 = plt.subplots()
fig.suptitle(f"Ultra 90, repoint {vrepoint}")
ax2 = ax1.twinx()
ax1.plot(tHrs,vcnts[:,2],'g')
ax1.set_ylabel('Counts in energy bin 2',color='g')
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_xlabel(f"Hours from start of repoint {vrepoint}")
ax2.plot(tHrs,vSum[1],'b')
ax2.set_ylabel('Minimum Voltage (V)',color='b')
ax2.tick_params(axis='y', labelcolor='b')
fig.tight_layout()
plt.savefig(figRoot+'deflection_voltage_example.png')
plt.show()

#SEP cull
# full time series
cntsum=0
spinbins=0
bintimes=0
lvMask=0
for ic,rp in enumerate(repointings90):
    print(ic)
    ii = np.where(c90['cullData'][rp].voltage_cull()['binMask'])[0]
    if ic==0:
        cntsum = c90['cullData'][rp].get_count_summary()[ii,:]
        spinbins = np.array(c90['cullData'][rp].spinbins)[ii,:]
        bintimes = c90['cullData'][rp].binTimes[ii,:]
        lvMask = c90['cullData'][rp].voltage_cull()['binMask']
    else:
        cntsum = np.concatenate((cntsum,c90['cullData'][rp].get_count_summary()[ii,:]),axis=0)
        spinbins = np.concatenate((spinbins,np.array(c90['cullData'][rp].spinbins)[ii,:]),axis=0)
        bintimes = np.concatenate((bintimes,c90['cullData'][rp].binTimes[ii,:]),axis=0)
        lvMask = np.concatenate((lvMask,c90['cullData'][rp].voltage_cull()['binMask']))



jj = np.where(np.logical_and(spinbins[:,0] > .857e6, spinbins[:,1] < .859e6))[0]
fig,ax = plt.subplots(6,figsize=(8,11),sharex='col')
for ic in range(6):
    ax[ic].plot(np.mean(spinbins[jj,:],axis=1),cntsum[jj,ic])
    ax[ic].set_ylabel(f"Counts Bin {ic}")
ax[5].set_xlabel("IMAP spin number")
fig.suptitle(f"Ultra 90 binned data during SEP event on {spiceTime.met_to_utc(bintimes[jj[50],0]).split('T')[0]}")
plt.savefig(figRoot+'weak_sep.png')
plt.show()

# sep thresholds
sep_threshold_per_spin = np.array([4., 2., 1.20, 0.45, 0.1, .1])
sep_thresh = sep_threshold_per_spin*c90['cullData'][130].spin_range
cumfrac = np.ndarray(1000)
cumbin = range(1,1001)
btot=np.size(spinbins)/2
for ic in range(1,1001):
    cumfrac[ic-1] =np.size(np.nonzero(cntsum[:, 5] < ic)[0])/btot


binsize = [20,5,5,2,1]
maxval = [700,150,80,60,25]
maxy=[30,15,6,6,2]
miny=[15,5,3,1,0.5]
revised_thresh=[450,75,40,34,8]

fig,ax = plt.subplots(5,figsize=(8,11))
for ic in range(5):
    corr, bedge, bn0 = binned_statistic(cntsum[:, 5], cntsum[:, ic], bins=range(0, maxval[ic], binsize[ic]))
    std, bedge, bn0 = binned_statistic(cntsum[:, 5], cntsum[:, ic], statistic='std', bins=range(0,maxval[ic], binsize[ic]))
    ax[ic].plot(bedge[0:-1], corr)
    ax[ic].plot([sep_thresh[ic], sep_thresh[ic]], [0, maxy[ic]])
    ax[ic].plot([revised_thresh[ic], revised_thresh[ic]], [miny[ic], maxy[ic]],'--')
    ax[ic].set_ylim(miny[ic], maxy[ic])
    ax[ic].set_ylabel(f"Counts Bin {ic}")
    ax2 = ax[ic].twinx()
    ax2.plot(cumbin,cumfrac,color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_xlim((0, maxval[ic]))
    ax2.set_ylabel("fraction of spins")
ax[4].set_xlabel("Counts in highest energy bin")
fig.tight_layout()
plt.savefig(figRoot+'sep_thresh.png')
plt.show()

#upstream cull

#find where upstream stuff is tripped
neb = np.shape(c90['upcull1'][repoint]['mask'])[0]

uc1evs = np.ndarray(np.size(repointings90))
uc2evs = np.ndarray(np.size(repointings90))
for ic in range(np.size(repointings90)):
    rp = repointings90[ic]
    nGoodV =np.sum(c90['vcull'][rp]['binMask']) #number of points passing voltage cull
    uc1evs[ic] = nGoodV - np.sum(c90['upcull1'][rp]['mask'][1])
    uc2evs[ic] = nGoodV - np.sum(c90['upcull2'][rp]['mask'][1])

plt.plot(uc1evs,'r.')
plt.plot(uc2evs,'b.')
plt.show()

rp = 163
ii = np.nonzero(c90['vcull'][rp]['binMask'])[0]
spin=(np.mean(c90['cullData'][rp].spinbins,axis=1))[ii]
cnts=c90['cnt_sum'][rp][ii,:]
um1=c90['upcull1'][rp]['mask'][1,ii]
um2=c90['upcull2'][rp]['mask'][1,ii]
jj1=np.nonzero(~um1)[0]
jj2=np.nonzero(~um2)[0]

fig, ax = plt.subplots(5, sharex=True)
for i in range(5):
    ax[i].plot(spin,cnts[:,i])
    ax[i].set_ylabel(f"counts ({i})")
    ax[i].plot(spin[jj1],cnts[jj1,i],'.',color='r')
    ax[i].plot(spin[jj2],cnts[jj2, i], '.', color='g')
ax[4].set_xlabel(f"Spin number")
fig.suptitle(f"Ultra 90, repointing {rp}")
plt.savefig(figRoot+'upstream_data.png')
plt.show()

for ic in np.arange(1,4,.5):
    print(f"{ic} sigma: {norm.cdf(ic)-norm.cdf(-ic)}")
for ic in np.arange(1,4,.5):
    print(f"{ic} sigma: {norm.cdf(ic)}")

## not earth images in earth_frag1.py