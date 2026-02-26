import numpy as np
from scipy.stats import binned_statistic
from scipy.stats import chisquare
from scipy.stats import stats
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import ultra_user.culling.cull_util as cull_util
import spiceypy
import time
import ultra_user.planets.ENA_planets as ENA_planets
import imap_processing.spice.time as spiceTime
import pytz
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


# repointings to focus (Nov 20 - Dec 6)
repointings90 = cull_util.get_pointings(27,163)
repointings45 = cull_util.get_pointings(27,163,sensor='45')

energy_ranges = cull_util.l1c_energy_ranges()
c90 = cull_util.runculls(repointings90,energy_ranges)
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45')

non_converged90 = list()
for repoint in repointings90:
    if len(np.nonzero(c90['scull'][repoint]['converge'][0:4])[0]) < 4:
        non_converged90.append(repoint)

non_converged45 = list()
for repoint in repointings45:
    if len(np.nonzero(c45['scull'][repoint]['converge'][0:4])[0]) < 4:
        non_converged45.append(repoint)

#non-converged report
cull=c45
nconv = non_converged45
nch=4
for repoint in nconv:
    mask = cull['cullData'][repoint].currentMask['bin_mask']
    conv = cull['scull'][repoint]['converge']
    csum = cull['cnt_sum'][repoint]
    line = f"{repoint}: "
    for ic in range(nch):
        if not conv[ic]:
            ii = np.nonzero(mask[ic, :])[0]
            cnts = csum[ii, ic]
            mean = np.mean(cnts)
            std = np.std(cnts)
            diff = std / np.sqrt(mean) - 1
            line += f"[{ic}: {conv[ic]} {len(cnts)}, {diff:.3f}, {len(cnts)}] "
    print(line)




cull_util.cullplot(c90)
cull_util.cullplot(c45)




ncroot = '/Users/demajr1/files/imap_ultra/culling/nonconverged/v0/'
cull=c90
for repoint in non_converged90:
    conv_summary =f"{repoint}: {cull['scull'][repoint]['converge'][0:4]} : {cull['scull'][repoint]['iterations'][0:4]}"
    print(conv_summary)
    nen=len(energy_ranges[:,0])
    cullData = cull['cullData'][repoint]
    csum = cullData.get_count_summary()
    spinStart = cullData.spinbins[:,0]

    premask = cull['ecull'][repoint]['mask']
    for ie in range(nen):
        premask[ie,:] = np.logical_and(premask[ie,:],cull['vcull'][repoint]['binMask'])
        scullMask = cull['scull'][repoint]['mask']
    fullMask = np.logical_and(premask,scullMask)

    ylims = [35,20,15,10]

    nch=4
    fig, axs = plt.subplots(nch)
    fig.suptitle(conv_summary)
    for ic in range(nch):
        ii = np.nonzero(premask[ic,:])[0]
        jj = np.nonzero(fullMask[ic,:])[0]
        axs[ic].plot(spinStart,csum[:,ic],'r')
        axs[ic].plot(spinStart[ii], csum[ii, ic], 'b')
        axs[ic].plot(spinStart[jj], csum[jj, ic], 'g')
        axs[ic].set_ylim(0,ylims[ic])
    plt.savefig(f"{ncroot}nonconv_{repoint}.png")
    print(repoint)
    plt.show()
    time.sleep(5)

#look at chi2
repoint=52

for ich in range(nch):
    ii = np.nonzero(c90['scull'][repoint]['mask'][ich,:])[0]
    #print(chisquare(c90['cnt_sum'][repoint][ii,ich]))
    print(stats.goodness_of_fit(stats.poisson,c90['cnt_sum'][repoint][ii,ich] ))


#plot like above with utc on axis for 1 pointing
repoint=52
cull=c90

nen=len(energy_ranges[:,0])
cullData = cull['cullData'][repoint]
csum = cullData.get_count_summary()
premask = cull['ecull'][repoint]['mask']
for ie in range(nen):
    premask[ie,:] = np.logical_and(premask[ie,:],cull['vcull'][repoint]['binMask'])
    scullMask = cull['scull'][repoint]['mask']
fullMask = np.logical_and(premask,scullMask)
ylims = [35, 20, 15, 10]
spinStart = cullData.spinbins[:,0]
et = np.mean(cullData.binTimes,axis=1)
time_arr = list()
for ic in range(len(et)):
    time_arr.append(np.datetime64(spiceTime.met_to_utc(et[ic])))
time_arr = np.array(time_arr)

nch = 4
fig, axs = plt.subplots(nch)
fig.suptitle(f"repoint {repoint}")
for ic in range(nch):
    ii = np.nonzero(premask[ic, :])[0]
    jj = np.nonzero(fullMask[ic, :])[0]
    axs[ic].plot(time_arr, csum[:, ic], 'r')
    axs[ic].plot(time_arr[ii], csum[ii, ic], 'b')
    axs[ic].plot(time_arr[jj], csum[jj, ic], 'g')
    axs[ic].set_ylim(0, ylims[ic])
plt.show()

