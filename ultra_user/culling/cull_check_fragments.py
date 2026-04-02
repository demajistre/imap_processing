import numpy as np
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
c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

# quick fraction check
nch=6
frac90 = np.ndarray( (len(repointings90), nch))
frac45 = np.ndarray( (len(repointings45), nch))
ic=0
for p in repointings90:
    frac90[ic,:] = c90['cullFrac'][p]
    ic +=1
ic=0
for p in repointings45:
    frac45[ic,:] = c45['cullFrac'][p]
    ic +=1

# pointing 90 is really clean

ut_ranges = [ ['2025-11-23','2025-11-30'],
              ['2025-11-30','2025-12-07'],
              ['2025-12-14','2025-12-21'],
              ['2025-12-21','2025-12-28']]
point_ranges = [[57,63],[64,70],[78,84],[85,92]]

pointing_list = repointings90
cull = c90
ipnt = 2
prange = point_ranges[ipnt]

elabel = list()
nen = len(energy_ranges[:,0])
ctofs = utils.kev2ctof(energy_ranges)
for ic in range(nen):
    elabel.append(f"{ctofs[ic,0]:.1f}-{ctofs[ic,1]:.1f}")
nch=5

csum = cull['cnt_sum'][prange[0]]
met = (cull['cullData'][prange[0]].binTimes[:,0]+cull['cullData'][prange[0]].binTimes[:,1])*.5
hichan = cull['cullData'][prange[0]].combine_spin_bins(5, 5)
mask = np.transpose(cull['cullData'][prange[0]].currentMask['bin_mask'])

for ip in range(prange[0]+1,prange[1]):
    if ip in pointing_list:
        csum = np.concatenate((csum,cull['cnt_sum'][ip]))
        cc = cull['cullData'][ip]
        met = np.concatenate((met, .5*(cc.binTimes[:,0]+cc.binTimes[:,1])))
        hichan = np.concatenate((hichan,cc.combine_spin_bins(4, 3)))
        mask = np.concatenate((mask,np.transpose(cc.currentMask['bin_mask'])))

ut = np.array(spiceTime.met_to_utc(met),dtype='datetime64')

chanlims=[40,30,15,10,5]
fig, axs = plt.subplots(nch+1,layout='constrained',sharex=True,figsize=(8,10))
axs[nch].plot(ut,hichan,'b')
axs[nch].set_yscale('log')
axs[nch].set_ylabel('< 8.9')
for ich in range(nch):
    #maskVal = np.zeros_like(ut,dtype=float)
    #maskVal[np.nonzero(mask[:,ich] == False)[0]] = chanlims[ich]*0.9
    ii = np.nonzero(mask[:,ich])[0]
    axs[ich].plot(ut,csum[:,ich],"r",lw=.5)
    axs[ich].plot(ut[ii],csum[ii,ich],"g",lw=2)
    axs[ich].set_ylim(0,chanlims[ich])
    axs[ich].set_ylabel(elabel[ich])
    #axs[ich].plot(ut,maskVal,'r')
plt.show()

myData = {'ut_range':ut_ranges[ipnt],'pointing_range':point_ranges[ipnt], 'hichan':hichan,
          'ut':ut,'met':met,'csum':csum,'mask':mask,'nch':nch,'chanlims':chanlims,'elabel':elabel,
          'energy_ranges':energy_ranges}
pickname = f"/Users/demajr1/tmp/cull_summary_{ipnt}_v5.pkl"

pickle.dump(myData,open(pickname,'wb'))


## focus on the 'problem area' in pointing 87
pointing = 87
c90a = cull_util.runculls([pointing],energy_ranges,cullPackage="hiEnergy_upstream_stat_v1")
cull = c90a

csum = cull['cnt_sum'][pointing]
met = (cull['cullData'][pointing].binTimes[:,0]+cull['cullData'][pointing].binTimes[:,1])*.5
hichan = cull['cullData'][pointing].combine_spin_bins(5, 5)
mask = np.transpose(cull['cullData'][pointing].currentMask['bin_mask'])
ut = np.array(spiceTime.met_to_utc(met),dtype='datetime64')

fig, axs = plt.subplots(nch+1,layout='constrained',sharex=True,figsize=(8,10))
axs[nch].plot(ut,hichan,'b')
axs[nch].set_yscale('log')
axs[nch].set_ylabel('< 8.9')
for ich in range(nch):
    ii = np.nonzero(mask[:,ich])[0]
    axs[ich].plot(ut,csum[:,ich],"r",lw=.5)
    axs[ich].plot(ut[ii],csum[ii,ich],"g",lw=2)
    axs[ich].set_ylim(0,chanlims[ich])
    axs[ich].set_ylabel(elabel[ich])
    axs[ich].xaxis.set_major_formatter(mdates.DateFormatter('%d:%H:%m'))
plt.show()

up1_mask = cull['upcull1'][87]['mask']
up2_mask = cull['upcull2'][87]['mask']


fig, axs = plt.subplots(4,layout='constrained',sharex=True,figsize=(8,10))
axs[0].plot(ut,up1_mask[0,:])
axs[1].plot(ut,up2_mask[0,:])
axs[2].plot(ut,c90['cullData'][87].currentMask['bin_mask'][0,:])
axs[3].plot(ut,hichan,'b')
axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%d:%H:%m'))
plt.show()

fig, axs = plt.subplots(3,layout='constrained',sharex=True,figsize=(8,10))
axs[0].plot(up1_mask[0,:])
axs[1].plot(up2_mask[0,:])
axs[2].plot(cull['cullData'][87].currentMask['bin_mask'][0,:])
plt.show()



up1 = cull['upcull1'][87]
up2 = cull['upcull2'][87]

plt.plot(up1['totalScaled'])
plt.plot(up2['totalScaled'])
plt.show()

up=up1
mu = np.mean(up['totalScaled'])
sig = np.std(up['totalScaled'])
thresh = mu+(3*np.sqrt(mu))

ii = np.nonzero(up['totalScaled'] < thresh)
jj = np.nonzero(up['totalScaled'] < up['thresh'])
m2 = np.zeros_like(up['mask'][0,:])
m2[ii] = True

plt.plot(up['mask'][0,:])
plt.plot(m2)
plt.show()

tt=np.array(range(len(up['totalScaled'])))
plt.plot(tt,up['totalScaled'])
plt.plot(tt[ii],up['totalScaled'][ii])
plt.plot(tt[jj],up['totalScaled'][jj])
plt.show()

reload(UltraCull0)
c90a = cull_util.runculls([pointing],energy_ranges,cullPackage="hiEnergy_upstream_stat_v1")

