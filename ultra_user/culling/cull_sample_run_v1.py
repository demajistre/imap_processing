import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import spiceypy

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
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2026_085_2026_113_002.ah.bc')

energy_ranges = cull_util.l1c_energy_ranges()
repointings90 = cull_util.get_pointings(96,183)
repointings45 = cull_util.get_pointings(96,183,sensor='45')

c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

cull_util.cullplot(c90)
cull_util.cullplot(c45)

#look at extended spin table - compare with masks
#Georges email pointings [96,127,159]
gps = [96,127,159]
xs = dict()
st = dict()
for gp in gps:
    xs[gp] = MyUltraFile.L1Bxspin(gp,sensor='45').data
    st[gp] = MyUltraFile.L1Bstatus(gp,sensor='45').data
xspinvars = xs[gps[0]].cdf_info().zVariables
statvars = st[gps[0]].cdf_info().zVariables


#voltage check
ip = gps[1]
spinGrid=np.mean(c45['cullData'][ip].spinbins,axis=1)
statSpin =np.interp(st[ip]['shcoarse'],xs[ip]['spin_start_time']+xs[ip]['spin_period']*0.5,xs[ip]['spin_number'])
plt.plot(xs[ip]['spin_number'],xs[ip]['quality_low_voltage'],label='low voltage flag')
plt.plot(spinGrid,c45['vcull'][ip]['binMask']*10,label='low voltage mask x10')
plt.plot(statSpin,np.minimum(st[ip]['rightdeflection_v'], st[ip]['leftdeflection_v'])*1.e-2,label='def V/100')
plt.legend(loc='upper center')
plt.show()

# don hates day 159 in particular
ip = gps[0]
spinbins=np.mean(c45['cullData'][ip].spinbins,axis=1)
sepmask = np.zeros((len(spinbins),6))
statmask = np.zeros((len(spinbins),6))
for ie in range(4,-1,-1):
    ii = np.nonzero(c45['ecull'][ip]['mask'][ie, :] == False)[0]
    jj = np.nonzero(c45['scull'][ip]['mask'][ie, :] == False)[0]
    if len(ii) > 0:
        sepmask[ii,ie] = ie+20
    if len(jj) > 0:
        statmask[jj,ie] = ie+1
upcull = np.zeros((len(spinbins),3))
upcull[np.nonzero(c45['upcull1'][ip]['mask'][0, :] == False)[0],0] = 10
upcull[np.nonzero(c45['upcull2'][ip]['mask'][0, :] == False)[0],0] = 11
upcull[np.nonzero(c45['speccull'][ip]['mask'][0, :] == False)[0],0] = 12

fig,axs = plt.subplots(8,1)
fig.set_size_inches(8, 15)
fig.suptitle(f"Repoint {ip}")
for ic in range(6):
    axs[ic].set_ylabel(f"{c45['cullData'][ip].energy_ranges[ic,0]:.1f} - {c45['cullData'][ip].energy_ranges[ic,1]:.1f} kev")
    axs[ic].plot(spinbins,c45['cnt_sum'][ip][:,ic])
axs[6].set_ylabel("SDC quality flags")
axs[6].plot(xs[ip]['spin_number'],xs[ip]['quality_high_energy'],label='SEP')
axs[6].plot(xs[ip]['spin_number'],xs[ip]['quality_upstream_ion_1'],label='up 1')
axs[6].plot(xs[ip]['spin_number'],xs[ip]['quality_upstream_ion_2'],label='up 2')
axs[6].plot(xs[ip]['spin_number'],xs[ip]['quality_spectral'],label='spectral')
axs[6].plot(xs[ip]['spin_number'],xs[ip]['quality_statistics'],label='stat')
axs[6].legend(loc='upper right')
axs[7].set_ylabel("Cull masks")
axs[7].plot(spinbins,sepmask,"r")
axs[7].plot(spinbins,statmask,"b")
axs[7].plot(spinbins,upcull,"g")
plt.show()


# focus on SEP and upstream
ip=gps[1]
spinbins=np.mean(c45['cullData'][ip].spinbins,axis=1)
fig,axs = plt.subplots(6,1)
for ic in range(6):
    axs[ic].plot(spinbins,c45['cnt_sum'][ip][:,ic])
plt.show()



#Georges email pointings [96,127,159]
focusPointings = [159]
c90f = cull_util.runculls(focusPointings,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45f = cull_util.runculls(focusPointings,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

cull_util.cullplot(c45f)


#energy bin checking
erange=cull_util.l1c_energy_ranges()
ctrange = utils.kev2ctof(erange)
