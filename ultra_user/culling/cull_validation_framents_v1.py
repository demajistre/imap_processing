import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.CullQFvalidation as CullVal
import spiceypy
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

energy_ranges = cull_util.l1c_energy_ranges()
#repointings90 = cull_util.get_pointings(96,183)
#repointings45 = cull_util.get_pointings(96,183,sensor='45')
repointings90 = cull_util.get_pointings(130,222) #3mo pointings
repointings45 = cull_util.get_pointings(130,222,sensor='45')

c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

xspin = MyUltraFile.L1Bxspin(repointings45[0],sensor='45').data
xspinvars = xspin.cdf_info().zVariables

# look at fractions
cfrac45 = np.ndarray((len(energy_ranges),len(repointings45)))
ic=0
for ip in repointings45:
    cfrac45[:,ic] = c45['cullFrac'][ip]
    ic +=1

cfrac90 = np.ndarray((len(energy_ranges),len(repointings90)))
ic=0
for ip in repointings90:
    cfrac90[:,ic] = c90['cullFrac'][ip]
    ic +=1

fig,ax = plt.subplots(len(energy_ranges)-1,sharex=True)
fig.set_size_inches(8,12)
for ie in range(len(energy_ranges)-1):
    ax[ie].plot(repointings45,cfrac45[ie,:],'.',label='45')
    ax[ie].plot(repointings90,cfrac90[ie,:],'.',label='90')
    ax[ie].set_title(f"Energy Range: {energy_ranges[ie,0]:.1f} - {energy_ranges[ie,1]:.1f}")
    ax[ie].set_ylabel("good time fraction")
    if ie == 4 :
        ax[ie].legend(loc='upper left')
        ax[ie].set_xlabel('pointing number')
plt.show()


#look at extended spin table - compare with masks (uses non-3mo pointings)
#Georges email pointings [96,127,159]
#gps = [96,127,159]
#example pointing
#ip = gps[1]

ip = 215

sdcXs = MyUltraFile.L1Bxspin(ip,sensor='45').data
cv = CullVal.CullQFvalidation(c45,ip)
sdcCulls = cv.SDCqf2cullmask.keys()

validationCulls = dict()
for c in sdcCulls:
    validationCulls[c] = cv.chanQFlags(c)

fig, axs = plt.subplots(np.size(list(sdcCulls)),layout='constrained',sharex=True,figsize=(8,10))
for ic,testKey in enumerate(sdcCulls):
    axs[ic].set_title(f"repoint {ip}:{testKey}")
    axs[ic].plot(sdcXs['spin_number'],sdcXs[testKey],'og',label='SDC')
    axs[ic].plot(cv.qfSpinBins,validationCulls[testKey],'+r',label='APL')
axs[0].legend()
plt.show()

#luisa's custom xs
vroot='data/imap/xsValidation0526/'

sdcVal = {127:cdflib.CDF(vroot+'imap_ultra_l1b_45sensor-extendedspin_20260115-repoint00127_v100.cdf'),
          166:cdflib.CDF(vroot+'imap_ultra_l1b_45sensor-extendedspin_20260223-repoint00166_v100.cdf')}

ip = 127


sdcXs = sdcVal[ip]
cv = CullVal.CullQFvalidation(c45,ip)
sdcCulls = cv.SDCqf2cullmask.keys()
validationCulls = dict()
for c in sdcCulls:
    validationCulls[c] = cv.chanQFlags(c)


fig, axs = plt.subplots(np.size(list(sdcCulls)),layout='constrained',sharex=True,figsize=(8,10))
for ic,testKey in enumerate(sdcCulls):
    axs[ic].set_title(f"repoint {ip}:{testKey}")
    axs[ic].plot(sdcXs['spin_number'],sdcXs[testKey],'og',label='SDC')
    axs[ic].plot(cv.qfSpinBins,validationCulls[testKey],'+r',label='APL')
axs[0].legend()
plt.show()
