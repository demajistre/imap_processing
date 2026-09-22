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
calPointings="c1"

energy_ranges = cull_util.l1c_energy_ranges()
repointings = cull_util.get_pointings_from_calPeriods(calPointings,sensor=sensor)

cull = cull_util.runculls(repointings,energy_ranges,sensor=sensor,cullPackage="hiEnergy_upstream_spectral_stat_v1")

cull_util.cullplot(cull,repointings)

cullCounts = cull_util.sepStats(cull,loud=True)
pickle.dump(cullCounts,
            open(pickleDir+
                 f"cullCounts_{sensor}_{calPointings}_{datetime.now().strftime('%Y-%m-%dT%H.%M.%S')}.pkl",'wb'))

cntShape = np.shape(cullCounts['cntsum'])

fig,ax = plt.subplots(5,figsize=(8,11))
for ic in range(0,cntShape[1]-1):
    ax[ic].plot(cullCounts['sepStat'][ic]['bcen'],cullCounts['sepStat'][ic]['avg'])
plt.show()

plt.plot(cullCounts['spinbins'],cullCounts['cntsum'][:,5],'.')
plt.yscale('log')
plt.show()


cullCountsU = cull_util.sepStats(cull,loud=True,binsize = [1,1,1,1,1],maxval=[100,100,100,100,100])


statsCCU = cullCountsU['sepStat']
nc = np.size(list(statsCCU.keys()))
nb = np.size(statsCCU[0]['avg'])
binspec = np.ndarray((nc, nb))
uncspec= np.ndarray((nc, nb))
for ic in range(nc):
    binspec[ic,:] = statsCCU[ic]['avg']
    uncspec[ic,:] = statsCCU[ic]['errMean']

for ic in range(nc):
    plt.plot(statsCCU[0]['bcen'],binspec[ic,:]/binspec[0,:],label=f"bin{ic}")
plt.legend()
plt.show()



### checking priority stuff
ucull ={'uc0': Ultracull0.UltraCull0(138,energy_ranges),
        'uc0r' : Ultracull0.UltraCull0(138,energy_ranges,useRawOnly=True),
        'uc1' : Ultracull0.UltraCull0(230,energy_ranges),
        'uc1r' : Ultracull0.UltraCull0(230,energy_ranges,useRawOnly=True)}

usum = dict()
for uc in ucull.keys():
    usum[uc] = ucull[uc].get_count_summary()

plt.plot(usum['uc0'][:,5],label='new')
plt.plot(usum['uc0r'][:,5],label='all raw')
plt.legend()
plt.show()