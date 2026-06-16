import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.CullQFvalidation as CullVal
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

energy_ranges = cull_util.l1c_energy_ranges()
#repointings90 = cull_util.get_pointings(96,183)
#repointings45 = cull_util.get_pointings(96,183,sensor='45')
repointings90 = cull_util.get_pointings(130,222) #3mo pointings
repointings45 = cull_util.get_pointings(130,222,sensor='45')

c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

#127 not in period
pointings=[215,217,219,221,207,205,195,191,171,167,165]
sensors = ['90','45']
culls = [c90,c45]
for ic in range(2):
    cull = culls[ic]
    sensor = sensors[ic]

    npoint = len(pointings)
    nchan = len(cull['scull'][pointings[0]]['converge'])

    conv = np.ndarray((npoint, nchan), dtype=bool)
    cfrac = np.ndarray((npoint, nchan), dtype=float)
    culledRate = np.ndarray((npoint, nchan), dtype=float)
    culledRateErr = np.ndarray((npoint, nchan), dtype=float)

    for ic in range(npoint):
        print(pointings[ic])
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
        myData = {'sensor': sensor, 'pointings': pointings, 'culledRate': culledRate, 'culledRateErr': culledRateErr,
                  'energy_ranges': energy_ranges, 'cullFraction': cfrac, 'convergence': conv}
        pickname = f"/Users/demajr1/tmp/pointing_rates_{sensor}_v0.pkl"
        pickle.dump(myData, open(pickname, 'wb'))


