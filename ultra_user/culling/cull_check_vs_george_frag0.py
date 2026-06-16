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

# George's totals
cntEst = {191:7833, 195:7722, 205: 5205, 207:7563, 215:6547, 217: 6252, 219:6726, 221:6661}


val = dict()
for ip in cntEst:
    print(ip)
    cntSum = c45['cullData'][ip].get_count_summary()
    mask = c45['cullData'][ip].currentMask['bin_mask']
    spins_per_bin = c45['cullData'][ip].spin_range
    counts = np.ndarray(5)
    bins = np.ndarray(5)
    spins = np.ndarray(5)
    total_counts = 0
    for ic in range(0, 5):
        ii = np.nonzero(mask[ic, :])[0]
        bins[ic] = np.size(ii)
        spins[ic] = bins[ic] * spins_per_bin
        counts[ic] = np.sum(cntSum[ii, ic])
        total_counts += counts[ic]
    val[ip] = {'cntEst': cntEst[ip],'totalCounts':total_counts,'cntRatio':total_counts/cntEst[ip],
               'counts':counts, 'bins': bins, 'spins':spins}

for ip in cntEst:
    print(f"repoint {ip}: hand {val[ip]['cntEst']}, cull {val[ip]['totalCounts']}, diff {val[ip]['cntRatio'] -1}")
