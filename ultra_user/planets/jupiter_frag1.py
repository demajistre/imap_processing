import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import ultra_user.planets.ENA_planets as planets
import ultra_user.culling.cull_util as cull_util
import imap_processing.spice.time as spiceTime
from importlib import reload

# change this to something better
spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')

spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260203_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_355_2025_357_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_066_001.ah.bc')

repoint = 90
de = MyUltraFile.L1Bde(repoint).data
xspin = MyUltraFile.L1Bxspin(repoint).data
xspinvars = xspin.cdf_info().zVariables
devars = de.cdf_info().zVariables
l1c = MyUltraFile.L1C(repoint).data
l1cvars = l1c.cdf_info().zVariables

# the cull
energy_ranges = cull_util.l1c_energy_ranges()
cull = cull_util.runculls([repoint], energy_ranges,cullPackage="hiEnergy_upstream_stat_v1")
csum =cull['cnt_sum'][repoint]
# for now use the whole pointing


# getting oriented
t0 = spiceypy.scs2e(-43,f"{np.mean(xspin['spin_start_time'])}")
tstart = spiceypy.scs2e(-43,f"{xspin['spin_start_time'][0]}")
tend = spiceypy.scs2e(-43,f"{xspin['spin_start_time'][-1]}")

jpos, lt = spiceypy.spkpos("JUPITER_BARYCENTER",np.array([tstart,tend]), 'IMAP_DPS','NONE','-43')
spos, lts = spiceypy.spkpos("SUN",t0, 'IMAP_DPS','NONE','-43')
epos, lte = spiceypy.spkpos("EARTH",t0, 'IMAP_DPS','NONE','-43')
rjp = np.sqrt(np.sum(jpos**2,axis=1))
jposu= np.zeros_like(jpos)
for ic in range(2):
    jposu[ic,:] = jpos[ic,:]/rjp[ic]
dailyang = np.degrees(np.arccos(np.sum(jposu[0,:]*jposu[1,:]))) # a little more than a degree/day



# position over time

repointings = {'90':cull_util.get_pointings(27,163),
                     '45':cull_util.get_pointings(27,163,sensor='45')}

sensor='90'

npoint = len(repointings[sensor])
et = np.zeros(npoint)
for ic in range(npoint):
    et[ic] = spiceypy.scs2e(-43, f"{np.mean(xspin['spin_start_time'])}")

