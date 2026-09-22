import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import ultra_user.planets.ENA_planets as planets
import ultra_user.culling.cull_util as cull_util
import imap_processing.spice.time as spiceTime
import ultra_user.utils.spiceHelpers as spiceHelp
from importlib import reload

# change this to something better
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
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')

repoint = 215
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
tslop = 300 # leave time for repointings
tstart = spiceypy.scs2e(-43,f"{xspin['spin_start_time'][0]}")+tslop
tend = spiceypy.scs2e(-43,f"{xspin['spin_start_time'][-1]}")-tslop

jpos, lt = spiceypy.spkpos("JUPITER_BARYCENTER",np.array([tstart,tend]), 'IMAP_DPS','NONE','-43')
spos, lts = spiceypy.spkpos("SUN",t0, 'IMAP_DPS','NONE','-43')
epos, lte = spiceypy.spkpos("EARTH",t0, 'IMAP_DPS','NONE','-43')
rjp = np.sqrt(np.sum(jpos**2,axis=1))
jposu= np.zeros_like(jpos)
for ic in range(2):
    jposu[ic,:] = jpos[ic,:]/rjp[ic]
dailyang = np.degrees(np.arccos(np.sum(jposu[0,:]*jposu[1,:])))

# position over time

repointings = {'90':cull_util.get_pointings(27,222),
                     '45':cull_util.get_pointings(27,222,sensor='45')}

sensor='90'
cull = cull_util.runculls(repointings[sensor],energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")


npp = np.size(repointings[sensor])
jpu = np.ndarray((3,npp))
tcen = np.ndarray(npp)
for ic in range(npp):
    xp = MyUltraFile.L1Bxspin(repointings[sensor][ic]).data
    tcen[ic] = spiceypy.scs2e(-43, f"{np.mean(xp['spin_start_time'])}")
    jp, lt = spiceypy.spkpos("JUPITER_BARYCENTER", tcen[ic], 'IMAP_DPS', 'NONE', '-43')
    r = np.linalg.norm(jp)
    jpu[:,ic] = jp/r

jaz = np.arctan2(jpu[1,:],jpu[0,:])
jel = np.arcsin(jpu[2,:])





npoint = len(repointings[sensor])
et = np.zeros(npoint)
for ic in range(npoint):
    et[ic] = spiceypy.scs2e(-43, f"{np.mean(xspin['spin_start_time'])}")

