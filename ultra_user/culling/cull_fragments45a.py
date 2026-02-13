import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import ultra_user.culling.cull_util as cull_util
import spiceypy
import ultra_user.planets.ENA_planets as ENA_planets
import imap_processing.spice.time as spiceTime
from importlib import reload
from importlib import reload


spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')

spiceypy.furnsh('data/imap/spice/spk/imap_recon_20250925_20260203_v01.bsp')
spiceypy.furnsh('data/imap/spice/spk/de440.bsp')
spiceypy.furnsh('data/imap/spice/fk/imap_130.tf')
spiceypy.furnsh('data/imap/spice/fk/imap_science_110.tf')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_270_2025_354_001.ah.bc')
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_037_002.ah.bc')

repoint = 47
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

c90 = cull_util.runculls(repointings90,energy_ranges)
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45')

cull_util.cullplot(c90)
cull_util.cullplot(c45)

# find unconverged series
unconv45 = list()
for repointing in repointings45:
    if not c45['scull'][repointing]['converge'][1]:
        unconv45.append(repointing)

unconv90 = list()
for repointing in repointings45:
    if not c90['scull'][repointing]['converge'][1]:
        unconv90.append(repointing)
