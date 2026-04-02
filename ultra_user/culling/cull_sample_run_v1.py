import numpy as np
import ultra_user.culling.cull_util as cull_util
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

energy_ranges = cull_util.l1c_energy_ranges()
repointings90 = cull_util.get_pointings(27,183)
repointings45 = cull_util.get_pointings(27,183,sensor='45')

c90 = cull_util.runculls(repointings90,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")
c45 = cull_util.runculls(repointings45,energy_ranges,sensor='45',cullPackage="hiEnergy_upstream_spectral_stat_v1")

cull_util.cullplot(c90)
cull_util.cullplot(c45)

