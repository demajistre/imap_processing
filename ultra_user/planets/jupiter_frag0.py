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
spiceypy.furnsh('data/imap/spice/ck/imap_dps_2025_359_2026_037_002.ah.bc')

#repoint = 47
repoint = 70
de = MyUltraFile.L1Bde(repoint).data
xspin = MyUltraFile.L1Bxspin(repoint).data
xspinvars = xspin.cdf_info().zVariables
devars = de.cdf_info().zVariables

energy_ranges = np.ndarray((6, 2))
energy_ranges[0, :] = [4.6, 10.27]
energy_ranges[1, :] = [10.27, 23.4444]
energy_ranges[2, :] = [23.4444, 52.2113]
energy_ranges[3, :] = [52.2113, 116.276]
energy_ranges[4, :] = [116.276, 258.95]
energy_ranges[5, :] = [258.95, 316.335]
ebin_range = [1, 19]
ieBin = 1
ii = np.nonzero(np.logical_and(
                np.logical_and(
                np.logical_and(np.logical_and(
                    np.logical_and(de['energy_spacecraft'] > energy_ranges[ieBin, 0],
                               de['energy_spacecraft'] < energy_ranges[ieBin, 1]),
                    de['quality_outliers'] == 0), de['quality_scattering'] == 0),
                de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))[0]

az=np.array(list())
lat=np.array(list())
repointings90 = cull_util.get_pointings(27,153)
for repoint in repointings90:
    de = MyUltraFile.L1Bde(repoint).data
    xspin = MyUltraFile.L1Bxspin(repoint).data
    t0 = spiceypy.scs2e(-43, f"{np.mean(xspin['spin_start_time'])}")
    ii = np.nonzero(np.logical_and(
        np.logical_and(
            np.logical_and(np.logical_and(
                np.logical_and(de['energy_spacecraft'] > energy_ranges[ieBin, 0],
                               de['energy_spacecraft'] < energy_ranges[ieBin, 1]),
                de['quality_outliers'] == 0), de['quality_scattering'] == 0),
            de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))[0]
    de_velocity = de['velocity_dps_sc'][ii, :]
    try:
        jupiter = planets.ENA_planets(t0, spice_ID="JUPITER_BARYCENTER")

        local_uv = jupiter.local_uvec(de_velocity)
        local_az = np.arctan2(local_uv[1, :], local_uv[0, :])
        local_el = np.arccos(local_uv[2, :])
        az0 = np.degrees(local_az)
        lat0 = 90-np.degrees(local_el)
        print(f"repoint {repoint}: {len(az0)} azimuthal, {np.max(az0)}, {np.min(az0)}")
        jj = np.nonzero(np.logical_and(np.abs(az0) < 20,np.abs(lat0) < 20))[0]
        np.append(az,az0[jj])
        np.append(lat,lat0[jj])
    except Exception:
        print(f"pointing {repoint} has no DPS frame")

# instrument coords plot
de_velocity_inst = de['velocity_sc'][ii,:]
de_vmag = np.sqrt(np.sum(de_velocity_inst**2, axis=1))
de_uv = de_velocity_inst*0
for ic in range(3):
    de_uv[:,ic] = de_velocity_inst[:,ic]/de_vmag
az = np.arctan2(de_uv[:,1],de_uv[:,0])
el = np.arccos(de_uv[:,2])
plt.hist2d(np.degrees(az),np.degrees(el), bins=100)
plt.show()

# jupiter pointed plot
t0 = spiceypy.scs2e(-43,f"{np.mean(xspin['spin_start_time'])}")
de_velocity = de['velocity_dps_sc'][ii,:]
jupiter = planets.ENA_planets(t0,spice_ID="JUPITER_BARYCENTER")

local_uv = jupiter.local_uvec(de_velocity)
local_az = np.arctan2(local_uv[1,:], local_uv[0,:])
local_el = np.arccos(local_uv[2,:])
plt.hist2d(np.degrees(local_az),90-np.degrees(local_el), bins=200)
plt.axis('equal')
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.show()
