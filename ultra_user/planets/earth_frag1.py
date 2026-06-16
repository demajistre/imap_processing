import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import ultra_user.culling.cull_util as cull_util
import spiceypy
import ultra_user.planets.ENA_planets as planets
import imap_processing.spice.time as spiceTime
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

energy_ranges = cull_util.l1c_energy_ranges(n_1cbins=2,maxbin_lim=20)
energy_ranges = energy_ranges[:-1]
nen = np.shape(energy_ranges)[0]

repointings45 = cull_util.get_pointings(130,222,sensor='45')

npoint = np.size(repointings45)
azGrid = np.arange(-45.,45)
elGrid = np.arange(-45.,45.)
angGrid = np.arange(0,90)
naz = np.shape(azGrid)[0]
nel = np.shape(elGrid)[0]
nang = np.shape(angGrid)[0]
ebin_range = [1, 19]
hrange = [[np.min(azGrid),np.max(azGrid)],[np.min(elGrid),np.max(elGrid)]]

cnt1d = np.ndarray((npoint,nen,nang),dtype=float)
cnt2d = np.ndarray((npoint,nen,nang,nang),dtype=float)


ic=0
for ip in repointings45:
    print(f"pointing {ip}")
    earthHist = list()
    eh1 = list()
    de = MyUltraFile.L1Bde(ip, sensor='45').data
    xspin = MyUltraFile.L1Bxspin(ip, sensor='45').data
    t0 = spiceypy.scs2e(-43, f"{np.mean(xspin['spin_start_time'])}")
    earth = planets.ENA_planets(t0)
    for ie in range(nen):
        erange = energy_ranges[ie,:]
        ii = np.nonzero(np.logical_and(
            np.logical_and(
                np.logical_and(np.logical_and(
                    np.logical_and(de['energy_spacecraft'] > erange[0],
                                   de['energy_spacecraft'] < erange[1]),
                    de['quality_outliers'] == 0), de['quality_scattering'] == 0),
                de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))[0]
        de_velocity = de['velocity_dps_sc'][ii, :]

        local_uv = earth.local_uvec(de_velocity)
        local_az = np.arctan2(local_uv[1, :], local_uv[0, :])
        local_el = np.arccos(local_uv[2, :])
        ang = np.degrees(np.arccos(local_uv[0,:]))
        h1d,bloc = np.histogram(ang,bins=90,range=(0,90))
        h2d,bloc1,bloc2 = np.histogram2d(np.degrees(local_az),90-np.degrees(local_el), bins=[90,90],range=hrange)
        cnt1d[ic,ie,:] = h1d
        cnt2d[ic,ie,:,:]=h2d
    ic += 1

sum1d = np.sum(cnt1d,axis=0)
sum2d = np.sum(cnt2d,axis=0)

epower = np.zeros_like(sum1d)
scale1d = 1./np.sin(np.radians(angGrid+0.5))

for ie in range(nen):
    epow = sum1d[ie,:]*scale1d
    epower[ie,:] = epow/np.max(epow)

for ie in range(nen):
    plt.plot(angGrid + .5, epower[ie, :],label=f"{energy_ranges[ie,0]:.2f}-{energy_ranges[ie,1]:.2f} kev")
plt.xlabel("Angle from Earth (degrees)")
plt.ylabel("Normalized Counts")
plt.legend()
plt.show()

for ie in range(nen):
    plt.plot((angGrid + .5)/.25, epower[ie, :],label=f"{energy_ranges[ie,0]:.2f}-{energy_ranges[ie,1]:.2f} kev")
plt.xlabel("Angle from Earth (Earth Radii)")
plt.ylabel("Normalized Counts")
plt.legend()
plt.show()


for ie in range(nen):
    plt.plot((angGrid + .5)/.25, epower[ie, :],label=f"{energy_ranges[ie,0]:.2f}-{energy_ranges[ie,1]:.2f} kev")
plt.yscale('log')
plt.xlim(0,200)
plt.xlabel("Angle from Earth (Earth Radii)")
plt.ylabel("Normalized Counts")
plt.legend(loc='upper right')
plt.show()
