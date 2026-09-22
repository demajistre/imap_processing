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

#NOTE - this is patterned after earth_frag1.py - we'll probably want to put this in a class

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

energy_ranges = cull_util.l1c_energy_ranges()
nen = np.size(energy_ranges[:,0])


sensor='90'  # stick with 90 for now, Earth gets in the way
repointings = cull_util.get_pointings(27,222,sensor=sensor)
cull=cull_util.runculls(repointings,energy_ranges,cullPackage="hiEnergy_upstream_spectral_stat_v1")

azGrid = np.arange(-45.,45)
elGrid = np.arange(-45.,45.)
angGrid = np.arange(0,90)
naz = np.shape(azGrid)[0]
nel = np.shape(elGrid)[0]
nang = np.shape(angGrid)[0]
ebin_range = [1, 19]
hrange = [[np.min(azGrid),np.max(azGrid)],[np.min(elGrid),np.max(elGrid)]]
spice_ID="JUPITER_BARYCENTER"

npoint = np.size(repointings)
cnt1d = np.ndarray((npoint,nen,nang),dtype=float)
cnt2d = np.ndarray((npoint,nen,nang,nang),dtype=float)
ic=0
for ip in repointings:
    print(f"pointing {ip}")
    earthHist = list()
    eh1 = list()
    cullDat = cull['cullData'][ip]
    de = cullDat.de
    xspin = cullDat.xspin
    t0 = spiceypy.scs2e(-43, f"{np.mean(xspin['spin_start_time'])}")
    body0 = planets.ENA_planets(t0,spice_ID=spice_ID)
    distance = np.sqrt(np.sum(body0.ePos**2))
    for ie in range(nen):
        erange = energy_ranges[ie,:]
        ii = cullDat.culledDE_indices(ie)
        if np.size(ii) >0:
            de_velocity = de['velocity_dps_sc'][ii, :]
            de_speed = np.sqrt(np.sum(de_velocity ** 2, axis=1))
            delt = distance/np.mean(de_speed)
            body = planets.ENA_planets(t0-delt, spice_ID=spice_ID)
            local_uv = body.local_uvec(de_velocity)
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

plt.imshow(sum2d[0,:,:])
plt.colorbar()
plt.plot([45],[45],'.',color='orange')
plt.show()