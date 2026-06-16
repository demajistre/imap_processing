import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
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

#repoint = 47
repoint = 70
de = MyUltraFile.L1Bde(repoint,sensor='45').data
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

# earth pointed plot
t0 = spiceypy.scs2e(-43,f"{np.mean(xspin['spin_start_time'])}")
de_velocity = de['velocity_dps_sc'][ii,:]
earth = planets.ENA_planets(t0)

local_uv = earth.local_uvec(de_velocity)
local_az = np.arctan2(local_uv[1,:], local_uv[0,:])
local_el = np.arccos(local_uv[2,:])
plt.hist2d(np.degrees(local_az),90-np.degrees(local_el), bins=200)
plt.axis('equal')
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.show()

# removing the earth (15 degrees)
anglim = np.radians(15)
coslim = np.cos(anglim)
jj = np.nonzero(np.abs(local_uv[0,:] < coslim))[0]
fig, ax = plt.subplots(2)
a1 = ax[0].hist2d(np.degrees(az),np.degrees(el), bins=50)
a2 = ax[1].hist2d(np.degrees(az[jj]),np.degrees(el[jj]), bins=50)
fig.colorbar(a1[3],ax=ax[0])
fig.colorbar(a2[3],ax=ax[1])
plt.show()




#raw material
t0 = spiceypy.scs2e(-43,f"{np.mean(xspin['spin_start_time'])}")

de_velocity = de['velocity_dps_sc'][ii,:]
de_vmag = np.sqrt(np.sum(de_velocity**2,1))
de_uv = de_velocity
for ic in range(3):
    de_uv[:,ic] = de_velocity[:,ic]/de_vmag

az = np.arctan2(de_uv[:,1], de_uv[:,0])
el = np.arccos(de_uv[:,2])

pos,lt = spiceypy.spkpos('EARTH',t0, 'IMAP_DPS','NONE','-43')
upos = pos/np.sqrt(np.sum(pos**2))

Xax = upos
Zax0 = np.cross(Xax,[0,1,0])
Zax = Zax0/np.sqrt(np.sum(Zax0**2))
Yax = np.cross(Zax,Xax)

Rot = np.array([Xax,Yax,Zax])

#uv_local = np.transpose(Rot)@np.transpose(de_uv)
uv_local = Rot@np.transpose(-de_uv)
az_local = np.arctan2(uv_local[1,:], uv_local[0,:])
el_local = np.arccos(uv_local[2,:])

plt.hist2d(np.degrees(az_local),90-(np.degrees(el_local)),bins=200)
plt.title('Counts (U45) in earth centered system')
plt.axis('equal')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Latitude (deg)')
plt.colorbar()
plt.show()

plt.hist2d(np.degrees(az_local),90-(np.degrees(el_local)),bins=200)
plt.title('Counts (U45) in earth centered system')
plt.axis('equal')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Latitude (deg)')
plt.xlim(-20,20)
plt.ylim(-20,20)
plt.colorbar()
plt.show()


plt.hist2d(az_local*1.5e6/6371,(np.pi*.5-el_local)*1.5e6/6371,bins=200)
plt.axis('equal')
plt.xlim(-50,50)
plt.ylim(-50,50)
plt.show()

ang0=de_vmag*0
for ic in range(len(de_vmag)):
    ang0[ic] = np.arccos(np.sum(-upos*de_uv[ic,:]))


planet = planets.ENA_planets(t0)
dist = planet.distance_from_planet(de_velocity)

h,dbin = np.histogram(dist,bins=range(0,100,1))
dcen=(dbin[0:-1]+dbin[1:])*.5

plt.plot(dcen,h/(dcen**2))
plt.show()

uv = planet.local_uvec(de_velocity)
az = np.arctan2(uv[1], uv[0])
el = np.arccos(uv[2])
plt.hist2d(np.degrees(az),90-np.degrees(el),bins=100)
plt.show()

#
v_sc = de['velocity_sc'][ii,:]
vv = np.sqrt(np.sum(v_sc**2,1))
uv_sc = v_sc*0
for ic in range(3):
    uv_sc[:,ic] = v_sc[:,ic]/vv
sc_az = np.arctan2(uv_sc[:,1],uv_sc[:,0])
sc_el = np.arccos(uv_sc[:,2])
plt.hist2d(np.degrees(sc_az),90-np.degrees(sc_el),bins=100)
plt.show()

