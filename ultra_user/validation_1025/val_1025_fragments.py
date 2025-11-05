import cdflib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

nside=32


rootdir = '/Users/demajr1/files/imap_ultra/validation/2025Oct/'
l1cFname = rootdir+'imap_ultra_l1c_90sensor-spacecraftpset-nonproton_20260926_v001.cdf'
l2Fname = rootdir+'imap_ultra_l2_u90-ena-h-hf-nsp-full-hae-2deg-0mo_20260926_v002.cdf'

l1c = cdflib.CDF(l1cFname)
l2= cdflib.CDF(l2Fname)

l1cNames = l1c.cdf_info().zVariables
for name in l1cNames:
    print(f'{name}: {np.shape(l1c[name])}')

l1cMask = dict()
npix = np.shape(l1c['counts'])[-1]
for name in l1cNames:
    if np.shape(l1c[name])[-1] == npix:
        l1cMask[name] = np.where(l1c[name] > -1.e-30 , True, False)


l2Names = l2.cdf_info().zVariables
for name in l2Names:
    print(f'{name}: {np.shape(l2[name])}')

l2Mask = dict()
expShape = np.shape(l2['exposure_factor'])[-2:]
for name in l2Names:
    if np.shape(l2[name])[-2:] == expShape :
        l2Mask[name] = np.where(l2[name] > -1.e-30 , True, False)


hp.mollview(l1c['counts'][0,10,:])
plt.show()

plt.imshow(l2['ena_intensity'][0,0,:,:])
plt.colorbar()
plt.show()

# L2 check
ic=6
mask = np.where(np.ndarray.flatten(l2Mask['ena_intensity'][0,ic,:,:]))
flux = np.ndarray.flatten(l2['ena_intensity'][0,ic,:,:])[mask]
unc = np.ndarray.flatten(l2['ena_intensity_stat_unc'][0,ic,:,:])[mask]
fluxmean = np.mean(flux)
fluxwmean = np.average(flux,weights=1/unc)
hist = np.histogram(flux)
whist = np.histogram(flux,weights=1/unc)

