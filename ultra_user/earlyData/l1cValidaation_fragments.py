import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.data_access.Valdata as Valdata
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import pandas as pd
import glob
import healpy as hp
import imap_processing.spice.time as spiceTime
from importlib import reload

# change this to something better
#spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
#spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
spiceypy.furnsh('data/imap_val0/spice/sclk/imap_sclk_0070.tsc')
spiceypy.furnsh('data/imap_val0/spice/lsk/naif0012.tls')

###
repoint = 55
#repoint = 31
sensor = '90'
rootDir = 'data/imap'
de = MyUltraFile.L1Bde(repoint, sensor=sensor,rootDir=rootDir).data
xspin = MyUltraFile.L1Bxspin(repoint, sensor=sensor,rootDir=rootDir).data
l1c = MyUltraFile.L1C(repoint, sensor=sensor,rootDir=rootDir).data
status = MyUltraFile.L1Bstatus(repoint, sensor=sensor,rootDir=rootDir).data



valset = Valdata.get_validation_set()

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

for var in l1cvars:
    print(f"{var}: {np.shape(l1c[var])}")
for var in devars:
    print(f"{var}: {np.shape(de[var])}")

l1c_ebins = np.transpose([l1c['energy_bin_geometric_mean'] - l1c['energy_delta_minus'],
                          l1c['energy_bin_geometric_mean'] + l1c['energy_delta_plus']])

tcounts_1c = np.sum(l1c['counts'], axis=2)[0]
val_cnts = valset['counts'].get_pointing_data(repoint)
#recommended values
#ebin_range = [5, 17]
#sdc_values
ebin_range = [1, 19]

ii = np.nonzero(de['energy_spacecraft'] > 0)[0]
jj = np.nonzero(np.logical_and(
    np.logical_and(np.logical_and(np.logical_and(np.logical_and(de['energy_spacecraft'] > np.min(l1c_ebins),
                                                                de['energy_spacecraft'] <= np.max(l1c_ebins)),
                                                 de['quality_outliers'] == 0), de['quality_scattering'] == 0),
                   de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))[0]

hcnts, eh = np.histogram(de['energy_spacecraft'][jj], l1c_ebins[:, 0])
vcnts = np.sum(val_cnts, axis=0)

fig, axs = plt.subplots(2)
axs[0].plot(l1c['energy_bin_geometric_mean'][0:-1], tcounts_1c[0:-1], label='L1C counts')
axs[0].plot(l1c['energy_bin_geometric_mean'][0:-1], hcnts, label='L1B histogram')
axs[0].plot(l1c['energy_bin_geometric_mean'][0:-1], vcnts[0:-1], label='validation cnts')
axs[0].set_title(f"Repointing {repoint}")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_ylabel('counts')
axs[0].legend()
axs[1].plot(l1c['energy_bin_geometric_mean'][0:-1], (tcounts_1c[0:-1] - vcnts[0:-1]), label='diff (l1c - val)')
axs[1].plot(l1c['energy_bin_geometric_mean'][0:-1], (hcnts-vcnts[0:-1]), label='diff (l1b - val)')
axs[1].set_xscale('log')
axs[1].set_ylabel('counts')
axs[1].set_xlabel('Center bin energy (keV)')
axs[1].legend()
plt.show()

# find the differences

# look at 1 channel
ich = 5
erange = l1c_ebins[ich, :]

kk0 = np.nonzero(
    np.logical_and(np.logical_and(np.logical_and(de['energy_spacecraft'] > erange[0],
                                                                               de['energy_spacecraft'] < erange[1]),
                                                                de['quality_outliers'] == 0),
                                                 de['quality_scattering'] == 0))
kk1 = np.nonzero(np.logical_and(de['energy_spacecraft'] > erange[0],de['energy_spacecraft'] < erange[1]))


kk = np.nonzero(
    np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(de['energy_spacecraft'] > erange[0],
                                                                               de['energy_spacecraft'] < erange[1]),
                                                                de['quality_outliers'] == 0),
                                                 de['quality_scattering'] == 0),
                                  de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))

nde = len(kk[0])
nval = np.sum(val_cnts,axis=0)[ich]
n1c = np.sum(l1c['counts'][0,ich,:])


