import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import imap_processing.spice.time as spiceTime
from importlib import reload




# change this to something better
spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0103.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')

# setup to get data variable names and energy ranges
repoint = 47
de = MyUltraFile.L1Bde(repoint).data
xspin = MyUltraFile.L1Bxspin(repoint).data
l1c = MyUltraFile.L1C(repoint).data
status = MyUltraFile.L1Bstatus(repoint).data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

l1c_ebins = np.transpose([l1c['energy_bin_geometric_mean']-l1c['energy_delta_minus'],
             l1c['energy_bin_geometric_mean']+l1c['energy_delta_plus']])

base_ebin = 4
n_1cbins=8
ebin_start = np.arange(base_ebin,len(l1c_ebins),n_1cbins,dtype=int)
ebin_end = np.append(ebin_start[1:]-1,len(l1c_ebins)-1)
nen= len(ebin_start)

eranges = np.ndarray((2,nen))
for ic in range(len(ebin_start)):
    eranges[:,ic] = [l1c_ebins[ebin_start[ic]][0],l1c_ebins[ebin_end[ic]][1]]
energy_ranges = np.transpose(eranges)


#repointings = np.append(range(27,48),range(50,64))
repointings = np.append(range(27,99),range(125,136))

#full cull
cullData = dict()
ecull = dict()
scull = dict()
vcull = dict()
cnt_sum = dict()
cullFrac0 = dict()
cullFrac = dict()
for repoint in repointings:
    print(repoint)
    cullData[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges,spin_range=20)
    cnt_sum[repoint] = cullData[repoint].get_count_summary()
    vcull[repoint] = cullData[repoint].voltage_cull()
    ecull[repoint] = cullData[repoint].high_energy_cull()
    cullFrac0[repoint] = cullData[repoint].currentCullFraction()
    scull[repoint] = cullData[repoint].statistical_cull()
    cullFrac[repoint] = cullData[repoint].currentCullFraction()


cFrac0 = np.ndarray((nen,len(repointings)))
cFrac1 = np.ndarray((nen,len(repointings)))
cFrac2 = np.ndarray((nen,len(repointings)))
for ic in range(len(repointings)):
    cFrac0[:,ic] = cullFrac0[repointings[ic]]
    cFrac1[:,ic] = cullFrac[repointings[ic]]
    cFrac2[:,ic] = cFrac1[:,ic]
    for jc in range(nen):
        if not scull[repointings[ic]]['converge'][jc]:
            cFrac2[jc, ic]=0

np.savetxt('/Users/demajr1/tmp/cfrac0.csv',np.transpose(cFrac0[0:4]),delimiter=',')
np.savetxt('/Users/demajr1/tmp/cfrac1.csv',np.transpose(cFrac1[0:4]),delimiter=',')
np.savetxt('/Users/demajr1/tmp/cfrac2.csv',np.transpose(cFrac2[0:4]),delimiter=',')

#looking at george's list these are the questionable pointings (good days with ~< 0.9 cfrac)
qps = [53,62,69,75,98,127,129]
qps.sort(key=lambda p:cullFrac0[p][0]) #sorted from worst to best

for repoint in qps:
    fig, ax = plt.subplots(5)
    ax[0].set_xlabel('MET seconds')
    ax2 = ax[0].twinx()
    ax[0].set_ylabel('High energy channel')
    ax2.set_ylabel('Mininum deflector voltage (V)')

c0=cullData[repoint]
minv=np.minimum(c0.status['rightdeflection_v'], c0.status['leftdeflection_v'])
cntsHi = c0.get_count_summary()[:,4]
ctime = c0.center_spin()['center_time']
fig, ax = plt.subplots(5)
ax[0].set_xlabel('MET seconds')
ax2 = ax[0].twinx()
ax[0].set_ylabel('High energy channel')
ax2.set_ylabel('Mininum deflector voltage (V)')
ax2.set_yscale('log')
ax2.set_ylim([1,100])
ax[0].plot(c0.status['shcoarse'],minv,'b',label='Mininum deflector voltage (V)')
ax2.plot(ctime,cntsHi,'r',label='Mininum deflector voltage (V)')
plt.show()
