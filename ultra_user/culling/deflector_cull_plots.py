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

#

thresh=range(0,3500,50)
nb10 = np.ndarray((len(repointings),len(thresh)))
nb20 = np.ndarray((len(repointings),len(thresh)))
nb100 = np.ndarray((len(repointings)))
nb200 = np.ndarray((len(repointings)))
ip=0
for repoint in repointings:
    cullData20 = UltraCull0.UltraCull0(repoint,energy_ranges,spin_range=20)
    cullData10 = UltraCull0.UltraCull0(repoint,energy_ranges,spin_range=10)
    vc200 = cullData20.voltage_cull(v_threshold=0,apply=False)
    vc100 = cullData10.voltage_cull(v_threshold=0,apply=False)
    nb200[ip]= len(np.nonzero(vc200['binMask'])[0])
    nb100[ip]= len(np.nonzero(vc100['binMask'])[0])
    it=0
    for iv in thresh:
        vc10 = cullData10.voltage_cull(v_threshold=iv, apply=False)
        vc20 = cullData20.voltage_cull(v_threshold=iv, apply=False)
        nb10[ip,it] = len(np.nonzero(vc10['binMask'])[0])
        nb20[ip,it] = len(np.nonzero(vc20['binMask'])[0])
        it=it+1
    ip=ip+1
    print(repoint)

nthresh = len(thresh)
frac10 = np.ndarray(nthresh)
frac20 = np.ndarray(nthresh)

n10total0 = np.sum(nb100)
n20total0 = np.sum(nb200)

frac20 = np.sum(nb20,0)/n20total0
frac10 = np.sum(nb10,0)/n10total0

#
cullData20 = UltraCull0.UltraCull0(28,energy_ranges,spin_range=20)
cullData10 = UltraCull0.UltraCull0(28,energy_ranges,spin_range=10)

vc10 = cullData10.voltage_cull()
vc20 = cullData20.voltage_cull()

plt.plot(thresh,frac10, label='10 spins/bin (~150s)')
plt.plot(thresh,frac20, label='20 spins/bin (~300s)')
plt.xlabel('Voltage Threhold (V)')
plt.ylabel('Fraction culled')
plt.legend()
plt.show()

plt.plot(thresh,frac10 - frac20)
plt.ylabel('fraction difference between 20 and 10 spins/bin')
plt.xlabel('Voltage Threhold (V)')
plt.show()

#full cull
cullData = dict()
ecull = dict()
scull = dict()
vcull = dict()
cnt_sum = dict()
cullFrac = dict()
for repoint in repointings:
    print(repoint)
    cullData[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges,spin_range=20)
    cnt_sum[repoint] = cullData[repoint].get_count_summary()
    vcull[repoint] = cullData[repoint].voltage_cull()
    ecull[repoint] = cullData[repoint].high_energy_cull()
    scull[repoint] = cullData[repoint].statistical_cull()
    cullFrac[repoint] = cullData[repoint].currentCullFraction()
