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
cullFrac = dict()
for repoint in repointings:
    print(repoint)
    cullData[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges,spin_range=20)
    cnt_sum[repoint] = cullData[repoint].get_count_summary()
    vcull[repoint] = cullData[repoint].voltage_cull()
    ecull[repoint] = cullData[repoint].high_energy_cull()
    scull[repoint] = cullData[repoint].statistical_cull()
    cullFrac[repoint] = cullData[repoint].currentCullFraction()

cFrac = np.ndarray((nen,len(repointings)))
cfrac2 = np.ndarray((nen,len(repointings)))
for ic in range(len(repointings)):
    cFrac[:,ic] = cullFrac[repointings[ic]]
    cfrac2[:,ic] = cFrac[:,ic]
    for jc in range(nen):
        if not scull[repointings[ic]]['converge'][jc]:
            cfrac2[jc, ic]=0

for ech in range(4):
    plt.plot(repointings,cFrac[ech,:],'.',label=f"Energy {ech}")
plt.legend()
plt.show()

for ech in range(4):
    plt.plot(repointings,cfrac2[ech,:],'.',label=f"Energy {ech}")
plt.legend()
plt.show()

for ic in range(len(repointings)):
    print(repointings[ic],cFrac[0:3,ic],cfrac2[0:3,ic])

# looking at deflection voltages
fig,ax1 = plt.subplots()
c1 = 'b'
ax1.set_xlabel('MET seconds')
ax1.set_ylabel('Mininum deflector voltage (V)', color=c1)
ax2 = ax1.twinx()
c2 = 'tab:orange'
ax2.set_ylabel('Bin 1 counts/bin', color=c2)  # we already handled the x-label with ax1
ax2.set_yscale('log')
ax2.set_ylim(5,5000)

for repoint in repointings:
    print(repoint)
    cbins = cullData[repoint].center_spin()
    status = cullData[repoint].status
    minv=np.minimum(status['rightdeflection_v'], status['leftdeflection_v'])

    ax1.plot(status['shcoarse'],minv,c1,label='minimum deflector voltage')
    ax2.plot(cbins['center_time'],cullData[repoint].get_count_summary()[:,1],c2,label='Counts in Energy Bin 1')
    ii = np.nonzero(vcull[repoint]['binMask'])[0]
    ax2.plot(cbins['center_time'][ii], cullData[repoint].get_count_summary()[ii, 1], 'g', label='culled')
plt.show()


# count rates in the channels
#%%

t0 = spiceypy.sce2t(-43,spiceypy.str2et("2025-01-01T00"))*2.e-5 +1#magic number time conversion
nch=5
fig,axs = plt.subplots(nch)
for ic in range(nch):
    axs[ic].set_yscale('log')
    axs[ic].set_ylim(1,1000)
    axs[ic].set_ylabel(f"counts ({ic})")
for repoint in repointings:
    tDay = (cullData[repoint].center_spin()['center_time'] - t0) / 86400
    for ech in range(nch):
        axs[ech].plot(tDay, cnt_sum[repoint][:, ech], 'b')
        axs[ech].plot([np.mean(tDay)], [np.mean(cnt_sum[repoint][:,ech])], '.r')
axs[nch-1].set_xlabel("day of 2025")
plt.show()

#full cull
t0 = spiceypy.sce2t(-43,spiceypy.str2et("2025-01-01T00"))*2.e-5 +1#magic number time conversion
nch=4
fig,axs = plt.subplots(nch)
for repoint in repointings:
    print(repoint)
    tDay = (cullData[repoint].center_spin()['center_time']-t0)/86400
    for ech in range(nch):
        cnts = cullData[repoint].get_count_summary()[:,ech]
        ii = np.nonzero(cullData[repoint].currentMask['bin_mask'][ech,:])[0]
        axs[ech].plot(tDay, cnts,'r')
        if len(ii) > 0:
            color='g'
            if scull[repoint]['converge'][ech] == False:
                color='b'
            axs[ech].plot(tDay[ii], cnts[ii], color)
for ech in range(nch):
    axs[ech].set_ylim(0,30)
    axs[ech].set_ylabel(f"counts ({ech})")
axs[nch-1].set_xlabel("day of 2025")
plt.show()

# look for non-converged
ich = 0
nonconv = list()
for repoint in repointings:
    if not scull[repoint]['converge'][ich]:
        nonconv.append(repoint)


# focus on 1 repointing (from scratch - then check against full run
pointing = nonconv[0] # non convergent

c0 = UltraCull0.UltraCull0(pointing,energy_ranges,spin_range=20)
ctime = c0.center_spin()['center_time']

cnts0 = c0.get_count_summary()[:,ich]
cntsHi = c0.get_count_summary()[:,4]
ctime0 = ctime
minv=np.minimum(c0.status['rightdeflection_v'], c0.status['leftdeflection_v'])
vc = c0.voltage_cull()
iiv = np.nonzero(vc['binMask'])[0]
ec = c0.high_energy_cull()
iie = np.nonzero(ec['mask'])[0]
sc = c0.statistical_cull()
iis = np.nonzero(sc['mask'][ich,:])[0]


fig,ax1 = plt.subplots()
ax1.set_xlabel('MET seconds')
ax2 = ax1.twinx()
ax2.set_ylabel('Mininum deflector voltage (V)')
ax1.set_ylabel(f"Bin {ich} counts/bin")  # we already handled the x-label with ax1
ax2.plot(c0.status['shcoarse'],minv,'r',label='Mininum deflector voltage (V)')
ax1.plot(ctime,cnts0,'b',label=f"Bin {ich} counts/bin")
ax1.legend()
ax2.legend()
plt.show()

fig,ax1 = plt.subplots()
ax1.set_xlabel('MET seconds')
ax2 = ax1.twinx()
ax2.set_ylabel('High energy channel')
ax1.set_ylabel(f"Bin {ich} counts/bin")  # we already handled the x-label with ax1
ax2.plot(ctime,cntsHi,'r',label=f"Counts in high energy Bin")
ax2.plot(ctime,cntsHi*0+25,'-r')
ax1.plot(ctime,cnts0,'b',label=f"Counts in Energy Bin {ich}")
ax1.plot(ctime[iis],cnts0[iis],'g')
ax1.legend()
ax2.legend(loc='upper right')
plt.show()

#convergence check
mu = np.mean(cnts0[iis])
sig = np.std(cnts0[iis])


#%% md
