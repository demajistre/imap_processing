import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import imap_processing.spice.time as spiceTime
from importlib import reload

# change this to something better
spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
###
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

eranges = np.ndarray((2,len(ebin_start)))
for ic in range(len(ebin_start)):
    eranges[:,ic] = [l1c_ebins[ebin_start[ic]][0],l1c_ebins[ebin_end[ic]][1]]
energy_ranges = np.transpose(eranges)

cnt_data = dict()

repointings = np.append(range(27,48),range(50,59))

#full cull
cullData = dict()
ecull = dict()
scull = dict()
vcull = dict()
cnt_sum = dict()
for repoint in repointings:
    print(repoint)
    cullData[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges)
    cnt_sum[repoint] = cullData[repoint].get_count_summary()
    vcull[repoint] = cullData[repoint].voltage_cull()
    ecull[repoint] = cullData[repoint].high_energy_cull()
    scull[repoint] = cullData[repoint].statistical_cull()

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

# voltage and high energy cull
t0 = spiceypy.sce2t(-43,spiceypy.str2et("2025-01-01T00"))*2.e-5 +1#magic number time conversion
nch=4
fig,axs = plt.subplots(nch)
for repoint in repointings:
    print(repoint)
    tDay = (cullData[repoint].center_spin()['center_time']-t0)/86400
    ii = np.nonzero(ecull[repoint]['mask'])[0]
    for ech in range(nch):
        cnts = cullData[repoint].get_count_summary()[:,ech]
        axs[ech].plot(tDay, cnts,'r')
        if len(ii) > 0:
            color='g'
            axs[ech].plot(tDay[ii], cnts[ii], color)
for ech in range(nch):
    axs[ech].set_ylim(0,30)
plt.show()

# just the counts
t0 = spiceypy.sce2t(-43,spiceypy.str2et("2025-01-01T00"))*2.e-5 +1#magic number time conversion
nch=5
fig,axs = plt.subplots(nch)
for repoint in repointings:
    print(repoint)
    tDay = (cullData[repoint].center_spin()['center_time'] - t0) / 86400
    for ech in range(nch):
        axs[ech].plot(tDay, cullData[repoint].get_count_summary()[:,ech], 'r')
        axs[ech].set_ylim(0,300)
        axs[ech].set_ylabel(f"counts ({ech})")
axs[nch-1].set_xlabel("day of 2025")
plt.show()

#############
for repoint in repointings:
    cnt_data[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges)

cntSummary = dict()
lvSpins_bins = dict()
lvSpins_spins = dict()
spinBins = dict()
centerSpins = dict()
vthresh = 3000
for repoint in repointings:
    print(repoint)
    spinBins[repoint] = cnt_data[repoint].spinbins
    cntSummary[repoint] = cnt_data[repoint].get_count_summary()
    lvSpins_bins[repoint] = cnt_data[repoint].lowVoltagespins(vthresh)
    centerSpins[repoint] = cnt_data[repoint].center_spin()

# build a big damn scatterplot
bigCnts = cnt_sum[repointings[0]][np.nonzero(vcull[repointings[0]]['binMask'])[0],:]
for repoint in repointings[1:]:
    bigCnts = np.concatenate((bigCnts,cnt_sum[repoint][np.nonzero(vcull[repoint]['binMask'])[0],:]))

hibins = range(0,250)
nbin=len(hibins)
cbin = 1
hbin = 4
idig = np.digitize(bigCnts[:,hbin],hibins)
cmean = np.zeros((nbin,6))
cstd = np.zeros((nbin,6))
cn = np.zeros(nbin)

for ic in range(1,nbin):
    ii = np.nonzero(idig == ic)[0]
    if len(ii) > 0:
        cn[ic]=len(ii)
        cmean[ic,:]=np.mean(bigCnts[ii,:],axis=0)
        cstd[ic,:]=np.std(bigCnts[ii,:],axis=0)

frac=np.zeros((nbin))
btot=np.sum(cn)
for ic in range(1,nbin):
    frac[ic] = np.sum(cn[0:ic])/btot

for ic in range(4):
    plt.plot(cmean[:,hbin],cmean[:,ic],label=f"channel {eranges[0,ic]:.1f}-{eranges[1,ic]:.1f}")
plt.xlim(0,200)
plt.ylim(0,20)
plt.legend()
plt.show()

plt.plot(cmean[:,hbin],cn)
plt.show()

plt.plot(frac)
plt.show()

plt.plot(frac)
plt.xlim(0,50)
plt.show()

# cull checking
repoint = 37 # for voltage
myCull  = UltraCull0.UltraCull0(repoint,energy_ranges)
status = cnt_data[repoint].status
minv=np.minimum(status['rightdeflection_v'], status['leftdeflection_v'])
t0=min(status['shcoarse'])
mask0 = myCull.currentMask['bin_mask']
v_result = myCull.voltage_cull()
mask1 = myCull.currentMask['bin_mask']
ich=1
ii0 = np.nonzero(mask0[ich,:])
ii1 = np.nonzero(mask1[ich,:])
ii1a = np.nonzero(myCull.currentMask['bin_mask'][ich,:])
plt.plot(np.array(myCull.center_spin()['center_time'])-t0, cntSummary[repoint][:,ich])
plt.plot(np.array(myCull.center_spin()['center_time'][ii0])-t0, cntSummary[repoint][ii0,ich][0])
plt.plot(np.array(myCull.center_spin()['center_time'][ii1])-t0, cntSummary[repoint][ii1,ich][0])
plt.plot(np.array(myCull.center_spin()['center_time'][ii1a])-t0, cntSummary[repoint][ii1a,ich][0])
plt.plot(np.array(status['shcoarse'])-t0,minv/3.5)
plt.xlim(12000,18000)
plt.show()


# looking at 30 - iterative std thresholding (looks pretty good)

#initial plot
repoint = 30
fig,axs = plt.subplots(6)
for ic in range(6):
    axs[ic].plot(centerSpins[repoint]['spin_bin'],cntSummary[repoint][:,ic])
plt.show()

mycull = UltraCull0.UltraCull0(repoint,energy_ranges)
cull0 = mycull.statistical_cull(apply=False,n_iter=5)

fig,axs = plt.subplots(6)
for ic in range(6):
    ii = np.nonzero(cull0['mask'][ic,:])[0]
    axs[ic].plot(centerSpins[repoint]['spin_bin'],cntSummary[repoint][:,ic],'r')
    axs[ic].plot(centerSpins[repoint]['spin_bin'][ii],cntSummary[repoint][ii,ic],'b')
plt.show()

#looking at energy channel thresholding -
mycull0 = UltraCull0.UltraCull0(30,energy_ranges)
mycull1 = UltraCull0.UltraCull0(40,energy_ranges)
cull0 = mycull0.high_energy_cull(apply=False)
cull1 = mycull1.high_energy_cull(apply=False)



# plot checking voltage threholds
repoint = 37
vbins = cullData[repoint].lowVoltagespins(3000)
cbins = cullData[repoint].center_spin()
status = cullData[repoint].status
xspin = cullData[repoint].xspin
tbin = cullData[repoint].binTimes
minv=np.minimum(status['rightdeflection_v'], status['leftdeflection_v'])
t0=min(status['shcoarse'])

plt.plot(status['shcoarse']-t0,minv,label='minimum deflector voltage')
for ispin in vbins['spin_index']:
    tspin = [xspin['spin_start_time'][ispin],xspin['spin_start_time'][ispin]+xspin['spin_period'][ispin]]
    plt.plot(np.array(tspin)-t0,[3250,3250],"r")
for ibin in vbins['bin_index']:
    tbins = [tbin[ibin,0],tbin[ibin,1]]
    plt.plot(np.array(tbins)-t0,[3210,3210],'b')
plt.xlim(0,20000)
plt.xlabel('MET')
plt.ylabel('voltage')
plt.show()

plt.plot(status['shcoarse']-t0,minv,label='minimum deflector voltage')
plt.plot(cbins['center_time']-t0,cullData[repoint].get_count_summary()[:,1])
plt.xlim(13000,18000)
plt.show()

# outdated - all channels
fig,axs = plt.subplots(2,3)
mxval = np.ndarray(6)
for repoint in repointings:
    for ichan in range(6):
        print(f"{repoint},{ichan}")
        i = int(np.floor(ichan/3))
        j =int(ichan % 3)
        spinCenter = (cnt_data[repoint].spinbins[:,0]+cnt_data[repoint].spinbins[:,1])*.5
        ii=np.nonzero(np.logical_and(spinCenter<1.e7,spinCenter>spinCenter[0]))[0]
        cdat = cnt_data[repoint].get_count_summary()[ii,ichan]
        axs[i,j].plot(spinCenter[ii],cdat,label=str(repoint))
        mxval[ichan]=max([mxval[ichan],np.median(cdat)*2])
for ichan in range(6):
    i = int(np.floor(ichan / 3))
    j = int(ichan % 3)
    axs[i,j].set_ylim([0,mxval[ichan]])
plt.show()

## demonstration of spin timimg problem
repointing = 37
de = MyUltraFile.L1Bde(repoint).data
xspin = MyUltraFile.L1Bxspin(repoint).data
l1c = MyUltraFile.L1C(repoint).data
status = MyUltraFile.L1Bstatus(repoint).data

ispin = np.max(np.nonzero(xspin['spin_start_time'] < de['de_event_met'][0]))
spindif = xspin['spin_number'][ispin] - de['spin'][0]
print(spindif)

############

spinave=10
n_spinbin = int(len(xspin['spin_number'])/spinave)

spinRange = [np.min(xspin['spin_number']),np.max(xspin['spin_number'])]

cnts = np.ndarray((n_spinbin,len(ebin_start)))
eranges = np.ndarray((2,len(ebin_start)))
spinranges = np.ndarray((2,int((spinRange[1]-spinRange[0])/spinave)))
for ic in range(len(ebin_start)):
    eranges[:,ic] = [l1c_ebins[ebin_start[ic]][0],l1c_ebins[ebin_end[ic]][1]]
    ii0=np.nonzero(np.logical_and(de['energy_spacecraft'] > eranges[0,ic],de['energy_spacecraft'] < eranges[1,ic]))
    spins = de['spin'][ii0]
    for jc in range(int((spinRange[1]-spinRange[0])/spinave)):
        spinranges[:,jc] = [jc*spinave+spinRange[0],(jc+1)*spinave+spinRange[0]]
        jj0 = np.nonzero(np.logical_and(spins >= jc*spinave+spinRange[0], spins < (jc+1)*spinave+spinRange[0]))
        cnts[jc,ic] = len(jj0[0])


energy_ranges = np.transpose(eranges)
cull0 = UltraCull0.UltraCull0(repoint,energy_ranges)
cnts2 = cull0.get_count_summary()

scaledCnts2 = np.zeros_like(cnts)
for ic in range(len(cnts[0,:])):
    scaledCnts2[:,ic] = (cnts[:,ic]-np.mean(cnts[:,ic]))/np.std(cnts[:,ic])

cnt_data = dict()
for repoint in range(35,48):
    cnt_data[repoint] = UltraCull0.UltraCull0(repoint,energy_ranges)

fig,axs = plt.subplots(2,3)
mxval = np.ndarray(6)
for repoint in range(35,51):
    for ichan in range(6):
        i = int(np.floor(ichan/3))
        j =int(ichan % 3)
        spinCenter = (cnt_data[repoint].spinbins[:,0]+cnt_data[repoint].spinbins[:,1])*.5
        ii=np.nonzero(np.logical_and(spinCenter<1.e7,spinCenter>spinCenter[0]))[0]
        cdat = cnt_data[repoint].get_count_summary()[ii,ichan]
        axs[i,j].plot(spinCenter[ii],cdat,label=str(repoint))
        mxval[ichan]=max([mxval[ichan],np.median(cdat)*2])
for ichan in range(6):
    i = int(np.floor(ichan / 3))
    j = int(ichan % 3)
    axs[i,j].set_ylim([0,mxval[ichan]])
plt.show()

ichan=2
mxd=0
for repoint in range(35,48):
    spinCenter = (cnt_data[repoint].spinbins[:, 0] + cnt_data[repoint].spinbins[:, 1]) * .5
    ii = np.nonzero(np.logical_and(spinCenter < 1.e7, spinCenter > spinCenter[0]))[0]
    cdat = cnt_data[repoint].get_count_summary()[ii, ichan]
    plt.plot(spinCenter[ii], cdat, label=str(repoint))
    mxd = max([mxd, np.median(cdat) * 2])
plt.ylim(0,mxd)
plt.show()
