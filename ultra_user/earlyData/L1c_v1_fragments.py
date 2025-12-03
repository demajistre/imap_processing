import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import ultra_user.data_access.MyUltraFile as MyUltraFile

l1c90all = MyUltraFile.L1C(24)
l1c90 = l1c90all.data
l1c45 = MyUltraFile.L1C(24, sensor='45').data
de90 = MyUltraFile.MyUltraFile('l1b', 'de', 24).data
de45 = MyUltraFile.MyUltraFile('l1b', 'de', 24, sensor='45').data

l1cVars = l1c90.cdf_info().zVariables


## check velocity directions (this looks like its been fixed since the last run)
fig,axs = plt.subplots(4,6)
for irow in range(4):
    for icol in range(6):
        plt.axes(axs[irow,icol])
        n=irow*6+icol
        hp.mollview(l1c45['exposure_factor'][0,n,:],hold=True,title=f"{l1c45['energy_bin_geometric_mean'][n]:5.1f}")
plt.show()

fig,axs = plt.subplots(4,6)
for irow in range(4):
    for icol in range(6):
        plt.axes(axs[irow,icol])
        n=irow*6+icol
        hp.mollview(l1c45['counts'][0,n,:],hold=True,title=f"{l1c45['energy_bin_geometric_mean'][n]:5.1f}",max=1)
plt.show()


##
plt.plot(l1c45['energy_bin_geometric_mean'],np.sum(l1c45['counts'][0,:,:],axis=1),"o--b",label='45')
plt.plot(l1c90['energy_bin_geometric_mean'],np.sum(l1c90['counts'][0,:,:],axis=1),"o--r",label='90')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mean channel energy')
plt.ylabel('Total Counts over pointing')
plt.title(l1c90all.fileName.split('/')[-1])
plt.legend()
plt.show()


fig,(ax1,ax2)= plt.subplots(2)
ax1.plot(l1c90['exposure_factor'][0,:,:])
ax1.set_title('90')
ax2.plot(l1c45['exposure_factor'][0,:,:])
ax2.set_title('45')
for ax in axs.flat:
    ax.set(xlabel='Energy Channel', ylabel='exposure factor')
plt.show()



fig,(ax1,ax2)= plt.subplots(2)
ax1.plot(l1c90['sensitivity'])
ax1.set_title('90')
ax2.plot(l1c45['sensitivity'])
ax2.set_title('45')
for ax in axs.flat:
    ax.set(xlabel='Energy Channel', ylabel='sensitivity')
plt.show()
