import cdflib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


droot = 'data/imap/ultra'
l1aRoot = droot+'l1a/2025/10/'
l1bRoot = droot+'l1b/2025/10/'
l1cRoot = droot+'l1c/2025/10/'
repoint_string = '20251021-repoint00024'
head_string = '45'
ver='v001'
de = cdflib.CDF(l1bRoot+'imap_ultra_l1b_'+head_string+'sensor-de_'+repoint_string+'_'+ver+'.cdf')
xspin = cdflib.CDF(l1bRoot+'imap_ultra_l1b_'+head_string+'sensor-extendedspin_'+repoint_string+'_'+ver+'.cdf')
l1c = cdflib.CDF(l1cRoot+'imap_ultra_l1c_'+head_string+'sensor-spacecraftpset_'+repoint_string+'_'+ver+'.cdf')


deVars = de.cdf_info().zVariables
xspinVars = xspin.cdf_info().zVariables
l1cVars = l1c.cdf_info().zVariables

np.shape(l1c['energy_bin_geometric_mean'])
plt.plot(l1c['energy_bin_geometric_mean'],'.')
plt.yscale('log')
plt.show()

plt.plot(l1c['spin_phase_step'],l1c['dead_time_ratio'])
plt.xlabel('Spin Phase')
plt.ylabel('Dead time ratio')
plt.show()

plt.plot(l1c['spin_phase_step'],l1c['dead_time_ratio']/np.mean(l1c['dead_time_ratio'])-1)
plt.xlabel('Spin Phase')
plt.ylabel('Fractional difference from mean DTR')
plt.show()


mapVars = ['counts', 'background_rates', 'exposure_factor', 'sensitivity','efficiency',
           'geometric_function','scatter_theta','scatter_phi']
for var in mapVars:
    print(f"{var}: {np.shape(l1c[var])}")

plt.plot(l1c['energy_bin_geometric_mean'],np.sum(l1c['geometric_function'],axis=1),'.')
plt.xscale('log')
plt.show()


plt.plot(l1c['energy_bin_geometric_mean'],np.sum(l1c['efficiency']*l1c['geometric_function'],axis=1))
plt.plot(l1c['energy_bin_geometric_mean'],np.sum(l1c['sensitivity'],axis=1))
plt.xlabel('mean channel energy')
plt.ylabel('Total Ultra 45 sensitivity')
plt.title('repoint: '+ repoint_string)
plt.xscale('log')
plt.show()


plt.plot(l1c['energy_bin_geometric_mean'],np.sum(l1c['counts'][0,:,:],axis=1),"o--b")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('mean channel energy')
plt.ylabel('Total Counts over pointing')
plt.show()

fig,axs = plt.subplots(4,6)
for irow in range(4):
    for icol in range(6):
        plt.axes(axs[irow,icol])
        n=irow*6+icol
        hp.mollview(l1c['exposure_factor'][0,n,:],hold=True,title=f"{l1c['energy_bin_geometric_mean'][n]:5.1f}")
plt.show()

fig,axs = plt.subplots(4,6)
for irow in range(4):
    for icol in range(6):
        plt.axes(axs[irow,icol])
        n=irow*6+icol
        hp.mollview(l1c['counts'][0,n,:],hold=True,title=f"{l1c['energy_bin_geometric_mean'][n]:5.1f}",max=1)
plt.show()
