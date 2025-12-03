import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import ultra_user.data_access.MyUltraFile as MyUltraFile

l1c90all = MyUltraFile.L1C(38)
l1c90 = l1c90all.data
l1cVars = l1c90.cdf_info().zVariables

# lets find a spatial bin we like - try 4 degrees latitude 8 degrees longitude (planning to take 4 days later)

lonRange=[170,190]
latRange=[-10,10]
erange = [5,100]
ii = np.nonzero((l1c90['longitude'][0] > lonRange[0]) & (l1c90['longitude'][0] < lonRange[1]) &
                (l1c90['latitude'][0] > latRange[0]) & (l1c90['latitude'][0] < latRange[1]))[0]

jj = np.nonzero((l1c90['energy_bin_geometric_mean']> erange[0]) & (l1c90['energy_bin_geometric_mean']<erange[1]))[0]

pointings = np.arange(35,40)
npoint = len(pointings)
ns = len(l1c90['latitude'][0])
ne = len(jj)


counts = np.zeros((npoint,ne,ns))
tex = np.zeros_like(counts)
sens = np.zeros_like(counts)

ic=0
for p in pointings:
    data = MyUltraFile.L1C(p).data
    counts[ic,:,:] = data['counts'][:,jj,:]
    tex[ic,:,:] = data['exposure_factor'][:,jj,:]
    sens[ic,:,:] = data['sensitivity'][jj,:]
    ic=ic+1

csum=np.sum(counts[:,:,ii],axis=2)
tsum=np.sum(tex[:,:,ii],axis=2)
ssum=np.sum(sens[:,:,ii],axis=2)
npsum = np.sum(counts[:,:,ii]*0+1,axis=2)

fluxSum = csum/(tsum*ssum)


ic=0
for p in pointings:
    plt.plot(l1c90['energy_bin_geometric_mean'][jj], fluxSum[ic, :], label=f"pointing {p}")
    ic=ic+1
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


fluxSumB = np.ndarray((npoint,int(ne/2)))
energyB = np.ndarray(int(ne/2))
for ie in np.arange(0,ne/2-1,dtype=int):
    fluxSumB[:,ie] = fluxSum[:,2*ie]+fluxSum[:,2*ie+1]
    energyB[ie] = np.sqrt(l1c90['energy_bin_geometric_mean'][jj[2*ie]]*l1c90['energy_bin_geometric_mean'][jj[2*ie+1]])

ic=0
for p in pointings:
    plt.plot(energyB, fluxSumB[ic, :], label=f"pointing {p}")
    ic=ic+1
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy (keV)')
plt.ylabel('Scaled FLux')
plt.legend()
plt.show()
