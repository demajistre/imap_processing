import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0


#example

repoint = 47
de = MyUltraFile.L1Bde(repoint).data
xspin = MyUltraFile.L1Bxspin(repoint).data
l1c = MyUltraFile.L1C(repoint).data
status = MyUltraFile.L1Bstatus(repoint).data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables


#ucull = UltraCull0.UltraCull0(repoint,energy_ranges, spin_range=1)
voltages = np.minimum(status['leftdeflection_v'],status['rightdeflection_v'])

#vhist, vb = np.histogram(voltages, bins=range(0,4000,10))

#2026-012 to 2026-024,  2026-029 to present
#125 to 136, 141 152

#repointings = range(125,137,1)
repointings = [141,142,143,144,145,146,148,149,150,151,152]
dv = 10
vhist, vb = np.histogram(voltages, bins=range(0,4000,dv))
vhist = vhist*0
for pointing in repointings:
    print(pointing)
    status = MyUltraFile.L1Bstatus(pointing).data
    voltages = np.minimum(status['leftdeflection_v'], status['rightdeflection_v'])
    vh, foo = np.histogram(voltages, bins=range(0,4000,dv))
    vhist = vhist + vh

fig, ax = plt.subplots(2)
fig.suptitle(f"Pointings {np.min(repointings)} - {np.max(repointings)}")
ax[0].plot(vb[0:-1],vhist)
ax[0].set_yscale('log')
ax[0].set_ylim(.1,1.e6)

ax[1].plot(vb[0:-1],vhist,'-+')
ax[1].set_yscale('log')
ax[1].set_ylim(.1,1.e6)
ax[1].set_xlim(3450,3600)
ax[1].set_xlabel('Minimum Deflector Voltage (V)')
plt.show()
