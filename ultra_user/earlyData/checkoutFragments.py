import cdflib
import numpy as np
import matplotlib.pyplot as plt

#droot = '/Users/demajr1/files/imap_ultra/earlydata/'
#de = cdflib.CDF(droot+'imap_ultra_l1b_45sensor-de_20251017-repoint00020_v001.cdf')
#deL1a = cdflib.CDF(droot+'imap_ultra_l1a_45sensor-de_20251017-repoint00020_v001.cdf')
#aux = cdflib.CDF(droot+'imap_ultra_l1a_45sensor-aux_20251017-repoint00020_v001.cdf')
#rates = cdflib.CDF(droot+'imap_ultra_l1a_45sensor-rates_20251017-repoint00020_v001.cdf')
#status = cdflib.CDF(droot+'imap_ultra_l1a_45sensor-status_20251017-repoint00020_v001.cdf')

droot = '/Users/demajr1/files/imap_ultra/SDC/imap_data/imap/ultra/'
l1aRoot = droot+'l1a/2025/10/'
l1bRoot = droot+'l1b/2025/10/'
de = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-de_20251019-repoint00022_v001.cdf')
deL1a= cdflib.CDF(l1aRoot+'imap_ultra_l1a_90sensor-de_20251019-repoint00022_v002.cdf')
aux = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-aux_20251019-repoint00022_v002.cdf')
rates = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-rates_20251019-repoint00022_v002.cdf')
status = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-status_20251019-repoint00022_v002.cdf')


deVars = de.cdf_info().zVariables
de1aVars = deL1a.cdf_info().zVariables
auxVars = aux.cdf_info().zVariables
rateVars = rates.cdf_info().zVariables
statusVars = status.cdf_info().zVariables

delt = (np.max(de['epoch']) - min(de['epoch']))*1.e-9
spins = np.unique(de['spin'])

#plot of counts in each spin
spin_cnts = np.histogram(de['spin'],bins=spins)
plt.plot(spin_cnts[0])
plt.show()

spin_angle = de['phase_angle']/2.

#energy/spin
ie = np.nonzero(de['energy_spacecraft'] > 0)

ie35 = np.nonzero(np.logical_and(de['energy_spacecraft'] > 3, de['energy_spacecraft'] < 5))
spin_hist35=np.histogram(spin_angle[ie35[0]],bins=60)
ie57 = np.nonzero(np.logical_and(de['energy_spacecraft'] > 5, de['energy_spacecraft'] < 7))
spin_hist57=np.histogram(spin_angle[ie57[0]],bins=60)
ie1525 = np.nonzero(np.logical_and(de['energy_spacecraft'] > 15, de['energy_spacecraft'] < 25))
spin_hist1525=np.histogram(spin_angle[ie1525[0]],bins=60)
ie2545 = np.nonzero(np.logical_and(de['energy_spacecraft'] > 25, de['energy_spacecraft'] < 45))
spin_hist2535=np.histogram(spin_angle[ie2545[0]],bins=60)


plt.plot(spin_hist35[0])
plt.plot(spin_hist57[0])
plt.plot(spin_hist1525[0])
plt.show()

smoothwin=100
spin_cnts = np.histogram(de['spin'][ie35[0]],bins=spins)
plt.plot(np.convolve(spin_cnts[0],np.ones(smoothwin)/smoothwin),label='3 -5 kev')
spin_cnts = np.histogram(de['spin'][ie57[0]],bins=spins)
plt.plot(np.convolve(spin_cnts[0],np.ones(smoothwin)/smoothwin),label='5 -7 kev')
spin_cnts = np.histogram(de['spin'][ie1525[0]],bins=spins)
plt.plot(np.convolve(spin_cnts[0],np.ones(smoothwin)/smoothwin),label='15 - 25 kev')
spin_cnts = np.histogram(de['spin'][ie2545[0]],bins=spins)
plt.plot(np.convolve(spin_cnts[0],np.ones(smoothwin)/smoothwin),label='25-45 kev')
plt.legend()
plt.show()

# look at spatial
htheta = np.histogram(de['phi'][ie[0]])

#n_de = len(de["spin"])

#plt.hist(de["spin"], bins=1000)
#plt.show()