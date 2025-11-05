import cdflib
import numpy as np
import matplotlib.pyplot as plt

droot = '/Users/demajr1/files/imap_ultra/SDC/imap_data/imap/ultra/'
l1aRoot = droot+'l1a/2025/10/'
l1bRoot = droot+'l1b/2025/10/'
de = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-de_20251019-repoint00022_v003.cdf')
xspin = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-extendedspin_20251019-repoint00022_v004.cdf')
good = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-goodtimes_20251019-repoint00022_v004.cdf')
bad = cdflib.CDF(l1bRoot+'imap_ultra_l1b_90sensor-badtimes_20251019-repoint00022_v002.cdf')


deVars = de.cdf_info().zVariables
xspinVars = xspin.cdf_info().zVariables
goodVars = good.cdf_info().zVariables
badVars = bad.cdf_info().zVariables

# looks like only bad spins are at the start and stop of the pointing
print(bad['spin_number'],min(good['spin_number']),max(good['spin_number']))

#check rate/period consistent
rate_err = np.max(np.abs(good['spin_period'] - 60/good['spin_rate']))
delt = np.max(good['spin_start_time']) - np.min(good['spin_start_time'])+good['spin_period'][-1]

plt.plot(good['spin_period'])
plt.show()

plt.plot(good['quality_attitude'],label='attitude')
for ic in range(5):
    plt.plot(good['quality_ena_rates'][ic],label=f"ena rates {good['energy_bin_geometric_mean'][ic]}")
plt.plot(good['quality_hk'],label='hk')
plt.plot(good['quality_instruments'],label='instruments')
plt.legend()
plt.show()

plt.plot(good['ena_rates'])
plt.show()

plt.plot(good['rejected_events_per_spin'])
plt.show()

plt.plot(good['spin_number'],good['start_pulses_per_spin'],'.',label='starts')
plt.plot(good['spin_number'],good['stop_pulses_per_spin'],'.',label='stops')
plt.plot(good['spin_number'],good['coin_pulses_per_spin'],'.',label='coins')
plt.legend()
plt.xlabel('spin number')
plt.show()

