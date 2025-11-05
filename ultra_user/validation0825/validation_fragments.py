import cdflib
import numpy as np
import matplotlib.pyplot as plt


f_base = '/Users/demajr1/files/imap_ultra/validation/2025Aug/'
f_xspin = f_base+'imap_ultra_l1b_90sensor-extendedspin_20260926_v001.cdf'
f_mask = f_base+'imap_ultra_l1b_90sensor-cullingmask_20260926_v001.cdf'
f_bad = f_base+'imap_ultra_l1b_90sensor-badtimes_20260926_v001.cdf'
#f_xspin = f_base+'imap_ultra_l1b_45sensor-extendedspin_20260926_v001.cdf'
#f_mask = f_base+'imap_ultra_l1b_45sensor-cullingmask_20260926_v001.cdf'
#f_bad = f_base+'imap_ultra_l1b_45sensor-badtimes_20260926_v001.cdf'

xspin = cdflib.CDF(f_xspin)
mask = cdflib.CDF(f_mask)
bad = cdflib.CDF(f_bad)

var_xspin = xspin.cdf_info().zVariables
var_mask = mask.cdf_info().zVariables
var_bad = bad.cdf_info().zVariables

ns = np.size(xspin["spin_number"])
nen=np.size(xspin['energy_bin_geometric_mean'])


plt.plot(xspin["spin_number"],xspin["spin_start_time"],".")
plt.xlabel("spin_number")
plt.ylabel("start time (s)")
plt.show()

plt.plot(xspin["spin_number"],xspin["spin_period"],".")
plt.plot(xspin["spin_number"],60./xspin["spin_rate"],"d")
plt.plot(xspin["spin_number"][1:],xspin["spin_start_time"][1:]-xspin["spin_start_time"][0:-1],"o")
plt.plot(xspin["spin_number"][1:],(xspin["epoch"][1:]-xspin["epoch"][0:-1])/1e9,"x")
plt.xlabel("spin_number")
plt.ylabel("spin rate (rpm)")
plt.show()

plt.plot(xspin["spin_start_time"],xspin["epoch"],".")
plt.xlabel("spin_start_time (s)")
plt.ylabel("epoch (s)")
plt.show()

plt.plot(xspin["spin_number"][1:],xspin["spin_start_time"][1:]-xspin["spin_start_time"][0:-1],"o")
plt.plot(xspin["spin_number"][1:],(xspin["epoch"][1:]-xspin["epoch"][0:-1])/1e9,".")
plt.xlabel("spin_number")
plt.ylabel("epoch (s)")
plt.show()

tsec = (xspin["epoch"]-xspin["epoch"][0])/1e9
plt.plot(tsec,xspin["spin_number"],".")
plt.xlabel("spin_number")
plt.ylabel("epoch - epoch[0]")
plt.show()

plt.plot(tsec[:-1],tsec[1:]-tsec[:-1],".")
plt.show()

plt.plot(xspin["epoch"])
plt.xlabel("cdf record number")
plt.ylabel("raw epoch (ns)")
plt.show()

#########

plt.plot(xspin["spin_number"],xspin["start_pulses_per_spin"],label="start")
plt.plot(xspin["spin_number"],xspin["stop_pulses_per_spin"],label="stop")
plt.plot(xspin["spin_number"],xspin["coin_pulses_per_spin"],label="coin")
plt.legend()
plt.xlabel("spin_number")
plt.ylabel("Pulses per spin")
plt.show()

plt.plot(xspin["spin_number"],xspin["quality_attitude"],label="attitude")
#plt.plot(xspin["spin_number"],xspin["quality_hk"],label="housekeeping")
#plt.plot(xspin["spin_number"],xspin["quality_instruments"],label="instruments")
for ei in range(nen):
    plt.plot(xspin["spin_number"], xspin["quality_ena_rates"][ei],
             label=f"Energy{xspin['energy_bin_geometric_mean'][ei]:.2f}")
plt.legend()
plt.xlabel("spin_number")
plt.ylabel("quality flags")
plt.show()

ic =10
ns = np.size(xspin["spin_number"])
nen=np.size(xspin['energy_bin_geometric_mean'])

# check thresholds
threshCheck = np.ndarray(nen)*0
myThresh = np.ndarray(nen)*0
for ic in range(nen):
    myThresh[ic] = np.max([np.mean(xspin["ena_rates"][ic, :]) + 6 * np.std(xspin["ena_rates"][ic, :], ddof=1),0.2])
    threshCheck[ic] = myThresh[ic] -xspin["ena_rates_threshold"][ic]


ic=16
plt.plot(xspin["ena_rates"][ic, :])
plt.plot(np.full(ns,xspin["ena_rates_threshold"][ic]))
plt.show()

exceedThresh = np.ndarray(nen,dtype=bool)
for ic in range(nen):
    exceedThresh[ic] = np.max(xspin["ena_rates"][ic,:]) >= xspin["ena_rates_threshold"][ic]

for ic in range(nen):
    plt.plot(xspin["spin_number"],xspin["ena_rates"][ic])
plt.show()


### below are for the large number of energy channels most recent only has 3
npan=2
ntrace = int(np.ceil(nen/(npan*npan)))
fig, ax = plt.subplots(npan,npan)
for ic in range(npan):
    for jc in range(npan):
        n = jc+ic*npan
        ei = np.arange(n * ntrace, (n + 1) * ntrace)
        for e in ei:
            if e < nen:
                ax[ic][jc].plot(xspin["spin_number"],xspin["ena_rates"][e],
                        label=f"{xspin['energy_bin_geometric_mean'][e]:.2f}")
        ax[ic][jc].legend(frameon=False,ncol=3,loc='upper center',fontsize='xx-small')
fig.show()

fig, ax = plt.subplots(1)
counts = np.sum(xspin["ena_rates"],axis=1)*15
plt.plot(counts)
plt.xlabel("Energy bin")
plt.ylabel("total counts/bin")
plt.yscale('log')
plt.ylim(.1)
plt.show()

###### culling mask

