import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.data_access.Valdata as Valdata
import ultra_user.data_access.Val1bdata as Val1bData
import ultra_user.data_access.NimbusData as NimbusData
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import pandas as pd
import glob
import healpy as hp
import imap_processing.spice.time as spiceTime
from importlib import reload
from matplotlib.backends.backend_pdf import PdfPages



# change this to something better
#spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0054.tsc')
#spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')
spiceypy.furnsh('data/imap_val0/spice/sclk/imap_sclk_0070.tsc')
spiceypy.furnsh('data/imap_val0/spice/lsk/naif0012.tls')

###
repoint = 55
#repoint = 31
sensor = '90'
rootDir = 'data/imap'
de = MyUltraFile.L1Bde(repoint, sensor=sensor,rootDir=rootDir).data
xspin = MyUltraFile.L1Bxspin(repoint, sensor=sensor,rootDir=rootDir).data
l1c = MyUltraFile.L1C(repoint, sensor=sensor,rootDir=rootDir).data
status = MyUltraFile.L1Bstatus(repoint, sensor=sensor,rootDir=rootDir).data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

v1bSet = Val1bData.Val1bdata()
v1b = v1bSet.get_eventData(repoint, sensor=sensor)
v1bSpin = v1bSet.get_spinData(repoint)

v1bvars = np.array(v1b.columns)
v1bSpinvars = np.array(v1bSpin.columns)

v1cSet = Valdata.get_validation_set()
v1c = v1cSet['counts'].get_pointing_data(repoint)


nim = NimbusData.NimbusData(file="data/imap/validation/nimbusEvents_ULTRA90.csv").get_data()
nimvars = np.array(nim.columns)


l1c_ebins = np.transpose([l1c['energy_bin_geometric_mean'] - l1c['energy_delta_minus'],
                          l1c['energy_bin_geometric_mean'] + l1c['energy_delta_plus']])

energy_bin =  5
ebin_range = [1, 19]
jj = np.nonzero(np.logical_and(
    np.logical_and(np.logical_and(np.logical_and(np.logical_and(de['energy_spacecraft'] > l1c_ebins[energy_bin,0],
                                                                de['energy_spacecraft'] <= l1c_ebins[energy_bin,1]),
                                                 de['quality_outliers'] == 0), de['quality_scattering'] == 0),
                   de['ebin'] >= ebin_range[0]), de['ebin'] <= ebin_range[1]))[0]
ii = np.nonzero(np.logical_and(v1b['energy_sc'] > l1c_ebins[energy_bin,0],
                               v1b['energy_sc'] <= l1c_ebins[energy_bin,1]))[0]

deID = de['event_id'][jj]
valID = np.int64([int(value,16) for value in v1b['eventID'][ii]])

# check against the JENAS L1c
len(valID) - np.sum(v1c[:,energy_bin])
# check the deIDs against SOC L1C
len(deID) - np.sum(l1c['counts'][0,energy_bin,:]) #this is OVER restricted


commonIDs,deCommonInd,valCommonInd=np.intersect1d(deID,valID,assume_unique=True,return_indices=True)
deNotCommon = np.setdiff1d(deID,commonIDs)
valNotCommon = np.setdiff1d(valID,commonIDs)

dummy,deNotCommonInd,dummy = np.intersect1d(deID,deNotCommon,assume_unique=True,return_indices=True)
dummy,valNotCommonInd,dummy = np.intersect1d(valID,valNotCommon,assume_unique=True,return_indices=True)

# quick check of matching events calculation in both sets. Need to double check with nimbus stuff
plt.plot(de['de_event_met'][jj[deCommonInd]], de['energy_spacecraft'][jj[deCommonInd]])
plt.plot(v1b['met (s)'][ii[valCommonInd]], v1b['energy_sc'][ii[valCommonInd]])
plt.show()

plt.plot(de['de_event_met'][jj[deCommonInd]], de['de_event_met'][jj[deCommonInd]]-v1b['met (s)'][ii[valCommonInd]])
plt.show()

plt.plot(de['de_event_met'][jj[deCommonInd]], de['energy_spacecraft'][jj[deCommonInd]]-v1b['energy_sc'][ii[valCommonInd]])
plt.show()

plt.plot(de['de_event_met'][jj[deCommonInd]], de['energy_spacecraft'][jj[deCommonInd]]/v1b['energy_sc'][ii[valCommonInd]]-1)
plt.show()

## energies of the non-matching events:
plt.plot(de['de_event_met'][jj[deNotCommonInd]], de['energy_spacecraft'][jj[deNotCommonInd]])
plt.plot(v1b['met (s)'][ii[valNotCommonInd]], v1b['energy_sc'][ii[valNotCommonInd]])
plt.show()


##############
## plot of what we're missing
plt.plot(de['de_event_met'][jj[deNotCommonInd]],de['energy_spacecraft'][jj[deNotCommonInd]],'o')
plt.plot(v1b['met (s)'][ii[valNotCommonInd]], v1b['energy_sc'][ii[valNotCommonInd]],'o')
plt.show()

### now get nimbus records
nimID = np.int64([int(value,16) for value in nim['EventID']])
dummy, nimCommonInd, dummy = np.intersect1d(nimID,commonIDs,return_indices=True)
dummy, nimDeNotCommonInd, dummy = np.intersect1d(nimID,deNotCommon,return_indices=True)
dummy, nimValNotCommonInd, dummy = np.intersect1d(nimID,valNotCommon,return_indices=True)

# computed bin vs tlm bin - one example not matching in de dataset
for ic in range(len(nimDeNotCommonInd)):
    print(f"{hex(nimID[nimDeNotCommonInd][ic])},{hex(de['event_id'][jj[deNotCommonInd][ic]])},"
          f"{nim['Bin'][nimDeNotCommonInd[ic]]}, {nim['ComputedBin'][nimDeNotCommonInd[ic]]}, {de['ebin'][jj[deNotCommonInd][ic]]}")

# All events in both DE and Val have same Bin and Computed bin (and SDC ebin) =7
# Events in validation and not SDC have several events where ComputedBin !=  bin (about half)
ibinMatch = np.nonzero((nim['Bin'][nimValNotCommonInd]-nim['ComputedBin'][nimValNotCommonInd] == 0))[0]
ibinNoMatch = np.nonzero((nim['Bin'][nimValNotCommonInd]-nim['ComputedBin'][nimValNotCommonInd] != 0))[0]

print(f"altered bin fraction :"
      f"{len(ibinNoMatch)/len(nimValNotCommonInd)}" )

#plots to find the rest

for nmkey in nim.keys():
    if type(nim[nmkey][0]) != str:
        plt.plot(nim[nmkey][nimCommonInd], '.',label='common')
        plt.plot(nim[nmkey][nimValNotCommonInd[ibinMatch]], '.',label='not common/same bin')
#       plt.plot(nim[nmkey][nimValNotCommonInd], '.',label='not common')
        plt.title(nmkey)
        plt.legend()
        plt.show()
        foo = input("next")

for nmkey in nim.keys():
       if type(nim[nmkey][0]) != str:
            plt.plot(nim[nmkey][nimCommonInd], '.', label='common')
            plt.plot(nim[nmkey][nimValNotCommonInd[ibinMatch]], '.', label='not common/same bin')
            #       plt.plot(nim[nmkey][nimValNotCommonInd], '.',label='not common')
            plt.title(nmkey)
            plt.legend()
            plt.savefig(f"/Users/demajr1/tmp/ultra_jenas_val0/{nmkey}.png")
            plt.show()




#checking more global sdc ebin stuff
# removing ebin restriction
kk = np.nonzero(
    np.logical_and(np.logical_and(np.logical_and(np.logical_and(de['energy_spacecraft'] > l1c_ebins[energy_bin,0],
                                                                de['energy_spacecraft'] <= l1c_ebins[energy_bin,1]),
                                                 de['quality_outliers'] == 0), de['quality_scattering'] == 0),
                   de['ebin'] >= ebin_range[0]))[0]

de2id = de['event_id'][kk]
nim2ids, nim2Ind, de2Ind = np.intersect1d(nimID,de2id,return_indices=True)

plt.plot(nim['MET'][nim2Ind],nim['Bin'][nim2Ind],'.')
plt.plot(nim['MET'][nim2Ind],nim['ComputedBin'][nim2Ind],'.')
plt.plot(de['de_event_met'][kk[de2Ind]],de['ebin'][kk[de2Ind]],'.')
plt.show()

plt.plot(nim['Bin'][nim2Ind],de['ebin'][kk[de2Ind]],'.')
plt.plot(nim['ComputedBin'][nim2Ind],de['ebin'][kk[de2Ind]],'+')
plt.show()

