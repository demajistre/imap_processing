import cdflib
import numpy as np
import pandas as pd

froot = "/Users/demajr1/files/imap_ultra/validation/2025Feb/"
l1a_file = cdflib.CDF(froot+"imap_ultra_l1a_45sensor-de_20240207_v001.cdf")
l1b_file = cdflib.CDF(froot+"imap_ultra_l1b_45sensor-de_20240207_v001.cdf")
ext_file = cdflib.CDF(froot+"imap_ultra_l1b_45sensor-extendedspin_20240207_v001.cdf")
valDat = pd.read_csv(froot+"ultra45_raw_sc_ultrarawimg_withFSWcalcs_FM45_40P_Phi28p5_BeamCal_LinearScan_phi2850_theta-000_20240207T102740.csv")

#-----
l1a_vars = l1a_file.cdf_info().zVariables
l1b_vars = l1b_file.cdf_info().zVariables
ext_vars = ext_file.cdf_info().zVariables

# copying into tracking spreadsheet
colName = [name.upper() for name in valDat.keys().astype(str)]
l1a_list =list(map(lambda x:x.upper(),l1a_vars))
l1b_list =list(map(lambda x:x.upper(),l1b_vars))
ext_list =list(map(lambda x:x.upper(),ext_vars))

for col in valDat.keys().astype(str):
    print(col)

for var in ext_vars:
    print(var)

#work out spin phase
rpm = ext_file['spin_rate'][0]
degSec = rpm * 360/60

# see if we can reproduce size of L1A (assuming 'epoch' error
il1 = list()
il1.append(0)
oldphase = valDat["PhaseAngle"][0]
for i in np.arange(1, len(valDat)):
    newphase = valDat["PhaseAngle"][i]
    if newphase < 0 or newphase != oldphase:
        il1.append(i)
    else:
        print(newphase)
    oldphase = newphase

imiss = list()
for i in np.arange(len(l1a_file["SHCOARSE"])):
    if valDat["MET"][il1[i]] != l1a_file["SHCOARSE"][i]:
        imiss.append(i)

## lets find the missing data - find number of records in each MET
valMet = set(valDat["MET"])
l1aMet = set(l1a_file["SHCOARSE"])
len(valMet.difference(l1aMet))  # this is indeed zero - the METS are the same.
# ID the METs that don't match
ival=0
il1 =0
found_met = list()
for met in valMet:
    for l1a_met in l1a_file["SHCOARSE"]:
        if (met ==l1a_met):
            il1=il1+1
    for val_met in valDat["MET"]:
        if (met == val_met):
            ival=ival+1
    if il1 != ival :
        found_met.append([met,ival,il1])
    il1=0
    ival=0
found_met = np.array(found_met)
ndiff0 = np.sum(found_met[:,1] - found_met[:,2])
ndiff1 = len(valDat) - len(l1a_file["SHCOARSE"]) #this checks out
#  now just check if it is the repeated phase angles
nval_uniq=0
for i in range(1,len(valDat)):
    if valDat["MET"][i] != valDat["MET"][i-1]:
        nval_uniq=nval_uniq+1
    else:
        if valDat["PhaseAngle"][i] != valDat["PhaseAngle"][i-1]:
            nval_uniq=nval_uniq+1
# still 1 off
found_uniq_phase =np.zeros_like(found_met[:,[0,1]])
for ic in range(len(found_met[:,0])):
    phase = set()
    for jc in range(len(valDat)):
        if valDat["MET"][jc] == found_met[ic,0]:
            phase.add(valDat["PhaseAngle"][jc])
    found_uniq_phase[ic,0] = found_met[ic,0]
    found_uniq_phase[ic, 1] = len(phase)

is_zero = np.sum(found_met[:,2]-found_uniq_phase[:,1]) # yep - thats the check

# again - try to build indexes to match val to L1A

l1i = [0]
ind=1
for ic in range(1,len(valDat["MET"])):
    if valDat["MET"][ic] == l1a_file["SHCOARSE"][ind] and (valDat["PhaseAngle"][ic] == l1a_file["PHASE_ANGLE"][ind] or valDat["PhaseAngle"][ic] <1):
        l1i.append(ind)
        ind=ind+1
