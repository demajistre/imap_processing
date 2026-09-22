import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ultra_user.instrumentConfig.calPointings as calPointings

cp = calPointings.CalPointings()
cp.getCalPointings("c1")
cp.getCalPeriod(200)
cp.get_priority_config_from_pointing(200)
cp.calPeriodList()
cp.get_cull_thresholds_from_pointing(200)


cdat = pd.read_csv(cp.calPointingFile)

pri_map = dict(zip(cdat['cal_config'], cdat['pri_config']))

thresholds = np.array(cdat[[istr for istr in np.array(cdat.keys()) if "cullThresh" in istr ]])
thresh_map = dict()
for ic,cfg in enumerate(cdat['cal_config']):
    thresh_map[cfg] = thresholds[ic,:]

