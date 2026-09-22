import numpy as np
import matplotlib.pyplot as plt
import ultra_user.culling.cull_util as cull_util
import ultra_user.culling.UltraCull0 as Ultracull0
import ultra_user.utils.ultra_utils as utils
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.CullQFvalidation as CullVal
import spiceypy
import imap_processing.spice.time as spiceTime
import pickle
import cdflib
import pickle
from scipy.stats import binned_statistic
from scipy.stats import norm
from datetime import datetime
import ultra_user.utils.spiceLoader as spiceLoader

from importlib import reload


spiceLoad = spiceLoader.spiceLoader()
spicefiles = spiceLoad.getFileList()
spiceLoad.loadSpice()

froot = '/Users/demajr1/files/imap_ultra/culling/version2/cullplots0/'

#sensor="90"
sensor="45"
energy_ranges = cull_util.l1c_energy_ranges()
cullList = ['c0','c1','c2','c3','c4']
pointings = dict()
culls=dict()
for cull in cullList:
    p = cull_util.get_pointings_from_calPeriods(cull,sensor=sensor)
    culls[cull] = cull_util.runculls(p, energy_ranges, sensor=sensor, cullPackage="hiEnergy_upstream_spectral_stat_v1")
    pvset = set(culls[cull]['vcull'].keys())
    p_missing = set(p).difference(pvset)
    for mp in p_missing:
        p.remove(mp)
    pointings[cull] = p

for cull in cullList:
    print(cull)
    cull_util.cullplot(culls[cull],pointings[cull],saveFile=froot+f"{sensor}_{cull}.png",addTitle=f" {cull}")
