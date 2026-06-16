import numpy as np
import matplotlib.pyplot as plt
import cdflib
from importlib import reload

fname = "/Users/demajr1/tmp/imap_ultra_l2_u90-enaflippedthetasixmo-h-hf-nsp-full-hae-6deg-custom-mapper_20251115_v000.cdf"

d = cdflib.CDF(fname)

l2vars = d.cdf_info().zVariables

ie=1

counts = d['counts'][0,ie,:,:]
intensity = d['ena_intensity'][0,ie,:,:]
intensity_err = d['ena_intensity_stat_uncert'][0,ie,:,:]
count_err = np.sqrt(counts)

plt.plot(counts.flatten(),count_err.flatten()/counts.flatten(),'.')
plt.plot(counts.flatten(),intensity_err.flatten()/intensity.flatten(),'.')
plt.show()

plt.plot(count_err.flatten()/counts.flatten(),intensity_err.flatten()/intensity.flatten(),'.')
plt.plot([0,1],[0,1],'r')
plt.show()
