import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import imap_processing.spice.time as spiceTime
from importlib import reload




# change this to something better
spiceypy.furnsh('data/imap/spice/sclk/imap_sclk_0103.tsc')
spiceypy.furnsh('data/imap/spice/lsk/naif0012.tls')

repoint = 32
de = MyUltraFile.L1Bde(repoint,sensor='45').data
xspin = MyUltraFile.L1Bxspin(repoint,sensor='45').data
l1c = MyUltraFile.L1C(repoint,sensor='45').data
status = MyUltraFile.L1Bstatus(repoint,sensor='45').data

devars = de.cdf_info().zVariables
xspinvars = xspin.cdf_info().zVariables
l1cvars = l1c.cdf_info().zVariables
statvars = status.cdf_info().zVariables

# good events
ebin_range = [1, 19]
ii = np.nonzero(np.logical_and(np.logical_and(np.logical_and(de['quality_outliers'] == 0,
                                                  de['quality_scattering'] == 0),
                                   de['ebin'] >= ebin_range[0]),de['ebin'] <= ebin_range[1]))[0]
plt.hist2d(de['phi'][ii],de['theta'][ii],bins=30)
plt.colorbar()
plt.show()
