import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd

constant_exposure = '~/files/imap_ultra/exposure/ultra_90_dps_exposure.csv'

d = pd.read_csv(constant_exposure)
hp.mollview(d["Exposure Time"])
plt.show()


hp.mollview(
    d["Exposure Time"],
    title='HEALPix Exposure – Energy Bin 0',
    unit='Exposure Time',
    cmap='viridis',
)
plt.show()

plt.plot(d["Exposure Time"],'.')
plt.xlim(75000,76000)
plt.ylim(4.998,5.002)
plt.show()

plt.subplot(1,2,1)
plt.plot(d['Declination (deg)'],d["Exposure Time"],'.')
plt.subplot(1,2,2)
plt.plot(d['Declination (deg)'][20000:35000],d["Exposure Time"][20000:35000],'.')
plt.show()

