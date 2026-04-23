import glob
import pandas as pd
import numpy as np
from datetime import datetime


class MyPointings:
    def __init__(self,repoint_file=None,template="data/imap/spice/repoint/*.repoint"):
        if repoint_file is None:
            pFiles = glob.glob(template)
            pFiles.sort()
            repoint_file = pFiles[-1]
        self.repoint_file = repoint_file
        self.pointingData = pd.read_csv(repoint_file)
        plen = np.shape(self.pointingData)[0]
        start_time = np.ndarray(plen,dtype=datetime)
        start_sclk = np.ndarray(plen,dtype=float)
        for ic in range(plen):
            start_time[ic] = datetime.fromisoformat(self.pointingData['repoint_start_utc'][ic])
            start_sclk[ic] = (self.pointingData['repoint_start_sec_sclk'][ic] +
                              self.pointingData['repoint_start_subsec_sclk'][ic]*1.e-9)
        self.pointingData['start_time'] = start_time
        self.pointingData['start_sclk'] = start_sclk
        self.pointingData.sort_values(by='start_time')


    def timerange(self,repoint,timeType='utc'):
        ii = np.nonzero(self.pointingData['repoint_id'] == repoint)[0][0]
        if timeType == 'utc':
            return self.pointingData['repoint_start_utc'][ii],self.pointingData['repoint_start_utc'][ii+1]

        elif timeType == 'sclk':
            return self.pointingData['start_sclk'][ii],self.pointingData['start_sclk'][ii+1]

        elif timeType == 'datetime':
            return self.pointingData['start_time'][ii],self.pointingData['start_time'][ii+1]

    def utcPointing(self,utc):

        if type(utc) != datetime:
            utc = datetime.fromisoformat(utc)
        plen = np.size(self.pointingData)
        for ic in range(plen):
            if utc < self.pointingData['start_time'][ic]:
                return self.pointingData['repoint_id'][ic-1]





