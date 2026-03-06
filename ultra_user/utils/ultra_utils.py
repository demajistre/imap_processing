import numpy as np
import glob
import pandas as pd
import imap_processing.spice.time as spTime
import spiceypy as spice


d_min = 400.96 # FSW parameter in funky units
e_conv = 0.104 # constant for conversion of J to Kev

def ctof2kev(ctof):
    return .5*e_conv*(d_min/ctof)**2

def kev2ctof(kev):
    return d_min*np.sqrt(.5*e_conv/kev)

class PointingDef:
    def __init__(self,pointing_dir="data/imap/spice/repoint/"):
        self.pointing_dir = pointing_dir
        self.repoint_file = sorted(glob.glob(pointing_dir+'*.repoint'))[-1]
        self.sclk_start,self.sclk_end,self.repoint = self.convert_table()
        self.repoint_index = dict()
        for ic in range(len(self.repoint)):
            self.repoint_index[self.repoint[ic]]=ic



    def convert_table(self):
        df = pd.read_csv(self.repoint_file)
        sclk_start = np.array(df['repoint_start_sec_sclk'] + df['repoint_start_subsec_sclk'])
        sclk_end = np.array(np.array(df['repoint_end_sec_sclk']+df['repoint_end_subsec_sclk']))
        repoint = np.array(df['repoint_id'],dtype=int)
        return sclk_start, sclk_end,repoint

    def get_pointing_timerange(self,repoint,format=None):
        index = self.repoint_index[repoint]
        sclk_start, sclk_end = self.sclk_start[index], self.sclk_end[index]
        start, end = sclk_start, sclk_end
        if format == 'et':
            start,end = spTime.met_to_ttj2000ns(sclk_start)*1.e-9,spTime.met_to_ttj2000ns(sclk_end)*1.e-9
        if format == 'utc':
            start,end = spTime.met_to_utc(sclk_start),spTime.met_to_utc(sclk_end)
        return start,end

    def get_pointing_for_time(self,time,format="met"):
        met=time
        if format=='et':
            met = spTime.et_to_met(time)
        if format == 'utc':
            met = spTime.et_to_met(spice.utc2et(time))
        index = np.searchsorted(self.sclk_start,met)
        return self.repoint[index]

