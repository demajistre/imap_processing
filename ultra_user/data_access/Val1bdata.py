import numpy as np
import pandas as pd
import glob
import numpy.typing as npt


class Val1bdata:
    def __init__(self, droot="data/imap/validation/l1b_inputs/"):
        self.droot = droot
        self.templateroots = {"90": droot + "AE-IMAP_ULTRA_90-p", "45": droot + "AE-IMAP_ULTRA_45-p",
                              "spin": droot + "SpinTable-p"}
        spinfiles = glob.glob(self.templateroots["spin"]+"*.csv")
        self.pointings = list()
        for spinfile in spinfiles:
            self.pointings.append(int(spinfile.split("-p")[1].replace(".csv","")))
        self.files90 = dict()
        self.files45 = dict()
        self.filesSpin = dict()
        for p in self.pointings:
            self.files90[p] = self.templateroots["90"]+f"{p}.csv"
            self.files45[p] = self.templateroots["45"]+f"{p}.csv"
            self.filesSpin[p] = self.templateroots["spin"]+f"{p}.csv"

    def get_eventData(self,repoint,sensor:"90"):
        if sensor == "90":
            return pd.read_csv(self.files90[repoint])
        else:
            return pd.read_csv(self.files45[repoint])

    def get_spinData(self,repoint):
        return pd.read_csv(self.filesSpin[repoint])


