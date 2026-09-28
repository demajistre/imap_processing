import numpy as np
import pandas as pd
import numpy.typing as npt


class CalPointings:
    def __init__(self, calRoot:str="resource/",sensor:str="90",calPointingFile:str=None):
        if calPointingFile is None:
            calPointingFile = calRoot + f"ultra{sensor}_cal_intervals.csv"
        self.calPointingFile = calPointingFile
        cdat = pd.read_csv(calPointingFile)

        cal_period_map = dict()
        iLast = cdat.shape[0] - 1
        for ic in range(iLast):
            cal_period_map[cdat['cal_config'][ic]] = [int(cdat['pointing'][ic] + 1), int(cdat['pointing'][ic + 1] - 1)]
        cal_period_map[cdat['cal_config'][iLast]] = [int(cdat['pointing'][iLast] + 1), int(100000)]
        self.cal_period_map = cal_period_map
        self.cal_priority_map = dict(zip(cdat['cal_config'], cdat['pri_config']))

        thresholds = np.array(cdat[[istr for istr in np.array(cdat.keys()) if "cullThresh" in istr]])
        thresh_map = dict()
        for ic, cfg in enumerate(cdat['cal_config']):
            thresh_map[cfg] = thresholds[ic, :]
        self.thresh_map = thresh_map

        self.voltage_thresh_map = dict(zip(cdat['cal_config'], cdat['deflector_Vthresh']))



    def getCalPointings(self,calPeriod:str)->list:
        return self.cal_period_map[calPeriod]


    def getCalPeriod(self,pointing:int)->str:
        for cp in self.cal_period_map.keys():
            cpRange = self.cal_period_map[cp]
            if cpRange[0] <= pointing <= cpRange[1]:
                return cp
        return "none"

    def calPeriodList(self):
        return list(self.cal_period_map.keys())

    def get_priority_config(self,calPeriod:str)->str:
        return self.cal_priority_map[calPeriod]

    def get_priority_config_from_pointing(self,pointing:int)->str:
        calPeriod = self.getCalPeriod(pointing)
        return self.cal_priority_map[calPeriod]

    def get_cull_thresholds(self,calPeriod:str)-> npt.NDArray[np.floating]:
        return self.thresh_map[calPeriod]

    def get_cull_thresholds_from_pointing(self,pointing:int)-> npt.NDArray[np.floating]:
        calPeriod = self.getCalPeriod(pointing)
        return self.thresh_map[calPeriod]

    def get_delector_Vthresh(self,calPeriod:str) -> float:
        return self.voltage_thresh_map[calPeriod]