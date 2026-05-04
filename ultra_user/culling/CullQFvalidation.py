import numpy as np


class CullQFvalidation:
    def __init__(self,cullSummary:dict,repoint:int):
        self.cullData = cullSummary['cullData'][repoint]
        self.sepCull = cullSummary['ecull'][repoint]
        self.statCull = cullSummary['scull'][repoint]
        self.lvCull = cullSummary['vcull'][repoint]
        self.up1Cull = cullSummary['upcull1'][repoint]
        self.up2Cull = cullSummary['upcull2'][repoint]
        self.specCull = cullSummary['speccull'][repoint]
        self.nchan = len(self.cullData.energy_ranges[:,0])
        self.flagVals = 2**(np.arange(self.nchan))
        self.xs = self.cullData.xspin
        self.qfSpinBins = self.xs['spin_number']
        self.SDCqf2cullmask = {'quality_low_voltage':self.lvCull,'quality_upstream_ion_1':self.up1Cull,
                               'quality_upstream_ion_2':self.up2Cull, 'quality_spectral':self.specCull,
                               'quality_statistics':self.statCull, 'quality_high_energy':self.sepCull}

# low voltage is special - not separated by channel
    def lvQFlags(self):
        qfLv = np.zeros_like(self.qfSpinBins)
        maskBins = self.cullData.spinbins
        fullFlag = np.sum(self.flagVals)
        for ic in range(len(maskBins[:, 0])):
            ii = np.nonzero(np.logical_and(self.qfSpinBins >= maskBins[ic, 0], self.qfSpinBins <= maskBins[ic, 1]))[0]
            if np.logical_not(self.lvCull['binMask'][ic]) and len(ii) > 0:
                qfLv[ii] = fullFlag
        return qfLv

    def chanQFlags(self,sdcCull:str):
        if sdcCull == 'quality_low_voltage':
            return self.lvQFlags()
        qfChan = np.zeros_like(self.qfSpinBins)
        mask = self.SDCqf2cullmask[sdcCull]['mask']
        maskBins = self.cullData.spinbins
        for ic in range(len(maskBins[:, 0])):
            ii = np.nonzero(np.logical_and(self.qfSpinBins >= maskBins[ic, 0], self.qfSpinBins <= maskBins[ic, 1]))[0]
            jj = np.nonzero(np.logical_not(mask[:, ic]))[0]
            if (len(ii) * len(jj)) > 0:
                qfChan[ii] = np.sum(self.flagVals[jj])
        return qfChan


