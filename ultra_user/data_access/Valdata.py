import numpy as np
import pandas as pd
import glob
import healpy as hp
import numpy.typing as npt

class Valdata:
    def __init__(self, droot="data/imap/validation/dps/", type="counts",sensor="90",frame='DPS-SC',
                 nside=32):
        self.droot = droot
        self.type = type
        self.sensor =sensor
        self.types = {'counts': 'Counts', 'eff': 'EFF', 'exposures': 'Exposures', 'gf': 'GF', 'sens': 'SENS',
                 'intensity': 'Intensity'}
        self.template = droot + type + '/' + self.types[type]+ f"-IMAP_ULTRA_{sensor}-IMAP_{frame}-nside{nside}-ebin*.csv"
        files = glob.glob(self.template)
        self.hasdata = False
        self.emap = dict()
        if len(files) != 0:
            for file in files:
                i = file.rfind("ebin") + 4
                j = file.rfind(".csv")
                ebin = int(file[i:j])
                self.emap[ebin] = file
            nen = len(self.emap.keys())
            npix = hp.nside2npix(nside)
            d0 = pd.read_csv(files[0])
            self.ra = d0['Right Ascension (deg)']
            self.dec = d0['Declination (deg)']
            self.pointings = list()
            for key in d0.keys():
                if key[0] == 'P':
                    self.pointings.append(int(key[1:]))
            self.pdat = dict()
            for pointing in self.pointings:
                self.pdat[pointing] = np.ndarray((npix, nen))

            for ebin in range(nen):
                d = pd.read_csv(self.emap[ebin])
                for p in self.pointings:
                    self.pdat[p][:, ebin] = d[f"P{p}"]
            self.hasdata = True
    def get_known_types(self)->dict[str:str]:
        return self.types

    def get_pointing_data(self,pointing:int)->npt.NDArray:
        if not self.hasdata:
            return np.ndarray(0)
        if not pointing in self.pointings:
            return np.ndarray(0)
        return self.pdat[pointing]

def get_validation_set(droot="data/imap/validation/dps/", type="counts",sensor="90",frame='DPS-SC',
                 nside=32)->dict:
    val_cnts = Valdata(droot=droot, type=type,sensor=sensor,frame=frame,
                 nside=nside)
    types = val_cnts.get_known_types().keys()
    valset = dict()
    for type in types:
        v = Valdata(type=type)
        if v.hasdata:
            valset[type] = v
    return valset


