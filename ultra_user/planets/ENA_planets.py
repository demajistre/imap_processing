import numpy as np
import spiceypy
import imap_processing.spice.time as spiceTime

class ENA_planets(object):
    def __init__(self,pointing_epoch:float,spice_ID:str='EARTH', radius_km:float=6371.):
        self.pointing_epoch = pointing_epoch
        self.spice_ID = spice_ID
        self.radius_km = radius_km
        self.ePos, lt = spiceypy.spkpos(spice_ID,pointing_epoch, 'IMAP_DPS','NONE','-43')
        self.rotMat = self.RotMat()

    def RotMat(self)->np.ndarray:
        edist = np.sqrt(np.sum(self.ePos ** 2))
        upos = self.ePos / edist
        Xax = upos
        Zax0 = np.cross(Xax, [0, 1, 0])
        Zax = Zax0 / np.sqrt(np.sum(Zax0 ** 2))
        Yax = np.cross(Zax, Xax)
        return np.array([Xax,Yax,Zax])

    def local_uvec(self,de_dps_velocity:np.array)->np.ndarray:
        vde = np.sqrt(np.sum(de_dps_velocity ** 2, 1))
        uv = de_dps_velocity
        for ic in range(3):
            uv[:, ic] = uv[:, ic] / vde

        return self.rotMat@ np.transpose(-uv)

    def distance_from_planet(self,de_velocity:np.array,radii=True):
        uv = self.local_uvec(de_velocity)
        angle = np.transpose(np.arccos(uv[0,:]))
        if radii is True:
            edist = np.sqrt(np.sum(self.ePos ** 2))
            return angle*edist/self.radius_km
        else:
            return angle





