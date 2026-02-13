import glob
import numpy as np
import cdflib


class MyUltraFile:
    def __init__(self, dataLevel, descriptor, repointNum, version='Latest', instrument='ultra', sensor='90',
                 rootDir='data/imap',silent=False):
        self.dataLevel = dataLevel
        self.descriptor = descriptor
        self.repointNum = repointNum
        self.version = version
        self.instrument = instrument
        self.sensor = sensor
        self.rootDir = rootDir
        self.silent = silent
        self.template = (f"{self.rootDir}/{self.instrument}/{self.dataLevel}/*/*/"
                         f"imap_{self.instrument}_{self.dataLevel}_{self.sensor}sensor-"
                         f"{self.descriptor}_*-repoint{self.repointNum:05d}*.cdf")
        self.fileName = self.findFile()
        self.data = None
        if self.fileName != '':
            self.data = cdflib.CDF(self.fileName)

    def findFile(self) -> str:
        candidates = self.fileCandidates()
        if len(candidates) == 0:
            if not self.silent:
                print(f"No files found matching the template:\n {self.template}")
            return ""
        fName = self.findVersion(candidates)
        if fName == '':
            if not self.silent:
                print(f"Version{self.version} not found")
        return fName

    def fileCandidates(self) -> list:
        candidates = glob.glob(self.template)
        return candidates

    def findVersion(self, candidates) -> str:
        verNames = list()
        vers = list()
        for candidate in candidates:
            vName = candidate.split('_')[-1].split('.')[0].lstrip('v')
            verNames.append(vName)
            v = int(vName)
            vers.append(v)
        fileName = candidates[np.argsort(vers)[-1]]
        if self.version != 'Latest':
            try:
                iver = verNames.index(self.version)
            except ValueError:
                return ""
            fileName = candidates[iver]
        return fileName


def L1C(repointNum, frame='spacecraft', version='Latest', sensor='90', rootDir='data/imap',
        descriptor=None,silent=False) -> MyUltraFile:
    if descriptor is None:
        if frame == 'spacecraft':
            descriptor = 'spacecraftpset'
        else:
            descriptor = 'heliopset'
    return MyUltraFile('l1c', descriptor, repointNum, version=version, sensor=sensor, rootDir=rootDir,
                       silent=silent)

def L1Bde(repointNum, version='Latest', sensor='90', rootDir='data/imap',silent=False) -> MyUltraFile:
    descriptor = 'de'
    return MyUltraFile('l1b', descriptor, repointNum, version=version, sensor=sensor, rootDir=rootDir,
                       silent=silent)

def L1Ade(repointNum, version='Latest', sensor='90', rootDir='data/imap',silent=False) -> MyUltraFile:
    descriptor = 'de'
    return MyUltraFile('l1a', descriptor, repointNum, version=version, sensor=sensor, rootDir=rootDir,
                       silent=silent)

def L1Bxspin(repointNum, version='Latest', sensor='90', rootDir='data/imap',silent=False) -> MyUltraFile:
    descriptor = 'extendedspin'
    return MyUltraFile('l1b', descriptor, repointNum, version=version, sensor=sensor, rootDir=rootDir,
                       silent=silent)

def L1Bstatus(repointNum, version='Latest', sensor='90', rootDir='data/imap',silent=False)-> MyUltraFile:
    descriptor = 'status'
    return MyUltraFile('l1b', descriptor, repointNum, version=version, sensor=sensor, rootDir=rootDir,
                       silent=silent)