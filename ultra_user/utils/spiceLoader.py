import requests
import spiceypy
from datetime import datetime
from pathlib import Path


# this uses the SDC API to return the files in the MK. the API is documented here:
# https://imap-processing.readthedocs.io/en/latest/data-access/spice-files.html

class spiceLoader():
    def __init__(self,spiceRoot='data/imap/spice/',dateStart:datetime=None,dateEnd:datetime=None):
        self.spiceRoot = spiceRoot
        if dateStart is None:
            dateStart = datetime.fromisoformat("2025-09-10")
        if dateEnd is None:
            dateEnd = datetime.now()
        self.dateStart = dateStart
        self.dateEnd = dateEnd

        self.fileTypes = ['attitude_history', 'attitude_predict', 'spin', 'spin',
                          'repoint', 'repoint', 'ephemeris_reconstructed', 'ephemeris_nominal',
                          'ephemeris_predicted', 'ephemeris_90days', 'ephemeris_long',
                          'ephemeris_launch', 'planetary_ephemeris', 'planetary_constants',
                          'leapseconds', 'pointing_attitude', 'spacecraft_clock', 'imap_frames',
                          'science_frames', 'metakernel', 'metakernel', 'thruster', 'lagrange_point',
                          'earth_attitude']
        self.url = "https://api.imap-mission.com/metakernel"
        self.extMap={'.tls':'lsk/','.tpc':'pck/','.bpc':'pck/','.tsc':'sclk/','.bsp':'spk/','.bc':'ck/','.tc':'ck/',
                     '.tf':'fk/'}

    def getFileList(self,fileTypes=None,loud=False):
        tdif0 = self.dateStart - datetime.fromisoformat("2000-01-01T12:00:00")
        tdif1 = self.dateEnd - datetime.fromisoformat("2000-01-01T12:00:00")
        etStart = tdif0.total_seconds()
        etEnd = tdif1.total_seconds()
        params = {
            "start_time": etStart,
            "end_time": etEnd}
        if fileTypes is None:
            params["list_files"] = True
        else:
            params["list_files"] = fileTypes
        if loud: print(f"params are {params}")
        response = requests.get(self.url, params=params, timeout=30)
        # Raise an exception for HTTP errors (404, 500, etc.)
        response.raise_for_status()
        if loud: print(f"url {response.url}: {response.headers.get('Content-Type')}")
        files = response.json()
        pathNames = list()
        for file in files:
            ext = Path(file).suffix
            if self.extMap.keys().__contains__(ext):
                pathNames.append(self.spiceRoot+self.extMap[ext]+file)
        return pathNames,response

    def loadSpice(self,fileTypes=None,loud=False):

        # this is approximate (we haven't loaded spice yet - should be good enough)

        files = self.getFileList(fileTypes=fileTypes,loud=loud)
        if loud is True:
            print(files)
        for file in files[0]:
            if loud:
                print(f"Loading {file}")
            if Path(file).is_file():
                ret = spiceypy.furnsh(file)
                if ret is not None:
                    print(f"file load {file} failed")
            else:
                print(f"file {file} does not exist spice directory. Ignored")
        return







