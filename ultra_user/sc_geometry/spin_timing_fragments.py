import numpy as np
import spiceypy
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

spinRoot = "data/imap/spice/spin/"
pointingTemplate = "data/imap/spice/repoint/*.repoint"
template = spinRoot + "*.spin"

pFiles = glob.glob(pointingTemplate)
pFiles.sort()
pfile = pFiles[-1]

fileDict = dict()
allFiles = glob.glob(template)
for file in allFiles:
    lastUnder = file.rfind('_')
    froot = file[:lastUnder]
    ver = int(file[lastUnder+1:lastUnder+3])
    if froot in fileDict:
        if fileDict[froot] < ver:
            fileDict[froot] = ver
    else:
        fileDict[froot] = ver

latestFiles = list()
for froot in fileDict.keys():
    latestFiles.append(froot+f"_{fileDict[froot]:02}.spin")




