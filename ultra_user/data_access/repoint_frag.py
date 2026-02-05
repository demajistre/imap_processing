import numpy as np
import pandas as pd


pointfile = 'data/imap/spice/repoint/imap_2026_029_01.repoint'

foo = pd.read_csv(pointfile)

pointingDates = dict()
for index, row in foo.iterrows():
    pointingDates[row['repoint_id']] = row['repoint_start_utc']



