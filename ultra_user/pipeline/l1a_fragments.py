# file for prototyping code fragments for l1a processing

import matplotlib.pyplot as plt
import numpy as np


from imap_processing import decom
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import create_dataset, ultra_l1a
from imap_processing.utils import group_by_apid

# stolen from unit test of l1a
ccsds_path = 'imap_processing/tests/ultra/test_data/l0/'
ccsds_file = (ccsds_path +
              'FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50_theta-0.00_20240207T102740.CCSDS')

test_data = ultra_l1a(
        ccsds_file, data_version="001", apid=ULTRA_EVENTS.apid[0]
    )
test_data[0] = test_data[0].sortby("epoch").groupby("epoch").first()
test_data_path = write_cdf(test_data[0])
print("cdf output path:", test_data_path)
#demonstrate loading l1a cdf into xarray
input_xarray_events = load_cdf(test_data_path)
