import requests
import imap_data_access
from pathlib import Path
import os

cfgScript = '/Users/demajr1/files/imap_ultra/SDC/imap_data_setup.sh'

with open(cfgScript) as f:
    for line in f:
        line = line.strip()
        if line.startswith('export IMAP_API_KEY='):
            api_key = line.split('=')[1]
        if line.startswith('export IMAP_DATA_DIR='):
            data_dir = line.split('=')[1]

imap_data_access.config['DATA_DIR']=Path(data_dir)
imap_data_access.config['API_KEY']=api_key


spice_types =["attitude_history","earth_attitude","ephemeris_nominal","ephemeris_predicted","ephemeris_reconstructed",
              "imap_frames","leapseconds","metakernel","planetary_constants","planetary_ephemeris","pointing_attitude",
              "science_frames","spacecraft_clock"]

url_template = 'https://api.imap-mission.com/spice-query?type=%s&start_time=0&end_time=1000000000'

files=[]
for type in spice_types:
    url=url_template%(type)
    files.extend(requests.get(url).json())

ic=0
for file in files:
    imap_data_access.download(file_path=file['file_name'])
    print(f"file {ic} of {len(files)}:{file['file_name']}")
    ic=ic+1
