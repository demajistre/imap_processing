import imap_data_access
import os

cfgScript = '/Users/demajr1/files/imap_ultra/SDC/imap_data_setup.sh'

with open(cfgScript) as f:
    for line in f:
        line = line.strip()
        if line.startswith('export IMAP_API_KEY='):
            api_key = line.split('=')[1]
        if line.startswith('export IMAP_DATA_DIR='):
            data_dir = line.split('=')[1]

imap_data_access.config['DATA_DIR']=data_dir
imap_data_access.config['API_KEY']=api_key

#files = imap_data_access.query(instrument='ultra',descriptor='90sensor-de')
files = imap_data_access.query(instrument='ultra',descriptor='90sensor-de')
files.extend(imap_data_access.query(instrument='ultra',descriptor='90sensor-rates'))
files.extend(imap_data_access.query(instrument='ultra',descriptor='90sensor-status'))
files.extend(imap_data_access.query(instrument='ultra',descriptor='90sensor-aux'))

# go for all l1a and l1b
files = imap_data_access.query(instrument='ultra',data_level='l1b')
files.extend(imap_data_access.query(instrument='ultra',data_level='l1a'))
files.extend(imap_data_access.query(instrument='ultra',data_level='l1c'))

# go for all l1b and l1c
files = imap_data_access.query(instrument='ultra',data_level='l1b')
files.extend(imap_data_access.query(instrument='ultra',data_level='l1c'))

ic=0
nf=len(files)
for f in files:
    print(f"{ic} of {nf}:   {f['file_path']}")
    imap_data_access.download(f['file_path'])
    ic=ic+1

