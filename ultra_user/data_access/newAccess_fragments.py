import ultra_user.data_access.MyUltraFile as MyUltraFile
import glob
from importlib import reload

rootDir=('/Volumes/ultra/data/from_sdc/imap')

myfile = MyUltraFile.MyUltraFile('l1b', 'de', 298, sensor='90', rootDir=rootDir)

fnames = glob.glob(myfile.template)
