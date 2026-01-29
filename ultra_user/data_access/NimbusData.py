import numpy as np
import pandas as pd
import glob
import healpy as hp
import numpy.typing as npt

class NimbusData:
    def __init__(self,file="~/ULTRAData/nimbus/nimbusEvents_ULTRA90.csv"):
        self.file = file
        self.df = pd.read_csv(file)

    def get_data(self) -> pd.DataFrame:
        return self.df