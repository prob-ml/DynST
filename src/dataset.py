import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Mimic3Dataset(Dataset):
    def __init__(
        self
    ):
        f = pd.read_hdf("data/all_hourly_data.h5")