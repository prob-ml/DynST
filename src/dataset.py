import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Mimic3Dataset(Dataset):
    def __init__(self, work_dir, seed=88):
        self.f = h5py.File(work_dir + "/data/mimic3_preprocessed.hdf5")
        np.random.seed(seed)
        self.outcomes = np.random.binomial(
            n=1, p=self.f["outcome_probs"][:, 0]
        )
        self.ix = self.f["index"]
        self.code_lookup = self.f["code_lookup"]


    def __len__(self):
        return len(self.f["treatment"])

    def __getitem__(self, index):
        item = {}
        j = self.ix[index]
        item["treatment"] = torch.tensor(self.f["treatment"][j])
        item["demog"] = torch.tensor(self.f["demog"][j])
        item["codes"] = torch.tensor(
            self.f["codes"][self.f["code_index"] == j]
        )
        item["vitals"] = torch.tensor(
            self.f["vitals"][self.f["vitals_index"] == j]
        )
        item["outcome"] = torch.tensor(self.outcomes[j])
        return item