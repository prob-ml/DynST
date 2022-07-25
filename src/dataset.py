import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Mimic3Dataset(Dataset):
    def __init__(self, work_dir, seed):
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
        item["treatment"] = self.f["treatment"][index]
        item["demog"] = self.f["demog"][index]
        item["codes"] = torch.tensor(
            self.f["codes"][self.f["code_index"] == j]
        )
        item["vitals"] = torch.tensor(
            self.f["vitals"][self.f["vitals_index"] == j]
        )
        item["outcome"] = self.outcomes[index]
        return item

def padded_collate(batch):
    res = {}
    res["treatment"] = torch.tensor([d["treatment"] for d in batch])
    res["demog"] = torch.tensor([d["demog"] for d in batch])
    res["outcome"] = torch.tensor([d["outcome"] for d in batch])


def pad_bincount(records, n_codes):
    # get counts of each cod
    records = np.bincount(records)
    # pad each vector to length T, all possible codes
    padded = np.zeros(n_codes)
    padded[: len(records)] = records
    return torch.from_numpy(padded).float()