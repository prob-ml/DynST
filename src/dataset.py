import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Mimic3Dataset(Dataset):
    def __init__(self, work_dir, seed, intervention=None):
        fdir = f"{work_dir}/data/preprocessed_{seed}"
        self.f = {}
        for fname in os.listdir(fdir):
            if fname.endswith(".npy"):
                self.f[fname[:-4]] = np.load(
                    f"{fdir}/{fname}", allow_pickle=True
                    )
        self.ix = self.f["patient_index"]
        self.code_lookup = np.insert(self.f["code_lookup"], 0, "pad")
        self.codes = self.f["codes"] + 1
        self.n_codes = len(self.code_lookup)
        self.n_vitals = self.f["vitals"].shape[1]
        self.n_demog = self.f["demog"].shape[1]
        self.pad_value = - 100
        # if supplied, represents treatment (True) or control (False)
        self.intervention = intervention



    def __len__(self):
        return len(self.f["treatment"])

    def __getitem__(self, index):
        item = {}
        j = self.ix[index]
        if self.intervention is None:
            item["treatment"] = self.f["treatment"][index]
        else:
            item["treatment"] = int(self.intervention)
        item["demog"] = self.f["demog"][index]
        item["codes"] = torch.tensor(
            self.pad_bincount(self.f["codes"][self.f["code_index"] == j])
        )
        item["vitals"] = torch.tensor(
            self.f["vitals"][self.f["hourly_index"] == j]
        ).float()
        item["survival"] = torch.tensor(
            self.f["survival"][self.f["hourly_index"] == j]
        )
        return item

    def pad_bincount(self, records):
        # get counts of each cod
        records = np.bincount(records)
        # pad each vector to length T, all possible codes
        padded = np.zeros(self.n_codes)
        padded[: len(records)] = records
        return torch.from_numpy(padded).float()

def padded_collate(batch, pad_index, causal=False):
    res = {}
    treatment = torch.tensor(np.array([d["treatment"] for d in batch]))
    demog = torch.tensor(np.array([d["demog"] for d in batch])).float()
    if causal:
        res["treatment"] = torch.tensor(np.array([d["treatment"] for d in batch]))
        res["static"] = torch.tensor(np.array([d["demog"] for d in batch])).float()
    else:
        res["static"] = torch.cat([demog, treatment.unsqueeze(1)], 1)
    res["codes"] = torch.stack([d["codes"] for d in batch])
    res["vitals"] = pad_sequence(
        [d["vitals"] for d in batch], batch_first=True, padding_value=pad_index
    )
    res["survival"] = pad_sequence(
        [d["survival"] for d in batch], batch_first=True, padding_value=pad_index
    )
    return res