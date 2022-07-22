import logging

import h5py
import numpy as np
import pandas as pd


class Mimic3Pipeline():
    def __init__(
        self, work_dir, length_range=(16,128), min_code_count=100, n_vitals=25
        ):
        self.work_dir = work_dir
        self.input = pd.HDFStore(work_dir + "/data/all_hourly_data.h5")
        self.output = h5py.File(work_dir + "/data/mimic3_preprocessed.hdf5")
        self.min_length = length_range[0]
        self.max_length = length_range[1]
        self.min_code_counts = min_code_count
        self.n_vitals = n_vitals
        self.stay_lengths = None
        self.arrays = {}

    def run(self):
        output_dir = self.work_dir + "/data/mimic3_preprocessed.hdf5"
        # build index of patients
        interventions = self.input["interventions"].reset_index()
        stay_lengths = interventions.groupby("subject_id").size()
        self.stay_lengths = stay_lengths[
            (stay_lengths >= self.min_length) & (stay_lengths <= self.max_length)
            ]
        self.stay_lengths.name="stay_length"
        self.arrays["index"] = self.stay_lengths.index
        # load semisynth label probabilities
        self.arrays["outcome_probs"] = pd.read_csv(
            self.work_dir + "/data/outcome_probs.csv", index_col=0
            )
        assert np.array_equal(
            self.arrays["index"], self.arrays["outcome_probs"].index
            )
        self.extract_treatment(interventions)
        self.process_patients_data()
        self.process_codes()
        self.process_vitals()
        # write out:
        for key, arr in self.arrays.items():
            self.output.create_dataset(
                key, data=arr
            )

        
    def extract_treatment(self, interventions):
        treatment = interventions.groupby("subject_id")["vent"].any().astype(int)
        treatment = treatment.to_frame().join(self.stay_lengths, how="right")["vent"]
        self.arrays["treatment"] = treatment.to_numpy()


    def process_patients_data(self):
        demog = self.input["patients"]
        demog = demog[["gender", "age"]]
        d = {"F":0, "M":1}
        demog["gender"] = demog["gender"].apply(lambda x: d.get(x)).astype(int)
        demog["age"] = demog["age"].clip(upper=90)
        demog["age"] = (demog["age"] - demog["age"].mean()) / demog["age"].std()
        demog = demog.reset_index().set_index("subject_id")[["gender", "age"]]
        demog = demog.join(self.stay_lengths, how="right")
        self.arrays["demog"] = demog.to_numpy()


    def process_codes(self):
        codes = self.input["codes"].reset_index()[["subject_id", "icd9_codes"]].drop_duplicates(["subject_id"])
        codes = codes.set_index("subject_id").join(self.stay_lengths, how="right")
        codes = codes.explode("icd9_codes")
        code_counts = codes["icd9_codes"].value_counts()
        code_counts = code_counts[code_counts >= self.min_code_counts]
        code_counts.name = "count"
        code_counts = code_counts.to_frame()
        codes = codes.merge(code_counts, left_on="icd9_codes", right_index=True, how="left")
        codes["icd9_codes"] = codes["icd9_codes"].mask(codes["count"].isna())
        codes["icd9_codes"] = codes["icd9_codes"].fillna("unk")
        self.arrays["code_index"] = codes.index
        self.arrays["code_lookup"], self.arrays["codes"] = np.unique(
            codes["icd9_codes"], return_inverse=True
            )

    def process_vitals(self):
        vitals = self.input["vitals_labs_mean"].droplevel(['hadm_id', 'icustay_id'])
        vitals.columns = vitals.columns.get_level_values(0)
        vitals_list = vitals.notna().sum(0).sort_values(ascending=False).head(self.n_vitals).index
        vitals = vitals[vitals_list]
        vitals = vitals.fillna(method="ffill")
        vitals = vitals.fillna(method="bfill")
        mean = np.mean(vitals, axis=0)
        std = np.std(vitals, axis=0)
        vitals = (vitals - mean) / std
        vitals = vitals.join(self.stay_lengths, how="right").drop(columns = "stay_length")
        self.arrays["vitals_index"] = vitals.index.get_level_values(0)
        self.arrays["vitals"] = vitals.to_numpy()
