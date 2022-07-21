import logging

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

class Mimic3Pipeline():
    def __init__(self, length_range=(16,128), min_code_count=100, n_vitals=25):
        self.f = pd.HDFStore("../data/all_hourly_data.h5")
        self.min_length = length_range[0]
        self.max_length = length_range[1]
        self.min_code_counts = min_code_count
        self.n_vitals = n_vitals
        self.stay_lengths = None

    def run(self):
        output_dir = "../data/mimic3_preprocessed.hdf5"
        self.extract_treatment().to_hdf(output_dir, key="treatment", mode="w")
        self.process_patients_data().to_hdf(output_dir, key="demog")
        self.extract_codes().to_hdf(output_dir, key="codes")
        self.process_vitals().to_hdf(output_dir, key="vitals")
        # load semisynth label probabilities
        outcome_probs = pd.read_csv("../data/outcome_probs.csv", index=0)
        outcome_probs.to_hdf(output_dir, key="outcome_probs")


    def extract_treatment(self):
        interventions = self.f["interventions"].reset_index()
        stay_lengths = interventions.groupby("subject_id").size()
        self.stay_lengths = stay_lengths[(stay_lengths >= self.min_length) & (stay_lengths <= self.max_length)]
        self.stay_lengths.name="stay_length"
        treatment = interventions.groupby("subject_id")["vent"].any().astype(int)
        return treatment.to_frame().join(stay_lengths, how="right")

    def process_patients_data(self):
        demog = self.f["patients"]
        demog = demog[["gender", "age"]]
        d = {"F":0, "M":1}
        demog["gender"] = demog["gender"].apply(lambda x: d.get(x))
        demog["age"] = demog["age"].clip(upper=90)
        demog["age"] = (demog["age"] - demog["age"].mean()) / demog["age"].std()
        demog = demog.reset_index().set_index("subject_id")[["gender", "age"]]
        return demog.join(self.stay_lengths, how="right")

    def extract_codes(self):
        codes = self.f["codes"].reset_index()[["subject_id", "icd9_codes"]].drop_duplicates(["subject_id"])
        codes = codes.set_index("subject_id").join(self.stay_lengths, how="right")
        codes = codes.explode("icd9_codes")
        code_counts = codes["icd9_codes"].value_counts()
        code_counts = code_counts[code_counts >= self.min_code_counts]
        code_counts.name = "count"
        code_counts = code_counts.to_frame()
        codes = codes.merge(code_counts, left_on="icd9_codes", right_index=True, how="left")
        codes["icd9_codes"] = codes["icd9_codes"].mask(codes["count"].isna())
        codes["icd9_codes"] = codes["icd9_codes"].fillna("unk")
        return codes

    def process_vitals(self):
        vitals = self.f["vitals_labs_mean"].droplevel(['hadm_id', 'icustay_id'])
        vitals.columns = vitals.columns.get_level_values(0)
        vitals_list = vitals.notna().sum(0).sort_values(ascending=False).head(self.n_vitals).index
        vitals = vitals[vitals_list]
        vitals = vitals.fillna(method="ffill")
        vitals = vitals.fillna(method="bfill")
        mean = np.mean(vitals, axis=0)
        std = np.std(vitals, axis=0)
        vitals = (vitals - mean) / std
        return vitals.join(self.stay_lengths, how="right")