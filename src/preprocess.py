import logging

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
import pdb

class Mimic3Pipeline():
    def __init__(
        self, work_dir, length_range=(16,128), min_code_count=100, n_vitals=25, seed=28
        ):
        self.work_dir = work_dir
        self.input = pd.HDFStore(work_dir + "/data/all_hourly_data.h5")
        self.output = h5py.File(work_dir + f"/data/mimic3_preprocessed_{seed}.hdf5", "w")
        self.min_length = length_range[0]
        self.max_length = length_range[1]
        self.min_code_counts = min_code_count
        self.n_vitals = n_vitals
        self.stay_lengths = None
        self.arrays = {}
        # baseline hazard
        self.H0 = 0.001
        # rate of hazard decay
        self.labda = 0.05
        self.seed = seed
        np.random.seed(seed)
        # static coefficients
        self.beta = np.random.uniform(0.5, 1, size=5)
        # dynamic coefficients
        self.gamma = np.random.uniform(-0.3, -0.1, size=4)
        # treatment effect on hazards
        self.alpha = -0.5

    def run(self):
        log.info("Beginning pipeline")
        # build index of patients
        interventions = self.input["interventions"].reset_index()
        stay_lengths = interventions.groupby("subject_id").size()
        self.stay_lengths = stay_lengths[
            (stay_lengths >= self.min_length) & (stay_lengths <= self.max_length)
            ]
        self.stay_lengths.name="stay_length"
        self.arrays["patient_index"] = self.stay_lengths.index.to_numpy()
        self.extract_treatment(interventions)
        self.process_patients_data()
        self.process_codes()
        self.process_vitals()
        self.arrays["hourly_index"] = self.vitals.index.get_level_values(0)
        # generate labels
        self.features = self.semisynth_features()
        df_sim =  self.generate_labels(self.features)
        self.arrays["survival"] = df_sim["corrected_survival"].to_numpy()
        self.arrays["hazards"] = df_sim["hazard"].to_numpy()
        # TODO: log some summary statistics
        self.summary_statistics(df_sim)

        log.info("Writing data")
        for key, arr in self.arrays.items():
            self.output.create_dataset(
                key, data=arr
            )
        df_sim.to_csv(self.work_dir + f"/data/mimic3_df_{self.seed}.csv")
        log.info("Pipeline completed")

        
    def extract_treatment(self, interventions):
        treatment = interventions.groupby("subject_id")["vent"].any().astype(int)
        self.treatment = treatment.to_frame().join(self.stay_lengths, how="right")["vent"]
        self.arrays["treatment"] = self.treatment.to_numpy()


    def process_patients_data(self):
        demog = self.input["patients"]
        demog = demog[["gender", "age"]]
        d = {"F":0, "M":1}
        demog["gender"] = demog["gender"].apply(lambda x: d.get(x)).astype(int)
        demog["age"] = demog["age"].clip(upper=90)
        demog["age"] = (demog["age"] - demog["age"].mean()) / demog["age"].std()
        demog = demog.reset_index().set_index("subject_id")[["gender", "age"]]
        self.demog = demog.join(self.stay_lengths, how="right")
        self.arrays["demog"] = self.demog[["gender", "age"]].to_numpy()


    def process_codes(self):
        log.info("Processing codes")
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
        self.codes = codes
        self.arrays["code_index"] = codes.index.to_numpy()
        self.arrays["code_lookup"], self.arrays["codes"] = np.unique(
            codes["icd9_codes"], return_inverse=True
            )
    

    def process_vitals(self):
        log.info("Processing vitals")
        vitals = self.input["vitals_labs_mean"].droplevel(['hadm_id', 'icustay_id'])
        vitals.columns = vitals.columns.get_level_values(0)
        vitals_list = vitals.notna().sum(0).sort_values(ascending=False).head(self.n_vitals).index
        vitals = vitals[vitals_list]
        vitals = vitals.fillna(method="ffill")
        vitals = vitals.fillna(method="bfill")
        mean = np.mean(vitals, axis=0)
        std = np.std(vitals, axis=0)
        vitals = (vitals - mean) / std
        self.vitals = vitals.join(self.stay_lengths, how="right").drop(columns = "stay_length")
        self.arrays["vitals"] = self.vitals.to_numpy()

    def generate_labels(self, df, treatment_col="vent"):
        df["treated"] = 1
        df["control"] = 0
        df["baseline_hazard"] = self.H0 * np.exp(- self.labda * df.index.get_level_values(1))
        # apply treatment
        # column can be "vent", "control" (all zero), or "treat" (all one)
        df["hazard"] = df["baseline_hazard"] * np.exp(self.alpha * df[treatment_col])
        # get static predictors
        # TODO: fix this
        X = df[["gender", "hypertension", "coronary_ath", "atrial_fib"]].to_numpy()
        df["hazard"] = df["hazard"] * np.exp((X * self.beta).sum(1))
        # get dynamic predictors
        V = df[["hematocrit", "hemoglobin", "platelets", "mean blood pressure"]].to_numpy()
        df["hazard"] = df["hazard"] * np.exp((V * self.gamma).sum(1))
        df["hazard"] = np.clip(df["hazard"], a_min = None, a_max = 0.1)
        # switch from hazards to cumulative survival probabilities
        df["q"] = 1 - df["hazard"]
        df["survival_prob"] = df.groupby("subject_id")["q"].cumprod()
        # bernoulli simulation of survival
        np.random.seed(self.seed)
        df["survives"] = np.random.binomial(1, df["survival_prob"])
        return self.corrected_survival_labels(df)


    def semisynth_features(self):
        df = self.demog[["gender", "stay_length"]].join(self.treatment)
        df["stay_length"] = (df["stay_length"] - df["stay_length"].mean()) / df["stay_length"].std()
        conf_codes = self.codes.copy()
        conf_codes["hypertension"] = (conf_codes["icd9_codes"] == "4019")
        conf_codes["coronary_ath"] = (conf_codes["icd9_codes"] == "41401")
        conf_codes["atrial_fib"] = (conf_codes["icd9_codes"] == "42731")
        conf_codes = conf_codes.groupby(conf_codes.index)[["hypertension", "coronary_ath", "atrial_fib"]].any().astype(int)
        conf_vitals = self.vitals[["hematocrit", "hemoglobin", "platelets", "mean blood pressure"]]
        return df.join(conf_codes).join(conf_vitals)

    @staticmethod
    def corrected_survival_labels(df):
        # identify timestep at which first failure occurs (if applicable)
        first_failure = df.reset_index(level="hours_in")
        first_failure = first_failure[first_failure["survives"] == 0].groupby(level=0).first()
        first_failure = first_failure.set_index("hours_in", append=True)
        first_failure["first_failure"] = True
        first_failure = first_failure["first_failure"]
        # label censored patients
        censored = df.reset_index(level="hours_in")
        censored = (censored["survives"] == 1).groupby(level=0).all()
        censored.name = "censored"
        # combine
        df_sim = df.join(first_failure, how="left")
        df_sim = df_sim.reset_index(level="hours_in").join(censored, how="left").\
            set_index("hours_in", append=True)
        # get corrected survival labels: 1 until first failure, then zero
        df_sim["corrected_survival"] = df_sim["first_failure"]
        df_sim["corrected_survival"] = df_sim.groupby(level=0)["corrected_survival"].bfill()
        df_sim["corrected_survival"] = df_sim["corrected_survival"].fillna(False)
        df_sim["corrected_survival"] = (df_sim["corrected_survival"] | df_sim["censored"]).astype(int)
        df_sim["corrected_survival"] = df_sim["corrected_survival"].mask(df_sim["first_failure"].fillna(False), 0)
        return df_sim

    def summary_statistics(self, df_sim):
        n = df_sim.reset_index()["subject_id"].nunique()
        c = df_sim[df_sim["first_failure"] == True].shape[0]
        tau = 16
        log.info(f"{n:,} total patients")
        log.info(f"{n - c:,} censored ({100*(n - c)/n:.2f} %)")
        lifetimes = df_sim.groupby(level=0)["corrected_survival"].sum().to_numpy()
        treated_ix = df_sim.groupby(level=0)["vent"].any()
        log.info(f"Mean time to censoring or failure: {np.mean(lifetimes):.2f} hours")
        rst = self.rmst(df_sim, tau)
        log.info(f"Mean restricted survival time: {np.mean(rst):.2f} hours, tau = {tau}")
        unadj_ate = rst[treated_ix].mean() - rst[~treated_ix].mean()
        log.info(f"Observed treatment effect: {unadj_ate:.2f} hours")


        df_treated = self.generate_labels(self.features, "treated")
        rmst_treated = np.mean(self.rmst(df_treated, tau))
        df_control = self.generate_labels(self.features, "control")
        rmst_control = np.mean(self.rmst(df_control, tau))
        log.info(f"True treatment effect: {rmst_treated - rmst_control:.2f} hours")

    @staticmethod
    def rmst(df, tau):
        restr = df.groupby(level=0)["corrected_survival"].head(tau)
        rst = restr.groupby(level=0).sum()
        return rst.to_numpy()












