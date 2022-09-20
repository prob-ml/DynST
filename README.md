# DynST
Dynamic Survival Models for Causal Inference with Electronic Health Records

### About

A deep learning model built in Pytorch Lightning. Dependencies are managed through Poetry, configurations through Hydra.

#### Data
Our semi-synthetic data derives from the MIMIC-III Clinical Database (https://physionet.org/content/mimiciii-demo/1.4/). We use the MIMIC-Extract pipeline to preprocess the data (https://github.com/MLforHealth/MIMIC_Extract).

#### How to Use

To generate the semi-synthetic dataset, you should have the MIMIC-Extract file `all_hourly_data.h5` saved in a directory called `data/`. Then run
```
python run.py preprocess.do=True
```
To launch a multirun of DynST sweeping over several hyperparameters, you can enter the following command:

```
python run -m model.d_model=32,64 model.alpha=0,0.1,0.2  
```