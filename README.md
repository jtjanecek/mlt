# Usage
```
usage: run.py [-h] --type {c,r} --input INPUT --outcome OUTCOME [--out OUT] [--timeout TIMEOUT] [--name NAME] [--random_state RANDOM_STATE] [--n_cores N_CORES]
              [--cv {LeaveOneOut,StratifiedKFold,KFold}] [--cv_splits CV_SPLITS] [--decision_thres DECISION_THRES] [--n_bootstraps N_BOOTSTRAPS]
              [--bootstrap_sampling_rate BOOTSTRAP_SAMPLING_RATE]

MLT

options:
  -h, --help            show this help message and exit
  --type {c,r}          Classification or regression
  --input INPUT         CSV input
  --outcome OUTCOME     Outcome variable in the CSV
  --out OUT             Output directory
  --timeout TIMEOUT     Timeout for AutoML fitting in minutes
  --name NAME           Name of the model to use as prefixes
  --random_state RANDOM_STATE
                        Random state for reproducibility
  --n_cores N_CORES     Number of CPUs to use. Use -1 for all cores
  --cv {LeaveOneOut,StratifiedKFold,KFold}
                        Cross validation metric to use. Default=LeaveOneOut
  --cv_splits CV_SPLITS
                        Number of splits for the CV to use. Not used for LOO CV. Default=5
  --decision_thres DECISION_THRES
                        The decision threshold to use for calculating statistics (not AUC). Default=0.5
  --n_bootstraps N_BOOTSTRAPS
                        The number of bootstraps to perform. Default=-1. If -1, then don't run bootstraps
  --bootstrap_sampling_rate BOOTSTRAP_SAMPLING_RATE
                        The number of bootstraps to perform. Default=0.8
```

# Installing
Here's how to install with anaconda. Use Python 3.12. Working and tested 2024-11-26:
```
conda create -n mlt python=3.12 -y
conda activate mlt
pip install -r requirements.txt

```

# Running without bootstrap
```
(mlt) johnjanecek@Johns-MacBook-Pro mlt_f1 % python run.py --type c --input ./data/cnmci_MDT.csv --outcome class --out ./f1                                                 
Namespace(type='c', input='./data/cnmci_MDT.csv', outcome='class', out='./f1', timeout=10, name=None, random_state=50, n_cores=1, cv='LeaveOneOut', cv_splits=5, decision_thres=0.5, n_bootstraps=-1, bootstrap_sampling_rate=0.8)
2024-11-26 16:09:54,387 mlt | INFO | Initializing ...
2024-11-26 16:09:54,387 mlt | DEBUG | Reading input csv at: ./data/cnmci_MDT.csv ...
2024-11-26 16:09:54,390 mlt | INFO | Found data with shape: (104, 14)
2024-11-26 16:09:54,397 mlt.classifier | INFO | Calculating LG Coefficients ...
2024-11-26 16:09:54,709 mlt.classifier | INFO | Calculating RF Gini Impurity Feature Importance ...
2024-11-26 16:09:54,854 mlt.classifier | INFO | Calculating permutation importance on training ...
2024-11-26 16:09:56,672 mlt.classifier | INFO | Running CV ...
2024-11-26 16:10:01,270 mlt.classifier | INFO | Running individual feature models ...
Running calc_stats_per_feature ...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:51<00:00,  4.00s/it]
Plotting rf individual feature models ...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  7.33it/s]
Running calc_stats_per_feature ...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 16.60it/s]
Plotting lg individual feature models ...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  7.76it/s]
```

# Running with bootstrap
```
(mlt) johnjanecek@Johns-MacBook-Pro mlt_f1 % python run.py --type c --input ./data/cnmci_MDT.csv --outcome class --out ./ci --n_bootstraps 100 --bootstrap_sampling_rate 1.0 
Namespace(type='c', input='./data/cnmci_MDT.csv', outcome='class', out='./ci', timeout=10, name=None, random_state=50, n_cores=1, cv='LeaveOneOut', cv_splits=5, decision_thres=0.5, n_bootstraps='100', bootstrap_sampling_rate='1.0')
2024-11-26 15:49:40,925 mlt | INFO | Initializing ...
2024-11-26 15:49:40,925 mlt | DEBUG | Reading input csv at: ./data/cnmci_MDT.csv ...
2024-11-26 15:49:40,926 mlt | INFO | Found data with shape: (104, 14)
2024-11-26 15:49:40,930 mlt.classifier | INFO | Calculating LG Coefficients ...
2024-11-26 15:49:41,209 mlt.classifier | INFO | Calculating RF Gini Impurity Feature Importance ...
2024-11-26 15:49:41,353 mlt.classifier | INFO | Calculating permutation importance on training ...
2024-11-26 15:49:43,167 mlt.classifier | INFO | Running CV ...
2024-11-26 15:49:47,686 mlt.classifier | INFO | Running individual feature models ...
Running calc_stats_per_feature ...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:53<00:00,  4.09s/it]
Plotting rf individual feature models ...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  7.01it/s]
Running calc_stats_per_feature ...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 15.48it/s]
Plotting lg individual feature models ...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  7.27it/s]
2024-11-26 15:50:45,367 mlt.classifier | INFO | Running Bootstraps ...
Executing bootstraps ...: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:06<00:00,  4.26s/it]
```

# Example singularity run
```
singularity run -B [your_local_output_folder]:/output,[your_local_data_folder]:/data mlt.simg --type c --input /data/[your_csv_in_data_folder] --outcome [your_outcome_variable] --out /output/
```


