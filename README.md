# Usage
```
usage: run.py [-h] --type {c,r} --input INPUT --outcome OUTCOME [--out OUT] [--timeout TIMEOUT] [--name NAME] [--random_state RANDOM_STATE] [--n_cores N_CORES]
              [--cv {LeaveOneOut,StratifiedKFold}] [--cv_splits CV_SPLITS]

MLT

optional arguments:
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
  --cv {LeaveOneOut,StratifiedKFold}
                        Cross validation metric to use. Default=LeaveOneOut
  --cv_splits CV_SPLITS
                        Number of splits for the CV to use. Not used for LOO CV. Default=5
```

# Example singularity run
```
singularity run -B [your_local_output_folder]:/output,[your_local_data_folder]:/data mlt.simg --type c --input /data/[your_csv_in_data_folder] --outcome [your_outcome_variable] --out /output/
```
