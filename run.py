import logging
logger = logging.getLogger('mlt')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s | %(levelname)s | %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

import pandas as pd
import os
import argparse

from models.classifier import Classifier
from models.regressor import Regressor
from models.profiler import Profiler

parser = argparse.ArgumentParser(description='MLT')
parser.add_argument('--type', help='Classification or regression', choices=['c','r'], required=True)
parser.add_argument('--input', help='CSV input', required=True)
parser.add_argument('--outcome', help='Outcome variable in the CSV', required=True)
parser.add_argument('--out', help='Output directory', default='output')
parser.add_argument('--timeout', help='Timeout for AutoML fitting in minutes', default=10)
parser.add_argument('--name', help='Name of the model to use as prefixes', default=None)
parser.add_argument('--random_state', help='Random state for reproducibility', default=50)
parser.add_argument('--n_cores', help='Number of CPUs to use. Use -1 for all cores', default=1)
parser.add_argument('--cv', help='Cross validation metric to use. Default=LeaveOneOut', choices=['LeaveOneOut','StratifiedKFold','KFold'], default="LeaveOneOut")
parser.add_argument('--cv_splits', help='Number of splits for the CV to use. Not used for LOO CV. Default=5', default=5)
parser.add_argument('--decision_thres', help='The decision threshold to use for calculating statistics (not AUC). Default=0.5', default=0.5)


cli_args = parser.parse_args()
print(cli_args)

csv = cli_args.input

logger.info("Initializing ...")

logger.debug(f"Reading input csv at: {csv} ...")
df = pd.read_csv(csv)

logger.info(f"Found data with shape: {df.shape}")

nans = df[df.isna().any(axis=1)]
if nans.shape[0] > 0:
	logger.warning(f" Found {nans.shape[0]} rows with NaN values! Dropping these rows:\n{nans}")

df = df[~df.isna().any(axis=1)]

assert os.path.isdir(cli_args.out)
os.chdir(cli_args.out)

name = os.path.basename(csv.strip(".csv")) if cli_args.name == None else cli_args.name

df.reset_index(drop=True, inplace=True)

numeric_check = df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all()
if not numeric_check:
	raise Exception("Found columns that are non-numeric! Please check your dataset!")

################

#Profiler(name, df)
	
if cli_args.type == 'c':
	Classifier(name, df, cli_args.outcome, cli_args.timeout, cli_args.random_state, cli_args.n_cores, cli_args.cv, cli_args.cv_splits, cli_args.decision_thres)
else:
	Regressor(name, df, cli_args.outcome, cli_args.timeout, cli_args.random_state, cli_args.n_cores, cli_args.cv, cli_args.cv_splits)



