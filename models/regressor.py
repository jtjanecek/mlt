import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import logging
logger = logging.getLogger('mlt.classifier')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


from scipy.stats import zscore

import numpy as np

from models.utils import *

from collections import defaultdict

import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 5})


class Regressor:
	def __init__(self, name, df, outcome, timeout, random_state, n_cores, cv, n_splits):
		self._name = name
		self._outcome = outcome
		self._timeout = timeout
		self._n_cores = n_cores
		self._random_state = random_state

		if cv != 'LeaveOneOut':
			self._cv = eval(f"{cv}(shuffle=True,n_splits={n_splits},random_state={random_state})")
		else:
			self._cv = eval(f"{cv}()")
		print(self._cv)

		self._y = df[outcome].values.flatten()
		df = df.drop([outcome], axis=1)
		self._columns = df.columns
		self._X = df.values
		self._X_z = pd.DataFrame(self._X).apply(zscore).values

		self._master_stats = defaultdict(dict)

		self._rf = RandomForestRegressor(random_state=self._random_state, n_jobs=self._n_cores)

		self.run_models()

	def run_models(self):
		### RF Gini impurity importance
		logger.info("Calculating RF Gini Impurity Feature Importance ...")
		self._rf.fit(self._X,self._y)
		feature_importance = 100 * (self._rf.feature_importances_ / max(self._rf.feature_importances_))
		rf_gi = pd.DataFrame({'feature': self._columns, 'gini_impurity_importance': feature_importance})
		rf_gi.to_csv(f'{self._name}_rf_gini_importance.csv',index=False)
		plot_importance(self._columns, feature_importance, f'{self._name}_rf_gini_importance')

		### Permutation Importance
		logger.info("Calculating permutation importance on training ...")
		rf_pi = calc_permutation_importance_training(self._rf, self._X, self._y, self._n_cores, 100, self._random_state, self._columns)
		rf_pi.to_csv(f'{self._name}_rf_permutation_importance.csv',index=False)

		## CV 
		logger.info("Running CV ...")
		rf_stats_cv = calc_stats_from_cv_reg(self._rf, self._X, self._y, self._cv)
		rf_stats_cv.to_csv(f'{self._name}_rf_cv_stats.csv', index=False)

		## Single feature per model
		logger.info("Running individual feature models ...")
		rf_singles = calc_stats_per_feature_reg(self._rf, self._X, self._y, self._cv)
		for i, column in enumerate(self._columns):
			rf_singles[i].to_csv(f'{self._name}_rf___stats_{column}.csv')



