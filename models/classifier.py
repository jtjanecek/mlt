import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import logging
logger = logging.getLogger('mlt.classifier')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


from scipy.stats import zscore

import numpy as np

from models.utils import *

import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 5})


class Classifier:
	def __init__(self, name, df, outcome, timeout, random_state, n_cores, cv, n_splits, decision_thres):
		self._name = name
		self._outcome = outcome
		self._timeout = timeout
		self._n_cores = n_cores
		self._random_state = random_state
		self._thres = decision_thres

		if cv != 'LeaveOneOut':
			self._cv = eval(f"{cv}(shuffle=True,n_splits={n_splits},random_state={random_state})")
		else:
			self._cv = eval(f"{cv}()")

		self._y = df[outcome].values.flatten()
		df = df.drop([outcome], axis=1)
		self._columns = df.columns
		self._X = df.values
		self._X_z = pd.DataFrame(self._X).apply(zscore).values

		if len(set(self._y)) != 2:
			raise Exception(f"Found {len(set(self._y))} possible outcomes, but expected 2. Outcomes: {set(self._y)}")

		self._rf = RandomForestClassifier(random_state=self._random_state, n_jobs=self._n_cores)
		self._lg = LogisticRegression(random_state=self._random_state, n_jobs=self._n_cores,     max_iter=10000)

		self.run_models()

	def run_models(self):
		### LG Coefficients
		logger.info("Calculating LG Coefficients ...")
		self._lg.fit(self._X_z,self._y)
		coef = self._lg.coef_.flatten()
		lg_coef_df = pd.DataFrame({'feature': self._columns, 'coefficient': coef})
		lg_coef_df.to_csv(f'{self._name}_lg_coefficients.csv',index=False)

		### RF Gini impurity importance
		logger.info("Calculating RF Gini Impurity Feature Importance ...")
		self._rf.fit(self._X,self._y)
		feature_importance = 100 * (self._rf.feature_importances_ / max(self._rf.feature_importances_))
		rf_gi = pd.DataFrame({'feature': self._columns, 'gini_impurity_importance': feature_importance})
		rf_gi.to_csv(f'{self._name}_rf_gini_importance.csv',index=False)

		### Permutation Importance
		logger.info("Calculating permutation importance on training ...")
		rf_pi = calc_permutation_importance_training(self._rf, self._X, self._y, self._n_cores, 100, self._random_state, self._columns)
		rf_pi.to_csv(f'{self._name}_rf_permutation_importance.csv',index=False)

		lg_pi = calc_permutation_importance_training(self._lg, self._X_z, self._y, self._n_cores, 100, self._random_state, self._columns)
		lg_pi.to_csv(f'{self._name}_lg_permutation_importance.csv',index=False)


		## CV 
		logger.info("Running CV ...")
		rf_stats_cv = calc_stats_from_cv(self._lg, self._X_z, self._y, self._cv, self._thres)
		rf_stats_cv.to_csv(f'{self._name}_rf_cv_stats.csv', index=False)

		lg_stats_cv = calc_stats_from_cv(self._lg, self._X_z, self._y, self._cv, self._thres)
		lg_stats_cv.to_csv(f'{self._name}_lg_cv_stats.csv', index=False)

		## Single feature per model
		logger.info("Running individual feature models ...")
		rf_singles = calc_stats_per_feature(self._rf, self._X, self._y, self._cv, self._thres)
		for i, column in enumerate(self._columns):
			rf_singles[i].to_csv(f'{self._name}_rf___single_feat_stats_{column}.csv')

		lg_singles = calc_stats_per_feature(self._lg, self._X_z, self._y, self._cv, self._thres)
		for i, column in enumerate(self._columns):
			lg_singles[i].to_csv(f'{self._name}_lg___single_feat_stats_{column}.csv')



