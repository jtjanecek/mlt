import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

from tqdm import tqdm

import logging
logger = logging.getLogger('mlt.classifier')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold

import random
from scipy.stats import zscore

import numpy as np

from models.utils import *

from collections import defaultdict

import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 5})


class Classifier:
	def __init__(self, name, df, outcome, timeout, random_state, n_cores, cv, n_splits, decision_thres, n_bootstraps, bootstrap_sampling_rate):
		self._name = name
		self._outcome = outcome
		self._timeout = timeout
		self._n_cores = n_cores
		self._random_state = random_state
		self._thres = decision_thres
		self._n_bootstraps = n_bootstraps
		self._bootstrap_sampling_rate = bootstrap_sampling_rate

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

		self._master_stats = defaultdict(dict)

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
		plot_importance(self._columns, coef, f'{self._name}_lg_coefficients')

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

		lg_pi = calc_permutation_importance_training(self._lg, self._X_z, self._y, self._n_cores, 100, self._random_state, self._columns)
		lg_pi.to_csv(f'{self._name}_lg_permutation_importance.csv',index=False)
		#plot_importance(self._columns, feature_importance, f'{self._name}_rf_gini_importance')

		## CV 
		logger.info("Running CV ...")
		rf_stats_cv = calc_stats_from_cv(self._rf, self._X, self._y, self._cv, self._thres)
		rf_stats_cv.to_csv(f'{self._name}_rf_cv_stats.csv', index=False)
		plot_roc_auc(rf_stats_cv, f'{self._name}_rf_roc_auc')
		m, l_ci, h_ci = mean_confidence_interval(rf_stats_cv['roc_auc'])
		self._master_stats['rf']['roc_auc_mean'] = m
		self._master_stats['rf']['roc_auc_l_ci'] = l_ci
		self._master_stats['rf']['roc_auc_h_ci'] = h_ci

		lg_stats_cv = calc_stats_from_cv(self._lg, self._X_z, self._y, self._cv, self._thres)
		lg_stats_cv.to_csv(f'{self._name}_lg_cv_stats.csv', index=False)
		plot_roc_auc(lg_stats_cv, f'{self._name}_lg_roc_auc')
		m, l_ci, h_ci = mean_confidence_interval(lg_stats_cv['roc_auc'])
		self._master_stats['lg']['roc_auc_mean'] = m
		self._master_stats['lg']['roc_auc_l_ci'] = l_ci
		self._master_stats['lg']['roc_auc_h_ci'] = h_ci


		## Single feature per model
		logger.info("Running individual feature models ...")
		rf_singles = calc_stats_per_feature(self._rf, self._X, self._y, self._cv, self._thres)
		for i in tqdm(range(len((self._columns))), desc="Plotting rf individual feature models ..."):
			column = self._columns[i]
			rf_singles[i].to_csv(f'{self._name}_rf___stats_{column}.csv')
			plot_roc_auc(rf_singles[i], f'{self._name}_rf___stats_{column}_roc_auc')
			m, l_ci, h_ci = mean_confidence_interval(rf_singles[i]['roc_auc'])
			self._master_stats[f'rf_{column}']['roc_auc_mean'] = m
			self._master_stats[f'rf_{column}']['roc_auc_l_ci'] = l_ci
			self._master_stats[f'rf_{column}']['roc_auc_h_ci'] = h_ci

		lg_singles = calc_stats_per_feature(self._lg, self._X_z, self._y, self._cv, self._thres)
		for i in tqdm(range(len((self._columns))), desc="Plotting lg individual feature models ..."):
			column = self._columns[i]
			lg_singles[i].to_csv(f'{self._name}_lg___stats_{column}.csv')
			plot_roc_auc(lg_singles[i], f'{self._name}_lg___stats_{column}_roc_auc')
			m, l_ci, h_ci = mean_confidence_interval(lg_singles[i]['roc_auc'])
			self._master_stats[f'lg_{column}']['roc_auc_mean'] = m
			self._master_stats[f'lg_{column}']['roc_auc_l_ci'] = l_ci
			self._master_stats[f'lg_{column}']['roc_auc_h_ci'] = h_ci


		## Save master stats
		master_stats = pd.DataFrame(self._master_stats).T
		master_stats = master_stats.sort_values(by=['roc_auc_mean'],ascending=False)
		master_stats.to_csv(f'{self._name}_all_model_stats.csv')
		


		if self._n_bootstraps != -1:
			logger.info("Running Bootstraps ...")

			n_to_sample = int(self._X.shape[0] * self._bootstrap_sampling_rate)

			all_bootstrap_results = []

			for i in tqdm(range(self._n_bootstraps), desc="Executing bootstraps ..."):
				# Get a subset of X and y with replacement
				idxes = random.sample(range(self._X.shape[0]), n_to_sample)

				X = self._X[idxes, :]
				y = self._y[idxes]

				self._rf.fit(self._X,self._y)
				rf_stats_bootstrap = calc_stats_from_cv(self._rf, X, y, self._cv, self._thres)

				all_bootstrap_results.append(rf_stats_bootstrap)

			bootstrap_df = pd.concat(all_bootstrap_results)

			bootstrap_df.to_csv(f'{self._name}_rf_bootstrap_cv_stats.csv', index=False)
			plot_roc_auc(bootstrap_df, f'{self._name}_rf_bootstrap_roc_auc')