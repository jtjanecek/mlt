import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import logging
logger = logging.getLogger('mlt.classifier')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneOut


from scipy.stats import zscore

import numpy as np


from autosklearn.classification import AutoSklearnClassifier
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 5})

class Classifier:
	def __init__(self, name, df, outcome, timeout, random_state, n_cores):
		self._name = name
		self._outcome = outcome
		self._timeout = timeout
		self._n_cores = n_cores
		self._random_state = random_state

		self._y = df[outcome].values.flatten()
		df = df.drop([outcome], axis=1)
		self._columns = df.columns
		self._X = df.values

		if len(set(self._y)) != 2:
			raise Exception(f"Found {len(set(self._y))} possible outcomes, but expected 2. Outcomes: {set(self._y)}")

		self._rf_importance()
		self._lg_importance()
		self._auto()


	def _rf_importance(self):
		logger.info("Calculating RF Gini Impurity Feature Importance ...")
		model = RandomForestClassifier(random_state=self._random_state)		
		model.fit(self._X,self._y)
		feature_importance = 100 * (model.feature_importances_ / max(model.feature_importances_))
		pi = permutation_importance(model, self._X, self._y, n_jobs=self._n_cores, n_repeats=100, random_state=self._random_state)

		# Save to CSVs
		rf_out_dict = {'feature': self._columns, 'gini_impurity_importance': feature_importance}
		rf_df = pd.DataFrame(rf_out_dict)

		rf_pi_dict = pd.DataFrame(pi['importances'])
		rf_pi_dict['feature'] = self._columns

		logger.info("Saving ...")
		rf_df.to_csv(f'{self._name}_rf_gini_importance.csv',index=False)
		rf_pi_dict.to_csv(f'{self._name}_rf_permutation_importance.csv',index=False)

		logger.info("Running LLO RF CV ...")
		loo = LeaveOneOut()
		positive_class_idx = list(model.classes_).index(1)
		ytrues = []
		ypreds = []
		for train_idx, test_idx in loo.split(self._X, self._y):
			X_train, X_test = self._X[train_idx], self._X[test_idx]
			y_train, y_test = self._y[train_idx], self._y[test_idx]

			model = RandomForestClassifier(random_state=self._random_state, n_jobs=self._n_cores)
			model.fit(X_train, y_train)
			ytrues.append(y_test.flatten())
			ypreds.append(model.predict_proba(X_test)[:,positive_class_idx].flatten())
		ytrues = np.array(ytrues).flatten()
		ypreds = np.array(ypreds).flatten()
	
		pred_df = pd.DataFrame({'ytrue': ytrues, 'ypred': ypreds})
		pred_df.to_csv(f'{self._name}_rf_loo_preds.csv', index=False)
	
		ypred_binary = [0 if z < 0.5 else 1 for z in ypreds]
		tn, fp, fn, tp = confusion_matrix(ytrues, ypred_binary).ravel()
		auc = roc_auc_score(ytrues, ypreds)
		precision = precision_score(ytrues, ypred_binary)
		recall = recall_score(ytrues, ypred_binary)
		basic_stats_df = pd.DataFrame({'decision_threshold': [.5], 'tn': [tn], 'fp': [fp],
			'fn': [fn], 'tp': [tp], 'roc_auc': [auc], 'precision': [precision],
			'recall': [recall]})
		basic_stats_df.to_csv(f'{self._name}_rf_loo_stats.csv', index=False)

		
	def _lg_importance(self):
		logger.info("Calculating LogisticRegression Coefficient Feature Importance ...")

		max_iter = 1000

		X_z = pd.DataFrame(self._X).apply(zscore).values

		model = LogisticRegression(random_state=self._random_state, n_jobs=self._n_cores, max_iter=max_iter)		
		model.fit(X_z,self._y)

		coef = model.coef_.flatten()
		pi = permutation_importance(model, X_z, self._y, n_jobs=self._n_cores, n_repeats=100, random_state=self._random_state)

		lg_out_dict = {'feature': self._columns, 'coefficient': coef}
		lg_df = pd.DataFrame(lg_out_dict)

		lg_pi_dict = pd.DataFrame(pi['importances'])
		lg_pi_dict['feature'] = self._columns

		logger.info("Saving ...")

		lg_df.to_csv(f'{self._name}_lg_coefficients.csv',index=False)
		lg_pi_dict.to_csv(f'{self._name}_lg_permutation_importance.csv',index=False)

		logger.info("Running LLO LG CV ...")
		loo = LeaveOneOut()
		positive_class_idx = list(model.classes_).index(1)
		ytrues = []
		ypreds = []
		for train_idx, test_idx in loo.split(self._X, self._y):
			X_train, X_test = self._X[train_idx], self._X[test_idx]
			y_train, y_test = self._y[train_idx], self._y[test_idx]
			
			model = LogisticRegression(random_state=self._random_state, n_jobs=self._n_cores, max_iter=max_iter)
			model.fit(X_train, y_train)
			ytrues.append(y_test.flatten())
			ypreds.append(model.predict_proba(X_test)[:,positive_class_idx].flatten())

		ytrues = np.array(ytrues).flatten()
		ypreds = np.array(ypreds).flatten()

		pred_df = pd.DataFrame({'ytrue': ytrues, 'ypred': ypreds})
		pred_df.to_csv(f'{self._name}_lg_loo_preds.csv', index=False)

		ypred_binary = [0 if z < 0.5 else 1 for z in ypreds]
		tn, fp, fn, tp = confusion_matrix(ytrues, ypred_binary).ravel()
		auc = roc_auc_score(ytrues, ypreds)
		precision = precision_score(ytrues, ypred_binary)
		recall = recall_score(ytrues, ypred_binary)
		basic_stats_df = pd.DataFrame({'decision_threshold': [.5], 'tn': [tn], 'fp': [fp],
                        'fn': [fn], 'tp': [tp], 'roc_auc': [auc], 'precision': [precision],
                        'recall': [recall]})
		basic_stats_df.to_csv(f'{self._name}_lg_loo_stats.csv', index=False)


	def _auto(self):
		logger.info("Running auto-sklearn ...")
		automl = AutoSklearnClassifier(
			time_left_for_this_task=120,
			per_run_time_limit=30,
			ensemble_size=1,
			n_jobs = self._n_cores
		)

		automl.fit(self._X, self._y, dataset_name=self._name)
		
