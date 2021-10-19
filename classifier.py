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
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


from scipy.stats import zscore

import numpy as np



import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 5})


def generate_df_from_cv_preds(ytrues, ypreds):
	tns = []
	fps = []
	fns = []
	tps = []
	aucs = []
	precisions = []
	recalls = []
		
	for cv_idx in range(len(ypreds)):
		cv_ytrue = ytrues[cv_idx]
		cv_ypred = ypreds[cv_idx]

		ypred_binary = [0 if z < 0.5 else 1 for z in cv_ypred]
		tn, fp, fn, tp = confusion_matrix(cv_ytrue, ypred_binary).ravel()
		auc = roc_auc_score(cv_ytrue, cv_ypred)
		precision = precision_score(cv_ytrue, ypred_binary)
		recall = recall_score(cv_ytrue, ypred_binary)

		tns.append(tn)
		fps.append(fp)
		fns.append(fn)
		tps.append(tp)
		aucs.append(auc)
		precisions.append(precision)
		recalls.append(recall)

	basic_stats_df = pd.DataFrame({'decision_threshold': [.5]*len(aucs), 'tn': tns, 'fp': fps,
		'fn': fns, 'tp': tps, 'roc_auc': aucs, 'precision': precisions,
		'recall': recalls})

	return basic_stats_df



class Classifier:
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

		self._y = df[outcome].values.flatten()
		df = df.drop([outcome], axis=1)
		self._columns = df.columns
		self._X = df.values

		if len(set(self._y)) != 2:
			raise Exception(f"Found {len(set(self._y))} possible outcomes, but expected 2. Outcomes: {set(self._y)}")

		self._rf_importance()
		self._lg_importance()
		#self._auto()


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

		logger.info("Running RF CV ...")
		positive_class_idx = list(model.classes_).index(1)
		ytrues = []
		ypreds = []
		for train_idx, test_idx in self._cv.split(self._X, self._y):
			X_train, X_test = self._X[train_idx], self._X[test_idx]
			y_train, y_test = self._y[train_idx], self._y[test_idx]

			model = RandomForestClassifier(random_state=self._random_state, n_jobs=self._n_cores)
			model.fit(X_train, y_train)
			ytrues.append(y_test.flatten())
			ypreds.append(model.predict_proba(X_test)[:,positive_class_idx].flatten())

		if 'LeaveOneOut' in str(self._cv):
			ytrues = [np.array(ytrues).flatten()]
			ypreds = [np.array(ypreds).flatten()]
			pred_df = pd.DataFrame({'ytrue': ytrues, 'ypred': ypreds})
			pred_df.to_csv(f'{self._name}_rf_loo_preds.csv', index=False)
	
		basic_stats_df = generate_df_from_cv_preds(ytrues, ypreds)
		basic_stats_df.to_csv(f'{self._name}_rf_cv_stats.csv', index=False)
		
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
		positive_class_idx = list(model.classes_).index(1)
		ytrues = []
		ypreds = []
		for train_idx, test_idx in self._cv.split(self._X, self._y):
			X_train, X_test = self._X[train_idx], self._X[test_idx]
			y_train, y_test = self._y[train_idx], self._y[test_idx]
			
			model = LogisticRegression(random_state=self._random_state, n_jobs=self._n_cores, max_iter=max_iter)
			model.fit(X_train, y_train)
			ytrues.append(y_test.flatten())
			ypreds.append(model.predict_proba(X_test)[:,positive_class_idx].flatten())

		if 'LeaveOneOut' in str(self._cv):
			ytrues = [np.array(ytrues).flatten()]
			ypreds = [np.array(ypreds).flatten()]
			pred_df = pd.DataFrame({'ytrue': ytrues, 'ypred': ypreds})
			pred_df.to_csv(f'{self._name}_lg_loo_preds.csv', index=False)
		
		basic_stats_df = generate_df_from_cv_preds(ytrues, ypreds)
		basic_stats_df.to_csv(f'{self._name}_lg_cv_stats.csv', index=False)

		# Single feature per LG
		tns = []
		fps = []
		fns = []
		tps = []
		aucs = []
		precisions = []
		recalls = []
		columns = self._columns

		for idx, column in enumerate(self._columns):
			this_X = self._X[:,idx]
			ytrues = []
			ypreds = []
			for train_idx, test_idx in self._cv.split(this_X, self._y):
				X_train, X_test = this_X[train_idx].reshape(-1, 1) , this_X[test_idx].reshape(-1, 1) 
				y_train, y_test = self._y[train_idx], self._y[test_idx]
	
				model = LogisticRegression(random_state=self._random_state, n_jobs=self._n_cores, max_iter=max_iter)
				model.fit(X_train, y_train)
				positive_class_idx = list(model.classes_).index(1)
				ypred = model.predict_proba(X_test)[:,positive_class_idx].flatten()
					
				ytrues.append(y_test)
				ypreds.append(ypred)
	
			ytrues = np.array(ytrues).flatten()
			ypreds = np.array(ypreds).flatten()
		
			ypred_binary = [0 if z < 0.5 else 1 for z in ypreds]
			tn, fp, fn, tp = confusion_matrix(ytrues, ypred_binary).ravel()
			auc = roc_auc_score(ytrues, ypreds)
			precision = precision_score(ytrues, ypred_binary)
			recall = recall_score(ytrues, ypred_binary)

			tns.append(tn)
			fps.append(fp)
			fns.append(tn)
			tps.append(tp)
			aucs.append(auc)
			precisions.append(precision)
			recalls.append(recall)
		
		single_model_df = pd.DataFrame({
			'feature': columns,
			'tn': tns,
			'fp': fps,
			'fn': fns,
			'tp': tps,
			'roc_auc': aucs,
			'precision': precisions,
			'recall': recalls
		})

		single_model_df.to_csv(f'{self._name}_lg_cv_model_per_feature.csv', index=False)
