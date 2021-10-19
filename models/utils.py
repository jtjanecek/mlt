import scipy
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

def calc_permutation_importance_training(model, X, y, n_cores, n_repeats, random_state, columns):
	pi = permutation_importance(model, X, y, n_jobs=n_cores, n_repeats=n_repeats, random_state=random_state)

	perm_dict = pd.DataFrame(pi['importances'])
	perm_dict['feature'] = columns
	return perm_dict
		
def generate_df_from_cv_preds(ytrues, ypreds, thres):
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

		ypred_binary = [0 if z < thres else 1 for z in cv_ypred]
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

	basic_stats_df = pd.DataFrame({'decision_threshold': [thres]*len(aucs), 'tn': tns, 'fp': fps, 'fn': fns, 'tp': tps, 'roc_auc': aucs, 'precision': precisions, 'recall': recalls})

	return basic_stats_df


def calc_stats_from_cv(model, X, y, cv, thres):
	positive_class_idx = list(model.classes_).index(1)
	ytrues = []
	ypreds = []
	for train_idx, test_idx in cv.split(X, y):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		model.fit(X_train, y_train)
		ytrues.append(y_test.flatten())
		ypreds.append(model.predict_proba(X_test)[:,positive_class_idx].flatten())

	if 'LeaveOneOut' in str(cv):
		ytrues = [np.array(ytrues).flatten()]
		ypreds = [np.array(ypreds).flatten()]
	basic_stats_df = generate_df_from_cv_preds(ytrues, ypreds, thres)
	return basic_stats_df


def calc_stats_per_feature(model, X, y, cv, thres):
	results = []
	for i in range(X.shape[1]):
		X_i = X[:,i].reshape(-1, 1)
		results.append(calc_stats_from_cv(model, X_i, y, cv, thres))
	return results


def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h
