import scipy

import numpy as np
from numpy import interp

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error


import pandas as pd

from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
#font = {'size'   : 1}
mpl.rcParams['figure.dpi'] = 150
#matplotlib.rc('font', **font)
mpl.rcParams.update({'font.size': 5})


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

	ytrues_str = []
	ypreds_str = []

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
		ytrues_str.append(','.join([str(v) for v in cv_ytrue]))
		ypreds_str.append(','.join([str(v) for v in cv_ypred]))

	basic_stats_df = pd.DataFrame({'decision_threshold': [thres]*len(aucs), 'tn': tns, 'fp': fps, 'fn': fns, 'tp': tps, 'roc_auc': aucs, 'precision': precisions, 'recall': recalls, 'ytrues': ytrues_str, 'ypreds': ypreds_str})

	basic_stats_df['ACC'] = (basic_stats_df['tn'] + basic_stats_df['tp']) / (basic_stats_df['tn'] + basic_stats_df['tp'] + basic_stats_df['fn'] + basic_stats_df['fp'])
	basic_stats_df['Sensitivity'] = (basic_stats_df['tp']) / (basic_stats_df['tp'] + basic_stats_df['fn'])
	basic_stats_df['Specificity'] = (basic_stats_df['tn']) / (basic_stats_df['tn'] + basic_stats_df['fp'])
	basic_stats_df['FPR'] = (basic_stats_df['fp']) / (basic_stats_df['fp'] + basic_stats_df['tn'])
	basic_stats_df['Positive Predictive Value'] = (basic_stats_df['tp']) / (basic_stats_df['tp'] + basic_stats_df['fp'])
	basic_stats_df['Negative Predictive Value'] = (basic_stats_df['tn']) / (basic_stats_df['tn'] + basic_stats_df['fn'])

	auc_mean = np.mean(basic_stats_df['roc_auc'].values.flatten())
	auc_sd = np.std(basic_stats_df['roc_auc'].values.flatten())
	samplesize = len(basic_stats_df['roc_auc'].values.flatten())
	basic_stats_df['roc_auc_mean'] = auc_mean
	basic_stats_df['roc_auc_lowci'] = auc_mean - 1.96 * (auc_sd/sqrt(samplesize))
	basic_stats_df['roc_auc_highci'] = auc_mean + 1.96 * (auc_sd/sqrt(samplesize))

	return basic_stats_df

		
def generate_df_from_cv_preds_reg(ytrues, ypreds):

	mae = []
	mae_lci = []
	mae_uci = []
	mse = []
	mse_lci = []
	mse_uci = []
	rmse = []
	rmse_lci = []
	rmse_uci = []

	ytrues_str = []
	ypreds_str = []

	for cv_idx in range(len(ypreds)):
		cv_ytrue = ytrues[cv_idx]
		cv_ypred = ypreds[cv_idx]
	
		this_mae = mean_absolute_error(cv_ytrue, cv_ypred)
		this_mse = mean_squared_error(cv_ytrue, cv_ypred)
		this_rmse = mean_squared_error(cv_ytrue, cv_ypred, squared=False)

		mae.append(this_mae)
		mse.append(this_mse)
		rmse.append(this_rmse)

		ytrues_str.append(','.join([str(v) for v in cv_ytrue]))
		ypreds_str.append(','.join([str(v) for v in cv_ypred]))

	basic_stats_df = pd.DataFrame({
		'mae': mae, 
		'mse': mse, 
		'rmse': rmse, 
		'ytrues': ytrues_str, 
		'ypreds': ypreds_str
	})

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


def calc_stats_from_cv_reg(model, X, y, cv):
	ytrues = []
	ypreds = []
	for train_idx, test_idx in cv.split(X, y):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		model.fit(X_train, y_train)
		ytrues.append(y_test.flatten())
		ypreds.append(model.predict(X_test).flatten())

	if 'LeaveOneOut' in str(cv):
		ytrues = [np.array(ytrues).flatten()]
		ypreds = [np.array(ypreds).flatten()]
	basic_stats_df = generate_df_from_cv_preds_reg(ytrues, ypreds)
	return basic_stats_df




def calc_stats_per_feature(model, X, y, cv, thres):
	results = []
	for i in range(X.shape[1]):
		X_i = X[:,i].reshape(-1, 1)
		results.append(calc_stats_from_cv(model, X_i, y, cv, thres))
	return results

def calc_stats_per_feature_reg(model, X, y, cv):
	results = []
	for i in range(X.shape[1]):
		X_i = X[:,i].reshape(-1, 1)
		results.append(calc_stats_from_cv_reg(model, X_i, y, cv))
	return results


def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h

def plot_importance(columns, importance, path):
	col_imp = list(zip(columns, importance))
	col_imp = sorted(col_imp, key=lambda x: x[1], reverse=True)

	feature_importance = np.array([x[1] for x in col_imp])
	feature_labels = np.array([x[0] for x in col_imp])

	fig, ax = plt.subplots()
	y_pos = np.arange(len(feature_labels))

	ax.barh(y_pos, feature_importance.astype(float), tick_label=feature_labels,align='center')
	#ax.set_yticks(y_pos)
	#ax.set_yticklabels(feature_labels)
	ax.invert_yaxis()
	ax.set_xlabel('Feature Importance')
	ax.set_title('Importance')
	plt.tight_layout()
	plt.savefig(path+'.eps', format='eps')
	plt.savefig(path+'.png', format='png')
	plt.close()

def plot_roc_auc(df, path):
	tprs = []
	aucs = []

	mean_fpr = np.linspace(0, 1, 100)

	for index, row in df.iterrows():
		ytrue = np.array([float(v) for v in row['ytrues'].split(",")])
		ypred = np.array([float(v) for v in row['ypreds'].split(",")])

		fpr, tpr, thresholds = roc_curve(ytrue, ypred)

		aucs.append(auc(fpr, tpr))
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0


	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0

	mean_auc = np.mean(aucs)
	std_auc = np.std(aucs)

	plt.figure()
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

	plt.plot(mean_fpr, mean_tpr, color='b',
		 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		 lw=2, alpha=.8)

	if df.shape[0] != 1: # It's not LOO, so plot the shaded area
		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				 label=r'$\pm$ SD')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate', fontsize=22)
	plt.ylabel('True Positive Rate', fontsize=22)
	plt.title('ROC', fontsize=22)
	plt.legend(loc="lower right", prop={'size': 8})
	plt.tick_params(axis='both', which='major', labelsize=16)

	plt.tight_layout()
	plt.savefig(path+'.eps', format='eps')
	plt.savefig(path+'.png', format='png')
	plt.close()


