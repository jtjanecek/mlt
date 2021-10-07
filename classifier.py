import logging
logger = logging.getLogger('mlt.classifier')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


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
		#self._lg_importance()


	def _rf_importance(self):
		logger.info("Calculating RF Gini Impurity Feature Importance ...")
		model = RandomForestClassifier(random_state=self._random_state)		
		model.fit(self._X,self._y)
		feature_importance = 100 * (model.feature_importances_ / max(model.feature_importances_))
		logger.info(self._columns)
		logger.info(feature_importance)
		pi = permutation_importance(model, self._X, self._y, n_jobs=self._n_cores, n_repeats=100, random_state=self._random_state)['importances_mean']
		logger.info(pi)
		logger.info("Saving ...")
		
		ypred = model.predict(self._X)
		logger.info(accuracy_score(self._y, ypred))
		
	
		
			

'''
1. Fit logistic regression, random forest for feature importance
2. Run logistic regression on each individual feature
3. Fit autoML model
4. Run Permutation importance on final model
'''
