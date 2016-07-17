
import numpy as np

class KFold(object):
	'''
	Takes in an unfitted sklearn model, an error function and a labelled data set
	and uses it to perform kfold validation
	'''

	def __init__(self, model, num_folds, error_func):
		'''
		Input: number of folds to use
			   an unfitted sklearn model
			   an error function for comparing predictions to results
		'''
		self.n_folds = num_folds
		self.model = model
		self.error_func = error_func
		self.errors = []]

	def fit(self, X, y):
		'''
		Input: numpy feature array and labels
		Output: self
		'''

		fold_indeces = np.split(np.random.shuffle(range(len(y))), self.n_folds)
		for index in fold_indeces:
			X_train, X_test = self._split_on_index(index,X)
			y_train, y_test = self._split_on_index(index,y)
			self.model.fit(X_train,y_train)
			errors.append(self.error_func(self.model.predict(), y_test))
		return self

	def _split_on_index(self, index, arr):
		everything_else = np.array(set(range(len(arr))) - set(index))
		return a[everything_else], a[index]
