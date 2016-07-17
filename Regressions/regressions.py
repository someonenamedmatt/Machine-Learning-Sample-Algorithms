import numpy as np
from gradient_descent import GradientAscent

# class Regression(object):
#     '''
#     A generic regression class which has a variety of cost functions
#     '''
#     def __init__(self, type, regularizer,  l = 0.0):
#         '''
#         Input: type of regression (OLS, logit)
#                Regularizer (ridge, lasso, anything else is nothing)
#         Output: None
#         '''
#
#         self.l = l
#         if type == 'OLS':
#             self.cost_func = self._OLS_cost
#
#         if regularizer == 'lasso':
#             self.cost_func = lambda X,y: self.cost_func(X,y) + _lasso_cost()
#
#
#
#     def fit(self, X, y, starting_guess = np.array([0])):
#         if type == 'OLS':
#
#
#
#     def _OLS_cost(self):
#         return lambda X, y: np.sum((self.predict(X)-y)**2)
#
#     def _lasso_cost(self):
#         return self.l * np.sum(self.coeff_**2)
