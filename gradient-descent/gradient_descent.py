import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import normalize

class GradientAscent(object):

    def __init__(self, cost, predict_func, fit_intercept=True, scale = True, h = .00001):
        '''
        INPUT: GradientAscent, function
        OUTPUT: None
        Initialize class variables. Takes functions:
        cost: the cost function to be minimized
        predict_func: how the model evaluates the predictions
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.is_scaled = False
        #checks to see if we've scaled X already
        self.mu = None
        self.sigma = None
        self.h = h


    def run(self, X, y, alpha=0.01, num_iterations=10000, step_size = .001):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None
        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        X= self._clean(X)
        self.gradient = _calc_grad(X.shape[1])
        self.coeffs = np.zeros(X.shape[1])
        for i in xrange(num_iterations):
            #import pdb; pdb.set_trace()
            #if i % 100 == 0:
                #print 'cost: ', self.cost(X,y, self.coeffs)
            current_cost = self.cost(X,y, self.coeffs)
            self.coeffs += alpha * self.gradient(X, y, self.coeffs)
            if np.abs(self.cost(X,y, self.coeffs) - current_cost) < step_size:
                #print "Completed in ", i, " iterations"
                break

    def runSGD(self, X, y, alpha=0.01, num_iterations=10000, step_size = .0001):
        '''
        INPUT: Stochastic GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None
        Run the Stochastic gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        X= self._clean(X)
        self.coeffs = np.zeros(X.shape[1])
        for j in xrange(num_iterations):
            for k,i in enumerate(np.random.choice(xrange(len(y)), len(y), replace=False)):
                self.coeffs += alpha * self.gradient(X[i,:], y[i], self.coeffs)
                current_cost = self.cost(X,y, self.coeffs)
                if np.abs(self.cost(X,y, self.coeffs) - current_cost) < step_size:
            #        print "Completed in ", k, " iterations"
                    break

            if np.abs(self.cost(X,y, self.coeffs) - current_cost) < step_size:
            #    print "Completed in ", j, " iterations"
                break

    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)
        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        X= self._clean(X)
        return self.predict_func(X, self.coeffs)

    def _clean(self, X):
        if self.scale:
            if not self.is_scaled:
                self.mu = np.mean(X, axis = 0)
                self.sigma = np.std(X,axis=0)
                self.is_scaled = True
            X = (X.astype(float) - self.mu)/self.sigma
        if self.fit_intercept:
            X = sm.add_constant(X)
        return X


    def _calc_grad(self, n):
        '''
        INPUT: number of dimensions in the cost functions (number of features in X)
        OUTPUT: None
        Defines a function which takes in n variables and outputs the gradient of the cost function at that point
        '''
        h_mat = np.diag([self.h for i in range(n)])
        self.grad = lambda x: [f(x+h_mat[:,i])-f(x)/self.h]
