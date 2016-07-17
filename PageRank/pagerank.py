import numpy as np

class PageRank(object):
        '''
        This class reproduces the PageRank algorithm
        '''

        def __init__(self, damp = .85):
            '''
            INPUT: damping factor (discount, how much future clicks are valued)
            OUTPUT: None

            '''
            self.d = damp
            self.states = None
            self.ranks = None

        def fit(self, trans_mat):
            '''
            INPUT: numpy array square transition matrix
            OUTPUT: fitted set of ranks
            '''
            self.states = np.ones(trans_mat.shape) * (1 - self.d)/len(trans_mat)
            Mhat  = self.d * trans_mat + states
            self.ranks =  np.linalg.eig(Mhat.T)[1][:,0]
            return self
            
