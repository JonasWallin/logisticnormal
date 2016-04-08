'''
Multivariate t distribution for estimating
Created on Apr 5, 2016

@author: jonaswallin
'''


from __future__ import division
import numpy as np
import numpy.random as npr
import scipy.special as sps

class tMd(object):
    
    def __init__(self, data = None):
        
        
        self.mu    = None
        self.Sigma = None
        self.nu    = 1
        
        if data is not None:
            self.set_data(data)
            
            
    def set_data(self, data):
        """
            setting the data
            
            data - (n x d) nunmpy arrray
        """
        
        self.data    = np.zeros_like( data)
        self.data[:] = data[:]
        self.mu    = np.zeros(data.shape[1])
        self.Sigma = np.zeros((data.shape[1], data.shape[1]))
        
    def loglik(self, mu = None, L = None, nu = None):
        """
            comupting the loglikelihood
            mu - the mean 
            L  - cholesky factor of precision matrix
            nu - degrees of freedom
        """
        if mu is None:
            mu  = self.mu
         
        if L is None:
            L = self.L
        
        if nu is None:
            nu   = self.nu
        
        if self.data is None:
            raise Exception('need to set data')
        
        
        d = self.data.shape[1]
        n = self.data.shape[0]
        loglik = sps.gammaln((nu + d) * 0.5) - sps.gammaln(0.5* nu) - 0.5 * d * np.log(nu)
        loglik +=  np.sum(np.log(np.diag(L)))
        loglik *= n
        
        res = np.dot(self.data - mu, L.T)
        loglik -= 0.5* (nu + d) * np.sum( np.log(1  + np.sum(res**2,1) / nu))
        return loglik
    
    def set_param(self, mu = None, L = None, nu = None):
        """
            setting the parameters
        """
        self.mu  = np.zeros_like(mu)
        self.mu[:] = mu[:]
        self.mu = self.mu.flatten()
        self.L   = np.zeros_like(L)
        self.L[:] = L[:]
        if nu is not None:
            self.nu  = nu
        
    def sample(self, n):
        """
            sampling from the prior distribution
        """
        d = self.mu.shape[0]
        X = np.zeros((n, d))
        for i in range(n):
            g = self.nu / npr.gamma(shape = self.nu)
            Xt = self.mu + np.sqrt(g) * np.linalg.solve(self.L,  npr.randn(d))
            X[i, :] = Xt
        return X
        

    
    def f_lik(self, theta):
        """
            function for minizing the loglikelihood
        """
        d = self.data.shape[1]
        mu = theta[:d]
        L = np.diag(np.exp(theta[d:(2*d)]))
        L[np.tril_indices(d, k = -1)] = theta[(2*d):((2*d) + d*(d -1)/2 )]
        l = - self.loglik(mu, L)
        return l