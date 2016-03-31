'''
	Various simple test for the scaling regression
	Created on Mar 31, 2016

@author: jonaswallin
'''
import unittest
import os
import numpy as np
import numpy.random as npr

from logisticnormal.distribution_cython import MultivariatenormalScalingCython as MultivariatenormalScaling  # @UnresolvedImport

class Test_scaling_cython(unittest.TestCase):
	
	
	def setUp(self):
		self.eps = 1e-4
		


	def tearDown(self):
		pass
	
	
	
	def startup(self, d):
		self.d = d
		self.beta = - 0.01*(npr.rand(d, 1) + 1)
		self.m = 5 #dimension of Y
		self.N = 10 # number of observations
		R = 0.1 * npr.randn(self.m, self.m)
		self.Sigma =  R * R.transpose() + np.eye(self.m)
		Q = np.linalg.inv(self.Sigma)
		self.Ys = np.zeros((self.N, self.m))
		self.Bs = np.zeros((self.N, self.m, self.d))	
		self.V = np.eye(d) # prior
		self.V[0,0] = 5
		self.mu     = npr.randn(d)
		self.Second_Der = - np.linalg.inv(self.V)
		self.beta = self.beta.flatten()
		self.lik_python = - 0.5* np.dot((self.beta - self.mu), 
							   np.linalg.solve(self.V, (self.beta - self.mu).transpose()))
		for i in range(self.N):
			
			B  = np.hstack( ( np.ones((self.m, 1)), npr.randn(self.m, d - 1) ))
			self.Bs[i, :,:] = B
			Sigmas = np.dot(np.diagflat(np.exp( np.dot(B, self.beta)  )) ,
							np.dot(self.Sigma,
							np.diagflat(np.exp( np.dot(B, self.beta)  ))))
			Rs = np.linalg.cholesky(Sigmas).transpose()
			self.Ys[i, :] =  np.dot(Rs, npr.randn(self.m, 1)).transpose()
			Y_scaled = np.dot(np.diagflat(np.exp( - np.dot(B, self.beta))) , self.Ys[i,:])
			self.lik_python -= np.sum(  np.dot(B, self.beta)) + 0.5 *  np.dot(Y_scaled.transpose(),
								np.dot(Q, Y_scaled) )
			
			D_ = np.dot(np.dot(np.diagflat(Y_scaled), Q), np.diagflat(Y_scaled))
			D2_ = np.dot(Q, np.outer(Y_scaled, Y_scaled))
			
			D2_ = np.dot( np.dot(B.transpose(), np.diagflat(Y_scaled) * np.diagflat(np.dot(Q, Y_scaled)) ),
								   B)
			self.Second_Der -=  np.dot( np.dot(B.transpose(), D_ ),
								   B)
			self.Second_Der -= D2_
			
	def test_lik(self):
		
		ds = [1, 2, 10]
		for d in ds:
			self.startup(d)
			self.lik_run()
			
			
	def test_gradient_Hessian(self):
		
		ds = [1, 2, 10]
		for d in ds:
			self.startup(d)
			self.grad_hessian_run()
	
	
	
	def grad_hessian_run(self):
		"""
			verifies that the gradient matches the likelhiood,
			and that the (not used in object) Hessian is correctly
			computed
		"""
		MVNscaleObj = MultivariatenormalScaling({'mu': self.mu, 'Sigma':self.V})
		
		
		beta = self.beta.flatten()
		MVNscaleObj.setY(self.Ys)
		MVNscaleObj.setB(self.Bs)
		MVNscaleObj.setSigmaY(self.Sigma)
		#lik = MVNscaleObj.loglik(beta.flatten())		
		
		grad = MVNscaleObj.gradlik(beta.flatten())
		beta_eps    = np.zeros_like(beta)
		beta_eps[:] = beta[:]
		grad_num     = np.zeros_like(beta)
		Delta_num    = np.zeros((self.d, self.d))
		grad_eps = np.zeros_like(beta)
		for i in range(beta.shape[0]):
			beta_eps[i] += self.eps 
			grad_num[i] = MVNscaleObj.loglik(beta_eps.flatten())
			grad_eps    = MVNscaleObj.gradlik(beta_eps.flatten())
			beta_eps[i] -= self.eps 
			beta_eps[i] -= self.eps 
			grad_num[i] -= MVNscaleObj.loglik(beta_eps.flatten())
			grad_eps    -= MVNscaleObj.gradlik(beta_eps.flatten())
			beta_eps[i] += self.eps 
			grad_num[i] /= 2 * self.eps
			grad_eps    /= 2 * self.eps
			Delta_num[:,i] += grad_eps
			Delta_num[i,:] += grad_eps
	
		Delta_num /= 2
		np.testing.assert_almost_equal(grad, 
									grad_num, 
									decimal = 3, 
									err_msg = "theortical gradient does not match numerical")
		
		np.testing.assert_almost_equal(self.Second_Der, 
									Delta_num, 
									decimal = 3, 
									err_msg = "theortical hessain does not match numerical")
		
		
	
	def lik_run(self):
		MVNscaleObj = MultivariatenormalScaling({'mu': self.mu, 'Sigma':self.V})
		
		
		beta = self.beta.flatten()
		MVNscaleObj.setY(self.Ys)
		MVNscaleObj.setB(self.Bs)
		MVNscaleObj.setSigmaY(self.Sigma)
		lik = MVNscaleObj.loglik(beta.flatten())
		
		np.testing.assert_array_almost_equal(lik, 
											self.lik_python,
											decimal = 6)
		


if __name__ == "__main__":
	#import sys;sys.argv = ['', 'Test.testStartuo']
	unittest.main()
	