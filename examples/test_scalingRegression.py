'''
test code for scalingRegression, verifying that gradient and Hessian is correctly calculated
Created on Mar 28, 2016

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import numpy.random as npr
import time

from logisticnormal.priors import multivariatenormal_scaling

beta = - 0.01*(npr.rand(2, 1) + 1)
d = beta.shape[0]  # number of covariates
m = 5 #dimension of Y
N = 10 # number of observations
R = 0.1 * npr.randn(m, m)
Sigma =  R * R.transpose() + np.eye(m)
Q = np.linalg.inv(Sigma)
Ys = np.zeros((N, m))
Bs = np.zeros((N, m, d))	
lik_python = 0.
V = np.eye(d) # prior
V[0,0] = 5
mu     = np.zeros(d)
Second_Der = - np.linalg.inv(V)
for i in range(N):
	
	B  = np.hstack( ( np.ones((m, 1)), npr.randn(m, d - 1) ))
	Bs[i, :,:] = B
	Sigmas = np.dot(np.diagflat(np.exp( np.dot(B, beta)  )) ,
					np.dot(Sigma,
					np.diagflat(np.exp( np.dot(B, beta)  ))))
	Rs = np.linalg.cholesky(Sigmas).transpose()
	Ys[i, :] =  np.dot(Rs, npr.randn(m, 1)).transpose()


print("****")
# setting up object

MVNscaleObj = multivariatenormal_scaling({'mu': mu, 'Sigma':V})


beta = beta.flatten()
MVNscaleObj.setY(Ys)
MVNscaleObj.setB(Bs)
MVNscaleObj.setSigmaY(Sigma)
MVNscaleObj.setX(beta)
for i in range(1000):
	beta_ = MVNscaleObj.sample()
	print(beta_)
print(beta)


