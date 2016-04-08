'''
Created on Apr 5, 2016

@author: jonaswallin
'''
import numpy as np
import numpy.random as npr
import scipy.optimize as spo
import matplotlib.pyplot as plt
from logisticnormal import tMd
n =  10000
d =  4


mu  = npr.randn(d)
L   = np.diag(npr.rand(d) + 1)
index = np.tril_indices(d, k = -1)
L[index]  = npr.randn(d * (d-1) / 2) 

tmObj = tMd()
tmObj.set_param(mu  = mu, L = L)
tmObj.nu = 0.1
X = tmObj.sample(n)
tmObj.set_data(X)
x0 =  np.hstack((mu, np.log(np.diag(L)),L[index]))
xest = spo.fmin(lambda x: tmObj.f_lik(x), x0)
tmObj.set_theta(xest)
mu_est = xest[:d]
L_est = np.diag(np.exp(xest[d:(2*d)]))
L_est[np.tril_indices(d, k = -1)] = xest[(2*d):((2*d) + d*(d -1)/2 )]

iV = tmObj.weights()
mu_est_w =  np.sum(iV.reshape((n,1))*tmObj.data/np.sum(iV),0)