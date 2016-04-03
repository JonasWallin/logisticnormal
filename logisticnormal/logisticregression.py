import numpy as np

from . import LogisticMNormal
from . import MultivariatenormalRegression, MultivariatenormalScaling
from . import invWishart


class LogisticRegression(object):

	def __init__(self, data=None, prior=None):
		"""

		"""

		if data is None:
			self.J = None
			self.d = None
		else:
			self.J, self.d = data.shape
			self.logistic_m_normals = [LogisticMNormal() for j in range(self.J)]  # @UnusedVariable

		self.multivariatenormal_regression = MultivariatenormalRegression()
		self.multivariatenormal_scaling = MultivariatenormalScaling()
		self.inv_wishart = invWishart()

		if not prior is None:
			self.set_prior(prior)

		if not data is None:
			self.set_data(data)

		self.Bs_mu = None
		self.Bs_sigma = None
		self._Sigma = None
		self._Sigmas = None
		self._beta_mu = None
		self._beta_sigma = None

	def set_prior(self, prior):
		'''
			prior   -   dictionary with attributes a_mu, V_mu, W, l
		'''
		if self.d is None:
			self.d = len(prior['a_mu'].flatten())
			self.inv_wishart.set_parameter({'theta': np.zeros((self.d-1,))})

		self.multivariatenormal_regression.set_prior(
			{'mu': prior['a_mu'], 'Sigma': prior['V_mu']})
		if 'a_sigma' in prior:
			self.multivariatenormal_scaling.set_prior(
				{'mu': prior['a_sigma'], 'Sigma': prior['V_sigma']})
		self.inv_wishart.set_prior({'Q': prior['W'], 'nu': prior['l']})

	def set_data(self, n):
		"""
			n - (J x d) numpy vector of observation count
		"""
		if self.d is None:
			self.d = n.shape[1]
			self.inv_wishart.set_parameter({'theta': np.zeros((self.d-1,))})

		if self.J is None:
			self.J, self.d = n.shape
			self.logistic_m_normals = [LogisticMNormal() for j in range(self.J)]  # @UnusedVariable

		for nj, lmn in zip(np.split(n, self.J, axis=0), self.logistic_m_normals):
			lmn.set_data(nj.reshape(self.d, 1))

	def set_covariates(self, Bs, mean_ind, cov_ind=None):
		'''
			Bs	   -   list (length self.J) of covariate matrices
							  (self.d x #covariates)
			mean_ind   -   indices of covariates affecting mean
			cov_ind -   indices of covariates affecting covariance
		'''
		self.Bs_mu = [B[:, mean_ind] for B in Bs]
		self.multivariatenormal_regression.setB(
			np.ascontiguousarray(np.vstack([Bj_mu[np.newaxis, :, :] for Bj_mu in self.Bs_mu]), dtype=np.float))
		if not cov_ind is None:
			self.Bs_sigma = [B[:, cov_ind] for B in Bs]
			self.multivariatenormal_scaling.setB(
				np.ascontiguousarray(np.vstack([Bj_sigma[np.newaxis, :, :] for Bj_sigma in self.Bs_sigma]), dtype=np.float))

	def init(self):
		"""
			sets up deafult starting values

		"""
		for lmn in self.logistic_m_normals:
			p = np.array(lmn.n, dtype=np.float)
			p[p == 0] = 1
			p /= np.sum(p)
			#print "p = {}".format(p)
			lmn.set_alpha_p(p)
		self.update_alphas()

		A = self.alphas.reshape(-1, 1)
		B = np.vstack(self.Bs_mu)
		self.beta_mu = np.linalg.lstsq(B, A)[0].reshape(-1)
		if not self.Bs_sigma is None:
			self.beta_sigma = np.zeros(self.Bs_sigma.shape[-1])

		r = self.alphas - self.mus
		self.Sigma = np.dot(r.T, r)*1./self.J
		self.Sigma0 = self.Sigma.copy()

	def sample(self):
		"""
			samples from the posterior distribution
			parameters are stored in:
			self.beta_mu
			self.beta_sigma
			self.Sigma
			self.alphas
		"""
		for lmn, mu, Sigma, invSigma in zip(self.logistic_m_normals, self.mus, self.Sigmas, self.invSigmas):
			lmn.set_prior({'mu': mu, 'Sigma': Sigma, 'Q': invSigma})
			lmn.sample()

		self.update_alphas()

		self.multivariatenormal_regression.setData(Y=self.alphas,
												   QY=np.vstack(self.invSigmas[np.newaxis, :, :]))
		self.beta_mu = self.multivariatenormal_regression.sample()
		if not self.Bs_sigma is None:
			# sample beta_sigma
			self.multivariatenormal_scaling.setData(Y=self.alphas-self.mus,
													SigmaY=self.Sigma0)
			self.beta_sigma = self.multivariatenormal_scaling.sample()

			self.inv_wishart.set_data(np.exp(-np.vstack(self.Bj_beta_sigmas))*(self.alphas-self.mus))
			self.Sigma0 = self.inv_wishart.sample()
			exp_Bj_beta_sigmas = [np.exp(Bj_beta_sigma) for Bj_beta_sigma in self.Bj_beta_sigmas]
			self.Sigmas = [exp_Bj_beta_sigma.reshape(-1, 1)*self.Sigma0*exp_Bj_beta_sigma.reshape(1, -1)
						   for exp_Bj_beta_sigma in exp_Bj_beta_sigmas]
		else:
			self.inv_wishart.set_data(self.alphas-self.mus)
			self.Sigma = self.inv_wishart.sample()  # this also defines self.Sigmas

	@property
	def beta_mu(self):
		return self._beta_mu

	@beta_mu.setter
	def beta_mu(self, beta_mu):
		if self._beta_mu is None:
			self._beta_mu = np.zeros_like(beta_mu)
		self._beta_mu[:] = beta_mu[:]
		self._mus = [np.dot(Bj_mu, self._beta_mu) for Bj_mu in self.Bs_mu]

	@property
	def mus(self):
		return self._mus

	@property
	def beta_sigma(self):
		return self._beta_sigma

	@beta_sigma.setter
	def beta_sigma(self, beta_sigma):
		if self._beta_sigma is None:
			self._beta_sigma = np.zeros_like(beta_sigma)
		self._beta_sigma[:] = beta_sigma[:]
		self._Bj_beta_sigmas = [np.dot(Bj_sigma, self._beta_sigma) for Bj_sigma in self.Bs_sigma]

	@property
	def Bj_beta_sigmas(self):
		return self._Bj_beta_sigmas

	@property
	def Sigmas(self):
		if self.Bs_sigma is None:
			return [self.Sigma]*self.J
		return self._Sigmas

	@Sigmas.setter
	def Sigmas(self, Sigmas):
		if self.Bs_sigma is None:
			raise ValueError('Not allowed to set variable Sigmas in common covariance mode')
		if self._Sigmas is None:
			self._Sigmas = [np.zeros_like(Sigma) for Sigma in Sigmas]
			self.invSigmas = [np.zeros_like(Sigma) for Sigma in Sigmas]
		for _Sigma, invSigma, Sigma in zip(self._Sigmas, self.invSigmas, self.Sigmas):
			_Sigma[:] = Sigma[:]
			invSigma[:] = np.linalg.inv(Sigma)[:]

	@property
	def invSigmas(self):
		if self.Bs_sigma is None:
			return [self.invSigma]*self.J
		return self._invSigmas

	@property
	def Sigma(self):
		return self._Sigma

	@Sigma.setter
	def Sigma(self, Sigma):
		if self._Sigma is None:
			self._Sigma = np.zeros_like(Sigma)
			self.invSigmas = np.zeros_like(Sigma)

		self._Sigma[:] = Sigma[:]
		self.invSigma[:] = np.linalg.inv(self._Sigma)[:]

	def update_alphas(self):
		self.alphas = np.vstack([lmn.alpha for lmn in self.logistic_m_normals])


class LogisticRegressionPrior(object):
	"""
		object for sampling
		beta_mu, beta_sigma, Sigma in logistic object
	
	"""

	def __init__(self, prior=None):
		"""
			prior see -> prior
		"""

		self.multivariatenormal_regression = MultivariatenormalRegression()
		self.multivariatenormal_scaling = MultivariatenormalScaling()
		self.inv_wishart = invWishart()

		if not prior is None:
			self.set_prior(prior)

		self.Bs_mu = None
		self.Bs_sigma = None
		self._Sigma = None
		self._Sigmas = None
		self._beta_mu = None
		self._beta_sigma = None

	def set_prior(self, prior):
		'''
			prior   -   dictionary with attributes 
						a_mu - prior mean for regerssion coeff 
						V_mu - prior variance for regression coeff
						W    - prior Wishart param 
						l    - prior wishart param
						otpional:
						a_sigma - prior mean for scaling coeff
						V_mu    - prior variance for scaling coeff
		'''
		if self.d is None:
			self.d = len(prior['a_mu'].flatten())
			self.inv_wishart.set_parameter({'theta': np.zeros((self.d-1,))})

		self.multivariatenormal_regression.set_prior(
			{'mu': prior['a_mu'], 'Sigma': prior['V_mu']})
		if 'a_sigma' in prior:
			self.multivariatenormal_scaling.set_prior(
				{'mu': prior['a_sigma'], 'Sigma': prior['V_sigma']})
		self.inv_wishart.set_prior({'Q': prior['W'], 'nu': prior['l']})


	def set_covariates(self, Bs, mean_ind, cov_ind=None):
		'''
			Bs	   -   list (length self.J) of covariate matrices
							  (self.d x #covariates)
			mean_ind   -   indices of covariates affecting mean
			cov_ind -   indices of covariates affecting covariance
		'''
		self.Bs_mu = [B[:, mean_ind] for B in Bs]
		self.multivariatenormal_regression.setB(
			np.ascontiguousarray(np.vstack([Bj_mu[np.newaxis, :, :] for Bj_mu in self.Bs_mu]), dtype=np.float))
		if not cov_ind is None:
			self.Bs_sigma = [B[:, cov_ind] for B in Bs]
			self.multivariatenormal_scaling.setB(
				np.ascontiguousarray(np.vstack([Bj_sigma[np.newaxis, :, :] for Bj_sigma in self.Bs_sigma]), dtype=np.float))

	def init(self, alphas = None):
		"""
			sets up deafult starting values

			alphas - vector of ceoffients
		"""
		if alphas is None:
			self.alphas = alphas

		A = self.alphas.reshape(-1, 1)
		B = np.vstack(self.Bs_mu)
		self.beta_mu = np.linalg.lstsq(B, A)[0].reshape(-1)
		if not self.Bs_sigma is None:
			self.beta_sigma = np.zeros(self.Bs_sigma.shape[-1])

		r = alphas - self.mus
		self.Sigma = np.dot(r.T, r)*1./self.J
		self.Sigma0 = self.Sigma.copy()

	def sample(self):
		"""
			samples from the posterior distribution
			parameters are stored in:
			self.beta_mu
			self.beta_sigma
			self.Sigma
			self.alphas
		"""

		self.multivariatenormal_regression.setData(Y=self.alphas,
												   QY=np.vstack(self.invSigmas[np.newaxis, :, :]))
		self.beta_mu = self.multivariatenormal_regression.sample()
		if not self.Bs_sigma is None:
			# sample beta_sigma
			self.multivariatenormal_scaling.setData(Y=self.alphas-self.mus,
													SigmaY=self.Sigma0)
			self.beta_sigma = self.multivariatenormal_scaling.sample()

			self.inv_wishart.set_data(np.exp(-np.vstack(self.Bj_beta_sigmas))*(self.alphas-self.mus))
			self.Sigma0 = self.inv_wishart.sample()
			exp_Bj_beta_sigmas = [np.exp(Bj_beta_sigma) for Bj_beta_sigma in self.Bj_beta_sigmas]
			self.Sigmas = [exp_Bj_beta_sigma.reshape(-1, 1)*self.Sigma0*exp_Bj_beta_sigma.reshape(1, -1)
						   for exp_Bj_beta_sigma in exp_Bj_beta_sigmas]
		else:
			self.inv_wishart.set_data(self.alphas-self.mus)
			self.Sigma = self.inv_wishart.sample()  # this also defines self.Sigmas

	@property
	def beta_mu(self):
		return self._beta_mu

	@beta_mu.setter
	def beta_mu(self, beta_mu):
		if self._beta_mu is None:
			self._beta_mu = np.zeros_like(beta_mu)
		self._beta_mu[:] = beta_mu[:]
		self._mus = [np.dot(Bj_mu, self._beta_mu) for Bj_mu in self.Bs_mu]

	@property
	def mus(self):
		return self._mus

	@property
	def beta_sigma(self):
		return self._beta_sigma

	@beta_sigma.setter
	def beta_sigma(self, beta_sigma):
		if self._beta_sigma is None:
			self._beta_sigma = np.zeros_like(beta_sigma)
		self._beta_sigma[:] = beta_sigma[:]
		self._Bj_beta_sigmas = [np.dot(Bj_sigma, self._beta_sigma) for Bj_sigma in self.Bs_sigma]

	@property
	def Bj_beta_sigmas(self):
		return self._Bj_beta_sigmas

	@property
	def Sigmas(self):
		if self.Bs_sigma is None:
			return [self.Sigma]*self.J
		return self._Sigmas

	@Sigmas.setter
	def Sigmas(self, Sigmas):
		if self.Bs_sigma is None:
			raise ValueError('Not allowed to set variable Sigmas in common covariance mode')
		if self._Sigmas is None:
			self._Sigmas = [np.zeros_like(Sigma) for Sigma in Sigmas]
			self.invSigmas = [np.zeros_like(Sigma) for Sigma in Sigmas]
		for _Sigma, invSigma, Sigma in zip(self._Sigmas, self.invSigmas, self.Sigmas):
			_Sigma[:] = Sigma[:]
			invSigma[:] = np.linalg.inv(Sigma)[:]

	@property
	def invSigmas(self):
		if self.Bs_sigma is None:
			return [self.invSigma]*self.J
		return self._invSigmas

	@property
	def Sigma(self):
		return self._Sigma

	@Sigma.setter
	def Sigma(self, Sigma):
		if self._Sigma is None:
			self._Sigma = np.zeros_like(Sigma)
			self.invSigmas = np.zeros_like(Sigma)

		self._Sigma[:] = Sigma[:]
		self.invSigma[:] = np.linalg.inv(self._Sigma)[:]

	def set_alphas(self, alphas):
		
		if self.alphas is None:
			self.alphas = np.zeros_like(alphas)
		self.alphas[:] = alphas[:]
	
	
