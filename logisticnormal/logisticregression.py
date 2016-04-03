import numpy as np

from . import LogisticMNormal
from . import MultivariatenormalRegression, MultivariatenormalScaling
from . import invWishart


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

        self.d = None

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
                        a_mu - prior mean for regression coeff
                        V_mu - prior variance for regression coeff
                        W    - prior Wishart param
                        l    - prior wishart param
                        optional:
                        a_sigma - prior mean for scaling coeff
                        V_mu    - prior variance for scaling coeff
        '''
        if self.d is None:
            self.d = prior['W'].shape[0]+1
            self.inv_wishart.set_parameter({'theta': np.zeros((self.d-1,))})

        self.multivariatenormal_regression.set_prior(
            {'mu': prior['a_mu'], 'Sigma': prior['V_mu']})
        if 'a_sigma' in prior:
            self.multivariatenormal_scaling.set_prior(
                {'mu': prior['a_sigma'], 'Sigma': prior['V_sigma']})
        self.inv_wishart.set_prior({'Q': prior['W'], 'nu': prior['l']})

    def set_prior_param0(self):
        '''
            Setting non-informative prior.

            Necessary to do set_covariates first!
        '''
        self.d = self.Bs_mu[0]
        K_mu = self.Bs_mu[0].shape[1]
        prior = {'a_mu': np.zeros(K_mu), 'V_mu': 1e6*np.eye(K_mu),
                 'W': 1e-6*np.eye(self.d), 'l': self.d}
        if not self.Bs_sigma is None:
            K_sigma = self.Bs_sigma[0].shape[1]
            prior.update({'a_sigma': np.zeros(K_sigma), 'V_sigma': 1e6*np.eye(K_sigma)})
        self.set_prior(prior)

    def set_covariates(self, Bs, mean_ind, cov_ind=None):
        '''
            Bs     -   list (length self.J) of covariate matrices
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

    def initialization(self, alphas=None):
        """
            sets up default starting values

            alphas - vector of ceoffients
        """
        if not alphas is None:
            self.alphas = alphas

        self.J = self.alphas.shape[0]

        A = self.alphas.reshape(-1, 1)
        B = np.vstack(self.Bs_mu)
        self.beta_mu = np.linalg.lstsq(B, A)[0].reshape(-1)
        if not self.Bs_sigma is None:
            self.beta_sigma = np.zeros(self.Bs_sigma[0].shape[-1])

        r = alphas - self.mus
        self.Sigma = np.dot(r.T, r)*1./self.J
        self.Sigma0 = self.Sigma.copy()

        self.multivariatenormal_scaling.setX(np.zeros())

    def sample(self):
        """
            samples from the posterior distribution
            parameters are stored in:
            self.beta_mu
            self.beta_sigma
            self.Sigmas
            self.alphas
        """

        self.multivariatenormal_regression.setData(
            Y=self.alphas, QY=np.vstack(
                [invSigma[np.newaxis, :, :] for invSigma in self.invSigmas]))
        self.beta_mu = self.multivariatenormal_regression.sample()
        if not self.Bs_sigma is None:
            self.multivariatenormal_scaling.setData(
                Y=self.alphas-self.mus, SigmaY=self.Sigma0)
            self.beta_sigma = self.multivariatenormal_scaling.sample()

            self.inv_wishart.set_data(np.exp(-np.vstack(self.Bj_beta_sigmas))*(self.alphas-self.mus))
            self.Sigma0 = self.inv_wishart.sample()  # this also defines self.Sigmas, self.invSigmas
        else:
            #print "len(self.mus) = {}".format(len(self.mus))
            self.inv_wishart.set_data(self.alphas-self.mus)
            self.Sigma = self.inv_wishart.sample()  # this also defines self.Sigmas, self.invSigmas

    @property
    def beta_mu(self):
        return self._beta_mu

    @beta_mu.setter
    def beta_mu(self, beta_mu):
        if self._beta_mu is None:
            self._beta_mu = np.zeros_like(beta_mu)
        self._beta_mu[:] = beta_mu[:]
        self._mus = np.vstack([np.dot(Bj_mu, self._beta_mu) for Bj_mu in self.Bs_mu])

    @property
    def mus(self):
        return self._mus.copy()

    @property
    def beta_sigma(self):
        return self._beta_sigma

    @beta_sigma.setter
    def beta_sigma(self, beta_sigma):
        if self._beta_sigma is None:
            self._beta_sigma = np.zeros_like(beta_sigma)
        self._beta_sigma[:] = beta_sigma[:]
        self._Bj_beta_sigmas = [np.dot(Bj_sigma, self._beta_sigma) for Bj_sigma in self.Bs_sigma]
        self._exp_Bj_beta_sigmas = [np.exp(Bj_beta_sigma) for Bj_beta_sigma in self._Bj_beta_sigmas]

    @property
    def Bj_beta_sigmas(self):
        return self._Bj_beta_sigmas

    @property
    def exp_Bj_beta_sigmas(self):
        return self._exp_Bj_beta_sigmas

    @property
    def Sigmas(self):
        if self.Bs_sigma is None:
            return [self.Sigma]*self.J
        return [exp_Bj_beta_sigma.reshape(-1, 1)*self.Sigma0*exp_Bj_beta_sigma.reshape(1, -1)
                for exp_Bj_beta_sigma in self.exp_Bj_beta_sigmas]

    @property
    def invSigmas(self):
        if self.Bs_sigma is None:
            return [self.invSigma]*self.J
        return [1./exp_Bj_beta_sigma.reshape(-1, 1)*self.invSigma0*1./exp_Bj_beta_sigma.reshape(1, -1)
                for exp_Bj_beta_sigma in self.exp_Bj_beta_sigmas]

    @property
    def Sigma0(self):
        return self._Sigma0

    @Sigma0.setter
    def Sigma0(self, Sigma0):
        self._Sigma0 = Sigma0
        self._invSigma0 = np.linalg.inv(Sigma0)

    @property
    def invSigma0(self):
        return self._invSigma0

    @property
    def Sigma(self):
        return self._Sigma

    @Sigma.setter
    def Sigma(self, Sigma):
        if self._Sigma is None:
            self._Sigma = np.zeros_like(Sigma)
            self.invSigma = np.zeros_like(Sigma)

        self._Sigma[:] = Sigma[:]
        self.invSigma[:] = np.linalg.inv(self._Sigma)[:]

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, alphas):
        if not hasattr(self, '_alphas'):
            self._alphas = np.zeros_like(alphas)
        self._alphas[:] = alphas[:]


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

        self.logistic_regression_prior = LogisticRegressionPrior(prior)

        if not data is None:
            self.set_data(data)

    def set_data(self, n):
        """
            n - (J x d) numpy vector of observation count
        """
        if self.d is None:
            self.d = n.shape[1]
            self.logistic_regression_prior.inv_wishart.set_parameter({'theta': np.zeros((self.d-1,))})

        if self.J is None:
            self.J, self.d = n.shape
            self.logistic_m_normals = [LogisticMNormal() for j in range(self.J)]  # @UnusedVariable

        for nj, lmn in zip(np.split(n, self.J, axis=0), self.logistic_m_normals):
            lmn.set_data(nj.reshape(self.d, 1))

    def set_prior(self, prior):
        self.logistic_regression_prior.set_prior(prior)

    def set_covariates(self, Bs, mean_ind, cov_ind=None):
        self.logistic_regression_prior.set_covariates(Bs, mean_ind, cov_ind)

    def initialization(self):
        """
            sets up default starting values

        """
        for lmn in self.logistic_m_normals:
            p = np.array(lmn.n, dtype=np.float)
            p[p == 0] = 1
            p /= np.sum(p)
            lmn.set_alpha_p(p)
        self.update_alphas()

        self.logistic_regression_prior.initialization(self.alphas)

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

        self.logistic_regression_prior.sample()

    @property
    def mus(self):
        return self.logistic_regression_prior.mus

    @property
    def Sigmas(self):
        return self.logistic_regression_prior.Sigmas

    @property
    def invSigmas(self):
        return self.logistic_regression_prior.invSigmas

    @property
    def beta_mu(self):
        return self.logistic_regression_prior.beta_mu

    @property
    def Sigma(self):
        return self.logistic_regression_prior.Sigma

    def update_alphas(self):
        self.alphas = np.vstack([lmn.alpha for lmn in self.logistic_m_normals])

