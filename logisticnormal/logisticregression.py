import numpy as np

from . import LogisticMNormal
from . import MultivariatenormalRegression
from . import invWishart


class LogisticRegression(object):

    def __init__(self, data=None, prior=None, J=None, d=None):
        if not J is None and not d is None:
            self.J, self.d = J, d
        else:
            self.J, self.d = data.shape

        if not prior is None:
            self.set_prior(prior)

        self.logistic_m_normals = [LogisticMNormal() for j in range(self.J)]
        self.multivariatenormal_regression = MultivariatenormalRegression()
        self.inv_wishart = invWishart()
        self.inv_wishart.set_param({'theta': np.zeros((self.d-1,))})

        if not data is None:
            self.set_data(data)

    def set_prior(self, prior):
        '''
            prior   -   dictionary with attributes a_mu, V_mu, W, l
        '''
        self.multivariatenormal_regression.set_prior(
            {'mu': prior['a_mu'], 'Sigma': prior['V_mu']})
        self.inv_wishart.set_prior({'Q': prior['W'], 'nu': prior['l']})

    def set_data(self, n):
        """
            n - (J x d) numpy vector of observation count
        """
        for nj, lmn in zip(np.split(n, self.J, axis=0), self.logistic_m_normals):
            lmn.set_data(nj.reshape(self.d, 1))

    def set_covariates(self, Bs, mean_ind, cov_ind):
        '''
            Bs          -   list of covariate matrices
            mean_ind    -   indices of covariates affecting mean
            cov_ind     -   indices of covariates affecting covariance
        '''
        self.Bs = Bs
        self.Bs_mu = [B[:, mean_ind] for B in self.Bs]
        self.multivariatenormal_regression.setB(np.vstack([B_i_mu[np.newaxis, :, :] for B_i_mu in self.Bs_mu]))
        self.Bs_sigma = [B[:, cov_ind] for B in self.Bs]  # not used currently

    # def init(self):
    #     p = 

    def sample(self):
        for lmn, mu in zip(self.logistic_m_normals, self.mus):
            lmn.set_prior({'mu': mu, 'Sigma': self.Sigma, 'Q': self.invSigma})
            lmn.sample()
        self.alphas = np.vstack([lmn.alpha for lmn in self.logistic_m_normals])

        self.multivariatenormal_regression.setData(Y=self.alphas, QY=np.tile(self.invSigma, (self.J, 1, 1)))
        self.beta_mu = self.multivariatenormal_regression.sample()
        self.mus = [np.dot(B_i_mu, self.beta_mu) for B_i_mu in self.Bs_mu]

        self.inv_wishart.set_data(self.alphas-self.mus)
        self.Sigma = self.inv_wishart.sample()
        self.invSigma = np.linalg.inv(self.Sigma)




