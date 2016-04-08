import unittest
import numpy as np

from logisticnormal import LogisticRegression
from logisticnormal import LogisticRegressionSampler


class TestLogisticRegression(unittest.TestCase):

    def startup1(self):
        J = 10
        d = 5
        self.lr = LogisticRegression()

        # data
        N = 10000
        p = np.arange(d, 0, -1, dtype=np.float)
        p /= np.sum(p)
        n = np.random.multinomial(N, p, size=J)
        self.lr.set_data(n)

        # covariates
        K = 2
        covars = np.vstack([np.arange(j, K+j) for j in range(J)])
        Bs = [np.hstack([covars[j, k]*np.eye(d-1) for k in range(K)]) for j in range(J)]  # [np.tile(range(j, K+j), (d-1, 1)) for j in range(J)]
        mean_ind = np.arange(K*(d-1))
        self.lr.set_covariates(Bs, mean_ind)

        # prior
        prior = {'a_mu': np.zeros(K*(d-1)), 'V_mu': np.eye(K*(d-1)),
                 'W': np.eye(d-1), 'l': d+3}
        self.lr.set_prior(prior)

        # initialization
        self.lr.initialization()

    def startup2(self):
        J = 10
        d = 5
        self.lr = LogisticRegression()

        # data
        N = 10000
        p = np.arange(d, 0, -1, dtype=np.float)
        p /= np.sum(p)
        n = np.random.multinomial(N, p, size=J)
        self.lr.set_data(n)

        # covariates
        K = 2
        covars = np.vstack([np.arange(j, K+j) for j in range(J)])
        covars = np.hstack([np.ones((J, 1)), covars])
        Bs = [np.hstack([covars[j, k]*np.eye(d-1) for k in range(K)]) for j in range(J)]  # [np.tile(range(j, K+j), (d-1, 1)) for j in range(J)]
        mean_ind = np.arange(K*(d-1))
        cov_ind = np.arange(d-1, K*(d-1))
        self.lr.set_covariates(Bs, mean_ind, cov_ind)

        # prior
        prior = {'a_mu': np.zeros(K*(d-1)), 'V_mu': np.eye(K*(d-1)),
                 'a_sigma': np.zeros((K-1)*(d-1)), 'V_sigma': np.eye((K-1)*(d-1)),
                 'W': np.eye(d-1), 'l': d+3}
        self.lr.set_prior(prior)

        # initialization
        self.lr.initialization()

    def test_setup1(self):
        self.startup1()

    def test_setup2(self):
        self.startup2()

    def test_sampling(self):
        sim = 10
        self.startup1()
        for i in range(sim):
            self.lr.sample()

    def test_sampling_with_scaling(self):
        sim = 10
        self.startup2()
        for i in range(sim):
            self.lr.sample()

    def test_logisticregressionsampler(self):
        self.startup1()
        lrs = LogisticRegressionSampler(20, 10, 10)
        lrs.set_sampling_object(self.lr)
        lrs.run()


if __name__ == '__main__':
 #   suite = unittest.TestLoader().loadTestsFromTestCase(TestLogisticRegression)
 #   suite.debug()
    unittest.main()
