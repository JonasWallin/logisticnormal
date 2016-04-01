import unittest
import numpy as np

from logisticnormal import LogisticRegression
from logisticnormal import LogisticRegressionSampler


class TestLogisticRegression(unittest.TestCase):

    def startup(self):
        J = 10
        d = 5
        self.lr = LogisticRegression(d=d)

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
        self.lr.init()

    def test_setup(self):
        self.startup()

    def test_sampling(self):
        sim = 10
        self.startup()
        for i in range(sim):
            self.lr.sample()

    def test_logisticregressionsampler(self):
        self.startup()
        lrs = LogisticRegressionSampler(20, 10, 10)
        lrs.set_sampling_object(self.lr)
        lrs.run()


if __name__ == '__main__':
    unittest.main()
