import numpy as np
import time


class MCMCSampler(object):

    def __init__(self, number_samples, burnin, trace_len=None,
                 sampling_object=None):
        self.sim = number_samples
        self.burnin = burnin
        if trace_len is None:
            self.save_freq = 1
            self.nsave = self.sim
        else:
            self.save_freq = int(np.ceil(self.sim*1./trace_len))
            self.nsave = self.sim/self.save_freq

        self.traces = {}
        self.moments = {}
        self.sampling_object = sampling_object

    def set_sampling_object(self, sampling_object):
        self.sampling_object = sampling_object

    def add_trace_variable(self, var):
        self.traces[var] = None

    def add_moment_variable(self, var, n_moments=2):
        self.moments[var] = [None]*n_moments

    def get_sampling_var(self, var):
        val = getattr(self.sampling_object, var)
        try:
            val**2
        except TypeError:
            val = np.vstack([val_[np.newaxis, ...] for val_ in val])
        return val

    def run(self):
        self.sampling_object.initialization()
        self.sampling_object.sample()

        for var in self.traces:
            val = self.get_sampling_var(var)
            self.traces[var] = np.empty((self.nsave,)+val.shape, dtype=val.dtype)

        for var in self.moments:
            val = self.get_sampling_var(var)

            self.moments[var] = [np.zeros_like(val) for m in range(1, len(self.moments[var])+1)]

        t0 = time.time()
        for i in range(self.sim):
            self.sampling_object.sample()

            if np.mod(i, self.save_freq) == 0:
                for var in self.traces:
                    val = self.get_sampling_var(var)
                    self.traces[var][i/self.save_freq, ...] = val

            if i >= self.burnin:
                for var in self.moments:
                    val = self.get_sampling_var(var)
                    for m, mom in enumerate(self.moments[var]):
                        mom += val**(m+1)
        t1 = time.time()
        print("Time per iteration: = {}".format((t1-t0)/self.sim))


class LogisticRegressionSampler(MCMCSampler):

    def __init__(self, number_samples, burnin, trace_len=None):
        super(LogisticRegressionSampler, self).__init__(number_samples, burnin, trace_len)
        for var in ['alphas', 'beta_mu']:
            self.add_trace_variable(var)
        for var in ['alphas', 'beta_mu', 'Sigma']:
            self.add_moment_variable(var)

    def run(self):
        if hasattr(self.sampling_object.logistic_regression_prior, 'Bs_sigma') and \
                not self.sampling_object.logistic_regression_prior.Bs_sigma is None:
            for var in ['beta_sigma']:
                self.add_trace_variable(var)
            for var in ['beta_sigma']:
                self.add_moment_variable(var)
        super(LogisticRegressionSampler, self).run()
