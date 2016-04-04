
import numpy as np
cimport numpy as np
cimport cython
import cPickle as pickle
cdef extern void outer_Y(double* YYt, const double* Y, const int n, const int d) nogil


cdef class LogisticMNormal:
    """
        Class for sampling and storing logistic Multivariate normal distribution
        
        param
        n - (Kx1) numpy vector of observation count
        K - (int) number of classes
    """
    cdef public int d, amcmc_batch
    cdef public np.ndarray Q, mu, Sigma, n, alpha, grad, neg_Hessian, L, Lg, LtLg
    cdef public bint AMCMC
    cdef public double sigma_MCMC, n_count, count_mcmc, accept_mcmc,  sum_n, llik
    cdef public double amcmc_delta_max, amcmc_desired_accept, amcmc_delta_rate
    cdef dict prior
    cdef np.ndarray Q_mu
    cdef double amcmc_count, amcmc_accept

    @cython.boundscheck(False)
    @cython.wraparound(False)            
    def __init__(self, prior = None):
        '''
            prior:
            prior['mu']    - np.array(dim=1) (d-1)
            prior['Sigma'] = np.array(dim=2) (d-1) x (d-1)
            
            
        '''
        self.prior = None
        self.n_count = 0
        self.n = None
        self.alpha = None
        self.d = -1
        self.mu  = None
        self.Q = None
        if prior != None:
            self.set_prior(prior)
            
        self.AMCMC = False
        self.sigma_MCMC = 0.99 #not equal to one
        self.count_mcmc = 0.
        self.accept_mcmc = 0.
        self.amcmc_count  = 0.
        self.amcmc_accept = 0.
    
    def pickle(self, filename):
        """
            store object in file
        """
        f = file(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        """
            load object from file
            use:
            
            object = Multivariatenormal.unpickle(filename)
        """
        with file(filename, 'rb') as f:
            return pickle.load(f)    



    @cython.boundscheck(False)
    @cython.wraparound(False)    
    def set_prior(self, prior):
        """

                prior:
                    prior['mu']    - np.array(dim=1) (d-1)
                    prior['Sigma'] = np.array(dim=2) (d-1) x (d-1)
                    prior['Q'] = np.array(dim=2) (d-1) x (d-1)
        """
        
        
        
        
        self.mu       = np.zeros_like(prior['mu'].flatten())
        self.mu[:]    = prior['mu'].flatten()[:]
        self.Sigma    = np.zeros_like(prior['Sigma'])
        self.Sigma[:] = prior['Sigma'][:]
        if 'Q' in prior:
            self.Q = np.zeros_like(prior['Q'])
            self.Q = prior['Q'][:]
        else:
            self.Q        = np.linalg.inv(self.Sigma)
        self.Q_mu     = np.dot(self.Q,self.mu)
        self.d        = self.Q_mu.shape[0] + 1
        if self.alpha is None:
            self.alpha = np.zeros(self.d - 1)
        else:
            if self.alpha.shape[0] != self.d - 1:
                raise Exception('logistincormal.set_prior dimension error')
        
    def get_lprior_grad_hess(self, alpha = None):
    
        if alpha is None:
            alpha = self.alpha
        
        alpha_mu = alpha - self.mu
        grad = - np.dot(self.Q, alpha_mu)
        llik =   np.dot(alpha_mu.T, grad)/2.
        Hessian = np.zeros_like(self.Q)
        Hessian[:] = - self.Q[:]
        return llik, grad, Hessian
    
    
    def get_f_grad_hess(self, alpha = None):
        """
            Calculating value, the gradient , Heissan of the density
            
            alpha - (d-1) values
        """
        llik, grad, Hessian    = self.get_llik_grad_hess(alpha)
        llik2, grad2, Hessian2 = self.get_lprior_grad_hess(alpha)
        llik += llik2
        grad += grad2
        Hessian += Hessian2
        return llik, grad, Hessian
    
    def get_llik_grad_hess(self, np.ndarray[np.double_t, ndim=1] alpha = None):
        """
            gradient and Hessian of the log likelihood component of the model
            
            alpha - (d-1 , ) vector of probability components 
        """
        if alpha is None:
            alpha = self.alpha
        alpha = alpha.flatten()
        
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] exp_alpha = np.exp(alpha)
        cdef double c_alpha = (1 + np.sum(exp_alpha))
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] grad = - self.sum_n * exp_alpha / c_alpha

        exp_alpha *= np.sqrt(self.sum_n)/c_alpha
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] Hessian  =  np.empty((self.d -1 ,self.d -1))
        outer_Y(<double *>  &Hessian[0,0],<double *>  &exp_alpha[0], 1, exp_alpha.shape[0])
        #Hessian =   np.outer(exp_alpha.T, exp_alpha) 
        Hessian +=  np.diag(grad)
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] n_alpha = self.n[1:]
        grad += n_alpha

        cdef double llik =  np.sum(n_alpha * alpha) - self.sum_n * np.log(c_alpha)


        return llik, grad, Hessian
        
    def get_p(self, alpha = None):
        """
            get the probabilities from the object 
        """
        if alpha is None:
            alpha = self.alpha
            
        p = np.hstack((1.,np.exp(alpha)))
        p /= np.sum(p)
        return p
        
    def set_alpha(self,np.ndarray alpha): 
        """
            setting alpha parameter
        
        """
        self.alpha = np.zeros_like(alpha.flatten())
        self.alpha[:] = alpha.flatten()[:]
        #if not self.n is None and not self.mu is None:
        #    self.update_llik()
            
            
    def set_alpha_p(self,np.ndarray p):
        """
            setting alpha through p
            
            p - (dx1) simplex vector
        """
        
        alpha = np.zeros(len(p)-1)
        sum_exp = 1./ p[0]
        for i in range(1,len(p)):
            alpha[i-1] = np.log(p[i] * sum_exp)
        
        self.set_alpha(alpha)           
            
    def update_llik(self, np.ndarray[np.double_t, ndim=1]  alpha = None):
        """
            Update components of the likelihood used in MALA
        """
        
        store = False
        if alpha is None:
            store = True
            alpha = self.alpha
            
        llik, grad,  neg_Hessian = self.get_f_grad_hess(alpha)
        neg_Hessian *= -1
        L    = np.linalg.cholesky(neg_Hessian)
        Lg   = np.linalg.solve(L, grad)
        LtLg = np.linalg.solve(L.T, 0.5 * Lg)
        
        if store:
            self.llik, self.grad, self.neg_Hessian = llik, grad, neg_Hessian
            self.L    = L
            self.Lg   = Lg
            self.LtLg = LtLg
        
        return llik, grad, neg_Hessian, L, Lg, LtLg
    

        
    def sample(self, z = None):
        """
            Sampling using AMCMC MALA with preconditioner as Hessian
        """
        if self.alpha is None:
            raise Exception('alpha must be set use set_alpha')
        
        self.count_mcmc   += 1
        self.amcmc_count  += 1
        if z is None:
            z =np.random.randn(self.d-1)
        
        
        llik_old, grad_old, neg_Hessian_old, L_old, Lg_old, LtLg_old = self.update_llik(self.alpha)
        
        mu         = self.alpha  + LtLg_old *self.sigma_MCMC**2
        alpha_star = np.linalg.solve(L_old.T, self.sigma_MCMC*z) + mu 
        llik_star, grad_star, neg_Hessian_star, _, __, LtLg_star = self.update_llik(alpha_star)
        
        mu_star = alpha_star + LtLg_star * self.sigma_MCMC**2
        res_old  = self.alpha - mu_star
        res_star  = alpha_star - mu
        q_o = -(0.5/self.sigma_MCMC**2) *  np.dot( res_old, np.dot(neg_Hessian_star,
                                                   res_old))   
        q_s = -(0.5/self.sigma_MCMC**2) *  np.dot( res_star.T, np.dot(neg_Hessian_old,
                                                   res_star) )
        
        U = np.random.rand(1)

        if np.log(U) < llik_star - llik_old + q_o - q_s:       
            self.accept_mcmc  += 1
            self.amcmc_accept += 1
            self.alpha = alpha_star
        
            
        if self.AMCMC:
            
            self.update_AMCMC()
            
        alpha    = np.zeros_like(self.alpha)
        alpha[:] = self.alpha[:]
        
        return alpha
        
    def set_AMCMC(self, batch = 50, accpate = 0.574, delta_rate = 1.):
        """
            Using AMCMC
            
            batch      - (int) how often to update sigma_MCMC
            accpate    - [0,1] desired accpance rate (0.574)
            delta_rate - [0,1] updating ratio for the amcmc
        """
        
        
        
        self.amcmc_delta_max    = 0.1
        self.amcmc_desired_accept = accpate
        self.amcmc_batch        = batch
        self.amcmc_delta_rate   =  delta_rate
        self.AMCMC = True


    def update_AMCMC(self):
        """
        Using roberts and rosenthal method for tunning the acceptance rate
        """
    
    
        if (self.amcmc_count +1) % self.amcmc_batch == 0:

            delta = np.min([self.amcmc_delta_max, (self.count_mcmc/self.amcmc_batch)**(-self.amcmc_delta_rate)])
            
            if self.amcmc_accept / self.amcmc_count > self.amcmc_desired_accept:
                self.sigma_MCMC *= np.exp(delta) 
            else:
                self.sigma_MCMC /= np.exp(delta)
            
            self.amcmc_count  = 0.
            self.amcmc_accept = 0.

    def set_data(self, np.ndarray n_in):
        """
            n - (dx1) numpy vector of observation count
        """
        self.n    = np.zeros(n_in.size)
        self.n[:] = n_in.flatten()[:]
        self.sum_n = np.sum(self.n)
        
        #if (self.alpha is not None) and (self.mu is not None):
        #    self.llik, self.grad, self.Hessian = self.get_f_grad_hess(self.alpha)
        #    self.L = np.linalg.cholesky(self.Hessian)