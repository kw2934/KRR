import numpy as np
from sklearn.kernel_ridge import KernelRidge

def f_star(x, fcn): # get true regression function
    if fcn == 'C':
        return np.cos(x * 2 * np.pi) - 1
    if fcn == 'S':
        return np.sin(x * 2 * np.pi)    
    if fcn == 'V':
        return np.abs(x - 1/2) - 1/2
    if fcn == 'W':
        f_1 = np.clip( np.abs(4 * x - 1), 0, 1)
        f_2 = np.clip( np.abs(4 * x - 3), 0, 1)
        return f_1 + f_2 - 2
    if fcn == 'x':
        return x * np.sin(4 * np.pi * x)


def Kernel(Z_1, Z_2): # compute kernel matrix given two sets of covariates
    n_1, n_2 = len(Z_1), len(Z_2)
    K = np.minimum(np.ones((n_1, 1)) @ Z_2.reshape(1, -1), Z_1.reshape(-1, 1) @ np.ones((1, n_2)) )
    return K / n_2


class KRR_covariate_shift:
    def __init__(self, n, n_0, B, sigma, fcn, seed): # preparations, sample generation
        assert B >= 1
        np.random.seed(seed)        
        self.B = B
        
        # source data
        self.n = n        
        self.fcn = fcn
        tmp = int(n * self.B / (self.B + 1))
        self.X = np.concatenate((np.random.rand(tmp) / 2, 1/2 + np.random.rand(n - tmp) / 2))
        np.random.shuffle(self.X)
        self.y = f_star(self.X, self.fcn) + sigma * np.random.randn(n)
        
        # target data
        tmp = int(n_0 * self.B / (self.B + 1))
        self.X_0 = np.concatenate((np.random.rand(n_0 - tmp) / 2, 1/2 + np.random.rand(tmp) / 2))
        self.y_0 = f_star(self.X_0, self.fcn) # noiseless responses
    
    
    def estimate(self, rho = 0.2, beta = 2): # KRR under covariate shift
        assert beta > 1
        assert 0 < rho and rho < 1
        
        # data splitting
        self.n_1 = int( (1 - rho) * self.n )
        self.n_2 = self.n - self.n_1
        self.X_1, self.y_1 = self.X[0:self.n_1], self.y[0:self.n_1]
        self.X_2, self.y_2 = self.X[self.n_1:self.n], self.y[self.n_1:self.n]        
        
        # penalty parameters: one for imputation, a geometric sequence for training
        lbd_tilde = 0.1 / self.n_2 # for the imputation model
        lbd_min, lbd_max = 0.1 / self.n_1, 1 # min and max for training
        m = np.log(lbd_max / lbd_min) / np.log(beta)
        m = max( int(np.ceil(m)) , 2 ) + 1
        self.Lambda = lbd_min * ( beta ** np.array( range(m) ) ) # for training

        # pseudo-labeling (call the KRR solver in sklearn)
        krr = KernelRidge(kernel = 'precomputed', alpha = lbd_tilde)
        krr.fit( Kernel(self.X_2, self.X_2) , self.y_2)
        self.alpha_tilde = krr.dual_coef_
        self.y_tilde = Kernel(self.X_0, self.X_2) @ self.alpha_tilde

        
        # training (call the KRR solver in sklearn)
        self.Alpha = np.zeros((m, self.n_1))
        self.err_est_naive = np.zeros(m)  
        self.err_est_pseudo = np.zeros(m)
        self.err_est_real = np.zeros(m)    
        for (j, lbd) in enumerate(self.Lambda):
            # KRR
            krr = KernelRidge(kernel = 'precomputed', alpha = lbd)
            krr.fit( Kernel(self.X_1, self.X_1) , self.y_1)
            self.Alpha[j] = krr.dual_coef_
            
            # naive estimate of loss (using source data)
            self.err_est_naive[j] = np.mean( (Kernel(self.X_2, self.X_1) @ self.Alpha[j] - self.y_2) ** 2 )
            
            # pseudo and real labels
            y_lbd = Kernel(self.X_0, self.X_1) @ self.Alpha[j]
            self.err_est_pseudo[j] = np.mean( (y_lbd - self.y_tilde) ** 2 )
            self.err_est_real[j] = np.mean( (y_lbd - self.y_0) ** 2 )
            
        
        # selection
        self.j_naive = np.argmin(self.err_est_naive)
        self.lbd_naive = self.Lambda[self.j_naive]
        self.alpha_naive = self.Alpha[self.j_naive]        
       
        self.j_pseudo = np.argmin(self.err_est_pseudo)
        self.lbd_pseudo = self.Lambda[self.j_pseudo]
        self.alpha_pseudo = self.Alpha[self.j_pseudo] 
        
        self.j_real = np.argmin(self.err_est_real)
        self.lbd_real = self.Lambda[self.j_real]
        self.alpha_real = self.Alpha[self.j_real]

        
    def evaluate(self, N_test, seed): # evaluate MSE on the target distribution using newly generated samples
        np.random.seed(seed)
        tmp = int(N_test * self.B / (self.B + 1))
        self.X_test_0 = np.concatenate((np.random.rand(N_test - tmp) / 2, 1/2 + np.random.rand(tmp) / 2))
        K_test = Kernel(self.X_test_0, self.X_1)
        self.y_true_test_0 = f_star(self.X_test_0, self.fcn)
        
        self.y_naive = K_test @ self.alpha_naive
        self.err_naive = np.mean( (self.y_true_test_0 - self.y_naive) ** 2 )
        
        self.y_pseudo = K_test @ self.alpha_pseudo
        self.err_pseudo = np.mean( (self.y_true_test_0 - self.y_pseudo) ** 2 )

        self.y_real = K_test @ self.alpha_real
        self.err_real = np.mean( (self.y_true_test_0 - self.y_real) ** 2 )        

    
    def predict(self, X_new): # make predictions given covariates
        self.y_new_true = f_star(X_new, self.fcn)
        self.y_new_tilde = Kernel(X_new, self.X_2) @ self.alpha_tilde

        K = Kernel(X_new, self.X_1)
        self.y_new_naive = K @ self.alpha_naive
        self.y_new_pseudo = K @ self.alpha_pseudo
        self.y_new_real = K @ self.alpha_real

        m = len(self.Lambda)
        self.y_new_candidates = np.zeros((m, len(X_new)))
        for j in range(m):
            self.y_new_candidates[j] = K @ self.Alpha[j]


