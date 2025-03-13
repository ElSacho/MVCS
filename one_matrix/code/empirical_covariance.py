import numpy as np
from functions import *

class EmpiricalCovariance:
    def __init__(self):
        self.q = 2
        pass

    def fit(self, residuals):
        self.mu = np.mean(residuals, axis=0)
        cov = np.linalg.inv(np.cov(residuals.T))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  
        self.Lambda = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
        self.eta = - self.Lambda @ self.mu

    def conformalize(self, residuals_calibration, alpha):
        n = residuals_calibration.shape[0]
        p = n - int(np.ceil((n+1)*(1-alpha)))
        if p < 0:
            raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")
        self.nu = get_nu_with_mu(residuals_calibration, self.Lambda, self.mu, p, self.q)
    
    def get_Lambda(self):
        return self.Lambda
    
    def get_normalized_Lambda(self):
        return self.Lambda / self.nu
    
    def get_eta(self):
        return self.eta
    
    def get_mu(self):
        return self.mu
    
    def get_nu(self):
        return self.nu
    
    def get_coverage(self, residuals_test):
        return 1 - proportion_outliers(residuals_test, self.Lambda, self.eta, self.nu, self.q)

    def get_volume(self):
        return get_volume(self.Lambda, self.nu, self.q)