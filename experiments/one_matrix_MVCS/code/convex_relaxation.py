import cvxpy as cp
import numpy as np
from functions import *



def minimize_convex_relaxation(y, p, q):
    # Variables
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y

    Lambda = cp.Variable((k, k), PSD=True)  
    eta = cp.Variable(k)
    nu = cp.Variable()  

    constraints = []
    constraints.append(Lambda >> 0)
    sigma_p = 0

    for i in range(n):
        sigma_p += cp.pos( cp.norm(Lambda @ y[i, :] + eta, q) - nu )
    
    objective = - cp.log_det(Lambda) + sigma_p + p * nu
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    if problem.status != cp.OPTIMAL:
        print("Problem not solved")
        print(problem.status)
        print(problem.value)

    return Lambda.value, eta.value, nu.value, problem.value


class ConvexRelaxation:
    def __init__(self):
        pass

    def fit(self, residuals, alpha, q):
        p = int(int(residuals.shape[0] * alpha))
        self.q = q
        self.Lambda, self.eta, radius, self.obj = minimize_convex_relaxation(residuals, p, q)
        self.radius = radius
        self.mu = - np.linalg.inv(self.Lambda) @ self.eta
    
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