import cvxpy as cp
import numpy as np
from scipy.special import gammaln
from functions import *

def order_points(y, Lambda, eta, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    
    return y


def linearization(y, Lambda, eta, Lambda_old, eta_old, p):
    """
    Linearize f(Lambda, eta) = ||Lambda y + eta||_p around (Lambda_old, eta_old).

    Parameters:
    y : numpy.ndarray
        Input vector y.
    Lambda : numpy.ndarray
        Current Lambda matrix.
    eta : numpy.ndarray
        Current eta vector.
    Lambda_old : numpy.ndarray
        Previous Lambda matrix (point of linearization).
    eta_old : numpy.ndarray
        Previous eta vector (point of linearization).
    p : float
        p-norm parameter.

    Returns:
    cp.Expression
        Linearized function.
    """
    # Compute v_old = Lambda_old @ y + eta_old
    v_old = Lambda_old @ y + eta_old

    # Compute the norm of v_old
    norm_v_old = np.linalg.norm(v_old, ord=p)

    # Avoid division by zero
    if norm_v_old == 0:
        raise ValueError("norm_v_old is zero, linearization is not well-defined.")

    # Compute the gradient components
    abs_v_old = np.abs(v_old)
    sign_v_old = np.sign(v_old)
    grad_factor = abs_v_old ** (p - 1) / (norm_v_old ** (p - 1))
    grad_eta = grad_factor * sign_v_old  # Gradient w.r.t eta
    grad_Lambda = np.outer(grad_factor * sign_v_old, y)  # Gradient w.r.t Lambda

    # Linearized function
    linearized_f = (
        cp.trace(grad_Lambda.T @ (Lambda - Lambda_old)) +  # Linear term for Lambda
        grad_eta.T @ (eta - eta_old)  # Linear term for eta
    )

    return linearized_f

def H(q, k):
    term1 = k * gammaln(1 + 1/q)  
    term2 = gammaln(1 + k/q)      
    term3 = k * np.log(2)
    return term1 - term2 + term3

def calculate_true_objective(y, Lambda, eta, p, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    nu = distances[sorted_indices[p-1]]
    
    obj = - np.log(np.linalg.det(Lambda)) + nu

    return obj

def calculate_volume(y, Lambda, eta, p, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    nu = distances[sorted_indices[p-1]]
    
    volume = - np.log(np.linalg.det(Lambda)) + k * np.log(nu) + np.exp(H(q, k))
    
    return np.exp(volume)

def get_nu_with_mu(y, Lambda, mu, p, q):
    n = len(y) 
    k = len(y[0])
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ (y[i, :] - mu) , ord=q)
    
    sorted_indices = np.argsort(distances)[::-1]
    nu = distances[sorted_indices[p-1]]

    return nu



def one_step_DC(y, Lambda_old, eta_old, p, q):
    # Variables
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y

    y = order_points(y, Lambda_old, eta_old, q)
    
    Lambda = cp.Variable((k, k), PSD=True)
    eta = cp.Variable(k)
    nu = cp.Variable()  

    Lambda.value = Lambda_old
    eta.value = eta_old

    constraints = []
    constraints.append(Lambda >> 0)
    
    sigma_p = 0
    sigma_p_minus_one = 0

    for i in range(n):
        sigma_p += cp.pos( cp.norm(Lambda @ y[i, :] + eta, q) - nu )
        if i < p - 1:
            sigma_i = linearization(y[i], Lambda, eta, Lambda_old, eta_old, q)
            sigma_p_minus_one += sigma_i
    
    objective = - cp.log_det(Lambda) + sigma_p + p * nu - sigma_p_minus_one
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    if problem.status != cp.OPTIMAL:
        print("Problem not solved")
        print(problem.status)
        print(problem.value)

    return Lambda.value, eta.value, nu.value, problem.value


def one_step_DC_one_p(y, Lambda_old, eta_old, p, q):
    # Variables
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y

    y = order_points(y, Lambda_old, eta_old, q)
    
    Lambda = cp.Variable((k, k), PSD=True)
    eta = cp.Variable(k)
    nu = cp.Variable()  

    Lambda.value = Lambda_old
    eta.value = eta_old

    constraints = []
    constraints.append(Lambda >> 0)
    
    sigma_p = cp.norm(Lambda @ y[p-1, :] + eta, q)

    objective = - cp.log_det(Lambda) + sigma_p 

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    if problem.status != cp.OPTIMAL:
        print("Problem not solved")
        print(problem.status)
        print(problem.value)

    return Lambda.value, eta.value, nu.value, problem.value

def DC_algorithm(y, p, q, beta, max_iter=100, tol=1e-6, verbose=False):
    # Initialization
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y
    
    # Initial values
    mu_old = np.mean(y, axis=0)
    M_old = np.linalg.inv(np.cov(y.T))
    eigenvalues, eigenvectors = np.linalg.eigh(M_old)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  
    Lambda_old = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    eta_old = - np.linalg.inv(Lambda_old) @ mu_old

    center = mu_old
    Lambda = Lambda_old
    vol_old = np.inf
    obj_old = np.inf
    true_obj_old = np.inf

    beta_0 = beta
    v_Lambda_k = Lambda_old
    v_eta_k = eta_old
    
    for i in range(max_iter):
        
        Lambda, eta, nu, obj = one_step_DC(y, Lambda_old, eta_old, p, q)
        vol = calculate_volume(y, Lambda, eta, p, q)
        true_obj = calculate_true_objective(y, Lambda, eta, p, q)
        
        if verbose:
            # print(f"Iteration {i}, obj = {obj}")
            print(f"Volume = {vol}")
            print(f"True Objective = {true_obj}")

        if np.abs(true_obj - true_obj_old) < tol:
            break

        beta = beta_0 / ( i + 1)
        v_Lambda_k = beta*v_Lambda_k + (1 - beta) * (Lambda - Lambda_old)
        v_eta_k = beta*v_eta_k + (1 - beta) * (eta - eta_old)
        Lambda = Lambda_old + v_Lambda_k
        eta = eta_old + v_eta_k
        
        # Update the values
        Lambda_old = Lambda
        eta_old = eta
        true_obj_old = true_obj
        
    return Lambda, eta, nu, obj


class DC:
    def __init__(self):
        pass

    def fit(self, residuals, alpha, q, beta = 0.0, max_iter=100, tol=1e-6, verbose=False):
        p = int(int(residuals.shape[0] * alpha))
        self.q = q
        self.Lambda, self.eta, radius, self.obj = DC_algorithm(residuals, p, q, beta, max_iter, tol, verbose)
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