import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

import numpy as np
from functions import *


def get_p_indice_mu(y, mu, Lambda, p, q):
    # Calcul des résidus
    values = []

    for i in range(y.shape[0]):
        # val = torch.linalg.norm(Lambda @ (y[i] - mu), ord=q).item()
        val = calculate_norm_q(Lambda @ (y[i] - mu), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return p_indice


def get_p_value_mu(y, mu, Lambda, p, q):
    # Calcul des résidus
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambda @ (y[i] - mu), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]


def H(q, k):
    """
    Computes the function H(q, k) where q is a PyTorch tensor.
    
    Parameters:
        q (torch.Tensor): Input tensor for q.
        k (float or torch.Tensor): Scalar or tensor for k.
    
    Returns:
        torch.Tensor: Result of the computation.
    """
    term1 = k * torch.special.gammaln(1 + 1 / q)
    term2 = torch.special.gammaln(1 + k / q)
    term3 = k * torch.log(torch.tensor(2.0))
    return term1 - term2 + term3

def calculate_norm_q(z, q):
    """
    Calculates the q-norm of a vector z manually in PyTorch.
    
    Parameters:
        z (torch.Tensor): Input vector (1D tensor).
        q (float): Order of the norm (q > 0).
    
    Returns:
        torch.Tensor: The q-norm of the input vector.
    """
    if q <= 0:
        raise ValueError("The order of the norm (q) must be greater than 0.")
    
    # Compute the absolute values of the vector elements raised to the power q
    abs_powers = torch.abs(z) ** q
    
    # Sum up the values
    sum_abs_powers = torch.sum(abs_powers)
    
    # Take the q-th root
    norm_q = sum_abs_powers ** (1 / q)
    
    return norm_q


def compute_loss(y, mu, Lambda, q, p, k):
    with torch.no_grad():
        idx_p = get_p_indice_mu(y, mu, Lambda, p, q)
    loss = - torch.linalg.slogdet(Lambda)[1] + calculate_norm_q(Lambda @ (y[idx_p] - mu), q) + H(q, k)
    return loss


class TorchMinimizer:
    def __init__(self):
        pass

    def fit(self, residuals, alpha, stop_residuals = None, lr=0.01, num_epochs=1000, verbose = 0):

        p = int(residuals.shape[0]*alpha)
        
        residuals_tensor = torch.tensor(residuals, dtype=torch.float32)
        if stop_residuals is not None:
            stop_residuals_tensor = torch.tensor(stop_residuals, dtype=torch.float32)
        
        k = residuals.shape[1]
        mu = nn.Parameter(torch.randn(k, requires_grad=True))
        mat_A = nn.Parameter(torch.randn(k, k, requires_grad=True))
        q = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)

        if verbose==1:
            print_every = num_epochs // 10
        elif verbose==2:
            print_every = 1

        # Optimiseur
        optimizer = optim.SGD([mu, mat_A, q], lr=lr)
        tab_loss = []
        tab_q = []
        best_loss = np.inf
        for epoch in range(num_epochs):
            # Reset des gradients
            optimizer.zero_grad()
            Lambda = mat_A @ mat_A.T
            
            loss = compute_loss(residuals_tensor, mu, Lambda, q, p, k)
            tab_loss.append(loss.item())
            tab_q.append(q.item())
            
            loss.backward()
            optimizer.step()

            if stop_residuals is not None:
                with torch.no_grad():
                    stop_loss = compute_loss(residuals_tensor, mu, Lambda, q, p, k)

                if stop_loss.item() < best_loss:
                    best_mu = copy.deepcopy(mu)
                    best_mat_A = copy.deepcopy(mat_A)
                    best_Lambda = best_mat_A @ best_mat_A.T
                    best_q = copy.deepcopy(q)
                    best_loss = stop_loss.item()

            else:
                if loss.item() < best_loss:
                    best_mu = copy.deepcopy(mu)
                    best_mat_A = copy.deepcopy(mat_A)
                    best_Lambda = best_mat_A @ best_mat_A.T
                    best_q = copy.deepcopy(q)
                    best_loss = loss.item()
            
            if verbose!=0 and epoch % print_every == 0:
                if stop_residuals is not None:
                    print(f"Epoch {epoch}: Loss = {loss.item()} - Stop loss = {stop_loss.item()}")
                else:
                    print(f"Epoch {epoch}: Loss = {loss.item()}")
            
        if verbose != 0:
            print(f"Finished - Best loss: {best_loss}")

        self.Lambda = best_Lambda.detach().numpy()
        self.mu = best_mu.detach().numpy()
        self.eta = - self.Lambda @ self.mu
        self.q = best_q.detach().numpy()
        self.tab_loss = tab_loss
        self.tab_q = tab_q

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
        return  get_volume(self.Lambda, self.nu, self.q)