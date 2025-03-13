import torch
import torch.nn as nn
import numpy as np

def calculate_out_of_box_multiple_matrices(y, f_x, rotations, tab_diag, tab_q, tab_weights, nu):
    rotated_residuals = torch.bmm(rotations, (y - f_x).unsqueeze(-1)).squeeze(-1)
    idx_Lambdas = get_idx_Lambdas(rotated_residuals, tab_weights)
    
    values = []
    for i in range(rotated_residuals.shape[0]):
        val = calculate_norm_q(tab_diag[idx_Lambdas[i]][i] @ rotated_residuals[i], tab_q[idx_Lambdas[i]]).item()
        values.append(val)
    
    values = torch.tensor(values)
    return torch.sum(values > nu) / len(values)

def get_idx_Lambdas(rotated_residuals, split):
    idx_Lambdas = split.get_idx(rotated_residuals)

    return idx_Lambdas

def get_p_value_mu_for_covariance(y, f_x, Lambdas, alpha, q):
    # Calcul des résidus
    p = int(alpha * y.shape[0])
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]


def get_p_indice(rotated_residuals, tab_diags, tab_q, tab_weights, alpha):
    # Calcul des résidus
    p = int(alpha * rotated_residuals.shape[0])

    idx_Lambdas = get_idx_Lambdas(rotated_residuals, tab_weights)
    
    values = []

    for i in range(rotated_residuals.shape[0]):
        val = calculate_norm_q(tab_diags[idx_Lambdas[i]][i] @ rotated_residuals[i], tab_q[idx_Lambdas[i]]).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]
    idx_Lambda = idx_Lambdas[p_indice]

    return p_indice, idx_Lambda

def get_p_value_multiple_Lambdas(y, f_x, rotations, tab_diag, tab_q, tab_weights, alpha):
    rotated_residuals = torch.bmm(rotations, (y - f_x).unsqueeze(-1)).squeeze(-1)
    idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, alpha)
    
    sigma_p = calculate_norm_q(tab_diag[idx_Lambda][idx_p] @ rotated_residuals[idx_p], tab_q[idx_Lambda])
    
    return sigma_p
    
def compute_loss_weighted_ellipsoids(y, f_x, rotations, tab_diag, tab_q, tab_weights, split, alpha, k, strategy="log"):
    if strategy == "log":
        residuals = y - f_x
        rotated_residuals = torch.bmm(rotations, residuals.unsqueeze(-1)).squeeze(-1) 

        with torch.no_grad():
            idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, alpha)
        
        diag_p = tab_diag[idx_Lambda][idx_p]
        sigma_p = calculate_norm_q( diag_p @ rotated_residuals[idx_p], tab_q[idx_Lambda])

        tab_volume = []    
        for i, diag in enumerate(tab_diag):
            dets = torch.linalg.det(diag)
            volumes =  1 / dets * torch.exp(H(tab_q[i], k))
            tab_volume.append(volumes.mean())
        
        tab_volume = torch.stack(tab_volume)
        
        values = tab_weights * tab_volume
        loss = torch.log(torch.sum(values)) + k * torch.log(sigma_p)
        return loss

    residuals = y - f_x
    rotated_residuals = torch.bmm(rotations, residuals.unsqueeze(-1)).squeeze(-1) 

    with torch.no_grad():
        idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, alpha)
    
    diag_p = tab_diag[idx_Lambda][idx_p]
    sigma_p = calculate_norm_q( diag_p @ rotated_residuals[idx_p], tab_q[idx_Lambda])

    tab_volume = []
    
    for i, diag in enumerate(tab_diag):
        dets = torch.linalg.det(diag)
        volumes =  sigma_p**k / dets * torch.exp(H(tab_q[i], k))
        tab_volume.append(volumes.mean())
    
    tab_volume = torch.stack(tab_volume)
    
    values = tab_weights * tab_volume
    loss = torch.sum(values)
    return loss

def compute_volume_weighted_ellipsoids(tab_diag, tab_q, tab_weights, nu, k):
    tab_volume = []
    
    for i, diag in enumerate(tab_diag):
        dets = torch.linalg.det(diag)
        volumes =  1 / dets * torch.exp(H(tab_q[i], k))
        tab_volume.append(volumes.mean())
    
    tab_volume = torch.stack(tab_volume)
    
    values = tab_weights * tab_volume
    volume = torch.sum(values) * torch.pow(nu, k)
    return volume

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

def calculate_volume(Lambdas, nu, q, k):
    _, logdet = torch.linalg.slogdet(Lambdas)
    volume = - logdet.mean() + k * torch.log(nu) + H(q, k)
    
    return np.exp(volume.item())


def calculate_out_of_box(y, f_x, Lambdas, nu, q):
    # Calcul des résidus
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    count = torch.sum(values > nu)

    return count/len(values)