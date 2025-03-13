import torch
import torch.nn as nn
import numpy as np

def get_p_indice_mu(y, f_x, Lambdas, alpha, q):
    """
        Get the p-th indice of the \|Lambda(y - f(x))\|_q for a coverage 1-alpha.

        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            alpha (float): Coverage proportion.
            q (float): Order of the norm.

        Returns:
            (int): The p-th indice of the residuals.
    """
    p = int(alpha * y.shape[0])
    values = []

    for i in range(y.shape[0]):
        # val = torch.linalg.norm(Lambda @ (y[i] - mu), ord=q).item()
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return p_indice

def get_alpha_value_mu(y, f_x, Lambdas, alpha, q):
    """
        Get the p-th value of the \|Lambda(y - f(x))\|_q for a coverage 1-alpha.
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            p (int): Proportion of the dataset to consider.
            q (float): Order of the norm.
            
        Returns:
            (float): The p-th value of the residuals.   
    """
    p = int(alpha * y.shape[0])
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]

def get_p_value_mu(y, f_x, Lambdas, p, q):
    """
        Get the p-th value of the \|Lambda(y - f(x))\|_q for a coverage with n - p + 1 points .
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            p (int): Proportion of the dataset to consider.
            q (float): Order of the norm.
            
        Returns:
            (float): The p-th value of the residuals.   
    """
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]

    return values[p_indice]

def H(q, k):
    """
    Computes the function H(q, k) where q is a PyTorch tensor. It is the log of the volume of the unit ball in q-norm.
    
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
        q (torch.Tensor): Order of the norm (q > 0).
    
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

def calculate_averaged_volume(Lambdas, nu, q):
    """
        Calculates the average volumes of the ellipsoids in the batch.

        Parameters:
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            (float): The average volume of the ellipsoids.
    """
    all_volumes = calculate_all_volumes(Lambdas, nu, q)

    return all_volumes.mean().item()

def calculate_all_volumes(Lambdas, nu, q):
    """
        Calculates the volumes of the ellipsoids for each sample in the batch.

        Parameters:
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            torch.Tensor: The volumes of the ellipsoids for each sample in the batch - (BatchSize , ).
    """
    k = Lambdas.shape[1]
    _, logdet = torch.linalg.slogdet(Lambdas)
    volumes = - logdet + k * torch.log(nu) + H(q, k)
    
    return torch.exp(volumes)

def compute_loss(y, f_x, Lambdas, q, alpha, loss_strategy="log_volume", use_epsilon=False):
    """
        Computes the Adaptative MVCS loss function.
        
        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            q (float): Order of the norm.
            alpha (float): Proportion of the dataset to consider.
            loss_strategy (str): Strategy to compute the loss.
            use_epsilon (bool): Whether to add epsilon to the determinant.

        Returns:
            torch.Tensor: The loss value
    """
    k = y.shape[1]
    with torch.no_grad():
        idx_p = get_p_indice_mu(y, f_x, Lambdas, alpha, q)
    if loss_strategy == "exact_volume":
        det = torch.linalg.det(Lambdas)
        if use_epsilon:
            det = det + 1e-8
        # loss =  (1 / det).mean() * calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q) ** k * torch.exp(H(q, k))
        loss = (1 / det).mean() * torch.exp( k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k) )
    elif loss_strategy == "log_volume": 
        det = torch.linalg.det(Lambdas)
        if use_epsilon:
            det = det + 1e-8
        loss = torch.log( (1 / det).mean() ) + k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k)
    else:
        raise ValueError("The strategy must be either 'exact_volume' or 'log'.")
    return loss

def calculate_coverage(y, f_x, Lambdas, nu, q):
    """
        Calculates the coverage of the ellipsoids for each sample in the batch.

        Parameters:
            y (torch.Tensor): Target tensor shape (BatchSize, k).
            f_x (torch.Tensor): Predicted tensor shape (BatchSize, k).
            Lambdas (torch.Tensor): Lambda tensor shape (BatchSize, k, k).
            nu (torch.Tensor): Threshold value.
            q (torch.Tensor): Order of the norm.
    
        Returns:
            torch.Tensor: The coverage of the ellipsoids.
    """
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    count = torch.sum(values < nu)

    return count/len(values)