import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functions import get_volume

import copy

from network import *


def calculate_out_of_box_multiple_matrices(y, f_x, rotations, tab_diag, tab_q, tab_weights, split, nu):
    rotated_residuals = torch.bmm(rotations, (y - f_x).unsqueeze(-1)).squeeze(-1)
    idx_Lambdas = get_idx_Lambdas(rotated_residuals, split)
    
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


def get_p_indice(rotated_residuals, tab_diags, tab_q, tab_weights, split, alpha):
    # Calcul des résidus
    p = int(alpha * rotated_residuals.shape[0])

    idx_Lambdas = get_idx_Lambdas(rotated_residuals, split)
    
    values = []

    for i in range(rotated_residuals.shape[0]):
        val = calculate_norm_q(tab_diags[idx_Lambdas[i]][i] @ rotated_residuals[i], tab_q[idx_Lambdas[i]]).item()
        values.append(val)

    values = torch.tensor(values)
    sorted_indices = torch.argsort(values, descending=True)
    p_indice = sorted_indices[p]
    idx_Lambda = idx_Lambdas[p_indice]

    return p_indice, idx_Lambda

def get_p_value_multiple_Lambdas(y, f_x, rotations, tab_diag, tab_q, tab_weights, split, alpha):
    rotated_residuals = torch.bmm(rotations, (y - f_x).unsqueeze(-1)).squeeze(-1)
    idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, split, alpha)
    
    sigma_p = calculate_norm_q(tab_diag[idx_Lambda][idx_p] @ rotated_residuals[idx_p], tab_q[idx_Lambda])
    
    return sigma_p
    
def compute_loss_weighted_ellipsoids(y, f_x, rotations, tab_diag, tab_q, tab_weights, split, alpha, k, strategy="log"):
    if strategy == "log":
        residuals = y - f_x
        rotated_residuals = torch.bmm(rotations, residuals.unsqueeze(-1)).squeeze(-1) 

        with torch.no_grad():
            idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, split, alpha)
        
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
        idx_p, idx_Lambda = get_p_indice(rotated_residuals, tab_diag, tab_q, tab_weights, split, alpha)
    
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



class Split:
    def __init__(self, k, n_splits):
        """
        Splits the space along `k` axes from the canonical basis into 2^n_splits subspaces.
        :param k: Number of axes to split the space along.
        :param n_splits: Number of binary splits.
        """
        if n_splits > k:
            raise ValueError("n_splits cannot exceed k")

        # Use first k axes from the canonical basis (e.g., e_1 = (1, 0, 0), e_2 = (0, 1, 0), ...)
        axes = torch.eye(k)
        self.splits = axes[:n_splits]

    def get_idx(self, points):
        """
        Returns the indices of the subspaces for the given points.
        :param points: A 2D tensor where each row is a point in the space.
        :return: A list of indices corresponding to the subspaces.
        """
        if len(self.splits) == 0:
            raise ValueError("Space has not been split yet. Call split_space first.")

        points = torch.tensor(points, dtype=torch.float32)  # Ensure points are float tensors
        indices = []

        # Compute binary indices based on the sign of the dot product
        for point in points:
            binary_idx = 0
            for i, axis in enumerate(self.splits):
                if torch.dot(point, axis) >= 0:
                    binary_idx |= (1 << i)
            indices.append(binary_idx)

        return indices






class HalfEllipsoidPredictor:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.tab_q = [torch.tensor(2.0, requires_grad=True) for _ in range(2**self.n_splits)]
        

    def fit(self, y_train, alpha, 
            y_stop=None,
            num_epochs_warm_start = 10,
            warm_start_epochs_mat = 0,
            num_epochs_our_loss = 500,
            lr_warm_start = 0.001,
            lr_model = 0.001,
            lr_q = 0.01,
            lr_matrix_model = 0.001,
            batch_size_warm_start = 32,
            batch_size_our_loss = 1_000,
            use_lr_scheduler = False,
            hidden_dim = 5, 
            hidden_dim_matrix = 100,
            verbose = 0
        ):

        x_train = np.ones_like(y_train)
        if y_stop is None:
            y_stop = y_train
        x_stop = np.ones_like(y_stop)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_train = x_train
        self.y_train = y_train
        x_stop_tensor = torch.tensor(x_stop, dtype=torch.float32)
        y_stop_tensor = torch.tensor(y_stop, dtype=torch.float32)

        k = y_train.shape[1]
        d = x_train.shape[1]
        self.k = y_train.shape[1]

        rotationMatrix_model = ParametricRotationMatrix2D(d, k, hidden_dim=hidden_dim_matrix)
        model_half = Network(d, k, hidden_dim=hidden_dim, n_hidden_layers=1)
        model_diag = DiagNetwork(d, k, hidden_dim=hidden_dim)

        self.model = model_half
        self.tab_diag_models = [copy.deepcopy(model_diag) for _ in range(2**self.n_splits)]
        self.rotationMatrix_model = rotationMatrix_model

        self.alpha = alpha
    
        self.split = Split(self.k , int(self.n_splits))
        self.tab_weights = torch.tensor([1/len(self.tab_diag_models) for _ in range(len(self.tab_diag_models))], dtype=torch.float32)

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_warm_start, shuffle=True)
        stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size_warm_start, shuffle=True)
        train_losses, stop_losses = self.model.fit_and_plot(dataloader, stoploader, epochs=num_epochs_warm_start, lr=lr_warm_start)
        
        self.warm_start_model = copy.deepcopy(self.model)
        
        self.warm_start_train_losses = train_losses
        self.warm_start_stop_losses = stop_losses
        
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_our_loss, shuffle=True)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': lr_model},  # Learning rate for the main model
            {'params': [param for diag_model in self.tab_diag_models for param in diag_model.parameters()], 'lr': lr_matrix_model},  # Learning rate for the matrix model
            {'params': self.tab_q, 'lr': lr_q}  # Learning rate for q
        ])

        k = y_train.shape[1]

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_our_loss)

        tab_loss = []
        tab_loss_stop = []
        tab_stop_loss = []
        
        print_every = num_epochs_our_loss // 10
        best_stop_loss = np.inf

        if verbose ==1:
            print_every = num_epochs_our_loss // 10
        elif verbose == 2:
            print_every = 1

        for epoch in range(num_epochs_our_loss):
            epoch_loss = 0.0
            for x, y in dataloader:
                optimizer.zero_grad()
                rotations = self.rotationMatrix_model(x)

                tab_diag_matrices = []
                for diag_model in self.tab_diag_models:
                    diag = diag_model(x)
                    diag_matrices = torch.stack([torch.diag(d) for d in diag], dim=0)
                    tab_diag_matrices.append(diag_matrices)
                
                if epoch < warm_start_epochs_mat:
                    with torch.no_grad():
                        f_x = self.model( x )
                
                else: f_x = self.model( x )

                loss = compute_loss_weighted_ellipsoids(y, f_x, rotations, tab_diag_matrices, self.tab_q, self.tab_weights, self.split, alpha, k, strategy="log")    
                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            tab_loss.append(epoch_loss/len(dataloader))


            if x_stop is not None:
                with torch.no_grad():
                        rotations = self.rotationMatrix_model(x_stop_tensor)

                        tab_diag_matrices = []
                        for diag_model in self.tab_diag_models:
                            diag = diag_model(x_stop_tensor)
                            diag_matrices = torch.stack([torch.diag(d) for d in diag], dim=0)
                            tab_diag_matrices.append(diag_matrices)
                        
                        f_x = self.model( x_stop_tensor )

                        stop_loss = compute_loss_weighted_ellipsoids(y_stop_tensor, f_x, rotations, tab_diag_matrices, self.tab_q, self.tab_weights, self.split, alpha, k, strategy="log")    

                tab_stop_loss.append(stop_loss.item())

                if stop_loss.item() < best_stop_loss:
                    best_stop_loss = stop_loss.item()
                    best_model = copy.deepcopy(self.model)
                    best_diag_models = [copy.deepcopy(diag_model) for diag_model in self.tab_diag_models]
                    best_q = copy.deepcopy(self.tab_q)
                    best_rotationMatrix_model = copy.deepcopy(self.rotationMatrix_model)

            
                if verbose !=0 and epoch % print_every == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item()} - Stop Loss : {stop_loss.item()} - Best stop loss = {best_stop_loss}")

            else:
                if loss.item() < best_stop_loss:
                    best_stop_loss = loss.item()
                    best_model = copy.deepcopy(self.model)
                    best_diag_models = [copy.deepcopy(diag_model) for diag_model in self.tab_diag_models]
                    best_q = copy.deepcopy(self.tab_q)
                    best_rotationMatrix_model = copy.deepcopy(self.rotationMatrix_model)
                if verbose !=0 and epoch % print_every == 0:
                    print(f"Epoch {epoch}: Loss = {loss.item()} - Best loss = {best_stop_loss}")
                    
            if use_lr_scheduler:
                scheduler.step()

        if verbose!=0:
            if x_stop is not None:
                print(f"Last epoch {epoch}: Loss = {loss.item()} - Stop Loss : {stop_loss.item()} - Best stop loss = {best_stop_loss}")
            else:
                print(f"Last epoch {epoch}: Loss = {loss.item()} - Best loss = {best_stop_loss}")


        self.mu = best_model(x_train_tensor)[0].detach().numpy()
        self.tab_diag = [torch.diag(diag_model(x_train_tensor)[0]).detach().numpy() for diag_model in best_diag_models]
        self.tab_q = [q.item() for q in best_q]
        self.rotationMatrix = best_rotationMatrix_model(x_train_tensor)[0].detach().numpy()

        self.model = best_model
        self.tab_diag_models = best_diag_models
        self.rotationMatrix_model = best_rotationMatrix_model

        self.tab_loss = tab_loss
        self.tab_loss_stop = tab_loss_stop
        
    def conformalize_ellipsoids(self, y_calibration, alpha):
        x_calibration = np.ones_like(y_calibration)
        x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float32)
        y_calibration_tensor = torch.tensor(y_calibration, dtype=torch.float32)

        f_x_calibration = self.model(x_calibration_tensor)

        rotations_calibration = self.rotationMatrix_model(x_calibration_tensor)
        tab_diag_matrices_calibration = []
        for diag_model in self.tab_diag_models:
            diag = diag_model(x_calibration_tensor)
            diag_matrices = torch.stack([torch.diag(d) for d in diag], dim=0)
            tab_diag_matrices_calibration.append(diag_matrices)

        self.nu = get_p_value_multiple_Lambdas(y_calibration_tensor, f_x_calibration, rotations_calibration, tab_diag_matrices_calibration, self.tab_q, self.tab_weights, self.split, alpha).item()

        return self.nu
    
    def get_volume(self):
        volume = 0
        for idx, diag in enumerate(self.tab_diag):
            volume += get_volume(diag, self.nu, self.tab_q[idx])
        return volume/len(self.tab_diag)
            

        
    def get_volume_and_coverage(self, y_test, x_test=None):
        if x_test is None:
            x_test = np.ones_like(y_test)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        f_x_test = self.model(x_test_tensor)

        rotations_test = self.rotationMatrix_model(x_test_tensor)
 
        tab_diag_matrices_test = []
        for diag_model in self.tab_diag_models:
            diag = diag_model(x_test_tensor)
            diag_matrices = torch.stack([torch.diag(d) for d in diag], dim=0)
            tab_diag_matrices_test.append(diag_matrices)
        
        volume = compute_volume_weighted_ellipsoids(tab_diag_matrices_test, self.tab_q, self.tab_weights, self.nu_conformal, self.k)
        out_of_box = calculate_out_of_box_multiple_matrices(y_test_tensor, f_x_test, rotations_test, tab_diag_matrices_test, self.tab_q, self.tab_weights, self.split, self.nu_conformal).item()        
        
        return volume, out_of_box
    
