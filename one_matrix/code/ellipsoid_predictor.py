import numpy as np
from utils import *
from functions import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from network import *


def get_p_indice_mu(y, f_x, Lambdas, alpha, q):
    # Calcul des résidus
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


def get_p_value_mu(y, f_x, Lambdas, alpha, q):
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

def calculate_averaged_volume(Lambdas, nu, q, k):
    volumes = calculate_all_volumes(Lambdas, nu, q, k)
    return volumes.mean().item()

def calculate_all_volumes(Lambdas, nu, q, k, replace_inf=True):
    _, logdet = torch.linalg.slogdet(Lambdas)
    volumes = - logdet + k * torch.log(nu) + H(q, k)
    volumes = torch.exp(volumes)
    if replace_inf:
        max_value = torch.max(volumes[volumes != float('inf')])
        volumes[volumes == float('inf')] = max_value
    return volumes

def compute_loss(y, f_x, Lambdas, q, alpha, k, strategy="real"):
    with torch.no_grad():
        idx_p = get_p_indice_mu(y, f_x, Lambdas, alpha, q)
    
    if strategy == "log":
        _, logdet = torch.linalg.slogdet(Lambdas)
        loss = - logdet.mean() + k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k)
    else:
        det = torch.linalg.det(Lambdas)
        loss = torch.log( (1 / det).mean() ) + k * torch.log(calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q)) + H(q, k)
        # raise ValueError("Unknown strategy")
        # loss = - logdet.mean() + calculate_norm_q(Lambdas[idx_p] @ (y[idx_p] - f_x[idx_p]), q) + H(q, k)
    return loss


def calculate_out_of_box(y, f_x, Lambdas, nu, q):
    # Calcul des résidus
    values = []

    for i in range(y.shape[0]):
        val = calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), q).item()
        values.append(val)

    values = torch.tensor(values)
    count = torch.sum(values > nu)

    return count/len(values)


class EllipsoidPredictor:
    def __init__(self):
        self.nu = None

    def fit(self,y_train, alpha,
            y_stop = None,
            num_epochs_warm_start = 1000,
            warm_start_epochs_mat = 50,
            num_epochs_our_loss = 500,
            lr_warm_start = 0.001,
            lr_model = 0.001,
            lr_q = 0.01,
            lr_matrix_model = 0.001,
            batch_size_warm_start = 32,
            batch_size_our_loss = 1_000,
            loss_strategy = "log",
            use_lr_scheduler = False,
            keep_best = True,
            hidden_dim = 128,
            hidden_dim_matrix = 128,
            n_hidden_layers_matrix = 1,
            verbose = False
        ):

        self.alpha = alpha
        

        x_train = np.ones_like(y_train)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_train = x_train
        self.y_train = y_train

        if y_stop is not None:
            x_stop = np.ones_like(y_stop)
            x_stop_tensor = torch.tensor(x_stop, dtype=torch.float32)
            y_stop_tensor = torch.tensor(y_stop, dtype=torch.float32)

        d = x_train.shape[1]
        self.k = y_train.shape[1]
        k = y_train.shape[1]

        model = Network(d, k, hidden_dim=hidden_dim, n_hidden_layers=1)
        matrix_model = MatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix, n_hidden_layers=n_hidden_layers_matrix)
        self.q = torch.tensor(2.0, requires_grad=True)
        self.model = model
        self.matrix_model = matrix_model

        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_warm_start, shuffle=True)
        if y_stop is not None:
            stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size_warm_start, shuffle=True)
            train_losses, stop_losses = self.model.fit_and_plot(dataloader, stoploader, epochs=num_epochs_warm_start, lr=lr_warm_start, keep_best=keep_best)
        else:
            train_losses, _ = self.model.fit_and_plot(dataloader, dataloader, epochs=num_epochs_warm_start, lr=lr_warm_start, keep_best=keep_best)
        
        self.warm_start_model = copy.deepcopy(self.model)
        
        self.warm_start_train_losses = train_losses
        if y_stop is not None:
            self.warm_start_stop_losses = stop_losses

        
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_our_loss, shuffle=True)

        # optimizer = optim.Adam([*self.model.parameters(), *self.matrix_model.parameters(), q], lr=lr_our_loss)

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': lr_model},  # Learning rate for self.model
            {'params': self.matrix_model.parameters(), 'lr': lr_matrix_model},  # Learning rate for self.matrix_model
            {'params': self.q, 'lr': lr_q}  # Learning rate for q
        ])

        k = y_train.shape[1]

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_our_loss)

        tab_loss = []
        tab_loss_calibration = []
        tab_q = []
        best_stop_loss = np.inf

        last_model_weight = self.model.state_dict()
        last_lambdas_weight = self.matrix_model.state_dict()
        last_q = self.q.item()

        if verbose != 0:
            print_every = num_epochs_our_loss // 10

        for epoch in range(num_epochs_our_loss):
            for x, y in dataloader:
                optimizer.zero_grad()
                mat_A = self.matrix_model(x)
                Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
                if epoch < warm_start_epochs_mat:
                    with torch.no_grad():
                        f_x = self.model( x )
                
                else: f_x = self.model( x )

                loss = compute_loss(y, f_x, Lambdas, self.q, alpha, k, strategy=loss_strategy)
                tab_q.append(self.q.item())
                loss.backward()
                optimizer.step()
                
                if torch.isnan(loss):
                    self.model.load_state_dict(last_model_weight)
                    self.matrix_model.load_state_dict(last_lambdas_weight)
                    self.q = torch.tensor(last_q)
                    break
                else:
                    last_model_weight = self.model.state_dict()
                    last_lambdas_weight = self.matrix_model.state_dict()
                    last_q = self.q.item()
            
            with torch.no_grad():
                mat_A = self.matrix_model(x_train_tensor) 
                Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
                epoch_loss = compute_loss(y_train_tensor, self.model( x_train_tensor ), Lambdas, self.q, alpha, k)
                tab_loss.append(epoch_loss.item())

                if y_stop is not None:
                    mat_A = self.matrix_model(x_stop_tensor)
                    Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
                    epoch_loss_stop = compute_loss(y_stop_tensor, self.model(x_stop_tensor), Lambdas, self.q, alpha, k)
                    tab_loss_calibration.append(epoch_loss_stop.item())
                    
                    if best_stop_loss > epoch_loss_stop.item():
                        best_stop_loss = epoch_loss_stop.item()
                        best_model_weight = self.model.state_dict()
                        best_lambdas_weight = self.matrix_model.state_dict()
                        best_q = self.q.item()

                    if verbose != 0 and epoch % print_every == 0 :
                        print(f"Epoch {epoch}: Loss = {epoch_loss.item()} - Stop loss {epoch_loss_stop.item()} - Best Stop Loss {best_stop_loss}")
                
                else : 
                    if best_stop_loss > epoch_loss.item():
                        best_stop_loss = epoch_loss.item()
                        best_model_weight = self.model.state_dict()
                        best_lambdas_weight = self.matrix_model.state_dict()
                        best_q = self.q.item()
                
                    if verbose != 0 and epoch % print_every == 0 :
                        print(f"Epoch {epoch}: Loss = {epoch_loss.item()} - Best Loss {best_stop_loss}")
    
            if use_lr_scheduler:
                scheduler.step()

        if verbose != 0:
            if y_stop is not None:
                print(f"Last epoch : Loss = {epoch_loss.item()} - Stop loss {epoch_loss_stop.item()} - Best Stop Loss {best_stop_loss}")
            else:    
                print(f"Last epoch : Loss = {epoch_loss.item()} - Best Loss {best_stop_loss}")

        if keep_best:
            self.model.load_state_dict(best_model_weight)
            self.matrix_model.load_state_dict(best_lambdas_weight)
            self.q = torch.tensor(best_q)

        
        self.Lambda = self.get_Lambdas(x_train)[0].detach().numpy()
        self.mu = self.model(x_train_tensor)[0].detach().numpy()
        self.q = self.q.item()

        self.tab_loss = tab_loss
        self.tab_loss_calibration = tab_loss_calibration
        self.tab_q = tab_q
        
    def conformalize(self, y_calibration, alpha):
        x_calibration = np.ones_like(y_calibration)
        x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float32)
        y_calibration_tensor = torch.tensor(y_calibration, dtype=torch.float32)

        f_x_calibration = self.model(x_calibration_tensor)

        Lambdas_calibration = self.get_Lambdas_whithout_inf(x_calibration)

        self.nu = get_p_value_mu(y_calibration_tensor, f_x_calibration, Lambdas_calibration, alpha, torch.tensor(self.q) ).item()
        return self.nu

   
    def get_volume_and_coverage(self, x_test, y_test):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        f_x_test = self.model(x_test_tensor)
        
        Lambdas_test = self.get_Lambdas_whithout_inf(x_test)
        
        volume = calculate_all_volumes(Lambdas_test, self.nu, self.q, self.k).mean().item()
        out_of_box = calculate_out_of_box(y_test_tensor, f_x_test, Lambdas_test, self.nu, self.q).item()
        
        return volume, out_of_box
    
    def is_inside(self, x, y, nu=None):
        pass
    
    def get_volume(self):
        return get_volume(self.Lambda, self.nu, self.q)
        
    def get_Lambdas(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        mat_A = self.matrix_model(x_tensor)
        Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
        return Lambdas
    
    def get_Lambdas_whithout_inf(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        mat_A = self.matrix_model(x_tensor)
        Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
        dets = torch.det(Lambdas)
        det_zero_mask = (dets == 0)
        max_values = Lambdas.max(dim=-1, keepdim=True).values
        Lambdas[det_zero_mask] = max_values[det_zero_mask]
        return Lambdas