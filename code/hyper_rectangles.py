import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import copy

from network import *

import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        """
        Pinball Loss for quantile regression.
        
        Args:
            quantile (float): The quantile to estimate (0 < quantile < 1).
        """
        super(PinballLoss, self).__init__()
        if not (0 < quantile < 1):
            raise ValueError("Quantile should be between 0 and 1.")
        self.quantile = quantile

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the pinball loss.
        
        Args:
            y_pred (torch.Tensor): Predicted values (batch_size, ).
            y_true (torch.Tensor): Ground truth values (batch_size, ).
        
        Returns:
            torch.Tensor: The computed pinball loss.
        """
        error = y_true - y_pred
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return loss.mean()


class PinballLoss(nn.Module):
    def __init__(self, tau: float):
        """
        tau: Quantile to estimate, must be between 0 and 1
        """
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y_pred, y_true):
        """
        y_pred: Predicted values (output of the model)
        y_true: True values (ground truth)
        """
        error = y_true - y_pred
        loss = torch.max(self.tau * error, (self.tau - 1) * error)
        return torch.mean(loss)


def calculate_scores(y_true, quantiles_low, quantiles_high):
    """
    Calculate the scores for the quantile predictions.
    
    Args:
        y_true (torch.Tensor): The true values (batch_size, ).
        quantiles_low (torch.Tensor): The low quantile predictions (batch_size, ).
        quantiles_high (torch.Tensor): The high quantile predictions (batch_size, ).
    """
    # Compute the predictions
    score_low = quantiles_low - y_true
    score_high = y_true - quantiles_high

    scores = torch.max(score_low, score_high)
    scores = torch.max(scores, dim=1).values

    return scores

class HyperRectanglePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, alpha, hidden_dim = 10, n_layers = 1, device = 'cpu'):
        self.device = device
        alpha_tilde = 1 - np.power(1 - alpha, 1/output_dim)
        self.alpha_low = alpha_tilde/2
        self.alpha_high = 1 - self.alpha_low
        self.tab_model_alpha_low = []
        self.tab_model_alpha_high = []
        for i in range(output_dim):
            self.tab_model_alpha_low.append(Network(input_dim, 1, hidden_dim, n_layers).to(self.device))
            self.tab_model_alpha_high.append(Network(input_dim, 1, hidden_dim, n_layers).to(self.device))
        self.alpha = alpha
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.alpha_tilde = alpha_tilde
        

    def fit(self, x_train, y_train, x_calibration, y_calibration,
            batch_size = 32,
            num_epochs = 100, 
            lr = 0.001, 
            verbose = False,
            use_lr_scheduler = False,
            keep_best = False):
        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_train = x_train
        self.y_train = y_train

        x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float32).to(self.device)
        y_calibration_tensor = torch.tensor(y_calibration, dtype=torch.float32).to(self.device)


        # Create the dataloaders
        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        calibration_dataset = torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizers for all models
        optimizers_low = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.tab_model_alpha_low]
        optimizers_high = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.tab_model_alpha_high]

        self.tab_loss = []
        self.tab_loss_stop = []

        tab_best_models_low = [model.state_dict() for model in self.tab_model_alpha_low]
        tab_best_models_high = [model.state_dict() for model in self.tab_model_alpha_high]

        if use_lr_scheduler:
            scheduler_low = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) for optimizer in optimizers_low]
            scheduler_high = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) for optimizer in optimizers_high]
            
        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1

        # Training loop
        for epoch in range(num_epochs):
        # for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0.0
            
            # Shuffle training data
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch_x, batch_y in train_loader:
                # Train low-alpha models
                for i, model_low in enumerate(self.tab_model_alpha_low):
                    optimizers_low[i].zero_grad()
                    predictions_low = model_low(batch_x).squeeze()
                    # loss_low = pinball_loss(predictions_low, batch_y[:, i], self.alpha_low)
                    loss_low = PinballLoss(self.alpha_low)(predictions_low, batch_y[:, i])
                    loss_low.backward()
                    optimizers_low[i].step()
                    
                # Train high-alpha models
                for i, model_high in enumerate(self.tab_model_alpha_high):
                    optimizers_high[i].zero_grad()
                    predictions_high = model_high(batch_x).squeeze()
                    # loss_high = pinball_loss(predictions_high, batch_y[:, i], self.alpha_high)
                    loss_high = PinballLoss(self.alpha_high)(predictions_high, batch_y[:, i])
                    loss_high.backward()
                    optimizers_high[i].step()
            
            with torch.no_grad():
                calibration_loss = 0.0
                epoch_loss = 0.0
                for batch_x, batch_y in calibration_loader:
                    for i in range(self.output_dim):
                        calibration_loss += PinballLoss(self.alpha_low)(self.tab_model_alpha_low[i](batch_x).squeeze(), batch_y[:, i]).item()
                        calibration_loss += PinballLoss(self.alpha_high)(self.tab_model_alpha_high[i](batch_x).squeeze(), batch_y[:, i]).item()
                for batch_x, batch_y in train_loader:
                    for i in range(self.output_dim):
                        epoch_loss += PinballLoss(self.alpha_low)(self.tab_model_alpha_low[i](batch_x).squeeze(), batch_y[:, i]).item()
                        epoch_loss += PinballLoss(self.alpha_high)(self.tab_model_alpha_high[i](batch_x).squeeze(), batch_y[:, i]).item()
        
                calibration_loss = calibration_loss/(len(calibration_loader.dataset))
                epoch_loss = epoch_loss/(len(train_loader.dataset))
                self.tab_loss_stop.append(calibration_loss)
                self.tab_loss.append(epoch_loss)

                if keep_best:
                    if calibration_loss < min(self.tab_loss_stop):
                        tab_best_models_low = [copy.deepcopy(model.state_dict()) for model in self.tab_model_alpha_low]
                        tab_best_models_high = [copy.deepcopy(model.state_dict()) for model in self.tab_model_alpha_high]
            
                if verbose != 0:
                    if epoch % print_every == 0:
                        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} - Stop Loss : {calibration_loss:.4f} - Best Stop Loss: {min(self.tab_loss_stop):.4f} ")

            if use_lr_scheduler:
                for scheduler in scheduler_low:
                    scheduler.step()
                for scheduler in scheduler_high:
                    scheduler.step()

        if keep_best:   
            for i, model in enumerate(self.tab_model_alpha_low):
                model.load_state_dict(tab_best_models_low[i])
            for i, model in enumerate(self.tab_model_alpha_high):
                model.load_state_dict(tab_best_models_high[i])
                    

    def conformalize(self, x_calibration, y_calibration):
        x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float32)
        y_calibration_tensor = torch.tensor(y_calibration, dtype=torch.float32)

        quantiles_low = torch.zeros((x_calibration.shape[0], self.output_dim))
        quantiles_high = torch.zeros((x_calibration.shape[0], self.output_dim))

        for i in range(self.output_dim):
            quantiles_low[:, i] = self.tab_model_alpha_low[i](x_calibration_tensor).squeeze()
            quantiles_high[:, i] = self.tab_model_alpha_high[i](x_calibration_tensor).squeeze()
        
        scores = calculate_scores(y_calibration_tensor, quantiles_low, quantiles_high)
        scores_sorted = torch.sort(scores, dim=0, descending=True).values
    
        # Calculate p and ensure it's a valid index
        n = y_calibration.shape[0]
        p = n - int(np.ceil((n+1)*(1-self.alpha)))
        if p < 0:
            raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

        # p = min(max(0, int((1 - self.alpha) * (len(x_calibration) + 1)) - 1), len(scores_sorted) - 1)

        # Get the conformal value
        conformal_value = scores_sorted[p]
        self.conformal_value = conformal_value

    def calculate_volume_and_coverage(self, x_test, y_test):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        quantiles_low = torch.zeros((x_test.shape[0], self.output_dim))
        quantiles_high = torch.zeros((x_test.shape[0], self.output_dim))

        section_lenghts = torch.zeros((x_test.shape[0], self.output_dim))
        for i in range(self.output_dim):    
            quantiles_low[:, i] = self.tab_model_alpha_low[i](x_test_tensor).squeeze()
            quantiles_high[:, i] = self.tab_model_alpha_high[i](x_test_tensor).squeeze()
            section_lenghts[:, i] = quantiles_high[:, i] - quantiles_low[:, i] + 2 * self.conformal_value
        
        volume = torch.prod(section_lenghts, dim=1)
        average_volume = torch.mean(volume)

        scores = calculate_scores(y_test_tensor, quantiles_low, quantiles_high)

        coverage = torch.sum(scores <= self.conformal_value)/len(scores)
        
        return average_volume.item(), coverage.item()
    
    def get_all_volumes(self, x_test, y_test):
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        quantiles_low = torch.zeros((x_test.shape[0], self.output_dim))
        quantiles_high = torch.zeros((x_test.shape[0], self.output_dim))

        section_lenghts = torch.zeros((x_test.shape[0], self.output_dim))
        for i in range(self.output_dim):    
            quantiles_low[:, i] = self.tab_model_alpha_low[i](x_test_tensor).squeeze()
            quantiles_high[:, i] = self.tab_model_alpha_high[i](x_test_tensor).squeeze()
            section_lenghts[:, i] = quantiles_high[:, i] - quantiles_low[:, i] + 2 * self.conformal_value
        
        volumes = torch.prod(section_lenghts, dim=1)
        return volumes