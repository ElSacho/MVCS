import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_functions import *

class MVCSPredictor:
    def __init__(self, center_model, matrix_model, q=torch.tensor(2.0, requires_grad=True)):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model that represents the center of the sets.
        matrix_model : torch.nn.Module
            The model that represents the matrix A of the ellipsoids. The matrix Lambda is obtained as the product of A by its transpose.
        q : torch.Tensor, optional
            The q parameter of the q-norm. The default is torch.tensor(2.0, requires_grad=True).
        """

        self.center_model = center_model
        self.matrix_model = matrix_model
        self.q = q
        self._nu_conformal = None

    def get_Lambdas(self, x):
        """
        Compute the matrix Lambda = A @ A^T for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        Lambdas : torch.Tensor
            The matrix Lambda = A(x) @ A(x)^T of shape (n, k, k)
        """
        mat_A = self.matrix_model(x)
        Lambdas = torch.einsum('nij,nkj->nik', mat_A, mat_A)
        return Lambdas
    
    @property
    def nu_conformal(self):
        return self._nu_conformal

    def fit(self,
            trainloader,
            stoploader,
            alpha,
            num_epochs=1000,
            num_epochs_mat_only=50,
            lr_model=0.001,
            lr_q=0.001,
            lr_matrix_model=0.001,
            use_lr_scheduler=False,
            verbose=0,
            stop_on_best=False,
            loss_strategy="log_volume",
            use_epsilon=False
        ):
        """"
        Parameters
        
        trainloader : torch.utils.data.DataLoader
            The DataLoader of the training set.
        stoploader : torch.utils.data.DataLoader
            The DataLoader of the validation set : Warning: do not use the calibration set as a stopping criterion as you would lose the coverage property.
        alpha : float
            The level of the confidence sets.
        num_epochs : int, optional
            The total number of epochs. The default is 1000.
        num_epochs_mat_only : int, optional
            The number of epochs where only the matrix model is trained. The default is 50.
        lr_model : float, optional
            The learning rate for the model. The default is 0.001.
        lr_q : float, optional
            The learning rate for the q parameter. The default is 0.001.
        lr_matrix_model : float, optional
            The learning rate for the matrix model. The default is 0.001.
        use_lr_scheduler : bool, optional
            Whether to use a learning rate scheduler. The default is False.
        verbose : int, optional
            The verbosity level. The default is 0.
        stop_on_best : bool, optional   
            Whether to stop on the best model. The default is False.
        loss_strategy : str, optional
            The strategy to compute the loss. The default is "exact_volume".
        use_epsilon : bool, optional
            Whether to use the epsilon parameter. The default is False.
            """
        
        if stop_on_best:
            self.best_stop_loss = np.inf
            self.best_model_weight = None
            self.best_lambdas_weight = None
            self.best_q = None

        self.alpha = alpha

        optimizer = torch.optim.Adam([
            {'params': self.center_model.parameters(), 'lr': lr_model},  # Learning rate for self.center_model
            {'params': self.matrix_model.parameters(), 'lr': lr_matrix_model},  # Learning rate for self.matrix_model
            {'params': self.q, 'lr': lr_q}  # Learning rate for q
        ])

        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.tab_train_loss = []
        self.tab_stop_loss = []

        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1

        last_model_weight = self.center_model.state_dict()
        last_lambdas_weight = self.matrix_model.state_dict()
        last_q = self.q.item()

        for epoch in range(num_epochs):

            epoch_loss = 0.0
            for x, y in trainloader:    
                optimizer.zero_grad()

                Lambdas = self.get_Lambdas(x)
    
                if epoch < num_epochs_mat_only:
                    with torch.no_grad():
                        f_x = self.center_model(x)
                else:
                    f_x = self.center_model(x)


                loss = compute_loss(y, f_x, Lambdas, self.q, alpha, loss_strategy=loss_strategy, use_epsilon=use_epsilon)

                if torch.isnan(loss):
                    self.center_model.load_state_dict(last_model_weight)
                    self.matrix_model.load_state_dict(last_lambdas_weight)
                    self.q = torch.tensor(last_q, requires_grad=True)
                    break

                loss.backward()
                optimizer.step()
                epoch_loss += loss

                last_model_weight = copy.deepcopy(self.center_model.state_dict())
                last_lambdas_weight = copy.deepcopy(self.matrix_model.state_dict())
                last_q = self.q.item()

            epoch_loss = self.eval(trainloader, loss_strategy)
            self.tab_train_loss.append(epoch_loss.item())

            epoch_stop_loss = self.eval(stoploader, loss_strategy)
            self.tab_stop_loss.append(epoch_stop_loss.item())

            if stop_on_best and self.best_stop_loss > epoch_stop_loss.item():
                if verbose == 2:
                    print(f"New best stop loss: {epoch_stop_loss.item()}")
                self.best_stop_loss = epoch_stop_loss
                self.best_model_weight = copy.deepcopy(self.center_model.state_dict())
                self.best_lambdas_weight = copy.deepcopy(self.matrix_model.state_dict())
                self.best_q = self.q.item()
                self.load_best_model()
                            
                
            if verbose != 0:
                if epoch % print_every == 0:
                    print(f"Epoch {epoch}: Loss = {epoch_loss.item()} - Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")

            if use_lr_scheduler:
                scheduler.step()

        epoch_loss = self.eval(trainloader, loss_strategy)
        epoch_stop_loss = self.eval(stoploader, loss_strategy)
        if stop_on_best:
            self.load_best_model()
        epoch_loss = self.eval(trainloader, loss_strategy)
        epoch_stop_loss = self.eval(stoploader, loss_strategy)
        if verbose != 0:
            print(f"Final Loss = {epoch_loss.item()} - Final Stop Loss = {epoch_stop_loss.item()} - Best Stop Loss = {self.best_stop_loss}")

    def load_best_model(self):
        """
        Load the best model.    
        """
        if self.best_model_weight is not None:
            self.center_model.load_state_dict(self.best_model_weight)
            self.matrix_model.load_state_dict(self.best_lambdas_weight)
            self.q = torch.tensor(self.best_q, requires_grad=True)
        else:
            raise ValueError("You must call the `fit` method with the `stop_on_best` parameter set to True.")

    def eval(self,
             dataloader, loss_strategy):
        """"
        Evaluate the loss on a given DataLoader.
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader of the dataset on which to evaluate the loss.
        loss_strategy : str
            The strategy to compute the loss.
        """
        
        with torch.no_grad():
            loss = 0.0
            empty = True
            for x, y in dataloader:
                Lambdas = self.get_Lambdas(x)
                f_x = self.center_model( x )

                if empty:
                    f_x_eval = f_x
                    y_eval = y
                    Lambdas_eval = Lambdas
                    empty = False

                else:
                    Lambdas_eval = torch.cat((Lambdas_eval, Lambdas), dim=0)
                    f_x_eval = torch.cat((f_x_eval, f_x), 0)
                    y_eval = torch.cat((y_eval, y), 0)

            loss = compute_loss(y_eval, f_x_eval, Lambdas_eval, self.q, self.alpha, loss_strategy = loss_strategy)
            return loss
    
    def conformalize(self, calibrationloader=None, x_calibration=None, y_calibration=None, alpha=0.1):
        """
        Compute the quantile value to conformalize the ellipsoids on a unseen dataset.

        Parameters
        ----------
        calibrationloader : torch.utils.data.DataLoader, optional
            The DataLoader of the calibration set. The default is None.
        x_calibration : torch.Tensor, optional
            The input tensor of the calibration set. The default is None.  The shape is (n, d).
        y_calibration : torch.Tensor, optional
            The output tensor of the calibration set. The default is None. The shape is (n, k).
        alpha : float, optional
            The level of the confidence sets. The default is 0.1.    
        """
        with torch.no_grad():
            if calibrationloader is not None:
                # Case where we use a DataLoader
                empty = True

                for x, y in calibrationloader:
                    f_x = self.center_model(x)
                    Lambdas = self.get_Lambdas(x)

                    if empty:
                        f_x_calibration = f_x
                        y_calibration = y
                        Lambdas_calibration = Lambdas
                        empty = False
                    else:
                        Lambdas_calibration = torch.cat((Lambdas_calibration, Lambdas), dim=0)
                        f_x_calibration = torch.cat((f_x_calibration, f_x), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                f_x_calibration = self.center_model(x_calibration)
                Lambdas_calibration = self.get_Lambdas(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")

            n = y_calibration.shape[0]
            p = n - int(np.ceil((n+1)*(1-alpha)))
            if p < 0:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

            self._nu_conformal = get_p_value_mu(y_calibration, f_x_calibration, Lambdas_calibration, p, self.q)

    def get_volumes(self, testloader=None, x_test=None):
        """
        Compute the volumes of the confidence sets on a given dataset.

        Parameters
        ----------
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None. The shape is (n, d).

        Returns
        -------
        volumes : torch.Tensor
            The volumes of each confidence sets. The shape is (n,).
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")

            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, _ in testloader:
                    Lambdas = self.get_Lambdas(x)
                    if empty:
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                    
            elif x_test is not None:
                # Case where we directly give tensors
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You need to either provide a `testloader`, or `x_test`.")
            
            return calculate_all_volumes(Lambdas_test, self._nu_conformal, self.q)

    def get_averaged_volume(self, testloader=None, x_test=None):
        """
        Compute the averaged volume of the confidence sets on a given dataset.

        Parameters
        ----------
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None.

        Returns
        -------
        volume : torch.Tensor
            The averaged volume of the confidence sets.
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, _ in testloader:
                    Lambdas = self.get_Lambdas(x)
                    if empty:
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                    
            elif x_test is not None:
                # Case where we directly give tensors
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You need to either provide a `testloader`, or `x_test`.")
            
            return calculate_averaged_volume(Lambdas_test, self._nu_conformal, self.q)
    
    
    def get_coverage(self, testloader=None, x_test=None, y_test=None):  
        """
        Compute the coverage of the confidence sets on a given dataset.
        
        Parameters
        ----------
        testloader : torch.utils.data.DataLoader, optional
            The DataLoader of the test set. The default is None.
        x_test : torch.Tensor, optional
            The input tensor of the test set. The default is None. The shape is (n, d).
        y_test : torch.Tensor, optional
            The output tensor of the test set. The default is None. The shape is (n, k).

        returns
        -------
        coverage : float
            The coverage of the confidence sets (between 0 and 1).
        """
        with torch.no_grad(): 
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")    
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in testloader:
                    f_x = self.center_model(x)
                    Lambdas = self.get_Lambdas(x)

                    if empty:
                        f_x_test = f_x
                        y_test = y
                        Lambdas_test = Lambdas
                        empty = False
                    else:
                        Lambdas_test = torch.cat((Lambdas_test, Lambdas), dim=0)
                        f_x_test = torch.cat((f_x_test, f_x), 0)
                        y_test = torch.cat((y_test, y), 0)

            elif x_test is not None and y_test is not None:
                # Case where x_test and y_test are given as tensors
                f_x_test = self.center_model(x_test)
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You must provide either `testloader` or both `x_test` and `y_test`.")
            
            coverage = calculate_coverage(y_test, f_x_test, Lambdas_test, self._nu_conformal, self.q).mean().item()
                
            return coverage

    
    def is_inside(self, x, y):
        """"
            Check if a given point is inside the confidence set.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor of shape (n, d)
            y : torch.Tensor  
                The output tensor of shape (n, k)

            Returns
            ------- 
            is_inside : bool
                Whether the point is inside the confidence set or not.
        """
        with torch.no_grad():
            if self._nu_conformal is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")

            f_x = self.center_model(x)
            Lambdas = self.get_Lambdas(x)
            
            norm_values = torch.tensor([calculate_norm_q(Lambdas[i] @ (y[i] - f_x[i]), self.q) for i in range(len(y))])  # (n,)
        
            return norm_values <= self.nu_conformal