import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_functions import *


def get_lambda_cov(y_train, f_x):
    k = y_train.shape[1]
    residuals = y_train - f_x
    cov = torch.atleast_2d(torch.cov(residuals.T))
    Lambda_cov_torch = torch.linalg.inv(cov)
    eigenvalues, eigenvectors = torch.linalg.eigh(Lambda_cov_torch)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    Lambda_cov_torch = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return Lambda_cov_torch


def find_local_covariances(y_train, f_x_train, x_train, x_calibration, n_neighbors):
    k = y_train.shape[1]
    tab_Lambda = torch.zeros((len(x_calibration), k, k), dtype=y_train.dtype)
    for idx, x in enumerate(x_calibration):
        distances = torch.norm(x_train - x, dim=1)
        nearest_neighbors = torch.argsort(distances)[:n_neighbors]
        y_neighbors = y_train[nearest_neighbors]
        f_x_neighbors = f_x_train[nearest_neighbors]
    
        Lambda_x = get_lambda_cov(y_neighbors, f_x_neighbors)
        tab_Lambda[idx] = Lambda_x
        
    return tab_Lambda


class LocalCovariancePredictor:
    def __init__(self, model, n_neighbors = 100, dtype = torch.float32):
        self._model = model
        self._nu_local_covariance = None
        self._n_neighbors = n_neighbors
        self._q_fix = torch.tensor(2, dtype=dtype)

    @property
    def nu_local_covariance(self):
        return self._nu_local_covariance
        
    @property
    def n_neighbors(self):
        return self._n_neighbors
        
    def fit(self, trainloader = None, x_train = None, y_train = None):
        with torch.no_grad():
            if trainloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in trainloader:
                    f_x = self._model(x)
                    
                    if empty:
                        x_train = x
                        f_x_train = f_x
                        y_train = y
                        empty = False
                    else:
                        x_train = torch.cat((x_train, x), 0)
                        f_x_train = torch.cat((f_x_train, f_x), 0)
                        y_train = torch.cat((y_train, y), 0)
            
            elif x_train is not None and y_train is not None:
                # Case where we directly give tensors
                f_x_train = self._model(x_train)

            self._x_train = x_train
            self._f_x_train = f_x_train
            self._y_train = y_train

    def get_Lambdas(self, x):
        return find_local_covariances(self._y_train, self._f_x_train, self._x_train, x, self._n_neighbors)   

    def conformalize(self, calibrationloader = None, x_calibration =None, y_calibration = None, alpha = 0.1, model_cov = None):
        if self._x_train is None:
            raise ValueError("You need to fit the model first.")
        
        with torch.no_grad():
            if calibrationloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in calibrationloader:
                    f_x = self._model(x)
                    
                    if empty:
                        x_calibration = x
                        f_x_calibration = f_x
                        y_calibration = y
                        empty = False
                    else:
                        x_calibration = torch.cat((x_calibration, x), 0)
                        f_x_calibration = torch.cat((f_x_calibration, f_x), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                f_x_calibration = self._model(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")
    
        k = y_calibration.shape[1]
        Lambdas_calibration = self.get_Lambdas(x_calibration)

        n = y_calibration.shape[0]
        p = n - int(np.ceil((n+1)*(1-alpha)))
        if p < 0:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

        self._nu_local_covariance = get_p_value_mu(y_calibration, f_x_calibration, Lambdas_calibration, p, self._q_fix)
        return self._nu_local_covariance
    
    def get_volumes(self, testloader=None, x_test=None):
        if self._x_train is None:
            raise ValueError("You must call the `fit` method before.")
        if self._nu_local_covariance is None:
            raise ValueError("You must call the `conformalize_ellipsoids` method before.")
        
        if testloader is not None:
            # Case where we use a DataLoader
            empty = True
            for x in testloader:
                if empty:
                    x_test = x
                    empty = False
                else:
                    x_test = torch.cat((x_test, x), dim=0)

        Lambdas_test = self.get_Lambdas(x_test)
        return calculate_all_volumes(Lambdas_test, self._nu_local_covariance, self._q_fix)

    def get_averaged_volume(self, testloader=None, x_test=None):
        return self.get_volumes(testloader=testloader, x_test=x_test).mean().item()
        
    def get_coverage(self, testloader=None, x_test=None, y_test=None):   
        with torch.no_grad(): 
            if self._x_train is None:
                raise ValueError("You must call the `fit` method before.")
            if self._nu_local_covariance is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")    
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in testloader:
                    f_x = self._model(x)

                    if empty:
                        x_test = x
                        f_x_test = f_x
                        y_test = y
                        empty = False
                    else:
                        x_test = torch.cat((x_test, x), 0)
                        f_x_test = torch.cat((f_x_test, f_x), 0)
                        y_test = torch.cat((y_test, y), 0)
                self.get_Lambdas(x_test)

            elif x_test is not None and y_test is not None:
                # Case where x_test and y_test are given as tensors
                f_x_test = self._model(x_test)
                Lambdas_test = self.get_Lambdas(x_test)

            else:
                raise ValueError("You must provide either `testloader` or both `x_test` and `y_test`.")
            
            coverage = calculate_coverage(y_test, f_x_test, Lambdas_test, self._nu_local_covariance, self._q_fix).mean().item()
                
            return coverage

    def is_inside(self, x, y):
        with torch.no_grad():
            if self._x_train is None:
                raise ValueError("You must call the `fit` method before.")
            if self._nu_local_covariance is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")
            k = y.shape[1]
            f_x = self._model(x)
            Lambdas = self.get_Lambdas(x)
            
            return calculate_norm_q(Lambdas @ (y - f_x), self._q_fix).item() <= self._nu_local_covariance
