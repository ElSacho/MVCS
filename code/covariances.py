import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_functions import *



def get_lambdas_cov(y_train, f_x):
    k = y_train.shape[1]
    residuals = y_train - f_x
    cov = torch.atleast_2d(torch.cov(residuals.T))
    Lambda_cov_torch = torch.linalg.inv(cov)
    eigenvalues, eigenvectors = torch.linalg.eigh(Lambda_cov_torch)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    Lambda_cov_torch = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return Lambda_cov_torch


class CovariancePredictor:
    def __init__(self, model):
        self._model = model
        self._nu_covariance = None
        self._Lambda_cov = None
        self._q_fix = torch.tensor(2, dtype=torch.float32)

    @property
    def nu_covariance(self):
        return self._nu_covariance
        
    @property
    def Lambda_cov(self):
        return self._Lambda_cov
    
    @property
    def model(self):
        return self._model
        
    def fit(self, trainloader = None, x_train = None, y_train = None):
        with torch.no_grad():
            if trainloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in trainloader:
                    f_x = self._model(x)
                    
                    if empty:
                        f_x_train = f_x
                        y_train = y
                        empty = False
                    else:
                        f_x_train = torch.cat((f_x_train, f_x), 0)
                        y_train = torch.cat((y_train, y), 0)
            
            elif x_train is not None and y_train is not None:
                # Case where we directly give tensors
                f_x_train = self._model(x_train)
            
            self._Lambda_cov = get_lambdas_cov(y_train, f_x_train)
    

    def conformalize(self, calibrationloader = None, x_calibration =None, y_calibration = None, alpha = 0.1, model_cov = None):
        if self._Lambda_cov is None:
            raise ValueError("You need to fit the model first.")
        
        with torch.no_grad():
            if calibrationloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in calibrationloader:
                    f_x = self._model(x)
                    
                    if empty:
                        f_x_calibration = f_x
                        y_calibration = y
                        empty = False
                    else:
                        f_x_calibration = torch.cat((f_x_calibration, f_x), 0)
                        y_calibration = torch.cat((y_calibration, y), 0)
            
            elif x_calibration is not None and y_calibration is not None:
                # Case where we directly give tensors
                f_x_calibration = self._model(x_calibration)
            
            else:
                raise ValueError("You need to provide a `calibrationloader`, or `x_calibration` and `y_calibration`.")
    
        k = y_calibration.shape[1]
        Lambdas_calibration = self._Lambda_cov.unsqueeze(0).expand(y_calibration.shape[0], k, k).clone()
        
        n = y_calibration.shape[0]
        p = n - int(np.ceil((n+1)*(1-alpha)))
        print(p)
        if p < 0:
                raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")

        self._nu_covariance = get_p_value_mu(y_calibration, f_x_calibration, Lambdas_calibration, p, self._q_fix)
        return self._nu_covariance
    
    def get_volumes(self, testloader=None, x_test=None):
        
        if self._Lambda_cov is None:
            raise ValueError("You must call the `fit` method before.")
        if self._nu_covariance is None:
            raise ValueError("You must call the `conformalize_ellipsoids` method before.")
        k = self._Lambda_cov.shape[1]
        if testloader is not None:
            # Case where we use a DataLoader
            Lambdas_test = self._Lambda_cov.unsqueeze(0).expand(len(testloader.dataset), k, k).clone()
        elif x_test is not None:
            # Case where we directly give tensors
            Lambdas_test = self._Lambda_cov.unsqueeze(0).expand(x_test.shape[0], k, k).clone()

        return calculate_all_volumes(Lambdas_test, self._nu_covariance, self._q_fix)

    def get_averaged_volume(self, testloader=None, x_test=None):
        if self._Lambda_cov is None:
            raise ValueError("You must call the `fit` method before.")
        if self._nu_covariance is None:
            raise ValueError("You must call the `conformalize_ellipsoids` method before.")
        return calculate_all_volumes(self._Lambda_cov, self._nu_covariance, self._q_fix).mean().item()
    
    def get_Lambdas(self, x_test):
        if self._Lambda_cov is None:
            raise ValueError("You must call the `fit` method before.")
        k = self._Lambda_cov.shape[1]
        return self._Lambda_cov.unsqueeze(0).expand(x_test.shape[0], k, k).clone()
        
    def get_coverage(self, testloader=None, x_test=None, y_test=None):   
        with torch.no_grad(): 
            if self._Lambda_cov is None:
                raise ValueError("You must call the `fit` method before.")
            if self._nu_covariance is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")    
            
            if testloader is not None:
                # Case where we use a DataLoader
                empty = True
                for x, y in testloader:
                    f_x = self._model(x)

                    if empty:
                        f_x_test = f_x
                        y_test = y
                        empty = False
                    else:
                        f_x_test = torch.cat((f_x_test, f_x), 0)
                        y_test = torch.cat((y_test, y), 0)
                        
                Lambdas_test = self._Lambda_cov.unsqueeze(0).expand(y_test.shape[0], y_test.shape[1], y_test.shape[1]).clone()

            elif x_test is not None and y_test is not None:
                # Case where x_test and y_test are given as tensors
                f_x_test = self._model(x_test)
                Lambdas_test = self._Lambda_cov.unsqueeze(0).expand(y_test.shape[0], y_test.shape[1], y_test.shape[1]).clone()

            else:
                raise ValueError("You must provide either `testloader` or both `x_test` and `y_test`.")
            
            
            coverage = calculate_coverage(y_test, f_x_test, Lambdas_test, self._nu_covariance, self._q_fix).mean().item()
                
            return coverage

    def is_inside(self, x, y):
        with torch.no_grad():
            if self._Lambda_cov is None:
                raise ValueError("You must call the `fit` method before.")
            if self._nu_covariance is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")
            k = y.shape[1]
            f_x = self._model(x)
            Lambdas = self._Lambda_cov.unsqueeze(0).expand(y.shape[0], k, k).clone()
            
            return calculate_norm_q(Lambdas @ (y - f_x), self._q_fix).item() <= self._nu_covariance
