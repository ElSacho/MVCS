import numpy as np

from utils import *
from generate_data import * 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

from network import *

from torch_functions import *

from plot_data import *
from data_loading import *

import json
import pickle
import argparse

from ellipsoid_predictor import *
from hyper_rectangles import *
from covariances import *
from local_covariances import *

seed_everything(43)


parser = argparse.ArgumentParser(description="Script avec argument config_name")
parser.add_argument("config_name", type=str, help="Nom de la configuration")

# Parser les arguments
args = parser.parse_args()

# Stocker la valeur de config_name dans une variable
config_name = args.config_name

config_path = "../parameters/" + config_name + ".json"
with open(config_path, 'r') as file : 
    parameters = json.load(file)

alpha = parameters["alpha"]

results = {}
tab_results = {}

for experiment in range(parameters["n_experiments"]):
    print(f"Experiment {experiment}/{parameters['n_experiments']} for alpha = {alpha}")

    prop_train = parameters["prop_train"]
    prop_calibration = parameters["prop_calibration"]

    # Chemin d'entrée et de sortie
    load_path = "../data/processed_data/" + parameters["load_name"] + ".npz"

    # Exemple d'utilisation du loader
    X, Y = load_data(load_path)

    # Load Data
    load_path = "../data/processed_data/" + parameters["load_name"] + ".npz"
    X, Y = load_data(load_path)


    normalize = parameters["normalize"]
    splits = [parameters["prop_train"], parameters["prop_stop"], parameters["prop_calibration"], parameters["prop_test"]]

    subsets = split_and_preprocess(X, Y, splits=splits, normalize=normalize)

    x_train, y_train, x_calibration, y_calibration, x_test, y_test, x_stop, y_stop = subsets["X_train"], subsets["Y_train"], subsets["X_calibration"], subsets["Y_calibration"], subsets["X_test"], subsets["Y_test"], subsets["X_stop"], subsets["Y_stop"]

    print("X_train shape:", x_train.shape, "Y_train shape:", y_train.shape)
    print("X_cal shape:", x_calibration.shape, "Y_cal shape:", y_calibration.shape)
    print("X_test shape:", x_test.shape, "Y_test shape:", y_test.shape)
    print("X_stop shape:", x_stop.shape, "Y_stop shape:", y_stop.shape)

    d = x_train.shape[1]
    k = y_train.shape[1]

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_calibration = x_calibration.shape[0]
    n_stop = x_stop.shape[0]

    hidden_dim = parameters["hidden_dim"]
    hidden_dim_matrix = parameters["hidden_dim_matrix"]
    n_hidden_layers = parameters["n_hidden_layers"]
    n_hidden_layers_matrix = parameters["n_hidden_layers_matrix"]

    num_epochs_warm_start = parameters["num_epochs_warm_start"]
    warm_start_epochs_mat = parameters["warm_start_epochs_mat"]
    num_epochs_our_loss = parameters["num_epochs_our_loss"]

    lr_warm_start = parameters["lr_warm_start"]
    lr_model = parameters["lr_model"]
    lr_matrix = parameters["lr_matrix"]
    lr_q = parameters["lr_q"]

    batch_size_warm_start = parameters["batch_size_warm_start"]
    batch_size_our_loss = parameters["batch_size_our_loss"]

    n_neighbors = parameters["n_neighbors"]
    use_lr_scheduler = parameters["use_lr_scheduler"]
    keep_best = parameters["keep_best"]
    loss_strategy = parameters["loss_strategy"]
    use_epsilon = parameters["use_epsilon"]

    num_epochs_rectangles = parameters["num_epochs_rectangles"]
    lr_rectangles = parameters["lr_rectangles"]
    batch_size_rectangles = parameters["batch_size_rectangles"]

    

    x_train_tensor = torch.tensor(x_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64)
    x_stop_tensor = torch.tensor(x_stop, dtype=torch.float64)
    y_stop_tensor = torch.tensor(y_stop, dtype=torch.float64)
    x_calibration_tensor = torch.tensor(x_calibration, dtype=torch.float64)
    y_calibration_tensor = torch.tensor(y_calibration, dtype=torch.float64)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64)

    matrix_model = MatrixPredictor(d, k, k, hidden_dim=hidden_dim_matrix).double()
    model = Network(d, k, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers).double()
    q = torch.tensor(2.0, requires_grad=True)

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_warm_start, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size_warm_start, shuffle=True)
    train_losses, stop_losses = model.fit_and_plot(trainloader, stoploader, epochs=num_epochs_warm_start, lr=lr_warm_start, keep_best=keep_best)

    warm_start_model = copy.deepcopy(model).double()

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size= batch_size_our_loss, shuffle=True)
    stoploader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_stop_tensor, y_stop_tensor), batch_size= batch_size_our_loss, shuffle=True)
    calibrationloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_calibration_tensor, y_calibration_tensor), batch_size= batch_size_our_loss, shuffle=True)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor), batch_size= batch_size_our_loss, shuffle=True)

    ellipsoid_predictor = EllipsoidPredictor(model, matrix_model, q)

    ellipsoid_predictor.fit(trainloader, 
                            stoploader, 
                            alpha,
                            num_epochs = num_epochs_our_loss,
                            num_epochs_mat_only = warm_start_epochs_mat,
                            lr_model = lr_model,
                            lr_q = lr_q,
                            lr_matrix_model = lr_matrix,
                            use_lr_scheduler = use_lr_scheduler,
                            verbose = 1,
                            stop_on_best = keep_best,
                            loss_strategy = loss_strategy,
                            use_epsilon = use_epsilon
                            )

    model_final_loss_calibration = ellipsoid_predictor.model.eval(calibrationloader)
    warm_start_model_loss_calibration = warm_start_model.eval(calibrationloader)

    if model_final_loss_calibration < warm_start_model_loss_calibration:
        covariance_predictor = CovariancePredictor(ellipsoid_predictor.model)
        local_covariance_predictor = LocalCovariancePredictor(ellipsoid_predictor.model, n_neighbors=n_neighbors)

        covariance_predictor.fit(trainloader)
        local_covariance_predictor.fit(trainloader)
        print('keeping the final model')
    else:
        covariance_predictor = CovariancePredictor(warm_start_model)
        local_covariance_predictor = LocalCovariancePredictor(warm_start_model, n_neighbors=n_neighbors)
        covariance_predictor.fit(trainloader)
        local_covariance_predictor.fit(trainloader)
        print('keeping the warm started model')


    covariance_predictor.conformalize(calibrationloader, alpha = alpha)
    local_covariance_predictor.conformalize(calibrationloader, alpha = alpha)
    ellipsoid_predictor.conformalize(calibrationloader, alpha = alpha)

    ellipsoid_coverage = ellipsoid_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)
    covariance_coverage = covariance_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)
    local_covariance_coverage = local_covariance_predictor.get_coverage(x_test=x_test_tensor, y_test=y_test_tensor)

    
    ellipsoid_volume = ellipsoid_predictor.get_averaged_volume(x_test=x_test_tensor)
    covariance_volume = covariance_predictor.get_averaged_volume(x_test = x_test_tensor)
    local_covariance_volume = local_covariance_predictor.get_averaged_volume(x_test = x_test_tensor)
    

    hyper_rectangle_predictor = HyperRectanglePredictor(d, k, alpha, hidden_dim = hidden_dim, n_layers = n_hidden_layers)

    hyper_rectangle_predictor.fit(x_train, y_train, x_stop, y_stop,
                            num_epochs = num_epochs_rectangles, 
                            lr = lr_rectangles,
                            batch_size = batch_size_rectangles,
                            use_lr_scheduler = use_lr_scheduler,
                            keep_best = keep_best)
    

    hyper_rectangle_predictor.conformalize(x_calibration, y_calibration)
    volume_hyper_rectangle, coverage_hyper_rectangles = hyper_rectangle_predictor.calculate_volume_and_coverage(x_test, y_test)

    results[experiment] = {"volume_ellipsoid": ellipsoid_volume, 
                    "volume_covariance":covariance_volume,
                    "volume_local_covariance":local_covariance_volume,
                    "volume_hyper_rectangle": volume_hyper_rectangle,
                    "coverage_ellipsoid": ellipsoid_coverage,
                    "coverage_covariance": covariance_coverage,
                    "coverage_local_covariance": local_covariance_coverage,
                    "coverage_hyper_rectangle": coverage_hyper_rectangles
                        }
    
    tab_results[experiment] = {"warm_start_train_MSE_loss" : train_losses, 
                                "warm_start_stop_MSE_loss" : stop_losses,
                                "our_loss_train" : ellipsoid_predictor.tab_train_loss, 
                                "our_loss_stop" : ellipsoid_predictor.tab_stop_loss, 
                                "hyper_rectangle_train_loss" : hyper_rectangle_predictor.tab_loss,
                                "hyper_rectangle_stop_loss":  hyper_rectangle_predictor.tab_loss_stop
                                }
    
    print(f"Volume ellipsoid: {ellipsoid_volume:.3f}, \nVolume covariance: {covariance_volume:.3f}, \nVolume local covariance: {local_covariance_volume:.3f}, \nVolume hyper rectangle: {volume_hyper_rectangle:.3f}")
    
save_path = f"../results/volume/{parameters['load_name']}_alpha_{alpha}.pkl"
save_tab_path = f"../results/tab_results/{parameters['load_name']}_alpha_{alpha}.pkl"

with open(save_path, "wb") as f:
    pickle.dump(results, f)

with open(save_tab_path, "wb") as f:
    pickle.dump(tab_results, f)

mean_results = {}
for key in results[0].keys():
    # Extract values for the current key across experiments
    values = [results[exp][key] for exp in results]
    
    # Remove the minimum and maximum values
    values_sorted = sorted(values)
    # values_trimmed = values_sorted[1:-1]  # Exclude first (min) and last (max)
    
    # Calculate the mean of the remaining values
    mean_results[key] = np.mean(values_sorted)

print(mean_results)