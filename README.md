# Minimum Volume Conformal Sets for Multivariate Regression

This package (PyTorch-based) provides tools for constructing and evaluating conformal prediction sets for multivariate regression, implementing our proposed **Minimum Volume Conformal Sets (MVCS)** method. It accompanies our paper:

**Minimum Volume Conformal Sets for Multivariate Regression**.

## Installation

Clone this repository and install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Overview

The package is structured as follows:

- **code/**: Contains the core implementation.
  - `torch_functions.py`: Defines loss functions used in training.
  - `MVCS.py`: Implements the `MVCSPredictor` class.
  - `example_usage_MVCS_predictor.ipynb`: Demonstrates how to use the implemented methods.
- **experiments/**: Contains experimental evaluations.
  - `one_matrix_MVCS/`: Experiments for Section 2 of the paper.
  - `adaptive_MVCS/`: Experiments for Sections 3 and 5.
    - `code/`: Code for adaptive MVCS experiments.
    - `parameters/`: JSON files defining hyperparameters for different strategies.

## Using `MVCSPredictor`

The `MVCSPredictor` class requires two models as input:
- A **center model**, which should ideally be pre-trained.
- A **matrix model**, used to construct conformal sets.

### Basic Usage

```python
from code.MVCS import MVCSPredictor

# Initialize with pre-trained models
predictor = MVCSPredictor(center_model, matrix_model)

# Fit the models to data
predictor.fit(trainloader)

# Conformalize the prediction sets
predictor.conformalize(calibrationloader)

# Get volume and coverage
volume = predictor.get_volume(x_test)
coverage = predictor.get_coverage(testloader)
```

## Running Experiments

### Section 2: One-Matrix MVCS
Navigate to the `experiments/one_matrix_MVCS/` folder and run the Jupyter notebooks to reproduce results from Section 2.

### Sections 3 and 5: Adaptive MVCS
Navigate to `experiments/adaptive_MVCS/code/` and run:
```bash
python generate_experiment.py name_of_the_parameter_file
```
where `name_of_the_parameter_file.json` is in `experiments/adaptive_MVCS/parameters/` and contains the strategy's hyperparameters.
Run the file `see_results_normalized.ipynb` to generate the tables present in the manuscript.

### Plotting Results
Plots can be generated using Jupyter notebooks in `experiments/adaptive_MVCS/code/`.

## Citation
If you use this repository for research purposes, please cite our paper:

**Minimum Volume Conformal Sets for Multivariate Regression**.

For any questions, feel free to open an issue or contact us.

