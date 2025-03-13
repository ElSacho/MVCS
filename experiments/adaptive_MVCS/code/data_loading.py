import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Charger les données depuis le fichier compressé
    data = np.load(file_path)
    return data['X'], data['Y']


def split_and_preprocess(X, Y, splits=[0.7, 0.1, 0.1, 0.1], normalize=True, random_state=None, strategy="quantile"):
    """
    Sépare les données selon les proportions spécifiées et applique une normalisation par quantiles.
    
    Args:
        X (numpy.ndarray): Les caractéristiques.
        Y (numpy.ndarray): Les labels.
        splits (list): Liste des pourcentages de répartition des données.
        random_state (int): Graine aléatoire pour la reproductibilité.
        
    Returns:
        dict: Un dictionnaire contenant les ensembles de données transformés.
    """
    if not np.isclose(sum(splits), 1.0, atol=1e-10):
        raise ValueError("Proportions need to sum to 1.0.")
    
    # Déterminer les tailles absolues des sous-ensembles
    n_samples = X.shape[0]
    split_sizes = [int(n_samples * p ) for p in splits]
    split_sizes[-1] = n_samples - sum(split_sizes[:-1])  # Ajustement pour éviter des erreurs d'arrondi
    
    # Fixer la graine aléatoire si random_state est fourni
    if random_state is not None:
        np.random.seed(random_state)
    
    # Mélanger les données
    indices = np.random.permutation(n_samples)
    X, Y = X[indices], Y[indices]
    
    # Découper les ensembles selon les tailles définies
    start = 0
    subsets = {}
    subset_names = ["X_train", "X_stop", "X_calibration", "X_test"] if len(splits) == 4 else ["X_train", "X_calibration", "X_test"]
    
    for i, name in enumerate(subset_names):
        end = start + split_sizes[i]
        subsets[name] = X[start:end]
        subsets[name.replace("X_", "Y_")] = Y[start:end]
        start = end
    
    if not normalize:
        return subsets
    
    if strategy == "StandardScaler":
        # Normalisation par StandardScaler
        print("StandardScaler")
        scaler = StandardScaler()
        subsets["X_train"] = scaler.fit_transform(subsets["X_train"])
        for name in subset_names[1:]:
            subsets[name] = scaler.transform(subsets[name])
        Y_scaler = StandardScaler()
        subsets["Y_train"] = Y_scaler.fit_transform(subsets["Y_train"])
        for name in [n.replace("X_", "Y_") for n in subset_names[1:]]:
            subsets[name] = Y_scaler.transform(subsets[name])
        return subsets
    
    # # Normalisation par quantiles
    # transformer = QuantileTransformer(output_distribution='normal')
    # subsets["X_train"] = transformer.fit_transform(subsets["X_train"])
    # for name in subset_names[1:]:  # Ne pas refitter, appliquer la même transformation aux autres sets
    #     subsets[name] = transformer.transform(subsets[name])

    # Normalisation par quantiles pour X
    x_transformer = QuantileTransformer(output_distribution='normal')
    subsets["X_train"] = x_transformer.fit_transform(subsets["X_train"])
    for name in subset_names[1:]:  # Ne pas refitter, appliquer la même transformation aux autres sets
        subsets[name] = x_transformer.transform(subsets[name])

    # Normalisation par quantiles pour Y
    y_transformer = QuantileTransformer(output_distribution='normal')
    subsets["Y_train"] = y_transformer.fit_transform(subsets["Y_train"])
    for name in [n.replace("X_", "Y_") for n in subset_names[1:]]:
        subsets[name] = y_transformer.transform(subsets[name])

    
    return subsets


