import pandas as pd
import random
import numpy as np
import torch



class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Initialize the KMeans class.

        Parameters:
        - n_clusters (int): Number of clusters to form.
        - max_iter (int): Maximum number of iterations of the algorithm.
        - tol (float): Tolerance for convergence. The algorithm stops if the centroids change less than this.
        - random_state (int): Seed for random number generator (for reproducibility).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans model to the data.

        Parameters:
        - X (numpy.ndarray): Data to cluster, of shape (n_samples, n_features).
        """
        np.random.seed(self.random_state)
        # Randomly initialize the centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # Assign each point to the nearest centroid
            labels = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters:
        - X (numpy.ndarray): Data to predict, of shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Index of the cluster each sample belongs to.
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """
        Assign each point in X to the nearest centroid.

        Parameters:
        - X (numpy.ndarray): Data to cluster, of shape (n_samples, n_features).

        Returns:
        - labels (numpy.ndarray): Index of the cluster each sample belongs to.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Création du DataFrame
def print_results(results):
    df_results = pd.DataFrame(results)

    # Arrondir les valeurs de 'Logdet' et 'Non relaxed problem cost' à deux chiffres significatifs
    df_results["Logdet"] = df_results["Logdet"].apply(lambda x: f"{x:.3g}")
    df_results["Non relaxed problem cost"] = df_results["Non relaxed problem cost"].apply(lambda x: f"{x:.3g}")

    # Masquer les répétitions de la colonne "Perturbation"
    df_results.loc[df_results.duplicated(subset=["Perturbation"]), "Perturbation"] = ""

    # Fonction pour ajouter des lignes doubles entre perturbations
    def add_double_border(s):
        border_style = "border-top: 3px double black;"
        return [border_style if isinstance(s.index[i], int) and i > 0 and df_results["Perturbation"].iloc[i] == "" else "" for i in range(len(s))]

    # Fonction pour styliser les valeurs minimales en gras dans les colonnes "Non relaxed problem cost" et "Logdet"
    # def highlight_min(df, column):
    #     styles = [''] * len(df)  # Initialiser une liste vide pour stocker les styles
    #     # Parcourir chaque groupe de perturbation
    #     for pert, group in df.groupby("Perturbation"):
    #         min_value = group[column].min()  # Trouver la plus petite valeur pour la colonne dans le groupe
    #         for idx, val in group.iterrows():  # Parcourir chaque ligne du groupe
    #             if val[column] == min_value:  # Vérifier si la valeur correspond au minimum
    #                 styles[idx] = 'font-weight: bold'  # Appliquer le style en gras
    #     return styles

    # Appliquer les styles au DataFrame
    styled_df = (
        df_results.style
        .apply(add_double_border, axis=1)
        # .apply(lambda x: highlight_min(df_results, "Non relaxed problem cost"), subset=["Non relaxed problem cost"], axis=0)
        # .apply(lambda x: highlight_min(df_results, "Logdet"), subset=["Logdet"], axis=0)
    )
    # Afficher le DataFrame stylisé
    return styled_df

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  