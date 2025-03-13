import pandas as pd
import random
import numpy as np
import torch

def generate_uniform_points_in_p_norm_ball(n, k, p):
    points = []
    while len(points) < n:
        # Générer un point uniformément dans le cube [-1,1]^k
        x = np.random.uniform(-1, 1, k)
        # Calculer la norme p du point
        norm_p = np.sum(np.abs(x) ** p) ** (1 / p)
        # Vérifier si le point est dans la boule unité de la norme p
        if norm_p <= 1:
            points.append(x)
    return np.array(points)

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