import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

import plotly.graph_objs as go
from plotly.subplots import make_subplots

def surface_ellipse_3d(center, Lambda, q, conformal_value):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """
    
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    # Scale points to match the conformal value under the q-norm
    scaling = conformal_value * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipsoid to the center
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))
    
    # Reshape to original grid shape
    X = ellipsoid_points[0, :].reshape(x.shape)
    Y = ellipsoid_points[1, :].reshape(y.shape)
    Z = ellipsoid_points[2, :].reshape(z.shape)
    
    return [X, Y, Z]


def add_ellipse(ax, center, Lambda, q, conformal_value, color='red', label = None, size=3):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """

    # Generate a set of points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    
    # Scale the unit circle points to match the conformal value under the q-norm
    scaling = conformal_value * np.power(np.sum(np.abs(unit_circle) ** q, axis=0), -1/q)
    scaled_points = unit_circle * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipse to the center
    ellipse_points = transformed_points + np.reshape(center, (2, 1))
    
    # Add the ellipse to the existing plot
    ax.plot(ellipse_points[0, :], ellipse_points[1, :], label=label, c=color, linewidth=size)
   #  ax.scatter(*center, color='red')
    ax.legend()
    
    return ax



def add_ellipse_multiple_lambdas(ax, mu, rotation, tab_diag, tab_q, split, conformal_value, color='red', label=None, size = 3):
   """
      plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
      conformal_value = radius ** q
   """
   # cmap = cm.get_cmap('tab10', len(tab_diag))
   tab_weights = np.array([1/4, 1/4, 1/4, 1/4])
   cumulative_weight = np.cumsum(tab_weights)
   points = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32)
   idx_lambdas = split.get_idx(points)
   for j in range(4):
      i = idx_lambdas[j]
      Lambda =  tab_diag[i]
      # Generate a set of points on a unit circle      
      if j > 0:
         theta = np.linspace(2 * np.pi * cumulative_weight[j-1], 2 * np.pi * cumulative_weight[j], 500)
      else:
         theta = np.linspace(0, 2 * np.pi * cumulative_weight[j], 500)
      
      unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
      
      # Scale the unit circle points to match the conformal value under the q-norm
      scaling = conformal_value * np.power(np.sum(np.abs(unit_circle) ** tab_q[i], axis=0), -1/tab_q[i])
      scaled_points = unit_circle * scaling
      
      # Apply the linear transformation Lambda
      transformed_points = np.linalg.inv(Lambda) @ scaled_points

      # print("transformed points : ", transformed_points.shape)
      
      # Translate the ellipse to the center
      ellipse_points = rotation.T @ transformed_points + mu.reshape(-1, 1)
      # ellipse_points = transformed_points 
      
      # Add the ellipse to the existing plot
      if j == 0:
         ax.plot(ellipse_points[0, :], ellipse_points[1, :], c=color, label=label,  linewidth=size)
      else:
         ax.plot(ellipse_points[0, :], ellipse_points[1, :], c=color, linewidth=size)

      if j != 0:
         first_point = ellipse_points[:, 0]
         ax.plot([last_point[0], first_point[0]], [last_point[1], first_point[1]], c=color, linewidth=size)
 
      else:
         first_point_boucle = ellipse_points[:, 0].copy()
      
      last_point = ellipse_points[:, -1]
      
      if j == len(tab_diag) - 1 and len(tab_diag) > 2:
         # pass
         ax.plot([last_point[0], first_point_boucle[0]], [last_point[1], first_point_boucle[1]], c=color, linewidth=size)
      
      ax.legend()
   
   return ax




def add_ellipse_3d_interactif(center, Lambda, q, conformal_value, color='red'):
    """
    Crée un ellipsoïde interactif avec Plotly.
    """
    # Génération des points d'une sphère unitaire
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    # Mise à l'échelle avec la norme q
    scaling = conformal_value * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling

    # Application de la transformation linéaire Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points

    # Translation de l'ellipsoïde vers le centre
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))

    # Remise en forme pour le tracé
    X = ellipsoid_points[0, :].reshape(x.shape)
    Y = ellipsoid_points[1, :].reshape(y.shape)
    Z = ellipsoid_points[2, :].reshape(z.shape)

    # Création de la surface Plotly
    surface = go.Surface(x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]], opacity=0.5, showscale=False)
    
    return surface


def add_ellipse_3d(ax, center, Lambda, q, conformal_value, color='red', label=None):
    """
       plot -> {y, \| \Lambda (u - center) \|_q^q \leq conformal_value}  }
       conformal_value = radius ** q
    """
    
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    unit_sphere = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    
    # Scale points to match the conformal value under the q-norm
    scaling = conformal_value * np.power(np.sum(np.abs(unit_sphere) ** q, axis=0), -1/q)
    scaled_points = unit_sphere * scaling
    
    # Apply the linear transformation Lambda
    transformed_points = np.linalg.inv(Lambda) @ scaled_points
    
    # Translate the ellipsoid to the center
    ellipsoid_points = transformed_points + np.reshape(center, (3, 1))
    
    # Reshape to original grid shape
    X = ellipsoid_points[0, :].reshape(x.shape)
    Y = ellipsoid_points[1, :].reshape(y.shape)
    Z = ellipsoid_points[2, :].reshape(z.shape)
    
    # Plot the ellipsoid
    ax.plot_surface(X, Y, Z, color=color, alpha=0.5, edgecolor='k', label=label)
    
    return ax