import numpy as np
import scipy.special

def get_nu_with_mu(y, Lambda, mu, p, q):
    n = len(y) 
    k = len(y[0])
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ (y[i, :] - mu) , ord=q)
    
    sorted_indices = np.argsort(distances)[::-1]
    nu = distances[sorted_indices[p-1]]

    return nu

def proportion_outliers(y, Lambda, eta, nu, q):
    n = len(y) 
    count = 0
    for i in range(n):
        if np.linalg.norm(Lambda @ y[i, :] + eta, ord=q) > nu:
            count += 1

    return count / n

def H_np(q, k):
    term1 = k * scipy.special.gammaln(1 + 1 / q)
    term2 = scipy.special.gammaln(1 + k / q)
    term3 = k * np.log(2.0)
    return term1 - term2 + term3

def get_volume(Lambda, nu, q):
    k  = Lambda.shape[0]
    return 1/np.linalg.det(Lambda/nu) * np.exp(H_np(q, k))
