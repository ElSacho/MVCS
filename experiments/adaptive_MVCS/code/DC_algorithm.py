import cvxpy as cp
import numpy as np
from scipy.special import gammaln
from sklearn.cluster import KMeans
from torch_functions import *
import torch


def order_points(y, Lambda, eta, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    
    return y


def proportion_outliers(y, Lambda, eta, nu, q):
    n = len(y) 
    count = 0
    for i in range(n):
        if np.linalg.norm(Lambda @ y[i, :] + eta, ord=q) > nu:
            count += 1

    return count / n

def get_volume(Lambda, nu, q):
    k  = Lambda.shape[0]
    return 1/np.linalg.det(Lambda/nu) * np.exp(H_dc(q, k))

def linearization(y, Lambda, eta, Lambda_old, eta_old, p):
    """
    Linearize f(Lambda, eta) = ||Lambda y + eta||_p around (Lambda_old, eta_old).

    Parameters:
    y : numpy.ndarray
        Input vector y.
    Lambda : numpy.ndarray
        Current Lambda matrix.
    eta : numpy.ndarray
        Current eta vector.
    Lambda_old : numpy.ndarray
        Previous Lambda matrix (point of linearization).
    eta_old : numpy.ndarray
        Previous eta vector (point of linearization).
    p : float
        p-norm parameter.

    Returns:
    cp.Expression
        Linearized function.
    """
    # Compute v_old = Lambda_old @ y + eta_old
    v_old = Lambda_old @ y + eta_old

    # Compute the norm of v_old
    norm_v_old = np.linalg.norm(v_old, ord=p)

    # Avoid division by zero
    if norm_v_old == 0:
        raise ValueError("norm_v_old is zero, linearization is not well-defined.")

    # Compute the gradient components
    abs_v_old = np.abs(v_old)
    sign_v_old = np.sign(v_old)
    grad_factor = abs_v_old ** (p - 1) / (norm_v_old ** (p - 1))
    grad_eta = grad_factor * sign_v_old  # Gradient w.r.t eta
    grad_Lambda = np.outer(grad_factor * sign_v_old, y)  # Gradient w.r.t Lambda

    # Linearized function
    linearized_f = (
        cp.trace(grad_Lambda.T @ (Lambda - Lambda_old)) +  # Linear term for Lambda
        grad_eta.T @ (eta - eta_old)  # Linear term for eta
    )

    return linearized_f

def H_dc(q, k):
    term1 = k * gammaln(1 + 1/q)  
    term2 = gammaln(1 + k/q)      
    term3 = k * np.log(2)
    return term1 - term2 + term3

def calculate_true_objective(y, Lambda, eta, p, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    nu = distances[sorted_indices[p-1]]
    
    obj = - np.log(np.linalg.det(Lambda)) + nu

    return obj

def calculate_volume(y, Lambda, eta, p, q):
    n = len(y) 
    k = len(y[0]) 
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ y[i, :] + eta, ord=q)
    sorted_indices = np.argsort(distances)[::-1]
    y = y[sorted_indices, :]
    nu = distances[sorted_indices[p-1]]
    
    volume = - np.log(np.linalg.det(Lambda)) + k * np.log(nu) + np.exp(H_dc(q, k))
    
    return np.exp(volume)

def get_nu_with_mu(y, Lambda, mu, p, q):
    n = len(y) 
    k = len(y[0])
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.linalg.norm(Lambda @ (y[i, :] - mu) , ord=q)
    
    sorted_indices = np.argsort(distances)[::-1]
    nu = distances[sorted_indices[p-1]]

    return nu



def one_step_DC(y, Lambda_old, eta_old, p, q):
    # Variables
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y

    y = order_points(y, Lambda_old, eta_old, q)
    
    Lambda = cp.Variable((k, k), PSD=True)
    eta = cp.Variable(k)
    nu = cp.Variable()  

    Lambda.value = Lambda_old
    eta.value = eta_old

    constraints = []
    constraints.append(Lambda >> 0)
    
    sigma_p = 0
    sigma_p_minus_one = 0

    for i in range(n):
        sigma_p += cp.pos( cp.norm(Lambda @ y[i, :] + eta, q) - nu )
        if i < p - 1:
            sigma_i = linearization(y[i], Lambda, eta, Lambda_old, eta_old, q)
            sigma_p_minus_one += sigma_i
    
    objective = - cp.log_det(Lambda) + sigma_p + p * nu - sigma_p_minus_one
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    if problem.status != cp.OPTIMAL:
        print("Problem not solved")
        print(problem.status)
        print(problem.value)

    return Lambda.value, eta.value, nu.value, problem.value


def one_step_DC_one_p(y, Lambda_old, eta_old, p, q):
    # Variables
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y

    y = order_points(y, Lambda_old, eta_old, q)
    
    Lambda = cp.Variable((k, k), PSD=True)
    eta = cp.Variable(k)
    nu = cp.Variable()  

    Lambda.value = Lambda_old
    eta.value = eta_old

    constraints = []
    constraints.append(Lambda >> 0)
    
    sigma_p = cp.norm(Lambda @ y[p-1, :] + eta, q)

    objective = - cp.log_det(Lambda) + sigma_p 

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    if problem.status != cp.OPTIMAL:
        print("Problem not solved")
        print(problem.status)
        print(problem.value)

    return Lambda.value, eta.value, nu.value, problem.value

def DC_algorithm(y, p, q, beta, max_iter=100, tol=1e-6, verbose=False):
    # Initialization
    n = len(y)  # Number of samples
    k = len(y[0])  # Dimension of y
    
    # Initial values
    mu_old = np.mean(y, axis=0)
    M_old = np.linalg.inv(np.cov(y.T))
    eigenvalues, eigenvectors = np.linalg.eigh(M_old)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  
    Lambda_old = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    eta_old = - np.linalg.inv(Lambda_old) @ mu_old

    center = mu_old
    Lambda = Lambda_old
    vol_old = np.inf
    obj_old = np.inf
    true_obj_old = np.inf

    beta_0 = beta
    v_Lambda_k = Lambda_old
    v_eta_k = eta_old
    
    for i in range(max_iter):
        
        Lambda, eta, nu, obj = one_step_DC(y, Lambda_old, eta_old, p, q)
        vol = calculate_volume(y, Lambda, eta, p, q)
        true_obj = calculate_true_objective(y, Lambda, eta, p, q)
        
        if verbose:
            # print(f"Iteration {i}, obj = {obj}")
            print(f"Volume = {vol}")
            print(f"True Objective = {true_obj}")

        if np.abs(true_obj - true_obj_old) < tol:
            break

        beta = beta_0 / ( i + 1)
        v_Lambda_k = beta*v_Lambda_k + (1 - beta) * (Lambda - Lambda_old)
        v_eta_k = beta*v_eta_k + (1 - beta) * (eta - eta_old)
        Lambda = Lambda_old + v_Lambda_k
        eta = eta_old + v_eta_k
        
        # Update the values
        Lambda_old = Lambda
        eta_old = eta
        true_obj_old = true_obj
        
    return Lambda, eta, nu, obj


# class DC:
#     def __init__(self, model):
#         self.model = model
#         pass

#     def get_all_f_x(self, dataloader):
#         empty = True
#         for (x, _) in dataloader:    
#             if empty:
#                 f_x = self.model(x).detach().numpy()
#                 empty = False
#             else:
#                 f_x = np.vstack((f_x, self.model(x).detach().numpy()))
            
#         return f_x


#     def get_all_f_x_y(self, dataloader):
#         empty = True
#         for (x, y_) in dataloader:    
#             if empty:
#                 f_x = self.model(x).detach().numpy()
#                 y = y_.numpy()
#                 empty = False
#             else:
#                 f_x = np.vstack((f_x, self.model(x).detach().numpy()))
#                 y = np.vstack((y, y_.numpy()))
            
#         return f_x, y

#     def fit(self, trainloader, alpha, q, beta = 0.0, max_iter=100, tol=1e-6, verbose=False):
#         print("Starting the DC algorithm")
#         f_x, y = self.get_all_f_x_y(trainloader)
#         residuals = y - f_x
#         p = int(residuals.shape[0] * alpha)
#         self.q = q
#         self.Lambda, self.eta, radius, self.obj = DC_algorithm(residuals, p, q, beta, max_iter, tol, verbose)
#         self.mu = - np.linalg.inv(self.Lambda) @ self.eta
#         print("DC algorithm finished")
    
#     def conformalize(self, calibration_loader, alpha):
#         f_x, y = self.get_all_f_x_y(calibration_loader)
#         residuals_calibration = y - f_x
#         n = residuals_calibration.shape[0]
#         p = n - int(np.ceil((n+1)*(1-alpha)))
#         if p < 0:
#             raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")
#         self.nu = get_nu_with_mu(residuals_calibration, self.Lambda, self.mu, p, self.q)
    
#     def get_Lambda(self):
#         return self.Lambda
    
#     def get_normalized_Lambda(self):
#         return self.Lambda / self.nu
    
#     def get_eta(self):
#         return self.eta
    
#     def get_mu(self):
#         return self.mu
    
#     def get_nu(self):
#         return self.nu
    
#     def get_coverage(self, test_loader=None, x_test=None, y_test=None):
#         if test_loader is None:
#             f_x, y = self.get_all_f_x_y(test_loader)
#         else:
#             f_x = self.model(x_test).detach().numpy()
#             y = y_test.numpy()
#         residuals_test = y - f_x
#         return 1 - proportion_outliers(residuals_test, self.Lambda, self.eta, self.nu, self.q)

#     def get_averaged_volume(self, test_loader=None, x_test=None, y_test=None):
#         return get_volume(self.Lambda, self.nu, self.q)
    
#     def get_volume(self, test_loader=None, x_test=None, y_test=None):
#         return get_volume(self.Lambda, self.nu, self.q)

def get_all_x_y(dataloader):
    X, Y = [], []
    for (x, y_) in dataloader:
        X.append(x.numpy())
        Y.append(y_.numpy())
    
    X = np.vstack(X)
    Y = np.vstack(Y)
    
    return X, Y



class DC:
    def __init__(self, model):
        self.model = model
        pass

    def get_all_f_x(self, dataloader):
        empty = True
        for (x, _) in dataloader:    
            if empty:
                f_x = self.model(x).detach().numpy()
                empty = False
            else:
                f_x = np.vstack((f_x, self.model(x).detach().numpy()))
            
        return f_x



    def fit(self, trainloader, alpha, q=2, beta=0.0, max_iter=100, tol=1e-6, verbose=False, n_clusters="auto"):
        # Retrieve x and y from the dataloader
        X, Y = get_all_x_y(trainloader)
        self.q = torch.tensor(q, dtype=torch.float32)
        # Perform clustering on X
        self.n_clusters = n_clusters
        if isinstance(n_clusters, str) and n_clusters.lower() == "auto":
            n_data = X.shape[0]
            n_clusters = max(1, n_data // 500 + 1)  # Évite d'avoir 0 clusters
            if verbose:
                print(f"Fitting DC with {n_clusters} clusters")
        elif isinstance(n_clusters, str) and n_clusters.lower() == "none":
            residuals = Y - self.model(torch.tensor(X)).detach().numpy()
            p = int(residuals.shape[0] * alpha)
            self.Lambda, self.eta, radius, obj = DC_algorithm(residuals, p, q, beta, max_iter, tol, verbose)
            self.mu = - np.linalg.inv(self.Lambda) @ self.eta
            return
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(X)
        
        self.cluster_models = []
        
        for cluster_idx in range(n_clusters):
            # Select data points belonging to the cluster
            cluster_mask = cluster_labels == cluster_idx
            X_cluster = X[cluster_mask]
            Y_cluster = Y[cluster_mask]

            if verbose:
                print(f"Processing cluster {cluster_idx+1}/{n_clusters} of size {X_cluster.shape[0]}")
            
            # Compute f(x) for the cluster
            f_x_cluster = self.model(torch.tensor(X_cluster)).detach().numpy()
            residuals = Y_cluster - f_x_cluster
            p = int(residuals.shape[0] * alpha)
            
            # Run the DC algorithm on this cluster
            Lambda, eta, radius, obj = DC_algorithm(residuals, p, q, beta, max_iter, tol)
            mu = -np.linalg.inv(Lambda) @ eta
            
            # Store the results for this cluster
            self.cluster_models.append({
                "cluster_idx": cluster_idx,
                "Lambda": Lambda/radius,
                "eta": eta,
                "mu": mu,
                "obj": obj,
            })

    def get_Lambdas(self, x):
        # Prédire le cluster de chaque x_test
        if isinstance(self.n_clusters, str) and self.n_clusters.lower() == "none":
            Lambdas = [self.Lambda for _ in range(x.shape[0])]
            mus = [self.mu for _ in range(x.shape[0])]
            return Lambdas, mus
        cluster_assignments = self.kmeans.predict(x)
        
        # Récupérer les matrices Lambda correspondantes
        Lambdas = [self.cluster_models[cluster_idx]["Lambda"] for cluster_idx in cluster_assignments]
        mus = [self.cluster_models[cluster_idx]["mu"] for cluster_idx in cluster_assignments]
        
        return Lambdas, mus
    
    def conformalize(self, calibration_loader, alpha):
        x_calibration, y_calibration = get_all_x_y(calibration_loader)
        f_x_calibration = self.model(torch.tensor(x_calibration))
        Lambdas_calibration, mu_calibration = self.get_Lambdas(x_calibration)
        Lambdas_calibration = torch.tensor(Lambdas_calibration, dtype=torch.float32)
        mu_calibration = torch.tensor(mu_calibration, dtype=torch.float32)
        y_calibration = torch.tensor(y_calibration, dtype=torch.float32)
        f_x_calibration_corrected = f_x_calibration + mu_calibration
        n = y_calibration.shape[0]
        p = n - int(np.ceil((n+1)*(1-alpha)))
        if p < 0:
            raise ValueError("The number of calibration samples is too low to reach the desired alpha level.")
        
        q = torch.tensor(self.q, dtype=torch.float32)
        self.nu = get_p_value_mu(y_calibration, f_x_calibration_corrected, Lambdas_calibration, p, q)


    def get_averaged_volume(self, testloader=None, x_test=None):
        with torch.no_grad():
            if self.nu is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")
            
            if testloader is not None:
                x_test, y_test = get_all_x_y(testloader)

            elif x_test is not None:
                x_test = x_test.detach().numpy()

            else:
                raise ValueError("You must provide either a testloader or x_test and y_test.")
                    
            f_x_test = self.model(torch.tensor(x_test))
            Lambdas_test, mu_test = self.get_Lambdas(x_test)
            Lambdas_test = torch.tensor(Lambdas_test, dtype=torch.float32)


            
            return calculate_averaged_volume(Lambdas_test, self.nu, torch.tensor(self.q, dtype=torch.float32))


    def get_coverage(self, testloader=None, x_test=None, y_test=None):   
        with torch.no_grad(): 
            if self.nu is None:
                raise ValueError("You must call the `conformalize_ellipsoids` method before.")    
                    
            if testloader is not None:
                x_test, y_test = get_all_x_y(testloader)

            elif x_test is not None and y_test is not None:
                x_test = x_test.detach().numpy()
                y_test = y_test.detach().numpy()

            else:
                raise ValueError("You must provide either a testloader or x_test and y_test.")
                
                    
            f_x_test = self.model(torch.tensor(x_test, dtype=torch.float32))
            Lambdas_test, mu_test = self.get_Lambdas(x_test)
            Lambdas_test = torch.tensor(Lambdas_test, dtype=torch.float32)
            mu_test = torch.tensor(mu_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            f_x_test_corrected = f_x_test + mu_test
              
            coverage = calculate_coverage(y_test, f_x_test_corrected, Lambdas_test, self.nu, torch.tensor(self.q, dtype=torch.float32)).mean().item()
                
            return coverage

