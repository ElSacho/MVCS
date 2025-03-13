import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.linalg import cholesky

class dataGenerator:
    def __init__(self, input_dim, output_dim, noise_type, f=None, local_perturbation = None, scale_noise=1.0, beta_star=None, covariance_matrix = None):
        self.noise_type = noise_type
        self.scale_noise = scale_noise
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        else:
            A = np.random.randn(output_dim, output_dim)
            self.covariance_matrix = A @ A.T
        self.beta_star = np.random.randn(input_dim, output_dim) if beta_star is None else beta_star
        self.local_perturbation = local_perturbation
        if f is None:
            def f(x, beta):
                return np.dot(x, beta)
            self.f = f
        else:
            self.f = f
        self.d = input_dim
        self.k = output_dim

    def generate_specific_x(self, n):
        x_specific = np.random.randn(1, self.d-1)
        x_specific = np.repeat(x_specific, n, axis=0)
        x_specific = np.hstack([np.ones((n, 1)), x_specific])

        f_star = self.f(x_specific, self.beta_star)
        data_perturbation_specific = get_perturbation(self.noise_type, n, self.k, covariance_matrix=self.covariance_matrix)
        if self.local_perturbation is not None:
            rotation = self.local_perturbation.get(x_specific)
            data_perturbation_specific = np.einsum('njk,nk->nj', rotation, data_perturbation_specific)
        y_specific = f_star + data_perturbation_specific
        
        return x_specific, y_specific
    
    def generate_y_given_x(self, x_specific, n=1):
        if n != 1:
            x_specifics = np.repeat(x_specific, n, axis=0)
        else:
            x_specifics = x_specific
        
        f_star = self.f(x_specifics, self.beta_star)
        data_perturbation_specific = get_perturbation(self.noise_type, n, self.k, covariance_matrix=self.covariance_matrix)
        if self.local_perturbation is not None:
            rotation = self.local_perturbation.get(x_specifics)
            data_perturbation_specific = np.einsum('njk,nk->nj', rotation, data_perturbation_specific)
        y_specific = f_star + data_perturbation_specific
        return x_specifics, y_specific

    def generate(self, n):
        x_train = np.random.randn(n, self.d-1)
        x_train = np.hstack([np.ones((n, 1)), x_train])
        
        f_star = self.f(x_train, self.beta_star)
        data_perturbation_train = get_perturbation(self.noise_type, n, self.k, covariance_matrix=self.covariance_matrix)
        if self.local_perturbation is not None:
            rotation = self.local_perturbation.get(x_train)
            data_perturbation_train = np.einsum('njk,nk->nj', rotation, data_perturbation_train)
        y_train = f_star + data_perturbation_train
        
        return x_train, y_train
    
    def generate_train_calibration_test(self, n_train, n_calibration, n_test):
        x_train, y_train = self.generate(n_train)
        x_calibration, y_calibration = self.generate(n_calibration)
        x_test, y_test = self.generate(n_test)
        return x_train, y_train, x_calibration, y_calibration, x_test, y_test
    
    def generate_perturbation(self, n):
        data_perturbation_train = get_perturbation(self.noise_type, n, self.k, covariance_matrix=self.covariance_matrix)
        return data_perturbation_train
    
    def generate_perturbation_train_calibration_test(self, n_train, n_calibration, n_test):
        data_perturbation_train = self.generate_perturbation(n_train)
        data_perturbation_calibration = self.generate_perturbation(n_calibration)
        data_perturbation_test = self.generate_perturbation(n_test)
        return data_perturbation_train, data_perturbation_calibration, data_perturbation_test
    
def get_perturbation(perturbation, n_train, k, covariance_matrix=None):
    mean = np.zeros(k)
    if perturbation == "laplace":
        data_perturbation_train = np.random.laplace(0, 1, (n_train, k))
    elif perturbation == "gaussian":
        if covariance_matrix is None:
            raise ValueError("Covariance matrix must be provided for gaussian perturbation")
        data_perturbation_train = np.random.multivariate_normal(mean, covariance_matrix, n_train)
    elif perturbation == "uniform":
        data_perturbation_train = np.random.uniform(-1, 1, (n_train, k))
    elif perturbation == "pareto":
        data_perturbation_train = np.random.pareto(a=2.0, size=(n_train, k))  # a est le paramètre de forme
    elif perturbation == "lognormal":
        data_perturbation_train = np.random.lognormal(mean=0.0, sigma=1.0, size=(n_train, k))
    elif perturbation == "exponential":
        data_perturbation_train = np.random.exponential(scale=1.0, size=(n_train, k))
    elif perturbation == "chisquare":
        data_perturbation_train = np.random.chisquare(df=2, size=(n_train, k))  # df est le nombre de degrés de liberté
    elif perturbation == "poisson":
        data_perturbation_train = np.random.poisson(lam=3.0, size=(n_train, k))
    elif perturbation == "gamma":
        data_perturbation_train = np.random.gamma(shape=2.0, scale=1.0, size=(n_train, k))
    elif perturbation == "beta":
        data_perturbation_train = np.random.standard_t(df=3, size=(n_train, k))  # df est le nombre de degrés de liberté
    elif perturbation == "cauchy":
        data_perturbation_train = np.random.standard_cauchy(size=(n_train, k))
    else:
        data_perturbation_train = np.zeros((n_train, k))
    return data_perturbation_train


# def inverse_distance(point1, point2):
#     """
#     Calculates the inverse-distance between two points
#     :param point1: First point's coordinates
#     :param point2: Second point's coordinates
#     :return: Inverse-distance value
#     """
#     return (1 / (np.linalg.norm(point1 - point2) + 1e-6)) ** 4

# def point_cov(point):
#     """
#     Generates a covariance matrix for a point influenced by the 4 $\mu_i$ points used to simulate data
#     :param point: The point's coordinates
#     :return: Local covariance matrix for the point
#     """
#     # first part.
#     mu_1 = np.array([-5, 5])
#     cov_1 = np.array([[0.1, -0.09], [-0.09, 0.1]])

#     # second part.
#     mu_2 = np.array([5, 5])
#     cov_2 = np.array([[0.1, 0.09], [0.09, 0.1]])

#     # first part.
#     mu_3 = np.array([-5, -5])
#     cov_3 = np.array([[0.1, 0.09], [0.09, 0.1]])

#     # second part.
#     mu_4 = np.array([5, -5])
#     cov_4 = np.array([[0.1, -0.09], [-0.09, 0.1]])

#     sum_inv_dist = (
#         inverse_distance(point, mu_1)
#         + inverse_distance(point, mu_2)
#         + inverse_distance(point, mu_3)
#         + inverse_distance(point, mu_4)
#     )
#     cov = (
#         (inverse_distance(point, mu_1) / sum_inv_dist) * cov_1
#         + (inverse_distance(point, mu_2) / sum_inv_dist) * cov_2
#         + (inverse_distance(point, mu_3) / sum_inv_dist) * cov_3
#         + (inverse_distance(point, mu_4) / sum_inv_dist) * cov_4
#     )
#     return cov


# def gen_noise_from_x(x, noise_type):
#     """
#     Generates noise from the point according to the covariance matrix generated above
#     :param x: The point's coordinates
#     :return: The point's noisy coordinates
#     """
#     cov = point_cov(x)
#     if noise_type == "gaussian":
#         b = np.random.normal(0, 1, size=(1, 2))
#     elif noise_type == "cauchy":
#         b = np.random.standard_cauchy(size=(1, 2))*0.01
#     elif noise_type == "exponential":
#         b = np.random.exponential(1, size=(1, 2))*0.1
#     else:
#         raise ValueError("Noise type not supported")
#     c = b @ cholesky(cov)
#     return c.squeeze()


# def generate_synthetic_data(n=10_000, noise_type='gaussian'):
#     """
#     Generates a synthetic data set
#     :param n: Desired number of instances
#     :return: x_train, y_train, x_calibration, y_calibration
#     """
#     df = pd.DataFrame(columns=["x1", "x2", "y1", "y2"], index=range(n))
#     df["x1"] = np.random.uniform(-5, 5, size=n)
#     df["x2"] = np.random.uniform(-5, 5, size=n)

#     df["y1"] = 0.7 * df["x1"] + 0.3 * df["x2"]
#     df["y2"] = 0.2 * df["x1"] + 0.8 * df["x2"]

#     df["noise_np"] = df.apply(
#         lambda x: gen_noise_from_x(np.array([x["x1"], x["x2"]]), noise_type), axis=1
#     )
    
#     df["epsilon_1"] = df.noise_np.apply(lambda x: x[0])
#     df["epsilon_2"] = df.noise_np.apply(lambda x: x[1])

#     df["y1"] = df["y1"] + df["epsilon_1"]
#     df["y2"] = df["y2"] + df["epsilon_2"]

#     train, test = train_test_split(df, test_size=0.2)
#     train, cal = train_test_split(train, test_size=0.2)

#     x_train = train[["x1", "x2"]].to_numpy()
#     y_train = train[["y1", "y2"]].to_numpy()

#     x_test = test[["x1", "x2"]].to_numpy()
#     y_test = test[["y1", "y2"]].to_numpy()

#     x_calibration = cal[["x1", "x2"]].to_numpy()
#     y_calibration = cal[["y1", "y2"]].to_numpy()

#     return x_train, y_train, x_test, y_test, x_calibration, y_calibration

# def generate_synthetic_data_one_x(n=10_000, noise_type='gaussian'):
#     """
#     Generates a synthetic data set
#     :param n: Desired number of instances
#     :return: x_train, y_train, x_calibration, y_calibration
#     """
#     df = pd.DataFrame(columns=["x1", "x2", "y1", "y2"], index=range(n))
#     val1 = np.random.uniform(-5, 5, size=1)
#     val2 = np.random.uniform(-5, 5, size=1)
#     df["x1"] = [val1[0]] * n
#     df["x2"] = [val2[0]] * n

#     df["y1"] = 0.7 * df["x1"] + 0.3 * df["x2"]
#     df["y2"] = 0.2 * df["x1"] + 0.8 * df["x2"]

#     df["noise_np"] = df.apply(
#         lambda x: gen_noise_from_x(np.array([x["x1"], x["x2"]]), noise_type), axis=1
#     )
    
#     df["epsilon_1"] = df.noise_np.apply(lambda x: x[0])
#     df["epsilon_2"] = df.noise_np.apply(lambda x: x[1])

#     df["y1"] = df["y1"] + df["epsilon_1"]
#     df["y2"] = df["y2"] + df["epsilon_2"]

#     x_train = df[["x1", "x2"]].to_numpy()
#     y_train = df[["y1", "y2"]].to_numpy()

#     return x_train, y_train

# if __name__=="__main__":
#     import matplotlib.pyplot as plt
    
#     x_train, y_train, x_calibration, y_calibration = generate_synthetic_data(n=100)
#     print(y_train.shape)
#     # plt.plot(y_train)
#     plt.scatter(y_train[:,0], y_train[:,1])
#     plt.show()
