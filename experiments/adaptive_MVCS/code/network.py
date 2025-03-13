import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class LinearMatrixPredictor(nn.Module):
    def __init__(self, input_dim, output_rows, output_cols):
        super(LinearMatrixPredictor, self).__init__()
        self.fc = nn.Linear(input_dim, output_rows * output_cols)
        self.output_rows = output_rows
        self.output_cols = output_cols

    def forward(self, x):
        output = self.fc(x)
        return output.view(-1, self.output_rows, self.output_cols)
    
class MatrixPredictor(nn.Module):
    def __init__(self, input_dim, output_rows, output_cols, hidden_dim = 100, n_hidden_layers = 0):
        super(MatrixPredictor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.n_hidden_layers = n_hidden_layers
        if n_hidden_layers > 0:
            self.tab_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_rows * output_cols)
        self.output_rows = output_rows
        self.output_cols = output_cols

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        if self.n_hidden_layers > 0:
            for hidden in self.tab_hidden:
                x = F.relu(hidden(x))
        output = self.fc2(x)
        return output.view(-1, self.output_rows, self.output_cols)
    
    def init_weights(self, m, std = 0.02):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)  # Loi normale N(0, 0.02^2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class WeightModel(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim = 10):
        super(WeightModel, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def fit(self, train_loader, epochs, verbose = True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')


    def eval(self, X_val, Y_val):
        with torch.no_grad():
            output = self(X_val)
            loss = F.mse_loss(output, Y_val)
            print(f'Validation Loss: {loss.item()}')

class Network(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim = 100, n_hidden_layers = 1):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.tab_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.fc3 = nn.Linear(hidden_dim, output_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for hidden in self.tab_hidden:
            x = F.relu(hidden(x))
        x = self.fc3(x)
        return x

    def fit(self, train_loader, epochs, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    
    def fit_and_plot(self, train_loader, test_loader, epochs, keep_best = False, lr = 0.001, verbose=False):
        if not keep_best:
            print("Training... (keep the last model)")
        best_weights = None
        best_loss = float('inf')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            train_loss = self.eval(train_loader)
            test_loss = self.eval(test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if keep_best and test_loss < best_loss:
                best_loss = test_loss
                best_weights = copy.deepcopy(self.state_dict())
            if verbose:
                print(f'Epoch: {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        if keep_best:
            self.load_state_dict(best_weights)
        return train_losses, test_losses

    def freeze_all_but_last_layer(self):
        for name, param in self.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False

    def eval(self, loader):
        with torch.no_grad():
            loss = 0
            for i, (X_batch, Y_batch) in enumerate(loader):
                output = self(X_batch)
                loss += F.mse_loss(output, Y_batch)
            return loss / len(loader)
        

class LinearNetwork(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_channels, output_channels)


    def forward(self, x):
        x = self.fc1(x)
        return x

    def fit(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')


    def eval(self, X_val, Y_val):
        with torch.no_grad():
            output = self(X_val)
            loss = F.mse_loss(output, Y_val)
            print(f'Validation Loss: {loss.item()}')


class DiagNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim = 10):
        super(DiagNetwork, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.abs(x)
    

class ParametricRotationMatrix2D(nn.Module):
    def __init__(self, input_dim, n, hidden_dim = 100):
        """
        Modèle générant une matrice de rotation paramétrée par des angles.

        Args:
            input_dim (int): Dimension de l'entrée x.
            n (int): Taille de la matrice de rotation (n x n).
        """
        super(ParametricRotationMatrix2D, self).__init__()
        self.input_dim = input_dim
        self.n = n
        self.num_angles = n * (n - 1) // 2  # Nombre d'angles nécessaires

        # Couche linéaire pour générer les angles
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_angles)

    def forward(self, x):
        """
        Calcule la matrice de rotation.

        Args:
            x (torch.Tensor): Entrée de taille (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Matrice de rotation de taille (batch_size, n, n).
        """
        batch_size = x.size(0)

        # Générer les angles
        angles = self.fc(x)  # (batch_size, num_angles)
        angles = F.relu(angles)
        angles = self.fc2(angles)
        angles = angles % (2 * torch.pi)  # Normaliser les angles dans [0, 2*pi]

        # Initialiser la matrice d'identité
        rotation_matrix = torch.eye(self.n, device=x.device).repeat(batch_size, 1, 1)  # (batch_size, n, n)

        # Appliquer des rotations successives
        index = 0
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                # Créer une matrice de rotation dans le plan (i, j)
                cos_theta = torch.cos(angles[:, index])
                sin_theta = torch.sin(angles[:, index])

                # Matrice d'identité temporaire
                sub_matrix = torch.eye(self.n, device=x.device).repeat(batch_size, 1, 1)
                sub_matrix[:, i, i] = cos_theta
                sub_matrix[:, i, j] = -sin_theta
                sub_matrix[:, j, i] = sin_theta
                sub_matrix[:, j, j] = cos_theta

                # Multiplier par la rotation accumulée
                rotation_matrix = torch.bmm(rotation_matrix, sub_matrix)  # (batch_size, n, n)
                index += 1

        return rotation_matrix
    



class RotationMatrixNet(nn.Module):
    def __init__(self, input_dim, k, hidden_dim = 100):
        """
        Initialise le réseau de neurones.
        :param input_dim: Dimension des features d'entrée.
        :param k: Taille de la matrice de rotation (k x k).
        """
        super(RotationMatrixNet, self).__init__()
        self.input_dim = input_dim
        self.k = k

        # Réseau de base pour générer une matrice quelconque
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k * k)  # Sortie : k x k éléments
        )

    def forward(self, x):
        """
        Propagation avant.
        :param x: Entrée de taille (batch_size, input_dim).
        :return: Matrice de rotation de taille (batch_size, k, k).
        """
        # Générer une matrice quelconque
        batch_size = x.size(0)
        matrix_flat = self.fc(x)  # Sortie : (batch_size, k*k)
        matrix = matrix_flat.view(batch_size, self.k, self.k)  # Reshape en (batch_size, k, k)

        # Décomposition QR pour obtenir une matrice orthogonale
        q, r = torch.linalg.qr(matrix)  # q : orthogonale, r : triangulaire supérieure

        # S'assurer que la matrice est une rotation (det(q) = 1)
        det = torch.det(q)
        if self.k % 2 == 0:
            # q[:, 0, :] = torch.where(det.unsqueeze(1) < 0, -q[:, 0, :], q[:, 0, :])
            D = torch.eye(self.k, device=q.device).repeat(batch_size, 1, 1)  # (batch_size, k, k)
            D[:, 0, 0] = torch.where(det < 0, -1.0, 1.0)  # Flip sign only if det(q) < 0
            # Perform matrix multiplication
            q = D @ q  # Apply the correction
        else:            
            q = torch.where(det.unsqueeze(-1).unsqueeze(-1) < 0, -q, q)

        return q


